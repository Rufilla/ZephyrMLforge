# _____________________________________________________________________________
#
# @file hil.py
# @brief Hardware-in-the-Loop (HIL) runner for physical board execution
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Hardware-in-the-Loop (HIL) runner for physical board execution.

Builds and flashes the generated application, then talks to the firmware over its
console: one command asks for a fresh latency measurement, and one asks for a
prediction, which is how accuracy can be measured on the hardware itself.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import numpy as np

from ..models.evaluation import MetricScore, evaluate_tflite, score_predictions
from .base import ExecutionResult
from .qemu import QEMURunner, parse_arena_used, parse_device_error, parse_latency_ms
from .serial_link import DeviceError, DeviceLink, wait_for_serial_port
from .zephyr import ZephyrRunner

logger = logging.getLogger(__name__)

# Seconds allowed for the board to answer a BENCH request, which runs the whole
# benchmark loop before it prints anything.
BENCHMARK_REPLY_TIMEOUT_S = 20.0


class HILRunner(ZephyrRunner):
    """Runner that builds, flashes and measures on physical hardware."""

    is_interactive = True

    # Shared with the emulator runner, which has the same orphaned-child problem.
    _terminate_process_group = staticmethod(QEMURunner._terminate_process_group)

    @property
    def board(self) -> str:
        """The real target board."""
        return self.config.hardware.board

    def run(
        self,
        build_dir: Path,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> ExecutionResult:
        """Flash the board, measure latency on it, and score the model."""
        logger.info(f"Flashing and running on {self.board}")

        elf_path = self.west_build_dir / "zephyr" / "zephyr.elf"
        if not elf_path.exists():
            return ExecutionResult(
                success=False,
                execution_env="hil",
                error_message=f"Built binary not found: {elf_path}",
            )

        flash_error = self._flash_device()
        if flash_error:
            return ExecutionResult(
                success=False, execution_env="hil", error_message=flash_error
            )

        environment = self.config.execution_environment
        logger.info("Flashed; waiting for the serial port to come back")
        port = wait_for_serial_port(environment.serial_port, timeout_s=environment.run_timeout_s)
        if port is None:
            return ExecutionResult(
                success=False,
                execution_env="hil",
                error_message=(
                    f"No serial port appeared within {environment.run_timeout_s}s of flashing "
                    f"(expected {environment.serial_port})"
                ),
            )

        try:
            with DeviceLink(port, environment.serial_baud) as link:
                return self._measure(link, test_data)
        except DeviceError as error:
            return ExecutionResult(
                success=False, execution_env="hil", error_message=str(error)
            )

    def _measure(
        self,
        link: DeviceLink,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> ExecutionResult:
        """Collect latency and accuracy over an open link."""
        # The board runs its own benchmark at boot before it listens, and a slow model
        # can hold it there for many seconds, so the wait tracks the run timeout rather
        # than a fixed few attempts.
        environment = self.config.execution_environment
        ping_attempts = max(3, environment.run_timeout_s // 2)

        # Logged because nothing else is: between the flash and the first metric the
        # board can hold the run for the best part of a minute, and silence there is
        # indistinguishable from a hang.
        logger.info(f"Waiting up to {ping_attempts * 2}s for the board to answer PING")

        if not link.ping(attempts=ping_attempts):
            # The firmware reports its own failures on this line, and reading them back
            # is the difference between a cause and a timeout.
            startup = link.read_until_marker("STATUS=", timeout_s=2.0)
            reported = parse_device_error(startup)

            return ExecutionResult(
                success=False,
                execution_env="hil",
                error_message=(
                    f"Firmware reported {reported}"
                    if reported
                    else f"Board did not answer PING within {environment.run_timeout_s}s; "
                    "it faulted during setup, or is still running its start-up benchmark"
                ),
                stdout=startup,
            )

        logger.info("Requesting a fresh benchmark from the board")
        output = link.request_benchmark(
            timeout_s=max(BENCHMARK_REPLY_TIMEOUT_S, float(environment.run_timeout_s))
        )

        device_error = parse_device_error(output)
        if device_error:
            return ExecutionResult(
                success=False,
                execution_env="hil",
                error_message=f"Firmware reported {device_error}",
                stdout=output,
            )

        latency_ms = parse_latency_ms(output)
        if latency_ms is None:
            return ExecutionResult(
                success=False,
                execution_env="hil",
                error_message=(
                    "No usable latency in the METRICS block: absent, or reported "
                    "as zero because the target has no working cycle counter"
                ),
                stdout=output,
            )

        try:
            score = self._evaluate(link, test_data)
        except DeviceError as error:
            return ExecutionResult(
                success=False,
                execution_env="hil",
                error_message=f"On-device evaluation failed: {error}",
                stdout=output,
            )

        return ExecutionResult(
            success=True,
            execution_env="hil",
            latency_ms=latency_ms,
            accuracy_value=score.value if score else None,
            accuracy_standard_error=score.standard_error if score else None,
            accuracy_sample_count=score.sample_count if score else None,
            arena_used_bytes=parse_arena_used(output),
            stdout=output,
        )

    def _evaluate(
        self,
        link: DeviceLink,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> MetricScore | None:
        """Score the model, on the board or on the host interpreter.

        Raises:
            DeviceError: the board stopped answering during on-device evaluation.
        """
        features, labels = test_data

        if not self.config.accuracy.evaluate_on_device:
            if self._current_tflite_path is None:
                return None
            return evaluate_tflite(
                self._current_tflite_path, test_data, self.config.accuracy.metric
            )

        # One serial round trip per sample, so the whole validation set is not used.
        sample_count = min(self.config.accuracy.on_device_samples, len(features))
        logger.info(f"Evaluating {sample_count} samples on the board")

        # Flattened: the firmware reads one value per input element, and an image
        # sample is (28, 28, 1) rather than a flat vector.
        predictions = [
            link.request_inference(np.asarray(features[index]).ravel().tolist())
            for index in range(sample_count)
        ]

        score = score_predictions(
            np.array(predictions), labels[:sample_count], self.config.accuracy.metric
        )
        logger.info(f"On-device evaluation: {self.config.accuracy.metric.value}={score}")
        return score

    def _flash_device(self) -> str | None:
        """Flash the built image.

        Returns:
            An error message, or None when the flash succeeded.
        """
        env = os.environ.copy()
        env["ZEPHYR_BASE"] = str(self.zephyr_base)

        environment = self.config.execution_environment
        command = ["west", "flash", "-d", str(self.west_build_dir)]

        # The board default is whatever runners.yaml lists first, which on the
        # FRDM-MCXN947 is LinkServer, an NXP download rather than a pip install.
        if environment.flash_runner:
            command += ["-r", environment.flash_runner]

        timeout_s = environment.flash_timeout_s
        logger.info(f"Running: {' '.join(command)}")

        # Its own session: west launches the flash runner as a grandchild, and killing
        # only west leaves that holding the debug probe, so every later flash fails.
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
        except FileNotFoundError:
            return "west not found on PATH; activate the project environment"

        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._terminate_process_group(process)
            return f"Flash timed out after {timeout_s}s"

        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

        if completed.returncode != 0:
            return completed.stderr or completed.stdout

        return None
