# _____________________________________________________________________________
#
# @file qemu.py
# @brief QEMU execution runner for Zephyr applications
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
QEMU execution runner for Zephyr applications.

Handles building and running Zephyr applications in the QEMU emulator, plus the
simulated runner used when no Zephyr toolchain is available and the auto-transitioning
runner that moves from QEMU to hardware.
"""

from __future__ import annotations

import logging
import os
import re
import select
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from ..core.config import PipelineConfig
from ..models.evaluation import evaluate_tflite, predict_tflite, score_predictions
from .base import BaseRunner, BuildResult, ExecutionResult
from .zephyr import ZephyrRunner

logger = logging.getLogger(__name__)

# Printed by the firmware once the METRICS block is complete.
COMPLETION_MARKER = "Inference complete"


def parse_latency_ms(output: str) -> float | None:
    """Read the mean inference latency out of a METRICS block.

    Nanoseconds are preferred: an inference faster than a microsecond truncates to zero
    in the other two fields, which would read as a free model rather than a fast one.

    Returns None for a reported zero, which means the target has no working cycle
    counter rather than an instantaneous inference; accepting it would satisfy every
    latency constraint.
    """
    for pattern, divisor in (
        (r"latency_ns=(\d+)", 1_000_000),
        (r"latency_ms=([0-9.]+)", 1),
        (r"latency_us=(\d+)", 1000),
    ):
        match = re.search(pattern, output)
        if match:
            latency_ms = float(match.group(1)) / divisor
            return latency_ms if latency_ms > 0 else None

    return None


def parse_arena_used(output: str) -> int | None:
    """Read the tensor arena bytes the interpreter claimed."""
    match = re.search(r"arena_used=(\d+)", output)
    return int(match.group(1)) if match else None


def parse_device_error(output: str) -> str | None:
    """Return the firmware's own failure reason, when it reported one.

    A board that fails to allocate its arena says so and then goes quiet; without this
    the host reports a timeout and hides the real cause.
    """
    match = re.search(r"STATUS=error\s+(.+)", output)
    return match.group(1).strip() if match else None


class QEMURunner(ZephyrRunner):
    """Runner that builds and executes Zephyr apps in QEMU."""

    is_interactive = False

    @property
    def board(self) -> str:
        """QEMU stand-in board; the real target rarely has emulator support."""
        return self.config.hardware.qemu_board

    def run(
        self,
        build_dir: Path,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> ExecutionResult:
        """Run the built application under QEMU and collect its metrics."""
        logger.info("Running application in QEMU")

        elf_path = self.west_build_dir / "zephyr" / "zephyr.elf"
        if not elf_path.exists():
            return ExecutionResult(
                success=False,
                execution_env="qemu",
                error_message=f"Built binary not found: {elf_path}",
            )

        stdout, stderr, has_completed = self._run_qemu()

        device_error = parse_device_error(stdout)
        if device_error:
            return ExecutionResult(
                success=False,
                execution_env="qemu",
                error_message=f"Firmware reported {device_error}",
                stdout=stdout,
                stderr=stderr,
            )

        if not has_completed:
            return ExecutionResult(
                success=False,
                execution_env="qemu",
                error_message=(
                    f"QEMU did not print '{COMPLETION_MARKER}' within "
                    f"{self.config.execution_environment.qemu_timeout_s}s"
                ),
                stdout=stdout,
                stderr=stderr,
            )

        latency_ms = parse_latency_ms(stdout)
        if latency_ms is None:
            return ExecutionResult(
                success=False,
                execution_env="qemu",
                error_message=(
                    "No usable latency in the METRICS block: absent, or reported "
                    "as zero because the target has no working cycle counter"
                ),
                stdout=stdout,
                stderr=stderr,
            )

        # QEMU has no channel to the guest console, so predictions come from the host
        # interpreter running the same quantized model.
        score = None
        if self._current_tflite_path is not None:
            score = evaluate_tflite(
                self._current_tflite_path, test_data, self.config.accuracy.metric
            )

        return ExecutionResult(
            success=True,
            execution_env="qemu",
            latency_ms=latency_ms,
            accuracy_value=score.value if score else None,
            accuracy_standard_error=score.standard_error if score else None,
            accuracy_sample_count=score.sample_count if score else None,
            arena_used_bytes=parse_arena_used(stdout),
            stdout=stdout,
            stderr=stderr,
        )

    def _run_qemu(self) -> tuple[str, str, bool]:
        """Start QEMU and read its console until the firmware signals completion.

        Returns:
            Captured stdout, stderr, and whether the completion marker arrived.
        """
        env = os.environ.copy()
        env["ZEPHYR_BASE"] = str(self.zephyr_base)

        command = ["west", "build", "-d", str(self.west_build_dir), "-t", "run"]
        timeout_s = self.config.execution_environment.qemu_timeout_s

        # Zephyr's run target refuses to start against a pid file left by a previous run.
        pid_file = self.west_build_dir / "qemu.pid"
        pid_file.unlink(missing_ok=True)

        has_completed = False

        # Its own session: west runs the emulator through ninja and a shell, and
        # terminating west alone leaves the emulator holding the console, so the next
        # iteration's run sees no output at all.
        # Binary, and read with os.read rather than readline: select() reports on the
        # file descriptor, while readline() on a text pipe pulls up to 8 KiB into
        # Python's own buffer and returns only the first line. The rest of a burst is
        # then invisible to select(), and the completion marker is never seen.
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []

        try:
            deadline = time.monotonic() + timeout_s

            while process.poll() is None and time.monotonic() < deadline:
                readable, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

                for stream in readable:
                    chunk = os.read(stream.fileno(), 4096)
                    if not chunk:
                        continue
                    if stream is process.stdout:
                        stdout_chunks.append(chunk)
                    else:
                        stderr_chunks.append(chunk)

                if COMPLETION_MARKER.encode() in b"".join(stdout_chunks):
                    has_completed = True
                    break
        finally:
            # QEMU runs forever once the firmware halts, so it is always stopped here.
            self._terminate_process_group(process)

        return (
            b"".join(stdout_chunks).decode("utf-8", errors="replace"),
            b"".join(stderr_chunks).decode("utf-8", errors="replace"),
            has_completed,
        )

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        """Stop the whole west/ninja/emulator group started by _run_qemu."""
        try:
            group_id = os.getpgid(process.pid)
        except ProcessLookupError:
            return

        for signal_number in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(group_id, signal_number)
            except ProcessLookupError:
                return

            try:
                process.wait(timeout=10)
                return
            except subprocess.TimeoutExpired:
                logger.warning("QEMU did not stop on SIGTERM; sending SIGKILL")


class SimulatedRunner(BaseRunner):
    """
    Simulated runner for development without a Zephyr toolchain.

    Runs the quantized model on the host interpreter. The latency it reports is host
    interpreter time, not an estimate of the target: it exists so the iteration loop can
    be exercised end to end, and it is not comparable with a QEMU or hardware figure.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._tflite_path: Path | None = None

    def build(
        self, tflite_path: Path, build_dir: Path, host_model_path: Path | None = None
    ) -> BuildResult:
        """Check the model exists and estimate its footprint from its size."""
        logger.info("Simulated build (no Zephyr compilation)")

        if not tflite_path.exists():
            return BuildResult(
                success=False,
                error_message=f"TFLite model not found: {tflite_path}",
            )

        self._tflite_path = tflite_path
        build_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(tflite_path, build_dir / "model.tflite")

        # Rough: the model plus a fixed allowance for the application and the arena.
        model_kb = tflite_path.stat().st_size // 1024

        return BuildResult(
            success=True,
            flash_usage_kb=model_kb + 10,
            ram_usage_kb=model_kb + self.config.hardware.tensor_arena_kb,
            build_dir=build_dir,
        )

    def run(
        self,
        build_dir: Path,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> ExecutionResult:
        """Run the model on the host interpreter and time it."""
        logger.info("Simulated execution (host TFLite interpreter)")

        tflite_path = build_dir / "model.tflite"
        if not tflite_path.exists():
            return ExecutionResult(
                success=False,
                execution_env="simulated",
                error_message=f"TFLite model not found: {tflite_path}",
            )

        features, labels = test_data

        try:
            start_time = time.perf_counter()
            predictions = predict_tflite(tflite_path, features)
            elapsed_s = time.perf_counter() - start_time

            score = score_predictions(predictions, labels, self.config.accuracy.metric)
        except Exception as error:
            logger.exception("Simulated execution failed")
            return ExecutionResult(
                success=False,
                execution_env="simulated",
                error_message=str(error),
            )

        return ExecutionResult(
            success=True,
            execution_env="simulated",
            latency_ms=elapsed_s / len(features) * 1000,
            accuracy_value=score.value,
            accuracy_standard_error=score.standard_error,
            accuracy_sample_count=score.sample_count,
        )


class AutoRunner(BaseRunner):
    """
    Auto-transitioning runner implementing the 'QEMU first, then HIL' policy.

    QEMU validates that the generated application builds and runs at all, which is
    cheap; hardware then supplies the timing that matters.
    """

    def __init__(self, config: PipelineConfig, zephyr_base: Path | None = None):
        from .hil import HILRunner

        self.config = config
        self._qemu_runner = QEMURunner(config, zephyr_base)
        self._hil_runner = HILRunner(config, zephyr_base)
        self._is_using_hil = False

    @property
    def uses_neutron(self) -> bool:
        """Whether the runners build for the accelerator."""
        return self._hil_runner.uses_neutron

    @uses_neutron.setter
    def uses_neutron(self, value: bool) -> None:
        """Forward to both runners; setting it here alone would reach neither."""
        self._qemu_runner.uses_neutron = value
        self._hil_runner.uses_neutron = value

    @property
    def current_runner(self) -> BaseRunner:
        """The runner handling this iteration."""
        return self._hil_runner if self._is_using_hil else self._qemu_runner

    @property
    def has_transitioned_to_hil(self) -> bool:
        """True once a QEMU run has succeeded and hardware has taken over."""
        return self._is_using_hil

    def build(
        self, tflite_path: Path, build_dir: Path, host_model_path: Path | None = None
    ) -> BuildResult:
        """Build with the runner handling this iteration."""
        logger.info(f"AutoRunner building with {'HIL' if self._is_using_hil else 'QEMU'}")
        return self.current_runner.build(tflite_path, build_dir, host_model_path)

    def run(
        self,
        build_dir: Path,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> ExecutionResult:
        """Run, then transition to hardware after the first successful QEMU run."""
        environment_name = "hil" if self._is_using_hil else "qemu"
        result = self.current_runner.run(build_dir, test_data)

        if not self._is_using_hil and result.success:
            logger.info("QEMU run succeeded; transitioning to hardware")
            self._is_using_hil = True

        result.execution_env = f"auto:{environment_name}"
        return result
