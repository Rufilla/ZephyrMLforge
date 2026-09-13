# _____________________________________________________________________________
#
# @file serial_link.py
# @brief Console line protocol client for the target board
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Console line protocol client for the target board.

The firmware prints a METRICS block after boot and then answers one command per line.
This module owns both halves of that conversation.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from collections.abc import Sequence

import serial

logger = logging.getLogger(__name__)


class DeviceError(RuntimeError):
    """The board answered with an error, or did not answer at all."""


def wait_for_serial_port(preferred_port: str, timeout_s: int = 15) -> str | None:
    """Wait for a serial port to appear after a flash.

    The USB CDC endpoint disappears while the MCU resets and may come back under a
    different name, so the configured port is preferred and any other ttyACM device is
    accepted as a fallback.

    Returns:
        The port that appeared, or None if none did within the timeout.
    """
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        if os.path.exists(preferred_port):
            return preferred_port

        candidates = sorted(glob.glob("/dev/ttyACM*"))
        if candidates:
            logger.warning(
                f"Configured port {preferred_port} absent; using {candidates[0]}. "
                "A second board on this machine would be picked up here."
            )
            return candidates[0]

        time.sleep(0.5)

    return None


class DeviceLink:
    """Reads metrics from the board and asks it for predictions."""

    def __init__(self, port: str, baud: int, read_timeout_s: float = 2.0):
        self.port = port
        self.baud = baud
        self.read_timeout_s = read_timeout_s
        self._serial: serial.Serial | None = None

    def __enter__(self) -> "DeviceLink":
        self.open()
        return self

    def __exit__(self, *exception_details) -> None:
        self.close()

    def open(self) -> None:
        """Open the port.

        Raises:
            DeviceError: the port could not be opened.
        """
        try:
            # A write timeout as well as a read one: a board that stops draining the USB
            # CDC endpoint fills the host buffer, and the default of None then blocks the
            # whole run inside write() with nothing logged.
            self._serial = serial.Serial(
                self.port,
                self.baud,
                timeout=self.read_timeout_s,
                write_timeout=self.read_timeout_s,
            )
        except serial.SerialException as error:
            raise DeviceError(f"Could not open {self.port}: {error}") from error

    def close(self) -> None:
        """Close the port if it is open."""
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def read_until_marker(self, marker: str, timeout_s: float) -> str:
        """Collect console output until ``marker`` appears or the timeout expires.

        Returns:
            Everything read, whether or not the marker arrived; the caller decides
            whether a partial capture is still useful.
        """
        deadline = time.monotonic() + timeout_s
        lines: list[str] = []

        while time.monotonic() < deadline:
            line = self._read_line()
            if line is None:
                continue

            lines.append(line)
            logger.debug(f"Serial: {line}")

            if marker in line:
                break

        return "\n".join(lines)

    def ping(self, attempts: int = 3) -> bool:
        """Check the command loop is answering."""
        for _ in range(attempts):
            self._write_line("PING")
            if "PONG" in self.read_until_marker("PONG", timeout_s=2.0):
                return True
        return False

    def request_benchmark(self, timeout_s: float) -> str:
        """Ask for a fresh latency measurement and return the console output.

        The block the firmware prints at boot is usually lost while the USB port is
        still enumerating, so it is re-requested rather than waited for.
        """
        self._write_line("BENCH")
        return self.read_until_marker("METRICS_END", timeout_s=timeout_s)

    def request_inference(self, features: Sequence[float]) -> list[float]:
        """Send one INFER command and return the dequantised outputs.

        Raises:
            DeviceError: the board reported an error or did not reply in time.
        """
        self._write_line("INFER " + ",".join(f"{value:.6f}" for value in features))

        deadline = time.monotonic() + self.read_timeout_s * 2
        while time.monotonic() < deadline:
            line = self._read_line()
            if line is None:
                continue

            if line.startswith("OUT="):
                return [float(value) for value in line[len("OUT=") :].split(",") if value]

            if line.startswith("ERR="):
                raise DeviceError(f"Board rejected the request: {line}")

        raise DeviceError("No reply to INFER within the read timeout")

    def _write_line(self, text: str) -> None:
        """Send one command line.

        Raises:
            DeviceError: the port is not open, or the board stopped reading it.
        """
        if self._serial is None:
            raise DeviceError("Serial port is not open")

        self._serial.reset_input_buffer()

        try:
            self._serial.write((text + "\n").encode("ascii"))
            self._serial.flush()
        except serial.SerialTimeoutException as error:
            raise DeviceError(
                f"The board stopped reading its console within {self.read_timeout_s}s: "
                "it faulted, or is busy in an inference that never returns"
            ) from error

    def _read_line(self) -> str | None:
        if self._serial is None:
            raise DeviceError("Serial port is not open")

        raw = self._serial.readline()
        if not raw:
            return None

        line = raw.decode("utf-8", errors="replace").strip()
        return line or None
