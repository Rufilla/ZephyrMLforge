"""The console protocol, against a pseudo-terminal standing in for the board."""

from __future__ import annotations

import os
import pty
import threading

import pytest

from pipeline.runners.serial_link import DeviceError, DeviceLink

METRICS_BLOCK = (
    "METRICS_START\n"
    "latency_ns=35462\n"
    "latency_us=35\n"
    "latency_ms=0.035462\n"
    "arena_used=772\n"
    "num_runs=100\n"
    "output_val=0.482906\n"
    "METRICS_END\n"
    "Inference complete\n"
)


class FakeDevice:
    """Answers the firmware's command protocol on one end of a pseudo-terminal.

    The replies are those observed from the firmware under QEMU, including the console
    echo of the command itself, which the host has to read past.
    """

    def __init__(self, replies: dict[str, str] | None = None):
        self.controller_fd, device_fd = pty.openpty()
        self.port = os.ttyname(device_fd)
        self.replies = replies or {}
        self.received: list[str] = []

        self._is_running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        buffer = b""
        while self._is_running:
            try:
                buffer += os.read(self.controller_fd, 128)
            except OSError:
                return

            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                command = line.decode().strip()
                if not command:
                    continue
                self.received.append(command)
                reply = self.replies.get(command, self._default_reply(command))
                os.write(self.controller_fd, f"{command}\n{reply}".encode())

    @staticmethod
    def _default_reply(command: str) -> str:
        if command == "PING":
            return "PONG\n"
        if command == "BENCH":
            return METRICS_BLOCK
        if command.startswith("INFER"):
            return "OUT=0.482906\n"
        return "ERR=unknown_command\n"

    def close(self) -> None:
        self._is_running = False
        os.close(self.controller_fd)


@pytest.fixture
def device():
    """A fake board on a pseudo-terminal."""
    fake = FakeDevice()
    yield fake
    fake.close()


def test_ping_is_answered(device):
    with DeviceLink(device.port, 115200) as link:
        assert link.ping() is True


def test_benchmark_request_returns_the_metrics_block(device):
    with DeviceLink(device.port, 115200) as link:
        output = link.request_benchmark(timeout_s=5)

    assert "METRICS_END" in output
    assert "latency_ns=35462" in output
    assert device.received == ["BENCH"]


def test_inference_request_sends_every_feature(device):
    with DeviceLink(device.port, 115200) as link:
        outputs = link.request_inference([0.5, -0.25])

    assert outputs == [0.482906]
    assert device.received == ["INFER 0.500000,-0.250000"]


def test_inference_reads_past_the_console_echo(device):
    # The board echoes the command before answering; taking the echo as the reply
    # would parse a float out of the command line.
    with DeviceLink(device.port, 115200) as link:
        assert link.request_inference([0.5]) == [0.482906]


def test_multiple_output_values_are_all_returned():
    fake = FakeDevice(replies={"INFER 0.500000": "OUT=0.1,0.7,0.2\n"})
    try:
        with DeviceLink(fake.port, 115200) as link:
            assert link.request_inference([0.5]) == [0.1, 0.7, 0.2]
    finally:
        fake.close()


def test_device_error_is_raised_not_parsed():
    fake = FakeDevice(replies={"INFER 0.500000": "ERR=expected_inputs=2\n"})
    try:
        with DeviceLink(fake.port, 115200) as link:
            with pytest.raises(DeviceError, match="expected_inputs=2"):
                link.request_inference([0.5])
    finally:
        fake.close()


def test_silence_raises_rather_than_hanging():
    fake = FakeDevice(replies={"INFER 0.500000": ""})
    try:
        with DeviceLink(fake.port, 115200, read_timeout_s=0.2) as link:
            with pytest.raises(DeviceError, match="No reply to INFER"):
                link.request_inference([0.5])
    finally:
        fake.close()


def test_opening_a_port_that_does_not_exist_is_reported():
    with pytest.raises(DeviceError, match="Could not open"):
        DeviceLink("/dev/ttyACM-absent", 115200).open()


def test_a_board_that_stops_reading_is_reported_rather_than_blocking(monkeypatch):
    # Observed on hardware: the port accepts no bytes, and without a write timeout the
    # run sits inside write() indefinitely with nothing logged.
    import serial

    fake = FakeDevice()
    try:
        with DeviceLink(fake.port, 115200, read_timeout_s=0.2) as link:

            def refuse(_data):
                raise serial.SerialTimeoutException("Write timeout")

            monkeypatch.setattr(link._serial, "write", refuse)

            with pytest.raises(DeviceError, match="stopped reading its console"):
                link.request_inference([0.5])
    finally:
        fake.close()
