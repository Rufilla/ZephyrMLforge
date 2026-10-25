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
        if command.startswith("INFER") or command == "INVOKE":
            return "OUT=0.482906\n"
        if command.startswith("LOAD"):
            return f"OK loaded={len(command.split(' ', 2)[2].split(','))}\n"
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


def test_motor_command_reads_past_the_echo_to_its_acknowledgement():
    fake = FakeDevice(replies={"MOTOR 2": "OK\n"})
    try:
        with DeviceLink(fake.port, 115200) as link:
            link.set_motors(2)
    finally:
        fake.close()

    assert fake.received == ["MOTOR 2"]


def test_motor_command_on_firmware_without_vibration_support_is_an_error(device):
    with DeviceLink(device.port, 115200) as link:
        with pytest.raises(DeviceError, match="Board rejected MOTOR: ERR=unknown_command"):
            link.set_motors(1)


def test_window_returns_its_duration_and_values():
    #  The echo 'WINDOW' also starts with 'WINDOW', so the reply is told
    # apart by its 'us=' field.
    fake = FakeDevice(replies={"WINDOW": "WINDOW us=128500 v=0.012,-0.034,1.500\n"})
    try:
        with DeviceLink(fake.port, 115200) as link:
            assert link.capture_window() == (128500, [0.012, -0.034, 1.5])
    finally:
        fake.close()


def test_window_overrun_is_raised_not_parsed():
    fake = FakeDevice(replies={"WINDOW": "ERR=sample_overrun rate_hz=1000\n"})
    try:
        with DeviceLink(fake.port, 115200) as link:
            with pytest.raises(DeviceError, match="sample_overrun"):
                link.capture_window()
    finally:
        fake.close()


def test_classify_returns_the_class_scores():
    fake = FakeDevice(replies={"CLASSIFY": "OUT=0.1,0.0,0.8,0.1\n"})
    try:
        with DeviceLink(fake.port, 115200) as link:
            assert link.classify() == [0.1, 0.0, 0.8, 0.1]
    finally:
        fake.close()


def test_long_inputs_are_sent_as_load_lines_short_enough_for_the_console(device):
    #  Zephyr's uart_console wraps a line past 255 characters.
    with DeviceLink(device.port, 115200) as link:
        outputs = link.request_inference([0.5] * 45)

    assert outputs == [0.482906]
    assert [command.split(" ")[:2] for command in device.received] == [
        ["LOAD", "0"], ["LOAD", "26"], ["INVOKE"]
    ]
    assert max(len(command) for command in device.received) < 256


def test_a_board_loading_fewer_inputs_than_sent_fails_the_inference(device):
    device.replies["LOAD 26 " + ",".join(["0.500000"] * 19)] = "OK loaded=4\n"

    with DeviceLink(device.port, 115200) as link:
        with pytest.raises(DeviceError, match="Board loaded 4 of the 19 inputs sent from offset 26"):
            link.request_inference([0.5] * 45)

    assert "INVOKE" not in device.received


def test_load_lines_stay_short_when_values_are_wide(device):
    #  -123.456789 is 11 characters; a fixed count of values a line would
    # pass 255 here.
    with DeviceLink(device.port, 115200) as link:
        link.request_inference([-123.456789] * 384)

    loads = [command for command in device.received if command.startswith("LOAD")]
    assert max(len(command) for command in loads) <= 240
    assert sum(len(command.split(" ", 2)[2].split(",")) for command in loads) == 384


def test_short_inputs_still_go_as_one_infer_line(device):
    with DeviceLink(device.port, 115200) as link:
        link.request_inference([0.5] * 20)

    assert len(device.received) == 1
    assert device.received[0].startswith("INFER ")
