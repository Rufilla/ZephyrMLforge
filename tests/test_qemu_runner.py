"""Reading the emulator's console, and forwarding accelerator settings."""

from __future__ import annotations

import subprocess
import sys

import pytest

from pipeline.runners.qemu import AutoRunner, QEMURunner
from tests.test_config import build_config

# Prints the whole METRICS block in one burst, then stays alive. A reader that takes
# only the first line of each burst never sees the completion marker.
BURST_SCRIPT = (
    "import time\n"
    "print('*** Booting Zephyr OS ***')\n"
    "print('STATUS=ready')\n"
    "print('METRICS_START')\n"
    "print('latency_ns=41200')\n"
    "print('arena_used=772')\n"
    "print('METRICS_END')\n"
    "print('Inference complete')\n"
    "time.sleep(30)\n"
)


@pytest.fixture
def runner(tmp_path):
    """A QEMU runner whose build directory exists."""
    runner = QEMURunner(build_config(), zephyr_base=tmp_path)
    runner._shared_build_dir = tmp_path / "shared"
    (tmp_path / "shared" / "build").mkdir(parents=True)
    return runner


def test_a_burst_of_console_output_is_captured_whole(runner, monkeypatch):
    # select() reports readability on the descriptor, but readline() on a text pipe
    # buffers the rest of the burst where select() cannot see it.
    real_popen = subprocess.Popen
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda command, **kwargs: real_popen([sys.executable, "-u", "-c", BURST_SCRIPT], **kwargs),
    )

    stdout, _, has_completed = runner._run_qemu()

    assert has_completed is True
    assert "latency_ns=41200" in stdout
    assert "METRICS_END" in stdout


def test_the_emulator_is_stopped_even_though_it_outlives_the_marker(runner, monkeypatch):
    # The script sleeps for 30s after printing; the runner must not leave it behind.
    real_popen = subprocess.Popen
    started = []

    def fake_popen(command, **kwargs):
        process = real_popen([sys.executable, "-u", "-c", BURST_SCRIPT], **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    runner._run_qemu()

    assert started[0].poll() is not None


def test_the_auto_runner_forwards_the_accelerator_to_both_runners(tmp_path):
    # Set on the AutoRunner alone it reached neither, so an accelerator-compiled model
    # was built into firmware that never registers the custom operator.
    runner = AutoRunner(build_config(), zephyr_base=tmp_path)

    runner.uses_neutron = True

    assert runner._qemu_runner.uses_neutron is True
    assert runner._hil_runner.uses_neutron is True
    assert runner.uses_neutron is True
