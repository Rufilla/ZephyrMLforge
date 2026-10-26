"""Scoring a model on the board over its serial link."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.core.config import PipelineConfig
from pipeline.runners import hil
from pipeline.runners.hil import HILRunner


class FakeLink:
    """ Stands in for DeviceLink; answers each inference with the reply
    the test chooses for that sample's first feature."""

    def __init__(self, reply):
        self.reply = reply
        self.requested: list[float] = []

    def ping(self, attempts: int = 3) -> bool:
        return True

    def request_benchmark(self, timeout_s: float) -> str:
        return "METRICS_START\nlatency_ns=1000000\narena_used=100\nMETRICS_END\n"

    def request_inference(self, features: list[float]) -> list[float]:
        self.requested.append(features[0])
        return self.reply(features[0])


def build_runner(tmp_path, on_device_samples: int) -> HILRunner:
    config = PipelineConfig.model_validate(
        {
            "hardware": {"board": "frdm_mcxn947/mcxn947/cpu0", "soc": "mcxn947",
                         "ram_kb": 512, "flash_kb": 2048},
            "accuracy": {"metric": "accuracy", "target": 0.95, "max_error": 0.80,
                         "evaluate_on_device": True,
                         "on_device_samples": on_device_samples},
            "performance": {"max_latency_ms": 5, "max_ram_kb": 128, "max_flash_kb": 256},
            "ai_agent": {"provider": "pseudo"},
            "synthetic_data": {"task": "classification"},
        }
    )
    return HILRunner(config, zephyr_base=tmp_path)


def ordered_by_class() -> tuple[np.ndarray, np.ndarray]:
    """ Nine samples of three classes, grouped by class as a vibration
    validation round is; each sample's only feature is its label."""
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return labels.astype(np.float32).reshape(-1, 1), labels


def one_hot(label: float) -> list[float]:
    return [1.0 if index == int(label) else 0.0 for index in range(3)]


def test_on_device_samples_are_spread_across_the_validation_set(tmp_path):
    link = FakeLink(one_hot)

    build_runner(tmp_path, on_device_samples=3)._measure(link, ordered_by_class())

    assert link.requested == [0.0, 1.0, 2.0]


def test_on_device_score_counts_only_the_samples_sent(tmp_path):
    def wrong_on_class_two(label: float) -> list[float]:
        return one_hot(0 if label == 2 else label)

    result = build_runner(tmp_path, on_device_samples=3)._measure(
        FakeLink(wrong_on_class_two), ordered_by_class()
    )

    assert result.accuracy_value == pytest.approx(2 / 3)
    assert result.accuracy_sample_count == 3


def test_a_model_giving_every_sample_the_same_output_fails(tmp_path):
    result = build_runner(tmp_path, on_device_samples=9)._measure(
        FakeLink(lambda _: [0.332, 0.332, 0.332]), ordered_by_class()
    )

    assert not result.success
    assert result.error_message == (
        "On-device evaluation failed: the model gave all 9 samples the same output, "
        "[0.332, 0.332, 0.332]"
    )


def runner_with_a_build(tmp_path) -> HILRunner:
    runner = build_runner(tmp_path, on_device_samples=3)
    runner._shared_build_dir = tmp_path
    elf_path = runner.west_build_dir / "zephyr" / "zephyr.elf"
    elf_path.parent.mkdir(parents=True)
    elf_path.touch()
    return runner


def test_a_failed_flash_is_reported_as_a_failed_run(tmp_path, monkeypatch):
    runner = runner_with_a_build(tmp_path)
    monkeypatch.setattr(runner, "_flash_device", lambda: "probe not found")

    result = runner.run(tmp_path, ordered_by_class())

    assert not result.success
    assert result.error_message == "Flash failed: probe not found"


def test_no_port_after_flashing_is_reported_as_a_failed_run(tmp_path, monkeypatch):
    runner = runner_with_a_build(tmp_path)
    monkeypatch.setattr(runner, "_flash_device", lambda: None)
    monkeypatch.setattr(hil, "wait_for_serial_port", lambda port, timeout_s: None)

    result = runner.run(tmp_path, ordered_by_class())

    assert not result.success
    assert result.error_message.startswith("No serial port appeared within")
