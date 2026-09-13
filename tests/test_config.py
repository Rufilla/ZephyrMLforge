"""Configuration schema, metric direction and credential handling."""

from __future__ import annotations

import pytest
import yaml

from pipeline.core.config import AccuracyMetric, PipelineConfig, TaskDescription

MINIMAL_CONFIG = {
    "hardware": {"board": "frdm_mcxn947", "soc": "mcxn947", "ram_kb": 512, "flash_kb": 2048},
    "accuracy": {"metric": "mse", "target": 0.01, "max_error": 0.05},
    "performance": {"max_latency_ms": 5, "max_ram_kb": 128, "max_flash_kb": 256},
    "ai_agent": {"provider": "pseudo"},
    "synthetic_data": {"task": "regression"},
}


def build_config(**overrides) -> PipelineConfig:
    """Build a valid configuration, with sections replaced wholesale."""
    return PipelineConfig.model_validate({**MINIMAL_CONFIG, **overrides})


def test_error_metric_treats_lower_as_better():
    accuracy = build_config().accuracy

    assert accuracy.metric.is_higher_better is False
    assert accuracy.is_within_limit(0.04) is True
    assert accuracy.is_within_limit(0.06) is False
    assert accuracy.meets_target(0.005) is True
    assert accuracy.is_better(0.01, than=0.02) is True


def test_accuracy_metric_treats_higher_as_better():
    accuracy = build_config(
        accuracy={"metric": "accuracy", "target": 0.95, "max_error": 0.80},
        synthetic_data={"task": "classification"},
    ).accuracy

    assert accuracy.metric.is_higher_better is True
    assert accuracy.is_within_limit(0.85) is True
    assert accuracy.is_within_limit(0.75) is False
    assert accuracy.meets_target(0.96) is True
    assert accuracy.is_better(0.9, than=0.8) is True


def test_rejects_error_target_looser_than_its_limit():
    with pytest.raises(ValueError, match="must be <= max_error"):
        build_config(accuracy={"metric": "mse", "target": 0.5, "max_error": 0.05})


def test_rejects_accuracy_target_below_its_limit():
    with pytest.raises(ValueError, match="must be >= max_error"):
        build_config(
            accuracy={"metric": "accuracy", "target": 0.5, "max_error": 0.9},
            synthetic_data={"task": "classification"},
        )


def test_rejects_classification_scored_by_an_error_metric():
    with pytest.raises(ValueError, match="Classification requires"):
        build_config(synthetic_data={"task": "classification"})


def test_rejects_arena_larger_than_the_ram_budget():
    with pytest.raises(ValueError, match="tensor_arena_kb"):
        build_config(
            hardware={**MINIMAL_CONFIG["hardware"], "tensor_arena_kb": 256},
            performance={"max_latency_ms": 5, "max_ram_kb": 128, "max_flash_kb": 256},
        )


def test_rejects_ram_budget_larger_than_the_hardware():
    with pytest.raises(ValueError, match="exceeds hardware RAM"):
        build_config(performance={"max_latency_ms": 5, "max_ram_kb": 4096, "max_flash_kb": 256})


def test_expands_an_environment_password_without_exposing_it(monkeypatch):
    monkeypatch.setenv("TEST_PIPELINE_PASSWORD", "s3cret-value")

    config = build_config(user={"username": "owen", "password": "${TEST_PIPELINE_PASSWORD}"})

    assert config.user.password.get_secret_value() == "s3cret-value"
    assert "s3cret-value" not in repr(config.user)


def test_snapshot_writes_the_reference_not_the_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_PIPELINE_PASSWORD", "s3cret-value")
    config = build_config(user={"username": "owen", "password": "${TEST_PIPELINE_PASSWORD}"})

    snapshot_path = tmp_path / "config_snapshot.yaml"
    config.to_yaml(snapshot_path)
    written = snapshot_path.read_text()

    assert "s3cret-value" not in written
    assert yaml.safe_load(written)["user"]["password"] == "${TEST_PIPELINE_PASSWORD}"


def test_snapshot_round_trips_back_into_a_valid_configuration(tmp_path):
    snapshot_path = tmp_path / "config_snapshot.yaml"
    build_config().to_yaml(snapshot_path)

    reloaded = PipelineConfig.from_yaml(snapshot_path)

    assert reloaded.accuracy.metric is AccuracyMetric.MSE
    assert reloaded.hardware.board == "frdm_mcxn947"


def test_constraints_summary_omits_credentials(monkeypatch):
    monkeypatch.setenv("TEST_PIPELINE_PASSWORD", "s3cret-value")
    config = build_config(user={"username": "owen", "password": "${TEST_PIPELINE_PASSWORD}"})

    assert "s3cret-value" not in str(config.get_constraints_summary(TaskDescription()))


def test_task_description_is_found_from_a_subdirectory(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.zephyr-ml-forge.task]\ndescription = "Approximate a sine wave"\n'
    )
    nested = tmp_path / "config" / "deeper"
    nested.mkdir(parents=True)

    assert TaskDescription.discover(nested).description == "Approximate a sine wave"


def test_task_description_is_empty_when_no_pyproject_exists(tmp_path):
    assert TaskDescription.discover(tmp_path).description == ""


def test_reports_a_target_below_the_mean_squared_error_floor():
    # Labels carry N(0, 0.1) noise, so a perfect model still scores 0.01.
    config = build_config(
        accuracy={"metric": "mse", "target": 0.01, "max_error": 0.05},
        synthetic_data={"task": "regression", "noise_stddev": 0.1},
    )

    warning = config.unreachable_target()

    assert warning is not None
    assert "no model can reach it" in warning


def test_accepts_a_target_above_the_floor():
    config = build_config(
        accuracy={"metric": "mse", "target": 0.02, "max_error": 0.05},
        synthetic_data={"task": "regression", "noise_stddev": 0.1},
    )

    assert config.unreachable_target() is None


def test_reports_a_target_below_the_absolute_error_floor():
    # The half-normal mean, 0.1 * sqrt(2/pi) = 0.0798.
    config = build_config(
        accuracy={"metric": "mae", "target": 0.05, "max_error": 0.2},
        synthetic_data={"task": "regression", "noise_stddev": 0.1},
    )

    assert "0.079788" in config.unreachable_target()


def test_a_noiseless_dataset_has_no_floor():
    config = build_config(
        accuracy={"metric": "mse", "target": 0.0001, "max_error": 0.05},
        synthetic_data={"task": "regression", "noise_stddev": 0.0},
    )

    assert config.unreachable_target() is None


def test_classification_has_no_additive_noise_floor():
    config = build_config(
        accuracy={"metric": "accuracy", "target": 0.99, "max_error": 0.8},
        synthetic_data={"task": "classification", "noise_stddev": 0.1},
    )

    assert config.unreachable_target() is None
