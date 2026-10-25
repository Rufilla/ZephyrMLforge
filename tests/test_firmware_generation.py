"""Generation of the Zephyr application tree."""

from __future__ import annotations

import re
import subprocess

import pytest

from pipeline.core.config import PipelineConfig, VibrationConfig
from pipeline.runners.firmware import DEFAULT_APP_SOURCE, FirmwareSource
from pipeline.runners.hil import HILRunner

MODEL_BYTES = bytes(range(256)) * 3 + b"\x01\x02\x03"


@pytest.fixture
def model_file(tmp_path):
    """A stand-in .tflite file whose bytes are easy to compare."""
    path = tmp_path / "model.tflite"
    path.write_bytes(MODEL_BYTES)
    return path


def extract_model_bytes(source: str) -> bytes:
    """Read the g_model array back out of a generated model.cpp."""
    body = source.split("g_model[] = {")[1].split("};")[0]
    return bytes(int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{2})", body))


def test_model_source_round_trips_every_byte(tmp_path, model_file):
    FirmwareSource().write_model(tmp_path, model_file)

    source = (tmp_path / "model.cpp").read_text()

    assert extract_model_bytes(source) == MODEL_BYTES
    assert f"const unsigned int g_model_len = {len(MODEL_BYTES)};" in source


def test_model_header_declares_an_unsigned_length(tmp_path, model_file):
    # A signed declaration against an unsigned definition is a link error.
    FirmwareSource().write_model(tmp_path, model_file)

    assert "extern const unsigned int g_model_len;" in (tmp_path / "model.hpp").read_text()


def test_model_array_is_aligned_for_the_flatbuffer_reader(tmp_path, model_file):
    FirmwareSource().write_model(tmp_path, model_file)

    assert "alignas(8) const unsigned char g_model[]" in (tmp_path / "model.cpp").read_text()


def test_app_config_carries_the_configured_values(tmp_path):
    FirmwareSource().write_app_config(tmp_path, 16384, 250, is_interactive=True)

    written = (tmp_path / "app_config.h").read_text()

    assert "#define MLFORGE_TENSOR_ARENA_SIZE 16384" in written
    assert "#define MLFORGE_BENCHMARK_RUNS 250" in written
    assert "#define MLFORGE_INTERACTIVE 1" in written


def test_app_config_compiles_out_the_command_loop_when_not_interactive(tmp_path):
    FirmwareSource().write_app_config(tmp_path, 8192, 100, is_interactive=False)

    assert "#define MLFORGE_INTERACTIVE 0" in (tmp_path / "app_config.h").read_text()


def test_prepared_application_builds_from_the_checked_in_tree(tmp_path, model_file):
    app_dir = tmp_path / "app"

    FirmwareSource().prepare_app(
        app_dir, model_file, tensor_arena_bytes=4096, benchmark_runs=10, is_interactive=False
    )

    assert (app_dir / "CMakeLists.txt").is_file()
    assert (app_dir / "prj.conf").is_file()
    for name in ("main.c", "main_functions.cpp", "output_handler.cpp", "model.cpp"):
        assert (app_dir / "src" / name).is_file()


def test_prepared_application_uses_the_generated_model_not_the_placeholder(tmp_path, model_file):
    app_dir = tmp_path / "app"

    FirmwareSource().prepare_app(
        app_dir, model_file, tensor_arena_bytes=4096, benchmark_runs=10, is_interactive=False
    )

    source = (app_dir / "src" / "model.cpp").read_text()
    assert extract_model_bytes(source) == MODEL_BYTES


def test_reports_a_missing_source_tree_clearly(tmp_path, model_file):
    firmware = FirmwareSource(app_source=tmp_path / "absent")

    with pytest.raises(FileNotFoundError, match="pip install -e"):
        firmware.prepare_app(
            tmp_path / "app", model_file, tensor_arena_bytes=4096, benchmark_runs=10,
            is_interactive=False,
        )


def test_app_config_enables_vibration_with_its_window_settings(tmp_path):
    vibration = VibrationConfig(sample_rate_hz=800, window_samples=64)

    FirmwareSource().write_app_config(tmp_path, 8192, 100, is_interactive=True, vibration=vibration)

    written = (tmp_path / "app_config.h").read_text()
    assert "#define MLFORGE_VIBRATION 1" in written
    assert "#define MLFORGE_WINDOW_SAMPLES 64" in written
    assert "#define MLFORGE_SAMPLE_RATE_HZ 800" in written


def test_app_config_leaves_vibration_off_by_default(tmp_path):
    FirmwareSource().write_app_config(tmp_path, 8192, 100, is_interactive=True)

    assert "#define MLFORGE_VIBRATION 0" in (tmp_path / "app_config.h").read_text()


def test_capture_firmware_keeps_the_placeholder_model(tmp_path):
    app_dir = tmp_path / "app"

    FirmwareSource().prepare_app(
        app_dir, None, tensor_arena_bytes=4096, benchmark_runs=10, is_interactive=True,
        vibration=VibrationConfig(),
    )

    assert (app_dir / "src" / "model.cpp").read_text() == (
        DEFAULT_APP_SOURCE / "src" / "model.cpp"
    ).read_text()
    assert (app_dir / "vibration.overlay").is_file()


def west_build_command(monkeypatch, tmp_path, source: str) -> list[str]:
    """ The west command a hardware runner issues, with west replaced."""
    config = PipelineConfig.model_validate(
        {
            "hardware": {"board": "frdm_mcxn947/mcxn947/cpu0", "soc": "mcxn947",
                         "ram_kb": 512, "flash_kb": 2048},
            "accuracy": {"metric": "accuracy", "target": 0.95, "max_error": 0.80},
            "performance": {"max_latency_ms": 5, "max_ram_kb": 128, "max_flash_kb": 256},
            "ai_agent": {"provider": "pseudo"},
            "synthetic_data": {"task": "classification"},
            "dataset": {"source": source},
        }
    )
    commands = []

    def fake_run(command, **_):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    HILRunner(config, zephyr_base=tmp_path).build_capture_firmware(tmp_path / "build")
    return commands[0]


def test_vibration_build_adds_the_click_overlay_and_configuration(monkeypatch, tmp_path):
    command = west_build_command(monkeypatch, tmp_path, "vibration")

    assert command[-2:] == [
        "-DEXTRA_DTC_OVERLAY_FILE=vibration.overlay", "-DEXTRA_CONF_FILE=vibration.conf"
    ]


def test_image_build_leaves_the_click_overlay_out(monkeypatch, tmp_path):
    command = west_build_command(monkeypatch, tmp_path, "fashion_mnist")

    assert not any("vibration" in argument for argument in command)
