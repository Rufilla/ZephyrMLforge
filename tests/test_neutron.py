"""Compilation of a model for the Neutron NPU."""

from __future__ import annotations

import subprocess

import pytest

from pipeline.models.neutron import NeutronCompilerError, _read_summary, compile_for_neutron

# The compiler's report, as observed converting a three-layer MLP for mcxn94x.
COMPILER_REPORT = """
Neutron Compiler
  Operators:
    Operators converted                 = 0,1,2,
  Memory:
    Total weights = 816 (bytes) (Weights)
  Latency:
    Cycle estimation   = 456 (cycles) (NPU only)
"""

NOTHING_CONVERTED_REPORT = """
  Operators:
    Operators converted                 =
  Latency:
    Cycle estimation   = 0 (cycles) (NPU only)
"""


def test_reads_the_operator_count_and_cycle_estimate(tmp_path):
    model = tmp_path / "model_npu.tflite"
    model.write_bytes(b"x" * 1888)

    result = _read_summary(COMPILER_REPORT, model)

    assert result.converted_operator_count == 3
    assert result.estimated_cycles == 456
    assert result.output_bytes == 1888
    assert result.uses_npu is True


def test_a_model_with_nothing_mapped_does_not_use_the_npu(tmp_path):
    # The run would otherwise report the accelerator while measuring the CPU.
    model = tmp_path / "model_npu.tflite"
    model.write_bytes(b"x")

    assert _read_summary(NOTHING_CONVERTED_REPORT, model).uses_npu is False


def test_reports_a_missing_compiler_with_the_install_command(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.models.neutron.find_neutron_compiler", lambda: None)

    with pytest.raises(NeutronCompilerError, match="eiq-neutron-sdk"):
        compile_for_neutron(tmp_path / "in.tflite", tmp_path / "out.tflite", "mcxn94x")


def test_reports_a_compiler_that_rejected_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.models.neutron.find_neutron_compiler", lambda: "neutron_compiler")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "unsupported operator"),
    )

    with pytest.raises(NeutronCompilerError, match="unsupported operator"):
        compile_for_neutron(tmp_path / "in.tflite", tmp_path / "out.tflite", "mcxn94x")


def test_reports_a_success_that_wrote_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.models.neutron.find_neutron_compiler", lambda: "neutron_compiler")
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, COMPILER_REPORT, "")
    )

    with pytest.raises(NeutronCompilerError, match="wrote no model"):
        compile_for_neutron(tmp_path / "in.tflite", tmp_path / "absent.tflite", "mcxn94x")


def test_passes_the_boards_target_to_the_compiler(tmp_path, monkeypatch):
    recorded = {}
    output = tmp_path / "out.tflite"

    def fake_run(command, **kwargs):
        recorded["command"] = command
        output.write_bytes(b"x" * 1712)
        return subprocess.CompletedProcess(command, 0, COMPILER_REPORT, "")

    monkeypatch.setattr("pipeline.models.neutron.find_neutron_compiler", lambda: "neutron_compiler")
    monkeypatch.setattr(subprocess, "run", fake_run)

    compile_for_neutron(tmp_path / "in.tflite", output, "mcxn94x")

    assert recorded["command"][recorded["command"].index("--target") + 1] == "mcxn94x"


#  The description a compiled model carries, as written by converter 3.0.1.
CONVERTER_301_DESCRIPTION = (
    b"MLIR Converted.\nConverted with eIQ neutron-converter\n"
    b"Version: 3.0.1+0Xd5d7918e\nTarget: mcxn94x\n"
)

MODULE_YML = """
blobs:
  - path: neutron/mcxn/libNeutronDriver.a
    sha256: 452b1077a015265d1ab34831db3a25a4ef266c28c94afa6a308b28794baac358
    type: lib
    version: '{version}'
"""


def build_west_tree(tmp_path, driver_version: str):
    """ A west workspace holding only hal_nxp's module.yml; returns its
    zephyr_base."""
    module_yml = tmp_path / "modules" / "hal" / "nxp" / "zephyr" / "module.yml"
    module_yml.parent.mkdir(parents=True)
    module_yml.write_text(MODULE_YML.format(version=driver_version))
    return tmp_path / "zephyr"


def compile_with_converter_301(tmp_path, monkeypatch, zephyr_base):
    output = tmp_path / "out.tflite"

    def fake_run(command, **kwargs):
        output.write_bytes(b"TFL3" + CONVERTER_301_DESCRIPTION)
        return subprocess.CompletedProcess(command, 0, COMPILER_REPORT, "")

    monkeypatch.setattr("pipeline.models.neutron.find_neutron_compiler", lambda: "neutron_compiler")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return compile_for_neutron(tmp_path / "in.tflite", output, "mcxn94x", zephyr_base=zephyr_base)


def test_refuses_a_model_from_a_converter_newer_than_the_driver_blob(tmp_path, monkeypatch):
    zephyr_base = build_west_tree(tmp_path, driver_version="3.0.0")

    with pytest.raises(NeutronCompilerError) as raised:
        compile_with_converter_301(tmp_path, monkeypatch, zephyr_base)

    assert str(raised.value) == (
        "The model was built by Neutron converter 3.0.1, but the firmware links driver "
        "blob 3.0.0. Install the matching converter:\n"
        "  pip install --index-url https://eiq.nxp.com/repository "
        "eiq-neutron-sdk==3.0.0 eiq_nsys"
    )


def test_accepts_a_model_from_the_driver_blobs_version(tmp_path, monkeypatch):
    zephyr_base = build_west_tree(tmp_path, driver_version="3.0.1")

    result = compile_with_converter_301(tmp_path, monkeypatch, zephyr_base)

    assert result.converted_operator_count == 3


def test_warns_when_the_driver_blob_version_cannot_be_found(tmp_path, monkeypatch, caplog):
    result = compile_with_converter_301(tmp_path, monkeypatch, tmp_path / "zephyr")

    assert result.converted_operator_count == 3
    assert "Could not compare the Neutron converter version (3.0.1) with the driver " \
        "blob's (unknown)" in caplog.text
