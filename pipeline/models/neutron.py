# _____________________________________________________________________________
#
# @file neutron.py
# @brief Compilation of a quantized model for the NXP eIQ Neutron NPU
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Compilation of a quantized model for the NXP eIQ Neutron NPU.

The compiler rewrites a quantized TFLite model, replacing the operators the NPU can
run with a single ``NeutronGraph`` custom operator; anything it cannot map stays on
the CPU. The firmware must register that operator, which it does only when built for
the NPU.

The compiler ships in the eIQ Neutron SDK, installed from NXP's package index. Its
version must match the driver blobs in hal_nxp, 3.0.0 on the FRDM-MCXN947:

    pip install --index-url https://eiq.nxp.com/repository eiq-neutron-sdk==3.0.0 eiq_nsys
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# The executable was renamed between SDK versions, and the version must match the
# driver blobs, so both names are accepted.
NEUTRON_COMPILER_NAMES = ("neutron_compiler", "neutron_converter")

# 'Operators converted = 0,1,2,' in the compiler's summary.
CONVERTED_OPERATORS_PATTERN = re.compile(r"Operators converted\s*=\s*([\d,]*)")

# 'Cycle estimation   = 456 (cycles) (NPU only)'
CYCLE_ESTIMATE_PATTERN = re.compile(r"Cycle estimation\s*=\s*(\d+)")

#  The converter writes 'Converted with eIQ neutron-converter\nVersion:
# 3.0.1+0Xd5d7918e' into the model's description, so the version checked is the one
# that built the model the board will run.
CONVERTER_VERSION_PATTERN = re.compile(rb"neutron-converter\nVersion: (\d+\.\d+\.\d+)")

#  The driver blob each compiler target runs on, as listed in hal_nxp's
# zephyr/module.yml.
DRIVER_BLOBS = {"mcxn94x": "neutron/mcxn/libNeutronDriver.a"}


class NeutronCompilerError(RuntimeError):
    """The Neutron compiler is missing, or refused the model."""


def find_neutron_compiler() -> Path | None:
    """Locate the compiler, on PATH or beside the running interpreter.

    It is a pip package installed into the same environment as the pipeline, so it
    sits next to this interpreter whether or not the environment has been activated.
    """
    for name in NEUTRON_COMPILER_NAMES:
        on_path = shutil.which(name)
        if on_path:
            return Path(on_path)

        alongside = Path(sys.executable).parent / name
        if alongside.is_file():
            return alongside

    return None


@dataclass
class NeutronResult:
    """What the compiler made of a model."""

    output_path: Path
    converted_operator_count: int
    estimated_cycles: int | None
    output_bytes: int

    @property
    def uses_npu(self) -> bool:
        """False when nothing in the model could be mapped to the accelerator."""
        return self.converted_operator_count > 0


def compile_for_neutron(
    tflite_path: Path, output_path: Path, target: str, zephyr_base: Path | None = None
) -> NeutronResult:
    """Rewrite a quantized model so the NPU runs what it can.

    Args:
        tflite_path: Quantized int8 model to convert.
        output_path: Where to write the converted model.
        target: Compiler target for the board, such as 'mcxn94x'.
        zephyr_base: The Zephyr tree the firmware builds against. When given, the
            converter's version is checked against that tree's driver blob.

    Returns:
        What was converted, for the iteration record.

    Raises:
        NeutronCompilerError: the compiler is absent, rejected the model, or is a
            different version from the driver blob.
    """
    compiler = find_neutron_compiler()
    if compiler is None:
        raise NeutronCompilerError(
            f"Neither {' nor '.join(NEUTRON_COMPILER_NAMES)} was found. "
            "Install the eIQ Neutron SDK, matching the version of the driver blobs:\n"
            "  pip install --index-url https://eiq.nxp.com/repository "
            "eiq-neutron-sdk==3.0.0 eiq_nsys"
        )

    command = [
        str(compiler),
        "--input",
        str(tflite_path),
        "--output",
        str(output_path),
        "--target",
        target,
    ]
    logger.info(f"Running: {' '.join(command)}")

    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired as error:
        raise NeutronCompilerError("Neutron compilation timed out after 300s") from error

    if completed.returncode != 0:
        raise NeutronCompilerError(
            f"Neutron compilation failed: {(completed.stderr or completed.stdout).strip()[:500]}"
        )

    if not output_path.exists():
        raise NeutronCompilerError("Neutron compilation reported success but wrote no model")

    if zephyr_base is not None:
        check_driver_match(output_path, zephyr_base, target)

    result = _read_summary(completed.stdout, output_path)

    if not result.uses_npu:
        # Worth saying plainly: the run will measure the CPU while claiming the NPU.
        logger.warning(
            "The Neutron compiler mapped no operators to the NPU; this model runs "
            "entirely on the CPU despite execution.npu being set"
        )
    else:
        logger.info(
            f"Neutron compiled {result.converted_operator_count} operators to the NPU "
            f"({result.output_bytes} bytes, estimate {result.estimated_cycles} NPU cycles)"
        )

    return result


def converter_version(model_path: Path) -> str | None:
    """The converter version recorded in a compiled model, such as '3.0.1'."""
    match = CONVERTER_VERSION_PATTERN.search(model_path.read_bytes())
    return match.group(1).decode() if match else None


def driver_blob_version(zephyr_base: Path, target: str) -> str | None:
    """The version of the Neutron driver blob that firmware for target links.

    Returns:
        The version from hal_nxp's zephyr/module.yml, or None when the target has no
        known blob or the file is not at the usual west location beside zephyr_base.
    """
    blob = DRIVER_BLOBS.get(target)
    module_yml = zephyr_base.parent / "modules" / "hal" / "nxp" / "zephyr" / "module.yml"
    if blob is None or not module_yml.is_file():
        return None

    for entry in yaml.safe_load(module_yml.read_text()).get("blobs", []):
        if entry.get("path") == blob:
            return str(entry.get("version"))
    return None


def check_driver_match(model_path: Path, zephyr_base: Path, target: str) -> None:
    """Refuse a model built by a converter of another version than the driver blob.

    The driver's own check does not catch every mismatch: on the FRDM-MCXN947, models
    from converter 3.0.1 passed it on the 3.0.0 driver, then gave every input the same
    output or hard-faulted the board.

    Raises:
        NeutronCompilerError: the two versions differ.
    """
    built_with = converter_version(model_path)
    driver = driver_blob_version(zephyr_base, target)

    if built_with is None or driver is None:
        logger.warning(
            f"Could not compare the Neutron converter version ({built_with or 'unknown'}) "
            f"with the driver blob's ({driver or 'unknown'}) for {target}; a mismatch "
            "runs without an error but gives wrong outputs"
        )
        return

    if built_with != driver:
        raise NeutronCompilerError(
            f"The model was built by Neutron converter {built_with}, but the firmware "
            f"links driver blob {driver}. Install the matching converter:\n"
            "  pip install --index-url https://eiq.nxp.com/repository "
            f"eiq-neutron-sdk=={driver} eiq_nsys"
        )


def _read_summary(stdout: str, output_path: Path) -> NeutronResult:
    """Pull the operator count and cycle estimate out of the compiler's report."""
    operators = CONVERTED_OPERATORS_PATTERN.search(stdout)
    converted = [index for index in (operators.group(1).split(",") if operators else []) if index]

    cycles = CYCLE_ESTIMATE_PATTERN.search(stdout)

    return NeutronResult(
        output_path=output_path,
        converted_operator_count=len(converted),
        estimated_cycles=int(cycles.group(1)) if cycles else None,
        output_bytes=output_path.stat().st_size,
    )
