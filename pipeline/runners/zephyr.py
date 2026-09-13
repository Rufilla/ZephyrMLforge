# _____________________________________________________________________________
#
# @file zephyr.py
# @brief Shared Zephyr build behaviour for the QEMU and hardware runners
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Shared Zephyr build behaviour for the QEMU and hardware runners.

Both targets generate the same application, invoke west the same way and read memory
usage out of the same build output; only the board and the way results are collected
differ.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

from ..core.config import PipelineConfig
from .base import BaseRunner, BuildResult
from .firmware import FirmwareSource

logger = logging.getLogger(__name__)

# Zephyr's end-of-build summary, e.g. 'FLASH:      27700 B         2 MB      1.32%'.
MEMORY_REGION_PATTERN = re.compile(
    r"^\s*(?P<region>[\w_]+):\s+(?P<used>\d+)\s*(?P<unit>B|KB|MB)\s", re.MULTILINE
)

UNIT_MULTIPLIERS = {"B": 1, "KB": 1024, "MB": 1024 * 1024}

# Region names Zephyr uses for each kind of memory, lower-cased.
FLASH_REGION_NAMES = ("flash", "rom", "code")
RAM_REGION_NAMES = ("ram", "sram", "dram")


class ZephyrRunner(BaseRunner):
    """Builds the generated application with west; subclasses decide how it is run."""

    # QEMU has no channel to the guest console, so only hardware serves host requests.
    is_interactive = False

    # Set by the caller when the board has an accelerator and the run asked for it.
    uses_neutron = False

    def __init__(
        self,
        config: PipelineConfig,
        zephyr_base: Path | None = None,
        firmware: FirmwareSource | None = None,
    ):
        self.config = config
        self.zephyr_base = zephyr_base or Path(os.environ.get("ZEPHYR_BASE", ""))
        self.firmware = firmware or FirmwareSource()

        self._shared_build_dir: Path | None = None
        self._is_first_build_done = False

        # What the host interpreter scores. Distinct from what the firmware embeds: a
        # Neutron-compiled model carries a custom operator the host cannot resolve.
        self._current_tflite_path: Path | None = None

    @property
    def board(self) -> str:
        """Board passed to west build."""
        raise NotImplementedError

    @property
    def build_identifier(self) -> str:
        """Board name reduced to a directory name.

        Distinct per runner: in auto mode the QEMU and hardware runners are both live,
        and one shared directory would have them wiping each other's build.
        """
        return self.board.replace("/", "_")

    @property
    def west_build_dir(self) -> Path:
        """Directory west writes into, below the shared build directory."""
        if self._shared_build_dir is None:
            raise RuntimeError("build() has not run yet")
        return self._shared_build_dir / "build"

    @property
    def app_dir(self) -> Path:
        """Generated application sources."""
        if self._shared_build_dir is None:
            raise RuntimeError("build() has not run yet")
        return self._shared_build_dir / "app"

    def build(
        self,
        tflite_path: Path,
        build_dir: Path,
        host_model_path: Path | None = None,
    ) -> BuildResult:
        """Generate the application and build it for the target board.

        Only model.cpp changes between iterations, so after the first build the tree is
        reused and ninja rebuilds one translation unit instead of the whole of TFLite
        Micro. The directory actually used is reported back in the result.
        """
        if not self.zephyr_base.exists():
            return BuildResult(
                success=False,
                error_message=f"ZEPHYR_BASE not found: {self.zephyr_base}",
            )

        if self._shared_build_dir is None:
            self._shared_build_dir = build_dir.parent / f"shared_build_{self.build_identifier}"
        self._shared_build_dir.mkdir(parents=True, exist_ok=True)

        self._current_tflite_path = host_model_path or tflite_path

        if self._is_first_build_done:
            logger.info(f"Incremental build for {self.board}")
            self.firmware.write_model(self.app_dir / "src", tflite_path)
            is_pristine = False
        else:
            logger.info(f"Initial build for {self.board}")
            self.firmware.prepare_app(
                self.app_dir,
                tflite_path,
                tensor_arena_bytes=self.config.hardware.tensor_arena_kb * 1024,
                benchmark_runs=self.config.execution_environment.benchmark_runs,
                is_interactive=self.is_interactive,
                uses_neutron=self.uses_neutron,
            )
            is_pristine = True

        try:
            result = self._run_west_build(is_pristine=is_pristine)
        except Exception as error:
            logger.exception("Build failed")
            return BuildResult(success=False, error_message=str(error))

        result.build_dir = self.west_build_dir
        result.app_dir = self.app_dir

        if not result.success:
            return result

        self._is_first_build_done = True
        result.flash_usage_kb, result.ram_usage_kb = self._parse_memory_usage(result.build_log)

        return result

    def _run_west_build(self, is_pristine: bool) -> BuildResult:
        """Invoke west build for the configured board."""
        env = os.environ.copy()
        env["ZEPHYR_BASE"] = str(self.zephyr_base)

        command = [
            "west",
            "build",
            "-b",
            self.board,
            "-d",
            str(self.west_build_dir),
            str(self.app_dir),
        ]
        if is_pristine:
            command.append("--pristine")

        if self.uses_neutron:
            # Pulls in the NPU driver and the TFLM integration from hal_nxp.
            command += ["--", "-DCONFIG_TFLM_NXP_NEUTRON=y"]

        timeout_s = self.config.execution_environment.build_timeout_s
        logger.info(f"Running: {' '.join(command)}")

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except FileNotFoundError:
            return BuildResult(
                success=False,
                error_message="west not found on PATH; activate the project environment",
            )
        except subprocess.TimeoutExpired:
            return BuildResult(
                success=False,
                error_message=f"Build timed out after {timeout_s}s",
            )

        build_log = completed.stdout + completed.stderr

        if completed.returncode != 0:
            return BuildResult(
                success=False,
                error_message=completed.stderr or completed.stdout,
                build_log=build_log,
            )

        return BuildResult(success=True, build_log=build_log)

    def _parse_memory_usage(self, build_log: str) -> tuple[int | None, int | None]:
        """Read flash and RAM use in KB from the build output.

        Zephyr's own end-of-build summary is authoritative and covers every board; the
        ELF program headers are only consulted when it is absent, as classifying
        segments by address needs per-SoC knowledge the summary already has.
        """
        flash_kb, ram_kb = self._parse_memory_regions(build_log)
        if flash_kb is not None or ram_kb is not None:
            return flash_kb, ram_kb

        stat_file = self.west_build_dir / "zephyr" / "zephyr.stat"
        if stat_file.exists():
            try:
                flash_bytes, ram_bytes = self._parse_stat_file(stat_file.read_text())
            except OSError as error:
                logger.warning(f"Could not read {stat_file}: {error}")
            else:
                if flash_bytes is not None or ram_bytes is not None:
                    return (
                        flash_bytes // 1024 if flash_bytes else None,
                        ram_bytes // 1024 if ram_bytes else None,
                    )

        logger.warning("No memory usage found in the build output")
        return None, None

    @staticmethod
    def _parse_memory_regions(build_log: str) -> tuple[int | None, int | None]:
        """Parse the 'Memory region  Used Size' table west prints after a link."""
        flash_bytes = None
        ram_bytes = None

        for match in MEMORY_REGION_PATTERN.finditer(build_log):
            region = match.group("region").lower()
            used_bytes = int(match.group("used")) * UNIT_MULTIPLIERS[match.group("unit")]

            if any(name in region for name in FLASH_REGION_NAMES):
                flash_bytes = (flash_bytes or 0) + used_bytes
            elif any(name in region for name in RAM_REGION_NAMES):
                ram_bytes = (ram_bytes or 0) + used_bytes

        return (
            flash_bytes // 1024 if flash_bytes is not None else None,
            ram_bytes // 1024 if ram_bytes is not None else None,
        )

    @staticmethod
    def _parse_stat_file(content: str) -> tuple[int | None, int | None]:
        """Sum LOAD segments from readelf output in zephyr.stat.

        A segment whose physical and virtual addresses differ is initialised data: it
        occupies flash in the image and RAM at run time, so it counts towards both.
        Otherwise an executable segment is flash and the rest is RAM.
        """
        flash_bytes = 0
        ram_bytes = 0
        is_in_program_headers = False

        for line in content.splitlines():
            if "Program Headers:" in line:
                is_in_program_headers = True
                continue

            if not is_in_program_headers:
                continue

            if "Section to Segment" in line:
                break

            match = re.match(
                r"\s*LOAD\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+"
                r"0x([0-9a-f]+)\s+0x([0-9a-f]+)\s+(?P<flags>[RWE ]+)",
                line,
                re.I,
            )
            if not match:
                continue

            virtual_address = int(match.group(2), 16)
            physical_address = int(match.group(3), 16)
            file_size = int(match.group(4), 16)
            memory_size = int(match.group(5), 16)
            is_executable = "E" in match.group("flags")

            if virtual_address != physical_address:
                flash_bytes += file_size
                ram_bytes += memory_size
            elif is_executable:
                flash_bytes += file_size
            else:
                ram_bytes += memory_size

        if flash_bytes == 0 and ram_bytes == 0:
            return None, None

        logger.info(f"Memory usage from zephyr.stat: flash={flash_bytes} B, RAM={ram_bytes} B")
        return flash_bytes or None, ram_bytes or None
