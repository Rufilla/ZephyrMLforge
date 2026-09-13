# _____________________________________________________________________________
#
# @file firmware.py
# @brief Generation of the Zephyr application tree from zephyr_app/
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Generation of the Zephyr application tree from zephyr_app/.

The firmware has one source, the ``zephyr_app`` directory at the repository root. A
run copies that tree and overwrites three generated files in the copy, so a pipeline
build and a manual ``west build zephyr_app`` compile the same code.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# pipeline/runners/firmware.py -> pipeline/runners -> pipeline -> repository root
DEFAULT_APP_SOURCE = Path(__file__).resolve().parents[2] / "zephyr_app"

# Not needed to build, and copying them invites edits to the copy instead of the source.
EXCLUDED_FROM_COPY = shutil.ignore_patterns("build", "build.*", "README.md", "__pycache__")

# Bytes of model data per generated source line.
MODEL_BYTES_PER_LINE = 12


class FirmwareSource:
    """Materialises a buildable Zephyr application for one model."""

    def __init__(self, app_source: Path | None = None):
        self.app_source = app_source or DEFAULT_APP_SOURCE

    def prepare_app(
        self,
        app_dir: Path,
        tflite_path: Path,
        tensor_arena_bytes: int,
        benchmark_runs: int,
        is_interactive: bool,
        uses_neutron: bool = False,
    ) -> None:
        """Copy the application tree into ``app_dir`` and write the generated files.

        Raises:
            FileNotFoundError: the zephyr_app source tree is missing, which happens when
                the package was installed non-editable and the tree was left behind.
        """
        if not self.app_source.is_dir():
            raise FileNotFoundError(
                f"Firmware source tree not found at {self.app_source}. "
                "Install the project editable ('pip install -e .') so it stays on disk."
            )

        app_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.app_source, app_dir, dirs_exist_ok=True, ignore=EXCLUDED_FROM_COPY)

        source_dir = app_dir / "src"
        self.write_app_config(
            source_dir, tensor_arena_bytes, benchmark_runs, is_interactive, uses_neutron
        )
        self.write_model(source_dir, tflite_path)

        logger.info(f"Prepared Zephyr application in {app_dir} from {self.app_source}")

    def write_app_config(
        self,
        source_dir: Path,
        tensor_arena_bytes: int,
        benchmark_runs: int,
        is_interactive: bool,
        uses_neutron: bool = False,
    ) -> None:
        """Write src/app_config.h with the values taken from the pipeline configuration."""
        content = f"""/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Written by the pipeline from the run configuration. Edit zephyr_app/src/app_config.h
 * for the checked-in defaults; edits here are overwritten on the next iteration.
 */

#ifndef ZEPHYR_ML_FORGE_APP_CONFIG_H_
#define ZEPHYR_ML_FORGE_APP_CONFIG_H_

#define MLFORGE_TENSOR_ARENA_SIZE {tensor_arena_bytes}
#define MLFORGE_BENCHMARK_RUNS {benchmark_runs}
#define MLFORGE_INTERACTIVE {1 if is_interactive else 0}
#define MLFORGE_NEUTRON {1 if uses_neutron else 0}

#endif /* ZEPHYR_ML_FORGE_APP_CONFIG_H_ */
"""
        (source_dir / "app_config.h").write_text(content)

    def write_model(self, source_dir: Path, tflite_path: Path) -> None:
        """Write src/model.cpp and src/model.hpp from the quantized model.

        This is the only file that changes between iterations, which is what makes the
        incremental build worthwhile.
        """
        model_bytes = tflite_path.read_bytes()

        lines = []
        for offset in range(0, len(model_bytes), MODEL_BYTES_PER_LINE):
            chunk = model_bytes[offset : offset + MODEL_BYTES_PER_LINE]
            lines.append("\t" + ", ".join(f"0x{byte:02x}" for byte in chunk))
        body = ",\n".join(lines)

        header = """/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Model data declarations. Written by the pipeline from the trained .tflite file and
 * retained as a build artefact for inspection or reuse.
 */

#ifndef ZEPHYR_ML_FORGE_MODEL_HPP_
#define ZEPHYR_ML_FORGE_MODEL_HPP_

extern const unsigned char g_model[];
extern const unsigned int g_model_len;

#endif /* ZEPHYR_ML_FORGE_MODEL_HPP_ */
"""
        (source_dir / "model.hpp").write_text(header)

        source = f"""/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Model data written by the pipeline from the trained .tflite file.
 * Model size: {len(model_bytes)} bytes
 */

#include "model.hpp"

/* TFLite Micro reads the flatbuffer in place and requires 8-byte alignment. */
alignas(8) const unsigned char g_model[] = {{
{body}
}};

const unsigned int g_model_len = {len(model_bytes)};
"""
        (source_dir / "model.cpp").write_text(source)

        logger.info(f"Generated model.cpp and model.hpp ({len(model_bytes)} bytes)")
