""" Write trained models, & the Fashion-MNIST test images, into the apps.

Run from the project root with the project environment active, naming any of the three
models to replace; the others are left as they are:

    python apps/export_models.py --sine artifacts/iteration_0001/model.tflite
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

APPS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APPS_DIR.parent))

from pipeline.core.boards import BoardRegistry  # noqa: E402
from pipeline.models.neutron import NeutronCompilerError, compile_for_neutron  # noqa: E402
from pipeline.runners.firmware import FirmwareSource  # noqa: E402

#  The board the apps are built for; its Neutron target comes from pyproject.toml.
BOARD = "frdm_mcxn947"

FASHION_MNIST_CLASSES = (
    "t-shirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
)

# The first 100 of the Keras test split. The pipeline validated on that
# split, so these images steered early stopping but were never trained on.
TEST_IMAGE_COUNT = 100


def write_test_images(source_dir: Path) -> None:
    """Write src/test_images.cpp & .hpp; Keras downloads the set on first use."""
    import tensorflow as tf

    (_, _), (images, labels) = tf.keras.datasets.fashion_mnist.load_data()
    images, labels = images[:TEST_IMAGE_COUNT], labels[:TEST_IMAGE_COUNT]

    rows = ",\n".join(
        "\t{" + ", ".join(str(int(pixel)) for pixel in image.ravel()) + "}" for image in images
    )
    names = ", ".join(f'"{name}"' for name in FASHION_MNIST_CLASSES)

    (source_dir / "test_images.hpp").write_text(
        f"""/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 * Written by apps/export_models.py.
 */

#ifndef ZEPHYR_ML_FORGE_TEST_IMAGES_HPP_
#define ZEPHYR_ML_FORGE_TEST_IMAGES_HPP_

#include <stdint.h>

constexpr int kTestImageCount = {TEST_IMAGE_COUNT};
constexpr int kImagePixels = {images[0].size};
constexpr int kClassCount = {len(FASHION_MNIST_CLASSES)};

/*  Grayscale pixels, 0 to 255, row by row. */
extern const uint8_t g_test_images[kTestImageCount][kImagePixels];
extern const uint8_t g_test_labels[kTestImageCount];
extern const char *const g_class_names[kClassCount];

#endif /* ZEPHYR_ML_FORGE_TEST_IMAGES_HPP_ */
"""
    )
    (source_dir / "test_images.cpp").write_text(
        f"""/*
 * Copyright (c) 2024 ZephyrMLForge
 * SPDX-License-Identifier: Apache-2.0
 *
 *  Written by apps/export_models.py: the first {TEST_IMAGE_COUNT} images of
 * the Fashion-MNIST test split.
 */

#include "test_images.hpp"

const uint8_t g_test_images[kTestImageCount][kImagePixels] = {{
{rows}
}};

const uint8_t g_test_labels[kTestImageCount] = {{
\t{", ".join(str(int(label)) for label in labels)}
}};

const char *const g_class_names[kClassCount] = {{{names}}};
"""
    )
    print(f"Wrote {TEST_IMAGE_COUNT} test images, labels {np.bincount(labels).tolist()}")


def neutron_target() -> str:
    """The converter target for BOARD, from pyproject.toml."""
    target = BoardRegistry.discover(APPS_DIR).for_board(BOARD).neutron_target
    if target is None:
        raise SystemExit(f"error: pyproject.toml gives {BOARD} no neutron_target")
    return target


def zephyr_base() -> Path:
    """The Zephyr tree whose Neutron driver blob the converter's version must match."""
    if not os.environ.get("ZEPHYR_BASE"):
        raise SystemExit("error: ZEPHYR_BASE is unset; run 'source activate.sh' first")
    path = Path(os.environ["ZEPHYR_BASE"])
    if not path.is_dir():
        raise SystemExit(f"error: ZEPHYR_BASE does not exist: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sine", type=Path, help="model.tflite for apps/sine_predict")
    parser.add_argument("--fashion", type=Path, help="model.tflite for apps/fashion_mnist_test")
    parser.add_argument("--vibration", type=Path, help="model.tflite for apps/vibration_guess")
    args = parser.parse_args()

    targets = {
        "sine_predict": args.sine,
        "fashion_mnist_test": args.fashion,
        "vibration_guess": args.vibration,
    }
    if not any(targets.values()):
        parser.error("name at least one model")

    firmware = FirmwareSource()
    for app, model in targets.items():
        if model is None:
            continue
        source_dir = APPS_DIR / app / "src"
        if app == "fashion_mnist_test":
            #  Embedded compiled for the NPU, which its prj.conf enables. Compiled
            # & the images written first, so a failure in either leaves the old model in place.
            with tempfile.TemporaryDirectory() as work_dir:
                npu_model = Path(work_dir) / "model_npu.tflite"
                try:
                    compile_for_neutron(model, npu_model, neutron_target(), zephyr_base())
                except NeutronCompilerError as error:
                    raise SystemExit(f"error: {error}") from error
                write_test_images(source_dir)
                firmware.write_model(source_dir, npu_model)
            print(f"Wrote {model}, compiled for the NPU, into {source_dir}")
        else:
            firmware.write_model(source_dir, model)
            print(f"Wrote {model} into {source_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
