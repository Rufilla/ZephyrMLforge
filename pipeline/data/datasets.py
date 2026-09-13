# _____________________________________________________________________________
#
# @file datasets.py
# @brief Image datasets for classification runs
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Image datasets for classification runs.

Fashion-MNIST and MNIST share a shape, 28x28 greyscale over ten classes, and both are
fetched by Keras with no account or API token. Fashion-MNIST is the better subject for
a search: a small convolutional network reaches about 99% on MNIST immediately, leaving
the agent nothing to improve, where Fashion-MNIST sits nearer 92%.

Only a subset is used. The full 60,000 images make each iteration slow enough to change
how the pipeline feels to run, and the point here is architecture search rather than the
last point of accuracy.
"""

from __future__ import annotations

import logging

import numpy as np

from ..core.config import DatasetConfig, DatasetSource
from ..core.output import import_tensorflow

logger = logging.getLogger(__name__)

DataTuple = tuple[np.ndarray, np.ndarray]

# Greyscale, so one channel; the trailing dimension is what a Conv2D expects.
IMAGE_SHAPE = (28, 28, 1)

# Highest value a pixel takes before scaling.
PIXEL_MAXIMUM = 255.0

CLASS_NAMES = {
    DatasetSource.FASHION_MNIST: (
        "t-shirt", "trouser", "pullover", "dress", "coat",
        "sandal", "shirt", "sneaker", "bag", "ankle boot",
    ),
    DatasetSource.MNIST: tuple(str(digit) for digit in range(10)),
}


class ImageDataset:
    """Loads a Keras image dataset and hands back a train and validation split."""

    def __init__(self, config: DatasetConfig, seed: int = 42):
        if not config.source.is_image:
            raise ValueError(f"{config.source.value} is not an image dataset")

        self.config = config
        self.seed = seed

    @property
    def class_names(self) -> tuple[str, ...]:
        """Human-readable labels, for reports."""
        return CLASS_NAMES[self.config.source]

    def generate(self) -> tuple[DataTuple, DataTuple]:
        """Load the dataset and draw the configured subsets.

        Returns:
            (train_features, train_labels), (val_features, val_labels), with features
            scaled to [0, 1] and shaped (samples, 28, 28, 1).

        Raises:
            RuntimeError: TensorFlow is not installed, or the download failed.
        """
        tf = import_tensorflow()

        loaders = {
            DatasetSource.FASHION_MNIST: tf.keras.datasets.fashion_mnist,
            DatasetSource.MNIST: tf.keras.datasets.mnist,
        }

        logger.info(f"Loading {self.config.source.value}")
        try:
            (train_images, train_labels), (val_images, val_labels) = loaders[
                self.config.source
            ].load_data()
        except Exception as error:
            raise RuntimeError(
                f"Could not load {self.config.source.value}: {error}. The first run "
                "downloads it to ~/.keras/datasets and needs network access."
            ) from error

        rng = np.random.default_rng(self.seed)
        train = self._subset(rng, train_images, train_labels, self.config.train_samples)
        validation = self._subset(rng, val_images, val_labels, self.config.val_samples)

        logger.info(
            f"Loaded {len(train[0])} training and {len(validation[0])} validation images "
            f"of {IMAGE_SHAPE}"
        )
        return train, validation

    @staticmethod
    def _subset(
        rng: np.random.Generator,
        images: np.ndarray,
        labels: np.ndarray,
        count: int,
    ) -> DataTuple:
        """Draw a reproducible subset, scaled and shaped for a convolutional model."""
        available = min(count, len(images))
        indices = rng.choice(len(images), available, replace=False)

        features = images[indices].astype(np.float32) / PIXEL_MAXIMUM
        features = features.reshape(available, *IMAGE_SHAPE)

        return features, labels[indices].astype(np.int32)


def create_data_source(
    dataset: DatasetConfig,
    synthetic_generator,
    seed: int = 42,
):
    """Return the generator for the configured dataset.

    The synthetic generator is passed in rather than built here, so that the synthetic
    path keeps its own configuration and this module stays about real data.
    """
    if dataset.source.is_image:
        return ImageDataset(dataset, seed=seed)
    return synthetic_generator
