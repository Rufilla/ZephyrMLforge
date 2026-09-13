# _____________________________________________________________________________
#
# @file synthetic.py
# @brief Synthetic data generation module
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Synthetic data generation module.

Generates deterministic synthetic datasets for pipeline testing
and validation without external dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

import numpy as np

from ..core.config import SyntheticDataConfig, SyntheticFunction, SyntheticTask

logger = logging.getLogger(__name__)

# Type alias for data tuple: (features, labels)
DataTuple = tuple[np.ndarray, np.ndarray]


class SyntheticDataGenerator:
    """Generator for synthetic training and validation data."""

    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def generate(self) -> tuple[DataTuple, DataTuple]:
        """
        Generate synthetic training and validation data.

        Returns:
            Tuple of (train_data, val_data) where each is (features, labels)
        """
        logger.info(
            f"Generating {self.config.samples} samples for {self.config.task.value} task"
        )

        if self.config.task == SyntheticTask.REGRESSION:
            features, labels = self._generate_regression_data()
        else:
            features, labels = self._generate_classification_data()

        # Split into train/validation
        split_idx = int(len(features) * self.config.train_split)

        # Shuffle before splitting
        indices = self._rng.permutation(len(features))
        features = features[indices]
        labels = labels[indices]

        train_data = (features[:split_idx], labels[:split_idx])
        val_data = (features[split_idx:], labels[split_idx:])

        logger.info(f"Train samples: {len(train_data[0])}, Val samples: {len(val_data[0])}")

        return train_data, val_data

    def _generate_regression_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate regression data based on configured function."""
        func_map: dict[SyntheticFunction, Callable[[np.ndarray], np.ndarray]] = {
            SyntheticFunction.SINE: self._sine_function,
            SyntheticFunction.QUADRATIC: self._quadratic_function,
            SyntheticFunction.STEP: self._step_function,
        }

        func = func_map[self.config.function]

        # Generate input features in [-pi, pi] range for sine, [0, 1] for others
        if self.config.function == SyntheticFunction.SINE:
            x = self._rng.uniform(-np.pi, np.pi, size=(self.config.samples, 1))
        else:
            x = self._rng.uniform(0, 1, size=(self.config.samples, 1))

        # Generate clean labels
        y = func(x)

        # Add noise
        noise = self._rng.normal(0, self.config.noise_stddev, size=y.shape)
        y_noisy = y + noise

        return x.astype(np.float32), y_noisy.astype(np.float32)

    def _generate_classification_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate classification data with separable clusters."""
        n_classes = 2  # Binary classification

        # An odd sample count would otherwise silently produce one sample fewer than
        # configured, which matters when the count is small.
        class_sizes = [
            self.config.samples // n_classes + (1 if index < self.config.samples % n_classes else 0)
            for index in range(n_classes)
        ]

        features_list = []
        labels_list = []

        # Fixed, separated centres: drawing them at random sometimes placed the two
        # classes on top of each other, and no model could then reach an accuracy
        # target. Overlap is controlled by noise_stddev, as the specification intends.
        centers = [(-0.5, -0.5), (0.5, 0.5)]

        for class_idx in range(n_classes):
            samples_per_class = class_sizes[class_idx]
            center = np.array(centers[class_idx])

            # Generate points around center
            points = self._rng.normal(
                loc=center,
                scale=0.3,  # Cluster spread
                size=(samples_per_class, 2),
            )

            # Add noise
            noise = self._rng.normal(0, self.config.noise_stddev, size=points.shape)
            points = points + noise

            features_list.append(points)
            labels_list.append(np.full(samples_per_class, class_idx))

        features = np.vstack(features_list).astype(np.float32)
        labels = np.hstack(labels_list).astype(np.int32)

        return features, labels

    @staticmethod
    def _sine_function(x: np.ndarray) -> np.ndarray:
        """Sine wave function: y = sin(x)."""
        return np.sin(x)

    @staticmethod
    def _quadratic_function(x: np.ndarray) -> np.ndarray:
        """Quadratic function: y = x^2."""
        return x ** 2

    @staticmethod
    def _step_function(x: np.ndarray) -> np.ndarray:
        """Step function: y = 1 if x > 0.5 else 0."""
        return (x > 0.5).astype(np.float32)

    def get_input_shape(self) -> tuple[int, ...]:
        """Get the input feature shape for model definition."""
        if self.config.task == SyntheticTask.REGRESSION:
            return (1,)  # Single feature
        else:
            return (2,)  # 2D features for classification

    def get_output_shape(self) -> tuple[int, ...]:
        """Get the output shape for model definition."""
        if self.config.task == SyntheticTask.REGRESSION:
            return (1,)  # Single output
        else:
            return (1,)  # Binary classification (sigmoid output)


def create_representative_dataset(
    features: np.ndarray, num_samples: int = 100, seed: int = 42
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """
    Create a representative dataset generator for TFLite quantization.

    Args:
        features: Full feature dataset
        num_samples: Number of samples to use for calibration
        seed: Seed for the calibration subset

    Returns:
        Generator function for the TFLite converter, yielding one batch per sample
    """
    # Seeded: the calibration subset fixes the quantization ranges, so drawing it from
    # the global RNG made an otherwise deterministic run produce different models.
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), min(num_samples, len(features)), replace=False)
    calibration_data = features[indices]

    def representative_dataset():
        for sample in calibration_data:
            yield [np.expand_dims(sample, axis=0).astype(np.float32)]

    return representative_dataset
