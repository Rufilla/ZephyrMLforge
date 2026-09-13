"""Synthetic data generation and quantization calibration."""

from __future__ import annotations

import numpy as np

from pipeline.core.config import SyntheticDataConfig
from pipeline.data.synthetic import SyntheticDataGenerator, create_representative_dataset


def build_generator(**overrides) -> SyntheticDataGenerator:
    """Build a generator over a small deterministic dataset."""
    defaults = {"task": "regression", "function": "sine", "samples": 100, "seed": 42}
    return SyntheticDataGenerator(SyntheticDataConfig(**{**defaults, **overrides}))


def test_same_seed_produces_the_same_data():
    (first_features, first_labels), _ = build_generator().generate()
    (second_features, second_labels), _ = build_generator().generate()

    assert np.array_equal(first_features, second_features)
    assert np.array_equal(first_labels, second_labels)


def test_different_seeds_produce_different_data():
    (first_features, _), _ = build_generator(seed=1).generate()
    (second_features, _), _ = build_generator(seed=2).generate()

    assert not np.array_equal(first_features, second_features)


def test_split_follows_the_configured_ratio():
    train_data, val_data = build_generator(samples=100, train_split=0.8).generate()

    assert len(train_data[0]) == 80
    assert len(val_data[0]) == 20


def test_regression_labels_are_a_column_of_floats():
    train_data, _ = build_generator().generate()

    assert train_data[0].shape[1] == 1
    assert train_data[1].shape[1] == 1


def test_classification_keeps_every_configured_sample():
    # An odd count used to lose one sample to integer division.
    train_data, val_data = build_generator(task="classification", samples=101).generate()

    assert len(train_data[0]) + len(val_data[0]) == 101


def test_classification_classes_are_separated():
    train_data, val_data = build_generator(
        task="classification", samples=200, noise_stddev=0.05
    ).generate()

    features = np.vstack([train_data[0], val_data[0]])
    labels = np.hstack([train_data[1], val_data[1]])

    # Fixed centres at (-0.5, -0.5) and (0.5, 0.5); random centres sometimes coincided
    # and left the task unlearnable.
    assert features[labels == 0].mean() < features[labels == 1].mean()


def test_calibration_subset_is_the_same_for_the_same_seed():
    features = np.arange(200, dtype=np.float32).reshape(200, 1)

    first = [batch[0] for batch in create_representative_dataset(features, 10, seed=7)()]
    second = [batch[0] for batch in create_representative_dataset(features, 10, seed=7)()]

    assert np.array_equal(np.array(first), np.array(second))


def test_calibration_subset_honours_the_requested_size():
    features = np.arange(200, dtype=np.float32).reshape(200, 1)

    batches = list(create_representative_dataset(features, 10, seed=7)())

    assert len(batches) == 10
    assert batches[0][0].shape == (1, 1)


def test_calibration_subset_is_capped_by_the_dataset_size():
    features = np.arange(5, dtype=np.float32).reshape(5, 1)

    assert len(list(create_representative_dataset(features, 100, seed=7)())) == 5
