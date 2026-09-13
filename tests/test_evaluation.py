"""Scoring of predictions against labels."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.core.config import AccuracyMetric
from pipeline.models.evaluation import MetricScore, score_predictions


def test_scores_column_predictions_against_flat_labels():
    predictions = np.array([[1.0], [2.0], [3.0]])
    labels = np.array([1.5, 2.5, 3.5])

    # Subtracting these as given broadcasts to a 3x3 matrix and yields 1.5833.
    assert score_predictions(predictions, labels, AccuracyMetric.MSE).value == pytest.approx(0.25)


def test_scores_matching_shapes_without_reshaping():
    predictions = np.array([[1.0], [2.0]])
    labels = np.array([[2.0], [4.0]])

    assert score_predictions(predictions, labels, AccuracyMetric.MSE).value == pytest.approx(2.5)


def test_mean_absolute_error_uses_absolute_differences():
    predictions = np.array([[1.0], [5.0]])
    labels = np.array([[2.0], [2.0]])

    assert score_predictions(predictions, labels, AccuracyMetric.MAE).value == pytest.approx(2.0)


def test_accuracy_takes_the_highest_scoring_class():
    predictions = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    labels = np.array([0, 1, 1])

    score = score_predictions(predictions, labels, AccuracyMetric.ACCURACY)

    assert score.value == pytest.approx(2 / 3)


def test_accuracy_thresholds_a_single_sigmoid_output():
    predictions = np.array([[0.9], [0.2], [0.6]])
    labels = np.array([1, 0, 0])

    score = score_predictions(predictions, labels, AccuracyMetric.ACCURACY)

    assert score.value == pytest.approx(2 / 3)


def test_accuracy_accepts_one_hot_labels():
    predictions = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([[1, 0], [0, 1]])

    score = score_predictions(predictions, labels, AccuracyMetric.ACCURACY)

    assert score.value == pytest.approx(1.0)


def test_rejects_a_prediction_count_that_differs_from_the_labels():
    with pytest.raises(ValueError, match="does not match label count"):
        score_predictions(np.array([[1.0], [2.0]]), np.array([1.0]), AccuracyMetric.MSE)


def test_rejects_prediction_width_that_differs_from_the_labels():
    predictions = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([[1.0], [3.0]])

    with pytest.raises(ValueError, match="does not match label shape"):
        score_predictions(predictions, labels, AccuracyMetric.MSE)


def test_standard_error_of_an_accuracy_follows_the_sample_count():
    predictions = np.array([[0.9, 0.1]] * 100)
    labels = np.array([0] * 90 + [1] * 10)

    score = score_predictions(predictions, labels, AccuracyMetric.ACCURACY)

    # sqrt(p(1-p)/n) for p=0.9 over 100 samples, give or take the ddof=1 correction.
    assert score.value == pytest.approx(0.9)
    assert score.standard_error == pytest.approx(0.03, abs=0.002)
    assert score.sample_count == 100


def test_standard_error_of_an_error_metric_is_the_spread_between_samples():
    predictions = np.array([[1.0], [3.0], [5.0], [7.0]])
    labels = np.array([[1.0], [1.0], [1.0], [1.0]])

    score = score_predictions(predictions, labels, AccuracyMetric.MSE)

    # Per-sample squared errors 0, 4, 16, 36: mean 14, sample deviation 15.8 over 4.
    assert score.value == pytest.approx(14.0)
    assert score.standard_error == pytest.approx(np.std([0, 4, 16, 36], ddof=1) / 2)


def test_a_single_sample_reports_no_spread_rather_than_nan():
    score = score_predictions(np.array([[1.0]]), np.array([[2.0]]), AccuracyMetric.MSE)

    assert score.standard_error == 0.0


def test_a_gap_inside_the_error_bar_does_not_separate_two_scores():
    best = MetricScore(value=0.872, standard_error=0.0106, sample_count=1000)
    runner_up = MetricScore(value=0.870, standard_error=0.0106, sample_count=1000)

    assert not best.separates(runner_up)


def test_a_gap_wider_than_the_error_bar_separates_two_scores():
    best = MetricScore(value=0.872, standard_error=0.0034, sample_count=10000)
    runner_up = MetricScore(value=0.850, standard_error=0.0034, sample_count=10000)

    assert best.separates(runner_up)
