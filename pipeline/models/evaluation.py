# _____________________________________________________________________________
#
# @file evaluation.py
# @brief Scoring of TFLite models against a validation set
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Scoring of TFLite models against a validation set.

Both the host interpreter and the on-target runners produce predictions in the same
shape, so scoring lives here and every execution path is measured the same way.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config import AccuracyMetric
from ..core.output import import_tensorflow

logger = logging.getLogger(__name__)


@dataclass
class MetricScore:
    """A metric value, and how precisely this validation set could measure it.

    Every metric here is a mean over samples, so its own sampling error is the standard
    error of that mean. Two models whose scores differ by less than it have not been
    told apart, however many decimal places the figures are printed to.
    """

    value: float
    standard_error: float
    sample_count: int

    def __str__(self) -> str:
        return f"{self.value:.6f} ± {self.standard_error:.6f} on {self.sample_count} samples"

    def separates(self, other: "MetricScore") -> bool:
        """True when the gap to another score is wider than this one's own error bar.

        The yardstick is one standard error rather than the two combined: both scores
        are measured on the same validation set, so their errors move together, and
        adding them in quadrature would demand more of a gap than the data warrants.
        """
        return abs(self.value - other.value) > self.standard_error


def score_predictions(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric: AccuracyMetric,
) -> MetricScore:
    """Score predictions against labels under the configured metric.

    Args:
        predictions: Model outputs, shape (samples,) or (samples, outputs).
        labels: Ground truth, shape (samples,), (samples, 1) or one-hot (samples, classes).
        metric: Metric to apply.

    Returns:
        The metric value and its standard error. Lower is better except for ``accuracy``.

    Raises:
        ValueError: prediction and label shapes disagree after normalisation.
    """
    per_sample = _per_sample_scores(predictions, labels, metric)

    return MetricScore(
        value=float(np.mean(per_sample)),
        standard_error=_standard_error(per_sample),
        sample_count=len(per_sample),
    )


def _standard_error(per_sample: np.ndarray) -> float:
    """Standard error of the mean of one score per sample.

    A single sample says nothing about spread, so its error is reported as zero rather
    than as the nan numpy would return for it.
    """
    if len(per_sample) < 2:
        return 0.0

    return float(np.std(per_sample, ddof=1) / math.sqrt(len(per_sample)))


def _per_sample_scores(
    predictions: np.ndarray,
    labels: np.ndarray,
    metric: AccuracyMetric,
) -> np.ndarray:
    """One score per sample, whose mean is the metric.

    Raises:
        ValueError: prediction and label shapes disagree after normalisation.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if len(predictions) != len(labels):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match label count {len(labels)}"
        )

    if metric is AccuracyMetric.ACCURACY:
        return (_to_class_index(predictions) == _to_class_index(labels)).astype(np.float64)

    # Reshape rather than subtract as given: (N,) labels against (N, 1) predictions
    # broadcast to an (N, N) matrix and yield a plausible but meaningless figure.
    predictions = predictions.reshape(len(predictions), -1)
    labels = labels.reshape(len(labels), -1)

    if predictions.shape != labels.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match label shape {labels.shape}"
        )

    error = predictions.astype(np.float64) - labels.astype(np.float64)

    # Averaged over each sample's outputs first, so what is left is one figure per
    # sample and the spread the standard error is taken over is the spread between
    # samples, which is what drawing a different validation set would change.
    if metric is AccuracyMetric.MAE:
        return np.mean(np.abs(error), axis=1)
    return np.mean(np.square(error), axis=1)


def _to_class_index(values: np.ndarray) -> np.ndarray:
    """Reduce scores, one-hot rows or a sigmoid column to one class index per sample."""
    values = np.asarray(values)

    if values.ndim == 1:
        return values.astype(np.int64)

    if values.shape[-1] == 1:
        # Single sigmoid output: the decision boundary is the midpoint.
        return (values.reshape(len(values)) >= 0.5).astype(np.int64)

    return values.reshape(len(values), -1).argmax(axis=-1).astype(np.int64)


def predict_tflite(tflite_path: Path, features: np.ndarray) -> np.ndarray:
    """Run every feature row through the quantized model on the host interpreter.

    The whole set goes through in one invocation where the model's input tensor can be
    resized to hold it, which is some twenty times faster than a sample at a time and is
    what makes a validation set large enough to rank models affordable. A graph that
    refuses the resize is run a sample at a time instead.

    Returns:
        Dequantised outputs, shape (samples, outputs).

    Raises:
        RuntimeError: TensorFlow is not installed.
    """
    features = np.asarray(features, dtype=np.float32)
    if len(features) == 0:
        return np.empty((0, 0), dtype=np.float32)

    interpreter, batch_size = _batched_interpreter(tflite_path, len(features))

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    predictions = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]

        interpreter.set_tensor(input_detail["index"], _quantize(batch, input_detail))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        predictions.append(_dequantize(output, output_detail))

    return np.concatenate(predictions, axis=0)


def _batched_interpreter(tflite_path: Path, sample_count: int):
    """An allocated interpreter, and how many samples one invocation of it takes.

    Raises:
        RuntimeError: TensorFlow is not installed.
    """
    interpreter = _make_interpreter(tflite_path)
    input_detail = interpreter.get_input_details()[0]

    try:
        interpreter.resize_tensor_input(
            input_detail["index"], [sample_count, *input_detail["shape"][1:]]
        )
        interpreter.allocate_tensors()
    except (RuntimeError, ValueError) as error:
        # A failed resize leaves the interpreter half-configured, so the fallback starts
        # from a fresh one rather than from whatever state the attempt left behind.
        logger.debug(f"Batched evaluation unavailable, scoring one sample at a time: {error}")
        interpreter = _make_interpreter(tflite_path)
        interpreter.allocate_tensors()
        return interpreter, 1

    return interpreter, sample_count


def _make_interpreter(tflite_path: Path):
    """Build an interpreter over the model, without allocating its tensors.

    Raises:
        RuntimeError: TensorFlow is not installed.
    """
    tf = import_tensorflow()

    # Without default delegates: XNNPACK refuses to prepare some fully int8 graphs and
    # fails the whole evaluation, and the reference kernels are the closer match to
    # TFLite Micro on the target, which has no delegate at all.
    return tf.lite.Interpreter(
        model_path=str(tflite_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )


def _quantize(batch: np.ndarray, input_detail: dict) -> np.ndarray:
    """Convert a float batch into whatever dtype the model's input tensor expects."""
    if input_detail["dtype"] != np.int8:
        return batch

    scale, zero_point = input_detail["quantization"]
    return np.clip(np.round(batch / scale + zero_point), -128, 127).astype(np.int8)


def _dequantize(output: np.ndarray, output_detail: dict) -> np.ndarray:
    """Convert a model output batch back to floats."""
    if output_detail["dtype"] != np.int8:
        return output

    scale, zero_point = output_detail["quantization"]
    return (output.astype(np.float32) - zero_point) * scale


def evaluate_tflite(
    tflite_path: Path,
    test_data: tuple[np.ndarray, np.ndarray],
    metric: AccuracyMetric,
) -> MetricScore:
    """Score a TFLite model on the host interpreter.

    Raises:
        RuntimeError: TensorFlow is not installed.
        ValueError: prediction and label shapes disagree.
    """
    features, labels = test_data
    predictions = predict_tflite(tflite_path, features)
    score = score_predictions(predictions, labels, metric)

    logger.info(f"Host evaluation: {metric.value}={score}")
    return score
