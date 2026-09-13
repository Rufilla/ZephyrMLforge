# _____________________________________________________________________________
#
# @file trainer.py
# @brief Model training and TFLite conversion module
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Model training and TFLite conversion module.

Handles dynamic model creation from AI-generated code, training, and conversion to
TFLite format for embedded deployment.

Security note: ``_create_model_from_code`` executes code returned by a third-party API
in this process, with the privileges of the user running the pipeline. The guardrails
here bound model *size*, not what the code may do. Run the pipeline only with an agent
provider and API key you trust.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import QuantizationConfig, QuantizationType, TrainingConfig
from ..core.output import import_tensorflow, suppressed_native_output
from ..core.pipeline import WeightMetadata
from ..data.synthetic import create_representative_dataset

logger = logging.getLogger(__name__)

# Lazy import TensorFlow to avoid startup overhead
tf = None


def _ensure_tf():
    """Import TensorFlow on first use and apply the configured verbosity."""
    global tf
    if tf is None:
        tf = import_tensorflow()


@dataclass
class WeightReuseResult:
    """Result of a weight reuse attempt for a single layer."""

    layer_name: str
    success: bool
    reason: str = ""  # Empty if success, description of incompatibility if failed


@dataclass
class TrainingMetrics:
    """Metrics from model training."""

    final_loss: float
    final_val_loss: float
    epochs_trained: int
    history: dict[str, list[float]]
    weight_reuse_results: list[WeightReuseResult] = field(default_factory=list)


class ModelTrainer:
    """Handles model training and conversion."""

    def __init__(self, training: TrainingConfig, seed: int = 42):
        self.training = training
        self.seed = seed

    def train(
        self,
        model_code: str,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        weight_reuse_layers: list[str] | None = None,
        previous_weights_path: Path | None = None,
    ) -> tuple[Any, TrainingMetrics]:
        """
        Train a model from AI-generated code.

        Args:
            model_code: Python code defining the model
            train_data: Tuple of (features, labels) for training
            val_data: Tuple of (features, labels) for validation
            weight_reuse_layers: Layer names to copy weights from previous model
            previous_weights_path: Path to previous saved model

        Returns:
            Tuple of (trained model, training metrics)

        Raises:
            ValueError: the generated code is invalid or the model breaches a guardrail
        """
        _ensure_tf()

        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)

        model = self._create_model_from_code(model_code, train_data[0].shape[1:])

        weight_reuse_results: list[WeightReuseResult] = []
        if weight_reuse_layers and previous_weights_path:
            weight_reuse_results = self._reuse_weights(
                model, weight_reuse_layers, previous_weights_path
            )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.training.early_stopping_patience,
                restore_best_weights=True,
            ),
        ]

        logger.info(f"Training for up to {self.training.epochs} epochs")
        history = model.fit(
            train_data[0],
            train_data[1],
            validation_data=val_data,
            epochs=self.training.epochs,
            batch_size=self.training.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        metrics = TrainingMetrics(
            final_loss=float(history.history["loss"][-1]),
            final_val_loss=float(history.history["val_loss"][-1]),
            epochs_trained=len(history.history["loss"]),
            history={key: [float(v) for v in values] for key, values in history.history.items()},
            weight_reuse_results=weight_reuse_results,
        )

        logger.info(
            f"Training complete: {metrics.epochs_trained} epochs, "
            f"final val_loss={metrics.final_val_loss:.6f}"
        )

        if metrics.epochs_trained >= self.training.epochs:
            # Early stopping never fired, so the budget ended training rather than
            # convergence. Comparing such iterations ranks how fast an architecture
            # trains as much as how good it is, and penalises the deeper ones.
            logger.warning(
                f"Training stopped at the {self.training.epochs}-epoch cap without "
                "converging; this model is under-trained, and comparisons with it "
                "reflect training budget as much as architecture. Raise training.epochs."
            )

        return model, metrics

    def _create_model_from_code(self, model_code: str, input_shape: tuple) -> Any:
        """
        Create a Keras model from AI-generated code.

        The code is executed in this process; see the module docstring.

        Raises:
            ValueError: the code fails to execute, defines no create_model, returns
                something other than a Keras model, or breaches a size guardrail.
        """
        _ensure_tf()

        namespace = {"tf": tf, "np": np}

        try:
            exec(model_code, namespace)
        except Exception as error:
            logger.error(f"Failed to execute model code: {error}")
            raise ValueError(f"Invalid model code: {error}") from error

        if "create_model" not in namespace:
            raise ValueError("Model code must define a 'create_model(input_shape)' function")

        try:
            model = namespace["create_model"](input_shape)
        except Exception as error:
            logger.error(f"Failed to create model: {error}")
            raise ValueError(f"Model creation failed: {error}") from error

        if not isinstance(model, tf.keras.Model):
            raise ValueError("create_model must return a tf.keras.Model instance")

        parameter_count = model.count_params()
        layer_count = len(model.layers)

        if parameter_count > self.training.max_parameters:
            raise ValueError(
                f"Model exceeds parameter limit: {parameter_count} > "
                f"{self.training.max_parameters}. Reduce layer sizes or number of layers."
            )

        if layer_count > self.training.max_layers:
            raise ValueError(
                f"Model exceeds layer limit: {layer_count} > {self.training.max_layers}. "
                "Simplify the model architecture."
            )

        logger.info(f"Created model with {parameter_count} parameters, {layer_count} layers")
        model.summary(print_fn=lambda line: logger.debug(line))

        return model

    def _reuse_weights(
        self,
        model: Any,
        layer_names: list[str],
        weights_path: Path,
    ) -> list[WeightReuseResult]:
        """
        Load weights from the previous model for the named layers.

        Returns:
            One result per requested layer, successful or not, for agent feedback.
        """
        _ensure_tf()

        results: list[WeightReuseResult] = []
        logger.info(f"Attempting to reuse weights for layers: {layer_names}")

        try:
            previous_model = tf.keras.models.load_model(weights_path, compile=False)
        except Exception as error:
            logger.warning(f"Could not load previous model: {error}")
            return [
                WeightReuseResult(
                    layer_name=layer_name,
                    success=False,
                    reason=f"Could not load previous model: {error}",
                )
                for layer_name in layer_names
            ]

        for layer_name in layer_names:
            try:
                source_weights = previous_model.get_layer(layer_name).get_weights()
                target_layer = model.get_layer(layer_name)
                target_weights = target_layer.get_weights()
            except ValueError as error:
                logger.warning(f"Could not reuse weights for {layer_name}: {error}")
                results.append(
                    WeightReuseResult(layer_name=layer_name, success=False, reason=str(error))
                )
                continue

            if len(source_weights) != len(target_weights):
                reason = (
                    f"Weight count mismatch: source has {len(source_weights)} tensors, "
                    f"target has {len(target_weights)}"
                )
                logger.warning(f"Layer {layer_name}: {reason}")
                results.append(
                    WeightReuseResult(layer_name=layer_name, success=False, reason=reason)
                )
                continue

            mismatches = [
                f"tensor {index}: source shape {source.shape} != target shape {target.shape}"
                for index, (source, target) in enumerate(zip(source_weights, target_weights))
                if source.shape != target.shape
            ]

            if mismatches:
                reason = f"Shape mismatch: {'; '.join(mismatches)}"
                logger.warning(f"Layer {layer_name}: {reason}")
                results.append(
                    WeightReuseResult(layer_name=layer_name, success=False, reason=reason)
                )
                continue

            target_layer.set_weights(source_weights)
            logger.info(f"Reused weights for layer: {layer_name}")
            results.append(WeightReuseResult(layer_name=layer_name, success=True))

        return results

    def save_weights(self, model: Any, path: Path) -> None:
        """Save the trained model in the native Keras format."""
        _ensure_tf()
        model.save(path)
        logger.info(f"Saved model to {path}")

    def save_model_visualization(self, model: Any, path: Path) -> bool:
        """
        Save a diagram of the model architecture as PNG.

        Returns:
            True if written, False if pydot or graphviz is missing.
        """
        _ensure_tf()

        try:
            from tensorflow.keras.utils import plot_model

            path.parent.mkdir(parents=True, exist_ok=True)
            plot_model(
                model,
                to_file=str(path),
                show_shapes=True,
                show_layer_names=True,
                show_layer_activations=True,
                expand_nested=True,
                dpi=150,
            )
            logger.info(f"Saved model visualization to {path}")
            return True

        except ImportError as error:
            logger.warning(f"Cannot create model visualization: {error}")
            logger.warning("Install pydot and graphviz: pip install pydot && apt install graphviz")
            return False

        except Exception as error:
            logger.warning(f"Failed to create model visualization: {error}")
            return False

    def get_weights_metadata(self, model: Any) -> list[WeightMetadata]:
        """Describe each layer's weights for AI agent context.

        One entry per layer, named exactly as the layer is: the agent quotes these names
        back in weight_reuse_layers, and a per-tensor name such as 'dense_1_bias' matches
        no layer, so every reuse request built from one used to fail.
        """
        _ensure_tf()

        metadata = []
        for layer in model.layers:
            weights = layer.get_weights()
            if not weights:
                continue

            metadata.append(
                WeightMetadata(
                    layer_name=layer.name,
                    shape=tuple(weights[0].shape),
                    bias_shape=tuple(weights[1].shape) if len(weights) > 1 else None,
                    dtype=str(weights[0].dtype),
                    trainable=layer.trainable,
                )
            )

        return metadata

    def convert_to_tflite(
        self,
        model: Any,
        output_path: Path,
        quantization_config: QuantizationConfig,
        representative_data: np.ndarray | None = None,
    ) -> int:
        """
        Convert a Keras model to TFLite with the configured quantization.

        Returns:
            Size of the TFLite model in bytes.

        Raises:
            RuntimeError: the TFLite converter rejected the model.
        """
        _ensure_tf()

        logger.info("Converting model to TFLite format")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        is_float16_only = (
            QuantizationType.FLOAT16 in quantization_config.allowed
            and QuantizationType.INT8 not in quantization_config.allowed
        )

        if is_float16_only:
            logger.warning(
                "Using float16 quantization. int8 is required by the specification for "
                "embedded deployment; add 'int8' to quantization.allowed."
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        else:
            per_channel = "per-channel" if quantization_config.per_channel else "per-tensor"
            logger.info(f"Applying int8 {per_channel} quantization")

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

            if not quantization_config.per_channel:
                # Private converter flag: there is no public switch for per-tensor
                # weights, and it is silently ignored if TensorFlow renames it.
                converter._experimental_disable_per_channel = True

            if representative_data is not None:
                converter.representative_dataset = create_representative_dataset(
                    representative_data, seed=self.seed
                )
            else:
                logger.warning("No representative data provided; int8 ranges will be guessed")

        try:
            with suppressed_native_output("the TFLite converter"):
                tflite_model = converter.convert()
        except Exception as error:
            logger.error(f"TFLite conversion failed: {error}")
            raise RuntimeError(f"TFLite conversion failed: {error}") from error

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(tflite_model)

        logger.info(f"Saved TFLite model ({len(tflite_model)} bytes) to {output_path}")
        return len(tflite_model)
