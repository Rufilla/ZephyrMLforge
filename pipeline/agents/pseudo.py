# _____________________________________________________________________________
#
# @file pseudo.py
# @brief Pseudo agent implementation for testing without API calls
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Pseudo agent implementation for testing without API calls.
"""

from __future__ import annotations

import logging
from typing import Any

from ..core.config import AIAgentConfig
from .base import BaseAgent, ModelProposal

logger = logging.getLogger(__name__)

# Hidden layer widths proposed on successive iterations, narrowing as it goes.
HIDDEN_WIDTHS = ((16, 8), (12, 6), (8, 4))

# Convolution filter counts for image inputs, narrowing likewise.
CONVOLUTION_FILTERS = ((16, 32), (8, 16), (8, 8))


class PseudoAgent(BaseAgent):
    """Mock AI agent returning pre-defined proposals without API calls."""

    DEFAULT_MODEL = "pseudo"

    def __init__(self, api_key: str = "", config: AIAgentConfig | None = None):
        super().__init__(api_key, config)

    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Never called: propose_model is overridden and makes no request."""
        raise NotImplementedError("The pseudo agent does not call a provider")

    @staticmethod
    def _convolutional_body(iteration: int) -> tuple[str, str]:
        """Two convolution blocks for an image input, narrowing on later iterations."""
        first, second = CONVOLUTION_FILTERS[(iteration - 1) % len(CONVOLUTION_FILTERS)]

        body = (
            f"        tf.keras.layers.Conv2D({first}, 3, activation='relu', "
            "padding='same', name='conv_1'),\n"
            "        tf.keras.layers.MaxPooling2D(2, name='pool_1'),\n"
            f"        tf.keras.layers.Conv2D({second}, 3, activation='relu', "
            "padding='same', name='conv_2'),\n"
            "        tf.keras.layers.MaxPooling2D(2, name='pool_2'),\n"
            "        tf.keras.layers.Flatten(name='flatten'),"
        )
        return body, f"Convolutional network with {first} and {second} filters"

    def propose_model(self, context: dict[str, Any]) -> ModelProposal:
        """Return a proposal suited to the configured task."""
        iteration = context.get("iteration", 1)
        task_type = context.get("task", {}).get("type", "regression")

        first_width, second_width = HIDDEN_WIDTHS[(iteration - 1) % len(HIDDEN_WIDTHS)]
        is_classification = task_type == "classification"
        class_count = context.get("class_count") or 2

        input_shape = context.get("input_shape") or []

        if is_classification:
            output_layer = (
                f"tf.keras.layers.Dense({class_count}, activation='softmax', name='output')"
            )
            compile_call = (
                "model.compile(optimizer='adam', "
                "loss='sparse_categorical_crossentropy', metrics=['accuracy'])"
            )
        else:
            output_layer = "tf.keras.layers.Dense(1, name='output')"
            compile_call = "model.compile(optimizer='adam', loss='mse', metrics=['mae'])"

        is_image = len(input_shape) > 1

        if is_image:
            body, description = self._convolutional_body(iteration)
        else:
            body = (
                f"        tf.keras.layers.Dense({first_width}, activation='relu', "
                "name='dense_1'),\n"
                f"        tf.keras.layers.Dense({second_width}, activation='relu', "
                "name='dense_2'),"
            )
            description = f"MLP with {first_width} and {second_width} hidden units"

        model_code = f"""import tensorflow as tf

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input'),
{body}
        {output_layer}
    ])
    {compile_call}
    return model
"""

        proposal = ModelProposal(
            architecture_description=f"{description} for {task_type}",
            model_code=model_code,
            weight_reuse_layers=[],
            rationale="Fixed sequence of narrowing MLPs; no provider is consulted.",
        )

        logger.info(f"Pseudo agent returning: {proposal.architecture_description}")
        return proposal
