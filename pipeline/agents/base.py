# _____________________________________________________________________________
#
# @file base.py
# @brief Base class for AI agent interfaces
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Base class for AI agent interfaces.

Defines the common interface that all AI providers must implement, and owns the
request, parse and retry loop so that providers only have to make one API call.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.config import AIAgentConfig

logger = logging.getLogger(__name__)

# JSON schema for model proposal validation
PROPOSAL_SCHEMA = {
    "required": ["architecture_description", "model_code"],
    "optional": ["weight_reuse_layers", "rationale", "reasoning", "stop", "stop_reason"],
    "types": {
        "architecture_description": str,
        "model_code": str,
        "weight_reuse_layers": list,
        "rationale": str,
        "reasoning": str,  # Ignored but allowed (LLMs sometimes emit it)
        "stop": bool,
        "stop_reason": str,
    },
}

# A proposal that ends the run carries no model, so the usual required fields do not
# apply to it.
STOP_REQUIRED_FIELDS = ["stop_reason"]

# Operations the firmware's op resolver registers. Widening this list without widening
# the resolver in zephyr_app/src/main_functions.cpp produces a model that builds, flashes
# and then hard-faults on the first inference.
SUPPORTED_LAYERS = (
    "Dense (FullyConnected), Conv2D, DepthwiseConv2D, MaxPooling2D, AveragePooling2D, "
    "GlobalAveragePooling2D, Flatten, Reshape, ZeroPadding2D, Add"
)
SUPPORTED_ACTIVATIONS = "relu, relu6, sigmoid (logistic), softmax"

# Example model code, shown to the agent as part of the template below.
EXAMPLE_MODEL_CODE = """import tensorflow as tf

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name='input'),
        tf.keras.layers.Dense(8, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(4, activation='relu', name='dense_2'),
        tf.keras.layers.Dense(1, name='output')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
"""

# Shown on a parse failure, and as the worked example in the system prompt. Built with
# json.dumps so the escaping is always right for the format the agent must return.
FALLBACK_TEMPLATE = json.dumps(
    {
        "architecture_description": (
            "Tiny MLP for sine regression: Dense(8) -> Dense(4) -> Dense(1)."
        ),
        "model_code": EXAMPLE_MODEL_CODE,
        "weight_reuse_layers": [],
        "rationale": (
            "Small model to fit RAM/flash and keep latency low while meeting the target."
        ),
    },
    indent=2,
)


@dataclass
class ModelProposal:
    """Proposal from AI agent for a model architecture."""

    # Human-readable description of the architecture
    architecture_description: str

    # Executable Python code defining the model
    model_code: str

    # List of layer names to reuse weights from (empty = train from scratch)
    weight_reuse_layers: list[str] = field(default_factory=list)

    # Optional explanation of changes from previous iteration
    rationale: str = ""

    # Set when the agent judges that further iteration cannot improve the result
    should_stop: bool = False
    stop_reason: str = ""


class BaseAgent(ABC):
    """Abstract base class for AI agent implementations."""

    DEFAULT_MODEL = ""

    def __init__(self, api_key: str, config: AIAgentConfig | None = None):
        self.api_key = api_key
        self.config = config or AIAgentConfig(provider="pseudo")
        self._conversation_history: list[dict[str, str]] = []

    @property
    def model(self) -> str:
        """Model id for this provider, from configuration or the provider default."""
        return self.config.model or self.DEFAULT_MODEL

    @abstractmethod
    def _request(self, system_prompt: str, messages: list[dict[str, str]]) -> str:
        """Send one request to the provider and return the raw response text.

        Raises:
            RuntimeError: the provider call failed; this is not retried as a parse error.
        """

    def propose_model(self, context: dict[str, Any]) -> ModelProposal:
        """
        Request a model proposal, retrying when the response cannot be parsed.

        Args:
            context: Dictionary containing:
                - constraints: Hardware and performance constraints
                - iteration: Current iteration number
                - previous_iteration: Results from previous iteration (if any)
                - available_weights: Weight metadata from previous model (if any)
                - best_so_far: Best results achieved so far

        Returns:
            ModelProposal with architecture and code

        Raises:
            RuntimeError: the provider failed, or no attempt produced a valid proposal.
        """
        system_prompt = self._build_system_prompt(context)
        user_prompt = self._build_user_prompt(context)

        logger.info(f"Requesting model proposal from {type(self).__name__} ({self.model})")
        logger.debug(f"User prompt:\n{user_prompt}")

        messages = [*self._conversation_history, {"role": "user", "content": user_prompt}]
        last_error = ""

        for attempt in range(self.config.max_retries + 1):
            response_text = self._request(system_prompt, messages)
            logger.debug(f"Response (attempt {attempt + 1}):\n{response_text}")

            try:
                proposal = self._parse_response(response_text)
            except ValueError as error:
                last_error = str(error)
                logger.warning(f"Parse failed (attempt {attempt + 1}): {last_error}")
                messages = [
                    *messages,
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": self._build_retry_prompt(last_error)},
                ]
                continue

            self._remember_exchange(user_prompt, response_text)
            logger.info(f"Received proposal: {proposal.architecture_description}")
            return proposal

        raise RuntimeError(
            f"No valid proposal after {self.config.max_retries + 1} attempts: {last_error}"
        )

    def _remember_exchange(self, user_prompt: str, response_text: str) -> None:
        """Keep the last ten exchanges so the agent can see its own history."""
        self._conversation_history.append({"role": "user", "content": user_prompt})
        self._conversation_history.append({"role": "assistant", "content": response_text})
        self._conversation_history = self._conversation_history[-20:]

    def _build_system_prompt(self, context: dict[str, Any] | None = None) -> str:
        """Build the system prompt for the AI agent."""
        constraints = (context or {}).get("constraints", {})
        performance = constraints.get("performance", {})
        limits = constraints.get("model_limits", {})

        return f"""You are an expert ML engineer specializing in TinyML and embedded
systems. Your task is to design neural network architectures that run efficiently on
resource-constrained devices.

CRITICAL OUTPUT RULES:
- Return JSON ONLY. No markdown code fences. No extra text before or after.
- Response must be a single valid JSON object that can be parsed directly.

HARD LIMITS (proposals exceeding these are rejected before training):
- Maximum model parameters: {limits.get("max_parameters", 10000)}
- Maximum layers: {limits.get("max_layers", 10)}
- Maximum RAM usage: {performance.get("max_ram_kb", 128)} KB
- Maximum Flash usage: {performance.get("max_flash_kb", 256)} KB

MODEL REQUIREMENTS:
- Use TensorFlow/Keras Sequential or Functional API
- Must be convertible to TensorFlow Lite with int8 quantization
- Define a function `create_model(input_shape)` that returns a compiled Keras model
- Include optimizer and loss in model.compile()

TFLITE MICRO OPERATION CONSTRAINTS (CRITICAL):
- The firmware registers exactly these operations; anything else faults on the device
- Supported layers: {SUPPORTED_LAYERS}
- Supported activations: {SUPPORTED_ACTIVATIONS}
- DO NOT use LSTM, GRU, BatchNormalization, LayerNormalization, Attention,
  MultiHeadAttention, Dropout, SeparableConv2D, or any custom layer
- Match the architecture to the input: Dense layers for a flat feature vector,
  convolution for an image

RESPONSE FORMAT (JSON only, no code fences):
{{
  "architecture_description": "Brief description of the model architecture",
  "model_code": "import tensorflow as tf\\n\\ndef create_model(input_shape):\\n    ...",
  "weight_reuse_layers": [],
  "rationale": "Explanation of design choices"
}}

EXAMPLE RESPONSE:
{FALLBACK_TEMPLATE}

{self._describe_stopping(context)}WEIGHT REUSE RULES (CRITICAL):
- Name a layer in "weight_reuse_layers" only when its weights are shape-compatible
- Both must hold: the layer name matches a layer listed under AVAILABLE WEIGHTS, and
  your layer has the same kernel shape as the one listed there
- Changing a layer's width or its input dimension makes its weights incompatible
- When in doubt leave "weight_reuse_layers" empty; training from scratch is safer
"""

    @staticmethod
    def _describe_stopping(context: dict[str, Any] | None) -> str:
        """Offer the stop response, when the run allows the agent to end it.

        Interpolated into the system prompt's f-string, so the braces below are not
        re-scanned and need no doubling.
        """
        if not (context or {}).get("may_stop"):
            return ""

        return """ENDING THE RUN:
- If further iteration cannot improve the result, return
  {"stop": true, "stop_reason": "..."} instead of a model. No other field is required.
- Stop when the metric has reached the irreducible floor quoted in the request, when
  several iterations have produced no material improvement, or when every remaining
  change would breach a hard limit.
- The floor is the best score the data admits: labels carry noise, so a model that
  recovers the underlying function exactly still scores it. Within a few percent of the
  floor means the task is solved, not that the search is stuck.
- Do not stop because an iteration was rejected or failed; that is feedback to act on.
  Stop only when no further proposal of yours would do better.

"""

    def _build_user_prompt(self, context: dict[str, Any]) -> str:
        """Build the user prompt from context."""
        constraints = context.get("constraints", {})
        hardware = constraints.get("hardware", {})
        performance = constraints.get("performance", {})
        accuracy = constraints.get("accuracy", {})
        task = context.get("task", {})

        task_type = task.get("type", "regression")
        metric = accuracy.get("metric", "mse")
        is_higher_better = accuracy.get("higher_is_better", False)

        prompt_parts = [
            f"Iteration {context.get('iteration', 1)}: design a neural network for this task:",
            "",
            f"Task type: {task_type}",
            *self._describe_data(context),
            self._describe_output_layer(task_type, context),
            "",
            "HARDWARE CONSTRAINTS:",
            f"- Board: {hardware.get('board', 'unknown')}",
            f"- Maximum RAM: {performance.get('max_ram_kb', 128)} KB",
            f"- Maximum Flash: {performance.get('max_flash_kb', 256)} KB",
            f"- Execution unit: {'CPU and NPU' if hardware.get('npu_enabled') else 'CPU'}",
            "",
            "PERFORMANCE CONSTRAINTS:",
            f"- Maximum latency: {performance.get('max_latency_ms', 5)} ms",
            f"- Metric: {metric} ({'higher' if is_higher_better else 'lower'} is better)",
            f"- Target {metric}: {accuracy.get('target', 0.01)}",
            f"- Rejected beyond: {accuracy.get('max_error', 0.05)}",
            *self._describe_floor(accuracy),
            f"- Quantization: {constraints.get('quantization', {}).get('allowed', ['int8'])}",
            "",
        ]

        if hardware.get("npu_enabled"):
            prompt_parts.append(
                "NOTE: the board has a neural accelerator, and the quantised model is "
                "compiled for it after training. Operators it cannot take stay on the "
                "CPU. Design as normal; no accelerator-specific layers are needed."
            )
            prompt_parts.append("")

        prompt_parts.extend(self._describe_previous_iteration(context))
        prompt_parts.extend(self._describe_available_weights(context))
        prompt_parts.extend(self._describe_best_so_far(context, metric))

        prompt_parts.append("Provide your model proposal in the specified JSON format.")

        return "\n".join(prompt_parts)

    @staticmethod
    def _describe_floor(accuracy: dict[str, Any]) -> list[str]:
        """State the best score the data admits, when there is one."""
        floor = accuracy.get("irreducible_floor")
        if floor is None:
            return []

        return [f"- Irreducible floor: {floor:.6f} (labels carry noise; nothing scores better)"]

    @staticmethod
    def _describe_data(context: dict[str, Any]) -> list[str]:
        """State what one sample looks like, which decides the architecture."""
        lines = []

        dataset = context.get("dataset")
        if dataset and dataset != "synthetic":
            lines.append(f"Dataset: {dataset}")
        else:
            lines.append(f"Data function: {context.get('task', {}).get('function', 'sine')}")

        input_shape = context.get("input_shape")
        if input_shape:
            lines.append(f"Input shape per sample: {tuple(input_shape)}")
        if context.get("class_count"):
            lines.append(f"Classes: {context['class_count']}")

        return lines

    @staticmethod
    def _describe_output_layer(task_type: str, context: dict[str, Any] | None = None) -> str:
        """State the output layer the task needs, in the model's own vocabulary."""
        if task_type == "classification":
            classes = (context or {}).get("class_count", "one per class")
            return (
                f"Output layer: Dense({classes}) with softmax activation, compiled with "
                "sparse_categorical_crossentropy and metrics=['accuracy']. Labels are "
                "integer class indices, not one-hot."
            )
        return (
            "Output layer: Dense(1) with linear activation, compiled with the mse loss. "
            "Use relu for hidden layers."
        )

    @staticmethod
    def _describe_previous_iteration(context: dict[str, Any]) -> list[str]:
        """Report the previous iteration's outcome, including why it was rejected."""
        previous = context.get("previous_iteration")
        if not previous:
            return []

        lines = ["PREVIOUS ITERATION RESULTS:", f"- Status: {previous.get('status', 'unknown')}"]

        metrics = previous.get("metrics", {})
        if metrics.get("accuracy_value") is not None:
            measured = f"- Measured metric: {metrics['accuracy_value']:.6f}"
            if metrics.get("accuracy_standard_error") is not None:
                measured += (
                    f" ± {metrics['accuracy_standard_error']:.6f} (standard error on "
                    f"{metrics.get('accuracy_sample_count')} validation samples)"
                )
            lines.append(measured)
        if metrics.get("latency_ms") is not None:
            lines.append(f"- Latency: {metrics['latency_ms']:.3f} ms")
        if metrics.get("ram_usage_kb") is not None:
            lines.append(f"- RAM usage: {metrics['ram_usage_kb']} KB")
        if metrics.get("flash_usage_kb") is not None:
            lines.append(f"- Flash usage: {metrics['flash_usage_kb']} KB")
        if metrics.get("arena_used_bytes") is not None:
            lines.append(f"- Tensor arena used: {metrics['arena_used_bytes']} bytes")

        if previous.get("is_under_trained"):
            lines.append(
                f"- WARNING: training stopped at the epoch cap after "
                f"{previous.get('epochs_trained')} epochs without converging, so that "
                "model was under-trained. A deeper model may have scored worse only "
                "because it had less chance to converge, not because it is worse."
            )

        failure = previous.get("failure")
        if failure and failure.get("message"):
            lines.append(f"- Failure: {failure['message']}")

        reuse_failures = previous.get("weight_reuse", {}).get("failures", [])
        if reuse_failures:
            lines.append("")
            lines.append("WEIGHT REUSE FAILURES (layers that could not take previous weights):")
            lines.extend(
                f"- {failure_info['layer_name']}: {failure_info['reason']}"
                for failure_info in reuse_failures
            )
            lines.append("Adjust those layers to match, or stop requesting reuse for them.")

        lines.append("")
        return lines

    @staticmethod
    def _describe_available_weights(context: dict[str, Any]) -> list[str]:
        """List reusable weights by layer name and kernel shape."""
        weights = context.get("available_weights")
        if not weights:
            return []

        lines = [
            "AVAILABLE WEIGHTS FOR REUSE:",
            "(Request a layer only if your layer has the same name and kernel shape)",
        ]
        for entry in weights:
            bias = f", bias={tuple(entry['bias_shape'])}" if entry.get("bias_shape") else ""
            lines.append(f"- {entry['layer_name']}: kernel={tuple(entry['shape'])}{bias}")

        lines.append("")
        return lines

    @staticmethod
    def _describe_best_so_far(context: dict[str, Any], metric: str) -> list[str]:
        """Report the best iteration so far, and how much of its lead is real."""
        best = context.get("best_so_far")
        if not best or best.get("accuracy") is None:
            return []

        latency = f"{best['latency_ms']:.3f}ms" if best.get("latency_ms") is not None else "unknown"
        score = f"{metric}={best['accuracy']:.6f}"
        if best.get("standard_error") is not None:
            score += f" ± {best['standard_error']:.6f}"

        lines = [
            "BEST RESULT SO FAR:",
            f"- Iteration {best['iteration']}: {score}, latency={latency}",
        ]

        unresolved = context.get("unresolved_ranking")
        if unresolved:
            lines.append(f"- WARNING: {unresolved}")
            lines.append(
                "- Chasing a difference that small is wasted effort. Either change the "
                "architecture enough to move the metric well beyond that error bar, or "
                "stop and say the search has converged."
            )

        lines.append("")
        return lines

    def _extract_json(self, response_text: str) -> str:
        """Extract the JSON object from a response, tolerating fences and prose."""
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if fenced:
            return fenced.group(1)

        start_index = response_text.find("{")
        if start_index != -1:
            depth = 0
            for offset, character in enumerate(response_text[start_index:], start_index):
                if character == "{":
                    depth += 1
                elif character == "}":
                    depth -= 1
                    if depth == 0:
                        return response_text[start_index : offset + 1]

        return response_text.strip()

    def _validate_schema(self, data: dict) -> list[str]:
        """Validate parsed data against the proposal schema. Returns a list of errors."""
        errors = []

        required = STOP_REQUIRED_FIELDS if data.get("stop") else PROPOSAL_SCHEMA["required"]

        for field_name in required:
            if field_name not in data:
                errors.append(f"Missing required field: {field_name}")
            elif not data[field_name]:
                errors.append(f"Required field is empty: {field_name}")

        for field_name, expected_type in PROPOSAL_SCHEMA["types"].items():
            if data.get(field_name) is not None and not isinstance(data[field_name], expected_type):
                errors.append(
                    f"Field '{field_name}' has wrong type: expected {expected_type.__name__}, "
                    f"got {type(data[field_name]).__name__}"
                )

        model_code = data.get("model_code", "")
        if model_code and "def create_model" not in model_code:
            errors.append("model_code must define a 'create_model(input_shape)' function")

        return errors

    def _parse_response(self, response_text: str) -> ModelProposal:
        """Parse the agent's response into a ModelProposal.

        Raises:
            ValueError: the response was not JSON, or did not satisfy the schema.
        """
        json_text = self._extract_json(response_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as error:
            logger.debug(f"Response text: {response_text[:500]}")
            raise ValueError(f"Response was not valid JSON: {error}") from error

        errors = self._validate_schema(data)
        if errors:
            raise ValueError(f"Response validation failed: {'; '.join(errors)}")

        if data.get("stop"):
            return ModelProposal(
                architecture_description="",
                model_code="",
                should_stop=True,
                stop_reason=data["stop_reason"],
            )

        # json.loads has already turned escapes into real characters. Replacing '\n'
        # again here corrupted any literal backslash-n inside the generated code.
        model_code = data["model_code"]

        fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", model_code, re.DOTALL)
        if fenced:
            model_code = fenced.group(1)

        return ModelProposal(
            architecture_description=data["architecture_description"],
            model_code=model_code,
            weight_reuse_layers=data.get("weight_reuse_layers") or [],
            rationale=data.get("rationale", ""),
        )

    def _build_retry_prompt(self, error_message: str) -> str:
        """Ask the agent to correct an unparseable response."""
        return f"""Your previous response could not be parsed.

ERROR: {error_message}

Please respond with ONLY a valid JSON object. No markdown, no code fences, no extra text.

Use this exact format:
{FALLBACK_TEMPLATE}

Respond with JSON only:"""
