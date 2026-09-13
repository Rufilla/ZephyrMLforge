# _____________________________________________________________________________
#
# @file config.py
# @brief Configuration schema and validation for the pipeline
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Configuration schema and validation for the pipeline.

This module defines the Pydantic models that validate YAML configuration files.
The YAML file is the authoritative contract, so every tunable the pipeline honours
is declared here, with its own field description.
"""

from __future__ import annotations

import math
import os
import re
from enum import Enum
from pathlib import Path

import tomllib
import yaml
from pydantic import BaseModel, Field, SecretStr, model_validator

# Matches a whole-value environment reference such as '${AI_PIPELINE_PASSWORD}'.
ENV_REFERENCE_PATTERN = re.compile(r"^\$\{([^}]+)\}$")


class ExecutionMode(str, Enum):
    """Execution environment modes."""

    QEMU = "qemu"
    HIL = "hil"
    AUTO = "auto"


class AIProvider(str, Enum):
    """Supported AI agent providers."""

    CHATGPT = "chatgpt"
    CLAUDE = "claude"
    CLAUDE_CODE = "claude_code"
    PSEUDO = "pseudo"

    @property
    def needs_api_key(self) -> bool:
        """True when the provider authenticates with an environment key."""
        return self in (AIProvider.CHATGPT, AIProvider.CLAUDE)


class AccuracyMetric(str, Enum):
    """Supported accuracy/error metrics."""

    MSE = "mse"
    MAE = "mae"
    ACCURACY = "accuracy"

    @property
    def is_higher_better(self) -> bool:
        """True when a larger value is a better model."""
        return self is AccuracyMetric.ACCURACY


class DatasetSource(str, Enum):
    """Where the training data comes from."""

    SYNTHETIC = "synthetic"
    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"

    @property
    def is_image(self) -> bool:
        """True for the image datasets, which are always classification."""
        return self is not DatasetSource.SYNTHETIC


class SyntheticTask(str, Enum):
    """Synthetic data task types."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"


class SyntheticFunction(str, Enum):
    """Synthetic data generation functions for regression."""

    SINE = "sine"
    QUADRATIC = "quadratic"
    STEP = "step"


class QuantizationType(str, Enum):
    """Supported quantization types."""

    INT8 = "int8"
    FLOAT16 = "float16"


class HardwareConfig(BaseModel):
    """Hardware configuration for target board."""

    board: str = Field(..., description="Zephyr board identifier")
    soc: str = Field(..., description="SoC identifier")
    ram_kb: int = Field(..., gt=0, description="Total RAM in KB")
    flash_kb: int = Field(..., gt=0, description="Total flash in KB")
    tensor_arena_kb: int = Field(
        default=8, gt=0, description="TFLite Micro tensor arena size in KB"
    )
    qemu_board: str = Field(
        default="qemu_cortex_m3",
        description="Board used for QEMU runs; the real target rarely has QEMU support",
    )


class ExecutionConfig(BaseModel):
    """Execution unit configuration."""

    cpu: bool = Field(default=True, description="Enable CPU execution")
    npu: bool = Field(default=False, description="Enable NPU execution")


class ExecutionEnvironmentConfig(BaseModel):
    """Execution environment, serial link and subprocess timeouts."""

    mode: ExecutionMode = Field(
        default=ExecutionMode.QEMU, description="Execution mode: qemu, hil, or auto"
    )
    serial_port: str = Field(default="/dev/ttyACM0", description="Serial port for HIL")
    serial_baud: int = Field(default=115200, gt=0, description="Serial baud rate")
    flash_runner: str | None = Field(
        default=None,
        description="west flash runner (pyocd, linkserver, jlink); the board default when unset",
    )
    build_timeout_s: int = Field(default=300, gt=0, description="west build timeout in seconds")
    flash_timeout_s: int = Field(default=60, gt=0, description="west flash timeout in seconds")
    run_timeout_s: int = Field(default=30, gt=0, description="On-target run timeout in seconds")
    qemu_timeout_s: int = Field(default=120, gt=0, description="QEMU run timeout in seconds")
    benchmark_runs: int = Field(
        default=100, gt=0, description="Inferences averaged for the latency measurement"
    )


class AccuracyConfig(BaseModel):
    """Model accuracy/error requirements.

    For error metrics (mse, mae) lower is better, so ``target <= max_error``. For
    the accuracy metric higher is better, so the two bounds invert and
    ``max_error`` is the lowest accuracy still accepted.
    """

    metric: AccuracyMetric = Field(..., description="Metric type")
    target: float = Field(..., description="Value to optimise toward")
    max_error: float = Field(..., description="Worst acceptable value; beyond this is rejected")
    evaluate_on_device: bool = Field(
        default=False,
        description="Evaluate on the board over the serial protocol, not the host interpreter",
    )
    on_device_samples: int = Field(
        default=64,
        gt=0,
        description=(
            "Validation samples sent to the board, when on device; each is a serial "
            "round trip, and the error bar on the score they give is wide"
        ),
    )

    @model_validator(mode="after")
    def validate_error_bounds(self) -> "AccuracyConfig":
        """Check that the target is at least as demanding as the rejection bound."""
        if self.metric.is_higher_better:
            if self.target < self.max_error:
                raise ValueError(
                    f"For {self.metric.value}, target ({self.target}) must be >= "
                    f"max_error ({self.max_error})"
                )
        elif self.target > self.max_error:
            raise ValueError(
                f"For {self.metric.value}, target ({self.target}) must be <= "
                f"max_error ({self.max_error})"
            )
        return self

    def is_within_limit(self, value: float) -> bool:
        """True when the measured value is acceptable under ``max_error``."""
        return value >= self.max_error if self.metric.is_higher_better else value <= self.max_error

    def meets_target(self, value: float) -> bool:
        """True when the measured value reaches ``target``."""
        return value >= self.target if self.metric.is_higher_better else value <= self.target

    def is_better(self, value: float, than: float) -> bool:
        """True when ``value`` is the better of the two measurements."""
        return value > than if self.metric.is_higher_better else value < than


class PerformanceConfig(BaseModel):
    """Performance constraints."""

    max_latency_ms: float = Field(..., gt=0, description="Maximum inference latency in ms")
    max_ram_kb: int = Field(..., gt=0, description="Maximum RAM usage in KB")
    max_flash_kb: int = Field(..., gt=0, description="Maximum flash usage in KB")


class QuantizationConfig(BaseModel):
    """Quantization policy configuration."""

    allowed: list[QuantizationType] = Field(
        default=[QuantizationType.INT8], description="Allowed quantization types"
    )
    per_channel: bool = Field(default=True, description="Enable per-channel quantization")


class AIAgentConfig(BaseModel):
    """AI agent provider and request parameters."""

    provider: AIProvider = Field(..., description="AI provider: chatgpt, claude or pseudo")
    model: str | None = Field(
        default=None, description="Provider model id; the provider default is used when unset"
    )
    temperature: float = Field(
        default=0.7, ge=0, le=2,
        description="Sampling temperature; chatgpt only, the Messages API removed it",
    )
    max_tokens: int = Field(
        default=16000,
        gt=0,
        description="Response token budget; thinking is drawn from it, so keep it ample",
    )
    max_retries: int = Field(
        default=2, ge=0, description="Retries offered to the agent after an unparseable response"
    )
    cli_path: str = Field(
        default="claude", description="Claude Code executable, for the claude_code provider"
    )
    cli_timeout_s: int = Field(
        default=300, gt=0, description="Seconds allowed for one Claude Code invocation"
    )

    def get_api_key(self) -> str:
        """Read the provider API key from the environment.

        Raises:
            ValueError: the provider needs a key and its variable is unset.
        """
        if not self.provider.needs_api_key:
            return ""
        env_var = "OPENAI_API_KEY" if self.provider == AIProvider.CHATGPT else "ANTHROPIC_API_KEY"
        key = os.environ.get(env_var)
        if not key:
            raise ValueError(f"Environment variable {env_var} not set")
        return key


class IterationConfig(BaseModel):
    """Iteration control configuration."""

    max_iterations: int = Field(default=50, gt=0, description="Maximum iterations")
    early_stop_on_target: bool = Field(default=True, description="Stop early when targets are met")
    allow_agent_stop: bool = Field(
        default=True,
        description="Let the agent end the run when it judges further search pointless",
    )


class TrainingConfig(BaseModel):
    """Host-side training hyperparameters and proposal guardrails."""

    epochs: int = Field(default=100, gt=0, description="Maximum training epochs")
    batch_size: int = Field(default=32, gt=0, description="Training batch size")
    early_stopping_patience: int = Field(
        default=10, gt=0, description="Epochs without val_loss improvement before stopping"
    )
    max_parameters: int = Field(
        default=10_000, gt=0, description="Proposals above this parameter count are rejected"
    )
    max_layers: int = Field(
        default=10, gt=0, description="Proposals above this layer count are rejected"
    )


class SyntheticDataConfig(BaseModel):
    """Synthetic data generation configuration."""

    task: SyntheticTask = Field(..., description="Task type")
    function: SyntheticFunction = Field(
        default=SyntheticFunction.SINE, description="Generation function for regression"
    )
    samples: int = Field(default=1000, gt=0, description="Number of samples")
    noise_stddev: float = Field(default=0.1, ge=0, description="Noise standard deviation")
    train_split: float = Field(default=0.8, gt=0, lt=1, description="Training split ratio")
    seed: int = Field(default=42, description="Random seed for reproducibility")

    def irreducible_error(self, metric: AccuracyMetric) -> float | None:
        """Best error any model can reach against these labels.

        The labels carry additive Gaussian noise, so a model that recovers the
        underlying function exactly is still scored against the noise: mean squared
        error bottoms out at the variance, mean absolute error at the half-normal mean.
        Returns None where the floor does not apply.
        """
        if self.task is not SyntheticTask.REGRESSION:
            return None

        if metric is AccuracyMetric.MSE:
            return self.noise_stddev**2
        if metric is AccuracyMetric.MAE:
            return self.noise_stddev * math.sqrt(2 / math.pi)

        return None


class DatasetConfig(BaseModel):
    """Which dataset to train on, and how much of it."""

    source: DatasetSource = Field(
        default=DatasetSource.SYNTHETIC, description="synthetic | mnist | fashion_mnist"
    )
    train_samples: int = Field(
        default=6000, gt=0, description="Training images drawn from the set, for tractable runs"
    )
    val_samples: int = Field(
        default=10000,
        gt=0,
        description=(
            "Validation images drawn from the set; the accuracy it reports carries a "
            "standard error of about sqrt(p(1-p)/val_samples), so a thousand samples "
            "cannot separate models a point apart"
        ),
    )


class TaskDescription(BaseModel):
    """ML task description from pyproject.toml for AI agent context."""

    description: str = Field(default="", description="Plain language task description")
    input_description: str = Field(default="", description="Description of input data")
    output_description: str = Field(default="", description="Description of expected output")
    domain: str = Field(default="", description="Problem domain")
    complexity_hint: str = Field(default="", description="Hint about complexity")

    @classmethod
    def from_pyproject(cls, project_root: Path | None = None) -> "TaskDescription":
        """Load the task description from ``project_root/pyproject.toml``.

        Returns an empty description when the file or the table is absent.
        """
        root = project_root if project_root is not None else Path.cwd()
        pyproject_path = root / "pyproject.toml"
        if not pyproject_path.exists():
            return cls()

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        task_config = data.get("tool", {}).get("zephyr-ml-forge", {}).get("task", {})
        return cls.model_validate(task_config)

    @classmethod
    def discover(cls, start: Path | None = None, source: str | None = None) -> "TaskDescription":
        """Search upward from ``start`` for the task description.

        A named source reads [tool.zephyr-ml-forge.tasks.<source>], falling back to the
        single [tool.zephyr-ml-forge.task] table. The pipeline is often run from a
        subdirectory, where a plain cwd lookup silently yields an empty description and
        the agent is left designing for a task nobody described.
        """
        current = (start or Path.cwd()).resolve()

        for candidate in [current, *current.parents]:
            pyproject_path = candidate / "pyproject.toml"
            if not pyproject_path.exists():
                continue

            with open(pyproject_path, "rb") as f:
                tool_table = tomllib.load(f).get("tool", {}).get("zephyr-ml-forge", {})

            if source:
                named = tool_table.get("tasks", {}).get(source)
                if named:
                    return cls.model_validate(named)

            default = tool_table.get("task")
            if default:
                return cls.model_validate(default)

        return cls()


class BuildConfig(BaseModel):
    """Build directory configuration."""

    build_dir: str = Field(
        default="/tmp/zephyr-build",
        description="Directory for Zephyr builds; /tmp avoids symlink issues on shared folders",
    )

    def get_build_path(self) -> Path:
        """Return the build directory with environment variables expanded."""
        return Path(os.path.expandvars(self.build_dir))


class UserConfig(BaseModel):
    """User credentials.

    The password is held as a ``SecretStr`` so that it cannot reach a log line or an
    artefact by accident; ``password_reference`` keeps the unexpanded '${VAR}' form so
    configuration snapshots stay replayable without carrying the secret.
    """

    username: str = Field(default="", description="Username for authenticated access")
    password: SecretStr = Field(
        default=SecretStr(""), description="Password (use env var expansion)"
    )
    password_reference: str = Field(
        default="", exclude=True, description="Original '${VAR}' reference, for snapshots"
    )

    @model_validator(mode="before")
    @classmethod
    def expand_password_reference(cls, data):
        """Expand a whole-value '${VAR}' password, keeping the reference for snapshots."""
        if not isinstance(data, dict):
            return data

        raw = data.get("password")
        if not isinstance(raw, str) or not raw:
            return data

        match = ENV_REFERENCE_PATTERN.match(raw)
        if match:
            data = dict(data)
            data["password_reference"] = raw
            data["password"] = os.environ.get(match.group(1), "")
        return data


class PipelineConfig(BaseModel):
    """Root configuration model for the pipeline."""

    hardware: HardwareConfig
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    execution_environment: ExecutionEnvironmentConfig = Field(
        default_factory=ExecutionEnvironmentConfig
    )
    accuracy: AccuracyConfig
    performance: PerformanceConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    ai_agent: AIAgentConfig
    iteration: IterationConfig = Field(default_factory=IterationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    synthetic_data: SyntheticDataConfig
    build: BuildConfig = Field(default_factory=BuildConfig)
    user: UserConfig = Field(default_factory=UserConfig)

    @model_validator(mode="after")
    def validate_memory_constraints(self) -> "PipelineConfig":
        """Check that performance constraints fit within the declared hardware."""
        if self.performance.max_ram_kb > self.hardware.ram_kb:
            raise ValueError(
                f"max_ram_kb ({self.performance.max_ram_kb}) exceeds hardware RAM "
                f"({self.hardware.ram_kb})"
            )
        if self.performance.max_flash_kb > self.hardware.flash_kb:
            raise ValueError(
                f"max_flash_kb ({self.performance.max_flash_kb}) exceeds hardware flash "
                f"({self.hardware.flash_kb})"
            )
        if self.hardware.tensor_arena_kb > self.performance.max_ram_kb:
            raise ValueError(
                f"tensor_arena_kb ({self.hardware.tensor_arena_kb}) exceeds max_ram_kb "
                f"({self.performance.max_ram_kb})"
            )
        return self

    @model_validator(mode="after")
    def validate_dataset_pairing(self) -> "PipelineConfig":
        """Check the task and metric suit the dataset."""
        if not self.dataset.source.is_image:
            return self

        if self.synthetic_data.task is not SyntheticTask.CLASSIFICATION:
            raise ValueError(
                f"dataset.source '{self.dataset.source.value}' is a classification set; "
                "set synthetic_data.task to 'classification'"
            )
        return self

    @model_validator(mode="after")
    def validate_metric_task_pairing(self) -> "PipelineConfig":
        """Check that the metric suits the task; the pairing decides how models are scored."""
        is_classification = self.synthetic_data.task == SyntheticTask.CLASSIFICATION
        if is_classification and self.accuracy.metric != AccuracyMetric.ACCURACY:
            raise ValueError(
                f"Classification requires accuracy.metric 'accuracy', not "
                f"'{self.accuracy.metric.value}'"
            )
        if not is_classification and self.accuracy.metric == AccuracyMetric.ACCURACY:
            raise ValueError(
                "accuracy.metric 'accuracy' requires synthetic_data.task 'classification'"
            )
        return self

    def unreachable_target(self) -> str | None:
        """Say why the accuracy target cannot be met, when it cannot.

        A target at or below the noise floor is never reached, so the run always
        spends its whole iteration budget however good the models are.
        """
        if self.dataset.source.is_image:
            return None

        floor = self.synthetic_data.irreducible_error(self.accuracy.metric)
        if floor is None or self.accuracy.target > floor:
            return None

        return (
            f"accuracy.target {self.accuracy.target} is at or below the "
            f"{self.accuracy.metric.value} floor of {floor:.6f} implied by "
            f"synthetic_data.noise_stddev {self.synthetic_data.noise_stddev}: no model "
            "can reach it, and every run will use its whole iteration budget. Raise the "
            "target above the floor, or lower noise_stddev."
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Raises:
            FileNotFoundError: the path does not exist.
            ValidationError: the file parses but violates the schema.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write the configuration to YAML with the credential left unexpanded.

        The snapshot is replayable: an environment-referenced password is written back as
        its '${VAR}' reference, and a literal one is written as an empty string rather
        than copied into the artefact.
        """
        data = self.model_dump(mode="json")
        data["user"]["password"] = self.user.password_reference
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_constraints_summary(self, task_description: TaskDescription | None = None) -> dict:
        """Summarise the constraints handed to the AI agent.

        Credentials are deliberately absent; this dictionary is sent to a third-party API.
        """
        task = task_description if task_description is not None else TaskDescription.discover()

        return {
            "task_description": {
                "description": task.description,
                "input": task.input_description,
                "output": task.output_description,
                "domain": task.domain,
                "complexity_hint": task.complexity_hint,
            },
            "hardware": {
                "board": self.hardware.board,
                "ram_kb": self.hardware.ram_kb,
                "flash_kb": self.hardware.flash_kb,
                "cpu_enabled": self.execution.cpu,
                "npu_enabled": self.execution.npu,
            },
            "accuracy": {
                "metric": self.accuracy.metric.value,
                "target": self.accuracy.target,
                "max_error": self.accuracy.max_error,
                "higher_is_better": self.accuracy.metric.is_higher_better,
                # The agent cannot tell a converged search from a stalled one without
                # knowing the best score the data admits.
                "irreducible_floor": self.synthetic_data.irreducible_error(self.accuracy.metric),
            },
            "performance": {
                "max_latency_ms": self.performance.max_latency_ms,
                "max_ram_kb": self.performance.max_ram_kb,
                "max_flash_kb": self.performance.max_flash_kb,
            },
            "quantization": {
                "allowed": [q.value for q in self.quantization.allowed],
                "per_channel": self.quantization.per_channel,
            },
            "model_limits": {
                "max_parameters": self.training.max_parameters,
                "max_layers": self.training.max_layers,
            },
        }
