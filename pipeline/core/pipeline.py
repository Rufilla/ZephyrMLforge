# _____________________________________________________________________________
#
# @file pipeline.py
# @brief Core pipeline orchestration module
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Core pipeline orchestration module.

This module implements the main iteration loop that coordinates:
- AI agent communication
- Model training and conversion
- Zephyr build and execution
- Metrics collection and feedback
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .boards import BoardCapabilities
from .config import PipelineConfig, TaskDescription

if TYPE_CHECKING:
    from ..agents.base import BaseAgent
    from ..data.synthetic import SyntheticDataGenerator
    from ..metrics.collector import MetricsCollector
    from ..models.evaluation import MetricScore
    from ..models.trainer import ModelTrainer
    from ..runners.base import BaseRunner, BuildResult

logger = logging.getLogger(__name__)

# Build failures that repeat on every iteration, so the run is stopped rather than
# spending the remaining iterations rediscovering them.
UNRECOVERABLE_BUILD_ERRORS = (
    'unknown command "build"',
    "west not found",
    "zephyr_base not found",
    "firmware source tree not found",
)


class IterationStatus(str, Enum):
    """Status of a pipeline iteration."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"  # Failed constraints
    STOPPED = "stopped"  # The agent ended the run instead of proposing


class FailureType(str, Enum):
    """Classification of iteration failures."""

    CONFIGURATION = "configuration"
    BUILD = "build"
    RUNTIME = "runtime"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ACCURACY_REGRESSION = "accuracy_regression"
    WEIGHT_REUSE_INCOMPATIBILITY = "weight_reuse_incompatibility"


@dataclass
class WeightMetadata:
    """Metadata about one layer's weights for AI agent context."""

    layer_name: str
    shape: tuple[int, ...]
    dtype: str
    trainable: bool
    bias_shape: tuple[int, ...] | None = None


@dataclass
class IterationResult:
    """Result of a single pipeline iteration."""

    iteration_number: int
    status: IterationStatus
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Model info
    model_architecture: str | None = None
    model_code: str | None = None

    # Metrics (populated after execution)
    accuracy_value: float | None = None

    # What the validation set could resolve, and over how many samples. Without it the
    # run cannot say whether one iteration beat another or merely drew luckier data.
    accuracy_standard_error: float | None = None
    accuracy_sample_count: int | None = None

    latency_ms: float | None = None
    ram_usage_kb: int | None = None
    flash_usage_kb: int | None = None
    arena_used_bytes: int | None = None

    # Weight reuse info
    weights_reused: bool = False
    reused_layers: list[str] = field(default_factory=list)
    weight_reuse_failures: list[dict[str, str]] = field(default_factory=list)

    # Failure info
    failure_type: FailureType | None = None
    failure_message: str | None = None

    # Set when the agent ended the run rather than proposing a model
    stop_recommendation: str | None = None

    # Execution environment
    execution_env: str = "qemu"

    # Which processor ran the inference, and how much of the model the NPU took
    accelerator: str = "cpu"
    npu_operator_count: int | None = None

    # Training budget, and whether it rather than convergence ended the fit
    epochs_trained: int | None = None
    is_under_trained: bool = False

    # Artifact paths
    artifact_dir: Path | None = None

    @property
    def score(self) -> "MetricScore | None":
        """The measured metric with its error bar, when the iteration was measured."""
        if self.accuracy_value is None or self.accuracy_standard_error is None:
            return None

        # Imported here rather than at the top of the module: the evaluation module
        # reaches back into this package for its config, and importing it eagerly closes
        # the loop.
        from ..models.evaluation import MetricScore

        return MetricScore(
            value=self.accuracy_value,
            standard_error=self.accuracy_standard_error,
            sample_count=self.accuracy_sample_count or 0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "iteration_number": self.iteration_number,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "model_architecture": self.model_architecture,
            "metrics": {
                "accuracy_value": self.accuracy_value,
                "accuracy_standard_error": self.accuracy_standard_error,
                "accuracy_sample_count": self.accuracy_sample_count,
                "latency_ms": self.latency_ms,
                "ram_usage_kb": self.ram_usage_kb,
                "flash_usage_kb": self.flash_usage_kb,
                "arena_used_bytes": self.arena_used_bytes,
            },
            "weight_reuse": {
                "reused": self.weights_reused,
                "layers": self.reused_layers,
                "failures": self.weight_reuse_failures,
            },
            "failure": {
                "type": self.failure_type.value if self.failure_type else None,
                "message": self.failure_message,
            }
            if self.failure_type
            else None,
            "stop_recommendation": self.stop_recommendation,
            "execution_env": self.execution_env,
            "accelerator": self.accelerator,
            "npu_operator_count": self.npu_operator_count,
            "epochs_trained": self.epochs_trained,
            "is_under_trained": self.is_under_trained,
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
        }

    def meets_constraints(self, config: PipelineConfig) -> tuple[bool, list[str]]:
        """Check the iteration against every hard limit.

        An unmeasured metric is a violation, not a pass: treating a missing accuracy as
        satisfied marked untested models as successes and left the agent optimising
        against a number nobody had.
        """
        violations = []

        if self.accuracy_value is None:
            violations.append("Accuracy was not measured")
        elif not config.accuracy.is_within_limit(self.accuracy_value):
            violations.append(
                f"{config.accuracy.metric.value} {self.accuracy_value:.4f} is outside "
                f"max_error {config.accuracy.max_error}"
            )

        if self.latency_ms is None:
            violations.append("Latency was not measured")
        elif self.latency_ms > config.performance.max_latency_ms:
            violations.append(
                f"Latency {self.latency_ms:.3f}ms exceeds max {config.performance.max_latency_ms}ms"
            )

        # Memory is read from the build output and may legitimately be absent for the
        # simulated runner, so it is only checked when a figure exists.
        if self.ram_usage_kb is not None and self.ram_usage_kb > config.performance.max_ram_kb:
            violations.append(
                f"RAM {self.ram_usage_kb}KB exceeds max {config.performance.max_ram_kb}KB"
            )

        if (
            self.flash_usage_kb is not None
            and self.flash_usage_kb > config.performance.max_flash_kb
        ):
            violations.append(
                f"Flash {self.flash_usage_kb}KB exceeds max {config.performance.max_flash_kb}KB"
            )

        return len(violations) == 0, violations

    def meets_targets(self, config: PipelineConfig) -> bool:
        """Check the iteration against the optimisation target, not just the limits."""
        if self.accuracy_value is None:
            return False

        meets_constraints, _ = self.meets_constraints(config)
        return config.accuracy.meets_target(self.accuracy_value) and meets_constraints


@dataclass
class PipelineState:
    """State of the pipeline across iterations."""

    config: PipelineConfig
    iterations: list[IterationResult] = field(default_factory=list)
    current_iteration: int = 0
    best_iteration: int | None = None
    stopped: bool = False
    stop_reason: str | None = None

    def get_previous_weights_metadata(self) -> list[WeightMetadata] | None:
        """Read weight metadata from the most recent successful iteration."""
        for result in reversed(self.iterations):
            if result.status == IterationStatus.SUCCESS and result.artifact_dir:
                metadata_path = result.artifact_dir / "weights_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        data = json.load(f)
                    return [WeightMetadata(**entry) for entry in data]
        return None

    def get_best_result(self) -> IterationResult | None:
        """Return the best successful iteration under the configured metric."""
        best = None

        for result in self.iterations:
            if result.status != IterationStatus.SUCCESS or result.accuracy_value is None:
                continue
            if best is None or self.config.accuracy.is_better(
                result.accuracy_value, best.accuracy_value
            ):
                best = result

        return best

    def unresolved_ranking(self) -> str | None:
        """Say when the best iteration's lead is inside the measurement's own error.

        The reported result is the best of every iteration, so sampling noise alone
        pushes it upward: whichever model happened to draw the kinder validation samples
        wins. Where the gap to the runner-up is narrower than one standard error, the
        two have not been told apart, and a bigger dataset.val_samples is the fix.
        """
        best = self.get_best_result()
        if best is None or best.score is None:
            return None

        runner_up = self._runner_up(best)
        if runner_up is None or runner_up.score is None:
            return None

        if best.score.separates(runner_up.score):
            return None

        metric = self.config.accuracy.metric.value
        return (
            f"Iteration {best.iteration_number} ({metric} {best.accuracy_value:.6f}) and "
            f"iteration {runner_up.iteration_number} ({runner_up.accuracy_value:.6f}) differ "
            f"by less than the standard error of {best.score.standard_error:.6f} on "
            f"{best.score.sample_count} validation samples, so the two are not "
            "distinguished by this measurement. Raise dataset.val_samples to rank them."
        )

    def _runner_up(self, best: IterationResult) -> IterationResult | None:
        """The second-best successful iteration, or None when there is only one."""
        others = [
            result
            for result in self.iterations
            if result is not best
            and result.status == IterationStatus.SUCCESS
            and result.accuracy_value is not None
        ]
        if not others:
            return None

        return sorted(
            others,
            key=lambda result: result.accuracy_value,
            reverse=self.config.accuracy.metric.is_higher_better,
        )[0]


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(
        self,
        config: PipelineConfig,
        agent: "BaseAgent",
        data_generator: "SyntheticDataGenerator",
        trainer: "ModelTrainer",
        runner: "BaseRunner",
        metrics_collector: "MetricsCollector",
        artifacts_dir: Path,
        task_description: TaskDescription | None = None,
        board: BoardCapabilities | None = None,
    ):
        self.config = config
        self.agent = agent
        self.data_generator = data_generator
        self.trainer = trainer
        self.runner = runner
        self.metrics_collector = metrics_collector
        self.artifacts_dir = artifacts_dir
        self.task_description = task_description or TaskDescription.discover()
        self.board = board or BoardCapabilities()

        self.state = PipelineState(config=config)
        self._input_shape: tuple[int, ...] = ()
        self._class_count: int | None = None

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def uses_npu(self) -> bool:
        """True when the run asked for an accelerator and the board has one."""
        return self.config.execution.npu and self.board.has_npu

    def run(self) -> PipelineState:
        """Run the pipeline iteration loop."""
        logger.info("Starting pipeline execution")
        logger.info(f"Max iterations: {self.config.iteration.max_iterations}")
        logger.info(f"Target {self.config.accuracy.metric.value}: {self.config.accuracy.target}")

        unreachable = self.config.unreachable_target()
        if unreachable:
            logger.warning(unreachable)

        logger.info("Generating synthetic training data")
        train_data, val_data = self.data_generator.generate()
        logger.info(f"Generated {len(train_data[0])} training samples")

        # The agent cannot design an input layer without knowing the sample's shape.
        self._input_shape = tuple(train_data[0].shape[1:])
        if self.config.synthetic_data.task.value == "classification":
            self._class_count = int(train_data[1].max()) + 1

        while not self._should_stop():
            self.state.current_iteration += 1
            iteration_number = self.state.current_iteration

            logger.info(f"=== Iteration {iteration_number} ===")

            iter_dir = self.artifacts_dir / f"iteration_{iteration_number:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            self.config.to_yaml(iter_dir / "config_snapshot.yaml")

            result = self._run_iteration(iteration_number, iter_dir, train_data, val_data)
            self.state.iterations.append(result)

            if result.status is IterationStatus.STOPPED:
                self.state.stopped = True
                self.state.stop_reason = f"Agent ended the run: {result.stop_recommendation}"
                logger.info(self.state.stop_reason)
                with open(iter_dir / "result.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                break

            if result.status == IterationStatus.SUCCESS:
                best = self.state.get_best_result()
                if best and best.iteration_number == iteration_number:
                    self.state.best_iteration = iteration_number
                    logger.info(f"New best iteration: {iteration_number}")

            with open(iter_dir / "result.json", "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            abandon_reason = self._abandon_reason(result)
            if abandon_reason:
                self.state.stopped = True
                self.state.stop_reason = abandon_reason
                logger.error(abandon_reason)
                break

            if self.config.iteration.early_stop_on_target and result.meets_targets(self.config):
                self.state.stopped = True
                self.state.stop_reason = "All targets met"
                logger.info("All targets met, stopping early")
                break

        logger.info("Generating metrics graphs")
        self.metrics_collector.generate_graphs(
            self.state.iterations,
            self.artifacts_dir / "graphs",
            self.config.get_constraints_summary(self.task_description),
        )

        unresolved = self.state.unresolved_ranking()
        if unresolved:
            logger.warning(unresolved)

        logger.info(f"Pipeline complete. Best iteration: {self.state.best_iteration}")
        return self.state

    def _should_stop(self) -> bool:
        """Check whether the iteration loop should end."""
        if self.state.stopped:
            return True
        if self.state.current_iteration >= self.config.iteration.max_iterations:
            self.state.stop_reason = "Max iterations reached"
            return True
        return False

    def _abandon_reason(self, result: IterationResult) -> str | None:
        """Why the run should stop now, or None to keep iterating."""
        if self._is_unrecoverable_failure(result):
            return f"Unrecoverable build failure: {result.failure_message}"

        repeated = self._repeated_failure_message()
        if repeated:
            return f"Same failure twice in a row, so it will not clear: {repeated}"

        return None

    def _repeated_failure_message(self) -> str | None:
        """The message of a failure that has now happened twice running.

        A misconfigured agent or an absent toolchain fails identically every time. Left
        alone it consumes the whole iteration budget, and each attempt may cost money.
        """
        recent = self.state.iterations[-2:]

        if len(recent) < 2 or any(r.status != IterationStatus.FAILED for r in recent):
            return None

        first, second = recent
        if first.failure_message and first.failure_message == second.failure_message:
            return second.failure_message

        return None

    @staticmethod
    def _is_unrecoverable_failure(result: IterationResult) -> bool:
        """True for build failures that every later iteration would hit as well."""
        if result.status != IterationStatus.FAILED or result.failure_type != FailureType.BUILD:
            return False
        if not result.failure_message:
            return False

        message = result.failure_message.lower()
        return any(error in message for error in UNRECOVERABLE_BUILD_ERRORS)

    def _run_iteration(
        self,
        iteration_number: int,
        iter_dir: Path,
        train_data: tuple,
        val_data: tuple,
    ) -> IterationResult:
        """Run a single pipeline iteration."""
        result = IterationResult(
            iteration_number=iteration_number,
            status=IterationStatus.RUNNING,
            artifact_dir=iter_dir,
        )

        try:
            context = self._build_agent_context()

            logger.info("Requesting model proposal from AI agent")
            proposal = self.agent.propose_model(context)

            if proposal.should_stop:
                # No model was proposed, so nothing is trained, built or flashed.
                result.status = IterationStatus.STOPPED
                result.stop_recommendation = proposal.stop_reason
                self.metrics_collector.record(result)
                return result

            result.model_architecture = proposal.architecture_description
            result.model_code = proposal.model_code

            (iter_dir / "model_code.py").write_text(proposal.model_code)

            if proposal.weight_reuse_layers:
                result.weights_reused = True
                result.reused_layers = proposal.weight_reuse_layers

            logger.info("Training model")
            model, training_metrics = self.trainer.train(
                model_code=proposal.model_code,
                train_data=train_data,
                val_data=val_data,
                weight_reuse_layers=proposal.weight_reuse_layers,
                previous_weights_path=self._get_previous_weights_path(),
            )

            result.epochs_trained = training_metrics.epochs_trained
            result.is_under_trained = (
                training_metrics.epochs_trained >= self.config.training.epochs
            )

            if training_metrics.weight_reuse_results:
                result.reused_layers = [
                    reuse.layer_name
                    for reuse in training_metrics.weight_reuse_results
                    if reuse.success
                ]
                result.weight_reuse_failures = [
                    {"layer_name": reuse.layer_name, "reason": reuse.reason}
                    for reuse in training_metrics.weight_reuse_results
                    if not reuse.success
                ]
                if result.weight_reuse_failures:
                    logger.warning(
                        f"Weight reuse failed for {len(result.weight_reuse_failures)} layers"
                    )

            self.trainer.save_weights(model, iter_dir / "weights.keras")
            weights_metadata = self.trainer.get_weights_metadata(model)
            with open(iter_dir / "weights_metadata.json", "w") as f:
                json.dump([entry.__dict__ for entry in weights_metadata], f, indent=2)

            self.trainer.save_model_visualization(model, iter_dir / "model_architecture.png")

            logger.info("Converting to TFLite")
            tflite_path = iter_dir / "model.tflite"
            self.trainer.convert_to_tflite(
                model, tflite_path, self.config.quantization, representative_data=train_data[0]
            )

            model_for_target = self._compile_for_accelerator(tflite_path, iter_dir, result)

            logger.info("Building Zephyr application")
            build_dir = self.config.build.get_build_path() / f"iteration_{iteration_number:04d}"
            build_dir.mkdir(parents=True, exist_ok=True)
            build_result = self.runner.build(
                model_for_target, build_dir, host_model_path=tflite_path
            )

            self._copy_build_artifacts(iter_dir, build_result)

            if not build_result.success:
                result.status = IterationStatus.FAILED
                result.failure_type = FailureType.BUILD
                result.failure_message = build_result.error_message
                self.metrics_collector.record(result)
                return result

            result.flash_usage_kb = build_result.flash_usage_kb
            result.ram_usage_kb = build_result.ram_usage_kb

            logger.info("Running inference")
            exec_result = self.runner.run(build_dir, val_data)

            if not exec_result.success:
                result.status = IterationStatus.FAILED
                result.failure_type = FailureType.RUNTIME
                result.failure_message = exec_result.error_message
                self.metrics_collector.record(result)
                return result

            result.latency_ms = exec_result.latency_ms
            result.accuracy_value = exec_result.accuracy_value
            result.accuracy_standard_error = exec_result.accuracy_standard_error
            result.accuracy_sample_count = exec_result.accuracy_sample_count
            result.arena_used_bytes = exec_result.arena_used_bytes
            result.execution_env = exec_result.execution_env

            meets_constraints, violations = result.meets_constraints(self.config)
            if not meets_constraints:
                result.status = IterationStatus.REJECTED
                result.failure_type = FailureType.CONSTRAINT_VIOLATION
                result.failure_message = "; ".join(violations)
                logger.warning(f"Iteration rejected: {result.failure_message}")
            else:
                result.status = IterationStatus.SUCCESS
                measured = result.score or f"{result.accuracy_value:.6f}"
                logger.info(
                    f"Iteration successful: {self.config.accuracy.metric.value}={measured}, "
                    f"latency={result.latency_ms:.3f}ms"
                )

            self.metrics_collector.record(result)

        except Exception as error:
            logger.exception(f"Iteration {iteration_number} failed with exception")
            result.status = IterationStatus.FAILED
            result.failure_type = FailureType.RUNTIME
            result.failure_message = str(error)
            self.metrics_collector.record(result)

        return result

    def _compile_for_accelerator(
        self,
        tflite_path: Path,
        iter_dir: Path,
        result: IterationResult,
    ) -> Path:
        """Compile the model for the board's accelerator, when the run uses one.

        Returns the model the firmware should embed: the compiled one when an
        accelerator is in play, the original otherwise.
        """
        if not self.uses_npu:
            return tflite_path

        from ..models.neutron import compile_for_neutron

        logger.info(f"Compiling for the {self.board.accelerator.value} accelerator")
        npu_path = iter_dir / "model_npu.tflite"
        neutron = compile_for_neutron(tflite_path, npu_path, self.board.neutron_target)

        result.accelerator = self.board.accelerator.value
        result.npu_operator_count = neutron.converted_operator_count

        return npu_path

    def _build_agent_context(self) -> dict[str, Any]:
        """Build the context dictionary handed to the AI agent."""
        context = {
            "constraints": self.config.get_constraints_summary(self.task_description),
            "iteration": self.state.current_iteration,
            "max_iterations": self.config.iteration.max_iterations,
            "task": {
                "type": self.config.synthetic_data.task.value,
                "function": self.config.synthetic_data.function.value,
            },
            "may_stop": self.config.iteration.allow_agent_stop,
            "dataset": self.config.dataset.source.value,
            "input_shape": list(self._input_shape),
            "class_count": self._class_count,
        }

        if self.state.iterations:
            context["previous_iteration"] = self.state.iterations[-1].to_dict()

        weights_metadata = self.state.get_previous_weights_metadata()
        if weights_metadata:
            context["available_weights"] = [
                {
                    "layer_name": entry.layer_name,
                    "shape": entry.shape,
                    "bias_shape": entry.bias_shape,
                    "dtype": entry.dtype,
                    "trainable": entry.trainable,
                }
                for entry in weights_metadata
            ]

        best = self.state.get_best_result()
        if best:
            context["best_so_far"] = {
                "iteration": best.iteration_number,
                "accuracy": best.accuracy_value,
                "standard_error": best.accuracy_standard_error,
                "sample_count": best.accuracy_sample_count,
                "latency_ms": best.latency_ms,
            }

        # Told outright when the scores it is comparing are inside the measurement's own
        # error, so it spends the remaining iterations on something that would show.
        context["unresolved_ranking"] = self.state.unresolved_ranking()

        return context

    def _get_previous_weights_path(self) -> Path | None:
        """Path to the most recent successful iteration's saved model."""
        for result in reversed(self.state.iterations):
            if result.status == IterationStatus.SUCCESS and result.artifact_dir:
                weights_path = result.artifact_dir / "weights.keras"
                if weights_path.exists():
                    return weights_path
        return None

    def _copy_build_artifacts(self, iter_dir: Path, build_result: "BuildResult") -> None:
        """Copy the Zephyr build outputs into the iteration artifact directory.

        The runners share one build directory between iterations, so the outputs are
        taken from the directory the build reports rather than the one it was offered.
        """
        build_artifacts_dir = iter_dir / "zephyr_build"
        build_artifacts_dir.mkdir(parents=True, exist_ok=True)

        if build_result.build_log:
            (build_artifacts_dir / "build.log").write_text(build_result.build_log)

        # Only on success: the build directory is shared between iterations, so after a
        # failed build it still holds the previous iteration's binaries, and copying
        # those would label one iteration's output as another's.
        if build_result.success and build_result.build_dir:
            zephyr_dir = build_result.build_dir / "zephyr"
            for name, destination in (
                ("zephyr.map", "zephyr.map"),
                ("zephyr.elf", "zephyr.elf"),
                ("zephyr.bin", "zephyr.bin"),
                (".config", "zephyr.config"),
            ):
                source = zephyr_dir / name
                if source.exists():
                    shutil.copy(source, build_artifacts_dir / destination)

        if build_result.success and build_result.app_dir and build_result.app_dir.exists():
            app_artifacts_dir = build_artifacts_dir / "app_src"
            if app_artifacts_dir.exists():
                shutil.rmtree(app_artifacts_dir)
            shutil.copytree(build_result.app_dir, app_artifacts_dir)

            # Kept at the top level as well: the specification calls for the compiled
            # model to be inspectable without digging through the build tree.
            model_files_dir = iter_dir / "model_files"
            model_files_dir.mkdir(parents=True, exist_ok=True)
            for name in ("model.cpp", "model.hpp"):
                source = build_result.app_dir / "src" / name
                if source.exists():
                    shutil.copy(source, model_files_dir / name)

        logger.info(f"Copied build artifacts to {build_artifacts_dir}")
