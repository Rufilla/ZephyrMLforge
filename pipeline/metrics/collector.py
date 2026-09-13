# _____________________________________________________________________________
#
# @file collector.py
# @brief Metrics collection and graphing module
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Metrics collection and graphing module.

Collects iteration metrics and generates visualization graphs for tracking
optimization progress.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from ..core.pipeline import IterationResult

logger = logging.getLogger(__name__)

PREFERRED_STYLE = "seaborn-v0_8-whitegrid"


@dataclass
class MetricRecord:
    """Record of metrics for a single iteration."""

    iteration: int
    timestamp: str
    status: str
    accuracy_value: float | None
    accuracy_standard_error: float | None
    latency_ms: float | None
    ram_usage_kb: int | None
    flash_usage_kb: int | None
    arena_used_bytes: int | None
    execution_env: str
    weights_reused: bool


class MetricsCollector:
    """Collects and manages iteration metrics."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir
        self.records: list[MetricRecord] = []

    def record(self, result: "IterationResult") -> None:
        """Record metrics from an iteration result."""
        self.records.append(
            MetricRecord(
                iteration=result.iteration_number,
                timestamp=result.timestamp,
                status=result.status.value,
                accuracy_value=result.accuracy_value,
                accuracy_standard_error=result.accuracy_standard_error,
                latency_ms=result.latency_ms,
                ram_usage_kb=result.ram_usage_kb,
                flash_usage_kb=result.flash_usage_kb,
                arena_used_bytes=result.arena_used_bytes,
                execution_env=result.execution_env,
                weights_reused=result.weights_reused,
            )
        )

        logger.info(
            f"Recorded iteration {result.iteration_number}: "
            f"metric={result.accuracy_value}, latency={result.latency_ms}ms"
        )

        if self.output_dir:
            self._save_metrics()

    def _save_metrics(self) -> None:
        """Write the accumulated records to metrics.json."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump([asdict(record) for record in self.records], f, indent=2)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to a pandas DataFrame."""
        return pd.DataFrame([asdict(record) for record in self.records])

    def generate_graphs(
        self,
        results: list["IterationResult"],
        output_dir: Path,
        constraints: dict | None = None,
    ) -> list[Path]:
        """
        Generate visualization graphs from iteration results.

        Args:
            results: List of iteration results
            output_dir: Directory to save graphs
            constraints: Constraint summary; without it the target and limit lines,
                the axis labels and the metric direction all fall back to defaults

        Returns:
            List of generated graph file paths
        """
        logger.info(f"Generating graphs in {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(
            [
                {
                    "iteration": result.iteration_number,
                    "accuracy": result.accuracy_value,
                    "accuracy_error": result.accuracy_standard_error,
                    "latency_ms": result.latency_ms,
                    "ram_kb": result.ram_usage_kb,
                    "flash_kb": result.flash_usage_kb,
                    "status": result.status.value,
                }
                for result in results
            ]
        )

        self._apply_style()

        plots = {
            "accuracy_vs_iteration.png": self._plot_accuracy_vs_iteration,
            "latency_vs_iteration.png": self._plot_latency_vs_iteration,
            "accuracy_vs_latency.png": self._plot_accuracy_vs_latency,
            "memory_vs_iteration.png": self._plot_memory_vs_iteration,
        }

        generated_files = []
        for filename, plot in plots.items():
            figure, axes = plt.subplots(figsize=(10, 6))
            plot(frame, axes, constraints)
            path = output_dir / filename
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            generated_files.append(path)
            logger.info(f"Generated: {path}")

        # One dashboard figure, saved in both formats: rendering it twice doubled the
        # slowest part of graphing for an identical picture.
        dashboard = self._create_dashboard(frame, constraints)
        for filename, options in (("dashboard.png", {"dpi": 150}), ("dashboard.svg", {})):
            path = output_dir / filename
            dashboard.savefig(path, bbox_inches="tight", **options)
            generated_files.append(path)
            logger.info(f"Generated: {path}")
        plt.close(dashboard)

        logger.info(f"Generated {len(generated_files)} graphs")
        return generated_files

    @staticmethod
    def _apply_style() -> None:
        """Apply the preferred style, falling back to the default when unavailable."""
        try:
            plt.style.use(PREFERRED_STYLE)
        except OSError:
            logger.debug(f"Style {PREFERRED_STYLE} unavailable; using the matplotlib default")

    @staticmethod
    def _metric_label(constraints: dict | None) -> str:
        """Axis label for the configured metric."""
        if not constraints:
            return "Error"
        return constraints.get("accuracy", {}).get("metric", "error").upper()

    @staticmethod
    def _is_higher_better(constraints: dict | None) -> bool:
        """True when a larger metric value is a better model."""
        if not constraints:
            return False
        return bool(constraints.get("accuracy", {}).get("higher_is_better", False))

    @staticmethod
    def _annotate_iterations(
        axes: plt.Axes,
        frame: pd.DataFrame,
        x_column: str,
        y_column: str,
        offset: tuple[int, int] = (0, 10),
        color: str | None = None,
    ) -> None:
        """Label each point with its iteration number."""
        for _, row in frame.iterrows():
            axes.annotate(
                f"{int(row['iteration'])}",
                (row[x_column], row[y_column]),
                textcoords="offset points",
                xytext=offset,
                ha="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    def _plot_accuracy_vs_iteration(
        self, frame: pd.DataFrame, axes: plt.Axes, constraints: dict | None
    ) -> None:
        """Plot the accuracy or error metric against iteration."""
        label = self._metric_label(constraints)
        valid = frame.dropna(subset=["accuracy"])

        if valid.empty:
            axes.text(0.5, 0.5, "No accuracy data", ha="center", va="center")
            axes.set_title(f"{label} vs Iteration")
            return

        # Drawn with its error bars: a bare point invites reading a wobble of a fraction
        # of a point as progress, when another draw of the validation set would move it
        # as far on its own.
        error = valid["accuracy_error"].fillna(0.0)
        axes.errorbar(
            valid["iteration"],
            valid["accuracy"],
            yerr=error if error.any() else None,
            marker="o",
            linewidth=2,
            markersize=8,
            capsize=4,
            color="#2196F3",
            label=label,
        )
        # Beside the point rather than above it, where the error bar's upper cap now is.
        self._annotate_iterations(axes, valid, "iteration", "accuracy", offset=(10, 0))

        if constraints:
            accuracy = constraints.get("accuracy", {})
            if "target" in accuracy:
                axes.axhline(
                    y=accuracy["target"],
                    color="green",
                    linestyle="--",
                    label=f"Target: {accuracy['target']}",
                )
            if "max_error" in accuracy:
                axes.axhline(
                    y=accuracy["max_error"],
                    color="red",
                    linestyle="--",
                    label=f"Rejected beyond: {accuracy['max_error']}",
                )

        axes.set_xlabel("Iteration", fontsize=12)
        axes.set_ylabel(label, fontsize=12)
        axes.set_title(f"Model {label} vs Iteration", fontsize=14)
        axes.legend()

        # A log axis suits an error falling towards zero; an accuracy in [0, 1] is
        # unreadable on one.
        if not self._is_higher_better(constraints):
            axes.set_yscale("log")

    def _plot_latency_vs_iteration(
        self, frame: pd.DataFrame, axes: plt.Axes, constraints: dict | None
    ) -> None:
        """Plot latency against iteration."""
        valid = frame.dropna(subset=["latency_ms"])

        if valid.empty:
            axes.text(0.5, 0.5, "No latency data", ha="center", va="center")
            axes.set_title("Latency vs Iteration")
            return

        axes.plot(
            valid["iteration"],
            valid["latency_ms"],
            marker="s",
            linewidth=2,
            markersize=8,
            color="#FF9800",
            label="Latency",
        )
        self._annotate_iterations(axes, valid, "iteration", "latency_ms")

        if constraints:
            performance = constraints.get("performance", {})
            if "max_latency_ms" in performance:
                axes.axhline(
                    y=performance["max_latency_ms"],
                    color="red",
                    linestyle="--",
                    label=f"Max: {performance['max_latency_ms']}ms",
                )

        axes.set_xlabel("Iteration", fontsize=12)
        axes.set_ylabel("Latency (ms)", fontsize=12)
        axes.set_title("Inference Latency vs Iteration", fontsize=14)
        axes.legend()

    def _plot_accuracy_vs_latency(
        self, frame: pd.DataFrame, axes: plt.Axes, constraints: dict | None
    ) -> None:
        """Plot the accuracy and latency trade-off."""
        label = self._metric_label(constraints)
        valid = frame.dropna(subset=["accuracy", "latency_ms"])

        if valid.empty:
            axes.text(0.5, 0.5, "No data", ha="center", va="center")
            axes.set_title(f"{label} vs Latency Trade-off")
            return

        scatter = axes.scatter(
            valid["latency_ms"],
            valid["accuracy"],
            c=valid["iteration"],
            cmap="viridis",
            s=100,
            edgecolors="black",
            linewidths=0.5,
        )
        self._annotate_iterations(axes, valid, "latency_ms", "accuracy", offset=(8, 0))

        colorbar = plt.colorbar(scatter, ax=axes)
        colorbar.set_label("Iteration")

        best_index = (
            valid["accuracy"].idxmax()
            if self._is_higher_better(constraints)
            else valid["accuracy"].idxmin()
        )
        best_latency = valid.loc[best_index, "latency_ms"]
        best_accuracy = valid.loc[best_index, "accuracy"]

        axes.scatter(
            best_latency,
            best_accuracy,
            marker="*",
            s=300,
            c="red",
            edgecolors="black",
            linewidths=1,
            zorder=5,
        )

        # Annotated rather than put in a legend: a one-entry legend for a star marker
        # lands inside the axes and reads as another measurement.
        axes.annotate(
            "best",
            (best_latency, best_accuracy),
            textcoords="offset points",
            xytext=(0, -18),
            ha="center",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

        if constraints:
            performance = constraints.get("performance", {})
            accuracy = constraints.get("accuracy", {})
            if "max_latency_ms" in performance:
                axes.axvline(
                    x=performance["max_latency_ms"], color="red", linestyle="--", alpha=0.7
                )
            if "max_error" in accuracy:
                axes.axhline(y=accuracy["max_error"], color="red", linestyle="--", alpha=0.7)

        axes.set_xlabel("Latency (ms)", fontsize=12)
        axes.set_ylabel(label, fontsize=12)
        axes.set_title(f"{label} vs Latency Trade-off", fontsize=14)

        if not self._is_higher_better(constraints):
            axes.set_yscale("log")

    def _plot_memory_vs_iteration(
        self, frame: pd.DataFrame, axes: plt.Axes, constraints: dict | None
    ) -> None:
        """Plot RAM and flash usage against iteration."""
        ram = frame.dropna(subset=["ram_kb"])
        flash = frame.dropna(subset=["flash_kb"])

        if ram.empty and flash.empty:
            axes.text(0.5, 0.5, "No memory data", ha="center", va="center")
            axes.set_title("Memory Usage vs Iteration")
            return

        if not ram.empty:
            axes.plot(
                ram["iteration"],
                ram["ram_kb"],
                marker="o",
                linewidth=2,
                markersize=8,
                color="#4CAF50",
                label="RAM (KB)",
            )
            self._annotate_iterations(axes, ram, "iteration", "ram_kb", color="#4CAF50")

        if not flash.empty:
            axes.plot(
                flash["iteration"],
                flash["flash_kb"],
                marker="s",
                linewidth=2,
                markersize=8,
                color="#9C27B0",
                label="Flash (KB)",
            )

        if constraints:
            performance = constraints.get("performance", {})
            if "max_ram_kb" in performance:
                axes.axhline(
                    y=performance["max_ram_kb"],
                    color="green",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Max RAM: {performance['max_ram_kb']}KB",
                )
            if "max_flash_kb" in performance:
                axes.axhline(
                    y=performance["max_flash_kb"],
                    color="purple",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Max Flash: {performance['max_flash_kb']}KB",
                )

        axes.set_xlabel("Iteration", fontsize=12)
        axes.set_ylabel("Memory (KB)", fontsize=12)
        axes.set_title("Memory Usage vs Iteration", fontsize=14)
        axes.legend()

    def _create_dashboard(self, frame: pd.DataFrame, constraints: dict | None) -> plt.Figure:
        """Combine every plot into one figure."""
        figure, axes = plt.subplots(2, 2, figsize=(14, 10))
        figure.suptitle("Pipeline Optimization Dashboard", fontsize=16, fontweight="bold")

        self._plot_accuracy_vs_iteration(frame, axes[0, 0], constraints)
        self._plot_latency_vs_iteration(frame, axes[0, 1], constraints)
        self._plot_accuracy_vs_latency(frame, axes[1, 0], constraints)
        self._plot_memory_vs_iteration(frame, axes[1, 1], constraints)

        figure.tight_layout()
        return figure

    @classmethod
    def from_artifacts(cls, artifacts_dir: Path) -> "MetricsCollector":
        """
        Load metrics from the result.json files under an artifacts directory.

        Args:
            artifacts_dir: Directory containing iteration artifacts

        Returns:
            MetricsCollector with loaded records
        """
        collector = cls(output_dir=artifacts_dir)

        for iteration_dir in sorted(artifacts_dir.glob("iteration_*")):
            result_path = iteration_dir / "result.json"
            if not result_path.exists():
                continue

            with open(result_path) as f:
                data = json.load(f)

            metrics = data["metrics"]
            collector.records.append(
                MetricRecord(
                    iteration=data["iteration_number"],
                    timestamp=data["timestamp"],
                    status=data["status"],
                    accuracy_value=metrics["accuracy_value"],
                    latency_ms=metrics["latency_ms"],
                    ram_usage_kb=metrics["ram_usage_kb"],
                    flash_usage_kb=metrics["flash_usage_kb"],
                    arena_used_bytes=metrics.get("arena_used_bytes"),
                    execution_env=data["execution_env"],
                    weights_reused=data["weight_reuse"]["reused"],
                )
            )

        logger.info(f"Loaded {len(collector.records)} records from artifacts")
        return collector
