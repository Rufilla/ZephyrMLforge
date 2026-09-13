# _____________________________________________________________________________
#
# @file cli.py
# @brief Command-line interface for the Embedded AI Pipeline
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Command-line interface for the Embedded AI Pipeline.

Provides entry points for running the pipeline and related utilities.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .agents import create_agent
from .core.boards import BoardCapabilities, BoardRegistry
from .core.config import ExecutionMode, PipelineConfig, TaskDescription
from .core.output import OutputSettings, Verbosity, configure_output
from .core.pipeline import Pipeline
from .data.datasets import create_data_source
from .data.synthetic import SyntheticDataGenerator
from .metrics.collector import MetricsCollector
from .models.trainer import ModelTrainer
from .runners import AutoRunner, HILRunner, QEMURunner, SimulatedRunner

console = Console()

# Directories searched for a Zephyr SDK when ZEPHYR_SDK_INSTALL_DIR is unset.
SDK_SEARCH_ROOTS = (Path.home(), Path.home() / ".local", Path("/opt"))
SDK_DIRECTORY_GLOB = "zephyr-sdk-*"

# Previous runs kept as zip archives beside the artifacts directory.
ARCHIVES_KEPT = 5


@dataclass
class ZephyrEnvironment:
    """Zephyr toolchain and SDK environment information."""

    zephyr_base: Path | None
    zephyr_sdk: Path | None
    zephyr_base_valid: bool
    zephyr_sdk_valid: bool
    sdk_auto_detected: bool = False

    @property
    def is_valid(self) -> bool:
        """True when ZEPHYR_BASE is usable; the SDK is recommended but not required."""
        return self.zephyr_base_valid

    @classmethod
    def detect(cls) -> "ZephyrEnvironment":
        """Detect the Zephyr environment from environment variables and common paths."""
        base_path = os.environ.get("ZEPHYR_BASE", "")
        sdk_path = os.environ.get("ZEPHYR_SDK_INSTALL_DIR", "")

        zephyr_base = Path(base_path) if base_path else None
        zephyr_sdk = Path(sdk_path) if sdk_path else None
        sdk_auto_detected = False

        if zephyr_sdk is None or not zephyr_sdk.exists():
            zephyr_sdk = cls._find_sdk()
            sdk_auto_detected = zephyr_sdk is not None

        return cls(
            zephyr_base=zephyr_base,
            zephyr_sdk=zephyr_sdk,
            zephyr_base_valid=zephyr_base is not None and zephyr_base.exists(),
            zephyr_sdk_valid=zephyr_sdk is not None and zephyr_sdk.exists(),
            sdk_auto_detected=sdk_auto_detected,
        )

    @staticmethod
    def _find_sdk() -> Path | None:
        """Find the newest installed SDK.

        Globbed rather than listed by version: a hardcoded list of versions stops
        finding the SDK the day a new one is released.
        """
        candidates = [
            path
            for root in SDK_SEARCH_ROOTS
            if root.is_dir()
            for path in root.glob(SDK_DIRECTORY_GLOB)
            if path.is_dir()
        ]
        return max(candidates, key=lambda path: path.name) if candidates else None


def setup_logging(verbosity: Verbosity) -> None:
    """Configure logging and third-party output for the chosen verbosity.

    Runs before TensorFlow is imported; its C++ logger reads its level once, at import.
    """
    logging.basicConfig(
        level=verbosity.log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )

    configure_output(verbosity)


def resolve_verbosity(args: argparse.Namespace) -> Verbosity:
    """Command-line flags override the pyproject.toml setting."""
    if getattr(args, "quiet", False):
        return Verbosity.QUIET
    if getattr(args, "verbose", False):
        return Verbosity.VERBOSE
    return OutputSettings.discover().verbosity


def _archive_existing_artifacts(artifacts_dir: Path) -> Path | None:
    """Archive an existing artifacts directory to a timestamped zip.

    Returns:
        Path to the created archive, or None when there was nothing to archive.
    """
    if not artifacts_dir.exists() or not any(artifacts_dir.iterdir()):
        return None

    archive_stem = artifacts_dir.parent / f"artifacts_{datetime.now():%Y%m%d_%H%M%S}"
    console.print(f"[yellow]Archiving existing artifacts to {archive_stem.name}.zip[/yellow]")

    shutil.make_archive(str(archive_stem), "zip", artifacts_dir)
    shutil.rmtree(artifacts_dir)

    archive_path = Path(f"{archive_stem}.zip")
    console.print(f"[green]Archived previous run to {archive_path}[/green]")

    _prune_old_archives(artifacts_dir.parent)
    return archive_path


def _prune_old_archives(directory: Path) -> None:
    """Delete all but the most recent archives.

    Each archive carries every iteration's ELF, binary and map file, so an unbounded
    series of them fills the disk of anyone iterating often.
    """
    archives = sorted(directory.glob("artifacts_*.zip"))

    for stale in archives[:-ARCHIVES_KEPT]:
        stale.unlink()
        console.print(f"[dim]Removed old archive {stale.name}[/dim]")


def _apply_cli_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    """Apply command-line overrides onto the loaded configuration."""
    if args.hil:
        config.execution_environment.mode = ExecutionMode.HIL
    if args.serial_port:
        config.execution_environment.serial_port = args.serial_port


def _create_runner(config: PipelineConfig, args: argparse.Namespace, zephyr_base: Path | None):
    """Select the runner for this run and say which one was chosen."""
    if args.simulate:
        _announce(args, "[yellow]Using simulated runner (no Zephyr build)[/yellow]")
        return SimulatedRunner(config)

    mode = config.execution_environment.mode
    port = config.execution_environment.serial_port

    if mode == ExecutionMode.HIL:
        _announce(args, f"[green]Using HIL runner on {port}[/green]")
        return HILRunner(config, zephyr_base)

    if mode == ExecutionMode.AUTO:
        _announce(args, f"[green]Using auto runner: QEMU first, then HIL on {port}[/green]")
        return AutoRunner(config, zephyr_base)

    _announce(args, f"[green]Using QEMU runner on {config.hardware.qemu_board}[/green]")
    return QEMURunner(config, zephyr_base)


def _announce(args: argparse.Namespace, message: str) -> None:
    """Print progress that is not a warning, unless the run asked for quiet."""
    if getattr(args, "verbosity", Verbosity.NORMAL) is not Verbosity.QUIET:
        console.print(message)


def cmd_run(args: argparse.Namespace) -> int:
    """Run the pipeline."""
    logger = logging.getLogger(__name__)

    try:
        config = PipelineConfig.from_yaml(args.config)
    except Exception as error:
        console.print(f"[red]Error loading config: {error}[/red]")
        return 1

    _announce(args, f"[green]Loaded configuration from {args.config}[/green]")
    _apply_cli_overrides(config, args)

    zephyr_env = ZephyrEnvironment.detect()

    if not args.simulate and not zephyr_env.is_valid:
        console.print("\n[bold red]Zephyr environment check failed[/bold red]")
        if zephyr_env.zephyr_base:
            console.print(f"[red]  ZEPHYR_BASE does not exist: {zephyr_env.zephyr_base}[/red]")
        else:
            console.print("[red]  ZEPHYR_BASE is not set[/red]")
        console.print("\n[yellow]Set up the Zephyr environment, or pass --simulate.[/yellow]")
        console.print(
            "[yellow]See: https://docs.zephyrproject.org/latest/develop/getting_started/[/yellow]"
        )
        return 1

    if not args.simulate and not zephyr_env.zephyr_sdk_valid:
        console.print(
            "[yellow]WARNING: no Zephyr SDK found. Builds fail unless the toolchain "
            "is already on PATH.[/yellow]"
        )

    board = BoardRegistry.discover().for_board(config.hardware.board)

    if config.execution.npu and args.simulate:
        _announce(
            args,
            "[yellow]Ignoring execution.npu: the simulated runner builds no "
            "firmware.[/yellow]",
        )
        board = BoardCapabilities()
    elif config.execution.npu and not board.has_npu:
        console.print(
            f"[red]execution.npu is set, but {config.hardware.board} has no accelerator "
            "declared in pyproject.toml.[/red]"
        )
        console.print(
            "[yellow]Boards with one: "
            f"{', '.join(BoardRegistry.discover().boards_with_npu()) or 'none declared'}. "
            "Set execution.npu to false, or add the board under "
            r"\[tool.zephyr-ml-forge.boards].[/yellow]"
        )
        return 1
    elif config.execution.npu:
        _announce(
            args,
            f"[green]Accelerator: {board.accelerator.value} "
            f"(compiler target {board.neutron_target})[/green]",
        )

    if getattr(args, "verbosity", Verbosity.NORMAL) is not Verbosity.QUIET:
        _print_config_summary(config, zephyr_env)

    artifacts_dir = Path(args.artifacts_dir)
    _archive_existing_artifacts(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        agent = create_agent(config.ai_agent)
        _announce(
            args,
            f"[green]AI agent: {config.ai_agent.provider.value} ({agent.model})[/green]",
        )

        data_generator = create_data_source(
            config.dataset,
            SyntheticDataGenerator(config.synthetic_data),
            seed=config.synthetic_data.seed,
        )
        trainer = ModelTrainer(config.training, seed=config.synthetic_data.seed)
        runner = _create_runner(config, args, zephyr_env.zephyr_base)
        runner.uses_neutron = config.execution.npu and board.has_npu
        metrics_collector = MetricsCollector(output_dir=artifacts_dir)
    except Exception as error:
        console.print(f"[red]Error initializing components: {error}[/red]")
        logger.exception("Initialization failed")
        return 1

    pipeline = Pipeline(
        config=config,
        agent=agent,
        data_generator=data_generator,
        trainer=trainer,
        runner=runner,
        metrics_collector=metrics_collector,
        artifacts_dir=artifacts_dir,
        task_description=TaskDescription.discover(
            Path(args.config).resolve().parent, source=config.dataset.source.value
        ),
        board=board,
    )

    _announce(args, "\n[bold blue]Starting pipeline execution[/bold blue]\n")

    try:
        state = pipeline.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        return 130
    except Exception as error:
        console.print(f"[red]Pipeline failed: {error}[/red]")
        logger.exception("Pipeline failed")
        return 1

    _print_results_summary(state, config)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a configuration file."""
    try:
        config = PipelineConfig.from_yaml(args.config)
    except Exception as error:
        console.print(f"[red]Configuration error: {error}[/red]")
        return 1

    console.print(f"[green]Configuration is valid: {args.config}[/green]")
    _print_config_summary(config)
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    """Visualize a TFLite model."""
    from .models.visualizer import visualize_tflite

    if not args.model.exists():
        console.print(f"[red]Model not found: {args.model}[/red]")
        return 1

    try:
        result = visualize_tflite(
            model_path=args.model,
            method=args.method,
            output_path=args.output,
            open_browser=not args.no_browser,
        )
    except Exception as error:
        console.print(f"[red]Visualization failed: {error}[/red]")
        return 1

    if result and args.method in ("summary", "json"):
        console.print(result)
    if args.output:
        console.print(f"[green]Output saved to {args.output}[/green]")

    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    """Generate graphs from existing artifacts."""
    from .core.pipeline import IterationResult, IterationStatus

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        console.print(f"[red]Artifacts directory not found: {artifacts_dir}[/red]")
        return 1

    collector = MetricsCollector.from_artifacts(artifacts_dir)
    if not collector.records:
        console.print("[yellow]No metrics found in artifacts[/yellow]")
        return 1

    constraints = None
    snapshot_path = next(iter(sorted(artifacts_dir.glob("iteration_*/config_snapshot.yaml"))), None)
    if snapshot_path:
        constraints = PipelineConfig.from_yaml(snapshot_path).get_constraints_summary()

    results = [
        IterationResult(
            iteration_number=record.iteration,
            status=IterationStatus(record.status),
            timestamp=record.timestamp,
            accuracy_value=record.accuracy_value,
            latency_ms=record.latency_ms,
            ram_usage_kb=record.ram_usage_kb,
            flash_usage_kb=record.flash_usage_kb,
            arena_used_bytes=record.arena_used_bytes,
            execution_env=record.execution_env,
            weights_reused=record.weights_reused,
        )
        for record in collector.records
    ]

    output_dir = artifacts_dir / "graphs"
    files = collector.generate_graphs(results, output_dir, constraints)
    console.print(f"[green]Generated {len(files)} graphs in {output_dir}[/green]")
    return 0


def _print_config_summary(
    config: PipelineConfig, zephyr_env: ZephyrEnvironment | None = None
) -> None:
    """Print a configuration summary table."""
    table = Table(title="Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    if zephyr_env:
        table.add_row(
            "Zephyr Base",
            str(zephyr_env.zephyr_base) if zephyr_env.zephyr_base_valid else "[red]Not found[/red]",
        )
        if zephyr_env.zephyr_sdk_valid:
            suffix = " [dim](auto-detected)[/dim]" if zephyr_env.sdk_auto_detected else ""
            table.add_row("Zephyr SDK", f"{zephyr_env.zephyr_sdk}{suffix}")
        else:
            table.add_row("Zephyr SDK", "[yellow]Not found (optional)[/yellow]")

    rows = (
        ("Board", config.hardware.board),
        ("RAM", f"{config.hardware.ram_kb} KB"),
        ("Flash", f"{config.hardware.flash_kb} KB"),
        ("Tensor arena", f"{config.hardware.tensor_arena_kb} KB"),
        ("NPU Enabled", str(config.execution.npu)),
        ("Exec Mode", config.execution_environment.mode.value),
        ("Serial Port", config.execution_environment.serial_port),
        ("AI Provider", config.ai_agent.provider.value),
        ("AI Model", config.ai_agent.model or "provider default"),
        ("Max Iterations", str(config.iteration.max_iterations)),
        ("Metric", config.accuracy.metric.value),
        ("Target", str(config.accuracy.target)),
        ("Rejected beyond", str(config.accuracy.max_error)),
        ("On-device accuracy", str(config.accuracy.evaluate_on_device)),
        ("Max Latency", f"{config.performance.max_latency_ms} ms"),
        ("Max RAM", f"{config.performance.max_ram_kb} KB"),
        ("Max Flash", f"{config.performance.max_flash_kb} KB"),
        ("Dataset", config.dataset.source.value),
        ("Task Type", config.synthetic_data.task.value),
        ("Data Function", config.synthetic_data.function.value),
        ("Samples", str(config.synthetic_data.samples)),
        ("Build Dir", config.build.build_dir),
    )
    for name, value in rows:
        table.add_row(name, value)

    console.print(table)


def _print_results_summary(state, config: PipelineConfig) -> None:
    """Print the per-iteration results table and the best result."""
    console.print("\n[bold]Pipeline Results Summary[/bold]\n")

    metric_label = config.accuracy.metric.value.upper()

    table = Table(title="Iteration Results")
    table.add_column("Iter", style="cyan")
    table.add_column("Status", style="green")
    table.add_column(metric_label, style="yellow")
    table.add_column("Latency (ms)", style="magenta")
    table.add_column("RAM (KB)", style="blue")
    table.add_column("Flash (KB)", style="blue")

    for result in state.iterations:
        status_color = {
            "success": "green",
            "failed": "red",
            "rejected": "yellow",
            "stopped": "cyan",
        }.get(result.status.value, "white")

        # Compared against None, not truthiness: a legitimate 0.0 rendered as '-'.
        table.add_row(
            str(result.iteration_number),
            f"[{status_color}]{result.status.value}[/{status_color}]",
            _format_accuracy(result),
            "-" if result.latency_ms is None else f"{result.latency_ms:.3f}",
            "-" if result.ram_usage_kb is None else str(result.ram_usage_kb),
            "-" if result.flash_usage_kb is None else str(result.flash_usage_kb),
        )

    console.print(table)

    best = state.get_best_result()
    if best:
        console.print(f"\n[bold green]Best Result: Iteration {best.iteration_number}[/bold green]")
        console.print(f"  {metric_label}: {_format_accuracy(best)}")
        if best.latency_ms is not None:
            console.print(f"  Latency: {best.latency_ms:.3f} ms")
    else:
        console.print("\n[yellow]No iteration met the constraints[/yellow]")

    # The best of several iterations is the one that drew the kindest validation data as
    # well as the best architecture, so a lead inside the error bar is worth saying out
    # loud rather than leaving the reader to infer it from the figures.
    unresolved = state.unresolved_ranking()
    if unresolved:
        console.print(f"\n[yellow]{unresolved}[/yellow]")

    if state.stop_reason:
        console.print(f"\n[bold]Stop Reason:[/bold] {state.stop_reason}")


def _format_accuracy(result) -> str:
    """Render a measured metric with its error bar, or '-' when it was not measured."""
    # Compared against None, not truthiness: a legitimate 0.0 rendered as '-'.
    if result.accuracy_value is None:
        return "-"
    if result.accuracy_standard_error is None:
        return f"{result.accuracy_value:.6f}"

    return f"{result.accuracy_value:.6f} ± {result.accuracy_standard_error:.6f}"


def main() -> int:
    """Main entry point."""
    load_dotenv()

    # Shared so that the flags work on either side of the subcommand. SUPPRESS keeps a
    # subparser from overwriting a flag given before it with its own default.
    verbosity_flags = argparse.ArgumentParser(add_help=False)
    verbosity_flags.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
        help="Show debug logging; overrides the pyproject.toml verbosity",
    )
    verbosity_flags.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
        help="Show warnings and errors only; overrides the pyproject.toml verbosity",
    )
    parser = argparse.ArgumentParser(
        prog="zephyr-ml-forge",
        description="Embedded AI Model Iteration Pipeline for Zephyr RTOS",
        parents=[verbosity_flags],
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", parents=[verbosity_flags], help="Run the pipeline")
    run_parser.add_argument(
        "-c", "--config", type=Path, default=Path("config/pipeline_config.yaml"),
        help="Path to configuration file",
    )
    run_parser.add_argument(
        "-o", "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Directory for artifacts output",
    )
    run_parser.add_argument(
        "--simulate", action="store_true", help="Use the simulated runner instead of Zephyr",
    )
    run_parser.add_argument(
        "--hil", action="store_true", help="Override the configured mode to hardware-in-the-loop",
    )
    run_parser.add_argument(
        "--serial-port", type=str, default=None, help="Override the configured serial port",
    )
    run_parser.set_defaults(func=cmd_run)

    validate_parser = subparsers.add_parser(
        "validate", parents=[verbosity_flags], help="Validate configuration"
    )
    validate_parser.add_argument(
        "-c", "--config", type=Path, required=True, help="Path to configuration file"
    )
    validate_parser.set_defaults(func=cmd_validate)

    graph_parser = subparsers.add_parser(
        "graph", parents=[verbosity_flags], help="Generate graphs from artifacts"
    )
    graph_parser.add_argument(
        "-a", "--artifacts-dir", type=Path, default=Path("artifacts"),
        help="Directory containing artifacts",
    )
    graph_parser.set_defaults(func=cmd_graph)

    viz_parser = subparsers.add_parser(
        "visualize", parents=[verbosity_flags], help="Visualize a TFLite model"
    )
    viz_parser.add_argument("model", type=Path, help="Path to TFLite model file")
    viz_parser.add_argument(
        "-m", "--method", choices=["netron", "summary", "json"], default="netron",
        help="Visualization method (default: netron)",
    )
    viz_parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output file path (summary/json)",
    )
    viz_parser.add_argument(
        "--no-browser", action="store_true", help="Do not open a browser for netron",
    )
    viz_parser.set_defaults(func=cmd_visualize)

    args = parser.parse_args()
    args.verbosity = resolve_verbosity(args)
    setup_logging(args.verbosity)

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
