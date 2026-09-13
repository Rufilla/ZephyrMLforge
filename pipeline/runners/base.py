# _____________________________________________________________________________
#
# @file base.py
# @brief Base classes for execution runners
# @version 0.2
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Base classes for execution runners.

Defines interfaces for building and running Zephyr applications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BuildResult:
    """Result of building a Zephyr application."""

    success: bool
    flash_usage_kb: int | None = None
    ram_usage_kb: int | None = None
    error_message: str | None = None
    build_log: str = ""

    # Where the build actually happened. Runners share one build directory across
    # iterations for incremental builds, so it is rarely the directory they were
    # handed, and the caller cannot find the ELF or the map file without being told.
    build_dir: Path | None = None
    app_dir: Path | None = None


@dataclass
class ExecutionResult:
    """Result of running a Zephyr application."""

    success: bool
    execution_env: str = "unknown"  # qemu, hil, simulated
    latency_ms: float | None = None
    accuracy_value: float | None = None

    # How precisely the validation set could measure that accuracy, and over how many
    # samples, so a caller can tell a real improvement from a different draw of data.
    accuracy_standard_error: float | None = None
    accuracy_sample_count: int | None = None

    arena_used_bytes: int | None = None
    error_message: str | None = None
    stdout: str = ""
    stderr: str = ""


class BaseRunner(ABC):
    """Abstract base class for execution runners."""

    @abstractmethod
    def build(
        self,
        tflite_path: Path,
        build_dir: Path,
        host_model_path: Path | None = None,
    ) -> BuildResult:
        """
        Build Zephyr application with TFLite model.

        Args:
            tflite_path: Model embedded in the firmware, which for an accelerator run
                carries a custom operator the host interpreter cannot execute
            build_dir: Directory the caller suggests for build output
            host_model_path: Model the host scores against; defaults to tflite_path

        Returns:
            BuildResult with status, metrics and the directory actually used
        """

    @abstractmethod
    def run(
        self,
        build_dir: Path,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> ExecutionResult:
        """
        Run the built application and measure performance.

        Args:
            build_dir: Directory containing built application
            test_data: Tuple of (features, labels) for testing

        Returns:
            ExecutionResult with performance metrics
        """
