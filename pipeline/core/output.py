# _____________________________________________________________________________
#
# @file output.py
# @brief Terminal verbosity, including TensorFlow's own output
# @version 0.1
# @date 2025-08-23
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Terminal verbosity, including TensorFlow's own output.

Most of what a run prints comes from TensorFlow rather than from this package, through
four channels that each need silencing separately: the C++ logger behind an environment
variable read at import, the absl Python logger, Keras's export messages, and the
warnings module. Verbosity therefore lives here rather than in the logging setup.

It is configured in pyproject.toml rather than the pipeline YAML because it changes
nothing about the run: a quiet run and a loud one produce identical artefacts, and the
YAML is the contract for what the pipeline does, not for how much it says.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import warnings
from collections.abc import Iterator
from enum import Enum
from pathlib import Path

import tomllib
from pydantic import BaseModel, Field

# TensorFlow reads this at import: 0 shows everything, 1 hides INFO, 2 also hides
# WARNING, 3 also hides ERROR.
TENSORFLOW_LOG_ENV = "TF_CPP_MIN_LOG_LEVEL"

# Libraries that log a line per HTTP request at INFO.
NOISY_LIBRARIES = ("httpx", "httpcore", "openai", "anthropic", "urllib3", "matplotlib")


class Verbosity(str, Enum):
    """How much a run prints."""

    QUIET = "quiet"
    NORMAL = "normal"
    VERBOSE = "verbose"
    DEBUG = "debug"

    @property
    def log_level(self) -> int:
        """Level for this package's own loggers."""
        return {
            Verbosity.QUIET: logging.WARNING,
            Verbosity.NORMAL: logging.INFO,
            Verbosity.VERBOSE: logging.DEBUG,
            Verbosity.DEBUG: logging.DEBUG,
        }[self]

    @property
    def tensorflow_log_level(self) -> str:
        """Value for TF_CPP_MIN_LOG_LEVEL.

        Errors are kept below 'quiet': a failed conversion should still say why.
        """
        return "3" if self is Verbosity.QUIET else "0" if self is Verbosity.DEBUG else "2"

    @property
    def shows_library_chatter(self) -> bool:
        """True when third-party logs and warnings are left alone."""
        return self is Verbosity.DEBUG


class OutputSettings(BaseModel):
    """Terminal output settings, from [tool.zephyr-ml-forge.output] in pyproject.toml."""

    verbosity: Verbosity = Field(
        default=Verbosity.NORMAL, description="quiet | normal | verbose | debug"
    )

    @classmethod
    def discover(cls, start: Path | None = None) -> "OutputSettings":
        """Search upward from ``start`` for a pyproject.toml carrying the settings.

        Returns the defaults when none is found, so the pipeline runs from anywhere.
        """
        current = (start or Path.cwd()).resolve()

        for candidate in [current, *current.parents]:
            pyproject_path = candidate / "pyproject.toml"
            if not pyproject_path.exists():
                continue

            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)

            table = data.get("tool", {}).get("zephyr-ml-forge", {}).get("output")
            if table is not None:
                return cls.model_validate(table)

        return cls()


def configure_output(verbosity: Verbosity) -> None:
    """Apply the verbosity to this process.

    Must run before TensorFlow is imported: its C++ logger reads the environment
    variable once, at import, and ignores later changes. The package imports
    TensorFlow lazily so that this is possible.
    """
    os.environ[TENSORFLOW_LOG_ENV] = verbosity.tensorflow_log_level

    if verbosity.shows_library_chatter:
        return

    for library in NOISY_LIBRARIES:
        logging.getLogger(library).setLevel(logging.WARNING)

    # The converter warns about missing input statistics on every quantisation, and
    # nothing in the pipeline acts on it.
    for module in ("tensorflow.*", "keras.*"):
        warnings.filterwarnings("ignore", category=UserWarning, module=module)
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=module)


def quieten_tensorflow(tf_module) -> None:
    """Silence the loggers that only exist once TensorFlow has been imported.

    Called from the lazy import. Keras prints its export summary - the saved artifact
    path and every endpoint and capture - through its own interactive logging, which no
    logger level reaches.
    """
    if os.environ.get(TENSORFLOW_LOG_ENV) == "0":
        return

    tf_module.get_logger().setLevel(logging.ERROR)
    tf_module.autograph.set_verbosity(0)

    try:
        import absl.logging

        absl.logging.set_verbosity(absl.logging.ERROR)
    except ImportError:
        pass

    try:
        import keras

        keras.config.disable_interactive_logging()
    except (ImportError, AttributeError):
        pass


@contextlib.contextmanager
def suppressed_native_output(description: str) -> Iterator[None]:
    """Capture writes that native code makes straight to the process's file descriptors.

    TensorFlow prints some lines before absl is initialised, and the converter prints
    its quantisation summary to stdout, so neither honours any logger. Redirecting the
    descriptors is the only way to reach them.

    The output is captured rather than discarded: it is logged at debug level, and at
    warning level if the block raised, so a failed conversion still says why.
    """
    if os.environ.get(TENSORFLOW_LOG_ENV) == "0":
        yield
        return

    logger = logging.getLogger(__name__)

    with tempfile.TemporaryFile(mode="w+") as capture:
        stdout_copy = os.dup(1)
        stderr_copy = os.dup(2)
        failed = False

        try:
            os.dup2(capture.fileno(), 1)
            os.dup2(capture.fileno(), 2)
            yield
        except Exception:
            failed = True
            raise
        finally:
            os.dup2(stdout_copy, 1)
            os.dup2(stderr_copy, 2)
            os.close(stdout_copy)
            os.close(stderr_copy)

            capture.seek(0)
            captured = capture.read().strip()

            if captured and failed:
                logger.warning(f"Output from {description}:\n{captured}")
            elif captured:
                logger.debug(f"Output from {description}:\n{captured}")


def import_tensorflow():
    """Import TensorFlow with its start-up output captured, and apply the verbosity.

    Returns:
        The imported tensorflow module.

    Raises:
        RuntimeError: TensorFlow is not installed.
    """
    try:
        with suppressed_native_output("the TensorFlow import"):
            import tensorflow

            # Enumerate devices inside the capture. The CUDA probe fails at error level
            # on a machine without a GPU, once, at whatever point something first
            # touches a device - which is otherwise in the middle of a training run.
            tensorflow.config.list_physical_devices()
    except ImportError as error:
        raise RuntimeError("TensorFlow is required for this operation") from error

    quieten_tensorflow(tensorflow)
    return tensorflow
