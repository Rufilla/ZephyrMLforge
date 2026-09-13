"""Terminal verbosity and the capture of native output."""

from __future__ import annotations

import logging
import os

import pytest

from pipeline.core.output import (
    TENSORFLOW_LOG_ENV,
    OutputSettings,
    Verbosity,
    configure_output,
    suppressed_native_output,
)


def write_to_file_descriptors(message: str) -> None:
    """Write straight to the descriptors, as native code does."""
    os.write(1, f"{message} on stdout\n".encode())
    os.write(2, f"{message} on stderr\n".encode())


@pytest.fixture(autouse=True)
def restore_tensorflow_env(monkeypatch):
    """Keep the environment variable from leaking between tests."""
    monkeypatch.delenv(TENSORFLOW_LOG_ENV, raising=False)


@pytest.mark.parametrize(
    "verbosity,log_level,tensorflow_level",
    [
        (Verbosity.QUIET, logging.WARNING, "3"),
        (Verbosity.NORMAL, logging.INFO, "2"),
        (Verbosity.VERBOSE, logging.DEBUG, "2"),
        (Verbosity.DEBUG, logging.DEBUG, "0"),
    ],
)
def test_each_verbosity_sets_both_log_levels(verbosity, log_level, tensorflow_level):
    assert verbosity.log_level == log_level
    assert verbosity.tensorflow_log_level == tensorflow_level


def test_only_debug_leaves_library_output_alone():
    assert Verbosity.DEBUG.shows_library_chatter is True
    assert Verbosity.VERBOSE.shows_library_chatter is False


def test_configure_output_sets_the_variable_tensorflow_reads():
    configure_output(Verbosity.NORMAL)

    assert os.environ[TENSORFLOW_LOG_ENV] == "2"


def test_settings_are_read_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.zephyr-ml-forge.output]\nverbosity = "quiet"\n'
    )

    assert OutputSettings.discover(tmp_path).verbosity is Verbosity.QUIET


def test_settings_are_found_from_a_subdirectory(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.zephyr-ml-forge.output]\nverbosity = "verbose"\n'
    )
    nested = tmp_path / "config" / "deeper"
    nested.mkdir(parents=True)

    assert OutputSettings.discover(nested).verbosity is Verbosity.VERBOSE


def test_settings_default_when_no_pyproject_declares_them(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')

    assert OutputSettings.discover(tmp_path).verbosity is Verbosity.NORMAL


def test_rejects_a_verbosity_that_is_not_a_level(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        '[tool.zephyr-ml-forge.output]\nverbosity = "loud"\n'
    )

    with pytest.raises(ValueError, match="verbosity"):
        OutputSettings.discover(tmp_path)


def test_native_writes_are_kept_off_the_terminal(capfd):
    configure_output(Verbosity.NORMAL)

    with suppressed_native_output("a test"):
        write_to_file_descriptors("noise")

    captured = capfd.readouterr()
    assert "noise" not in captured.out
    assert "noise" not in captured.err


def test_the_descriptors_still_work_afterwards(capfd):
    configure_output(Verbosity.NORMAL)

    with suppressed_native_output("a test"):
        write_to_file_descriptors("hidden")
    write_to_file_descriptors("visible")

    captured = capfd.readouterr()
    assert "visible on stdout" in captured.out
    assert "visible on stderr" in captured.err


def test_captured_output_is_logged_for_a_debug_reader(capfd, caplog):
    configure_output(Verbosity.NORMAL)

    with caplog.at_level(logging.DEBUG, logger="pipeline.core.output"):
        with suppressed_native_output("a test"):
            write_to_file_descriptors("detail")

    capfd.readouterr()
    assert "detail on stdout" in caplog.text


def test_a_failing_block_reports_what_it_printed(capfd, caplog):
    # A conversion that fails explains itself on stderr; discarding that would leave
    # the caller with an exception and no cause.
    configure_output(Verbosity.NORMAL)

    with caplog.at_level(logging.WARNING, logger="pipeline.core.output"):
        with pytest.raises(RuntimeError, match="conversion failed"):
            with suppressed_native_output("the converter"):
                write_to_file_descriptors("the reason")
                raise RuntimeError("conversion failed")

    capfd.readouterr()
    assert "the reason on stderr" in caplog.text
    assert "the converter" in caplog.text


def test_debug_leaves_native_output_on_the_terminal(capfd):
    configure_output(Verbosity.DEBUG)

    with suppressed_native_output("a test"):
        write_to_file_descriptors("wanted")

    assert "wanted on stdout" in capfd.readouterr().out
