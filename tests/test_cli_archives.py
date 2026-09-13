"""Archiving of a previous run's artifacts."""

from __future__ import annotations

from pipeline.cli import ARCHIVES_KEPT, _archive_existing_artifacts


def make_run(artifacts_dir):
    """Write one file into the artifacts directory, so it is worth archiving."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "metrics.json").write_text("[]")


def test_archives_a_previous_run_and_clears_the_directory(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    make_run(artifacts_dir)

    archive = _archive_existing_artifacts(artifacts_dir)

    assert archive is not None and archive.is_file()
    assert not artifacts_dir.exists()


def test_archives_nothing_when_there_is_no_previous_run(tmp_path):
    assert _archive_existing_artifacts(tmp_path / "artifacts") is None


def test_archives_nothing_when_the_directory_is_empty(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    assert _archive_existing_artifacts(artifacts_dir) is None


def test_keeps_only_the_most_recent_archives(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    for index in range(ARCHIVES_KEPT + 3):
        # Timestamped to the second, so the names are written directly.
        (tmp_path / f"artifacts_2025010{index}_000000.zip").write_text("old")
    make_run(artifacts_dir)

    _archive_existing_artifacts(artifacts_dir)

    assert len(list(tmp_path.glob("artifacts_*.zip"))) == ARCHIVES_KEPT
