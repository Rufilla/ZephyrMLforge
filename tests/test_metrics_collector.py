"""Metrics reloaded from a finished run's artifacts."""

from __future__ import annotations

import json

from pipeline.metrics.collector import MetricsCollector


def test_records_reload_from_result_files_with_their_error_bars(tmp_path):
    iteration_dir = tmp_path / "iteration_0001"
    iteration_dir.mkdir()
    (iteration_dir / "result.json").write_text(
        json.dumps(
            {
                "iteration_number": 1,
                "timestamp": "2026-10-24T17:40:00+00:00",
                "status": "success",
                "metrics": {
                    "accuracy_value": 0.95,
                    "accuracy_standard_error": 0.016,
                    "latency_ms": 6.95,
                    "ram_usage_kb": 95,
                    "flash_usage_kb": 194,
                    "arena_used_bytes": 6404,
                },
                "execution_env": "hil",
                "weight_reuse": {"reused": False},
            }
        )
    )

    records = MetricsCollector.from_artifacts(tmp_path).records

    assert [(r.iteration, r.accuracy_standard_error) for r in records] == [(1, 0.016)]
