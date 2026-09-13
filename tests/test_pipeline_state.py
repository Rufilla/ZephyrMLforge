"""Constraint checking, target checking and best-iteration selection."""

from __future__ import annotations

from pipeline.core.config import TaskDescription
from pipeline.core.pipeline import (
    FailureType,
    IterationResult,
    IterationStatus,
    Pipeline,
    PipelineState,
)
from tests.test_config import build_config


def make_result(iteration: int = 1, **fields) -> IterationResult:
    """Build a successful iteration result with every metric measured."""
    defaults = {
        "status": IterationStatus.SUCCESS,
        "accuracy_value": 0.02,
        "latency_ms": 1.0,
        "ram_usage_kb": 40,
        "flash_usage_kb": 80,
    }
    return IterationResult(iteration_number=iteration, **{**defaults, **fields})


def test_accepts_a_result_inside_every_limit():
    meets, violations = make_result().meets_constraints(build_config())

    assert meets is True
    assert violations == []


def test_rejects_a_result_whose_accuracy_was_never_measured():
    meets, violations = make_result(accuracy_value=None).meets_constraints(build_config())

    assert meets is False
    assert violations == ["Accuracy was not measured"]


def test_rejects_a_result_whose_latency_was_never_measured():
    meets, violations = make_result(latency_ms=None).meets_constraints(build_config())

    assert meets is False
    assert violations == ["Latency was not measured"]


def test_rejects_an_error_above_the_limit():
    meets, violations = make_result(accuracy_value=0.06).meets_constraints(build_config())

    assert meets is False
    assert "is outside max_error 0.05" in violations[0]


def test_rejects_a_latency_above_the_limit():
    meets, violations = make_result(latency_ms=9.0).meets_constraints(build_config())

    assert meets is False
    assert "exceeds max 5.0ms" in violations[0]


def test_rejects_memory_above_the_limit():
    meets, violations = make_result(ram_usage_kb=200, flash_usage_kb=900).meets_constraints(
        build_config()
    )

    assert meets is False
    assert len(violations) == 2


def test_accepts_absent_memory_figures():
    # The simulated runner legitimately reports none.
    meets, violations = make_result(ram_usage_kb=None, flash_usage_kb=None).meets_constraints(
        build_config()
    )

    assert meets is True
    assert violations == []


def test_rejects_an_accuracy_below_its_limit_when_higher_is_better():
    config = build_config(
        accuracy={"metric": "accuracy", "target": 0.95, "max_error": 0.80},
        synthetic_data={"task": "classification"},
    )

    meets, violations = make_result(accuracy_value=0.70).meets_constraints(config)

    assert meets is False
    assert "is outside max_error 0.8" in violations[0]


def test_meets_targets_only_when_the_target_is_reached():
    config = build_config()

    assert make_result(accuracy_value=0.005).meets_targets(config) is True
    assert make_result(accuracy_value=0.02).meets_targets(config) is False


def test_meets_no_target_without_a_measurement():
    assert make_result(accuracy_value=None).meets_targets(build_config()) is False


def test_best_result_takes_the_lowest_error():
    state = PipelineState(config=build_config())
    state.iterations = [
        make_result(1, accuracy_value=0.04),
        make_result(2, accuracy_value=0.01),
        make_result(3, accuracy_value=0.03),
    ]

    assert state.get_best_result().iteration_number == 2


def test_best_result_takes_the_highest_accuracy():
    config = build_config(
        accuracy={"metric": "accuracy", "target": 0.95, "max_error": 0.80},
        synthetic_data={"task": "classification"},
    )
    state = PipelineState(config=config)
    state.iterations = [
        make_result(1, accuracy_value=0.88),
        make_result(2, accuracy_value=0.97),
        make_result(3, accuracy_value=0.91),
    ]

    assert state.get_best_result().iteration_number == 2


def test_best_result_ignores_iterations_that_did_not_succeed():
    state = PipelineState(config=build_config())
    state.iterations = [
        make_result(1, accuracy_value=0.04),
        make_result(2, accuracy_value=0.001, status=IterationStatus.REJECTED),
    ]

    assert state.get_best_result().iteration_number == 1


def test_no_best_result_before_any_iteration_succeeds():
    assert PipelineState(config=build_config()).get_best_result() is None


class Unused:
    """Stands in for a collaborator the tested path never touches."""


def make_pipeline(tmp_path, config=None) -> Pipeline:
    """Build a pipeline whose collaborators are never called."""
    return Pipeline(
        config=config or build_config(),
        agent=Unused(),
        data_generator=Unused(),
        trainer=Unused(),
        runner=Unused(),
        metrics_collector=Unused(),
        artifacts_dir=tmp_path / "artifacts",
        task_description=TaskDescription(),
    )


def failed_result(iteration: int, message: str, failure_type=FailureType.RUNTIME):
    """An iteration that failed for the given reason."""
    return make_result(
        iteration,
        status=IterationStatus.FAILED,
        accuracy_value=None,
        latency_ms=None,
        failure_type=failure_type,
        failure_message=message,
    )


def test_abandons_a_run_after_the_same_failure_twice(tmp_path):
    # A misconfigured agent fails identically every time; without this the run spends
    # its whole iteration budget, and every attempt may cost money.
    pipeline = make_pipeline(tmp_path)
    message = "Messages.create() got an unexpected keyword argument 'temperature'"
    pipeline.state.iterations = [failed_result(1, message), failed_result(2, message)]

    reason = pipeline._abandon_reason(pipeline.state.iterations[-1])

    assert reason is not None
    assert "twice in a row" in reason
    assert "temperature" in reason


def test_keeps_going_when_two_failures_differ(tmp_path):
    pipeline = make_pipeline(tmp_path)
    pipeline.state.iterations = [
        failed_result(1, "TFLite conversion failed"),
        failed_result(2, "Board did not answer PING"),
    ]

    assert pipeline._abandon_reason(pipeline.state.iterations[-1]) is None


def test_keeps_going_after_a_single_failure(tmp_path):
    pipeline = make_pipeline(tmp_path)
    pipeline.state.iterations = [failed_result(1, "TFLite conversion failed")]

    assert pipeline._abandon_reason(pipeline.state.iterations[-1]) is None


def test_keeps_going_when_a_success_separates_the_failures(tmp_path):
    pipeline = make_pipeline(tmp_path)
    message = "Board did not answer PING"
    pipeline.state.iterations = [
        failed_result(1, message),
        make_result(2),
        failed_result(3, message),
    ]

    assert pipeline._abandon_reason(pipeline.state.iterations[-1]) is None


def test_abandons_immediately_on_an_unrecoverable_build_failure(tmp_path):
    pipeline = make_pipeline(tmp_path)
    result = failed_result(1, "west not found on PATH", failure_type=FailureType.BUILD)
    pipeline.state.iterations = [result]

    reason = pipeline._abandon_reason(result)

    assert reason is not None
    assert "Unrecoverable build failure" in reason


def test_a_stopped_iteration_is_neither_best_nor_a_failure(tmp_path):
    pipeline = make_pipeline(tmp_path)
    stopped = make_result(
        2,
        status=IterationStatus.STOPPED,
        accuracy_value=None,
        latency_ms=None,
        stop_recommendation="At the noise floor",
    )
    pipeline.state.iterations = [make_result(1, accuracy_value=0.02), stopped]

    assert pipeline.state.get_best_result().iteration_number == 1
    assert pipeline._abandon_reason(stopped) is None


def test_a_stop_recommendation_reaches_the_result_file(tmp_path):
    result = make_result(
        1,
        status=IterationStatus.STOPPED,
        accuracy_value=None,
        latency_ms=None,
        stop_recommendation="No proposal would do better",
    )

    assert result.to_dict()["stop_recommendation"] == "No proposal would do better"
    assert result.to_dict()["status"] == "stopped"


class StoppingAgent:
    """Proposes once, then ends the run."""

    def __init__(self, proposals_before_stopping: int = 1):
        self.proposals_before_stopping = proposals_before_stopping
        self.calls = 0
        self.contexts: list[dict] = []

    def propose_model(self, context):
        from pipeline.agents.base import ModelProposal

        self.calls += 1
        self.contexts.append(context)

        if self.calls > self.proposals_before_stopping:
            return ModelProposal(
                architecture_description="",
                model_code="",
                should_stop=True,
                stop_reason="Within 3% of the noise floor",
            )

        raise AssertionError("the test stops on the first call")


class RecordingCollector:
    """Records what the pipeline reports, and draws nothing."""

    def __init__(self):
        self.records = []

    def record(self, result):
        self.records.append(result)

    def generate_graphs(self, results, output_dir, constraints=None):
        return []


def test_the_loop_ends_when_the_agent_asks_to_stop(tmp_path):
    # Nothing is trained, built or flashed: the agent declines before proposing.
    collector = RecordingCollector()
    pipeline = Pipeline(
        config=build_config(),
        agent=StoppingAgent(proposals_before_stopping=0),
        data_generator=SimpleGenerator(),
        trainer=Unused(),
        runner=Unused(),
        metrics_collector=collector,
        artifacts_dir=tmp_path / "artifacts",
        task_description=TaskDescription(),
    )

    state = pipeline.run()

    assert state.stopped is True
    assert "Within 3% of the noise floor" in state.stop_reason
    assert len(state.iterations) == 1
    assert state.iterations[0].status is IterationStatus.STOPPED
    assert (tmp_path / "artifacts" / "iteration_0001" / "result.json").is_file()


def test_the_agent_is_told_whether_it_may_stop(tmp_path):
    agent = StoppingAgent(proposals_before_stopping=0)
    config = build_config(
        iteration={"max_iterations": 4, "early_stop_on_target": True, "allow_agent_stop": False}
    )
    pipeline = Pipeline(
        config=config,
        agent=agent,
        data_generator=SimpleGenerator(),
        trainer=Unused(),
        runner=Unused(),
        metrics_collector=RecordingCollector(),
        artifacts_dir=tmp_path / "artifacts",
        task_description=TaskDescription(),
    )

    pipeline.run()

    assert agent.contexts[0]["may_stop"] is False


class SimpleGenerator:
    """Returns a tiny dataset without touching numpy's RNG."""

    def generate(self):
        import numpy as np

        features = np.zeros((4, 1), dtype=np.float32)
        labels = np.zeros((4, 1), dtype=np.float32)
        return (features, labels), (features, labels)


def test_an_under_trained_model_is_recorded_as_such():
    # A fit ended by the epoch cap rather than by convergence makes an iteration
    # comparison rank training budget as much as architecture.
    result = make_result(1, epochs_trained=60, is_under_trained=True)

    assert result.to_dict()["epochs_trained"] == 60
    assert result.to_dict()["is_under_trained"] is True


def test_a_converged_model_is_not_marked_under_trained():
    assert make_result(1, epochs_trained=23).to_dict()["is_under_trained"] is False


def test_a_lead_inside_the_error_bar_is_reported_as_unresolved():
    # Six iterations are six draws of the same validation set, so the best of them is
    # partly the luckiest of them. A gap this narrow is not a result.
    state = PipelineState(config=build_config())
    state.iterations = [
        make_result(
            1, accuracy_value=0.0200, accuracy_standard_error=0.0011, accuracy_sample_count=1000
        ),
        make_result(
            2, accuracy_value=0.0198, accuracy_standard_error=0.0011, accuracy_sample_count=1000
        ),
    ]

    message = state.unresolved_ranking()

    assert "not distinguished" in message
    assert "dataset.val_samples" in message


def test_a_lead_wider_than_the_error_bar_is_not_reported():
    state = PipelineState(config=build_config())
    state.iterations = [
        make_result(
            1, accuracy_value=0.0400, accuracy_standard_error=0.0003, accuracy_sample_count=10000
        ),
        make_result(
            2, accuracy_value=0.0200, accuracy_standard_error=0.0003, accuracy_sample_count=10000
        ),
    ]

    assert state.unresolved_ranking() is None


def test_a_ranking_is_not_questioned_when_the_error_bar_was_never_measured():
    state = PipelineState(config=build_config())
    state.iterations = [make_result(1, accuracy_value=0.02), make_result(2, accuracy_value=0.019)]

    assert state.unresolved_ranking() is None


def test_a_single_iteration_has_no_ranking_to_question():
    state = PipelineState(config=build_config())
    state.iterations = [
        make_result(
            1, accuracy_value=0.02, accuracy_standard_error=0.001, accuracy_sample_count=1000
        )
    ]

    assert state.unresolved_ranking() is None


def test_the_runner_up_is_the_second_best_rather_than_the_previous_iteration():
    # Ranked under the configured metric: the iteration before the best one may be the
    # worst of the run, and comparing against it would call any lead resolved.
    state = PipelineState(config=build_config())
    state.iterations = [
        make_result(
            1, accuracy_value=0.0201, accuracy_standard_error=0.0011, accuracy_sample_count=1000
        ),
        make_result(
            2,
            accuracy_value=0.0900,
            status=IterationStatus.REJECTED,
            accuracy_standard_error=0.0011,
            accuracy_sample_count=1000,
        ),
        make_result(
            3, accuracy_value=0.0200, accuracy_standard_error=0.0011, accuracy_sample_count=1000
        ),
    ]

    assert "iteration 1" in state.unresolved_ranking()
