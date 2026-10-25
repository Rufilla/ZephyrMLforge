"""Recording & splitting ML Vibro Sens Click windows."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.core.config import VibrationConfig
from pipeline.data.vibration import Recording, VibrationDataset, record
from pipeline.runners.serial_link import DeviceError

WINDOW_SAMPLES = 4


def build_vibration_config(tmp_path, **overrides) -> VibrationConfig:
    defaults = {
        "recording": str(tmp_path / "vibration.npz"),
        "sample_rate_hz": 1000,
        "window_samples": WINDOW_SAMPLES,
        "windows_per_class": 2,
        "rounds": 2,
        "val_rounds": 1,
        "settle_s": 0,
    }
    return VibrationConfig(**{**defaults, **overrides})


class FakeLink:
    """ Stands in for DeviceLink; each window's values carry its motor
    state, so a test can see which state a window was recorded under."""

    def __init__(self, duration_us: int = 4000, values_per_window: int = WINDOW_SAMPLES * 3):
        self.duration_us = duration_us
        self.values_per_window = values_per_window
        self.motor_states: list[int] = []

    def set_motors(self, state: int) -> None:
        self.motor_states.append(state)

    def capture_window(self) -> tuple[int, list[float]]:
        return self.duration_us, [float(self.motor_states[-1])] * self.values_per_window


def build_recording(
    rounds: list[int], sample_rate_hz: int = 1000, sessions: list[int] | None = None
) -> Recording:
    count = len(rounds)
    return Recording(
        features=np.zeros((count, WINDOW_SAMPLES, 1, 3), dtype=np.float32),
        labels=np.arange(count, dtype=np.int32) % 3,
        rounds=np.asarray(rounds, dtype=np.int32),
        sample_rate_hz=sample_rate_hz,
        measured_rate_hz=float(sample_rate_hz),
        sessions=None if sessions is None else np.asarray(sessions, dtype=np.int32),
    )


def save_recording(path, rounds: list[int], sample_rate_hz: int = 1000) -> None:
    build_recording(rounds, sample_rate_hz).save(path)


def test_record_labels_each_window_by_the_motor_state_it_was_taken_under(tmp_path):
    link = FakeLink()

    recording = record(link, build_vibration_config(tmp_path))

    assert recording.labels.tolist() == [0, 1, 2, 0, 1, 2]
    assert recording.features[:, 0, 0, 0].tolist() == [0, 1, 2, 0, 1, 2]


def test_record_visits_every_state_once_per_round(tmp_path):
    recording = record(FakeLink(), build_vibration_config(tmp_path))

    assert recording.rounds.tolist() == [0, 0, 0, 1, 1, 1]


def test_record_shapes_windows_as_samples_by_one_by_three_axes(tmp_path):
    recording = record(FakeLink(), build_vibration_config(tmp_path))

    assert recording.features.shape == (6, WINDOW_SAMPLES, 1, 3)


def test_record_switches_the_motors_off_after_a_failed_window(tmp_path):
    link = FakeLink(values_per_window=5)

    with pytest.raises(DeviceError, match="sent 5 values for a window, expected 12"):
        record(link, build_vibration_config(tmp_path))

    assert link.motor_states[-1] == 0


def test_record_rejects_a_board_sampling_at_another_rate(tmp_path):
    #  4 samples in 8 ms is 500 Hz against the configured 1000.
    with pytest.raises(DeviceError, match="sampled at 500 Hz against the configured 1000"):
        record(FakeLink(duration_us=8000), build_vibration_config(tmp_path))


def test_record_rejects_another_rate_on_the_first_window(tmp_path):
    link = FakeLink(duration_us=8000)

    with pytest.raises(DeviceError, match="sampled at 500 Hz"):
        record(link, build_vibration_config(tmp_path))

    assert link.motor_states == [0, 0]


class BoardDyingMidCapture(FakeLink):
    def __init__(self):
        super().__init__()
        self.has_died = False

    def set_motors(self, state: int) -> None:
        if self.has_died:
            raise DeviceError("No reply to MOTOR within the read timeout")
        super().set_motors(state)

    def capture_window(self) -> tuple[int, list[float]]:
        self.has_died = True
        raise DeviceError("No reply to WINDOW within the read timeout")


def test_record_reports_the_capture_failure_when_the_motors_cannot_be_stopped(tmp_path):
    with pytest.raises(DeviceError, match="No reply to WINDOW"):
        record(BoardDyingMidCapture(), build_vibration_config(tmp_path))


def test_dataset_holds_out_the_final_rounds(tmp_path):
    config = build_vibration_config(tmp_path, rounds=3, val_rounds=1)
    save_recording(tmp_path / "vibration.npz", rounds=[0, 0, 1, 1, 2, 2])

    (train_features, _), (val_features, val_labels) = VibrationDataset(config).generate()

    assert len(train_features) == 4
    assert val_labels.tolist() == [1, 2]


def test_dataset_refuses_a_recording_taken_at_another_rate(tmp_path):
    save_recording(tmp_path / "vibration.npz", rounds=[0, 1], sample_rate_hz=500)

    with pytest.raises(ValueError, match="at 500 Hz, but the configuration asks for 4 at 1000"):
        VibrationDataset(build_vibration_config(tmp_path)).generate()


def test_dataset_refuses_too_few_rounds_to_split(tmp_path):
    save_recording(tmp_path / "vibration.npz", rounds=[0, 0])

    with pytest.raises(ValueError, match="Session 0 of .* holds 1 rounds, too few"):
        VibrationDataset(build_vibration_config(tmp_path)).generate()


def test_missing_recording_names_the_capture_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="zephyr-ml-forge capture"):
        VibrationDataset(build_vibration_config(tmp_path)).generate()


def test_dataset_refuses_a_recording_of_another_class_set(tmp_path):
    path = tmp_path / "vibration.npz"
    np.savez(
        path, features=np.zeros((2, WINDOW_SAMPLES, 1, 3)), labels=[0, 3], rounds=[0, 1],
        class_names=np.array(["off", "balanced", "unbalanced", "both"]),
        sample_rate_hz=1000, measured_rate_hz=1000.0,
    )

    with pytest.raises(ValueError, match="holds classes off, balanced, unbalanced, both"):
        VibrationDataset(build_vibration_config(tmp_path)).generate()


def test_appended_session_rounds_follow_the_earlier_ones():
    merged = build_recording(rounds=[0, 0, 1, 1]).followed_by(build_recording(rounds=[0, 1]))

    assert merged.rounds.tolist() == [0, 0, 1, 1, 2, 3]


def test_appended_session_weights_the_measured_rate_by_window_count():
    earlier = build_recording(rounds=[0, 0, 0])
    earlier.measured_rate_hz = 990.0
    later = build_recording(rounds=[0])
    later.measured_rate_hz = 1010.0

    assert earlier.followed_by(later).measured_rate_hz == pytest.approx(995.0)


def test_append_refuses_a_session_at_another_rate():
    with pytest.raises(ValueError, match="at 500 Hz to a recording of 4-sample windows at 1000"):
        build_recording(rounds=[0]).followed_by(build_recording(rounds=[0], sample_rate_hz=500))


def test_appended_session_is_numbered_after_the_earlier_ones():
    earlier = build_recording(rounds=[0, 1], sessions=[0, 1])

    merged = earlier.followed_by(build_recording(rounds=[0, 1]))

    assert merged.sessions.tolist() == [0, 1, 2, 2]


def test_dataset_holds_out_the_final_round_of_each_session(tmp_path):
    config = build_vibration_config(tmp_path, val_rounds=1)
    build_recording(rounds=[0, 1, 2, 3, 4], sessions=[0, 0, 1, 1, 1]).save(
        tmp_path / "vibration.npz"
    )

    (_, train_labels), (_, val_labels) = VibrationDataset(config).generate()

    assert train_labels.tolist() == [0, 2, 0]
    assert val_labels.tolist() == [1, 1]


def test_dataset_refuses_a_session_with_too_few_rounds_to_split(tmp_path):
    build_recording(rounds=[0, 1, 2], sessions=[0, 0, 1]).save(tmp_path / "vibration.npz")

    with pytest.raises(ValueError, match="Session 1 of .* holds 1 rounds, too few"):
        VibrationDataset(build_vibration_config(tmp_path)).generate()


def test_recording_saved_without_sessions_loads_as_one_session(tmp_path):
    path = tmp_path / "vibration.npz"
    np.savez(
        path, features=np.zeros((2, WINDOW_SAMPLES, 1, 3)), labels=[0, 1], rounds=[0, 1],
        class_names=np.array(["off", "balanced", "unbalanced"]),
        sample_rate_hz=1000, measured_rate_hz=1000.0,
    )

    assert Recording.load(path).sessions.tolist() == [0, 0]
