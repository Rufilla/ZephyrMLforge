# _____________________________________________________________________________
#
# @file vibration.py
# @brief Recorded accelerometer windows from the ML Vibro Sens Click
# @version 0.1
# @date 2026-10-24
# _____________________________________________________________________________
#
# Copyright (C) 2025 Rufilla Ltd
# _____________________________________________________________________________

"""
Recorded accelerometer windows from the ML Vibro Sens Click.

The click carries an FXLS8974 accelerometer & two motors, one balanced & one
unbalanced, which the firmware switches itself. A recording is labeled by the motor
state that was set while it was taken, so capture needs no rig & no hand labeling.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.config import VibrationConfig
from ..runners.serial_link import DeviceError, DeviceLink

logger = logging.getLogger(__name__)

DataTuple = tuple[np.ndarray, np.ndarray]

# Indexed by the firmware's MOTOR state. Both motors running simultaneously is not a state:
# on one FRDM-MCXN947 & click, the balanced motor's vibration did not reliably show under
# the unbalanced one's.
CLASS_NAMES = ("off", "balanced", "unbalanced")

AXES = 3

#  The firmware paces samples off the kernel tick, which on the
# FRDM-MCXN947 gave 128.5 ms for 128 samples at 1 kHz, 0.4% slow.
RATE_TOLERANCE = 0.02


@dataclass
class Recording:
    """Labeled windows, shaped (windows, window_samples, 1, 3).

    Rounds are numbered across the whole recording; sessions number the captures
    appended to it, so that validation can draw on each.
    """

    features: np.ndarray
    labels: np.ndarray
    rounds: np.ndarray
    sample_rate_hz: int
    measured_rate_hz: float
    sessions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.sessions is None:
            self.sessions = np.zeros(len(self.labels), dtype=np.int32)

    @property
    def window_samples(self) -> int:
        return int(self.features.shape[1])

    def followed_by(self, later: "Recording") -> "Recording":
        """This recording with a later session's windows after it.

        The later session's rounds & sessions are numbered after this one's.

        Raises:
            ValueError: the sessions differ in sample rate or window size.
        """
        if (
            later.sample_rate_hz != self.sample_rate_hz
            or later.window_samples != self.window_samples
        ):
            raise ValueError(
                f"Cannot append {later.window_samples}-sample windows at "
                f"{later.sample_rate_hz} Hz to a recording of {self.window_samples}-sample "
                f"windows at {self.sample_rate_hz} Hz"
            )

        count, later_count = len(self.labels), len(later.labels)
        return Recording(
            features=np.concatenate([self.features, later.features]),
            labels=np.concatenate([self.labels, later.labels]),
            rounds=np.concatenate([self.rounds, later.rounds + int(self.rounds.max()) + 1]),
            sessions=np.concatenate(
                [self.sessions, later.sessions + int(self.sessions.max()) + 1]
            ),
            sample_rate_hz=self.sample_rate_hz,
            measured_rate_hz=(
                self.measured_rate_hz * count + later.measured_rate_hz * later_count
            )
            / (count + later_count),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            features=self.features,
            labels=self.labels,
            rounds=self.rounds,
            sessions=self.sessions,
            class_names=np.array(CLASS_NAMES),
            sample_rate_hz=self.sample_rate_hz,
            measured_rate_hz=self.measured_rate_hz,
        )

    @classmethod
    def load(cls, path: Path) -> "Recording":
        """Labeled windows, shaped (windows, window_samples, 1, 3)."""
        if not path.exists():
            raise FileNotFoundError(
                f"No vibration recording at {path}. Record one with "
                "'zephyr-ml-forge capture -c <config>' with the board connected."
            )

        with np.load(path) as data:
            if tuple(data["class_names"].tolist()) != CLASS_NAMES:
                raise ValueError(
                    f"{path} holds classes {', '.join(data['class_names'].tolist())}, not "
                    f"{', '.join(CLASS_NAMES)}. Record again with 'zephyr-ml-forge capture'."
                )
            return cls(
                features=data["features"],
                labels=data["labels"],
                rounds=data["rounds"],
                sample_rate_hz=int(data["sample_rate_hz"]),
                measured_rate_hz=float(data["measured_rate_hz"]),
                #  Recordings saved before sessions existed hold one.
                sessions=data["sessions"] if "sessions" in data.files else None,
            )


@contextmanager
def motors_off_afterwards(link: DeviceLink) -> Iterator[None]:
    try:
        yield
    except BaseException:
        try:
            link.set_motors(0)
        except DeviceError as motor_error:
            logger.error(f"Could not switch the motors off after the failure: {motor_error}")
        raise
    link.set_motors(0)


def measured_rate_hz(durations_us: list[int], config: VibrationConfig) -> float:
    rate_hz = config.window_samples * 1e6 / float(np.mean(durations_us))
    if abs(rate_hz / config.sample_rate_hz - 1) > RATE_TOLERANCE:
        raise DeviceError(
            f"Board sampled at {rate_hz:.0f} Hz against the configured "
            f"{config.sample_rate_hz} Hz; it runs firmware built for another rate"
        )
    return rate_hz


def record(
    link: DeviceLink,
    config: VibrationConfig,
    on_progress: Callable[[int, int], None] | None = None,
) -> Recording:
    """Record config.windows_per_class windows of every motor state.

    The states are visited in turn, once per round, so slow drift over the session is
    shared by every class rather than separating them. The motors are switched off
    however the capture ends.

    Raises:
        DeviceError: the board failed a command, returned a window of the wrong size,
            or sampled away from the configured rate.
    """
    per_round = config.windows_per_round
    total = per_round * config.rounds * len(CLASS_NAMES)

    windows: list[list[float]] = []
    labels: list[int] = []
    rounds: list[int] = []
    durations_us: list[int] = []

    with motors_off_afterwards(link):
        for round_index in range(config.rounds):
            for state in range(len(CLASS_NAMES)):
                link.set_motors(state)
                time.sleep(config.settle_s)

                for _ in range(per_round):
                    duration_us, values = link.capture_window()
                    if len(values) != config.window_values:
                        raise DeviceError(
                            f"Board sent {len(values)} values for a window, expected "
                            f"{config.window_values}; it runs firmware built for another "
                            "vibration.window_samples"
                        )

                    windows.append(values)
                    labels.append(state)
                    rounds.append(round_index)
                    durations_us.append(duration_us)
                    if len(durations_us) == 1:
                        measured_rate_hz(durations_us, config)

                    if on_progress:
                        on_progress(len(windows), total)

    rate_hz = measured_rate_hz(durations_us, config)

    features = np.asarray(windows, dtype=np.float32).reshape(
        -1, config.window_samples, 1, AXES
    )
    return Recording(
        features=features,
        labels=np.asarray(labels, dtype=np.int32),
        rounds=np.asarray(rounds, dtype=np.int32),
        sample_rate_hz=config.sample_rate_hz,
        measured_rate_hz=rate_hz,
    )


class VibrationDataset:
    """Splits a recording into training & validation rounds."""

    def __init__(self, config: VibrationConfig):
        self.config = config

    @property
    def class_names(self) -> tuple[str, ...]:
        return CLASS_NAMES

    def generate(self) -> tuple[DataTuple, DataTuple]:
        """Load the recording & hold out the last config.val_rounds rounds of each session.

        Each session is held out from, so a recording that mixes cold & warm motors
        is validated on both.

        Returns:
            (train_features, train_labels), (val_features, val_labels), with features
            shaped (windows, window_samples, 1, 3) in m/s^2.

        Raises:
            FileNotFoundError: there is no recording at config.recording.
            ValueError: the recording was taken at another rate or window size, or a
                session holds fewer rounds than the split needs.
        """
        recording = Recording.load(Path(self.config.recording))

        if (
            recording.sample_rate_hz != self.config.sample_rate_hz
            or recording.window_samples != self.config.window_samples
        ):
            raise ValueError(
                f"{self.config.recording} holds {recording.window_samples}-sample windows "
                f"at {recording.sample_rate_hz} Hz, but the configuration asks for "
                f"{self.config.window_samples} at {self.config.sample_rate_hz} Hz. "
                "Record again with 'zephyr-ml-forge capture'."
            )

        is_validation = np.zeros(len(recording.labels), dtype=bool)
        for session in np.unique(recording.sessions):
            in_session = recording.sessions == session
            session_rounds = np.unique(recording.rounds[in_session])
            if self.config.val_rounds >= len(session_rounds):
                raise ValueError(
                    f"Session {session} of {self.config.recording} holds "
                    f"{len(session_rounds)} rounds, too few to hold out "
                    f"vibration.val_rounds ({self.config.val_rounds}) & still train"
                )
            held_out = session_rounds[-self.config.val_rounds :]
            is_validation |= in_session & np.isin(recording.rounds, held_out)
        train = (recording.features[~is_validation], recording.labels[~is_validation])
        validation = (recording.features[is_validation], recording.labels[is_validation])

        logger.info(
            f"Loaded {len(train[0])} training & {len(validation[0])} validation windows "
            f"of {recording.window_samples} samples at {recording.sample_rate_hz} Hz"
        )
        return train, validation
