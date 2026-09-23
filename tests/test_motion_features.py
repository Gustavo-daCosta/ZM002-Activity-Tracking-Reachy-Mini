import numpy as np
import pytest

from motion_helpers import feed, motion_frames
from core.motion.features import FEATURE_NAMES, count_reversals, features_vector, window_features
from core.motion.window import KeypointWindow


def features_of(frames):
    window = feed(KeypointWindow(1.5, 0.5, 1.0), frames)
    return window_features(*window.frames())


def test_count_reversals_ignores_jitter():
    assert count_reversals(np.array([0.0, 0.05, 0.0, 0.05, 0.0, 0.04]), 0.1) == 0


def test_count_reversals_counts_sine_half_cycles():
    t = np.arange(30) / 20
    assert count_reversals(0.5 * np.sin(2 * np.pi * 2 * t), 0.1) >= 5


def test_wave_features():
    features = features_of(motion_frames("wave"))
    assert list(features) == list(FEATURE_NAMES)
    assert features["above_frac"] == pytest.approx(1.0)
    assert features["height_mean"] == pytest.approx(0.5, abs=0.05)
    assert 0.8 <= features["x_amplitude"] <= 1.0
    assert features["reversals"] >= 4
    assert features["reversal_rate"] >= 2.5
    assert features["elbow_below_frac"] == pytest.approx(1.0)
    assert features["valid_frac"] == pytest.approx(1.0)
    assert features_vector(features).shape == (9,)


def test_still_arm_has_no_reversals():
    features = features_of(motion_frames("still"))
    assert features["reversals"] == 0
    assert features["above_frac"] == 0


def test_active_wrist_is_the_moving_one():
    features = features_of(motion_frames("wave", hand="left"))
    assert features["above_frac"] == pytest.approx(1.0)
    assert features["reversals"] >= 4


def test_not_enough_data_returns_none():
    assert features_of(motion_frames("wave", duration_s=0.5)) is None
    half_missing = [(t, None if i % 2 else kps) for i, (t, kps) in enumerate(motion_frames("wave"))]
    assert features_of(half_missing) is None
