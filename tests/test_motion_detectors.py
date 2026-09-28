import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from motion_helpers import feed, motion_frames
from core.motion.detectors import ClassifierWaveDetector, RuleWaveDetector
from core.motion.features import FEATURE_NAMES, features_vector, window_features
from core.motion.forest import export_forest
from core.motion.window import KeypointWindow

WAVE = {"above_frac": 1.0, "height_mean": 0.5, "x_amplitude": 0.9, "y_amplitude": 0.1, "reversals": 5.0,
        "reversal_rate": 3.4, "speed_mean": 2.0, "elbow_below_frac": 1.0, "valid_frac": 1.0}
STILL = {"above_frac": 0.0, "height_mean": -1.5, "x_amplitude": 0.02, "y_amplitude": 0.02, "reversals": 0.0,
         "reversal_rate": 0.0, "speed_mean": 0.1, "elbow_below_frac": 0.0, "valid_frac": 1.0}


def features_of(kind):
    window = feed(KeypointWindow(1.5, 0.5, 1.0), motion_frames(kind))
    return window_features(*window.frames())


def wave_forest(path, threshold=0.5):
    rng = np.random.default_rng(0)
    x = np.vstack([features_vector(WAVE) + rng.normal(scale=0.05, size=(20, len(FEATURE_NAMES))),
                   features_vector(STILL) + rng.normal(scale=0.05, size=(20, len(FEATURE_NAMES)))])
    y = np.array([1] * 20 + [0] * 20)
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(x, y)
    return export_forest(model, path, threshold=threshold)


def test_rules_fire_on_wave():
    detection = RuleWaveDetector().detect(WAVE)
    assert detection.is_wave and detection.score == pytest.approx(1.0)
    assert detection.details == {"above_frac": 1.0, "reversals": 5.0, "x_amplitude": 0.9}


def test_rules_fire_on_synthetic_wave_but_not_on_non_waves():
    rules = RuleWaveDetector()
    assert rules.detect(features_of("wave")).is_wave
    for kind in ("still", "hand_on_face", "swing_low", "stretch"):
        assert not rules.detect(features_of(kind)).is_wave, kind


def test_rules_partial_score_and_no_data():
    partial = {**WAVE, "reversals": 0.0, "x_amplitude": 0.15}
    detection = RuleWaveDetector().detect(partial)
    assert not detection.is_wave
    assert detection.score == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert RuleWaveDetector().detect(None) is None


def test_classifier_round_trip(tmp_path):
    path = wave_forest(tmp_path / "wave.npz")

    detector = ClassifierWaveDetector(path)
    assert detector.available
    detection = detector.detect(WAVE)
    assert detection.is_wave and 0.5 <= detection.score <= 1.0
    assert not detector.detect(STILL).is_wave
    assert detector.detect(None) is None


def test_joblib_path_maps_to_the_npz_next_to_it(tmp_path):
    wave_forest(tmp_path / "wave.npz")
    assert ClassifierWaveDetector(tmp_path / "wave.joblib").available


def test_mismatched_model_is_rejected(tmp_path):
    rng = np.random.default_rng(0)
    model = RandomForestClassifier(n_estimators=2, random_state=0).fit(rng.normal(size=(10, 2)), [0, 1] * 5)
    path = export_forest(model, tmp_path / "old.npz", feature_names=["a", "b"])
    with pytest.raises(ValueError, match="retrain"):
        ClassifierWaveDetector(path)


def test_missing_model_is_unavailable(tmp_path):
    detector = ClassifierWaveDetector(tmp_path / "missing.npz")
    assert not detector.available
    assert detector.detect(WAVE) is None


def test_detector_uses_the_threshold_stored_with_the_forest(tmp_path):
    path = wave_forest(tmp_path / "wave.npz", threshold=0.9)

    assert ClassifierWaveDetector(path).threshold == 0.9
    assert ClassifierWaveDetector(path, threshold=0.3).threshold == 0.3
