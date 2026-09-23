import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from motion_helpers import feed, motion_frames
from core.motion.detectors import ClassifierWaveDetector, RuleWaveDetector, load_classifier, save_classifier
from core.motion.features import features_vector, window_features
from core.motion.window import KeypointWindow

WAVE = {"above_frac": 1.0, "height_mean": 0.5, "x_amplitude": 0.9, "y_amplitude": 0.1, "reversals": 5.0,
        "reversal_rate": 3.4, "speed_mean": 2.0, "elbow_below_frac": 1.0, "valid_frac": 1.0}
STILL = {"above_frac": 0.0, "height_mean": -1.5, "x_amplitude": 0.02, "y_amplitude": 0.02, "reversals": 0.0,
         "reversal_rate": 0.0, "speed_mean": 0.1, "elbow_below_frac": 0.0, "valid_frac": 1.0}


def features_of(kind):
    window = feed(KeypointWindow(1.5, 0.5, 1.0), motion_frames(kind))
    return window_features(*window.frames())


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
    model = LogisticRegression().fit(np.array([features_vector(WAVE), features_vector(STILL)]), [1, 0])
    path = tmp_path / "wave.joblib"
    save_classifier(model, path)

    detector = ClassifierWaveDetector(path)
    assert detector.available
    detection = detector.detect(WAVE)
    assert detection.is_wave and 0.5 <= detection.score <= 1.0
    assert not detector.detect(STILL).is_wave
    assert detector.detect(None) is None


def test_mismatched_model_is_rejected(tmp_path):
    path = tmp_path / "old.joblib"
    joblib.dump({"model": None, "feature_names": ["something_else"], "window_s": 1.5}, path)
    with pytest.raises(ValueError, match="retrain"):
        load_classifier(path)
    with pytest.raises(ValueError):
        ClassifierWaveDetector(path)


def test_missing_model_is_unavailable(tmp_path):
    detector = ClassifierWaveDetector(tmp_path / "missing.joblib")
    assert not detector.available
    assert detector.detect(WAVE) is None


def test_detector_uses_the_threshold_stored_with_the_forest(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    from core.motion.detectors import ClassifierWaveDetector
    from core.motion.forest import export_forest

    rng = np.random.default_rng(0)
    x = rng.normal(size=(80, 9))
    y = (x[:, 0] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=8, random_state=0).fit(x, y)
    path = tmp_path / "wave.npz"
    export_forest(model, path, threshold=0.9)

    detector = ClassifierWaveDetector(path=tmp_path / "wave.joblib", forest_path=path)
    assert detector.threshold == 0.9
    assert ClassifierWaveDetector(path=tmp_path / "wave.joblib", forest_path=path, threshold=0.3).threshold == 0.3
