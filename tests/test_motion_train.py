import pytest

from motion_helpers import synthetic_session
from core.motion.detectors import ClassifierWaveDetector, save_classifier
from training.train import format_report, train_and_evaluate

# 8 waves per session: with only a few, all waves of a session can share a speed and the forest learns that shortcut.
KINDS = ("wave", "still", "wave", "hand_on_face", "wave", "swing_low", "wave", "stretch") * 2


def test_training_on_two_sessions_is_grouped_by_session(tmp_path):
    sessions = [synthetic_session(seed=seed, round_kinds=KINDS, round_s=4.0) for seed in (0, 1)]
    result = train_and_evaluate(sessions)

    assert result["groups"] == "session" and result["sessions"] == 2
    assert result["wave_windows"] > 0 and result["windows"] > result["wave_windows"]
    assert result["classifier"]["f1"] > 0.9
    assert result["rules"]["f1"] > 0.9
    assert set(result["importances"]) == set(result["medians"])

    report = format_report(result)
    assert "rules" in report and "classifier" in report and "grouped by session" in report

    save_classifier(result["model"], tmp_path / "wave.joblib")
    assert ClassifierWaveDetector(tmp_path / "wave.joblib").available


def test_single_session_is_grouped_by_round():
    result = train_and_evaluate([synthetic_session(seed=3, round_kinds=KINDS, round_s=4.0)])
    assert result["groups"] == "round"


def test_training_needs_both_classes():
    with pytest.raises(ValueError, match="WAVE and NOT WAVE"):
        train_and_evaluate([synthetic_session(round_kinds=("wave", "wave"), round_s=3.0)])
    with pytest.raises(ValueError, match="No usable windows"):
        train_and_evaluate([])
