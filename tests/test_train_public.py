"""Training on public dataset windows, evaluated by subject and on external sets."""

import numpy as np
import pytest

from motion_helpers import motion_frames
from core.motion.features import window_features
from training.train_public import evaluate_windows, to_arrays, train_grouped
from core.motion.window import WINDOW_S, KeypointWindow


def windows_of(kind, subject, seed=0, duration_s=5.0):
    """(features, label, group) triples like public_data.dataset_windows produces."""
    window = KeypointWindow(WINDOW_S, 0.5, 16 / 9)
    out, next_emit = [], None
    for timestamp, keypoints in motion_frames(kind, duration_s=duration_s, seed=seed, freq_hz=2.0 + seed % 3 * 0.3):
        window.add(timestamp, keypoints)
        if window.span() < WINDOW_S - 0.5:
            continue
        if next_emit is None or timestamp >= next_emit:
            features = window_features(*window.frames())
            if features is not None:
                out.append((features, int(kind == "wave"), subject))
            next_emit = timestamp + 0.5
    return out


def dataset(subjects=6):
    data = []
    for subject in range(subjects):
        data += windows_of("wave", f"P{subject:03d}", seed=subject)
        data += windows_of("still", f"P{subject:03d}", seed=subject + 10)
        data += windows_of("hand_on_face", f"P{subject:03d}", seed=subject + 20)
    return data


def test_to_arrays_keeps_labels_and_groups_aligned():
    X, y, groups = to_arrays(dataset(subjects=2))

    assert X.shape[0] == y.shape[0] == groups.shape[0]
    assert set(np.unique(y)) == {0, 1}
    assert set(groups) == {"P000", "P001"}


def test_training_is_grouped_by_subject_and_learns_the_wave():
    result = train_grouped(dataset())

    assert result["subjects"] == 6
    assert result["classifier"]["f1"] > 0.9  # evaluated on subjects the model never saw
    assert "rules" in result and result["model"] is not None


def test_training_needs_both_classes():
    only_waves = [w for w in dataset() if w[1] == 1]
    with pytest.raises(ValueError, match="WAVE and NOT WAVE"):
        train_grouped(only_waves)


def test_evaluate_windows_scores_an_external_set():
    result = train_grouped(dataset())
    external = windows_of("wave", "external", seed=99) + windows_of("still", "external", seed=98)

    metrics = evaluate_windows(result["model"], external)

    assert 0.0 <= metrics["recall"] <= 1.0
    assert metrics["windows"] == len(external)


def test_threshold_is_tuned_on_held_out_folds():
    result = train_grouped(dataset())

    assert 0.05 <= result["threshold"] <= 0.95
    assert result["classifier"]["f1"] >= result["classifier_at_half"]["f1"]  # tuning never hurts F1


def test_threshold_curve_reports_the_trade_off():
    result = train_grouped(dataset())
    curve = {round(t, 2): m for t, m in result["threshold_curve"]}

    assert len(curve) >= 5
    precisions = [m["precision"] for _, m in sorted(result["threshold_curve"])]
    assert precisions[-1] >= precisions[0]  # a stricter threshold is at least as precise
