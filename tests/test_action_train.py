"""Training helpers: array conversion, per-class floors, grouped evaluation, honesty diagnostics.

The diagnostics are tested as hard as the scores. A forest trained on studio NTU footage for `wave` and
`clapping` and on handheld UCF101 footage for `pushup`, `squat` and `jumping_jacks` can score well by
recognizing the *camera*, so the per-source breakdown, the per-fold spread of the thinnest class and the
feature importances are deliverables, not decoration: a test that lets them silently disappear would let
the whole report lie.
"""

import numpy as np
import pytest

from core.motion.actions import ACTION_WINDOW_S, ACTIONS, NONE
from core.motion.actions.features import ACTION_FEATURE_NAMES
from training.actions.train import (
    crops_of, format_report, recorded_windows, spans_of, to_arrays, train_grouped, tune_floors,
)


def window(value, label, group, span=3.0, crop="full"):
    """A window whose features are all `value`: enough for a classifier to separate on."""
    features = {name: float(value) for name in ACTION_FEATURE_NAMES}
    features["span"] = float(span)
    features["crop"] = crop
    return (features, label, group)


# --- arrays -----------------------------------------------------------------------------------------


def test_to_arrays_keeps_the_feature_order_and_the_groups():
    windows = [window(1.0, "wave", "ntu:P001"), window(2.0, "none", "ucf:PushUps_g01")]
    X, y, groups = to_arrays(windows)
    assert X.shape == (2, len(ACTION_FEATURE_NAMES))
    assert X.dtype == np.float32
    assert list(y) == ["wave", "none"]
    assert list(groups) == ["ntu:P001", "ucf:PushUps_g01"]


def test_to_arrays_ignores_the_span_diagnostic():
    """`span` travels with every window but is deliberately not a column.

    It correlates with both the source and the class -- NTU clips are short, UCF101 clips are long -- so a
    span column would hand the forest a "short window => NTU => wave or clapping" shortcut that still
    generalizes inside a dataset and inflates the score.
    """
    X, _, _ = to_arrays([window(1.0, "wave", "ntu:P001", span=2.1)])
    assert X.shape == (1, len(ACTION_FEATURE_NAMES))
    assert "span" not in ACTION_FEATURE_NAMES
    assert list(spans_of([window(1.0, "wave", "n:1", span=2.1),
                          window(1.0, "none", "n:2", span=3.0)])) == [2.1, 3.0]


# --- floors -----------------------------------------------------------------------------------------


def test_tune_floors_picks_a_floor_per_class():
    classes = ["none", "wave"]
    # Waves scored 0.6, non-waves 0.4: any floor in between is perfect, above 0.6 loses every wave.
    probabilities = np.array([[0.4, 0.6]] * 10 + [[0.6, 0.4]] * 10)
    y = np.array(["wave"] * 10 + ["none"] * 10)
    floors = tune_floors(probabilities, y, classes)
    assert set(floors) == {"none", "wave"}
    assert floors["wave"] <= 0.6


def test_tune_floors_raises_a_floor_to_shed_false_positives():
    classes = ["none", "wave"]
    # Real waves are confident (0.9); ten `none` windows leak in at 0.55. A floor above 0.55 is perfect.
    probabilities = np.array([[0.1, 0.9]] * 10 + [[0.45, 0.55]] * 10)
    y = np.array(["wave"] * 10 + ["none"] * 10)
    floors = tune_floors(probabilities, y, classes)
    assert 0.55 < floors["wave"] <= 0.9
    assert floors[NONE] == 0.0   # `none` is the fallback, it never has a floor of its own


# --- grouped training -------------------------------------------------------------------------------


def test_train_grouped_separates_two_obvious_classes_across_subjects():
    windows = []
    for subject in range(6):
        for _ in range(8):
            windows.append(window(10.0, "wave", f"ntu:P{subject:03d}"))
            windows.append(window(-10.0, "none", f"ntu:P{subject:03d}"))
    result = train_grouped(windows, n_splits=3, seed=0)

    assert result["classes"] == [name for name in ACTIONS if name in {"none", "wave"}]
    assert set(result["floors"]) == {"none", "wave"}
    assert result["report"]["wave"]["f1"] > 0.9
    # The confusion matrix is square over the classes present, and counts every window once.
    confusion = result["confusion"]
    assert confusion.shape == (2, 2)
    assert confusion.sum() == len(windows)


def test_train_grouped_never_splits_a_subject_across_folds():
    windows = [window(float(i % 3), "wave" if i % 2 else "none", f"ntu:P{i % 4:03d}")
               for i in range(80)]
    result = train_grouped(windows, n_splits=4, seed=0)
    # Every window got an out-of-fold prediction: nothing was silently dropped.
    assert result["confusion"].sum() == len(windows)
    for fold in result["folds"]:
        assert not (set(fold["test_groups"]) & set(fold["train_groups"]))


def test_train_grouped_refuses_fewer_groups_than_folds():
    windows = [window(1.0, "wave", "ntu:P001"), window(0.0, "none", "ntu:P001")]
    with pytest.raises(ValueError, match="groups"):
        train_grouped(windows, n_splits=5)


def test_train_grouped_reports_every_class_per_fold():
    """The thinnest class rests on 19 groups; its pooled F1 hides how much it moves between folds."""
    windows = []
    for subject in range(6):
        for _ in range(6):
            windows.append(window(10.0, "wave", f"ntu:P{subject:03d}"))
            windows.append(window(-10.0, "none", f"ucf:Other_g{subject:02d}"))
        windows.append(window(3.0, "pushup", f"ucf:PushUps_g{subject:02d}"))
    result = train_grouped(windows, n_splits=3, seed=0)

    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert set(fold["report"]) <= set(result["classes"])
        assert fold["report"]["wave"]["support"] > 0
    # Each fold's supports add up to the pooled support: every window is out-of-fold exactly once.
    for name in result["classes"]:
        pooled = result["report"][name]["support"]
        assert sum(f["report"].get(name, {"support": 0})["support"] for f in result["folds"]) == pooled


def test_train_grouped_records_a_fold_that_never_saw_a_class():
    """A class absent from a fold's training half cannot be predicted there; say so, never hide it."""
    windows = []
    for subject in range(4):
        for _ in range(6):
            windows.append(window(10.0, "wave", f"ntu:P{subject:03d}"))
            windows.append(window(-10.0, "none", f"ucf:Other_g{subject:02d}"))
    # A single squat group: exactly one fold holds it out, and that fold trains without the class.
    for _ in range(4):
        windows.append(window(5.0, "squat", "ucf:Squats_g99"))
    result = train_grouped(windows, n_splits=4, seed=0)

    missing = [name for fold in result["folds"] for name in fold["missing_in_train"]]
    assert "squat" in missing
    assert result["confusion"].sum() == len(windows)


# --- the honesty diagnostics ------------------------------------------------------------------------


def test_train_grouped_breaks_none_down_by_source():
    """If NTU negatives and UCF101 negatives score very differently, the model is a camera detector."""
    windows = []
    for subject in range(6):
        for _ in range(5):
            windows.append(window(10.0, "wave", f"ntu:P{subject:03d}"))
            windows.append(window(-10.0, NONE, f"ntu:P{subject:03d}"))
            windows.append(window(-9.0, NONE, f"ucf:Other_g{subject:02d}"))
    result = train_grouped(windows, n_splits=3, seed=0)

    per_source = result["per_source"]
    assert set(per_source) == {"ntu", "ucf"}
    assert per_source["ntu"]["report"][NONE]["support"] == 30
    assert per_source["ucf"]["report"][NONE]["support"] == 30
    # A per-source confusion matrix over the same classes, in the same order.
    assert per_source["ntu"]["confusion"].shape == (len(result["classes"]),) * 2
    assert sum(source["confusion"].sum() for source in per_source.values()) == len(windows)


def test_train_grouped_reports_feature_importances_highest_first():
    windows = []
    rng = np.random.default_rng(0)
    for subject in range(6):
        for _ in range(8):
            noise = {name: float(rng.normal()) for name in ACTION_FEATURE_NAMES}
            for label, value in (("wave", 10.0), (NONE, -10.0)):
                features = dict(noise, span=3.0)
                features["wrist_y_amplitude"] = value    # the only informative column
                windows.append((features, label, f"ntu:P{subject:03d}"))
    result = train_grouped(windows, n_splits=3, seed=0)

    importances = result["importances"]
    assert [name for name, _ in importances][:1] == ["wrist_y_amplitude"]
    assert len(importances) == len(ACTION_FEATURE_NAMES)
    assert all(earlier >= later for (_, earlier), (_, later) in zip(importances, importances[1:]))


def test_train_grouped_correlates_span_with_the_predictions():
    """Span is not a feature, but a prediction that tracks it is the shortcut we refused to hand over."""
    windows = []
    for subject in range(6):
        for _ in range(8):
            windows.append(window(10.0, "wave", f"ntu:P{subject:03d}", span=2.0))
            windows.append(window(-10.0, NONE, f"ucf:Other_g{subject:02d}", span=3.0))
    result = train_grouped(windows, n_splits=3, seed=0)

    correlation = result["span_correlation"]
    assert set(correlation) == set(result["classes"])
    # Waves are exactly the short windows here, so the correlation is strongly negative.
    assert correlation["wave"] < -0.5


def test_format_report_shows_the_diagnostics_and_the_external_sets():
    windows = []
    for subject in range(6):
        for _ in range(6):
            windows.append(window(10.0, "wave", f"ntu:P{subject:03d}"))
            windows.append(window(-10.0, NONE, f"ucf:Other_g{subject:02d}"))
    result = train_grouped(windows, n_splits=3, seed=0)
    result["external"] = {"Our own recordings (waves only)": {"wave": {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 7}}}
    text = format_report(result)

    assert str(ACTION_WINDOW_S) in text
    assert "Our own recordings (waves only)" in text
    for needle in ("per fold", "by source", "importances", "span"):
        assert needle in text.lower()
    assert "wrist_y_amplitude" in text        # the importances are printed by name


# --- our own recordings, through the generalized make_windows ---------------------------------------


def bobbing_session(rounds=2, round_s=4.0, fps=20, amplitude=0.06, freq_hz=1.0):
    """A session whose whole body rises and falls in the frame: labelled wave, then other.

    It exists to prove the trunk-height channel is wired through `make_windows`. Trunk height is the
    shoulder midpoint's offset from the frame centre in torso lengths, so a body that bobs vertically is
    the one thing that makes `trunk_height_amplitude` nonzero -- and with `extra=None` it would be exactly
    the 0.0 "not measured" sentinel instead.
    """
    t, keypoints, labels, round_ids = [], [], [], []
    for round_id in range(rounds):
        for i in range(int(round_s * fps)):
            local = i / fps
            offset = amplitude * np.sin(2 * np.pi * freq_hz * local)
            frame = np.zeros((17, 3), np.float32)
            for index, (x, y) in {
                0: (0.50, 0.30), 5: (0.42, 0.42), 6: (0.58, 0.42), 7: (0.38, 0.55), 8: (0.62, 0.55),
                9: (0.36, 0.68), 10: (0.64, 0.68), 11: (0.45, 0.62), 12: (0.55, 0.62),
                13: (0.45, 0.78), 14: (0.55, 0.78), 15: (0.45, 0.92), 16: (0.55, 0.92),
            }.items():
                frame[index] = (x, y + offset, 0.9)
            t.append(round_id * (round_s + 1.0) + local)
            keypoints.append(frame)
            labels.append(1 if round_id == 0 else 0)
            round_ids.append(round_id)
    return {"t": np.array(t, np.float64), "keypoints": np.array(keypoints, np.float32),
            "labels": np.array(labels, np.int8), "rounds": np.array(round_ids, np.int16),
            "pose_model": "synthetic", "aspect_ratio": 1.0}


def test_recorded_windows_uses_the_action_features_and_the_trunk_height_channel(tmp_path):
    from training.dataset import save_session

    session = bobbing_session()
    save_session(tmp_path / "session_20260923_120000.npz", session["t"], session["keypoints"],
                 session["labels"], session["rounds"], "synthetic", 1.0)
    windows = recorded_windows(tmp_path)

    assert windows, "the recordings must yield action windows"
    features, label, group = windows[0]
    assert set(features) >= set(ACTION_FEATURE_NAMES)
    assert {label for _, label, _ in windows} == {"wave", NONE}
    assert group == "rec:session0"        # namespaced like every dataset group key
    from training.actions.train import source_of

    assert source_of(group) == "rec"
    # Wired-in trunk-height channel: without `extra=trunk_height_parts` this is the 0.0 sentinel.
    assert max(f["trunk_height_amplitude"] for f, _, _ in windows) > 0.01
    # The 3 s action window, not the wave pipeline's 1.5 s one.
    assert max(f["span"] for f, _, _ in windows) > 2.0


def test_make_windows_without_the_extra_channel_leaves_trunk_height_unmeasured(tmp_path):
    """The complement of the test above: it is the `extra` channel doing the work, not luck."""
    from core.motion.actions.features import action_features
    from core.motion.actions.normalize import normalize_to_torso
    from training.dataset import make_windows

    windows = make_windows(bobbing_session(), window_s=ACTION_WINDOW_S,
                           features_of=action_features, normalizer=normalize_to_torso)
    assert windows
    assert all(f["trunk_height_amplitude"] == 0.0 for f, _, _ in windows)


# --- crop augmentation: the two regimes must be reported apart ---------------------------------------


def test_to_arrays_ignores_the_crop_diagnostic():
    """`crop` says which regime a window came from. `lower_body_valid_frac` is the feature that tells the
    forest the same thing, and unlike the string it is a real measurement a live frame can produce."""
    X, _, _ = to_arrays([window(1.0, "wave", "ntu:P001", crop="waist")])
    assert X.shape == (1, len(ACTION_FEATURE_NAMES))
    assert "crop" not in ACTION_FEATURE_NAMES
    assert list(crops_of([window(1.0, "wave", "n:1", crop="full"),
                          window(1.0, "none", "n:2", crop="waist")])) == ["full", "waist"]


def test_train_grouped_scores_each_crop_regime_separately():
    """A pooled number would hide exactly the failure the augmentation exists to fix."""
    windows = []
    for subject in range(6):
        for _ in range(5):
            for crop in ("full", "legs", "waist"):
                windows.append(window(10.0, "wave", f"ntu:P{subject:03d}", crop=crop))
                windows.append(window(-10.0, NONE, f"ntu:P{subject:03d}", crop=crop))
    result = train_grouped(windows, n_splits=3, seed=0)

    per_regime = result["per_regime"]
    assert list(per_regime) == ["full", "legs", "waist"]
    for block in per_regime.values():
        assert block["report"]["wave"]["support"] == 30
        assert block["confusion"].shape == (len(result["classes"]),) * 2
    assert sum(block["confusion"].sum() for block in per_regime.values()) == len(windows)
    text = format_report(result)
    assert "by crop regime" in text.lower()
    for name in per_regime:
        assert name in text


def test_recorded_windows_are_tagged_as_their_own_regime(tmp_path):
    from training.dataset import save_session

    session = bobbing_session()
    save_session(tmp_path / "session_20260923_120000.npz", session["t"], session["keypoints"],
                 session["labels"], session["rounds"], "synthetic", 1.0)
    windows = recorded_windows(tmp_path)
    # Not "full" and not a synthetic crop either: this is what the robot's camera actually produced.
    assert {features["crop"] for features, _, _ in windows} == {"recorded"}


# --- the crop fraction: a measured trade-off, not a fixed doubling ------------------------------------


def both_regimes(n_subjects=6, per=5):
    windows = []
    for subject in range(n_subjects):
        for _ in range(per):
            for crop in ("full", "legs", "waist"):
                windows.append(window(10.0, "wave", f"ntu:P{subject:03d}", crop=crop))
                windows.append(window(-10.0, NONE, f"ntu:P{subject:03d}", crop=crop))
    return windows


def test_crop_fraction_subsamples_only_the_training_half():
    """The evaluation set is *fixed* across settings; only the training mix moves.

    Generating fewer cropped windows would change the test set too, and then the curve would compare
    scores measured on different data - unreadable. So every window is always scored out-of-fold, and
    `crop_fraction` decides only how many cropped windows a fold is allowed to *learn* from.
    """
    windows = both_regimes()
    cropped = sum(1 for features, _, _ in windows if features["crop"] != "full")
    full = len(windows) - cropped

    for fraction, expected in ((0.0, full), (0.5, full + cropped // 2), (1.0, len(windows))):
        result = train_grouped(windows, n_splits=3, seed=0, crop_fraction=fraction)
        assert result["n_train_windows"] == expected, fraction
        assert result["crop_fraction"] == fraction
        # Every window still gets an out-of-fold prediction, whatever the fraction.
        assert result["confusion"].sum() == len(windows)
        assert set(result["per_regime"]) == {"full", "legs", "waist"}


def test_crop_fraction_zero_never_trains_on_a_cropped_window():
    """Which is the baseline this whole fix exists to beat, so it has to be reproducible exactly."""
    result = train_grouped(both_regimes(), n_splits=3, seed=0, crop_fraction=0.0)
    assert result["n_train_windows"] * 3 == len(both_regimes())


def test_crop_fraction_is_reproducible_and_rejects_nonsense():
    windows = both_regimes()
    first = train_grouped(windows, n_splits=3, seed=0, crop_fraction=0.5)
    again = train_grouped(windows, n_splits=3, seed=0, crop_fraction=0.5)
    assert first["n_train_windows"] == again["n_train_windows"]
    assert np.array_equal(first["confusion"], again["confusion"])
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="crop_fraction"):
            train_grouped(windows, n_splits=3, crop_fraction=bad)


def test_each_fold_is_scored_per_regime_too():
    """A 2-point gap between two crop settings is only meaningful against the per-fold spread.

    Deciding between `crop_fraction` settings on a pooled full-body number alone invites reading fold
    noise as a difference, which is the mistake the push-up per-fold spread already warned about.
    """
    windows = both_regimes(n_subjects=6, per=5)
    result = train_grouped(windows, n_splits=3, seed=0, crop_fraction=0.5)

    for fold in result["folds"]:
        assert set(fold["per_regime"]) == {"full", "legs", "waist"}
        for regime, report in fold["per_regime"].items():
            assert set(report) <= set(result["classes"])
    # Each regime's per-fold supports add up to that regime's pooled support: no window scored twice.
    for regime, block in result["per_regime"].items():
        for name, scores in block["report"].items():
            per_fold = sum(fold["per_regime"][regime].get(name, {"support": 0})["support"]
                           for fold in result["folds"])
            assert per_fold == scores["support"], (regime, name)


# --- the artefact that actually ships ----------------------------------------------------------------


def test_the_shipped_forest_matches_the_feature_contract():
    """The `.npz` is the only file the robot runs, and nothing else in the suite opens it.

    The robot has no scikit-learn, so the joblib bundle is a Mac-side convenience and this export is the
    deployed model. It is a tracked file, so its absence is a defect rather than a missing optional
    asset: this asserts it exists instead of skipping. An earlier version skipped when the path was
    missing, which let a directory rename silently retire the only test that opens the shipped model.
    """
    from pathlib import Path

    from core.motion.actions import ACTIONS
    from core.motion.forest import load_forest

    path = Path(__file__).resolve().parents[1] / "core" / "models" / "action_classifier.npz"
    assert path.exists(), f"{path} is tracked and must be present; retrain with training.actions.train"

    forest = load_forest(path, ACTION_FEATURE_NAMES, ACTION_WINDOW_S)
    assert tuple(forest.feature_names) == ACTION_FEATURE_NAMES
    assert len(forest.feature_names) == len(ACTION_FEATURE_NAMES)
    # All six actions, in ACTIONS order: `export_multiclass_forest` reorders each tree's columns to match,
    # so a wrong order here would mean every probability is attributed to the wrong action.
    assert list(forest.classes) == list(ACTIONS)
    # The floors documented in the report and the commit message for the shipped crop_fraction=0.25 model.
    assert forest.thresholds == pytest.approx(
        {"none": 0.00, "wave": 0.45, "pushup": 0.05, "squat": 0.05,
         "clapping": 0.40, "jumping_jacks": 0.05})
    # It predicts one of its own classes from a real feature vector, rather than only loading.
    name, probability = forest.predict(np.zeros(len(ACTION_FEATURE_NAMES), np.float32))
    assert name in ACTIONS and 0.0 <= probability <= 1.0
