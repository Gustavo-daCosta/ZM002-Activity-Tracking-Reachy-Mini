"""Train the exercise classifier (none, squat, arm_raise, side_bend) on the group's own recordings.

    reachy_mini_env/bin/python -m training.train_exercises

Reads every `collect/recordings/<person>/session_*.npz` written by `collect/record_exercises.py`, cuts
3 s windows inside each labeled round with the same `KeypointWindow` the live monitor uses, and scores
the forest on people it never saw (leave-one-person-out up to 5 people): the number printed is what a new
person should get.
The final model is fitted on everyone and exported as a numpy forest for `ActionMonitor`.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

from core.motion.actions import EXERCISE_MODEL
from core.motion.features import ACTION_FEATURE_NAMES, action_features_vector
from core.motion.forest import export_forest
from core.motion.window import ACTION_WINDOW_S
from training.data import ACTION_WINDOW, make_windows
from training.train_actions import N_ESTIMATORS, _predict_with_floors, tune_floors

RECORDINGS = Path(__file__).resolve().parents[1] / "collect" / "recordings"
STEP_S = 0.5
# MoveNet's own default: at 2-3 m it scores shoulders ~0.45, so the 0.5 of the public-data pipeline drops
# whole sessions. The live app uses the same value (`MoveNetBackend.default_min_score`).
MIN_SCORE = 0.3
# Each fold holds out whole people. Leave-one-person-out up to MAX_FOLDS people; beyond that the folds group
# people together, so every child the coach adds does not cost one more fit on the robot's CPU.
MAX_FOLDS = 5


def load_windows(recordings=RECORDINGS):
    """(X, y, groups): one row per window, labeled with the class name, grouped by person.

    `recordings` is one folder or several (the coach keeps children's sessions apart from the group's);
    a folder given twice is read once.
    """
    folders = [recordings] if isinstance(recordings, (str, Path)) else recordings
    folders = list(dict.fromkeys(Path(folder).resolve() for folder in folders))
    rows, labels, people = [], [], []
    for path in sorted(path for folder in folders for path in folder.glob("*/session_*.npz")):
        with np.load(path) as data:
            session = {key: data[key] for key in ("t", "keypoints", "labels", "rounds")}
            session["aspect_ratio"] = float(data["aspect_ratio"])
            classes = [str(name) for name in data["classes"]]
        windows = make_windows(session, **ACTION_WINDOW, step_s=STEP_S, min_score=MIN_SCORE)
        print(f"  {path.parent.name}/{path.name}: {len(windows)} windows")
        for features, label, _ in windows:
            rows.append(action_features_vector(features))
            labels.append(classes[label])
            people.append(path.parent.name)
    if not rows:
        raise SystemExit(f"No sessions under {recordings}")
    return np.array(rows), np.array(labels), np.array(people)


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="+", default=[str(RECORDINGS)],
                        help="one or more folders with <person>/session_*.npz")
    parser.add_argument("--out", default=str(EXERCISE_MODEL), help="where to write the model")
    args = parser.parse_args()

    print(f"Loading {', '.join(args.recordings)}...")
    X, y, people = load_windows(args.recordings)
    classes = sorted(set(y.tolist()))
    print(f"{len(y)} windows, {len(set(people))} people: "
          + ", ".join(f"{name} {int((y == name).sum())}" for name in classes))

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced_subsample",
                                   random_state=0, n_jobs=-1)
    # Every training fold holds all four classes, so predict_proba columns follow `classes` (sorted).
    n_people = len(set(people.tolist()))
    if n_people < 2:
        raise SystemExit(f"Need recordings of at least 2 people to score the model, found {n_people}.")
    folds = GroupKFold(n_splits=min(MAX_FOLDS, n_people))
    probabilities = cross_val_predict(model, X, y, groups=people, cv=folds, method="predict_proba")
    floors = tune_floors(probabilities, y, classes)
    predicted = _predict_with_floors(probabilities, classes, floors)

    print(f"\n{folds.n_splits}-fold by person (the model never saw the person it is scored on):")
    print(classification_report(y, predicted, labels=classes, digits=3, zero_division=0))
    print("Confusion (rows = truth, columns = predicted):", " ".join(classes))
    print(confusion_matrix(y, predicted, labels=classes))
    for person in sorted(set(people)):
        mine = people == person
        print(f"  {person}: accuracy {float((predicted[mine] == y[mine]).mean()):.3f}")
    print("Confidence floors:", floors)

    model.fit(X, y)
    path = export_forest(model, args.out, feature_names=ACTION_FEATURE_NAMES, window_s=ACTION_WINDOW_S,
                         classes=classes, thresholds=floors)
    print(f"\nSaved numpy forest to {path}")


if __name__ == "__main__":
    main()
