"""Train the CMAS classifier (none, chair_rise, pick_up) on NTU RGB+D, optionally with our own recordings.

    reachy_mini_env/bin/python -m training.train_cmas --ntu datasets/ntu60_hrnet.pkl
    reachy_mini_env/bin/python -m training.train_cmas --ntu datasets/ntu60_hrnet.pkl --with-recordings --out /tmp/cmas_rec.npz

Recognizes that CMAS items 12 (chair rise) and 14 (pick-up) were performed; it does not score them. Items 7
and 8 (arms above the head, held) are a rule in `core/motion/cmas.py`, not this model.

Default (the shipped model): NTU A9 standing up and A6 pick up, other NTU actions as `none` (A8 sitting
down skipped). It scored higher on our one room session than the model also trained on our recordings, and
worked live on the Mac and the robot. `--with-recordings` adds the windows of `collect/recordings/` as
`none` except the squats (see `main`), and the sessions of `collect/recordings_cmas/`
(`record_exercises.py --protocol cmas`), whose arms-up rounds are `none` too. Folds hold out whole people and
a person's two folders share one group, so the `cmas` rows of the report come from a model that never saw
that person in a room.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict

from core.motion.actions import CMAS_MODEL
from core.motion.features import ACTION_FEATURE_NAMES
from core.motion.forest import NONE, export_forest
from core.motion.window import ACTION_WINDOW_S
from training.data import NTU_CMAS_LABELS, ntu_windows
from training.train_actions import N_ESTIMATORS, _predict_with_floors, to_arrays, tune_floors
from training.train_exercises import RECORDINGS, load_windows

CMAS_RECORDINGS = Path(__file__).resolve().parents[1] / "collect" / "recordings_cmas"
CLASSES = (NONE, "chair_rise", "pick_up")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ntu", required=True, help="path to ntu60_hrnet.pkl")
    parser.add_argument("--max-clips-per-class", type=int, default=20,
                        help="cap on NTU clips per negative class (positives are never capped)")
    parser.add_argument("--splits", type=int, default=5, help="GroupKFold folds")
    parser.add_argument("--with-recordings", action="store_true",
                        help="also train on (and score) collect/recordings and collect/recordings_cmas")
    parser.add_argument("--out", default=str(CMAS_MODEL), help="where to write the model")
    args = parser.parse_args()

    print(f"Loading NTU windows from {args.ntu}...", flush=True)
    # Full-body only: chair rise and pick-up are leg movements, a waist-up camera cannot see them.
    X, y, groups = to_arrays(ntu_windows(args.ntu, args.max_clips_per_class, labels=NTU_CMAS_LABELS,
                                         crop_levels={"full": ()}))
    sources = np.full(len(y), "ntu", dtype=object)
    for source, folder in (("physio", RECORDINGS), ("cmas", CMAS_RECORDINGS)) if args.with_recordings else ():
        if not any(folder.glob("*/session_*.npz")):
            print(f"No sessions under {folder}, skipped")
            continue
        X_rec, y_rec, people = load_windows(folder)
        # A squat is a chair rise without the chair: as `none` it made the model miss every rise in a room
        # (and fire pick_up on 73 squats). A CMAS session never asks for squats, so they are left out.
        keep = y_rec != "squat"
        X_rec, y_rec, people = X_rec[keep], y_rec[keep], people[keep]
        y_rec =np.where(np.isin(y_rec, CLASSES), y_rec, NONE).astype(object)
        X, y = np.concatenate([X, X_rec]), np.concatenate([y, y_rec])
        groups = np.concatenate([groups, np.array([f"rec:{person}" for person in people], dtype=object)])
        sources = np.concatenate([sources, np.full(len(y_rec), source, dtype=object)])

    classes = sorted(set(y.tolist()))
    print(f"{len(y)} windows, {len(set(groups.tolist()))} people: "
          + ", ".join(f"{name} {int((y == name).sum())}" for name in classes))
    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, class_weight="balanced_subsample",
                                   random_state=0, n_jobs=-1)
    folds = GroupKFold(n_splits=args.splits)
    probabilities = cross_val_predict(model, X, y, groups=groups, cv=folds, method="predict_proba")
    floors = tune_floors(probabilities, y, classes)
    predicted = _predict_with_floors(probabilities, classes, floors)

    print(f"\n{folds.n_splits}-fold by person (the model never saw the person it is scored on).")
    print("Confidence floors:", floors)
    for source in ("ntu", "physio", "cmas"):
        rows = sources == source
        if not rows.any():
            continue
        print(f"\n--- {source} ({int(rows.sum())} windows) ---")
        print(classification_report(y[rows], predicted[rows], labels=classes, digits=3, zero_division=0))
        print("Confusion (rows = truth, columns = predicted):", " ".join(classes))
        print(confusion_matrix(y[rows], predicted[rows], labels=classes))

    model.fit(X, y)
    path = export_forest(model, args.out, feature_names=ACTION_FEATURE_NAMES, window_s=ACTION_WINDOW_S,
                         classes=classes, thresholds=floors)
    print(f"\nSaved numpy forest to {path}")


if __name__ == "__main__":
    main()
