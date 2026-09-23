"""Train the wave classifier on public skeleton datasets, then check it where it has to work: the robot.

    reachy_mini_env/bin/python -m training.train_public --ntu <ntu60_hrnet.pkl> \
        [--hmdb <hmdb51_2d.pkl>] [--max-clips-per-class 150] [--out core/models/wave_classifier_ntu.joblib]

Our own recordings are one person, one camera, one room. NTU RGB+D 60 gives ~950 waving clips from 40
subjects seen by 3 cameras, plus 59 other actions as labelled negatives. Evaluation is grouped by subject,
so the reported numbers are for people the model never saw; HMDB51 (movies) and the robot recordings are
kept as external test sets, never trained on.
"""

import argparse
from pathlib import Path

import numpy as np

from training.dataset import DATA_DIR, list_sessions, load_session, make_windows
from core.motion.detectors import RuleWaveDetector, save_classifier
from core.motion.features import FEATURE_NAMES, features_vector
from core.motion.forest import export_forest
from training.public_data import HMDB_WAVE_LABEL, NTU_WAVE_LABEL, dataset_windows
from training.train import classification_metrics, make_model

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "models" / "wave_classifier_ntu.joblib"
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]


def to_arrays(windows):
    X = np.array([features_vector(f) for f, _, _ in windows], np.float32).reshape(-1, len(FEATURE_NAMES))
    y = np.array([label for _, label, _ in windows], int)
    groups = np.array([group for _, _, group in windows])
    return X, y, groups


def train_grouped(windows, n_splits=5):
    """Cross-validate by subject (never splitting one person between train and test), then fit on everything."""
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X, y, groups = to_arrays(windows)
    if len(y) == 0:
        raise ValueError("No usable windows in the dataset")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training needs both WAVE and NOT WAVE windows")

    unique = np.unique(groups)
    cv = GroupKFold(n_splits=min(n_splits, len(unique)))
    # Probabilities, not labels: the decision threshold is chosen on these held-out predictions, never on
    # data the model was fitted on, and then travels with the saved model.
    probabilities = cross_val_predict(make_model(), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    curve = [(t, classification_metrics(y, (probabilities >= t).astype(int))) for t in THRESHOLDS]
    threshold, metrics = max(curve, key=lambda item: item[1]["f1"])

    rules = RuleWaveDetector()
    rule_predictions = np.array([int(rules.detect(f).is_wave) for f, _, _ in windows])

    return {
        "windows": int(len(y)),
        "wave_windows": int(y.sum()),
        "subjects": int(len(unique)),
        "folds": cv.get_n_splits(),
        "rules": classification_metrics(y, rule_predictions),
        "classifier": metrics,
        "classifier_at_half": classification_metrics(y, (probabilities >= 0.5).astype(int)),
        "threshold": float(threshold),
        "threshold_curve": curve,
        "model": make_model().fit(X, y),
    }


def evaluate_windows(model, windows, threshold=0.5):
    """Metrics of a fitted model on a set it was never trained on, at the tuned threshold."""
    X, y, _ = to_arrays(windows)
    predictions = (model.predict_proba(X)[:, list(model.classes_).index(1)] >= threshold).astype(int)
    metrics = classification_metrics(y, predictions)
    metrics["windows"] = int(len(y))
    metrics["wave_windows"] = int(y.sum())
    return metrics


def session_windows(data_dir=DATA_DIR):
    """Our robot-viewpoint recordings, in the same (features, label, group) shape."""
    windows = []
    for index, path in enumerate(list_sessions(data_dir)):
        for features, label, _ in make_windows(load_session(path)):
            windows.append((features, int(label == 1), f"session{index}"))
    return windows


def format_metrics(name, metrics):
    return (f"{name:<28}{metrics['accuracy']:>9.3f}{metrics['precision']:>10.3f}{metrics['recall']:>8.3f}"
            f"{metrics['f1']:>7.3f}   {metrics['confusion']}")


def format_report(result, external):
    lines = [
        f"Windows: {result['windows']} (wave {result['wave_windows']}) from {result['subjects']} subjects, "
        f"{result['folds']}-fold grouped by subject; decision threshold tuned to {result['threshold']:.2f} "
        f"(at 0.50: f1 {result['classifier_at_half']['f1']:.3f})",
        "",
        f"{'set':<28}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>7}   confusion [[TN FP] [FN TP]]",
        format_metrics("rules (same windows)", result["rules"]),
        format_metrics("classifier (unseen subjects)", result["classifier"]),
    ]
    for name, metrics in external.items():
        lines.append(format_metrics(f"{name} ({metrics['windows']} windows)", metrics))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ntu", required=True, help="path to ntu60_hrnet.pkl")
    parser.add_argument("--hmdb", help="path to hmdb51_2d.pkl (external test set)")
    parser.add_argument("--max-clips-per-class", type=int, default=150,
                        help="clips taken per action class, so the negatives stay balanced (default 150)")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="our recorded sessions (external test set)")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(argv)

    print(f"Loading NTU windows from {args.ntu} (up to {args.max_clips_per_class} clips per class)...")
    windows = dataset_windows(args.ntu, NTU_WAVE_LABEL, max_clips_per_class=args.max_clips_per_class)
    print(f"  {len(windows)} windows")
    result = train_grouped(windows)

    external = {}
    if args.hmdb:
        print(f"Loading HMDB51 windows from {args.hmdb}...")
        hmdb = dataset_windows(args.hmdb, HMDB_WAVE_LABEL, max_clips_per_class=args.max_clips_per_class)
        if hmdb:
            external["HMDB51 (movies)"] = evaluate_windows(result["model"], hmdb, result["threshold"])
    recorded = session_windows(args.data_dir)
    if recorded:
        external["our robot recordings"] = evaluate_windows(result["model"], recorded, result["threshold"])

    print()
    print(format_report(result, external))

    save_classifier(result["model"], args.out, threshold=result["threshold"])
    forest_path = export_forest(result["model"], Path(args.out).with_suffix(".npz"),
                                threshold=result["threshold"])
    print(f"\nSaved classifier to {args.out}")
    print(f"Saved numpy forest (robot runtime) to {forest_path}")


if __name__ == "__main__":
    main()
