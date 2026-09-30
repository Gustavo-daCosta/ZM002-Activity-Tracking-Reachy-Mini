"""Train the wave classifier on NTU RGB+D 60, then check it on HMDB51 and on our robot recordings.

    reachy_mini_env/bin/python -m training.train_wave --ntu datasets/ntu60_hrnet.pkl \
        [--hmdb datasets/hmdb51_2d.pkl] [--max-clips-per-class 150] [--out core/models/wave_classifier_ntu.npz]

Neighbouring windows are almost identical, so evaluation is grouped by subject: the numbers are for
people the model never saw. HMDB51 and our recordings are external test sets, never trained on.
"""

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, cross_val_predict

from core.motion.features import FEATURE_NAMES, features_vector
from core.motion.forest import export_forest
from core.motion.wave import DEFAULT_MODEL_PATH, RuleWaveDetector
from training.data import (
    DATA_DIR, HMDB_WAVE_LABEL, NTU_WAVE_LABEL, list_sessions, load_session, make_windows, wave_dataset_windows,
)

THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]


def make_model():
    """The wave forest, untrained."""
    return RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=0)


def classification_metrics(y_true, y_pred):
    """Accuracy, precision, recall, F1 and the [[TN FP] [FN TP]] confusion matrix for the wave class."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1,
                                                               zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1),
            "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}


def to_arrays(windows):
    """(X, y, groups) from (features, label, group) triples."""
    X = np.array([features_vector(f) for f, _, _ in windows], np.float32).reshape(-1, len(FEATURE_NAMES))
    y = np.array([label for _, label, _ in windows], int)
    return X, y, np.array([group for _, _, group in windows])


def train_grouped(windows, n_splits=5):
    """Cross-validate by subject, tune the threshold on held-out probabilities, then fit on everything.

    Raises:
        ValueError: No windows, or only one class.
    """
    X, y, groups = to_arrays(windows)
    if len(y) == 0:
        raise ValueError("No usable windows in the dataset")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training needs both WAVE and NOT WAVE windows")
    unique = np.unique(groups)
    cv = GroupKFold(n_splits=min(n_splits, len(unique)))
    probabilities = cross_val_predict(make_model(), X, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
    curve = [(t, classification_metrics(y, (probabilities >= t).astype(int))) for t in THRESHOLDS]
    threshold, metrics = max(curve, key=lambda item: item[1]["f1"])
    rules = RuleWaveDetector()
    rule_predictions = np.array([int(rules.detect(f).is_wave) for f, _, _ in windows])
    return {
        "windows": int(len(y)), "wave_windows": int(y.sum()), "subjects": int(len(unique)),
        "folds": cv.get_n_splits(), "rules": classification_metrics(y, rule_predictions), "classifier": metrics,
        "classifier_at_half": classification_metrics(y, (probabilities >= 0.5).astype(int)),
        "threshold": float(threshold), "model": make_model().fit(X, y),
    }


def evaluate_windows(model, windows, threshold=0.5):
    """Metrics of a fitted model on windows it was never trained on, at `threshold`."""
    X, y, _ = to_arrays(windows)
    predictions = (model.predict_proba(X)[:, list(model.classes_).index(1)] >= threshold).astype(int)
    return dict(classification_metrics(y, predictions), windows=int(len(y)), wave_windows=int(y.sum()))


def session_windows(data_dir=DATA_DIR):
    """Our robot-viewpoint recordings as (features, label, group) triples."""
    return [(features, int(label == 1), f"session{index}")
            for index, path in enumerate(list_sessions(data_dir))
            for features, label, _ in make_windows(load_session(path))]


def format_metrics(name, metrics):
    """One report row."""
    return (f"{name:<28}{metrics['accuracy']:>9.3f}{metrics['precision']:>10.3f}{metrics['recall']:>8.3f}"
            f"{metrics['f1']:>7.3f}   {metrics['confusion']}")


def format_report(result, external):
    """Render `train_grouped` output plus the external sets as text."""
    lines = [
        f"Windows: {result['windows']} (wave {result['wave_windows']}) from {result['subjects']} subjects, "
        f"{result['folds']}-fold grouped by subject; decision threshold tuned to {result['threshold']:.2f} "
        f"(at 0.50: f1 {result['classifier_at_half']['f1']:.3f})",
        "",
        f"{'set':<28}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>7}   confusion [[TN FP] [FN TP]]",
        format_metrics("rules (same windows)", result["rules"]),
        format_metrics("classifier (unseen subjects)", result["classifier"]),
    ]
    lines += [format_metrics(f"{name} ({metrics['windows']} windows)", metrics) for name, metrics in external.items()]
    return "\n".join(lines)


def main(argv=None):
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ntu", required=True, help="path to ntu60_hrnet.pkl")
    parser.add_argument("--hmdb", help="path to hmdb51_2d.pkl (external test set)")
    parser.add_argument("--max-clips-per-class", type=int, default=150,
                        help="clips taken per action class, so the negatives stay balanced (default 150)")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="our recorded sessions (external test set)")
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args(argv)

    print(f"Loading NTU windows from {args.ntu} (up to {args.max_clips_per_class} clips per class)...")
    windows = wave_dataset_windows(args.ntu, NTU_WAVE_LABEL, max_clips_per_class=args.max_clips_per_class)
    print(f"  {len(windows)} windows")
    result = train_grouped(windows)

    external = {}
    if args.hmdb:
        print(f"Loading HMDB51 windows from {args.hmdb}...")
        hmdb = wave_dataset_windows(args.hmdb, HMDB_WAVE_LABEL, max_clips_per_class=args.max_clips_per_class)
        if hmdb:
            external["HMDB51 (movies)"] = evaluate_windows(result["model"], hmdb, result["threshold"])
    recorded = session_windows(args.data_dir)
    if recorded:
        external["our robot recordings"] = evaluate_windows(result["model"], recorded, result["threshold"])

    print()
    print(format_report(result, external))
    print(f"\nSaved numpy forest to {export_forest(result['model'], args.out, threshold=result['threshold'])}")


if __name__ == "__main__":
    main()
