"""Train the wave classifier on recorded sessions and compare it with the rules.

    reachy_mini_env/bin/python -m training.train [--data-dir training/data/wave] [--out core/models/wave_classifier.joblib]

Neighbouring windows are almost identical, so evaluation never mixes them between train and test:
with 2+ sessions it leaves one session out at a time, with 1 session it groups by round.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict

from training.dataset import DATA_DIR, list_sessions, load_session, make_windows
from core.motion.detectors import DEFAULT_MODEL_PATH, RuleWaveDetector, save_classifier
from core.motion.features import FEATURE_NAMES, features_vector
from core.motion.forest import export_forest


def make_model():
    return RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=0)


def classification_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def build_dataset(sessions):
    features, labels, session_ids, round_ids = [], [], [], []
    for session_index, session in enumerate(sessions):
        for window_features, label, round_id in make_windows(session):
            features.append(window_features)
            labels.append(label)
            session_ids.append(session_index)
            round_ids.append(session_index * 1000 + round_id)
    X = np.array([features_vector(f) for f in features], np.float32).reshape(-1, len(FEATURE_NAMES))
    return features, X, np.array(labels, int), np.array(session_ids), np.array(round_ids)


def train_and_evaluate(sessions):
    features, X, y, session_ids, round_ids = build_dataset(sessions)
    if len(y) == 0:
        raise ValueError("No usable windows: record a session first (python -m training.record)")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Training needs both WAVE and NOT WAVE windows; record a full session")

    if len(np.unique(session_ids)) >= 2:
        groups, cv, grouping = session_ids, LeaveOneGroupOut(), "session"
    else:
        groups, grouping = round_ids, "round"
        cv = GroupKFold(n_splits=min(5, len(np.unique(round_ids))))
    out_of_fold = cross_val_predict(make_model(), X, y, groups=groups, cv=cv)

    rules = RuleWaveDetector()
    rule_predictions = np.array([int(rules.detect(f).is_wave) for f in features])

    model = make_model().fit(X, y)
    return {
        "windows": int(len(y)),
        "wave_windows": int(y.sum()),
        "sessions": int(len(np.unique(session_ids))),
        "groups": grouping,
        "rules": classification_metrics(y, rule_predictions),
        "classifier": classification_metrics(y, out_of_fold),
        "importances": dict(zip(FEATURE_NAMES, map(float, model.feature_importances_))),
        "medians": {
            name: {"wave": float(np.median(X[y == 1, i])), "other": float(np.median(X[y == 0, i]))}
            for i, name in enumerate(FEATURE_NAMES)
        },
        "model": model,
    }


def format_report(result):
    lines = [
        f"Sessions: {result['sessions']}  windows: {result['windows']} (wave {result['wave_windows']})  "
        f"evaluation grouped by {result['groups']}",
        "",
        f"{'detector':<11}{'accuracy':>9}{'precision':>10}{'recall':>8}{'f1':>7}   confusion [[TN FP] [FN TP]]",
    ]
    for name in ("rules", "classifier"):
        m = result[name]
        lines.append(
            f"{name:<11}{m['accuracy']:>9.3f}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}   {m['confusion']}"
        )
    lines += ["", f"{'feature':<18}{'importance':>11}{'median wave':>13}{'median other':>14}"]
    for name in FEATURE_NAMES:
        medians = result["medians"][name]
        lines.append(
            f"{name:<18}{result['importances'][name]:>11.3f}{medians['wave']:>13.2f}{medians['other']:>14.2f}"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--out", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args(argv)

    paths = list_sessions(args.data_dir)
    print(f"Loading {len(paths)} session(s) from {args.data_dir}")
    try:
        result = train_and_evaluate([load_session(path) for path in paths])
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(format_report(result))
    save_classifier(result["model"], args.out)
    forest_path = export_forest(result["model"], Path(args.out).with_suffix(".npz"))
    print(f"\nSaved classifier to {args.out}")
    print(f"Saved numpy forest (robot runtime, no scikit-learn) to {forest_path}")


if __name__ == "__main__":
    main()
