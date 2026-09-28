"""Train the multiclass action classifier on public skeleton datasets, and say honestly what it can do.

    reachy_mini_env/bin/python -m training.actions.train \
        --ntu datasets/ntu60_hrnet.pkl --ucf datasets/ucf101_hrnet.pkl \
        --hmdb datasets/hmdb51_2d.pkl --max-clips-per-class 3 --crop-fraction 0.25

Validation is grouped by person: windows from one clip overlap heavily and a random split would report a
score we would not see live.

Squats, push-ups and jumping jacks exist only in handheld UCF101 footage; waving and clapping only in
studio NTU footage. A forest can therefore learn which *dataset* a window came from, so the report also
gives per-source `none` recall, feature importances, per-fold scores and the correlation of the
predictions with the window span (deliberately not a feature).
"""

import argparse
from pathlib import Path

import numpy as np

from core.motion.actions import ACTION_WINDOW_S, ACTIONS, NONE
from core.motion.actions.features import (
    ACTION_FEATURE_NAMES, action_features, action_features_vector,
)
from core.motion.actions.normalize import normalize_to_torso, trunk_height_parts
from core.motion.forest import export_forest
from training.actions.datasets import CROP_LEVELS, hmdb_windows, ntu_windows, ucf_windows

FLOORS = [round(0.05 * step, 2) for step in range(1, 19)]
N_ESTIMATORS = 300
# core/models, not training/models: the exported model ships with core/ to the robot.
MODELS_DIR = Path(__file__).resolve().parents[2] / "core" / "models"
DEFAULT_OUT = MODELS_DIR / "action_classifier.npz"


def to_arrays(windows):
    """(X, y, groups) from (features, label, group) triples, in `ACTION_FEATURE_NAMES` order."""
    X = np.array([action_features_vector(features) for features, _, _ in windows], np.float32)
    y = np.array([label for _, label, _ in windows], dtype=object)
    groups = np.array([group for _, _, group in windows], dtype=object)
    return X, y, groups


def spans_of(windows):
    """Per-window span in seconds (0.0 when missing); diagnostics only."""
    return np.array([float(features.get("span", 0.0)) for features, _, _ in windows], np.float64)


def crops_of(windows):
    """Per-window crop regime ("full" when missing)."""
    return [str(features.get("crop", "full")) for features, _, _ in windows]


def _regime_order(name):
    return list(CROP_LEVELS).index(name) if name in CROP_LEVELS else len(CROP_LEVELS)


def source_of(group):
    """The dataset a group came from: the namespace prefix of its key ("ntu:P001" -> "ntu")."""
    return str(group).split(":", 1)[0]


def _predict_with_floors(probabilities, classes, floors):
    """Argmax, replaced by `none` when the winner does not clear its own floor."""
    winners = np.argmax(probabilities, axis=1)
    best = probabilities[np.arange(len(probabilities)), winners]
    names = np.array([classes[index] for index in winners], dtype=object)
    thresholds = np.array([floors.get(classes[index], 0.0) for index in winners])
    return np.where(best >= thresholds, names, NONE)


def _prf(predicted, actual, name):
    """Precision, recall, F1 and support of one class."""
    true_positive = int(((predicted == name) & (actual == name)).sum())
    predicted_positive = int((predicted == name).sum())
    actual_positive = int((actual == name).sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual_positive}


def _report(predicted, actual, classes):
    return {name: _prf(predicted, actual, name) for name in classes}


def _confusion(predicted, actual, classes):
    index = {name: position for position, name in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), int)
    for truth, guess in zip(actual, predicted):
        if truth in index and guess in index:
            matrix[index[truth], index[guess]] += 1
    return matrix


def tune_floors(probabilities, y, classes):
    """Per-class confidence floor maximizing that class's F1 on out-of-fold probabilities.

    Each class is swept independently; `none` is the fallback and never gets a floor.
    """
    floors = {}
    for name in classes:
        if name == NONE:
            floors[name] = 0.0
            continue
        best, best_f1 = FLOORS[0], -1.0
        for candidate in FLOORS:
            trial = dict.fromkeys(classes, 0.0)
            trial[name] = candidate
            f1 = _prf(_predict_with_floors(probabilities, classes, trial), y, name)["f1"]
            if f1 > best_f1:
                best, best_f1 = candidate, f1
        floors[name] = best
    return floors


def _fold_probabilities(model, X, y, train, test, classes):
    """Fit on `train`; return `test` probabilities in `classes` order plus the classes never seen.

    Columns are placed by name: `predict_proba` orders them by the classes present in the training half,
    which can be fewer than `classes`. Absent classes get an all-zero column.
    """
    from sklearn.base import clone

    fitted = clone(model).fit(X[train], y[train])
    raw = fitted.predict_proba(X[test])
    probabilities = np.zeros((len(test), len(classes)), np.float64)
    seen = list(fitted.classes_)
    for column, name in enumerate(seen):
        probabilities[:, classes.index(name)] = raw[:, column]
    return probabilities, [name for name in classes if name not in seen]


def train_grouped(windows, n_splits=5, seed=0, crop_fraction=1.0):
    """Fit on everything, and score with out-of-fold predictions grouped by person.

    Args:
        windows: (features, label, group) triples.
        n_splits: GroupKFold folds.
        seed: Forest and crop-sampling seed.
        crop_fraction: Share of the cropped windows a fold may *learn* from (0 = full-body only). The
            evaluation half is never touched, so every setting is scored on the same windows.

    Returns:
        Dict with the fitted `model`, `floors`, `classes`, pooled / per-fold / per-regime / per-source
        reports and the diagnostics.

    Raises:
        ValueError: Fewer groups than folds, or `crop_fraction` outside [0, 1].
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold

    X, y, groups = to_arrays(windows)
    spans = spans_of(windows)
    unique = len(set(groups.tolist()))
    if unique < n_splits:
        raise ValueError(f"need at least {n_splits} groups for {n_splits} folds, got {unique}")
    if not 0.0 <= crop_fraction <= 1.0:
        raise ValueError(f"crop_fraction must be in [0, 1], got {crop_fraction}")

    regimes = np.array(crops_of(windows), dtype=object)
    trainable = regimes == "full"
    cropped_rows = np.flatnonzero(~trainable)
    if crop_fraction > 0 and len(cropped_rows):
        chosen = np.random.default_rng(seed).choice(
            cropped_rows, size=int(len(cropped_rows) * crop_fraction), replace=False)
        trainable[chosen] = True

    classes = [name for name in ACTIONS if name in set(y.tolist())]
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, class_weight="balanced_subsample", random_state=seed, n_jobs=-1
    )

    # Folds are walked by hand: per-fold scores are a deliverable, and `cross_val_predict` raises when a
    # fold's training half lacks a class, which is worth reporting rather than crashing on.
    probabilities = np.zeros((len(y), len(classes)), np.float64)
    fold_index = np.full(len(y), -1, int)
    folds = []
    for number, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        part, missing = _fold_probabilities(
            model, X, y, train[trainable[train]], test, classes)
        probabilities[test] = part
        fold_index[test] = number
        folds.append({
            "fold": number, "missing_in_train": missing,
            "train_groups": sorted(set(groups[train].tolist())),
            "test_groups": sorted(set(groups[test].tolist())),
        })
    assert (fold_index >= 0).all(), "every window must get exactly one out-of-fold prediction"

    floors = tune_floors(probabilities, y, classes)
    predicted = _predict_with_floors(probabilities, classes, floors)

    for fold in folds:
        rows = fold_index == fold["fold"]
        present = [name for name in classes if (y[rows] == name).any()]
        fold["report"] = _report(predicted[rows], y[rows], present)
        fold["n_windows"] = int(rows.sum())
        fold["per_regime"] = {}
        for regime in sorted(set(regimes.tolist()), key=_regime_order):
            part = rows & (regimes == regime)
            fold["per_regime"][regime] = _report(
                predicted[part], y[part],
                [name for name in classes if (y[part] == name).any()])

    per_regime = {}
    for regime in sorted(set(regimes.tolist()), key=_regime_order):
        rows = regimes == regime
        present = [name for name in classes if (y[rows] == name).any()]
        per_regime[regime] = {
            "report": _report(predicted[rows], y[rows], present),
            "confusion": _confusion(predicted[rows], y[rows], classes),
            "n_windows": int(rows.sum()),
        }

    per_source = {}
    sources = np.array([source_of(group) for group in groups], dtype=object)
    for source in sorted(set(sources.tolist())):
        rows = sources == source
        present = [name for name in classes if (y[rows] == name).any()]
        per_source[source] = {
            "report": _report(predicted[rows], y[rows], present),
            "confusion": _confusion(predicted[rows], y[rows], classes),
            "n_windows": int(rows.sum()),
        }

    span_correlation = {}
    for name in classes:
        indicator = (predicted == name).astype(np.float64)
        if spans.std() < 1e-12 or indicator.std() < 1e-12:
            span_correlation[name] = 0.0
        else:
            span_correlation[name] = float(np.corrcoef(spans, indicator)[0, 1])

    model.fit(X[trainable], y[trainable])
    importances = sorted(zip(ACTION_FEATURE_NAMES, model.feature_importances_.tolist()),
                         key=lambda pair: -pair[1])
    return {
        "model": model, "floors": floors, "classes": classes,
        "report": _report(predicted, y, classes),
        "confusion": _confusion(predicted, y, classes),
        "folds": folds, "per_source": per_source, "per_regime": per_regime,
        "importances": importances,
        "span_correlation": span_correlation, "span_mean": float(spans.mean()),
        "n_windows": len(windows), "n_groups": unique, "n_splits": n_splits,
        "crop_fraction": crop_fraction, "n_train_windows": int(trainable.sum()),
        "class_counts": {name: int((y == name).sum()) for name in classes},
        "group_counts": {name: len({group for group, label in zip(groups, y) if label == name})
                         for name in classes},
    }


def evaluate(model, windows, floors):
    """Per-class scores of a fitted model on windows it never saw."""
    X, y, _ = to_arrays(windows)
    classes = list(model.classes_)
    predicted = _predict_with_floors(model.predict_proba(X), classes, floors)
    present = [name for name in ACTIONS if name in set(y.tolist())]
    return _report(predicted, y, present)


def evaluate_confusion(model, windows, floors):
    """(classes, confusion matrix) of an external set, truth by row."""
    X, y, _ = to_arrays(windows)
    classes = list(model.classes_)
    predicted = _predict_with_floors(model.predict_proba(X), classes, floors)
    ordered = [name for name in ACTIONS if name in set(classes)]
    return ordered, _confusion(predicted, y, ordered)


def _table(report):
    lines = [f"  {'action':<14} {'prec':>6} {'recall':>7} {'f1':>6} {'support':>8}"]
    for name, scores in report.items():
        lines.append(f"  {name:<14} {scores['precision']:>6.3f} {scores['recall']:>7.3f} "
                     f"{scores['f1']:>6.3f} {scores['support']:>8d}")
    return "\n".join(lines)


def _matrix(classes, confusion):
    lines = ["  " + " ".join(f"{name[:9]:>10}" for name in [""] + classes)]
    for name, row in zip(classes, confusion):
        lines.append(f"  {name[:9]:>10} " + " ".join(f"{count:>10d}" for count in row))
    return "\n".join(lines)


def format_report(result):
    """Render `train_grouped` output (plus any `external` sets) as text."""
    classes = result["classes"]
    lines = [
        f"{result['n_windows']} windows, {result['n_groups']} groups, {len(classes)} classes, "
        f"window {ACTION_WINDOW_S} s, {len(ACTION_FEATURE_NAMES)} features, "
        f"{result.get('n_splits', '?')} folds",
        f"  crop_fraction {result.get('crop_fraction', 1.0)}: "
        f"{result.get('n_train_windows', result['n_windows'])} of {result['n_windows']} windows are "
        f"trainable; all {result['n_windows']} are scored out-of-fold",
        "  windows per class: " + ", ".join(
            f"{name} {count}" for name, count in result["class_counts"].items()),
        "  groups per class:  " + ", ".join(
            f"{name} {count}" for name, count in result["group_counts"].items()),
        "",
        "Held-out subjects (out-of-fold):",
        _table(result["report"]),
        "",
        "  floors: " + ", ".join(f"{name} {floor:.2f}" for name, floor in result["floors"].items()),
        "",
        "Confusion (rows = truth, columns = prediction):",
        _matrix(classes, result["confusion"]),
        "",
        "Per fold (the spread is the honest number for a thin class):",
    ]
    for fold in result["folds"]:
        missing = (" [absent from this fold's training half: "
                   + ", ".join(fold["missing_in_train"]) + "]") if fold["missing_in_train"] else ""
        lines += [f"  fold {fold['fold']}: {fold['n_windows']} windows, "
                  f"{len(fold['test_groups'])} held-out groups{missing}",
                  _table(fold["report"])]
    f1_by_class = {name: [fold["report"][name]["f1"] for fold in result["folds"]
                          if name in fold["report"]] for name in classes}
    lines += ["", "  f1 per fold:"]
    for name, values in f1_by_class.items():
        spread = f"{min(values):.3f}..{max(values):.3f}" if values else "-"
        lines.append(f"    {name:<14} " + " ".join(f"{value:.3f}" for value in values)
                     + f"   (spread {spread})")

    for regime in result["per_regime"]:
        lines += ["", f"  f1 per fold, {regime} regime:"]
        for name in classes:
            values = [fold["per_regime"][regime][name]["f1"] for fold in result["folds"]
                      if name in fold["per_regime"].get(regime, {})]
            if not values:
                continue
            array = np.array(values)
            lines.append(f"    {name:<14} " + " ".join(f"{value:.3f}" for value in values)
                         + f"   (mean {array.mean():.3f}, sd {array.std(ddof=1):.3f})")

    lines += ["", "By crop regime (the robot sees a waist-up person; a pooled number would hide it):"]
    for regime, block in result["per_regime"].items():
        lines += [f"  {regime} ({block['n_windows']} windows):", _table(block["report"]),
                  "  confusion:", _matrix(classes, block["confusion"])]

    lines += ["", "By source (the source-correlated confound: `none` is the only class drawn from both):"]
    for source, block in result["per_source"].items():
        lines += [f"  {source} ({block['n_windows']} windows):", _table(block["report"]),
                  "  confusion:", _matrix(classes, block["confusion"])]
    recalls = {source: block["report"].get(NONE, {}).get("recall")
               for source, block in result["per_source"].items()}
    measured = {source: value for source, value in recalls.items() if value is not None}
    if len(measured) > 1:
        lines.append("  `none` recall per source: " + ", ".join(
            f"{source} {value:.3f}" for source, value in measured.items())
            + f"   (spread {max(measured.values()) - min(measured.values()):.3f})")

    lines += ["", f"Feature importances (mean span {result['span_mean']:.2f} s):"]
    for name, value in result["importances"]:
        lines.append(f"  {value:.4f}  {name}")
    lines += ["", "corr(span, predicted == class) -- span is not a feature; a strong value here means "
                  "something else carries it:"]
    for name, value in result["span_correlation"].items():
        lines.append(f"  {name:<14} {value:+.3f}")

    for title, report in result.get("external", {}).items():
        lines += ["", f"{title}:", _table(report)]
        matrix = result.get("external_confusion", {}).get(title)
        if matrix is not None:
            lines += ["  confusion:", _matrix(*matrix)]
    return "\n".join(lines)


def recorded_windows(data_dir=None):
    """Our own robot-viewpoint recordings as action windows (waves and pauses only).

    Tagged with the "recorded" crop regime and the "rec:" group namespace; used as an external test set.
    """
    from training.dataset import DATA_DIR, list_sessions, load_session, make_windows

    out = []
    for index, path in enumerate(list_sessions(DATA_DIR if data_dir is None else data_dir)):
        session = load_session(path)
        for features, label, _ in make_windows(
            session, window_s=ACTION_WINDOW_S, features_of=action_features,
            normalizer=normalize_to_torso, extra=trunk_height_parts,
        ):
            out.append((dict(features, crop="recorded"), "wave" if label == 1 else NONE,
                        f"rec:session{index}"))
    return out


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ntu", required=True, help="path to ntu60_hrnet.pkl (wave, clapping)")
    parser.add_argument("--ucf", required=True, help="path to ucf101_hrnet.pkl (pushup, squat, jacks)")
    parser.add_argument("--hmdb", help="path to hmdb51_2d.pkl (external test set, never trained on)")
    parser.add_argument("--data-dir", default=None, help="our recorded sessions (external test set)")
    parser.add_argument("--max-clips-per-class", type=int, default=3,
                        help="cap on clips per negative source class, per dataset")
    parser.add_argument("--splits", type=int, default=5, help="GroupKFold folds")
    parser.add_argument("--crop-fraction", type=float, default=0.25,
                        help="share of the crop augmentation that reaches the training half (0..1)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="where to write the model")
    args = parser.parse_args()

    print(f"Loading NTU windows from {args.ntu}...", flush=True)
    windows = ntu_windows(args.ntu, max_clips_per_class=args.max_clips_per_class)
    print(f"  {len(windows)} windows", flush=True)
    print(f"Loading UCF101 windows from {args.ucf}...", flush=True)
    windows += ucf_windows(args.ucf, max_clips_per_class=args.max_clips_per_class)
    print(f"  {len(windows)} windows total", flush=True)

    recorded = recorded_windows(args.data_dir)
    hmdb = hmdb_windows(args.hmdb, max_clips_per_class=args.max_clips_per_class) if args.hmdb else []
    if args.hmdb:
        print(f"  {len(hmdb)} HMDB51 windows (external test set)", flush=True)

    print(f"\nTraining with crop_fraction={args.crop_fraction}...", flush=True)
    result = train_grouped(windows, n_splits=args.splits, crop_fraction=args.crop_fraction)

    result["external"], result["external_confusion"] = {}, {}
    # HMDB51 is full-body movie footage, so its cropped copies are the only out-of-distribution check on
    # the waist-up regime.
    for regime in CROP_LEVELS:
        part = [triple for triple in hmdb if triple[0].get("crop") == regime]
        if not part:
            continue
        title = f"HMDB51 {regime} (movies, incl. situps as none)"
        result["external"][title] = evaluate(result["model"], part, result["floors"])
        result["external_confusion"][title] = evaluate_confusion(result["model"], part, result["floors"])
    if recorded:
        title = "Our own recordings (waves only, natively waist-up)"
        result["external"][title] = evaluate(result["model"], recorded, result["floors"])
        result["external_confusion"][title] = evaluate_confusion(result["model"], recorded, result["floors"])

    print()
    print(format_report(result), flush=True)

    path = export_forest(
        result["model"], args.out, feature_names=ACTION_FEATURE_NAMES, window_s=ACTION_WINDOW_S,
        thresholds=result["floors"], classes=result["classes"],
    )
    print(f"\nSaved numpy forest to {path}")


if __name__ == "__main__":
    main()
