"""Random forest exported to plain numpy arrays: no scikit-learn needed to run it.

Why: on the robot (Raspberry Pi CM4) `RandomForestClassifier.predict_proba` costs ~33 ms for a single sample -
almost all of it scikit-learn's per-call validation and joblib dispatch, against ~3 ms of actual tree work.
Walking the exported trees in numpy costs a fraction of that, and the robot venv does not need scikit-learn
(which also removes the version coupling of the joblib pickle).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.motion.actions import NONE
from core.motion.features import FEATURE_NAMES
from core.motion.window import WINDOW_S

LEAF = -1


@dataclass
class NumpyForest:
    """Decision trees as flat arrays; `class_proba[t][node]` holds one probability per class.

    Binary wave exports (a single class-1 probability per node) are widened to two columns at load time,
    so a model exported before the multiclass support keeps working untouched.
    """

    children_left: list
    children_right: list
    feature: list
    split_threshold: list   # the value each internal node compares its feature against
    class_proba: list       # per tree: (nodes, n_classes)
    feature_names: list
    classes: list = field(default_factory=lambda: [0, 1])
    threshold: float = 0.5  # binary decision threshold tuned at training time
    thresholds: dict = field(default_factory=dict)  # multiclass: per-class confidence floor

    def _leaf_probabilities(self, sample):
        sample = np.asarray(sample, dtype=np.float64).ravel()
        if sample.size != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {sample.size}")
        total = np.zeros(len(self.classes), np.float64)
        for left, right, feature, split, proba in zip(
            self.children_left, self.children_right, self.feature, self.split_threshold, self.class_proba
        ):
            node = 0
            while left[node] != LEAF:
                node = left[node] if sample[feature[node]] <= split[node] else right[node]
            total += proba[node]
        return total / len(self.children_left)

    def probabilities(self, sample) -> dict:
        """Mean per-class probability over the trees (same as sklearn's predict_proba)."""
        return dict(zip(self.classes, self._leaf_probabilities(sample).tolist()))

    def wave_probability(self, sample) -> float:
        """Probability of the wave class, for the binary wave model."""
        return float(self._leaf_probabilities(sample)[self.classes.index(1)])

    def predict_from(self, probabilities: dict):
        """Same as `predict`, but from an already-computed probability dict.

        Walking 300 trees costs ~0.8 ms, so a caller that also wants the probabilities (the live action
        detector does, for its panel) would otherwise pay for the walk twice on every frame.
        """
        name = max(probabilities, key=probabilities.get)
        probability = probabilities[name]
        if probability < self.thresholds.get(name, 0.0):
            return NONE, 0.0
        return name, float(probability)

    def predict(self, sample):
        """(class name, probability): the argmax, unless it fails its own confidence floor."""
        return self.predict_from(self.probabilities(sample))


def export_forest(model, path, feature_names=None, window_s=WINDOW_S, threshold=0.5):
    """Save a fitted RandomForestClassifier as the arrays NumpyForest needs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wave_column = list(model.classes_).index(1)
    arrays = {"n_trees": len(model.estimators_), "window_s": window_s, "threshold": float(threshold),
              "feature_names": np.array(list(feature_names if feature_names is not None else FEATURE_NAMES))}
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        counts = tree.value[:, 0, :]  # (nodes, classes): class proportions per node in recent sklearn
        arrays[f"left_{index}"] = tree.children_left.astype(np.int32)
        arrays[f"right_{index}"] = tree.children_right.astype(np.int32)
        arrays[f"feature_{index}"] = tree.feature.astype(np.int32)
        arrays[f"threshold_{index}"] = tree.threshold.astype(np.float64)
        arrays[f"proba_{index}"] = (counts[:, wave_column] / counts.sum(axis=1)).astype(np.float64)
    np.savez_compressed(path, **arrays)
    return path


def export_multiclass_forest(model, path, feature_names, window_s, thresholds, classes):
    """Save a fitted multiclass RandomForestClassifier as the arrays NumpyForest needs.

    `classes` is the label order to store; the model's own classes_ may be sorted differently, so each
    tree's columns are reordered to match it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(classes)
    order = [list(model.classes_).index(name) for name in classes]
    arrays = {
        "n_trees": len(model.estimators_), "window_s": float(window_s),
        "feature_names": np.array(list(feature_names)),
        "classes": np.array(classes),
        "thresholds": np.array([float(thresholds[name]) for name in classes], np.float64),
    }
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        counts = tree.value[:, 0, :]  # (nodes, classes): class proportions per node in recent sklearn
        proba = counts[:, order] / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        arrays[f"left_{index}"] = tree.children_left.astype(np.int32)
        arrays[f"right_{index}"] = tree.children_right.astype(np.int32)
        arrays[f"feature_{index}"] = tree.feature.astype(np.int32)
        arrays[f"threshold_{index}"] = tree.threshold.astype(np.float64)
        arrays[f"proba_{index}"] = proba.astype(np.float64)
    np.savez_compressed(path, **arrays)
    return path


def load_forest(path, feature_names=FEATURE_NAMES, window_s=WINDOW_S) -> NumpyForest:
    """Load an exported forest, checking it was trained on the features the caller expects."""
    data = np.load(path, allow_pickle=False)
    stored_names = [str(name) for name in data["feature_names"]]
    if tuple(stored_names) != tuple(feature_names) or float(data["window_s"]) != float(window_s):
        raise ValueError(
            f"{path} was exported with different features or window size; retrain it "
            "(`python -m training.train` for the wave model, "
            "`python -m training.actions.train` for the action model)"
        )
    count = int(data["n_trees"])
    probabilities = [np.atleast_2d(data[f"proba_{i}"]) for i in range(count)]
    if "classes" in data:
        classes = [str(name) for name in data["classes"]]
        thresholds = dict(zip(classes, data["thresholds"].tolist()))
    else:
        # A binary wave export: one column of class-1 probability per node. Widen it to [P(0), P(1)].
        classes, thresholds = [0, 1], {}
        probabilities = [np.column_stack([1.0 - p.ravel(), p.ravel()]) for p in probabilities]
    return NumpyForest(
        children_left=[data[f"left_{i}"] for i in range(count)],
        children_right=[data[f"right_{i}"] for i in range(count)],
        feature=[data[f"feature_{i}"] for i in range(count)],
        split_threshold=[data[f"threshold_{i}"] for i in range(count)],
        class_proba=probabilities,
        feature_names=stored_names,
        classes=classes,
        threshold=float(data["threshold"]) if "threshold" in data else 0.5,
        thresholds=thresholds,
    )
