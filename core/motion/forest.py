"""Random forest exported to plain numpy arrays, so the robot runs it without scikit-learn.

On the Raspberry Pi CM4 `predict_proba` costs ~33 ms per sample (mostly validation overhead); walking
the exported trees costs a fraction of that and removes the pickle's version coupling.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.motion.features import FEATURE_NAMES
from core.motion.window import WINDOW_S

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
NONE = "none"
LEAF = -1


@dataclass
class NumpyForest:
    """Decision trees as flat arrays, one entry per tree in every list.

    Attributes:
        children_left: Left child index per node (LEAF for leaves).
        children_right: Right child index per node.
        feature: Feature index compared at each node.
        split_threshold: Value each node compares its feature against.
        class_proba: (nodes, n_classes) class probabilities.
        feature_names: Names the sample vector must follow.
        classes: Class labels, in column order.
        threshold: Binary decision threshold (wave model).
        thresholds: Per-class confidence floor (multiclass model).
    """

    children_left: list
    children_right: list
    feature: list
    split_threshold: list
    class_proba: list
    feature_names: list
    classes: list = field(default_factory=lambda: [0, 1])
    threshold: float = 0.5
    thresholds: dict = field(default_factory=dict)

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
        """Mean per-class probability over the trees (same as sklearn's `predict_proba`)."""
        return dict(zip(self.classes, self._leaf_probabilities(sample).tolist()))

    def wave_probability(self, sample) -> float:
        """Probability of the wave class, for the binary wave model."""
        return float(self._leaf_probabilities(sample)[self.classes.index(1)])

    def predict(self, sample):
        """(class name, probability): the argmax, or (NONE, 0.0) when it fails its class floor."""
        probabilities = self.probabilities(sample)
        name = max(probabilities, key=probabilities.get)
        if probabilities[name] < self.thresholds.get(name, 0.0):
            return NONE, 0.0, probabilities
        return name, float(probabilities[name]), probabilities


def export_forest(model, path, feature_names=FEATURE_NAMES, window_s=WINDOW_S, threshold=0.5,
                  classes=None, thresholds=None):
    """Save a fitted RandomForestClassifier as the arrays `NumpyForest` needs.

    Args:
        model: Fitted `RandomForestClassifier`.
        path: Output `.npz`.
        feature_names: Feature order the model was trained on.
        window_s: Window length the features were computed over.
        threshold: Binary decision threshold (ignored when `classes` is given).
        classes: Multiclass label order to store; None exports the binary wave format.
        thresholds: Per-class confidence floors, required with `classes`.

    Returns:
        The output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"n_trees": len(model.estimators_), "window_s": float(window_s),
              "feature_names": np.array(list(feature_names))}
    if classes is None:
        columns = [list(model.classes_).index(1)]
        arrays["threshold"] = float(threshold)
    else:
        classes = list(classes)
        columns = [list(model.classes_).index(name) for name in classes]
        arrays["classes"] = np.array(classes)
        arrays["thresholds"] = np.array([float(thresholds[name]) for name in classes], np.float64)
    for index, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        counts = tree.value[:, 0, :]
        proba = counts[:, columns] / np.maximum(counts.sum(axis=1, keepdims=True), 1e-12)
        arrays[f"left_{index}"] = tree.children_left.astype(np.int32)
        arrays[f"right_{index}"] = tree.children_right.astype(np.int32)
        arrays[f"feature_{index}"] = tree.feature.astype(np.int32)
        arrays[f"threshold_{index}"] = tree.threshold.astype(np.float64)
        arrays[f"proba_{index}"] = (proba[:, 0] if classes is None else proba).astype(np.float64)
    np.savez_compressed(path, **arrays)
    return path


def load_forest(path, feature_names=FEATURE_NAMES, window_s=WINDOW_S) -> NumpyForest:
    """Load an exported forest.

    Raises:
        ValueError: The file was exported with other features or another window length.
    """
    data = np.load(path, allow_pickle=False)
    stored_names = [str(name) for name in data["feature_names"]]
    if tuple(stored_names) != tuple(feature_names) or float(data["window_s"]) != float(window_s):
        raise ValueError(
            f"{path} was exported with different features or window size; retrain it "
            "(`python -m training.train_wave` or `python -m training.train_actions`)"
        )
    count = int(data["n_trees"])
    probabilities = [np.atleast_2d(data[f"proba_{i}"]) for i in range(count)]
    if "classes" in data:
        classes = [str(name) for name in data["classes"]]
        thresholds = dict(zip(classes, data["thresholds"].tolist()))
    else:
        # Binary export: one column of class-1 probability, widened to [P(0), P(1)].
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
