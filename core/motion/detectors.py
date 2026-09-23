"""Wave detectors: hand-written rules and a trained scikit-learn classifier, on the same window features."""

from pathlib import Path

import numpy as np

from core.motion import Detection
from core.motion.features import FEATURE_NAMES, features_vector
from core.motion.window import WINDOW_S

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
# Trained on NTU RGB+D 60 (40 subjects, 3 cameras) rather than on our two recordings: measured on those same
# recordings it scores f1 0.951 (precision 0.976) against 0.884 for the self-trained one, which stays in
# wave_classifier.joblib and can be selected with --classifier.
DEFAULT_MODEL_PATH = MODELS_DIR / "wave_classifier_ntu.joblib"
SELF_TRAINED_MODEL_PATH = MODELS_DIR / "wave_classifier.joblib"
# Same forest as plain numpy arrays: what the robot uses (no scikit-learn, ~30 ms less per evaluation).
DEFAULT_FOREST_PATH = DEFAULT_MODEL_PATH.with_suffix(".npz")


class RuleWaveDetector:
    name = "rules"

    def __init__(self, min_above_frac=0.7, min_reversals=3, min_x_amplitude=0.3):
        self.thresholds = {"above_frac": min_above_frac, "reversals": min_reversals, "x_amplitude": min_x_amplitude}

    def detect(self, features):
        if features is None:
            return None
        is_wave = all(features[name] >= threshold for name, threshold in self.thresholds.items())
        score = float(np.mean([min(1.0, features[name] / threshold) for name, threshold in self.thresholds.items()]))
        return Detection(is_wave, score, {name: features[name] for name in self.thresholds})


def save_classifier(model, path, threshold=0.5, feature_names=FEATURE_NAMES, window_s=WINDOW_S):
    """Persist a fitted sklearn model with the features and window length it was trained on.

    `feature_names` and `window_s` default to the wave model's, which is every existing caller. The action
    trainer passes `ACTION_FEATURE_NAMES` and `ACTION_WINDOW_S`: the bundle is only a Mac-side convenience
    (the robot has no scikit-learn and runs the `.npz` export instead), but it must still record which
    feature contract it holds, or `load_classifier` would happily hand a wave detector an action model.
    """
    import joblib
    import sklearn

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": list(feature_names), "window_s": window_s,
         "threshold": float(threshold), "sklearn_version": sklearn.__version__},
        path,
    )


def load_classifier(path):
    import joblib

    bundle = joblib.load(path)
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES or bundle.get("window_s") != WINDOW_S:
        raise ValueError(
            f"{path} was trained with different features or window size; "
            "retrain with `python -m training.train`"
        )
    return bundle["model"]


class ClassifierWaveDetector:
    name = "classifier"

    def __init__(self, path=DEFAULT_MODEL_PATH, threshold=None, forest_path=None):
        from core.motion.forest import load_forest

        self.path = Path(path)
        # The numpy export lives next to the pickle (same name, .npz), so both follow `path`.
        forest_path = self.path.with_suffix(".npz") if forest_path is None else Path(forest_path)
        # Prefer the numpy export (works without scikit-learn); fall back to the joblib pickle.
        self._forest = load_forest(forest_path) if Path(forest_path).exists() else None
        self._model = load_classifier(self.path) if self._forest is None and self.path.exists() else None
        # The threshold tuned at training time travels with the model; an explicit one always wins.
        stored = self._forest.threshold if self._forest is not None else 0.5
        self.threshold = stored if threshold is None else threshold

    @property
    def available(self):
        return self._forest is not None or self._model is not None

    def detect(self, features):
        if features is None or not self.available:
            return None
        vector = features_vector(features)
        if self._forest is not None:
            probability = self._forest.wave_probability(vector)
        else:
            probabilities = self._model.predict_proba(vector[None])[0]
            probability = float(probabilities[list(self._model.classes_).index(1)])
        return Detection(probability >= self.threshold, probability, {"p": probability})
