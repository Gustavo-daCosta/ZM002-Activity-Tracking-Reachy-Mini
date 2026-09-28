"""Wave detectors: hand-written rules and the exported forest, on the same window features."""

from pathlib import Path

import numpy as np

from core.motion import Detection
from core.motion.features import features_vector

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
# Trained on NTU RGB+D 60 (40 subjects, 3 cameras): f1 0.951 on our recordings vs 0.884 for the
# self-trained wave_classifier.npz.
DEFAULT_MODEL_PATH = MODELS_DIR / "wave_classifier_ntu.npz"


class RuleWaveDetector:
    """Threshold rules on the wave features."""

    name = "rules"

    def __init__(self, min_above_frac=0.7, min_reversals=3, min_x_amplitude=0.3):
        self.thresholds = {"above_frac": min_above_frac, "reversals": min_reversals, "x_amplitude": min_x_amplitude}

    def detect(self, features):
        """Apply the rules.

        Args:
            features: Dict from `window_features`, or None.

        Returns:
            A `Detection`, or None without features.
        """
        if features is None:
            return None
        is_wave = all(features[name] >= threshold for name, threshold in self.thresholds.items())
        score = float(np.mean([min(1.0, features[name] / threshold) for name, threshold in self.thresholds.items()]))
        return Detection(is_wave, score, {name: features[name] for name in self.thresholds})


class ClassifierWaveDetector:
    """The exported numpy forest behind the same interface as the rules."""

    name = "classifier"

    def __init__(self, path=DEFAULT_MODEL_PATH, threshold=None):
        """Load the forest if it exists.

        Args:
            path: The `.npz` export (a `.joblib` path is mapped to the `.npz` next to it).
            threshold: Decision threshold; None uses the one stored with the model.
        """
        from core.motion.forest import load_forest

        self.path = Path(path).with_suffix(".npz")
        self._forest = load_forest(self.path) if self.path.exists() else None
        stored = self._forest.threshold if self._forest is not None else 0.5
        self.threshold = stored if threshold is None else threshold

    @property
    def available(self):
        """Whether a model was loaded."""
        return self._forest is not None

    def detect(self, features):
        """Classify one feature dict.

        Args:
            features: Dict from `window_features`, or None.

        Returns:
            A `Detection`, or None without features or model.
        """
        if features is None or not self.available:
            return None
        probability = self._forest.wave_probability(features_vector(features))
        return Detection(probability >= self.threshold, probability, {"p": probability})
