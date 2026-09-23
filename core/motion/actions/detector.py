"""Multiclass action detector: the exported numpy forest behind the same interface as the wave ones.

Only numpy is needed to run it, which is the point: the robot's venv has no scikit-learn, and the `.npz`
carries the per-class confidence floors with it, so the thresholding is not reimplemented here.
"""

from dataclasses import dataclass, field
from pathlib import Path

from core.motion.actions import ACTION_WINDOW_S, NONE
from core.motion.actions.features import ACTION_FEATURE_NAMES, action_features_vector
from core.motion.forest import load_forest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_ACTION_MODEL = MODELS_DIR / "action_classifier.npz"


@dataclass
class ActionDetection:
    action: str
    score: float
    probabilities: dict = field(default_factory=dict)

    @property
    def is_action(self):
        return self.action != NONE


class ActionDetector:
    """Loads the exported forest, if it is there, and turns a feature dict into one detection."""

    name = "actions"

    def __init__(self, path=DEFAULT_ACTION_MODEL):
        self.path = Path(path)
        self._forest = None
        if self.path.exists():
            # `load_forest` refuses a model exported with different features or a different window, so a
            # stale `.npz` is an error here rather than a silently misaligned feature vector at runtime.
            self._forest = load_forest(
                self.path, feature_names=ACTION_FEATURE_NAMES, window_s=ACTION_WINDOW_S
            )

    @property
    def available(self):
        return self._forest is not None

    def detect(self, features):
        """`ActionDetection`, or None when the window produced no features or there is no model.

        `predict` applies the per-class confidence floors stored in the `.npz` (none 0.00, wave 0.45,
        clapping 0.40, the rest 0.05) and answers `none` when the argmax fails its own floor.
        """
        if features is None or not self.available:
            return None
        vector = action_features_vector(features)
        # One tree walk per frame: `predict_from` reuses these probabilities instead of walking the 300
        # trees a second time (measured 1.85 ms -> 0.89 ms per detect() on the Mac, 300 trees).
        probabilities = self._forest.probabilities(vector)
        action, score = self._forest.predict_from(probabilities)
        return ActionDetection(action, score, probabilities)
