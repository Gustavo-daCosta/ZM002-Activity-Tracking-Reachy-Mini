"""Multiclass action detector: the exported numpy forest behind the same interface as the wave ones."""

from dataclasses import dataclass, field
from pathlib import Path

from core.motion.actions import ACTION_WINDOW_S, NONE
from core.motion.actions.features import ACTION_FEATURE_NAMES, action_features_vector
from core.motion.forest import load_forest

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_ACTION_MODEL = MODELS_DIR / "action_classifier.npz"


@dataclass
class ActionDetection:
    """Result of one action detection.

    Attributes:
        action: Predicted class name (`NONE` when nothing clears its floor).
        score: Probability of `action`.
        probabilities: Probability per class.
    """

    action: str
    score: float
    probabilities: dict = field(default_factory=dict)

    @property
    def is_action(self):
        """Whether something other than `NONE` was detected."""
        return self.action != NONE


class ActionDetector:
    """Loads the exported forest, if present, and turns a feature dict into one detection."""

    name = "actions"

    def __init__(self, path=DEFAULT_ACTION_MODEL):
        """Load the model.

        Args:
            path: The `.npz` export.

        Raises:
            ValueError: The file was exported with other features or another window length.
        """
        self.path = Path(path)
        self._forest = None
        if self.path.exists():
            self._forest = load_forest(self.path, feature_names=ACTION_FEATURE_NAMES, window_s=ACTION_WINDOW_S)

    @property
    def available(self):
        """Whether a model was loaded."""
        return self._forest is not None

    def detect(self, features):
        """Classify one feature dict using the per-class floors stored in the `.npz`.

        Args:
            features: Dict from `action_features`, or None.

        Returns:
            An `ActionDetection`, or None without features or model.
        """
        if features is None or not self.available:
            return None
        probabilities = self._forest.probabilities(action_features_vector(features))
        action, score = self._forest.predict_from(probabilities)
        return ActionDetection(action, score, probabilities)
