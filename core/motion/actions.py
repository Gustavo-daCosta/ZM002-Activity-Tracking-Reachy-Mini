"""Six-class action recognition on a 3 s window, and the live monitor that answers with the antennas.

Trained on COCO-17 (not BlazePose's 33 landmarks) so one model runs on the Mac and on the robot. Unlike
`WaveMonitor`: `panel_lines` is a generator, `antennas(now)` already includes the neutral pose, and every
frame is classified (25 ms mean per `update()` on the CM4).
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from core.motion.antennas import NEUTRAL_ANTENNAS, AntennaWave, antennas_with_neutral
from core.motion.features import ACTION_FEATURE_NAMES, action_features, action_features_vector
from core.motion.forest import MODELS_DIR, NONE, load_forest
from core.motion.window import ACTION_WINDOW_S, KeypointWindow, normalize_to_torso, trunk_height_parts

ACTIONS = (NONE, "wave", "pushup", "squat", "clapping", "jumping_jacks")
DEFAULT_ACTION_MODEL = MODELS_DIR / "action_classifier.npz"
GREEN, GREY, YELLOW, CYAN = (0, 255, 0), (200, 200, 200), (255, 200, 0), (0, 255, 255)

# One antenna signature per action: (amplitude deg, Hz, whole cycles). Whole cycles end at sin = 0, the
# neutral pose, instead of snapping there from wherever the sine stopped.
REACTIONS = {
    "wave": (20.0, 1.5, 2), "clapping": (12.0, 3.0, 3), "pushup": (30.0, 0.8, 1),
    "squat": (25.0, 1.1, 2), "jumping_jacks": (35.0, 2.2, 4),
}


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

    def __init__(self, path=DEFAULT_ACTION_MODEL):
        """Load the model at `path` (a `.npz` export), if it exists."""
        self.path = Path(path)
        self._forest = None
        if self.path.exists():
            self._forest = load_forest(self.path, feature_names=ACTION_FEATURE_NAMES, window_s=ACTION_WINDOW_S)

    @property
    def available(self):
        """Whether a model was loaded."""
        return self._forest is not None

    def detect(self, features):
        """Classify one `action_features` dict with the per-class floors stored in the `.npz`; None without."""
        if features is None or not self.available:
            return None
        return ActionDetection(*self._forest.predict(action_features_vector(features)))


class ActionMonitor:
    """Holds the window, the detector and one `AntennaWave` per action.

    Attributes:
        window: The keypoint window; set `window.aspect_ratio` from the real frame.
        detector: The `ActionDetector`.
        features: Latest feature dict.
        detection: Latest `ActionDetection`.
        counts: Occurrences recognized per action.
        reactions: `AntennaWave` per action.
    """

    def __init__(self, aspect_ratio, min_score=0.5, model_path=DEFAULT_ACTION_MODEL, detector=None):
        """Build the monitor.

        Args:
            aspect_ratio: The camera's real frame width / height. No default: a wrong value shifts every
                angle feature (16/9 on 340x256 clips moved `elbow_angle_amplitude` by 17 degrees).
            min_score: Keypoint confidence threshold.
            model_path: Action model `.npz`, used when `detector` is None.
            detector: An already-built `ActionDetector` (parsing the 5 MB `.npz` twice is slow on the CM4).

        Raises:
            ValueError: No model available.
        """
        self.window = KeypointWindow(ACTION_WINDOW_S, min_score, aspect_ratio,
                                     normalizer=normalize_to_torso, extra=trunk_height_parts)
        self.min_score = min_score
        self.detector = detector if detector is not None else ActionDetector(model_path)
        if not self.detector.available:
            raise ValueError(f"No action model at {self.detector.path}. Train it with "
                             "`python -m training.train_actions --ntu ... --ucf ...`")
        self.features = None
        self.detection = None
        self.counts = {action: 0 for action in ACTIONS}
        self.reactions = {action: AntennaWave(amplitude, hz, cycles / hz)
                          for action, (amplitude, hz, cycles) in REACTIONS.items()}
        self._answered_at = {action: None for action in ACTIONS}

    def note(self, action, now):
        """Record a detected action; returns True when the antennas answered it.

        A repeat is counted only once the window has fully refilled since the last answer (a 3 s window
        keeps matching for seconds after the movement ends). Every occurrence is counted, but the antennas
        answer only when no reaction is already running.
        """
        if action == NONE:
            return False
        oldest = self.window.frames()[0]
        answered = self._answered_at[action]
        if not (answered is None or (len(oldest) > 0 and oldest[0] > answered)):
            return False
        self.counts[action] += 1
        self._answered_at[action] = now
        if self._active_reaction(now) is not None:
            return False
        return self.reactions[action].trigger(now)

    def update(self, now, keypoints):
        """Add a frame and re-detect. Returns the milliseconds the motion stage took."""
        started = time.perf_counter()
        self.window.add(now, keypoints)
        self.features = action_features(*self.window.frames(), min_score=self.min_score,
                                        trunk_heights=self.window.extras())
        self.detection = self.detector.detect(self.features)
        if self.detection is not None:
            self.note(self.detection.action, now)
        return (time.perf_counter() - started) * 1000.0

    def _active_reaction(self, now):
        return next((reaction for reaction in self.reactions.values() if reaction.active(now)), None)

    def antennas(self, now):
        """[right, left] antenna targets: the active reaction added to the neutral pose."""
        reaction = self._active_reaction(now)
        return list(NEUTRAL_ANTENNAS) if reaction is None else antennas_with_neutral(reaction.angles(now))

    def panel_lines(self, now):
        """Yield (text, colour) rows for the camera overlay."""
        if self.detection is None:
            yield ("actions: no window yet", GREY)
            return
        yield (f"{self.detection.action} {self.detection.score:.2f}", GREEN if self.detection.is_action else GREY)
        top = sorted(self.detection.probabilities.items(), key=lambda item: -item[1])[:3]
        yield ("  " + "  ".join(f"{name} {p:.2f}" for name, p in top), GREY)
        counted = " ".join(f"{action}:{count}" for action, count in self.counts.items() if count)
        yield ("  counts " + (counted or "-"), YELLOW)
        active = [action for action, reaction in self.reactions.items() if reaction.active(now)]
        if active:
            yield ("  antennas: " + ", ".join(active), CYAN)
