"""Hand-wave recognition: rules and the exported forest on the same 1.5 s window, plus the live monitor."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.motion.antennas import AntennaWave
from core.motion.features import features_vector, window_features
from core.motion.forest import MODELS_DIR, load_forest
from core.motion.window import WINDOW_S, KeypointWindow

# Recording labels (training/record.py): per-frame class of a session.
WAVE, OTHER, PAUSE = 1, 0, -1
# Trained on NTU RGB+D 60 (40 subjects): f1 0.951 on our recordings.
DEFAULT_MODEL_PATH = MODELS_DIR / "wave_classifier_ntu.npz"
GREEN, GREY, YELLOW = (0, 255, 0), (200, 200, 200), (0, 255, 255)


@dataclass
class Detection:
    """Result of one wave detector.

    Attributes:
        is_wave: Whether the window was classified as a wave.
        score: Confidence in [0, 1].
        details: Detector-specific values shown on the overlay.
    """

    is_wave: bool
    score: float
    details: dict = field(default_factory=dict)


class RuleWaveDetector:
    """Threshold rules on the wave features."""

    def __init__(self, min_above_frac=0.7, min_reversals=3, min_x_amplitude=0.3):
        self.thresholds = {"above_frac": min_above_frac, "reversals": min_reversals, "x_amplitude": min_x_amplitude}

    def detect(self, features):
        """Apply the rules to a `window_features` dict; None without features."""
        if features is None:
            return None
        is_wave = all(features[name] >= threshold for name, threshold in self.thresholds.items())
        score = float(np.mean([min(1.0, features[name] / threshold) for name, threshold in self.thresholds.items()]))
        return Detection(is_wave, score, {name: features[name] for name in self.thresholds})


class ClassifierWaveDetector:
    """The exported numpy forest behind the same interface as the rules."""

    def __init__(self, path=DEFAULT_MODEL_PATH, threshold=None):
        """Load the forest if it exists.

        Args:
            path: The `.npz` export.
            threshold: Decision threshold; None uses the one stored with the model.
        """
        self.path = Path(path)
        self._forest = load_forest(self.path) if self.path.exists() else None
        stored = self._forest.threshold if self._forest is not None else 0.5
        self.threshold = stored if threshold is None else threshold

    @property
    def available(self):
        """Whether a model was loaded."""
        return self._forest is not None

    def detect(self, features):
        """Classify one `window_features` dict; None without features or model."""
        if features is None or not self.available:
            return None
        probability = self._forest.wave_probability(features_vector(features))
        return Detection(probability >= self.threshold, probability, {"p": probability})


class WaveMonitor:
    """Runs the rule and classifier detectors on every frame and triggers the antenna wave.

    Attributes:
        window: The keypoint window; set `window.aspect_ratio` from the real frame.
        reaction: The `AntennaWave` answering detected waves.
        detections: Latest `Detection` per detector name.
        triggers: Rising-edge count per detector name (distinct waves seen).
        reactions: Waves the antennas actually answered.
        trigger_name: Which detector drives the reaction.
    """

    def __init__(self, trigger="auto", model_path=DEFAULT_MODEL_PATH, aspect_ratio=16 / 9, min_score=0.5,
                 eval_interval_s=0.1):
        """Build the detectors.

        Args:
            trigger: "rules", "classifier" or "auto" (classifier when its model exists).
            model_path: Classifier model.
            aspect_ratio: Frame width / height.
            min_score: Keypoint confidence threshold.
            eval_interval_s: Seconds between detector evaluations.

        Raises:
            ValueError: "classifier" requested without a model.
        """
        self.window = KeypointWindow(WINDOW_S, min_score, aspect_ratio)
        self.min_score = min_score
        self.eval_interval_s = eval_interval_s
        self.rules = RuleWaveDetector()
        self.classifier = ClassifierWaveDetector(model_path)
        if trigger == "auto":
            trigger = "classifier" if self.classifier.available else "rules"
        if trigger == "classifier" and not self.classifier.available:
            raise ValueError(f"No wave classifier at {model_path}: train it (python -m training.train_wave) "
                             "or use --wave-trigger rules")
        self.trigger_name = trigger
        self.reaction = AntennaWave()
        self.detections = {"rules": None, "classifier": None}
        self.triggers = {"rules": 0, "classifier": 0}
        self.reactions = 0
        self._was_wave = {"rules": False, "classifier": False}
        self._next_eval = float("-inf")
        self._reacted_at = float("-inf")

    def update(self, now, keypoints):
        """Add a pose and, every `eval_interval_s`, run both detectors. Returns the milliseconds spent."""
        self.window.add(now, keypoints)
        if now < self._next_eval:
            return 0.0
        self._next_eval = now + self.eval_interval_s
        start = time.perf_counter()
        features = window_features(*self.window.frames(), min_score=self.min_score)
        self.detections = {"rules": self.rules.detect(features), "classifier": self.classifier.detect(features)}
        # A repeat answer needs evidence recorded after the last one (the window keeps 1.5 s of history).
        oldest = self.window.frames()[0]
        fresh = len(oldest) == 0 or oldest[0] >= self._reacted_at
        for name, detection in self.detections.items():
            is_wave = detection is not None and detection.is_wave
            started = is_wave and not self._was_wave[name]
            if started:
                self.triggers[name] += 1
            if name == self.trigger_name and is_wave and (started or fresh) and self.reaction.trigger(now):
                self.reactions += 1
                self._reacted_at = now
            self._was_wave[name] = is_wave
        return (time.perf_counter() - start) * 1000

    def antennas(self, now):
        """[right, left] wave offsets in radians (zeros when not waving); add `NEUTRAL_ANTENNAS` yourself."""
        return self.reaction.angles(now) or [0.0, 0.0]

    def panel_lines(self, now):
        """(text, colour) rows for the camera overlay."""
        rules = self.detections["rules"]
        if rules is None:
            lines = [("rules: no data", GREY)]
        else:
            lines = [(f"rules {'WAVE' if rules.is_wave else '-'}  score {rules.score:.2f}  "
                      f"rev {rules.details['reversals']:.0f}", GREEN if rules.is_wave else GREY)]
        classifier = self.detections["classifier"]
        if not self.classifier.available:
            lines.append(("clf: no model (run python -m training.train_wave)", GREY))
        elif classifier is None:
            lines.append(("clf: no data", GREY))
        else:
            lines.append((f"clf {'WAVE' if classifier.is_wave else '-'}  p={classifier.score:.2f}",
                          GREEN if classifier.is_wave else GREY))
        if self.reaction.active(now):
            lines.append((f"WAVE! (trigger: {self.trigger_name})", YELLOW))
        return lines
