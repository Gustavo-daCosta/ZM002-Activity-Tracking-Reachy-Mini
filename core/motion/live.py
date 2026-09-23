"""Live wave recognition: both detectors on a sliding window, antenna reaction, panel text."""

import time

from core.motion.detectors import DEFAULT_MODEL_PATH, ClassifierWaveDetector, RuleWaveDetector
from core.motion.features import window_features
from core.motion.reaction import AntennaWave
from core.motion.window import WINDOW_S, KeypointWindow

GREEN, GREY, YELLOW = (0, 255, 0), (200, 200, 200), (0, 255, 255)


class WaveMonitor:
    def __init__(self, trigger="auto", model_path=DEFAULT_MODEL_PATH, aspect_ratio=16 / 9, min_score=0.5,
                 eval_interval_s=0.1):
        self.window = KeypointWindow(WINDOW_S, min_score, aspect_ratio)
        self.min_score = min_score
        self.eval_interval_s = eval_interval_s
        self.rules = RuleWaveDetector()
        self.classifier = ClassifierWaveDetector(model_path)
        if trigger == "auto":
            trigger = "classifier" if self.classifier.available else "rules"
        if trigger == "classifier" and not self.classifier.available:
            raise ValueError(
                f"No wave classifier at {model_path}: record and train it "
                "(python -m training.record, python -m training.train) or use --wave-trigger rules"
            )
        self.trigger_name = trigger
        self.reaction = AntennaWave()
        self.detections = {"rules": None, "classifier": None}
        self.triggers = {"rules": 0, "classifier": 0}
        self.reactions = 0  # waves the antennas actually answered (the rest fell inside the cooldown)
        self._was_wave = {"rules": False, "classifier": False}
        self._next_eval = float("-inf")
        self._reacted_at = float("-inf")

    def update(self, now, keypoints):
        """Add a pose; every eval_interval_s run both detectors. Returns the milliseconds spent."""
        self.window.add(now, keypoints)
        if now < self._next_eval:
            return 0.0
        self._next_eval = now + self.eval_interval_s
        start = time.perf_counter()
        features = window_features(*self.window.frames(), min_score=self.min_score)
        self.detections = {"rules": self.rules.detect(features), "classifier": self.classifier.detect(features)}
        # The window keeps 1.5 s of history, so right after a wave ends it still looks like a wave. A repeat
        # answer therefore needs *fresh* evidence: every frame in the window recorded after the last answer.
        oldest = self.window.frames()[0]
        fresh = len(oldest) == 0 or oldest[0] >= self._reacted_at
        for name, detection in self.detections.items():
            is_wave = detection is not None and detection.is_wave
            started = is_wave and not self._was_wave[name]
            if started:
                self.triggers[name] += 1  # rising edges: how many distinct waves were seen
            # Answer a wave that just started, or one that is still going on with evidence we have not
            # answered yet - so waving without stopping keeps the antennas going.
            if name == self.trigger_name and is_wave and (started or fresh) and self.reaction.trigger(now):
                self.reactions += 1
                self._reacted_at = now
            self._was_wave[name] = is_wave
        return (time.perf_counter() - start) * 1000

    def antennas(self, now):
        return self.reaction.angles(now) or [0.0, 0.0]

    def panel_lines(self, now):
        rules = self.detections["rules"]
        if rules is None:
            lines = [("rules: no data", GREY)]
        else:
            lines = [(f"rules {'WAVE' if rules.is_wave else '-'}  score {rules.score:.2f}  "
                      f"rev {rules.details['reversals']:.0f}", GREEN if rules.is_wave else GREY)]
        classifier = self.detections["classifier"]
        if not self.classifier.available:
            lines.append(("clf: no model (run python -m training.train)", GREY))
        elif classifier is None:
            lines.append(("clf: no data", GREY))
        else:
            lines.append((f"clf {'WAVE' if classifier.is_wave else '-'}  p={classifier.score:.2f}",
                          GREEN if classifier.is_wave else GREY))
        if self.reaction.active(now):
            lines.append((f"WAVE! (trigger: {self.trigger_name})", YELLOW))
        return lines
