"""Live action recognition: window -> features -> detector -> one antenna answer per occurrence.

Unlike `WaveMonitor`: `panel_lines` is a generator, `antennas(now)` already includes the neutral pose,
and every frame is classified (measured on the CM4: 25 ms mean per `update()`, 42% of the loop).
"""

import time

from core.motion.actions import ACTION_WINDOW_S, ACTIONS, NONE
from core.motion.actions.detector import DEFAULT_ACTION_MODEL, ActionDetector
from core.motion.actions.features import action_features
from core.motion.actions.normalize import normalize_to_torso, trunk_height_parts
from core.motion.reaction import AntennaWave, NEUTRAL_ANTENNAS, antennas_with_neutral
from core.motion.window import KeypointWindow

GREEN, GREY, YELLOW, CYAN = (0, 255, 0), (200, 200, 200), (255, 200, 0), (0, 255, 255)

# One antenna signature per action. Durations are whole cycles so every reaction ends at sin(2*pi*n) = 0,
# the neutral pose, instead of snapping there from wherever the sine stopped.
REACTION_SIGNATURES = {
    "wave":          {"amplitude_deg": 20.0, "freq_hz": 1.5, "cycles": 2},
    "clapping":      {"amplitude_deg": 12.0, "freq_hz": 3.0, "cycles": 3},
    "pushup":        {"amplitude_deg": 30.0, "freq_hz": 0.8, "cycles": 1},
    "squat":         {"amplitude_deg": 25.0, "freq_hz": 1.1, "cycles": 2},
    "jumping_jacks": {"amplitude_deg": 35.0, "freq_hz": 2.2, "cycles": 4},
}
REACTIONS = {
    action: {"amplitude_deg": signature["amplitude_deg"], "freq_hz": signature["freq_hz"],
             "duration_s": signature["cycles"] / signature["freq_hz"]}
    for action, signature in REACTION_SIGNATURES.items()
}


class ActionMonitor:
    """Holds the window, the detector and one `AntennaWave` per action.

    The window carries the `trunk_height_parts` extra channel, as the training windows do: two of the
    model's top features come from it and normalization destroys the quantity they need.

    Attributes:
        window: The keypoint window; set `window.aspect_ratio` from the real frame.
        detector: The `ActionDetector`.
        features: Latest feature dict.
        detection: Latest `ActionDetection`.
        counts: Occurrences recognized per action.
        reactions: `AntennaWave` per action.
    """

    def __init__(self, aspect_ratio, min_score=0.5, model_path=DEFAULT_ACTION_MODEL,
                 window_s=ACTION_WINDOW_S, detector=None):
        """Build the monitor.

        Args:
            aspect_ratio: The camera's real frame width / height. No default: a wrong value shifts every
                angle feature (16/9 on 340x256 clips moved `elbow_angle_amplitude` by 17 degrees).
            min_score: Keypoint confidence threshold.
            model_path: Action model `.npz`, used when `detector` is None.
            window_s: Window length in seconds.
            detector: An already-built `ActionDetector` (parsing the 5.3 MB `.npz` twice is slow on the CM4).

        Raises:
            ValueError: No model at `model_path`.
        """
        self.window = KeypointWindow(window_s, min_score, aspect_ratio,
                                     normalizer=normalize_to_torso, extra=trunk_height_parts)
        self.min_score = min_score
        self.detector = None
        if detector is not None:
            self.detector = detector
            model_path = detector.path
        elif model_path is not None:
            self.detector = ActionDetector(model_path)
        if self.detector is not None and not self.detector.available:
            raise ValueError(
                f"No action model at {model_path}. Train it with "
                "`python -m training.actions.train --ntu ... --ucf ...`"
            )
        self.features = None
        self.detection = None
        self.counts = {action: 0 for action in ACTIONS}
        self.reactions = {action: AntennaWave(**settings) for action, settings in REACTIONS.items()}
        self._last = NONE
        self._answered_at = {action: None for action in ACTIONS}

    def note(self, action, now):
        """Record a detected action.

        A repeat is counted only once the window has fully refilled since the last answer (a 3 s window
        keeps matching for seconds after the movement ends). Every occurrence is counted, but the antennas
        answer only when no reaction is already running: starting one mid-swing would step the target.

        Args:
            action: Detected class name.
            now: Timestamp in seconds.

        Returns:
            True when the antennas answered this occurrence.
        """
        if action == NONE:
            self._last = NONE
            return False
        oldest = self.window.frames()[0]
        answered = self._answered_at[action]
        fresh = answered is None or (len(oldest) > 0 and oldest[0] > answered)
        self._last = action
        if not fresh:
            return False
        self.counts[action] += 1
        self._answered_at[action] = now
        if self._active_reaction(now) is not None:
            return False
        return self.reactions[action].trigger(now)

    def update(self, now, keypoints):
        """Add a frame and re-detect.

        Args:
            now: Timestamp in seconds.
            keypoints: (17, 3) keypoints or None.

        Returns:
            Milliseconds the motion stage took.
        """
        started = time.perf_counter()
        self.window.add(now, keypoints)
        self.features = action_features(*self.window.frames(), min_score=self.min_score,
                                        trunk_heights=self.window.extras())
        self.detection = self.detector.detect(self.features) if self.detector is not None else None
        if self.detection is not None:
            self.note(self.detection.action, now)
        return (time.perf_counter() - started) * 1000.0

    def _active_reaction(self, now):
        for reaction in self.reactions.values():
            if reaction.active(now):
                return reaction
        return None

    def antennas(self, now):
        """[right, left] antenna targets: the active reaction added to the neutral pose."""
        reaction = self._active_reaction(now)
        return list(NEUTRAL_ANTENNAS) if reaction is None else antennas_with_neutral(reaction.angles(now))

    def panel_lines(self, now):
        """Yield (text, colour) rows for the camera overlay."""
        if self.detection is None:
            yield ("actions: no window yet", GREY)
            return
        yield (f"{self.detection.action} {self.detection.score:.2f}",
               GREEN if self.detection.is_action else GREY)
        top = sorted(self.detection.probabilities.items(), key=lambda item: -item[1])[:3]
        yield ("  " + "  ".join(f"{name} {p:.2f}" for name, p in top), GREY)
        counted = " ".join(f"{action}:{count}" for action, count in self.counts.items() if count)
        yield ("  counts " + (counted or "-"), YELLOW)
        active = [action for action, reaction in self.reactions.items() if reaction.active(now)]
        if active:
            yield ("  antennas: " + ", ".join(active), CYAN)
