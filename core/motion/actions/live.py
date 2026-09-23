"""Live action recognition: window -> features -> detector -> one antenna answer per occurrence.

Three deliberate differences from `core.motion.live.WaveMonitor`, for whoever wires this up:

* `panel_lines` is a **generator**, not a list: do not `len()` it and do not iterate it twice.
* `antennas(now)` returns the **neutral-offset** target, ready to command, while
  `WaveMonitor.antennas` returns raw offsets. Do not add `NEUTRAL_ANTENNAS` to it again.
* there is no evaluation throttle (`WaveMonitor.eval_interval_s`): every frame is classified. At the
  robot's ~10 FPS that is affordable -- `update()` measures a mean of 0.23 ms and a max of 1.4 ms on the
  Mac against a 100 ms frame budget -- but a much faster camera would want one.
"""

import time

from core.motion.actions import ACTION_WINDOW_S, ACTIONS, NONE
from core.motion.actions.detector import DEFAULT_ACTION_MODEL, ActionDetector
from core.motion.actions.features import action_features
from core.motion.actions.normalize import normalize_to_torso, trunk_height_parts
from core.motion.reaction import AntennaWave, NEUTRAL_ANTENNAS, antennas_with_neutral
from core.motion.window import KeypointWindow

GREEN, GREY, YELLOW, CYAN = (0, 255, 0), (200, 200, 200), (255, 200, 0), (0, 255, 255)

# One antenna signature per action, so you can tell from across the room what the robot recognized.
# `pushup` has one because the class exists, not because it can be demonstrated: it scores an F1 of 0.381
# full-body and 0.137 waist-up, so nothing here should be built on the assumption that it ever fires.
#
# Each signature is stated as a **whole number of cycles**, and the duration is derived from it, because
# `AntennaWave.angles` is a raw sine that stops returning angles the instant the duration elapses: the last
# commanded angle is wherever the sine happened to be, and the antenna then snaps to neutral in one control
# step. At a non-integer cycle count that step is real hardware motion -- the earlier table ended a push-up
# reaction at +29.5 degrees, its full amplitude, and a wave at -19.0 degrees. An integer count ends every
# reaction at sin(2*pi*n) = 0, which is already the neutral pose, so the reaction closes smoothly.
# (The wave path's own defaults, 2.5 Hz for 1.2 s, are 3.00 cycles and were right by luck.)
#
# The amplitude/frequency pairs stay pairwise distinct: telling the actions apart across the room is the
# whole point of having five signatures.
REACTION_SIGNATURES = {
    "wave":          {"amplitude_deg": 20.0, "freq_hz": 1.5, "cycles": 2},   # 1.333 s
    "clapping":      {"amplitude_deg": 12.0, "freq_hz": 3.0, "cycles": 3},   # 1.000 s
    "pushup":        {"amplitude_deg": 30.0, "freq_hz": 0.8, "cycles": 1},   # 1.250 s
    "squat":         {"amplitude_deg": 25.0, "freq_hz": 1.1, "cycles": 2},   # 1.818 s
    "jumping_jacks": {"amplitude_deg": 35.0, "freq_hz": 2.2, "cycles": 4},   # 1.818 s
}

# The AntennaWave keyword arguments, with `duration_s` computed from the cycle count so the two can never
# drift apart in an edit.
REACTIONS = {
    action: {"amplitude_deg": signature["amplitude_deg"], "freq_hz": signature["freq_hz"],
             "duration_s": signature["cycles"] / signature["freq_hz"]}
    for action, signature in REACTION_SIGNATURES.items()
}


class ActionMonitor:
    """Holds the window, the detector and one AntennaWave per action.

    The window carries the `trunk_height_parts` extra channel, exactly as the training pipeline's windows
    do. It is not optional: `trunk_height_amplitude` and `trunk_height_reversal_rate` are the two most
    important features of the trained model, they are the only squat signal that survives the robot's
    waist-up framing, and normalization destroys the quantity they are computed from, so it cannot be
    recovered from `frames()`. Without the channel both would quietly read their 0.0 "not measured"
    sentinel and the live feature vector would stop matching the one the model was trained on -- the model
    would appear to work while being wrong.

    `aspect_ratio` (the camera's frame width / height) is required and deliberately has no default.
    `normalize_to_torso` multiplies x by it before measuring anything, and the training windows used each
    clip's own `width / height`, so a wrong value rescales every horizontal quantity and, through the joint
    chains, every angle. Measured against the training path on 220 real dataset windows: with the real
    ratio the two paths agree bit for bit (max |difference| 0.0 over all 34 features), while a live path
    fixed at 16/9 shifts UCF101's 340x256 clips by up to 17.2 degrees of `elbow_angle_amplitude` and 14.2
    of `knee_angle_amplitude`, flipping the predicted class in 24% of jumping-jack and 29% of push-up
    windows. A plausible-looking default is exactly how that ships unnoticed.

    The "fresh evidence" rule is per action: a 3 s window keeps matching for seconds after the movement
    ends, so an occurrence is answered again only once the window has refilled since the last answer.
    """

    def __init__(self, aspect_ratio, min_score=0.5, model_path=DEFAULT_ACTION_MODEL,
                 window_s=ACTION_WINDOW_S, detector=None):
        self.window = KeypointWindow(window_s, min_score, aspect_ratio,
                                     normalizer=normalize_to_torso, extra=trunk_height_parts)
        self.min_score = min_score
        self.detector = None
        # `detector` lets a caller that already built one (the entry points check the model before waking the
        # robot) hand it over: parsing the 5.3 MB .npz twice costs a visible fraction of a second on the CM4.
        if detector is not None:
            self.detector = detector
            model_path = detector.path
        elif model_path is not None:
            self.detector = ActionDetector(model_path)
        if self.detector is not None:
            if not self.detector.available:
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

    def reaction_for(self, action):
        return self.reactions.get(action)

    def note(self, action, now):
        """Record a detected action; return True when the antennas answered this occurrence.

        A repeat of the same action is gated on window freshness alone. An earlier version also counted
        whenever the action differed from the previous frame's, which sounds like "a new occurrence
        started" but is really "the previous frame disagreed": a single frame in which the model reads
        `none` re-arms that gate, and the frame after it counts the *same* movement again. Measured on a
        synthetic 2.5 s squat at 10 FPS, two one-frame dropouts turned one occurrence into three, which is
        exactly what a real webcam session reported (one squat counted 2-3 times). Freshness already
        admits the first occurrence of an action (`answered is None`) and is tracked per action, so two
        different actions in a row are still counted separately. The cost is that two genuine repetitions
        closer together than one window refill count as one -- with a 3 s window, a fast repetition is
        indistinguishable from the tail of the previous one anyway.

        Recognition and reaction are deliberately separate. Every new occurrence is **counted**, but its
        antenna reaction starts only when no reaction is already running: the antennas have one pair of
        joints and `angles()` is an absolute target, so starting a second reaction mid-swing steps the
        target discontinuously -- measured at 22.7 degrees per antenna when a wave interrupted a squat
        0.29 s in, and more with a jumping jack's 35 degrees. So the monitor counts everything it
        recognizes and the antennas answer what they can without interrupting themselves. Losing the count
        instead would trade a visible glitch for an invisible one.
        """
        if action == NONE:
            self._last = NONE
            return False
        oldest = self.window.frames()[0]
        answered = self._answered_at[action]
        # Never answered yet, or the window has fully refilled since the last answer. An empty window
        # cannot prove freshness, so it counts as stale and only `started` can trigger. The comparison is
        # strict: a frame recorded at exactly the answer's timestamp was part of the evidence that was
        # already answered, so it does not make the window fresh. (The wave path uses `>=` there, which is
        # unobservable for it because its reaction refuses to retrigger that soon anyway; here counting is
        # decoupled from the reaction, so the boundary became visible.)
        fresh = answered is None or (len(oldest) > 0 and oldest[0] > answered)
        self._last = action
        if not fresh:
            return False
        # `_answered_at` is the occurrence bookkeeping and must move even when the antennas stay busy:
        # leaving it unset would make this same occurrence look new again on the very next frame.
        self.counts[action] += 1
        self._answered_at[action] = now
        if self._active_reaction(now) is not None:
            return False
        return self.reactions[action].trigger(now)

    def update(self, now, keypoints):
        """Add a frame, re-detect, and return how long the motion stage took in milliseconds."""
        started = time.perf_counter()
        self.window.add(now, keypoints)
        self.features = action_features(*self.window.frames(), min_score=self.min_score,
                                        trunk_heights=self.window.extras())
        self.detection = self.detector.detect(self.features) if self.detector is not None else None
        if self.detection is not None:
            self.note(self.detection.action, now)
        return (time.perf_counter() - started) * 1000.0

    def _active_reaction(self, now):
        """The reaction currently moving the antennas, or None. `note` keeps this at most one."""
        for reaction in self.reactions.values():
            if reaction.active(now):
                return reaction
        return None

    def antennas(self, now):
        """[right, left] antenna targets: the active reaction added to the neutral pose.

        Because `note` never starts a reaction while another is running, at most one is ever active, so
        this is continuous: it leaves neutral only at a reaction's first sample and returns to it at the
        last (every signature is a whole number of cycles). The iteration order of `self.reactions` is
        therefore not a behavioural choice.
        """
        reaction = self._active_reaction(now)
        return list(NEUTRAL_ANTENNAS) if reaction is None else antennas_with_neutral(reaction.angles(now))

    def panel_lines(self, now):
        """(text, colour) rows for the camera overlay."""
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
