"""The live path: window -> features -> forest -> one antenna answer per action occurrence.

The synthetic bodies come from `test_action_features`, so the live path is exercised against the same
anthropometrically real skeletons (elbows placed by two-link IK) the feature module is tested with.
"""

import math

import numpy as np
import pytest
from test_action_features import FPS, FRAMES, squatting, upright

from core.motion.actions import ACTIONS, ACTION_WINDOW_S, NONE
from core.motion.actions.features import ACTION_FEATURE_NAMES


class FakeForest:
    """Stands in for the exported forest: returns whatever the test queues up."""

    feature_names = list(ACTION_FEATURE_NAMES)
    classes = list(ACTIONS)

    def __init__(self):
        self.next = (NONE, 0.0)
        self.probs = dict.fromkeys(ACTIONS, 0.0)
        self.samples = []

    def predict(self, sample):
        return self.predict_from(self.probabilities(sample))

    def predict_from(self, probabilities):
        return self.next

    def probabilities(self, sample):
        # The detector walks the trees exactly once per detect(): this is where the sample arrives now.
        self.samples.append(np.asarray(sample))
        return dict(self.probs)


def fake_model(tmp_path):
    """A real file, so ActionDetector takes the loading branch and hits the patched load_forest."""
    path = tmp_path / "action_classifier.npz"
    path.write_bytes(b"")
    return path


def feed(monitor, pose_at, frames=FRAMES, start=0.0):
    """Drive a monitor with a synthetic pose function, one frame at a time, like a camera loop."""
    for index in range(frames):
        t = start + index / FPS
        keypoints = np.zeros((17, 3), np.float32)
        for joint, (x, y) in pose_at(t).items():
            keypoints[joint] = (x, y, 0.9)   # only the joints the pose provides are confident
        monitor.update(t, keypoints)
    return monitor


# --- the detector -----------------------------------------------------------------------------------


def test_detector_reports_the_predicted_action(monkeypatch, tmp_path):
    from core.motion.actions import detector as module

    forest = FakeForest()
    monkeypatch.setattr(module, "load_forest", lambda *a, **k: forest)
    detector = module.ActionDetector(path=fake_model(tmp_path))
    assert detector.available

    forest.next = ("squat", 0.81)
    forest.probs = dict(dict.fromkeys(ACTIONS, 0.0), squat=0.81)
    detection = detector.detect(dict.fromkeys(ACTION_FEATURE_NAMES, 0.0))
    assert detection.action == "squat"
    assert detection.score == pytest.approx(0.81)
    assert detection.is_action
    assert detection.probabilities["squat"] == pytest.approx(0.81)

    forest.next = (NONE, 0.0)
    quiet = detector.detect(dict.fromkeys(ACTION_FEATURE_NAMES, 0.0))
    assert quiet.action == NONE
    assert not quiet.is_action


def test_detector_loads_with_the_action_contract(monkeypatch, tmp_path):
    """The floors live inside the .npz; the detector must not reimplement thresholding."""
    from core.motion.actions import detector as module

    seen = {}

    def spy(path, feature_names=None, window_s=None):
        seen.update(path=path, feature_names=feature_names, window_s=window_s)
        return FakeForest()

    monkeypatch.setattr(module, "load_forest", spy)
    module.ActionDetector(path=fake_model(tmp_path))
    assert tuple(seen["feature_names"]) == tuple(ACTION_FEATURE_NAMES)
    assert seen["window_s"] == ACTION_WINDOW_S


def test_detector_passes_the_features_in_name_order(monkeypatch, tmp_path):
    from core.motion.actions import detector as module

    forest = FakeForest()
    monkeypatch.setattr(module, "load_forest", lambda *a, **k: forest)
    detector = module.ActionDetector(path=fake_model(tmp_path))
    features = {name: float(index) for index, name in enumerate(ACTION_FEATURE_NAMES)}
    detector.detect(features)
    assert forest.samples[0].tolist() == [float(i) for i in range(len(ACTION_FEATURE_NAMES))]


def test_detector_returns_none_without_features(monkeypatch, tmp_path):
    from core.motion.actions import detector as module

    monkeypatch.setattr(module, "load_forest", lambda *a, **k: FakeForest())
    detector = module.ActionDetector(path=fake_model(tmp_path))
    assert detector.detect(None) is None


def test_detector_without_a_model_is_unavailable(tmp_path):
    from core.motion.actions.detector import ActionDetector

    detector = ActionDetector(path=tmp_path / "nothing.npz")
    assert not detector.available
    assert detector.detect(dict.fromkeys(ACTION_FEATURE_NAMES, 0.0)) is None


def test_the_shipped_model_is_loadable():
    from core.motion.actions.detector import DEFAULT_ACTION_MODEL, ActionDetector

    assert DEFAULT_ACTION_MODEL.exists()
    assert ActionDetector().available


# --- the trunk-height channel -----------------------------------------------------------------------


def test_the_monitor_measures_trunk_height():
    """The top two features of the trained model come from an extra per-frame channel.

    Without `extra=trunk_height_parts` on the window they silently fall back to their "not measured"
    sentinel and the live vector stops matching the trained one -- a model that looks like it works.
    """
    from core.motion.actions.live import ActionMonitor

    monitor = feed(ActionMonitor(1.0, model_path=None), squatting)
    assert monitor.features is not None
    assert monitor.features["trunk_height_amplitude"] > 0.05
    assert monitor.features["trunk_height_reversal_rate"] > 0.0

    still = feed(ActionMonitor(1.0, model_path=None), upright)
    assert still.features["trunk_height_amplitude"] < 0.01      # a quiet body really is quiet


def test_the_monitor_survives_frames_with_no_body():
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    for index in range(FRAMES):
        monitor.update(index / FPS, np.zeros((17, 3), np.float32))   # nothing confident
    assert monitor.features is None
    assert monitor.detection is None
    assert list(monitor.panel_lines(1.0))


# --- one answer per occurrence ----------------------------------------------------------------------


def test_one_occurrence_is_answered_once():
    """A 3 s window keeps matching after the movement stops; that must not count as a second one."""
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    assert monitor.note("clapping", now=0.0) is True     # the occurrence starts
    assert monitor.note("clapping", now=0.3) is False    # same occurrence, stale evidence
    assert monitor.note("clapping", now=0.6) is False
    assert monitor.counts["clapping"] == 1


def test_a_second_occurrence_with_fresh_evidence_is_answered():
    """The movement stops and starts again, and the window has refilled in between."""
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    monitor.window.add(0.0, np.zeros((17, 3), np.float32))
    assert monitor.note("wave", now=0.0) is True
    assert monitor.note(NONE, now=2.0) is False          # the movement stopped
    monitor.window.add(4.0, np.zeros((17, 3), np.float32))   # drops every pre-answer frame
    assert monitor.note("wave", now=4.0) is True         # a new one
    assert monitor.counts["wave"] == 2


def test_a_dropout_mid_occurrence_does_not_count_a_second_time():
    """A single frame the model reads as something else must not re-count the movement in progress.

    This is what a real webcam session hit: one squat counted two or three times. Recognition is not
    stable frame to frame, so "the previous frame said something else" cannot mean "a new occurrence".
    """
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    frame = np.zeros((17, 3), np.float32)
    for index in range(25):                              # 2.5 s of squat at 10 FPS
        now = index / 10.0
        monitor.window.add(now, frame)
        monitor.note(NONE if index in (8, 17) else "squat", now)
    assert monitor.counts["squat"] == 1


def test_a_continuing_action_is_answered_again_once_the_window_refilled():
    """Waving without pausing must keep the antennas going, but only on evidence not already answered."""
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    monitor.window.add(0.0, np.zeros((17, 3), np.float32))
    assert monitor.note("wave", now=0.0) is True
    monitor.window.add(1.0, np.zeros((17, 3), np.float32))
    assert monitor.note("wave", now=1.0) is False        # the window still holds pre-answer frames
    monitor.window.add(3.5, np.zeros((17, 3), np.float32))   # drops everything older than 0.5 s
    assert monitor.note("wave", now=3.5) is True
    assert monitor.counts["wave"] == 2


def test_each_action_is_counted_separately():
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    assert monitor.note("wave", now=0.0) is True
    assert monitor.note("squat", now=5.0) is True        # a different action, answered independently
    assert monitor.counts["wave"] == 1
    assert monitor.counts["squat"] == 1
    assert monitor.counts["pushup"] == 0


def test_an_occurrence_during_an_active_reaction_is_counted_but_not_answered():
    """Recognition and reaction come apart: the antennas cannot serve two actions at once.

    Starting a second reaction mid-swing steps the antenna target discontinuously (22.7 degrees per
    antenna when a wave interrupted a squat 0.29 s in), so the occurrence is counted and left unanswered.
    """
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    assert monitor.note("squat", now=0.0) is True
    assert monitor.note("wave", now=0.30) is False       # the squat reaction is still running
    assert monitor.counts["wave"] == 1                   # but it was recognized, so it is counted
    assert monitor.counts["squat"] == 1
    assert monitor.reaction_for("wave").angles(0.30) is None   # and its antennas never started


def test_the_antenna_target_stays_continuous_when_a_second_action_arrives():
    """The measured glitch: this asserts the step across that moment is small, not merely that it exists."""
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    monitor.note("squat", now=0.0)
    before = monitor.antennas(0.29999)
    monitor.note("wave", now=0.30)
    after = monitor.antennas(0.30001)
    step = max(abs(a - b) for a, b in zip(before, after))
    assert step < math.radians(1.0)      # was 22.7 degrees per antenna with first-wins arbitration


def test_none_is_never_answered():
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    assert monitor.note(NONE, now=0.0) is False
    assert monitor.counts[NONE] == 0
    assert monitor.reaction_for(NONE) is None


# --- the antennas -----------------------------------------------------------------------------------


def test_every_action_has_its_own_antenna_reaction():
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    for action in ACTIONS:
        if action == NONE:
            assert monitor.reaction_for(action) is None
        else:
            reaction = monitor.reaction_for(action)
            assert reaction is not None
            assert reaction.amplitude_deg > 0
    # Distinct actions must not look the same on the antennas.
    signatures = {(monitor.reaction_for(a).amplitude_deg, monitor.reaction_for(a).freq_hz)
                  for a in ACTIONS if a != NONE}
    assert len(signatures) == len(ACTIONS) - 1


def test_the_antennas_return_to_neutral_between_reactions():
    from core.motion.reaction import NEUTRAL_ANTENNAS
    from core.motion.actions.live import ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    assert monitor.antennas(0.0) == pytest.approx(list(NEUTRAL_ANTENNAS))
    monitor.note("jumping_jacks", now=0.0)
    moved = monitor.antennas(0.13)
    assert moved != pytest.approx(list(NEUTRAL_ANTENNAS))
    reaction = monitor.reaction_for("jumping_jacks")
    assert monitor.antennas(reaction.duration_s + 1.0) == pytest.approx(list(NEUTRAL_ANTENNAS))


def test_every_reaction_ends_where_it_started():
    """`angles()` stops returning angles the instant the duration elapses, so the LAST angle is commanded.

    A non-integer cycle count leaves the antenna mid-swing and the next control step snaps it to neutral:
    measured +29.5 degrees (full amplitude) at the end of every push-up reaction, -19.0 for a wave. Sampling
    only after the reaction ends cannot see it, so this samples just before.
    """
    from core.motion.actions.live import REACTION_SIGNATURES, ActionMonitor

    monitor = ActionMonitor(1.0, model_path=None)
    for action, signature in REACTION_SIGNATURES.items():
        reaction = monitor.reaction_for(action)
        assert signature["cycles"] == int(signature["cycles"])
        assert reaction.duration_s == pytest.approx(signature["cycles"] / signature["freq_hz"])
        assert reaction.trigger(0.0)
        last = reaction.angles(reaction.duration_s - 1e-6)
        assert last is not None
        assert max(abs(angle) for angle in last) < math.radians(0.01), action

    # And the signatures are still pairwise distinct, which is why there are five of them.
    assert len({(s["amplitude_deg"], s["freq_hz"]) for s in REACTION_SIGNATURES.values()}) == \
        len(REACTION_SIGNATURES)


def test_panel_lines_describe_the_state(monkeypatch, tmp_path):
    from core.motion.actions import detector as module
    from core.motion.actions.live import ActionMonitor

    forest = FakeForest()
    monkeypatch.setattr(module, "load_forest", lambda *a, **k: forest)
    forest.next = ("squat", 0.77)
    forest.probs = dict(dict.fromkeys(ACTIONS, 0.0), squat=0.77, none=0.2)
    monitor = ActionMonitor(1.0, model_path=fake_model(tmp_path))

    assert "no window yet" in list(monitor.panel_lines(0.0))[0][0]
    feed(monitor, squatting)
    text = " ".join(line for line, _colour in monitor.panel_lines(FRAMES / FPS))
    assert "squat" in text and "0.77" in text
    assert "squat:1" in text


def test_update_reports_its_cost_and_the_detection(monkeypatch, tmp_path):
    from core.motion.actions import detector as module
    from core.motion.actions.live import ActionMonitor

    forest = FakeForest()
    monkeypatch.setattr(module, "load_forest", lambda *a, **k: forest)
    forest.next = ("clapping", 0.6)
    monitor = ActionMonitor(1.0, model_path=fake_model(tmp_path))
    milliseconds = [monitor.update(index / FPS, _confident_upright(index / FPS))
                    for index in range(FRAMES)]
    # A real bound, not `>= 0.0`, which is unconditionally true of a perf_counter delta. Measured mean
    # 0.226 ms, p95 1.015 ms, max 1.409 ms on the Mac; an estimated 4-10 ms on the Pi against a 100 ms
    # frame budget. 20 ms catches a pathological regression without being flaky on a loaded machine.
    assert max(milliseconds) < 20.0
    assert monitor.detection.action == "clapping"
    assert monitor.counts["clapping"] == 1


def test_a_missing_model_is_a_clear_error(tmp_path):
    from core.motion.actions.live import ActionMonitor

    with pytest.raises(ValueError, match="No action model"):
        ActionMonitor(1.0, model_path=tmp_path / "nothing.npz")


def _confident_upright(t):
    keypoints = np.zeros((17, 3), np.float32)
    for joint, (x, y) in upright(t).items():
        keypoints[joint] = (x, y, 0.9)
    return keypoints


# --- the live path must produce the training path's vector -------------------------------------------


def _annotation(pose_at, width=340, height=256, frames=FRAMES * 3, fps=30.0):
    """One clip in the published MMAction2/PySkl layout, in pixels, for `clip_windows`.

    Deliberately 340x256 (an aspect ratio of 1.328, UCF101's own) rather than 16/9, because that is the
    shape whose ratio a live path with a fixed default gets wrong.
    """
    keypoints = np.zeros((1, frames, 17, 3), np.float32)
    scores = np.zeros((1, frames, 17), np.float32)
    for index in range(frames):
        for joint, (x, y) in pose_at(index / fps).items():
            keypoints[0, index, joint] = (x * width, y * height, 0.0)
            scores[0, index, joint] = 0.9
    return {"img_shape": (height, width), "keypoint": keypoints[..., :2].copy(),
            "keypoint_score": scores, "frame_dir": "v_Synthetic_g01_c01", "label": 0}


def _training_windows(annotation):
    from core.motion.actions.features import MIN_SPAN_S
    from core.motion.actions.normalize import normalize_to_torso, trunk_height_parts
    from core.motion.actions.features import action_features
    from training.public_data import clip_windows

    return clip_windows(annotation, window_s=ACTION_WINDOW_S, features_of=action_features,
                        normalizer=normalize_to_torso, extra=trunk_height_parts, min_span_s=MIN_SPAN_S)


def _live_windows(annotation, aspect_ratio):
    """The same frames through ActionMonitor, emitted on clip_windows' own gate.

    The stride, the `min_span_s` gate and the `STEP_S` cadence below are a deliberate copy of
    `public_data.clip_windows`: the live path has no emission gate of its own (it classifies every frame),
    so there is nothing to unify. **Keep this copy in step with `clip_windows`** -- if its gate changes and
    this does not, the two window sequences stop lining up and the equivalence test silently compares
    different windows.
    """
    from core.motion.actions.features import MIN_SPAN_S
    from core.motion.actions.live import ActionMonitor
    from training.public_data import MIN_SCORE, SOURCE_FPS, ROBOT_FPS, STEP_S

    height, width = annotation["img_shape"]
    keypoints, scores = annotation["keypoint"][0], annotation["keypoint_score"][0]
    monitor = ActionMonitor(aspect_ratio, min_score=MIN_SCORE, model_path=None,
                            window_s=ACTION_WINDOW_S)
    out, next_emit = [], None
    for index in range(0, len(keypoints), max(1, round(SOURCE_FPS / ROBOT_FPS))):
        frame = np.zeros((17, 3), np.float32)
        # The published arrays are float16 pixels: cast before dividing, or a norm overflows to nan.
        frame[:, 0] = np.asarray(keypoints[index, :, 0], np.float64) / width
        frame[:, 1] = np.asarray(keypoints[index, :, 1], np.float64) / height
        frame[:, 2] = scores[index]
        timestamp = index / SOURCE_FPS
        monitor.update(timestamp, frame)
        if monitor.window.span() < MIN_SPAN_S:
            continue
        if next_emit is None or timestamp >= next_emit:
            if monitor.features is not None:
                out.append(monitor.features)
            next_emit = timestamp + STEP_S
    return out


def test_the_live_path_reproduces_the_training_vector_exactly():
    """The live feature vector must be the one the model was trained on, feature for feature.

    Checked on real data too: over 220 windows of NTU wave/clapping and UCF101 squats, jumping jacks and
    push-ups, the two paths agree bit for bit (max |difference| 0.0 over all 34 features) when the live
    monitor is given the clip's own aspect ratio.
    """
    from core.motion.actions.features import action_features_vector

    annotation = _annotation(squatting)
    training = _training_windows(annotation)
    height, width = annotation["img_shape"]
    live = _live_windows(annotation, width / height)
    assert len(training) > 1 and len(live) == len(training)
    for expected, actual in zip(training, live):
        assert action_features_vector(actual) == pytest.approx(action_features_vector(expected))


def test_the_aspect_ratio_is_required_and_changes_the_features():
    """A wrong aspect ratio rescales x before anything is measured, so it must not have a default.

    Measured against the training path on real clips: a live path fixed at 16/9 against UCF101's 340x256
    shifts elbow_angle_amplitude by up to 17.2 degrees and flips the predicted class in 24% of
    jumping-jack and 29% of push-up windows. That is a silent wrong answer, so the caller must state it.
    """
    from core.motion.actions.live import ActionMonitor

    with pytest.raises(TypeError):
        ActionMonitor(model_path=None)          # no default: the camera's ratio is not guessable

    annotation = _annotation(squatting)
    height, width = annotation["img_shape"]
    right = _live_windows(annotation, width / height)[0]
    wrong = _live_windows(annotation, 16 / 9)[0]
    assert right["bbox_aspect"] != pytest.approx(wrong["bbox_aspect"])
    assert right["knee_angle_mean"] != pytest.approx(wrong["knee_angle_mean"], abs=1.0)
