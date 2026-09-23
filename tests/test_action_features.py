"""Whole-body features, checked against synthetic skeletons: one geometry per action.

Every skeleton here is built from `SHOULDER_WIDTH_TO_TORSO_LENGTH` and the other measured NTU-60
proportions, exactly as `tests/test_action_normalize.py` does, so both test files describe the same body.
An eyeballed fixture (shoulders 0.10 apart against a 0.30 torso, a ratio of 3.0 where real people measure
1.666) would calibrate these thresholds against a body nobody has.

Only the joints a pose actually provides are marked confident; every other joint keeps score 0.0. A real
pose backend never reports high confidence for a joint it cannot see, and a confident joint parked at the
frame origin dominates every extremum and bounding box in this module.
"""

import math

import numpy as np
import pytest

from core.motion.actions import ACTION_WINDOW_S
from core.motion.features import amplitude
from core.motion.actions.features import (
    ACTION_FEATURE_NAMES, action_features, action_features_vector,
)
from core.motion.actions.normalize import (
    SHOULDER_WIDTH_TO_TORSO_LENGTH, normalize_to_torso, torso_scale, trunk_height,
    trunk_height_parts,
)
from core.motion.window import KeypointWindow
from core.pose_backends import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)

FPS = 10.0
FRAMES = int(ACTION_WINDOW_S * FPS)

# --- the body ---------------------------------------------------------------------------------------
# All proportions are expressed in torso lengths and derived from the measured constant, never hard-coded.
SHOULDER_WIDTH = 0.10
TORSO = SHOULDER_WIDTH * SHOULDER_WIDTH_TO_TORSO_LENGTH          # 0.1666 frame units
LEG = 1.416 * TORSO                # hip -> knee -> ankle, NTU-60 median, split evenly between segments
UPPER_ARM = 0.50 * TORSO           # shoulder -> elbow -> wrist sums to 0.99 torso lengths (NTU median)
FOREARM = 0.49 * TORSO
ARM = UPPER_ARM + FOREARM
NECK = 0.45 * TORSO                # shoulder line -> nose
SHOULDER_Y = 0.30
HIP_Y = SHOULDER_Y + TORSO
KNEE_Y = HIP_Y + LEG / 2
ANKLE_Y = HIP_Y + LEG
HIP_HALF_WIDTH = 0.04              # hips are slightly narrower than the shoulders
SHOULDER_HALF = SHOULDER_WIDTH / 2
HAND_Y = 0.60 + 0.55 * ARM         # the push-up's planted hands, fixed in the image while the body moves


def elbow(shoulder, wrist, sign):
    """Place the elbow anatomically: UPPER_ARM from the shoulder, FOREARM to the wrist.

    Two-link inverse kinematics in the image plane, bending toward `sign` (-1 left, +1 right). Placing the
    elbow by eye instead lets a fixture claim a forearm that is not FOREARM long, which shows up directly
    as a wrong elbow angle - the very feature under test.
    """
    s = np.array(shoulder, float)
    w = np.array(wrist, float)
    reach = float(np.linalg.norm(w - s))
    unit = (w - s) / max(reach, 1e-9)
    reach = min(reach, ARM * 0.999)              # a real arm cannot be straighter than straight
    along = (reach ** 2 + UPPER_ARM ** 2 - FOREARM ** 2) / (2 * reach)
    across = math.sqrt(max(UPPER_ARM ** 2 - along ** 2, 0.0))
    normal = np.array([-unit[1], unit[0]]) * sign
    return tuple(s + along * unit + across * normal)


def build(pose_at):
    """Run `pose_at(t) -> {joint: (x, y)}` through a torso-normalized window and return its features."""
    window = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso)
    for index in range(FRAMES):
        t = index / FPS
        keypoints = np.zeros((17, 3), np.float32)
        for joint, (x, y) in pose_at(t).items():
            keypoints[joint] = (x, y, 0.9)      # only the joints we were given are confident
        window.add(t, keypoints)
    return action_features(*window.frames(), min_score=0.5)


def upright(t, override=None):
    """A person standing still, facing the camera, at NTU-median proportions.

    `override` is a positional dict: its keys are COCO joint indices, which cannot be **kwargs.
    """
    pose = {
        NOSE: (0.50, SHOULDER_Y - NECK),
        LEFT_SHOULDER: (0.50 - SHOULDER_HALF, SHOULDER_Y),
        RIGHT_SHOULDER: (0.50 + SHOULDER_HALF, SHOULDER_Y),
        LEFT_ELBOW: (0.50 - SHOULDER_HALF, SHOULDER_Y + UPPER_ARM),
        RIGHT_ELBOW: (0.50 + SHOULDER_HALF, SHOULDER_Y + UPPER_ARM),
        LEFT_WRIST: (0.50 - SHOULDER_HALF, SHOULDER_Y + ARM),
        RIGHT_WRIST: (0.50 + SHOULDER_HALF, SHOULDER_Y + ARM),
        LEFT_HIP: (0.50 - HIP_HALF_WIDTH, HIP_Y), RIGHT_HIP: (0.50 + HIP_HALF_WIDTH, HIP_Y),
        LEFT_KNEE: (0.50 - HIP_HALF_WIDTH, KNEE_Y), RIGHT_KNEE: (0.50 + HIP_HALF_WIDTH, KNEE_Y),
        LEFT_ANKLE: (0.50 - HIP_HALF_WIDTH, ANKLE_Y), RIGHT_ANKLE: (0.50 + HIP_HALF_WIDTH, ANKLE_Y),
    }
    pose.update(override or {})
    return pose


def test_the_synthetic_body_is_anthropometrically_real():
    """Guards the fixture: the thresholds below only mean something for a body someone could have."""
    pose = upright(0.0)
    shoulder_width = pose[RIGHT_SHOULDER][0] - pose[LEFT_SHOULDER][0]
    torso = (pose[LEFT_HIP][1] + pose[RIGHT_HIP][1]) / 2 - pose[LEFT_SHOULDER][1]
    assert torso / shoulder_width == pytest.approx(1.666, abs=0.01)   # NTU square-on median
    leg = pose[LEFT_ANKLE][1] - pose[LEFT_HIP][1]
    assert leg / torso == pytest.approx(1.416, abs=0.01)              # NTU median, hip -> knee -> ankle
    arm = pose[LEFT_WRIST][1] - pose[LEFT_SHOULDER][1]
    assert arm / torso == pytest.approx(0.99, abs=0.01)               # NTU median, shoulder -> wrist


def test_feature_names_are_the_agreed_features():
    """34 now, not 35: every oscillation is a rate per second, and the wave's raw `reversals` count

    would have become a second copy of its own `reversal_rate`, so it is not embedded any more.
    """
    assert len(ACTION_FEATURE_NAMES) == 34
    assert len(set(ACTION_FEATURE_NAMES)) == 34
    assert "reversals" not in ACTION_FEATURE_NAMES       # the count; `reversal_rate` is the column
    assert not any(name.endswith("_reversals") for name in ACTION_FEATURE_NAMES)
    assert "trunk_height_amplitude" in ACTION_FEATURE_NAMES
    assert "trunk_height_reversal_rate" in ACTION_FEATURE_NAMES
    assert ACTION_FEATURE_NAMES[0] == "torso_angle_mean"
    assert "lower_body_valid_frac" in ACTION_FEATURE_NAMES
    assert "upper_body_valid_frac" in ACTION_FEATURE_NAMES


def test_vector_follows_the_name_order():
    features = build(upright)
    vector = action_features_vector(features)
    assert vector.dtype == np.float32
    assert vector.shape == (len(ACTION_FEATURE_NAMES),)
    assert vector[0] == pytest.approx(features["torso_angle_mean"])


def test_standing_still_is_upright_and_quiet():
    features = build(upright)
    assert features["torso_angle_mean"] < 15.0       # degrees from vertical
    assert features["hip_y_amplitude"] < 0.05
    assert features["knee_angle_mean"] > 160.0       # legs straight
    assert features["lower_body_valid_frac"] == pytest.approx(1.0)


def pushup(t):
    """A real push-up: hands planted, elbows tucked back toward the ribs, the trunk rising and falling.

    Seen from in front and above, head to the left, so the body axis runs along image x and the two sides
    are separated in image y by about a shoulder width. Three things the brief's fixture got wrong and that
    matter here: the hands stay at a fixed point (they are planted, it is the body that moves), the arm is
    strongly foreshortened because it points away from the camera rather than down the image plane, and the
    elbow travels back along the ribs as the chest drops instead of splaying outwards. The splayed, full-
    length arm inflated the body's vertical extent and dragged bbox_aspect down to 1.676.
    """
    phase = (1 + np.cos(2 * np.pi * 0.4 * t)) / 2         # 1 = top, arms straight; 0 = bottom, chest low
    shoulder_x = 0.30
    shoulder_y = HAND_Y - (0.20 + 0.35 * phase) * ARM
    lift = shoulder_y - (HAND_Y - 0.55 * ARM)             # the body pivots about the planted toes
    elbow_x = shoulder_x + 0.55 * UPPER_ARM * (1 - phase) + 0.05 * ARM * phase
    elbow_y = HAND_Y - (0.13 + 0.15 * phase) * ARM
    hip_y = shoulder_y - 0.6 * lift + 0.02
    knee_y = shoulder_y - 0.3 * lift + 0.04
    ankle_y = shoulder_y - lift + 0.06
    width = SHOULDER_WIDTH
    return {
        NOSE: (shoulder_x - NECK, shoulder_y + width / 2),
        LEFT_SHOULDER: (shoulder_x, shoulder_y), RIGHT_SHOULDER: (shoulder_x, shoulder_y + width),
        LEFT_ELBOW: (elbow_x, elbow_y), RIGHT_ELBOW: (elbow_x, elbow_y + width),
        LEFT_WRIST: (shoulder_x + 0.01, HAND_Y), RIGHT_WRIST: (shoulder_x + 0.01, HAND_Y + width),
        LEFT_HIP: (shoulder_x + TORSO, hip_y), RIGHT_HIP: (shoulder_x + TORSO, hip_y + width),
        LEFT_KNEE: (shoulder_x + TORSO + LEG / 2, knee_y),
        RIGHT_KNEE: (shoulder_x + TORSO + LEG / 2, knee_y + width),
        LEFT_ANKLE: (shoulder_x + TORSO + LEG, ankle_y),
        RIGHT_ANKLE: (shoulder_x + TORSO + LEG, ankle_y + width),
    }


def test_pushup_is_a_horizontal_torso():
    features = build(pushup)
    assert features["torso_angle_mean"] > 70.0       # near horizontal; measured 85.9
    assert features["bbox_aspect"] > 1.5             # wide and flat; measured 2.341
    assert build(upright)["bbox_aspect"] < 1.0       # the contrast that separates them; measured 0.209


def test_pushup_repetitions_show_up_in_the_elbow_not_in_the_static_shape():
    """A held plank has the same torso_angle and bbox_aspect. Only the elbow cycles."""
    features = build(pushup)
    assert features["elbow_angle_amplitude"] > 60.0  # measured 121.3
    assert features["elbow_angle_min"] < 90.0        # measured 45.5, the bottom of the repetition
    # The legs are what a desk camera crops, and the trunk-height signal dies with them: the elbow does not.
    def waist_up(t):
        return {joint: xy for joint, xy in pushup(t).items()
                if joint not in (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)}

    cropped = build(waist_up)
    assert cropped["shoulder_y_amplitude"] == 0.0
    assert cropped["elbow_angle_amplitude"] > 60.0


def squatting(t):
    """The ankles stay planted, the whole trunk drops by up to 0.4 torso lengths and the knees travel
    outwards, which is what bends the hip-knee-ankle angle in a frontal view.

    The *whole* trunk, head and arms descend together with the hips, not the hips alone: the torso is a rigid
    segment, so lowering the hips while pinning the shoulders lengthens it by 0.4 torso lengths, which is a
    body nobody has. That version also silently drove the torso scale itself up and down, which is where its
    `shoulder_y_amplitude` of 0.658 came from -- an oscillating denominator rather than a moving body -- and
    it left `trunk_height` at exactly 0.0 for a squat, the one action this feature exists to see.
    """
    phase = (1 - np.cos(2 * np.pi * 0.4 * t)) / 2         # down and back up
    drop = 0.40 * TORSO * phase
    pose = upright(t)
    # Everything from the hips up is rigid and descends together; the knees descend by a third as much
    # (they bend), and the feet stay planted.
    for joint in (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW,
                  LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP):
        x, y = pose[joint]
        pose[joint] = (x, y + drop)
    pose[LEFT_KNEE] = (0.50 - HIP_HALF_WIDTH - 0.35 * TORSO * phase, KNEE_Y + drop / 3)
    pose[RIGHT_KNEE] = (0.50 + HIP_HALF_WIDTH + 0.35 * TORSO * phase, KNEE_Y + drop / 3)
    return pose


def test_squat_bends_the_knees_without_tilting_the_torso():
    features = build(squatting)
    assert features["knee_angle_amplitude"] > 10.0
    assert features["knee_angle_min"] < features["knee_angle_mean"]
    assert features["torso_angle_mean"] < 25.0       # still upright, unlike a push-up
    assert features["hip_y_reversal_rate"] >= 0.3      # measured 0.345 = 1 reversal over 2.9 s
    # The trunk really does rise and fall relative to the planted feet: measured 0.380 torso lengths (it
    # read 0.658 while the fixture stretched the torso instead of lowering it -- see `squatting`).
    assert features["shoulder_y_amplitude"] > 0.2
    # Arms hanging straight throughout: the elbow is the push-up's signal, not the squat's.
    assert features["elbow_angle_amplitude"] < 5.0   # measured 0.0
    assert features["elbow_angle_mean"] > 160.0      # measured 180.0


def clapping(t):
    """Hands in front of the chest, the gap closing and opening at 1.5 Hz, elbows placed by anatomy."""
    gap = TORSO * (0.05 + 0.45 * (1 + np.sin(2 * np.pi * 1.5 * t)) / 2)
    chest_y = SHOULDER_Y + 0.75 * TORSO
    left_wrist = (0.50 - gap, chest_y)
    right_wrist = (0.50 + gap, chest_y)
    return upright(t, {
        LEFT_WRIST: left_wrist, RIGHT_WRIST: right_wrist,
        LEFT_ELBOW: elbow((0.50 - SHOULDER_HALF, SHOULDER_Y), left_wrist, -1),
        RIGHT_ELBOW: elbow((0.50 + SHOULDER_HALF, SHOULDER_Y), right_wrist, 1),
    })


def test_clapping_brings_the_wrists_together_repeatedly():
    features = build(clapping)
    assert features["wrist_distance_reversal_rate"] >= 1.3   # measured 2.759 = 8 over 2.9 s
    assert features["wrist_distance_min"] < features["wrist_distance_mean"]
    assert features["wrists_above_head_frac"] == 0.0
    # Clapping is arms held folded: the elbow's *mean* separates it from a squat or a jumping jack (both
    # ~180). Measured 101.5 here, and real data agrees - median 86.2 for clapping against 142.3 for waving.
    assert features["elbow_angle_mean"] < 130.0
    # There is deliberately NO assertion on elbow_angle_amplitude here, and please do not add one. This
    # fixture measures 6.6, because its hands sweep an arc at a perfectly constant radius from the shoulder
    # and the elbow angle depends only on that radius. Real clapping does not work that way: people's elbows
    # flex considerably, and on real data elbow_angle_amplitude *favours* clapping - median 127 against
    # waving's 69, the opposite sign. An assertion that passes for a fixture reason while stating something
    # false about the world is worse than no assertion: it would read as evidence that clapping has a flat
    # elbow signal, and a model could be tuned against it.


def jumping_jacks(t):
    """Straight arms sweeping an arc from the hips to overhead, feet spreading in time with them.

    The hand follows an arc about the shoulder at a near-constant radius, because that is what a straight
    arm does. A straight *line* from hip height to overhead passes close to the shoulder and would fold the
    elbow to 18 degrees halfway through - an artefact of the fixture, not of the action.
    """
    phase = (1 + np.sin(2 * np.pi * 1.0 * t)) / 2
    swing = math.radians(160.0 * phase)        # 0 = hanging down, 160 = overhead and a little outwards
    radius = 0.995 * ARM
    spread = 0.36 * TORSO * phase              # each foot; a 1.20-torso stance at the widest
    override = {
        LEFT_ANKLE: (0.50 - HIP_HALF_WIDTH - spread, ANKLE_Y),
        RIGHT_ANKLE: (0.50 + HIP_HALF_WIDTH + spread, ANKLE_Y),
    }
    for shoulder_x, sign, elbow_joint, wrist_joint in (
            (0.50 - SHOULDER_HALF, -1, LEFT_ELBOW, LEFT_WRIST),
            (0.50 + SHOULDER_HALF, 1, RIGHT_ELBOW, RIGHT_WRIST)):
        shoulder = (shoulder_x, SHOULDER_Y)
        wrist = (shoulder_x + sign * radius * math.sin(swing), SHOULDER_Y + radius * math.cos(swing))
        override[wrist_joint] = wrist
        override[elbow_joint] = elbow(shoulder, wrist, sign)
    return upright(t, override)


def test_jumping_jacks_raise_the_wrists_above_the_head_and_spread_the_ankles():
    features = build(jumping_jacks)
    assert features["wrists_above_head_frac"] > 0.2   # measured 0.400
    assert features["ankle_separation_amplitude"] > 0.3   # measured 0.685
    assert features["wrist_y_reversal_rate"] >= 0.68  # measured 2.069 = 6 over 2.9 s
    # Done with straight arms, so the elbow carries nothing: this is the contrast against a push-up.
    assert features["elbow_angle_amplitude"] < 5.0    # measured 0.0
    assert features["elbow_angle_mean"] > 160.0       # measured 168.5


def waving(t):
    """One arm raised, the upper arm still, the forearm swinging about the elbow at 2 Hz."""
    elbow_xy = (0.50 - SHOULDER_HALF - 0.25 * TORSO, SHOULDER_Y - 0.35 * TORSO)
    angle = math.radians(75 + 35 * np.sin(2 * np.pi * 2.0 * t))
    return upright(t, {
        LEFT_ELBOW: elbow_xy,
        LEFT_WRIST: (elbow_xy[0] - FOREARM * math.cos(angle), elbow_xy[1] - FOREARM * math.sin(angle)),
    })


def test_waving_swings_one_forearm_while_the_body_stays_still():
    features = build(waving)
    # The embedded wave features still work; measured 3.793 = 11 reversals over 2.9 s.
    assert features["reversal_rate"] >= 1.4
    assert features["above_frac"] == pytest.approx(1.0)   # the waving hand is above its shoulder
    assert features["knee_angle_mean"] > 160.0        # standing, legs straight
    assert features["torso_angle_mean"] < 15.0
    # Both arms are pooled, so the still arm holds the mean high while the waving one supplies the spread.
    assert features["elbow_angle_min"] < 140.0        # measured 126.2
    assert features["elbow_angle_amplitude"] > 20.0   # measured 53.8, the waving arm's own span


def legless(t):
    """Upright, but framed from the waist up -- the robot's usual view from a desk."""
    return {joint: xy for joint, xy in upright(t).items()
            if joint not in (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)}


def test_invisible_legs_are_reported_not_guessed():
    features = build(legless)
    assert features is not None                      # an upper-body window is still usable
    assert features["lower_body_valid_frac"] == 0.0
    assert features["knee_angle_mean"] == 0.0        # a sentinel, never an invented angle
    assert features["elbow_angle_mean"] > 160.0      # the arms are still visible, so this is measured


def test_vertical_oscillation_needs_the_ankles_as_its_ground_reference():
    """Pins the fix for a dead feature: shoulder_y/hip_y must not be measured from the shoulder origin.

    Normalization puts the origin on the shoulder midpoint and makes the shoulder-to-hip distance exactly
    1, so a shoulder-relative `shoulder_y` is identically 0 and `hip_y` identically cos(trunk tilt). Both
    measured a span of 0.0000 across a 0.4-torso squat before this was referenced to the ankles. With the
    legs out of frame there is no ground reference at all, and the honest answer is the 0.0 sentinel.
    """
    features = build(legless)
    assert features["shoulder_y_amplitude"] == 0.0
    assert features["hip_y_amplitude"] == 0.0
    assert features["hip_y_reversal_rate"] == 0.0


def motionless_but_asymmetric(t):
    """Nothing moves at all: the left arm hangs straight, the right forearm is folded to 90 degrees.

    Real people stand asymmetrically, and a one-sided occlusion biases one limb systematically, so this is
    ordinary rather than contrived.
    """
    shoulder = (0.50 + SHOULDER_HALF, SHOULDER_Y)
    elbow_xy = (shoulder[0], shoulder[1] + UPPER_ARM)
    return upright(t, {
        RIGHT_ELBOW: elbow_xy,
        RIGHT_WRIST: (elbow_xy[0] - FOREARM, elbow_xy[1]),   # forearm horizontal: 90 degrees at the elbow
    })


def test_a_static_left_right_difference_is_not_repetition():
    """Regression: the amplitude is per side, not over both sides' concatenated angle series.

    Pooling made this motionless skeleton report an elbow amplitude of 90.0 - larger than a real wave's
    42.4 - and a one-knee-outward pose report 52.6, inside squat territory (66.2). A false repetition
    signal in the two features that exist to measure repetition.
    """
    features = build(motionless_but_asymmetric)
    assert features["elbow_angle_amplitude"] < 1.0    # measured 0.0; was 90.0 when pooled
    assert features["elbow_angle_min"] == pytest.approx(90.0, abs=1.0)   # the fold is still reported
    assert features["elbow_angle_mean"] == pytest.approx(135.0, abs=1.0)  # mean still pools both sides

    def one_knee_out(t):
        return upright(t, {RIGHT_KNEE: (0.50 + HIP_HALF_WIDTH + 0.35 * TORSO, KNEE_Y)})

    assert build(one_knee_out)["knee_angle_amplitude"] < 1.0   # measured 0.0; was 52.6 when pooled


def test_validity_fractions_disambiguate_a_zero_angle():
    """0.0 is not a unique sentinel - a fully folded arm really does read 0 - so each limb pair has a flag."""
    full = build(upright)
    assert full["upper_body_valid_frac"] == pytest.approx(1.0)
    assert full["lower_body_valid_frac"] == pytest.approx(1.0)

    def armless(t):
        return {joint: xy for joint, xy in upright(t).items()
                if joint not in (LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)}

    cropped = build(armless)
    assert cropped["elbow_angle_mean"] == 0.0          # not measurable
    assert cropped["upper_body_valid_frac"] == 0.0      # ... and this is what says so
    assert cropped["lower_body_valid_frac"] == pytest.approx(1.0)


def test_too_short_a_window_has_no_features():
    window = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso)
    window.add(0.0, np.zeros((17, 3), np.float32))
    assert action_features(*window.frames(), min_score=0.5) is None


# --- task 2b: the trunk-height channel, which is what makes a waist-up squat visible at all -----------

def build_with_trunk_height(pose_at):
    """Like `build`, but also feeds the trunk-height extra channel through the window."""
    window = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso,
                            extra=trunk_height_parts)
    for index in range(FRAMES):
        t = index / FPS
        keypoints = np.zeros((17, 3), np.float32)
        for joint, (x, y) in pose_at(t).items():
            keypoints[joint] = (x, y, 0.9)
        window.add(t, keypoints)
    times, keypoints = window.frames()
    return action_features(times, keypoints, min_score=0.5, trunk_heights=window.extras())


def waist_up(pose_at):
    """Drop everything from the hips down: the Reachy Mini's own framing from a desk."""
    dropped = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)

    def cropped(t):
        return {joint: xy for joint, xy in pose_at(t).items() if joint not in dropped}

    return cropped


def test_a_waist_up_squat_is_no_longer_identical_to_standing_still():
    """The load-bearing test. Before task 2b these two vectors were bit-identical on all 33 slots."""
    squat = build_with_trunk_height(waist_up(squatting))
    still = build_with_trunk_height(waist_up(upright))
    assert squat["trunk_height_amplitude"] > 0.1     # measured 0.380, the same as the full-body window
    # measured 0.345 = one repetition over the window's 2.9 s
    assert squat["trunk_height_reversal_rate"] >= 0.3
    assert still["trunk_height_amplitude"] < 0.01
    assert still["trunk_height_reversal_rate"] == 0.0
    # ... and these two features are the ONLY thing that separates them: every other slot is equal.
    # `abs=1e-5` rather than exact equality because the normalized keypoints are float32 and the squat's
    # whole trunk sits at a different frame y, so the same normalized coordinate is rounded differently:
    # the largest observed difference over all the other slots is 1.8e-07, which is float32 rounding at this
    # magnitude and not a signal. Any real difference between these two windows would be orders larger.
    for name in ACTION_FEATURE_NAMES:
        if name in ("trunk_height_amplitude", "trunk_height_reversal_rate"):
            continue
        assert squat[name] == pytest.approx(still[name], abs=1e-5), name


def test_a_full_body_squat_keeps_its_knee_and_ankle_referenced_signals():
    """Nothing regressed: the leg signals are unchanged by the new channel."""
    features = build_with_trunk_height(squatting)
    assert features["knee_angle_amplitude"] > 10.0
    assert features["hip_y_reversal_rate"] >= 0.3       # measured 0.345
    assert features["shoulder_y_amplitude"] > 0.2
    assert features["trunk_height_amplitude"] > 0.1     # and the new one agrees with them
    without = build(squatting)
    for name in ACTION_FEATURE_NAMES:
        if name in ("trunk_height_amplitude", "trunk_height_reversal_rate"):
            continue
        assert features[name] == pytest.approx(without[name]), name
    assert without["trunk_height_amplitude"] == 0.0     # no channel given -> no invented signal
    assert without["trunk_height_reversal_rate"] == 0.0


def test_trunk_height_is_invariant_to_apparent_size():
    """The property the whole design rests on: it must measure the pose, not the distance to the camera.

    Moving away from a pinhole camera scales the image about the principal point, so that is the transform
    the feature has to survive.
    """
    pose = upright(0.0)
    keypoints = np.zeros((17, 3), np.float32)
    for joint, (x, y) in pose.items():
        keypoints[joint] = (x, y, 0.9)
    reference = trunk_height(keypoints, 0.5, 1.0)
    assert np.isfinite(reference)
    for factor in (0.5, 2.0):
        scaled = keypoints.copy()
        scaled[:, :2] = 0.5 + (keypoints[:, :2] - 0.5) * factor
        scaled[:, 2] = keypoints[:, 2]
        assert trunk_height(scaled, 0.5, 1.0) == pytest.approx(reference, rel=0.02)


def test_trunk_height_is_nan_exactly_where_normalization_has_no_scale():
    from tests.test_action_normalize import skeleton

    occluded_hip = skeleton({
        LEFT_SHOULDER: (0.30, 0.50), RIGHT_SHOULDER: (0.3005, 0.50),
        LEFT_HIP: (0.60, 0.52), RIGHT_HIP: (0.6005, 0.52),
        LEFT_KNEE: (0.75, 0.55), LEFT_ANKLE: (0.90, 0.58),
    })
    occluded_hip[[LEFT_HIP, RIGHT_HIP], 2] = 0.1
    only_collapsed_shoulders = skeleton({
        LEFT_SHOULDER: (0.30, 0.50), RIGHT_SHOULDER: (0.3005, 0.50),
    })
    for degenerate in (occluded_hip, only_collapsed_shoulders, np.zeros((17, 3), np.float32)):
        assert normalize_to_torso(degenerate, min_score=0.5, aspect_ratio=1.0) is None
        assert torso_scale(degenerate, min_score=0.5, aspect_ratio=1.0) is None
        assert math.isnan(trunk_height(degenerate, min_score=0.5, aspect_ratio=1.0))
    good = skeleton({joint: xy for joint, xy in upright(0.0).items()})
    assert torso_scale(good, min_score=0.5, aspect_ratio=1.0) is not None
    assert np.isfinite(trunk_height(good, min_score=0.5, aspect_ratio=1.0))


def test_a_window_without_an_extra_channel_behaves_exactly_as_before():
    """The wave path shares KeypointWindow, so `extra=None` must move nothing."""
    plain = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso)
    withchannel = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso,
                                 extra=trunk_height_parts)
    for index in range(FRAMES):
        t = index / FPS
        keypoints = np.zeros((17, 3), np.float32)
        for joint, (x, y) in upright(t).items():
            keypoints[joint] = (x, y, 0.9)
        plain.add(t, keypoints)
        withchannel.add(t, keypoints)
    plain_times, plain_keypoints = plain.frames()
    other_times, other_keypoints = withchannel.frames()
    assert np.array_equal(plain_times, other_times)
    assert np.array_equal(np.nan_to_num(plain_keypoints), np.nan_to_num(other_keypoints))
    extras = plain.extras()
    assert extras.shape == (FRAMES,)          # one scalar nan per frame, the extra=None contract
    assert np.isnan(extras).all()
    assert withchannel.extras().shape == (FRAMES, 2)   # (centre-referenced y, torso scale) per frame
    assert np.isfinite(withchannel.extras()).all()


def test_a_misaligned_trunk_height_channel_raises_instead_of_reporting_a_sentinel():
    """A channel of the wrong length is a caller bug, not missing data, so it must be loud.

    Reporting the 0.0 sentinel would make it indistinguishable from "the legs are not visible" and could
    quietly poison a whole training run.
    """
    window = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso,
                            extra=trunk_height_parts)
    for index in range(FRAMES):
        t = index / FPS
        keypoints = np.zeros((17, 3), np.float32)
        for joint, (x, y) in squatting(t).items():
            keypoints[joint] = (x, y, 0.9)
        window.add(t, keypoints)
    times, keypoints = window.frames()
    heights = window.extras()
    assert action_features(times, keypoints, 0.5, heights)["trunk_height_amplitude"] > 0.1
    with pytest.raises(ValueError, match="does not belong to these frames"):
        action_features(times, keypoints, 0.5, heights[:-3])
    with pytest.raises(ValueError, match=r"\(frames, 2\)"):
        action_features(times, keypoints, 0.5, heights[:, 0])


def turning_on_the_spot(t):
    """Waist-up, perfectly still vertically, but rotating the shoulders left and right.

    Ordinary desk behaviour, and the reason the window's scale must be held fixed: with the hips out of
    frame `torso_scale` is the shoulder width, which projects as cos(rotation), so a per-frame denominator
    turns this motionless-in-y body into a trunk-height oscillation.
    """
    angle = math.radians(50.0) * (1 + np.sin(2 * np.pi * 0.7 * t)) / 2
    half = SHOULDER_HALF * math.cos(angle)
    pose = waist_up(upright)(t)
    pose[LEFT_SHOULDER] = (0.50 - half, SHOULDER_Y)
    pose[RIGHT_SHOULDER] = (0.50 + half, SHOULDER_Y)
    return pose


def test_a_rotating_denominator_is_not_vertical_motion():
    """Regression for the window-median scale: the person's shoulders never change height at all."""
    features = build_with_trunk_height(turning_on_the_spot)
    assert features["trunk_height_amplitude"] < 0.01     # measured 0.0000
    assert features["trunk_height_reversal_rate"] == 0.0

    # What the per-frame denominator would have made of the same frames, for the contrast.
    per_frame = []
    for index in range(FRAMES):
        keypoints = np.zeros((17, 3), np.float32)
        for joint, (x, y) in turning_on_the_spot(index / FPS).items():
            keypoints[joint] = (x, y, 0.9)
        per_frame.append(trunk_height(keypoints, 0.5, 1.0))
    assert amplitude(np.array(per_frame)) > 0.3          # measured 0.4634 of pure denominator jitter
