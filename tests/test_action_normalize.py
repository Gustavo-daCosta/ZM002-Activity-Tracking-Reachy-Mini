"""Torso normalization: the scale that survives a side-view push-up."""

import math

import numpy as np
import pytest

from core.motion.actions import ACTION_WINDOW_S, ACTIONS
from core.motion.actions.normalize import SHOULDER_WIDTH_TO_TORSO_LENGTH, normalize_to_torso
from core.motion.window import KeypointWindow, normalize_to_shoulders
from core.pose_backends import (
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_HIP, RIGHT_SHOULDER,
)


def skeleton(positions):
    """A (17, 3) skeleton: only the given joints are confident, at the given (x, y); the rest are absent.

    A real pose backend never reports high confidence for a joint it cannot see, so unset joints get
    score 0.0 rather than sitting confidently at the origin -- a confident (0, 0) would pad the
    confident-keypoints bounding box with spread that is not really in the frame.
    `positions` is passed positionally because its keys are COCO joint indices: ints cannot be **kwargs.
    """
    keypoints = np.zeros((17, 3), np.float32)
    for index, (x, y) in positions.items():
        keypoints[index] = (x, y, 0.9)
    return keypoints


# The standing fixture's proportions are the measured medians of datasets/ntu60_hrnet.pkl, not eyeballed
# numbers, because later tasks in this plan calibrate feature thresholds against this very skeleton: a
# fixture that is not a body anyone has would calibrate them against a body that does not exist. The
# previous fixture put the shoulders 0.10 apart with a 0.30 torso -- a ratio of 3.0, where the dataset
# measures 1.666 square-on. Every proportion below is expressed in torso lengths and is derived from the
# constant rather than hard-coded, so the fixture cannot silently drift away from it.
SHOULDER_WIDTH = 0.10
TORSO = SHOULDER_WIDTH * SHOULDER_WIDTH_TO_TORSO_LENGTH
# hip -> knee -> ankle, summed, over 5 281 755 frames: p25 1.286, median 1.416, p75 1.589 torso lengths.
# Split evenly between the two segments (femur and tibia are close to equal in length).
LEG = 1.416 * TORSO
# Straight-line shoulder -> wrist for the raised waving arm. Summed shoulder->elbow->wrist over 4 945 393
# frames has median 0.99 torso lengths; a raised arm is bent, so the straight-line distance is shorter.
ARM_REACH = 0.8 * TORSO
SHOULDER_Y = 0.30
HIP_Y = SHOULDER_Y + TORSO
KNEE_Y = HIP_Y + LEG / 2
ANKLE_Y = HIP_Y + LEG


def standing():
    """A square-on standing person with a raised left arm, at NTU-median proportions."""
    # The raised arm points up and out at 1:2 (out:up), normalized to ARM_REACH.
    reach = ARM_REACH / math.hypot(1.0, 2.0)
    return skeleton({
        LEFT_SHOULDER: (0.45, SHOULDER_Y), RIGHT_SHOULDER: (0.55, SHOULDER_Y),
        LEFT_HIP: (0.46, HIP_Y), RIGHT_HIP: (0.54, HIP_Y),
        LEFT_KNEE: (0.46, KNEE_Y), LEFT_ANKLE: (0.46, ANKLE_Y),
        LEFT_WRIST: (0.45 - reach, SHOULDER_Y - 2 * reach),
    })


def test_the_standing_fixture_is_anthropometrically_real():
    """Guards the fixture itself: later tasks calibrate thresholds against these proportions."""
    standing_pose = standing()
    shoulder_width = standing_pose[RIGHT_SHOULDER, 0] - standing_pose[LEFT_SHOULDER, 0]
    torso = (standing_pose[LEFT_HIP, 1] + standing_pose[RIGHT_HIP, 1]) / 2 - standing_pose[LEFT_SHOULDER, 1]
    assert torso / shoulder_width == pytest.approx(1.666, abs=0.01)   # NTU square-on median
    leg = standing_pose[LEFT_ANKLE, 1] - standing_pose[LEFT_HIP, 1]
    assert leg / torso == pytest.approx(1.416, abs=0.01)              # NTU median, hip -> knee -> ankle
    reach = math.dist(standing_pose[LEFT_WRIST, :2], standing_pose[LEFT_SHOULDER, :2])
    assert reach / torso == pytest.approx(0.8, abs=0.01)


def test_both_scale_paths_agree_on_a_square_on_body():
    """Finding 1: identical motion must not change magnitude just because the hips dropped out.

    This is the property `SHOULDER_WIDTH_TO_TORSO_LENGTH` exists for. It holds exactly here because the
    fixture is square-on and at the measured ratio; under rotation the fallback over-estimates (see
    `test_person_turned_60_degrees_from_camera_is_still_accepted`) and nothing 2D can prevent that.
    """
    by_torso = normalize_to_torso(standing(), min_score=0.5, aspect_ratio=1.0)
    no_hips = standing()
    no_hips[[LEFT_HIP, RIGHT_HIP], 2] = 0.1
    by_shoulder_width = normalize_to_torso(no_hips, min_score=0.5, aspect_ratio=1.0)
    assert np.abs(by_shoulder_width[:, :2] - by_torso[:, :2]).max() < 0.01


def test_actions_are_the_six_agreed_labels():
    assert ACTIONS == ("none", "wave", "pushup", "squat", "clapping", "jumping_jacks")
    assert ACTION_WINDOW_S == 3.0


def test_scale_is_the_torso_so_the_hips_sit_one_unit_below_the_shoulders():
    normalized = normalize_to_torso(standing(), min_score=0.5, aspect_ratio=1.0)
    hip_mid_y = (normalized[LEFT_HIP, 1] + normalized[RIGHT_HIP, 1]) / 2
    assert hip_mid_y == pytest.approx(1.0, abs=0.01)


def test_side_view_pushup_stays_bounded_where_the_shoulder_scale_explodes():
    # Seen from the side, the two shoulders project onto nearly the same pixel.
    pushup = skeleton({
        LEFT_SHOULDER: (0.30, 0.50), RIGHT_SHOULDER: (0.3005, 0.50),
        LEFT_HIP: (0.60, 0.52), RIGHT_HIP: (0.6005, 0.52),
    })
    by_shoulders = normalize_to_shoulders(pushup, min_score=0.5, aspect_ratio=1.0)
    by_torso = normalize_to_torso(pushup, min_score=0.5, aspect_ratio=1.0)
    assert np.abs(by_shoulders[:, :2]).max() > 100      # the failure this task exists for
    # Measured 1.663: the bound is dominated by the absent joints parked at the frame origin, 1.66 torso
    # lengths from this pose's shoulder midpoint -- the point is that it is O(1), not O(100).
    assert np.abs(by_torso[:, :2]).max() < 2           # torso length is unaffected by the viewpoint

def test_falls_back_to_shoulder_width_without_confident_hips():
    no_hips = standing()
    no_hips[[LEFT_HIP, RIGHT_HIP], 2] = 0.1
    normalized = normalize_to_torso(no_hips, min_score=0.5, aspect_ratio=1.0)
    assert normalized is not None
    # The fallback unit is an estimated *torso* length, not a shoulder width, so shoulders 0.10 apart span
    # 1 / 1.666 = 0.600 of it -- which is exactly what the hip-confident path gives for this fixture too.
    separation = normalized[RIGHT_SHOULDER, 0] - normalized[LEFT_SHOULDER, 0]
    assert separation == pytest.approx(1.0 / SHOULDER_WIDTH_TO_TORSO_LENGTH, abs=0.01)


def test_returns_none_without_confident_shoulders():
    faceless = standing()
    faceless[[LEFT_SHOULDER, RIGHT_SHOULDER], 2] = 0.1
    assert normalize_to_torso(faceless, min_score=0.5, aspect_ratio=1.0) is None


def test_window_uses_the_normalizer_it_is_given():
    window = KeypointWindow(ACTION_WINDOW_S, 0.5, 1.0, normalizer=normalize_to_torso)
    window.add(0.0, standing())
    _, keypoints = window.frames()
    hip_mid_y = (keypoints[0, LEFT_HIP, 1] + keypoints[0, RIGHT_HIP, 1]) / 2
    assert hip_mid_y == pytest.approx(1.0, abs=0.01)


def test_window_still_defaults_to_the_shoulder_normalizer():
    window = KeypointWindow(1.5, 0.5, 1.0)
    window.add(0.0, standing())
    _, keypoints = window.frames()
    width = keypoints[0, RIGHT_SHOULDER, 0] - keypoints[0, LEFT_SHOULDER, 0]
    assert width == pytest.approx(1.0, abs=0.01)


def test_side_view_pushup_with_occluded_hip_is_rejected():
    # Seen edge-on with a hip occluded, the shoulders (collapsed to one pixel) carry no scale, even
    # though other joints (knee, ankle) are visible and spread across the frame -- so the confident-
    # keypoints bounding box is not itself degenerate. There is no credible scale in this frame, so it
    # must be rejected rather than scaled by a shoulder "width" that isn't really a width.
    pushup = skeleton({
        LEFT_SHOULDER: (0.30, 0.50), RIGHT_SHOULDER: (0.3005, 0.50),
        LEFT_HIP: (0.60, 0.52), RIGHT_HIP: (0.6005, 0.52),
        LEFT_KNEE: (0.75, 0.55), LEFT_ANKLE: (0.90, 0.58),
    })
    pushup[[LEFT_HIP, RIGHT_HIP], 2] = 0.1  # hip occluded/unreliable from the side
    assert normalize_to_torso(pushup, min_score=0.5, aspect_ratio=1.0) is None


def test_person_turned_60_degrees_from_camera_is_still_accepted():
    # A person standing at an angle to a desk robot is ordinary conversational range, not an edge case.
    # Shoulder width shrinks as cos(angle) with rotation, so reproduce that: same standing pose, hips not
    # confident (forcing the shoulder-width fallback), shoulders' horizontal separation scaled by
    # cos(60 degrees). The frame must still be accepted, with bounded normalized coordinates.
    turned = standing()
    turned[[LEFT_HIP, RIGHT_HIP], 2] = 0.1
    mid_x = (turned[LEFT_SHOULDER, 0] + turned[RIGHT_SHOULDER, 0]) / 2
    half_width = (turned[RIGHT_SHOULDER, 0] - turned[LEFT_SHOULDER, 0]) / 2 * math.cos(math.radians(60))
    turned[LEFT_SHOULDER, 0] = mid_x - half_width
    turned[RIGHT_SHOULDER, 0] = mid_x + half_width
    normalized = normalize_to_torso(turned, min_score=0.5, aspect_ratio=1.0)
    assert normalized is not None
    # Measured 6.02, down from 12.0 before SHOULDER_WIDTH_TO_TORSO_LENGTH existed. The residual is the
    # 1 / cos(60) = 2x rotation over-estimate that no purely 2D fallback can remove without the hips, times
    # the 3.01 this fixture reaches square-on. So 7 bounds it with a little headroom for the fixture's
    # proportions, and proves the frame is accepted and bounded rather than rejected.
    assert np.abs(normalized[:, :2]).max() < 7
    square_on = normalize_to_torso(standing(), min_score=0.5, aspect_ratio=1.0)
    over_estimate = np.abs(normalized[:, :2]).max() / np.abs(square_on[:, :2]).max()
    assert over_estimate == pytest.approx(1.0 / math.cos(math.radians(60)), abs=0.05)


def test_only_collapsed_shoulders_visible_is_rejected():
    # Nothing else in frame -- the normal case for a desk-height camera cropping the legs. The confident-
    # keypoints bounding box is then itself degenerate (just the two collapsed shoulders), so this is
    # rejected by the absolute body-extent floor rather than the shoulder/diagonal ratio.
    only_shoulders = skeleton({
        LEFT_SHOULDER: (0.30, 0.50), RIGHT_SHOULDER: (0.3005, 0.50),
    })
    assert normalize_to_torso(only_shoulders, min_score=0.5, aspect_ratio=1.0) is None
