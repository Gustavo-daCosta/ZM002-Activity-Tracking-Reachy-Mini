"""Keypoints normalized to the torso: the scale that survives a person lying down.

Also home to `trunk_height`, the one quantity here that deliberately does NOT remove global translation.
`normalize_to_torso` puts the origin on the shoulder midpoint, which is what makes the features
framing-independent -- and which also deletes the whole of a squat, since a squat is mostly global vertical
translation of an otherwise upright body. From the Reachy Mini's desk-height camera the legs are out of
frame, so every leg-based squat signal is unmeasurable and a waist-up squat is bit-identical to standing
still on all the other features. `trunk_height` is the translation-preserving, scale-normalized quantity
that fills that gap, and it needs no legs.

Known limitation: `trunk_height` is measured in the camera frame, so **camera motion injects fake
oscillation** -- a panning or handheld camera moves the shoulder midpoint through the frame exactly as a
crouch does. Fixed-camera studio footage (NTU-60) is clean; in-the-wild footage (UCF101, where the push-up,
squat and jumping-jack training clips come from) is not, and the training task should expect a
systematically noisier trunk-height signal for the UCF101 classes than for the NTU ones rather than be
surprised by it. Measured on 3 s windows: static NTU classes reach a p90 trunk-height amplitude of
0.04-0.09, static UCF101 classes 0.13-0.17 -- the same median but a 2-4x heavier tail, which is the camera
moving, not the person. The feature pipeline divides by the window's *median* scale for this reason (see
`trunk_height_parts`), which cuts that tail by about 10x in the robot's own waist-up framing.
"""

import numpy as np

from core.pose_backends import LEFT_HIP, LEFT_SHOULDER, RIGHT_HIP, RIGHT_SHOULDER

MIN_TORSO_LENGTH = 1e-3
# A confident-keypoints bounding box smaller than this (in frame units) is not a scalable body at all --
# e.g. only the two shoulders visible and collapsed onto each other by a side-on view. There is no scale
# to recover in that frame, so it must be rejected rather than floored to an arbitrary number.
MIN_BODY_EXTENT = 0.05
# When the hips are not confident, the shoulders' width is used as the fallback scale -- it is anatomical
# and framing-independent, unlike a bounding-box diagonal, which shrinks or grows with the crop. But if
# that width is below this fraction of the confident keypoints' bounding-box diagonal, the person is being
# seen edge-on (shoulders collapsed) while other joints are still visible spread across the frame: the
# shoulders themselves carry no scale information in that pose, so the frame must be rejected rather than
# scaled by a width that is not really the shoulders' width.
#
# Kept deliberately low: shoulder width shrinks as cos(angle) with rotation away from the camera, and a
# person turning up to ~82 degrees from square-on is ordinary conversational range for a desk robot, not
# an edge case -- 0.02 accepts out to about that angle (acos(0.02 / 0.1397), where 0.1397 is the measured
# square-on ratio). The degenerate side-view-with-occluded-hip case measures ratio ~0.000826, so 0.02 still
# keeps ~24x margin above it. The costs are asymmetric: a false reject silently drops every frame at a
# normal angle, while a false accept is protected by that ~24x margin -- so this threshold errs low.
MIN_SHOULDER_WIDTH_FRACTION_OF_BODY_EXTENT = 0.02
# The two scale paths must be commensurate: a shoulder width is systematically *shorter* than a torso
# length, so dividing by it would inflate every coordinate and make the same motion produce different
# feature amplitudes depending only on whether the hips happened to be confident. This constant converts
# an observed shoulder width into an estimated torso length, so the fallback unit means the same thing as
# the real one.
#
# Measured on datasets/ntu60_hrnet.pkl (COCO-17, HRNet), over frames where all four trunk joints
# (both shoulders, both hips) score >= 0.5. Cast the stored float16 keypoints to float64 before any norm:
# at 1920-pixel coordinates float16 overflows to inf and the ratio comes out nan.
#
# torso_length / shoulder_width, over all 5 348 703 such frames:
#     p5 1.224   p25 1.573   median 1.940   p75 3.463   p95 10.196
# That distribution is not anthropometry -- it is anthropometry convolved with viewing angle. Shoulder
# width projects as cos(angle) while torso length does not, so every frame where the person is turned away
# from the camera inflates the ratio, which is what produces that long upper tail. Taking its median would
# bake a partial rotation correction into the constant, and this normalizer deliberately does not correct
# for rotation (it cannot: with no confident hips there is nothing 2D left to estimate the angle from).
#
# So the constant is the *square-on* ratio, estimated by taking, in each of the 56 562 sequences, the
# single frame where the shoulders project widest -- the frame closest to facing the camera:
#     p5 1.070   p25 1.438   median 1.666   p75 2.280   p95 5.420
# Cross-check, over all 891 465 frames within 95% of their sequence's widest shoulders: median 1.540.
#
# 1.666 is therefore the measured median square-on ratio, not a round number -- do not "tidy" it.
# What it buys: at 0 degrees of rotation, for a real body, the fallback and the torso path agree. Under
# rotation the projected shoulder width shrinks while the true torso length does not, so a residual
# over-estimate of about 1/cos(angle) (~2x at 60 degrees) remains. No purely 2D fallback can do better
# without the hips.
SHOULDER_WIDTH_TO_TORSO_LENGTH = 1.666
# The frame's vertical centre, which `trunk_height` measures from. Moving away from a pinhole camera scales
# the image about the principal point, so a height measured from the frame centre is invariant to how far
# away the person stands, while a height measured from the top edge of the frame is not.
FRAME_CENTRE_Y = 0.5


def torso_scale(keypoints, min_score, aspect_ratio):
    """The body scale `normalize_to_torso` divides by, in aspect-corrected frame units, or None.

    Factored out so `trunk_height` measures its height in exactly the same unit: two copies of this logic
    would drift, and a height and a scale in different units make a meaningless ratio. Returns None in
    exactly the cases `normalize_to_torso` rejects -- no confident shoulders, a confident-keypoints bounding
    box below `MIN_BODY_EXTENT`, or shoulders collapsed below
    `MIN_SHOULDER_WIDTH_FRACTION_OF_BODY_EXTENT` of that box with no confident hips to fall back on.
    """
    if keypoints is None:
        return None
    if keypoints[LEFT_SHOULDER, 2] < min_score or keypoints[RIGHT_SHOULDER, 2] < min_score:
        return None
    xy = keypoints[:, :2] * np.array([aspect_ratio, 1.0], np.float32)
    shoulder_mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    if keypoints[LEFT_HIP, 2] >= min_score and keypoints[RIGHT_HIP, 2] >= min_score:
        hip_mid = (xy[LEFT_HIP] + xy[RIGHT_HIP]) / 2
        scale = float(np.linalg.norm(hip_mid - shoulder_mid))
    else:
        shoulder_width = float(np.linalg.norm(xy[LEFT_SHOULDER] - xy[RIGHT_SHOULDER]))
        confident_xy = xy[keypoints[:, 2] >= min_score]
        diagonal = float(np.linalg.norm(confident_xy.max(axis=0) - confident_xy.min(axis=0)))
        if diagonal < MIN_BODY_EXTENT:
            return None
        if shoulder_width < MIN_SHOULDER_WIDTH_FRACTION_OF_BODY_EXTENT * diagonal:
            return None
        scale = shoulder_width * SHOULDER_WIDTH_TO_TORSO_LENGTH
    return max(scale, MIN_TORSO_LENGTH)


def trunk_height(keypoints, min_score, aspect_ratio):
    """The shoulder midpoint's height in the frame, in torso lengths, or nan with no credible scale.

    Measured from `FRAME_CENTRE_Y` and divided by `torso_scale`, so it is invariant to how far away the
    person stands and to where the crop begins, and it needs no legs.

    **Image y points down, so a person who crouches gets a LARGER value.** That inverts the intuition the
    name suggests; only differences of this quantity are ever used as features, so the sign matters only
    when reading a raw value.

    Returns nan in exactly the frames `normalize_to_torso` returns None for.

    This is the single-frame form, dividing by that frame's own scale. The feature pipeline uses
    `trunk_height_parts` instead and divides by the window's median scale; see that function for why.
    """
    height, scale = trunk_height_parts(keypoints, min_score, aspect_ratio)
    return height / scale


def trunk_height_parts(keypoints, min_score, aspect_ratio):
    """`trunk_height` before the division: (centre-referenced frame y, torso scale), or (nan, nan).

    The two halves are kept apart so that a *window* of frames can divide by one scale for all of them --
    the median of the window's finite scales -- instead of by each frame's own. Both are legitimate; the
    window-median form is much quieter in the robot's waist-up framing, and the reason is in the
    denominator, not the numerator. With the hips out of frame `torso_scale` falls back to the shoulder
    width, which projects as cos(rotation), so a person merely turning their shoulders makes a per-frame
    ratio oscillate with no vertical motion at all. A body's apparent size barely changes within 3 seconds
    while that rotation error changes a lot, so holding the scale fixed removes the jitter and leaves the
    motion. Measured over cropped 3 s windows, per-frame -> window-median, amplitude p90: NTU
    play-with-phone 0.852 -> 0.066, NTU drink-water 0.616 -> 0.117, UCF101 squats 11.466 -> 6.505, with the
    squat median unmoved (1.684 -> 1.644, -2.4%). The spurious reversals go with it: those same static
    classes drop from a reversals p90 of 25.0 and 19.8 to 0.0 and 0.0, while UCF101 squats settle from a
    median of 6.0 to 3.0 -- which is the number of repetitions actually in 3 seconds.

    The case this could have traded away is a subject who genuinely walks toward or away from the camera:
    their true scale changes within the window, so one median misstates it. Measured on UCF101
    WalkingWithDog it does not bite -- cropped amplitude median 1.271 -> 0.573 and p90 7.173 -> 1.459, i.e.
    it improves, because even there the rotation jitter the median removes is larger than the genuine
    apparent-size change it misstates.
    """
    scale = torso_scale(keypoints, min_score, aspect_ratio)
    if scale is None:
        return float("nan"), float("nan")
    shoulder_mid_y = float(keypoints[LEFT_SHOULDER, 1] + keypoints[RIGHT_SHOULDER, 1]) / 2
    return shoulder_mid_y - FRAME_CENTRE_Y, scale


def normalize_to_torso(keypoints, min_score, aspect_ratio):
    """Frame-normalized (17, 3) keypoints -> x, y in torso lengths from the shoulder midpoint.

    Scale is the distance from the shoulder midpoint to the hip midpoint, which barely changes when the
    person rotates or lies down; the shoulders' width does, and collapses to zero in a side-view push-up.
    When the hips are not confident it falls back to the shoulder width, rescaled by
    `SHOULDER_WIDTH_TO_TORSO_LENGTH` so that the fallback unit is still an (estimated) torso length and
    the two paths stay commensurate. When there is no credible body scale
    at all -- too few confident keypoints, or shoulders collapsed with no confident hips to fall back on --
    returns None rather than inventing a scale: the window/feature pipeline already drops such frames via
    `valid_frac`, and a frame with no scale is better predicted from nothing than from garbage coordinates.
    Returns None without two confident shoulders. Image y points down, so positive y is below the shoulders.
    """
    scale = torso_scale(keypoints, min_score, aspect_ratio)
    if scale is None:
        return None
    xy = keypoints[:, :2] * np.array([aspect_ratio, 1.0], np.float32)
    shoulder_mid = (xy[LEFT_SHOULDER] + xy[RIGHT_SHOULDER]) / 2
    normalized = keypoints.astype(np.float32).copy()
    normalized[:, :2] = (xy - shoulder_mid) / scale
    return normalized
