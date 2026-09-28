"""Whole-body features of a window of torso-normalized COCO-17 keypoints.

Lengths are in torso lengths, angles in degrees. Every statistic is taken over score-masked frames and
joints: an unseen joint sits at frame (0, 0) and would otherwise dominate any extremum. A feature that
could not be measured reports 0.0; `upper_body_valid_frac` / `lower_body_valid_frac` say whether the limb
was ever visible, since 0.0 is also a legitimate angle.
"""

import numpy as np

from core.motion.features import (
    FEATURE_NAMES, REVERSAL_HYSTERESIS, amplitude, count_reversals, window_features,
)
from core.pose_backends import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)

MIN_SPAN_S = 2.0
MIN_VALID_FRAC = 0.6
MIN_JOINT_FRAMES = 5

# The wave features embedded as a subset; `valid_frac` is recomputed here and `reversals` is replaced by
# rates, which do not depend on the window span (NTU clips are shorter than UCF101's).
WAVE_SUBFEATURES = tuple(name for name in FEATURE_NAMES if name not in ("valid_frac", "reversals"))

ACTION_FEATURE_NAMES = (
    "torso_angle_mean", "torso_angle_std", "bbox_aspect",
    "shoulder_y_amplitude", "shoulder_y_reversal_rate",
    "hip_y_amplitude", "hip_y_reversal_rate",
    "wrist_y_amplitude", "wrist_y_reversal_rate",
    "trunk_height_amplitude", "trunk_height_reversal_rate",
    "knee_angle_mean", "knee_angle_min", "knee_angle_amplitude",
    "elbow_angle_mean", "elbow_angle_min", "elbow_angle_amplitude",
    "ankle_separation_mean", "ankle_separation_amplitude", "wrists_above_head_frac",
    "wrist_distance_mean", "wrist_distance_min", "wrist_distance_amplitude", "wrist_distance_reversal_rate",
) + WAVE_SUBFEATURES + ("valid_frac", "upper_body_valid_frac", "lower_body_valid_frac")

UPPER_BODY = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST)
LOWER_BODY = (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE)

# (proximal, vertex, distal) per side.
LEG_CHAINS = ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
ARM_CHAINS = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))


def _confident(keypoints, joints, min_score):
    """Mask of frames that were normalized and where every one of `joints` is confident."""
    mask = ~np.isnan(keypoints[:, 0, 0])
    for joint in joints:
        mask &= np.nan_to_num(keypoints[:, joint, 2]) >= min_score
    return mask


def _midpoint(keypoints, left, right, axis):
    return (keypoints[:, left, axis] + keypoints[:, right, axis]) / 2


def _oscillation(values, prefix, out, span):
    """Store `<prefix>_amplitude` and `<prefix>_reversal_rate` of a 1D signal (zeros when too short)."""
    if len(values) < MIN_JOINT_FRAMES:
        out[f"{prefix}_amplitude"] = 0.0
        out[f"{prefix}_reversal_rate"] = 0.0
        return
    out[f"{prefix}_amplitude"] = amplitude(values)
    out[f"{prefix}_reversal_rate"] = float(count_reversals(values, REVERSAL_HYSTERESIS)) / span


def _angle_at(vertex, first, second):
    """Angle in degrees at `vertex` between the two limb segments, per frame."""
    a = first - vertex
    b = second - vertex
    cosine = (a * b).sum(axis=1) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-6
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _joint_angles(keypoints, chains, prefix, out, min_score):
    """Store mean, min and amplitude of a limb joint's angle over both sides.

    Mean and min pool both sides; the amplitude is the larger per-side value, because pooling reads a
    static left/right difference as motion and dilutes a one-sided movement.
    """
    per_side = []
    for proximal, vertex, distal in chains:
        limb = _confident(keypoints, (proximal, vertex, distal), min_score)
        if limb.sum() < MIN_JOINT_FRAMES:
            continue
        per_side.append(
            _angle_at(keypoints[limb][:, vertex, :2], keypoints[limb][:, proximal, :2],
                      keypoints[limb][:, distal, :2])
        )
    if not per_side:
        out[f"{prefix}_mean"] = out[f"{prefix}_min"] = out[f"{prefix}_amplitude"] = 0.0
        return
    pooled = np.concatenate(per_side)
    out[f"{prefix}_mean"] = float(pooled.mean())
    out[f"{prefix}_min"] = float(pooled.min())
    out[f"{prefix}_amplitude"] = max(amplitude(side) for side in per_side)


def _trunk_heights(parts, valid):
    """Trunk heights over the window's median scale, dropping frames without a scale.

    Args:
        parts: (frames, 2) of (centre-referenced y, torso scale) from `trunk_height_parts`, or None.
        valid: Per-frame normalization mask.

    Raises:
        ValueError: `parts` does not match the window's frames.
    """
    if parts is None:
        return np.empty(0)
    parts = np.asarray(parts, np.float64)
    if len(parts) != len(valid):
        raise ValueError(
            f"trunk_heights has {len(parts)} frames but the window has {len(valid)}: "
            "the channel does not belong to these frames"
        )
    if parts.ndim != 2 or parts.shape[1] != 2:
        raise ValueError(
            f"trunk_heights must be (frames, 2) from normalize.trunk_height_parts, got {parts.shape}"
        )
    usable = np.isfinite(parts).all(axis=1) & valid
    if not usable.any():
        return np.empty(0)
    scale = float(np.median(parts[usable, 1]))
    return parts[usable, 0] / scale


def action_features(times, keypoints, min_score=0.5, trunk_heights=None):
    """Compute the action features of a window.

    Args:
        times: (frames,) timestamps.
        keypoints: (frames, 17, 3) torso-normalized keypoints, NaN rows for rejected frames.
        min_score: Joint confidence threshold.
        trunk_heights: The `KeypointWindow.extras()` channel of `trunk_height_parts`; the only squat
            signal that survives a waist-up framing.

    Returns:
        Dict with `ACTION_FEATURE_NAMES` plus a `span` diagnostic (never a feature: it correlates with the
        dataset), or None when the window is too short or too empty.
    """
    if len(times) < 2:
        return None
    span = float(times[-1] - times[0])
    valid = ~np.isnan(keypoints[:, 0, 0])
    valid_frac = float(valid.mean())
    if span < MIN_SPAN_S or valid_frac < MIN_VALID_FRAC:
        return None

    out = {"valid_frac": valid_frac, "span": span}

    trunk = _confident(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP), min_score)
    if trunk.sum() >= MIN_JOINT_FRAMES:
        dx = _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 0)[trunk] - _midpoint(
            keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 0)[trunk]
        dy = _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 1)[trunk] - _midpoint(
            keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 1)[trunk]
        # Angle from the vertical: 0 standing, 90 lying down.
        angles = np.degrees(np.arctan2(np.abs(dx), np.maximum(np.abs(dy), 1e-6)))
        out["torso_angle_mean"] = float(angles.mean())
        out["torso_angle_std"] = float(angles.std())
    else:
        out["torso_angle_mean"] = out["torso_angle_std"] = 0.0

    confident = np.nan_to_num(keypoints[:, :, 2]) >= min_score
    confident &= valid[:, None]
    if confident.any():
        xs = keypoints[:, :, 0][confident]
        ys = keypoints[:, :, 1][confident]
        height = max(float(ys.max() - ys.min()), 1e-6)
        out["bbox_aspect"] = float(xs.max() - xs.min()) / height
    else:
        out["bbox_aspect"] = 0.0

    # Vertical oscillation is measured above the ankle midpoint: the shoulder origin makes shoulder y
    # identically 0 and hip y a function of tilt, so the ankles are the only static ground reference.
    ground = _confident(keypoints, (LEFT_ANKLE, RIGHT_ANKLE), min_score)
    ankle_y = _midpoint(keypoints, LEFT_ANKLE, RIGHT_ANKLE, 1)
    shoulders = _confident(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER), min_score) & ground
    _oscillation((ankle_y - _midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 1))[shoulders],
                 "shoulder_y", out, span)
    hips = _confident(keypoints, (LEFT_HIP, RIGHT_HIP), min_score) & ground
    _oscillation((ankle_y - _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 1))[hips], "hip_y", out, span)
    wrists = _confident(keypoints, (LEFT_WRIST, RIGHT_WRIST), min_score)
    _oscillation(_midpoint(keypoints, LEFT_WRIST, RIGHT_WRIST, 1)[wrists], "wrist_y", out, span)
    _oscillation(_trunk_heights(trunk_heights, valid), "trunk_height", out, span)

    _joint_angles(keypoints, LEG_CHAINS, "knee_angle", out, min_score)
    _joint_angles(keypoints, ARM_CHAINS, "elbow_angle", out, min_score)

    ankles = _confident(keypoints, (LEFT_ANKLE, RIGHT_ANKLE), min_score)
    if ankles.sum() >= MIN_JOINT_FRAMES:
        separation = np.abs(keypoints[ankles][:, LEFT_ANKLE, 0] - keypoints[ankles][:, RIGHT_ANKLE, 0])
        out["ankle_separation_mean"] = float(separation.mean())
        out["ankle_separation_amplitude"] = amplitude(separation)
    else:
        out["ankle_separation_mean"] = out["ankle_separation_amplitude"] = 0.0

    head = _confident(keypoints, (NOSE, LEFT_WRIST, RIGHT_WRIST), min_score)
    if head.sum() >= MIN_JOINT_FRAMES:
        nose_y = keypoints[head][:, NOSE, 1]
        above = ((keypoints[head][:, LEFT_WRIST, 1] < nose_y)
                 & (keypoints[head][:, RIGHT_WRIST, 1] < nose_y))
        out["wrists_above_head_frac"] = float(above.mean())
    else:
        out["wrists_above_head_frac"] = 0.0

    if wrists.sum() >= MIN_JOINT_FRAMES:
        left = keypoints[wrists][:, LEFT_WRIST, :2]
        right = keypoints[wrists][:, RIGHT_WRIST, :2]
        distance = np.linalg.norm(left - right, axis=1)
        out["wrist_distance_mean"] = float(distance.mean())
        out["wrist_distance_min"] = float(distance.min())
        out["wrist_distance_amplitude"] = amplitude(distance)
        out["wrist_distance_reversal_rate"] = float(
            count_reversals(distance, REVERSAL_HYSTERESIS)) / span
    else:
        for name in ("mean", "min", "amplitude", "reversal_rate"):
            out[f"wrist_distance_{name}"] = 0.0

    # `window_features` is None when no wrist is visible enough; that must not discard the window.
    wave = window_features(times, keypoints, min_score=min_score)
    for name in WAVE_SUBFEATURES:
        out[name] = 0.0 if wave is None else float(wave[name])

    out["upper_body_valid_frac"] = float(_confident(keypoints, UPPER_BODY, min_score).mean())
    out["lower_body_valid_frac"] = float(_confident(keypoints, LOWER_BODY, min_score).mean())
    return out


def action_features_vector(features):
    """The feature dict as a float32 vector in `ACTION_FEATURE_NAMES` order."""
    return np.array([features[name] for name in ACTION_FEATURE_NAMES], np.float32)
