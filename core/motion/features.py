"""Feature sets computed over a keypoint window.

`window_features` (9 features, wave model): the most active wrist, in shoulder widths.
`action_features` (34 features, action model): whole body, in torso lengths, angles in degrees. Every
statistic is taken over score-masked frames and joints; a feature that could not be measured reports 0.0,
and the `*_valid_frac` features say whether the limb was ever visible.
"""

import numpy as np

from core.pose_backends import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)

FEATURE_NAMES = (
    "above_frac", "height_mean", "x_amplitude", "y_amplitude", "reversals",
    "reversal_rate", "speed_mean", "elbow_below_frac", "valid_frac",
)
MIN_SPAN_S = 1.0
MIN_VALID_FRAC = 0.6
MIN_WRIST_FRAMES = 5
REVERSAL_HYSTERESIS = 0.1
ARMS = ((LEFT_WRIST, LEFT_SHOULDER, LEFT_ELBOW), (RIGHT_WRIST, RIGHT_SHOULDER, RIGHT_ELBOW))

ACTION_MIN_SPAN_S = 2.0
MIN_JOINT_FRAMES = 5
# The wave features embedded as a subset; `valid_frac` is recomputed and `reversals` replaced by rates,
# which do not depend on the window span (NTU clips are shorter than UCF101's).
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
LEG_CHAINS = ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
ARM_CHAINS = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))


def count_reversals(values, hysteresis):
    """Count direction changes in a 1D signal; a change counts only after moving back more than `hysteresis`."""
    if len(values) < 2:
        return 0
    direction, extremum, reversals = 0, values[0], 0
    for value in values[1:]:
        if direction == 0:
            if value - extremum > hysteresis:
                direction, extremum = 1, value
            elif extremum - value > hysteresis:
                direction, extremum = -1, value
        elif direction == 1:
            if value > extremum:
                extremum = value
            elif extremum - value > hysteresis:
                direction, extremum, reversals = -1, value, reversals + 1
        else:
            if value < extremum:
                extremum = value
            elif value - extremum > hysteresis:
                direction, extremum, reversals = 1, value, reversals + 1
    return reversals


def amplitude(values):
    """Robust peak-to-peak: the 10th-to-90th percentile span."""
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def window_features(times, keypoints, min_score=0.5):
    """Compute the wave features of a window.

    Args:
        times: (frames,) timestamps from `KeypointWindow.frames()`.
        keypoints: (frames, 17, 3) shoulder-normalized keypoints, NaN rows for rejected frames.
        min_score: Wrist/elbow confidence threshold.

    Returns:
        Dict in `FEATURE_NAMES` order, or None without enough data.
    """
    if len(times) < 2:
        return None
    span = float(times[-1] - times[0])
    valid = ~np.isnan(keypoints[:, 0, 0])
    valid_frac = float(valid.mean())
    if span < MIN_SPAN_S or valid_frac < MIN_VALID_FRAC:
        return None

    best = None
    for wrist, shoulder, elbow in ARMS:
        usable = valid & (np.nan_to_num(keypoints[:, wrist, 2]) >= min_score)
        if usable.sum() < MIN_WRIST_FRAMES:
            continue
        x_amplitude = amplitude(keypoints[usable, wrist, 0])
        if best is None or x_amplitude > best[0]:
            best = (x_amplitude, wrist, shoulder, elbow, usable)
    if best is None:
        return None

    x_amplitude, wrist, shoulder, elbow, usable = best
    t = times[usable]
    x, y = keypoints[usable, wrist, 0], keypoints[usable, wrist, 1]
    shoulder_y, elbow_y, elbow_score = (keypoints[usable, shoulder, 1], keypoints[usable, elbow, 1],
                                        keypoints[usable, elbow, 2])
    reversals = count_reversals(x, REVERSAL_HYSTERESIS)
    speeds = np.abs(np.diff(x)) / np.maximum(np.diff(t), 1e-6)
    return {
        "above_frac": float(np.mean(y < shoulder_y)),
        "height_mean": float(np.mean(shoulder_y - y)),
        "x_amplitude": x_amplitude,
        "y_amplitude": amplitude(y),
        "reversals": float(reversals),
        "reversal_rate": reversals / span,
        "speed_mean": float(speeds.mean()) if len(speeds) else 0.0,
        "elbow_below_frac": float(np.mean((elbow_score >= min_score) & (elbow_y > y))),
        "valid_frac": valid_frac,
    }


def features_vector(features):
    """The wave feature dict as a float32 vector in `FEATURE_NAMES` order."""
    return np.array([features[name] for name in FEATURE_NAMES], np.float32)


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
        out[f"{prefix}_amplitude"] = out[f"{prefix}_reversal_rate"] = 0.0
        return
    out[f"{prefix}_amplitude"] = amplitude(values)
    out[f"{prefix}_reversal_rate"] = float(count_reversals(values, REVERSAL_HYSTERESIS)) / span


def _angle_at(vertex, first, second):
    """Angle in degrees at `vertex` between the two limb segments, per frame."""
    a, b = first - vertex, second - vertex
    cosine = (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _joint_angles(keypoints, chains, prefix, out, min_score):
    """Store mean, min (pooled over both sides) and amplitude (larger side) of a limb joint's angle."""
    per_side = []
    for proximal, vertex, distal in chains:
        limb = _confident(keypoints, (proximal, vertex, distal), min_score)
        if limb.sum() < MIN_JOINT_FRAMES:
            continue
        per_side.append(_angle_at(keypoints[limb][:, vertex, :2], keypoints[limb][:, proximal, :2],
                                  keypoints[limb][:, distal, :2]))
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
    if len(parts) != len(valid) or parts.ndim != 2 or parts.shape[1] != 2:
        raise ValueError(f"trunk_heights must be (frames, 2) aligned with the window, got {parts.shape}")
    usable = np.isfinite(parts).all(axis=1) & valid
    if not usable.any():
        return np.empty(0)
    return parts[usable, 0] / float(np.median(parts[usable, 1]))


def action_features(times, keypoints, min_score=0.5, trunk_heights=None):
    """Compute the action features of a window.

    Args:
        times: (frames,) timestamps.
        keypoints: (frames, 17, 3) torso-normalized keypoints, NaN rows for rejected frames.
        min_score: Joint confidence threshold.
        trunk_heights: The `KeypointWindow.extras()` channel of `trunk_height_parts`.

    Returns:
        Dict with `ACTION_FEATURE_NAMES` plus a `span` diagnostic (never a feature), or None when the
        window is too short or too empty.
    """
    if len(times) < 2:
        return None
    span = float(times[-1] - times[0])
    valid = ~np.isnan(keypoints[:, 0, 0])
    valid_frac = float(valid.mean())
    if span < ACTION_MIN_SPAN_S or valid_frac < MIN_VALID_FRAC:
        return None
    out = {"valid_frac": valid_frac, "span": span}

    trunk = _confident(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP), min_score)
    if trunk.sum() >= MIN_JOINT_FRAMES:
        dx = (_midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 0) - _midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 0))[trunk]
        dy = (_midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 1) - _midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 1))[trunk]
        angles = np.degrees(np.arctan2(np.abs(dx), np.maximum(np.abs(dy), 1e-6)))  # 0 standing, 90 lying
        out["torso_angle_mean"], out["torso_angle_std"] = float(angles.mean()), float(angles.std())
    else:
        out["torso_angle_mean"] = out["torso_angle_std"] = 0.0

    confident = (np.nan_to_num(keypoints[:, :, 2]) >= min_score) & valid[:, None]
    if confident.any():
        xs, ys = keypoints[:, :, 0][confident], keypoints[:, :, 1][confident]
        out["bbox_aspect"] = float(xs.max() - xs.min()) / max(float(ys.max() - ys.min()), 1e-6)
    else:
        out["bbox_aspect"] = 0.0

    # Vertical oscillation is measured above the ankle midpoint, the only static ground reference.
    ground = _confident(keypoints, (LEFT_ANKLE, RIGHT_ANKLE), min_score)
    ankle_y = _midpoint(keypoints, LEFT_ANKLE, RIGHT_ANKLE, 1)
    shoulders = _confident(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER), min_score) & ground
    _oscillation((ankle_y - _midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 1))[shoulders], "shoulder_y", out, span)
    hips = _confident(keypoints, (LEFT_HIP, RIGHT_HIP), min_score) & ground
    _oscillation((ankle_y - _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 1))[hips], "hip_y", out, span)
    wrists = _confident(keypoints, (LEFT_WRIST, RIGHT_WRIST), min_score)
    _oscillation(_midpoint(keypoints, LEFT_WRIST, RIGHT_WRIST, 1)[wrists], "wrist_y", out, span)
    _oscillation(_trunk_heights(trunk_heights, valid), "trunk_height", out, span)

    _joint_angles(keypoints, LEG_CHAINS, "knee_angle", out, min_score)
    _joint_angles(keypoints, ARM_CHAINS, "elbow_angle", out, min_score)

    if ground.sum() >= MIN_JOINT_FRAMES:
        separation = np.abs(keypoints[ground][:, LEFT_ANKLE, 0] - keypoints[ground][:, RIGHT_ANKLE, 0])
        out["ankle_separation_mean"], out["ankle_separation_amplitude"] = float(separation.mean()), amplitude(separation)
    else:
        out["ankle_separation_mean"] = out["ankle_separation_amplitude"] = 0.0

    head = _confident(keypoints, (NOSE, LEFT_WRIST, RIGHT_WRIST), min_score)
    if head.sum() >= MIN_JOINT_FRAMES:
        nose_y = keypoints[head][:, NOSE, 1]
        above = (keypoints[head][:, LEFT_WRIST, 1] < nose_y) & (keypoints[head][:, RIGHT_WRIST, 1] < nose_y)
        out["wrists_above_head_frac"] = float(above.mean())
    else:
        out["wrists_above_head_frac"] = 0.0

    if wrists.sum() >= MIN_JOINT_FRAMES:
        distance = np.linalg.norm(keypoints[wrists][:, LEFT_WRIST, :2] - keypoints[wrists][:, RIGHT_WRIST, :2], axis=1)
        out["wrist_distance_mean"], out["wrist_distance_min"] = float(distance.mean()), float(distance.min())
        out["wrist_distance_amplitude"] = amplitude(distance)
        out["wrist_distance_reversal_rate"] = float(count_reversals(distance, REVERSAL_HYSTERESIS)) / span
    else:
        for name in ("mean", "min", "amplitude", "reversal_rate"):
            out[f"wrist_distance_{name}"] = 0.0

    # None when no wrist is visible enough; that must not discard the window.
    wave = window_features(times, keypoints, min_score=min_score)
    for name in WAVE_SUBFEATURES:
        out[name] = 0.0 if wave is None else float(wave[name])
    out["upper_body_valid_frac"] = float(_confident(keypoints, UPPER_BODY, min_score).mean())
    out["lower_body_valid_frac"] = float(_confident(keypoints, LOWER_BODY, min_score).mean())
    return out


def action_features_vector(features):
    """The action feature dict as a float32 vector in `ACTION_FEATURE_NAMES` order."""
    return np.array([features[name] for name in ACTION_FEATURE_NAMES], np.float32)
