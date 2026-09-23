"""Features of the most active wrist over a keypoint window (units: shoulder widths)."""

import numpy as np

FEATURE_NAMES = (
    "above_frac", "height_mean", "x_amplitude", "y_amplitude", "reversals",
    "reversal_rate", "speed_mean", "elbow_below_frac", "valid_frac",
)
MIN_SPAN_S = 1.0
MIN_VALID_FRAC = 0.6
MIN_WRIST_FRAMES = 5
REVERSAL_HYSTERESIS = 0.1

# (wrist, same-side shoulder, same-side elbow) COCO indices.
ARMS = ((9, 5, 7), (10, 6, 8))


def count_reversals(values, hysteresis):
    """Direction changes in a 1D signal; a change counts only after moving back more than `hysteresis`."""
    if len(values) < 2:
        return 0
    direction = 0  # +1 rising, -1 falling, 0 not established
    extremum = values[0]
    reversals = 0
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
    """Robust peak-to-peak: the 10th-to-90th percentile span, so one bad keypoint cannot inflate it."""
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def window_features(times, keypoints, min_score=0.5):
    """Features dict (FEATURE_NAMES order) of a window from KeypointWindow.frames(), or None if not enough data."""
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
    x = keypoints[usable, wrist, 0]
    y = keypoints[usable, wrist, 1]
    shoulder_y = keypoints[usable, shoulder, 1]
    elbow_y = keypoints[usable, elbow, 1]
    elbow_score = keypoints[usable, elbow, 2]
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
    return np.array([features[name] for name in FEATURE_NAMES], np.float32)
