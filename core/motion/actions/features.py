"""Whole-body features of a window of torso-normalized COCO-17 keypoints.

The wave model looks at one wrist. Push-ups, squats, clapping and jumping jacks need the trunk and the
legs, so this adds trunk orientation, vertical oscillation of the shoulders/hips/wrists, knee and elbow
angles, ankle separation and wrist separation, and embeds the wave features as a subset.

The knee angle carries the squat's repetition and the elbow angle the push-up's: without it nothing here
separates a push-up from a held plank, since both are a horizontal trunk and only the elbow cycles.

All lengths are in torso lengths (see actions/normalize.py); all angles are in degrees. Joints that are
not confident contribute nothing, and how much of the body was actually visible is itself a feature -
on the robot the legs are usually outside the frame, and the classifier has to be able to answer
"nothing" instead of inferring a squat from half a skeleton.

A feature that could not be measured reports 0.0. That is *not* a unique sentinel, and earlier versions of
this docstring wrongly claimed it was: on real NTU-60 skeletons `elbow_angle_min` is exactly 0.0000 in 9 of
103 windows, a fully folded arm or a coincident elbow and wrist keypoint. `upper_body_valid_frac` and
`lower_body_valid_frac` are what disambiguate it - they say whether the limb pair was ever confident at
all - so any model reading a 0.0 angle must read the matching validity fraction with it.

Every amplitude, extremum and bounding box below is taken over score-masked frames and joints, never over
the raw array. A joint the backend could not see is reported at frame coordinate (0, 0), which normalizes
to -shoulder_mid / scale: measured 3.0012 over all 17 rows of a standing skeleton against 2.4160 over the
confident rows only, so an unmasked maximum measures a detector artifact rather than a body part. A frame
that `normalize_to_torso` rejected is a NaN row, and `_confident` folds that per-frame validity into every
per-joint mask so a rejected frame cannot contribute to any statistic either.
"""

import numpy as np

from core.motion.features import (
    FEATURE_NAMES, REVERSAL_HYSTERESIS, amplitude, count_reversals, window_features,
)
from core.pose_backends import (
    LEFT_ANKLE, LEFT_ELBOW, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST, NOSE,
    RIGHT_ANKLE, RIGHT_ELBOW, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
)

MIN_SPAN_S = 2.0        # a push-up or squat repetition needs at least this much of the 3 s window
MIN_VALID_FRAC = 0.6
MIN_JOINT_FRAMES = 5

# The wave features, computed by core.motion.features and embedded here as a subset. valid_frac is
# excluded because this module computes its own, and the raw `reversals` count because this module states
# every oscillation as a rate per second (see below): divided by the span it would be `reversal_rate`
# twice over, and core.motion.features already publishes that column.
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

# (proximal, vertex, distal) per side. Note the order differs from core.motion.features.ARMS, which
# is (wrist, shoulder, elbow) and serves a different purpose.
LEG_CHAINS = ((LEFT_HIP, LEFT_KNEE, LEFT_ANKLE), (RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE))
ARM_CHAINS = ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST))


def _confident(keypoints, joints, min_score):
    """Boolean mask of frames that were normalized and where every one of `joints` is confident."""
    mask = ~np.isnan(keypoints[:, 0, 0])
    for joint in joints:
        mask &= np.nan_to_num(keypoints[:, joint, 2]) >= min_score
    return mask


def _midpoint(keypoints, left, right, axis):
    return (keypoints[:, left, axis] + keypoints[:, right, axis]) / 2


def _oscillation(values, prefix, out, span):
    """Amplitude and reversal *rate* of a 1D signal, or zeros when there is not enough of it.

    The rate, not the count, because emitted windows do not all span the same time: a clip shorter than
    `ACTION_WINDOW_S` yields a shorter window (see `public_data.clip_windows`), and NTU clips are much
    shorter than UCF101's, so a count would encode which dataset -- and therefore which class -- a window
    came from. Measured on NTU wave, `wrist_y_reversals` had a median of 1.0 in 2.0 s windows against 4.0
    in 3.0 s ones, corr(span) = +0.31, while a rate is span-free. It is also the better quantity on its
    own: a wave is how fast the hand oscillates, not how many reversals happened to fit in our window.

    `span` is the whole window's span, matching how `core.motion.features` computes `reversal_rate`
    -- the reversals are counted over the score-masked frames but divided by the window's duration, so a
    partly occluded signal reports a lower rate rather than a spuriously high one.
    """
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
    """Mean, minimum and amplitude of a limb joint's angle over both sides.

    A side contributes only in the frames where all three of its joints are confident and the frame was
    normalized; a side with too few such frames contributes nothing at all. With neither side measurable
    all three features are 0.0 - see the module docstring on why that is not a unique sentinel.

    Mean and minimum pool both sides. The amplitude must NOT, for two independent reasons.

    First, a percentile span over the two sides' concatenated series reads a *static* left/right difference
    as motion - a false repetition signal in the two features that exist for repetition. Measured on a
    completely motionless skeleton, pooling gave 90.0 for one forearm folded to 90 degrees and 52.6 for one
    knee displaced outwards: the first larger than the waving fixture's 42.4, the second inside squat
    territory (66.2). Real people stand asymmetrically and one-sided occlusion biases one limb
    systematically, so this is the ordinary case, not a contrived one.

    Second, and in the opposite direction, pooling *diluted* genuine one-sided motion: the still limb's
    constant angle flattened the percentile span of the limb that was actually moving. The one-armed waving
    fixture reads 53.8 per side against 42.4 pooled, so the pooled form was weakening the wave signal at the
    same time as it was faking the squat one.

    Hence: amplitude per side, and the larger taken. The max rather than the mean of the two sides is what
    keeps a one-armed wave visible, and both static cases above drop to 0.0.
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
    """The window's trunk heights, in torso lengths: each frame's y over the window's *median* scale.

    `parts` is a (frames, 2) array of (centre-referenced frame y, torso scale) from
    `normalize.trunk_height_parts`, or None. One scale for the whole window rather than each frame's own:
    within 3 seconds a body's apparent size barely changes, while the shoulder-width fallback's
    cos(rotation) error changes a lot, so a per-frame denominator injects a rotation jitter that reads as
    vertical motion. See `normalize.trunk_height_parts` for the measured effect. The median, not the mean,
    so one badly scaled frame cannot move the denominator.

    Frames with no computable scale contribute nothing: they are nan on both halves, so they are excluded
    from the median and dropped from the returned series. `valid` (the per-frame normalization mask) is
    folded in as well -- redundant by construction, since a rejected frame is already nan here, and kept so
    the masking cannot drift if either side changes.
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
    """Feature dict in ACTION_FEATURE_NAMES order, or None when the window is too short or too empty.

    `trunk_heights` is the optional `KeypointWindow.extras()` channel of `normalize.trunk_height_parts`:
    a (frames, 2) array of (centre-referenced frame y, torso scale), aligned with `keypoints`. It is the
    only squat signal that survives the robot's waist-up framing; without it `trunk_height_amplitude` and
    `trunk_height_reversal_rate` are 0.0 like any other unmeasurable feature. A channel whose length does not
    match `keypoints` raises `ValueError`: that is a caller bug, not missing data, and reporting it as the
    0.0 sentinel would make it indistinguishable from "the legs are not visible" and could quietly poison
    a whole training run.
    """
    if len(times) < 2:
        return None
    span = float(times[-1] - times[0])
    valid = ~np.isnan(keypoints[:, 0, 0])
    valid_frac = float(valid.mean())
    if span < MIN_SPAN_S or valid_frac < MIN_VALID_FRAC:
        return None

    # `span` is a diagnostic that rides along with every window, NOT a feature: it is absent from
    # ACTION_FEATURE_NAMES and so never reaches `action_features_vector`. It is here because span
    # correlates with both the source and the class -- NTU clips are short, UCF101 clips are long -- so a
    # span column would hand a classifier a "short window => NTU => wave or clapping" shortcut that still
    # generalizes within a dataset and inflates the score. Training reports its correlation with the
    # predictions instead, which is how that shortcut would be caught if the features leaked it anyway.
    out = {"valid_frac": valid_frac, "span": span}

    # --- trunk orientation and overall shape -------------------------------------------------------
    trunk = _confident(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP), min_score)
    if trunk.sum() >= MIN_JOINT_FRAMES:
        dx = _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 0)[trunk] - _midpoint(
            keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 0)[trunk]
        dy = _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 1)[trunk] - _midpoint(
            keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 1)[trunk]
        # Angle from the vertical: 0 standing, 90 lying down. abs(dy) so upside down still reads upright.
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

    # --- vertical oscillation ----------------------------------------------------------------------
    # Measured as height above the ankle midpoint, not above the shoulder midpoint. Normalization puts the
    # origin on the shoulder midpoint and makes the shoulder-to-hip distance exactly 1, so in normalized
    # coordinates the shoulder midpoint's y is identically 0 and the hip midpoint's y is identically
    # cos(trunk tilt): measured spans of 0.0000 over a 0.4-torso squat and over a bobbing push-up, i.e.
    # both signals are blind to the very oscillation they are named for, and the hip one only restates
    # torso_angle. The ankles are the one static ground reference a normalized skeleton has - feet stay
    # planted through a squat while the trunk drops - so the signal is (ankle_y - joint_y), positive up.
    # With no confident ankles there is no ground reference and these report 0.0; that is the robot's usual
    # framing, where the legs are out of frame and a squat is not visible at all.
    ground = _confident(keypoints, (LEFT_ANKLE, RIGHT_ANKLE), min_score)
    ankle_y = _midpoint(keypoints, LEFT_ANKLE, RIGHT_ANKLE, 1)
    shoulders = _confident(keypoints, (LEFT_SHOULDER, RIGHT_SHOULDER), min_score) & ground
    _oscillation((ankle_y - _midpoint(keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, 1))[shoulders],
                 "shoulder_y", out, span)
    hips = _confident(keypoints, (LEFT_HIP, RIGHT_HIP), min_score) & ground
    _oscillation((ankle_y - _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, 1))[hips], "hip_y", out, span)
    wrists = _confident(keypoints, (LEFT_WRIST, RIGHT_WRIST), min_score)
    _oscillation(_midpoint(keypoints, LEFT_WRIST, RIGHT_WRIST, 1)[wrists], "wrist_y", out, span)

    # The trunk's height in the *camera frame*, in torso lengths -- the one squat signal that survives a
    # waist-up crop, because it does not need the legs and is not destroyed by the shoulder-origin
    # normalization. Only its oscillation is used: the absolute height says where the person stands in the
    # frame, not what they are doing, and a mean would let the forest learn the camera's geometry. See
    # actions/normalize.py on why camera motion is the limitation of this feature.
    _oscillation(_trunk_heights(trunk_heights, valid), "trunk_height", out, span)

    # --- joint angles: the squat (knees) and the push-up (elbows) ----------------------------------
    _joint_angles(keypoints, LEG_CHAINS, "knee_angle", out, min_score)
    # The elbow is what tells a push-up repetition from a held plank: the trunk is horizontal in both, so
    # torso_angle and bbox_aspect (static shape) cannot separate them, and the ankle-referenced trunk
    # heights are the 0.0 sentinel whenever the legs are out of frame - the robot's usual desk framing.
    # The elbow is visible from a waist-up crop, so it survives that framing.
    _joint_angles(keypoints, ARM_CHAINS, "elbow_angle", out, min_score)

    # --- ankles and wrists above the head: the jumping jack ----------------------------------------
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

    # --- wrist separation: the clap ----------------------------------------------------------------
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

    # --- the wave features, as a subset ------------------------------------------------------------
    wave = window_features(times, keypoints, min_score=min_score)
    for name in WAVE_SUBFEATURES:
        # window_features returns None when no arm has enough confident wrist frames - a squat with the
        # hands hidden, for instance. That must not discard the whole window.
        out[name] = 0.0 if wave is None else float(wave[name])

    # --- how much of the body we actually saw ------------------------------------------------------
    # These are what disambiguate a 0.0 angle: a limb pair that was never confident against one that was
    # measured and happened to read 0. Both limb pairs need one - the elbow triple's sentinel would
    # otherwise be undiscountable, while the knees already had `lower_body_valid_frac`.
    out["upper_body_valid_frac"] = float(_confident(keypoints, UPPER_BODY, min_score).mean())
    out["lower_body_valid_frac"] = float(_confident(keypoints, LOWER_BODY, min_score).mean())
    return out


def action_features_vector(features):
    return np.array([features[name] for name in ACTION_FEATURE_NAMES], np.float32)
