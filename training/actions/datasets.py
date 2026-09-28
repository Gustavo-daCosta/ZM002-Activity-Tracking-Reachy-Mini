"""Labelled, grouped action windows from the public skeleton datasets.

Groups keep one person on one side of a fold: NTU encodes the subject in the clip name; UCF101 encodes a
per-class group of clips cut from the same footage, so the class is part of the key.

Negatives are drawn from both sources: squats, push-ups and jumping jacks exist only in handheld UCF101
footage and NTU is a studio camera, so `none` windows from NTU alone would let a classifier recognize
the camera. `max_clips_per_class` caps clips per *original* class so the negatives stay spread.

Provenance, licenses and the class lists: datasets/SOURCES.md
"""

import re

from core.motion.actions import ACTION_WINDOW_S, NONE
from core.motion.actions.features import MIN_SPAN_S, action_features
from core.motion.actions.normalize import normalize_to_torso, trunk_height_parts
from core.pose_backends import (
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE,
)
from training.public_data import clip_windows, load_annotations, ntu_subject

# Crop augmentation: each clip is windowed as published and once per cropped regime (the robot sees a
# waist-up person; the datasets never do). Two cropped levels because "legs" and "waist" take
# different scale paths in `normalize_to_torso`.
CROP_LEVELS = {
    "full": (),
    "legs": (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE),
    "waist": (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE),
}


def crop_scores(annotation, joints):
    """A shallow copy of `annotation` with `joints` made unconfident, coordinates untouched.

    Applied to the raw scores before windowing so the whole pipeline sees a genuinely cropped skeleton,
    the way an out-of-frame joint looks from a pose backend.
    """
    if not joints:
        return annotation
    scores = annotation["keypoint_score"].copy()
    scores[:, :, list(joints)] = 0.0
    return dict(annotation, keypoint_score=scores)


NTU_LABELS = {22: "wave", 9: "clapping"}  # A23 hand waving, A10 clapping (0-based)

# HMDB51 classes are alphabetical; indices verified against the clip names. 38 = situp is deliberately
# not mapped, so a situp called a pushup stays the false positive it is. External test set only.
HMDB_LABELS = {50: "wave", 29: "pushup", 4: "clapping"}
HMDB_SITUP_LABEL = 38

UCF_LABELS = {"PushUps": "pushup", "BodyWeightSquats": "squat", "JumpingJack": "jumping_jacks"}
UCF_NAME = re.compile(r"v_([A-Za-z]+)_(g\d+)_c\d+")


def ntu_group(frame_dir):
    """Group key of an NTU clip ("ntu:<subject>")."""
    return f"ntu:{ntu_subject(frame_dir)}"


def ucf_class(frame_dir):
    """`PushUps` from `v_PushUps_g08_c02`; the whole name when it does not match."""
    match = UCF_NAME.match(frame_dir)
    return match.group(1) if match else frame_dir


def ucf_group(frame_dir):
    """Group key of a UCF101 clip ("ucf:<class>_<group>")."""
    match = UCF_NAME.match(frame_dir)
    return f"ucf:{match.group(1)}_{match.group(2)}" if match else f"ucf:{frame_dir}"


def _action_window(annotation):
    """The 3 s torso-normalized action windows of one clip, at every crop level, tagged with `crop`.

    The emit gate is `MIN_SPAN_S` rather than the wave pipeline's `window_s - step_s`: the stricter gate
    dropped 47% of NTU's short clips.
    """
    out = []
    for level, joints in CROP_LEVELS.items():
        for features in clip_windows(
            crop_scores(annotation, joints), window_s=ACTION_WINDOW_S, features_of=action_features,
            normalizer=normalize_to_torso, extra=trunk_height_parts, min_span_s=MIN_SPAN_S,
        ):
            out.append(dict(features, crop=level))
    return out


def _windows(path, label_of, group_of, class_of, max_clips_per_class):
    """(features, action name, group) for every clip, capping clips per negative source class.

    Positives are never capped: there are far fewer of them than negatives.
    """
    taken, out = {}, []
    for annotation in load_annotations(path):
        action = label_of(annotation)
        if action == NONE and max_clips_per_class is not None:
            if taken.get(class_of(annotation), 0) >= max_clips_per_class:
                continue
        clip = _action_window(annotation)
        if not clip:
            continue
        taken[class_of(annotation)] = taken.get(class_of(annotation), 0) + 1
        group = group_of(annotation["frame_dir"])
        out += [(features, action, group) for features in clip]
    return out


def ntu_windows(path, max_clips_per_class=None):
    """Action windows from ntu60_hrnet.pkl: waving and clapping, everything else `none`."""
    return _windows(
        path,
        label_of=lambda annotation: NTU_LABELS.get(annotation["label"], NONE),
        group_of=ntu_group,
        class_of=lambda annotation: ("ntu", annotation["label"]),
        max_clips_per_class=max_clips_per_class,
    )


def hmdb_windows(path, max_clips_per_class=None):
    """Action windows from hmdb51_2d.pkl: waving, push-ups and clapping, everything else `none`.

    External test set only: movie footage with moving cameras and crowds.
    """
    return _windows(
        path,
        label_of=lambda annotation: HMDB_LABELS.get(annotation["label"], NONE),
        group_of=lambda frame_dir: f"hmdb:{frame_dir}",
        class_of=lambda annotation: ("hmdb", annotation["label"]),
        max_clips_per_class=max_clips_per_class,
    )


def ucf_windows(path, max_clips_per_class=None):
    """Action windows from ucf101_hrnet.pkl: push-ups, squats and jumping jacks, everything else `none`."""
    return _windows(
        path,
        label_of=lambda annotation: UCF_LABELS.get(ucf_class(annotation["frame_dir"]), NONE),
        group_of=ucf_group,
        class_of=lambda annotation: ("ucf", ucf_class(annotation["frame_dir"])),
        max_clips_per_class=max_clips_per_class,
    )
