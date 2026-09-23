"""Labelled, grouped action windows from the public skeleton datasets.

Two sources, two grouping schemes, one namespace: validation must never put the same person on both
sides of a fold. NTU encodes the subject in the clip name; UCF101 encodes a group of clips cut from the
same source footage, and that group id is per class - g08 of PushUps has nothing to do with g08 of
BodyWeightSquats - so the class is part of the key.

**The negatives are deliberately drawn from both sources.** Squats, push-ups and jumping jacks exist only
in UCF101 (handheld YouTube footage) while NTU is a fixed studio camera, so `none` windows taken only from
NTU would let a classifier score well by recognizing the *camera* rather than the action: on static
classes the handheld footage shows a 2-4x heavier motion tail. `ucf_windows` therefore labels every
non-target UCF101 class `none` as well, and `max_clips_per_class` caps clips **per original class**
(the name for UCF101, the integer label for NTU) so those negatives stay spread over many classes instead
of being dominated by whichever ones come first in the file.

Provenance, licenses and the class lists: datasets/SOURCES.md
"""

import re

from core.motion.actions import ACTION_WINDOW_S, NONE
from core.motion.actions.features import MIN_SPAN_S, action_features
from core.motion.actions.normalize import normalize_to_torso, trunk_height_parts
from training.public_data import clip_windows, load_annotations, ntu_subject
from core.pose_backends import (
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE,
)

# Crop augmentation: every clip is windowed once as published and once per cropped regime, because the
# robot's camera sees a waist-up person and NTU and UCF101 never do. Measured over both of our own
# recordings (7225 frames), the confidence pattern is not one regime but two: knees and ankles are
# confident in 0.000 of frames while the hips are confident in 0.19 of them. Those two send
# `normalize_to_torso` down *different* paths -- the hip-to-shoulder distance for the scale in the first,
# the rescaled shoulder-width fallback in the second -- so collapsing them into one would train on only
# half of what the robot produces. Hence three levels, generated from the same clip.
CROP_LEVELS = {
    "full": (),
    "legs": (LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE),
    "waist": (LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE),
}


def crop_scores(annotation, joints):
    """A shallow copy of `annotation` with `joints` made unconfident, leaving the coordinates alone.

    The crop is applied to the **raw scores, before windowing and normalization**, so the whole pipeline
    sees a genuinely cropped skeleton: `torso_scale`'s fallback, `normalize_to_torso`'s rejection rules,
    the window's median scale and every score mask downstream. Doctoring a computed feature vector instead
    would train the forest on vectors no live frame can produce.

    Only confidence is removed, never the coordinates: that is what an out-of-frame joint looks like from a
    pose backend, and `_confident` masks on score, so a kept coordinate cannot leak into any statistic.
    """
    if not joints:
        return annotation
    scores = annotation["keypoint_score"].copy()
    scores[:, :, list(joints)] = 0.0
    return dict(annotation, keypoint_score=scores)


NTU_LABELS = {22: "wave", 9: "clapping"}                  # A23 hand waving, A10 clapping (0-based)

# HMDB51's 51 classes are alphabetical, so the indices are positional. Verified against
# datasets/hmdb51_2d.pkl by reading the clip names, not assumed: 4 = clap (127 clips, e.g.
# "My_Lil__Man_clapping_his_Hands_clap_u_nm_np1_fr_med_1"), 29 = pushup (103), 50 = wave (104).
# 38 = situp (105) is deliberately NOT mapped: we have no such class, so every situp window the model calls
# a pushup stays the false positive it is. HMDB51 is an external test set only and is never trained on.
HMDB_LABELS = {50: "wave", 29: "pushup", 4: "clapping"}
HMDB_SITUP_LABEL = 38

UCF_LABELS = {"PushUps": "pushup", "BodyWeightSquats": "squat", "JumpingJack": "jumping_jacks"}
UCF_NAME = re.compile(r"v_([A-Za-z]+)_(g\d+)_c\d+")


def ntu_group(frame_dir):
    return f"ntu:{ntu_subject(frame_dir)}"


def ucf_class(frame_dir):
    """`PushUps` from `v_PushUps_g08_c02`; the whole name when it does not match."""
    match = UCF_NAME.match(frame_dir)
    return match.group(1) if match else frame_dir


def ucf_group(frame_dir):
    match = UCF_NAME.match(frame_dir)
    return f"ucf:{match.group(1)}_{match.group(2)}" if match else f"ucf:{frame_dir}"


def _action_window(annotation):
    """The 3 s torso-normalized action windows of one clip, at every crop level, trunk-height included.

    One clip yields one set of windows per entry in `CROP_LEVELS`, all carrying the same group, so a
    cropped copy and its full-body original always land on the same side of a fold: they are near
    duplicates, and splitting them would report a score we would not see live.

    The emit gate is `MIN_SPAN_S`, what `action_features` itself accepts, not the wave pipeline's
    `window_s - step_s`: NTU clips are short (median 2.3 s) and the stricter gate silently dropped 47% of
    them, taking the subject diversity the dataset is here for with it. Short clips now yield genuine
    short windows instead of nothing.
    """
    out = []
    for level, joints in CROP_LEVELS.items():
        for features in clip_windows(
            crop_scores(annotation, joints), window_s=ACTION_WINDOW_S, features_of=action_features,
            normalizer=normalize_to_torso, extra=trunk_height_parts, min_span_s=MIN_SPAN_S,
        ):
            # Which regime this window came from: a diagnostic for reporting the two separately, never a
            # column. `lower_body_valid_frac` is the *feature* that tells the forest which regime it is in,
            # and it is a real measurement a live frame can produce; this string is not.
            out.append(dict(features, crop=level))
    return out


def _windows(path, label_of, group_of, class_of, max_clips_per_class):
    """(features, action name, group) for every clip, capping how many clips each negative class gives.

    The cap counts clips per *original* dataset class (`class_of`), not per action: `none` is one action
    but dozens of source classes, and counting them together would let the first few classes in the file
    use up the whole budget. Positives are never capped: there are far fewer of them than negatives, and
    dropping them would trade away the recall we are trying to buy.
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

    External test set only -- never trained on. Movie footage with moving cameras, crowds and heavy
    cropping, so a weak score here is partly the dataset and partly the model, and the report says which
    where it can. The cap counts clips per source class like the other two loaders, so a single populous
    negative class cannot dominate the negatives.
    """
    return _windows(
        path,
        label_of=lambda annotation: HMDB_LABELS.get(annotation["label"], NONE),
        group_of=lambda frame_dir: f"hmdb:{frame_dir}",
        class_of=lambda annotation: ("hmdb", annotation["label"]),
        max_clips_per_class=max_clips_per_class,
    )


def ucf_windows(path, max_clips_per_class=None):
    """Action windows from ucf101_hrnet.pkl: push-ups, squats and jumping jacks, everything else `none`.

    The action comes from the clip name, not from the integer label: the name is readable and survives a
    different label ordering. Every other UCF101 class becomes `none`, which is how the negative set stops
    being a proxy for "studio camera" - see the module docstring.
    """
    return _windows(
        path,
        label_of=lambda annotation: UCF_LABELS.get(ucf_class(annotation["frame_dir"]), NONE),
        group_of=ucf_group,
        class_of=lambda annotation: ("ucf", ucf_class(annotation["frame_dir"])),
        max_clips_per_class=max_clips_per_class,
    )
