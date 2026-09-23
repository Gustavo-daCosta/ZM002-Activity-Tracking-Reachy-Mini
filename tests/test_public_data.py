"""Windows from the public skeleton datasets (NTU RGB+D / HMDB51 pkl files from MMAction2)."""

import pickle

import numpy as np
import pytest

from motion_helpers import motion_frames
from training.public_data import (
    NTU_WAVE_LABEL, ROBOT_FPS, SOURCE_FPS, clip_windows, dataset_windows, ntu_subject,
)

IMG_SHAPE = (1080, 1920)


def annotation(kind, frame_dir="S001C001P001R001A023", label=NTU_WAVE_LABEL, duration_s=4.0, people=1):
    """Build a pkl-style annotation from our synthetic motion helper (pixels, like the real files)."""
    frames = motion_frames(kind, duration_s=duration_s)
    height, width = IMG_SHAPE
    keypoints = np.stack([np.stack([kp[:, 0] * width, kp[:, 1] * height], -1) for _, kp in frames])
    scores = np.stack([kp[:, 2] for _, kp in frames])
    return {
        "frame_dir": frame_dir,
        "label": label,
        "img_shape": IMG_SHAPE,
        "total_frames": len(frames),
        "keypoint": np.repeat(keypoints[None], people, axis=0).astype(np.float32),
        "keypoint_score": np.repeat(scores[None], people, axis=0).astype(np.float32),
    }


def test_ntu_subject_is_parsed_from_the_clip_name():
    assert ntu_subject("S001C001P015R002A023") == "P015"


def test_unknown_naming_falls_back_to_the_clip_name():
    assert ntu_subject("prelinger_LetsBeGo1953_wave_u_cm_np10_ba_med_3").startswith("prelinger")


def test_clip_windows_are_resampled_to_the_robot_rate():
    windows = clip_windows(annotation("wave"), source_fps=SOURCE_FPS, target_fps=ROBOT_FPS)

    assert windows, "a 4 s clip must yield windows"
    assert all(0.55 <= w["valid_frac"] <= 1.0 for w in windows)


def test_waving_clips_look_like_waves_and_still_ones_do_not():
    wave = clip_windows(annotation("wave"), source_fps=SOURCE_FPS, target_fps=ROBOT_FPS)
    still = clip_windows(annotation("still"), source_fps=SOURCE_FPS, target_fps=ROBOT_FPS)

    assert np.median([w["reversals"] for w in wave]) >= 3
    assert np.median([w["reversals"] for w in still]) == 0


def test_short_clips_yield_nothing():
    assert clip_windows(annotation("wave", duration_s=0.5), source_fps=SOURCE_FPS, target_fps=ROBOT_FPS) == []


def test_dataset_windows_labels_and_groups(tmp_path):
    path = tmp_path / "fake.pkl"
    data = {
        "split": {"train": ["a"], "test": ["b"]},
        "annotations": [
            annotation("wave", "S001C001P001R001A023", NTU_WAVE_LABEL),
            annotation("still", "S001C001P002R001A001", 0),
            annotation("hand_on_face", "S001C001P003R001A002", 1),
        ],
    }
    path.write_bytes(pickle.dumps(data))

    windows = dataset_windows(path, wave_label=NTU_WAVE_LABEL)

    labels = {label for _, label, _ in windows}
    groups = {group for _, _, group in windows}
    assert labels == {0, 1}
    assert groups == {"P001", "P002", "P003"}  # grouped by subject, so folds never split a person
    assert all(label == 1 for _, label, group in windows if group == "P001")


def test_clips_per_class_are_capped(tmp_path):
    path = tmp_path / "fake.pkl"
    annotations = [annotation("wave", f"S001C001P{i:03d}R001A023", NTU_WAVE_LABEL) for i in range(5)]
    annotations += [annotation("still", f"S001C001P{i:03d}R001A001", 0) for i in range(5)]
    path.write_bytes(pickle.dumps({"split": {}, "annotations": annotations}))

    windows = dataset_windows(path, wave_label=NTU_WAVE_LABEL, max_clips_per_class=2)

    assert len({group for _, label, group in windows if label == 1}) == 5  # every wave clip is kept
    assert len({group for _, label, group in windows if label == 0}) == 2  # negatives are capped


def test_wave_clips_have_their_own_cap(tmp_path):
    path = tmp_path / "fake.pkl"
    annotations = [annotation("wave", f"S001C001P{i:03d}R001A023", NTU_WAVE_LABEL) for i in range(5)]
    annotations += [annotation("still", f"S001C001P{i:03d}R001A001", 0) for i in range(5)]
    path.write_bytes(pickle.dumps({"split": {}, "annotations": annotations}))

    windows = dataset_windows(path, wave_label=NTU_WAVE_LABEL, max_clips_per_class=4, max_wave_clips=3)

    assert len({group for _, label, group in windows if label == 1}) == 3


def test_missing_person_keypoints_are_skipped(tmp_path):
    empty = annotation("wave")
    empty["keypoint"] = np.zeros((0, 10, 17, 2), np.float32)
    empty["keypoint_score"] = np.zeros((0, 10, 17), np.float32)
    path = tmp_path / "fake.pkl"
    path.write_bytes(pickle.dumps({"split": {}, "annotations": [empty]}))

    assert dataset_windows(path, wave_label=NTU_WAVE_LABEL) == []
