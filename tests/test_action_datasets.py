"""Dataset readers: label maps, group keys, and the caps that keep negatives balanced."""

import pickle

import numpy as np

from training.actions.datasets import (
    CROP_LEVELS, HMDB_LABELS, NTU_LABELS, UCF_LABELS, crop_scores, hmdb_windows, ntu_group, ntu_windows,
    ucf_class, ucf_group, ucf_windows,
)


def fake_clip(frame_dir, label, frames=90, height=480, width=640, moving=True, bob=0.0):
    """One annotation in the PySKL pickle format: a standing person, optionally waving one arm.

    `bob` moves the whole body up and down in pixels, which is what the trunk-height channel measures.
    """
    keypoints = np.zeros((1, frames, 17, 2), np.float32)
    base = {0: (320, 110), 5: (290, 150), 6: (350, 150), 7: (280, 210), 8: (360, 210),
            9: (275, 270), 10: (365, 270), 11: (295, 300), 12: (345, 300),
            13: (295, 375), 14: (345, 375), 15: (295, 450), 16: (345, 450)}
    phase = np.sin(np.linspace(0, 12 * np.pi, frames))
    for joint, (x, y) in base.items():
        keypoints[0, :, joint] = (x, y)
        if bob:
            keypoints[0, :, joint, 1] = y + bob * phase
    if moving:
        keypoints[0, :, 10, 0] = 365 + 40 * phase
        keypoints[0, :, 10, 1] = 120
    return {
        "frame_dir": frame_dir, "label": label, "img_shape": (height, width),
        "total_frames": frames,
        "keypoint": keypoints, "keypoint_score": np.full((1, frames, 17), 0.9, np.float32),
    }


def write_pickle(path, clips):
    with open(path, "wb") as handle:
        pickle.dump({"annotations": clips, "split": {}}, handle)
    return path


def test_ucf_class_and_group_come_from_the_clip_name():
    assert ucf_class("v_PushUps_g08_c02") == "PushUps"
    assert ucf_class("v_BodyWeightSquats_g11_c01") == "BodyWeightSquats"
    # The group id is per class in UCF101: g08 of PushUps is not the same people as g08 of squats.
    assert ucf_group("v_PushUps_g08_c02") == "ucf:PushUps_g08"
    assert ucf_group("v_BodyWeightSquats_g08_c01") == "ucf:BodyWeightSquats_g08"
    assert ucf_group("v_PushUps_g08_c02") != ucf_group("v_BodyWeightSquats_g08_c01")


def test_ntu_group_is_the_subject():
    assert ntu_group("S001C002P003R002A023") == "ntu:P003"
    assert ntu_group("S014C001P003R001A010") == "ntu:P003"   # same subject, other setup


def test_label_maps_only_name_actions_we_agreed_on():
    from core.motion.actions import ACTIONS

    assert set(NTU_LABELS.values()) == {"wave", "clapping"}
    assert set(UCF_LABELS.values()) == {"pushup", "squat", "jumping_jacks"}
    assert set(NTU_LABELS.values()) | set(UCF_LABELS.values()) | {"none"} == set(ACTIONS)


def test_ntu_windows_label_the_wave_class_and_call_everything_else_none(tmp_path):
    path = write_pickle(tmp_path / "ntu.pkl", [
        fake_clip("S001C001P001R001A023", 22),   # wave
        fake_clip("S001C001P002R001A010", 9),    # clapping
        fake_clip("S001C001P003R001A001", 0),    # drink water -> none
    ])
    windows = ntu_windows(path)
    labels = {label for _, label, _ in windows}
    assert labels == {"wave", "clapping", "none"}
    assert {group for _, _, group in windows} == {"ntu:P001", "ntu:P002", "ntu:P003"}
    assert all(isinstance(features, dict) for features, _, _ in windows)


def test_ucf_windows_use_the_name_not_the_integer_label(tmp_path):
    # Deliberately wrong integer labels: the reader must ignore them and trust the clip name.
    path = write_pickle(tmp_path / "ucf.pkl", [
        fake_clip("v_PushUps_g08_c01", 999),
        fake_clip("v_Basketball_g02_c01", 999),
    ])
    windows = ucf_windows(path)
    by_group = {group: label for _, label, group in windows}
    assert by_group["ucf:PushUps_g08"] == "pushup"
    assert by_group["ucf:Basketball_g02"] == "none"


def test_max_clips_per_class_caps_the_negatives_only(tmp_path):
    clips = [fake_clip(f"S001C001P00{i}R001A001", 0) for i in range(1, 6)]
    clips += [fake_clip(f"S001C001P01{i}R001A023", 22) for i in range(1, 6)]
    path = write_pickle(tmp_path / "ntu.pkl", clips)
    windows = ntu_windows(path, max_clips_per_class=2)
    groups_by_label = {}
    for _, label, group in windows:
        groups_by_label.setdefault(label, set()).add(group)
    assert len(groups_by_label["none"]) == 2      # capped
    assert len(groups_by_label["wave"]) == 5      # positives are kept


def test_ucf_negatives_are_capped_per_original_class_not_in_bulk(tmp_path):
    """UCF101 integer labels are ignored, so the cap must key on the class in the clip name.

    Otherwise the negatives collapse onto whichever class comes first in the file, and `none` becomes a
    proxy for that one class instead of a spread of static UCF101 footage.
    """
    clips = [fake_clip(f"v_PlayingGuitar_g0{i}_c01", 999) for i in range(1, 4)]
    clips += [fake_clip(f"v_ApplyEyeMakeup_g0{i}_c01", 999) for i in range(1, 4)]
    path = write_pickle(tmp_path / "ucf.pkl", clips)
    groups = {group for _, label, group in ucf_windows(path, max_clips_per_class=1)
              if label == "none"}
    assert groups == {"ucf:PlayingGuitar_g01", "ucf:ApplyEyeMakeup_g01"}


def full_body(windows):
    """Only the uncropped windows of a clip: one clip now yields one set per `CROP_LEVELS` entry."""
    return [triple for triple in windows if triple[0]["crop"] == "full"]


def test_features_have_the_action_feature_names(tmp_path):
    from core.motion.actions.features import ACTION_FEATURE_NAMES

    path = write_pickle(tmp_path / "ntu.pkl", [fake_clip("S001C001P001R001A023", 22)])
    features, _, _ = ntu_windows(path)[0]
    # `span` and `crop` ride along for diagnostics and are deliberately not columns: see
    # core/motion/actions/features.py and datasets.CROP_LEVELS.
    assert set(features) == set(ACTION_FEATURE_NAMES) | {"span", "crop"}
    assert "span" not in ACTION_FEATURE_NAMES and "crop" not in ACTION_FEATURE_NAMES


def test_windows_carry_the_trunk_height_channel(tmp_path):
    """The two trunk-height features are the only squat signal; they need the extra channel wired in.

    A window built without it silently reports 0.0 for both, which is indistinguishable from a person
    standing still - so a bobbing body must come out with a non-zero amplitude.
    """
    still = write_pickle(tmp_path / "still.pkl", [fake_clip("v_PlayingGuitar_g01_c01", 0, moving=False)])
    bobbing = write_pickle(tmp_path / "bob.pkl",
                           [fake_clip("v_BodyWeightSquats_g01_c01", 0, moving=False, bob=60.0)])
    still_features = full_body(ucf_windows(still))[0][0]
    bob_features = full_body(ucf_windows(bobbing))[0][0]
    assert still_features["trunk_height_amplitude"] == 0.0 or \
        still_features["trunk_height_amplitude"] < 0.05
    assert bob_features["trunk_height_amplitude"] > 0.5
    assert bob_features["trunk_height_reversal_rate"] > 0


def test_windows_span_the_action_window_not_the_wave_window(tmp_path):
    """3 s windows, emitted every STEP_S once the window holds MIN_SPAN_S of motion."""
    from core.motion.actions import ACTION_WINDOW_S
    from core.motion.actions.features import MIN_SPAN_S
    from training.public_data import STEP_S, clip_windows

    clip = fake_clip("S001C001P001R001A023", 22, frames=90)
    assert ACTION_WINDOW_S == 3.0 and STEP_S == 0.5
    assert len(clip_windows(clip)) > 1                     # the untouched 1.5 s wave default

    longer = write_pickle(tmp_path / "long.pkl", [fake_clip("S001C001P001R001A023", 22, frames=180)])
    # Per crop level: emits at t = 2.0, 2.5, ... 5.5 within 5.9 s.
    assert len(full_body(ntu_windows(longer))) == 8
    assert len(ntu_windows(longer)) == 8 * len(CROP_LEVELS)


def test_short_clips_still_yield_a_window(tmp_path):
    """NTU clips are short (median 2.3 s); the emit gate must be MIN_SPAN_S, not window_s - step_s.

    A gate at 2.5 s discarded 47% of NTU and halved the number of subjects, although `action_features`
    accepts anything from MIN_SPAN_S up. A 2.2 s clip must give a genuine 2.2 s window.
    """
    from core.motion.actions.features import MIN_SPAN_S
    from training.public_data import STEP_S

    frames = 66                                            # 2.2 s at 30 FPS, under window_s - step_s
    path = write_pickle(tmp_path / "short.pkl", [fake_clip("S001C001P001R001A023", 22, frames=frames)])
    assert MIN_SPAN_S <= 2.1 < (3.0 - STEP_S)
    assert len(full_body(ntu_windows(path))) == 1


def test_the_wave_pipeline_keeps_the_stricter_gate():
    """`clip_windows` with no `min_span_s` must gate at `window_s - step_s`, exactly as before."""
    from training.public_data import clip_windows

    short = fake_clip("S001C001P001R001A023", 22, frames=30)   # 0.9 s of samples, under 1.5 - 0.5
    assert clip_windows(short) == []
    assert len(clip_windows(fake_clip("x", 22, frames=36))) == 1   # 1.1 s, over the gate
    # The parameter only moves that gate; raising it emits later and so emits fewer windows.
    long_clip = fake_clip("S001C001P001R001A023", 22, frames=90)
    assert len(clip_windows(long_clip, min_span_s=2.5)) < len(clip_windows(long_clip))


# --- HMDB51: external test set only, never trained on -----------------------------------------------


def test_hmdb_labels_name_only_actions_we_have():
    from core.motion.actions import ACTIONS

    assert set(HMDB_LABELS.values()) <= set(ACTIONS)
    # Verified against datasets/hmdb51_2d.pkl: 104 wave clips, 103 pushup, 127 clap.
    assert HMDB_LABELS == {50: "wave", 29: "pushup", 4: "clapping"}


def test_hmdb_windows_label_the_mapped_classes_and_call_everything_else_none(tmp_path):
    clips = [fake_clip("AmericanGangster_wave_u_cm_np1_fr_med_57", 50),
             fake_clip("Pushups_Workout_pushup_f_cm_np1_ri_bad_2", 29),
             fake_clip("Abs__Situps__Crunches_situp_u_cm_np1_ri_goo_1", 38)]
    path = write_pickle(tmp_path / "hmdb.pkl", clips)
    windows = hmdb_windows(path)

    assert {label for _, label, _ in windows} == {"wave", "pushup", "none"}
    # Situps stay `none` on purpose: the model has no such class, so every situp window it calls a
    # pushup must be counted as the false positive it is.
    situps = {label for _, label, group in windows if "situp" in group}
    assert situps == {"none"}
    assert all(group.startswith("hmdb:") for _, _, group in windows)


def test_hmdb_cap_counts_clips_per_source_class(tmp_path):
    clips = [fake_clip(f"movie{i}_situp_u_cm_np1_ri_goo_{i}", 38) for i in range(3)]
    clips += [fake_clip(f"movie{i}_shoot_gun_u_cm_np1_ri_goo_{i}", 40) for i in range(3)]
    path = write_pickle(tmp_path / "hmdb.pkl", clips)
    groups = {group for _, _, group in hmdb_windows(path, max_clips_per_class=1)}
    assert len(groups) == 2   # one clip of each of the two negative source classes


# --- crop augmentation: the robot sees a waist-up person, the datasets never do ----------------------


def test_crop_levels_reproduce_the_measured_recordings():
    """The levels are measured from our own recordings, not guessed.

    Over both sessions (7225 frames): knees and ankles are confident in 0.000 of frames, hips in 0.19.
    So the real pattern spans two regimes, not one -- legs gone with the hips still visible, and hips gone
    as well -- and both are generated, because they send `normalize_to_torso` down different paths: the
    hip-to-shoulder scale in the first and the shoulder-width fallback in the second.
    """
    assert CROP_LEVELS["full"] == ()
    assert set(CROP_LEVELS["legs"]) == {13, 14, 15, 16}            # knees and ankles
    assert set(CROP_LEVELS["waist"]) == {11, 12, 13, 14, 15, 16}   # the hips go too
    assert list(CROP_LEVELS) == ["full", "legs", "waist"]


def test_crop_scores_zeroes_only_the_named_joints_and_does_not_mutate():
    clip = fake_clip("S001C001P001R001A023", 22)
    before = clip["keypoint_score"].copy()
    cropped = crop_scores(clip, CROP_LEVELS["waist"])

    assert np.array_equal(clip["keypoint_score"], before), "the original annotation must be untouched"
    assert (cropped["keypoint_score"][:, :, 11:17] == 0.0).all()
    assert (cropped["keypoint_score"][:, :, :11] == before[:, :, :11]).all()
    # The keypoints themselves are never moved: only confidence is taken away, as a real crop does.
    assert cropped["keypoint"] is clip["keypoint"]


def test_windows_come_at_every_crop_level_and_say_which(tmp_path):
    path = write_pickle(tmp_path / "ntu.pkl", [fake_clip("S001C001P001R001A023", 22)])
    windows = ntu_windows(path)

    assert {features["crop"] for features, _, _ in windows} == set(CROP_LEVELS)
    assert "crop" not in ACTION_FEATURE_NAMES_FOR_TEST()
    # A cropped window really is legless once it reaches the features, which is the whole point: the crop
    # is applied to the raw scores before windowing and normalization, not to a computed vector.
    for features, _, _ in windows:
        if features["crop"] == "full":
            assert features["lower_body_valid_frac"] > 0.5
        else:
            assert features["lower_body_valid_frac"] == 0.0
    legs = [f for f, _, _ in windows if f["crop"] == "legs"]
    assert all(f["knee_angle_mean"] == 0.0 for f in legs)   # no knees: the angle is unmeasurable


def test_crops_of_one_clip_share_its_group(tmp_path):
    """A cropped copy is a near-duplicate of its original, so both must land in the same fold."""
    path = write_pickle(tmp_path / "ntu.pkl", [fake_clip("S001C001P001R001A023", 22)])
    by_level = {}
    for features, _, group in ntu_windows(path):
        by_level.setdefault(features["crop"], set()).add(group)
    assert len(by_level) == 3
    assert len({frozenset(groups) for groups in by_level.values()}) == 1


def test_the_cap_counts_a_clip_once_however_many_crops_it_yields(tmp_path):
    clips = [fake_clip(f"v_PlayingGuitar_g0{i}_c01", 999) for i in range(1, 4)]
    path = write_pickle(tmp_path / "ucf.pkl", clips)
    groups = {group for _, _, group in ucf_windows(path, max_clips_per_class=2)}
    assert groups == {"ucf:PlayingGuitar_g01", "ucf:PlayingGuitar_g02"}


def ACTION_FEATURE_NAMES_FOR_TEST():
    from core.motion.actions.features import ACTION_FEATURE_NAMES

    return ACTION_FEATURE_NAMES
