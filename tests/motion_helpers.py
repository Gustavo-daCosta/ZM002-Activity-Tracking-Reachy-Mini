"""Synthetic keypoint sequences for motion tests (20 Hz, one person facing the camera, aspect ratio 1)."""

import numpy as np

from core.motion import OTHER, PAUSE, WAVE

FPS = 20
SCORE = 0.9


def pose(wrist_xy=(0.65, 0.8), elbow_xy=(0.65, 0.65), hand="right", score=SCORE):
    """Frame-normalized COCO keypoints. Shoulders at (0.4, 0.5) and (0.6, 0.5): shoulder width 0.2.

    `wrist_xy` / `elbow_xy` place the moving arm given in right-arm coordinates; for hand="left" they are
    mirrored (x -> 1 - x). The other arm rests down.
    """
    kps = np.zeros((17, 3), np.float32)
    kps[0] = (0.5, 0.35, score)
    kps[5] = (0.4, 0.5, score)
    kps[6] = (0.6, 0.5, score)
    kps[7] = (0.35, 0.65, score)
    kps[8] = (0.65, 0.65, score)
    kps[9] = (0.35, 0.8, score)
    kps[10] = (0.65, 0.8, score)
    if hand == "right":
        kps[10, :2], kps[8, :2] = wrist_xy, elbow_xy
    else:
        kps[9, :2] = (1 - wrist_xy[0], wrist_xy[1])
        kps[7, :2] = (1 - elbow_xy[0], elbow_xy[1])
    return kps


def arm_positions(kind, t, freq_hz=2.0, amplitude=0.1):
    """(wrist_xy, elbow_xy) of the moving arm at local time t for a movement kind."""
    swing = amplitude * np.sin(2 * np.pi * freq_hz * t)
    if kind == "wave":
        return (0.65 + swing, 0.4), (0.65, 0.45)
    if kind == "still":
        return (0.65, 0.8), (0.65, 0.65)
    if kind == "hand_on_face":
        return (0.52, 0.38), (0.62, 0.6)
    if kind == "swing_low":
        return (0.65 + swing, 0.8), (0.65, 0.65)
    if kind == "stretch":
        progress = min(1.0, t / 2.0)
        return (0.65, 0.8 - 0.5 * progress), (0.65, 0.65 - 0.3 * progress)
    raise ValueError(kind)


def motion_frames(kind, duration_s=2.0, start=0.0, hand="right", freq_hz=2.0, amplitude=0.1, jitter=0.004, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(int(round(duration_s * FPS))):
        local = i / FPS
        wrist, elbow = arm_positions(kind, local, freq_hz, amplitude)
        wrist = (wrist[0] + rng.normal(0, jitter), wrist[1] + rng.normal(0, jitter))
        frames.append((start + local, pose(wrist, elbow, hand)))
    return frames


def feed(window, frames):
    for t, keypoints in frames:
        window.add(t, keypoints)
    return window


def synthetic_session(seed=0, round_kinds=("wave", "still", "wave", "hand_on_face"), round_s=4.0, pause_s=1.0):
    """Session dict like dataset.load_session: pause (still) before every round, random wave speed/size."""
    rng = np.random.default_rng(seed)
    t, keypoints, labels, rounds = [], [], [], []
    clock = 0.0

    def extend(frames, label, round_id):
        for time_s, kps in frames:
            t.append(time_s)
            keypoints.append(kps)
            labels.append(label)
            rounds.append(round_id)

    for round_id, kind in enumerate(round_kinds):
        extend(motion_frames("still", pause_s, clock, seed=int(rng.integers(2**31))), PAUSE, -1)
        clock += pause_s
        frames = motion_frames(
            kind, round_s, clock,
            freq_hz=rng.uniform(1.5, 3.0), amplitude=rng.uniform(0.07, 0.12), seed=int(rng.integers(2**31)),
        )
        extend(frames, WAVE if kind == "wave" else OTHER, round_id)
        clock += round_s
    return {
        "t": np.array(t, np.float64),
        "keypoints": np.array(keypoints, np.float32),
        "labels": np.array(labels, np.int8),
        "rounds": np.array(rounds, np.int16),
        "pose_model": "synthetic",
        "aspect_ratio": 1.0,
    }
