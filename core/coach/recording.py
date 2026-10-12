"""Guided recording of the coach's exercises, in the `collect/` kit's `.npz` format.

The pipeline announces each round, calls `start_round`, then feeds every frame to `add`; when the round's
time is up `add` returns a `RoundResult` after checking that the child stayed in view and really moved. A
bad round is dropped (its frames get `rounds = -1`, which `training.data.make_windows` skips) and repeated
once. Classes, schedule and file keys mirror `collect/record_exercises.py`, a standalone kit that cannot
import this module: keep the two in step.
"""

import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.motion.window import normalize_to_torso

CLASSES = ("none", "squat", "arm_raise", "side_bend")
# COCO-17 parts that must be visible (either side of a pair is enough), as in collect/record_exercises.py.
REQUIRED = {"head": (0,), "hips": (11, 12), "knees": (13, 14), "ankles": (15, 16)}
# Elbows, wrists, hips, knees: what moves relative to the shoulders in all three exercises.
MOVING_JOINTS = [7, 8, 9, 10, 11, 12, 13, 14]
# Calibrated on the group's 192 rounds (collect/recordings, 2026-10-12): the whole body was visible in
# >= 88 % of the frames of every round; exercise rounds moved >= 0.31 torso lengths (5th percentile 0.41),
# standing-still `none` rounds 0.05-0.13. Both are tunable from the entry points.
MIN_VISIBLE = 0.7
MIN_MOTION = 0.2


def missing_parts(keypoints, min_score):
    """Names of the REQUIRED body parts not visible in this frame (all of them without a person)."""
    if keypoints is None:
        return list(REQUIRED)
    return [part for part, indices in REQUIRED.items() if (keypoints[list(indices), 2] < min_score).all()]


def round_quality(keypoints, aspect_ratio, min_score):
    """(visible fraction, motion) of one round's (N, 17, 3) keypoints.

    Motion is the largest 5-95 percentile range of any MOVING_JOINTS coordinate, in torso lengths from the
    shoulder midpoint: where the child stands and how far from the camera do not matter.
    """
    if len(keypoints) == 0:
        return 0.0, 0.0
    visible = float(np.mean([not missing_parts(k, min_score) for k in keypoints]))
    normalized = [normalize_to_torso(k, min_score, aspect_ratio) for k in keypoints]
    joints = np.array([n[MOVING_JOINTS, :2] for n in normalized if n is not None])
    if len(joints) < 5:
        return visible, 0.0
    spread = np.percentile(joints, 95, axis=0) - np.percentile(joints, 5, axis=0)
    return visible, float(spread.max())


def sanitize(name):
    """'João Silva' -> 'joao_silva': safe as a folder and file name on every OS (as in the collect kit)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9_]", "", ascii_name.strip().lower().replace(" ", "_"))


def next_session_index(person_dir):
    """1 + the highest existing session number, so a rerun never overwrites earlier sessions."""
    numbers = [int(match.group(1)) for path in Path(person_dir).glob("session_*.npz")
               if (match := re.fullmatch(r"session_(\d+)\.npz", path.name))]
    return max(numbers, default=0) + 1


@dataclass
class RoundResult:
    """Outcome of one round.

    Attributes:
        name: The class that was recorded.
        ok: The round passed the check and is kept.
        visible: Fraction of frames with the whole body visible.
        motion: Movement in torso lengths (see `round_quality`).
        retry: The same class was queued again, to be recorded next.
        gone: The child was out of view when the round ended; a failed round then keeps its redo.
    """

    name: str
    ok: bool
    visible: float
    motion: float
    retry: bool
    gone: bool


class GuidedRecorder:
    """Round schedule plus labeled keypoint rows for one session.

    Attributes:
        total: Rounds planned (`rounds_per_class` x 4); redos do not add to it.
        done: Rounds finished, kept or given up after their redo.
        seed: Shuffle seed, saved with the session.
        rows: Lists "t", "keypoints", "labels", "rounds", one entry per recorded frame.
    """

    def __init__(self, aspect_ratio, min_score=0.3, rounds_per_class=6, round_s=8.0, seed=None,
                 min_visible=MIN_VISIBLE, min_motion=MIN_MOTION):
        self.aspect_ratio = aspect_ratio
        self.min_score = min_score
        self.round_s = round_s
        self.min_visible = min_visible
        self.min_motion = min_motion
        self.seed = random.randrange(2 ** 31) if seed is None else seed
        order = [name for name in CLASSES for _ in range(rounds_per_class)]
        # Shuffled so tiredness and position in the session do not correlate with the class.
        random.Random(self.seed).shuffle(order)
        self._queue = [(name, False) for name in order]  # (class, is a redo)
        self.total = len(order)
        self.done = 0
        self.rows = {"t": [], "keypoints": [], "labels": [], "rounds": []}
        self._current = None  # (class, is a redo, round id, start time)
        self._next_round_id = 0
        self._t0 = None

    @property
    def pending(self):
        """Class of the next round, or None when the schedule is done."""
        return self._queue[0][0] if self._queue else None

    @property
    def current(self):
        """Class being recorded, or None between rounds."""
        return self._current[0] if self._current else None

    @property
    def finished(self):
        """Whether every round is done."""
        return not self._queue and self._current is None

    @property
    def has_rounds(self):
        """Whether any frame was recorded (quitting before the first round saves nothing)."""
        return bool(self.rows["t"])

    def start_round(self, now):
        """Start recording the next class; returns its name."""
        name, redo = self._queue.pop(0)
        if self._t0 is None:
            self._t0 = now
        self._current = (name, redo, self._next_round_id, now)
        self._next_round_id += 1
        return name

    def time_left(self, now):
        """Seconds left in the current round (0 between rounds)."""
        return max(0.0, self._current[3] + self.round_s - now) if self._current else 0.0

    def add(self, now, keypoints):
        """Record one frame of the current round. Returns a `RoundResult` once `round_s` has passed."""
        name, _, round_id, started = self._current
        keypoints = np.zeros((17, 3), np.float32) if keypoints is None else np.asarray(keypoints, np.float32)
        self.rows["t"].append(now - self._t0)
        self.rows["keypoints"].append(keypoints)
        self.rows["labels"].append(CLASSES.index(name))
        self.rows["rounds"].append(round_id)
        if now - started < self.round_s:
            return None
        return self._finish(keypoints)

    def _finish(self, last_keypoints):
        name, redo, round_id, _ = self._current
        self._current = None
        mine = np.flatnonzero(np.asarray(self.rows["rounds"]) == round_id)
        visible, motion = round_quality(np.asarray(self.rows["keypoints"])[mine], self.aspect_ratio,
                                        self.min_score)
        ok = visible >= self.min_visible and (name == "none" or motion >= self.min_motion)
        gone = bool(missing_parts(last_keypoints, self.min_score))
        retry = False
        if not ok:
            for index in mine:
                self.rows["rounds"][index] = -1  # make_windows skips these frames
            if gone:  # the child walked away: not their fault, the redo is not used up
                self._queue.insert(0, (name, redo))
                retry = True
            elif not redo:
                self._queue.insert(0, (name, True))
                retry = True
        if not retry:
            self.done += 1
        return RoundResult(name, ok, visible, motion, retry, gone)

    def save(self, path, person, session, pose_model):
        """Write the rows as `.npz` with the keys of `collect/record_exercises.py`. Returns the path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, t=np.asarray(self.rows["t"], np.float64),
            keypoints=np.asarray(self.rows["keypoints"], np.float32).reshape(-1, 17, 3),
            labels=np.asarray(self.rows["labels"], np.int8), rounds=np.asarray(self.rows["rounds"], np.int16),
            classes=np.array(CLASSES), pose_model=np.array(pose_model),
            aspect_ratio=np.float64(self.aspect_ratio), person=np.array(person),
            session=np.int16(session), seed=np.int64(self.seed))
        return path
