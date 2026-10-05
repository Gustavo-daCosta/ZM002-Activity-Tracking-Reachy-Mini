"""Guided recording of physiotherapy exercises, to train the Reachy Mini action classifier.

    python record_exercises.py --list-cameras
    python record_exercises.py --name alice [--camera 0]

Follow the instructions on screen. One session takes about 4.5 minutes; record two, changing clothes,
position or lighting in between. When you finish, a zip is written to the current folder: send it back.
"""

import argparse
import math
import os
import random
import re
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "movenet-lightning.onnx"
POSE_MODEL = "movenet-lightning"
INPUT_SIZE = 192
MIN_SCORE = 0.3
WINDOW = "Exercise recorder"

CLASSES = ("none", "squat", "arm_raise", "side_bend")
TITLES = {"none": "NO EXERCISE", "squat": "SQUAT", "arm_raise": "ARM RAISE", "side_bend": "SIDE BEND"}
HINTS = {
    "squat": "feet shoulder-width, sit back, stand up",
    "arm_raise": "raise both arms sideways to shoulder height, lower",
    "side_bend": "hands on hips, lean left, then right",
}
NONE_HINTS = ("stand still", "check your phone", "fix your hair", "take a step", "cross your arms",
              "scratch your head")
COLORS = {"none": (0, 140, 255), "squat": (200, 120, 0), "arm_raise": (0, 160, 0), "side_bend": (160, 0, 160)}
PAUSE_COLOR = (120, 120, 120)
PREP_S = 5.0
COUNTDOWN_S = 3.0
AUTO_START_S = 5.0  # whole body visible this long -> the session starts by itself

# COCO-17 body parts that must be visible before a session starts (either side of a pair is enough).
REQUIRED = {"head": (0,), "hips": (11, 12), "knees": (13, 14), "ankles": (15, 16)}
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


class CameraError(RuntimeError):
    """The camera stopped delivering frames."""


class Pose:
    """MoveNet SinglePose Lightning on onnxruntime (CPU), the same model as the robot's pose backend."""

    def __init__(self, path=MODEL_PATH):
        if not path.exists():
            sys.exit(f"{path.name} not found next to the script. Unzip the whole kit folder and run from it.")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, frame):
        """(17, 3) float32 keypoints: x, y normalized to the frame, score. All zeros when nobody is found."""
        h, w = frame.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        new_w, new_h = round(w * scale), round(h * scale)
        pad_x, pad_y = (INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2
        square = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        square[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = cv2.resize(frame, (new_w, new_h))
        square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        raw = self.session.run(None, {self.input: square[None].astype(np.int32)})[0].reshape(17, 3)
        keypoints = np.zeros((17, 3), np.float32)
        if (raw[:, 2] >= MIN_SCORE).any():
            keypoints[:, 0] = (raw[:, 1] * INPUT_SIZE - pad_x) / scale / w
            keypoints[:, 1] = (raw[:, 0] * INPUT_SIZE - pad_y) / scale / h
            keypoints[:, 2] = raw[:, 2]
        return keypoints


def capture(index):
    """VideoCapture on DirectShow on Windows: the default MSMF backend can take up to a minute to open."""
    return cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)


def key_pressed():
    """The key pressed in the video window as a lowercase character ('' for none, 'esc', ' ')."""
    key = cv2.waitKey(1) & 0xFF
    return "esc" if key == 27 else "" if key == 255 else chr(key).lower()


def open_camera(index):
    """Open camera `index` at 640x480 (keeps the videos small), or exit with a hint."""
    cap = capture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened() or not cap.read()[0]:
        sys.exit(f"Could not open camera {index} (is another app using it?). "
                 "Run with --list-cameras to see which ones work.")
    return cap


def read(cap):
    """The next frame, or CameraError."""
    ok, frame = cap.read()
    if not ok:
        raise CameraError("Could not read a frame from the camera. Run with --list-cameras to check it.")
    return frame


def list_cameras(max_index=6):
    """Print which camera indices deliver frames."""
    for index in range(max_index):
        cap = capture(index)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        print(f"camera {index}: {f'{frame.shape[1]}x{frame.shape[0]}' if ok else 'not available'}")
        cap.release()


def draw_skeleton(frame, keypoints):
    """Draw confident keypoints and the bones between them."""
    h, w = frame.shape[:2]
    confident = keypoints[:, 2] >= MIN_SCORE
    points = [(int(x * w), int(y * h)) for x, y, _ in keypoints]
    for a, b in SKELETON:
        if confident[a] and confident[b]:
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    for i, point in enumerate(points):
        if confident[i]:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)


def show(frame, keypoints):
    """A mirrored copy with the skeleton, so people see themselves as in a mirror. The frame is untouched."""
    view = frame.copy()
    draw_skeleton(view, keypoints)
    return cv2.flip(view, 1)


def banner(frame, color, title, subtitle=""):
    """Colored bar at the top with a big title and a smaller line under it."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), color, -1)
    cv2.putText(frame, title, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, subtitle, (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def status(frame, text, row, color=(255, 255, 255)):
    """One line of text under the banner, `row` 1-based."""
    cv2.putText(frame, text, (15, 90 + 30 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def missing_parts(keypoints):
    """Names of the REQUIRED body parts not visible in this frame."""
    return [part for part, indices in REQUIRED.items() if (keypoints[list(indices), 2] < MIN_SCORE).all()]


@dataclass
class Phase:
    """One segment of a session.

    Attributes:
        start: Start time in seconds from the session start.
        end: End time in seconds.
        label: Index into CLASSES, -1 during the countdown before a round.
        round_id: Round index, -1 during countdowns.
        title: Big text on the banner.
        hint: Small text under it.
        color: Banner color (BGR).
    """

    start: float
    end: float
    label: int
    round_id: int
    title: str
    hint: str
    color: tuple


def build_schedule(rounds, round_s, seed):
    """`rounds` rounds of every class in a shuffled order, each preceded by a countdown naming it.

    Shuffled so tiredness and position in the session do not correlate with the class.
    """
    order = [label for label in range(len(CLASSES)) for _ in range(rounds)]
    random.Random(seed).shuffle(order)
    schedule, clock, none_count = [], 0.0, 0
    for round_id, label in enumerate(order):
        name = CLASSES[label]
        if name == "none":
            hint = NONE_HINTS[none_count % len(NONE_HINTS)]
            none_count += 1
        else:
            hint = HINTS[name]
        wait = PREP_S if round_id == 0 else COUNTDOWN_S
        schedule.append(Phase(clock, clock + wait, -1, -1, f"Next: {TITLES[name]}", hint, PAUSE_COLOR))
        clock += wait
        schedule.append(Phase(clock, clock + round_s, label, round_id, TITLES[name], hint, COLORS[name]))
        clock += round_s
    return schedule


def phase_at(schedule, elapsed):
    """The phase active at `elapsed` seconds, or None after the schedule ends."""
    return next((phase for phase in schedule if phase.start <= elapsed < phase.end), None)


def framing_screen(cap, pose):
    """Live preview until the whole body stays visible for AUTO_START_S (or SPACE is pressed).

    Returns:
        Frames per second of this loop (read + pose, the same work as recording), or None to quit.
    """
    stamps = []
    visible_since = None
    while True:
        frame = read(cap)
        keypoints = pose(frame)
        now = time.perf_counter()
        stamps = (stamps + [now])[-30:]
        view = show(frame, keypoints)
        missing = missing_parts(keypoints)
        if missing:
            visible_since = None
            banner(view, (0, 0, 200), "STEP BACK", "not visible: " + ", ".join(missing))
        else:
            visible_since = visible_since or now
            left = AUTO_START_S - (now - visible_since)
            banner(view, (0, 160, 0), f"READY - starting in {max(math.ceil(left), 1)}", "stay where you are")
        status(view, "whole body in view, 2-3 m from the camera   Q = quit", 1)
        cv2.imshow(WINDOW, view)
        key = key_pressed()
        if key in ("esc", "q"):
            return None
        if not missing and len(stamps) > 1 and (key == " " or now - visible_since >= AUTO_START_S):
            return (len(stamps) - 1) / (stamps[-1] - stamps[0])


def record_session(cap, pose, schedule, video_path, fps):
    """Record one session: every frame goes to the mp4 and gets one labeled keypoints row.

    Stops early on q / ESC, Ctrl+C or a camera failure and keeps what was recorded.

    Returns:
        (rows, aspect_ratio, stopped): rows holds lists "t", "keypoints", "labels", "rounds".
    """
    rows = {"t": [], "keypoints": [], "labels": [], "rounds": []}
    writer, aspect_ratio, stopped = None, 4 / 3, False
    total = sum(phase.round_id >= 0 for phase in schedule)
    start = time.perf_counter()
    try:
        while True:
            frame = read(cap)
            elapsed = time.perf_counter() - start
            phase = phase_at(schedule, elapsed)
            if phase is None:
                break
            if writer is None:
                h, w = frame.shape[:2]
                aspect_ratio = w / h
                # Relative path: OpenCV on Windows cannot write to paths with accents (C:\Users\João\...).
                writer = cv2.VideoWriter(os.path.relpath(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise CameraError(f"Could not write {video_path}. Move the kit folder to a path without "
                                      "accents or spaces (e.g. C:\\collect) and run from inside it.")
            keypoints = pose(frame)
            writer.write(frame)
            rows["t"].append(elapsed)
            rows["keypoints"].append(keypoints)
            rows["labels"].append(phase.label)
            rows["rounds"].append(phase.round_id)

            view = show(frame, keypoints)
            banner(view, phase.color, phase.title, phase.hint)
            round_text = f"round {phase.round_id + 1}/{total}" if phase.round_id >= 0 else "get ready"
            status(view, f"{phase.end - elapsed:.1f} s   {round_text}   Q = stop", 1)
            missing = missing_parts(keypoints)
            if missing:
                status(view, "not visible: " + ", ".join(missing) + " - step back", 2, (0, 0, 255))
            cv2.imshow(WINDOW, view)
            if key_pressed() in ("esc", "q"):
                stopped = True
                break
    except KeyboardInterrupt:
        stopped = True
    except CameraError as error:
        print(error)
        stopped = True
    finally:
        if writer is not None:
            writer.release()
    return rows, aspect_ratio, stopped


def save_session(path, rows, aspect_ratio, person, session, seed):
    """Write one session's rows as `.npz`: a superset of what training/data.py `load_session` reads."""
    np.savez_compressed(
        path, t=np.asarray(rows["t"], np.float64),
        keypoints=np.asarray(rows["keypoints"], np.float32).reshape(-1, 17, 3),
        labels=np.asarray(rows["labels"], np.int8), rounds=np.asarray(rows["rounds"], np.int16),
        classes=np.array(CLASSES), pose_model=np.array(POSE_MODEL), aspect_ratio=np.float64(aspect_ratio),
        person=np.array(person), session=np.int16(session), seed=np.int64(seed))


def next_session_screen(cap, session):
    """Between sessions: N records another one, Q finishes. Returns True for another session."""
    while True:
        view = cv2.flip(read(cap), 1)
        banner(view, (0, 160, 0), f"SESSION {session} SAVED", "change clothes, position or lighting")
        status(view, "N = record another session (2 recommended)", 1)
        status(view, "Q = finish and write the zip", 2)
        cv2.imshow(WINDOW, view)
        key = key_pressed()
        if key == "n":
            return True
        if key in ("esc", "q"):
            return False


def sanitize(name):
    """'João Silva' -> 'joao_silva': safe as a folder and file name on every OS."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9_]", "", ascii_name.strip().lower().replace(" ", "_"))


def next_session_index(out_dir):
    """1 + the highest existing session number, so a rerun never overwrites earlier sessions."""
    numbers = [int(match.group(1)) for path in out_dir.glob("session_*.npz")
               if (match := re.fullmatch(r"session_(\d+)\.npz", path.name))]
    return max(numbers, default=0) + 1


def frame_counts(labels):
    """'none 120, squat 118, ...': frames per class."""
    return ", ".join(f"{name} {int((labels == index).sum())}" for index, name in enumerate(CLASSES))


def write_zip(out_dir, name):
    """Zip every session in `out_dir` into the current folder. mp4s are stored, they are already compressed."""
    path = Path.cwd() / f"exercise_data_{name}_{time.strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(out_dir.glob("session_*")):
            compression = zipfile.ZIP_STORED if file.suffix == ".mp4" else zipfile.ZIP_DEFLATED
            archive.write(file, f"{name}/{file.name}", compress_type=compression)
    return path


def main():
    """Entry point."""
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", help="your name (used in the file names)")
    parser.add_argument("--camera", type=int, default=0, help="camera index, see --list-cameras (default 0)")
    parser.add_argument("--rounds", type=int, default=6, help="rounds per exercise per session (default 6)")
    parser.add_argument("--round-seconds", type=float, default=8.0, help="seconds per round (default 8)")
    parser.add_argument("--list-cameras", action="store_true", help="print which cameras work and exit")
    args = parser.parse_args()
    if args.list_cameras:
        list_cameras()
        return
    name = sanitize(args.name or "")
    if not name:
        parser.error("--name is required (letters, digits, spaces)")

    out_dir = HERE / "recordings" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pose = Pose()
    cap = open_camera(args.camera)
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    session = next_session_index(out_dir)
    try:
        while True:
            fps = framing_screen(cap, pose)
            if fps is None:
                break
            seed = random.randrange(2 ** 31)
            video = out_dir / f"session_{session}.mp4"
            rows, aspect_ratio, stopped = record_session(
                cap, pose, build_schedule(args.rounds, args.round_seconds, seed), video, fps)
            if any(round_id >= 0 for round_id in rows["rounds"]):  # quitting in a countdown saves nothing
                save_session(out_dir / f"session_{session}.npz", rows, aspect_ratio, name, session, seed)
                print(f"session {session}: {len(rows['t'])} frames ({frame_counts(np.array(rows['labels']))})")
                session += 1
            else:
                video.unlink(missing_ok=True)
            if stopped or not next_session_screen(cap, session - 1):
                break
    except CameraError as error:
        print(error)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not any(out_dir.glob("session_*.npz")):
        print("Nothing recorded, no zip written.")
        return
    path = write_zip(out_dir, name)
    print(f"\nDone. Send this file back: {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
