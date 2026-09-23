"""Guided recording of wave / not-wave examples (keypoints only, no video). No robot needed.

    reachy_mini_env/bin/python -m training.record [--rounds 10] [--round-seconds 4]

Follow the text on screen: WAVE rounds and NOT WAVE rounds (try the suggested non-wave movements),
separated by short countdowns. q / ESC stops early and still saves.
"""

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np

from core.motion import OTHER, PAUSE, WAVE
from training.dataset import DATA_DIR, save_session
from core.pose_backends import BACKENDS, create_backend
from core.vision import add_camera_argument, draw_coco_skeleton, open_camera, put_text

OTHER_HINTS = (
    "stand still",
    "touch your face",
    "stretch your arms",
    "cross your arms",
    "drink / scratch your head",
    "move your arms below the shoulders",
)
COLORS = {WAVE: (0, 200, 0), OTHER: (0, 140, 255), PAUSE: (160, 160, 160)}


@dataclass
class Phase:
    start: float
    end: float
    label: int
    round_id: int
    text: str


def build_schedule(rounds, round_s, pause_s, prep_s=3.0):
    """Preparation, then 2 * rounds rounds alternating WAVE / NOT WAVE with a countdown pause between them."""
    schedule = [Phase(0.0, prep_s, PAUSE, -1, "Get ready: WAVE")]
    clock = prep_s
    for k in range(2 * rounds):
        label = WAVE if k % 2 == 0 else OTHER
        name = "WAVE" if label == WAVE else "NOT WAVE"
        if k > 0:
            schedule.append(Phase(clock, clock + pause_s, PAUSE, -1, f"Next: {name}"))
            clock += pause_s
        text = name if label == WAVE else f"{name} - {OTHER_HINTS[(k // 2) % len(OTHER_HINTS)]}"
        schedule.append(Phase(clock, clock + round_s, label, k, text))
        clock += round_s
    return schedule


def phase_at(schedule, elapsed):
    return next((phase for phase in schedule if phase.start <= elapsed < phase.end), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, default=10, help="rounds per class (default 10)")
    parser.add_argument("--round-seconds", type=float, default=4.0)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    parser.add_argument("--model", choices=list(BACKENDS), default="blazepose-lite", help="pose model")
    add_camera_argument(parser)
    args = parser.parse_args()

    schedule = build_schedule(args.rounds, args.round_seconds, args.pause_seconds)
    total_rounds = 2 * args.rounds
    cap = open_camera(args.camera)
    backend = create_backend(args.model)
    times, keypoints, labels, rounds = [], [], [], []
    aspect_ratio = 16 / 9
    print(f"Recording {total_rounds} rounds (~{schedule[-1].end:.0f} s). q / ESC to stop early.")

    start = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame.")
                break
            frame = cv2.flip(frame, 1)
            elapsed = time.monotonic() - start
            phase = phase_at(schedule, elapsed)
            if phase is None:
                break
            aspect_ratio = frame.shape[1] / frame.shape[0]

            result = backend.infer(frame)
            kps = result.keypoints if result.keypoints is not None else np.zeros((17, 3), np.float32)
            times.append(elapsed)
            keypoints.append(kps)
            labels.append(phase.label)
            rounds.append(phase.round_id)

            if result.keypoints is not None:
                draw_coco_skeleton(frame, result.keypoints, backend.default_min_score)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), COLORS[phase.label], -1)
            cv2.putText(frame, phase.text, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
            round_text = f"round {phase.round_id + 1}/{total_rounds}" if phase.round_id >= 0 else "countdown"
            put_text(frame, f"{phase.end - elapsed:.1f} s left   {round_text}", 4)
            if result.keypoints is None:
                put_text(frame, "NO PERSON", 5, (0, 0, 255))
            elif (result.keypoints[[9, 10], 2] < backend.default_min_score).all():
                # Without a visible wrist the window is discarded in training.
                put_text(frame, "HANDS NOT VISIBLE - step back / raise hands into view", 5, (0, 0, 255))
            else:
                put_text(frame, "person and hand visible", 5, (0, 255, 0))

            cv2.imshow("Reachy Mini - record wave examples", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        cap.release()
        cv2.destroyAllWindows()

    if not times:
        print("Nothing recorded.")
        return
    path = DATA_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    save_session(path, times, keypoints, labels, rounds, backend.name, aspect_ratio)
    labels = np.array(labels)
    print(f"Saved {path}")
    print(f"frames: wave {int((labels == WAVE).sum())}, not wave {int((labels == OTHER).sum())}, "
          f"countdown {int((labels == PAUSE).sum())}")


if __name__ == "__main__":
    main()
