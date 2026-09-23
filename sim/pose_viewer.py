"""MediaPipe pose on the laptop webcam, without the robot (sanity check for the vision part).

    reachy_mini_env/bin/python -m sim.pose_viewer [--camera FaceTime] [--model lite|full]
"""

import argparse
import time

import cv2

from core.vision import PoseDetector, add_camera_argument, draw_pose, open_camera, put_text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_argument(parser)
    parser.add_argument("--model", choices=["lite", "full"], default="lite")
    args = parser.parse_args()

    cap = open_camera(args.camera)
    detector = PoseDetector(args.model)
    print("ESC or q to quit.")
    fps, frames, t0 = 0.0, 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame.")
                break
            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(frame)
            if landmarks:
                draw_pose(frame, landmarks)

            frames += 1
            if time.perf_counter() - t0 >= 1.0:
                fps, frames, t0 = frames / (time.perf_counter() - t0), 0, time.perf_counter()
            put_text(frame, f"FPS: {fps:.1f}", 1)

            cv2.imshow("MediaPipe Pose", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
