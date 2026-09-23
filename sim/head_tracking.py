"""Webcam head tracking on the simulated Reachy Mini.

The MuJoCo simulation has no camera, so the laptop webcam stands in for it: MediaPipe finds
the nose, its position in the image becomes head yaw/pitch, sent with set_target().

    # terminal 1
    sim/start.sh
    # terminal 2
    reachy_mini_env/bin/python -m sim.head_tracking [--camera FaceTime]
"""

import argparse
import time

import cv2

from core.tracking_math import HeadFollower, image_error_to_angles
from core.vision import NOSE, PoseDetector, add_camera_argument, draw_pose, open_camera, put_text


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_argument(parser)
    parser.add_argument("--model", choices=["lite", "full"], default="lite", help="pose model size")
    parser.add_argument("--host", default="localhost", help="daemon host (default: local simulation)")
    parser.add_argument("--rate", type=float, default=20.0, help="max head updates per second")
    parser.add_argument("--max-yaw", type=float, default=40.0, help="degrees at the image edge")
    parser.add_argument("--max-pitch", type=float, default=25.0, help="degrees at the image edge")
    parser.add_argument("--deadzone", type=float, default=0.03, help="ignored offset from center (0..0.5)")
    parser.add_argument("--smoothing", type=float, default=0.2, help="0..1, higher = faster")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import the SDK late so --help works instantly.
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose

    cap = open_camera(args.camera)
    detector = PoseDetector(args.model)
    follower = HeadFollower(smoothing=args.smoothing, return_speed=0.05)
    update_interval = 1.0 / args.rate

    print(f"Connecting to the daemon on {args.host}...")
    connection_mode = "localhost_only" if args.host in ("localhost", "127.0.0.1") else "network"
    # no_media: the simulation has no camera/microphone, we use the webcam instead.
    with ReachyMini(host=args.host, connection_mode=connection_mode, media_backend="no_media") as mini:
        print("Connected. ESC or q to quit.")
        mini.goto_target(head=create_head_pose(), duration=1.0)

        last_update = 0.0
        fps, frames, t0 = 0.0, 0, time.perf_counter()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Could not read a frame.")
                    break
                frame = cv2.flip(frame, 1)  # mirror: behaves like looking at the robot

                landmarks = detector.detect(frame)
                target = None
                if landmarks:
                    nose = landmarks[NOSE]
                    target = image_error_to_angles(nose.x, nose.y, args.max_yaw, args.max_pitch, args.deadzone)
                    draw_pose(frame, landmarks)
                    h, w = frame.shape[:2]
                    cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, (0, 255, 255), -1)
                # No person: drift slowly back to center instead of freezing.
                yaw, pitch = follower.update(target)

                now = time.perf_counter()
                if now - last_update >= update_interval:
                    mini.set_target(head=create_head_pose(yaw=yaw, pitch=pitch, degrees=True))
                    last_update = now

                frames += 1
                if now - t0 >= 1.0:
                    fps, frames, t0 = frames / (now - t0), 0, now
                put_text(frame, f"FPS: {fps:.1f}", 1)
                put_text(frame, f"Yaw: {yaw:+.1f}  Pitch: {pitch:+.1f}", 2, (0, 255, 0))
                if target is None:
                    put_text(frame, "No person detected", 3, (0, 0, 255))

                cv2.imshow("Reachy Mini sim - webcam tracking", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            print("Returning head to center...")
            mini.goto_target(head=create_head_pose(), duration=1.0)
            detector.close()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
