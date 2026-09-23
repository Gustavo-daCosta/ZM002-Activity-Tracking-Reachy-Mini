"""Show the laptop webcam (no robot, no MediaPipe) and list the available cameras.

    reachy_mini_env/bin/python -m sim.camera_check --list
    reachy_mini_env/bin/python -m sim.camera_check --camera FaceTime
"""

import argparse

import cv2

from core.vision import add_camera_argument, list_cameras, open_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_camera_argument(parser)
    parser.add_argument("--list", action="store_true", help="print cameras (index and name) and exit")
    args = parser.parse_args()

    if args.list:
        for index, name in list_cameras():
            print(f"{index}: {name}")
        return

    cap = open_camera(args.camera)
    print("ESC or q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read a frame.")
                break
            cv2.imshow(f"Webcam {args.camera}", cv2.flip(frame, 1))
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
