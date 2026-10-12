"""Run MoveNet on a laptop and send keypoints to the Reachy Mini workout session.

The Reachy must be running `robot.remote_camera_server --session`, which serves
an MJPEG camera stream and accepts keypoints at its /result endpoint.

Example:
    python -m robot.apps.remote_session_inference --model movenet-lightning
      --remote-camera http://10.42.0.66:8765/stream.mjpg
      --result-url http://10.42.0.66:8766/result --width 192 --window
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import cv2

from core.pose_backends import BACKENDS, create_backend
from core.vision import LatestJsonPoster, RemoteMjpegSource, downscale, draw_coco_skeleton, put_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(BACKENDS), default="movenet-lightning")
    parser.add_argument("--remote-camera", required=True, help="Reachy MJPEG stream URL")
    parser.add_argument("--result-url", required=True, help="Reachy /result URL receiving keypoints")
    parser.add_argument("--width", type=int, default=192, help="maximum frame width before inference")
    parser.add_argument("--min-score", type=float, default=None, help="keypoint confidence threshold for overlay")
    parser.add_argument("--send-rate", type=float, default=20.0, help="maximum keypoint POST rate")
    parser.add_argument("--window", action="store_true", help="show the camera with pose overlay")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds; 0 means Ctrl+C")
    args = parser.parse_args()

    if args.width < 1:
        parser.error("--width must be positive")
    if args.send_rate <= 0:
        parser.error("--send-rate must be positive")

    source = None
    backend = None
    poster = None
    try:
        source = RemoteMjpegSource(args.remote_camera)
        backend = create_backend(args.model)
        min_score = args.min_score if args.min_score is not None else backend.default_min_score
        poster = LatestJsonPoster(args.result_url, rate_hz=args.send_rate)

        providers_session = getattr(backend, "_session", None)
        if providers_session is not None:
            print("ONNX Runtime providers:", providers_session.get_providers())

        print(f"Sending keypoints to Reachy session: {args.result_url}")
        print("Press q/ESC in the window or Ctrl+C to stop.")

        frame_times: deque[float] = deque()
        frame_count = 0
        started = time.perf_counter()
        last_report = started

        while True:
            frame = source.read()
            if frame is None:
                time.sleep(0.005)
                continue

            frame = downscale(frame, args.width)
            result = backend.infer(frame)
            now = time.perf_counter()
            frame_times.append(now)
            frame_count += 1

            # Keep a one-second rolling FPS window for diagnostics; inference remains on the laptop.
            while frame_times and frame_times[0] < now - 1.0:
                frame_times.popleft()
            fps = float(len(frame_times))

            keypoints = result.keypoints
            poster.update({
                "timestamp": time.time(),
                "model": backend.name,
                "fps": fps,
                "keypoints": keypoints.tolist() if keypoints is not None else None,
            })

            if args.window:
                view = frame.copy()
                if keypoints is not None:
                    draw_coco_skeleton(view, keypoints, min_score)
                put_text(view, f"REMOTE SESSION INFERENCE | {backend.name}", 1)
                put_text(view, f"FPS: {fps:.1f} | input: {frame.shape[1]}x{frame.shape[0]}", 2)
                put_text(view, f"Sending keypoints: {args.result_url}", 3, (0, 255, 255))
                cv2.imshow("Reachy remote session inference", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if now - last_report >= 2.0:
                print(
                    f"FPS={fps:.1f}; frames={frame_count}; "
                    f"keypoints={'yes' if keypoints is not None else 'none'}"
                )
                last_report = now

            if args.seconds and now - started >= args.seconds:
                break

    except KeyboardInterrupt:
        pass
    finally:
        if poster is not None:
            poster.close()
        if source is not None:
            source.close()
        if backend is not None:
            backend.close()
        if args.window:
            cv2.destroyAllWindows()
        print(f"Remote session inference stopped after {frame_count if 'frame_count' in locals() else 0} frames.")


if __name__ == "__main__":
    main()
