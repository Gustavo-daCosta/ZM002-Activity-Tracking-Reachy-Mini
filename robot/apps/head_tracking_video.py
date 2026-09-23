"""Camera window with a square on the tracked face.

Run: reachy_mini_env/bin/python -m robot.apps.head_tracking_video  (press 'q' to quit)
"""

import cv2
import numpy as np

from robot.connect import connect_robot

BOX_SIZE = 180  # px on the 1280x720 video: the tracker only returns the face center

with connect_robot(media=True, needs=("tracking",), sleep_on_exit=True) as mini:
    mini.start_head_tracking(weight=1.0)
    print("Head tracking ON. Press 'q' in the window (or Ctrl+C) to stop.")
    try:
        while True:
            frame = mini.media.get_frame()
            if frame is None:
                continue
            frame = frame.copy()  # SDK frames are read-only; OpenCV draws in place

            h, w = frame.shape[:2]
            face = mini.get_tracked_face(wait=False)

            if face.detected:
                # x, y are the nose position normalized to [-1, 1]
                cx = (face.x + 1) / 2 * (w - 1)
                cy = (face.y + 1) / 2 * (h - 1)
                angle = np.degrees(face.roll or 0.0)

                # Square centered on the face, rotated by the head roll
                box = cv2.boxPoints(((cx, cy), (BOX_SIZE, BOX_SIZE), angle))
                cv2.polylines(frame, [box.astype(np.int32)], True, (0, 255, 0), 3)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                label = f"x={face.x:+.2f} y={face.y:+.2f} roll={angle:+.0f}deg"
                color = (0, 255, 0)
            elif face.ts is None:
                label, color = "Waiting for the tracker...", (0, 200, 255)
            else:
                label, color = "No face detected", (0, 0, 255)

            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Reachy Mini camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
