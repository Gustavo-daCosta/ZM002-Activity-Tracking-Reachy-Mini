"""Click on the camera window and the robot looks at that point.

Run: reachy_mini_env/bin/python -m robot.apps.look_at_click
Keys: 'c' recenter the head, 'q' quit.
"""

import cv2

from robot.connect import connect_robot
from reachy_mini.utils import create_head_pose

WINDOW = "Reachy Mini - click to look"
MOVE_DURATION = 0.4  # s; look_at_image blocks while the head moves

clicks: list[tuple[int, int]] = []


def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))


with connect_robot(media=True, sleep_on_exit=True) as mini:
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)  # autosize: window pixels == image pixels
    cv2.setMouseCallback(WINDOW, on_mouse)
    print("Click on the video to make the robot look there. 'c' recenter, 'q' quit.")

    try:
        while True:
            frame = mini.media.get_frame()
            if frame is None:
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                continue
            frame = frame.copy()  # SDK frames are read-only
            h, w = frame.shape[:2]

            if clicks:
                u, v = clicks.pop()
                clicks.clear()  # ignore clicks made while the head was moving
                # look_at_image requires a pixel strictly inside the image
                u, v = min(max(u, 1), w - 2), min(max(v, 1), h - 2)
                mini.look_at_image(u, v, duration=MOVE_DURATION)

            # After a click the chosen point should end up under this center cross
            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
            cv2.putText(frame, "click = look | c = center | q = quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                mini.goto_target(head=create_head_pose(), duration=0.6)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
