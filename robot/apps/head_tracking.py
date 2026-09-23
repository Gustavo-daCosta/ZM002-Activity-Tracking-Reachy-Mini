"""Daemon-side face tracking, printed in the terminal.

Run: reachy_mini_env/bin/python -m robot.apps.head_tracking
"""

import time

from robot.connect import connect_robot

# media=False: tracking runs on the robot; no video needed on this computer.
with connect_robot(media=False, needs=("tracking",), sleep_on_exit=True) as mini:
    mini.start_head_tracking(weight=1.0)
    print("Head tracking ON. Stand in front of the camera. Ctrl+C to stop.")
    try:
        while True:
            face = mini.get_tracked_face()
            if face.detected:
                print(f"Face: x={face.x:+.2f}  y={face.y:+.2f}  roll={face.roll or 0.0:+.2f}")
            elif face.ts is None:
                print("Waiting for the camera feed...")
            else:
                print("No face detected")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
