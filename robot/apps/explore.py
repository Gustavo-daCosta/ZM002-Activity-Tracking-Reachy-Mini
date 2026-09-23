"""Tour of the main SDK features. Run: reachy_mini_env/bin/python -m robot.apps.explore"""

import numpy as np

from robot.connect import connect_robot
from reachy_mini.motion.recorded_move import RecordedMoves
from reachy_mini.utils import create_head_pose

with connect_robot(media=False, sleep_on_exit=True) as mini:
    # 1. Read state (API: /api/state/*)
    np.set_printoptions(precision=3, suppress=True)
    print("Head pose (4x4):\n", mini.get_current_head_pose())
    print("Antennas (rad):", mini.get_present_antenna_joint_positions())
    head_joints, antenna_joints = mini.get_current_joint_positions()
    print("Head joints:", head_joints)

    # 2. Head: position in mm, angles in degrees (API: /api/move/goto)
    print("Head: look up, down, tilt, turn...")
    mini.goto_target(head=create_head_pose(pitch=-15), duration=1.0)
    mini.goto_target(head=create_head_pose(pitch=15), duration=1.0)
    mini.goto_target(head=create_head_pose(roll=20), duration=1.0)
    mini.goto_target(head=create_head_pose(yaw=30), duration=1.0)
    mini.goto_target(head=create_head_pose(z=10, mm=True), duration=1.0)
    mini.goto_target(head=create_head_pose(), duration=1.0)

    # 3. Body rotation (radians)
    print("Body: turn left and right...")
    mini.goto_target(body_yaw=np.deg2rad(30), duration=1.0)
    mini.goto_target(body_yaw=np.deg2rad(-30), duration=1.0)
    mini.goto_target(body_yaw=0.0, duration=1.0)

    # 4. Everything at once
    mini.goto_target(
        head=create_head_pose(yaw=20, pitch=-10),
        antennas=[0.6, 0.6],
        body_yaw=np.deg2rad(15),
        duration=1.5,
    )

    # 5. Look at a point in space (meters, robot frame: x forward, y left, z up)
    print("Look at a point in front / to the side...")
    mini.look_at_world(0.5, 0.2, 0.1, duration=1.0)
    mini.look_at_world(0.5, -0.2, 0.0, duration=1.0)

    # 6. Recorded emotions (API: /api/move/play/recorded-move-dataset/...)
    # Downloads the dataset from HuggingFace on first run.
    print("Playing recorded emotions...")
    emotions = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    for name in ["cheerful1", "curious1", "yes_sad1"]:
        print("  ->", name)
        mini.play_move(emotions.get(name), initial_goto_duration=1.0, sound=False)

    # 7. Gravity compensation is skipped: it needs the Placo kinematics engine and this
    #    robot's daemon runs AnalyticalKinematics (the daemon answers with an error).

    print("Done! (going to sleep)")
