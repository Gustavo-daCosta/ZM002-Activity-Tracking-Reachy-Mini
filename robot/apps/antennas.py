"""Wiggle the antennas. Run: reachy_mini_env/bin/python -m robot.apps.antennas"""

from robot.connect import connect_robot

with connect_robot(media=False) as mini:
    print("Wiggling antennas...")
    mini.goto_target(antennas=[0.5, -0.5], duration=0.5)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.5)
    mini.goto_target(antennas=[0, 0], duration=0.5)
    print("Done!")
