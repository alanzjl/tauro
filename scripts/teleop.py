import time
from pathlib import Path

from tauro.common.robots.so100_follower.so100_follower_end_effector import (
    SO100FollowerEndEffector,
)
from tauro.common.robots.so100_follower.config_so100_follower import (
    SO100FollowerEndEffectorConfig,
)
from tauro.common.teleoperators.keyboard import (
    KeyboardEndEffectorTeleop,
    KeyboardEndEffectorTeleopConfig,
)

calibration_dir = (
    Path(__file__).parent.parent / ".calibrations" / "robots" / "so100_follower"
)


robot = SO100FollowerEndEffector(
    config=SO100FollowerEndEffectorConfig(
        port="/dev/ttyACM0", id="right", calibration_dir=calibration_dir
    )
)
teleop = KeyboardEndEffectorTeleop(
    config=KeyboardEndEffectorTeleopConfig(use_gripper=True)
)

robot.connect()
teleop.connect()

while True:
    action = teleop.get_action()
    robot.send_action(action)
    time.sleep(0.1)
