import time
from pathlib import Path

from tauro.common.robots.so100_follower import (
    SO100FollowerEndEffector,
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

for i in range(10):
    action = action_dict = {
        "delta_x": 0,
        "delta_y": 0,
        "delta_z": 0.1,
    }
    robot.send_action(action)
    time.sleep(1)
