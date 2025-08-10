#!/usr/bin/env python3
"""
Script to calibrate a robot via gRPC.
"""

import argparse
import logging

from tauro_inference.client import RobotClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calibrate_robot(robot: RobotClient):
    """Calibrate the robot and display calibration data."""
    print(f"\n=== Calibrating Robot {robot.robot_id} ===")

    # Perform calibration
    calibration_data = robot.calibrate()

    if calibration_data:
        print("\nCalibration successful!")
        print("\nCalibration data:")
        print("-" * 50)

        for motor_name, calibration in calibration_data.items():
            print(f"\nMotor: {motor_name}")
            print(f"  Homing offset: {calibration.homing_position}")
            print(f"  Range: [{calibration.min_position}, {calibration.max_position}]")
            if hasattr(calibration, "zero_position"):
                print(f"  Zero position: {calibration.zero_position}")

        print("\n" + "=" * 50)
        return True
    else:
        print("\nCalibration failed!")
        return False


def main(args):
    # Connect to robot
    print(f"Connecting to robot {args.robot_id} at {args.host}:{args.port}...")

    with RobotClient(
        robot_id=args.robot_id, robot_type=args.robot_type, host=args.host, port=args.port
    ) as robot:
        print("Connected successfully!")

        # Calibrate the robot
        success = calibrate_robot(robot)

        if success:
            print("\nCalibration completed successfully!")
        else:
            print("\nCalibration failed. Please check the robot and try again.")
            exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a robot via gRPC")
    parser.add_argument("--robot-id", default="robot_001", help="Robot ID")
    parser.add_argument("--robot-type", default="so100_follower", help="Robot type")
    parser.add_argument("--host", default="127.0.0.1", help="Robot server host")
    parser.add_argument("--port", type=int, default=50051, help="Robot server port")

    args = parser.parse_args()
    main(args)
