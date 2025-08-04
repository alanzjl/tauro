#!/usr/bin/env python3
"""
Sample script demonstrating robot control via gRPC.
Shows how to read current robot pose and apply small deltas in both joint and end effector space.
"""

import argparse
import time

from tauro_inference.client import RemoteRobot


def goto_zero_position(robot):
    """Test joint space control by applying small deltas to current positions."""
    print("\n=== Testing Joint Space Control ===")

    # Get current observation
    obs = robot.get_observation()
    print(f"Current observation keys: {list(obs.keys())}")

    print(obs["joints"])

    # Extract joint positions
    joint_positions = {}
    for motor_name, val in obs["joints"]["position"].items():
        joint_positions[motor_name] = 0
        print(f"{motor_name}: {val}")

    target = {}
    for motor_name in joint_positions.keys():
        target[motor_name] = 0.0

    action = {"joints": {"position": target}}
    robot.send_action(action)
    time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description="Test robot control via gRPC")
    parser.add_argument("--robot-id", default="robot_001", help="Robot ID")
    parser.add_argument("--robot-type", default="so100_follower", help="Robot type")
    parser.add_argument("--host", default="127.0.0.1", help="Robot server host")
    parser.add_argument("--port", type=int, default=50051, help="Robot server port")
    args = parser.parse_args()

    # Connect to robot
    print(f"Connecting to robot {args.robot_id} at {args.host}:{args.port}...")

    with RemoteRobot(
        robot_id=args.robot_id, robot_type=args.robot_type, host=args.host, port=args.port
    ) as robot:
        print("Connected successfully!")

        # Note: The robot should automatically load calibration from cache if it exists
        # The calibrate() method will run the full calibration process even if cached
        # calibration exists, so we skip calling it here to use the cached calibration

        # Run requested tests
        goto_zero_position(robot)
        print("\nTest completed!")


if __name__ == "__main__":
    main()
