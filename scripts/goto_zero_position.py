#!/usr/bin/env python3
"""
Sample script demonstrating robot control via gRPC.
Shows how to read current robot pose and apply small deltas in both joint and end effector space.
"""

import argparse
import time

from tauro_inference.client import RobotClient


def print_joint_positions(robot):
    print("Current joint positions:")
    state = robot.get_robot_state()
    joint_positions = {k: v.position for k, v in state.joints.items()}
    print(joint_positions)


def goto_zero_position(robot):
    """Test joint space control by applying small deltas to current positions."""
    print("\n=== Testing Joint Space Control ===")

    # Get current observation
    state = robot.get_robot_state()

    # Extract joint positions
    joint_positions = {}
    for motor_name, val in state.joints.items():
        joint_positions[motor_name] = val.position
    print_joint_positions(robot)

    # Send action to move to zero position
    time.sleep(1)
    action = {"joints": {"position": {k: 0.0 for k in joint_positions.keys()}}}
    robot.send_action(action)
    print_joint_positions(robot)
    time.sleep(3)

    # Go to a random position
    action = {"joints": {"position": {k: 20.0 for k in joint_positions.keys()}}}
    robot.send_action(action)
    print_joint_positions(robot)
    time.sleep(3)

    # Go to a random position
    action = {"joints": {"position": {k: -20.0 for k in joint_positions.keys()}}}
    robot.send_action(action)
    print_joint_positions(robot)
    time.sleep(3)

    # Go back to the home position
    robot.goto_home_position()
    time.sleep(3)


async def goto_zero_position_async(robot):
    """Test joint space control by applying small deltas to current positions."""
    print("\n=== Testing Joint Space Control ===")
    state = await robot.get_robot_state()
    joint_positions = {k: v.position for k, v in state.joints.items()}
    print(joint_positions)

    # Send action to move to zero position
    time.sleep(1)
    action = {"joints": {"position": {k: 0.0 for k in joint_positions.keys()}}}
    await robot.send_action(action)
    time.sleep(1)


def main(args):
    # Connect to robot
    print(f"Connecting to robot {args.robot_id} at {args.host}:{args.port}...")

    with RobotClient(
        robot_id=args.robot_id, robot_type=args.robot_type, host=args.host, port=args.port
    ) as robot:
        print("Connected successfully!")

        # Note: The robot should automatically load calibration from cache if it exists
        # The calibrate() method will run the full calibration process even if cached
        # calibration exists, so we skip calling it here to use the cached calibration

        # Run requested tests
        goto_zero_position(robot)
        print("\nTest completed!")


async def main_async(args):
    async with RobotClient(
        robot_id=args.robot_id, robot_type=args.robot_type, host=args.host, port=args.port
    ) as robot:
        print("Connected successfully!")
        await goto_zero_position_async(robot)
        print("\nTest completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test robot control via gRPC")
    parser.add_argument("--robot-id", default="robot_001", help="Robot ID")
    parser.add_argument("--robot-type", default="so100_follower", help="Robot type")
    parser.add_argument("--host", default="127.0.0.1", help="Robot server host")
    parser.add_argument("--port", type=int, default=50051, help="Robot server port")
    args = parser.parse_args()

    main(args)
    time.sleep(1)
    # asyncio.run(main_async(args))
