#!/usr/bin/env python3
"""Test script for keyboard teleoperation with the refactored codebase."""

import argparse
import time

from tauro_inference.client import RemoteRobot
from tauro_inference.teleop import (
    KeyboardEndEffectorTeleop,
    KeyboardEndEffectorTeleopConfig,
    KeyboardTeleop,
    KeyboardTeleopConfig,
)


def test_joint_space_teleop(robot_address: str, robot_id: str, robot_type: str):
    """Test joint space keyboard teleoperation."""
    print("\n=== Joint Space Keyboard Teleoperation Test ===")

    # Connect to robot
    host, port = robot_address.split(":")
    robot = RemoteRobot(robot_id=robot_id, robot_type=robot_type, host=host, port=int(port))

    try:
        print(f"Connecting to robot at {robot_address}...")
        robot.connect()
        print("✓ Robot connected")

        # Create keyboard teleop
        config = KeyboardTeleopConfig(
            repeat_delay=0.1,
            magnitude=5.0,  # degrees
        )
        teleop = KeyboardTeleop(config, robot)

        print("\nStarting keyboard teleoperation...")
        print("Press ESC to exit")
        print("\nNote: Joint space control not fully implemented yet")

        teleop.connect()

        # Run for a short time
        time.sleep(5)

        teleop.disconnect()
        print("\n✓ Joint space teleop test completed")

    finally:
        robot.disconnect()


def test_ee_space_teleop(robot_address: str, robot_id: str, robot_type: str):
    """Test end-effector space keyboard teleoperation."""
    print("\n=== End-Effector Space Keyboard Teleoperation Test ===")

    # Connect to robot
    host, port = robot_address.split(":")
    robot = RemoteRobot(robot_id=robot_id, robot_type=robot_type, host=host, port=int(port))

    try:
        print(f"Connecting to robot at {robot_address}...")
        robot.connect()
        print("✓ Robot connected")

        # Create EE keyboard teleop
        config = KeyboardEndEffectorTeleopConfig(
            repeat_delay=0.1,
            magnitude=0.01,  # 1cm steps
            use_gripper=True,
        )
        teleop = KeyboardEndEffectorTeleop(config, robot)

        print("\nStarting EE keyboard teleoperation...")
        print("\nControls:")
        print("  w/s: Move forward/backward (X)")
        print("  a/d: Move left/right (Y)")
        print("  q/e: Move up/down (Z)")
        print("  r/f: Open/close gripper")
        print("  ESC: Exit")

        teleop.connect()

        # Main control loop
        print("\nTeleoperation active. Use keyboard to control...")
        start_time = time.time()

        while time.time() - start_time < 30:  # Run for 30 seconds max
            # Get action from keyboard
            action = teleop.get_action()

            if action:
                # Convert EE deltas to joint commands
                # This is a simplified version - in practice you'd use proper IK
                joint_action = {}

                if "delta_x" in action and action["delta_x"] != 0:
                    # X motion affects shoulder_pan
                    current_obs = robot.get_observation()
                    if "shoulder_pan.pos" in current_obs:
                        joint_action["shoulder_pan.pos"] = (
                            current_obs["shoulder_pan.pos"] + action["delta_x"] * 100
                        )

                if "delta_y" in action and action["delta_y"] != 0:
                    # Y motion affects shoulder_lift and elbow
                    current_obs = robot.get_observation()
                    if "shoulder_lift.pos" in current_obs:
                        joint_action["shoulder_lift.pos"] = (
                            current_obs["shoulder_lift.pos"] - action["delta_y"] * 100
                        )
                    if "elbow_flex.pos" in current_obs:
                        joint_action["elbow_flex.pos"] = (
                            current_obs["elbow_flex.pos"] + action["delta_y"] * 50
                        )

                if "delta_z" in action and action["delta_z"] != 0:
                    # Z motion affects shoulder_lift and elbow
                    current_obs = robot.get_observation()
                    if "shoulder_lift.pos" in current_obs:
                        joint_action["shoulder_lift.pos"] = (
                            current_obs["shoulder_lift.pos"] - action["delta_z"] * 50
                        )
                    if "elbow_flex.pos" in current_obs:
                        joint_action["elbow_flex.pos"] = (
                            current_obs["elbow_flex.pos"] - action["delta_z"] * 100
                        )

                if "gripper" in action:
                    # Gripper control: 0=close, 1=stay, 2=open
                    if action["gripper"] == 0:
                        joint_action["gripper.pos"] = -50.0  # Close
                    elif action["gripper"] == 2:
                        joint_action["gripper.pos"] = 50.0  # Open

                # Send joint commands
                if joint_action:
                    robot.send_action(joint_action)
                    print(f"\rAction sent: {joint_action}", end="", flush=True)

            time.sleep(0.05)  # Small delay

        teleop.disconnect()
        print("\n\n✓ EE space teleop test completed")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        robot.disconnect()


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test keyboard teleoperation")
    parser.add_argument(
        "--robot-address",
        type=str,
        default="localhost:50051",
        help="Robot server address (default: localhost:50051)",
    )
    parser.add_argument(
        "--robot-id", type=str, default="robot_001", help="Robot ID (default: robot_001)"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="so100_follower",
        help="Robot type (default: so100_follower)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["joint", "ee", "both"],
        default="ee",
        help="Test mode: joint space, EE space, or both (default: ee)",
    )

    args = parser.parse_args()

    print("=== Keyboard Teleoperation Test ===")
    print(f"Robot: {args.robot_id} ({args.robot_type})")
    print(f"Address: {args.robot_address}")
    print(f"Mode: {args.mode}")

    if args.mode in ["joint", "both"]:
        test_joint_space_teleop(args.robot_address, args.robot_id, args.robot_type)

    if args.mode in ["ee", "both"]:
        test_ee_space_teleop(args.robot_address, args.robot_id, args.robot_type)

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
