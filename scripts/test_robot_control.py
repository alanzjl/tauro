#!/usr/bin/env python3
"""
Sample script demonstrating robot control via gRPC.
Shows how to read current robot pose and apply small deltas in both joint and end effector space.
"""

import argparse
import time

from tauro_inference.client import RemoteRobot


def test_joint_control(robot):
    """Test joint space control by applying small deltas to current positions."""
    print("\n=== Testing Joint Space Control ===")

    # Get current observation
    obs = robot.get_observation()
    print(f"Current observation keys: {list(obs.keys())}")

    print(obs["joints"])

    # Extract joint positions
    joint_positions = {}
    for motor_name, val in obs["joints"]["position"].items():
        joint_positions[motor_name] = val
        print(f"{motor_name}: {val}")

    # Apply small delta to each joint
    print("\nApplying small delta to joints...")
    delta = 5.0  # degrees or normalized units depending on robot config

    for motor_name, current_pos in joint_positions.items():
        if motor_name != "gripper":  # Skip gripper for safety
            new_pos = current_pos + delta
            action = {f"{motor_name}.pos": new_pos}
            print(f"Moving {motor_name} from {current_pos} to {new_pos}")
            robot.send_action(action)
            time.sleep(0.5)  # Small delay between movements

    # Return to original position
    print("\nReturning to original positions...")
    for motor_name, original_pos in joint_positions.items():
        if motor_name != "gripper":
            action = {f"{motor_name}.pos": original_pos}
            robot.send_action(action)
            time.sleep(0.5)


def test_end_effector_control(robot, control_gripper=True):
    """Test end effector space control by applying small Cartesian deltas."""
    print("\n=== Testing End Effector Space Control ===")
    if not control_gripper:
        print("Note: Gripper control disabled to avoid overload errors")

    # Get current observation
    obs = robot.get_observation()

    # Check if end effector state is available
    if "end_effector" in obs and "position" in obs["end_effector"]:
        ee_pos = obs["end_effector"]["position"]
        print(
            f"Current end effector position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}"
        )
    else:
        print("End effector state not available.")
        print(f"Available observation keys: {list(obs.keys())}")
        if "end_effector" in obs:
            print(f"End effector keys: {list(obs['end_effector'].keys())}")
        return

    # Define test movements
    # Note: gripper value of 1.0 means "maintain current position"
    # Use 0.0 to close, 2.0 to open relative to current position
    movements = [
        {"delta_x": 0.02, "delta_y": 0.0, "delta_z": 0.0},  # Move 2cm in +X
        {"delta_x": -0.02, "delta_y": 0.0, "delta_z": 0.0},  # Move back
        {"delta_x": 0.0, "delta_y": 0.02, "delta_z": 0.0},  # Move 2cm in +Y
        {"delta_x": 0.0, "delta_y": -0.02, "delta_z": 0.0},  # Move back
        {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.02},  # Move 2cm in +Z
        {"delta_x": 0.0, "delta_y": 0.0, "delta_z": -0.02},  # Move back
    ]

    # Add gripper control if enabled
    if control_gripper:
        for movement in movements:
            movement["gripper"] = 1.0

    for i, movement in enumerate(movements):
        direction = []
        if movement["delta_x"] != 0:
            direction.append(f"X={movement['delta_x']:+.3f}")
        if movement["delta_y"] != 0:
            direction.append(f"Y={movement['delta_y']:+.3f}")
        if movement["delta_z"] != 0:
            direction.append(f"Z={movement['delta_z']:+.3f}")

        print(f"\nMovement {i + 1}: {', '.join(direction)}")

        # Send end effector command
        action = {"end_effector": movement}
        robot.send_action(action)

        time.sleep(1.0)  # Wait for movement to complete

        # Read new position
        try:
            obs = robot.get_observation()
            if "end_effector" in obs and "position" in obs["end_effector"]:
                ee_pos = obs["end_effector"]["position"]
                print(f"New position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")
        except Exception as e:
            print(f"Error reading position after movement: {e}")
            print("Note: Motor overload errors may occur if the gripper is physically obstructed")
            break


def test_gripper_control(robot):
    """Test gripper control."""
    print("\n=== Testing Gripper Control ===")

    # Get current gripper position
    obs = robot.get_observation()
    if "gripper.pos" in obs:
        current_gripper = obs["gripper.pos"]
        print(f"Current gripper position: {current_gripper:.3f}")
    else:
        print("Gripper position not found")
        return

    # Open gripper
    print("Opening gripper...")
    robot.send_action({"gripper.pos": 50.0})  # Adjust max value based on robot
    time.sleep(1.0)

    # Close gripper
    print("Closing gripper...")
    robot.send_action({"gripper.pos": 5.0})  # Minimum safe value
    time.sleep(1.0)

    # Return to original position
    print("Returning to original position...")
    robot.send_action({"gripper.pos": current_gripper})
    time.sleep(1.0)


def main():
    parser = argparse.ArgumentParser(description="Test robot control via gRPC")
    parser.add_argument("--robot-id", default="robot_001", help="Robot ID")
    parser.add_argument("--robot-type", default="so100_follower", help="Robot type")
    parser.add_argument("--host", default="127.0.0.1", help="Robot server host")
    parser.add_argument("--port", type=int, default=50051, help="Robot server port")
    parser.add_argument(
        "--mode",
        choices=["joint", "end_effector", "gripper", "all"],
        default="all",
        help="Control mode to test",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Disable gripper control in end effector mode to avoid overload errors",
    )
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
        if args.mode == "joint" or args.mode == "all":
            test_joint_control(robot)

        if args.mode == "end_effector" or args.mode == "all":
            test_end_effector_control(robot, control_gripper=not args.no_gripper)

        if args.mode == "gripper" or args.mode == "all":
            test_gripper_control(robot)

        print("\nTest completed!")


if __name__ == "__main__":
    main()
