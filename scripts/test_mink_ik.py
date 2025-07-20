#!/usr/bin/env python3
"""Test script for mink-based inverse kinematics."""

import time

import numpy as np

from tauro_inference.client import RemoteRobot


def test_mink_ik():
    """Test the mink IK implementation with various movements."""
    print("=== Testing Mink-based Inverse Kinematics ===\n")

    # Connect to robot
    with RemoteRobot(
        robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
    ) as robot:
        print("Connected to robot")

        # Get initial observation
        obs = robot.get_observation()

        # Print initial end-effector position
        if "end_effector" in obs and "position" in obs["end_effector"]:
            ee_pos = obs["end_effector"]["position"]
            print(f"Initial EE position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Test 1: Small movements in each axis
        print("\n--- Test 1: Small movements in each axis ---")
        test_movements = [
            {"delta_x": 0.02, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": -0.02, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": 0.0, "delta_y": 0.02, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": 0.0, "delta_y": -0.02, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.02, "gripper": 1.0},
            {"delta_x": 0.0, "delta_y": 0.0, "delta_z": -0.02, "gripper": 1.0},
        ]

        for i, move in enumerate(test_movements):
            print(
                f"\nMovement {i+1}: dx={move['delta_x']}, dy={move['delta_y']}, dz={move['delta_z']}"
            )

            # Send end-effector command
            action = {"end_effector": move}
            success = robot.send_action(action)
            print(f"Command sent: {'✓' if success else '✗'}")

            time.sleep(1.5)

            # Get new position
            obs = robot.get_observation()
            if "end_effector" in obs and "position" in obs["end_effector"]:
                ee_pos = obs["end_effector"]["position"]
                print(f"New position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Test 2: Diagonal movement
        print("\n--- Test 2: Diagonal movement ---")
        diagonal_move = {"delta_x": 0.02, "delta_y": 0.02, "delta_z": 0.01, "gripper": 1.0}
        print(f"Moving diagonally: {diagonal_move}")

        action = {"end_effector": diagonal_move}
        robot.send_action(action)
        time.sleep(2.0)

        obs = robot.get_observation()
        if "end_effector" in obs and "position" in obs["end_effector"]:
            ee_pos = obs["end_effector"]["position"]
            print(f"Final position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Test 3: Circular motion
        print("\n--- Test 3: Circular motion in XY plane ---")
        radius = 0.02
        steps = 8

        for i in range(steps):
            angle = (i / steps) * 2 * np.pi
            dx = radius * (np.cos(angle + np.pi / steps) - np.cos(angle))
            dy = radius * (np.sin(angle + np.pi / steps) - np.sin(angle))

            move = {"delta_x": dx, "delta_y": dy, "delta_z": 0.0, "gripper": 1.0}
            action = {"end_effector": move}

            robot.send_action(action)
            print(f"Step {i+1}/{steps}: dx={dx:.3f}, dy={dy:.3f}")
            time.sleep(0.5)

        print("\n✓ All tests completed!")


def test_ik_limits():
    """Test IK behavior at workspace limits."""
    print("\n=== Testing IK at Workspace Limits ===\n")

    with RemoteRobot(
        robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
    ) as robot:
        print("Connected to robot")

        # Test large movements that might hit limits
        print("\nTesting large movements (may hit workspace limits)...")

        large_moves = [
            {"delta_x": 0.1, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": -0.1, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.1, "gripper": 1.0},
        ]

        for move in large_moves:
            print(f"\nAttempting large move: {move}")
            action = {"end_effector": move}

            # Get position before
            obs_before = robot.get_observation()
            ee_before = obs_before.get("end_effector", {}).get("position", [0, 0, 0])

            # Send command
            robot.send_action(action)
            time.sleep(2.0)

            # Get position after
            obs_after = robot.get_observation()
            ee_after = obs_after.get("end_effector", {}).get("position", [0, 0, 0])

            # Calculate actual movement
            actual_dx = ee_after[0] - ee_before[0]
            actual_dy = ee_after[1] - ee_before[1]
            actual_dz = ee_after[2] - ee_before[2]

            print(f"Requested: dx={move['delta_x']}, dy={move['delta_y']}, dz={move['delta_z']}")
            print(f"Actual:    dx={actual_dx:.3f}, dy={actual_dy:.3f}, dz={actual_dz:.3f}")

            if (
                abs(actual_dx - move["delta_x"]) > 0.01
                or abs(actual_dy - move["delta_y"]) > 0.01
                or abs(actual_dz - move["delta_z"]) > 0.01
            ):
                print("⚠️  Movement was limited (likely hit workspace boundary)")

        print("\n✓ Limit test completed!")


def main():
    """Run all IK tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Test mink-based IK implementation")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Robot server host")
    parser.add_argument("--port", type=int, default=50051, help="Robot server port")
    parser.add_argument(
        "--test",
        type=str,
        choices=["basic", "limits", "all"],
        default="all",
        help="Which test to run",
    )

    args = parser.parse_args()

    try:
        if args.test in ["basic", "all"]:
            test_mink_ik()

        if args.test in ["limits", "all"]:
            test_ik_limits()

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
