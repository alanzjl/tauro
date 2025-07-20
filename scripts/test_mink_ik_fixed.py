#!/usr/bin/env python3
"""Test mink IK with correct step size scaling."""

import time

import numpy as np

from tauro_inference.client import RemoteRobot


def main():
    """Test mink IK with proper scaling."""
    print("=== Testing Mink IK with Correct Scaling ===\n")

    with RemoteRobot(
        robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
    ) as robot:
        print("Connected to robot")

        # Get initial observation
        obs = robot.get_observation()
        if "end_effector" in obs:
            ee_pos = obs["end_effector"]["position"]
            print(f"Initial EE position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Test movements with correct scaling
        # Step size is 0.02m, so to move 4cm we need delta=2
        print("\n--- Testing scaled movements ---")
        test_moves = [
            {"name": "Forward 4cm", "delta_x": 2.0, "delta_y": 0.0, "delta_z": 0.0},
            {"name": "Back 4cm", "delta_x": -2.0, "delta_y": 0.0, "delta_z": 0.0},
            {"name": "Right 4cm", "delta_y": 2.0, "delta_x": 0.0, "delta_z": 0.0},
            {"name": "Left 4cm", "delta_y": -2.0, "delta_x": 0.0, "delta_z": 0.0},
            {"name": "Up 2cm", "delta_z": 1.0, "delta_x": 0.0, "delta_y": 0.0},
            {"name": "Down 2cm", "delta_z": -1.0, "delta_x": 0.0, "delta_y": 0.0},
        ]

        for move in test_moves:
            print(f"\n{move['name']}:")

            # Get position before
            obs_before = robot.get_observation()
            ee_before = obs_before["end_effector"]["position"]

            # Send action
            action = {
                "end_effector": {
                    "delta_x": move["delta_x"],
                    "delta_y": move["delta_y"],
                    "delta_z": move["delta_z"],
                    "gripper": 1.0,
                }
            }
            success = robot.send_action(action)
            print(f"  Command sent: {'✓' if success else '✗'}")

            time.sleep(1.5)

            # Get position after
            obs_after = robot.get_observation()
            ee_after = obs_after["end_effector"]["position"]

            # Calculate actual movement
            actual_dx = ee_after[0] - ee_before[0]
            actual_dy = ee_after[1] - ee_before[1]
            actual_dz = ee_after[2] - ee_before[2]

            print(f"  Before: [{ee_before[0]:.3f}, {ee_before[1]:.3f}, {ee_before[2]:.3f}]")
            print(f"  After:  [{ee_after[0]:.3f}, {ee_after[1]:.3f}, {ee_after[2]:.3f}]")
            print(f"  Actual movement: dx={actual_dx:.3f}, dy={actual_dy:.3f}, dz={actual_dz:.3f}")

        # Test circular motion
        print("\n--- Testing circular motion ---")
        radius_steps = 2.0  # 4cm radius in step units
        n_points = 8

        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            next_angle = 2 * np.pi * (i + 1) / n_points

            dx = radius_steps * (np.cos(next_angle) - np.cos(angle))
            dy = radius_steps * (np.sin(next_angle) - np.sin(angle))

            action = {
                "end_effector": {"delta_x": dx, "delta_y": dy, "delta_z": 0.0, "gripper": 1.0}
            }

            robot.send_action(action)
            print(f"  Step {i+1}/{n_points}")
            time.sleep(0.5)

        print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
