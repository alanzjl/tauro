#!/usr/bin/env python3
"""Test end-effector control without gripper to avoid overload errors."""

import time

from tauro_inference.client import RemoteRobot


def main():
    print("Testing end-effector control without gripper movement...\n")

    with RemoteRobot(
        robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
    ) as robot:
        print("Connected to robot")

        # Get initial state
        obs = robot.get_observation()
        if "end_effector" in obs and "position" in obs["end_effector"]:
            ee_pos = obs["end_effector"]["position"]
            print(f"Initial position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Test movements with gripper=1.0 (maintain current position)
        movements = [
            {"delta_x": 0.01, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0},
            {"delta_x": -0.01, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0},
        ]

        for i, move in enumerate(movements):
            print(f"\nTest {i + 1}: Moving {move}")
            action = {"end_effector": move}

            try:
                robot.send_action(action)
                time.sleep(2.0)

                obs = robot.get_observation()
                if "end_effector" in obs and "position" in obs["end_effector"]:
                    ee_pos = obs["end_effector"]["position"]
                    print(f"New position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")
            except Exception as e:
                print(f"Error: {e}")
                break

        print("\nTest completed!")


if __name__ == "__main__":
    main()
