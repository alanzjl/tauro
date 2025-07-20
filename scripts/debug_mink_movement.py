#!/usr/bin/env python3
"""Debug script to understand why robot isn't moving with mink IK."""

import time

from tauro_inference.client import RemoteRobot


def main():
    """Debug mink IK movement."""
    print("=== Debugging Mink IK Movement ===\n")

    with RemoteRobot(
        robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
    ) as robot:
        print("Connected to robot")

        # Get initial observation
        obs = robot.get_observation()
        print("\nInitial observation:")
        print(f"  Joint positions: {obs['joints']['position']}")
        if "end_effector" in obs:
            ee_pos = obs["end_effector"]["position"]
            print(
                f"  End effector position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}"
            )

        # Test 1: Send joint command directly
        print("\n--- Test 1: Direct joint command ---")
        joint_action = {
            "shoulder_pan.pos": 10.0,
            "shoulder_lift.pos": -10.0,
            "elbow_flex.pos": 20.0,
            "wrist_flex.pos": -10.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 30.0,
        }
        print(f"Sending joint action: {joint_action}")
        success = robot.send_action(joint_action)
        print(f"Action sent: {'✓' if success else '✗'}")

        time.sleep(2)

        # Check new position
        obs = robot.get_observation()
        print(f"\nNew joint positions: {obs['joints']['position']}")
        if "end_effector" in obs:
            ee_pos = obs["end_effector"]["position"]
            print(f"New EE position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Test 2: Send end effector command
        print("\n--- Test 2: End effector command (large movement) ---")
        ee_action = {
            "end_effector": {
                "delta_x": 0.05,  # 5cm
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 1.0,
            }
        }
        print(f"Sending EE action: {ee_action}")
        success = robot.send_action(ee_action)
        print(f"Action sent: {'✓' if success else '✗'}")

        time.sleep(2)

        # Check final position
        obs = robot.get_observation()
        print(f"\nFinal joint positions: {obs['joints']['position']}")
        if "end_effector" in obs:
            ee_pos = obs["end_effector"]["position"]
            print(f"Final EE position: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

        # Return to home
        print("\n--- Returning to home position ---")
        home_action = {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 30.0,
        }
        robot.send_action(home_action)
        time.sleep(2)

        print("\n✓ Debug complete!")


if __name__ == "__main__":
    main()
