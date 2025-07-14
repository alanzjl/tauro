#!/usr/bin/env python3
"""Simple debug test for robot control."""

import time

from tauro_inference.client import RemoteRobot


def main():
    print("=== Robot Control Debug Test ===")

    # Connect to robot
    print("\n1. Connecting to robot...")
    with RemoteRobot(
        robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
    ) as robot:
        print("✓ Connected successfully!")

        # Get initial state
        print("\n2. Reading robot state...")
        obs = robot.get_observation()
        print(f"✓ Got {len(obs)} observation values")

        # Show joint positions
        print("\n3. Current joint positions:")
        for key in sorted(obs.keys()):
            if key.endswith(".pos"):
                print(f"   {key}: {obs[key]:.3f}")

        # Send a single joint command
        print("\n4. Sending test command to shoulder_pan...")
        current_pos = obs.get("shoulder_pan.pos", 0)
        new_pos = current_pos + 10
        action = {"shoulder_pan.pos": new_pos}

        print(f"   Current position: {current_pos:.3f}")
        print(f"   Target position: {new_pos:.3f}")

        success = robot.send_action(action)
        print(f"   Command sent: {'✓ Success' if success else '✗ Failed'}")

        # Wait and read new state
        print("\n5. Waiting 1 second...")
        time.sleep(1)

        print("\n6. Reading new state...")
        new_obs = robot.get_observation()
        new_shoulder_pos = new_obs.get("shoulder_pan.pos", 0)
        print(f"   New shoulder_pan position: {new_shoulder_pos:.3f}")
        print(f"   Position changed: {abs(new_shoulder_pos - current_pos) > 0.1}")

        # Return to original position
        print("\n7. Returning to original position...")
        robot.send_action({"shoulder_pan.pos": current_pos})
        print("✓ Command sent")

    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()
