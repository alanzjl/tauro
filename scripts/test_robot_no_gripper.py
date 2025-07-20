#!/usr/bin/env python3
"""Test robot control without gripper motor to diagnose connection issues."""

import time

from tauro_inference.client import RemoteRobot


def test_basic_connection():
    """Test basic robot connection and observation."""
    print("=== Testing Basic Robot Connection ===\n")

    try:
        # Connect to robot
        robot = RemoteRobot(
            robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
        )

        print("Attempting to connect...")
        robot.connect()
        print("✓ Connected successfully!")

        # Get observation
        print("\nGetting observation...")
        obs = robot.get_observation()

        print("\nRobot state:")
        for key in sorted(obs.keys()):
            print(f"  {key}: {obs[key]}")

        # Try a simple motion on working motors
        print("\n\nTesting motor control (excluding gripper)...")

        # Test shoulder pan
        if "shoulder_pan.pos" in obs:
            current_pos = obs["shoulder_pan.pos"]
            new_pos = current_pos + 5  # Move 5 degrees
            print(f"\nMoving shoulder_pan from {current_pos:.1f} to {new_pos:.1f}")

            action = {"shoulder_pan.pos": new_pos}
            success = robot.send_action(action)
            print(f"Command sent: {'✓' if success else '✗'}")

            time.sleep(1)

            # Move back
            print(f"Moving back to {current_pos:.1f}")
            action = {"shoulder_pan.pos": current_pos}
            success = robot.send_action(action)
            print(f"Command sent: {'✓' if success else '✗'}")

        robot.disconnect()
        print("\n✓ Test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


def test_keyboard_control_no_gripper():
    """Test keyboard control without using gripper."""
    print("\n\n=== Testing Keyboard Control (No Gripper) ===\n")

    from tauro_inference.teleop import (
        KeyboardEndEffectorTeleop,
        KeyboardEndEffectorTeleopConfig,
    )

    try:
        # Connect to robot
        robot = RemoteRobot(
            robot_id="robot_001", robot_type="so100_follower", host="127.0.0.1", port=50051
        )

        print("Connecting to robot...")
        robot.connect()
        print("✓ Connected!")

        # Create keyboard teleop without gripper
        config = KeyboardEndEffectorTeleopConfig(
            repeat_delay=0.1,
            magnitude=0.01,
            use_gripper=False,  # Disable gripper
        )

        teleop = KeyboardEndEffectorTeleop(config, robot)

        print("\nStarting keyboard control (5 seconds)...")
        print("Controls: w/s (X), a/d (Y), q/e (Z)")

        teleop.connect()

        # Run for 25 seconds
        start_time = time.time()
        while time.time() - start_time < 25:
            action = teleop.get_action()
            if action and any(action.get(k, 0) != 0 for k in ["delta_x", "delta_y", "delta_z"]):
                print(f"\nAction: {action}")

                # Send end-effector action directly
                # The robot will handle IK internally
                ee_command = {
                    "end_effector": {
                        "delta_x": action.get("delta_x", 0),
                        "delta_y": action.get("delta_y", 0),
                        "delta_z": action.get("delta_z", 0),
                        "gripper": 1.0,  # Maintain gripper position
                    }
                }

                print(f"Sending EE command: {ee_command['end_effector']}")
                success = robot.send_action(ee_command)
                print(f"Command sent: {'✓' if success else '✗'}")

            time.sleep(0.05)

        teleop.disconnect()
        robot.disconnect()

        print("\n\n✓ Keyboard test completed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run diagnostic tests."""
    print("Robot Diagnostic Test\n")
    print("This will test robot connection and control without using the gripper motor.\n")

    # Test basic connection
    test_basic_connection()

    # Test keyboard control
    test_keyboard_control_no_gripper()

    print("\n\n=== All tests completed ===")


if __name__ == "__main__":
    main()
