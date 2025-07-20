#!/usr/bin/env python3
"""Simple keyboard teleoperation test that works with current robot setup."""

import argparse
import threading
import time

from tauro_inference.client import RemoteRobot
from tauro_inference.teleop import (
    KeyboardEndEffectorTeleop,
    KeyboardEndEffectorTeleopConfig,
)


def print_controls():
    """Print control instructions."""
    print("\n" + "=" * 50)
    print("Keyboard Teleoperation Controls")
    print("=" * 50)
    print("\nMovement:")
    print("  w/s : Move forward/backward (X)")
    print("  a/d : Move left/right (Y)")
    print("  q/e : Move up/down (Z)")
    print("  r/f : Open/close gripper")
    print("\nOther:")
    print("  ESC : Quit")
    print("=" * 50 + "\n")


def test_keyboard_teleop_simple(robot_address: str, robot_id: str, robot_type: str):
    """Simple keyboard teleoperation test."""
    print("\n=== Simple Keyboard Teleoperation Test ===")

    # Parse address
    host, port = robot_address.split(":")

    # Try to connect with context manager for automatic cleanup
    try:
        with RemoteRobot(
            robot_id=robot_id, robot_type=robot_type, host=host, port=int(port)
        ) as robot:
            print(f"✓ Connected to robot {robot_id}")

            # Get initial observation
            obs = robot.get_observation()
            print("\nAvailable features:")
            for key in sorted(obs.keys()):
                if key.endswith(".pos"):
                    print(f"  {key}: {obs[key]:.2f}")

            # Create keyboard teleop
            config = KeyboardEndEffectorTeleopConfig(
                repeat_delay=0.1,
                magnitude=0.005,  # Small movements
                use_gripper=False,  # Disable gripper to avoid motor 6 issues
            )
            teleop = KeyboardEndEffectorTeleop(config, robot)

            print_controls()

            # Connect keyboard
            teleop.connect()
            print("Keyboard teleoperation active...")

            # Control loop
            running = True

            def check_exit():
                nonlocal running
                while running:
                    if hasattr(teleop, "stop_event") and teleop.stop_event.is_set():
                        running = False
                        break
                    time.sleep(0.1)

            # Start exit checker thread
            exit_thread = threading.Thread(target=check_exit)
            exit_thread.daemon = True
            exit_thread.start()

            while running:
                try:
                    # Get action from keyboard
                    action = teleop.get_action()

                    if action and any(
                        v != 0 for v in action.values() if isinstance(v, int | float)
                    ):
                        # Simple mapping: just move joints based on deltas
                        joint_commands = {}

                        if action.get("delta_x", 0) != 0:
                            # X motion -> shoulder_pan
                            current = obs.get("shoulder_pan.pos", 0)
                            joint_commands["shoulder_pan.pos"] = current + action["delta_x"] * 50

                        if action.get("delta_y", 0) != 0:
                            # Y motion -> elbow_flex
                            current = obs.get("elbow_flex.pos", 0)
                            joint_commands["elbow_flex.pos"] = current + action["delta_y"] * 50

                        if action.get("delta_z", 0) != 0:
                            # Z motion -> shoulder_lift
                            current = obs.get("shoulder_lift.pos", 0)
                            joint_commands["shoulder_lift.pos"] = current - action["delta_z"] * 50

                        if joint_commands:
                            print(f"\rSending: {joint_commands}", end="", flush=True)
                            robot.send_action(joint_commands)

                            # Update observation
                            obs = robot.get_observation()

                    time.sleep(0.05)

                except KeyboardInterrupt:
                    print("\n\nInterrupted by user")
                    running = False
                    break
                except Exception as e:
                    print(f"\nError in control loop: {e}")
                    time.sleep(0.1)

            # Disconnect
            teleop.disconnect()
            print("\n✓ Teleoperation ended")

    except Exception as e:
        print(f"\n✗ Connection error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Simple keyboard teleoperation test")
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

    args = parser.parse_args()

    test_keyboard_teleop_simple(args.robot_address, args.robot_id, args.robot_type)


if __name__ == "__main__":
    main()
