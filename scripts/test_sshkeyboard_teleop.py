#!/usr/bin/env python3
"""Test script for the sshkeyboard-based KeyboardTeleop."""

import logging
import time
from tauro.common.teleoperators.keyboard.teleop_keyboard import (
    KeyboardEndEffectorTeleop,
)
from tauro.common.teleoperators.keyboard.configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_keyboard_teleop():
    """Test the KeyboardEndEffectorTeleop with sshkeyboard."""
    print("Testing KeyboardEndEffectorTeleop with sshkeyboard")
    print("-" * 40)

    # Create a mock config
    config = KeyboardEndEffectorTeleopConfig(mock=False)

    # Create the teleop instance
    teleop = KeyboardEndEffectorTeleop(config)

    # Test connection
    print("Connecting keyboard teleop...")
    try:
        teleop.connect()
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return

    # Check connection status
    print(f"Is connected: {teleop.is_connected}")

    # Test keyboard input
    print("\nPress some keys to test (press ESC to disconnect):")
    print("Try pressing and releasing different keys...")

    try:
        # Run for a while to capture key events
        start_time = time.time()
        while teleop.is_connected and (time.time() - start_time) < 30:
            try:
                action = teleop.get_action()
                if action:
                    print(f"Active keys: {action}")
            except Exception as e:
                print(f"Error getting action: {e}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Test disconnection
    print("\nDisconnecting...")
    try:
        teleop.disconnect()
        print("✓ Disconnected successfully")
    except Exception as e:
        print(f"✗ Disconnection failed: {e}")

    print(f"Is connected after disconnect: {teleop.is_connected}")
    print("\nTest complete!")


if __name__ == "__main__":
    test_keyboard_teleop()
