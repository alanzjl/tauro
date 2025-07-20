#!/usr/bin/env python3
"""Interactive test for sshkeyboard functionality."""

import sshkeyboard


def test_direct_sshkeyboard():
    """Test sshkeyboard directly to ensure it works in this environment."""
    print("Testing sshkeyboard directly")
    print("-" * 40)
    print("Press some keys (ESC to quit):")
    print("This test will show if sshkeyboard works in your terminal")

    pressed_keys = set()

    def on_press(key):
        pressed_keys.add(key)
        print(f"Key pressed: '{key}' | Currently pressed: {pressed_keys}")

    def on_release(key):
        if key in pressed_keys:
            pressed_keys.remove(key)
        print(f"Key released: '{key}' | Currently pressed: {pressed_keys}")

    try:
        sshkeyboard.listen_keyboard(on_press=on_press, on_release=on_release, until="esc")
    except Exception as e:
        print(f"\nError: {e}")
        print(
            "\nMake sure you're running this in an interactive terminal (not through a script runner)"
        )
        print("Try running directly: python scripts/test_keyboard_interactive.py")


if __name__ == "__main__":
    test_direct_sshkeyboard()
