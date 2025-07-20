import logging
import sys
import threading
import time
from queue import Queue
from typing import Any

try:
    import sshkeyboard

    SSHKEYBOARD_AVAILABLE = True
except ImportError:
    SSHKEYBOARD_AVAILABLE = False


from tauro_inference.client import RemoteRobot

from ..teleoperator import Teleoperator
from .configuration_keyboard import (
    KeyboardEndEffectorTeleopConfig,
    KeyboardTeleopConfig,
)


class KeyboardTeleop(Teleoperator):
    """
    Teleop class to use keyboard inputs for control.
    """

    config_class = KeyboardTeleopConfig
    name = "keyboard"

    def __init__(self, config: KeyboardTeleopConfig, robot: RemoteRobot | None = None):
        super().__init__(config, robot)
        self.config = config

        self.event_queue = Queue()
        self.listener_thread = None
        self.stop_event = threading.Event()
        self.logs = {}
        self.current_pressed_keys = set()
        self.last_repeat_time = time.monotonic()

    @property
    def action_features(self) -> dict:
        if self.robot:
            obs = self.robot.get_observation()
            motor_names = [k.replace(".pos", "") for k in obs.keys() if k.endswith(".pos")]
            return {
                "dtype": "float32",
                "shape": (len(motor_names),),
                "names": {"motors": motor_names},
            }
        return {
            "dtype": "float32",
            "shape": (0,),
            "names": {"motors": []},
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return (
            self.listener_thread is not None
            and self.listener_thread.is_alive()
            and not self.stop_event.is_set()
        )

    @property
    def is_calibrated(self) -> bool:
        pass

    def connect(self) -> None:
        if self.is_connected:
            logging.warning("Keyboard is already connected.")
            return

        if not SSHKEYBOARD_AVAILABLE:
            logging.warning("sshkeyboard not available - keyboard input disabled")
            return

        # Check if we're in an interactive terminal
        if not sys.stdin.isatty():
            logging.warning("Not running in an interactive terminal - keyboard input disabled")
            logging.info("Run directly in a terminal to enable keyboard control")
            return

        logging.info("Starting sshkeyboard listener.")
        self.stop_event.clear()
        self.listener_thread = threading.Thread(target=self._keyboard_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

    def calibrate(self) -> None:
        pass

    def _keyboard_listener(self):
        """Run sshkeyboard listener in a separate thread."""

        def on_press(key):
            if not self.stop_event.is_set():
                self.event_queue.put((key, True))
                if key == "esc":
                    logging.info("ESC pressed, disconnecting.")
                    self.stop_event.set()
                    sshkeyboard.stop_listening()

        def on_release(key):
            if not self.stop_event.is_set():
                self.event_queue.put((key, False))

        sshkeyboard.listen_keyboard(
            on_press=on_press,
            on_release=on_release,
            until=None,  # Don't stop on any key, we'll use stop_listening() instead
            sequential=False,
        )

    def _drain_pressed_keys(self):
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed_keys.add(key_char)
            else:
                self.current_pressed_keys.remove(key_char)

    def configure(self):
        pass

    def get_action(self) -> dict[str, Any]:
        raise NotImplementedError("KeyboardTeleop does not support get_action()")

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            logging.warning("KeyboardTeleop is not connected.")
            return
        self.stop_event.set()
        if SSHKEYBOARD_AVAILABLE and sys.stdin.isatty():
            try:
                sshkeyboard.stop_listening()
            except Exception:
                pass  # Ignore errors when stopping
        if self.listener_thread is not None:
            self.listener_thread.join(timeout=2.0)


class KeyboardEndEffectorTeleop(KeyboardTeleop):
    """
    Teleop class to use keyboard inputs for end effector control.
    Designed to be used with the `So100FollowerEndEffector` robot.
    """

    config_class = KeyboardEndEffectorTeleopConfig
    name = "keyboard_ee"

    def __init__(self, config: KeyboardEndEffectorTeleopConfig, robot: RemoteRobot | None = None):
        super().__init__(config, robot)
        self.config = config
        self.misc_keys_queue = Queue()

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            logging.warning("KeyboardTeleop is not connected.")
            return {}

        self._drain_pressed_keys()

        if time.monotonic() - self.last_repeat_time < self.config.repeat_delay:
            return {}

        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1  # default gripper action is to stay

        # Generate action based on current key states
        for key in self.current_pressed_keys:
            if key == "w":
                delta_x = self.config.magnitude
            elif key == "s":
                delta_x = -self.config.magnitude
            elif key == "a":
                delta_y = self.config.magnitude
            elif key == "d":
                delta_y = -self.config.magnitude
            elif key == "q":
                delta_z = -self.config.magnitude
            elif key == "e":
                delta_z = self.config.magnitude
            elif key == "r":
                # Gripper actions are expected to be between 0 (close), 1 (stay), 2 (open)
                gripper_action = 2
            elif key == "f":
                gripper_action = 0
            elif key:
                # If the key is pressed, add it to the misc_keys_queue
                # this will record key presses that are not part of the delta_x, delta_y, delta_z
                # this is useful for retrieving other events like interventions for RL, episode success, etc.
                self.misc_keys_queue.put(key)

        action_dict = {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
        }

        if self.config.use_gripper:
            action_dict["gripper"] = gripper_action

        self.last_repeat_time = time.monotonic()

        return action_dict
