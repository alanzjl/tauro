"""Remote keyboard teleoperation using gRPC client."""

import asyncio
import logging
import threading
import time
from queue import Queue
from typing import Optional

import numpy as np
from sshkeyboard import listen_keyboard, stop_listening

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT
from tauro_common.types.robot_types import ControlCommand, ControlMode
from tauro_common.utils.utils import log_say
from tauro_inference.client.robot_client import RobotClient

logger = logging.getLogger(__name__)


class RemoteKeyboardTeleoperator:
    """Keyboard teleoperation for remote robots via gRPC."""

    def __init__(
        self,
        robot_id: str,
        robot_type: str,
        host: str = DEFAULT_GRPC_HOST,
        port: int = DEFAULT_GRPC_PORT,
        use_degrees: bool = True,
        speed_scale: float = 1.0,
    ):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.client = RobotClient(host=host, port=port)
        self.use_degrees = use_degrees
        self.speed_scale = speed_scale

        self._connected = False
        self._streaming = False
        self._stop_event = threading.Event()
        self._command_queue = Queue()
        self._stream_thread: Optional[threading.Thread] = None

        # Control state
        self.motor_names = []
        self.motor_positions = {}
        self.selected_motor_idx = 0

        # Movement parameters
        self.step_size = 5.0 if use_degrees else 0.05  # degrees or normalized
        self.continuous_speed = 30.0 if use_degrees else 0.3  # degrees/s or normalized/s

    def connect(self):
        """Connect to the remote robot."""
        if self._connected:
            raise RuntimeError("Already connected")

        # Connect to server
        self.client.connect_to_server()

        # Connect to robot
        success = self.client.connect_robot(self.robot_id, self.robot_type)
        if not success:
            self.client.disconnect_from_server()
            raise ConnectionError(f"Failed to connect to robot {self.robot_id}")

        # Get initial state
        state = self.client.get_robot_state(self.robot_id)
        if state and state.joints:
            self.motor_names = sorted(state.joints.keys())
            self.motor_positions = {name: state.joints[name].position for name in self.motor_names}

        self._connected = True
        logger.info(f"Connected to remote robot {self.robot_id}")

    def disconnect(self):
        """Disconnect from the remote robot."""
        if not self._connected:
            return

        # Stop streaming if active
        if self._streaming:
            self.stop_streaming()

        try:
            self.client.disconnect_robot(self.robot_id)
        finally:
            self.client.disconnect_from_server()
            self._connected = False
            logger.info(f"Disconnected from remote robot {self.robot_id}")

    def calibrate(self):
        """Calibrate the remote robot."""
        if not self._connected:
            raise RuntimeError("Not connected")

        log_say("Starting calibration...")
        calibrations = self.client.calibrate_robot(self.robot_id)

        if calibrations:
            log_say("Calibration complete!")
            return True
        else:
            log_say("Calibration failed!")
            return False

    def start_streaming(self):
        """Start streaming control mode."""
        if not self._connected:
            raise RuntimeError("Not connected")

        if self._streaming:
            return

        self._streaming = True
        self._stop_event.clear()

        # Start streaming thread
        self._stream_thread = threading.Thread(target=self._streaming_loop)
        self._stream_thread.start()

        logger.info("Started streaming control")

    def stop_streaming(self):
        """Stop streaming control mode."""
        if not self._streaming:
            return

        self._streaming = False
        self._stop_event.set()

        if self._stream_thread:
            self._stream_thread.join()
            self._stream_thread = None

        logger.info("Stopped streaming control")

    def _streaming_loop(self):
        """Background thread for streaming control."""

        async def async_streaming():
            await self.client.start_streaming_control(self.robot_id)

            try:
                while not self._stop_event.is_set():
                    # Check for commands
                    if not self._command_queue.empty():
                        command = self._command_queue.get()
                        await self.client.send_streaming_command(command)

                    # Get state update
                    state = await self.client.get_streaming_state()
                    if state and state.joints:
                        self.motor_positions = {
                            name: state.joints[name].position for name in self.motor_names
                        }

                    await asyncio.sleep(0.01)  # 100Hz

            finally:
                await self.client.stop_streaming_control()

        # Run async loop
        asyncio.run(async_streaming())

    def send_position_command(self, motor_name: str, position: float):
        """Send position command for a specific motor."""
        command = ControlCommand(
            timestamp=time.time(),
            robot_id=self.robot_id,
            joint_commands={motor_name: position},
            control_mode=ControlMode.POSITION,
        )

        if self._streaming:
            self._command_queue.put(command)
        else:
            # Send as single action
            self.client.send_action(self.robot_id, {motor_name: position})

    def move_motor(self, motor_idx: int, delta: float):
        """Move a motor by a delta amount."""
        if motor_idx >= len(self.motor_names):
            return

        motor_name = self.motor_names[motor_idx]
        current_pos = self.motor_positions.get(motor_name, 0.0)
        new_pos = current_pos + delta

        # Apply limits based on mode
        if self.use_degrees:
            new_pos = np.clip(new_pos, -180, 180)
        else:
            new_pos = np.clip(new_pos, -1.0, 1.0)

        self.motor_positions[motor_name] = new_pos
        self.send_position_command(motor_name, new_pos)

    def _on_key_press(self, key):
        """Handle keyboard key press."""
        # Motor selection (number keys)
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(self.motor_names):
                self.selected_motor_idx = idx
                logger.info(f"Selected motor {idx}: {self.motor_names[idx]}")

        # Motor movement
        elif key == "q":  # Move negative
            self.move_motor(self.selected_motor_idx, -self.step_size)
        elif key == "w":  # Move positive
            self.move_motor(self.selected_motor_idx, self.step_size)

        # All motors movement
        elif key == "a":  # All motors negative
            for i in range(len(self.motor_names)):
                self.move_motor(i, -self.step_size)
        elif key == "s":  # All motors positive
            for i in range(len(self.motor_names)):
                self.move_motor(i, self.step_size)

        # Calibration
        elif key == "c":
            self.calibrate()

        # Exit
        elif key == "esc":
            stop_listening()

    def run(self):
        """Run the keyboard teleoperation."""
        if not self._connected:
            raise RuntimeError("Not connected to robot")

        # Start streaming for better responsiveness
        self.start_streaming()

        logger.info("Starting keyboard teleoperation...")
        logger.info("Controls:")
        logger.info("  1-9: Select motor")
        logger.info("  q/w: Move selected motor -/+")
        logger.info("  a/s: Move all motors -/+")
        logger.info("  c: Calibrate")
        logger.info("  ESC: Exit")

        try:
            listen_keyboard(on_press=self._on_key_press, until="esc")
        finally:
            self.stop_streaming()


def main():
    """Example usage of remote keyboard teleoperation."""
    import argparse

    parser = argparse.ArgumentParser(description="Remote Keyboard Teleoperation")
    parser.add_argument("--robot-id", type=str, default="robot_001", help="Robot ID")
    parser.add_argument("--robot-type", type=str, default="so100_follower", help="Robot type")
    parser.add_argument("--host", type=str, default=DEFAULT_GRPC_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_GRPC_PORT, help="Server port")
    parser.add_argument(
        "--use-degrees", action="store_true", help="Use degrees instead of normalized"
    )
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Speed scaling factor")

    args = parser.parse_args()

    teleop = RemoteKeyboardTeleoperator(
        robot_id=args.robot_id,
        robot_type=args.robot_type,
        host=args.host,
        port=args.port,
        use_degrees=args.use_degrees,
        speed_scale=args.speed_scale,
    )

    try:
        teleop.connect()
        teleop.run()
    finally:
        teleop.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
