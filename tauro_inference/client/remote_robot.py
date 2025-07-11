"""Remote robot interface that communicates with tauro_edge via gRPC."""

import logging
from typing import Any, Optional

import numpy as np

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT
from tauro_common.types.robot_types import RobotState
from tauro_inference.client.robot_client import RobotClient

logger = logging.getLogger(__name__)


class RemoteRobot:
    """Remote robot that communicates with a physical robot through tauro_edge server."""

    def __init__(
        self,
        robot_id: str,
        robot_type: str,
        host: str = DEFAULT_GRPC_HOST,
        port: int = DEFAULT_GRPC_PORT,
        config: Optional[dict] = None,
    ):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.config = config or {}
        self.client = RobotClient(host=host, port=port)
        self._connected = False
        self._robot_info = None
        self._latest_state: Optional[RobotState] = None

    def connect(self):
        """Connect to the robot through the edge server."""
        if self._connected:
            raise RuntimeError(f"Robot {self.robot_id} is already connected")

        # Connect to server
        self.client.connect_to_server()

        # Connect to robot
        success = self.client.connect_robot(self.robot_id, self.robot_type, self.config)
        if not success:
            self.client.disconnect_from_server()
            raise ConnectionError(f"Failed to connect to robot {self.robot_id}")

        self._connected = True
        logger.info(f"Connected to remote robot {self.robot_id}")

    def disconnect(self):
        """Disconnect from the robot."""
        if not self._connected:
            return

        try:
            self.client.disconnect_robot(self.robot_id)
        finally:
            self.client.disconnect_from_server()
            self._connected = False
            logger.info(f"Disconnected from remote robot {self.robot_id}")

    def calibrate(self) -> bool:
        """Calibrate the robot."""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        calibrations = self.client.calibrate_robot(self.robot_id)
        return calibrations is not None

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from the robot."""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        state = self.client.get_robot_state(self.robot_id)
        if state is None:
            raise RuntimeError("Failed to get robot state")

        self._latest_state = state

        # Convert state to observation format
        obs = {}

        # Extract joint states into observation.state
        if state.joints:
            motor_names = sorted(state.joints.keys())
            num_motors = len(motor_names)

            positions = [state.joints[name].position for name in motor_names]
            # velocities = [state.joints[name].velocity for name in motor_names]
            # torques = [state.joints[name].torque for name in motor_names]

            for name in motor_names:
                obs[f"{name}.pos"] = state.joints[name].position
                # obs[f"{name}.vel"] = state.joints[name].velocity
                # obs[f"{name}.torque"] = state.joints[name].torque

        # # Add sensor data
        # for key, data in state.sensors.items():
        #     obs[key] = data

        return obs

    def send_action(self, action: dict[str, Any]) -> bool:
        """Send action command to the robot."""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        # Extract action array
        if "action" not in action:
            raise ValueError("Action dict must contain 'action' key")

        action_array = action["action"]
        if not isinstance(action_array, np.ndarray):
            action_array = np.array(action_array, dtype=np.float32)

        # Convert to joint commands
        if self._latest_state and self._latest_state.joints:
            motor_names = sorted(self._latest_state.joints.keys())
            if len(action_array) != len(motor_names):
                raise ValueError(
                    f"Action size {len(action_array)} doesn't match "
                    f"number of motors {len(motor_names)}"
                )

            joint_commands = {name: float(action_array[i]) for i, name in enumerate(motor_names)}
        else:
            # Fallback: assume action indices map to motor indices
            joint_commands = {f"motor_{i}": float(val) for i, val in enumerate(action_array)}

        return self.client.send_action(self.robot_id, joint_commands)

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        if not self._connected or self._latest_state is None:
            return False

        # Check if all joints are calibrated
        if self._latest_state.joints:
            return all(joint.is_calibrated for joint in self._latest_state.joints.values())

        return False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
