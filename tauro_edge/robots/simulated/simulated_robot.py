"""MuJoCo-based simulated robot implementation."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from tauro_edge.motors import MotorCalibration
from tauro_edge.robots.robot import Robot

logger = logging.getLogger(__name__)


@dataclass
class SimulatedRobotConfig:
    """Configuration for simulated robot."""

    id: str
    robot_type: str = "so100_follower"
    model_path: Path | None = None
    calibration_dir: Path | None = None
    sim_timestep: float = 0.002  # 2ms timestep for 500Hz simulation
    control_timestep: float = 0.01  # 10ms control timestep for 100Hz control


class SimulatedRobot(Robot):
    """
    MuJoCo-based simulated robot implementation.

    This class provides a simulated robot that mimics the behavior of real robots,
    including proper handling of calibration offsets and joint limits.
    """

    name = "simulated"

    def __init__(self, config: SimulatedRobotConfig):
        """Initialize simulated robot.

        Args:
            config: Simulated robot configuration
        """
        super().__init__(config)

        self.config = config
        self._connected = False
        self._calibrated = False

        # Set up model path
        if config.model_path is None:
            # Default to SO100 model
            self.model_path = (
                Path(__file__).parent.parent.parent.parent.parent
                / "tauro_common"
                / "models"
                / "so_arm100"
                / "so_arm100.xml"
            )
        else:
            self.model_path = config.model_path

        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model not found at {self.model_path}")

        # MuJoCo objects (initialized on connect)
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

        # Joint mapping based on robot type
        if config.robot_type == "so100_follower":
            self.joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
            self.motor_names = [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]
        elif config.robot_type == "so101_follower":
            self.joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
            self.motor_names = [
                "waist",
                "shoulder",
                "elbow",
                "wrist_angle",
                "wrist_rotate",
                "gripper",
            ]
        else:
            raise ValueError(f"Unsupported robot type: {config.robot_type}")

        # Motor configurations (mimicking real robot)
        self.motor_configs = {name: {"id": i} for i, name in enumerate(self.motor_names)}

        # Observation and action spaces
        self._setup_spaces()

        # Control state
        self.target_positions = np.zeros(len(self.joint_names))
        self.last_update_time = 0

        # Calibration-related
        self.calibration_offsets = np.zeros(len(self.joint_names))
        self.calibration_ranges = [(0, 4095)] * len(
            self.joint_names
        )  # Default 12-bit encoder range

    def _setup_spaces(self):
        """Set up observation and action spaces."""
        # Joint space observation
        joint_space = {
            "position": {
                motor: {"shape": (), "dtype": np.dtype(np.float32), "names": None}
                for motor in self.motor_names
            },
            "velocity": {
                motor: {"shape": (), "dtype": np.dtype(np.float32), "names": None}
                for motor in self.motor_names
            },
        }

        # End effector observation
        end_effector_space = {
            "position": {
                "shape": (3,),
                "dtype": np.dtype(np.float32),
                "names": ["x", "y", "z"],
            },
            "orientation": {
                "shape": (9,),
                "dtype": np.dtype(np.float32),
                "names": None,
            },
        }

        self.observation_space = {
            "joint": joint_space,
            "end_effector": end_effector_space,
        }

        # Action space
        self.action_space = {
            f"{motor}.pos": {"shape": (), "dtype": np.dtype(np.float32), "names": None}
            for motor in self.motor_names
        }

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        """Check if robot is calibrated."""
        return self._calibrated

    def connect(self, calibrate: bool = True) -> None:
        """Connect to simulated robot.

        Args:
            calibrate: Whether to calibrate after connecting
        """
        if self._connected:
            logger.warning(f"Robot {self.id} already connected")
            return

        try:
            # Load MuJoCo model
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.data = mujoco.MjData(self.model)

            # Set simulation timestep
            self.model.opt.timestep = self.config.sim_timestep

            # Initialize to home position
            home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
            else:
                # Set default home position
                for name in self.joint_names:
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    if joint_id >= 0:
                        self.data.qpos[joint_id] = 0

            # Forward dynamics to update state
            mujoco.mj_forward(self.model, self.data)

            self._connected = True
            self.last_update_time = time.time()

            logger.info(f"Connected to simulated robot {self.id}")

            # Load calibration if available
            if self.calibration_fpath.is_file():
                self._load_calibration()
                self._apply_calibration()
                self._calibrated = True
            elif calibrate:
                self.calibrate()

        except Exception as e:
            logger.error(f"Failed to connect simulated robot: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from simulated robot."""
        if not self._connected:
            logger.warning(f"Robot {self.id} not connected")
            return

        self._connected = False
        self.model = None
        self.data = None

        logger.info(f"Disconnected simulated robot {self.id}")

    def calibrate(self) -> None:
        """Calibrate the simulated robot."""
        if not self._connected:
            raise RuntimeError("Robot must be connected before calibration")

        # For simulation, we create synthetic calibration data
        # that mimics real robot calibration
        self.calibration = {}

        for i, motor_name in enumerate(self.motor_names):
            # Generate realistic calibration values
            # Homing offset: random offset to shift zero reference
            homing_offset = np.random.randint(-2048, 2048)

            # Range limits based on joint limits in MuJoCo model
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[i])
            if joint_id >= 0:
                joint_range = self.model.jnt_range[joint_id]
                # Convert radians to encoder counts (12-bit encoder)
                range_min = int((joint_range[0] + np.pi) * 4095 / (2 * np.pi))
                range_max = int((joint_range[1] + np.pi) * 4095 / (2 * np.pi))
            else:
                range_min = 1024
                range_max = 3072

            self.calibration[motor_name] = MotorCalibration(
                id=i,
                drive_mode=0,
                homing_offset=homing_offset,
                range_min=range_min,
                range_max=range_max,
            )

            self.calibration_offsets[i] = homing_offset * 2 * np.pi / 4095  # Convert to radians
            self.calibration_ranges[i] = (range_min, range_max)

        self._calibrated = True

        # Save calibration
        self._save_calibration()

        logger.info(f"Calibrated simulated robot {self.id}")

    def _apply_calibration(self):
        """Apply loaded calibration to the robot."""
        if not self.calibration:
            return

        for i, motor_name in enumerate(self.motor_names):
            if motor_name in self.calibration:
                cal = self.calibration[motor_name]
                # Convert homing offset from encoder counts to radians
                self.calibration_offsets[i] = cal.homing_offset * 2 * np.pi / 4095
                self.calibration_ranges[i] = (cal.range_min, cal.range_max)

    def configure(self) -> None:
        """Configure the simulated robot."""
        # No special configuration needed for simulation
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current observation from simulated robot.

        Returns:
            Dictionary containing joint positions and velocities
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")

        # Step simulation if needed
        self._step_simulation()

        obs = {}

        # Get joint states
        positions = {}
        velocities = {}

        for i, (joint_name, motor_name) in enumerate(
            zip(self.joint_names, self.motor_names, strict=False)
        ):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                # Get raw position and velocity from MuJoCo
                raw_pos = self.data.qpos[joint_id]
                raw_vel = self.data.qvel[joint_id]

                # Apply calibration offset to position (mimicking real motor behavior)
                # In real motors: Present_Position = Actual_Position - Homing_Offset
                calibrated_pos = raw_pos - self.calibration_offsets[i]

                # Convert to encoder counts
                encoder_pos = (calibrated_pos + np.pi) * 4095 / (2 * np.pi)

                # Normalize based on calibration range and motor type
                range_min, range_max = self.calibration_ranges[i]

                if motor_name == "gripper":
                    # Gripper uses [0, 100] normalization
                    norm_pos = ((encoder_pos - range_min) / (range_max - range_min)) * 100
                    norm_pos = np.clip(norm_pos, 0, 100)
                else:
                    # Other joints use [-100, 100] normalization
                    norm_pos = (((encoder_pos - range_min) / (range_max - range_min)) * 200) - 100
                    norm_pos = np.clip(norm_pos, -100, 100)

                positions[motor_name] = float(norm_pos)
                velocities[motor_name] = float(raw_vel)  # Velocity in rad/s
            else:
                positions[motor_name] = 0.0
                velocities[motor_name] = 0.0

        # Build observation dictionary matching real robot format
        obs["joints"] = {
            "position": positions,
            "velocity": velocities,
        }

        # TODO: Add end effector observation if needed
        # For now, just return empty end effector data
        obs["end_effector"] = {
            "position": np.zeros(3, dtype=np.float32),
            "orientation": np.eye(3, dtype=np.float32).flatten(),
        }

        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action command to simulated robot.

        Args:
            action: Dictionary containing action commands

        Returns:
            The action actually applied
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")

        applied_action = {}

        # Handle joint position commands
        if "joints" in action and "position" in action["joints"]:
            positions = action["joints"]["position"]
            for motor_name, position in positions.items():
                if motor_name in self.motor_names:
                    idx = self.motor_names.index(motor_name)

                    # Unnormalize position based on motor type
                    range_min, range_max = self.calibration_ranges[idx]

                    if motor_name == "gripper":
                        # Gripper: [0, 100] -> encoder counts
                        encoder_pos = (position / 100) * (range_max - range_min) + range_min
                    else:
                        # Other joints: [-100, 100] -> encoder counts
                        encoder_pos = ((position + 100) / 200) * (range_max - range_min) + range_min

                    # Convert to radians
                    rad_pos = encoder_pos * 2 * np.pi / 4095 - np.pi

                    # Apply calibration offset (inverse of reading)
                    # In real motors: Actual_Position = Present_Position + Homing_Offset
                    calibrated_pos = rad_pos + self.calibration_offsets[idx]

                    # Clip to joint limits
                    joint_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[idx]
                    )
                    if joint_id >= 0:
                        joint_range = self.model.jnt_range[joint_id]
                        calibrated_pos = np.clip(calibrated_pos, joint_range[0], joint_range[1])

                    self.target_positions[idx] = calibrated_pos
                    applied_action[motor_name] = position

        # Handle individual motor commands (e.g., "shoulder_pan.pos": 50)
        for key, value in action.items():
            if key.endswith(".pos"):
                motor_name = key[:-4]
                if motor_name in self.motor_names:
                    idx = self.motor_names.index(motor_name)

                    # Unnormalize based on motor type
                    range_min, range_max = self.calibration_ranges[idx]

                    if motor_name == "gripper":
                        # Gripper: [0, 100] -> encoder counts
                        encoder_pos = (value / 100) * (range_max - range_min) + range_min
                    else:
                        # Other joints: [-100, 100] -> encoder counts
                        encoder_pos = ((value + 100) / 200) * (range_max - range_min) + range_min

                    # Convert to radians
                    rad_pos = encoder_pos * 2 * np.pi / 4095 - np.pi

                    # Apply calibration offset
                    calibrated_pos = rad_pos + self.calibration_offsets[idx]

                    # Clip to joint limits
                    joint_id = mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, self.joint_names[idx]
                    )
                    if joint_id >= 0:
                        joint_range = self.model.jnt_range[joint_id]
                        calibrated_pos = np.clip(calibrated_pos, joint_range[0], joint_range[1])

                    self.target_positions[idx] = calibrated_pos
                    applied_action[motor_name] = value

        # Step simulation to apply actions
        self._step_simulation()

        return applied_action

    def _step_simulation(self):
        """Step the MuJoCo simulation."""
        if not self._connected:
            return

        current_time = time.time()
        dt = current_time - self.last_update_time

        # Only step if enough time has passed
        if dt < self.config.control_timestep:
            return

        # Set control targets
        for i, joint_name in enumerate(self.joint_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = self.target_positions[i]

        # Step simulation
        steps = int(dt / self.config.sim_timestep)
        for _ in range(min(steps, 100)):  # Cap at 100 steps to prevent hanging
            mujoco.mj_step(self.model, self.data)

        self.last_update_time = current_time
