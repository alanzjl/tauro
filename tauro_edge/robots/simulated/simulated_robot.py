"""MuJoCo-based simulated robot implementation."""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

try:
    import mujoco.viewer

    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

from tauro_edge.motors import MotorCalibration
from tauro_edge.robots.robot import Robot
from tauro_edge.utils.robot_config import (
    denormalize_position,
    get_robot_config,
    normalize_position,
    setup_action_space,
    setup_observation_space,
)

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
    enable_visualization: bool = True  # Enable MuJoCo viewer
    use_torque_control: bool = False  # Use torque control instead of position control


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
                Path(__file__).parent.parent.parent.parent
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
        self.viewer = None  # MuJoCo viewer (if visualization enabled)

        # Joint mapping based on robot type
        robot_cfg = get_robot_config(config.robot_type)
        self.joint_names = robot_cfg["joint_names"]
        self.motor_names = robot_cfg["motor_names"]

        # Motor configurations (mimicking real robot)
        self.motor_configs = {name: {"id": i} for i, name in enumerate(self.motor_names)}

        # Observation and action spaces
        self._setup_spaces()

        # Control state
        self.target_positions = np.zeros(len(self.joint_names))
        self.last_update_time = 0

        # Joint ranges will be loaded from MuJoCo model
        self.joint_ranges = []

        # Simulation thread
        self._sim_thread = None
        self._sim_running = False
        self._sim_lock = threading.Lock()

        # Track last known viewer control values to detect manual changes
        self._last_viewer_ctrl = None

    def _setup_spaces(self):
        """Set up observation and action spaces."""
        self.observation_space = setup_observation_space(self.motor_names)
        self.action_space = setup_action_space(self.motor_names)

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

            # Initialize viewer if visualization is enabled
            if self.config.enable_visualization and HAS_VIEWER:
                # On macOS, require mjpython for visualization
                logger.warning("!!!!Visualization on macOS requires mjpython!!!!")

                try:
                    # Launch passive viewer
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    logger.info(f"MuJoCo passive viewer launched for robot {self.id}")
                except Exception as e:
                    logger.warning(f"Failed to launch MuJoCo viewer: {e}")
                    self.viewer = None

            # Set simulation timestep
            self.model.opt.timestep = self.config.sim_timestep

            # Load joint ranges from model
            self.joint_ranges = []
            for joint_name in self.joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    self.joint_ranges.append(tuple(self.model.jnt_range[joint_id]))
                else:
                    # Default range if joint not found
                    self.joint_ranges.append((-np.pi, np.pi))

            # Initialize to home position
            home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
            if home_key >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
                # Set target positions to match home position
                for i, joint_name in enumerate(self.joint_names):
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:
                        self.target_positions[i] = self.data.qpos[joint_id]
                logger.info(
                    f"Initialized to home position, target_positions: {self.target_positions}"
                )
                logger.info(f"Home qpos: {self.data.qpos[:len(self.joint_names)]}")
            else:
                # Set default home position
                for i, joint_name in enumerate(self.joint_names):
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    if joint_id >= 0:
                        self.data.qpos[joint_id] = 0
                        self.target_positions[i] = 0

            # Forward dynamics to update state
            mujoco.mj_forward(self.model, self.data)

            self._connected = True
            self.last_update_time = time.time()

            # Start simulation thread
            self._start_simulation_thread()

            logger.info(f"Connected to simulated robot {self.id}")

            # For simulated robots, we always use perfect calibration
            self._calibrated = True
            self._setup_perfect_calibration()
            logger.info(f"Robot calibrated: {self._calibrated}, joint_ranges: {self.joint_ranges}")

        except Exception as e:
            logger.error(f"Failed to connect simulated robot: {e}")
            raise

    def disconnect(self) -> None:
        """Disconnect from simulated robot."""
        if not self._connected:
            logger.warning(f"Robot {self.id} not connected")
            return

        # Stop simulation thread
        self._stop_simulation_thread()

        # Close viewer if it exists
        if self.viewer is not None:
            try:
                self.viewer.close()
            except AttributeError:
                pass
            self.viewer = None

        self._connected = False
        self.model = None
        self.data = None

        logger.info(f"Disconnected simulated robot {self.id}")

    def calibrate(self) -> None:
        """Calibrate the simulated robot."""
        if not self._connected:
            raise RuntimeError("Robot must be connected before calibration")

        # For simulated robots, calibration is always perfect
        self._setup_perfect_calibration()
        logger.info(f"Calibrated simulated robot {self.id} (perfect calibration)")

    def _setup_perfect_calibration(self):
        """Set up perfect calibration for simulated robot."""
        # Create calibration data that represents perfect calibration
        self.calibration = {}

        for i, motor_name in enumerate(self.motor_names):
            # Perfect calibration: no offset, full range
            self.calibration[motor_name] = MotorCalibration(
                id=i,
                drive_mode=0,
                homing_offset=0,  # No offset for perfect calibration
                range_min=0,
                range_max=4095,  # Full encoder range
            )

        self._calibrated = True

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

        obs = {}
        logger.debug("Getting observation")

        # Get joint states (thread-safe)
        with self._sim_lock:
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

                    # Normalize position
                    norm_pos = normalize_position(raw_pos, self.joint_ranges[i], motor_name)

                    positions[motor_name] = float(norm_pos)
                    velocities[motor_name] = float(raw_vel)  # Velocity in rad/s
                    logger.debug(
                        f"Obs {motor_name}: raw_pos={raw_pos:.3f}, norm_pos={norm_pos:.1f}, range={self.joint_ranges[i]}"
                    )
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

        logger.debug(f"send_action called with: {action}")
        applied_action = {}

        with self._sim_lock:
            # Handle joint position commands
            if "joints" in action and "position" in action["joints"]:
                positions = action["joints"]["position"]
                for motor_name, position in positions.items():
                    if motor_name in self.motor_names:
                        idx = self.motor_names.index(motor_name)

                        # Denormalize position
                        target_pos = denormalize_position(
                            position, self.joint_ranges[idx], motor_name
                        )

                        self.target_positions[idx] = target_pos
                        applied_action[motor_name] = position
                        logger.debug(
                            f"Set {motor_name} (idx={idx}) target: {position} -> {target_pos:.3f} rad (range: {self.joint_ranges[idx]})"
                        )

            # Handle individual motor commands (e.g., "shoulder_pan.pos": 50)
            for key, value in action.items():
                if key.endswith(".pos"):
                    motor_name = key[:-4]
                    if motor_name in self.motor_names:
                        idx = self.motor_names.index(motor_name)

                        # Denormalize position
                        target_pos = denormalize_position(value, self.joint_ranges[idx], motor_name)

                        self.target_positions[idx] = target_pos
                        applied_action[motor_name] = value

        logger.debug(f"Applied action: {applied_action}")
        logger.debug(f"Target positions: {self.target_positions}")
        return applied_action

    def _simulation_thread_loop(self):
        """Continuous simulation loop that runs in a separate thread."""
        logger.info("Starting simulation thread")
        last_step_time = time.time()

        while self._sim_running:
            current_time = time.time()
            dt = current_time - last_step_time

            # Step at regular intervals
            if dt >= self.config.sim_timestep:
                with self._sim_lock:
                    if self.data is not None and self.model is not None:
                        # First, check if viewer sliders have been moved manually
                        viewer_changed = False
                        if self.viewer is not None and self._last_viewer_ctrl is not None:
                            try:
                                for i, joint_name in enumerate(self.joint_names):
                                    actuator_id = mujoco.mj_name2id(
                                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                                    )
                                    if actuator_id >= 0:
                                        current_ctrl = self.data.ctrl[actuator_id]
                                        # Check if viewer slider was moved (changed from last known value)
                                        if (
                                            abs(current_ctrl - self._last_viewer_ctrl[actuator_id])
                                            > 0.001
                                        ):
                                            # Viewer was manually changed, update our target
                                            self.target_positions[i] = current_ctrl
                                            viewer_changed = True
                                            logger.debug(
                                                f"Viewer manually moved {joint_name} to {current_ctrl:.3f}"
                                            )
                            except Exception:
                                pass  # Viewer might be closed

                        # If viewer wasn't changed, apply our target positions to control
                        if not viewer_changed:
                            for i, joint_name in enumerate(self.joint_names):
                                actuator_id = mujoco.mj_name2id(
                                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                                )
                                if actuator_id >= 0:
                                    # Set control to our target position
                                    self.data.ctrl[actuator_id] = self.target_positions[i]

                        # Remember current control values for next iteration
                        if self._last_viewer_ctrl is None:
                            self._last_viewer_ctrl = np.zeros(self.model.nu)
                        self._last_viewer_ctrl[:] = self.data.ctrl[:]

                        # Step physics
                        mujoco.mj_step(self.model, self.data)

                        # Update viewer
                        if self.viewer is not None:
                            try:
                                self.viewer.sync()
                            except Exception:
                                pass  # Viewer might be closed

                last_step_time = current_time

            # Small sleep to prevent CPU spinning
            time.sleep(0.001)

        logger.info("Simulation thread stopped")

    def _start_simulation_thread(self):
        """Start the continuous simulation thread."""
        if self._sim_thread is None or not self._sim_thread.is_alive():
            self._sim_running = True
            self._sim_thread = threading.Thread(target=self._simulation_thread_loop, daemon=True)
            self._sim_thread.start()
            logger.info("Simulation thread started")

    def _stop_simulation_thread(self):
        """Stop the continuous simulation thread."""
        if self._sim_thread is not None and self._sim_thread.is_alive():
            self._sim_running = False
            self._sim_thread.join(timeout=1.0)
            self._sim_thread = None
            logger.info("Simulation thread stopped")
