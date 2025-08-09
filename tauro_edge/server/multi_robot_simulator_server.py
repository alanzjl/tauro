"""Multi-robot MuJoCo simulator server."""

import asyncio
import logging
import threading
import time
from concurrent import futures
from pathlib import Path

import grpc
import mujoco
import numpy as np
import yaml

try:
    import mujoco.viewer

    HAS_VIEWER = True
except ImportError:
    HAS_VIEWER = False

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT, MAX_MESSAGE_SIZE
from tauro_common.proto import robot_service_pb2, robot_service_pb2_grpc
from tauro_common.types.robot_types import (
    FeatureInfo,
    JointState,
    RobotState,
    RobotStatus,
)
from tauro_common.utils.proto_utils import (
    feature_info_to_proto,
    motor_calibration_to_proto,
    robot_state_dict_to_proto,
    robot_state_to_proto,
    timestamp_to_proto,
)
from tauro_edge.motors import MotorCalibration
from tauro_edge.utils.robot_config import (
    denormalize_position,
    get_robot_config,
    normalize_position,
    setup_observation_space,
)
from tauro_edge.utils.scene_builder import SceneBuilder

logger = logging.getLogger(__name__)


class RobotInstance:
    """Represents a single robot in the simulation."""

    def __init__(self, robot_id: str, robot_type: str, joint_indices: list, actuator_indices: list):
        self.id = robot_id
        self.type = robot_type
        self.joint_indices = joint_indices
        self.actuator_indices = actuator_indices
        self.is_calibrated = False
        self.calibration = {}

        # Motor naming based on robot type
        try:
            robot_cfg = get_robot_config(robot_type)
            self.motor_names = robot_cfg["motor_names"]
            self.joint_names = robot_cfg["joint_names"]
        except ValueError:
            # Default naming for unknown robot types
            self.motor_names = [f"joint_{i}" for i in range(len(joint_indices))]
            self.joint_names = [f"joint_{i}" for i in range(len(joint_indices))]

        # Observation and action spaces
        self.observation_space = self._setup_observation_space()
        self.action_space = self._setup_action_space()

    def _setup_observation_space(self):
        """Set up observation space for this robot."""
        return setup_observation_space(self.motor_names)

    def _setup_action_space(self):
        """Set up action space for this robot."""
        # Note: multi-robot server uses a slightly different action space format
        # but we can still use the motor names
        return {
            "joints": {
                "shape": (len(self.motor_names),),
                "dtype": np.dtype(np.float32),
                "names": self.motor_names,
            }
        }

    def calibrate(self):
        """Generate synthetic calibration data."""
        self.calibration = {}
        for motor_name in self.motor_names:
            # Generate synthetic calibration offsets
            self.calibration[motor_name] = MotorCalibration(
                homing_offset=np.random.uniform(-5, 5),
                gear_ratio=1.0,
                direction=1,
                min_position=-100,
                max_position=100,
            )
        self.is_calibrated = True
        logger.info(f"Robot {self.id} calibrated with synthetic offsets")


class MultiRobotSimulatorServicer(robot_service_pb2_grpc.RobotControlServiceServicer):
    """gRPC service for multi-robot simulation."""

    def __init__(
        self,
        scene_config: Path,
        enable_visualization: bool = True,
        sim_timestep: float = 0.002,
        control_timestep: float = 0.01,
    ):
        """Initialize multi-robot simulator.

        Args:
            scene_config: Path to scene configuration YAML
            enable_visualization: Whether to show MuJoCo viewer
            sim_timestep: Simulation timestep
            control_timestep: Control update timestep
        """
        self.scene_config = scene_config
        self.enable_visualization = enable_visualization and HAS_VIEWER
        self.sim_timestep = sim_timestep
        self.control_timestep = control_timestep

        # Load configuration
        with open(scene_config) as f:
            self.config = yaml.safe_load(f)

        # Build scene
        base_scene_name = self.config.get("scene", {}).get("base_scene", "empty_scene.xml")
        base_scene_path = Path("tauro_edge/sim_scenes") / base_scene_name

        builder = SceneBuilder(base_scene_path, scene_config)
        self.scene_xml = builder.build_scene()

        logger.info(f"Built scene from {scene_config}")
        logger.info(f"Scene XML: {self.scene_xml}")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        # Parse robot instances from the model
        self.robots: dict[str, RobotInstance] = {}
        self._parse_robots()

        # Simulation state
        self._sim_lock = threading.Lock()
        self._sim_thread = None
        self._sim_running = False
        self.viewer = None

        # Control streams
        self._control_streams: dict[str, asyncio.Queue] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}

        # Start simulation
        self._start_simulation()

    def _parse_robots(self):
        """Parse robot instances from the MuJoCo model."""
        # Get robot definitions from config
        robot_configs = self.config.get("robots", [])

        for robot_cfg in robot_configs:
            robot_id = robot_cfg["id"]
            robot_type = robot_cfg.get("type", "so100_follower")

            # Find joints and actuators for this robot
            joint_indices = []
            actuator_indices = []

            # Expected joint names for this robot
            if robot_type in ["so100_follower", "so101_follower"]:
                joint_suffixes = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
            else:
                joint_suffixes = []

            # Find joints
            for suffix in joint_suffixes:
                joint_name = f"{robot_id}/{suffix}"
                try:
                    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                    joint_indices.append(joint_id)
                except Exception:
                    logger.warning(f"Joint {joint_name} not found in model")

            # Find actuators
            for suffix in joint_suffixes:
                act_name = f"{robot_id}/{suffix}"
                try:
                    act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                    actuator_indices.append(act_id)
                except Exception:
                    logger.warning(f"Actuator {act_name} not found in model")

            if joint_indices and actuator_indices:
                robot = RobotInstance(robot_id, robot_type, joint_indices, actuator_indices)
                self.robots[robot_id] = robot
                logger.info(f"Registered robot {robot_id} with {len(joint_indices)} joints")

    def _start_simulation(self):
        """Start the simulation thread."""
        self._sim_running = True
        self._sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self._sim_thread.start()
        logger.info("Simulation thread started")

    def _stop_simulation(self):
        """Stop the simulation thread."""
        self._sim_running = False
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)
        if self.viewer:
            self.viewer.close()
        logger.info("Simulation thread stopped")

    def _simulation_loop(self):
        """Main simulation loop."""
        if self.enable_visualization:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        while self._sim_running:
            # Step simulation
            with self._sim_lock:
                mujoco.mj_step(self.model, self.data)

            # Update viewer if enabled
            if self.viewer and self.viewer.is_running():
                self.viewer.sync()
            elif self.viewer and not self.viewer.is_running():
                # Viewer was closed
                self._sim_running = False
                break

            # Control rate limiting
            time.sleep(self.sim_timestep)

    def _get_robot_observation(self, robot_id: str) -> dict:
        """Get observation for a specific robot."""
        robot = self.robots.get(robot_id)
        if not robot:
            return {}

        with self._sim_lock:
            # Get joint positions and velocities
            positions = {}
            velocities = {}

            for joint_idx, motor_name in zip(robot.joint_indices, robot.motor_names, strict=False):
                # Get raw position and velocity
                raw_pos = self.data.qpos[joint_idx]
                raw_vel = self.data.qvel[joint_idx]

                # Apply calibration if available
                if robot.is_calibrated and motor_name in robot.calibration:
                    cal = robot.calibration[motor_name]
                    # Apply calibration offset
                    calibrated_pos = (raw_pos - cal.homing_offset) * cal.direction
                else:
                    calibrated_pos = raw_pos

                # Normalize position
                joint_range = self.model.jnt_range[joint_idx]
                normalized_pos = normalize_position(calibrated_pos, joint_range, motor_name)

                positions[motor_name] = float(normalized_pos)
                velocities[motor_name] = float(raw_vel)

        return {
            "joints": {
                "position": positions,
                "velocity": velocities,
            },
            "end_effector": {
                "position": np.zeros(3, dtype=np.float32),  # Placeholder for now
                "orientation": np.eye(3, dtype=np.float32).flatten(),  # Identity matrix flattened
            },
        }

    def _send_robot_action(self, robot_id: str, action: dict):
        """Send action to a specific robot."""
        robot = self.robots.get(robot_id)
        if not robot:
            return

        with self._sim_lock:
            if "joints" in action and "position" in action["joints"]:
                positions = action["joints"]["position"]

                for i, (act_idx, motor_name) in enumerate(
                    zip(robot.actuator_indices, robot.motor_names, strict=False)
                ):
                    if motor_name in positions:
                        normalized_pos = positions[motor_name]

                        # Denormalize position
                        joint_idx = robot.joint_indices[i]
                        joint_range = self.model.jnt_range[joint_idx]
                        raw_pos = denormalize_position(normalized_pos, joint_range, motor_name)

                        # Apply calibration if available
                        if robot.is_calibrated and motor_name in robot.calibration:
                            cal = robot.calibration[motor_name]
                            raw_pos = raw_pos / cal.direction + cal.homing_offset

                        # Set control target
                        self.data.ctrl[act_idx] = np.clip(raw_pos, joint_range[0], joint_range[1])

    # gRPC service methods

    def Connect(self, request, context):
        """Handle robot connection request."""
        robot_id = request.robot_id
        robot_type = request.robot_type

        # Check if robot exists in scene
        if robot_id not in self.robots:
            return robot_service_pb2.ConnectResponse(
                success=False,
                message=f"Robot {robot_id} not found in scene. Available robots: {list(self.robots.keys())}",
            )

        robot = self.robots[robot_id]

        # Update robot type if provided
        if robot_type:
            robot.type = robot_type

        # Prepare robot info
        robot_info = robot_service_pb2.RobotInfo(
            robot_id=robot_id,
            robot_type=robot.type,
            motor_names=robot.motor_names,
        )

        # Add observation space
        for space, space_feature in robot.observation_space.items():
            if space == "joint":
                for space_type, type_feature in space_feature.items():
                    for joint_name, joint_feature in type_feature.items():
                        robot_info.observation_space[f"{space}.{space_type}.{joint_name}"].CopyFrom(
                            feature_info_to_proto(
                                FeatureInfo(
                                    shape=joint_feature["shape"],
                                    dtype=joint_feature["dtype"].name,
                                    names=joint_feature.get("names"),
                                )
                            )
                        )
            elif space == "end_effector":
                for space_type, type_feature in space_feature.items():
                    robot_info.observation_space[f"{space}.{space_type}"].CopyFrom(
                        feature_info_to_proto(
                            FeatureInfo(
                                shape=type_feature["shape"],
                                dtype=type_feature["dtype"].name,
                                names=type_feature.get("names"),
                            )
                        )
                    )

        # Add action space
        for name, feature in robot.action_space.items():
            robot_info.action_space[name].CopyFrom(
                feature_info_to_proto(
                    FeatureInfo(
                        shape=feature["shape"],
                        dtype=feature["dtype"].name,
                        names=feature.get("names"),
                    )
                )
            )

        return robot_service_pb2.ConnectResponse(
            success=True,
            message=f"Successfully connected to robot {robot_id}",
            robot_info=robot_info,
        )

    def Disconnect(self, request, context):
        """Handle robot disconnection request."""
        robot_id = request.robot_id

        if robot_id not in self.robots:
            return robot_service_pb2.DisconnectResponse(
                success=False, message=f"Robot {robot_id} not found"
            )

        # Stop any active control streams
        if robot_id in self._stream_tasks:
            self._stream_tasks[robot_id].cancel()
            del self._stream_tasks[robot_id]

        if robot_id in self._control_streams:
            del self._control_streams[robot_id]

        return robot_service_pb2.DisconnectResponse(
            success=True, message=f"Successfully disconnected robot {robot_id}"
        )

    def Calibrate(self, request, context):
        """Handle robot calibration request."""
        robot_id = request.robot_id

        if robot_id not in self.robots:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Robot {robot_id} not found")
            return robot_service_pb2.CalibrateResponse(success=False, message="Robot not found")

        robot = self.robots[robot_id]
        robot.calibrate()

        # Return calibration data
        calibrations = {}
        for motor_name in robot.motor_names:
            if motor_name in robot.calibration:
                calibrations[motor_name] = motor_calibration_to_proto(robot.calibration[motor_name])

        return robot_service_pb2.CalibrateResponse(
            success=True,
            message=f"Successfully calibrated robot {robot_id}",
            calibrations=calibrations,
        )

    def GetState(self, request, context):
        """Get current robot state."""
        robot_id = request.robot_id

        if robot_id not in self.robots:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Robot {robot_id} not found")
            return robot_service_pb2.RobotState()

        robot = self.robots[robot_id]
        obs = self._get_robot_observation(robot_id)

        robot_state = robot_state_dict_to_proto(obs, robot.is_calibrated, robot_id, time.time())
        return robot_state

    def SendAction(self, request, context):
        """Send action command to robot."""
        robot_id = request.robot_id

        if robot_id not in self.robots:
            return robot_service_pb2.ActionResponse(
                success=False, message=f"Robot {robot_id} not found"
            )

        # Parse action command
        if request.HasField("joint_command"):
            joint_cmd = request.joint_command
            action = {"joints": {"position": dict(joint_cmd.positions)}}
        elif request.HasField("end_effector_command"):
            return robot_service_pb2.ActionResponse(
                success=False, message="End effector control not yet supported"
            )
        else:
            return robot_service_pb2.ActionResponse(success=False, message="Invalid action command")

        # Send action
        self._send_robot_action(robot_id, action)

        return robot_service_pb2.ActionResponse(success=True, message="Action sent successfully")

    async def StreamControl(self, request_iterator, context):
        """Handle bidirectional streaming for real-time control."""
        robot_id = None

        try:
            async for command in request_iterator:
                robot_id = command.robot_id

                if robot_id not in self.robots:
                    logger.error(f"Robot {robot_id} not found in stream")
                    continue

                robot = self.robots[robot_id]

                # Process control command
                if command.HasField("joint_command"):
                    joint_cmd = command.joint_command
                    action = {"joints": {"position": dict(joint_cmd.positions)}}
                    self._send_robot_action(robot_id, action)

                # Get current state
                obs = self._get_robot_observation(robot_id)

                # Convert to RobotState
                joints = {}
                if "joints" in obs:
                    joint_data = obs["joints"]
                    for motor_name in robot.motor_names:
                        joints[motor_name] = JointState(
                            position=float(joint_data["position"].get(motor_name, 0)),
                            velocity=float(joint_data["velocity"].get(motor_name, 0)),
                            torque=0.0,
                            temperature=25.0,
                            is_calibrated=robot.is_calibrated,
                        )

                robot_state = RobotState(
                    timestamp=time.time(),
                    robot_id=robot_id,
                    joints=joints,
                    sensors={},
                    status=RobotStatus.READY if robot.is_calibrated else RobotStatus.CONNECTED,
                )

                yield robot_state_to_proto(robot_state)

        except asyncio.CancelledError:
            logger.info(f"Stream cancelled for robot {robot_id}")
        except Exception as e:
            logger.error(f"Error in control stream: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def HealthCheck(self, request, context):
        """Check health status of the simulator."""
        components = {}

        # Check each robot
        for robot_id, robot in self.robots.items():
            components[robot_id] = robot_service_pb2.ComponentHealth(
                is_healthy=True,
                status="ready" if robot.is_calibrated else "connected",
                message=f"Robot {robot_id} operational",
            )

        # Check simulator
        components["simulator"] = robot_service_pb2.ComponentHealth(
            is_healthy=self._sim_running,
            status="running" if self._sim_running else "stopped",
            message="Multi-robot simulator",
        )

        return robot_service_pb2.HealthStatus(
            is_healthy=all(c.is_healthy for c in components.values()),
            components=components,
            timestamp=timestamp_to_proto(time.time()),
        )

    def __del__(self):
        """Cleanup on deletion."""
        self._stop_simulation()


def serve(
    host: str = DEFAULT_GRPC_HOST,
    port: int = DEFAULT_GRPC_PORT,
    scene_config: Path = Path("tauro_edge/configs/sim_two_robots.yaml"),
    enable_visualization: bool = True,
):
    """Run the multi-robot simulator server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    servicer = MultiRobotSimulatorServicer(scene_config, enable_visualization)
    robot_service_pb2_grpc.add_RobotControlServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting multi-robot simulator server on {listen_addr}")
    logger.info(f"Scene config: {scene_config}")
    logger.info(f"Visualization: {'enabled' if enable_visualization else 'disabled'}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down simulator server...")
        servicer._stop_simulation()
        server.stop(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Robot MuJoCo Simulator Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50051, help="Port to bind to")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("tauro_edge/configs/sim_two_robots.yaml"),
        help="Path to scene configuration YAML",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualization")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    serve(args.host, args.port, args.scene, enable_visualization=not args.no_vis)
