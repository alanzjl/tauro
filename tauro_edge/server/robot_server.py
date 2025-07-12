"""gRPC server implementation for robot control."""

import asyncio
import logging
import time
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc
import numpy as np
import yaml

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
    proto_to_control_mode,
    proto_to_timestamp,
    robot_state_to_proto,
    timestamp_to_proto,
)
from tauro_edge.robots.robot import Robot

logger = logging.getLogger(__name__)


class RobotControlServicer(robot_service_pb2_grpc.RobotControlServiceServicer):
    """gRPC service implementation for robot control."""

    def __init__(self):
        self.robots: dict[str, Robot] = {}
        self._control_streams: dict[str, asyncio.Queue] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._robot_configs = self._load_robot_configs()

    def _load_robot_configs(self) -> dict[str, dict]:
        """Load robot configurations from YAML file."""
        config_path = Path(__file__).parent.parent / "configs" / "robot_ports.yaml"
        if not config_path.exists():
            logger.warning(f"Robot config file not found at {config_path}")
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return config.get("robots", {})
        except Exception as e:
            logger.error(f"Error loading robot configs: {e}")
            return {}

    def _get_robot(self, robot_id: str) -> Optional[Robot]:
        """Get robot instance by ID."""
        return self.robots.get(robot_id)

    def _register_robot(self, robot_id: str, robot: Robot):
        """Register a robot instance."""
        self.robots[robot_id] = robot
        logger.info(f"Registered robot: {robot_id}")

    def Connect(self, request, context):
        """Handle robot connection request."""
        robot_id = request.robot_id
        robot_type = request.robot_type

        # Check if robot already connected
        if robot_id in self.robots:
            return robot_service_pb2.ConnectResponse(
                success=False, message=f"Robot {robot_id} is already connected"
            )

        # Check if robot is configured
        if robot_id not in self._robot_configs:
            return robot_service_pb2.ConnectResponse(
                success=False,
                message=f"Robot {robot_id} not found in configuration. Please add it to robot_ports.yaml",
            )

        robot_config = self._robot_configs[robot_id]
        configured_type = robot_config.get("type")
        port = robot_config.get("port")

        # Verify robot type matches configuration
        if robot_type != configured_type:
            return robot_service_pb2.ConnectResponse(
                success=False,
                message=f"Robot type mismatch: requested {robot_type} but {robot_id} is configured as {configured_type}",
            )

        try:
            # Import and instantiate robot based on type
            if robot_type == "so100_follower":
                from tauro_edge.robots.so100_follower import SO100Follower, SO100FollowerConfig

                config = SO100FollowerConfig(id=robot_id, port=port)
                robot = SO100Follower(config)
            elif robot_type == "so101_follower":
                from tauro_edge.robots.so101_follower import SO101Follower, SO101FollowerConfig

                config = SO101FollowerConfig(id=robot_id, port=port)
                robot = SO101Follower(config)
            else:
                return robot_service_pb2.ConnectResponse(
                    success=False, message=f"Unknown robot type: {robot_type}"
                )

            # Connect to robot without forcing calibration
            robot.connect(calibrate=False)
            self._register_robot(robot_id, robot)

            # Prepare robot info
            robot_info = robot_service_pb2.RobotInfo(
                robot_id=robot_id,
                robot_type=robot_type,
                motor_names=list(robot.motor_configs.keys()),
            )

            # Add observation and action space info
            for name, feature in robot.observation_space.items():
                robot_info.observation_space[name].CopyFrom(
                    feature_info_to_proto(
                        FeatureInfo(
                            shape=feature["shape"],
                            dtype=feature["dtype"].name,
                            names=feature.get("names"),
                        )
                    )
                )

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

        except Exception as e:
            logger.error(f"Error connecting to robot {robot_id}: {e}")
            return robot_service_pb2.ConnectResponse(
                success=False, message=f"Error connecting to robot: {str(e)}"
            )

    def Disconnect(self, request, context):
        """Handle robot disconnection request."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            return robot_service_pb2.DisconnectResponse(
                success=False, message=f"Robot {robot_id} not found"
            )

        try:
            # Stop any active control streams
            if robot_id in self._stream_tasks:
                self._stream_tasks[robot_id].cancel()
                del self._stream_tasks[robot_id]

            if robot_id in self._control_streams:
                del self._control_streams[robot_id]

            # Disconnect robot
            robot.disconnect()
            del self.robots[robot_id]

            return robot_service_pb2.DisconnectResponse(
                success=True, message=f"Successfully disconnected robot {robot_id}"
            )

        except Exception as e:
            logger.error(f"Error disconnecting robot {robot_id}: {e}")
            return robot_service_pb2.DisconnectResponse(
                success=False, message=f"Error disconnecting robot: {str(e)}"
            )

    def Calibrate(self, request, context):
        """Handle robot calibration request."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Robot {robot_id} not found")
            return robot_service_pb2.CalibrateResponse(success=False, message="Robot not found")

        try:
            # Perform calibration
            robot.calibrate()

            # Get calibration data
            calibrations = {}
            for motor_name, motor in robot.motors.items():
                calibrations[motor_name] = motor_calibration_to_proto(motor.calibration)

            return robot_service_pb2.CalibrateResponse(
                success=True,
                message=f"Successfully calibrated robot {robot_id}",
                calibrations=calibrations,
            )

        except Exception as e:
            logger.error(f"Error calibrating robot {robot_id}: {e}")
            return robot_service_pb2.CalibrateResponse(
                success=False, message=f"Error calibrating robot: {str(e)}"
            )

    def GetState(self, request, context):
        """Get current robot state."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Robot {robot_id} not found")
            return robot_service_pb2.RobotState()

        try:
            # Get observation from robot
            obs = robot.get_observation()

            # Convert to RobotState
            joints = {}
            motor_names = list(robot.motor_configs.keys())

            # Check if observation contains individual motor positions
            motor_positions = {}
            for key, value in obs.items():
                if key.endswith(".pos"):
                    motor_name = key.removesuffix(".pos")
                    motor_positions[motor_name] = value

            if motor_positions:
                # Use individual motor positions
                for motor_name in motor_names:
                    position = motor_positions.get(motor_name, 0.0)
                    joints[motor_name] = JointState(
                        position=float(position),
                        velocity=0.0,  # Not available in current implementation
                        torque=0.0,  # Not available in current implementation
                        temperature=0.0,  # Not available in current implementation
                        is_calibrated=robot.is_calibrated,
                    )
            elif "observation.state" in obs:
                # Use packed state format
                state_data = obs["observation.state"]
                num_motors = len(motor_names)

                for i, motor_name in enumerate(motor_names):
                    joints[motor_name] = JointState(
                        position=float(state_data[i]),
                        velocity=float(state_data[i + num_motors]),
                        torque=float(state_data[i + 2 * num_motors]),
                        temperature=0.0,  # Not available in current implementation
                        is_calibrated=robot.is_calibrated,
                    )

            # Prepare sensor data
            sensors = {}
            end_effector_state = None

            for key, value in obs.items():
                if key == "end_effector.pos":
                    # Extract end effector position
                    if end_effector_state is None:
                        end_effector_state = {}
                    end_effector_state["position"] = value
                elif key == "end_effector.orientation":
                    # Extract end effector orientation
                    if end_effector_state is None:
                        end_effector_state = {}
                    end_effector_state["orientation"] = value
                elif key != "observation.state" and not key.endswith(".pos"):
                    # Other sensor data
                    if isinstance(value, np.ndarray):
                        sensors[key] = value
                    else:
                        sensors[key] = np.array(value)

            robot_state = RobotState(
                timestamp=time.time(),
                robot_id=robot_id,
                joints=joints,
                sensors=sensors,
                status=RobotStatus.READY if robot.is_calibrated else RobotStatus.CONNECTED,
                end_effector=end_effector_state,
            )

            return robot_state_to_proto(robot_state)

        except Exception as e:
            logger.error(f"Error getting state for robot {robot_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return robot_service_pb2.RobotState()

    def SendAction(self, request, context):
        """Send single action command to robot."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            return robot_service_pb2.ActionResponse(
                success=False, message=f"Robot {robot_id} not found"
            )

        try:
            # Check which type of command was sent
            if request.HasField("joint_command"):
                # Joint space command
                joint_cmd = request.joint_command
                action = {}
                for motor_name, position in joint_cmd.positions.items():
                    action[f"{motor_name}.pos"] = position
            elif request.HasField("end_effector_command"):
                # End effector space command
                ee_cmd = request.end_effector_command
                action = {
                    "end_effector": {
                        "delta_x": ee_cmd.delta_x,
                        "delta_y": ee_cmd.delta_y,
                        "delta_z": ee_cmd.delta_z,
                        "gripper": ee_cmd.gripper,
                    }
                }
                if ee_cmd.delta_orientation:
                    action["end_effector"]["delta_orientation"] = list(ee_cmd.delta_orientation)
            else:
                raise ValueError("Invalid action command")

            # Send action
            robot.send_action(action)

            return robot_service_pb2.ActionResponse(
                success=True, message="Action sent successfully"
            )

        except Exception as e:
            logger.error(f"Error sending action to robot {robot_id}: {e}")
            return robot_service_pb2.ActionResponse(
                success=False, message=f"Error sending action: {str(e)}"
            )

    async def StreamControl(self, request_iterator, context):
        """Handle bidirectional streaming for real-time control."""
        robot_id = None

        try:
            async for command in request_iterator:
                robot_id = command.robot_id
                robot = self._get_robot(robot_id)

                if not robot:
                    logger.error(f"Robot {robot_id} not found in stream")
                    continue

                # Process control command
                timestamp = proto_to_timestamp(command.timestamp)
                control_mode = proto_to_control_mode(command.control_mode)

                # Check which type of command was sent
                if command.HasField("joint_command"):
                    # Joint space command
                    joint_cmd = command.joint_command
                    action = {}
                    for motor_name, position in joint_cmd.positions.items():
                        action[f"{motor_name}.pos"] = position
                    robot.send_action(action)
                elif command.HasField("end_effector_command"):
                    # End effector space command
                    ee_cmd = command.end_effector_command
                    action = {
                        "end_effector": {
                            "delta_x": ee_cmd.delta_x,
                            "delta_y": ee_cmd.delta_y,
                            "delta_z": ee_cmd.delta_z,
                            "gripper": ee_cmd.gripper,
                        }
                    }
                    if ee_cmd.delta_orientation:
                        action["end_effector"]["delta_orientation"] = list(ee_cmd.delta_orientation)
                    robot.send_action(action)
                else:
                    logger.warning(f"No valid command in ControlCommand for robot {robot_id}")

                # Get current state and yield back
                obs = robot.get_observation()

                # Convert to RobotState
                joints = {}
                if "observation.state" in obs:
                    state_data = obs["observation.state"]
                    num_motors = len(motor_names)

                    for i, motor_name in enumerate(motor_names):
                        joints[motor_name] = JointState(
                            position=float(state_data[i]),
                            velocity=float(state_data[i + num_motors]),
                            torque=float(state_data[i + 2 * num_motors]),
                            temperature=0.0,
                            is_calibrated=robot.is_calibrated,
                        )

                sensors = {k: v for k, v in obs.items() if k != "observation.state"}

                robot_state = RobotState(
                    timestamp=time.time(),
                    robot_id=robot_id,
                    joints=joints,
                    sensors=sensors,
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
        """Check health status of the server and connected robots."""
        components = {}

        # Check each connected robot
        for robot_id, robot in self.robots.items():
            try:
                # Try to get observation as health check
                obs = robot.get_observation()
                components[robot_id] = robot_service_pb2.ComponentHealth(
                    is_healthy=True, status="connected", message="Robot responding normally"
                )
            except Exception as e:
                components[robot_id] = robot_service_pb2.ComponentHealth(
                    is_healthy=False, status="error", message=str(e)
                )

        # Server is healthy if we can respond
        components["server"] = robot_service_pb2.ComponentHealth(
            is_healthy=True, status="running", message="Server is running"
        )

        return robot_service_pb2.HealthStatus(
            is_healthy=all(c.is_healthy for c in components.values()),
            components=components,
            timestamp=timestamp_to_proto(time.time()),
        )


async def serve_async(host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT):
    """Run the gRPC server asynchronously."""
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ]
    )

    servicer = RobotControlServicer()
    robot_service_pb2_grpc.add_RobotControlServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        await server.stop(5)


def serve(host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT):
    """Run the gRPC server (blocking)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    servicer = RobotControlServicer()
    robot_service_pb2_grpc.add_RobotControlServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting gRPC server on {listen_addr}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
