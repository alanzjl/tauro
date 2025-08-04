"""MuJoCo-based simulator server for robot control."""

import asyncio
import logging
import time
from concurrent import futures
from pathlib import Path

import grpc
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
    robot_state_to_proto,
    timestamp_to_proto,
)
from tauro_edge.robots.simulated import SimulatedRobot, SimulatedRobotConfig

logger = logging.getLogger(__name__)


class SimulatorServicer(robot_service_pb2_grpc.RobotControlServiceServicer):
    """gRPC service implementation for simulated robot control."""

    def __init__(self, config_path: Path | None = None):
        self.robots: dict[str, SimulatedRobot] = {}
        self._control_streams: dict[str, asyncio.Queue] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}

        # Load configuration if provided
        self._robot_configs = {}
        if config_path:
            self._load_robot_configs(config_path)

    def _load_robot_configs(self, config_path: Path) -> dict:
        """Load robot configurations from YAML file."""
        if not config_path.exists():
            logger.warning(f"Robot config file not found at {config_path}")
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                self._robot_configs = config.get("simulated_robots", {})
                logger.info(f"Loaded {len(self._robot_configs)} simulated robot configs")
                return self._robot_configs
        except Exception as e:
            logger.error(f"Error loading robot configs: {e}")
            return {}

    def _get_robot(self, robot_id: str) -> SimulatedRobot | None:
        """Get robot instance by ID."""
        return self.robots.get(robot_id)

    def _register_robot(self, robot_id: str, robot: SimulatedRobot):
        """Register a robot instance."""
        self.robots[robot_id] = robot
        logger.info(f"Registered simulated robot: {robot_id}")

    def Connect(self, request, context):
        """Handle robot connection request."""
        robot_id = request.robot_id
        robot_type = request.robot_type

        # Check if robot already connected
        if robot_id in self.robots:
            return robot_service_pb2.ConnectResponse(
                success=False, message=f"Simulated robot {robot_id} is already connected"
            )

        try:
            # Create simulated robot configuration
            config = SimulatedRobotConfig(
                id=robot_id,
                robot_type=robot_type,
            )

            # Create and connect simulated robot
            robot = SimulatedRobot(config)
            robot.connect(calibrate=False)  # Don't force calibration on connect
            self._register_robot(robot_id, robot)

            # Prepare robot info
            robot_info = robot_service_pb2.RobotInfo(
                robot_id=robot_id,
                robot_type=robot_type,
                motor_names=list(robot.motor_configs.keys()),
            )

            # Add observation space info
            for space, space_feature in robot.observation_space.items():
                if space == "joint":
                    for space_type, type_feature in space_feature.items():
                        # "position" or "velocity"
                        for joint_name, joint_feature in type_feature.items():
                            robot_info.observation_space[
                                f"{space}.{space_type}.{joint_name}"
                            ].CopyFrom(
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

            # Add action space info
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
                message=f"Successfully connected to simulated robot {robot_id}",
                robot_info=robot_info,
            )

        except Exception as e:
            logger.error(f"Error connecting to simulated robot {robot_id}: {e}")
            return robot_service_pb2.ConnectResponse(
                success=False, message=f"Error connecting to simulated robot: {str(e)}"
            )

    def Disconnect(self, request, context):
        """Handle robot disconnection request."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            return robot_service_pb2.DisconnectResponse(
                success=False, message=f"Simulated robot {robot_id} not found"
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
                success=True, message=f"Successfully disconnected simulated robot {robot_id}"
            )

        except Exception as e:
            logger.error(f"Error disconnecting simulated robot {robot_id}: {e}")
            return robot_service_pb2.DisconnectResponse(
                success=False, message=f"Error disconnecting simulated robot: {str(e)}"
            )

    def Calibrate(self, request, context):
        """Handle robot calibration request."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Simulated robot {robot_id} not found")
            return robot_service_pb2.CalibrateResponse(success=False, message="Robot not found")

        try:
            # Perform calibration
            robot.calibrate()

            # Get calibration data
            calibrations = {}
            for motor_name in robot.motor_names:
                if motor_name in robot.calibration:
                    calibrations[motor_name] = motor_calibration_to_proto(
                        robot.calibration[motor_name]
                    )

            return robot_service_pb2.CalibrateResponse(
                success=True,
                message=f"Successfully calibrated simulated robot {robot_id}",
                calibrations=calibrations,
            )

        except Exception as e:
            logger.error(f"Error calibrating simulated robot {robot_id}: {e}")
            return robot_service_pb2.CalibrateResponse(
                success=False, message=f"Error calibrating robot: {str(e)}"
            )

    def GetState(self, request, context):
        """Get current robot state."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Simulated robot {robot_id} not found")
            return robot_service_pb2.RobotState()

        try:
            # Get observation from robot
            obs = robot.get_observation()

            # Convert to proto format
            joints = {}
            if "joints" in obs:
                joint_data = obs["joints"]
                for motor_name in robot.motor_names:
                    joints[motor_name] = JointState(
                        position=float(joint_data["position"].get(motor_name, 0)),
                        velocity=float(joint_data["velocity"].get(motor_name, 0)),
                        torque=0.0,  # Simulated robot doesn't provide torque yet
                        temperature=25.0,  # Default temperature
                        is_calibrated=robot.is_calibrated,
                    )

            # Build robot state
            robot_state = RobotState(
                timestamp=time.time(),
                robot_id=robot_id,
                joints=joints,
                sensors={},  # No additional sensors for now
                status=RobotStatus.READY if robot.is_calibrated else RobotStatus.CONNECTED,
            )

            return robot_state_to_proto(robot_state)

        except Exception as e:
            logger.error(f"Error getting state for simulated robot {robot_id}: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return robot_service_pb2.RobotState()

    def SendAction(self, request, context):
        """Send single action command to robot."""
        robot_id = request.robot_id
        robot = self._get_robot(robot_id)

        if not robot:
            return robot_service_pb2.ActionResponse(
                success=False, message=f"Simulated robot {robot_id} not found"
            )

        try:
            # Check which type of command was sent
            if request.HasField("joint_command"):
                # Joint space command
                joint_cmd = request.joint_command
                action = {"joints": {"position": dict(joint_cmd.positions)}}
            elif request.HasField("end_effector_command"):
                # End effector space command (not yet supported in simulation)
                return robot_service_pb2.ActionResponse(
                    success=False, message="End effector control not yet supported in simulation"
                )
            else:
                # Handle individual motor commands
                action = {}
                # The request might have individual motor commands embedded
                # For now, return error if no valid command type
                return robot_service_pb2.ActionResponse(
                    success=False, message="Invalid action command"
                )

            # Send action
            robot.send_action(action)

            return robot_service_pb2.ActionResponse(
                success=True, message="Action sent successfully"
            )

        except Exception as e:
            logger.error(f"Error sending action to simulated robot {robot_id}: {e}")
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
                    logger.error(f"Simulated robot {robot_id} not found in stream")
                    continue

                # Process control command
                if command.HasField("joint_command"):
                    # Joint space command
                    joint_cmd = command.joint_command
                    action = {}
                    for motor_name, position in joint_cmd.positions.items():
                        action[f"{motor_name}.pos"] = position
                    robot.send_action(action)
                elif command.HasField("end_effector_command"):
                    # End effector space command (not yet supported)
                    logger.warning(
                        f"End effector control not yet supported for simulated robot {robot_id}"
                    )
                else:
                    logger.warning(
                        f"No valid command in ControlCommand for simulated robot {robot_id}"
                    )

                # Get current state and yield back
                obs = robot.get_observation()

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
            logger.info(f"Stream cancelled for simulated robot {robot_id}")
        except Exception as e:
            logger.error(f"Error in control stream: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

    def HealthCheck(self, request, context):
        """Check health status of the simulator and connected robots."""
        components = {}

        # Check each connected robot
        for robot_id, robot in self.robots.items():
            try:
                # Try to get observation as health check
                robot.get_observation()
                components[robot_id] = robot_service_pb2.ComponentHealth(
                    is_healthy=True,
                    status="connected",
                    message="Simulated robot responding normally",
                )
            except Exception as e:
                components[robot_id] = robot_service_pb2.ComponentHealth(
                    is_healthy=False, status="error", message=str(e)
                )

        # Simulator is healthy if we can respond
        components["simulator"] = robot_service_pb2.ComponentHealth(
            is_healthy=True, status="running", message="Simulator is running"
        )

        return robot_service_pb2.HealthStatus(
            is_healthy=all(c.is_healthy for c in components.values()),
            components=components,
            timestamp=timestamp_to_proto(time.time()),
        )


async def serve_async(
    host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT, config_path: Path | None = None
):
    """Run the simulator gRPC server asynchronously."""
    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ]
    )

    servicer = SimulatorServicer(config_path)
    robot_service_pb2_grpc.add_RobotControlServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting simulator gRPC server on {listen_addr}")
    await server.start()

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down simulator server...")
        await server.stop(5)


def serve(
    host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT, config_path: Path | None = None
):
    """Run the simulator gRPC server (blocking)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
        ],
    )

    servicer = SimulatorServicer(config_path)
    robot_service_pb2_grpc.add_RobotControlServiceServicer_to_server(servicer, server)

    listen_addr = f"{host}:{port}"
    server.add_insecure_port(listen_addr)

    logger.info(f"Starting simulator gRPC server on {listen_addr}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down simulator server...")
        server.stop(5)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo Robot Simulator Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=50053, help="Port to bind to")
    parser.add_argument("--config", type=Path, help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    serve(args.host, args.port, args.config)
