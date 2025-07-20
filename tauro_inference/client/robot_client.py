"""gRPC client for communicating with tauro_edge server."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator

import grpc
from google.protobuf import empty_pb2

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT, MAX_MESSAGE_SIZE
from tauro_common.proto import robot_service_pb2, robot_service_pb2_grpc
from tauro_common.types.robot_types import (
    ControlCommand,
    MotorCalibration,
    RobotState,
)
from tauro_common.utils.proto_utils import (
    control_mode_to_proto,
    proto_to_motor_calibration,
    proto_to_robot_state,
    timestamp_to_proto,
)

logger = logging.getLogger(__name__)


class RobotClient:
    """Client for communicating with tauro_edge robot control server."""

    def __init__(self, host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT):
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.channel: grpc.Channel | None = None
        self.stub: robot_service_pb2_grpc.RobotControlServiceStub | None = None
        self._stream_task: asyncio.Task | None = None
        self._command_queue: asyncio.Queue | None = None
        self._state_queue: asyncio.Queue | None = None

    def connect_to_server(self):
        """Establish connection to the gRPC server."""
        if self.channel is not None:
            raise RuntimeError("Already connected to server")

        logger.info(f"Connecting to gRPC server at {self.address}")

        self.channel = grpc.insecure_channel(
            self.address,
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        self.stub = robot_service_pb2_grpc.RobotControlServiceStub(self.channel)

        # Test connection with health check
        try:
            health = self.health_check()
            logger.info(f"Connected to server. Health: {health.is_healthy}")
        except grpc.RpcError as e:
            self.channel.close()
            self.channel = None
            self.stub = None
            raise ConnectionError(f"Failed to connect to server: {e}") from e

    def disconnect_from_server(self):
        """Close connection to the gRPC server."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
            self.stub = None
            logger.info("Disconnected from server")

    def connect_robot(self, robot_id: str, robot_type: str, config: dict | None = None) -> bool:
        """Connect to a specific robot through the server."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.ConnectRequest(
            robot_id=robot_id, robot_type=robot_type, config=config or {}
        )

        try:
            response = self.stub.Connect(request)
            if response.success:
                logger.info(f"Connected to robot {robot_id}: {response.message}")
                return True
            else:
                logger.error(f"Failed to connect to robot {robot_id}: {response.message}")
                return False
        except grpc.RpcError as e:
            logger.error(f"gRPC error connecting to robot: {e}")
            return False

    def disconnect_robot(self, robot_id: str) -> bool:
        """Disconnect from a specific robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.DisconnectRequest(robot_id=robot_id)

        try:
            response = self.stub.Disconnect(request)
            if response.success:
                logger.info(f"Disconnected from robot {robot_id}")
                return True
            else:
                logger.error(f"Failed to disconnect from robot {robot_id}: {response.message}")
                return False
        except grpc.RpcError as e:
            logger.error(f"gRPC error disconnecting from robot: {e}")
            return False

    def calibrate_robot(self, robot_id: str) -> dict[str, MotorCalibration] | None:
        """Calibrate a robot and return calibration data."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.CalibrateRequest(robot_id=robot_id)

        try:
            response = self.stub.Calibrate(request)
            if response.success:
                logger.info(f"Calibrated robot {robot_id}")
                calibrations = {
                    name: proto_to_motor_calibration(calib)
                    for name, calib in response.calibrations.items()
                }
                return calibrations
            else:
                logger.error(f"Failed to calibrate robot {robot_id}: {response.message}")
                return None
        except grpc.RpcError as e:
            logger.error(f"gRPC error calibrating robot: {e}")
            return None

    def get_robot_state(self, robot_id: str) -> RobotState | None:
        """Get current state of a robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.GetStateRequest(robot_id=robot_id)

        try:
            proto_state = self.stub.GetState(request)
            return proto_to_robot_state(proto_state)
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting robot state: {e}")
            return None

    def send_action(self, robot_id: str, action: dict) -> bool:
        """Send a single action command to a robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        # Create control command based on action type
        control_cmd = robot_service_pb2.ControlCommand(
            timestamp=timestamp_to_proto(time.time()),
            robot_id=robot_id,
        )

        # Check if it's a joint command or end effector command
        if "positions" in action or all(k.endswith(".pos") for k in action.keys()):
            # Joint command
            joint_cmd = robot_service_pb2.JointCommand()
            if "positions" in action:
                joint_cmd.positions.update(action["positions"])
            else:
                # Convert motor_name.pos format to positions dict
                positions = {k.removesuffix(".pos"): v for k, v in action.items()}
                joint_cmd.positions.update(positions)
            control_cmd.joint_command.CopyFrom(joint_cmd)
        elif "end_effector" in action:
            # End effector command
            ee_data = action["end_effector"]
            ee_cmd = robot_service_pb2.EndEffectorCommand(
                delta_x=ee_data.get("delta_x", 0.0),
                delta_y=ee_data.get("delta_y", 0.0),
                delta_z=ee_data.get("delta_z", 0.0),
                gripper=ee_data.get("gripper", 0.0),
            )
            if "delta_orientation" in ee_data:
                ee_cmd.delta_orientation.extend(ee_data["delta_orientation"])
            control_cmd.end_effector_command.CopyFrom(ee_cmd)
        else:
            raise ValueError(f"Invalid action format: {action}")

        try:
            response = self.stub.SendAction(control_cmd)
            if not response.success:
                logger.error(f"Failed to send action: {response.message}")
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error sending action: {e}")
            return False

    def health_check(self) -> robot_service_pb2.HealthStatus:
        """Check health of the server and connected robots."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        try:
            return self.stub.HealthCheck(empty_pb2.Empty())
        except grpc.RpcError as e:
            logger.error(f"gRPC error in health check: {e}")
            raise

    async def start_streaming_control(self, robot_id: str):
        """Start streaming control session with a robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        if self._stream_task is not None:
            raise RuntimeError("Streaming already active")

        self._command_queue = asyncio.Queue()
        self._state_queue = asyncio.Queue()

        self._stream_task = asyncio.create_task(self._streaming_control_loop(robot_id))

        logger.info(f"Started streaming control for robot {robot_id}")

    async def stop_streaming_control(self):
        """Stop the streaming control session."""
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
            self._command_queue = None
            self._state_queue = None
            logger.info("Stopped streaming control")

    async def send_streaming_command(self, command: ControlCommand):
        """Send a command in streaming mode."""
        if self._command_queue is None:
            raise RuntimeError("Streaming not active")
        await self._command_queue.put(command)

    async def get_streaming_state(self) -> RobotState | None:
        """Get the latest state in streaming mode."""
        if self._state_queue is None:
            raise RuntimeError("Streaming not active")
        try:
            return await asyncio.wait_for(self._state_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    async def _streaming_control_loop(self, robot_id: str):
        """Internal loop for handling streaming control."""
        try:
            # Create async stub
            async_channel = grpc.aio.insecure_channel(
                self.address,
                options=[
                    ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                ],
            )
            async_stub = robot_service_pb2_grpc.RobotControlServiceStub(async_channel)

            # Generator for sending commands
            async def command_generator():
                while True:
                    command = await self._command_queue.get()
                    proto_command = robot_service_pb2.ControlCommand(
                        timestamp=timestamp_to_proto(command.timestamp),
                        robot_id=command.robot_id,
                        control_mode=control_mode_to_proto(command.control_mode),
                    )
                    # Set the appropriate command type
                    if hasattr(command, "joint_command") and command.joint_command:
                        proto_command.joint_command.CopyFrom(command.joint_command)
                    elif hasattr(command, "end_effector_command") and command.end_effector_command:
                        proto_command.end_effector_command.CopyFrom(command.end_effector_command)
                    yield proto_command

            # Start streaming
            stream = async_stub.StreamControl(command_generator())

            # Process incoming states
            async for proto_state in stream:
                state = proto_to_robot_state(proto_state)
                # Clear queue and add latest state
                while not self._state_queue.empty():
                    try:
                        self._state_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                await self._state_queue.put(state)

        except asyncio.CancelledError:
            logger.info("Streaming control cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in streaming control: {e}")
            raise
        finally:
            await async_channel.close()


class AsyncRobotClient:
    """Async version of the robot client."""

    def __init__(self, host: str = DEFAULT_GRPC_HOST, port: int = DEFAULT_GRPC_PORT):
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.channel: grpc.aio.Channel | None = None
        self.stub: robot_service_pb2_grpc.RobotControlServiceStub | None = None

    async def connect_to_server(self):
        """Establish connection to the gRPC server."""
        if self.channel is not None:
            raise RuntimeError("Already connected to server")

        logger.info(f"Connecting to gRPC server at {self.address}")

        self.channel = grpc.aio.insecure_channel(
            self.address,
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
                ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ],
        )
        self.stub = robot_service_pb2_grpc.RobotControlServiceStub(self.channel)

        # Test connection with health check
        try:
            health = await self.health_check()
            logger.info(f"Connected to server. Health: {health.is_healthy}")
        except grpc.RpcError as e:
            await self.channel.close()
            self.channel = None
            self.stub = None
            raise ConnectionError(f"Failed to connect to server: {e}") from e

    async def disconnect_from_server(self):
        """Close connection to the gRPC server."""
        if self.channel is not None:
            await self.channel.close()
            self.channel = None
            self.stub = None
            logger.info("Disconnected from server")

    async def connect_robot(
        self, robot_id: str, robot_type: str, config: dict | None = None
    ) -> bool:
        """Connect to a specific robot through the server."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.ConnectRequest(
            robot_id=robot_id, robot_type=robot_type, config=config or {}
        )

        try:
            response = await self.stub.Connect(request)
            if response.success:
                logger.info(f"Connected to robot {robot_id}: {response.message}")
                return True
            else:
                logger.error(f"Failed to connect to robot {robot_id}: {response.message}")
                return False
        except grpc.RpcError as e:
            logger.error(f"gRPC error connecting to robot: {e}")
            return False

    async def disconnect_robot(self, robot_id: str) -> bool:
        """Disconnect from a specific robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.DisconnectRequest(robot_id=robot_id)

        try:
            response = await self.stub.Disconnect(request)
            if response.success:
                logger.info(f"Disconnected from robot {robot_id}")
                return True
            else:
                logger.error(f"Failed to disconnect from robot {robot_id}: {response.message}")
                return False
        except grpc.RpcError as e:
            logger.error(f"gRPC error disconnecting from robot: {e}")
            return False

    async def calibrate_robot(self, robot_id: str) -> dict[str, MotorCalibration] | None:
        """Calibrate a robot and return calibration data."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.CalibrateRequest(robot_id=robot_id)

        try:
            response = await self.stub.Calibrate(request)
            if response.success:
                logger.info(f"Calibrated robot {robot_id}")
                calibrations = {
                    name: proto_to_motor_calibration(calib)
                    for name, calib in response.calibrations.items()
                }
                return calibrations
            else:
                logger.error(f"Failed to calibrate robot {robot_id}: {response.message}")
                return None
        except grpc.RpcError as e:
            logger.error(f"gRPC error calibrating robot: {e}")
            return None

    async def get_robot_state(self, robot_id: str) -> RobotState | None:
        """Get current state of a robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        request = robot_service_pb2.GetStateRequest(robot_id=robot_id)

        try:
            proto_state = await self.stub.GetState(request)
            return proto_to_robot_state(proto_state)
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting robot state: {e}")
            return None

    async def send_action(self, robot_id: str, action: dict) -> bool:
        """Send a single action command to a robot."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        # Create control command based on action type
        control_cmd = robot_service_pb2.ControlCommand(
            timestamp=timestamp_to_proto(time.time()),
            robot_id=robot_id,
        )

        # Check if it's a joint command or end effector command
        if "positions" in action or all(k.endswith(".pos") for k in action.keys()):
            # Joint command
            joint_cmd = robot_service_pb2.JointCommand()
            if "positions" in action:
                joint_cmd.positions.update(action["positions"])
            else:
                # Convert motor_name.pos format to positions dict
                positions = {k.removesuffix(".pos"): v for k, v in action.items()}
                joint_cmd.positions.update(positions)
            control_cmd.joint_command.CopyFrom(joint_cmd)
        elif "end_effector" in action:
            # End effector command
            ee_data = action["end_effector"]
            ee_cmd = robot_service_pb2.EndEffectorCommand(
                delta_x=ee_data.get("delta_x", 0.0),
                delta_y=ee_data.get("delta_y", 0.0),
                delta_z=ee_data.get("delta_z", 0.0),
                gripper=ee_data.get("gripper", 0.0),
            )
            if "delta_orientation" in ee_data:
                ee_cmd.delta_orientation.extend(ee_data["delta_orientation"])
            control_cmd.end_effector_command.CopyFrom(ee_cmd)
        else:
            raise ValueError(f"Invalid action format: {action}")

        try:
            response = await self.stub.SendAction(control_cmd)
            if not response.success:
                logger.error(f"Failed to send action: {response.message}")
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error sending action: {e}")
            return False

    async def health_check(self) -> robot_service_pb2.HealthStatus:
        """Check health of the server and connected robots."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        try:
            return await self.stub.HealthCheck(empty_pb2.Empty())
        except grpc.RpcError as e:
            logger.error(f"gRPC error in health check: {e}")
            raise

    async def stream_control(
        self, robot_id: str, command_iterator: AsyncIterator[ControlCommand]
    ) -> AsyncIterator[RobotState]:
        """Stream control commands and receive state updates."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

        # Convert commands to protobuf
        async def proto_command_generator():
            async for command in command_iterator:
                proto_command = robot_service_pb2.ControlCommand(
                    timestamp=timestamp_to_proto(command.timestamp),
                    robot_id=command.robot_id,
                    control_mode=control_mode_to_proto(command.control_mode),
                )
                # Set the appropriate command type
                if hasattr(command, "joint_command") and command.joint_command:
                    proto_command.joint_command.CopyFrom(command.joint_command)
                elif hasattr(command, "end_effector_command") and command.end_effector_command:
                    proto_command.end_effector_command.CopyFrom(command.end_effector_command)
                yield proto_command

        # Stream and convert responses
        stream = self.stub.StreamControl(proto_command_generator())
        async for proto_state in stream:
            yield proto_to_robot_state(proto_state)
