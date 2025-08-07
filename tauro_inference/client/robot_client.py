"""gRPC client for communicating with tauro_edge server."""

import asyncio
import atexit
import inspect
import logging
import time
from collections.abc import AsyncIterator, Callable
from functools import wraps
from typing import Any

import grpc
import grpc.aio
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

# Global event loop for synchronous operations
_sync_loop = None
_sync_loop_thread = None


def get_sync_loop():
    """Get or create the global event loop for synchronous operations."""
    global _sync_loop, _sync_loop_thread

    if _sync_loop is None or _sync_loop.is_closed():
        import threading

        # Create a new event loop in a separate thread
        def run_loop():
            global _sync_loop
            _sync_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_sync_loop)
            _sync_loop.run_forever()

        _sync_loop_thread = threading.Thread(target=run_loop, daemon=True)
        _sync_loop_thread.start()

        # Wait for the loop to be ready
        while _sync_loop is None:
            time.sleep(0.001)

    return _sync_loop


def cleanup_sync_loop():
    """Clean up the sync event loop."""
    global _sync_loop, _sync_loop_thread
    if _sync_loop is not None and not _sync_loop.is_closed():
        _sync_loop.call_soon_threadsafe(_sync_loop.stop)
        if _sync_loop_thread is not None:
            _sync_loop_thread.join(timeout=1.0)
        _sync_loop = None
        _sync_loop_thread = None


# Register cleanup on exit
atexit.register(cleanup_sync_loop)


def sync_async_method(async_func: Callable) -> Callable:
    """Decorator to create both sync and async versions of a method."""

    @wraps(async_func)
    def wrapper(self, *args, **kwargs):
        if inspect.iscoroutinefunction(async_func):
            # Check if we're already in an async context
            try:
                loop = asyncio.get_running_loop()
                # Already in async context, return coroutine
                return async_func(self, *args, **kwargs)
            except RuntimeError:
                # No running loop, use sync mode
                loop = get_sync_loop()
                future = asyncio.run_coroutine_threadsafe(async_func(self, *args, **kwargs), loop)
                return future.result()
        else:
            return async_func(self, *args, **kwargs)

    return wrapper


class RobotClient:
    """
    Unified client for communicating with tauro_edge robot control server.
    Supports both synchronous and asynchronous operations.

    All methods can be called both synchronously and asynchronously:
    - Sync: client.connect()
    - Async: await client.connect()
    """

    def __init__(
        self,
        robot_id: str,
        robot_type: str,
        host: str = DEFAULT_GRPC_HOST,
        port: int = DEFAULT_GRPC_PORT,
        config: dict | None = None,
    ):
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.config = config

        # Unified channel and stub (async by default)
        self.channel: grpc.aio.Channel | None = None
        self.stub: robot_service_pb2_grpc.RobotControlServiceStub | None = None

        # Streaming control state
        self._stream_task: asyncio.Task | None = None
        self._command_queue: asyncio.Queue | None = None
        self._state_queue: asyncio.Queue | None = None

        # Track if we're using sync mode
        self._sync_mode = False

    def __enter__(self):
        """Sync context manager entry."""
        self._sync_mode = True
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        self.disconnect()
        self._sync_mode = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    @sync_async_method
    async def connect(self):
        """Establish connection to the gRPC server and connect to robot."""
        if self.channel is not None:
            raise RuntimeError("Already connected to server")

        logger.info(f"Connecting to gRPC server at {self.address}")

        # Use the sync event loop if in sync mode
        if self._sync_mode:
            loop = get_sync_loop()

            # Create channel with the sync loop
            def create_channel():
                asyncio.set_event_loop(loop)
                return grpc.aio.insecure_channel(
                    self.address,
                    options=[
                        ("grpc.max_send_message_length", MAX_MESSAGE_SIZE),
                        ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
                    ],
                )

            import concurrent.futures

            future = concurrent.futures.ThreadPoolExecutor().submit(create_channel)
            self.channel = future.result()
        else:
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

        # Connect to the specific robot
        self._ensure_connected()

        request = robot_service_pb2.ConnectRequest(
            robot_id=self.robot_id, robot_type=self.robot_type, config=self.config or {}
        )

        try:
            response = await self.stub.Connect(request)
            if response.success:
                logger.info(f"Connected to robot {self.robot_id}: {response.message}")
                return True
            else:
                logger.error(f"Failed to connect to robot {self.robot_id}: {response.message}")
                return False
        except grpc.RpcError as e:
            logger.error(f"gRPC error connecting to robot: {e}")
            return False

    @sync_async_method
    async def disconnect(self):
        """Disconnect from the robot then close connection to the gRPC server."""
        if self.stub is not None:
            request = robot_service_pb2.DisconnectRequest(robot_id=self.robot_id)
            try:
                response = await self.stub.Disconnect(request)
                if response.success:
                    logger.info(f"Disconnected from robot {self.robot_id}")
                else:
                    logger.error(
                        f"Failed to disconnect from robot {self.robot_id}: {response.message}"
                    )
            except grpc.RpcError as e:
                logger.error(f"gRPC error disconnecting from robot: {e}")

        if self.channel is not None:
            await self.channel.close()
            self.channel = None
            self.stub = None
            logger.info("Disconnected from server")
        return True

    @sync_async_method
    async def calibrate(self) -> dict[str, MotorCalibration] | None:
        """Calibrate the robot and return calibration data."""
        self._ensure_connected()

        request = robot_service_pb2.CalibrateRequest(robot_id=self.robot_id)

        try:
            response = await self.stub.Calibrate(request)
            if response.success:
                logger.info(f"Calibrated robot {self.robot_id}")
                calibrations = {
                    name: proto_to_motor_calibration(calib)
                    for name, calib in response.calibrations.items()
                }
                return calibrations
            else:
                logger.error(f"Failed to calibrate robot {self.robot_id}: {response.message}")
                return None
        except grpc.RpcError as e:
            logger.error(f"gRPC error calibrating robot: {e}")
            return None

    @sync_async_method
    async def get_robot_state(self) -> RobotState | None:
        """Get current state of a robot."""
        self._ensure_connected()

        request = robot_service_pb2.GetStateRequest(robot_id=self.robot_id)

        try:
            proto_state = await self.stub.GetState(request)
            return proto_to_robot_state(proto_state)
        except grpc.RpcError as e:
            logger.error(f"gRPC error getting robot state: {e}")
            return None

    @sync_async_method
    async def get_observation(self) -> dict[str, Any]:
        """Get current observation from the robot (alias for get_robot_state with dict conversion)."""
        state = await self.get_robot_state()
        if state is None:
            return {}

        # Convert RobotState to observation dict format
        from tauro_common.utils.proto_utils import robot_state_to_dict

        return robot_state_to_dict(state)

    @sync_async_method
    async def send_action(self, action: dict) -> bool:
        """
        Send a single action command to the robot.
        Supported formats:
        {
            "joints": {
                "position": {
                    "jnt1": 0.0,
                    "jnt2": 0.0,
                    "jnt3": 0.0,
                    ...
                }
            }
        }
        or
        {
            "end_effector": {
                "delta_x": 0.0,
                "delta_y": 0.0,
                "delta_z": 0.0,
                "gripper": 0.0,
                "delta_orientation": [0.0, 0.0, 0.0],
            }
        }
        """
        self._ensure_connected()

        control_cmd = self._prepare_control_command(self.robot_id, action)

        try:
            response = await self.stub.SendAction(control_cmd)
            if not response.success:
                logger.error(f"Failed to send action: {response.message}")
            return response.success
        except grpc.RpcError as e:
            logger.error(f"gRPC error sending action: {e}")
            return False

    @sync_async_method
    async def health_check(self) -> robot_service_pb2.HealthStatus:
        """Check health of the server and connected robots."""
        self._ensure_connected()

        try:
            return await self.stub.HealthCheck(empty_pb2.Empty())
        except grpc.RpcError as e:
            logger.error(f"gRPC error in health check: {e}")
            raise

    # ============= Streaming Methods (Async Only) =============

    async def start_streaming_control(self):
        """Start streaming control session with the robot."""
        if self.channel is None:
            await self.connect()

        if self._stream_task is not None:
            raise RuntimeError("Streaming already active")

        self._command_queue = asyncio.Queue()
        self._state_queue = asyncio.Queue()

        self._stream_task = asyncio.create_task(self._streaming_control_loop())

        logger.info(f"Started streaming control for robot {self.robot_id}")

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

    async def stream_control(
        self, command_iterator: AsyncIterator[ControlCommand]
    ) -> AsyncIterator[RobotState]:
        """Stream control commands and receive state updates."""
        if self.channel is None:
            await self.connect()

        # Convert commands to protobuf
        async def proto_command_generator():
            async for command in command_iterator:
                yield self._convert_command_to_proto(command)

        # Stream and convert responses
        stream = self.stub.StreamControl(proto_command_generator())
        async for proto_state in stream:
            yield proto_to_robot_state(proto_state)

    # ============= Private Helper Methods =============

    def _ensure_connected(self):
        """Ensure client is connected to server."""
        if self.stub is None:
            raise RuntimeError("Not connected to server")

    def _prepare_control_command(self, robot_id: str, action: dict) -> Any:
        """Prepare a control command from an action dictionary."""
        control_cmd = robot_service_pb2.ControlCommand(
            timestamp=timestamp_to_proto(time.time()),
            robot_id=robot_id,
        )

        # Check if it's a joint command or end effector command
        if "joints" in action:
            # Joint command
            joint_cmd = robot_service_pb2.JointCommand()
            joint_cmd.positions.update(action["joints"]["position"])
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

        return control_cmd

    def _convert_command_to_proto(self, command: ControlCommand) -> Any:
        """Convert a ControlCommand to protobuf format."""
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
        return proto_command

    async def _streaming_control_loop(self):
        """Internal loop for handling streaming control."""
        try:
            # Generator for sending commands
            async def command_generator():
                while True:
                    command = await self._command_queue.get()
                    yield self._convert_command_to_proto(command)

            # Start streaming
            stream = self.stub.StreamControl(command_generator())

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
