"""Tests for the gRPC robot client."""

import time
from unittest.mock import MagicMock, patch

import grpc
import pytest

from tauro_common.proto import robot_service_pb2
from tauro_common.types.robot_types import (
    ControlCommand,
    ControlMode,
    RobotStatus,
)
from tauro_inference.client.robot_client import AsyncRobotClient, RobotClient


class MockStub:
    """Mock gRPC stub for testing."""

    def __init__(self):
        self.connected_robots = {}
        self.calibrated_robots = set()

    def Connect(self, request):
        if request.robot_id in self.connected_robots:
            return robot_service_pb2.ConnectResponse(success=False, message="Already connected")

        self.connected_robots[request.robot_id] = request.robot_type

        robot_info = robot_service_pb2.RobotInfo(
            robot_id=request.robot_id,
            robot_type=request.robot_type,
            motor_names=["motor_1", "motor_2"],
        )

        return robot_service_pb2.ConnectResponse(
            success=True, message="Connected", robot_info=robot_info
        )

    def Disconnect(self, request):
        if request.robot_id not in self.connected_robots:
            return robot_service_pb2.DisconnectResponse(success=False, message="Not found")

        del self.connected_robots[request.robot_id]
        return robot_service_pb2.DisconnectResponse(success=True, message="Disconnected")

    def Calibrate(self, request):
        if request.robot_id not in self.connected_robots:
            raise grpc.RpcError()

        self.calibrated_robots.add(request.robot_id)

        calibrations = {
            "motor_1": robot_service_pb2.MotorCalibration(motor_name="motor_1", offset=0.1),
            "motor_2": robot_service_pb2.MotorCalibration(motor_name="motor_2", offset=0.2),
        }

        return robot_service_pb2.CalibrateResponse(
            success=True, message="Calibrated", calibrations=calibrations
        )

    def GetState(self, request):
        if request.robot_id not in self.connected_robots:
            raise grpc.RpcError()

        state = robot_service_pb2.RobotState()
        state.robot_id = request.robot_id
        state.timestamp.GetCurrentTime()
        state.status = robot_service_pb2.ROBOT_STATUS_READY

        state.joints["motor_1"].CopyFrom(
            robot_service_pb2.JointState(
                position=1.0,
                velocity=0.1,
                torque=5.0,
                temperature=20.0,
                is_calibrated=request.robot_id in self.calibrated_robots,
            )
        )

        state.joints["motor_2"].CopyFrom(
            robot_service_pb2.JointState(
                position=2.0,
                velocity=0.2,
                torque=6.0,
                temperature=21.0,
                is_calibrated=request.robot_id in self.calibrated_robots,
            )
        )

        return state

    def SendAction(self, request):
        if request.robot_id not in self.connected_robots:
            return robot_service_pb2.ActionResponse(success=False, message="Robot not found")

        return robot_service_pb2.ActionResponse(success=True, message="Action sent")

    def HealthCheck(self, request):
        health = robot_service_pb2.HealthStatus()
        health.is_healthy = True
        health.timestamp.GetCurrentTime()

        health.components["server"].CopyFrom(
            robot_service_pb2.ComponentHealth(is_healthy=True, status="running", message="OK")
        )

        return health


@pytest.fixture
def mock_channel():
    """Create mock gRPC channel."""
    return MagicMock(spec=grpc.Channel)


@pytest.fixture
def mock_stub():
    """Create mock gRPC stub."""
    return MockStub()


@pytest.fixture
def client(mock_channel, mock_stub):
    """Create test client with mocked connection."""
    client = RobotClient()
    client.channel = mock_channel
    client.stub = mock_stub
    return client


class TestRobotClient:
    def test_connect_to_server(self):
        """Test connecting to server."""
        client = RobotClient()

        with patch("grpc.insecure_channel") as mock_channel:
            with patch.object(client, "health_check") as mock_health:
                mock_health.return_value = MagicMock(is_healthy=True)

                client.connect_to_server()

                assert client.channel is not None
                assert client.stub is not None
                mock_channel.assert_called_once()

    def test_connect_to_server_already_connected(self):
        """Test connecting when already connected."""
        client = RobotClient()
        client.channel = MagicMock()

        with pytest.raises(RuntimeError, match="Already connected"):
            client.connect_to_server()

    def test_disconnect_from_server(self, client):
        """Test disconnecting from server."""
        client.disconnect_from_server()

        client.channel.close.assert_called_once()
        assert client.channel is None
        assert client.stub is None

    def test_connect_robot(self, client):
        """Test connecting to a robot."""
        success = client.connect_robot("test_robot", "so100_follower")

        assert success is True
        assert "test_robot" in client.stub.connected_robots

    def test_connect_robot_failure(self, client):
        """Test failed robot connection."""
        # Connect once
        client.connect_robot("test_robot", "so100_follower")

        # Try to connect again
        success = client.connect_robot("test_robot", "so100_follower")
        assert success is False

    def test_disconnect_robot(self, client):
        """Test disconnecting from a robot."""
        # First connect
        client.connect_robot("test_robot", "so100_follower")

        # Then disconnect
        success = client.disconnect_robot("test_robot")

        assert success is True
        assert "test_robot" not in client.stub.connected_robots

    def test_calibrate_robot(self, client):
        """Test calibrating a robot."""
        # Connect first
        client.connect_robot("test_robot", "so100_follower")

        # Calibrate
        calibrations = client.calibrate_robot("test_robot")

        assert calibrations is not None
        assert len(calibrations) == 2
        assert "motor_1" in calibrations
        assert calibrations["motor_1"].offset == 0.1

    def test_get_robot_state(self, client):
        """Test getting robot state."""
        # Connect first
        client.connect_robot("test_robot", "so100_follower")

        # Get state
        state = client.get_robot_state("test_robot")

        assert state is not None
        assert state.robot_id == "test_robot"
        assert state.status == RobotStatus.READY
        assert len(state.joints) == 2

    def test_send_action(self, client):
        """Test sending action to robot."""
        # Connect first
        client.connect_robot("test_robot", "so100_follower")

        # Send action
        success = client.send_action("test_robot", {"motor_1": 0.5, "motor_2": -0.5})

        assert success is True

    def test_health_check(self, client):
        """Test health check."""
        health = client.health_check()

        assert health.is_healthy is True
        assert "server" in health.components


class TestAsyncRobotClient:
    @pytest.mark.asyncio
    async def test_connect_to_server(self):
        """Test async connecting to server."""
        client = AsyncRobotClient()

        with patch("grpc.aio.insecure_channel") as mock_channel:
            with patch.object(client, "health_check") as mock_health:
                mock_health.return_value = MagicMock(is_healthy=True)

                await client.connect_to_server()

                assert client.channel is not None
                assert client.stub is not None

    @pytest.mark.asyncio
    async def test_stream_control(self):
        """Test streaming control."""
        client = AsyncRobotClient()

        # Mock stub with async streaming
        mock_stub = MagicMock()

        async def mock_stream(command_iter):
            async for cmd in command_iter:
                state = robot_service_pb2.RobotState()
                state.robot_id = cmd.robot_id
                state.timestamp.GetCurrentTime()
                yield state

        mock_stub.StreamControl = mock_stream
        client.stub = mock_stub

        # Create command generator
        async def command_generator():
            for i in range(3):
                yield ControlCommand(
                    timestamp=time.time(),
                    robot_id="test_robot",
                    joint_commands={"motor_1": float(i)},
                    control_mode=ControlMode.POSITION,
                )

        # Stream control
        states = []
        async for state in client.stream_control("test_robot", command_generator()):
            states.append(state)

        assert len(states) == 3
        assert all(s.robot_id == "test_robot" for s in states)
