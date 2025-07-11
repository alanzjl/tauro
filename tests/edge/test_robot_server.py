"""Tests for the gRPC robot server."""

from concurrent import futures
from unittest.mock import MagicMock, patch

import grpc
import numpy as np
import pytest
from google.protobuf import empty_pb2

from tauro_common.proto import robot_service_pb2, robot_service_pb2_grpc
from tauro_edge.server.robot_server import RobotControlServicer


class MockRobot:
    """Mock robot for testing."""

    def __init__(self, robot_id: str):
        self.id = robot_id
        self.name = "mock_robot"
        self.motor_configs = {
            "motor_1": {},
            "motor_2": {},
        }
        self.motors = {
            "motor_1": MagicMock(calibration=MagicMock(offset=0.1)),
            "motor_2": MagicMock(calibration=MagicMock(offset=0.2)),
        }
        self.is_calibrated = False
        self._connected = False

        self.observation_space = {
            "observation.state": {
                "shape": (6,),
                "dtype": np.dtype(np.float32),
                "names": ["pos_1", "pos_2", "vel_1", "vel_2", "torq_1", "torq_2"],
            }
        }

        self.action_space = {
            "action": {
                "shape": (2,),
                "dtype": np.dtype(np.float32),
                "names": ["motor_1", "motor_2"],
            }
        }

    def connect(self):
        if self._connected:
            raise RuntimeError("Already connected")
        self._connected = True

    def disconnect(self):
        if not self._connected:
            raise RuntimeError("Not connected")
        self._connected = False

    def calibrate(self):
        self.is_calibrated = True

    def get_observation(self):
        return {"observation.state": np.array([1.0, 2.0, 0.1, 0.2, 5.0, 6.0], dtype=np.float32)}

    def send_action(self, action_dict):
        # Mock action sending
        pass


@pytest.fixture
def server():
    """Create test server."""
    return RobotControlServicer()


@pytest.fixture
def grpc_server():
    """Create gRPC test server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    servicer = RobotControlServicer()
    robot_service_pb2_grpc.add_RobotControlServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")
    stub = robot_service_pb2_grpc.RobotControlServiceStub(channel)

    yield stub, servicer

    server.stop(0)


class TestRobotControlServicer:
    def test_connect_robot(self, server):
        """Test robot connection."""
        with patch("tauro_edge.server.robot_server.SO100Follower", MockRobot):
            request = robot_service_pb2.ConnectRequest(
                robot_id="test_robot", robot_type="so100_follower"
            )

            response = server.Connect(request, None)

            assert response.success is True
            assert "test_robot" in server.robots
            assert response.robot_info.robot_id == "test_robot"
            assert response.robot_info.robot_type == "so100_follower"
            assert len(response.robot_info.motor_names) == 2

    def test_connect_duplicate_robot(self, server):
        """Test connecting same robot twice."""
        # First connection
        server.robots["test_robot"] = MockRobot("test_robot")

        request = robot_service_pb2.ConnectRequest(
            robot_id="test_robot", robot_type="so100_follower"
        )

        response = server.Connect(request, None)
        assert response.success is False
        assert "already connected" in response.message

    def test_disconnect_robot(self, server):
        """Test robot disconnection."""
        # Add robot
        mock_robot = MockRobot("test_robot")
        mock_robot.connect()
        server.robots["test_robot"] = mock_robot

        request = robot_service_pb2.DisconnectRequest(robot_id="test_robot")
        response = server.Disconnect(request, None)

        assert response.success is True
        assert "test_robot" not in server.robots

    def test_disconnect_nonexistent_robot(self, server):
        """Test disconnecting robot that doesn't exist."""
        request = robot_service_pb2.DisconnectRequest(robot_id="nonexistent")
        response = server.Disconnect(request, None)

        assert response.success is False
        assert "not found" in response.message

    def test_calibrate_robot(self, server):
        """Test robot calibration."""
        # Add robot
        mock_robot = MockRobot("test_robot")
        mock_robot.connect()
        server.robots["test_robot"] = mock_robot

        request = robot_service_pb2.CalibrateRequest(robot_id="test_robot")
        response = server.Calibrate(request, None)

        assert response.success is True
        assert mock_robot.is_calibrated is True
        assert len(response.calibrations) == 2

    def test_get_robot_state(self, server):
        """Test getting robot state."""
        # Add robot
        mock_robot = MockRobot("test_robot")
        mock_robot.connect()
        mock_robot.is_calibrated = True
        server.robots["test_robot"] = mock_robot

        request = robot_service_pb2.GetStateRequest(robot_id="test_robot")
        context = MagicMock()

        state = server.GetState(request, context)

        assert state.robot_id == "test_robot"
        assert state.status == robot_service_pb2.ROBOT_STATUS_READY
        assert len(state.joints) == 2
        assert state.joints["motor_1"].position == 1.0
        assert state.joints["motor_2"].position == 2.0

    def test_send_action(self, server):
        """Test sending action to robot."""
        # Add robot
        mock_robot = MockRobot("test_robot")
        mock_robot.connect()
        server.robots["test_robot"] = mock_robot

        request = robot_service_pb2.ActionCommand(
            robot_id="test_robot", actions={"motor_1": 0.5, "motor_2": -0.5}
        )

        response = server.SendAction(request, None)

        assert response.success is True
        mock_robot.send_action.assert_called_once()

    def test_health_check(self, server):
        """Test health check."""
        # Add robot
        mock_robot = MockRobot("test_robot")
        mock_robot.connect()
        server.robots["test_robot"] = mock_robot

        health = server.HealthCheck(empty_pb2.Empty(), None)

        assert health.is_healthy is True
        assert "server" in health.components
        assert "test_robot" in health.components
        assert health.components["server"].is_healthy is True
        assert health.components["test_robot"].is_healthy is True


class TestGRPCIntegration:
    def test_full_workflow(self, grpc_server):
        """Test complete workflow through gRPC."""
        stub, servicer = grpc_server

        with patch("tauro_edge.server.robot_server.SO100Follower", MockRobot):
            # Connect robot
            connect_req = robot_service_pb2.ConnectRequest(
                robot_id="test_robot", robot_type="so100_follower"
            )
            connect_resp = stub.Connect(connect_req)
            assert connect_resp.success is True

            # Calibrate
            cal_req = robot_service_pb2.CalibrateRequest(robot_id="test_robot")
            cal_resp = stub.Calibrate(cal_req)
            assert cal_resp.success is True

            # Get state
            state_req = robot_service_pb2.GetStateRequest(robot_id="test_robot")
            state = stub.GetState(state_req)
            assert state.robot_id == "test_robot"

            # Send action
            action_req = robot_service_pb2.ActionCommand(
                robot_id="test_robot", actions={"motor_1": 0.5}
            )
            action_resp = stub.SendAction(action_req)
            assert action_resp.success is True

            # Health check
            health = stub.HealthCheck(empty_pb2.Empty())
            assert health.is_healthy is True

            # Disconnect
            disc_req = robot_service_pb2.DisconnectRequest(robot_id="test_robot")
            disc_resp = stub.Disconnect(disc_req)
            assert disc_resp.success is True
