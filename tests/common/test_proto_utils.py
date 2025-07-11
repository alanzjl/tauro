"""Tests for protobuf conversion utilities."""

import numpy as np

from tauro_common.proto import robot_service_pb2 as proto
from tauro_common.types.robot_types import (
    ControlMode,
    FeatureInfo,
    JointState,
    MotorCalibration,
    RobotState,
    RobotStatus,
)
from tauro_common.utils.proto_utils import (
    control_mode_to_proto,
    feature_info_to_proto,
    joint_state_to_proto,
    motor_calibration_to_proto,
    numpy_to_proto_sensor_data,
    proto_to_control_mode,
    proto_to_feature_info,
    proto_to_joint_state,
    proto_to_motor_calibration,
    proto_to_numpy_sensor_data,
    proto_to_robot_state,
    proto_to_robot_status,
    proto_to_timestamp,
    robot_state_to_proto,
    robot_status_to_proto,
    timestamp_to_proto,
)


class TestTimestampConversion:
    def test_timestamp_to_proto(self):
        timestamp = 1234567890.123456
        proto_ts = timestamp_to_proto(timestamp)
        assert proto_ts.seconds == 1234567890
        assert abs(proto_ts.nanos - 123456000) < 1000  # Allow small floating point error

    def test_proto_to_timestamp(self):
        proto_ts = proto.google.protobuf.timestamp_pb2.Timestamp()
        proto_ts.seconds = 1234567890
        proto_ts.nanos = 123456000
        timestamp = proto_to_timestamp(proto_ts)
        assert abs(timestamp - 1234567890.123456) < 1e-6


class TestEnumConversions:
    def test_control_mode_conversions(self):
        # Test all control modes
        modes = [
            (ControlMode.POSITION, proto.CONTROL_MODE_POSITION),
            (ControlMode.VELOCITY, proto.CONTROL_MODE_VELOCITY),
            (ControlMode.TORQUE, proto.CONTROL_MODE_TORQUE),
            (ControlMode.IMPEDANCE, proto.CONTROL_MODE_IMPEDANCE),
        ]

        for py_mode, proto_mode in modes:
            assert control_mode_to_proto(py_mode) == proto_mode
            assert proto_to_control_mode(proto_mode) == py_mode

    def test_robot_status_conversions(self):
        # Test all robot statuses
        statuses = [
            (RobotStatus.UNKNOWN, proto.ROBOT_STATUS_UNKNOWN),
            (RobotStatus.DISCONNECTED, proto.ROBOT_STATUS_DISCONNECTED),
            (RobotStatus.CONNECTED, proto.ROBOT_STATUS_CONNECTED),
            (RobotStatus.CALIBRATING, proto.ROBOT_STATUS_CALIBRATING),
            (RobotStatus.READY, proto.ROBOT_STATUS_READY),
            (RobotStatus.ERROR, proto.ROBOT_STATUS_ERROR),
        ]

        for py_status, proto_status in statuses:
            assert robot_status_to_proto(py_status) == proto_status
            assert proto_to_robot_status(proto_status) == py_status


class TestFeatureInfoConversion:
    def test_feature_info_to_proto(self):
        info = FeatureInfo(shape=(3, 4, 5), dtype="float32", names=["x", "y", "z"])
        proto_info = feature_info_to_proto(info)

        assert list(proto_info.shape) == [3, 4, 5]
        assert proto_info.dtype == "float32"
        assert list(proto_info.names) == ["x", "y", "z"]

    def test_proto_to_feature_info(self):
        proto_info = proto.FeatureInfo()
        proto_info.shape.extend([3, 4, 5])
        proto_info.dtype = "float32"
        proto_info.names.extend(["x", "y", "z"])

        info = proto_to_feature_info(proto_info)
        assert info.shape == (3, 4, 5)
        assert info.dtype == "float32"
        assert info.names == ["x", "y", "z"]


class TestJointStateConversion:
    def test_joint_state_to_proto(self):
        state = JointState(
            position=1.5, velocity=0.5, torque=10.0, temperature=25.5, is_calibrated=True
        )
        proto_state = joint_state_to_proto(state)

        assert proto_state.position == 1.5
        assert proto_state.velocity == 0.5
        assert proto_state.torque == 10.0
        assert proto_state.temperature == 25.5
        assert proto_state.is_calibrated is True

    def test_proto_to_joint_state(self):
        proto_state = proto.JointState(
            position=1.5, velocity=0.5, torque=10.0, temperature=25.5, is_calibrated=True
        )
        state = proto_to_joint_state(proto_state)

        assert state.position == 1.5
        assert state.velocity == 0.5
        assert state.torque == 10.0
        assert state.temperature == 25.5
        assert state.is_calibrated is True


class TestSensorDataConversion:
    def test_float_array_conversion(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        proto_data = numpy_to_proto_sensor_data(data)

        assert proto_data.HasField("float_array")
        assert list(proto_data.float_array.values) == [1.0, 2.0, 3.0]

        # Convert back
        recovered = proto_to_numpy_sensor_data(proto_data)
        np.testing.assert_array_equal(recovered, data)

    def test_int_array_conversion(self):
        data = np.array([1, 2, 3], dtype=np.int32)
        proto_data = numpy_to_proto_sensor_data(data)

        assert proto_data.HasField("int_array")
        assert list(proto_data.int_array.values) == [1, 2, 3]

        # Convert back
        recovered = proto_to_numpy_sensor_data(proto_data)
        np.testing.assert_array_equal(recovered, data)

    def test_raw_bytes_conversion(self):
        data = np.array([1, 2, 3], dtype=np.complex64)  # Unsupported type
        proto_data = numpy_to_proto_sensor_data(data)

        assert proto_data.HasField("raw_bytes")

        # Convert back (will be uint8)
        recovered = proto_to_numpy_sensor_data(proto_data)
        assert recovered.dtype == np.uint8

    def test_shaped_array_recovery(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        proto_data = numpy_to_proto_sensor_data(data)

        # Recover with shape
        recovered = proto_to_numpy_sensor_data(proto_data, shape=(2, 2))
        np.testing.assert_array_equal(recovered, data)
        assert recovered.shape == (2, 2)


class TestMotorCalibrationConversion:
    def test_motor_calibration_to_proto(self):
        calib = MotorCalibration(
            motor_name="motor_1", offset=0.5, homing_position=1.0, index_position=0.0
        )
        proto_calib = motor_calibration_to_proto(calib)

        assert proto_calib.motor_name == "motor_1"
        assert proto_calib.offset == 0.5
        assert proto_calib.homing_position == 1.0
        assert proto_calib.index_position == 0.0

    def test_proto_to_motor_calibration(self):
        proto_calib = proto.MotorCalibration(
            motor_name="motor_1", offset=0.5, homing_position=1.0, index_position=0.0
        )
        calib = proto_to_motor_calibration(proto_calib)

        assert calib.motor_name == "motor_1"
        assert calib.offset == 0.5
        assert calib.homing_position == 1.0
        assert calib.index_position == 0.0


class TestRobotStateConversion:
    def test_robot_state_full_conversion(self):
        # Create a complex robot state
        joints = {
            "joint_1": JointState(
                position=1.0, velocity=0.1, torque=5.0, temperature=20.0, is_calibrated=True
            ),
            "joint_2": JointState(
                position=2.0, velocity=0.2, torque=6.0, temperature=21.0, is_calibrated=False
            ),
        }

        sensors = {
            "camera": np.array([[1, 2], [3, 4]], dtype=np.uint8),
            "force": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        }

        state = RobotState(
            timestamp=1234567890.123,
            robot_id="test_robot",
            joints=joints,
            sensors=sensors,
            status=RobotStatus.READY,
        )

        # Convert to proto
        proto_state = robot_state_to_proto(state)

        # Verify proto fields
        assert abs(proto_to_timestamp(proto_state.timestamp) - 1234567890.123) < 1e-6
        assert proto_state.robot_id == "test_robot"
        assert proto_state.status == proto.ROBOT_STATUS_READY
        assert len(proto_state.joints) == 2
        assert len(proto_state.sensors) == 2

        # Convert back
        recovered = proto_to_robot_state(proto_state)

        assert abs(recovered.timestamp - state.timestamp) < 1e-6
        assert recovered.robot_id == state.robot_id
        assert recovered.status == state.status
        assert len(recovered.joints) == len(state.joints)
        assert len(recovered.sensors) == len(state.sensors)

        # Check joints
        for name in state.joints:
            assert name in recovered.joints
            orig_joint = state.joints[name]
            rec_joint = recovered.joints[name]
            assert rec_joint.position == orig_joint.position
            assert rec_joint.velocity == orig_joint.velocity
            assert rec_joint.torque == orig_joint.torque
            assert rec_joint.temperature == orig_joint.temperature
            assert rec_joint.is_calibrated == orig_joint.is_calibrated
