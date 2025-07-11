"""Utilities for converting between protobuf messages and Python types."""

from typing import Optional

import numpy as np
from google.protobuf import timestamp_pb2
from numpy.typing import NDArray

from tauro_common.proto import robot_service_pb2 as proto
from tauro_common.types.robot_types import (
    ControlMode,
    EndEffectorState,
    FeatureInfo,
    JointState,
    MotorCalibration,
    RobotState,
    RobotStatus,
)


def timestamp_to_proto(timestamp: float) -> timestamp_pb2.Timestamp:
    """Convert float timestamp to protobuf Timestamp."""
    ts = timestamp_pb2.Timestamp()
    ts.FromSeconds(int(timestamp))
    ts.nanos = int((timestamp - int(timestamp)) * 1e9)
    return ts


def proto_to_timestamp(ts: timestamp_pb2.Timestamp) -> float:
    """Convert protobuf Timestamp to float timestamp."""
    return ts.seconds + ts.nanos / 1e9


def control_mode_to_proto(mode: ControlMode) -> proto.ControlMode:
    """Convert Python ControlMode enum to protobuf enum."""
    mapping = {
        ControlMode.POSITION: proto.CONTROL_MODE_POSITION,
        ControlMode.VELOCITY: proto.CONTROL_MODE_VELOCITY,
        ControlMode.TORQUE: proto.CONTROL_MODE_TORQUE,
        ControlMode.IMPEDANCE: proto.CONTROL_MODE_IMPEDANCE,
    }
    return mapping[mode]


def proto_to_control_mode(mode: proto.ControlMode) -> ControlMode:
    """Convert protobuf ControlMode enum to Python enum."""
    mapping = {
        proto.CONTROL_MODE_POSITION: ControlMode.POSITION,
        proto.CONTROL_MODE_VELOCITY: ControlMode.VELOCITY,
        proto.CONTROL_MODE_TORQUE: ControlMode.TORQUE,
        proto.CONTROL_MODE_IMPEDANCE: ControlMode.IMPEDANCE,
    }
    return mapping[mode]


def robot_status_to_proto(status: RobotStatus) -> proto.RobotStatus:
    """Convert Python RobotStatus enum to protobuf enum."""
    mapping = {
        RobotStatus.UNKNOWN: proto.ROBOT_STATUS_UNKNOWN,
        RobotStatus.DISCONNECTED: proto.ROBOT_STATUS_DISCONNECTED,
        RobotStatus.CONNECTED: proto.ROBOT_STATUS_CONNECTED,
        RobotStatus.CALIBRATING: proto.ROBOT_STATUS_CALIBRATING,
        RobotStatus.READY: proto.ROBOT_STATUS_READY,
        RobotStatus.ERROR: proto.ROBOT_STATUS_ERROR,
    }
    return mapping[status]


def proto_to_robot_status(status: proto.RobotStatus) -> RobotStatus:
    """Convert protobuf RobotStatus enum to Python enum."""
    mapping = {
        proto.ROBOT_STATUS_UNKNOWN: RobotStatus.UNKNOWN,
        proto.ROBOT_STATUS_DISCONNECTED: RobotStatus.DISCONNECTED,
        proto.ROBOT_STATUS_CONNECTED: RobotStatus.CONNECTED,
        proto.ROBOT_STATUS_CALIBRATING: RobotStatus.CALIBRATING,
        proto.ROBOT_STATUS_READY: RobotStatus.READY,
        proto.ROBOT_STATUS_ERROR: RobotStatus.ERROR,
    }
    return mapping[status]


def feature_info_to_proto(info: FeatureInfo) -> proto.FeatureInfo:
    """Convert Python FeatureInfo to protobuf."""
    proto_info = proto.FeatureInfo()
    proto_info.shape.extend(info.shape)
    proto_info.dtype = info.dtype
    if info.names:
        proto_info.names.extend(info.names)
    return proto_info


def proto_to_feature_info(proto_info: proto.FeatureInfo) -> FeatureInfo:
    """Convert protobuf FeatureInfo to Python."""
    return FeatureInfo(
        shape=tuple(proto_info.shape),
        dtype=proto_info.dtype,
        names=list(proto_info.names) if proto_info.names else None,
    )


def joint_state_to_proto(state: JointState) -> proto.JointState:
    """Convert Python JointState to protobuf."""
    return proto.JointState(
        position=state.position,
        velocity=state.velocity,
        torque=state.torque,
        temperature=state.temperature,
        is_calibrated=state.is_calibrated,
    )


def proto_to_joint_state(proto_state: proto.JointState) -> JointState:
    """Convert protobuf JointState to Python."""
    return JointState(
        position=proto_state.position,
        velocity=proto_state.velocity,
        torque=proto_state.torque,
        temperature=proto_state.temperature,
        is_calibrated=proto_state.is_calibrated,
    )


def numpy_to_proto_sensor_data(data: NDArray) -> proto.SensorData:
    """Convert numpy array to protobuf SensorData."""
    sensor_data = proto.SensorData()

    if data.dtype in [np.float32, np.float64]:
        float_array = proto.FloatArray()
        float_array.values.extend(data.flatten().astype(np.float32).tolist())
        sensor_data.float_array.CopyFrom(float_array)
    elif data.dtype in [np.int32, np.int64, np.uint8, np.uint16]:
        int_array = proto.IntArray()
        int_array.values.extend(data.flatten().astype(np.int32).tolist())
        sensor_data.int_array.CopyFrom(int_array)
    else:
        # Fallback to raw bytes
        sensor_data.raw_bytes = data.tobytes()

    return sensor_data


def proto_to_numpy_sensor_data(
    proto_data: proto.SensorData, shape: Optional[tuple] = None
) -> NDArray:
    """Convert protobuf SensorData to numpy array."""
    if proto_data.HasField("float_array"):
        data = np.array(proto_data.float_array.values, dtype=np.float32)
    elif proto_data.HasField("int_array"):
        data = np.array(proto_data.int_array.values, dtype=np.int32)
    elif proto_data.HasField("raw_bytes"):
        data = np.frombuffer(proto_data.raw_bytes, dtype=np.uint8)
    else:
        raise ValueError("SensorData has no data field set")

    if shape is not None:
        data = data.reshape(shape)

    return data


def end_effector_state_to_proto(state: EndEffectorState | dict) -> proto.EndEffectorState:
    """Convert Python EndEffectorState to protobuf."""
    proto_ee = proto.EndEffectorState()

    if isinstance(state, dict):
        # Handle dict format
        if "position" in state:
            proto_ee.position.extend(
                state["position"].tolist()
                if hasattr(state["position"], "tolist")
                else state["position"]
            )
        if "orientation" in state:
            proto_ee.orientation.extend(
                state["orientation"].tolist()
                if hasattr(state["orientation"], "tolist")
                else state["orientation"]
            )
    else:
        # Handle EndEffectorState object
        if state.position is not None:
            proto_ee.position.extend(state.position.tolist())
        if state.orientation is not None:
            proto_ee.orientation.extend(state.orientation.tolist())
        if state.linear_velocity is not None:
            proto_ee.linear_velocity.extend(state.linear_velocity.tolist())
        if state.angular_velocity is not None:
            proto_ee.angular_velocity.extend(state.angular_velocity.tolist())
        if state.force is not None:
            proto_ee.force.extend(state.force.tolist())
        if state.torque is not None:
            proto_ee.torque.extend(state.torque.tolist())

    return proto_ee


def proto_to_end_effector_state(proto_ee: proto.EndEffectorState) -> EndEffectorState:
    """Convert protobuf EndEffectorState to Python."""
    return EndEffectorState(
        position=np.array(proto_ee.position) if proto_ee.position else None,
        orientation=np.array(proto_ee.orientation) if proto_ee.orientation else None,
        linear_velocity=np.array(proto_ee.linear_velocity) if proto_ee.linear_velocity else None,
        angular_velocity=np.array(proto_ee.angular_velocity) if proto_ee.angular_velocity else None,
        force=np.array(proto_ee.force) if proto_ee.force else None,
        torque=np.array(proto_ee.torque) if proto_ee.torque else None,
    )


def robot_state_to_proto(state: RobotState) -> proto.RobotState:
    """Convert Python RobotState to protobuf."""
    proto_state = proto.RobotState()
    proto_state.timestamp.CopyFrom(timestamp_to_proto(state.timestamp))
    proto_state.robot_id = state.robot_id
    proto_state.status = robot_status_to_proto(state.status)

    # Convert joints
    for name, joint in state.joints.items():
        proto_state.joints[name].CopyFrom(joint_state_to_proto(joint))

    # Convert sensors
    for name, data in state.sensors.items():
        proto_state.sensors[name].CopyFrom(numpy_to_proto_sensor_data(data))

    # Convert end effector state if present
    if state.end_effector is not None:
        proto_state.end_effector.CopyFrom(end_effector_state_to_proto(state.end_effector))

    return proto_state


def proto_to_robot_state(proto_state: proto.RobotState) -> RobotState:
    """Convert protobuf RobotState to Python."""
    joints = {name: proto_to_joint_state(joint) for name, joint in proto_state.joints.items()}

    sensors = {name: proto_to_numpy_sensor_data(data) for name, data in proto_state.sensors.items()}

    end_effector = None
    if proto_state.HasField("end_effector"):
        end_effector = proto_to_end_effector_state(proto_state.end_effector)

    return RobotState(
        timestamp=proto_to_timestamp(proto_state.timestamp),
        robot_id=proto_state.robot_id,
        joints=joints,
        sensors=sensors,
        status=proto_to_robot_status(proto_state.status),
        end_effector=end_effector,
    )


def motor_calibration_to_proto(calib: MotorCalibration) -> proto.MotorCalibration:
    """Convert Python MotorCalibration to protobuf."""
    return proto.MotorCalibration(
        motor_name=calib.motor_name,
        offset=calib.offset,
        homing_position=calib.homing_position,
        index_position=calib.index_position,
    )


def proto_to_motor_calibration(proto_calib: proto.MotorCalibration) -> MotorCalibration:
    """Convert protobuf MotorCalibration to Python."""
    return MotorCalibration(
        motor_name=proto_calib.motor_name,
        offset=proto_calib.offset,
        homing_position=proto_calib.homing_position,
        index_position=proto_calib.index_position,
    )
