"""Common type definitions for robot control."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

from numpy.typing import NDArray


class ControlMode(Enum):
    """Robot control modes."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"


class RobotStatus(Enum):
    """Robot connection and operational status."""

    UNKNOWN = "unknown"
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    CALIBRATING = "calibrating"
    READY = "ready"
    ERROR = "error"


@dataclass
class FeatureInfo:
    """Information about observation or action features."""

    shape: tuple[int, ...]
    dtype: str
    names: Optional[list[str]] = None


@dataclass
class JointState:
    """State of a single joint."""

    position: float
    velocity: float
    torque: float
    temperature: float
    is_calibrated: bool = False


@dataclass
class EndEffectorState:
    """End effector state in Cartesian space."""

    position: Optional[NDArray] = None  # [x, y, z]
    orientation: Optional[NDArray] = None  # Quaternion or rotation matrix
    linear_velocity: Optional[NDArray] = None  # [vx, vy, vz]
    angular_velocity: Optional[NDArray] = None  # [wx, wy, wz]
    force: Optional[NDArray] = None  # [fx, fy, fz]
    torque: Optional[NDArray] = None  # [tx, ty, tz]


@dataclass
class RobotState:
    """Complete robot state."""

    timestamp: float
    robot_id: str
    joints: dict[str, JointState]
    sensors: dict[str, NDArray]
    status: RobotStatus
    end_effector: Optional[Union[EndEffectorState, dict]] = None


@dataclass
class ControlCommand:
    """Command to control robot joints."""

    timestamp: float
    robot_id: str
    joint_commands: dict[str, float]
    control_mode: ControlMode = ControlMode.POSITION


@dataclass
class MotorCalibration:
    """Motor calibration data."""

    motor_name: str
    offset: float
    homing_position: float = 0.0
    index_position: float = 0.0
