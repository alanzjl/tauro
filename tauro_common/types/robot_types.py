"""Common type definitions for robot control."""

from dataclasses import dataclass
from enum import Enum

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
    names: list[str] | None = None


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

    position: NDArray | None = None  # [x, y, z]
    orientation: NDArray | None = None  # Quaternion or rotation matrix
    linear_velocity: NDArray | None = None  # [vx, vy, vz]
    angular_velocity: NDArray | None = None  # [wx, wy, wz]
    force: NDArray | None = None  # [fx, fy, fz]
    torque: NDArray | None = None  # [tx, ty, tz]


@dataclass
class RobotState:
    """Complete robot state."""

    timestamp: float
    robot_id: str
    joints: dict[str, JointState]
    sensors: dict[str, NDArray]
    status: RobotStatus
    end_effector: EndEffectorState | dict | None = None


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
