from dataclasses import dataclass, field

from ..config import RobotConfig


@dataclass
class SO100FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False

    # Enable end effector control mode
    enable_end_effector_control: bool = False

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )


@dataclass
class SO100FollowerEndEffectorConfig(SO100FollowerConfig):
    """Configuration for the SO100FollowerEndEffector robot."""

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
        }
    )
