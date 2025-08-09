"""Shared robot configuration and utilities for simulated robots."""


import numpy as np

# Robot type to motor/joint mapping
ROBOT_CONFIGS = {
    "so100_follower": {
        "motor_names": [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ],
        "joint_names": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
    },
    "so101_follower": {
        "motor_names": [
            "waist",
            "shoulder",
            "elbow",
            "wrist_angle",
            "wrist_rotate",
            "gripper",
        ],
        "joint_names": ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
    },
}


def get_robot_config(robot_type: str) -> dict:
    """Get robot configuration by type.

    Args:
        robot_type: Type of robot (e.g., "so100_follower")

    Returns:
        Dictionary with motor_names and joint_names

    Raises:
        ValueError: If robot type is not supported
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    return ROBOT_CONFIGS[robot_type].copy()


def normalize_position(raw_pos: float, joint_range: tuple[float, float], motor_name: str) -> float:
    """Normalize position from radians to [-100, 100] or [0, 100].

    Args:
        raw_pos: Raw position in radians
        joint_range: (min, max) joint range in radians
        motor_name: Name of the motor (gripper uses [0, 100], others use [-100, 100])

    Returns:
        Normalized position
    """
    joint_min, joint_max = joint_range

    if motor_name == "gripper":
        # Gripper uses [0, 100] normalization
        if joint_max > joint_min:
            norm_pos = ((raw_pos - joint_min) / (joint_max - joint_min)) * 100
        else:
            norm_pos = 0
        return np.clip(norm_pos, 0, 100)
    else:
        # Other joints use [-100, 100] normalization
        if joint_max > joint_min:
            norm_pos = ((raw_pos - joint_min) / (joint_max - joint_min)) * 200 - 100
        else:
            norm_pos = 0
        return np.clip(norm_pos, -100, 100)


def denormalize_position(
    norm_pos: float, joint_range: tuple[float, float], motor_name: str
) -> float:
    """Denormalize position from [-100, 100] or [0, 100] to radians.

    Args:
        norm_pos: Normalized position
        joint_range: (min, max) joint range in radians
        motor_name: Name of the motor (gripper uses [0, 100], others use [-100, 100])

    Returns:
        Position in radians
    """
    joint_min, joint_max = joint_range

    if motor_name == "gripper":
        # Gripper: [0, 100] -> radians
        raw_pos = (norm_pos / 100) * (joint_max - joint_min) + joint_min
    else:
        # Other joints: [-100, 100] -> radians
        raw_pos = ((norm_pos + 100) / 200) * (joint_max - joint_min) + joint_min

    return np.clip(raw_pos, joint_min, joint_max)


def setup_observation_space(motor_names: list) -> dict:
    """Set up observation space for a robot.

    Args:
        motor_names: List of motor names

    Returns:
        Observation space dictionary
    """
    joint_space = {
        "position": {
            motor: {"shape": (), "dtype": np.dtype(np.float32), "names": None}
            for motor in motor_names
        },
        "velocity": {
            motor: {"shape": (), "dtype": np.dtype(np.float32), "names": None}
            for motor in motor_names
        },
    }

    end_effector_space = {
        "position": {
            "shape": (3,),
            "dtype": np.dtype(np.float32),
            "names": ["x", "y", "z"],
        },
        "orientation": {
            "shape": (9,),
            "dtype": np.dtype(np.float32),
            "names": None,
        },
    }

    return {"joint": joint_space, "end_effector": end_effector_space}


def setup_action_space(motor_names: list) -> dict:
    """Set up action space for a robot.

    Args:
        motor_names: List of motor names

    Returns:
        Action space dictionary
    """
    return {
        f"{motor}.pos": {"shape": (), "dtype": np.dtype(np.float32), "names": None}
        for motor in motor_names
    }
