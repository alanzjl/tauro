from enum import Enum

class ActionSpace(Enum):
    """Command types for robot actions."""
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_TORQUE = "joint_torque"
    END_EFFECTOR_POSITION = "end_effector_position"
    END_EFFECTOR_VELOCITY = "end_effector_velocity"
    END_EFFECTOR_TORQUE = "end_effector_torque"
