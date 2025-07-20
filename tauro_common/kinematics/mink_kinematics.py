#!/usr/bin/env python3
"""Mink-based kinematics for SO-ARM100 robot."""

import logging
import os
from pathlib import Path

import mujoco
import numpy as np
from mink import SE3, Configuration, FrameTask, solve_ik
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MinkKinematics:
    """Inverse kinematics solver using mink for SO-ARM100."""

    def __init__(self, xml_path: str | None = None):
        """Initialize the IK solver with the robot model.

        Args:
            xml_path: Path to the MuJoCo XML file. If None, uses the default SO-ARM100 model.
        """
        if xml_path is None:
            # Use the default path relative to this file
            models_dir = Path(__file__).parent.parent / "models" / "so_arm100"
            xml_path = str(models_dir / "so_arm100.xml")

        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"MuJoCo XML file not found at {xml_path}. "
                "Please download it using: python scripts/download_model.py"
            )

        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Create mink configuration
        self.configuration = Configuration(self.model)

        # Joint names mapping (from MuJoCo model)
        # Get actuator names from the model
        self.joint_names = []
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                self.joint_names.append(name)

        # Our joint names mapping to actuators
        self.motor_names = [
            "Rotation",  # shoulder_pan
            "Pitch",  # shoulder_lift
            "Elbow",  # elbow_flex
            "Wrist_Pitch",  # wrist_flex
            "Wrist_Roll",  # wrist_roll
            "Jaw",  # gripper
        ]

        # Frame names - we'll use body names from MuJoCo
        self.ee_frame = "Wrist_Pitch_Roll"  # End effector frame based on MuJoCo model
        self.gripper_frame = "Moving_Jaw"  # Gripper frame

        logger.info(f"Initialized MinkKinematics with model from {xml_path}")
        logger.info(f"Model has {self.model.nq} DOF and {self.model.nu} actuators")
        logger.info(f"Joint names: {self.joint_names}")

    def forward_kinematics(
        self, joint_positions_deg: NDArray[np.float32], frame: str = "Wrist_Pitch_Roll"
    ) -> NDArray[np.float32]:
        """Compute forward kinematics.

        Args:
            joint_positions_deg: Joint positions in degrees [5 or 7 values]
            frame: Target frame name

        Returns:
            4x4 homogeneous transformation matrix
        """
        # Convert to radians
        q_rad = np.deg2rad(joint_positions_deg)

        # Handle different input sizes
        if len(q_rad) == 5:
            # Add gripper joint (default closed)
            gripper_pos = 0.0  # Default closed
            q_rad = np.concatenate([q_rad, [gripper_pos]])
        elif len(q_rad) > 6:
            # Take only first 6 joints
            q_rad = q_rad[:6]

        # Update configuration and MuJoCo data
        self.configuration = Configuration(self.model, q_rad)
        self.data.qpos[:] = q_rad

        # Update MuJoCo forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Get body ID for the frame
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame)
        if body_id == -1:
            raise ValueError(f"Body '{frame}' not found in model")

        # Get transformation matrix
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[body_id]

        return T.astype(np.float32)

    def solve_ik(
        self,
        target_position: NDArray[np.float32],
        target_orientation: NDArray[np.float32] | None = None,
        current_joint_pos_deg: NDArray[np.float32] | None = None,
        frame: str = "Wrist_Pitch_Roll",
        position_weight: float = 1.0,
        orientation_weight: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-3,
        damping: float = 1e-3,
    ) -> NDArray[np.float32]:
        """Solve inverse kinematics using mink.

        Args:
            target_position: Desired 3D position
            target_orientation: Desired orientation as 3x3 rotation matrix (optional)
            current_joint_pos_deg: Initial joint positions in degrees [5 values]
            frame: Target frame name
            position_weight: Weight for position task
            orientation_weight: Weight for orientation task (0 = position only)
            max_iterations: Maximum iterations for solver
            tolerance: Convergence tolerance
            damping: Damping factor for numerical stability

        Returns:
            Joint positions in degrees [5 values] achieving the target pose
        """
        # Initialize configuration
        if current_joint_pos_deg is not None:
            q_rad = np.deg2rad(current_joint_pos_deg)
            if len(q_rad) == 5:
                # Add gripper joint
                q_rad = np.concatenate([q_rad, [0.0]])
            elif len(q_rad) > 6:
                q_rad = q_rad[:6]
            self.configuration = Configuration(self.model, q_rad)
            self.data.qpos[:] = q_rad
        else:
            # Use zero configuration
            self.configuration = Configuration(self.model, np.zeros(self.model.nq))

        # Get current orientation if not provided
        if target_orientation is None:
            # Keep current orientation
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame)
            if body_id == -1:
                raise ValueError(f"Body '{frame}' not found in model")
            mujoco.mj_forward(self.model, self.data)
            target_orientation = self.data.xmat[body_id].reshape(3, 3)

        # Create target transform using quaternion and position
        # Convert rotation matrix to quaternion
        from scipy.spatial.transform import Rotation

        quat = Rotation.from_matrix(target_orientation).as_quat()  # returns [x, y, z, w]
        # SE3 expects [w, x, y, z, x, y, z]
        wxyz_xyz = np.concatenate([[quat[3]], quat[:3], target_position])
        target_transform = SE3(wxyz_xyz)

        # Create task for body frame
        task = FrameTask(
            frame_name=frame,
            frame_type="body",
            position_cost=position_weight,
            orientation_cost=orientation_weight,
        )

        # Set target
        task.set_target(target_transform)

        # Solve IK iteratively
        dt = 0.1  # Integration time step (larger for faster convergence)
        for i in range(max_iterations):
            # Check convergence
            error = task.compute_error(self.configuration)
            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                logger.debug(f"IK converged in {i} iterations with error {error_norm}")
                break

            # Solve for joint velocities
            dq = solve_ik(
                configuration=self.configuration,
                tasks=[task],
                dt=dt,
                solver="quadprog",  # Use quadprog solver
                damping=damping,
            )

            # Update configuration by integrating velocity
            new_q = self.configuration.integrate(dq, dt)

            # Update MuJoCo data manually since Configuration doesn't expose q setter
            self.data.qpos[:] = new_q
            mujoco.mj_forward(self.model, self.data)

            # Recreate configuration with new q
            self.configuration = Configuration(self.model, new_q)

        # Extract joint positions (first 5 joints only)
        joint_pos_rad = self.configuration.q[:5]
        joint_pos_deg = np.rad2deg(joint_pos_rad)

        return joint_pos_deg.astype(np.float32)

    def solve_ik_delta(
        self,
        current_joint_pos_deg: NDArray[np.float32],
        delta_position: NDArray[np.float32],
        frame: str = "Wrist_Pitch_Roll",
        **kwargs,
    ) -> NDArray[np.float32]:
        """Solve IK for a delta movement from current position.

        Args:
            current_joint_pos_deg: Current joint positions in degrees [5 values]
            delta_position: Desired position change [dx, dy, dz]
            frame: Target frame name
            **kwargs: Additional arguments passed to solve_ik

        Returns:
            Joint positions in degrees [5 values] achieving the target
        """
        # Get current end effector position
        current_transform = self.forward_kinematics(current_joint_pos_deg, frame)
        current_position = current_transform[:3, 3]

        # Calculate target position
        target_position = current_position + delta_position

        # Solve IK
        return self.solve_ik(
            target_position=target_position,
            current_joint_pos_deg=current_joint_pos_deg,
            frame=frame,
            **kwargs,
        )

    def get_jacobian(
        self, joint_positions_deg: NDArray[np.float32], frame: str = "Wrist_Pitch_Roll"
    ) -> NDArray[np.float32]:
        """Compute the Jacobian matrix.

        Args:
            joint_positions_deg: Joint positions in degrees [5 values]
            frame: Target frame name

        Returns:
            6x5 Jacobian matrix (spatial velocity)
        """
        # Convert to radians
        q_rad = np.deg2rad(joint_positions_deg)
        if len(q_rad) == 5:
            q_rad = np.concatenate([q_rad, [0.0]])
        elif len(q_rad) > 6:
            q_rad = q_rad[:6]

        # Update configuration and MuJoCo data
        self.configuration = Configuration(self.model, q_rad)
        self.data.qpos[:] = q_rad
        mujoco.mj_forward(self.model, self.data)

        # Get body Jacobian
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame)
        if body_id == -1:
            raise ValueError(f"Body '{frame}' not found in model")

        # Compute Jacobian using MuJoCo
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian

        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)

        # Stack to get full 6x7 Jacobian
        J = np.vstack([jacp, jacr])

        # Return only the first 5 columns (arm joints, excluding gripper)
        return J[:, :5].astype(np.float32)
