#!/usr/bin/env python3
"""Unit tests for MinkKinematics class."""

from pathlib import Path

import numpy as np
import pytest

from tauro_common.kinematics.mink_kinematics import MinkKinematics


class TestMinkKinematics:
    """Test suite for MinkKinematics."""

    @pytest.fixture
    def kinematics(self):
        """Create a MinkKinematics instance for testing."""
        # Check if model exists, skip if not
        models_dir = Path(__file__).parent.parent.parent / "tauro_common" / "models" / "so_arm100"
        xml_path = models_dir / "so_arm100.xml"

        if not xml_path.exists():
            pytest.skip(
                f"Model file not found at {xml_path}. Run 'python scripts/download_model.py' first."
            )

        return MinkKinematics()

    def test_initialization(self, kinematics):
        """Test that MinkKinematics initializes correctly."""
        assert kinematics is not None
        assert kinematics.model is not None
        assert kinematics.data is not None
        assert kinematics.configuration is not None

        # Check joint names
        assert len(kinematics.joint_names) == 6
        assert kinematics.joint_names[0] == "Rotation"
        assert kinematics.joint_names[-1] == "Jaw"

        # Check motor names
        assert len(kinematics.motor_names) == 6

        # Check frame names
        assert kinematics.ee_frame == "Wrist_Pitch_Roll"
        assert kinematics.gripper_frame == "Moving_Jaw"

    def test_forward_kinematics_zero_position(self, kinematics):
        """Test forward kinematics at zero position."""
        joint_pos = np.zeros(5, dtype=np.float32)
        T = kinematics.forward_kinematics(joint_pos)

        # Check that we get a 4x4 homogeneous transformation matrix
        assert T.shape == (4, 4)
        assert np.allclose(T[3, :], [0, 0, 0, 1])

        # Check that rotation part is orthonormal
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)

        # Store the zero position for reference
        zero_pos = T[:3, 3]
        assert zero_pos.shape == (3,)

    def test_forward_kinematics_various_positions(self, kinematics):
        """Test forward kinematics at various joint configurations."""
        test_configs = [
            np.array([0, 0, 0, 0, 0], dtype=np.float32),
            np.array([30, -45, 60, -30, 0], dtype=np.float32),
            np.array([45, 0, -45, 0, 90], dtype=np.float32),
            np.array([-30, 30, -60, 45, -45], dtype=np.float32),
            np.array([90, -90, 90, -90, 180], dtype=np.float32),
        ]

        previous_pos = None
        for joints in test_configs:
            T = kinematics.forward_kinematics(joints)

            # Basic checks
            assert T.shape == (4, 4)
            assert np.allclose(T[3, :], [0, 0, 0, 1])

            # Check rotation matrix properties
            R = T[:3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)

            # Check that position changes with different configurations
            current_pos = T[:3, 3]
            if previous_pos is not None:
                assert not np.allclose(current_pos, previous_pos)
            previous_pos = current_pos

    def test_forward_kinematics_with_gripper(self, kinematics):
        """Test forward kinematics with 6 DOF (including gripper)."""
        # Test with 6 joints (including gripper)
        joint_pos = np.array([0, 0, 0, 0, 0, 30], dtype=np.float32)
        T = kinematics.forward_kinematics(joint_pos)

        assert T.shape == (4, 4)
        # Gripper value shouldn't affect end-effector frame position
        T_no_gripper = kinematics.forward_kinematics(joint_pos[:5])
        assert np.allclose(T, T_no_gripper)

    def test_inverse_kinematics_identity(self, kinematics):
        """Test that IK can recover the same joint configuration."""
        # Start with a known joint configuration
        initial_joints = np.array([10, -20, 30, -15, 5], dtype=np.float32)

        # Get the forward kinematics
        T = kinematics.forward_kinematics(initial_joints)
        target_pos = T[:3, 3]

        # Solve inverse kinematics
        solution = kinematics.solve_ik(
            target_position=target_pos,
            current_joint_pos_deg=initial_joints,
            max_iterations=200,
            tolerance=1e-3,
        )

        # Check that we get a valid solution
        assert solution.shape == (5,)
        assert not np.any(np.isnan(solution))

        # Verify the solution reaches the target
        T_solution = kinematics.forward_kinematics(solution)
        pos_error = np.linalg.norm(T_solution[:3, 3] - target_pos)
        assert pos_error < 0.005  # 5mm tolerance

    def test_inverse_kinematics_reachable_targets(self, kinematics):
        """Test IK for various reachable target positions."""
        # Start from home position
        home_joints = np.zeros(5, dtype=np.float32)
        T_home = kinematics.forward_kinematics(home_joints)
        home_pos = T_home[:3, 3]

        # Test small movements from home
        test_deltas = [
            np.array([0.02, 0, 0]),  # 2cm forward
            np.array([0, 0.02, 0]),  # 2cm right
            np.array([0, 0, 0.02]),  # 2cm up
            np.array([0.01, 0.01, 0.01]),  # diagonal
        ]

        for delta in test_deltas:
            target_pos = home_pos + delta

            solution = kinematics.solve_ik(
                target_position=target_pos,
                current_joint_pos_deg=home_joints,
                max_iterations=200,
                tolerance=1e-3,
            )

            # Verify solution
            T_solution = kinematics.forward_kinematics(solution)
            actual_pos = T_solution[:3, 3]
            error = np.linalg.norm(actual_pos - target_pos)

            assert error < 0.02  # 20mm tolerance for this test
            assert not np.any(np.isnan(solution))
            assert np.all(np.abs(solution) <= 180)  # Joint limits

    def test_inverse_kinematics_unreachable_targets(self, kinematics):
        """Test IK behavior for unreachable targets."""
        # Test very far targets that are likely unreachable
        unreachable_targets = [
            np.array([1.0, 0, 0.1]),  # 1 meter forward
            np.array([0, 1.0, 0.1]),  # 1 meter to the side
            np.array([0, 0, 2.0]),  # 2 meters up
        ]

        home_joints = np.zeros(5, dtype=np.float32)

        for target_pos in unreachable_targets:
            solution = kinematics.solve_ik(
                target_position=target_pos,
                current_joint_pos_deg=home_joints,
                max_iterations=100,
                tolerance=5e-3,
            )

            # Should still return a solution (best effort)
            assert solution.shape == (5,)
            assert not np.any(np.isnan(solution))

            # But likely won't reach the exact target
            T_solution = kinematics.forward_kinematics(solution)
            actual_pos = T_solution[:3, 3]
            error = np.linalg.norm(actual_pos - target_pos)

            # For unreachable targets, we expect larger errors
            assert error > 0.01  # More than 1cm error expected

    def test_solve_ik_delta(self, kinematics):
        """Test the solve_ik_delta method."""
        current_joints = np.array([0, -15, 30, -15, 0], dtype=np.float32)
        delta_pos = np.array([0.01, 0.01, 0], dtype=np.float32)

        # Get current position
        T_current = kinematics.forward_kinematics(current_joints)
        current_pos = T_current[:3, 3]

        # Solve for delta movement
        new_joints = kinematics.solve_ik_delta(
            current_joint_pos_deg=current_joints, delta_position=delta_pos
        )

        # Verify the movement
        T_new = kinematics.forward_kinematics(new_joints)
        new_pos = T_new[:3, 3]
        actual_delta = new_pos - current_pos

        # Check that we moved approximately in the right direction
        assert np.linalg.norm(actual_delta - delta_pos) < 0.005

    def test_jacobian(self, kinematics):
        """Test Jacobian computation."""
        joint_pos = np.array([10, -20, 30, -15, 5], dtype=np.float32)

        J = kinematics.get_jacobian(joint_pos)

        # Check shape
        assert J.shape == (6, 5)  # 6 DOF output (position + orientation), 5 joints

        # Check that Jacobian is not zero
        assert not np.allclose(J, 0)

        # Verify Jacobian numerically using finite differences
        eps = 0.1  # degrees
        T0 = kinematics.forward_kinematics(joint_pos)

        for i in range(5):
            # Perturb joint i
            joint_pos_pert = joint_pos.copy()
            joint_pos_pert[i] += eps

            # Compute numerical derivative
            T_pert = kinematics.forward_kinematics(joint_pos_pert)

            # Position difference (convert eps to radians for derivative)
            dp = (T_pert[:3, 3] - T0[:3, 3]) / np.deg2rad(eps)

            # Compare with Jacobian columns (position part)
            J_pos_numerical = dp
            J_pos_analytical = J[:3, i]

            # Allow some numerical error
            assert np.allclose(J_pos_analytical, J_pos_numerical, atol=1e-2)

    def test_joint_limit_handling(self, kinematics):
        """Test that IK respects joint limits."""
        # Test with extreme initial position
        extreme_joints = np.array([170, -170, 170, -170, 170], dtype=np.float32)

        # Try to move further
        T = kinematics.forward_kinematics(extreme_joints)
        target_pos = T[:3, 3] + np.array([0.05, 0, 0])

        solution = kinematics.solve_ik(
            target_position=target_pos, current_joint_pos_deg=extreme_joints, max_iterations=100
        )

        # Check that solution respects limits
        assert np.all(np.abs(solution) <= 180)

    def test_ik_with_orientation(self, kinematics):
        """Test IK with orientation constraints."""
        # Get a reference pose
        ref_joints = np.array([20, -30, 45, -20, 10], dtype=np.float32)
        T_ref = kinematics.forward_kinematics(ref_joints)

        # Move position but try to keep orientation
        target_pos = T_ref[:3, 3] + np.array([0.02, 0.01, 0])
        target_orient = T_ref[:3, :3]

        solution = kinematics.solve_ik(
            target_position=target_pos,
            target_orientation=target_orient,
            current_joint_pos_deg=ref_joints,
            position_weight=1.0,
            orientation_weight=1.0,
            max_iterations=200,
        )

        # Check result
        T_solution = kinematics.forward_kinematics(solution)
        pos_error = np.linalg.norm(T_solution[:3, 3] - target_pos)

        # Orientation error (using Frobenius norm)
        orient_error = np.linalg.norm(T_solution[:3, :3] - target_orient, "fro")

        # With orientation constraint, we might not reach exact position
        assert pos_error < 0.03  # 3cm tolerance when orientation is constrained
        assert orient_error < 0.5  # Some orientation error is expected

    def test_configuration_update(self, kinematics):
        """Test that configuration is properly updated during IK solving."""
        # This tests the internal state management
        joints1 = np.array([10, -10, 20, -10, 0], dtype=np.float32)
        joints2 = np.array([20, -20, 40, -20, 0], dtype=np.float32)

        # Compute FK for both
        T1 = kinematics.forward_kinematics(joints1)
        T2 = kinematics.forward_kinematics(joints2)

        # Results should be different
        assert not np.allclose(T1[:3, 3], T2[:3, 3])

        # Computing T1 again should give same result
        T1_again = kinematics.forward_kinematics(joints1)
        assert np.allclose(T1, T1_again)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
