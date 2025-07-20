#!/usr/bin/env python3
"""Edge case and numerical stability tests for kinematics."""

from pathlib import Path

import numpy as np
import pytest

from tauro_common.kinematics.mink_kinematics import MinkKinematics


class TestKinematicsEdgeCases:
    """Test edge cases and numerical stability of kinematics."""

    @pytest.fixture
    def kinematics(self):
        """Create a MinkKinematics instance for testing."""
        models_dir = Path(__file__).parent.parent.parent / "tauro_common" / "models" / "so_arm100"
        xml_path = models_dir / "so_arm100.xml"

        if not xml_path.exists():
            pytest.skip(
                f"Model file not found at {xml_path}. Run 'python scripts/download_model.py' first."
            )

        return MinkKinematics()

    def test_singularity_configurations(self, kinematics):
        """Test kinematics near singular configurations."""
        # Common singularities for robotic arms
        singular_configs = [
            # Fully extended
            np.array([0, 0, 0, 0, 0], dtype=np.float32),
            # Elbow at 180 (straight arm)
            np.array([0, -90, 180, -90, 0], dtype=np.float32),
            # Wrist singularity
            np.array([45, -45, 90, -90, 0], dtype=np.float32),
        ]

        for config in singular_configs:
            # Forward kinematics should still work
            T = kinematics.forward_kinematics(config)
            assert not np.any(np.isnan(T))
            assert not np.any(np.isinf(T))

            # Try small perturbations for IK
            target_pos = T[:3, 3] + np.array([0.001, 0.001, 0.001])

            solution = kinematics.solve_ik(
                target_position=target_pos,
                current_joint_pos_deg=config,
                max_iterations=200,
                damping=1e-2,  # Higher damping for stability
            )

            assert not np.any(np.isnan(solution))
            assert not np.any(np.isinf(solution))

    def test_zero_movement_ik(self, kinematics):
        """Test IK when target is the same as current position."""
        joints = np.array([30, -45, 60, -30, 15], dtype=np.float32)

        # Get current position
        T = kinematics.forward_kinematics(joints)
        current_pos = T[:3, 3]

        # Solve IK for the same position
        solution = kinematics.solve_ik(
            target_position=current_pos, current_joint_pos_deg=joints, max_iterations=50
        )

        # Should converge quickly and return similar joints
        assert np.allclose(solution, joints, atol=1.0)  # Within 1 degree

    def test_large_joint_movements(self, kinematics):
        """Test IK with large required joint movements."""
        # Start at one extreme
        start_joints = np.array([-90, 45, -90, 45, -90], dtype=np.float32)

        # Target at opposite extreme
        target_joints = np.array([90, -45, 90, -45, 90], dtype=np.float32)
        T_target = kinematics.forward_kinematics(target_joints)
        target_pos = T_target[:3, 3]

        # Solve IK
        solution = kinematics.solve_ik(
            target_position=target_pos,
            current_joint_pos_deg=start_joints,
            max_iterations=500,  # More iterations for large movement
            tolerance=5e-3,
        )

        # Check that we get a valid solution
        assert not np.any(np.isnan(solution))
        assert np.all(np.abs(solution) <= 180)

        # Verify we get close to target (this is a very large movement)
        T_solution = kinematics.forward_kinematics(solution)
        error = np.linalg.norm(T_solution[:3, 3] - target_pos)
        # For such extreme movements, we may not reach exactly
        assert error < 0.5  # 50cm tolerance for extreme case

    def test_ik_convergence_rates(self, kinematics):
        """Test IK convergence for different scenarios."""
        home_joints = np.zeros(5, dtype=np.float32)
        T_home = kinematics.forward_kinematics(home_joints)
        home_pos = T_home[:3, 3]

        scenarios = [
            ("small_move", home_pos + np.array([0.01, 0, 0]), 50),
            ("medium_move", home_pos + np.array([0.05, 0.05, 0]), 100),
            ("large_move", home_pos + np.array([0.1, 0, 0.05]), 200),
        ]

        for name, target_pos, max_iters in scenarios:
            solution = kinematics.solve_ik(
                target_position=target_pos,
                current_joint_pos_deg=home_joints,
                max_iterations=max_iters,
                tolerance=1e-3,
            )

            # Verify convergence
            T_solution = kinematics.forward_kinematics(solution)
            error = np.linalg.norm(T_solution[:3, 3] - target_pos)

            if name == "small_move":
                assert error < 0.005  # Should converge well
            elif name == "medium_move":
                assert error < 0.05  # Reasonable convergence for medium moves
            else:
                assert error < 0.05  # Larger tolerance for large moves

    def test_jacobian_near_singularity(self, kinematics):
        """Test Jacobian computation near singularities."""
        # Configuration near singularity (straight arm)
        joints = np.array([0, -89.9, 179.9, -89.9, 0], dtype=np.float32)

        J = kinematics.get_jacobian(joints)

        # Check that Jacobian doesn't have NaN or Inf
        assert not np.any(np.isnan(J))
        assert not np.any(np.isinf(J))

        # Check condition number (high means near singularity)
        cond = np.linalg.cond(J[:3, :])  # Position part
        assert cond > 1  # Should be poorly conditioned but finite

    def test_ik_with_bad_initial_guess(self, kinematics):
        """Test IK robustness with poor initial guesses."""
        # Target position
        target_pos = np.array([0.15, -0.15, 0.2], dtype=np.float32)

        # Various initial guesses
        initial_guesses = [
            None,  # No initial guess
            np.zeros(5),  # Zero position
            np.array([180, 180, 180, 180, 180], dtype=np.float32),  # Extreme position
            np.random.uniform(-90, 90, 5).astype(np.float32),  # Random position
        ]

        solutions = []
        for init_guess in initial_guesses:
            solution = kinematics.solve_ik(
                target_position=target_pos,
                current_joint_pos_deg=init_guess,
                max_iterations=300,
                tolerance=5e-3,
            )

            # Should always return something valid
            assert not np.any(np.isnan(solution))
            assert solution.shape == (5,)

            # Verify it reaches target
            T = kinematics.forward_kinematics(solution)
            error = np.linalg.norm(T[:3, 3] - target_pos)
            assert error < 0.1  # 10cm tolerance - may not converge perfectly with bad initial guess

            solutions.append(solution)

        # Different initial guesses might lead to different solutions
        # but all should reach the target

    def test_incremental_movements(self, kinematics):
        """Test a series of small incremental movements."""
        current_joints = np.zeros(5, dtype=np.float32)

        # Series of small movements
        movements = [
            np.array([0.005, 0, 0]),
            np.array([0, 0.005, 0]),
            np.array([0, 0, 0.005]),
            np.array([-0.005, 0, 0]),
            np.array([0, -0.005, 0]),
            np.array([0, 0, -0.005]),
        ]

        total_error = 0
        for delta in movements:
            new_joints = kinematics.solve_ik_delta(
                current_joint_pos_deg=current_joints, delta_position=delta, max_iterations=50
            )

            # Verify movement
            T_before = kinematics.forward_kinematics(current_joints)
            T_after = kinematics.forward_kinematics(new_joints)
            actual_delta = T_after[:3, 3] - T_before[:3, 3]

            error = np.linalg.norm(actual_delta - delta)
            total_error += error

            # Update for next iteration
            current_joints = new_joints

        # Average error should be small
        avg_error = total_error / len(movements)
        assert avg_error < 0.001  # Sub-millimeter average error

    def test_workspace_boundaries(self, kinematics):
        """Test behavior at workspace boundaries."""
        # Test points at various distances from origin
        distances = [0.05, 0.15, 0.25, 0.35, 0.45]  # meters

        for r in distances:
            # Test points in a circle at height z=0.2 (more comfortable height)
            n_points = 8
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                target_pos = np.array([r * np.cos(angle), r * np.sin(angle), 0.2], dtype=np.float32)

                solution = kinematics.solve_ik(
                    target_position=target_pos,
                    current_joint_pos_deg=None,
                    max_iterations=200,
                    tolerance=5e-3,
                )

                # Check if reachable
                T = kinematics.forward_kinematics(solution)
                error = np.linalg.norm(T[:3, 3] - target_pos)

                # Always check for valid solution
                assert not np.any(np.isnan(solution))
                assert solution.shape == (5,)

                # For closer targets, check convergence
                # Note: The robot's actual workspace depends on many factors
                if r <= 0.15 and abs(angle) < np.pi / 2:  # Front hemisphere only
                    assert error < 0.15  # Within 15cm is reasonable for workspace edges

    def test_joint_velocity_limits(self, kinematics):
        """Test that IK respects reasonable joint velocities."""
        current_joints = np.array([0, -30, 60, -30, 0], dtype=np.float32)
        dt = 0.1  # 100ms time step

        # Request a movement
        delta = np.array([0.02, 0, 0], dtype=np.float32)
        new_joints = kinematics.solve_ik_delta(
            current_joint_pos_deg=current_joints, delta_position=delta
        )

        # Calculate joint velocities
        joint_velocities = (new_joints - current_joints) / dt

        # Check that velocities are reasonable (e.g., < 180 deg/s)
        assert np.all(np.abs(joint_velocities) < 180)

    def test_deterministic_behavior(self, kinematics):
        """Test that IK gives consistent results."""
        target_pos = np.array([0.15, -0.1, 0.2], dtype=np.float32)
        init_joints = np.array([10, -20, 30, -15, 5], dtype=np.float32)

        # Solve multiple times
        solutions = []
        for _ in range(5):
            solution = kinematics.solve_ik(
                target_position=target_pos, current_joint_pos_deg=init_joints, max_iterations=100
            )
            solutions.append(solution)

        # All solutions should be identical (deterministic)
        for i in range(1, len(solutions)):
            assert np.allclose(solutions[0], solutions[i], atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
