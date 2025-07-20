#!/usr/bin/env python3
"""Standalone test for mink-based inverse kinematics (no robot server required)."""

import numpy as np

from tauro_common.kinematics.mink_kinematics import MinkKinematics


def test_forward_kinematics():
    """Test forward kinematics with various joint configurations."""
    print("\n=== Testing Forward Kinematics ===")
    kin = MinkKinematics()

    test_configs = [
        ([0, 0, 0, 0, 0], "Home position"),
        ([30, -45, 60, -30, 0], "Typical reaching pose"),
        ([45, 0, -45, 0, 90], "Extended with wrist rotation"),
        ([-30, 30, -60, 45, -45], "Negative joint angles"),
    ]

    for joints, description in test_configs:
        joint_pos = np.array(joints, dtype=np.float32)
        T = kin.forward_kinematics(joint_pos)
        pos = T[:3, 3]
        print(f"\n{description}:")
        print(f"  Joints (deg): {joints}")
        print(f"  Position (m): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")


def test_inverse_kinematics():
    """Test inverse kinematics for various targets."""
    print("\n=== Testing Inverse Kinematics ===")
    kin = MinkKinematics()

    # Start from home position
    home_joints = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    T_home = kin.forward_kinematics(home_joints)
    home_pos = T_home[:3, 3]

    print(f"\nHome position: [{home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f}]")

    # Test various target positions
    test_targets = [
        (home_pos + np.array([0.02, 0, 0]), "Move +2cm in X"),
        (home_pos + np.array([0, 0.02, 0]), "Move +2cm in Y"),
        (home_pos + np.array([0, 0, 0.02]), "Move +2cm in Z"),
        (home_pos + np.array([0.01, 0.01, 0.01]), "Move diagonally"),
    ]

    for target_pos, description in test_targets:
        print(f"\n{description}:")
        print(f"  Target: [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")

        # Solve IK
        new_joints = kin.solve_ik(
            target_pos, current_joint_pos_deg=home_joints, max_iterations=100, tolerance=1e-3
        )

        # Verify solution
        T_new = kin.forward_kinematics(new_joints)
        actual_pos = T_new[:3, 3]
        error = np.linalg.norm(actual_pos - target_pos)

        print(
            f"  Solution joints (deg): [{new_joints[0]:.2f}, {new_joints[1]:.2f}, {new_joints[2]:.2f}, {new_joints[3]:.2f}, {new_joints[4]:.2f}]"
        )
        print(f"  Actual position: [{actual_pos[0]:.4f}, {actual_pos[1]:.4f}, {actual_pos[2]:.4f}]")
        print(f"  Error: {error*1000:.2f} mm")


def test_workspace_limits():
    """Test IK behavior at workspace boundaries."""
    print("\n=== Testing Workspace Limits ===")
    kin = MinkKinematics()

    # Test extreme positions
    extreme_targets = [
        ([0.3, 0, 0.1], "Far forward reach"),
        ([0, 0.3, 0.1], "Far side reach"),
        ([0, 0, 0.4], "High reach"),
        ([-0.1, -0.1, 0.05], "Behind and low"),
    ]

    for target, description in extreme_targets:
        print(f"\n{description}: {target}")
        target_pos = np.array(target, dtype=np.float32)

        try:
            joints = kin.solve_ik(
                target_pos,
                current_joint_pos_deg=np.zeros(5),
                max_iterations=200,
                tolerance=5e-3,  # More tolerant for extreme positions
            )

            T = kin.forward_kinematics(joints)
            actual_pos = T[:3, 3]
            error = np.linalg.norm(actual_pos - target_pos)

            if error > 0.01:  # 1cm threshold
                print(f"  ⚠️  Large error ({error*1000:.1f}mm) - likely at workspace limit")
            else:
                print(f"  ✓ Reachable (error: {error*1000:.1f}mm)")
            print(
                f"  Joints: [{joints[0]:.1f}, {joints[1]:.1f}, {joints[2]:.1f}, {joints[3]:.1f}, {joints[4]:.1f}]°"
            )

        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_trajectory():
    """Test IK along a trajectory."""
    print("\n=== Testing Trajectory Following ===")
    kin = MinkKinematics()

    # Create a circular trajectory
    center = np.array([0.15, -0.2, 0.15], dtype=np.float32)
    radius = 0.05
    n_points = 16

    print(f"Circular trajectory: center={center}, radius={radius}m")

    # Start from a position near the circle
    init_joints = np.array([20, -30, 45, -15, 0], dtype=np.float32)
    current_joints = init_joints.copy()

    errors = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        target_pos = center + radius * np.array([np.cos(angle), np.sin(angle), 0])

        # Solve IK from current position
        new_joints = kin.solve_ik(
            target_pos, current_joint_pos_deg=current_joints, max_iterations=50, tolerance=1e-3
        )

        # Verify
        T = kin.forward_kinematics(new_joints)
        actual_pos = T[:3, 3]
        error = np.linalg.norm(actual_pos - target_pos)
        errors.append(error)

        # Update for next iteration
        current_joints = new_joints

        if i % 4 == 0:  # Print every 4th point
            print(f"  Point {i+1}/{n_points}: error={error*1000:.2f}mm")

    avg_error = np.mean(errors) * 1000
    max_error = np.max(errors) * 1000
    print(f"\nTrajectory stats: avg error={avg_error:.2f}mm, max error={max_error:.2f}mm")


def main():
    """Run all tests."""
    print("=== Mink IK Standalone Tests ===")
    print("Testing the SO-ARM100 robot kinematics with mink solver")

    try:
        test_forward_kinematics()
        test_inverse_kinematics()
        test_workspace_limits()
        test_trajectory()

        print("\n✅ All tests completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
