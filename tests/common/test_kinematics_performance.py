#!/usr/bin/env python3
"""Performance benchmarks for kinematics computations."""

import time
from pathlib import Path

import numpy as np
import pytest

from tauro_common.kinematics.mink_kinematics import MinkKinematics


class TestKinematicsPerformance:
    """Performance tests for kinematics operations."""

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

    def test_forward_kinematics_performance(self, kinematics):
        """Benchmark forward kinematics computation time."""
        n_iterations = 1000
        joint_configs = np.random.uniform(-90, 90, (n_iterations, 5)).astype(np.float32)

        start_time = time.perf_counter()
        for joints in joint_configs:
            _ = kinematics.forward_kinematics(joints)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / n_iterations

        print("\nForward Kinematics Performance:")
        print(f"  Total time for {n_iterations} iterations: {total_time:.3f}s")
        print(f"  Average time per computation: {avg_time*1000:.3f}ms")
        print(f"  Frequency: {1/avg_time:.1f} Hz")

        # Performance assertion - FK should be fast
        assert avg_time < 0.001  # Less than 1ms per computation

    def test_inverse_kinematics_performance(self, kinematics):
        """Benchmark inverse kinematics computation time."""
        n_iterations = 100

        # Generate random but reachable targets
        home_joints = np.zeros(5, dtype=np.float32)
        T_home = kinematics.forward_kinematics(home_joints)
        home_pos = T_home[:3, 3]

        # Small movements around home position
        targets = []
        for _ in range(n_iterations):
            delta = np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
            targets.append(home_pos + delta)

        times = []
        successes = 0

        for target_pos in targets:
            start_time = time.perf_counter()
            solution = kinematics.solve_ik(
                target_position=target_pos,
                current_joint_pos_deg=home_joints,
                max_iterations=100,
                tolerance=1e-3,
            )
            end_time = time.perf_counter()

            times.append(end_time - start_time)

            # Check if converged
            T = kinematics.forward_kinematics(solution)
            error = np.linalg.norm(T[:3, 3] - target_pos)
            if error < 0.005:
                successes += 1

        avg_time = np.mean(times)
        std_time = np.std(times)
        success_rate = successes / n_iterations

        print("\nInverse Kinematics Performance:")
        print(f"  Average time: {avg_time*1000:.3f}ms ± {std_time*1000:.3f}ms")
        print(f"  Min time: {min(times)*1000:.3f}ms")
        print(f"  Max time: {max(times)*1000:.3f}ms")
        print(f"  Success rate: {success_rate*100:.1f}%")
        print(f"  Frequency: {1/avg_time:.1f} Hz")

        # Performance assertions
        assert avg_time < 0.3  # Less than 300ms average (IK can be slower)
        assert success_rate > 0.1  # At least 10% success rate for random targets

    def test_jacobian_performance(self, kinematics):
        """Benchmark Jacobian computation time."""
        n_iterations = 1000
        joint_configs = np.random.uniform(-90, 90, (n_iterations, 5)).astype(np.float32)

        start_time = time.perf_counter()
        for joints in joint_configs:
            _ = kinematics.get_jacobian(joints)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / n_iterations

        print("\nJacobian Computation Performance:")
        print(f"  Total time for {n_iterations} iterations: {total_time:.3f}s")
        print(f"  Average time per computation: {avg_time*1000:.3f}ms")
        print(f"  Frequency: {1/avg_time:.1f} Hz")

        # Jacobian should also be fast
        assert avg_time < 0.002  # Less than 2ms per computation

    def test_ik_delta_performance(self, kinematics):
        """Benchmark solve_ik_delta performance."""
        n_iterations = 100
        current_joints = np.array([0, -30, 60, -30, 0], dtype=np.float32)

        # Small random deltas
        deltas = np.random.uniform(-0.01, 0.01, (n_iterations, 3)).astype(np.float32)

        start_time = time.perf_counter()
        for delta in deltas:
            new_joints = kinematics.solve_ik_delta(
                current_joint_pos_deg=current_joints, delta_position=delta, max_iterations=50
            )
            current_joints = new_joints  # Use result for next iteration
        end_time = time.perf_counter()

        total_time = end_time - start_time
        avg_time = total_time / n_iterations

        print("\nIK Delta Performance:")
        print(f"  Total time for {n_iterations} iterations: {total_time:.3f}s")
        print(f"  Average time per computation: {avg_time*1000:.3f}ms")
        print(f"  Frequency: {1/avg_time:.1f} Hz")

        # Delta IK should be efficient for small movements
        assert avg_time < 0.1  # Less than 100ms average

    def test_repeated_fk_caching(self, kinematics):
        """Test if repeated FK calls benefit from any caching."""
        joints = np.array([30, -45, 60, -30, 15], dtype=np.float32)
        n_iterations = 1000

        # First pass - might involve initialization
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            _ = kinematics.forward_kinematics(joints)
        first_pass_time = time.perf_counter() - start_time

        # Second pass - should be at least as fast
        start_time = time.perf_counter()
        for _ in range(n_iterations):
            _ = kinematics.forward_kinematics(joints)
        second_pass_time = time.perf_counter() - start_time

        print("\nRepeated FK Performance:")
        print(f"  First pass: {first_pass_time:.3f}s")
        print(f"  Second pass: {second_pass_time:.3f}s")
        print(f"  Speedup: {first_pass_time/second_pass_time:.2f}x")

        # Second pass should not be significantly slower
        assert second_pass_time <= first_pass_time * 1.3  # Allow 30% variance

    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_batch_processing_efficiency(self, kinematics, batch_size):
        """Test efficiency of processing multiple targets."""
        # Generate batch of targets
        home_joints = np.zeros(5, dtype=np.float32)
        T_home = kinematics.forward_kinematics(home_joints)
        home_pos = T_home[:3, 3]

        targets = []
        for _ in range(batch_size):
            delta = np.random.uniform(-0.03, 0.03, 3).astype(np.float32)
            targets.append(home_pos + delta)

        # Time batch processing
        start_time = time.perf_counter()
        solutions = []
        for target in targets:
            solution = kinematics.solve_ik(
                target_position=target, current_joint_pos_deg=home_joints, max_iterations=50
            )
            solutions.append(solution)
        batch_time = time.perf_counter() - start_time

        avg_time_per_target = batch_time / batch_size
        print(f"\nBatch Processing (size={batch_size}):")
        print(f"  Total time: {batch_time:.3f}s")
        print(f"  Average per target: {avg_time_per_target*1000:.3f}ms")

        # Efficiency should be maintained with larger batches
        assert avg_time_per_target < 0.15  # Less than 150ms per target

    def test_memory_stability(self, kinematics):
        """Test that repeated operations don't leak memory."""
        import gc

        # Force garbage collection
        gc.collect()

        # Get initial memory usage (rough estimate)
        initial_objects = len(gc.get_objects())

        # Perform many operations
        for _ in range(1000):
            joints = np.random.uniform(-90, 90, 5).astype(np.float32)
            T = kinematics.forward_kinematics(joints)

            if np.random.rand() < 0.5:  # 50% chance to do IK
                target = T[:3, 3] + np.random.uniform(-0.01, 0.01, 3)
                _ = kinematics.solve_ik(
                    target_position=target, current_joint_pos_deg=joints, max_iterations=20
                )

        # Force garbage collection again
        gc.collect()

        # Check object count didn't grow significantly
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        print("\nMemory Stability:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Growth: {object_growth}")

        # Allow some growth but not unbounded
        assert object_growth < 1000  # Arbitrary threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print outputs
