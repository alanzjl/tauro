#!/usr/bin/env python3
"""Example of using the refactored Tauro architecture for remote robot control."""

import argparse
import asyncio
import logging
import time

import numpy as np

from tauro_common.constants import DEFAULT_GRPC_HOST, DEFAULT_GRPC_PORT
from tauro_common.types.robot_types import ControlCommand, ControlMode
from tauro_inference.client.remote_robot import RemoteRobot
from tauro_inference.client.robot_client import AsyncRobotClient


def simple_control_example(host: str, port: int):
    """Simple synchronous control example."""
    print("=== Simple Control Example ===")

    # Connect to robot
    with RemoteRobot(
        robot_id="example_robot", robot_type="so100_follower", host=host, port=port
    ) as robot:
        print("Connected to robot")

        # Calibrate if needed
        if not robot.is_calibrated:
            print("Calibrating robot...")
            success = robot.calibrate()
            print(f"Calibration {'successful' if success else 'failed'}")

        # Get and print observation
        obs = robot.get_observation()
        print(f"Observation keys: {list(obs.keys())}")
        if "observation.state" in obs:
            print(f"Joint states: {obs['observation.state']}")

        # Send some actions
        print("\nSending test actions...")
        for i in range(5):
            # Create sinusoidal motion
            t = i * 0.1
            action = np.array(
                [
                    0.3 * np.sin(2 * np.pi * 0.5 * t),  # Motor 1
                    0.3 * np.cos(2 * np.pi * 0.5 * t),  # Motor 2
                ]
            )

            robot.send_action({"action": action})
            print(f"Sent action {i+1}: {action}")
            time.sleep(0.1)

        print("Done!")


async def streaming_control_example(host: str, port: int):
    """Advanced asynchronous streaming control example."""
    print("\n=== Streaming Control Example ===")

    client = AsyncRobotClient(host=host, port=port)

    try:
        # Connect to server
        await client.connect_to_server()
        print("Connected to server")

        # Connect to robot
        success = await client.connect_robot("stream_robot", "so100_follower")
        if not success:
            print("Failed to connect to robot")
            return
        print("Connected to robot")

        # Calibrate if needed
        print("Calibrating robot...")
        calibrations = await client.calibrate_robot("stream_robot")
        if calibrations:
            print(f"Calibration complete. Motor offsets: {list(calibrations.keys())}")

        # Create command generator for streaming
        async def sine_wave_commands():
            """Generate sinusoidal position commands."""
            start_time = time.time()
            frequency = 0.5  # Hz
            amplitude = 0.3  # normalized units

            for i in range(100):  # Send 100 commands
                t = time.time() - start_time

                # Generate sine wave for each motor
                positions = {
                    "motor_1": amplitude * np.sin(2 * np.pi * frequency * t),
                    "motor_2": amplitude * np.cos(2 * np.pi * frequency * t),
                }

                yield ControlCommand(
                    timestamp=time.time(),
                    robot_id="stream_robot",
                    joint_commands=positions,
                    control_mode=ControlMode.POSITION,
                )

                await asyncio.sleep(0.01)  # 100Hz control rate

        # Stream control and receive states
        print("Starting streaming control (10 seconds)...")
        state_count = 0

        async for state in client.stream_control("stream_robot", sine_wave_commands()):
            state_count += 1

            # Print state every 10th update
            if state_count % 10 == 0:
                positions = [f"{joint.position:.3f}" for joint in state.joints.values()]
                print(f"State {state_count}: positions={positions}")

        print(f"Streaming complete. Received {state_count} state updates")

        # Disconnect
        await client.disconnect_robot("stream_robot")

    finally:
        await client.disconnect_from_server()


async def multi_robot_example(host: str, port: int):
    """Example controlling multiple robots simultaneously."""
    print("\n=== Multi-Robot Control Example ===")

    client = AsyncRobotClient(host=host, port=port)

    try:
        await client.connect_to_server()

        # Connect multiple robots
        robot_ids = ["robot_1", "robot_2", "robot_3"]
        for robot_id in robot_ids:
            success = await client.connect_robot(robot_id, "so100_follower")
            print(f"Connected to {robot_id}: {success}")

        # Send synchronized actions to all robots
        print("Sending synchronized actions...")
        for i in range(20):
            t = i * 0.05

            # Send different actions to each robot
            for j, robot_id in enumerate(robot_ids):
                phase_offset = j * 2 * np.pi / len(robot_ids)
                action = {
                    "motor_1": 0.3 * np.sin(2 * np.pi * 0.5 * t + phase_offset),
                    "motor_2": 0.3 * np.cos(2 * np.pi * 0.5 * t + phase_offset),
                }

                await client.send_action(robot_id, action)

            await asyncio.sleep(0.05)

        # Get final states
        print("\nFinal robot states:")
        for robot_id in robot_ids:
            state = await client.get_robot_state(robot_id)
            if state:
                positions = [f"{j.position:.3f}" for j in state.joints.values()]
                print(f"{robot_id}: positions={positions}")

        # Disconnect all
        for robot_id in robot_ids:
            await client.disconnect_robot(robot_id)

    finally:
        await client.disconnect_from_server()


def health_monitoring_example(host: str, port: int):
    """Example of monitoring robot health."""
    print("\n=== Health Monitoring Example ===")

    from tauro_inference.client.robot_client import RobotClient

    client = RobotClient(host=host, port=port)

    try:
        client.connect_to_server()

        # Check initial health
        health = client.health_check()
        print(f"Server healthy: {health.is_healthy}")
        print(f"Components: {list(health.components.keys())}")

        # Connect a robot and monitor
        if client.connect_robot("monitor_robot", "so100_follower"):
            print("\nMonitoring robot health for 5 seconds...")

            for i in range(5):
                # Get robot state
                state = client.get_robot_state("monitor_robot")
                if state:
                    # Check joint temperatures
                    temps = [j.temperature for j in state.joints.values()]
                    avg_temp = sum(temps) / len(temps) if temps else 0
                    print(f"Time {i+1}s: Average temperature: {avg_temp:.1f}°C")

                # Check overall health
                health = client.health_check()
                robot_health = health.components.get("monitor_robot")
                if robot_health:
                    print(f"  Robot health: {robot_health.status}")

                time.sleep(1)

            client.disconnect_robot("monitor_robot")

    finally:
        client.disconnect_from_server()


def main():
    parser = argparse.ArgumentParser(description="Tauro Remote Control Examples")
    parser.add_argument("--host", type=str, default=DEFAULT_GRPC_HOST, help="gRPC server host")
    parser.add_argument("--port", type=int, default=DEFAULT_GRPC_PORT, help="gRPC server port")
    parser.add_argument(
        "--example",
        type=str,
        default="all",
        choices=["simple", "streaming", "multi", "health", "all"],
        help="Which example to run",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    print(f"Connecting to server at {args.host}:{args.port}\n")

    try:
        if args.example in ["simple", "all"]:
            simple_control_example(args.host, args.port)

        if args.example in ["streaming", "all"]:
            asyncio.run(streaming_control_example(args.host, args.port))

        if args.example in ["multi", "all"]:
            asyncio.run(multi_robot_example(args.host, args.port))

        if args.example in ["health", "all"]:
            health_monitoring_example(args.host, args.port)

    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the tauro_edge server is running:")
        print(f"  python -m tauro_edge.main --host {args.host} --port {args.port}")


if __name__ == "__main__":
    main()
