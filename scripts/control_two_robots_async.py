#!/usr/bin/env python3
"""
Async script demonstrating concurrent control of two robots via gRPC.
Shows how to send commands to multiple robots simultaneously using async/await.
"""

import argparse
import asyncio
import time

from tauro_inference.client import RobotClient


async def move_robot_sequence(robot: RobotClient, robot_name: str, offset: float = 0):
    """
    Move a single robot through a sequence of positions.

    Args:
        robot: The robot client
        robot_name: Name/ID for logging
        offset: Position offset to make robots move differently
    """
    print(f"\n[{robot_name}] Starting movement sequence")

    # Get current state
    state = await robot.get_robot_state()
    joint_positions = {k: v.position for k, v in state.joints.items()}
    print(f"[{robot_name}] Current positions: {joint_positions}")

    # Sequence of movements
    movements = [
        (0.0 + offset, "zero + offset"),
        (30.0 + offset, "positive"),
        (-30.0 + offset, "negative"),
        (0.0 + offset, "back to zero + offset"),
    ]

    for target, description in movements:
        print(f"[{robot_name}] Moving to {description} ({target:.1f})")
        action = {"joints": {"position": {k: target for k in joint_positions.keys()}}}
        success = await robot.send_action(action)
        if not success:
            print(f"[{robot_name}] Failed to send action!")

        # Wait for movement to complete
        await asyncio.sleep(2)

        # Get and print new position
        state = await robot.get_robot_state()
        new_positions = {k: v.position for k, v in state.joints.items()}
        print(f"[{robot_name}] New positions: {list(new_positions.values())[0]:.1f}")

    # Go to home position
    await robot.goto_home_position()
    print(f"[{robot_name}] Going to home position")
    await asyncio.sleep(2)

    print(f"[{robot_name}] Movement sequence complete")


async def control_two_robots_simultaneously(
    robot1_config: dict,
    robot2_config: dict,
):
    """
    Control two robots simultaneously using async operations.

    Args:
        robot1_config: Configuration for first robot
        robot2_config: Configuration for second robot
    """
    print("=== Controlling Two Robots Simultaneously ===")

    # Connect to both robots
    async with RobotClient(**robot1_config) as robot1, RobotClient(**robot2_config) as robot2:
        print(f"Connected to robot 1: {robot1_config['robot_id']}")
        print(f"Connected to robot 2: {robot2_config['robot_id']}")

        # Move both robots simultaneously with different offsets
        # Robot 1 will move to positions as-is
        # Robot 2 will move to positions with +10 offset
        await asyncio.gather(
            move_robot_sequence(robot1, robot1_config["robot_id"], offset=0),
            move_robot_sequence(robot2, robot2_config["robot_id"], offset=10),
        )

        print("\n=== Both robots completed their sequences ===")


async def control_two_robots_interleaved(
    robot1_config: dict,
    robot2_config: dict,
):
    """
    Control two robots with interleaved movements (alternating).

    Args:
        robot1_config: Configuration for first robot
        robot2_config: Configuration for second robot
    """
    print("\n=== Controlling Two Robots with Interleaved Movements ===")

    async with RobotClient(**robot1_config) as robot1, RobotClient(**robot2_config) as robot2:
        print(f"Connected to robot 1: {robot1_config['robot_id']}")
        print(f"Connected to robot 2: {robot2_config['robot_id']}")

        # Get initial states
        state1 = await robot1.get_robot_state()
        state2 = await robot2.get_robot_state()

        joints1 = list(state1.joints.keys())
        joints2 = list(state2.joints.keys())

        # Alternating movements
        positions = [0, 20, -20, 30, -30, 0]

        for i, pos in enumerate(positions):
            # Alternate between robots
            if i % 2 == 0:
                # Move robot 1
                print(f"\n[Robot 1] Moving to {pos}")
                action = {"joints": {"position": {k: pos for k in joints1}}}
                await robot1.send_action(action)
            else:
                # Move robot 2
                print(f"\n[Robot 2] Moving to {pos}")
                action = {"joints": {"position": {k: pos for k in joints2}}}
                await robot2.send_action(action)

            # Small delay to see the movement
            await asyncio.sleep(1.5)

        # Go to home position
        await robot1.goto_home_position()
        await robot2.goto_home_position()
        print("[Robot 1] Going to home position")
        print("[Robot 2] Going to home position")
        await asyncio.sleep(2)

        print("\n=== Interleaved control complete ===")


async def mirror_control(
    robot1_config: dict,
    robot2_config: dict,
    duration: float = 12.0,
):
    """
    Mirror control: Robot 2 mirrors Robot 1's position in real-time.

    Args:
        robot1_config: Configuration for leader robot
        robot2_config: Configuration for follower robot
        duration: How long to run the mirror control
    """
    print("\n=== Mirror Control: Robot 2 mirrors Robot 1 ===")

    async with RobotClient(**robot1_config) as robot1, RobotClient(**robot2_config) as robot2:
        print(f"Leader robot: {robot1_config['robot_id']}")
        print(f"Follower robot: {robot2_config['robot_id']}")

        start_time = time.time()

        # Shared state for the leader position
        leader_position = {"positions": None, "lock": asyncio.Lock()}

        # Move robot 1 through a smooth trajectory
        # Robot 2 will mirror its positions
        async def move_leader():
            """Move the leader robot through a trajectory."""
            positions = [0, 30, 0, -30, 0, 20, -20, 0]

            # Get initial joint names
            state = await robot1.get_robot_state()
            joints = list(state.joints.keys())

            for pos in positions:
                if time.time() - start_time > duration - 2:
                    break

                action = {"joints": {"position": {k: pos for k in joints}}}
                await robot1.send_action(action)

                # Update shared state with the target position
                async with leader_position["lock"]:
                    leader_position["positions"] = {k: pos for k in joints}

                print(f"[Robot 1] Moving to {pos}")
                await asyncio.sleep(1.5)

            # Go to home position
            await robot1.goto_home_position()
            print("[Robot 1] Going to home position")

            # Update shared state to home position
            async with leader_position["lock"]:
                leader_position["positions"] = {k: 0 for k in joints}

            await asyncio.sleep(2)

        async def follow_leader():
            """Make robot 2 follow robot 1's positions."""
            last_positions = None

            while time.time() - start_time < duration:
                # Get the leader's target position from shared state
                async with leader_position["lock"]:
                    current_positions = leader_position["positions"]

                # If we have positions and they've changed, update robot 2
                if current_positions and current_positions != last_positions:
                    action = {"joints": {"position": current_positions}}
                    await robot2.send_action(action)
                    print(f"[Robot 2] Mirroring to {list(current_positions.values())[0]:.1f}")
                    last_positions = current_positions.copy()

                # Update rate for following
                await asyncio.sleep(0.2)

            # Final home position for robot 2
            await robot2.goto_home_position()
            print("[Robot 2] Going to home position")

        # Run both tasks concurrently
        await asyncio.gather(
            move_leader(),
            follow_leader(),
        )

        print("\n=== Mirror control complete ===")


async def main():
    parser = argparse.ArgumentParser(
        description="Control two robots simultaneously using async operations"
    )

    # Robot 1 configuration
    parser.add_argument("--robot1-id", default="robot_001", help="First robot ID")
    parser.add_argument("--robot1-type", default="so100_follower", help="First robot type")
    parser.add_argument("--robot1-host", default="127.0.0.1", help="First robot server host")
    parser.add_argument("--robot1-port", type=int, default=50051, help="First robot server port")

    # Robot 2 configuration
    parser.add_argument("--robot2-id", default="robot_002", help="Second robot ID")
    parser.add_argument("--robot2-type", default="so100_follower", help="Second robot type")
    parser.add_argument("--robot2-host", default="127.0.0.1", help="Second robot server host")
    parser.add_argument("--robot2-port", type=int, default=50051, help="Second robot server port")

    # Control mode
    parser.add_argument(
        "--mode",
        choices=["simultaneous", "interleaved", "mirror", "all"],
        default="simultaneous",
        help="Control mode for the two robots",
    )

    args = parser.parse_args()

    # Create robot configurations
    robot1_config = {
        "robot_id": args.robot1_id,
        "robot_type": args.robot1_type,
        "host": args.robot1_host,
        "port": args.robot1_port,
    }

    robot2_config = {
        "robot_id": args.robot2_id,
        "robot_type": args.robot2_type,
        "host": args.robot2_host,
        "port": args.robot2_port,
    }

    print(f"Robot 1: {args.robot1_id} at {args.robot1_host}:{args.robot1_port}")
    print(f"Robot 2: {args.robot2_id} at {args.robot2_host}:{args.robot2_port}")

    # Run the selected mode
    if args.mode == "simultaneous" or args.mode == "all":
        await control_two_robots_simultaneously(robot1_config, robot2_config)

    if args.mode == "interleaved" or args.mode == "all":
        await control_two_robots_interleaved(robot1_config, robot2_config)

    if args.mode == "mirror" or args.mode == "all":
        await mirror_control(robot1_config, robot2_config, duration=10.0)

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())
