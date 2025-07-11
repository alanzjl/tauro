from tauro_inference.client import RemoteRobot

# Connect to remote robot
with RemoteRobot(
    robot_id="robot_001",
    robot_type="so100_follower",
    host="127.0.0.1",  # Robot's IP
    port=50051,
) as robot:
    # Calibrate if needed
    # robot.calibrate()

    # Get observation
    obs = robot.get_observation()
    print(f"Robot state: {obs}")

    # Get joint positions
    joint_positions = obs["observation.state"][:6]
    print(f"Joint positions: {joint_positions}")

    # Add a small offset to the joint positions
    goal_positions = joint_positions - 1
    print(f"Goal positions: {goal_positions}")

    # Send action
    action = {"action": goal_positions}  # Position commands
    robot.send_action(action)
