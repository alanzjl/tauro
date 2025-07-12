#!/usr/bin/env python3
"""Test script for controlling two robots simultaneously."""

import asyncio
import time
import math
from concurrent.futures import ThreadPoolExecutor
from tauro_inference.client import RemoteRobot


class DualRobotController:
    """Controller for managing two robots simultaneously."""
    
    def __init__(self, host="127.0.0.1", port=50051):
        self.host = host
        self.port = port
        self.robot1 = None
        self.robot2 = None
        
    def connect(self):
        """Connect to both robots."""
        print("=== Connecting to robots ===")
        
        # Connect robot 1
        print("\nConnecting to robot_001...")
        self.robot1 = RemoteRobot(
            robot_id="robot_001",
            robot_type="so100_follower",
            host=self.host,
            port=self.port
        )
        self.robot1.connect()
        print("✓ Robot_001 connected")
        
        # Connect robot 2
        print("\nConnecting to robot_002...")
        self.robot2 = RemoteRobot(
            robot_id="robot_002", 
            robot_type="so100_follower",
            host=self.host,
            port=self.port
        )
        self.robot2.connect()
        print("✓ Robot_002 connected")
        
    def disconnect(self):
        """Disconnect from both robots."""
        if self.robot1:
            self.robot1.disconnect()
        if self.robot2:
            self.robot2.disconnect()
        print("\n✓ Both robots disconnected")
        
    def get_states(self):
        """Get current state of both robots."""
        obs1 = self.robot1.get_observation()
        obs2 = self.robot2.get_observation()
        return obs1, obs2
    
    def send_actions(self, action1, action2):
        """Send actions to both robots simultaneously."""
        # Use threads for parallel execution
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self.robot1.send_action, action1)
            future2 = executor.submit(self.robot2.send_action, action2)
            
            success1 = future1.result()
            success2 = future2.result()
            
        return success1, success2
    
    def mirror_movement_test(self):
        """Test where robot2 mirrors robot1's movements."""
        print("\n=== Mirror Movement Test ===")
        print("Robot_002 will mirror Robot_001's movements\n")
        
        # Get initial states
        obs1, obs2 = self.get_states()
        
        # Store initial positions
        initial_positions = {}
        for key in obs1.keys():
            if key.endswith(".pos") and key != "gripper.pos":
                motor_name = key.removesuffix(".pos")
                initial_positions[motor_name] = obs1[key]
                print(f"{motor_name}: Robot1={obs1[key]:.2f}, Robot2={obs2.get(key, 0):.2f}")
        
        # Move robot1 and mirror with robot2
        print("\nPerforming mirrored movements...")
        movements = [
            ("shoulder_pan", 10),
            ("shoulder_lift", -10),
            ("elbow_flex", 15),
            ("wrist_flex", -15),
            ("wrist_roll", 20)
        ]
        
        for motor_name, delta in movements:
            print(f"\nMoving {motor_name} by {delta:+.1f} degrees...")
            
            # Calculate new positions
            new_pos1 = initial_positions[motor_name] + delta
            new_pos2 = initial_positions[motor_name] + delta  # Mirror movement
            
            # Send commands to both robots
            action1 = {f"{motor_name}.pos": new_pos1}
            action2 = {f"{motor_name}.pos": new_pos2}
            
            success1, success2 = self.send_actions(action1, action2)
            print(f"  Robot1: {'✓' if success1 else '✗'}, Robot2: {'✓' if success2 else '✗'}")
            
            time.sleep(0.5)
        
        # Return to initial positions
        print("\nReturning to initial positions...")
        for motor_name, pos in initial_positions.items():
            action = {f"{motor_name}.pos": pos}
            self.send_actions(action, action)
        
        time.sleep(1)
        print("✓ Mirror test completed")
    
    def synchronized_wave_test(self):
        """Test synchronized wave motion between robots."""
        print("\n=== Synchronized Wave Test ===")
        print("Both robots will perform synchronized wave motions\n")
        
        # Get initial states
        obs1, obs2 = self.get_states()
        
        # Store initial shoulder_pan positions
        initial_pan1 = obs1.get("shoulder_pan.pos", 0)
        initial_pan2 = obs2.get("shoulder_pan.pos", 0)
        
        print(f"Initial positions - Robot1: {initial_pan1:.2f}, Robot2: {initial_pan2:.2f}")
        
        # Perform wave motion
        print("\nPerforming synchronized wave (10 steps)...")
        amplitude = 30  # degrees
        steps = 10
        
        for i in range(steps):
            # Calculate sine wave position
            angle = (i / steps) * 2 * math.pi
            offset = amplitude * math.sin(angle)
            
            # Calculate positions for both robots
            pos1 = initial_pan1 + offset
            pos2 = initial_pan2 - offset  # Opposite phase
            
            # Send commands
            action1 = {"shoulder_pan.pos": pos1}
            action2 = {"shoulder_pan.pos": pos2}
            
            self.send_actions(action1, action2)
            print(f"  Step {i+1}/10: Robot1={pos1:.1f}, Robot2={pos2:.1f}")
            
            time.sleep(0.3)
        
        # Return to initial positions
        print("\nReturning to initial positions...")
        action1 = {"shoulder_pan.pos": initial_pan1}
        action2 = {"shoulder_pan.pos": initial_pan2}
        self.send_actions(action1, action2)
        
        print("✓ Wave test completed")
    
    def opposite_movement_test(self):
        """Test where robots move in opposite directions."""
        print("\n=== Opposite Movement Test ===")
        print("Robots will move joints in opposite directions\n")
        
        # Get initial states
        obs1, obs2 = self.get_states()
        
        # Test opposite movements
        tests = [
            ("shoulder_lift", 15, -15),
            ("elbow_flex", -20, 20),
            ("wrist_flex", 25, -25)
        ]
        
        for motor_name, delta1, delta2 in tests:
            initial1 = obs1.get(f"{motor_name}.pos", 0)
            initial2 = obs2.get(f"{motor_name}.pos", 0)
            
            print(f"\n{motor_name}: Robot1 {delta1:+.1f}°, Robot2 {delta2:+.1f}°")
            
            # Send opposite commands
            action1 = {f"{motor_name}.pos": initial1 + delta1}
            action2 = {f"{motor_name}.pos": initial2 + delta2}
            
            success1, success2 = self.send_actions(action1, action2)
            print(f"  Commands sent: Robot1 {'✓' if success1 else '✗'}, Robot2 {'✓' if success2 else '✗'}")
            
            time.sleep(1)
            
            # Return to initial
            action1 = {f"{motor_name}.pos": initial1}
            action2 = {f"{motor_name}.pos": initial2}
            self.send_actions(action1, action2)
            
            time.sleep(0.5)
        
        print("\n✓ Opposite movement test completed")


def main():
    """Main test function."""
    controller = DualRobotController()
    
    try:
        # Connect to both robots
        controller.connect()
        
        # Run tests
        controller.mirror_movement_test()
        controller.synchronized_wave_test()
        controller.opposite_movement_test()
        
        print("\n=== All tests completed successfully! ===")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Always disconnect
        controller.disconnect()


if __name__ == "__main__":
    main()