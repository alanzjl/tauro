# Migration Guide: Tauro Refactoring

This guide helps you migrate from the monolithic `tauro.common` structure to the new modular architecture with `tauro_edge`, `tauro_inference`, and `tauro_common`.

## Overview of Changes

### Module Reorganization
- `tauro.common.motors` → `tauro_edge.motors`
- `tauro.common.robots` → `tauro_edge.robots`
- `tauro.common.teleoperators` → `tauro_inference.teleop`
- Shared utilities → `tauro_common`

### New Features
- gRPC-based communication between edge and inference
- Remote robot control capability
- Streaming control for low-latency operation
- Better separation of concerns

## Import Changes

### Before
```python
from tauro.common.robots import Robot
from tauro.common.motors import FeetechMotorsBus
from tauro.common.teleoperators import KeyboardTeleoperator
from tauro.common.constants import default_calibration_path
```

### After
```python
# For edge (robot hardware) code:
from tauro_edge.robots import Robot
from tauro_edge.motors import FeetechMotorsBus

# For inference (control) code:
from tauro_inference.teleop import KeyboardTeleoperator
from tauro_inference.client import RemoteRobot

# For shared code:
from tauro_common.constants import default_calibration_path
from tauro_common.types import RobotState, ControlCommand
```

## Code Migration Examples

### 1. Direct Robot Control → Edge Server

**Before:**
```python
from tauro.common.robots.so100_follower import SO100Follower

robot = SO100Follower(config)
robot.connect()
robot.calibrate()

while True:
    obs = robot.get_observation()
    action = compute_action(obs)
    robot.send_action(action)
```

**After (Edge Server):**
```python
# This now runs as part of the edge server
# Start with: python -m tauro_edge.main

# The robot is managed by the gRPC server
# Client code connects remotely (see below)
```

**After (Inference Client):**
```python
from tauro_inference.client import RemoteRobot

# Connect to robot through edge server
with RemoteRobot(
    robot_id="robot_001",
    robot_type="so100_follower",
    host="robot.local",  # Robot's hostname/IP
    port=50051
) as robot:
    robot.calibrate()
    
    while True:
        obs = robot.get_observation()
        action = compute_action(obs)
        robot.send_action(action)
```

### 2. Teleoperation Migration

**Before:**
```python
from tauro.common.teleoperators.keyboard import KeyboardTeleoperator
from tauro.common.robots import make_robot

robot = make_robot("so100_follower", config)
teleop = KeyboardTeleoperator(robot)
teleop.run()
```

**After:**
```python
from tauro_inference.teleop.keyboard_remote import RemoteKeyboardTeleoperator

# Teleoperator now connects to remote robot
teleop = RemoteKeyboardTeleoperator(
    robot_id="robot_001",
    robot_type="so100_follower",
    host="robot.local",
    port=50051
)

teleop.connect()
teleop.run()
```

### 3. Custom Robot Implementation

**Before:**
```python
from tauro.common.robots import Robot

class MyRobot(Robot):
    def connect(self):
        # Hardware connection
        pass
```

**After:**
```python
# In tauro_edge/robots/my_robot.py
from tauro_edge.robots import Robot

class MyRobot(Robot):
    name = "my_robot"
    
    def connect(self):
        # Hardware connection
        pass

# Register in edge server for remote access
```

### 4. Streaming Control (New Feature)

```python
# High-frequency control with streaming
import asyncio
from tauro_inference.client import AsyncRobotClient
from tauro_common.types import ControlCommand

async def control_loop():
    client = AsyncRobotClient()
    await client.connect_to_server()
    await client.connect_robot("robot_001", "so100_follower")
    
    async def command_generator():
        while True:
            command = ControlCommand(
                timestamp=time.time(),
                robot_id="robot_001",
                joint_commands=compute_commands(),
                control_mode=ControlMode.POSITION
            )
            yield command
            await asyncio.sleep(0.001)  # 1kHz
    
    async for state in client.stream_control("robot_001", command_generator()):
        process_state(state)
```

## Configuration Changes

### Before
```python
config = SO100FollowerConfig(
    port="/dev/ttyUSB0",
    calibration_dir="~/.cache/tauro"
)
```

### After
Edge configuration is now handled by the server:
```bash
# Start edge server with robot config
python -m tauro_edge.main --config robot_config.yaml
```

Inference configuration:
```python
# Just specify connection details
robot = RemoteRobot(
    robot_id="robot_001",
    robot_type="so100_follower",
    host="192.168.1.100",
    port=50051
)
```

## Testing Changes

### Before
```python
def test_robot():
    robot = SO100Follower(config)
    robot.connect()
    # Test directly
```

### After
```python
# Test edge components
def test_robot():
    robot = SO100Follower(config)
    robot.connect()
    # Test directly (same as before)

# Test inference with mock server
def test_remote_control():
    with mock_grpc_server() as server:
        client = RemoteRobot("test", "mock", host="localhost")
        # Test through gRPC
```

## Deployment Changes

### Before
- Single process running everything
- Must run on robot hardware

### After
- Edge server runs on robot
- Inference can run anywhere
- Better scalability and flexibility

### Edge Deployment
```bash
# On robot
pip install tauro[edge]
python -m tauro_edge.main --host 0.0.0.0
```

### Inference Deployment
```bash
# On control computer
pip install tauro[inference]
python your_control_script.py --robot-host 192.168.1.100
```

## Common Issues and Solutions

### 1. ImportError after migration
- Update all imports to use new module names
- Run `pip install -e .` to reinstall

### 2. Connection refused
- Ensure edge server is running
- Check firewall settings
- Verify IP address and port

### 3. Performance degradation
- Use streaming API for high-frequency control
- Consider running inference on same machine as edge
- Check network latency

### 4. Missing features
- Some features may need adaptation for remote use
- File-based operations need different approach
- Consider using shared storage or streaming

## Gradual Migration Strategy

1. **Phase 1**: Update imports and test compilation
   ```bash
   # Update imports
   find . -name "*.py" -exec sed -i 's/tauro\.common\.robots/tauro_edge.robots/g' {} \;
   find . -name "*.py" -exec sed -i 's/tauro\.common\.motors/tauro_edge.motors/g' {} \;
   ```

2. **Phase 2**: Run edge server alongside old code
   - Start edge server
   - Gradually migrate control code to use remote API

3. **Phase 3**: Full migration
   - All control code uses inference module
   - Remove direct hardware access from inference

## Support

For migration help:
1. Check the examples in `examples/`
2. Run tests to verify functionality
3. Open an issue for migration problems