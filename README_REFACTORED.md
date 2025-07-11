# Tauro - Modular Robot Control Framework

Tauro is a modular Python framework for robot control, designed with a clean separation between edge computing (robot hardware control) and inference (control algorithms and teleoperation).

## Architecture Overview

The framework is split into three main modules:

### 1. `tauro_edge` - Robot Hardware Control
- Runs on the robot's onboard computer
- Direct hardware control (motors, sensors)
- gRPC server for remote control
- Real-time state management
- Calibration and safety features

### 2. `tauro_inference` - Control Algorithms
- Can run onboard or offboard
- Teleoperation interfaces (keyboard, gamepad, neural networks)
- gRPC client for communication with edge
- Visualization and monitoring tools

### 3. `tauro_common` - Shared Components
- Protocol definitions (protobuf)
- Common data types
- Utility functions
- Configuration schemas

## Installation

### Prerequisites
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-dev protobuf-compiler
```

### Development Setup
```bash
# Clone the repository
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"

# Install gRPC tools
uv pip install grpcio grpcio-tools

# Compile protobuf files
python scripts/compile_proto.py
```

## Quick Start

### 1. Start the Edge Server (on robot)
```bash
# Run the edge server
python -m tauro_edge.main --host 0.0.0.0 --port 50051

# With debug logging
python -m tauro_edge.main --log-level DEBUG
```

### 2. Connect from Inference Client
```python
from tauro_inference.client import RemoteRobot

# Connect to remote robot
with RemoteRobot(
    robot_id="robot_001",
    robot_type="so100_follower",
    host="192.168.1.100",  # Robot's IP
    port=50051
) as robot:
    # Calibrate if needed
    robot.calibrate()
    
    # Get observation
    obs = robot.get_observation()
    print(f"Robot state: {obs}")
    
    # Send action
    action = {"action": [0.0, 0.0]}  # Position commands
    robot.send_action(action)
```

### 3. Keyboard Teleoperation
```bash
# Run keyboard teleoperation
python -m tauro_inference.teleop.keyboard_remote \
    --robot-id robot_001 \
    --robot-type so100_follower \
    --host 192.168.1.100 \
    --use-degrees
```

## Module Details

### tauro_edge

The edge module handles all direct robot hardware interaction:

```python
# Example: Custom robot implementation
from tauro_edge.robots import Robot
from tauro_edge.motors import FeetechMotorsBus

class MyRobot(Robot):
    name = "my_robot"
    
    def __init__(self, config):
        super().__init__(config)
        self.motors_bus = FeetechMotorsBus(
            port=config.port,
            motor_configs=config.motor_configs
        )
    
    def connect(self):
        self.motors_bus.connect()
    
    def get_observation(self):
        positions = self.motors_bus.get_positions()
        return {"observation.state": positions}
```

### tauro_inference

The inference module provides control interfaces:

```python
# Example: Custom teleoperation
from tauro_inference.client import AsyncRobotClient
from tauro_common.types import ControlCommand

async def control_loop():
    client = AsyncRobotClient()
    await client.connect_to_server()
    await client.connect_robot("robot_001", "so100_follower")
    
    # Stream control commands
    async def command_generator():
        while True:
            yield ControlCommand(
                timestamp=time.time(),
                robot_id="robot_001",
                joint_commands={"motor_1": 0.5},
                control_mode=ControlMode.POSITION
            )
            await asyncio.sleep(0.01)  # 100Hz
    
    async for state in client.stream_control("robot_001", command_generator()):
        print(f"Robot state: {state.joints}")
```

### tauro_common

Common components shared between modules:

```python
# Example: Using common types
from tauro_common.types import RobotState, JointState, RobotStatus

state = RobotState(
    timestamp=time.time(),
    robot_id="robot_001",
    joints={
        "shoulder": JointState(position=0.5, velocity=0.0, torque=1.0, temperature=25.0),
        "elbow": JointState(position=-0.3, velocity=0.1, torque=0.5, temperature=26.0)
    },
    sensors={"camera": camera_data},
    status=RobotStatus.READY
)
```

## Communication Protocol

The modules communicate using gRPC with protobuf-defined messages:

### Available RPC Methods
- `Connect` - Establish connection to a robot
- `Disconnect` - Disconnect from a robot
- `Calibrate` - Calibrate robot motors
- `GetState` - Get current robot state
- `SendAction` - Send single action command
- `StreamControl` - Bidirectional streaming for real-time control
- `HealthCheck` - Check system health

### Message Types
- `RobotState` - Complete robot state including joints and sensors
- `ControlCommand` - Control commands with timestamps
- `JointState` - Individual joint state (position, velocity, torque, temperature)
- `MotorCalibration` - Motor calibration data

## Configuration

### Robot Configuration
```python
from dataclasses import dataclass
from tauro_edge.robots import RobotConfig

@dataclass
class MyRobotConfig(RobotConfig):
    port: str = "/dev/ttyUSB0"
    baudrate: int = 1000000
    motor_ids: list = field(default_factory=lambda: [1, 2, 3, 4])
```

### Server Configuration
```yaml
# config/edge_server.yaml
server:
  host: "0.0.0.0"
  port: 50051
  max_message_size: 10485760  # 10MB

robots:
  - id: "robot_001"
    type: "so100_follower"
    calibration_dir: "~/.cache/tauro/calibration"
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific module tests
pytest tests/edge/
pytest tests/inference/
pytest tests/common/

# Run with coverage
pytest --cov=tauro_edge --cov=tauro_inference --cov=tauro_common
```

## Deployment

### Edge Deployment (on robot)
```bash
# Create systemd service
sudo tee /etc/systemd/system/tauro-edge.service << EOF
[Unit]
Description=Tauro Edge Robot Control Server
After=network.target

[Service]
Type=simple
User=robot
WorkingDirectory=/home/robot/tauro
ExecStart=/home/robot/tauro/.venv/bin/python -m tauro_edge.main
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable tauro-edge
sudo systemctl start tauro-edge
```

### Network Configuration
- Default gRPC port: 50051
- Ensure firewall allows incoming connections on the gRPC port
- For local network only: bind to specific interface (e.g., `192.168.1.100`)
- For internet access: use proper security (TLS, authentication)

## Performance Optimization

### Edge Optimization
- Use motor bus batch operations
- Implement state caching
- Configure appropriate motor update rates

### Inference Optimization
- Use streaming for continuous control
- Batch multiple robot controls
- Implement client-side state prediction

### Network Optimization
- Enable gRPC compression for slow networks
- Adjust message sizes based on bandwidth
- Use appropriate keepalive settings

## Troubleshooting

### Common Issues

1. **Connection refused**
   - Check if edge server is running
   - Verify IP address and port
   - Check firewall settings

2. **Motor not responding**
   - Verify motor connections
   - Check motor IDs and configuration
   - Run motor diagnostic tools

3. **High latency**
   - Use streaming instead of individual commands
   - Check network connection
   - Consider running inference onboard

4. **Calibration failures**
   - Ensure motors have full range of motion
   - Check for mechanical obstructions
   - Verify motor configuration

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Use the modular architecture
2. Add tests for new features
3. Update documentation
4. Follow the existing code style
5. Submit pull requests with clear descriptions

## License

Apache License 2.0 - See LICENSE file for details