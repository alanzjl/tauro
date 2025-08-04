# Tauro

A modular Python framework for robot control with distributed architecture, designed for flexible deployment scenarios where edge computing (hardware control) can be separated from inference (control algorithms).

## Overview

Tauro provides a clean separation between:
- **Edge Computing**: Direct hardware control and real-time state management
- **Inference**: Control algorithms, teleoperation, and high-level planning
- **Common**: Shared protocol definitions and utilities

This architecture enables:
- Remote robot control over network connections
- Separation of compute-intensive tasks from real-time control
- Flexible deployment (single machine or distributed systems)
- Low-latency streaming control

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd tauro

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .

# For development (includes linting and formatting tools)
uv pip install -e ".[dev]"

# Optional features
uv pip install -e ".[feetech]"      # Feetech servo support
uv pip install -e ".[visualization]" # Visualization tools
uv pip install -e ".[realsense]"    # Intel RealSense support
```

### Basic Usage

1. **Start the robot server (on robot/edge device):**
```bash
# Real hardware
python -m tauro_edge.main --host 0.0.0.0 --port 50051 --log-level DEBUG

# Or use the simulator for testing
python -m tauro_edge simulator --host 0.0.0.0 --port 50051
```

2. **Control the robot (from any device):**
```python
from tauro_inference.client import RemoteRobot

# Connect to robot
robot = RemoteRobot(address="localhost:50051")
robot.connect()

# Get robot state
obs = robot.get_observation()
print(f"Joint positions: {obs['joint_pos']}")

# Send control commands
action = {"joint_pos": [0.0, 0.5, -0.5, 0.0, 0.0, 0.0]}
robot.send_action(action)

robot.disconnect()
```

3. **Teleoperate with keyboard:**
```bash
python scripts/keyboard_teleop.py --robot-address localhost:50051
```

## Architecture

### Modules

1. **tauro_edge** - Hardware control layer
   - gRPC server for remote access
   - Motor controllers (CyberGear, Feetech)
   - Robot implementations (SO100, SO101)
   - Real-time state management
   - Calibration system

2. **tauro_inference** - Control and algorithms layer
   - Remote robot client
   - Teleoperation interfaces
   - Visualization tools
   - Control algorithms

3. **tauro_common** - Shared components
   - Protocol definitions (protobuf)
   - Common data types
   - Configuration schemas
   - Utility functions

### Communication

Tauro uses gRPC for efficient communication between edge and inference layers:
- Supports both polling and streaming modes
- Low-latency control for real-time applications
- Automatic reconnection and error handling

## Robot Configuration

Configure robot connections in `tauro_edge/configs/robot_ports.yaml`:

```yaml
so100:
  motors:
    can0:
      - id: 127
        model: "cybergear"
        name: "base_yaw"
      - id: 126
        model: "cybergear"
        name: "shoulder_pitch"
```

## Features

### Supported Robots
- **SO100**: 6-DOF arm with optional gripper
- **SO101**: Similar architecture with different kinematics
- **Simulated robots**: MuJoCo-based physics simulation for testing

### Motor Support
- CyberGear motors (CAN bus)
- Feetech servos (serial)
- Unified calibration system
- Simulated motors with realistic calibration behavior

### Teleoperation
- Keyboard control (no X display required)
- Leader-follower mode
- Configurable control schemes

### Visualization
- Camera streaming (OpenCV)
- Intel RealSense depth cameras
- Web-based interfaces

### Simulation
- MuJoCo physics engine integration
- Realistic calibration and joint offset handling
- Drop-in replacement for hardware testing
- Multiple simulated robots support

## Development

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Testing

```bash
# Run tests
pytest

# Test specific module
pytest tests/tauro_edge
```

### Common Tasks

**Calibrate motors:**
```bash
python scripts/calibrate_motors.py --robot so100
```

**Change motor IDs:**
```bash
python scripts/change_motor_id.py --old-id 1 --new-id 127 --port /dev/ttyUSB0
```

**Stream camera:**
```bash
# Regular camera
python scripts/stream_camera.py --camera1 0 --port 5000

# RealSense camera
python scripts/stream_realsense.py --port 5001
```

**Run simulator:**
```bash
# Start MuJoCo simulator server
python -m tauro_edge simulator --port 50051

# Connect to simulated robot (same as real robot)
python scripts/keyboard_teleop.py --robot-address localhost:50051
```

## Deployment

### Single Machine
Both edge and inference components run on the same device:
```bash
# Terminal 1: Start edge server
python -m tauro_edge.main

# Terminal 2: Run inference/control
python scripts/keyboard_teleop.py
```

### Distributed System
Edge server runs on robot, inference runs on separate compute:
```bash
# On robot (edge device)
python -m tauro_edge.main --host 0.0.0.0 --robot-port 50051

# On compute device
python scripts/keyboard_teleop.py --robot-address robot.local:50051
```

## Troubleshooting

### Connection Issues
- Ensure firewall allows gRPC port (default: 50051)
- Check network connectivity: `ping robot.local`
- Verify server is running: `netstat -tlnp | grep 50051`

### Motor Issues
- Check motor connections and power
- Verify motor IDs match configuration
- Run calibration if movements are incorrect

### Performance
- Use streaming mode for low-latency control
- Monitor network latency for remote setups
- Consider local deployment for time-critical tasks

## License

[Add license information]

## Contributing

[Add contributing guidelines]