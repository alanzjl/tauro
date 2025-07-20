# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tauro is a modular Python framework for robot control with a distributed architecture. It separates edge computing (hardware control) from inference (control algorithms), enabling flexible deployment scenarios where robots can be controlled remotely or locally.

## Architecture

The project is organized into three main modules:

### 1. tauro_edge - Hardware Control Layer
- **Purpose**: Direct hardware control and real-time state management
- **Location**: `tauro_edge/`
- **Key Components**:
  - `server.py`: gRPC server for remote robot control
  - `robots/`: Robot implementations (SO100, SO101)
  - `motors/`: Motor controllers (CyberGear, Feetech)
  - `configs/`: Robot configuration files
  - Real-time state management and calibration

### 2. tauro_inference - Control & Algorithms Layer
- **Purpose**: High-level control, teleoperation, and algorithms
- **Location**: `tauro_inference/`
- **Key Components**:
  - `client.py`: RemoteRobot client for connecting to edge servers
  - `teleoperators/`: Teleoperation interfaces (keyboard, etc.)
  - `visualization/`: Visualization tools
  - Control algorithms and planning

### 3. tauro_common - Shared Components
- **Purpose**: Protocol definitions and shared utilities
- **Location**: `tauro_common/`
- **Key Components**:
  - `proto/`: Protocol buffer definitions (robot.proto)
  - `types.py`: Common data types
  - `utils/`: Shared utility functions
  - Configuration schemas

## Development Setup & Commands

### Environment Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Compile protocol buffers (if modified)
cd tauro_common/proto
./compile_protos.sh
```

### Code Quality Commands
```bash
# Format code
ruff format .

# Lint code
ruff check .

# Run pre-commit hooks manually
pre-commit run --all-files

# Run tests
pytest
```

## Key Patterns & Concepts

### 1. Robot Interface
- Robots implement the gRPC service defined in `tauro_common/proto/robot.proto`
- Standard lifecycle: `connect()` → `calibrate()` → `get_observation()`/`send_action()` → `disconnect()`
- Observation and action spaces defined as feature dictionaries

### 2. Communication Architecture
- **gRPC**: Primary communication protocol between edge and inference
- **Streaming Mode**: Low-latency continuous control
- **Polling Mode**: Request-response pattern for discrete control
- Example client usage:
  ```python
  from tauro_inference.client import RemoteRobot
  robot = RemoteRobot(address="localhost:50051")
  robot.connect()
  obs = robot.get_observation()
  robot.send_action(action)
  ```

### 3. Motor System
- Unified motor interface supporting multiple types
- Motor configuration in `tauro_edge/configs/robot_ports.yaml`
- Calibration data stored in `.cache/calibration/`
- Motor types:
  - CyberGear (CAN bus)
  - Feetech (serial)

### 4. Configuration Management
- Robot configurations: `tauro_edge/configs/robot_ports.yaml`
- Calibration data: `.cache/calibration/{robot_name}.json`
- Environment-specific settings via command-line args

## Common Development Tasks

### Adding a New Robot
1. Create robot class in `tauro_edge/robots/`
2. Implement gRPC service methods
3. Define observation/action spaces
4. Add configuration to `robot_ports.yaml`
5. Update server to recognize new robot type

### Working with Motors
```bash
# Configure motor IDs
python scripts/change_motor_id.py --old-id 1 --new-id 127 --port /dev/ttyUSB0

# Calibrate motors
python scripts/calibrate_motors.py --robot so100

# Test motor control
python scripts/test_motor.py --motor-id 127 --port /dev/ttyUSB0
```

### Testing & Debugging
```bash
# Start edge server
python -m tauro_edge.main --host 0.0.0.0 --port 50051 --log-level DEBUG

# Test with keyboard teleop
python scripts/keyboard_teleop.py --robot-address localhost:50051

# Stream camera for debugging
python scripts/stream_camera.py --camera1 0 --port 5000
```

## Project Structure
```
tauro/
├── tauro_edge/           # Hardware control
│   ├── __main__.py      # Module entry point
│   ├── server.py        # gRPC server
│   ├── robots/          # Robot implementations
│   ├── motors/          # Motor controllers
│   └── configs/         # Configuration files
├── tauro_inference/      # Control algorithms
│   ├── client.py        # Remote robot client
│   ├── teleoperators/   # Teleoperation interfaces
│   └── visualization/   # Viz tools
├── tauro_common/         # Shared components
│   ├── proto/           # Protocol definitions
│   ├── types.py         # Common types
│   └── utils/           # Utilities
├── scripts/              # Utility scripts
├── tests/               # Test suite
├── examples/            # Example code
└── lib/                 # External libraries
    └── librealsense/    # Intel RealSense
```

## Important Development Notes

1. **Import Style**: Use absolute imports for cross-module references
   ```python
   from tauro_common.proto import robot_pb2
   from tauro_edge.robots import SO100Robot
   ```

2. **Protocol Buffers**: Always recompile after modifying `.proto` files
   ```bash
   cd tauro_common/proto && ./compile_protos.sh
   ```

3. **Error Handling**: Use appropriate gRPC status codes for errors
   ```python
   import grpc
   context.abort(grpc.StatusCode.NOT_FOUND, "Robot not found")
   ```

4. **Threading**: Be aware of gRPC's threading model
   - Server handles requests in thread pool
   - Use locks for shared state access

5. **Testing**: Test edge and inference components separately
   - Mock gRPC services for unit tests
   - Integration tests with real hardware

## Deployment Considerations

- **Single Machine**: Both edge and inference on same device
- **Distributed**: Edge on robot, inference on remote compute
- **Network**: Ensure gRPC ports (default 50051) are accessible
- **Latency**: Use streaming mode for real-time control
- **Security**: Consider TLS for production deployments

## Common Issues & Solutions

1. **Import Errors**: Ensure all modules are installed with `uv pip install -e .`
2. **Proto Compilation**: Run compile script if seeing proto-related errors
3. **Connection Refused**: Check if edge server is running and port is correct
4. **Motor Not Responding**: Verify motor ID and power connections
5. **Calibration Issues**: Delete `.cache/calibration/` to force recalibration