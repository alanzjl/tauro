# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tauro is a Python package for robot control designed as a modular framework for controlling various types of robots. It provides standardized interfaces for robot control, teleoperation, and sensor integration.

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

# For RealSense camera support
source env_setup.sh
```

### Code Quality Commands
```bash
# Lint code
ruff check .

# Format code
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files
```

### Testing
```bash
# Run tests (pytest is installed but no test directory exists yet)
pytest
```

## Architecture Overview

### Core Abstractions

1. **Robot Interface** (`tauro/common/robots/robot.py`):
   - Abstract base class defining standard robot interface
   - Key lifecycle: `connect()` → `calibrate()` → `get_observation()`/`send_action()` → `disconnect()`
   - Built-in calibration system with JSON persistence
   - Properties define observation/action spaces similar to OpenAI Gym

2. **Motor System**:
   - Multiple motor types supported (CyberGear, Feetech)
   - Unified motor bus interface in `motors/motors_bus.py`
   - Motor calibration with offset management

3. **Teleoperation**:
   - Abstract teleoperator interface in `teleoperators/teleoperator.py`
   - Keyboard implementation uses `sshkeyboard` (no X display required)
   - Configurable for different robot types and control modes

### Key Patterns

- **Configuration Classes**: Each robot/teleoperator has a corresponding config dataclass
- **Feature Dictionaries**: Observation and action spaces defined as dicts with dtype, shape, and names
- **Calibration Persistence**: Motor offsets saved to JSON files in `.cache/calibration/`
- **Error Handling**: Custom exceptions in `common/errors.py` for device states

### Robot Implementations

- **SO100**: Follower robot with optional end effector control
- **SO101**: Similar architecture to SO100
- Each robot has leader/follower variants for teleoperation scenarios

### Camera Integration

- Regular cameras: Use OpenCV for streaming
- RealSense: Depth camera support with colormap visualization
- Web interface: Flask server with HTML/JS frontend for visualization

## Important Development Notes

1. **Import Style**: Use relative imports within the `tauro` package
2. **Motor IDs**: Each motor has unique ID that must be configured before use
3. **Calibration**: Always calibrate robots before use; calibration data persists
4. **Threading**: Keyboard teleoperation uses threading for non-blocking input
5. **Dependencies**: Optional dependencies like `feetech-servo-sdk` installed with `uv pip install -e ".[feetech]"`

## Common Tasks

### Adding a New Robot
1. Create new directory under `tauro/common/robots/`
2. Implement robot class inheriting from `Robot`
3. Define configuration dataclass
4. Implement required abstract methods
5. Define observation and action features

### Working with Motors
- Motor configuration: `scripts/change_motor_id.py`
- Motor calibration stored in `.cache/calibration/{robot_name}.json`

### Teleoperation Testing
```bash
# Basic teleoperation example
python scripts/teleop.py

# Camera streaming
python scripts/stream_camera.py --camera1 0 --port 5000
python scripts/stream_realsense.py --port 5000
```

## Project Structure
```
tauro/
├── common/
│   ├── robots/          # Robot implementations
│   ├── motors/          # Motor controllers
│   ├── teleoperators/   # Control interfaces
│   ├── model/           # Kinematics models
│   └── utils/           # Utilities
├── scripts/             # Example scripts and tools
└── lib/                 # External libraries (librealsense)
```