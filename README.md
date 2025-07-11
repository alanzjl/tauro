# Tauro

A Python package for robot control.

## Installation

```bash
pip install tauro
```

## Development Setup

This project uses uv for dependency management and ruff for code formatting and linting. To set up the development environment:

1. Install uv (if you haven't already):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   # .venv\Scripts\activate  # On Windows
   
   uv pip install -e ".[dev]"
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Quality

This project uses [ruff](https://github.com/astral-sh/ruff) for code formatting and linting. Ruff is configured to:
- Format code (similar to black)
- Sort imports (similar to isort)
- Lint code (similar to flake8)

You can run ruff manually with:
```bash
# Check your code
ruff check .

# Format your code
ruff format .
```

The pre-commit hooks will automatically run these checks before each commit.

## Usage

### Basic Robot Control

```python
from tauro_inference.client import RemoteRobot

# Connect to a remote robot
with RemoteRobot(
    robot_id="robot_001",
    robot_type="so100_follower", 
    host="127.0.0.1",
    port=50051
) as robot:
    # Get current robot state
    obs = robot.get_observation()
    
    # Send joint commands
    robot.send_action({"shoulder_pan.pos": 0.5})
    
    # Or send end effector commands (if enabled)
    robot.send_action({
        "end_effector": {
            "delta_x": 0.01,  # Move 1cm in X
            "delta_y": 0.0,
            "delta_z": 0.0,
            "gripper": 1.0
        }
    })
```

### Robot Configuration

Robot configurations are stored in `tauro_edge/configs/robot_ports.yaml`:

```yaml
robots:
  robot_001:
    port: "/dev/ttyACM0"
    type: "so100_follower"
```

### Testing Robot Control

A test script is provided to demonstrate reading robot pose and applying small deltas:

```bash
# Test all control modes (joint, end effector, gripper)
python scripts/test_robot_control.py --robot-id robot_001 --host 127.0.0.1

# Test only joint control
python scripts/test_robot_control.py --mode joint

# Test only end effector control (requires enable_end_effector_control=True in robot config)
python scripts/test_robot_control.py --mode end_effector
```

The test script demonstrates:
- Reading current joint positions and applying small deltas
- Moving the end effector in Cartesian space (X, Y, Z)
- Opening and closing the gripper

### Calibrating a Robot

To calibrate a robot before use:

```python
from tauro_inference.client import RemoteRobot

with RemoteRobot(robot_id="robot_001", robot_type="so100_follower") as robot:
    robot.calibrate()  # Interactive calibration process
```

## Visualize Camera views

For single or dual regular camera(s), run
```sh
python scripts/stream_camera.py --width 848 --height 480 --fps 30 --camera1 0 --camera2 2 --port 5000
```

For RealSense camera, run
```sh
python scripts/stream_realsense.py --width 848 --height 480 --fps 30 --colormap viridis --port 5000
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 