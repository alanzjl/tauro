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

```python
from tauro import Robot

# Example usage will be added here
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