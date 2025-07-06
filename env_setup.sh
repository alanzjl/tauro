#! /bin/bash

# Activate virtual environment
source .venv/bin/activate

# Add RealSense SDK to Python path
export PYTHONPATH=$PYTHONPATH:./lib/librealsense/build/Release
