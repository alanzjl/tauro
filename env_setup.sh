#! /bin/bash

# Activate virtual environment
source .venv/bin/activate

# Add RealSense SDK to Python path
export PYTHONPATH=$PYTHONPATH:/home/azhao/src/tauro/lib/librealsense/build/Release
