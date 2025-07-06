# Build Intel RealSense support

## Install dependencies

1. Make Ubuntu up-to-date including the latest stable kernel:
   ```sh
   sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
   ```
2. Install the core packages required to build _librealsense_ binaries and the affected kernel modules:
   ```sh
   sudo apt-get install libssl-dev libusb-1.0-0-dev libudev-dev pkg-config libgtk-3-dev
   ```
   **Cmake Note:** certain _librealsense_ [CMAKE](https://cmake.org/download/) flags (e.g. CUDA) require version 3.8+ which is currently not made available via apt manager for Ubuntu LTS.
3. Install build tools
   ```sh
   sudo apt-get install git wget cmake build-essential
   ```
4. Prepare Linux Backend and the Dev. Environment \
   Unplug any connected Intel RealSense camera and run:
   ```sh
   sudo apt-get install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev at libudev-dev
   ```

## Install librealsense2

1. Clone/Download the latest stable version of _librealsense2_ in one of the following ways:
   ```sh
   git submodule update --init --recursive
   cd lib/librealsense
   ```

2. Run Intel Realsense permissions script from _librealsense2_ root directory:
   ```sh
   ./scripts/setup_udev_rules.sh
   ```
   Notice: You can always remove permissions by running: `./scripts/setup_udev_rules.sh --uninstall`

## Building librealsense2 SDK

  * Navigate to _librealsense2_ root directory and run:
    ```sh
    mkdir build && cd build
    ```
  * Run cmake configure step, note that you have to run this ***within the venv***
    ```sh
    cmake ../ -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=false -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=$(which python)
    ```
  * Recompile and install _librealsense2_ binaries:
    ```sh
    sudo make uninstall && make clean && make && sudo make install
    ```
    **Note:** Only relevant to CPUs with more than 1 core: use `make -j$(($(nproc)-1)) install` allow parallel compilation.

    **Note:** The shared object will be installed in `/usr/local/lib`, header files in `/usr/local/include`. \
    The binary demos, tutorials and test files will be copied into `/usr/local/bin`

   * The compiled python bindings will be located at `lib/librealsense/build/Release`. Make sure to source `env_setup.sh` or do
      ```sh
      export PYTHONPATH=$PYTHONPATH:./lib/librealsense/build/Release
      ```