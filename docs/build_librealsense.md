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
   * Clone the _librealsense2_ repo
     ```sh
     git clone https://github.com/IntelRealSense/librealsense.git
     ```
   * Download and unzip the latest stable _librealsense2_ version from `master` branch \
     [IntelRealSense.zip](https://github.com/IntelRealSense/librealsense/archive/master.zip)

2. Run Intel Realsense permissions script from _librealsense2_ root directory:
   ```sh
   ./scripts/setup_udev_rules.sh
   ```
   Notice: You can always remove permissions by running: `./scripts/setup_udev_rules.sh --uninstall`

3. Build and apply patched kernel modules for:
    * Ubuntu 20/22 (focal/jammy) with LTS kernel 5.13, 5.15, 5.19, 6.2, 6.5 \
      `./scripts/patch-realsense-ubuntu-lts-hwe.sh`
    * Ubuntu 18/20 with LTS kernel (< 5.13) \
     `./scripts/patch-realsense-ubuntu-lts.sh`

    **Note:** What the *.sh script perform?
    The script above will download, patch and build realsense-affected kernel modules (drivers). \
    Then it will attempt to insert the patched module instead of the active one. If failed
    the original uvc modules will be restored.

   >  Check the patched modules installation by examining the generated log as well as inspecting the latest entries in kernel log: \
       `sudo dmesg | tail -n 50` \
       The log should indicate that a new _uvcvideo_ driver has been registered.  
       Refer to [Troubleshooting](#troubleshooting-installation-and-patch-related-issues) in case of errors/warning reports.
