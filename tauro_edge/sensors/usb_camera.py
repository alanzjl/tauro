"""USB camera implementation."""

import time
from typing import Any

import cv2

from tauro_edge.sensors.camera_base import (
    CameraBase,
    CameraConfig,
    CameraType,
    FrameMetadata,
    RGBDFrame,
    RGBFrame,
)


class USBCamera(CameraBase):
    """USB camera implementation using OpenCV."""

    def __init__(self, camera_id: str, device_id: str, config: dict[str, Any]):
        """Initialize USB camera.

        Args:
            camera_id: Logical camera ID (e.g., "left", "right")
            device_id: Physical device ID (e.g., "/dev/video0")
            config: Camera configuration from sensor_ports.yaml
        """
        super().__init__(camera_id, device_id, config)
        self.camera_type = CameraType.RGB
        self.cap: cv2.VideoCapture | None = None
        self.current_config: CameraConfig | None = None

    def open(self, camera_config: CameraConfig) -> bool:
        """Open USB camera with specified configuration.

        Args:
            camera_config: Camera configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert device path to camera index if needed
            if self.device_id.startswith("/dev/video"):
                try:
                    cam_index = int(self.device_id.replace("/dev/video", ""))
                except ValueError:
                    print(f"Invalid device ID: {self.device_id}")
                    return False
            else:
                cam_index = int(self.device_id)

            # Open camera
            self.cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.device_id}")
                return False

            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
            self.cap.set(cv2.CAP_PROP_FPS, camera_config.fps)

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Set MJPEG format if specified in config
            if self.config.get("format") == "MJPEG":
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

            # Verify settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"Camera {self.camera_id} opened: {actual_width}x{actual_height}@{actual_fps}fps")

            self.is_open = True
            self.current_config = camera_config
            self.frame_count = 0

            # Capture a test frame to ensure camera is working
            ret, _ = self.cap.read()
            if not ret:
                print(f"Failed to capture test frame from {self.camera_id}")
                self.close()
                return False

            return True

        except Exception as e:
            print(f"Error opening camera {self.camera_id}: {e}")
            self.close()
            return False

    def close(self) -> None:
        """Close USB camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_open = False
        print(f"Camera {self.camera_id} closed")

    def capture_frame(self) -> RGBFrame | RGBDFrame | None:
        """Capture a single RGB frame.

        Returns:
            RGBFrame or None if failed
        """
        if not self.is_open or self.cap is None:
            return None

        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to capture frame from {self.camera_id}")
                return None

            # Get frame metadata
            timestamp_ns = int(time.time_ns())

            # Try to get camera properties
            exposure_us = 0
            gain_db = 0.0

            try:
                # These properties might not be supported by all cameras
                exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                if exposure > 0:
                    exposure_us = int(exposure * 1000)  # Convert to microseconds

                gain = self.cap.get(cv2.CAP_PROP_GAIN)
                if gain >= 0:
                    gain_db = float(gain)
            except Exception:
                pass

            # Calculate brightness
            brightness = self.calculate_brightness(frame)

            # Create metadata
            metadata = FrameMetadata(
                exposure_us=exposure_us,
                gain_db=gain_db,
                hardware_timestamp_ns=timestamp_ns,
                brightness=brightness,
            )

            # Create RGB frame
            rgb_frame = RGBFrame(image=frame, metadata=metadata)

            self.frame_count += 1
            return rgb_frame

        except Exception as e:
            print(f"Error capturing frame from {self.camera_id}: {e}")
            return None

    def set_exposure(self, exposure_us: int) -> bool:
        """Set camera exposure time.

        Args:
            exposure_us: Exposure time in microseconds

        Returns:
            True if successful, False otherwise
        """
        if not self.is_open or self.cap is None:
            return False

        try:
            # Disable auto exposure
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual mode
            # Set exposure (convert to camera units)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_us / 1000.0)
            return True
        except Exception:
            return False

    def set_gain(self, gain_db: float) -> bool:
        """Set camera gain.

        Args:
            gain_db: Gain in decibels

        Returns:
            True if successful, False otherwise
        """
        if not self.is_open or self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_GAIN, gain_db)
            return True
        except Exception:
            return False
