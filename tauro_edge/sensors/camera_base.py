"""Base class for camera capture implementations."""

import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class CameraType(Enum):
    """Camera type enumeration."""

    RGB = "rgb"
    RGBD = "rgbd"


@dataclass
class CameraConfig:
    """Camera configuration."""

    width: int
    height: int
    fps: int
    jpeg_quality: int = 85
    skip_frames_on_lag: bool = False


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coeffs: np.ndarray | None = None
    width: int | None = None
    height: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for protobuf."""
        result = {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
        }
        if self.distortion_coeffs is not None:
            result["distortion_coeffs"] = self.distortion_coeffs.tolist()
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height
        return result


@dataclass
class FrameMetadata:
    """Frame metadata."""

    exposure_us: int = 0
    gain_db: float = 0.0
    hardware_timestamp_ns: int = 0
    brightness: float = 0.0


@dataclass
class RGBFrame:
    """RGB frame data."""

    image: np.ndarray  # BGR format
    metadata: FrameMetadata


@dataclass
class RGBDFrame:
    """RGB-D frame data."""

    color: np.ndarray  # BGR format
    depth: np.ndarray  # Depth in meters (float32)
    metadata: FrameMetadata
    depth_scale: float = 0.001  # Depth unit in meters
    min_depth: float = 0.1  # Minimum valid depth in meters
    max_depth: float = 10.0  # Maximum valid depth in meters


class CameraBase(ABC):
    """Base class for camera implementations."""

    def __init__(self, camera_id: str, device_id: str, config: dict[str, Any]):
        """Initialize camera.

        Args:
            camera_id: Logical camera ID (e.g., "left", "right")
            device_id: Physical device ID (e.g., "/dev/video0", "realsense://0")
            config: Camera configuration from sensor_ports.yaml
        """
        self.camera_id = camera_id
        self.device_id = device_id
        self.config = config
        self.is_open = False
        self.intrinsics: CameraIntrinsics | None = None
        self.frame_count = 0
        self.last_frame_time = 0
        self.camera_type = CameraType.RGB

    @abstractmethod
    def open(self, camera_config: CameraConfig) -> bool:
        """Open camera with specified configuration.

        Args:
            camera_config: Camera configuration

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close camera."""
        pass

    @abstractmethod
    def capture_frame(self) -> RGBFrame | RGBDFrame | None:
        """Capture a single frame.

        Returns:
            RGBFrame or RGBDFrame, or None if failed
        """
        pass

    def get_camera_type(self) -> CameraType:
        """Get camera type.

        Returns:
            Camera type (RGB or RGBD)
        """
        return self.camera_type

    def compress_rgb_frame(self, frame: RGBFrame, jpeg_quality: int = 85) -> dict[str, Any]:
        """Compress RGB frame for transmission.

        Args:
            frame: RGB frame data
            jpeg_quality: JPEG quality (0-100)

        Returns:
            Dictionary with compressed data
        """
        import cv2

        success, encoded = cv2.imencode(
            ".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        if not success:
            return {"jpg_data": {"jpeg_data": b""}}

        return {"jpg_data": {"jpeg_data": encoded.tobytes()}}

    def compress_rgbd_frame(self, frame: RGBDFrame, jpeg_quality: int = 85) -> dict[str, Any]:
        """Compress RGB-D frame for transmission.

        Args:
            frame: RGB-D frame data
            jpeg_quality: JPEG quality for color (0-100)

        Returns:
            Dictionary with compressed data
        """
        # Create RGB frame for color compression
        rgb_frame = RGBFrame(image=frame.color, metadata=frame.metadata)
        rgb_compressed = self.compress_rgb_frame(rgb_frame, jpeg_quality)

        # Get the JPEG data
        color_compressed = rgb_compressed["jpg_data"]["jpeg_data"]

        # Compress depth with zlib
        depth_compressed = zlib.compress(frame.depth.tobytes())

        return {
            "rgbd_data": {
                "color_compressed": color_compressed,
                "depth_compressed": depth_compressed,
                "depth_scale": frame.depth_scale,
                "min_depth": frame.min_depth,
                "max_depth": frame.max_depth,
            }
        }

    def calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness of image.

        Args:
            image: Image array

        Returns:
            Brightness value (0-1)
        """
        import cv2

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(np.mean(gray) / 255.0)

    def should_skip_frame(self, target_fps: int) -> bool:
        """Check if frame should be skipped for FPS control.

        Args:
            target_fps: Target frames per second

        Returns:
            True if frame should be skipped
        """
        if target_fps <= 0:
            return False

        current_time = time.time()
        min_frame_interval = 1.0 / target_fps

        if current_time - self.last_frame_time < min_frame_interval:
            return True

        self.last_frame_time = current_time
        return False

    def get_intrinsics(self) -> CameraIntrinsics | None:
        """Get camera intrinsics if available.

        Returns:
            Camera intrinsics or None
        """
        return self.intrinsics

    def set_intrinsics(self, intrinsics: CameraIntrinsics) -> None:
        """Set camera intrinsics.

        Args:
            intrinsics: Camera intrinsics
        """
        self.intrinsics = intrinsics
