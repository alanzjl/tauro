"""Intel RealSense camera implementation."""

from typing import Any

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    print("WARNING: pyrealsense2 not installed. RealSense cameras will not be available.")
    rs = None

from tauro_edge.sensors.camera_base import (
    CameraBase,
    CameraConfig,
    CameraIntrinsics,
    CameraType,
    FrameMetadata,
    RGBDFrame,
    RGBFrame,
)


class RealSenseCamera(CameraBase):
    """Intel RealSense camera implementation."""

    def __init__(self, camera_id: str, device_id: str, config: dict[str, Any]):
        """Initialize RealSense camera.

        Args:
            camera_id: Logical camera ID (e.g., "depth")
            device_id: Physical device ID (e.g., "realsense://0" or serial number)
            config: Camera configuration from sensor_ports.yaml
        """
        super().__init__(camera_id, device_id, config)
        self.camera_type = CameraType.RGBD
        self.pipeline: rs.pipeline | None = None
        self.pipeline_config: rs.config | None = None
        self.align: rs.align | None = None
        self.current_config: CameraConfig | None = None
        self.depth_scale = 0.001  # Default depth scale

    def open(self, camera_config: CameraConfig) -> bool:
        """Open RealSense camera with specified configuration.

        Args:
            camera_config: Camera configuration

        Returns:
            True if successful, False otherwise
        """
        if rs is None:
            print("pyrealsense2 not available")
            return False

        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.pipeline_config = rs.config()

            # Handle device ID
            if self.device_id.startswith("realsense://"):
                # Extract device index or serial
                device_spec = self.device_id.replace("realsense://", "")
                if device_spec and device_spec != "0":
                    # Use serial number if provided
                    self.pipeline_config.enable_device(device_spec)
            elif self.device_id:
                # Direct serial number
                self.pipeline_config.enable_device(self.device_id)

            # Enable streams
            self.pipeline_config.enable_stream(
                rs.stream.color,
                camera_config.width,
                camera_config.height,
                rs.format.bgr8,
                camera_config.fps,
            )
            self.pipeline_config.enable_stream(
                rs.stream.depth,
                camera_config.width,
                camera_config.height,
                rs.format.z16,
                camera_config.fps,
            )

            # Create align object to align depth frames to color frames
            self.align = rs.align(rs.stream.color)

            # Start pipeline
            profile = self.pipeline.start(self.pipeline_config)

            # Get depth sensor's depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            self.set_intrinsics(
                CameraIntrinsics(
                    fx=color_intrinsics.fx,
                    fy=color_intrinsics.fy,
                    cx=color_intrinsics.ppx,
                    cy=color_intrinsics.ppy,
                    distortion_coeffs=np.array(color_intrinsics.coeffs),
                    width=color_intrinsics.width,
                    height=color_intrinsics.height,
                )
            )

            # Test frame capture
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print(f"Failed to capture test frames from RealSense camera {self.camera_id}")
                self.close()
                return False

            self.is_open = True
            self.current_config = camera_config
            self.frame_count = 0

            print(
                f"RealSense camera {self.camera_id} opened: {camera_config.width}x{camera_config.height}@{camera_config.fps}fps"
            )
            print(f"Depth scale: {self.depth_scale}")

            return True

        except Exception as e:
            print(f"Error opening RealSense camera {self.camera_id}: {e}")
            self.close()
            return False

    def close(self) -> None:
        """Close RealSense camera."""
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
            self.pipeline_config = None
            self.align = None
        self.is_open = False
        print(f"RealSense camera {self.camera_id} closed")

    def capture_frame(self) -> RGBFrame | RGBDFrame | None:
        """Capture RGB-D frame.

        Returns:
            RGBDFrame or None if failed
        """
        if not self.is_open or self.pipeline is None or self.align is None:
            return None

        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()

            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert depth to meters
            depth_meters = depth_image.astype(np.float32) * self.depth_scale

            # Get timestamp
            timestamp_ns = int(color_frame.get_timestamp() * 1e6)  # Convert ms to ns

            # Get metadata (if available)
            exposure_us = 0
            gain_db = 0.0

            try:
                # Try to get exposure time
                if color_frame.supports_frame_metadata(rs.frame_metadata_value.actual_exposure):
                    exposure_us = int(
                        color_frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure)
                    )

                # Try to get gain
                if color_frame.supports_frame_metadata(rs.frame_metadata_value.gain_level):
                    gain_db = float(
                        color_frame.get_frame_metadata(rs.frame_metadata_value.gain_level)
                    )
            except Exception:
                pass

            # Calculate brightness
            brightness = self.calculate_brightness(color_image)

            # Create metadata
            metadata = FrameMetadata(
                exposure_us=exposure_us,
                gain_db=gain_db,
                hardware_timestamp_ns=timestamp_ns,
                brightness=brightness,
            )

            # Create RGB-D frame
            rgbd_frame = RGBDFrame(
                color=color_image,
                depth=depth_meters,
                metadata=metadata,
                depth_scale=self.depth_scale,
                min_depth=0.1,  # RealSense typical minimum
                max_depth=10.0,  # RealSense typical maximum
            )

            self.frame_count += 1
            return rgbd_frame

        except Exception as e:
            print(f"Error capturing frame from RealSense {self.camera_id}: {e}")
            return None

    def set_exposure(self, exposure_us: int) -> bool:
        """Set camera exposure time.

        Args:
            exposure_us: Exposure time in microseconds

        Returns:
            True if successful, False otherwise
        """
        if not self.is_open or self.pipeline is None:
            return False

        try:
            # Get color sensor
            device = self.pipeline.get_active_profile().get_device()
            color_sensor = device.query_sensors()[1]  # Color sensor is typically second

            # Disable auto exposure
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)

            # Set manual exposure
            color_sensor.set_option(rs.option.exposure, exposure_us)
            return True
        except Exception:
            return False
