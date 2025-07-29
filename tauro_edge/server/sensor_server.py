"""Sensor service gRPC server implementation."""

import asyncio
import time
from concurrent import futures
from pathlib import Path
from typing import Any

import google.protobuf.timestamp_pb2 as timestamp_pb2
import grpc
import yaml

from tauro_common.proto import camera_service_pb2, camera_service_pb2_grpc
from tauro_edge.sensors.camera_base import CameraBase, CameraConfig, CameraType
from tauro_edge.sensors.realsense_camera import RealSenseCamera
from tauro_edge.sensors.usb_camera import USBCamera


class CameraServicer(camera_service_pb2_grpc.CameraServiceServicer):
    """Camera service implementation."""

    def __init__(self, config_path: str = "tauro_edge/configs/sensor_ports.yaml"):
        """Initialize camera service.

        Args:
            config_path: Path to sensor configuration file
        """
        self.config_path = Path(config_path)
        self.cameras: dict[str, CameraBase] = {}
        self.active_streams: dict[str, asyncio.Task] = {}
        self.camera_configs: dict[str, dict[str, Any]] = {}
        self.frame_counter = 0

        # Load camera configurations
        self._load_configurations()

    def _load_configurations(self):
        """Load camera configurations from YAML file."""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f)
                self.camera_configs = config.get("cameras", {})
                print(f"Loaded {len(self.camera_configs)} camera configurations")
        except Exception as e:
            print(f"Error loading camera configurations: {e}")
            self.camera_configs = {}

    def _create_camera(self, camera_id: str) -> CameraBase | None:
        """Create camera instance based on configuration.

        Args:
            camera_id: Camera ID from configuration

        Returns:
            Camera instance or None if failed
        """
        if camera_id not in self.camera_configs:
            print(f"Camera {camera_id} not found in configuration")
            return None

        config = self.camera_configs[camera_id]
        device_id = config.get("device_id", "")
        camera_type = config.get("type", "usb")

        try:
            if camera_type == "realsense":
                return RealSenseCamera(camera_id, device_id, config)
            else:  # Default to USB camera
                return USBCamera(camera_id, device_id, config)
        except Exception as e:
            print(f"Error creating camera {camera_id}: {e}")
            return None

    def _get_or_create_camera(self, camera_id: str) -> CameraBase | None:
        """Get existing camera or create new one.

        Args:
            camera_id: Camera ID

        Returns:
            Camera instance or None if failed
        """
        if camera_id not in self.cameras:
            camera = self._create_camera(camera_id)
            if camera:
                self.cameras[camera_id] = camera
        return self.cameras.get(camera_id)

    def StreamImages(self, request, context):
        """Stream images from specified cameras."""
        camera_ids = request.camera_ids
        if not camera_ids:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No camera IDs specified")

        # Convert proto config to CameraConfig
        camera_config = CameraConfig(
            width=request.config.width or 640,
            height=request.config.height or 480,
            fps=request.config.fps or 30,
            jpeg_quality=request.config.jpeg_quality or 85,
            skip_frames_on_lag=request.config.skip_frames_on_lag,
        )

        # Open cameras
        active_cameras = []
        for camera_id in camera_ids:
            camera = self._get_or_create_camera(camera_id)
            if camera and camera.open(camera_config):
                active_cameras.append(camera)
            else:
                # Close already opened cameras
                for cam in active_cameras:
                    cam.close()
                context.abort(grpc.StatusCode.UNAVAILABLE, f"Failed to open camera {camera_id}")

        try:
            while context.is_active():
                # Capture frames from all cameras
                image_protos = []

                for camera in active_cameras:
                    # Check FPS throttling
                    if camera.should_skip_frame(camera_config.fps):
                        continue

                    # Capture frame
                    frame = camera.capture_frame()
                    if frame is None:
                        continue

                    # Compress frame based on camera type
                    if camera.get_camera_type() == CameraType.RGB:
                        compressed_data = camera.compress_rgb_frame(
                            frame, camera_config.jpeg_quality
                        )
                    else:  # RGB-D
                        compressed_data = camera.compress_rgbd_frame(
                            frame, camera_config.jpeg_quality
                        )

                    # Create image proto
                    image_proto = camera_service_pb2.CameraImage(
                        camera_id=camera.camera_id,
                        width=camera_config.width,
                        height=camera_config.height,
                        metadata=camera_service_pb2.FrameMetadata(
                            exposure_us=frame.metadata.exposure_us,
                            gain_db=frame.metadata.gain_db,
                            hardware_timestamp_ns=frame.metadata.hardware_timestamp_ns,
                            brightness=frame.metadata.brightness,
                        ),
                        status=camera_service_pb2.CameraStatus.CAMERA_STATUS_OK,
                    )

                    # Set compressed data
                    if "jpg_data" in compressed_data:
                        image_proto.jpg_data.CopyFrom(
                            camera_service_pb2.CompressedJPGData(**compressed_data["jpg_data"])
                        )
                    elif "rgbd_data" in compressed_data:
                        image_proto.rgbd_data.CopyFrom(
                            camera_service_pb2.CompressedRGBDData(**compressed_data["rgbd_data"])
                        )

                    # Set intrinsics if available
                    intrinsics = camera.get_intrinsics()
                    if intrinsics:
                        intrinsics_dict = intrinsics.to_dict()
                        image_proto.intrinsics.CopyFrom(
                            camera_service_pb2.CameraIntrinsics(**intrinsics_dict)
                        )

                    image_protos.append(image_proto)

                # Send frame if we have images
                if image_protos:
                    # Create timestamp
                    timestamp = timestamp_pb2.Timestamp()
                    timestamp.GetCurrentTime()

                    # Create and yield frame
                    frame_proto = camera_service_pb2.ImageFrame(
                        timestamp=timestamp, images=image_protos, frame_id=self.frame_counter
                    )
                    self.frame_counter += 1
                    yield frame_proto

                # Small delay to prevent CPU spinning
                time.sleep(0.001)

        finally:
            # Clean up cameras
            for camera in active_cameras:
                camera.close()

    def CaptureFrame(self, request, context):
        """Capture a single frame from specified cameras."""
        camera_ids = request.camera_ids
        if not camera_ids:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No camera IDs specified")

        # Convert proto config to CameraConfig
        camera_config = CameraConfig(
            width=request.config.width or 640,
            height=request.config.height or 480,
            fps=30,  # Not used for single capture
            jpeg_quality=request.config.jpeg_quality or 85,
            skip_frames_on_lag=False,
        )

        image_protos = []

        for camera_id in camera_ids:
            camera = self._get_or_create_camera(camera_id)
            if not camera:
                continue

            # Open camera if not already open
            was_open = camera.is_open
            if not was_open and not camera.open(camera_config):
                continue

            try:
                # Capture frame
                frame = camera.capture_frame()
                if frame is None:
                    continue

                # Compress frame
                if camera.get_camera_type() == CameraType.RGB:
                    compressed_data = camera.compress_rgb_frame(frame, camera_config.jpeg_quality)
                else:  # RGB-D
                    compressed_data = camera.compress_rgbd_frame(frame, camera_config.jpeg_quality)

                # Create image proto
                image_proto = camera_service_pb2.CameraImage(
                    camera_id=camera.camera_id,
                    width=camera_config.width,
                    height=camera_config.height,
                    metadata=camera_service_pb2.FrameMetadata(
                        exposure_us=frame.metadata.exposure_us,
                        gain_db=frame.metadata.gain_db,
                        hardware_timestamp_ns=frame.metadata.hardware_timestamp_ns,
                        brightness=frame.metadata.brightness,
                    ),
                    status=camera_service_pb2.CameraStatus.CAMERA_STATUS_OK,
                )

                # Set compressed data
                if "jpg_data" in compressed_data:
                    image_proto.jpg_data.CopyFrom(
                        camera_service_pb2.CompressedJPGData(**compressed_data["jpg_data"])
                    )
                elif "rgbd_data" in compressed_data:
                    image_proto.rgbd_data.CopyFrom(
                        camera_service_pb2.CompressedRGBDData(**compressed_data["rgbd_data"])
                    )

                # Set intrinsics if available
                intrinsics = camera.get_intrinsics()
                if intrinsics:
                    intrinsics_dict = intrinsics.to_dict()
                    image_proto.intrinsics.CopyFrom(
                        camera_service_pb2.CameraIntrinsics(**intrinsics_dict)
                    )

                image_protos.append(image_proto)

            finally:
                # Close camera if it wasn't open before
                if not was_open:
                    camera.close()

        # Create timestamp
        timestamp = timestamp_pb2.Timestamp()
        timestamp.GetCurrentTime()

        # Return frame
        return camera_service_pb2.ImageFrame(
            timestamp=timestamp, images=image_protos, frame_id=self.frame_counter
        )

    def UpdateStreamParameters(self, request, context):
        """Update stream parameters for active cameras."""
        # This would require tracking active streams and updating their parameters
        # For now, return not implemented
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "UpdateStreamParameters not implemented")

    def GetCameraInfo(self, request, context):
        """Get information about available cameras."""
        camera_details = {}

        for camera_id, config in self.camera_configs.items():
            # Create camera details
            details = camera_service_pb2.CameraDetails(
                device_id=config.get("device_id", ""),
                type=config.get("type", "usb"),
                status=camera_service_pb2.CameraStatus.CAMERA_STATUS_OK,
                properties=config,
            )

            # Add supported resolutions
            if "resolution" in config:
                res_str = config["resolution"]
                try:
                    width, height = map(int, res_str.split("x"))
                    resolution = camera_service_pb2.Resolution(
                        width=width, height=height, fps_options=[15, 30, 60]
                    )
                    details.supported_resolutions.append(resolution)
                except Exception:
                    pass

            # Try to get intrinsics if camera is available
            camera = self._get_or_create_camera(camera_id)
            if camera and camera.get_intrinsics():
                intrinsics_dict = camera.get_intrinsics().to_dict()
                details.intrinsics.CopyFrom(camera_service_pb2.CameraIntrinsics(**intrinsics_dict))

            camera_details[camera_id] = details

        return camera_service_pb2.CameraInfoResponse(cameras=camera_details)

    def StopAllStreams(self, request, context):
        """Stop all active camera streams."""
        streams_stopped = 0

        # Close all open cameras
        for camera in self.cameras.values():
            if camera.is_open:
                camera.close()
                streams_stopped += 1

        # Clear camera instances
        self.cameras.clear()

        return camera_service_pb2.StopStreamResponse(
            success=True,
            message=f"Stopped {streams_stopped} camera streams",
            streams_stopped=streams_stopped,
        )


def serve(host: str = "0.0.0.0", port: int = 50052, config_path: str | None = None):
    """Start the camera service gRPC server.

    Args:
        host: Host to bind to
        port: Port to bind to
        config_path: Path to sensor configuration file
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Create servicer with config path
    if config_path:
        servicer = CameraServicer(config_path)
    else:
        servicer = CameraServicer()

    camera_service_pb2_grpc.add_CameraServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"{host}:{port}")
    server.start()

    print(f"Camera service started on {host}:{port}")
    print("Press Ctrl+C to stop...")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\nShutting down camera service...")
        server.stop(0)
        print("Camera service stopped")


if __name__ == "__main__":
    serve()
