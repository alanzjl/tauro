#!/usr/bin/env python3
"""
gRPC Camera Streaming Script
Streams camera feeds from the gRPC camera service through Flask web interface
"""

import argparse
import logging
import os
import threading
import time
import zlib

import cv2
import grpc
import numpy as np
from flask import Flask, Response, jsonify, render_template
from google.protobuf import empty_pb2

from tauro_common.proto import camera_service_pb2, camera_service_pb2_grpc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(script_dir, "static")

app = Flask(__name__, template_folder=template_dir)


class GRPCCameraStreamer:
    def __init__(self, camera_id, grpc_address, width=640, height=480, fps=30, name=None):
        self.camera_id = camera_id
        self.grpc_address = grpc_address
        self.name = name or f"Camera {camera_id}"
        self.width = width
        self.height = height
        self.fps = fps
        self.channel = None
        self.stub = None
        self.stream_iterator = None
        self.frame = None
        self.depth_frame = None  # For RGB-D cameras
        self.is_rgbd = False  # Track if this is an RGB-D camera
        self.is_streaming = False
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_interval = 1.0 / fps

    def connect_grpc(self):
        """Connect to gRPC camera service"""
        try:
            self.channel = grpc.insecure_channel(self.grpc_address)
            self.stub = camera_service_pb2_grpc.CameraServiceStub(self.channel)

            # Test connection by getting camera info
            try:
                response = self.stub.GetCameraInfo(empty_pb2.Empty())
                if self.camera_id not in response.cameras:
                    logger.warning(f"Camera {self.camera_id} not found in service camera list")
                    # Don't fail here, camera might still work
            except Exception as e:
                logger.warning(f"Could not get camera info: {e}")
                # Continue anyway, the stream might still work

            logger.info(f"Connected to gRPC service at {self.grpc_address}")
            return True

        except Exception as e:
            logger.error(f"Error connecting to gRPC service: {e}")
            return False

    def start_streaming(self):
        """Start camera streaming from gRPC service"""
        if not self.connect_grpc():
            return False

        self.is_streaming = True
        self.streaming_thread = threading.Thread(target=self._receive_frames)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()

        logger.info(f"{self.name} started streaming")
        return True

    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if hasattr(self, "streaming_thread"):
            self.streaming_thread.join(timeout=2)
        if self.channel:
            self.channel.close()
        logger.info(f"{self.name} stopped streaming")

    def _receive_frames(self):
        """Receive frames from gRPC stream"""
        try:
            # Create stream request
            request = camera_service_pb2.StreamRequest(
                camera_ids=[self.camera_id],
                config=camera_service_pb2.CameraConfig(
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                    jpeg_quality=85,
                    skip_frames_on_lag=True,
                ),
            )

            # Start streaming
            self.stream_iterator = self.stub.StreamImages(request)

            for frame_proto in self.stream_iterator:
                if not self.is_streaming:
                    break

                # Find our camera's image in the frame
                for image in frame_proto.images:
                    if image.camera_id == self.camera_id:
                        frame = None

                        # Decode JPEG data (regular cameras)
                        if image.HasField("jpg_data"):
                            jpeg_bytes = image.jpg_data.jpeg_data
                            logger.debug(f"Received JPEG data: {len(jpeg_bytes)} bytes")
                            # Convert to numpy array
                            nparr = np.frombuffer(jpeg_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        # Decode RGB-D data (RealSense cameras)
                        elif image.HasField("rgbd_data"):
                            self.is_rgbd = True
                            rgbd = image.rgbd_data
                            logger.debug(f"Processing RGB-D frame for {self.camera_id}")

                            # Decode color image
                            if rgbd.color_compressed:
                                logger.debug(
                                    f"RGB-D color data: {len(rgbd.color_compressed)} bytes"
                                )
                                nparr = np.frombuffer(rgbd.color_compressed, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    logger.debug(f"Decoded color frame: {frame.shape}")
                                else:
                                    logger.warning("Failed to decode color frame")
                            else:
                                logger.warning("No color data in RGB-D frame")

                            # Decode depth image
                            if rgbd.depth_compressed:
                                logger.debug(
                                    f"RGB-D depth data: {len(rgbd.depth_compressed)} bytes"
                                )
                                logger.debug(
                                    f"Depth scale: {rgbd.depth_scale}, min: {rgbd.min_depth}, max: {rgbd.max_depth}"
                                )

                                try:
                                    # Depth is compressed with zlib
                                    depth_bytes = zlib.decompress(rgbd.depth_compressed)
                                    logger.debug(
                                        f"Decompressed depth data: {len(depth_bytes)} bytes"
                                    )

                                    if frame is not None:
                                        num_pixels = frame.shape[0] * frame.shape[1]
                                        bytes_per_pixel = len(depth_bytes) // num_pixels
                                        logger.debug(
                                            f"Detected {bytes_per_pixel} bytes per pixel for depth"
                                        )

                                        # Determine depth data type based on bytes per pixel
                                        if bytes_per_pixel == 2:
                                            # 16-bit depth
                                            depth_frame = np.frombuffer(
                                                depth_bytes, dtype=np.uint16
                                            ).reshape(frame.shape[0], frame.shape[1])
                                        elif bytes_per_pixel == 4:
                                            # 32-bit float depth
                                            depth_frame = np.frombuffer(
                                                depth_bytes, dtype=np.float32
                                            ).reshape(frame.shape[0], frame.shape[1])
                                        else:
                                            logger.error(
                                                f"Unexpected bytes per pixel: {bytes_per_pixel}"
                                            )
                                            depth_frame = None

                                        if depth_frame is not None:
                                            logger.debug(
                                                f"Decoded depth frame: {depth_frame.shape}, dtype: {depth_frame.dtype}"
                                            )
                                            logger.debug(
                                                f"Depth range: min={depth_frame.min()}, max={depth_frame.max()}"
                                            )

                                            # For float32, values are already in meters; for uint16, apply scale
                                            if depth_frame.dtype == np.float32:
                                                depth_m = depth_frame
                                            else:
                                                depth_scale = (
                                                    rgbd.depth_scale
                                                    if rgbd.depth_scale > 0
                                                    else 0.001
                                                )
                                                depth_m = (
                                                    depth_frame.astype(np.float32) * depth_scale
                                                )

                                            # Log depth statistics
                                            logger.debug(
                                                f"Depth in meters: min={depth_m.min():.3f}, max={depth_m.max():.3f}"
                                            )

                                            # Clip to reasonable range (0-10 meters)
                                            depth_m_clipped = np.clip(depth_m, 0, 10)

                                            # Normalize to 0-255 for visualization
                                            depth_normalized = (depth_m_clipped / 10 * 255).astype(
                                                np.uint8
                                            )

                                            # Apply colormap
                                            depth_colormap = cv2.applyColorMap(
                                                depth_normalized, cv2.COLORMAP_JET
                                            )
                                            logger.debug(
                                                f"Generated depth colormap: {depth_colormap.shape}"
                                            )

                                            with self.lock:
                                                self.depth_frame = depth_colormap
                                    else:
                                        logger.error(
                                            "No color frame available to determine depth dimensions"
                                        )

                                except Exception as e:
                                    logger.error(f"Failed to decode depth frame: {e}")
                            else:
                                logger.warning("No depth data in RGB-D frame")

                        if frame is not None:
                            with self.lock:
                                self.frame = frame
                                self.last_frame_time = time.time()
                        break

        except grpc.RpcError as e:
            logger.error(f"gRPC error: {e}")
        except Exception as e:
            logger.error(f"Error receiving frames: {e}")
        finally:
            self.is_streaming = False

    def get_frame(self):
        """Get the latest color frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_depth_frame(self):
        """Get the latest depth frame"""
        with self.lock:
            return self.depth_frame.copy() if self.depth_frame is not None else None

    def generate_frames(self, stream_type="color"):
        """Generate frames for Flask streaming"""
        logger.debug(f"Starting frame generation for {self.camera_id} - stream_type: {stream_type}")
        frame_count = 0

        while self.is_streaming:
            if stream_type == "depth":
                frame = self.get_depth_frame()
            else:
                frame = self.get_frame()

            if frame is not None:
                frame_count += 1
                if frame_count % 300 == 0:  # Log every 300 frames (~10 seconds at 30fps)
                    logger.debug(
                        f"Generated {frame_count} {stream_type} frames for {self.camera_id}"
                    )

                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                except Exception as e:
                    logger.error(f"Error encoding {stream_type} frame: {e}")
            else:
                if frame_count == 0 and stream_type == "depth":
                    # Only warn once at the beginning for missing depth frames
                    logger.debug(f"No {stream_type} frames available for {self.camera_id}")

            # Control frame rate
            time.sleep(self.frame_interval)


# Global camera streamers
camera_streamers = {}


@app.route("/")
def index():
    """Main page showing camera streams"""
    # Build camera info list with RGB-D status
    camera_info = []
    for camera_id, streamer in camera_streamers.items():
        camera_info.append((camera_id, streamer.is_rgbd))

    return render_template(
        "grpc_stream_camera.html",
        grpc_address=list(camera_streamers.values())[0].grpc_address if camera_streamers else "N/A",
        width=list(camera_streamers.values())[0].width if camera_streamers else 640,
        height=list(camera_streamers.values())[0].height if camera_streamers else 480,
        fps=list(camera_streamers.values())[0].fps if camera_streamers else 30,
        camera_info=camera_info,
    )


@app.route("/video_feed/<camera_id>")
def video_feed(camera_id):
    """Camera video streaming route (for regular cameras)"""
    if camera_id in camera_streamers and camera_streamers[camera_id].is_streaming:
        return Response(
            camera_streamers[camera_id].generate_frames("color"),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return f"Camera {camera_id} not available", 503


@app.route("/video_feed/<camera_id>/<stream_type>")
def video_feed_typed(camera_id, stream_type):
    """Camera video streaming route with stream type (color/depth)"""
    logger.debug(f"Video feed request: camera_id={camera_id}, stream_type={stream_type}")

    if camera_id in camera_streamers:
        streamer = camera_streamers[camera_id]
        logger.debug(
            f"Camera {camera_id}: is_streaming={streamer.is_streaming}, is_rgbd={streamer.is_rgbd}"
        )

        if streamer.is_streaming:
            return Response(
                streamer.generate_frames(stream_type),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    logger.warning(f"Camera {camera_id} {stream_type} stream not available")
    return f"Camera {camera_id} {stream_type} stream not available", 503


@app.route("/status")
def status():
    """API endpoint for camera status"""
    status_data = {}
    for camera_id, streamer in camera_streamers.items():
        status_data[camera_id] = {
            "streaming": streamer.is_streaming,
            "resolution": f"{streamer.width}x{streamer.height}",
            "fps": streamer.fps,
            "is_rgbd": streamer.is_rgbd,
            "last_frame": time.time() - streamer.last_frame_time
            if streamer.last_frame_time > 0
            else None,
        }
    return jsonify(status_data)


def main():
    parser = argparse.ArgumentParser(description="gRPC Camera Streaming Server")
    parser.add_argument(
        "--cameras",
        "-c",
        nargs="+",
        default=["top", "depth"],
        help="Camera IDs to stream (default: top depth)",
    )
    parser.add_argument(
        "--grpc-address",
        "-g",
        type=str,
        default="localhost:50052",
        help="gRPC service address (default: localhost:50052)",
    )
    parser.add_argument("--width", "-w", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument(
        "--height", "-ht", type=int, default=480, help="Frame height (default: 480)"
    )
    parser.add_argument("--fps", "-f", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Flask host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5001, help="Flask port to bind to (default: 5001)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("🎥 gRPC Camera Streaming Server")
    print("=" * 60)
    print(f"gRPC Service: {args.grpc_address}")
    print(f"Cameras: {', '.join(args.cameras)}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frame Rate: {args.fps} FPS")
    print(f"Web Server: http://{args.host}:{args.port}")
    print("=" * 60)

    # Initialize camera streamers
    cameras_started = 0
    for camera_id in args.cameras:
        streamer = GRPCCameraStreamer(
            camera_id=camera_id,
            grpc_address=args.grpc_address,
            width=args.width,
            height=args.height,
            fps=args.fps,
            name=f"Camera {camera_id}",
        )

        if streamer.start_streaming():
            camera_streamers[camera_id] = streamer
            cameras_started += 1
            print(f"✅ {camera_id}: Started")
        else:
            print(f"❌ {camera_id}: Failed to start")

    if cameras_started > 0:
        print(f"\n🌐 Open http://localhost:{args.port} in your browser")
        print("Press Ctrl+C to stop the server")

        try:
            app.run(
                host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False
            )
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
        finally:
            for streamer in camera_streamers.values():
                streamer.stop_streaming()
            print("✅ Server stopped successfully!")
    else:
        print("\n❌ Failed to start any camera streams!")
        print("Check if:")
        print("  - gRPC camera service is running")
        print("  - Service address is correct")
        print("  - Camera IDs match those in sensor_ports.yaml")


if __name__ == "__main__":
    main()
