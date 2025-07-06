#!/usr/bin/env python3
"""
RealSense Camera Streaming Script for Raspberry Pi
Streams color and depth feeds from RealSense camera through Flask web interface
"""

import argparse
import logging
import os
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, jsonify, render_template_string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class RealSenseStreamer:
    def __init__(self, width=640, height=480, fps=30, depth_colormap=cv2.COLORMAP_JET):
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_colormap = depth_colormap
        self.pipeline = None
        self.config = None
        self.align = None
        self.color_frame = None
        self.depth_frame = None
        self.is_streaming = False
        self.lock = threading.Lock()

    def initialize_camera(self):
        """Initialize the RealSense camera"""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure streams
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            self.config.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

            # Create align object to align depth frames to color frames
            self.align = rs.align(rs.stream.color)

            # Start pipeline
            profile = self.pipeline.start(self.config)

            # Get depth sensor's depth scale
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # Test frame capture
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                logger.error("Failed to capture test frames from RealSense camera")
                return False

            logger.info(
                f"RealSense camera initialized successfully at"
                f"{self.width}x{self.height}@{self.fps}fps"
            )
            logger.info(f"Depth scale: {self.depth_scale}")
            return True

        except Exception as e:
            logger.error(f"Error initializing RealSense camera: {e}")
            return False

    def start_streaming(self):
        """Start the camera streaming in a separate thread"""
        if self.initialize_camera():
            self.is_streaming = True
            self.streaming_thread = threading.Thread(target=self._capture_frames)
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
            return True
        return False

    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if hasattr(self, "streaming_thread"):
            self.streaming_thread.join(timeout=2)
        if self.pipeline:
            self.pipeline.stop()
            logger.info("RealSense camera released")

    def _capture_frames(self):
        """Continuously capture frames from RealSense camera"""
        while self.is_streaming:
            try:
                if self.pipeline is not None:
                    # Wait for frames
                    frames = self.pipeline.wait_for_frames()

                    # Align depth frame to color frame
                    aligned_frames = self.align.process(frames)

                    # Get frames
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if color_frame and depth_frame:
                        # Convert to numpy arrays
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data())

                        # Apply colormap to depth image for visualization
                        depth_colormap = cv2.applyColorMap(
                            cv2.convertScaleAbs(depth_image, alpha=0.03),
                            self.depth_colormap,
                        )

                        with self.lock:
                            self.color_frame = color_image.copy()
                            self.depth_frame = depth_colormap.copy()
                    else:
                        logger.warning("Failed to capture frames from RealSense camera")
                        time.sleep(0.1)
                else:
                    break

            except Exception as e:
                logger.error(f"Error capturing frames from RealSense camera: {e}")
                break

    def get_color_frame(self):
        """Get the latest color frame"""
        with self.lock:
            return self.color_frame.copy() if self.color_frame is not None else None

    def get_depth_frame(self):
        """Get the latest depth frame (colorized)"""
        with self.lock:
            return self.depth_frame.copy() if self.depth_frame is not None else None

    def generate_color_frames(self):
        """Generate color frames for streaming"""
        while self.is_streaming:
            frame = self.get_color_frame()
            if frame is not None:
                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                except Exception as e:
                    logger.error(f"Error encoding color frame: {e}")

            time.sleep(1.0 / self.fps)

    def generate_depth_frames(self):
        """Generate depth frames for streaming"""
        while self.is_streaming:
            frame = self.get_depth_frame()
            if frame is not None:
                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                except Exception as e:
                    logger.error(f"Error encoding depth frame: {e}")

            time.sleep(1.0 / self.fps)


# Global RealSense streamer instance
realsense_streamer = None

# HTML template for the RealSense streaming page
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(script_dir, "static", "stream_realsense.html")

HTML_TEMPLATE = open(html_path).read()


@app.route("/")
def index():
    """Main page showing both color and depth streams"""
    colormap_name = {
        cv2.COLORMAP_JET: "JET",
        cv2.COLORMAP_HOT: "HOT",
        cv2.COLORMAP_RAINBOW: "RAINBOW",
        cv2.COLORMAP_COOL: "COOL",
        cv2.COLORMAP_VIRIDIS: "VIRIDIS",
    }.get(
        realsense_streamer.depth_colormap if realsense_streamer else cv2.COLORMAP_JET,
        "JET",
    )

    return render_template_string(
        HTML_TEMPLATE,
        width=realsense_streamer.width if realsense_streamer else 640,
        height=realsense_streamer.height if realsense_streamer else 480,
        fps=realsense_streamer.fps if realsense_streamer else 30,
        colormap=colormap_name,
    )


@app.route("/color_feed")
def color_feed():
    """Color video streaming route"""
    if realsense_streamer and realsense_streamer.is_streaming:
        return Response(
            realsense_streamer.generate_color_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return "Color feed not available", 503


@app.route("/depth_feed")
def depth_feed():
    """Depth video streaming route"""
    if realsense_streamer and realsense_streamer.is_streaming:
        return Response(
            realsense_streamer.generate_depth_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return "Depth feed not available", 503


@app.route("/status")
def status():
    """API endpoint for camera status"""
    return jsonify(
        {
            "streaming": realsense_streamer.is_streaming
            if realsense_streamer
            else False,
            "resolution": f"{realsense_streamer.width}x{realsense_streamer.height}"
            if realsense_streamer
            else None,
            "fps": realsense_streamer.fps if realsense_streamer else None,
            "depth_scale": realsense_streamer.depth_scale
            if realsense_streamer and hasattr(realsense_streamer, "depth_scale")
            else None,
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="RealSense Camera Streaming Server for Raspberry Pi"
    )
    parser.add_argument(
        "--width", "-w", type=int, default=640, help="Frame width (default: 640)"
    )
    parser.add_argument(
        "--height", "-ht", type=int, default=480, help="Frame height (default: 480)"
    )
    parser.add_argument(
        "--fps", "-f", type=int, default=30, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--colormap",
        "-c",
        choices=["jet", "hot", "rainbow", "cool", "viridis"],
        default="jet",
        help="Depth colormap (default: jet)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5000, help="Port to bind to (default: 5000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Map colormap names to OpenCV constants
    colormap_dict = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "cool": cv2.COLORMAP_COOL,
        "viridis": cv2.COLORMAP_VIRIDIS,
    }

    global realsense_streamer

    print("📷🔍 Raspberry Pi RealSense Camera Streaming Server")
    print("=" * 60)
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frame Rate: {args.fps} FPS")
    print(f"Depth Colormap: {args.colormap.upper()}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    # Initialize RealSense streamer
    realsense_streamer = RealSenseStreamer(
        width=args.width,
        height=args.height,
        fps=args.fps,
        depth_colormap=colormap_dict[args.colormap],
    )

    # Start streaming
    if realsense_streamer.start_streaming():
        print("📷 RealSense Camera: ✅ Online")
        print(f"\n🌐 Open http://localhost:{args.port} in your browser")
        print("Press Ctrl+C to stop the server")

        try:
            app.run(
                host=args.host,
                port=args.port,
                debug=args.debug,
                threaded=True,
                use_reloader=False,
            )
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
        finally:
            realsense_streamer.stop_streaming()
            print("✅ Server stopped successfully!")
    else:
        print("❌ Failed to start RealSense camera streaming!")
        print("Check if:")
        print("  - RealSense camera is connected")
        print("  - RealSense drivers are installed")
        print("  - Camera permissions are correct")
        print("  - Another application isn't using the camera")
        print("  - Install pyrealsense2: pip install pyrealsense2")


if __name__ == "__main__":
    main()
