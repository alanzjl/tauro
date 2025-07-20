#!/usr/bin/env python3
"""
Dual USB Camera Streaming Script for Raspberry Pi
Streams two USB camera feeds through Flask web interface
"""

import argparse
import logging
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template_string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


class CameraStreamer:
    def __init__(self, camera_index, width=640, height=480, fps=30, name="Camera"):
        self.camera_index = camera_index
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.frame = None
        self.is_streaming = False
        self.lock = threading.Lock()

    def initialize_camera(self):
        """Initialize the USB camera"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)

            if not self.camera.isOpened():
                logger.error(f"Failed to open {self.name} at index {self.camera_index}")
                return False

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Test frame capture
            ret, frame = self.camera.read()
            if not ret:
                logger.error(f"Failed to capture test frame from {self.name}")
                return False

            logger.info(
                f"{self.name} initialized successfully at {self.width}x{self.height}@{self.fps}fps"
            )
            return True

        except Exception as e:
            logger.error(f"Error initializing {self.name}: {e}")
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
        if self.camera:
            self.camera.release()
            logger.info(f"{self.name} released")

    def _capture_frames(self):
        """Continuously capture frames from camera"""
        while self.is_streaming:
            try:
                if self.camera is not None:
                    ret, frame = self.camera.read()
                    if ret:
                        with self.lock:
                            self.frame = frame.copy()
                    else:
                        logger.warning(f"Failed to capture frame from {self.name}")
                        time.sleep(0.1)
                else:
                    break

                # Control frame rate
                time.sleep(1.0 / self.fps)

            except Exception as e:
                logger.error(f"Error capturing frame from {self.name}: {e}")
                break

    def get_frame(self):
        """Get the latest frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def generate_frames(self):
        """Generate frames for streaming"""
        while self.is_streaming:
            frame = self.get_frame()
            if frame is not None:
                try:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                except Exception as e:
                    logger.error(f"Error encoding frame from {self.name}: {e}")

            time.sleep(1.0 / self.fps)


# Global camera streamer instances
camera1_streamer = None
camera2_streamer = None

# HTML template for the dual camera streaming page
# HTML template for the RealSense streaming page
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
html_path = os.path.join(script_dir, "static", "stream_camera.html")

HTML_TEMPLATE = open(html_path).read()


@app.route("/")
def index():
    """Main page showing both camera streams"""
    return render_template_string(
        HTML_TEMPLATE,
        width=camera1_streamer.width if camera1_streamer else 640,
        height=camera1_streamer.height if camera1_streamer else 480,
        fps=camera1_streamer.fps if camera1_streamer else 30,
        camera1_index=camera1_streamer.camera_index if camera1_streamer else 0,
        camera2_index=camera2_streamer.camera_index if camera2_streamer else 1,
    )


@app.route("/video_feed1")
def video_feed1():
    """Camera 1 video streaming route"""
    if camera1_streamer and camera1_streamer.is_streaming:
        return Response(
            camera1_streamer.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return "Camera 1 not available", 503


@app.route("/video_feed2")
def video_feed2():
    """Camera 2 video streaming route"""
    if camera2_streamer and camera2_streamer.is_streaming:
        return Response(
            camera2_streamer.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return "Camera 2 not available", 503


@app.route("/status")
def status():
    """API endpoint for both cameras status"""
    return jsonify(
        {
            "camera1": {
                "streaming": camera1_streamer.is_streaming if camera1_streamer else False,
                "camera_index": camera1_streamer.camera_index if camera1_streamer else None,
                "resolution": f"{camera1_streamer.width}x{camera1_streamer.height}"
                if camera1_streamer
                else None,
                "fps": camera1_streamer.fps if camera1_streamer else None,
            },
            "camera2": {
                "streaming": camera2_streamer.is_streaming if camera2_streamer else False,
                "camera_index": camera2_streamer.camera_index if camera2_streamer else None,
                "resolution": f"{camera2_streamer.width}x{camera2_streamer.height}"
                if camera2_streamer
                else None,
                "fps": camera2_streamer.fps if camera2_streamer else None,
            },
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dual USB Camera Streaming Server for Raspberry Pi"
    )
    parser.add_argument("--camera1", "-c1", type=int, default=0, help="Camera 1 index (default: 0)")
    parser.add_argument("--camera2", "-c2", type=int, default=1, help="Camera 2 index (default: 2)")
    parser.add_argument("--width", "-w", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument(
        "--height", "-ht", type=int, default=480, help="Frame height (default: 480)"
    )
    parser.add_argument("--fps", "-f", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=5000, help="Port to bind to (default: 5000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    global camera1_streamer, camera2_streamer

    print("🎥📹 Raspberry Pi Dual USB Camera Streaming Server")
    print("=" * 60)
    print(f"Camera 1 Index: {args.camera1}")
    print(f"Camera 2 Index: {args.camera2}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frame Rate: {args.fps} FPS")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    # Initialize camera streamers
    camera1_streamer = CameraStreamer(
        camera_index=args.camera1,
        width=args.width,
        height=args.height,
        fps=args.fps,
        name="Camera 1",
    )

    camera2_streamer = CameraStreamer(
        camera_index=args.camera2,
        width=args.width,
        height=args.height,
        fps=args.fps,
        name="Camera 2",
    )

    # Start camera streaming
    camera1_started = camera1_streamer.start_streaming()
    camera2_started = camera2_streamer.start_streaming()

    if camera1_started or camera2_started:
        print("📷 Camera Status:")
        print(f"  Camera 1: {'✅ Online' if camera1_started else '❌ Failed'}")
        print(f"  Camera 2: {'✅ Online' if camera2_started else '❌ Failed'}")
        print(f"\n🌐 Open http://localhost:{args.port} in your browser")
        print("Press Ctrl+C to stop the server")

        try:
            app.run(
                host=args.host,
                port=args.port,
                debug=args.debug,
                threaded=True,
                use_reloader=False,  # Disable reloader to prevent camera conflicts
            )
        except KeyboardInterrupt:
            print("\n🛑 Shutting down server...")
        finally:
            camera1_streamer.stop_streaming()
            camera2_streamer.stop_streaming()
            print("✅ Server stopped successfully!")
    else:
        print("❌ Failed to start any camera streaming!")
        print("Check if:")
        print("  - USB cameras are connected")
        print("  - Camera permissions are correct")
        print("  - Another application isn't using the cameras")
        print("  - Try different camera indices with --camera1 and --camera2 parameters")
        print("  - Check available cameras with: ls /dev/video*")


if __name__ == "__main__":
    main()
