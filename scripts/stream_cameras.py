#!/usr/bin/env python3
"""
Multi-Camera Streaming Script
Streams any number of USB camera feeds through Flask web interface
"""

import argparse
import logging
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


# Global camera streamers dictionary
camera_streamers: dict[int, CameraStreamer] = {}

# HTML template for multi-camera streaming
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Multi-Camera Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
            text-align: center;
        }
        .cameras-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .camera-section {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            background-color: #fafafa;
        }
        .camera-title {
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }
        .camera-container {
            border: 2px solid #ddd;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            background-color: #f0f0f0;
        }
        .camera-stream {
            width: 100%;
            height: auto;
            display: block;
        }
        .camera-overlay {
            position: absolute;
            top: 10px;
            left: 10px;
            background-color: rgba(0,0,0,0.7);
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
        }
        .status {
            margin: 10px 0;
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
        }
        .status.online {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.offline {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .controls {
            text-align: center;
            margin: 20px 0;
        }
        button {
            background-color: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        .info {
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .camera-info {
            margin: 10px 0;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        @media (max-width: 768px) {
            .cameras-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    <script>
        function refreshPage() {
            location.reload();
        }

        function checkStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    Object.keys(data).forEach(cameraId => {
                        const status = document.getElementById(`status${cameraId}`);
                        if (status) {
                            if (data[cameraId].streaming) {
                                status.className = 'status online';
                                status.textContent = `${data[cameraId].name}: Online`;
                            } else {
                                status.className = 'status offline';
                                status.textContent = `${data[cameraId].name}: Offline`;
                            }
                        }
                    });
                })
                .catch(error => {
                    document.querySelectorAll('.status').forEach(status => {
                        status.className = 'status offline';
                        status.textContent = 'Error checking status';
                    });
                });
        }

        // Check status every 5 seconds
        setInterval(checkStatus, 5000);
        checkStatus(); // Initial check
    </script>
</head>
<body>
    <div class="container">
        <h1>📷 Multi-Camera Stream ({{ num_cameras }} camera{{ 's' if num_cameras != 1 else '' }})</h1>

        <div class="cameras-grid">
            {% for camera_id, camera in cameras.items() %}
            <div class="camera-section">
                <div class="camera-title">📷 {{ camera.name }} (Index: {{ camera.camera_index }})</div>
                <div id="status{{ camera_id }}" class="status">Checking camera status...</div>
                <div class="camera-container">
                    <img src="/video_feed/{{ camera_id }}" class="camera-stream" alt="{{ camera.name }} Feed"
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjQwIiBoZWlnaHQ9IjQ4MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZGRkIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCwgSGVsdmV0aWNhLCBzYW5zLXNlcmlmIiBmb250LXNpemU9IjE4IiBmaWxsPSIjOTk5IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBkeT0iLjNlbSI+Q2FtZXJhIE5vdCBBdmFpbGFibGU8L3RleHQ+PC9zdmc+';">
                    <div class="camera-overlay">{{ camera.name }}</div>
                </div>
            </div>
            {% endfor %}
        </div>

        <div class="controls">
            <button onclick="refreshPage()">🔄 Refresh Streams</button>
        </div>

        <div class="info">
            <h3>📋 Stream Information</h3>
            {% for camera_id, camera in cameras.items() %}
            <div class="camera-info">
                <h4>{{ camera.name }}</h4>
                <p><strong>Camera Index:</strong> {{ camera.camera_index }} | <strong>Resolution:</strong> {{ camera.width }}x{{ camera.height }} | <strong>Frame Rate:</strong> {{ camera.fps }} FPS | <strong>Stream URL:</strong> <code>/video_feed/{{ camera_id }}</code></p>
            </div>
            {% endfor %}
            <p><strong>Server:</strong> <code>http://{{ host }}:{{ port }}</code></p>
        </div>
    </div>
</body>
</html>
"""


@app.route("/")
def index():
    """Main page showing all camera streams"""
    host = app.config.get("host", "localhost")
    port = app.config.get("port", 5000)
    return render_template_string(
        HTML_TEMPLATE,
        cameras=camera_streamers,
        num_cameras=len(camera_streamers),
        host=host,
        port=port,
    )


@app.route("/video_feed/<int:camera_id>")
def video_feed(camera_id):
    """Video streaming route for specific camera"""
    if camera_id in camera_streamers and camera_streamers[camera_id].is_streaming:
        return Response(
            camera_streamers[camera_id].generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    else:
        return f"Camera {camera_id} not available", 503


@app.route("/status")
def status():
    """API endpoint for all cameras status"""
    status_data = {}
    for camera_id, streamer in camera_streamers.items():
        status_data[camera_id] = {
            "name": streamer.name,
            "streaming": streamer.is_streaming,
            "camera_index": streamer.camera_index,
            "resolution": f"{streamer.width}x{streamer.height}",
            "fps": streamer.fps,
        }
    return jsonify(status_data)


def detect_all_cameras() -> list[int]:
    """Detect all available camera indices by checking /dev/video* devices"""
    available_cameras = []

    # Method 1: Check /dev/video* devices (Linux)
    import glob
    import platform

    if platform.system() == "Linux":
        video_devices = glob.glob("/dev/video*")

        for device in sorted(video_devices):
            try:
                # Extract the index from /dev/video0, /dev/video1, etc.
                index = int(device.replace("/dev/video", ""))

                # Test if we can actually open and read from this device
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()

                    if ret:
                        available_cameras.append(index)
                        logger.info(f"Found camera at {device} (index {index})")

            except ValueError:
                logger.warning(f"Could not parse device {device}")
                continue

    else:
        # Fallback method for non-Linux systems: probe first 10 indices
        logger.info("Non-Linux system detected, using probe method")
        consecutive_failures = 0

        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()

                if ret:
                    available_cameras.append(i)
                    consecutive_failures = 0
                    logger.info(f"Found camera at index {i}")
                else:
                    consecutive_failures += 1
            else:
                consecutive_failures += 1

            # Stop after 3 consecutive failures
            if consecutive_failures >= 3:
                break

    return sorted(available_cameras)


def parse_camera_indices(indices_str: str) -> list[int]:
    """Parse camera indices from string (e.g., '0,1,2' or '0-3' or '0,2-4')"""
    indices = []
    parts = indices_str.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            # Handle range like '0-3'
            start, end = part.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            # Single index
            indices.append(int(part))

    return sorted(set(indices))  # Remove duplicates and sort


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Camera Streaming Server",
        epilog="Examples:\n"
        "  %(prog)s                       # Auto-detect and stream all cameras\n"
        "  %(prog)s --cameras auto        # Auto-detect and stream all cameras\n"
        "  %(prog)s --cameras 0,1,2      # Stream cameras at indices 0, 1, and 2\n"
        "  %(prog)s --cameras 0-3        # Stream cameras at indices 0, 1, 2, and 3\n"
        "  %(prog)s --cameras 0,2-4      # Stream cameras at indices 0, 2, 3, and 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cameras",
        "-c",
        type=str,
        default="auto",
        help="Camera indices to use (e.g., '0,1,2' or '0-3' or '0,2-4' or 'auto' to detect all)",
    )
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

    # Parse or detect camera indices
    if args.cameras.lower() == "auto":
        print("🔍 Auto-detecting available cameras...")
        camera_indices = detect_all_cameras()
        if not camera_indices:
            print("❌ No cameras detected!")
            print("Check if:")
            print("  - USB cameras are connected")
            print("  - Camera permissions are correct")
            print("  - Try specifying camera indices manually")
            print("  - Check available devices with: ls /dev/video*")
            return
        print(f"✅ Found {len(camera_indices)} camera(s) at indices: {camera_indices}")
    else:
        camera_indices = parse_camera_indices(args.cameras)

    print("\n📷 Multi-Camera Streaming Server")
    print("=" * 60)
    print(f"Camera Indices: {camera_indices}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Frame Rate: {args.fps} FPS")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    # Store config in app for template access
    app.config["host"] = args.host
    app.config["port"] = args.port

    # Initialize camera streamers
    successful_cameras = 0
    for idx, camera_index in enumerate(camera_indices):
        camera_name = f"Camera {idx + 1}"
        streamer = CameraStreamer(
            camera_index=camera_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            name=camera_name,
        )
        camera_streamers[idx] = streamer

        # Start streaming
        if streamer.start_streaming():
            successful_cameras += 1
            print(f"  {camera_name} (index {camera_index}): ✅ Online")
        else:
            print(f"  {camera_name} (index {camera_index}): ❌ Failed")

    if successful_cameras > 0:
        print(f"\n📷 Successfully started {successful_cameras}/{len(camera_indices)} cameras")
        print(f"🌐 Open http://localhost:{args.port} in your browser")
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
            for streamer in camera_streamers.values():
                streamer.stop_streaming()
            print("✅ Server stopped successfully!")
    else:
        print("❌ Failed to start any camera streaming!")
        print("Check if:")
        print("  - USB cameras are connected")
        print("  - Camera permissions are correct")
        print("  - Another application isn't using the cameras")
        print("  - Try different camera indices")
        print("  - Check available cameras with: ls /dev/video*")


if __name__ == "__main__":
    main()
