"""
Wrapper around picamera2 for simplified camera operations.
"""

import cv2
import numpy as np


class Picamera2Wrapper:
    """Simplified wrapper around Picamera2 for capture and display."""

    def __init__(self):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise ImportError(
                "picamera2 not installed (sudo apt install python3-picamera2)"
            )
        self._cam = Picamera2()
        self._configured = False
        self._current_size = None

    def start(self, width: int = 720, height: int = 576):
        """Start the camera with the specified resolution."""
        cam_config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._cam.configure(cam_config)
        self._cam.start()
        self._configured = True
        self._current_size = (width, height)

    def capture_array(self) -> np.ndarray:
        """Capture a single frame as an RGB numpy array."""
        if not self._configured:
            raise RuntimeError("Camera not started. Call start() first.")
        return self._cam.capture_array("main")  # RGB888, no extra copy

    @staticmethod
    def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
        """Resize a frame to the model input size."""
        return cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB frame to BGR for OpenCV display/drawing."""
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def show_window(self, frame_bgr: np.ndarray):
        """Display a frame in an OpenCV window (desktop debugging)."""
        cv2.imshow("preview", frame_bgr)

    @staticmethod
    def wait_key(timeout: int = 1) -> int:
        """Wait for a key press and return the key code."""
        return cv2.waitKey(timeout) & 0xFF

    @staticmethod
    def destroy_windows():
        """Destroy all OpenCV windows."""
        cv2.destroyAllWindows()

    def stop(self):
        """Stop the camera."""
        try:
            self._cam.stop()
        except Exception:
            pass
        self._configured = False
