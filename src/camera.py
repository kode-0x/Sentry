"""Webcam capture abstraction for Sentry.

Provides a context-managed camera interface with frame reading,
device selection, and proper resource cleanup.
"""

from __future__ import annotations

import time
from typing import Generator

import cv2
import numpy as np

from src.config import get_config
from src.logger import get_logger


class CameraError(Exception):
    """Raised when camera operations fail."""


class Camera:
    """OpenCV-based webcam capture interface.

    Supports context management for safe resource cleanup and
    configurable frame intervals.

    Attributes:
        device: Camera device index.
        frame_interval_ms: Minimum milliseconds between frame reads.
    """

    def __init__(
        self,
        device: int | None = None,
        frame_interval_ms: int | None = None,
    ) -> None:
        """Initialize camera with config or explicit parameters.

        Args:
            device: Camera device index. Defaults to config value.
            frame_interval_ms: Frame interval in ms. Defaults to config value.
        """
        config = get_config()
        camera_config = config.get("camera", {})

        self.device = device if device is not None else camera_config.get("device", 0)
        self.frame_interval_ms = (
            frame_interval_ms
            if frame_interval_ms is not None
            else camera_config.get("frame_interval_ms", 250)
        )

        self._capture: cv2.VideoCapture | None = None
        self._logger = get_logger()
        self._last_frame_time: float = 0.0

    @property
    def is_opened(self) -> bool:
        """Check if the camera is currently opened."""
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> None:
        """Open the camera device.

        Raises:
            CameraError: If the camera cannot be opened.
        """
        if self.is_opened:
            return

        self._capture = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)

        if not self._capture.isOpened():
            # Fallback: try without specifying backend
            self._capture = cv2.VideoCapture(self.device)

        if not self._capture.isOpened():
            self._capture = None
            raise CameraError(
                f"Cannot open camera device {self.device}. "
                "Ensure a webcam is connected and not in use by another application."
            )

        # Warmup: discard initial frames (Windows DirectShow needs time)
        for _ in range(10):
            self._capture.read()
            time.sleep(0.05)

        self._logger.info(f"Camera opened on device {self.device}")

    def close(self) -> None:
        """Release the camera device and free resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            self._logger.info("Camera released")

    def read_frame(self) -> np.ndarray:
        """Read a single frame from the camera.

        Returns:
            BGR image as a NumPy array.

        Raises:
            CameraError: If the camera is not opened or frame read fails.
        """
        if not self.is_opened:
            raise CameraError("Camera is not opened. Call open() first.")

        # Retry a few times — some backends need multiple reads
        for attempt in range(3):
            ret, frame = self._capture.read()  # type: ignore[union-attr]
            if ret and frame is not None:
                return frame
            time.sleep(0.05)

        raise CameraError("Failed to read frame from camera.")

    def read_frame_throttled(self) -> np.ndarray | None:
        """Read a frame respecting the configured frame interval.

        Returns:
            BGR image as a NumPy array, or None if called too soon.

        Raises:
            CameraError: If the camera is not opened or frame read fails.
        """
        now = time.time()
        elapsed_ms = (now - self._last_frame_time) * 1000

        if elapsed_ms < self.frame_interval_ms:
            return None

        frame = self.read_frame()
        self._last_frame_time = time.time()
        return frame

    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """Generate frames from the camera at the configured interval.

        Yields:
            BGR image as a NumPy array.

        Raises:
            CameraError: If the camera is not opened.
        """
        if not self.is_opened:
            raise CameraError("Camera is not opened. Call open() first.")

        while self.is_opened:
            frame = self.read_frame_throttled()
            if frame is not None:
                yield frame
            else:
                # Brief sleep to avoid busy-waiting
                time.sleep(0.01)

    def get_resolution(self) -> tuple[int, int]:
        """Get the current camera resolution.

        Returns:
            Tuple of (width, height).

        Raises:
            CameraError: If the camera is not opened.
        """
        if not self.is_opened:
            raise CameraError("Camera is not opened.")

        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # type: ignore[union-attr]
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # type: ignore[union-attr]
        return width, height

    def __enter__(self) -> Camera:
        """Context manager entry: open the camera."""
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        """Context manager exit: release the camera."""
        self.close()
