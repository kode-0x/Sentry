"""Tests for src/camera.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera import Camera, CameraError


class TestCamera:
    def test_not_opened_initially(self, isolated_config):
        camera = Camera()
        assert not camera.is_opened

    @patch("src.camera.cv2.VideoCapture")
    def test_open_success(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_class.return_value = mock_cap

        camera = Camera(device=0)
        camera.open()
        assert camera.is_opened

    @patch("src.camera.cv2.VideoCapture")
    def test_open_failure_raises(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cap_class.return_value = mock_cap

        camera = Camera(device=0)
        with pytest.raises(CameraError, match="Cannot open camera"):
            camera.open()

    @patch("src.camera.cv2.VideoCapture")
    def test_read_frame_success(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap_class.return_value = mock_cap

        camera = Camera(device=0)
        camera.open()
        result = camera.read_frame()
        assert result.shape == (480, 640, 3)

    @patch("src.camera.cv2.VideoCapture")
    def test_read_frame_without_open_raises(self, mock_cap_class, isolated_config):
        camera = Camera(device=0)
        with pytest.raises(CameraError, match="not opened"):
            camera.read_frame()

    @patch("src.camera.cv2.VideoCapture")
    def test_read_frame_failure_raises(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap_class.return_value = mock_cap

        camera = Camera(device=0)
        camera.open()
        with pytest.raises(CameraError, match="Failed to read frame"):
            camera.read_frame()

    @patch("src.camera.cv2.VideoCapture")
    def test_close_releases(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap_class.return_value = mock_cap

        camera = Camera(device=0)
        camera.open()
        camera.close()
        mock_cap.release.assert_called_once()
        assert not camera.is_opened

    @patch("src.camera.cv2.VideoCapture")
    def test_context_manager(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap_class.return_value = mock_cap

        with Camera(device=0) as camera:
            assert camera.is_opened
        mock_cap.release.assert_called_once()

    @patch("src.camera.cv2.VideoCapture")
    def test_read_frame_throttled_respects_interval(self, mock_cap_class, isolated_config):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cap_class.return_value = mock_cap

        camera = Camera(device=0, frame_interval_ms=1000)
        camera.open()

        # First call should succeed
        result1 = camera.read_frame_throttled()
        assert result1 is not None

        # Immediate second call should return None (throttled)
        result2 = camera.read_frame_throttled()
        assert result2 is None
