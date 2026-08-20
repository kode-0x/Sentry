"""Tests for the MonitoringEngine in src/main.py."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.main import MonitoringEngine


def _make_engine(isolated_config, mock_cam=None, mock_detector=None, mock_recognizer=None, mock_store=None):
    """Helper to create and start a MonitoringEngine with mocks."""
    if mock_cam is None:
        mock_cam = MagicMock()
        mock_cam.read_frame_throttled.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    if mock_detector is None:
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []
    if mock_recognizer is None:
        mock_recognizer = MagicMock()
    if mock_store is None:
        mock_store = MagicMock()
        mock_store.is_enrolled = True
        mock_store.count = 5

    engine = MonitoringEngine()
    engine.start(mock_cam, mock_detector, mock_recognizer, mock_store)
    return engine


class TestMonitoringEngine:
    def test_initial_state(self, isolated_config):
        engine = MonitoringEngine()
        assert not engine.is_running
        assert engine.consecutive_unknown == 0
        assert engine.total_frames == 0
        assert engine.total_alerts == 0

    def test_start(self, isolated_config):
        engine = _make_engine(isolated_config)
        assert engine.is_running

    def test_process_frame_no_face(self, isolated_config):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = []

        engine = _make_engine(isolated_config, mock_detector=mock_detector)
        status = engine.process_frame()
        assert "No face" in status
        assert engine.consecutive_unknown == 0

    @patch("src.main.send_alert")
    @patch("src.main.capture_screenshot")
    @patch("src.main.lock_screen")
    def test_intrusion_triggers_response(
        self, mock_lock, mock_screenshot, mock_alert, isolated_config
    ):
        mock_store = MagicMock()
        mock_store.is_enrolled = True
        mock_store.count = 5
        mock_store.is_authorized.return_value = False

        mock_cam = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cam.read_frame_throttled.return_value = frame

        face_detection = np.array([10, 10, 100, 100, 0, 0, 0, 0, 0, 0, 0.95])
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [face_detection]

        mock_recognizer = MagicMock()
        mock_recognizer.get_embedding.return_value = np.random.randn(128).astype(np.float32)

        mock_screenshot.return_value = Path("/tmp/test.png")
        mock_lock.return_value = True

        engine = _make_engine(
            isolated_config,
            mock_cam=mock_cam,
            mock_detector=mock_detector,
            mock_recognizer=mock_recognizer,
            mock_store=mock_store,
        )

        # Process frames until threshold (config: required_consecutive_frames=3)
        for i in range(2):
            status = engine.process_frame()
            assert "Unknown face" in status

        # Third frame triggers response
        status = engine.process_frame()
        assert "INTRUSION" in status
        mock_lock.assert_called_once()
        mock_screenshot.assert_called_once()
        mock_alert.assert_called_once()
        assert engine.total_alerts == 1

    def test_authorized_face_resets_counter(self, isolated_config):
        mock_store = MagicMock()
        mock_store.is_enrolled = True
        mock_store.count = 5
        mock_store.is_authorized.side_effect = [False, True]

        mock_cam = MagicMock()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cam.read_frame_throttled.return_value = frame

        face_detection = np.array([10, 10, 100, 100, 0, 0, 0, 0, 0, 0, 0.95])
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [face_detection]

        mock_recognizer = MagicMock()
        mock_recognizer.get_embedding.return_value = np.random.randn(128).astype(np.float32)

        engine = _make_engine(
            isolated_config,
            mock_cam=mock_cam,
            mock_detector=mock_detector,
            mock_recognizer=mock_recognizer,
            mock_store=mock_store,
        )

        # First frame: unknown
        engine.process_frame()
        assert engine.consecutive_unknown == 1

        # Second frame: authorized - resets counter
        engine.process_frame()
        assert engine.consecutive_unknown == 0

    def test_cooldown_period(self, isolated_config):
        engine = _make_engine(isolated_config)

        # Simulate cooldown
        engine._cooldown_until = time.time() + 100

        assert engine.is_in_cooldown
        status = engine.process_frame()
        assert "Cooldown" in status

    def test_stop(self, isolated_config):
        mock_cam = MagicMock()
        engine = _make_engine(isolated_config, mock_cam=mock_cam)
        assert engine.is_running

        engine.stop()
        assert not engine.is_running
        mock_cam.close.assert_called_once()
