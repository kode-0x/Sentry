"""Tests for src/screenshot.py."""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.screenshot import capture_screenshot, cleanup_old_screenshots


class TestCaptureScreenshot:
    @patch("src.screenshot._grab_screen")
    def test_captures_and_saves(self, mock_grab, isolated_config):
        # Create a test image
        test_image = Image.new("RGB", (100, 100), color="red")
        mock_grab.return_value = test_image

        result = capture_screenshot("test-event")

        assert result is not None
        assert result.exists()
        assert result.suffix == ".png"
        assert "test-event" in result.name

    @patch("src.screenshot._grab_screen")
    def test_returns_none_on_grab_failure(self, mock_grab, isolated_config):
        mock_grab.return_value = None

        result = capture_screenshot("test-event")
        assert result is None

    @patch("src.screenshot._grab_screen")
    def test_filename_format(self, mock_grab, isolated_config):
        test_image = Image.new("RGB", (100, 100), color="blue")
        mock_grab.return_value = test_image

        result = capture_screenshot("unknown-face")

        assert result is not None
        # Format: YYYY-MM-DD_HH-MM-SS_unknown-face.png
        name = result.name
        parts = name.split("_")
        assert len(parts) >= 3
        assert name.endswith(".png")

    @patch("src.screenshot._grab_screen")
    def test_sanitizes_event_label(self, mock_grab, isolated_config):
        test_image = Image.new("RGB", (100, 100), color="green")
        mock_grab.return_value = test_image

        result = capture_screenshot("bad/label with spaces!")

        assert result is not None
        assert "/" not in result.name
        assert " " not in result.name


class TestCleanupOldScreenshots:
    def test_deletes_old_files(self, isolated_config):
        screenshot_dir = Path(isolated_config) / "screenshots"

        # Create a "old" file with modified time in the past
        old_file = screenshot_dir / "2020-01-01_00-00-00_test.png"
        old_file.write_bytes(b"fake png")
        # Set mtime to 30 days ago
        import os
        old_time = time.time() - (30 * 86400)
        os.utime(old_file, (old_time, old_time))

        deleted = cleanup_old_screenshots(retention_days=7)
        assert deleted == 1
        assert not old_file.exists()

    def test_keeps_recent_files(self, isolated_config):
        screenshot_dir = Path(isolated_config) / "screenshots"

        # Create a recent file
        recent_file = screenshot_dir / "2026-08-20_12-00-00_test.png"
        recent_file.write_bytes(b"fake png")

        deleted = cleanup_old_screenshots(retention_days=7)
        assert deleted == 0
        assert recent_file.exists()

    def test_empty_directory(self, isolated_config):
        deleted = cleanup_old_screenshots(retention_days=7)
        assert deleted == 0
