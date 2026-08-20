"""Tests for src/screen_lock.py."""

import platform
from unittest.mock import MagicMock, patch

import pytest

from src.screen_lock import ScreenLockError, is_lock_available, lock_screen


class TestLockScreen:
    @patch("src.screen_lock.platform.system", return_value="Windows")
    @patch("src.screen_lock._lock_windows")
    def test_lock_windows_success(self, mock_lock, mock_sys, isolated_config):
        result = lock_screen()
        assert result is True
        mock_lock.assert_called_once()

    @patch("src.screen_lock.platform.system", return_value="Darwin")
    @patch("src.screen_lock._lock_macos")
    def test_lock_macos_success(self, mock_lock, mock_sys, isolated_config):
        result = lock_screen()
        assert result is True
        mock_lock.assert_called_once()

    @patch("src.screen_lock.platform.system", return_value="Linux")
    @patch("src.screen_lock._lock_linux")
    def test_lock_linux_success(self, mock_lock, mock_sys, isolated_config):
        result = lock_screen()
        assert result is True
        mock_lock.assert_called_once()

    @patch("src.screen_lock.platform.system", return_value="Windows")
    @patch("src.screen_lock._lock_windows", side_effect=ScreenLockError("failed"))
    def test_lock_failure_returns_false(self, mock_lock, mock_sys, isolated_config):
        result = lock_screen()
        assert result is False

    @patch("src.screen_lock.platform.system", return_value="UnknownOS")
    def test_unsupported_platform_returns_false(self, mock_sys, isolated_config):
        result = lock_screen()
        assert result is False

    @patch("src.screen_lock.platform.system", return_value="Windows")
    @patch("src.screen_lock._lock_windows", side_effect=RuntimeError("unexpected"))
    def test_unexpected_error_returns_false(self, mock_lock, mock_sys, isolated_config):
        result = lock_screen()
        assert result is False


class TestIsLockAvailable:
    @patch("src.screen_lock.platform.system", return_value="Windows")
    @patch("src.screen_lock.sys.platform", "win32")
    def test_windows_available(self, mock_sys):
        assert is_lock_available() is True

    @patch("src.screen_lock.platform.system", return_value="Darwin")
    def test_macos_available(self, mock_sys):
        assert is_lock_available() is True
