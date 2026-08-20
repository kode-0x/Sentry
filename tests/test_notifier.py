"""Tests for src/notifier.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.notifier import NotificationError, send_alert


class TestSendAlert:
    def test_no_channels_enabled(self, isolated_config):
        """When no channels are enabled, returns empty dict."""
        results = send_alert("unknown_face", "Test alert")
        assert results == {}

    @patch("src.notifier._send_email")
    def test_email_channel_success(self, mock_email, isolated_config, monkeypatch):
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        # Enable email in config
        import src.config
        config = src.config.get_config()
        config["notifications"]["email"] = True

        results = send_alert("unknown_face", "Test alert")
        assert results["email"] is True
        mock_email.assert_called_once()

    @patch("src.notifier._send_email", side_effect=NotificationError("SMTP error"))
    def test_email_channel_failure(self, mock_email, isolated_config, monkeypatch):
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        import src.config
        config = src.config.get_config()
        config["notifications"]["email"] = True

        results = send_alert("unknown_face", "Test alert")
        assert results["email"] is False

    @patch("src.notifier._send_whatsapp")
    def test_whatsapp_channel_success(self, mock_wa, isolated_config, monkeypatch):
        monkeypatch.setenv("WHATSAPP_ENABLED", "true")
        import src.config
        config = src.config.get_config()
        config["notifications"]["whatsapp"] = True

        results = send_alert("unknown_face", "Test alert")
        assert results["whatsapp"] is True
        mock_wa.assert_called_once()

    @patch("src.notifier._send_whatsapp", side_effect=NotificationError("API error"))
    def test_whatsapp_channel_failure(self, mock_wa, isolated_config, monkeypatch):
        monkeypatch.setenv("WHATSAPP_ENABLED", "true")
        import src.config
        config = src.config.get_config()
        config["notifications"]["whatsapp"] = True

        results = send_alert("unknown_face", "Test alert")
        assert results["whatsapp"] is False

    @patch("src.notifier._send_email")
    @patch("src.notifier._send_whatsapp", side_effect=NotificationError("failed"))
    def test_channels_independent(self, mock_wa, mock_email, isolated_config, monkeypatch):
        """One channel failing doesn't prevent the other from sending."""
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        monkeypatch.setenv("WHATSAPP_ENABLED", "true")
        import src.config
        config = src.config.get_config()
        config["notifications"]["email"] = True
        config["notifications"]["whatsapp"] = True

        results = send_alert("unknown_face", "Test alert")
        assert results["email"] is True
        assert results["whatsapp"] is False

    @patch("src.notifier._send_email")
    def test_screenshot_path_passed(self, mock_email, isolated_config, monkeypatch, tmp_path):
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        import src.config
        config = src.config.get_config()
        config["notifications"]["email"] = True

        screenshot = tmp_path / "test.png"
        screenshot.write_bytes(b"fake png data")

        send_alert("unknown_face", "Test", screenshot_path=screenshot)
        # Verify screenshot_path was passed to _send_email
        call_args = mock_email.call_args
        assert call_args[0][2] == screenshot
