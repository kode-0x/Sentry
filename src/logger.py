"""Structured event logging for Sentry.

Provides file-based and console logging with timestamps,
log rotation, and security event tracking.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_config, resolve_path


# Security event types
EVENT_UNKNOWN_FACE = "unknown_face"
EVENT_FACE_ENROLLED = "face_enrolled"
EVENT_SCREEN_LOCKED = "screen_locked"
EVENT_SCREENSHOT_TAKEN = "screenshot_taken"
EVENT_ALERT_SENT = "alert_sent"
EVENT_ALERT_FAILED = "alert_failed"
EVENT_MONITORING_STARTED = "monitoring_started"
EVENT_MONITORING_STOPPED = "monitoring_stopped"
EVENT_COOLDOWN_STARTED = "cooldown_started"
EVENT_COOLDOWN_ENDED = "cooldown_ended"


class SecurityEventFormatter(logging.Formatter):
    """Custom formatter that outputs structured log lines."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        base = f"[{timestamp}] [{record.levelname}] {record.getMessage()}"

        # Append extra structured data if present
        if hasattr(record, "event_data") and record.event_data:
            base += f" | {json.dumps(record.event_data)}"

        return base


class EventLogger:
    """Sentry event logger with file rotation and console output.

    Attributes:
        logger: The underlying Python logger instance.
        log_dir: Directory where log files are stored.
    """

    def __init__(self, name: str = "sentry") -> None:
        config = get_config()
        storage = config.get("storage", {})
        log_dir_str = storage.get("log_directory", "./logs")
        self.log_dir = resolve_path(log_dir_str)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Clear existing handlers to support re-initialization in tests
        # and prevent stale file handles to old directories
        self.logger.handlers.clear()
        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Configure file and console handlers."""
        formatter = SecurityEventFormatter()

        # Rotating file handler: 5MB per file, keep 5 backups
        log_file = self.log_dir / "sentry.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an informational message."""
        self.logger.info(message, extra={"event_data": kwargs} if kwargs else {})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(message, extra={"event_data": kwargs} if kwargs else {})

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(message, extra={"event_data": kwargs} if kwargs else {})

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(message, extra={"event_data": kwargs} if kwargs else {})

    def security_event(self, event_type: str, message: str, **kwargs: Any) -> None:
        """Log a security event with structured metadata.

        Args:
            event_type: One of the EVENT_* constants.
            message: Human-readable event description.
            **kwargs: Additional structured data to include.
        """
        event_data = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self.logger.warning(
            f"[SECURITY] {message}",
            extra={"event_data": event_data},
        )

    def get_recent_events(self, count: int = 20) -> list[str]:
        """Read the most recent log lines from the log file.

        Args:
            count: Number of recent lines to return.

        Returns:
            List of recent log lines (newest last).
        """
        log_file = self.log_dir / "sentry.log"
        if not log_file.exists():
            return []

        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            return [line.rstrip() for line in lines[-count:]]
        except OSError:
            return []


# Module-level singleton
_logger: EventLogger | None = None


def get_logger() -> EventLogger:
    """Get the application-wide EventLogger instance.

    Returns:
        The singleton EventLogger.
    """
    global _logger
    if _logger is None:
        _logger = EventLogger()
    return _logger
