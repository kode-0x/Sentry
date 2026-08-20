"""Tests for src/logger.py."""

from pathlib import Path

from src.logger import (
    EVENT_UNKNOWN_FACE,
    EventLogger,
    get_logger,
)


class TestEventLogger:
    def test_creates_log_file(self, isolated_config):
        logger = EventLogger("test_creates")
        logger.info("test message")
        # Flush handlers to ensure file is written
        for handler in logger.logger.handlers:
            handler.flush()
        log_file = Path(isolated_config) / "logs" / "sentry.log"
        assert log_file.exists()

    def test_info_logged_to_file(self, isolated_config):
        logger = EventLogger("test_info")
        logger.info("hello world")
        log_file = Path(isolated_config) / "logs" / "sentry.log"
        content = log_file.read_text()
        assert "hello world" in content

    def test_security_event_includes_metadata(self, isolated_config):
        logger = EventLogger("test_security")
        logger.security_event(EVENT_UNKNOWN_FACE, "intruder detected", score=0.3)
        log_file = Path(isolated_config) / "logs" / "sentry.log"
        content = log_file.read_text()
        assert "SECURITY" in content
        assert "intruder detected" in content
        assert "unknown_face" in content

    def test_get_recent_events(self, isolated_config):
        logger = EventLogger("test_recent")
        for i in range(5):
            logger.info(f"event {i}")

        events = logger.get_recent_events(3)
        assert len(events) == 3
        assert "event 4" in events[-1]

    def test_get_recent_events_empty_log(self, isolated_config):
        logger = EventLogger("test_empty")
        events = logger.get_recent_events()
        assert events == []


class TestGetLogger:
    def test_returns_event_logger(self, isolated_config):
        logger = get_logger()
        assert isinstance(logger, EventLogger)

    def test_singleton(self, isolated_config):
        logger1 = get_logger()
        logger2 = get_logger()
        assert logger1 is logger2
