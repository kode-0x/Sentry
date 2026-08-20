"""Shared test fixtures for Sentry tests."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Provide an isolated config environment for each test.

    Creates a temporary config.yaml and sets up directories
    so tests don't touch real project files.
    """
    # Create test directories
    (tmp_path / "models").mkdir()
    (tmp_path / "screenshots").mkdir()
    (tmp_path / "logs").mkdir()

    # Write test config
    config = {
        "camera": {"device": 0, "frame_interval_ms": 50},
        "detection": {
            "required_consecutive_frames": 3,
            "recognition_threshold": 0.55,
        },
        "security": {
            "lock_on_unknown_face": True,
            "screenshot_on_unknown_face": True,
            "cooldown_seconds": 10,
        },
        "storage": {
            "screenshot_directory": str(tmp_path / "screenshots"),
            "log_directory": str(tmp_path / "logs"),
            "model_directory": str(tmp_path / "models"),
            "retention_days": 7,
        },
        "notifications": {"email": False, "whatsapp": False},
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Clear env vars to prevent test pollution
    env_vars = [
        "EMAIL_ENABLED", "SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME",
        "SMTP_PASSWORD", "ALERT_EMAIL", "WHATSAPP_ENABLED",
        "WHATSAPP_API_URL", "WHATSAPP_API_TOKEN", "WHATSAPP_DESTINATION",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    # Reset the config singleton so tests get fresh config
    import src.config
    monkeypatch.setattr(src.config, "_config", None)
    monkeypatch.setattr(src.config, "ROOT_DIR", tmp_path)

    # Reset logger singleton
    import src.logger
    monkeypatch.setattr(src.logger, "_logger", None)

    return tmp_path


@pytest.fixture
def fake_frame():
    """Create a fake BGR image frame for testing."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def fake_embedding():
    """Create a fake 128-dimensional face embedding."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(128).astype(np.float32)
    # Normalize to unit vector (as SFace does)
    emb = emb / np.linalg.norm(emb)
    return emb


@pytest.fixture
def different_embedding():
    """Create a different fake embedding (low similarity to fake_embedding)."""
    rng = np.random.default_rng(99)
    emb = rng.standard_normal(128).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb
