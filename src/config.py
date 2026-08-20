"""Configuration loader for Sentry.

Loads settings from config.yaml and environment variables (.env).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Load .env file if it exists
_env_path = ROOT_DIR / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Optional path to config.yaml. Defaults to project root.

    Returns:
        Configuration dictionary.
    """
    if path is None:
        path = ROOT_DIR / "config.yaml"

    if not path.exists():
        return _default_config()

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config if config else _default_config()


def _default_config() -> dict[str, Any]:
    """Return default configuration values."""
    return {
        "camera": {
            "device": 0,
            "frame_interval_ms": 250,
        },
        "detection": {
            "required_consecutive_frames": 5,
            "recognition_threshold": 0.55,
        },
        "security": {
            "lock_on_unknown_face": True,
            "screenshot_on_unknown_face": True,
            "cooldown_seconds": 300,
        },
        "storage": {
            "screenshot_directory": "./screenshots",
            "log_directory": "./logs",
            "model_directory": "./models",
            "retention_days": 7,
        },
        "notifications": {
            "email": True,
            "whatsapp": False,
        },
    }


def get_env(key: str, default: str = "") -> str:
    """Get an environment variable value.

    Args:
        key: Environment variable name.
        default: Default value if not set.

    Returns:
        The environment variable value or default.
    """
    return os.environ.get(key, default)


def get_env_bool(key: str, default: bool = False) -> bool:
    """Get an environment variable as a boolean.

    Args:
        key: Environment variable name.
        default: Default boolean value.

    Returns:
        Boolean interpretation of the env var.
    """
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("true", "1", "yes", "on")


# Singleton config instance
_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Get the application configuration (cached singleton).

    Returns:
        Configuration dictionary.
    """
    global _config
    if _config is None:
        _config = load_yaml_config()
    return _config


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path against the project root.

    Args:
        relative_path: Path string from config (e.g., './screenshots').

    Returns:
        Resolved absolute Path.
    """
    path = Path(relative_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path
