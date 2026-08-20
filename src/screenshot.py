"""Desktop screenshot capture for Sentry.

Captures timestamped screenshots on security events using Pillow's
ImageGrab. Includes retention management to automatically clean up
old screenshots.
"""

from __future__ import annotations

import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import ImageGrab

from src.config import get_config, resolve_path
from src.logger import EVENT_SCREENSHOT_TAKEN, get_logger


class ScreenshotError(Exception):
    """Raised when screenshot capture fails."""


def capture_screenshot(event_label: str = "unknown-face") -> Path | None:
    """Capture a desktop screenshot and save it with a timestamp.

    Args:
        event_label: Label describing the event (used in filename).

    Returns:
        Path to the saved screenshot, or None if capture failed.
    """
    logger = get_logger()
    config = get_config()
    storage = config.get("storage", {})
    screenshot_dir = resolve_path(storage.get("screenshot_directory", "./screenshots"))

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Sanitize event label for filesystem safety
    safe_label = "".join(c if c.isalnum() or c in "-_" else "-" for c in event_label)
    filename = f"{timestamp}_{safe_label}.png"
    filepath = screenshot_dir / filename

    try:
        image = _grab_screen()

        if image is None:
            logger.error("Screenshot capture returned None")
            return None

        # Save with restrictive intent (Pillow doesn't set file permissions,
        # but we can set them after saving on Unix systems)
        image.save(str(filepath), "PNG", optimize=True)

        # Set restrictive permissions on Unix-like systems
        _set_restrictive_permissions(filepath)

        logger.security_event(
            EVENT_SCREENSHOT_TAKEN,
            f"Screenshot saved: {filename}",
            path=str(filepath),
            size_bytes=filepath.stat().st_size,
        )

        return filepath

    except Exception as e:
        logger.error(f"Failed to capture screenshot: {e}")
        return None


def _grab_screen():
    """Capture the screen using Pillow's ImageGrab.

    Handles platform differences for screen capture.

    Returns:
        PIL Image of the screen, or None if capture fails.
    """
    system = platform.system()

    try:
        if system == "Linux":
            # On Linux, ImageGrab may not work without extra dependencies.
            # Try ImageGrab first, fall back to scrot/gnome-screenshot.
            try:
                image = ImageGrab.grab()
                if image is not None:
                    return image
            except OSError:
                return _grab_screen_linux_fallback()
        else:
            # Windows and macOS: ImageGrab works natively
            return ImageGrab.grab()
    except Exception:
        if system == "Linux":
            return _grab_screen_linux_fallback()
        return None


def _grab_screen_linux_fallback():
    """Fallback screen capture for Linux using command-line tools.

    Returns:
        PIL Image of the screen, or None if all methods fail.
    """
    import tempfile

    from PIL import Image

    # Try scrot first, then gnome-screenshot, then import (ImageMagick)
    tmp_path = Path(tempfile.mktemp(suffix=".png"))

    commands = [
        ["scrot", str(tmp_path)],
        ["gnome-screenshot", "-f", str(tmp_path)],
        ["import", "-window", "root", str(tmp_path)],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )
            if result.returncode == 0 and tmp_path.exists():
                image = Image.open(tmp_path)
                image.load()  # Force load before deleting temp file
                tmp_path.unlink(missing_ok=True)
                return image
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue
        finally:
            tmp_path.unlink(missing_ok=True)

    return None


def _set_restrictive_permissions(filepath: Path) -> None:
    """Set restrictive file permissions (owner read/write only).

    Only applies on Unix-like systems. Silently skipped on Windows.

    Args:
        filepath: Path to the file to restrict.
    """
    if platform.system() == "Windows":
        return

    try:
        filepath.chmod(0o600)
    except OSError:
        pass


def cleanup_old_screenshots(retention_days: int | None = None) -> int:
    """Delete screenshots older than the configured retention period.

    Args:
        retention_days: Number of days to retain. Defaults to config value.

    Returns:
        Number of files deleted.
    """
    logger = get_logger()
    config = get_config()
    storage = config.get("storage", {})

    if retention_days is None:
        retention_days = storage.get("retention_days", 7)

    screenshot_dir = resolve_path(storage.get("screenshot_directory", "./screenshots"))

    if not screenshot_dir.exists():
        return 0

    cutoff_time = time.time() - (retention_days * 86400)
    deleted_count = 0

    for filepath in screenshot_dir.glob("*.png"):
        try:
            if filepath.stat().st_mtime < cutoff_time:
                filepath.unlink()
                deleted_count += 1
        except OSError:
            continue

    if deleted_count > 0:
        logger.info(
            f"Cleaned up {deleted_count} screenshot(s) older than {retention_days} days"
        )

    return deleted_count
