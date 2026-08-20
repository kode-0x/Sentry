"""Platform-specific screen locking for Sentry.

Supports Windows, macOS, and Linux with graceful fallback
when locking is unavailable on the current platform.
"""

from __future__ import annotations

import platform
import subprocess
import sys

from src.logger import EVENT_SCREEN_LOCKED, get_logger


class ScreenLockError(Exception):
    """Raised when screen locking fails."""


def lock_screen() -> bool:
    """Lock the workstation screen using the platform-appropriate method.

    Returns:
        True if the screen was locked successfully, False otherwise.

    The function fails safely rather than raising — monitoring should
    continue even if locking is unavailable.
    """
    logger = get_logger()
    system = platform.system()

    try:
        if system == "Windows":
            _lock_windows()
        elif system == "Darwin":
            _lock_macos()
        elif system == "Linux":
            _lock_linux()
        else:
            logger.warning(f"Screen locking not supported on platform: {system}")
            return False

        logger.security_event(
            EVENT_SCREEN_LOCKED,
            "Workstation screen locked",
            platform=system,
        )
        return True

    except ScreenLockError as e:
        logger.error(f"Failed to lock screen: {e}", platform=system)
        return False
    except Exception as e:
        logger.error(f"Unexpected error locking screen: {e}", platform=system)
        return False


def _lock_windows() -> None:
    """Lock the Windows workstation using ctypes.

    Calls LockWorkStation from user32.dll.

    Raises:
        ScreenLockError: If the lock call fails.
    """
    if sys.platform != "win32":
        raise ScreenLockError("Not running on Windows")

    import ctypes

    result = ctypes.windll.user32.LockWorkStation()  # type: ignore[attr-defined]
    if not result:
        raise ScreenLockError(
            "LockWorkStation returned False. "
            "This may happen if the workstation is already locked "
            "or if called from a service without the interactive desktop."
        )


def _lock_macos() -> None:
    """Lock macOS by activating the screensaver via osascript.

    Uses the system screensaver which requires the user to authenticate
    to dismiss (if password-on-wake is enabled in System Preferences).

    Raises:
        ScreenLockError: If the lock command fails.
    """
    # pmset displaysleepnow is another option but may not require password
    # Using login window via CGSession is more reliable for locking
    commands = [
        # Primary: use CGSession to switch to login window
        [
            "/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession",
            "-suspend",
        ],
        # Fallback: activate screensaver
        [
            "osascript",
            "-e",
            'tell application "System Events" to start current screen saver',
        ],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    raise ScreenLockError(
        "All macOS locking methods failed. "
        "Ensure 'Require password after sleep or screen saver begins' "
        "is enabled in System Preferences > Security & Privacy."
    )


def _lock_linux() -> None:
    """Lock the Linux desktop using available session managers.

    Tries multiple locking mechanisms in order of preference:
    1. loginctl lock-session (systemd)
    2. dbus-send to org.freedesktop.ScreenSaver
    3. xdg-screensaver lock
    4. gnome-screensaver-command --lock

    Raises:
        ScreenLockError: If no locking method succeeds.
    """
    commands = [
        # systemd logind (works on most modern distros)
        ["loginctl", "lock-session"],
        # D-Bus screensaver interface
        [
            "dbus-send",
            "--type=method_call",
            "--dest=org.freedesktop.ScreenSaver",
            "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.Lock",
        ],
        # XDG standard
        ["xdg-screensaver", "lock"],
        # GNOME specific
        ["gnome-screensaver-command", "--lock"],
        # KDE specific
        ["qdbus", "org.freedesktop.ScreenSaver", "/ScreenSaver", "Lock"],
    ]

    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    raise ScreenLockError(
        "All Linux locking methods failed. "
        "Ensure a screen locker is installed and the session supports locking. "
        "Tried: loginctl, dbus-send, xdg-screensaver, gnome-screensaver-command, qdbus."
    )


def is_lock_available() -> bool:
    """Check if screen locking is likely available on this platform.

    Returns:
        True if the platform is supported and basic prerequisites are met.
    """
    system = platform.system()

    if system == "Windows":
        # LockWorkStation is available on all modern Windows
        return sys.platform == "win32"

    elif system == "Darwin":
        # macOS always has CGSession or osascript
        return True

    elif system == "Linux":
        # Check if any locking command exists
        check_commands = ["loginctl", "xdg-screensaver", "gnome-screensaver-command"]
        for cmd in check_commands:
            try:
                result = subprocess.run(
                    ["which", cmd],
                    capture_output=True,
                    timeout=3,
                )
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        return False

    return False
