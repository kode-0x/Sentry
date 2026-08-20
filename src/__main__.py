"""Sentry entry point.

Run with:
    python -m src              — Launch Sentry (enrollment + monitoring)
    python -m src --cleanup    — Run screenshot cleanup and exit
"""

from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]

    if "--cleanup" in args or "cleanup" in args:
        from src.screenshot import cleanup_old_screenshots
        deleted = cleanup_old_screenshots()
        print(f"Cleaned up {deleted} old screenshot(s).")
        return

    if "--help" in args or "-h" in args:
        print(__doc__)
        return

    from src.main import main as run_app
    run_app()


if __name__ == "__main__":
    main()
