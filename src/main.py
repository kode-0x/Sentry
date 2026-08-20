"""Sentry — Unified TUI with enrollment and monitoring.

Single-command workflow: launches the TUI, auto-enrolls if no face
data exists, then seamlessly transitions to monitoring. The interface
is designed to be fast, responsive, and require minimal user interaction.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Footer, Header, Label, Log, ProgressBar, Rule, Static

from src.camera import Camera, CameraError
from src.config import get_config
from src.face_recognition import (
    EnrolledFaceStore,
    FaceDetector,
    FaceRecognizer,
    FaceRecognitionError,
)
from src.logger import (
    EVENT_COOLDOWN_ENDED,
    EVENT_COOLDOWN_STARTED,
    EVENT_FACE_ENROLLED,
    EVENT_MONITORING_STARTED,
    EVENT_MONITORING_STOPPED,
    EVENT_UNKNOWN_FACE,
    get_logger,
)
from src.notifier import send_alert
from src.screen_lock import lock_screen
from src.screenshot import capture_screenshot, cleanup_old_screenshots


# ─── Enrollment Engine ────────────────────────────────────────────────

ENROLLMENT_SAMPLES = 10
ENROLLMENT_INTERVAL = 1.2  # seconds between captures


class EnrollmentEngine:
    """Handles face enrollment — captures samples and saves embeddings."""

    def __init__(
        self,
        camera: Camera,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        store: EnrolledFaceStore,
    ) -> None:
        self._camera = camera
        self._detector = detector
        self._recognizer = recognizer
        self._store = store
        self._logger = get_logger()
        self._last_capture: float = 0.0
        self.samples_collected: int = 0
        self.target_samples: int = ENROLLMENT_SAMPLES
        self.is_complete: bool = False
        self.face_detected: bool = False
        self.message: str = "Look at the camera..."

    def process_frame(self) -> None:
        """Process one frame for enrollment."""
        try:
            frame = self._camera.read_frame()
        except CameraError:
            self.message = "Camera error — retrying..."
            return

        faces = self._detector.detect(frame)
        self.face_detected = len(faces) == 1

        if len(faces) == 0:
            self.message = "No face detected. Please face the camera."
            return
        if len(faces) > 1:
            self.message = "Multiple faces — only you should be visible."
            return

        now = time.time()
        if now - self._last_capture < ENROLLMENT_INTERVAL:
            remaining = ENROLLMENT_INTERVAL - (now - self._last_capture)
            self.message = f"Hold steady... ({remaining:.1f}s)"
            return

        try:
            embedding = self._recognizer.get_embedding(frame, faces[0])
            if np.isnan(embedding).any() or np.allclose(embedding, 0):
                self.message = "Poor capture. Adjust position."
                return

            self._store.add_embedding(embedding)
            self.samples_collected += 1
            self._last_capture = now

            remaining = self.target_samples - self.samples_collected
            if remaining <= 0:
                self._store.save()
                self.is_complete = True
                self.message = "Enrollment complete!"
                self._logger.security_event(
                    EVENT_FACE_ENROLLED,
                    f"Face enrolled with {self.samples_collected} samples",
                    samples=self.samples_collected,
                )
            else:
                self.message = f"Captured {self.samples_collected}/{self.target_samples} — try a different angle"
        except FaceRecognitionError:
            self.message = "Capture failed — try again."


# ─── Monitoring Engine ────────────────────────────────────────────────


class MonitoringEngine:
    """Core monitoring logic with multi-frame confirmation and cooldown."""

    def __init__(self) -> None:
        self._logger = get_logger()
        self._config = get_config()
        detection_cfg = self._config.get("detection", {})
        security_cfg = self._config.get("security", {})

        self.required_consecutive_frames: int = detection_cfg.get(
            "required_consecutive_frames", 5
        )
        self.recognition_threshold: float = detection_cfg.get(
            "recognition_threshold", 0.55
        )
        self.lock_enabled: bool = security_cfg.get("lock_on_unknown_face", True)
        self.screenshot_enabled: bool = security_cfg.get("screenshot_on_unknown_face", True)
        self.cooldown_seconds: int = security_cfg.get("cooldown_seconds", 30)

        self._camera: Camera | None = None
        self._detector: FaceDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._face_store: EnrolledFaceStore | None = None

        self._consecutive_unknown: int = 0
        self._cooldown_until: float = 0.0
        self._is_running: bool = False
        self._total_frames: int = 0
        self._total_detections: int = 0
        self._total_alerts: int = 0
        self._last_event: str = ""

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    @property
    def cooldown_remaining(self) -> int:
        return max(0, int(self._cooldown_until - time.time()))

    @property
    def consecutive_unknown(self) -> int:
        return self._consecutive_unknown

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def total_detections(self) -> int:
        return self._total_detections

    @property
    def total_alerts(self) -> int:
        return self._total_alerts

    @property
    def last_event(self) -> str:
        return self._last_event

    def start(
        self,
        camera: Camera,
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        face_store: EnrolledFaceStore,
    ) -> None:
        """Start monitoring with pre-initialized components."""
        self._camera = camera
        self._detector = detector
        self._recognizer = recognizer
        self._face_store = face_store
        self._is_running = True
        self._last_event = "Monitoring active"
        self._logger.security_event(
            EVENT_MONITORING_STARTED,
            "Monitoring started",
            enrolled_samples=face_store.count,
        )

    def stop(self) -> None:
        """Stop monitoring and release camera."""
        self._is_running = False
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        self._logger.security_event(
            EVENT_MONITORING_STOPPED,
            "Monitoring stopped",
            total_frames=self._total_frames,
            total_alerts=self._total_alerts,
        )
        self._last_event = "Stopped"

    def process_frame(self) -> str:
        """Process a single camera frame. Returns status message."""
        if not self._is_running or self._camera is None:
            return "Not running"

        if self.is_in_cooldown:
            remaining = self.cooldown_remaining
            if remaining <= 0:
                self._logger.security_event(EVENT_COOLDOWN_ENDED, "Cooldown ended")
                self._cooldown_until = 0.0
            else:
                return f"Cooldown ({remaining}s)"

        try:
            frame = self._camera.read_frame_throttled()
        except CameraError as e:
            return f"Camera error: {e}"

        if frame is None:
            return "Scanning..."

        self._total_frames += 1
        faces = self._detector.detect(frame)

        if not faces:
            self._consecutive_unknown = 0
            return "No face"

        any_unknown = False
        for face in faces:
            try:
                embedding = self._recognizer.get_embedding(frame, face)
                if not self._face_store.is_authorized(
                    embedding, self._recognizer, self.recognition_threshold
                ):
                    any_unknown = True
                    self._total_detections += 1
            except FaceRecognitionError:
                continue

        if any_unknown:
            self._consecutive_unknown += 1
            progress = f"{self._consecutive_unknown}/{self.required_consecutive_frames}"
            if self._consecutive_unknown >= self.required_consecutive_frames:
                self._trigger_response()
                self._consecutive_unknown = 0
                return "INTRUSION — Response triggered"
            self._last_event = f"Unknown ({progress})"
            return f"Unknown face ({progress})"
        else:
            self._consecutive_unknown = 0
            return "Authorized"

    def _trigger_response(self) -> None:
        """Execute intrusion response: lock, screenshot, notify."""
        self._logger.security_event(
            EVENT_UNKNOWN_FACE,
            "Unknown face confirmed",
            consecutive_frames=self.required_consecutive_frames,
        )
        self._total_alerts += 1
        self._last_event = "INTRUSION RESPONSE"

        screenshot_path: Path | None = None
        if self.lock_enabled:
            lock_screen()
        if self.screenshot_enabled:
            screenshot_path = capture_screenshot("unknown-face")
        send_alert(
            event_type="unknown_face",
            message="An unrecognized person was detected at your workstation.",
            screenshot_path=screenshot_path,
        )
        self._cooldown_until = time.time() + self.cooldown_seconds
        self._logger.security_event(
            EVENT_COOLDOWN_STARTED,
            f"Cooldown ({self.cooldown_seconds}s)",
            cooldown_seconds=self.cooldown_seconds,
        )

    def run_cleanup(self) -> None:
        """Delete old screenshots."""
        cleanup_old_screenshots()


# ─── Unified TUI App ─────────────────────────────────────────────────

# App phases
PHASE_INIT = "init"
PHASE_ENROLL = "enroll"
PHASE_MONITORING = "monitoring"
PHASE_PAUSED = "paused"
PHASE_ERROR = "error"


class SentryApp(App[None]):
    """Unified Sentry TUI — enrollment + monitoring in one flow."""

    TITLE = "Sentry"

    CSS = """
    Screen {
        background: $surface;
    }

    #header-bar {
        height: 3;
        background: $primary-background;
        padding: 0 2;
        content-align: center middle;
    }

    #header-bar Label {
        text-style: bold;
        width: 1fr;
        content-align: center middle;
    }

    #main {
        height: 1fr;
        padding: 1 2;
    }

    #status-row {
        height: auto;
        margin-bottom: 1;
    }

    #phase-box {
        height: auto;
        width: 1fr;
        border: heavy $accent;
        padding: 1 2;
        margin-right: 1;
    }

    #phase-box.enrolling {
        border: heavy $warning;
    }

    #phase-box.alert {
        border: heavy $error;
    }

    #phase-box.cooldown {
        border: heavy $warning;
    }

    #stats-box {
        height: auto;
        width: 1fr;
        border: round $primary;
        padding: 1 2;
    }

    #activity-label {
        color: $text;
        margin-top: 1;
    }

    .phase-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .stat-value {
        color: $accent;
    }

    #enrollment-section {
        height: auto;
        margin-bottom: 1;
        padding: 0 1;
    }

    #event-log-box {
        height: 1fr;
        border: round $primary;
    }

    #event-log-box > Label {
        background: $primary;
        color: $text;
        padding: 0 2;
        width: 100%;
        text-style: bold;
    }

    #event-log {
        height: 1fr;
        padding: 0 1;
    }

    .indicator-ok {
        color: $success;
        text-style: bold;
    }

    .indicator-warn {
        color: $warning;
        text-style: bold;
    }

    .indicator-error {
        color: $error;
        text-style: bold;
    }

    .indicator-dim {
        color: $text-muted;
    }

    .hidden {
        display: none;
    }
    """

    BINDINGS = [
        ("q", "quit_app", "Quit"),
        ("space", "toggle_monitor", "Pause/Resume"),
        ("r", "re_enroll", "Re-enroll"),
        ("d", "run_cleanup", "Cleanup"),
        ("l", "clear_log", "Clear Log"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._phase: str = PHASE_INIT
        self._engine = MonitoringEngine()
        self._enrollment: EnrollmentEngine | None = None
        self._camera: Camera | None = None
        self._detector: FaceDetector | None = None
        self._recognizer: FaceRecognizer | None = None
        self._face_store: EnrolledFaceStore | None = None
        self._timer = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Container(id="main"):
            with Horizontal(id="status-row"):
                with Vertical(id="phase-box"):
                    yield Label("INITIALIZING", id="phase-title", classes="phase-title")
                    yield Label("", id="status-indicator")
                    yield Label("", id="activity-label")
                with Vertical(id="stats-box"):
                    yield Label("STATS", classes="phase-title")
                    yield Label("Frames: 0", id="stat-frames")
                    yield Label("Unknowns: 0", id="stat-unknowns")
                    yield Label("Alerts: 0", id="stat-alerts")
                    yield Label("Streak: 0/5", id="stat-streak")
            with Vertical(id="enrollment-section", classes="hidden"):
                yield Label("", id="enroll-status")
                yield ProgressBar(id="enroll-progress", total=ENROLLMENT_SAMPLES)
            with Vertical(id="event-log-box"):
                yield Label(" Event Log ")
                yield Log(id="event-log", highlight=True, auto_scroll=True)
        yield Footer()

    def on_mount(self) -> None:
        """Boot sequence: init components, decide phase, start timer."""
        self._log("Sentry starting...")
        self.set_timer(0.3, self._boot)

    def _boot(self) -> None:
        """Initialize camera and models, then decide workflow path."""
        try:
            self._log("Loading face detection model...")
            self._detector = FaceDetector()
            self._log("Loading face recognition model...")
            self._recognizer = FaceRecognizer()
            self._face_store = EnrolledFaceStore()
            self._log("Opening camera...")
            self._camera = Camera()
            self._camera.open()
            self._log("Camera ready.")
        except (CameraError, FaceRecognitionError) as e:
            self._set_phase(PHASE_ERROR)
            self._log(f"[ERROR] {e}")
            self._update_indicator(f"Error: {e}", "indicator-error")
            return

        if self._face_store.is_enrolled:
            self._log(f"Face data found ({self._face_store.count} samples). Starting monitor...")
            self._start_monitoring()
        else:
            self._log("No face enrolled. Starting enrollment...")
            self._start_enrollment()

    def _start_enrollment(self) -> None:
        """Enter enrollment phase."""
        self._set_phase(PHASE_ENROLL)
        self._enrollment = EnrollmentEngine(
            self._camera, self._detector, self._recognizer, self._face_store
        )
        section = self.query_one("#enrollment-section")
        section.remove_class("hidden")
        self._update_indicator("Face the camera — capturing samples", "indicator-warn")
        self._timer = self.set_interval(0.15, self._enrollment_tick)

    def _enrollment_tick(self) -> None:
        """Process enrollment frame."""
        if self._enrollment is None:
            return
        self._enrollment.process_frame()

        # Update UI
        progress_bar = self.query_one("#enroll-progress", ProgressBar)
        progress_bar.progress = self._enrollment.samples_collected
        status = self.query_one("#enroll-status", Label)
        status.update(self._enrollment.message)

        face_indicator = "●" if self._enrollment.face_detected else "○"
        self.query_one("#activity-label", Label).update(
            f"{face_indicator} {self._enrollment.message}"
        )

        if self._enrollment.is_complete:
            self._timer.stop()
            self._timer = None
            section = self.query_one("#enrollment-section")
            section.add_class("hidden")
            self._log("Enrollment complete. Transitioning to monitoring...")
            self.set_timer(0.5, self._start_monitoring)

    def _start_monitoring(self) -> None:
        """Enter monitoring phase."""
        self._set_phase(PHASE_MONITORING)
        self._engine.start(
            self._camera, self._detector, self._recognizer, self._face_store
        )
        self._update_indicator("Monitoring active", "indicator-ok")
        self._log("Monitoring active — watching for unauthorized faces.")
        self._timer = self.set_interval(0.12, self._monitor_tick)

    def _monitor_tick(self) -> None:
        """Process one monitoring frame and update UI."""
        if not self._engine.is_running:
            return

        status_msg = self._engine.process_frame()

        # Update activity
        activity = self.query_one("#activity-label", Label)
        activity.update(f"  {status_msg}")

        # Stats
        req = self._engine.required_consecutive_frames
        self.query_one("#stat-frames", Label).update(f"Frames: {self._engine.total_frames}")
        self.query_one("#stat-unknowns", Label).update(f"Unknowns: {self._engine.total_detections}")
        self.query_one("#stat-alerts", Label).update(f"Alerts: {self._engine.total_alerts}")
        self.query_one("#stat-streak", Label).update(
            f"Streak: {self._engine.consecutive_unknown}/{req}"
        )

        # Phase styling
        if self._engine.is_in_cooldown:
            self._update_indicator(
                f"Cooldown ({self._engine.cooldown_remaining}s)", "indicator-warn"
            )
            self._set_phase_style("cooldown")
        elif self._engine.consecutive_unknown > 0:
            self._update_indicator(f"Unknown face ({status_msg})", "indicator-error")
            self._set_phase_style("alert")
        else:
            self._update_indicator("Monitoring active", "indicator-ok")
            self._set_phase_style("")

        # Log intrusions
        if "INTRUSION" in status_msg:
            self._log(f"[ALERT] {status_msg}")
        elif "Unknown face" in status_msg and "1/" in status_msg:
            self._log(f"[WARN] {status_msg}")

    # ─── Actions ──────────────────────────────────────────────────

    def action_toggle_monitor(self) -> None:
        """Pause or resume monitoring."""
        if self._phase == PHASE_MONITORING and self._engine.is_running:
            # Pause
            if self._timer:
                self._timer.stop()
                self._timer = None
            self._engine.stop()
            self._set_phase(PHASE_PAUSED)
            self._update_indicator("Paused", "indicator-dim")
            self._log("Monitoring paused.")
        elif self._phase == PHASE_PAUSED:
            # Resume — reopen camera
            try:
                self._camera = Camera()
                self._camera.open()
                self._start_monitoring()
                self._log("Monitoring resumed.")
            except CameraError as e:
                self._log(f"[ERROR] Cannot resume: {e}")

    def action_re_enroll(self) -> None:
        """Reset face data and re-enroll."""
        if self._timer:
            self._timer.stop()
            self._timer = None
        if self._engine.is_running:
            self._engine.stop()

        # Clear old data
        if self._face_store:
            self._face_store.clear()
        self._log("Face data cleared. Re-enrolling...")

        # Reopen camera if closed
        try:
            if self._camera is None or not self._camera.is_opened:
                self._camera = Camera()
                self._camera.open()
            self._start_enrollment()
        except CameraError as e:
            self._log(f"[ERROR] Camera: {e}")

    def action_run_cleanup(self) -> None:
        """Delete old screenshots."""
        self._engine.run_cleanup()
        self._log("Old screenshots cleaned up.")

    def action_clear_log(self) -> None:
        """Clear the event log display."""
        self.query_one("#event-log", Log).clear()

    def action_quit_app(self) -> None:
        """Shut down and exit."""
        if self._timer:
            self._timer.stop()
        if self._engine.is_running:
            self._engine.stop()
        elif self._camera and self._camera.is_opened:
            self._camera.close()
        self.exit()

    # ─── Helpers ──────────────────────────────────────────────────

    def _set_phase(self, phase: str) -> None:
        """Update the current phase."""
        self._phase = phase
        title = self.query_one("#phase-title", Label)
        titles = {
            PHASE_INIT: "INITIALIZING",
            PHASE_ENROLL: "ENROLLING FACE",
            PHASE_MONITORING: "MONITORING",
            PHASE_PAUSED: "PAUSED",
            PHASE_ERROR: "ERROR",
        }
        title.update(titles.get(phase, phase.upper()))

    def _set_phase_style(self, style: str) -> None:
        """Update phase box border style."""
        box = self.query_one("#phase-box")
        box.remove_class("enrolling", "alert", "cooldown")
        if style:
            box.add_class(style)

    def _update_indicator(self, text: str, css_class: str) -> None:
        """Update the main status indicator."""
        indicator = self.query_one("#status-indicator", Label)
        indicator.remove_class("indicator-ok", "indicator-warn", "indicator-error", "indicator-dim")
        indicator.add_class(css_class)
        indicator.update(text)

    def _log(self, message: str) -> None:
        """Write to the event log."""
        ts = datetime.now().strftime("%H:%M:%S")
        self.query_one("#event-log", Log).write_line(f"[{ts}] {message}")

    def on_unmount(self) -> None:
        if self._engine.is_running:
            self._engine.stop()
        elif self._camera and self._camera.is_opened:
            self._camera.close()


def main() -> None:
    """Single entry point — launches the unified Sentry TUI."""
    app = SentryApp()
    app.run()


if __name__ == "__main__":
    main()
