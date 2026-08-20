<p align="center">
  <img src="assets/Sentry.svg" alt="Sentry Banner" width="100%" />
</p>

**Face-based desktop intrusion detection with automated response.**

Sentry watches your webcam in real time, learns your face, and acts instantly when someone unauthorized sits down — locking the screen, capturing evidence, and alerting you via email or WhatsApp.

One command. No configuration required to get started.

```
python -m src
```

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV_5-DNN_Face-5C3EE8?logo=opencv)](https://opencv.org)
[![Textual](https://img.shields.io/badge/Textual-TUI-2D2D2D)](https://textual.textualize.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## How It Works

```
Launch → Enroll (if needed) → Monitor → Detect → Respond
```

1. **Enroll** — Sentry captures 10 face samples from your webcam (varied angles and expressions) and stores embeddings locally.
2. **Monitor** — Continuously reads frames, detects faces via YuNet DNN, and compares embeddings against the enrolled user using SFace.
3. **Confirm** — Requires multiple consecutive unrecognized frames before triggering (prevents false alarms from motion blur, lighting changes, or partial faces).
4. **Respond** — Locks the workstation, captures a timestamped screenshot, and sends alerts through configured channels.
5. **Cooldown** — Enters a configurable quiet period before resuming detection.

The entire workflow runs inside a single Textual TUI — no separate enrollment step, no CLI juggling.

---

## Features

- **Zero-config start** — launch once, enroll face, monitor indefinitely
- **Real-time face recognition** — OpenCV DNN (YuNet detector + SFace embedder)
- **Multi-frame confirmation** — configurable consecutive-frame threshold
- **Automatic screen lock** — Windows (ctypes), macOS (CGSession), Linux (loginctl/xdg)
- **Evidence capture** — timestamped PNG screenshots with auto-retention cleanup
- **Multi-channel alerts** — SMTP email with attachments, WhatsApp Business API
- **Modern TUI dashboard** — live status, stats, event log, keyboard-driven
- **Fault-tolerant** — each response action (lock, screenshot, alert) runs independently
- **Cross-platform** — Windows, macOS, Linux

---

## Quick Start

### Prerequisites

- Python 3.10+
- A webcam
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Install and Run

```bash
git clone https://github.com/your-username/sentry.git
cd sentry

# Using uv (recommended)
uv sync --extra dev
uv run python -m src

# Or with pip
python -m venv .venv
.venv/Scripts/activate      # Windows
source .venv/bin/activate   # macOS/Linux
pip install -e ".[dev]"
python -m src
```

On first launch, Sentry will:
1. Download face detection models (~250KB + ~36MB, one-time)
2. Open your camera and guide you through face enrollment
3. Automatically transition to monitoring

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Pause / Resume monitoring |
| `R` | Re-enroll face (clears existing data) |
| `D` | Delete old screenshots |
| `L` | Clear event log |
| `Q` | Quit |

---

## Configuration

### Runtime settings (`config.yaml`)

```yaml
camera:
  device: 0
  frame_interval_ms: 250

detection:
  required_consecutive_frames: 5
  recognition_threshold: 0.55

security:
  lock_on_unknown_face: true
  screenshot_on_unknown_face: true
  cooldown_seconds: 30

storage:
  screenshot_directory: "./screenshots"
  log_directory: "./logs"
  model_directory: "./models"
  retention_days: 7

notifications:
  email: true
  whatsapp: false
```

### Notification credentials (`.env`)

```bash
cp .env.example .env
```

```env
# Email
EMAIL_ENABLED=true
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=you@gmail.com
SMTP_PASSWORD=your-app-password
ALERT_EMAIL=you@gmail.com

# WhatsApp (official Business API)
WHATSAPP_ENABLED=false
WHATSAPP_API_URL=https://graph.facebook.com/v17.0/PHONE_ID/messages
WHATSAPP_API_TOKEN=your-token
WHATSAPP_DESTINATION=recipient-number
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Face Detection | OpenCV DNN — YuNet (ONNX) |
| Face Recognition | OpenCV DNN — SFace (ONNX, 128-dim embeddings) |
| Similarity | Cosine similarity via `FaceRecognizerSF` |
| TUI | Textual |
| Screenshots | Pillow (ImageGrab) |
| Email | smtplib (STARTTLS/SSL) |
| Messaging | WhatsApp Business Cloud API |
| Config | PyYAML + python-dotenv |
| Package Manager | uv / pip |
| Testing | pytest |

---

## Project Structure

```
sentry/
├── pyproject.toml           # Dependencies and build config
├── config.yaml              # Runtime configuration
├── .env.example             # Notification credentials template
├── src/
│   ├── __main__.py          # Entry point
│   ├── main.py              # Unified TUI app (enrollment + monitoring)
│   ├── config.py            # YAML/env config loader
│   ├── logger.py            # Rotating file + console logger
│   ├── camera.py            # OpenCV webcam abstraction
│   ├── face_recognition.py  # YuNet detector + SFace recognizer
│   ├── screen_lock.py       # Cross-platform screen locking
│   ├── screenshot.py        # Timestamped capture + retention
│   └── notifier.py          # Email + WhatsApp routing
├── models/                  # ONNX models + face embeddings (git-ignored)
├── screenshots/             # Captured evidence (git-ignored)
├── logs/                    # Event logs (git-ignored)
└── tests/                   # pytest test suite
```

---

## Architecture

```
┌──────────┐    ┌──────────────┐    ┌────────────────┐
│  Camera  │───▶│ YuNet Detect │───▶│ SFace Embedding│
└──────────┘    └──────────────┘    └───────┬────────┘
                                            │
                                   ┌────────▼────────┐
                                   │  Match Enrolled? │
                                   └──┬──────────┬───┘
                                   Yes│          │No (n frames)
                                      ▼          ▼
                                   Continue   ┌──────────────┐
                                              │ Lock Screen  │
                                              │ Screenshot   │
                                              │ Send Alert   │
                                              │ Cooldown     │
                                              └──────────────┘
```

---

## Security and Privacy

- **Biometric data stays local** — embeddings stored in `models/` as a pickle file, never transmitted
- **Credentials in `.env`** — never committed to git
- **Restrictive file permissions** — screenshots saved with `0600` on Unix
- **Auto-retention** — old screenshots deleted after configurable days
- **Fail-safe design** — notification failure doesn't prevent screen lock or evidence capture

> **Notice:** Only use Sentry on devices and workspaces you're authorized to monitor. Comply with applicable privacy, biometric, and workplace-monitoring laws.

---

## Testing

```bash
uv run python -m pytest      # 62 tests
uv run python -m pytest -v   # Verbose output
```

Tests cover: camera mocking, face store CRUD, config loading, screen lock dispatch, screenshot capture/retention, notification routing, monitoring engine state machine (multi-frame confirmation, cooldown, intrusion response).

---

## Running at Startup

| Platform | Method |
|----------|--------|
| Windows | Task Scheduler → `python -m src` |
| macOS | Launch Agent (`~/Library/LaunchAgents/`) |
| Linux | systemd user service or XDG autostart |

---

## License

[MIT](LICENSE)
