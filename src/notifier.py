"""Email and WhatsApp alert routing for Sentry.

Sends security notifications with optional screenshot attachments
via SMTP email and WhatsApp Business API. Fault-tolerant — a
notification failure never crashes the monitoring process.
"""

from __future__ import annotations

import email.mime.base
import email.mime.image
import email.mime.multipart
import email.mime.text
import smtplib
import ssl
from pathlib import Path

import requests

from src.config import get_config, get_env, get_env_bool
from src.logger import EVENT_ALERT_FAILED, EVENT_ALERT_SENT, get_logger


class NotificationError(Exception):
    """Raised when a notification attempt fails."""


def send_alert(
    event_type: str,
    message: str,
    screenshot_path: Path | None = None,
) -> dict[str, bool]:
    """Send a security alert through all enabled notification channels.

    Each channel operates independently — failure in one does not
    prevent attempts on others.

    Args:
        event_type: Type of security event (e.g., 'unknown_face').
        message: Human-readable alert message.
        screenshot_path: Optional path to screenshot to attach.

    Returns:
        Dictionary of channel names to success status.
        E.g., {'email': True, 'whatsapp': False}
    """
    logger = get_logger()
    config = get_config()
    notifications = config.get("notifications", {})
    results: dict[str, bool] = {}

    # Email channel
    if notifications.get("email", False) and get_env_bool("EMAIL_ENABLED", False):
        try:
            _send_email(event_type, message, screenshot_path)
            results["email"] = True
            logger.security_event(
                EVENT_ALERT_SENT,
                "Email notification sent",
                channel="email",
                alert_event_type=event_type,
            )
        except NotificationError as e:
            results["email"] = False
            logger.security_event(
                EVENT_ALERT_FAILED,
                f"Email notification failed: {e}",
                channel="email",
                alert_event_type=event_type,
                error=str(e),
            )

    # WhatsApp channel
    if notifications.get("whatsapp", False) and get_env_bool("WHATSAPP_ENABLED", False):
        try:
            _send_whatsapp(event_type, message, screenshot_path)
            results["whatsapp"] = True
            logger.security_event(
                EVENT_ALERT_SENT,
                "WhatsApp notification sent",
                channel="whatsapp",
                alert_event_type=event_type,
            )
        except NotificationError as e:
            results["whatsapp"] = False
            logger.security_event(
                EVENT_ALERT_FAILED,
                f"WhatsApp notification failed: {e}",
                channel="whatsapp",
                alert_event_type=event_type,
                error=str(e),
            )

    if not results:
        logger.debug("No notification channels enabled")

    return results


def _send_email(
    event_type: str,
    message: str,
    screenshot_path: Path | None = None,
) -> None:
    """Send an email alert via SMTP with optional screenshot attachment.

    Args:
        event_type: Security event type.
        message: Alert message body.
        screenshot_path: Optional screenshot to attach.

    Raises:
        NotificationError: If sending fails.
    """
    smtp_host = get_env("SMTP_HOST")
    smtp_port = int(get_env("SMTP_PORT", "587"))
    smtp_username = get_env("SMTP_USERNAME")
    smtp_password = get_env("SMTP_PASSWORD")
    alert_email = get_env("ALERT_EMAIL")

    if not all([smtp_host, smtp_username, smtp_password, alert_email]):
        raise NotificationError(
            "Email configuration incomplete. "
            "Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and ALERT_EMAIL."
        )

    # Build the email message
    msg = email.mime.multipart.MIMEMultipart()
    msg["From"] = smtp_username
    msg["To"] = alert_email
    msg["Subject"] = f"[Sentry] Security Alert: {event_type}"

    # Email body
    body = (
        f"Sentry Security Alert\n"
        f"{'=' * 40}\n\n"
        f"Event: {event_type}\n"
        f"Details: {message}\n\n"
        f"This is an automated alert from Sentry.\n"
        f"An unrecognized person was detected at your workstation.\n"
    )

    if screenshot_path and screenshot_path.exists():
        body += f"\nScreenshot attached: {screenshot_path.name}\n"

    msg.attach(email.mime.text.MIMEText(body, "plain"))

    # Attach screenshot if available
    if screenshot_path and screenshot_path.exists():
        try:
            with open(screenshot_path, "rb") as img_file:
                img_data = img_file.read()
            img_attachment = email.mime.image.MIMEImage(img_data, name=screenshot_path.name)
            img_attachment.add_header(
                "Content-Disposition",
                "attachment",
                filename=screenshot_path.name,
            )
            msg.attach(img_attachment)
        except OSError as e:
            # Proceed without attachment rather than failing entirely
            get_logger().warning(f"Could not attach screenshot: {e}")

    # Send via SMTP with TLS
    try:
        context = ssl.create_default_context()

        if smtp_port == 465:
            # SSL from the start
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=30) as server:
                server.login(smtp_username, smtp_password)
                server.sendmail(smtp_username, [alert_email], msg.as_string())
        else:
            # STARTTLS (port 587 typical)
            with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(smtp_username, smtp_password)
                server.sendmail(smtp_username, [alert_email], msg.as_string())

    except smtplib.SMTPAuthenticationError as e:
        raise NotificationError(f"SMTP authentication failed: {e}") from e
    except smtplib.SMTPException as e:
        raise NotificationError(f"SMTP error: {e}") from e
    except OSError as e:
        raise NotificationError(f"Network error sending email: {e}") from e


def _send_whatsapp(
    event_type: str,
    message: str,
    screenshot_path: Path | None = None,
) -> None:
    """Send a WhatsApp alert via the WhatsApp Business API.

    Uses the official Cloud API (graph.facebook.com) to send
    a text message. Image sending requires a media upload step.

    Args:
        event_type: Security event type.
        message: Alert message body.
        screenshot_path: Optional screenshot (uploaded as media if provided).

    Raises:
        NotificationError: If sending fails.
    """
    api_url = get_env("WHATSAPP_API_URL")
    api_token = get_env("WHATSAPP_API_TOKEN")
    destination = get_env("WHATSAPP_DESTINATION")

    if not all([api_url, api_token, destination]):
        raise NotificationError(
            "WhatsApp configuration incomplete. "
            "Set WHATSAPP_API_URL, WHATSAPP_API_TOKEN, and WHATSAPP_DESTINATION."
        )

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    # Send text message
    alert_text = (
        f"*Sentry Security Alert*\n\n"
        f"Event: {event_type}\n"
        f"Details: {message}\n\n"
        f"An unrecognized person was detected at your workstation."
    )

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": destination,
        "type": "text",
        "text": {"preview_url": False, "body": alert_text},
    }

    try:
        response = requests.post(
            api_url,
            json=payload,
            headers=headers,
            timeout=30,
        )

        if response.status_code not in (200, 201):
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = str(error_data.get("error", {}).get("message", ""))
            except (ValueError, KeyError):
                error_detail = response.text[:200]
            raise NotificationError(
                f"WhatsApp API returned {response.status_code}: {error_detail}"
            )

    except requests.exceptions.Timeout:
        raise NotificationError("WhatsApp API request timed out")
    except requests.exceptions.ConnectionError as e:
        raise NotificationError(f"WhatsApp API connection failed: {e}")
    except requests.exceptions.RequestException as e:
        raise NotificationError(f"WhatsApp API request error: {e}")

    # If screenshot exists, attempt to send as image (separate request)
    if screenshot_path and screenshot_path.exists():
        _send_whatsapp_image(api_url, api_token, destination, screenshot_path)


def _send_whatsapp_image(
    api_url: str,
    api_token: str,
    destination: str,
    image_path: Path,
) -> None:
    """Upload and send an image via WhatsApp Business API.

    This is a best-effort operation — failure is logged but does
    not raise, since the text alert already succeeded.

    Args:
        api_url: WhatsApp API messages endpoint URL.
        api_token: Bearer token.
        destination: Recipient phone number.
        image_path: Path to the image file.
    """
    logger = get_logger()

    # Derive media upload URL from messages URL
    # e.g., https://graph.facebook.com/v17.0/PHONE_ID/messages
    #     -> https://graph.facebook.com/v17.0/PHONE_ID/media
    media_url = api_url.rsplit("/", 1)[0] + "/media"

    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        # Upload the image
        with open(image_path, "rb") as img_file:
            upload_response = requests.post(
                media_url,
                headers=headers,
                files={"file": (image_path.name, img_file, "image/png")},
                data={"messaging_product": "whatsapp"},
                timeout=60,
            )

        if upload_response.status_code not in (200, 201):
            logger.warning(f"WhatsApp image upload failed: {upload_response.status_code}")
            return

        media_id = upload_response.json().get("id")
        if not media_id:
            logger.warning("WhatsApp image upload returned no media ID")
            return

        # Send the image message
        image_payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": destination,
            "type": "image",
            "image": {"id": media_id, "caption": "Sentry: Intruder screenshot"},
        }

        send_headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }

        send_response = requests.post(
            api_url,
            json=image_payload,
            headers=send_headers,
            timeout=30,
        )

        if send_response.status_code not in (200, 201):
            logger.warning(f"WhatsApp image send failed: {send_response.status_code}")

    except Exception as e:
        logger.warning(f"WhatsApp image send error (non-critical): {e}")
