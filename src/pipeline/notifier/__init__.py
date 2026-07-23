"""Notifier subsystem — Slack, Discord, and SMTP delivery.

Supports three configurable channels (Slack webhook, Discord webhook, SMTP email).
Channels are tried in priority order: Slack → Discord → Email.

SLA commitments (documented here per design spec §12):
- ``send_review_gate`` must dispatch within 60 seconds of gate trigger.
- ``send_failure_alert`` must dispatch within 60 seconds of failure detection.

Deduplication:
- Key: ``"{video_id}:{notification_type}:{stage_name or ''}"``
- Suppress duplicate notification if the same key was dispatched within the last 10 minutes.

Fallback chain:
- Each channel is attempted up to 2 times before moving to the next.
- If all channels fail the failure is logged; no exception is raised to callers.
- If no channel is configured a ``LogEntry`` warning is written via the optional
  ``log_writer`` callable.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from typing import Any, Callable, Literal, Optional

import httpx

from pipeline.models import (
    LogEntry,
    NotificationEvent,
    NotificationPayload,
    PipelineStatus,
    SmtpConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


@dataclass
class VideoSummary:
    """Summary of a single video included in a batch summary notification.

    Attributes:
        video_id: Pipeline-assigned identifier for the video.
        title: Human-readable title of the video.
        status: Current ``PipelineStatus`` of the video.
        scheduled_publish_datetime: When the video is scheduled to publish, or ``None``.
    """

    video_id: str
    title: str
    status: PipelineStatus
    scheduled_publish_datetime: Optional[datetime] = None


@dataclass
class NotifierConfig:
    """Configuration for the Notifier delivery channels.

    At least one channel should be configured.  If all are ``None`` the Notifier
    suppresses all notifications and writes a warning to the structured log.

    Attributes:
        slack_webhook_url: Full HTTPS URL of the Slack incoming webhook.
        discord_webhook_url: Full HTTPS URL of the Discord webhook.
        smtp: SMTP server settings for email delivery.
    """

    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    smtp: Optional[SmtpConfig] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEDUP_WINDOW = timedelta(minutes=10)
"""Duration within which a duplicate notification is suppressed."""

_PER_CHANNEL_ATTEMPTS = 2
"""Number of delivery attempts per channel before moving to the next."""


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _build_log_entry(
    *,
    event_type: Literal["api_call", "stage_transition", "retry_attempt", "error", "warning"],
    message: str,
    video_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    http_response_code: Optional[int] = None,
    retry_attempt: Optional[int] = None,
    metadata: Optional[dict] = None,  # type: ignore[type-arg]
) -> LogEntry:
    """Construct a ``LogEntry`` for the Notifier stage."""
    return LogEntry(
        timestamp=_utcnow(),
        event_type=event_type,
        stage_name="notifier",
        video_id=video_id,
        batch_id=batch_id,
        http_response_code=http_response_code,
        retry_attempt=retry_attempt,
        message=message,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier:
    """Dispatches notifications to configured delivery channels.

    Channels are tried in order: Slack → Discord → Email.  Each channel
    receives up to ``_PER_CHANNEL_ATTEMPTS`` attempts before the fallback
    chain advances to the next channel.

    Deduplication is enforced in-memory.  The key
    ``"{video_id}:{notification_type}:{stage_name or ''}"`` is checked
    against a dispatch registry; if the same key was sent within
    ``_DEDUP_WINDOW`` (10 minutes) the notification is suppressed.

    Args:
        config: ``NotifierConfig`` describing which channels are available.
        log_writer: Optional callable that receives a ``LogEntry`` and persists
            it to the Asset_Store structured log.  When ``None``, log entries
            fall back to the Python ``logging`` module.
    """

    def __init__(
        self,
        config: NotifierConfig,
        log_writer: Optional[Callable[[LogEntry], None]] = None,
    ) -> None:
        self._config = config
        self._log_writer = log_writer
        # dedup_key → dispatched_at (UTC-aware datetime)
        self._dedup_registry: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def send(self, event: NotificationEvent) -> None:
        """Dispatch a pre-built ``NotificationEvent`` through the channel chain."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in an async context — schedule as a fire-and-forget task.
            # Do NOT block with concurrent.futures.wait; that deadlocks the loop.
            asyncio.ensure_future(self._send_async(event), loop=loop)
        else:
            asyncio.run(self._send_async(event))

    def send_review_gate(
        self,
        video_id: str,
        gate_type: Literal["script", "final"],
        asset_links: list[str],
        action_prompt: str,
    ) -> None:
        """Notify the creator that a Review Gate is open and requires action.

        SLA: must dispatch within 60 seconds of gate trigger.

        Args:
            video_id: Pipeline video identifier.
            gate_type: ``"script"`` (Gate 1) or ``"final"`` (Gate 2).
            asset_links: Drive URLs for the assets to be reviewed.
            action_prompt: Human-readable description of the required action.
        """
        gate_label = "Script Review" if gate_type == "script" else "Final Review"
        title = f"[Review Required] {gate_label} for video {video_id}"
        body_lines = [
            f"Video ID: {video_id}",
            f"Gate: {gate_label}",
            "",
            action_prompt,
        ]
        if asset_links:
            body_lines += ["", "Asset links:"] + [f"  • {url}" for url in asset_links]

        event = NotificationEvent(
            event_id=uuid.uuid4(),
            video_id=video_id,
            notification_type="review_gate",
            stage_name=gate_type,
            channel="slack",  # resolved at dispatch time by the fallback chain
            payload=NotificationPayload(
                title=title,
                body="\n".join(body_lines),
                asset_links=asset_links,
                action_prompt=action_prompt,
            ),
            dedup_key=f"{video_id}:review_gate:{gate_type}",
            status="pending",
        )
        self.send(event)

    def send_failure_alert(
        self,
        video_id: str,
        stage_name: str,
        error_message: str,
        publish_datetime: Optional[datetime] = None,
    ) -> None:
        """Alert the creator that a pipeline stage has failed.

        SLA: must dispatch within 60 seconds of failure detection.

        Args:
            video_id: Pipeline video identifier.
            stage_name: Name of the stage that failed (e.g. ``"script_generation"``).
            error_message: Human-readable description of the failure.
            publish_datetime: Scheduled publish datetime, if known.
        """
        title = f"[Pipeline Failure] Stage '{stage_name}' failed for video {video_id}"
        body_lines = [
            f"Video ID: {video_id}",
            f"Failed stage: {stage_name}",
            "",
            f"Error: {error_message}",
        ]
        if publish_datetime is not None:
            body_lines.append(f"Scheduled publish: {publish_datetime.isoformat()}")

        event = NotificationEvent(
            event_id=uuid.uuid4(),
            video_id=video_id,
            notification_type="failure_alert",
            stage_name=stage_name,
            channel="slack",
            payload=NotificationPayload(
                title=title,
                body="\n".join(body_lines),
            ),
            dedup_key=f"{video_id}:failure_alert:{stage_name}",
            status="pending",
        )
        self.send(event)

    def send_batch_summary(
        self,
        batch_id: str,
        results: list[VideoSummary],
    ) -> None:
        """Send a batch completion summary containing exactly N video entries.

        Each entry includes: title, status, and ``scheduled_publish_datetime``.

        Args:
            batch_id: Identifier of the batch run.
            results: Exactly N ``VideoSummary`` entries — one per video in the batch.
        """
        title = f"[Batch Summary] Batch {batch_id} — {len(results)} video(s)"
        body_lines = [f"Batch ID: {batch_id}", f"Total videos: {len(results)}", ""]

        for i, vs in enumerate(results, start=1):
            publish_str = (
                vs.scheduled_publish_datetime.isoformat()
                if vs.scheduled_publish_datetime is not None
                else "not scheduled"
            )
            body_lines += [
                f"{i}. {vs.title}",
                f"   Video ID : {vs.video_id}",
                f"   Status   : {vs.status.value}",
                f"   Publish  : {publish_str}",
                "",
            ]

        # Use the first video_id as the notification anchor; stage_name empty for batch summaries
        anchor_video_id = results[0].video_id if results else ""
        event = NotificationEvent(
            event_id=uuid.uuid4(),
            video_id=anchor_video_id,
            notification_type="batch_summary",
            stage_name=None,
            channel="slack",
            payload=NotificationPayload(
                title=title,
                body="\n".join(body_lines),
            ),
            dedup_key=f"{batch_id}:batch_summary:",
            status="pending",
        )
        self.send(event)

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _send_async(self, event: NotificationEvent) -> None:
        """Core async dispatch with deduplication and channel fallback chain."""
        # ---- 1. No-channel guard ------------------------------------------
        if not self._has_any_channel():
            entry = _build_log_entry(
                event_type="warning",
                message="No notification channel configured",
                video_id=event.video_id,
            )
            self._write_log(entry)
            return

        # ---- 2. Deduplication check ----------------------------------------
        now = _utcnow()
        last_sent = self._dedup_registry.get(event.dedup_key)
        if last_sent is not None and (now - last_sent) < _DEDUP_WINDOW:
            logger.debug(
                "Notification suppressed (dedup): key=%s last_sent=%s",
                event.dedup_key,
                last_sent.isoformat(),
            )
            return

        # ---- 3. Build message body -----------------------------------------
        message_body = self._format_message(event)

        # ---- 4. Try channels in order: Slack → Discord → Email -------------
        channels_tried: list[str] = []
        delivered = False

        for channel_name, attempt_coro in self._channel_attempts(message_body, event):
            channels_tried.append(channel_name)
            success = await self._attempt_with_retry(
                channel_name=channel_name,
                attempt_coro_factory=attempt_coro,
                event=event,
            )
            if success:
                delivered = True
                # Record successful dispatch time for deduplication
                self._dedup_registry[event.dedup_key] = _utcnow()
                break

        # ---- 5. All channels exhausted -------------------------------------
        if not delivered:
            entry = _build_log_entry(
                event_type="error",
                message=(
                    f"All notification channels failed for event "
                    f"dedup_key={event.dedup_key!r}. "
                    f"Channels tried: {channels_tried}"
                ),
                video_id=event.video_id,
                metadata={"dedup_key": event.dedup_key, "channels_tried": channels_tried},
            )
            self._write_log(entry)
            logger.error(
                "Notification delivery failed on all channels: dedup_key=%s",
                event.dedup_key,
            )

    async def _attempt_with_retry(
        self,
        channel_name: str,
        attempt_coro_factory: Callable[[], asyncio.Future[None]],  # type: ignore[type-arg]
        event: NotificationEvent,
    ) -> bool:
        """Attempt delivery on a single channel up to ``_PER_CHANNEL_ATTEMPTS`` times.

        Args:
            channel_name: Human-readable channel identifier for logging.
            attempt_coro_factory: Callable that returns a coroutine performing one delivery.
            event: The notification event being dispatched.

        Returns:
            ``True`` if delivery succeeded within the allowed attempts, ``False`` otherwise.
        """
        for attempt in range(1, _PER_CHANNEL_ATTEMPTS + 1):
            try:
                await attempt_coro_factory()
                logger.debug(
                    "Notification delivered via %s (attempt %d): dedup_key=%s",
                    channel_name,
                    attempt,
                    event.dedup_key,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                entry = _build_log_entry(
                    event_type="retry_attempt",
                    message=(
                        f"Delivery attempt {attempt}/{_PER_CHANNEL_ATTEMPTS} "
                        f"failed on channel '{channel_name}': {exc}"
                    ),
                    video_id=event.video_id,
                    retry_attempt=attempt,
                    metadata={"channel": channel_name, "error": str(exc)},
                )
                self._write_log(entry)
                logger.warning(
                    "Notification delivery attempt %d/%d failed on %s: %s",
                    attempt,
                    _PER_CHANNEL_ATTEMPTS,
                    channel_name,
                    exc,
                )

        # All attempts on this channel exhausted — log and move to next channel
        entry = _build_log_entry(
            event_type="error",
            message=(
                f"Channel '{channel_name}' exhausted after {_PER_CHANNEL_ATTEMPTS} "
                f"attempts for dedup_key={event.dedup_key!r}. Trying next channel."
            ),
            video_id=event.video_id,
            metadata={"channel": channel_name, "dedup_key": event.dedup_key},
        )
        self._write_log(entry)
        return False

    # ------------------------------------------------------------------
    # Channel iterators
    # ------------------------------------------------------------------

    def _channel_attempts(
        self,
        message_body: str,
        event: Optional[NotificationEvent] = None,
    ) -> list[tuple[str, Callable[[], asyncio.Future[None]]]]:  # type: ignore[type-arg]
        """Build ordered (channel_name, coro_factory) pairs.

        For email, uses HTML formatting (with tap buttons) when the event is a
        review_gate notification.
        """
        attempts: list[tuple[str, Callable[[], asyncio.Future[None]]]] = []  # type: ignore[type-arg]

        if self._config.slack_webhook_url:
            slack_url = self._config.slack_webhook_url

            async def _slack(_url: str = slack_url, _body: str = message_body) -> None:
                await self._post_webhook(_url, {"text": _body})

            attempts.append(("slack", _slack))  # type: ignore[arg-type]

        if self._config.discord_webhook_url:
            discord_url = self._config.discord_webhook_url

            async def _discord(_url: str = discord_url, _body: str = message_body) -> None:
                await self._post_webhook(_url, {"content": _body})

            attempts.append(("discord", _discord))  # type: ignore[arg-type]

        if self._config.smtp is not None:
            smtp_cfg = self._config.smtp
            # Use rich HTML for review gate events; plain text for everything else.
            html_body = (
                self._format_html_email(event)
                if event and event.notification_type == "review_gate"
                else None
            )

            async def _email(
                _cfg: SmtpConfig = smtp_cfg,
                _plain: str = message_body,
                _html: Optional[str] = html_body,
            ) -> None:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._send_smtp, _cfg, _plain, _html)

            attempts.append(("email", _email))  # type: ignore[arg-type]

        return attempts

    # ------------------------------------------------------------------
    # Delivery primitives
    # ------------------------------------------------------------------

    async def _post_webhook(self, url: str, payload: dict[str, str]) -> None:
        """POST a JSON payload to a webhook URL using ``httpx.AsyncClient``.

        Args:
            url: Full HTTPS webhook URL.
            payload: JSON-serialisable dictionary (Slack or Discord format).

        Raises:
            httpx.HTTPStatusError: If the server returns a 4xx/5xx response.
            httpx.RequestError: On network-level failures.
        """
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

    def _send_smtp(self, cfg: SmtpConfig, body: str, html_body: Optional[str] = None) -> None:
        """Send an email via SMTP. Sends HTML when html_body is provided.

        Args:
            cfg: SMTP configuration.
            body: Plain-text fallback body.
            html_body: Optional HTML body (shown on modern email clients).
        """
        from email.mime.multipart import MIMEMultipart  # noqa: PLC0415

        if html_body:
            msg: Any = MIMEMultipart("alternative")
            msg.attach(MIMEText(body, "plain", "utf-8"))
            msg.attach(MIMEText(html_body, "html", "utf-8"))
        else:
            msg = MIMEText(body, "plain", "utf-8")

        msg["Subject"] = "AI YouTube Pipeline — Action Required"
        msg["From"]    = cfg.from_address
        msg["To"]      = cfg.to_address

        with smtplib.SMTP(cfg.host, cfg.port) as server:
            server.ehlo()
            if server.has_extn("STARTTLS"):
                server.starttls()
                server.ehlo()
            server.login(cfg.username, cfg.password)
            server.sendmail(cfg.from_address, [cfg.to_address], msg.as_string())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _has_any_channel(self) -> bool:
        """Return ``True`` if at least one delivery channel is configured."""
        return bool(
            self._config.slack_webhook_url
            or self._config.discord_webhook_url
            or self._config.smtp is not None
        )

    @staticmethod
    def _format_message(event: NotificationEvent) -> str:
        """Build a human-readable plain-text message from a ``NotificationEvent``."""
        lines = [
            f"**{event.payload.title}**",
            "",
            event.payload.body,
        ]
        if event.payload.asset_links:
            lines += ["", "Assets:"] + [f"  • {link}" for link in event.payload.asset_links]
        if event.payload.action_prompt:
            lines += ["", f"Action required: {event.payload.action_prompt}"]
        return "\n".join(lines)

    @staticmethod
    def _format_html_email(event: NotificationEvent) -> str:
        """Build an HTML email with tap-friendly approve/edit buttons.

        Parses approve and edit URLs out of the action_prompt so the email
        renders large, mobile-friendly buttons instead of raw links.
        """
        import re  # noqa: PLC0415

        prompt = event.payload.action_prompt or ""
        approve_url = ""
        edit_url = ""
        reject_url = ""

        # Extract URLs from the action prompt written by ReviewGate.trigger()
        approve_match = re.search(r"Tap to approve:\s+(https?://\S+)", prompt)
        edit_match    = re.search(r"Tap to (?:edit script|request re-gen):\s+(https?://\S+)", prompt)
        reject_match  = re.search(r"Tap to reject:\s+(https?://\S+)", prompt)
        if approve_match:
            approve_url = approve_match.group(1)
        if edit_match:
            edit_url = edit_match.group(1)
        if reject_match:
            reject_url = reject_match.group(1)

        asset_links_html = ""
        if event.payload.asset_links:
            items = "".join(
                f'<li style="margin-bottom:8px">'
                f'<a href="{url}" target="_blank" rel="noopener" style="color:#2563eb;word-break:break-all">'
                f'{url}</a></li>'
                for url in event.payload.asset_links
            )
            asset_links_html = (
                f'<p><strong>Assets to review:</strong></p>'
                f'<ul style="padding-left:20px;color:#374151">{items}</ul>'
            )

        buttons_html = ""
        if approve_url or edit_url or reject_url:
            btn_style = (
                "display:inline-block;padding:16px 32px;margin:8px;"
                "border-radius:12px;font-size:18px;font-weight:bold;"
                "text-decoration:none;color:#fff"
            )
            approve_btn = (
                f'<a href="{approve_url}" style="{btn_style};background:#16a34a">✅ Approve</a>'
                if approve_url else ""
            )
            edit_btn = (
                f'<a href="{edit_url}" style="{btn_style};background:#d97706">✏️ Edit Script</a>'
                if edit_url else ""
            )
            reject_btn = (
                f'<a href="{reject_url}" style="{btn_style};background:#dc2626">🚫 Reject & Regenerate</a>'
                if reject_url else ""
            )
            buttons_html = (
                f'<div style="text-align:center;margin:32px 0">'
                f'{approve_btn}{edit_btn}{reject_btn}'
                f'</div>'
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="font-family:-apple-system,BlinkMacSystemFont,sans-serif;
             background:#f9fafb;margin:0;padding:24px">
  <div style="max-width:600px;margin:0 auto;background:#fff;
              border-radius:16px;padding:32px;box-shadow:0 2px 16px rgba(0,0,0,.07)">
    <h2 style="color:#111827;margin-top:0">{event.payload.title}</h2>
    <p style="white-space:pre-wrap;color:#374151">{event.payload.body}</p>
    {asset_links_html}
    {buttons_html}
    <hr style="border:none;border-top:1px solid #e5e7eb;margin:24px 0">
    <p style="color:#9ca3af;font-size:13px">
      AI YouTube Content Pipeline — tap a button above or update the status
      in <a href="https://notion.so" style="color:#6b7280">Notion</a>.
    </p>
  </div>
</body></html>"""

    def _write_log(self, entry: LogEntry) -> None:
        """Persist a ``LogEntry`` via ``log_writer`` or fall back to ``logging``.

        Args:
            entry: Structured log entry to persist.
        """
        if self._log_writer is not None:
            try:
                self._log_writer(entry)
            except Exception as exc:  # noqa: BLE001
                logger.error("log_writer raised an exception: %s", exc)
        else:
            # Degrade gracefully when no structured writer is available
            level = logging.WARNING if entry.event_type == "warning" else logging.ERROR
            logger.log(level, "[%s] %s", entry.event_type, entry.message)


__all__ = [
    "Notifier",
    "NotifierConfig",
    "VideoSummary",
]
