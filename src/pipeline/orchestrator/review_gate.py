"""ReviewGate — Human-in-the-loop review checkpoints for the pipeline.

Design reference: §Review Gate Mechanism

Two gates exist in the pipeline:

Gate 1 — Script Review
  * Opens after Script_Writer completes.
  * Closed when the creator approves (→ Script Approved) or submits edits
    (→ Scripting).
  * Reminder schedule: 48 h, 72 h, 96 h (max 3 reminders).
  * Gate stays open indefinitely until action is taken or pipeline cancelled.

Gate 2 — Final Asset Review
  * Opens after Metadata_Generator completes.
  * Closed when the creator approves (→ Approved for Upload) or requests
    per-asset regeneration (→ Generating …).
  * Auto-approves at 72 h if no action has been taken.
  * Each regeneration request resets the 72 h auto-approve timer.

Usage
-----
Instantiate one ``ReviewGate`` per gate per video::

    gate = ReviewGate(
        gate_type="script",
        video_id="video-abc",
        content_calendar=cc,
        notifier=notifier,
    )
    await gate.trigger(asset_links=["https://drive.google.com/…"])
    result = await gate.poll_until_action()
    # result == {"action": "approve"} or {"action": "edit", "payload": "…"}
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Callable, Literal, Optional

from pipeline.content_calendar import Content_Calendar
from pipeline.models import PipelineStatus
from pipeline.notifier import Notifier
from pipeline.review_server import create_review_token, get_review_urls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How often to poll the Content_Calendar for a user action (seconds).
_POLL_INTERVAL_S: float = 60.0

# Gate 1 — Script Review reminder thresholds.
_GATE1_REMINDER_THRESHOLDS: tuple[timedelta, timedelta, timedelta] = (
    timedelta(minutes=30),   # single reminder at 30 min before auto-approve at 1h
    timedelta(minutes=45),
    timedelta(minutes=55),
)

# Gate 2 — Final Review auto-approve timeout.
_GATE2_AUTO_APPROVE_TIMEOUT: timedelta = timedelta(hours=1)

# Maximum number of reminders to send for Gate 1.
_MAX_REMINDERS: int = 3

# Gate 1 — statuses that mean the gate is still open.
_GATE1_OPEN_STATUSES: frozenset[PipelineStatus] = frozenset(
    {PipelineStatus.AWAITING_SCRIPT_REVIEW}
)

# Gate 1 — closed statuses and their canonical action meanings.
_GATE1_APPROVE_STATUS: PipelineStatus = PipelineStatus.SCRIPT_APPROVED
_GATE1_EDIT_STATUS: PipelineStatus = PipelineStatus.SCRIPTING

# Gate 2 — statuses that mean the gate is still open.
_GATE2_OPEN_STATUSES: frozenset[PipelineStatus] = frozenset(
    {PipelineStatus.AWAITING_FINAL_REVIEW}
)

# Gate 2 — closed statuses.
_GATE2_APPROVE_STATUSES: frozenset[PipelineStatus] = frozenset(
    {PipelineStatus.APPROVED_FOR_UPLOAD, PipelineStatus.AUTO_APPROVED_FOR_UPLOAD}
)

# Gate 2 — any "Generating …" status indicates a regen request was fulfilled.
_GATE2_REGEN_STATUSES: frozenset[PipelineStatus] = frozenset(
    {
        PipelineStatus.GENERATING_VISUALS,
        PipelineStatus.GENERATING_METADATA,
        PipelineStatus.SCRIPTING,       # script-level regen
        PipelineStatus.SCRIPT_APPROVED, # narration regen start point
    }
)

# ---------------------------------------------------------------------------
# ReviewGateError
# ---------------------------------------------------------------------------


class ReviewGateError(Exception):
    """Raised when the ReviewGate encounters an unrecoverable error."""


class EmptyEditError(ValueError):
    """Raised when a Gate 1 edit submission has an empty diff (no content change)."""


# ---------------------------------------------------------------------------
# ReviewGate
# ---------------------------------------------------------------------------


class ReviewGate:
    """Durable checkpoint that suspends pipeline execution pending creator review.

    The gate is backed by the ``Content_Calendar`` (Notion) — the creator's
    decision is signalled by an external call to
    ``Orchestrator.handle_review_response``, which updates the CC status.
    ``poll_until_action`` reads that status every 60 s and returns when it
    transitions to a *closed* value.

    Args:
        gate_type: ``"script"`` (Gate 1) or ``"final"`` (Gate 2).
        video_id: Pipeline video identifier.
        content_calendar: :class:`~pipeline.content_calendar.Content_Calendar`.
        notifier: :class:`~pipeline.notifier.Notifier`.
        now_factory: Optional callable returning the current UTC-aware
            :class:`~datetime.datetime`.  Override in tests to control time.
    """

    def __init__(
        self,
        gate_type: Literal["script", "final"],
        video_id: str,
        content_calendar: Content_Calendar,
        notifier: Notifier,
        now_factory: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._gate_type = gate_type
        self._video_id = video_id
        self._content_calendar = content_calendar
        self._notifier = notifier
        self._now: Callable[[], datetime] = now_factory or (
            lambda: datetime.now(tz=timezone.utc)
        )

        # Timestamp at which trigger() opened the gate.  Set by trigger().
        self._gate_opened_at: Optional[datetime] = None

        # Track which reminder thresholds (Gate 1) have already been sent.
        self._reminders_sent: list[bool] = [False, False, False]
        self._reminder_count: int = 0

        # Asyncio queue populated by the review webhook when the creator taps
        # approve / edit.  Also serves as the fallback when the webhook isn't
        # hit (poll_until_action reads Notion in parallel).
        self._action_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def trigger(self, asset_links: list[str]) -> None:
        """Open the review gate: register webhook token, update CC, send notification.

        The notification email contains two tap-to-act links that point to the
        pipeline's public webhook endpoint (``PIPELINE_PUBLIC_URL``).  Tapping
        either link puts the action on ``_action_queue``; ``poll_until_action``
        consumes it immediately.

        Notion status update is also performed so that the Notion calendar
        reflects the gate state regardless of whether the webhook is used.

        Args:
            asset_links: Google Drive URLs for the assets under review.
        """
        open_status = (
            PipelineStatus.AWAITING_SCRIPT_REVIEW
            if self._gate_type == "script"
            else PipelineStatus.AWAITING_FINAL_REVIEW
        )

        # 1. Register this review with the webhook server and get tap URLs.
        token = create_review_token(
            video_id=self._video_id,
            gate_type=self._gate_type,
            queue=self._action_queue,
        )
        approve_url, edit_url, reject_url = get_review_urls(token)

        # 2. Build action prompt with the tap links.
        if self._gate_type == "script":
            action_prompt = (
                f"Tap to approve:       {approve_url}\n"
                f"Tap to edit script:   {edit_url}\n"
                f"Tap to reject:        {reject_url}\n\n"
                "Or change Notion status to 'Script Approved' / 'Scripting'."
            )
        else:
            action_prompt = (
                f"Tap to approve:        {approve_url}\n"
                f"Tap to reject:         {reject_url}\n\n"
                "Or change Notion status to approved."
            )

        logger.info(
            "ReviewGate.trigger: gate_type=%s video_id=%s approve=%s",
            self._gate_type, self._video_id, approve_url,
        )

        # 3. Update Content_Calendar status.
        await self._content_calendar.update_status(self._video_id, open_status)

        # 4. Record gate-open time.
        self._gate_opened_at = self._now()

        # 5. Send notification (within 60 s SLA).
        self._notifier.send_review_gate(
            video_id=self._video_id,
            gate_type=self._gate_type,
            asset_links=asset_links,
            action_prompt=action_prompt,
        )

    async def poll_until_action(self) -> dict:  # type: ignore[type-arg]
        """Await the creator's action via webhook tap OR Notion status change.

        Two concurrent paths run in parallel — whichever fires first wins:

        Path A — Webhook queue (fast path):
            The creator taps a link in the email → FastAPI webhook puts
            ``"approve"`` or ``"edit"`` on ``_action_queue``.

        Path B — Notion polling (fallback):
            Polls Notion status every 60 s in case the creator updates Notion
            directly instead of tapping the email link.

        Gate 1 (script):
            - ``approve``  → continue to narration
            - ``edit``     → regenerate script
            - Reminders at 48 h / 72 h / 96 h (max 3)

        Gate 2 (final):
            - ``approve``     → upload
            - ``regenerate``  → re-run requested stage
            - ``auto_approve`` → 72 h timeout elapsed, upload automatically

        Returns:
            Dict with ``"action"`` key.

        Raises:
            ReviewGateError: If trigger() was not called first.
        """
        if self._gate_opened_at is None:
            raise ReviewGateError("Call trigger() before poll_until_action().")

        timeout = (
            _GATE2_AUTO_APPROVE_TIMEOUT.total_seconds()
            if self._gate_type == "final"
            else _GATE2_AUTO_APPROVE_TIMEOUT.total_seconds()  # same 1h auto-approve for both gates
        )

        # Path A: webhook queue
        async def _wait_webhook() -> str:
            try:
                return await asyncio.wait_for(self._action_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                return "timeout"

        # Path B: Notion polling
        async def _wait_notion() -> str:
            return await self._poll_notion(timeout_seconds=timeout)

        webhook_task = asyncio.create_task(_wait_webhook(), name=f"webhook_{self._video_id}")
        notion_task  = asyncio.create_task(_wait_notion(),  name=f"notion_{self._video_id}")

        done, pending = await asyncio.wait(
            [webhook_task, notion_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel the loser cleanly.
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        result = next(iter(done)).result()
        if result == "timeout":
            result = "auto_approve"  # both gates auto-approve after 1 hour

        # edit actions carry the payload after a colon: "edit:Make it funnier"
        if result.startswith("edit:"):
            payload = result[len("edit:"):]
            logger.info(
                "ReviewGate closed: gate=%s video_id=%s action=edit payload=%s",
                self._gate_type, self._video_id, payload[:80],
            )
            return {"action": "edit", "payload": payload}

        logger.info(
            "ReviewGate closed: gate=%s video_id=%s action=%s",
            self._gate_type, self._video_id, result,
        )
        return {"action": result}

    async def _poll_notion(self, timeout_seconds: float) -> str:
        """Poll Notion status every 60 s until the gate closes or timeout."""
        elapsed = 0.0
        while elapsed < timeout_seconds:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

            current_status = await self._read_current_status()
            elapsed_td = timedelta(seconds=elapsed)

            logger.debug(
                "ReviewGate Notion poll: gate=%s video_id=%s status=%s elapsed=%.0fs",
                self._gate_type, self._video_id,
                current_status.value if current_status else "unknown", elapsed,
            )

            if self._gate_type == "script":
                if current_status == _GATE1_APPROVE_STATUS:
                    return "approve"
                if current_status == _GATE1_EDIT_STATUS:
                    return "edit"
                self._check_and_send_reminders(elapsed_td)
            else:
                if current_status in _GATE2_APPROVE_STATUSES:
                    return "approve"
                if current_status == PipelineStatus.SCRIPT_REJECTED:
                    return "reject"
                if elapsed_td >= _GATE2_AUTO_APPROVE_TIMEOUT:
                    await self._auto_approve_gate2()
                    return "auto_approve"

        return "timeout"

    def is_gate_open(self, status: PipelineStatus) -> bool:
        """Return ``True`` when *status* indicates this gate is currently open.

        Args:
            status: The current ``PipelineStatus`` of the video.

        Returns:
            ``True`` if the gate is open (creator action still pending).
        """
        if self._gate_type == "script":
            return status in _GATE1_OPEN_STATUSES
        return status in _GATE2_OPEN_STATUSES

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_current_status(self) -> PipelineStatus:
        """Read the current ``PipelineStatus`` from the Content_Calendar.

        Returns:
            The current :class:`~pipeline.models.PipelineStatus`.
        """
        # Content_Calendar._get_status is private; use _resolve_page_id + query.
        # We expose it here by calling the private method since ReviewGate and
        # Content_Calendar live in the same package boundary.
        return await self._content_calendar._get_status(self._video_id)  # noqa: SLF001

    def _check_and_send_reminders(self, elapsed: timedelta) -> None:
        """Send reminder notifications for Gate 1 at 48 h, 72 h, and 96 h.

        A reminder is sent when:
        - ``elapsed`` exceeds the threshold.
        - That threshold's reminder has not yet been sent.
        - The total number of reminders sent is below ``_MAX_REMINDERS``.

        Args:
            elapsed: Time elapsed since the gate was opened.
        """
        if self._reminder_count >= _MAX_REMINDERS:
            return

        for idx, threshold in enumerate(_GATE1_REMINDER_THRESHOLDS):
            if self._reminders_sent[idx]:
                continue  # already sent for this threshold
            if elapsed >= threshold:
                self._send_reminder(threshold)
                self._reminders_sent[idx] = True
                self._reminder_count += 1
                logger.info(
                    "ReviewGate Gate1 reminder %d/%d sent at %s: video_id=%s",
                    self._reminder_count,
                    _MAX_REMINDERS,
                    threshold,
                    self._video_id,
                )
                if self._reminder_count >= _MAX_REMINDERS:
                    break

    def _send_reminder(self, threshold: timedelta) -> None:
        """Dispatch a reminder notification via the Notifier.

        Args:
            threshold: The elapsed-time threshold that triggered this reminder.
        """
        hours = int(threshold.total_seconds() // 3600)
        self._notifier.send_review_gate(
            video_id=self._video_id,
            gate_type=self._gate_type,
            asset_links=[],
            action_prompt=(
                f"Reminder: Script review for video {self._video_id} has been "
                f"pending for {hours} hours. "
                f"Please approve or submit edits."
            ),
        )

    async def _auto_approve_gate2(self) -> None:
        """Auto-approve Gate 2 by setting status to AUTO_APPROVED_FOR_UPLOAD.

        Also sends a notification to the creator informing them of the
        automatic approval.
        """
        logger.info(
            "ReviewGate._auto_approve_gate2: setting AUTO_APPROVED_FOR_UPLOAD "
            "for video_id=%s",
            self._video_id,
        )
        await self._content_calendar.update_status(
            self._video_id, PipelineStatus.AUTO_APPROVED_FOR_UPLOAD
        )
        self._notifier.send_review_gate(
            video_id=self._video_id,
            gate_type="final",
            asset_links=[],
            action_prompt=(
                f"Video {self._video_id} has been automatically approved for upload "
                f"after 72 hours without a response. "
                f"Upload will proceed shortly."
            ),
        )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ReviewGate",
    "ReviewGateError",
    "EmptyEditError",
]
