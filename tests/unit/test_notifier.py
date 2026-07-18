"""Unit tests for pipeline.notifier (Tasks 4.1 and 4.2).

Tests cover:
- send_review_gate dispatches a Slack/Discord/SMTP notification (4.1)
- send_failure_alert dispatches a notification with error details (4.1)
- send_batch_summary includes all N entries with required fields (4.1)
- No channel configured → suppresses notification and writes warning log (4.1)
- Fallback chain: primary channel fails → next channel used (4.2)
- Deduplication: second call within 10-minute window is suppressed (4.2)
- Deduplication: call outside 10-minute window is dispatched (4.2)
- All channels fail → delivery failure logged, no exception raised (4.2)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.models import LogEntry, PipelineStatus, SmtpConfig
from pipeline.notifier import Notifier, NotifierConfig, VideoSummary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_slack_config() -> NotifierConfig:
    return NotifierConfig(slack_webhook_url="https://hooks.slack.com/test")


def _make_all_channels_config() -> NotifierConfig:
    return NotifierConfig(
        slack_webhook_url="https://hooks.slack.com/test",
        discord_webhook_url="https://discord.com/api/webhooks/test",
        smtp=SmtpConfig(
            host="smtp.example.com",
            port=587,
            username="user",
            password="pass",
            from_address="from@example.com",
            to_address="to@example.com",
        ),
    )


def _make_video_summaries(n: int = 3) -> list[VideoSummary]:
    return [
        VideoSummary(
            video_id=f"video-{i}",
            title=f"Title {i}",
            status=PipelineStatus.SCHEDULED,
            scheduled_publish_datetime=_utcnow() + timedelta(days=i),
        )
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# Task 4.1 — Interface methods
# ---------------------------------------------------------------------------


class TestSendReviewGate:
    """send_review_gate dispatches the notification through the first available channel."""

    def test_dispatches_via_slack(self) -> None:
        config = _make_slack_config()
        notifier = Notifier(config)

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
            notifier.send_review_gate(
                video_id="vid-001",
                gate_type="script",
                asset_links=["https://drive.google.com/script"],
                action_prompt="Please review and approve the script.",
            )

        mock_post.assert_awaited_once()
        url, payload = mock_post.await_args.args
        assert url == "https://hooks.slack.com/test"
        assert "text" in payload
        assert "vid-001" in payload["text"]

    def test_review_gate_dedup_key_contains_gate_type(self) -> None:
        """The dedup key must encode gate_type to distinguish script vs final gates."""
        config = _make_slack_config()
        notifier = Notifier(config)

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock):
            notifier.send_review_gate("v1", "script", [], "approve")
            notifier.send_review_gate("v1", "final", [], "approve")

        # Both should have been dispatched (different dedup keys)
        assert "v1:review_gate:script" in notifier._dedup_registry
        assert "v1:review_gate:final" in notifier._dedup_registry


class TestSendFailureAlert:
    """send_failure_alert dispatches an alert with error details."""

    def test_dispatches_failure_alert(self) -> None:
        config = _make_slack_config()
        notifier = Notifier(config)

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
            notifier.send_failure_alert(
                video_id="vid-002",
                stage_name="script_generation",
                error_message="Word count out of range",
                publish_datetime=_utcnow() + timedelta(days=7),
            )

        mock_post.assert_awaited_once()
        payload = mock_post.await_args.args[1]
        assert "vid-002" in payload["text"]
        assert "script_generation" in payload["text"]
        assert "Word count out of range" in payload["text"]

    def test_failure_alert_without_publish_datetime(self) -> None:
        """publish_datetime is optional; omitting it should not raise."""
        config = _make_slack_config()
        notifier = Notifier(config)

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
            notifier.send_failure_alert(
                video_id="vid-003",
                stage_name="narration",
                error_message="ElevenLabs API error",
            )

        mock_post.assert_awaited_once()


class TestSendBatchSummary:
    """send_batch_summary includes exactly N entries with title, status, and scheduled_publish_datetime."""

    def test_includes_all_n_entries(self) -> None:
        config = _make_slack_config()
        notifier = Notifier(config)
        summaries = _make_video_summaries(4)

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
            notifier.send_batch_summary("batch-001", summaries)

        payload = mock_post.await_args.args[1]
        message = payload["text"]

        for vs in summaries:
            assert vs.title in message
            assert vs.status.value in message

    def test_includes_scheduled_datetime_for_each_entry(self) -> None:
        config = _make_slack_config()
        notifier = Notifier(config)
        summaries = _make_video_summaries(2)

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
            notifier.send_batch_summary("batch-002", summaries)

        payload = mock_post.await_args.args[1]
        message = payload["text"]

        for vs in summaries:
            assert vs.scheduled_publish_datetime is not None
            assert vs.scheduled_publish_datetime.isoformat() in message

    def test_includes_none_publish_datetime_gracefully(self) -> None:
        """Videos with no scheduled datetime should show 'not scheduled'."""
        config = _make_slack_config()
        notifier = Notifier(config)
        summaries = [
            VideoSummary(
                video_id="v-none",
                title="No Schedule",
                status=PipelineStatus.UNLISTED,
                scheduled_publish_datetime=None,
            )
        ]

        with patch.object(notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
            notifier.send_batch_summary("batch-003", summaries)

        payload = mock_post.await_args.args[1]
        assert "not scheduled" in payload["text"]


class TestNoChannelConfigured:
    """If no channel is configured, notification is suppressed and a warning is logged."""

    def test_suppresses_and_logs_warning(self) -> None:
        config = NotifierConfig()  # all None
        log_entries: list[LogEntry] = []
        notifier = Notifier(config, log_writer=log_entries.append)

        notifier.send_review_gate("v-empty", "script", [], "approve")

        assert len(log_entries) == 1
        entry = log_entries[0]
        assert entry.event_type == "warning"
        assert "No notification channel configured" in entry.message

    def test_no_exception_raised(self) -> None:
        config = NotifierConfig()
        notifier = Notifier(config)
        # Should not raise even without log_writer
        notifier.send_failure_alert("v-empty", "upload", "oops")


# ---------------------------------------------------------------------------
# Task 4.2 — Deduplication and fallback chain
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Deduplication suppresses duplicate notifications within 10 minutes."""

    def test_second_call_within_window_is_suppressed(self) -> None:
        config = _make_slack_config()
        notifier = Notifier(config)

        call_count = 0

        async def fake_post(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1

        with patch.object(notifier, "_post_webhook", side_effect=fake_post):
            notifier.send_failure_alert("v-dedup", "stage-x", "error 1")
            notifier.send_failure_alert("v-dedup", "stage-x", "error 2")

        assert call_count == 1, "Second notification within window should be suppressed"

    def test_call_after_window_expires_is_dispatched(self) -> None:
        config = _make_slack_config()
        notifier = Notifier(config)
        dedup_key = "v-dedup2:failure_alert:stage-y"

        # Pre-seed registry with a timestamp older than 10 minutes
        notifier._dedup_registry[dedup_key] = _utcnow() - timedelta(minutes=11)

        call_count = 0

        async def fake_post(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1

        with patch.object(notifier, "_post_webhook", side_effect=fake_post):
            notifier.send_failure_alert("v-dedup2", "stage-y", "error after window")

        assert call_count == 1, "Notification after window expiry should be dispatched"

    def test_different_stage_names_not_deduplicated(self) -> None:
        """Different stage_name values produce different dedup keys."""
        config = _make_slack_config()
        notifier = Notifier(config)
        call_count = 0

        async def fake_post(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1

        with patch.object(notifier, "_post_webhook", side_effect=fake_post):
            notifier.send_failure_alert("v-dd3", "stage-a", "err")
            notifier.send_failure_alert("v-dd3", "stage-b", "err")

        assert call_count == 2


class TestFallbackChain:
    """On per-channel failure after 2 attempts, the next channel is tried."""

    def test_slack_fail_uses_discord(self) -> None:
        config = NotifierConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            discord_webhook_url="https://discord.com/api/webhooks/test",
        )
        notifier = Notifier(config)
        discord_called = False

        async def fail_slack(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            nonlocal discord_called
            if "slack" in url:
                raise httpx.ConnectError("Slack down")
            discord_called = True

        import httpx  # noqa: PLC0415 — local import to avoid top-level collision

        async def failing_slack(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            nonlocal discord_called
            if "slack" in url:
                raise httpx.ConnectError("Slack down")
            discord_called = True

        with patch.object(notifier, "_post_webhook", side_effect=failing_slack):
            notifier.send_failure_alert("v-fb", "upload", "slack is down")

        assert discord_called, "Discord should have been called after Slack failed"

    def test_all_channels_fail_logs_error_no_exception(self) -> None:
        config = _make_all_channels_config()
        log_entries: list[LogEntry] = []
        notifier = Notifier(config, log_writer=log_entries.append)

        import httpx as _httpx  # noqa: PLC0415

        async def always_fail(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            raise _httpx.ConnectError("network failure")

        def smtp_fail(cfg: SmtpConfig, body: str) -> None:
            raise OSError("SMTP failure")

        with (
            patch.object(notifier, "_post_webhook", side_effect=always_fail),
            patch.object(notifier, "_send_smtp", side_effect=smtp_fail),
        ):
            # Should not raise
            notifier.send_failure_alert("v-allfail", "upload", "all down")

        # One final "all channels failed" error entry should be present
        error_entries = [e for e in log_entries if e.event_type == "error"]
        assert any("All notification channels failed" in e.message for e in error_entries)

    def test_each_channel_attempted_twice_before_fallback(self) -> None:
        """Slack should be tried exactly 2 times before Discord is tried."""
        config = NotifierConfig(
            slack_webhook_url="https://hooks.slack.com/test",
            discord_webhook_url="https://discord.com/api/webhooks/test",
        )
        notifier = Notifier(config)
        slack_attempts = 0
        discord_attempts = 0

        import httpx as _httpx  # noqa: PLC0415

        async def count_attempts(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            nonlocal slack_attempts, discord_attempts
            if "slack" in url:
                slack_attempts += 1
                raise _httpx.ConnectError("slack fail")
            else:
                discord_attempts += 1

        with patch.object(notifier, "_post_webhook", side_effect=count_attempts):
            notifier.send_failure_alert("v-retry", "stage", "err")

        assert slack_attempts == 2, f"Expected 2 Slack attempts, got {slack_attempts}"
        assert discord_attempts == 1, f"Expected 1 Discord attempt (success), got {discord_attempts}"

    def test_discord_format_uses_content_key(self) -> None:
        """Discord payload must use 'content' key; Slack uses 'text'."""
        config = NotifierConfig(
            discord_webhook_url="https://discord.com/api/webhooks/test"
        )
        notifier = Notifier(config)
        captured_payload: dict = {}  # type: ignore[type-arg]

        async def capture(url: str, payload: dict) -> None:  # type: ignore[type-arg]
            captured_payload.update(payload)

        with patch.object(notifier, "_post_webhook", side_effect=capture):
            notifier.send_failure_alert("v-discord", "stage", "err")

        assert "content" in captured_payload, "Discord payload must use 'content' key"
        assert "text" not in captured_payload
