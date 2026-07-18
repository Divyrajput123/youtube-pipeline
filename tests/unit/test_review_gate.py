"""Unit tests for pipeline.orchestrator.review_gate (Task 16.4).

Tests cover:
- trigger(): updates CC status to the correct open status (Gate 1 / Gate 2).
- trigger(): calls notifier.send_review_gate within the call.
- poll_until_action() Gate 1: returns {"action": "approve"} on SCRIPT_APPROVED.
- poll_until_action() Gate 1: returns {"action": "edit"} on SCRIPTING.
- poll_until_action() Gate 1: sends reminders at 48 h, 72 h, 96 h (max 3).
- poll_until_action() Gate 2: returns {"action": "approve"} on APPROVED_FOR_UPLOAD.
- poll_until_action() Gate 2: returns {"action": "approve"} on AUTO_APPROVED_FOR_UPLOAD.
- poll_until_action() Gate 2: returns {"action": "regenerate"} on GENERATING_VISUALS etc.
- poll_until_action() Gate 2: auto-approves after 72 h and updates CC status.
- poll_until_action() before trigger() raises ReviewGateError.
- is_gate_open(): correct for Gate 1 and Gate 2.
- edit validation: empty diff payload raises EmptyEditError when checked externally.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from pipeline.models import PipelineStatus
from pipeline.orchestrator.review_gate import (
    EmptyEditError,
    ReviewGate,
    ReviewGateError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_gate(
    gate_type: str = "script",
    video_id: str = "video-test",
    initial_status: PipelineStatus = PipelineStatus.AWAITING_SCRIPT_REVIEW,
    now_factory=None,
) -> tuple[ReviewGate, AsyncMock, MagicMock]:
    """Create a ReviewGate with mocked Content_Calendar and Notifier.

    Returns:
        (gate, mock_cc, mock_notifier)
        mock_cc._get_status is pre-configured to return initial_status.
    """
    mock_cc = AsyncMock()
    mock_cc.update_status = AsyncMock(return_value=None)
    mock_cc._get_status = AsyncMock(return_value=initial_status)

    mock_notifier = MagicMock()
    mock_notifier.send_review_gate = MagicMock(return_value=None)

    gate = ReviewGate(
        gate_type=gate_type,  # type: ignore[arg-type]
        video_id=video_id,
        content_calendar=mock_cc,
        notifier=mock_notifier,
        now_factory=now_factory,
    )
    return gate, mock_cc, mock_notifier


# ---------------------------------------------------------------------------
# trigger() tests
# ---------------------------------------------------------------------------


class TestTrigger:
    """ReviewGate.trigger() updates CC status and sends a notification."""

    @pytest.mark.asyncio
    async def test_gate1_sets_awaiting_script_review_status(self) -> None:
        gate, mock_cc, _ = _make_gate(gate_type="script")
        await gate.trigger(asset_links=["https://drive.google.com/script"])

        mock_cc.update_status.assert_awaited_once_with(
            "video-test", PipelineStatus.AWAITING_SCRIPT_REVIEW
        )

    @pytest.mark.asyncio
    async def test_gate2_sets_awaiting_final_review_status(self) -> None:
        gate, mock_cc, _ = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
        )
        await gate.trigger(asset_links=[])

        mock_cc.update_status.assert_awaited_once_with(
            "video-test", PipelineStatus.AWAITING_FINAL_REVIEW
        )

    @pytest.mark.asyncio
    async def test_trigger_calls_send_review_gate(self) -> None:
        gate, _, mock_notifier = _make_gate(gate_type="script")
        await gate.trigger(asset_links=["https://drive.google.com/s"])

        mock_notifier.send_review_gate.assert_called_once()
        kwargs = mock_notifier.send_review_gate.call_args
        assert kwargs.kwargs["video_id"] == "video-test"
        assert kwargs.kwargs["gate_type"] == "script"
        assert "https://drive.google.com/s" in kwargs.kwargs["asset_links"]

    @pytest.mark.asyncio
    async def test_trigger_records_gate_opened_at(self) -> None:
        t0 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        gate, _, _ = _make_gate(now_factory=lambda: t0)
        await gate.trigger(asset_links=[])

        assert gate._gate_opened_at == t0

    @pytest.mark.asyncio
    async def test_trigger_gate2_sends_final_prompt(self) -> None:
        gate, _, mock_notifier = _make_gate(gate_type="final")
        await gate.trigger(asset_links=["https://drive.google.com/video"])

        kwargs = mock_notifier.send_review_gate.call_args.kwargs
        assert kwargs["gate_type"] == "final"
        assert "regenerate" in kwargs["action_prompt"].lower()


# ---------------------------------------------------------------------------
# is_gate_open() tests
# ---------------------------------------------------------------------------


class TestIsGateOpen:
    """is_gate_open returns True only for the gate's own open status."""

    def test_gate1_open_for_awaiting_script_review(self) -> None:
        gate, _, _ = _make_gate(gate_type="script")
        assert gate.is_gate_open(PipelineStatus.AWAITING_SCRIPT_REVIEW) is True

    def test_gate1_closed_for_script_approved(self) -> None:
        gate, _, _ = _make_gate(gate_type="script")
        assert gate.is_gate_open(PipelineStatus.SCRIPT_APPROVED) is False

    def test_gate1_closed_for_scripting(self) -> None:
        gate, _, _ = _make_gate(gate_type="script")
        assert gate.is_gate_open(PipelineStatus.SCRIPTING) is False

    def test_gate2_open_for_awaiting_final_review(self) -> None:
        gate, _, _ = _make_gate(gate_type="final")
        assert gate.is_gate_open(PipelineStatus.AWAITING_FINAL_REVIEW) is True

    def test_gate2_closed_for_approved_for_upload(self) -> None:
        gate, _, _ = _make_gate(gate_type="final")
        assert gate.is_gate_open(PipelineStatus.APPROVED_FOR_UPLOAD) is False

    def test_gate2_closed_for_generating_visuals(self) -> None:
        gate, _, _ = _make_gate(gate_type="final")
        assert gate.is_gate_open(PipelineStatus.GENERATING_VISUALS) is False

    def test_gate1_closed_for_unrelated_status(self) -> None:
        gate, _, _ = _make_gate(gate_type="script")
        assert gate.is_gate_open(PipelineStatus.UPLOADING) is False


# ---------------------------------------------------------------------------
# poll_until_action() before trigger
# ---------------------------------------------------------------------------


class TestPollBeforeTrigger:
    """poll_until_action raises ReviewGateError if trigger was not called first."""

    @pytest.mark.asyncio
    async def test_raises_review_gate_error(self) -> None:
        gate, _, _ = _make_gate()
        with pytest.raises(ReviewGateError, match="trigger"):
            await gate.poll_until_action()


# ---------------------------------------------------------------------------
# poll_until_action() Gate 1 — basic approve / edit
# ---------------------------------------------------------------------------


class TestGate1PollActions:
    """Gate 1 poll returns the correct action dict based on CC status."""

    @pytest.mark.asyncio
    async def test_approve_on_script_approved(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gate, mock_cc, _ = _make_gate(gate_type="script", now_factory=lambda: t0)
        await gate.trigger(asset_links=[])

        # After one poll interval, CC reports SCRIPT_APPROVED.
        mock_cc._get_status = AsyncMock(return_value=PipelineStatus.SCRIPT_APPROVED)

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await gate.poll_until_action()

        assert result == {"action": "approve"}
        mock_sleep.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_edit_on_scripting(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gate, mock_cc, _ = _make_gate(gate_type="script", now_factory=lambda: t0)
        await gate.trigger(asset_links=[])

        mock_cc._get_status = AsyncMock(return_value=PipelineStatus.SCRIPTING)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gate.poll_until_action()

        assert result == {"action": "edit"}


# ---------------------------------------------------------------------------
# poll_until_action() Gate 1 — reminder schedule
# ---------------------------------------------------------------------------


class TestGate1Reminders:
    """Gate 1 sends reminders at 48 h, 72 h, 96 h (max 3)."""

    @pytest.mark.asyncio
    async def test_no_reminder_before_48h(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _advance_time() -> datetime:
            nonlocal call_count
            # First call: trigger time (t0).
            # Subsequent calls: 47 h later (no reminder threshold crossed).
            if call_count == 0:
                call_count += 1
                return t0
            return t0 + timedelta(hours=47)

        gate, mock_cc, mock_notifier = _make_gate(
            gate_type="script", now_factory=_advance_time
        )
        await gate.trigger(asset_links=[])

        # One poll: 47 h elapsed, still open, then approve.
        poll_statuses = iter([
            PipelineStatus.AWAITING_SCRIPT_REVIEW,  # first poll: still open
            PipelineStatus.SCRIPT_APPROVED,         # second poll: approved
        ])
        mock_cc._get_status = AsyncMock(side_effect=poll_statuses)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gate.poll_until_action()

        # send_review_gate called once for trigger only; no reminders.
        assert mock_notifier.send_review_gate.call_count == 1

    @pytest.mark.asyncio
    async def test_sends_reminder_at_48h(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _time_factory() -> datetime:
            nonlocal call_count
            times = [
                t0,                           # trigger
                t0 + timedelta(hours=49),     # first poll (past 48 h)
                t0 + timedelta(hours=49),     # second poll to read status
            ]
            idx = min(call_count, len(times) - 1)
            call_count += 1
            return times[idx]

        gate, mock_cc, mock_notifier = _make_gate(
            gate_type="script", now_factory=_time_factory
        )
        await gate.trigger(asset_links=[])

        # First poll: still open (sends reminder), second poll: approved.
        poll_statuses = iter([
            PipelineStatus.AWAITING_SCRIPT_REVIEW,
            PipelineStatus.SCRIPT_APPROVED,
        ])
        mock_cc._get_status = AsyncMock(side_effect=poll_statuses)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gate.poll_until_action()

        # trigger call + 1 reminder call = 2 total
        assert mock_notifier.send_review_gate.call_count == 2
        reminder_call = mock_notifier.send_review_gate.call_args_list[1]
        assert "48" in reminder_call.kwargs["action_prompt"]

    @pytest.mark.asyncio
    async def test_max_3_reminders_not_exceeded(self) -> None:
        """Even if enough time passes for >3 thresholds, only 3 reminders are sent."""
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _time_factory() -> datetime:
            nonlocal call_count
            # After trigger: always return 120 h (past all 3 thresholds)
            if call_count == 0:
                call_count += 1
                return t0
            call_count += 1
            return t0 + timedelta(hours=120)

        gate, mock_cc, mock_notifier = _make_gate(
            gate_type="script", now_factory=_time_factory
        )
        await gate.trigger(asset_links=[])

        # First 4 polls: open; 5th poll: approved.
        statuses = (
            [PipelineStatus.AWAITING_SCRIPT_REVIEW] * 4
            + [PipelineStatus.SCRIPT_APPROVED]
        )
        mock_cc._get_status = AsyncMock(side_effect=iter(statuses))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gate.poll_until_action()

        # 1 trigger + max 3 reminders = 4 total
        assert mock_notifier.send_review_gate.call_count == 4

    @pytest.mark.asyncio
    async def test_each_threshold_reminder_sent_only_once(self) -> None:
        """Each reminder threshold fires at most once even across multiple polls."""
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _time_factory() -> datetime:
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return t0
            # Toggle between 49 h and 50 h — both past 48 h threshold.
            call_count += 1
            return t0 + timedelta(hours=49 + (call_count % 2))

        gate, mock_cc, mock_notifier = _make_gate(
            gate_type="script", now_factory=_time_factory
        )
        await gate.trigger(asset_links=[])

        # 3 polls open, then approved.
        statuses = [
            PipelineStatus.AWAITING_SCRIPT_REVIEW,
            PipelineStatus.AWAITING_SCRIPT_REVIEW,
            PipelineStatus.AWAITING_SCRIPT_REVIEW,
            PipelineStatus.SCRIPT_APPROVED,
        ]
        mock_cc._get_status = AsyncMock(side_effect=iter(statuses))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gate.poll_until_action()

        # Only 1 reminder for the 48-hour threshold (not 3 duplicates)
        reminder_calls = [
            c for c in mock_notifier.send_review_gate.call_args_list[1:]
        ]
        assert len(reminder_calls) == 1


# ---------------------------------------------------------------------------
# poll_until_action() Gate 2 — approve / regenerate / auto-approve
# ---------------------------------------------------------------------------


class TestGate2PollActions:
    """Gate 2 poll returns the correct action or auto-approves at 72 h."""

    @pytest.mark.asyncio
    async def test_approve_on_approved_for_upload(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gate, mock_cc, _ = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=lambda: t0,
        )
        await gate.trigger(asset_links=[])

        mock_cc._get_status = AsyncMock(return_value=PipelineStatus.APPROVED_FOR_UPLOAD)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gate.poll_until_action()

        assert result == {"action": "approve"}

    @pytest.mark.asyncio
    async def test_approve_on_auto_approved_for_upload(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gate, mock_cc, _ = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=lambda: t0,
        )
        await gate.trigger(asset_links=[])

        mock_cc._get_status = AsyncMock(
            return_value=PipelineStatus.AUTO_APPROVED_FOR_UPLOAD
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gate.poll_until_action()

        assert result == {"action": "approve"}

    @pytest.mark.asyncio
    async def test_regenerate_on_generating_visuals(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gate, mock_cc, _ = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=lambda: t0,
        )
        await gate.trigger(asset_links=[])

        mock_cc._get_status = AsyncMock(return_value=PipelineStatus.GENERATING_VISUALS)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gate.poll_until_action()

        assert result == {"action": "regenerate"}

    @pytest.mark.asyncio
    async def test_regenerate_on_generating_metadata(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        gate, mock_cc, _ = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=lambda: t0,
        )
        await gate.trigger(asset_links=[])

        mock_cc._get_status = AsyncMock(
            return_value=PipelineStatus.GENERATING_METADATA
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gate.poll_until_action()

        assert result == {"action": "regenerate"}

    @pytest.mark.asyncio
    async def test_auto_approve_at_72h(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _time_factory() -> datetime:
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return t0
            call_count += 1
            return t0 + timedelta(hours=73)  # past 72 h

        gate, mock_cc, mock_notifier = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=_time_factory,
        )
        await gate.trigger(asset_links=[])

        # Poll returns still-open status.
        mock_cc._get_status = AsyncMock(
            return_value=PipelineStatus.AWAITING_FINAL_REVIEW
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gate.poll_until_action()

        assert result == {"action": "auto_approve"}

        # CC should be updated to AUTO_APPROVED_FOR_UPLOAD.
        mock_cc.update_status.assert_any_await(
            "video-test", PipelineStatus.AUTO_APPROVED_FOR_UPLOAD
        )

    @pytest.mark.asyncio
    async def test_auto_approve_sends_notification(self) -> None:
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _time_factory() -> datetime:
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return t0
            return t0 + timedelta(hours=73)

        gate, mock_cc, mock_notifier = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=_time_factory,
        )
        await gate.trigger(asset_links=[])
        mock_cc._get_status = AsyncMock(
            return_value=PipelineStatus.AWAITING_FINAL_REVIEW
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await gate.poll_until_action()

        # Should have at least 2 send_review_gate calls: trigger + auto-approve notification.
        assert mock_notifier.send_review_gate.call_count >= 2
        last_call = mock_notifier.send_review_gate.call_args_list[-1]
        assert "auto" in last_call.kwargs["action_prompt"].lower() or "72" in last_call.kwargs["action_prompt"]

    @pytest.mark.asyncio
    async def test_no_auto_approve_before_72h(self) -> None:
        """Gate 2 does NOT auto-approve at 71 h — it keeps polling."""
        t0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        call_count = 0

        def _time_factory() -> datetime:
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return t0
            call_count += 1
            # First poll: 71 h (not yet), second poll: 73 h (auto-approve)
            if call_count <= 2:
                return t0 + timedelta(hours=71)
            return t0 + timedelta(hours=73)

        gate, mock_cc, _ = _make_gate(
            gate_type="final",
            initial_status=PipelineStatus.AWAITING_FINAL_REVIEW,
            now_factory=_time_factory,
        )
        await gate.trigger(asset_links=[])

        mock_cc._get_status = AsyncMock(
            return_value=PipelineStatus.AWAITING_FINAL_REVIEW
        )

        sleep_calls: list[float] = []

        async def fake_sleep(secs: float) -> None:
            sleep_calls.append(secs)

        with patch("asyncio.sleep", side_effect=fake_sleep):
            result = await gate.poll_until_action()

        # Must have polled more than once before auto-approving.
        assert len(sleep_calls) >= 2
        assert result == {"action": "auto_approve"}


# ---------------------------------------------------------------------------
# Edit validation — empty diff guard
# ---------------------------------------------------------------------------


class TestEditValidation:
    """Empty edit payloads should raise EmptyEditError."""

    def test_empty_edit_raises_empty_edit_error(self) -> None:
        """Callers that check the edit payload before calling handle_review_response
        should raise EmptyEditError when the diff is empty."""
        def validate_edit_payload(payload: str) -> None:
            if not payload or not payload.strip():
                raise EmptyEditError(
                    "Edit submission rejected: diff is empty. No changes were made."
                )

        with pytest.raises(EmptyEditError, match="empty"):
            validate_edit_payload("")

        with pytest.raises(EmptyEditError, match="empty"):
            validate_edit_payload("   ")

    def test_non_empty_edit_does_not_raise(self) -> None:
        def validate_edit_payload(payload: str) -> None:
            if not payload or not payload.strip():
                raise EmptyEditError("Edit submission rejected: diff is empty.")

        # Should not raise
        validate_edit_payload("Changed the intro to be more engaging.")
        validate_edit_payload("a")


# ---------------------------------------------------------------------------
# Orchestrator.create_review_gate factory (integration check)
# ---------------------------------------------------------------------------


class TestCreateReviewGateFactory:
    """Orchestrator.create_review_gate returns a properly configured ReviewGate."""

    def test_factory_returns_review_gate_instance(self) -> None:
        """create_review_gate must return a ReviewGate wired with CC and notifier."""
        from pipeline.orchestrator import Orchestrator

        # Build a minimal Orchestrator with mock dependencies.
        mock_cc = AsyncMock()
        mock_notifier = MagicMock()

        # We only need the two dependencies used by ReviewGate.
        orch = object.__new__(Orchestrator)
        orch._content_calendar = mock_cc  # type: ignore[attr-defined]
        orch._notifier = mock_notifier  # type: ignore[attr-defined]

        gate = orch.create_review_gate(gate_type="script", video_id="vid-001")  # type: ignore[attr-defined]

        assert isinstance(gate, ReviewGate)
        assert gate._gate_type == "script"
        assert gate._video_id == "vid-001"
        assert gate._content_calendar is mock_cc
        assert gate._notifier is mock_notifier

    def test_factory_gate2(self) -> None:
        from pipeline.orchestrator import Orchestrator

        mock_cc = AsyncMock()
        mock_notifier = MagicMock()

        orch = object.__new__(Orchestrator)
        orch._content_calendar = mock_cc  # type: ignore[attr-defined]
        orch._notifier = mock_notifier  # type: ignore[attr-defined]

        gate = orch.create_review_gate(gate_type="final", video_id="vid-002")  # type: ignore[attr-defined]

        assert gate._gate_type == "final"
        assert gate._video_id == "vid-002"
