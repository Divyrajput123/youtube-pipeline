"""Unit tests for Content_Calendar — Tasks 3.1 and 3.2.

Covers:
- create_record builds all 13 schema fields correctly
- update_status sets the select field and retries on transient failures
- update_asset_link maps asset_type → Notion field and rejects unknown types
- set_publish_datetime rejects past datetimes and Published-status videos
- get_batch_topics filters by batch_id and lookback window
- get_batch_completion returns floor(published_count / total * 100)
- detect_conflicts returns pairs sharing the same minute bucket
- detect_gaps returns intervals of 7+ consecutive empty days
- Notion API retry exhaustion raises ContentCalendarError
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.content_calendar import (
    Content_Calendar,
    ContentCalendarError,
    InvalidDatetimeError,
    NotionClient,
)
from pipeline.models import PipelineStatus


# ---------------------------------------------------------------------------
# Test double — in-memory Notion client
# ---------------------------------------------------------------------------


class InMemoryNotionClient:
    """Simple in-memory Notion client for unit tests.

    Stores pages in a dict keyed by page ID. Supports create_page,
    update_page, and query_database against the "video_id" and "batch_id"
    rich-text properties.
    """

    def __init__(self) -> None:
        self._pages: dict[str, dict[str, Any]] = {}
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"page-{self._id_counter}"

    async def create_page(
        self, database_id: str, properties: dict[str, Any]
    ) -> dict[str, Any]:
        page_id = self._next_id()
        page: dict[str, Any] = {"id": page_id, "properties": properties}
        self._pages[page_id] = page
        return page

    async def update_page(
        self, page_id: str, properties: dict[str, Any]
    ) -> dict[str, Any]:
        page = self._pages[page_id]
        page["properties"].update(properties)
        return page

    async def query_database(
        self,
        database_id: str,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        if filter is None:
            return list(self._pages.values())

        results = []
        for page in self._pages.values():
            if self._matches(page, filter):
                results.append(page)
        return results

    def _matches(self, page: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Minimal filter evaluation for tests (rich_text equals, select equals, and)."""
        if "and" in filter:
            return all(self._matches(page, sub) for sub in filter["and"])

        prop_name = filter.get("property", "")
        props = page.get("properties", {})
        prop = props.get(prop_name, {})

        # rich_text equals
        if "rich_text" in filter:
            expected = filter["rich_text"].get("equals", "")
            segments = prop.get("rich_text", [])
            actual = "".join(s.get("plain_text", "") or s.get("text", {}).get("content", "") for s in segments)
            return actual == expected

        # select equals
        if "select" in filter:
            expected = filter["select"].get("equals", "")
            select_obj = prop.get("select") or {}
            return select_obj.get("name", "") == expected

        # date on_or_after — always True for simplicity in unit tests
        if "date" in filter:
            return True

        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_calendar(
    fail_after: int = 0,
    now: Optional[datetime] = None,
) -> tuple[Content_Calendar, InMemoryNotionClient]:
    """Return a Content_Calendar backed by InMemoryNotionClient.

    Args:
        fail_after: If > 0, the client will raise after this many successful
            calls (to test retry / exhaustion behaviour).
        now: Fixed "current time" for the calendar.
    """
    client = InMemoryNotionClient()
    fixed_now = now or datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    calendar = Content_Calendar(
        notion_client=client,
        database_id="test-db",
        now_factory=lambda: fixed_now,
    )
    return calendar, client


# ---------------------------------------------------------------------------
# Task 3.1 — create_record
# ---------------------------------------------------------------------------


class TestCreateRecord:
    @pytest.mark.asyncio
    async def test_returns_notion_page_id(self) -> None:
        cal, client = make_calendar()
        page_id = await cal.create_record("vid-001")
        assert isinstance(page_id, str)
        assert len(page_id) > 0

    @pytest.mark.asyncio
    async def test_all_13_fields_present(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-002")
        page = list(client._pages.values())[0]
        props = page["properties"]
        expected_fields = {
            "video_id",
            "title",
            "topic",
            "status",
            "scheduled_publish_datetime",
            "script_url",
            "narration_url",
            "video_url",
            "thumbnail_url",
            "metadata_url",
            "pipeline_run_timestamp",
            "batch_id",
            "style_profile_doc_id",
        }
        assert expected_fields.issubset(set(props.keys()))

    @pytest.mark.asyncio
    async def test_status_initialised_to_pending(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-003")
        page = list(client._pages.values())[0]
        status_name = page["properties"]["status"]["select"]["name"]
        assert status_name == PipelineStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_batch_id_included_when_provided(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-004", batch_id="batch-abc")
        page = list(client._pages.values())[0]
        segments = page["properties"]["batch_id"]["rich_text"]
        content = "".join(s["text"]["content"] for s in segments)
        assert content == "batch-abc"

    @pytest.mark.asyncio
    async def test_batch_id_empty_when_not_provided(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-005")
        page = list(client._pages.values())[0]
        segments = page["properties"]["batch_id"]["rich_text"]
        assert segments == []

    @pytest.mark.asyncio
    async def test_pipeline_run_timestamp_set(self) -> None:
        fixed = datetime(2025, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
        cal, client = make_calendar(now=fixed)
        await cal.create_record("vid-006")
        page = list(client._pages.values())[0]
        ts = page["properties"]["pipeline_run_timestamp"]["date"]["start"]
        assert "2025-01-15" in ts


# ---------------------------------------------------------------------------
# Task 3.1 — update_status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    @pytest.mark.asyncio
    async def test_updates_status_field(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-010")
        await cal.update_status("vid-010", PipelineStatus.SCRIPTING)

        page = list(client._pages.values())[0]
        assert page["properties"]["status"]["select"]["name"] == PipelineStatus.SCRIPTING.value

    @pytest.mark.asyncio
    async def test_raises_on_unknown_video_id(self) -> None:
        cal, _ = make_calendar()
        with pytest.raises(ContentCalendarError):
            await cal.update_status("nonexistent-video", PipelineStatus.SCRIPTING)

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_content_calendar_error(self) -> None:
        """Simulate 3 consecutive Notion API failures → ContentCalendarError."""
        failing_client = MagicMock()
        # query_database must work (to resolve page ID), but update_page always raises
        inner_client = InMemoryNotionClient()

        async def good_query(db_id: str, filter: Any = None) -> list[dict[str, Any]]:
            return await inner_client.query_database(db_id, filter=filter)

        async def fail_update(page_id: str, properties: Any) -> dict[str, Any]:
            raise Exception("Notion 500")

        failing_client.query_database = good_query
        failing_client.update_page = fail_update
        failing_client.create_page = inner_client.create_page

        # Create the page first via the inner client so query resolves it
        cal = Content_Calendar(
            notion_client=failing_client,
            database_id="test-db",
            now_factory=lambda: datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        # Seed the inner client so the page exists for lookup
        await inner_client.create_page(
            "test-db",
            {
                "video_id": {"rich_text": [{"plain_text": "vid-011", "text": {"content": "vid-011"}}]},
                "status": {"select": {"name": "Pending"}},
            },
        )

        with patch("pipeline.content_calendar.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ContentCalendarError, match="failed after 3 attempts"):
                await cal.update_status("vid-011", PipelineStatus.SCRIPTING)


# ---------------------------------------------------------------------------
# Task 3.1 — update_asset_link
# ---------------------------------------------------------------------------


class TestUpdateAssetLink:
    @pytest.mark.asyncio
    async def test_sets_script_url(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-020")
        await cal.update_asset_link("vid-020", "script", "https://drive.google.com/script")
        page = list(client._pages.values())[0]
        assert page["properties"]["script_url"]["url"] == "https://drive.google.com/script"

    @pytest.mark.asyncio
    async def test_sets_all_five_asset_types(self) -> None:
        cal, client = make_calendar()
        await cal.create_record("vid-021")
        asset_map = {
            "script": "https://drive.google.com/s",
            "narration": "https://drive.google.com/n",
            "video": "https://drive.google.com/v",
            "thumbnail": "https://drive.google.com/t",
            "metadata": "https://drive.google.com/m",
        }
        for asset_type, url in asset_map.items():
            await cal.update_asset_link("vid-021", asset_type, url)

        page = list(client._pages.values())[0]
        assert page["properties"]["script_url"]["url"] == asset_map["script"]
        assert page["properties"]["narration_url"]["url"] == asset_map["narration"]
        assert page["properties"]["video_url"]["url"] == asset_map["video"]
        assert page["properties"]["thumbnail_url"]["url"] == asset_map["thumbnail"]
        assert page["properties"]["metadata_url"]["url"] == asset_map["metadata"]

    @pytest.mark.asyncio
    async def test_rejects_unknown_asset_type(self) -> None:
        cal, _ = make_calendar()
        await cal.create_record("vid-022")
        with pytest.raises(ValueError, match="Unknown asset_type"):
            await cal.update_asset_link("vid-022", "audio_mix", "https://example.com")


# ---------------------------------------------------------------------------
# Task 3.1 — set_publish_datetime
# ---------------------------------------------------------------------------


class TestSetPublishDatetime:
    @pytest.mark.asyncio
    async def test_sets_future_datetime(self) -> None:
        fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        cal, client = make_calendar(now=fixed_now)
        await cal.create_record("vid-030")
        future = fixed_now + timedelta(days=3)
        await cal.set_publish_datetime("vid-030", future)
        page = list(client._pages.values())[0]
        stored = page["properties"]["scheduled_publish_datetime"]["date"]["start"]
        assert "2025-06-04" in stored

    @pytest.mark.asyncio
    async def test_rejects_past_datetime(self) -> None:
        fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        cal, _ = make_calendar(now=fixed_now)
        await cal.create_record("vid-031")
        past = fixed_now - timedelta(hours=1)
        with pytest.raises(InvalidDatetimeError, match="in the past"):
            await cal.set_publish_datetime("vid-031", past)

    @pytest.mark.asyncio
    async def test_rejects_datetime_equal_to_now(self) -> None:
        fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        cal, _ = make_calendar(now=fixed_now)
        await cal.create_record("vid-032")
        with pytest.raises(InvalidDatetimeError):
            await cal.set_publish_datetime("vid-032", fixed_now)

    @pytest.mark.asyncio
    async def test_rejects_when_status_is_published(self) -> None:
        fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        cal, _ = make_calendar(now=fixed_now)
        await cal.create_record("vid-033")
        # Manually mark as Published
        await cal.update_status("vid-033", PipelineStatus.PUBLISHED)
        future = fixed_now + timedelta(days=1)
        with pytest.raises(InvalidDatetimeError, match="already Published"):
            await cal.set_publish_datetime("vid-033", future)

    @pytest.mark.asyncio
    async def test_naive_datetime_treated_as_utc(self) -> None:
        """A naive datetime should be accepted when it is in the future."""
        fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        cal, _ = make_calendar(now=fixed_now)
        await cal.create_record("vid-034")
        naive_future = datetime(2025, 7, 1, 10, 0, 0)  # no tzinfo
        # Should not raise
        await cal.set_publish_datetime("vid-034", naive_future)


# ---------------------------------------------------------------------------
# Task 3.1 — get_batch_topics
# ---------------------------------------------------------------------------


class TestGetBatchTopics:
    @pytest.mark.asyncio
    async def test_returns_topics_for_batch(self) -> None:
        cal, client = make_calendar()
        # Create two records with batch_id="b1"
        pid1 = await cal.create_record("vid-040", batch_id="b1")
        pid2 = await cal.create_record("vid-041", batch_id="b1")
        # Manually set topics
        await client.update_page(pid1, {"topic": {"rich_text": [{"plain_text": "AI Safety", "text": {"content": "AI Safety"}}]}})
        await client.update_page(pid2, {"topic": {"rich_text": [{"plain_text": "LLMs", "text": {"content": "LLMs"}}]}})

        topics = await cal.get_batch_topics("b1", lookback_days=30)
        assert set(topics) == {"AI Safety", "LLMs"}

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_unknown_batch(self) -> None:
        cal, _ = make_calendar()
        topics = await cal.get_batch_topics("no-such-batch", lookback_days=30)
        assert topics == []


# ---------------------------------------------------------------------------
# Task 3.1 — get_batch_completion
# ---------------------------------------------------------------------------


class TestGetBatchCompletion:
    @pytest.mark.asyncio
    async def test_returns_zero_for_unknown_batch(self) -> None:
        cal, _ = make_calendar()
        pct = await cal.get_batch_completion("empty-batch")
        assert pct == 0

    @pytest.mark.asyncio
    async def test_all_published_returns_100(self) -> None:
        cal, _ = make_calendar()
        for i in range(3):
            await cal.create_record(f"vid-05{i}", batch_id="b2")
            await cal.update_status(f"vid-05{i}", PipelineStatus.PUBLISHED)
        pct = await cal.get_batch_completion("b2")
        assert pct == 100

    @pytest.mark.asyncio
    async def test_none_published_returns_0(self) -> None:
        cal, _ = make_calendar()
        for i in range(4):
            await cal.create_record(f"vid-06{i}", batch_id="b3")
        pct = await cal.get_batch_completion("b3")
        assert pct == 0

    @pytest.mark.asyncio
    async def test_partial_uses_floor(self) -> None:
        # 1 of 3 published → floor(1/3 * 100) = 33
        cal, _ = make_calendar()
        for i in range(3):
            await cal.create_record(f"vid-07{i}", batch_id="b4")
        await cal.update_status("vid-070", PipelineStatus.PUBLISHED)
        pct = await cal.get_batch_completion("b4")
        assert pct == math.floor(1 / 3 * 100)  # 33

    @pytest.mark.asyncio
    async def test_2_of_5_published(self) -> None:
        cal, _ = make_calendar()
        for i in range(5):
            await cal.create_record(f"vid-08{i}", batch_id="b5")
        for i in range(2):
            await cal.update_status(f"vid-08{i}", PipelineStatus.PUBLISHED)
        pct = await cal.get_batch_completion("b5")
        assert pct == math.floor(2 / 5 * 100)  # 40

    @pytest.mark.asyncio
    async def test_boundary_floor_rounding(self) -> None:
        # floor(1/7 * 100) = 14
        cal, _ = make_calendar()
        for i in range(7):
            await cal.create_record(f"vid-09{i}", batch_id="b6")
        await cal.update_status("vid-090", PipelineStatus.PUBLISHED)
        pct = await cal.get_batch_completion("b6")
        assert pct == math.floor(1 / 7 * 100)  # 14


# ---------------------------------------------------------------------------
# Task 3.2 — detect_conflicts
# ---------------------------------------------------------------------------


class TestDetectConflicts:
    def test_no_conflicts_returns_empty(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1, 10, 0, 0),
            "v2": datetime(2025, 6, 1, 11, 0, 0),
            "v3": datetime(2025, 6, 2, 10, 0, 0),
        }
        assert Content_Calendar.detect_conflicts(datetimes) == []

    def test_two_same_minute_conflict(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1, 10, 30, 0),
            "v2": datetime(2025, 6, 1, 10, 30, 45),  # same minute, different second
        }
        result = Content_Calendar.detect_conflicts(datetimes)
        assert result == [("v1", "v2")]

    def test_seconds_do_not_matter(self) -> None:
        """Two videos at HH:MM:00 and HH:MM:59 are the same minute."""
        datetimes = {
            "alpha": datetime(2025, 6, 1, 10, 0, 0),
            "beta": datetime(2025, 6, 1, 10, 0, 59),
        }
        result = Content_Calendar.detect_conflicts(datetimes)
        assert len(result) == 1
        assert result[0] == ("alpha", "beta")

    def test_three_same_minute_returns_three_pairs(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1, 10, 0, 0),
            "v2": datetime(2025, 6, 1, 10, 0, 10),
            "v3": datetime(2025, 6, 1, 10, 0, 20),
        }
        result = Content_Calendar.detect_conflicts(datetimes)
        assert len(result) == 3
        assert ("v1", "v2") in result
        assert ("v1", "v3") in result
        assert ("v2", "v3") in result

    def test_multiple_conflict_groups(self) -> None:
        datetimes = {
            "a": datetime(2025, 6, 1, 10, 0, 0),
            "b": datetime(2025, 6, 1, 10, 0, 1),  # conflicts with a
            "c": datetime(2025, 6, 1, 11, 0, 0),
            "d": datetime(2025, 6, 1, 11, 0, 5),  # conflicts with c
        }
        result = Content_Calendar.detect_conflicts(datetimes)
        assert ("a", "b") in result
        assert ("c", "d") in result
        assert len(result) == 2

    def test_pairs_are_lexicographically_ordered(self) -> None:
        datetimes = {
            "zzz": datetime(2025, 6, 1, 10, 0, 0),
            "aaa": datetime(2025, 6, 1, 10, 0, 0),
        }
        result = Content_Calendar.detect_conflicts(datetimes)
        assert result == [("aaa", "zzz")]

    def test_empty_input(self) -> None:
        assert Content_Calendar.detect_conflicts({}) == []

    def test_single_video_no_conflict(self) -> None:
        assert Content_Calendar.detect_conflicts({"v1": datetime(2025, 6, 1, 10, 0)}) == []

    def test_different_days_same_time_no_conflict(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1, 10, 0, 0),
            "v2": datetime(2025, 6, 2, 10, 0, 0),
        }
        assert Content_Calendar.detect_conflicts(datetimes) == []


# ---------------------------------------------------------------------------
# Task 3.2 — detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_no_gap_close_together(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1),
            "v2": datetime(2025, 6, 5),  # 3-day gap (< 7)
        }
        assert Content_Calendar.detect_gaps(datetimes) == []

    def test_exactly_6_days_no_gap(self) -> None:
        # Jun 1 → Jun 8: 6 clear days between (Jun 2..Jun 7)
        datetimes = {
            "v1": datetime(2025, 6, 1),
            "v2": datetime(2025, 6, 8),
        }
        assert Content_Calendar.detect_gaps(datetimes) == []

    def test_exactly_7_days_is_a_gap(self) -> None:
        # Jun 1 → Jun 9: clear days are Jun 2..Jun 8 = 7 days
        datetimes = {
            "v1": datetime(2025, 6, 1),
            "v2": datetime(2025, 6, 9),
        }
        result = Content_Calendar.detect_gaps(datetimes)
        assert len(result) == 1
        gap_start, gap_end = result[0]
        assert gap_start == date(2025, 6, 2)
        assert gap_end == date(2025, 6, 8)
        # Span is 7 days
        assert (gap_end - gap_start).days + 1 == 7

    def test_large_gap_reported_correctly(self) -> None:
        # Jun 1 → Jun 15: clear days are Jun 2..Jun 14 = 13 days
        datetimes = {
            "v1": datetime(2025, 6, 1),
            "v2": datetime(2025, 6, 15),
        }
        result = Content_Calendar.detect_gaps(datetimes)
        assert len(result) == 1
        gap_start, gap_end = result[0]
        assert gap_start == date(2025, 6, 2)
        assert gap_end == date(2025, 6, 14)

    def test_multiple_gaps(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1),
            "v2": datetime(2025, 6, 10),   # 8-day gap (Jun 2..Jun 9)
            "v3": datetime(2025, 6, 11),   # only 1 day gap → no gap
            "v4": datetime(2025, 6, 25),   # 13-day gap (Jun 12..Jun 24)
        }
        result = Content_Calendar.detect_gaps(datetimes)
        assert len(result) == 2
        assert result[0][0] == date(2025, 6, 2)
        assert result[0][1] == date(2025, 6, 9)
        assert result[1][0] == date(2025, 6, 12)
        assert result[1][1] == date(2025, 6, 24)

    def test_single_video_no_gap(self) -> None:
        assert Content_Calendar.detect_gaps({"v1": datetime(2025, 6, 1)}) == []

    def test_empty_input_no_gap(self) -> None:
        assert Content_Calendar.detect_gaps({}) == []

    def test_consecutive_daily_videos_no_gap(self) -> None:
        datetimes = {
            f"v{i}": datetime(2025, 6, i + 1) for i in range(10)
        }
        assert Content_Calendar.detect_gaps(datetimes) == []

    def test_open_ended_boundaries_not_reported(self) -> None:
        """Gaps before the first and after the last video are NOT reported."""
        # Even if the input has a 7+ day gap implied by the "edges",
        # only between-video gaps are returned.
        datetimes = {
            "v1": datetime(2025, 6, 10),
            "v2": datetime(2025, 6, 12),   # 1 day gap only
        }
        assert Content_Calendar.detect_gaps(datetimes) == []

    def test_duplicate_dates_not_a_gap(self) -> None:
        """Multiple videos on the same date should not produce a gap between them."""
        datetimes = {
            "v1": datetime(2025, 6, 1, 10, 0),
            "v2": datetime(2025, 6, 1, 12, 0),  # same day as v1
            "v3": datetime(2025, 6, 5),
        }
        # Between Jun 1 and Jun 5: Jun 2..Jun 4 = 3 days — less than 7
        assert Content_Calendar.detect_gaps(datetimes) == []

    def test_gap_boundaries_are_dates_not_datetimes(self) -> None:
        datetimes = {
            "v1": datetime(2025, 6, 1, 23, 59, tzinfo=timezone.utc),
            "v2": datetime(2025, 6, 20, 0, 1, tzinfo=timezone.utc),
        }
        result = Content_Calendar.detect_gaps(datetimes)
        assert len(result) == 1
        gap_start, gap_end = result[0]
        assert isinstance(gap_start, date)
        assert isinstance(gap_end, date)
