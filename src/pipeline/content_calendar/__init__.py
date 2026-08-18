"""Content_Calendar subsystem — Notion MCP wrapper.

Wraps the Notion API to maintain per-video and per-batch pipeline records.
Provides calendar-view utilities for conflict and gap detection.

Design reference: §11 Content_Calendar
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from pipeline.models import PipelineStatus, VideoRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

NotionPageID = str
DriveURL = str

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ContentCalendarError(Exception):
    """Raised when a Notion API operation fails after all retries are exhausted."""


class InvalidDatetimeError(ValueError):
    """Raised when a publish datetime is rejected (past datetime or video is Published)."""


# ---------------------------------------------------------------------------
# NotionClient Protocol — injectable for testability
# ---------------------------------------------------------------------------


@runtime_checkable
class NotionClient(Protocol):
    """Minimal Notion API surface required by Content_Calendar.

    Implementors: NotionMCPClient (production), or any test double.
    All methods are async to match the Notion MCP interface.
    """

    async def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a new Notion page in *database_id* with *properties*.

        Returns the raw Notion page object (must include ``"id"``).
        """
        ...

    async def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Update properties on an existing Notion page.

        Returns the raw Notion page object.
        """
        ...

    async def query_database(
        self,
        database_id: str,
        filter: Optional[dict[str, Any]] = None,
        sorts: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Query a Notion database, returning a list of raw page objects.

        *filter* is an optional Notion filter object (compound or single-property).
        *sorts* is an optional list of Notion sort objects.
        """
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_notion_id(raw_id: str) -> str:
    """Convert a bare 32-hex Notion ID to UUID format (8-4-4-4-12).

    Notion's REST API requires IDs in UUID format in URLs.
    The notion-client SDK handles this for page/database objects passed as
    arguments, but raw ``request()`` calls need the formatted ID directly.

    ``"39b14d5100e7802a8fd5000c52a58ce3"``
    → ``"39b14d51-00e7-802a-8fd5-000c52a58ce3"``

    Already-formatted UUIDs (containing hyphens) are returned unchanged.
    """
    raw_id = raw_id.strip()
    if "-" in raw_id:
        return raw_id  # already formatted
    if len(raw_id) == 32:
        return f"{raw_id[0:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:32]}"
    return raw_id  # unknown format — return as-is


# ---------------------------------------------------------------------------
# NotionMCPClient — production stub (calls the real Notion MCP)
# ---------------------------------------------------------------------------


class NotionMCPClient:
    """Production Notion client using notion-client SDK v2.x.

    Notion v2 splits the old "database" concept into two objects:
      - ``database``   — the view/wrapper (used for retrieve & update schema)
      - ``data_source`` — the actual data store (used for query & page creation)

    Both IDs are stored; ``database_id`` is used for schema operations and
    ``data_source_id`` (discovered from the database's ``data_sources`` list)
    is used for queries and page creation.
    """

    def __init__(self, auth_token: str, database_id: str) -> None:
        self._auth_token   = auth_token
        self._database_id  = _format_notion_id(database_id)
        self._data_source_id: Optional[str] = None
        from notion_client import AsyncClient  # noqa: PLC0415
        self._client: Any = AsyncClient(auth=auth_token)

    async def _resolve_data_source_id(self) -> str:
        """Return the data_source ID backing this database, caching on first call."""
        if self._data_source_id:
            return self._data_source_id
        db = await self._client.databases.retrieve(database_id=self._database_id)
        data_sources = db.get("data_sources", [])
        if data_sources:
            self._data_source_id = data_sources[0]["id"]
        else:
            # Older Notion database — data_source_id == database_id
            self._data_source_id = self._database_id
        return self._data_source_id

    async def create_page(
        self,
        database_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a page inside the data_source that backs this database."""
        ds_id = await self._resolve_data_source_id()
        return await self._client.pages.create(
            parent={"type": "data_source_id", "data_source_id": ds_id},
            properties=properties,
        )

    async def update_page(
        self,
        page_id: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._client.pages.update(page_id=page_id, properties=properties)

    async def query_database(
        self,
        database_id: str,
        filter: Optional[dict[str, Any]] = None,
        sorts: Optional[list[dict[str, Any]]] = None,
    ) -> list[dict[str, Any]]:
        """Query pages via the data_source endpoint (Notion v2)."""
        ds_id = await self._resolve_data_source_id()
        kwargs: dict[str, Any] = {"page_size": 100}
        if filter is not None:
            kwargs["filter"] = filter
        if sorts is not None:
            kwargs["sorts"] = sorts
        response = await self._client.data_sources.query(ds_id, **kwargs)
        return response.get("results", [])


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

_MAX_ATTEMPTS = 3
_BASE_DELAY_S = 5.0
_MAX_DELAY_S = 20.0


async def _retry_notion(
    coro_factory: Any,
    operation_name: str = "Notion API call",
) -> Any:
    """Execute *coro_factory()* with exponential back-off retry.

    Retry policy (design §11):
    - 3 attempts total
    - Back-off: ``min(5 * 2^(attempt-1), 20)`` seconds
      → delays: 5 s, 10 s, 20 s (capped)

    Raises ``ContentCalendarError`` after all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == _MAX_ATTEMPTS:
                break
            delay = min(_BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S)
            logger.warning(
                "Notion API error on %s (attempt %d/%d), retrying in %.0f s: %s",
                operation_name,
                attempt,
                _MAX_ATTEMPTS,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

    raise ContentCalendarError(
        f"{operation_name} failed after {_MAX_ATTEMPTS} attempts: {last_exc}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Content_Calendar
# ---------------------------------------------------------------------------


class Content_Calendar:
    """Notion-backed content calendar for the AI YouTube Content Pipeline.

    All datetimes are expected in UTC. The ``notion_client`` parameter accepts
    any object that satisfies the ``NotionClient`` protocol, enabling injection
    of test doubles without patching.

    Args:
        notion_client: A ``NotionClient``-compatible object.
        database_id: Notion database ID for the video records.
        now_factory: Optional callable returning the current UTC datetime;
            defaults to ``datetime.now(timezone.utc)``. Override in tests.
    """

    def __init__(
        self,
        notion_client: NotionClient,
        database_id: str,
        now_factory: Any = None,
    ) -> None:
        self._client = notion_client
        self._db_id = database_id
        self._now: Any = now_factory or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Task 3.1 — Core interface methods
    # ------------------------------------------------------------------

    async def create_record(
        self,
        video_id: str,
        batch_id: Optional[str] = None,
    ) -> NotionPageID:
        """Create a new Notion page for *video_id* with all 13 schema fields.

        All text fields are initialised to empty strings; numeric / datetime
        fields are left null.  The record is associated with *batch_id* when
        provided.

        Args:
            video_id: Pipeline-assigned identifier (1–128 alphanumeric/-/_).
            batch_id: Optional batch identifier; links the record to a batch.

        Returns:
            The Notion page ID for the newly created record.

        Raises:
            ContentCalendarError: If the Notion API fails after 3 attempts.
        """
        now_iso = self._now().isoformat()

        # Build Notion property map for all 13 required schema fields
        properties: dict[str, Any] = {
            # 1. video_id — rich text (primary key label)
            "video_id": {
                "rich_text": [{"text": {"content": video_id}}]
            },
            # 2. title — title property (Notion "Name" column)
            "title": {
                "title": [{"text": {"content": ""}}]
            },
            # 3. topic
            "topic": {
                "rich_text": [{"text": {"content": ""}}]
            },
            # 4. status — select (18 values driven by PipelineStatus enum)
            "status": {
                "select": {"name": PipelineStatus.PENDING.value}
            },
            # 5. scheduled_publish_datetime — date (UTC)
            "scheduled_publish_datetime": {
                "date": None
            },
            # 6. script_url
            "script_url": {
                "url": None
            },
            # 7. narration_url
            "narration_url": {
                "url": None
            },
            # 8. video_url
            "video_url": {
                "url": None
            },
            # 9. thumbnail_url
            "thumbnail_url": {
                "url": None
            },
            # 10. metadata_url
            "metadata_url": {
                "url": None
            },
            # 11. pipeline_run_timestamp — date (UTC, set at creation)
            "pipeline_run_timestamp": {
                "date": {"start": now_iso}
            },
            # 12. batch_id — nullable rich text
            "batch_id": {
                "rich_text": [{"text": {"content": batch_id}}]
                if batch_id is not None
                else []
            },
            # 13. style_profile_doc_id
            "style_profile_doc_id": {
                "rich_text": [{"text": {"content": ""}}]
            },
        }

        result = await _retry_notion(
            lambda: self._client.create_page(self._db_id, properties),
            operation_name=f"create_record({video_id})",
        )
        page_id: NotionPageID = result["id"]
        logger.info("Created Notion record for video_id=%s, page_id=%s", video_id, page_id)
        return page_id

    async def update_status(
        self,
        video_id: str,
        status: PipelineStatus,
    ) -> None:
        """Update the ``status`` field of the record identified by *video_id*.

        Complies with the 30-second SLA (design §11 Status Update SLA) — the
        caller is responsible for ensuring this coroutine is awaited promptly
        after a stage transition.

        Applies Notion API retry: 3 attempts, 5 s base, 20 s max exponential.

        Args:
            video_id: The pipeline video identifier.
            status: New ``PipelineStatus`` value.

        Raises:
            ContentCalendarError: If the Notion API fails after all retries.
        """
        page_id = await self._resolve_page_id(video_id)
        properties = {
            "status": {"select": {"name": status.value}}
        }
        await _retry_notion(
            lambda: self._client.update_page(page_id, properties),
            operation_name=f"update_status({video_id}, {status.value})",
        )
        logger.info("Updated status for video_id=%s to %s", video_id, status.value)

    async def update_topic(
        self,
        video_id: str,
        topic: str,
    ) -> None:
        """Set the ``topic`` field on the Notion record for *video_id*.

        Called right after topic research so deduplication works in future runs.

        Args:
            video_id: The pipeline video identifier.
            topic: The chosen topic title string.

        Raises:
            ContentCalendarError: If the Notion API fails after all retries.
        """
        page_id = await self._resolve_page_id(video_id)
        properties = {
            "topic": {"rich_text": [{"text": {"content": topic[:2000]}}]}
        }
        await _retry_notion(
            lambda: self._client.update_page(page_id, properties),
            operation_name=f"update_topic({video_id})",
        )
        logger.info("Updated topic for video_id=%s to %r", video_id, topic)

    async def update_asset_link(
        self,
        video_id: str,
        asset_type: str,
        url: DriveURL,
    ) -> None:
        """Set one of the five asset URL fields on the record for *video_id*.

        Args:
            video_id: The pipeline video identifier.
            asset_type: One of ``"script"``, ``"narration"``, ``"video"``,
                ``"thumbnail"``, ``"metadata"``.
            url: Google Drive URL for the asset.

        Raises:
            ValueError: If *asset_type* is not a recognised asset field name.
            ContentCalendarError: If the Notion API fails after all retries.
        """
        _ASSET_FIELD_MAP = {
            "script": "script_url",
            "narration": "narration_url",
            "video": "video_url",
            "thumbnail": "thumbnail_url",
            "metadata": "metadata_url",
        }
        field = _ASSET_FIELD_MAP.get(asset_type)
        if field is None:
            raise ValueError(
                f"Unknown asset_type '{asset_type}'. "
                f"Valid values: {list(_ASSET_FIELD_MAP.keys())}"
            )

        page_id = await self._resolve_page_id(video_id)
        properties = {field: {"url": url}}
        await _retry_notion(
            lambda: self._client.update_page(page_id, properties),
            operation_name=f"update_asset_link({video_id}, {asset_type})",
        )
        logger.info("Updated %s for video_id=%s", asset_type, video_id)

    async def set_youtube_url(
        self,
        video_id: str,
        youtube_video_id: str,
        unlisted_url: str,
    ) -> None:
        """Store the YouTube video ID and unlisted URL after a successful upload.

        Args:
            video_id: The pipeline video identifier.
            youtube_video_id: The YouTube-assigned video ID (e.g. "dQw4w9WgXcQ").
            unlisted_url: The full YouTube watch URL for the unlisted video.

        Raises:
            ContentCalendarError: If the Notion API fails after all retries.
        """
        page_id = await self._resolve_page_id(video_id)
        properties = {
            "youtube_video_id": {"rich_text": [{"text": {"content": youtube_video_id}}]},
            "unlisted_url": {"url": unlisted_url},
        }
        await _retry_notion(
            lambda: self._client.update_page(page_id, properties),
            operation_name=f"set_youtube_url({video_id})",
        )
        logger.info(
            "Stored YouTube video ID %s for video_id=%s", youtube_video_id, video_id
        )

    async def get_youtube_video_id(self, video_id: str) -> Optional[str]:
        """Return the stored YouTube video ID for *video_id*, or None if not set.

        Args:
            video_id: The pipeline video identifier.

        Returns:
            YouTube video ID string, or None.
        """
        try:
            notion_filter: dict[str, Any] = {
                "property": "video_id",
                "rich_text": {"equals": video_id},
            }
            pages = await self._client.query_database(self._db_id, filter=notion_filter)
            if pages:
                yt_id = self._extract_rich_text(pages[0], "youtube_video_id")
                return yt_id if yt_id else None
            return None
        except Exception as exc:
            logger.debug("get_youtube_video_id failed for %s: %s", video_id, exc)
            return None

    async def set_publish_datetime(
        self,
        video_id: str,
        dt: datetime,
    ) -> None:
        """Set the ``scheduled_publish_datetime`` for *video_id*.

        Validation rules (design §11 Validation, requirements 10.5 / 10.6):
        - Raises ``InvalidDatetimeError`` if *dt* is in the past (relative to
          the current UTC time returned by ``now_factory``).
        - Raises ``InvalidDatetimeError`` if the video's current status is
          ``Published``.

        Args:
            video_id: The pipeline video identifier.
            dt: The desired UTC publish datetime.

        Raises:
            InvalidDatetimeError: When *dt* is in the past or video is
                already Published.
            ContentCalendarError: If the Notion API fails after all retries.
        """
        # Normalise to UTC-aware datetime for comparison
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        now = self._now()

        # Guard: reject past datetimes
        if dt <= now:
            raise InvalidDatetimeError(
                f"Publish datetime {dt.isoformat()} is in the past "
                f"(current UTC: {now.isoformat()})."
            )

        # Guard: reject if video is already Published
        current_status = await self._get_status(video_id)
        if current_status == PipelineStatus.PUBLISHED:
            raise InvalidDatetimeError(
                f"Cannot set publish datetime for video_id={video_id} "
                "because it is already Published."
            )

        page_id = await self._resolve_page_id(video_id)
        properties = {
            "scheduled_publish_datetime": {
                "date": {"start": dt.isoformat()}
            }
        }
        await _retry_notion(
            lambda: self._client.update_page(page_id, properties),
            operation_name=f"set_publish_datetime({video_id})",
        )
        logger.info(
            "Set scheduled_publish_datetime for video_id=%s to %s",
            video_id,
            dt.isoformat(),
        )

    async def set_pipeline_end_time(
        self,
        video_id: str,
        end_time: Optional[datetime] = None,
    ) -> None:
        """Record when the pipeline run finished for *video_id*.

        Sets the ``pipeline_end_time`` date property in Notion. If *end_time*
        is not provided, the current UTC time is used.

        Args:
            video_id: The pipeline video identifier.
            end_time: UTC datetime when the pipeline finished. Defaults to now.

        Raises:
            ContentCalendarError: If the Notion API fails after all retries.
        """
        if end_time is None:
            end_time = self._now()
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)

        page_id = await self._resolve_page_id(video_id)
        properties = {
            "pipeline_end_time": {
                "date": {"start": end_time.isoformat()}
            }
        }
        await _retry_notion(
            lambda: self._client.update_page(page_id, properties),
            operation_name=f"set_pipeline_end_time({video_id})",
        )
        logger.info(
            "Set pipeline_end_time for video_id=%s to %s",
            video_id,
            end_time.isoformat(),
        )

    async def get_scheduled_datetimes(self) -> list[datetime]:
        """Return all future scheduled_publish_datetimes across all calendar records.

        Used by _find_next_free_slot to avoid scheduling conflicts when YouTube
        has not yet reflected recently assigned slots.

        Returns:
            List of UTC-aware datetimes that are in the future.
        """
        now = self._now()
        pages = await _retry_notion(
            lambda: self._client.query_database(self._db_id),
            operation_name="get_scheduled_datetimes",
        )
        result = []
        for page in pages:
            props = page.get("properties", {})
            dt_prop = props.get("scheduled_publish_datetime", {})
            date_val = dt_prop.get("date") or {}
            start = date_val.get("start")
            if not start:
                continue
            try:
                dt = datetime.fromisoformat(start)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt > now:
                    result.append(dt)
            except Exception:
                pass
        return result

    async def get_batch_topics(
        self,
        batch_id: str,
        lookback_days: int,
    ) -> list[str]:
        """Return topic strings for all videos in *batch_id* within the lookback window.

        Queries the Notion database for records whose ``batch_id`` matches and
        whose ``pipeline_run_timestamp`` falls within the past *lookback_days*
        calendar days.

        Used by Topic_Researcher to exclude recently used topics (requirement 2.6).

        Args:
            batch_id: The batch identifier.
            lookback_days: Number of days to look back from now.

        Returns:
            A list of topic strings (may be empty if no matching records exist).

        Raises:
            ContentCalendarError: If the Notion API fails after all retries.
        """
        cutoff = self._now() - timedelta(days=lookback_days)

        # If batch_id is empty, fetch ALL topics within the lookback window
        # (used by single-video runs to avoid repeating any past topic).
        # Otherwise, filter by the specific batch_id as well.
        # Always exclude videos that failed (Pipeline Error) — those topics
        # should be retried, not excluded.
        error_filter: dict[str, Any] = {
            "property": "status",
            "select": {"does_not_equal": "Pipeline Error"},
        }

        if batch_id:
            notion_filter: dict[str, Any] = {
                "and": [
                    {
                        "property": "batch_id",
                        "rich_text": {"equals": batch_id},
                    },
                    {
                        "property": "pipeline_run_timestamp",
                        "date": {"on_or_after": cutoff.isoformat()},
                    },
                    error_filter,
                ]
            }
        else:
            notion_filter = {
                "and": [
                    {
                        "property": "pipeline_run_timestamp",
                        "date": {"on_or_after": cutoff.isoformat()},
                    },
                    error_filter,
                ]
            }

        pages = await _retry_notion(
            lambda: self._client.query_database(self._db_id, filter=notion_filter),
            operation_name=f"get_batch_topics(batch_id={batch_id})",
        )

        topics: list[str] = []
        for page in pages:
            topic_val = self._extract_rich_text(page, "topic")
            if topic_val:
                topics.append(topic_val)

        logger.debug(
            "get_batch_topics(batch_id=%s, lookback_days=%d) → %d topics",
            batch_id,
            lookback_days,
            len(topics),
        )
        return topics

    async def list_videos_by_status(self, status: "PipelineStatus") -> list[dict]:
        """Return all Notion pages with the given pipeline status.

        Args:
            status: The PipelineStatus to filter by.

        Returns:
            List of dicts with at least 'video_id' key.
        """
        notion_filter: dict[str, Any] = {
            "property": "status",
            "select": {"equals": status.value},
        }
        # Sort by scheduled_publish_datetime ascending so the oldest scheduled
        # video is posted first — Reels go out in the same order as YouTube publishes.
        # Without this Notion defaults to last-edited descending, causing the
        # Reel Manager to always pick the most recently created video.
        notion_sorts: list[dict[str, Any]] = [
            {"property": "scheduled_publish_datetime", "direction": "ascending"}
        ]
        pages = await _retry_notion(
            lambda: self._client.query_database(
                self._db_id, filter=notion_filter, sorts=notion_sorts
            ),
            operation_name=f"list_videos_by_status(status={status.value})",
        )
        results = []
        for page in pages:
            video_id = self._extract_rich_text(page, "video_id")
            if video_id:
                results.append({"video_id": video_id, "page_id": page.get("id", "")})
        return results

    async def get_batch_completion(self, batch_id: str) -> int:
        """Return the completion percentage for *batch_id* as an integer 0–100.

        Formula (design §11, requirement 11.6):
            ``floor(published_count / total_count * 100)``

        Returns 0 if the batch has no records (avoids division by zero).

        Args:
            batch_id: The batch identifier.

        Returns:
            Integer percentage (0–100) rounded down.

        Raises:
            ContentCalendarError: If the Notion API fails after all retries.
        """
        notion_filter: dict[str, Any] = {
            "property": "batch_id",
            "rich_text": {"equals": batch_id},
        }

        pages = await _retry_notion(
            lambda: self._client.query_database(self._db_id, filter=notion_filter),
            operation_name=f"get_batch_completion(batch_id={batch_id})",
        )

        total_count = len(pages)
        if total_count == 0:
            logger.warning("get_batch_completion: no records found for batch_id=%s", batch_id)
            return 0

        published_count = sum(
            1
            for page in pages
            if self._extract_select(page, "status") == PipelineStatus.PUBLISHED.value
        )

        pct: int = math.floor(published_count / total_count * 100)
        logger.debug(
            "get_batch_completion(batch_id=%s): %d/%d published → %d%%",
            batch_id,
            published_count,
            total_count,
            pct,
        )
        return pct

    # ------------------------------------------------------------------
    # Task 3.2 — Calendar view: conflict and gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_conflicts(
        scheduled_datetimes: dict[str, datetime],
    ) -> list[tuple[str, str]]:
        """Return all pairs of video IDs sharing the same scheduled datetime (to the minute).

        Two videos conflict when their ``scheduled_publish_datetime`` values are
        identical after truncating to the minute (i.e. seconds and sub-second
        components are ignored). (Design §11 Calendar View Rules.)

        Args:
            scheduled_datetimes: Mapping of ``video_id → datetime``.  Datetimes
                may be timezone-aware (UTC) or naive.

        Returns:
            A sorted list of ``(video_id_a, video_id_b)`` tuples where
            ``video_id_a < video_id_b`` (lexicographic) to ensure deterministic
            ordering.  The list is empty when no conflicts exist.

        Example::

            conflicts = Content_Calendar.detect_conflicts({
                "vid-1": datetime(2025, 6, 1, 10, 0, 0),
                "vid-2": datetime(2025, 6, 1, 10, 0, 30),  # same minute as vid-1
                "vid-3": datetime(2025, 6, 2, 10, 0, 0),
            })
            # → [("vid-1", "vid-2")]
        """
        # Group video IDs by (year, month, day, hour, minute)
        minute_bucket: dict[tuple[int, int, int, int, int], list[str]] = defaultdict(list)
        for video_id, dt in scheduled_datetimes.items():
            key = (dt.year, dt.month, dt.day, dt.hour, dt.minute)
            minute_bucket[key].append(video_id)

        conflicts: list[tuple[str, str]] = []
        for video_ids in minute_bucket.values():
            if len(video_ids) < 2:
                continue
            # Produce all ordered pairs (a, b) with a < b
            video_ids_sorted = sorted(video_ids)
            for i in range(len(video_ids_sorted)):
                for j in range(i + 1, len(video_ids_sorted)):
                    conflicts.append((video_ids_sorted[i], video_ids_sorted[j]))

        # Sort the final list for deterministic output
        conflicts.sort()
        return conflicts

    @staticmethod
    def detect_gaps(
        scheduled_datetimes: dict[str, datetime],
    ) -> list[tuple[date, date]]:
        """Return date intervals of 7 or more consecutive days without a scheduled video.

        A *gap* is defined as a run of 7 or more calendar days with no video
        scheduled.  The gap boundaries returned are inclusive: ``(gap_start,
        gap_end)`` where ``gap_start`` is the first day with no video and
        ``gap_end`` is the last day with no video.  The span
        ``(gap_end - gap_start).days + 1 ≥ 7``.

        Detection is performed **between** existing scheduled dates only —
        open-ended periods before the first video or after the last video are
        **not** reported.

        Args:
            scheduled_datetimes: Mapping of ``video_id → datetime``.

        Returns:
            A list of ``(gap_start: date, gap_end: date)`` tuples sorted
            chronologically.  Empty when fewer than 2 videos are scheduled or
            no gaps of ≥ 7 days exist.

        Example::

            gaps = Content_Calendar.detect_gaps({
                "vid-1": datetime(2025, 6, 1),
                "vid-2": datetime(2025, 6, 15),   # 13 clear days between 1 and 15
            })
            # → [(date(2025, 6, 2), date(2025, 6, 14))]
        """
        if len(scheduled_datetimes) < 2:
            return []

        # Collect unique scheduled dates (ignoring time-of-day)
        scheduled_dates: list[date] = sorted(
            {dt.date() if hasattr(dt, "date") else dt for dt in scheduled_datetimes.values()}
        )

        gaps: list[tuple[date, date]] = []
        for i in range(len(scheduled_dates) - 1):
            current_day = scheduled_dates[i]
            next_day = scheduled_dates[i + 1]

            # The gap runs from the day after current to the day before next
            gap_start = current_day + timedelta(days=1)
            gap_end = next_day - timedelta(days=1)

            gap_length = (gap_end - gap_start).days + 1  # inclusive span
            if gap_length >= 7:
                gaps.append((gap_start, gap_end))

        return gaps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _resolve_page_id(self, video_id: str) -> str:
        """Lookup the Notion page ID for *video_id*.

        Queries the database with a filter on the ``video_id`` rich-text field.
        Raises ``ContentCalendarError`` if no record is found.
        """
        notion_filter: dict[str, Any] = {
            "property": "video_id",
            "rich_text": {"equals": video_id},
        }
        pages = await _retry_notion(
            lambda: self._client.query_database(self._db_id, filter=notion_filter),
            operation_name=f"_resolve_page_id({video_id})",
        )
        if not pages:
            raise ContentCalendarError(
                f"No Notion record found for video_id='{video_id}'."
            )
        return pages[0]["id"]

    async def _get_status(self, video_id: str) -> PipelineStatus:
        """Return the current ``PipelineStatus`` of *video_id*."""
        notion_filter: dict[str, Any] = {
            "property": "video_id",
            "rich_text": {"equals": video_id},
        }
        pages = await _retry_notion(
            lambda: self._client.query_database(self._db_id, filter=notion_filter),
            operation_name=f"_get_status({video_id})",
        )
        if not pages:
            raise ContentCalendarError(
                f"No Notion record found for video_id='{video_id}' when reading status."
            )
        raw_status = self._extract_select(pages[0], "status")
        try:
            return PipelineStatus(raw_status)
        except ValueError:
            # Default to a safe non-Published value if the stored string doesn't match
            logger.warning(
                "Unknown status value '%s' for video_id=%s; treating as Pending.",
                raw_status,
                video_id,
            )
            return PipelineStatus.PENDING

    # ------------------------------------------------------------------
    # Notion page property extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_rich_text(page: dict[str, Any], field: str) -> str:
        """Extract the plain-text value of a rich-text Notion property."""
        try:
            segments: list[dict[str, Any]] = (
                page.get("properties", {}).get(field, {}).get("rich_text", [])
            )
            return "".join(seg.get("plain_text", "") for seg in segments)
        except (KeyError, TypeError, AttributeError):
            return ""

    @staticmethod
    def _extract_select(page: dict[str, Any], field: str) -> str:
        """Extract the name of a select Notion property."""
        try:
            select_obj = page.get("properties", {}).get(field, {}).get("select")
            if select_obj is None:
                return ""
            return select_obj.get("name", "")
        except (KeyError, TypeError, AttributeError):
            return ""


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Content_Calendar",
    "ContentCalendarError",
    "InvalidDatetimeError",
    "NotionClient",
    "NotionMCPClient",
    "NotionPageID",
    "DriveURL",
]
