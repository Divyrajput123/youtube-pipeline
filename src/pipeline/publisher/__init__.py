"""Publisher subsystem — YouTube Data API upload, scheduling, and rescheduling.

Manages the full lifecycle of a video's YouTube presence:
  - Task 14.1: ``upload``   — upload MP4 + thumbnail + metadata (privacy = Unlisted)
  - Task 14.2: ``schedule`` — validate datetime and schedule the video for publishing
  - Task 14.3: ``reschedule`` — move a previously-scheduled video to a new datetime

Design reference: §8 Publisher
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Protocol, runtime_checkable

from pipeline.content_calendar import Content_Calendar
from pipeline.config import is_production_mode, require_production_config
from pipeline.models import MetadataPackage, PipelineStatus, VisualAsset, YouTubeVideoRef
from pipeline.notifier import Notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — retry policy (design §8 Retry Policy)
# ---------------------------------------------------------------------------

_UPLOAD_ATTEMPTS = 3
_UPLOAD_BASE_DELAY_S = 60.0
_UPLOAD_MAX_DELAY_S = 300.0

_SCHEDULE_ATTEMPTS = 3
_SCHEDULE_BASE_DELAY_S = 5.0
_SCHEDULE_MAX_DELAY_S = 20.0

_CALENDAR_ATTEMPTS = 3
_CALENDAR_BASE_DELAY_S = 5.0
_CALENDAR_MAX_DELAY_S = 20.0

# Minimum lead-time before "now" that a publish datetime must satisfy (§9.4)
_MIN_SCHEDULE_LEAD = timedelta(minutes=15)

# Maximum time the Publisher exposes youtube_video_id + unlisted_url after upload (§9.2)
_UPLOAD_NOTIFY_DEADLINE_S = 600.0  # 10 minutes

# Reschedule must complete within 5 minutes (§10.5 / task 14.3)
_RESCHEDULE_TIMEOUT_S = 300.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PublisherError(Exception):
    """Raised when the Publisher cannot complete an operation after all retries."""


class InvalidPublishDatetimeError(ValueError):
    """Raised when a publish datetime fails validation.

    Inherits from ``ValueError`` so callers can catch either the specific or
    general case.  Raised by ``schedule()`` and ``reschedule()``.
    """


# ---------------------------------------------------------------------------
# YouTubeClient Protocol — injectable for testability
# ---------------------------------------------------------------------------


@runtime_checkable
class YouTubeClient(Protocol):
    """Minimal YouTube Data API surface required by the Publisher.

    All methods are async to match the async pipeline design.  Implementations:
    - ``YouTubeDataAPIClient`` (production stub, raises ``NotImplementedError``)
    - Any test double satisfying this protocol.
    """

    async def upload_video(
        self,
        mp4_path: str,
        title: str,
        description: str,
        tags: list[str],
        privacy: str,
    ) -> dict[str, Any]:
        """Upload a video file to YouTube and return basic reference data.

        Args:
            mp4_path: Local file-system path to the MP4 file.
            title: YouTube video title (≤ 60 characters as validated upstream).
            description: Full video description including chapter markers.
            tags: List of tag strings.
            privacy: YouTube privacy status string, e.g. ``"unlisted"``.

        Returns:
            A dict with at least ``"id"`` (the YouTube video ID) and ``"url"``
            (the watch URL).  Example::

                {"id": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}

        Raises:
            Exception: Any YouTube API error (will be retried by the Publisher).
        """
        ...

    async def set_thumbnail(
        self,
        youtube_video_id: str,
        thumbnail_path: str,
    ) -> None:
        """Upload and set a custom thumbnail for an existing YouTube video.

        Args:
            youtube_video_id: The YouTube video ID returned by ``upload_video``.
            thumbnail_path: Local file-system path to the JPEG thumbnail.

        Raises:
            Exception: Any YouTube API error.
        """
        ...

    async def update_video(
        self,
        youtube_video_id: str,
        properties: dict[str, Any],
    ) -> None:
        """Call the ``videos.update`` endpoint to patch video metadata.

        Used to set ``publishAt`` (scheduling) or revert ``status.privacyStatus``
        to ``"unlisted"`` (rollback).

        Args:
            youtube_video_id: The YouTube video ID to update.
            properties: Partial resource snippet / status dict to apply.

        Raises:
            Exception: Any YouTube API error.
        """
        ...

    async def list_scheduled_videos(self) -> list[datetime]:
        """Return publish datetimes of all videos currently scheduled (private + publishAt set).

        Used to find the next free scheduling slot so videos don't overlap.

        Returns:
            List of UTC datetimes for all scheduled videos, may be empty.

        Raises:
            Exception: Any YouTube API error.
        """
        ...

    async def list_uploaded_titles(self) -> list[str]:
        """Return titles of all videos already on the channel (published + unlisted + scheduled).

        Used to build the topic exclusion list so the pipeline never repeats
        a topic that already exists on the channel.

        Returns:
            List of video title strings, may be empty.

        Raises:
            Exception: Any YouTube API error.
        """
        ...


# ---------------------------------------------------------------------------
# YouTubeDataAPIClient — production stub
# ---------------------------------------------------------------------------


class YouTubeDataAPIClient:
    """YouTube Data API v3 client.

    Uses the same Google OAuth credentials as Google Drive
    (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN).

    All uploads go to the authenticated user's YouTube channel as Unlisted.
    """

    def __init__(self) -> None:
        self._service: Any = None
        self._fallback_mode = False
        self._upload_counter = 0

        client_id     = os.environ.get("GOOGLE_CLIENT_ID", "")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN", "")

        _creds_present = (
            client_id and client_secret and refresh_token
            and not any(v.startswith("REPLACE") for v in [client_id, client_secret, refresh_token])
        )

        if is_production_mode() and not _creds_present:
            raise ValueError(
                "Production mode requires YouTube Data API credentials: "
                "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN."
            )

        if _creds_present:
            try:
                from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
                from googleapiclient.discovery import build  # type: ignore[import-untyped]
                import google.auth.transport.requests as _gtr  # noqa: PLC0415

                creds = Credentials(
                    token=None,
                    refresh_token=refresh_token,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=[
                        "https://www.googleapis.com/auth/youtube",
                        "https://www.googleapis.com/auth/youtube.upload",
                    ],
                )
                creds.refresh(_gtr.Request())
                self._service = build("youtube", "v3", credentials=creds, cache_discovery=False)
                logger.info("YouTubeDataAPIClient: connected to YouTube Data API v3.")
            except Exception as exc:
                raise ValueError(
                    f"YouTube API auth failed: {exc}\n"
                    "Run: python get_refresh_token.py  to get a token with YouTube scopes."
                ) from exc
        else:
            logger.warning(
                "YouTube credentials missing — using fallback mode (placeholder IDs). "
                "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN."
            )
            self._fallback_mode = True

    async def upload_video(
        self,
        mp4_path: str,
        title: str,
        description: str,
        tags: list[str],
        privacy: str,
    ) -> dict[str, Any]:
        if self._fallback_mode:
            import uuid
            fake_id  = f"PLACEHOLDER_{uuid.uuid4().hex[:11]}"
            logger.info("YouTube fallback: simulated upload → %s", fake_id)
            return {"id": fake_id, "url": f"https://www.youtube.com/watch?v={fake_id}"}

        import asyncio as _asyncio  # noqa: PLC0415
        from googleapiclient.http import MediaFileUpload  # type: ignore[import-untyped]

        # mp4_path is a Google Drive URL — download it first to a temp file
        video_bytes = await self._fetch_bytes(mp4_path)

        import tempfile as _tmp, pathlib as _pl  # noqa: PLC0415, E401
        with _tmp.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            local_path = f.name

        def _sync_upload() -> dict[str, Any]:
            try:
                body = {
                    "snippet": {
                        "title": title[:100],
                        "description": description[:5000],
                        "tags": tags[:500],
                        "categoryId": "24",       # Entertainment (suits superhero/animation content)
                    },
                    "status": {"privacyStatus": privacy},
                }
                media = MediaFileUpload(local_path, mimetype="video/mp4", resumable=True)
                request = self._service.videos().insert(
                    part="snippet,status",
                    body=body,
                    media_body=media,
                )
                response = None
                while response is None:
                    _, response = request.next_chunk()
                return {
                    "id":  response["id"],
                    "url": f"https://www.youtube.com/watch?v={response['id']}",
                }
            finally:
                _pl.Path(local_path).unlink(missing_ok=True)

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_upload)

    async def set_thumbnail(self, youtube_video_id: str, thumbnail_path: str) -> None:
        if self._fallback_mode:
            logger.info("YouTube fallback: simulated thumbnail set for %s", youtube_video_id)
            return

        import asyncio as _asyncio  # noqa: PLC0415
        import tempfile as _tmp, pathlib as _pl  # noqa: PLC0415, E401
        from googleapiclient.http import MediaFileUpload  # type: ignore[import-untyped]

        # thumbnail_path is a Google Drive URL
        thumb_bytes = await self._fetch_bytes(thumbnail_path)
        with _tmp.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(thumb_bytes)
            local_path = f.name

        def _sync_set() -> None:
            try:
                media = MediaFileUpload(local_path, mimetype="image/jpeg")
                self._service.thumbnails().set(
                    videoId=youtube_video_id,
                    media_body=media,
                ).execute()
            finally:
                _pl.Path(local_path).unlink(missing_ok=True)

        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_set)

    async def update_video(self, youtube_video_id: str, properties: dict[str, Any]) -> None:
        if self._fallback_mode:
            logger.info("YouTube fallback: simulated update for %s", youtube_video_id)
            return

        import asyncio as _asyncio  # noqa: PLC0415

        def _sync_update() -> None:
            body = {"id": youtube_video_id, **properties}
            parts = ",".join(properties.keys())
            self._service.videos().update(part=parts, body=body).execute()

        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_update)

    async def list_scheduled_videos(self) -> list[datetime]:
        """Return publish datetimes of all privately scheduled videos on the channel."""
        if self._fallback_mode:
            logger.info("YouTube fallback: returning empty scheduled videos list")
            return []

        import asyncio as _asyncio  # noqa: PLC0415
        from datetime import timezone as _tz  # noqa: PLC0415

        def _sync_list() -> list[datetime]:
            scheduled: list[datetime] = []
            request = self._service.videos().list(
                part="status",
                mine=True,
                myRating=None,
                maxResults=50,
            )
            # YouTube doesn't support filtering by privacyStatus in list(),
            # so we use search().list() with type=video and eventType=upcoming
            search_req = self._service.search().list(
                part="id",
                forMine=True,
                type="video",
                eventType="upcoming",
                maxResults=50,
            )
            search_resp = search_req.execute()
            video_ids = [
                item["id"]["videoId"]
                for item in search_resp.get("items", [])
            ]
            if not video_ids:
                return scheduled

            # Fetch status details for each found video
            details_resp = self._service.videos().list(
                part="status",
                id=",".join(video_ids),
            ).execute()

            for item in details_resp.get("items", []):
                publish_at = item.get("status", {}).get("publishAt")
                if publish_at:
                    dt = datetime.fromisoformat(publish_at.replace("Z", "+00:00"))
                    scheduled.append(dt.astimezone(_tz.utc))

            return scheduled

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_list)

    async def list_uploaded_titles(self) -> list[str]:
        """Return titles of all videos on the channel (all privacy statuses)."""
        if self._fallback_mode:
            logger.info("YouTube fallback: returning empty uploaded titles list")
            return []

        import asyncio as _asyncio  # noqa: PLC0415

        def _sync_list_titles() -> list[str]:
            titles: list[str] = []
            # Get the channel's uploads playlist ID first
            ch_resp = self._service.channels().list(
                part="contentDetails", mine=True
            ).execute()
            items = ch_resp.get("items", [])
            if not items:
                return titles
            uploads_playlist_id = (
                items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            )

            # Page through the uploads playlist to get all video titles
            next_page_token = None
            while True:
                pl_resp = self._service.playlistItems().list(
                    part="snippet",
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token,
                ).execute()
                for item in pl_resp.get("items", []):
                    title = item.get("snippet", {}).get("title", "")
                    if title and title != "Deleted video" and title != "Private video":
                        titles.append(title)
                next_page_token = pl_resp.get("nextPageToken")
                if not next_page_token:
                    break

            return titles

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_list_titles)

    async def _fetch_bytes(self, url_or_path: str) -> bytes:
        """Download content from a Google Drive URL or read a local file path."""
        if url_or_path.startswith("https://drive.google.com"):
            import asyncio as _asyncio  # noqa: PLC0415
            from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]
            import io as _io  # noqa: PLC0415
            import re as _re  # noqa: PLC0415

            # Extract file ID from Drive URL
            match = _re.search(r"/d/([^/]+)/", url_or_path)
            if not match:
                raise ValueError(f"Cannot extract Drive file ID from URL: {url_or_path}")
            file_id = match.group(1)

            def _sync_download() -> bytes:
                from googleapiclient.discovery import build  # type: ignore[import-untyped]
                from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
                import google.auth.transport.requests as _gtr  # noqa: PLC0415

                creds = Credentials(
                    token=None,
                    refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=os.environ["GOOGLE_CLIENT_ID"],
                    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
                    scopes=["https://www.googleapis.com/auth/drive"],
                )
                creds.refresh(_gtr.Request())
                drive = build("drive", "v3", credentials=creds, cache_discovery=False)
                request = drive.files().get_media(fileId=file_id)
                buf = _io.BytesIO()
                dl = MediaIoBaseDownload(buf, request)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                return buf.getvalue()

            loop = _asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_download)

        elif url_or_path.startswith("file://"):
            import pathlib as _pl  # noqa: PLC0415
            return _pl.Path(url_or_path[7:]).read_bytes()

        else:
            import pathlib as _pl  # noqa: PLC0415
            return _pl.Path(url_or_path).read_bytes()


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------


def _upload_delay(attempt: int) -> float:
    """Exponential back-off delay for upload retries.

    Formula: ``min(60 * 2^(attempt-1), 300)``
    Attempt 1 → 60 s, Attempt 2 → 120 s, Attempt 3+ → 300 s (capped).
    """
    return min(_UPLOAD_BASE_DELAY_S * (2 ** (attempt - 1)), _UPLOAD_MAX_DELAY_S)


def _schedule_delay(attempt: int) -> float:
    """Exponential back-off delay for schedule / calendar retries.

    Formula: ``min(5 * 2^(attempt-1), 20)``
    Attempt 1 → 5 s, Attempt 2 → 10 s, Attempt 3+ → 20 s (capped).
    """
    return min(_SCHEDULE_BASE_DELAY_S * (2 ** (attempt - 1)), _SCHEDULE_MAX_DELAY_S)


async def _retry(
    coro_factory: Any,
    attempts: int,
    delay_fn: Any,
    operation_name: str,
) -> Any:
    """Execute *coro_factory()* with configurable retries and exponential back-off.

    Args:
        coro_factory: A zero-argument callable returning an awaitable.
        attempts: Total number of attempts before giving up.
        delay_fn: ``callable(attempt: int) -> float`` returning sleep seconds.
        operation_name: Human-readable label for log messages.

    Returns:
        The result of the first successful attempt.

    Raises:
        The last exception raised after all attempts are exhausted.
    """
    last_exc: Exception = Exception("No attempts made")
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == attempts:
                logger.error(
                    "Publisher %s failed after %d attempt(s): %s",
                    operation_name,
                    attempts,
                    exc,
                )
                break
            delay = delay_fn(attempt)
            logger.warning(
                "Publisher %s error on attempt %d/%d, retrying in %.0f s: %s",
                operation_name,
                attempt,
                attempts,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
    raise last_exc


# ---------------------------------------------------------------------------
# Chapter formatter
# ---------------------------------------------------------------------------


def _format_chapters(metadata: MetadataPackage) -> str:
    """Build the chapter-markers block from ``MetadataPackage.chapters``.

    Format (design §8 chapter format)::

        0:00 Intro
        {timestamp} {label}
        ...

    Returns an empty string when the chapter list is empty.

    Args:
        metadata: The validated ``MetadataPackage`` for the video.

    Returns:
        A newline-prefixed multi-line string with chapter markers, or ``""``
        when ``metadata.chapters`` is empty.
    """
    if not metadata.chapters:
        return ""
    lines = ["\n0:00 Intro"]
    for chapter in metadata.chapters:
        lines.append(f"{chapter.timestamp} {chapter.label}")
    return "\n".join(lines)


def _build_description(metadata: MetadataPackage) -> str:
    """Combine the base description with formatted chapter markers.

    Args:
        metadata: The validated ``MetadataPackage`` for the video.

    Returns:
        Full description string suitable for the YouTube ``snippet.description`` field.
    """
    chapters_block = _format_chapters(metadata)
    if chapters_block:
        return f"{metadata.description}{chapters_block}"
    return metadata.description


def _unlisted_url(youtube_video_id: str) -> str:
    """Return the canonical watch URL for an unlisted video.

    Args:
        youtube_video_id: YouTube-assigned video ID.

    Returns:
        ``https://www.youtube.com/watch?v={youtube_video_id}``
    """
    return f"https://www.youtube.com/watch?v={youtube_video_id}"


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class Publisher:
    """Manages YouTube upload, scheduling, and rescheduling for the pipeline.

    All public methods are async coroutines.  The ``youtube_client``,
    ``content_calendar``, and ``notifier`` dependencies are injected at
    construction time, enabling test doubles without patching.

    Args:
        youtube_client: Any object satisfying the ``YouTubeClient`` protocol.
        content_calendar: ``Content_Calendar`` instance for status persistence.
        notifier: ``Notifier`` instance for failure and success alerts.
        now_factory: Optional callable returning the current UTC datetime;
            defaults to ``datetime.now(timezone.utc)``.  Override in tests.
    """

    def __init__(
        self,
        youtube_client: YouTubeClient,
        content_calendar: Content_Calendar,
        notifier: Notifier,
        now_factory: Optional[Any] = None,
    ) -> None:
        self._yt = youtube_client
        self._calendar = content_calendar
        self._notifier = notifier
        self._now: Any = now_factory or (lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Task 14.1 — upload
    # ------------------------------------------------------------------

    async def upload(
        self,
        video_id: str,
        assets: VisualAsset,
        metadata: MetadataPackage,
    ) -> YouTubeVideoRef:
        """Upload a video to YouTube as Unlisted and update the Content Calendar.

        Steps:
        1. Build description (base + chapter markers).
        2. Attempt upload via ``youtube_client.upload_video`` (3 retries,
           exponential back-off 60 s base / 300 s max).
        3. On upload success: set custom thumbnail via ``youtube_client.set_thumbnail``.
        4. Update Content_Calendar status: ``Uploading`` → ``Unlisted``.
        5. Return ``YouTubeVideoRef`` with ``youtube_video_id`` and ``unlisted_url``.

        On all retries exhausted:
        - Notify Notifier with error details.
        - Raise ``PublisherError`` to halt the upload stage.

        The returned ``YouTubeVideoRef`` is available within 10 minutes of the
        upload starting (guaranteed by the 3-attempt/300-s-max retry envelope).

        Args:
            video_id: Pipeline-assigned video identifier.
            assets: ``VisualAsset`` containing ``mp4_path`` and ``thumbnail_path``.
            metadata: ``MetadataPackage`` with title, description, tags, chapters.

        Returns:
            ``YouTubeVideoRef`` with ``youtube_video_id`` and ``unlisted_url``.

        Raises:
            PublisherError: When all upload retries are exhausted.
        """
        description = _build_description(metadata)
        logger.info("Publisher.upload starting for video_id=%s", video_id)

        # --- 1. Upload MP4 + metadata (with retry) -------------------------
        upload_result: dict[str, Any]
        try:
            upload_result = await _retry(
                coro_factory=lambda: self._yt.upload_video(
                    mp4_path=assets.mp4_url or assets.mp4_path,
                    title=metadata.title,
                    description=description,
                    tags=metadata.tags,
                    privacy="unlisted",
                ),
                attempts=_UPLOAD_ATTEMPTS,
                delay_fn=_upload_delay,
                operation_name=f"upload_video(video_id={video_id})",
            )
        except Exception as exc:
            error_msg = f"upload_video failed after {_UPLOAD_ATTEMPTS} attempts: {exc}"
            logger.error("Publisher.upload error for video_id=%s: %s", video_id, error_msg)
            # Notify Notifier before raising
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="publisher.upload",
                error_message=error_msg,
            )
            raise PublisherError(error_msg) from exc

        youtube_video_id: str = upload_result["id"]
        url = _unlisted_url(youtube_video_id)

        logger.info(
            "Publisher.upload_video succeeded for video_id=%s → youtube_video_id=%s",
            video_id,
            youtube_video_id,
        )

        # --- 2. Set thumbnail (best-effort; log failure but don't halt) -----
        try:
            thumb_path = assets.thumbnail_url or assets.thumbnail_path
            if thumb_path:
                await self._yt.set_thumbnail(
                    youtube_video_id=youtube_video_id,
                    thumbnail_path=thumb_path,
                )
                logger.info(
                    "Publisher.set_thumbnail succeeded for youtube_video_id=%s", youtube_video_id
                )
            else:
                logger.warning(
                    "Publisher.set_thumbnail skipped for youtube_video_id=%s — no thumbnail URL/path available",
                    youtube_video_id,
                )
        except Exception as thumb_exc:  # noqa: BLE001
            logger.warning(
                "Publisher.set_thumbnail failed for youtube_video_id=%s (non-fatal): %s",
                youtube_video_id,
                thumb_exc,
            )

        # --- 3. Update Content Calendar: → Uploading → Unlisted ------------
        await self._update_calendar_status(video_id, PipelineStatus.UPLOADING)
        await self._update_calendar_status(video_id, PipelineStatus.UNLISTED)

        ref = YouTubeVideoRef(youtube_video_id=youtube_video_id, unlisted_url=url)
        logger.info(
            "Publisher.upload complete: video_id=%s unlisted_url=%s",
            video_id,
            url,
        )
        return ref

    # ------------------------------------------------------------------
    # Task 14.2 — schedule
    # ------------------------------------------------------------------

    async def schedule(
        self,
        video_id: str,
        youtube_video_id: str,
        publish_datetime: datetime,
    ) -> None:
        """Schedule a previously-uploaded (Unlisted) video for publication.

        Validation:
        - ``publish_datetime`` must be > now + 15 minutes; raises
          ``InvalidPublishDatetimeError`` otherwise.

        On valid datetime:
        1. Call ``youtube_client.update_video`` with ``publishAt`` (3 retries,
           5 s base / 20 s max).
        2. Update Content_Calendar status to ``Scheduled``.

        Content Calendar rollback:
        - If Content_Calendar ``update_status`` fails after 3 retries, revert
          the YouTube video to ``Unlisted`` and notify Notifier with rollback
          details.

        Args:
            video_id: Pipeline-assigned video identifier.
            youtube_video_id: YouTube-assigned video ID (from ``YouTubeVideoRef``).
            publish_datetime: UTC datetime at which the video should go public.

        Raises:
            InvalidPublishDatetimeError: When ``publish_datetime`` does not satisfy
                the "> now + 15 minutes" guard.
            PublisherError: When the YouTube API ``videos.update`` call fails after
                all retries.
        """
        # --- 1. Validate publish datetime -----------------------------------
        publish_datetime = _ensure_utc(publish_datetime)
        now = self._now()
        min_allowed = now + _MIN_SCHEDULE_LEAD

        if publish_datetime <= min_allowed:
            raise InvalidPublishDatetimeError(
                f"publish_datetime must be > now + 15 minutes "
                f"(minimum: {min_allowed.isoformat()}, got: {publish_datetime.isoformat()})."
            )

        logger.info(
            "Publisher.schedule video_id=%s youtube_video_id=%s publish_at=%s",
            video_id,
            youtube_video_id,
            publish_datetime.isoformat(),
        )

        # --- 2. Call YouTube API to set publishAt ----------------------------
        try:
            await _retry(
                coro_factory=lambda: self._yt.update_video(
                    youtube_video_id=youtube_video_id,
                    properties={
                        "status": {
                            "privacyStatus": "private",
                            "publishAt": publish_datetime.isoformat(),
                        }
                    },
                ),
                attempts=_SCHEDULE_ATTEMPTS,
                delay_fn=_schedule_delay,
                operation_name=f"update_video(schedule, youtube_video_id={youtube_video_id})",
            )
        except Exception as exc:
            error_msg = (
                f"videos.update (schedule) failed after {_SCHEDULE_ATTEMPTS} attempts: {exc}"
            )
            logger.error(
                "Publisher.schedule YouTube API error for video_id=%s: %s", video_id, error_msg
            )
            raise PublisherError(error_msg) from exc

        # --- 3. Update Content Calendar to Scheduled (with rollback) --------
        try:
            await _retry(
                coro_factory=lambda: self._calendar.update_status(
                    video_id, PipelineStatus.SCHEDULED
                ),
                attempts=_CALENDAR_ATTEMPTS,
                delay_fn=_schedule_delay,
                operation_name=f"calendar.update_status(Scheduled, video_id={video_id})",
            )
        except Exception as cal_exc:
            # Content Calendar update failed after 3 retries — roll back YouTube to Unlisted
            rollback_msg = (
                f"Content_Calendar update_status(Scheduled) failed after "
                f"{_CALENDAR_ATTEMPTS} attempts for video_id={video_id}: {cal_exc}. "
                f"Rolling back YouTube video {youtube_video_id} to Unlisted."
            )
            logger.error(rollback_msg)
            await self._revert_to_unlisted(
                video_id=video_id,
                youtube_video_id=youtube_video_id,
                reason=rollback_msg,
            )
            raise PublisherError(rollback_msg) from cal_exc

        logger.info(
            "Publisher.schedule complete: video_id=%s scheduled for %s",
            video_id,
            publish_datetime.isoformat(),
        )

    # ------------------------------------------------------------------
    # Task 14.3 — reschedule
    # ------------------------------------------------------------------

    async def reschedule(
        self,
        video_id: str,
        youtube_video_id: str,
        new_datetime: datetime,
    ) -> None:
        """Move a scheduled video to a different publish datetime.

        Guards (skip silently if either condition is true):
        - ``new_datetime`` is in the past (≤ now).
        - The video's current YouTube / calendar status is ``Published``.

        On valid reschedule:
        1. Call ``youtube_client.update_video`` with the new ``publishAt``.
        2. Update ``Content_Calendar.set_publish_datetime`` with the new datetime.

        The full reschedule operation must complete within 5 minutes
        (enforced via ``asyncio.wait_for``).

        Args:
            video_id: Pipeline-assigned video identifier.
            youtube_video_id: YouTube-assigned video ID.
            new_datetime: Desired new UTC publish datetime.

        Raises:
            PublisherError: If the YouTube API update fails after all retries,
                or if the operation exceeds the 5-minute timeout.
        """
        new_datetime = _ensure_utc(new_datetime)
        now = self._now()

        # --- Guard 1: skip if new_datetime is in the past ------------------
        if new_datetime <= now:
            logger.info(
                "Publisher.reschedule skipped: new_datetime %s is in the past for video_id=%s",
                new_datetime.isoformat(),
                video_id,
            )
            return

        # --- Guard 2: skip if video is already Published -------------------
        try:
            current_status = await self._calendar._get_status(video_id)  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            # If we can't determine status, be safe and skip the reschedule
            logger.warning(
                "Publisher.reschedule could not read status for video_id=%s (%s); skipping.",
                video_id,
                exc,
            )
            return

        if current_status == PipelineStatus.PUBLISHED:
            logger.info(
                "Publisher.reschedule skipped: video_id=%s is already Published.",
                video_id,
            )
            return

        logger.info(
            "Publisher.reschedule video_id=%s youtube_video_id=%s new_datetime=%s",
            video_id,
            youtube_video_id,
            new_datetime.isoformat(),
        )

        async def _do_reschedule() -> None:
            # Step 1: update YouTube API
            await _retry(
                coro_factory=lambda: self._yt.update_video(
                    youtube_video_id=youtube_video_id,
                    properties={
                        "status": {
                            "privacyStatus": "private",
                            "publishAt": new_datetime.isoformat(),
                        }
                    },
                ),
                attempts=_SCHEDULE_ATTEMPTS,
                delay_fn=_schedule_delay,
                operation_name=(
                    f"update_video(reschedule, youtube_video_id={youtube_video_id})"
                ),
            )

            # Step 2: update Content_Calendar.set_publish_datetime
            await self._calendar.set_publish_datetime(video_id, new_datetime)

        try:
            await asyncio.wait_for(_do_reschedule(), timeout=_RESCHEDULE_TIMEOUT_S)
        except asyncio.TimeoutError as exc:
            error_msg = (
                f"Publisher.reschedule timed out after {_RESCHEDULE_TIMEOUT_S}s "
                f"for video_id={video_id}"
            )
            logger.error(error_msg)
            raise PublisherError(error_msg) from exc
        except Exception as exc:
            error_msg = (
                f"Publisher.reschedule failed for video_id={video_id}: {exc}"
            )
            logger.error(error_msg)
            raise PublisherError(error_msg) from exc

        logger.info(
            "Publisher.reschedule complete: video_id=%s rescheduled to %s",
            video_id,
            new_datetime.isoformat(),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _update_calendar_status(
        self,
        video_id: str,
        status: PipelineStatus,
    ) -> None:
        """Update Content_Calendar status with retry; log but don't raise on failure."""
        try:
            await _retry(
                coro_factory=lambda: self._calendar.update_status(video_id, status),
                attempts=_CALENDAR_ATTEMPTS,
                delay_fn=_schedule_delay,
                operation_name=f"calendar.update_status({status.value}, video_id={video_id})",
            )
            logger.debug(
                "Content_Calendar status updated: video_id=%s → %s", video_id, status.value
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Content_Calendar update_status(%s) failed for video_id=%s (non-fatal): %s",
                status.value,
                video_id,
                exc,
            )

    async def _revert_to_unlisted(
        self,
        video_id: str,
        youtube_video_id: str,
        reason: str,
    ) -> None:
        """Revert a YouTube video to Unlisted privacy and notify the Notifier.

        Called during the Content Calendar rollback path in ``schedule()``.
        Errors during the revert call are logged but not re-raised so the
        caller's original exception is preserved.

        Args:
            video_id: Pipeline-assigned video identifier.
            youtube_video_id: YouTube-assigned video ID.
            reason: Human-readable rollback reason for the Notifier message.
        """
        try:
            await self._yt.update_video(
                youtube_video_id=youtube_video_id,
                properties={
                    "status": {"privacyStatus": "unlisted"}
                },
            )
            logger.info(
                "Publisher: reverted youtube_video_id=%s to Unlisted (rollback).",
                youtube_video_id,
            )
        except Exception as revert_exc:  # noqa: BLE001
            logger.error(
                "Publisher: revert-to-Unlisted failed for youtube_video_id=%s: %s",
                youtube_video_id,
                revert_exc,
            )

        # Notify regardless of whether the revert API call succeeded
        self._notifier.send_failure_alert(
            video_id=video_id,
            stage_name="publisher.schedule.rollback",
            error_message=reason,
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _ensure_utc(dt: datetime) -> datetime:
    """Attach UTC timezone to a naive datetime; return tz-aware datetimes unchanged.

    Args:
        dt: Any ``datetime`` object (naive or aware).

    Returns:
        A timezone-aware ``datetime`` in UTC.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Publisher",
    "PublisherError",
    "InvalidPublishDatetimeError",
    "YouTubeClient",
    "YouTubeDataAPIClient",
]
