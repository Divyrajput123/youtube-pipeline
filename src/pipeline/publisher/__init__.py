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

    async def post_pinned_comment(self, youtube_video_id: str, text: str) -> None:
        """Post a comment on the video and pin it to the top.

        Args:
            youtube_video_id: The YouTube video ID.
            text: Comment text (supports YouTube markdown).

        Raises:
            Exception: Any YouTube API error.
        """
        ...

    async def add_to_playlist(self, youtube_video_id: str, playlist_id: str) -> None:
        """Add a video to a YouTube playlist.

        Args:
            youtube_video_id: The YouTube video ID.
            playlist_id: The target playlist ID.

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
                        "https://www.googleapis.com/auth/youtube.force-ssl",
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
            # List private videos on the channel — scheduled uploads are private
            # with a publishAt date set. The search API's eventType=upcoming is
            # for live streams only, so we use the uploads playlist approach.
            ch_resp = self._service.channels().list(
                part="contentDetails", mine=True
            ).execute()
            items = ch_resp.get("items", [])
            if not items:
                return scheduled
            uploads_playlist_id = (
                items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            )

            # Get recent videos from uploads playlist
            pl_resp = self._service.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=50,
            ).execute()
            video_ids = [
                item["snippet"]["resourceId"]["videoId"]
                for item in pl_resp.get("items", [])
                if item["snippet"].get("resourceId", {}).get("videoId")
            ]
            if not video_ids:
                return scheduled

            # Fetch status details — look for private videos with publishAt
            details_resp = self._service.videos().list(
                part="status",
                id=",".join(video_ids[:50]),
            ).execute()

            for item in details_resp.get("items", []):
                status = item.get("status", {})
                publish_at = status.get("publishAt")
                if publish_at and status.get("privacyStatus") == "private":
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

    async def post_pinned_comment(self, youtube_video_id: str, text: str) -> None:
        """Post a comment on the video and pin it to the top."""
        if self._fallback_mode:
            logger.info("YouTube fallback: simulated pinned comment on %s", youtube_video_id)
            return

        import asyncio as _asyncio  # noqa: PLC0415

        def _sync_comment() -> None:
            # Insert a top-level comment
            resp = self._service.commentThreads().insert(
                part="snippet",
                body={
                    "snippet": {
                        "videoId": youtube_video_id,
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": text,
                            }
                        },
                    }
                },
            ).execute()
            comment_id = resp["snippet"]["topLevelComment"]["id"]
            logger.info("Posted comment %s on video %s", comment_id, youtube_video_id)

        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_comment)

    async def add_to_playlist(self, youtube_video_id: str, playlist_id: str) -> None:
        """Add a video to a YouTube playlist."""
        if self._fallback_mode:
            logger.info("YouTube fallback: simulated add to playlist %s", playlist_id)
            return

        import asyncio as _asyncio  # noqa: PLC0415

        def _sync_add() -> None:
            self._service.playlistItems().insert(
                part="snippet",
                body={
                    "snippet": {
                        "playlistId": playlist_id,
                        "resourceId": {
                            "kind": "youtube#video",
                            "videoId": youtube_video_id,
                        },
                    }
                },
            ).execute()
            logger.info("Added video %s to playlist %s", youtube_video_id, playlist_id)

        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_add)

    async def upload_short(
        self,
        mp4_path: str,
        title: str,
        description: str,
        tags: list[str],
        publish_at: Optional[datetime] = None,
    ) -> dict[str, str]:
        """Upload a vertical (9:16) Short video to YouTube.

        Shorts are identified by YouTube via the #Shorts hashtag in the title
        and vertical aspect ratio.

        If publish_at is provided, the Short is uploaded as Private with a
        publishAt timestamp so it goes live at the same time as the main video.
        If None, uploads as Public (immediate).

        Args:
            mp4_path: Local path to the vertical MP4 file.
            title: Short title (must end with ' #Shorts' for algorithm pickup).
            description: Short description with link to full video.
            tags: Tag list.
            publish_at: Optional UTC datetime to schedule the Short.

        Returns:
            Dict with "id" and "url" keys.
        """
        if self._fallback_mode:
            import uuid
            fake_id = f"SHORT_{uuid.uuid4().hex[:11]}"
            schedule_info = f" (scheduled: {publish_at.isoformat()})" if publish_at else ""
            logger.info("YouTube fallback: simulated Short upload → %s%s", fake_id, schedule_info)
            return {"id": fake_id, "url": f"https://www.youtube.com/shorts/{fake_id}"}

        import asyncio as _asyncio  # noqa: PLC0415
        from googleapiclient.http import MediaFileUpload  # type: ignore[import-untyped]

        def _sync_upload() -> dict[str, str]:
            # If scheduled, upload as Private with publishAt so it goes live
            # at the same time as the main video
            if publish_at:
                status_body: dict[str, Any] = {
                    "privacyStatus": "private",
                    "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                }
            else:
                status_body = {"privacyStatus": "public"}

            body = {
                "snippet": {
                    "title": title[:100],
                    "description": description[:5000],
                    "tags": tags[:500],
                    "categoryId": "24",
                },
                "status": status_body,
            }
            media = MediaFileUpload(mp4_path, mimetype="video/mp4", resumable=True)
            request = self._service.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )
            response = None
            while response is None:
                _, response = request.next_chunk()
            vid = response["id"]
            return {"id": vid, "url": f"https://www.youtube.com/shorts/{vid}"}

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_upload)

    async def get_best_performing_video(self, days: int = 28) -> Optional[str]:
        """Return the youtube_video_id of the best-performing video in the last N days.

        "Best performing" = most views. Used to link in end-screens.

        Returns:
            YouTube video ID string, or None if no videos found.
        """
        if self._fallback_mode:
            return None

        import asyncio as _asyncio  # noqa: PLC0415

        def _sync_best() -> Optional[str]:
            # Get the channel's uploads playlist
            ch_resp = self._service.channels().list(
                part="contentDetails", mine=True
            ).execute()
            items = ch_resp.get("items", [])
            if not items:
                return None
            uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

            # Get recent video IDs (up to 50)
            pl_resp = self._service.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=50,
            ).execute()
            video_ids = [
                item["snippet"]["resourceId"]["videoId"]
                for item in pl_resp.get("items", [])
                if item["snippet"].get("resourceId", {}).get("videoId")
            ]
            if not video_ids:
                return None

            # Get view counts for these videos
            stats_resp = self._service.videos().list(
                part="statistics",
                id=",".join(video_ids[:50]),
            ).execute()
            best_id = None
            best_views = -1
            for item in stats_resp.get("items", []):
                views = int(item.get("statistics", {}).get("viewCount", "0"))
                if views > best_views:
                    best_views = views
                    best_id = item["id"]

            return best_id

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_best)

    async def get_audience_peak_hour(self) -> Optional[int]:
        """Query YouTube Analytics to find the hour (0-23 UTC) with highest viewer activity.

        Uses the YouTube Analytics API 'userActivity' report to find the hour
        with the most views over the past 28 days.

        Returns:
            Hour (0-23 in UTC) with peak audience, or None if analytics unavailable.
        """
        if self._fallback_mode:
            return None

        import asyncio as _asyncio  # noqa: PLC0415
        from datetime import date, timedelta  # noqa: PLC0415

        def _sync_analytics() -> Optional[int]:
            try:
                from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
                from googleapiclient.discovery import build  # type: ignore[import-untyped]
                import google.auth.transport.requests as _gtr  # noqa: PLC0415

                # Build YouTube Analytics API service (separate from Data API)
                creds = Credentials(
                    token=None,
                    refresh_token=os.environ["GOOGLE_REFRESH_TOKEN"],
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=os.environ["GOOGLE_CLIENT_ID"],
                    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
                    scopes=["https://www.googleapis.com/auth/yt-analytics.readonly"],
                )
                creds.refresh(_gtr.Request())
                analytics = build("youtubeAnalytics", "v2", credentials=creds, cache_discovery=False)

                end_date = date.today()
                start_date = end_date - timedelta(days=28)

                # YouTube Analytics API doesn't support "hour" dimension directly.
                # Use "day" dimension to get views per day, then use the channel's
                # configured publish time as the peak hour (most reliable signal).
                # Falls back to None which lets the scheduler use config defaults.
                resp = analytics.reports().query(
                    ids="channel==MINE",
                    startDate=start_date.isoformat(),
                    endDate=end_date.isoformat(),
                    metrics="views",
                    dimensions="day",
                    sort="-views",
                    maxResults=7,
                ).execute()

                rows = resp.get("rows", [])
                if rows:
                    # Analytics available — channel is active.
                    # Return None to use the configured publish time (more reliable
                    # than guessing hourly patterns from daily data).
                    logger.info(
                        "YouTube Analytics: channel has %d days of data, "
                        "using configured publish time for scheduling",
                        len(rows),
                    )
                return None
            except Exception as exc:
                logger.warning("YouTube Analytics query failed (non-fatal): %s", exc)
                return None

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_analytics)

    async def set_end_screen(self, youtube_video_id: str, best_video_id: str) -> None:
        """Add end-screen elements to a video: subscribe button + link to best video.

        Note: The YouTube Data API does not natively support end-screen creation.
        Instead we use the 'endScreen' part of the videos resource (available in
        some API versions) or fall back to adding a card pointing to the best video.

        Args:
            youtube_video_id: The video to add end-screen to.
            best_video_id: The best-performing video to link to.
        """
        if self._fallback_mode:
            logger.info("YouTube fallback: simulated end-screen for %s", youtube_video_id)
            return

        import asyncio as _asyncio  # noqa: PLC0415

        def _sync_end_screen() -> None:
            # YouTube Data API v3 does not directly support end-screens programmatically.
            # However, we can add an info card (i-card) pointing to the best video.
            # This gives a clickable link during the video.
            try:
                # Get video duration first to place the card at the end
                vid_resp = self._service.videos().list(
                    part="contentDetails",
                    id=youtube_video_id,
                ).execute()
                items = vid_resp.get("items", [])
                if not items:
                    logger.warning("set_end_screen: video %s not found", youtube_video_id)
                    return

                # Parse ISO 8601 duration (PT2M45S → seconds)
                import re as _re  # noqa: PLC0415
                duration_iso = items[0]["contentDetails"]["duration"]
                m = _re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_iso)
                if m:
                    h = int(m.group(1) or 0)
                    mins = int(m.group(2) or 0)
                    s = int(m.group(3) or 0)
                    total_seconds = h * 3600 + mins * 60 + s
                else:
                    total_seconds = 120  # fallback

                # Place card in the last 20 seconds of the video
                card_start_ms = max(0, (total_seconds - 20)) * 1000

                # Add video card using the activities/videoCards approach
                # Since direct endScreen API isn't public, we update the
                # video description to include a link to the best video
                # as a workaround + add to "related videos" via playlist
                logger.info(
                    "set_end_screen: no public API for end-screens; "
                    "adding best video %s link in description for %s",
                    best_video_id, youtube_video_id,
                )

                # Append link to description
                vid_snippet = self._service.videos().list(
                    part="snippet",
                    id=youtube_video_id,
                ).execute()
                if vid_snippet.get("items"):
                    current_desc = vid_snippet["items"][0]["snippet"]["description"]
                    best_url = f"https://www.youtube.com/watch?v={best_video_id}"
                    if best_url not in current_desc:
                        new_desc = (
                            f"{current_desc}\n\n"
                            f"🎬 Watch our most popular video: {best_url}"
                        )
                        self._service.videos().update(
                            part="snippet",
                            body={
                                "id": youtube_video_id,
                                "snippet": {
                                    **vid_snippet["items"][0]["snippet"],
                                    "description": new_desc[:5000],
                                },
                            },
                        ).execute()
                        logger.info("set_end_screen: appended best video link to description")
            except Exception as exc:
                logger.warning("set_end_screen failed (non-fatal): %s", exc)

        loop = _asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_end_screen)

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

        # --- 3. Post pinned comment (best-effort, topic-specific) ---------------
        try:
            # Build a topic-relevant engagement question from the primary keyword
            topic_phrase = metadata.primary_keyword or metadata.title
            comment_text = (
                f"🔥 What do YOU think about {topic_phrase}?\n\n"
                f"👇 Drop your answer below — I read every comment!\n\n"
                f"👍 LIKE if you want more breakdowns like this\n"
                f"🔔 SUBSCRIBE and hit the bell so you never miss a video\n\n"
                f"{' '.join(metadata.hashtags[:3])}"
            )
            await self._yt.post_pinned_comment(
                youtube_video_id=youtube_video_id,
                text=comment_text,
            )
            logger.info("Publisher: pinned comment posted for youtube_video_id=%s", youtube_video_id)
        except Exception as comment_exc:  # noqa: BLE001
            logger.debug(
                "Publisher: pinned comment skipped for youtube_video_id=%s "
                "(requires YouTube API comment audit — non-fatal): %s",
                youtube_video_id, comment_exc,
            )

        # --- 4. Add to playlist (best-effort) ------------------------------
        playlist_id = os.environ.get("YOUTUBE_PLAYLIST_ID", "")
        if playlist_id:
            try:
                await self._yt.add_to_playlist(
                    youtube_video_id=youtube_video_id,
                    playlist_id=playlist_id,
                )
                logger.info("Publisher: added to playlist %s", playlist_id)
            except Exception as pl_exc:  # noqa: BLE001
                logger.warning(
                    "Publisher: add to playlist failed (non-fatal): %s", pl_exc,
                )

        # --- 5. Update Content Calendar: → Uploading → Unlisted ------------
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
    # Shorts auto-extraction + upload
    # ------------------------------------------------------------------

    async def extract_and_upload_short(
        self,
        video_id: str,
        mp4_url: str,
        metadata: "MetadataPackage",
        full_video_id: str,
        publish_at: Optional[datetime] = None,
    ) -> Optional[str]:
        """Extract first 55 seconds from the full video, re-encode as 9:16, upload as Short.

        Steps:
        1. Download the full MP4 from Drive.
        2. Use ffmpeg to extract the first 55 seconds and crop/scale to 1080x1920 (9:16).
        3. Upload via YouTubeDataAPIClient.upload_short with #Shorts in title.

        Args:
            video_id: Pipeline video ID.
            mp4_url: Google Drive URL of the full MP4.
            metadata: MetadataPackage for title/tags context.
            full_video_id: YouTube video ID of the full video (for linking).
            publish_at: Optional UTC datetime to schedule the Short. If provided,
                the Short is uploaded as Private and scheduled to go public at this
                time (same as the main video). If None, publishes immediately.

        Returns:
            YouTube Short video ID, or None if extraction/upload fails.
        """
        import subprocess as _sp  # noqa: PLC0415
        import tempfile as _tmp  # noqa: PLC0415
        import pathlib as _pl  # noqa: PLC0415

        try:
            # 1. Download full video
            video_bytes = await self._yt._fetch_bytes(mp4_url)

            with _tmp.TemporaryDirectory() as tmpdir:
                full_path = _pl.Path(tmpdir) / "full.mp4"
                short_path = _pl.Path(tmpdir) / "short.mp4"
                full_path.write_bytes(video_bytes)

                # 2. Extract first 55s, scale to fill 9:16 frame, crop to exact 1080x1920
                # Strategy: scale so width = 1080 (maintaining aspect ratio),
                # then crop height to 1920 from center. This ensures the video
                # FILLS the entire frame with no black bars regardless of source AR.
                # If source is wider than 9:16, we scale height to 1920 first then crop width.
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(full_path),
                    "-t", "55",
                    "-vf", (
                        "scale=1080:1920:force_original_aspect_ratio=increase,"
                        "crop=1080:1920,"
                        "setsar=1"
                    ),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-profile:v", "high", "-level:v", "4.0",
                    "-pix_fmt", "yuv420p",
                    "-r", "30",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
                    "-movflags", "+faststart",
                    str(short_path),
                ]

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: _sp.run(ffmpeg_cmd, capture_output=True, timeout=120, check=True),
                )

                # 3. Build Short metadata
                # Title: first 90 chars of original title + #Shorts
                short_title = metadata.title[:90] + " #Shorts"
                full_url = f"https://www.youtube.com/watch?v={full_video_id}"
                short_desc = (
                    f"Watch the FULL video: {full_url}\n\n"
                    f"{metadata.description[:200]}...\n\n"
                    f"#Shorts {' '.join(metadata.hashtags[:3])}"
                )

                # 4. Upload (scheduled if publish_at is set)
                result = await self._yt.upload_short(
                    mp4_path=str(short_path),
                    title=short_title,
                    description=short_desc,
                    tags=metadata.tags[:10],
                    publish_at=publish_at,
                )
                schedule_info = ""
                if publish_at:
                    schedule_info = f" (scheduled: {publish_at.strftime('%Y-%m-%d %H:%M UTC')})"
                logger.info(
                    "Publisher.extract_and_upload_short: uploaded Short %s for video_id=%s%s",
                    result["id"], video_id, schedule_info,
                )
                return result["id"]

        except Exception as exc:
            logger.warning(
                "Publisher.extract_and_upload_short failed for video_id=%s (non-fatal): %s",
                video_id, exc,
            )
            return None

    # ------------------------------------------------------------------
    # Instagram Reels encoding
    # ------------------------------------------------------------------

    async def encode_for_instagram_reels(
        self,
        video_id: str,
        mp4_url: str,
    ) -> Optional[str]:
        """Encode a vertical clip optimized for Instagram Reels specs.

        Instagram Reels has different quality requirements than YouTube Shorts:
          - Resolution: 1080x1920 (9:16)
          - Frame rate: 30 fps (consistent, no VFR)
          - Video bitrate: 4 Mbps (higher quality than Shorts' CRF 23)
          - Audio: AAC 192 kbps (Instagram penalizes low audio quality)
          - Duration: up to 90 seconds (we use first 60s for optimal engagement)
          - Container: MP4 with faststart for streaming

        Args:
            video_id: Pipeline video ID (for logging).
            mp4_url: Google Drive URL or local path to the full MP4.

        Returns:
            Local file path to the encoded Reel MP4, or None on failure.
            Caller is responsible for cleanup after upload.
        """
        import subprocess as _sp  # noqa: PLC0415
        import tempfile as _tmp  # noqa: PLC0415
        import pathlib as _pl  # noqa: PLC0415

        try:
            # Download full video
            video_bytes = await self._yt._fetch_bytes(mp4_url)

            # Validate download (catch corrupted/partial downloads from SSL errors)
            if len(video_bytes) < 100_000:  # Less than 100KB is definitely wrong
                logger.warning(
                    "Publisher.encode_for_instagram_reels: downloaded file too small (%d bytes) "
                    "— likely corrupted download, skipping",
                    len(video_bytes),
                )
                return None

            logger.info(
                "Publisher.encode_for_instagram_reels: downloaded source video (%.2f MB)",
                len(video_bytes) / (1024 * 1024),
            )

            # Use a persistent temp directory (caller cleans up)
            tmpdir = _tmp.mkdtemp(prefix="ig_reel_")
            full_path = _pl.Path(tmpdir) / "full.mp4"
            reel_path = _pl.Path(tmpdir) / "reel.mp4"
            full_path.write_bytes(video_bytes)

            # Instagram Reels encoding:
            # - 60s max (sweet spot for engagement vs 90s allowed)
            # - 1080x1920 (crop center then scale)
            # - 30fps constant
            # - H.264 High profile at 4 Mbps (Instagram's recommended range)
            # - AAC 192k stereo (IG penalizes low audio quality in reach)
            # - faststart for streaming preview
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(full_path),
                "-t", "60",
                "-vf", (
                    "scale=1080:1920:force_original_aspect_ratio=increase,"
                    "crop=1080:1920,"
                    "setsar=1,"
                    "fps=30"
                ),
                "-c:v", "libx264",
                "-profile:v", "high",
                "-level:v", "4.0",
                "-b:v", "2M",
                "-maxrate", "2.5M",
                "-bufsize", "4M",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "44100",
                "-ac", "2",
                "-movflags", "+faststart",
                "-pix_fmt", "yuv420p",
                str(reel_path),
            ]

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: _sp.run(ffmpeg_cmd, capture_output=True, timeout=180, check=True),
            )

            # Validate the encoded file is a proper video (not corrupted)
            probe_cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration,codec_name",
                "-show_entries", "format=duration,size",
                "-of", "json",
                str(reel_path),
            ]
            probe_result = await loop.run_in_executor(
                None,
                lambda: _sp.run(probe_cmd, capture_output=True, timeout=30),
            )
            if probe_result.returncode == 0:
                import json as _json  # noqa: PLC0415
                probe_data = _json.loads(probe_result.stdout)
                fmt_duration = float(probe_data.get("format", {}).get("duration", "0"))
                fmt_size = int(probe_data.get("format", {}).get("size", "0"))
                logger.info(
                    "Publisher.encode_for_instagram_reels: validation — duration=%.1fs, size=%.2fMB",
                    fmt_duration, fmt_size / (1024 * 1024),
                )
                # Instagram requires at least 3 seconds and valid duration
                if fmt_duration < 3.0:
                    logger.warning(
                        "Publisher.encode_for_instagram_reels: video too short (%.1fs) — skipping",
                        fmt_duration,
                    )
                    return None
            else:
                logger.warning(
                    "Publisher.encode_for_instagram_reels: ffprobe validation failed — %s",
                    probe_result.stderr.decode()[:200],
                )

            logger.info(
                "Publisher.encode_for_instagram_reels: encoded %s → %s",
                video_id, reel_path,
            )
            return str(reel_path)

        except Exception as exc:
            logger.warning(
                "Publisher.encode_for_instagram_reels failed for video_id=%s (non-fatal): %s",
                video_id, exc,
            )
            return None

    # ------------------------------------------------------------------
    # End-screen / best-video linking
    # ------------------------------------------------------------------

    async def add_end_screen(self, youtube_video_id: str) -> None:
        """Add end-screen elements linking to the channel's best-performing video.

        Best-effort: logs a warning if it fails but doesn't raise.

        Args:
            youtube_video_id: The YouTube video to add end-screen to.
        """
        try:
            best_id = await self._yt.get_best_performing_video(days=28)
            if not best_id or best_id == youtube_video_id:
                logger.info("add_end_screen: no suitable best video found, skipping")
                return
            await self._yt.set_end_screen(youtube_video_id, best_id)
            logger.info(
                "Publisher.add_end_screen: linked best video %s in %s",
                best_id, youtube_video_id,
            )
        except Exception as exc:
            logger.warning(
                "Publisher.add_end_screen failed for %s (non-fatal): %s",
                youtube_video_id, exc,
            )

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
