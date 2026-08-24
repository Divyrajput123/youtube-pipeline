"""Instagram Reels subsystem — upload vertical short-form video as Instagram Reels.

Uses the Instagram Graph API (via Facebook's Content Publishing API) to:
  1. Generate an SEO-optimized Reel caption with hashtags, keywords, and CTA.
  2. Upload the 9:16 vertical clip (same one used for YouTube Shorts) as a Reel.
  3. Publish the Reel container once the upload is processed.

Requirements:
  - A Facebook Page connected to an Instagram Professional (Business/Creator) account.
  - A long-lived access token with permissions:
      instagram_basic, instagram_content_publish, pages_read_engagement,
      pages_show_list (required for facebook_reels_sync_data mirroring to a Facebook Page)
  - The Instagram Account ID (numeric, not the @username).

Retry policy:
  - Upload init: 3 attempts, 30 s exponential backoff.
  - Publish (container status polling): up to 60 s with 5 s intervals.

SEO Strategy for Reels:
  - Primary keyword in the first line (Instagram search indexes captions).
  - 20-30 relevant hashtags (mix of high-volume and niche).
  - Hook line + CTA for engagement signals (saves, shares, comments).
  - Mention the full YouTube video to drive cross-platform traffic.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from pipeline.config import is_production_mode
from pipeline.models import MetadataPackage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GRAPH_API_BASE = "https://graph.facebook.com/v21.0"

_UPLOAD_ATTEMPTS = 3
_UPLOAD_BASE_DELAY_S = 30.0
_UPLOAD_MAX_DELAY_S = 120.0

# Container status polling
_POLL_INTERVAL_S = 10.0
_POLL_TIMEOUT_S = 900.0  # 15 minutes — large video files need time to download + transcode

# Instagram caption limit
_CAPTION_CHAR_LIMIT = 2_200

# Maximum hashtags for Reels SEO (Instagram allows up to 30)
_MAX_HASHTAGS = 25
_MIN_HASHTAGS = 10


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InstagramReelsError(Exception):
    """Raised when a Reel upload or publish fails after retries."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReelResult:
    """Outcome of an Instagram Reel upload.

    Attributes:
        success: True if published successfully.
        reel_id: Instagram media ID of the published Reel.
        permalink: Direct link to the Reel on Instagram.
        error: Error message if failed.
    """

    success: bool
    reel_id: Optional[str] = None
    permalink: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SEO Caption Builder
# ---------------------------------------------------------------------------


def build_alt_text(metadata: MetadataPackage) -> str:
    """Build alt text for Instagram Reel (max 100 chars).

    Instagram uses alt text for:
      - Search/Explore ranking (confirmed by Instagram engineering)
      - Screen reader accessibility
      - Image recognition fallback

    Strategy: primary keyword + title, trimmed to 100 chars.
    """
    primary = metadata.primary_keyword or ""
    title = metadata.title or ""

    # Combine keyword and title, avoiding duplication
    if primary.lower() in title.lower():
        alt = title
    else:
        alt = f"{primary} — {title}"

    return alt[:100]


def build_reel_caption(
    metadata: MetadataPackage,
    youtube_url: str,
    extra_hashtags: Optional[list[str]] = None,
) -> str:
    """Build an SEO-optimized Instagram Reel caption.

    Structure (optimized for discoverability + YouTube redirect):
      1. YouTube redirect CTA — prominent first line so it's visible before "more".
      2. Hook line with primary keyword (Instagram search indexes first lines).
      3. Brief description (first 150 chars of video description).
      4. Engagement CTA (save/share/comment — boosts algorithm signals).
      5. Hashtag block (20-25 hashtags: metadata hashtags + niche expansions).

    Instagram shows only the first ~125 chars before "more", so the YouTube
    link and hook are placed first to maximize redirect clicks.

    Args:
        metadata: Video MetadataPackage with title, description, tags, hashtags.
        youtube_url: Full YouTube video URL for cross-promotion.
        extra_hashtags: Optional additional hashtags to include.

    Returns:
        Caption string within Instagram's 2200 char limit.
    """
    # --- 1. YouTube redirect (FIRST — visible before "more" tap) ---
    # Note: Instagram does not make URLs in captions clickable — this is a
    # platform limitation. Direct users to the link in bio instead.
    yt_redirect = f"🎬 Full video on YouTube — link in bio 👆"

    # --- 2. Hook line using primary keyword ---
    primary_kw = metadata.primary_keyword or metadata.title
    hook = f"{primary_kw}"

    # --- 3. Brief description ---
    desc_snippet = metadata.description[:150].rstrip()
    if len(metadata.description) > 150:
        desc_snippet += "..."

    # --- 4. Engagement CTA ---
    cta = (
        "Save this reel for later\n"
        "Share with someone who needs to see this\n"
        "Follow for more breakdowns like this"
    )

    # --- 5. Hashtag block (tiered strategy) ---
    # Tier 1: High-volume (1M+ posts) — broad discoverability
    # Tier 2: Mid-volume (100K-1M) — niche targeting
    # Tier 3: Low-volume (<100K) — topic-specific, less competition
    # Tier 4: Engagement boosters — signals for Explore page
    #
    # The mix ensures: broad reach (Tier 1) + relevant audience (Tier 2/3)
    # + algorithm boost (Tier 4). Instagram recommends 3-5 highly relevant
    # hashtags, but Reels perform better with 20-25 covering all tiers.

    hashtags: list[str] = []

    # Tier 1: High-volume entertainment/superhero tags (5 slots)
    tier1_pool = [
        "#marvel", "#dc", "#superhero", "#mcu", "#comics",
        "#avengers", "#dccomics", "#marvelcomics", "#anime", "#superheroes",
        "#batman", "#spiderman", "#xmen", "#manga", "#entertainment",
    ]

    # Tier 2: Mid-volume niche tags derived from video content (10 slots)
    # Built from the video's actual tags and primary keyword
    tier2_tags: list[str] = []
    # Use metadata hashtags first (they're already topic-specific)
    for ht in metadata.hashtags:
        if ht not in tier2_tags:
            tier2_tags.append(ht)
    # Convert video tags to hashtags
    for tag in metadata.tags:
        ht = "#" + tag.replace(" ", "").replace("-", "").lower()
        if ht not in tier2_tags and len(ht) <= 30:
            tier2_tags.append(ht)

    # Tier 3: Topic-specific long-tail tags (5 slots)
    # Generated from primary keyword — these have less competition
    primary_kw_tag = "#" + (metadata.primary_keyword or "").replace(" ", "").lower()
    tier3_tags: list[str] = []
    if primary_kw_tag and len(primary_kw_tag) > 1:
        tier3_tags.append(primary_kw_tag)
    # Add compound tags from title words
    title_words = [w.lower() for w in metadata.title.split() if len(w) > 3]
    for i in range(min(len(title_words) - 1, 4)):
        compound = "#" + title_words[i] + title_words[i + 1]
        if compound not in tier3_tags and len(compound) <= 30:
            tier3_tags.append(compound)

    # Tier 4: Engagement/discovery boosters (5 slots)
    tier4_tags = [
        "#reels", "#explorepage", "#viral",
        "#fyp", "#reelsinstagram",
    ]

    # Assemble: pick from each tier to fill 25 slots
    # Tier 1: pick 5 that are relevant (check if any match video tags)
    relevant_tier1 = [t for t in tier1_pool if any(
        t[1:] in tag.lower() for tag in metadata.tags
    )]
    remaining_tier1 = [t for t in tier1_pool if t not in relevant_tier1]
    selected_tier1 = (relevant_tier1 + remaining_tier1)[:5]

    # Tier 2: first 10
    selected_tier2 = tier2_tags[:10]

    # Tier 3: first 5
    selected_tier3 = tier3_tags[:5]

    # Tier 4: all 5
    selected_tier4 = tier4_tags[:5]

    # Combine all tiers, deduplicating
    for ht in selected_tier1 + selected_tier2 + selected_tier3 + selected_tier4:
        if ht not in hashtags and len(hashtags) < _MAX_HASHTAGS:
            hashtags.append(ht)

    # Add user's extra_hashtags from config (niche overrides)
    if extra_hashtags:
        for ht in extra_hashtags:
            if not ht.startswith("#"):
                ht = f"#{ht}"
            if ht not in hashtags and len(hashtags) < _MAX_HASHTAGS:
                hashtags.append(ht)

    hashtag_block = " ".join(hashtags[:_MAX_HASHTAGS])

    # --- Assemble caption ---
    # YouTube redirect is FIRST so it shows in the preview before "...more"
    caption = (
        f"{yt_redirect}\n\n"
        f"{hook}\n\n"
        f"{desc_snippet}\n\n"
        f"{cta}\n\n"
        f".\n.\n.\n"
        f"{hashtag_block}"
    )

    # Trim to Instagram limit if needed
    if len(caption) > _CAPTION_CHAR_LIMIT:
        # Reduce hashtags until it fits
        while len(caption) > _CAPTION_CHAR_LIMIT and len(hashtags) > _MIN_HASHTAGS:
            hashtags.pop()
            hashtag_block = " ".join(hashtags)
            caption = (
                f"{yt_redirect}\n\n"
                f"{hook}\n\n"
                f"{desc_snippet}\n\n"
                f"{cta}\n\n"
                f".\n.\n.\n"
                f"{hashtag_block}"
            )

    return caption[:_CAPTION_CHAR_LIMIT]


# ---------------------------------------------------------------------------
# Instagram Reels Client
# ---------------------------------------------------------------------------


class InstagramReelsClient:
    """Upload and publish Instagram Reels via the Graph API.

    Uses the Content Publishing API flow:
      1. POST /{ig-user-id}/media  — create a Reel container (video_url + caption).
      2. Poll GET /{container-id}?fields=status_code — wait for FINISHED.
      3. POST /{ig-user-id}/media_publish — publish the container.

    Token auto-refresh:
      On the first API call of each pipeline run, the client checks if the
      token is within 7 days of expiry and refreshes it automatically.
      Requires FACEBOOK_APP_ID and FACEBOOK_APP_SECRET in .env.

    Args:
        access_token: Long-lived Facebook/Instagram access token.
        instagram_account_id: Numeric Instagram Business/Creator account ID.

    The token needs these permissions:
      - instagram_basic
      - instagram_content_publish
      - pages_read_engagement
      - pages_show_list (required when facebook_page_id is set for Reel mirroring)
    """

    def __init__(
        self,
        access_token: str,
        instagram_account_id: str,
        facebook_page_id: Optional[str] = None,
    ) -> None:
        self._access_token = access_token
        self._ig_account_id = instagram_account_id
        self._facebook_page_id = facebook_page_id or ""
        self._fallback_mode = False
        self._token_checked = False  # Only check once per pipeline run

        if not access_token or not instagram_account_id:
            if is_production_mode():
                raise ValueError(
                    "Production mode requires Instagram Reels credentials: "
                    "INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID."
                )
            logger.warning(
                "Instagram Reels credentials missing — using fallback mode. "
                "Set INSTAGRAM_ACCESS_TOKEN and INSTAGRAM_ACCOUNT_ID to enable."
            )
            self._fallback_mode = True

        if facebook_page_id:
            logger.info(
                "InstagramReelsClient: Facebook Page mirroring enabled (page_id=%s)",
                facebook_page_id,
            )

    async def _ensure_valid_token(self) -> None:
        """Check and refresh the token if it's near expiry. Called once per run."""
        if self._token_checked or self._fallback_mode:
            return
        self._token_checked = True

        try:
            from pipeline.instagram_reels.token_refresh import check_and_refresh_token

            refreshed = await check_and_refresh_token(
                access_token=self._access_token,
                app_id=os.environ.get("FACEBOOK_APP_ID", ""),
                app_secret=os.environ.get("FACEBOOK_APP_SECRET", ""),
            )
            if refreshed != self._access_token:
                self._access_token = refreshed
                logger.info("Instagram token auto-refreshed for this session")
        except Exception as exc:
            logger.warning("Token refresh check failed (non-fatal): %s", exc)

    async def upload_reel(
        self,
        video_url: str,
        caption: str,
        cover_url: Optional[str] = None,
        share_to_feed: bool = True,
        scheduled_publish_time: Optional[datetime] = None,
        alt_text: Optional[str] = None,
    ) -> ReelResult:
        """Upload a video as an Instagram Reel.

        The video must be hosted at a publicly accessible URL. For local files,
        the caller must first upload to a public hosting service (e.g., the
        pipeline's Google Drive with link sharing enabled).

        Args:
            video_url: Public HTTPS URL to the vertical MP4 file (9:16, <=90s).
            caption: SEO-optimized caption (built via build_reel_caption).
            cover_url: Optional public URL to a custom cover image.
            share_to_feed: Whether to also share the Reel to the main feed grid.
            scheduled_publish_time: Optional UTC datetime to schedule the Reel.
                If provided, the Reel will be created as a container and published
                at the specified time (must be 10 min to 75 days in the future).
            alt_text: Optional accessibility text (up to 100 chars). Used by
                Instagram for search indexing and screen readers.
                If None, the Reel publishes immediately.

        Returns:
            ReelResult with success status, reel_id, and permalink.
        """
        # Auto-refresh token if near expiry (once per pipeline run)
        await self._ensure_valid_token()

        if self._fallback_mode:
            import uuid
            fake_id = f"REEL_{uuid.uuid4().hex[:11]}"
            schedule_info = ""
            if scheduled_publish_time:
                schedule_info = f" (scheduled for {scheduled_publish_time.isoformat()})"
            logger.info("Instagram Reels fallback: simulated upload -> %s%s", fake_id, schedule_info)
            return ReelResult(
                success=True,
                reel_id=fake_id,
                permalink=f"https://www.instagram.com/reel/{fake_id}/",
            )

        try:
            # Step 1: Create Reel container
            container_id = await self._create_container(
                video_url=video_url,
                caption=caption,
                cover_url=cover_url,
                share_to_feed=share_to_feed,
                scheduled_publish_time=scheduled_publish_time,
                alt_text=alt_text,
            )

            # Step 2: Poll until ready
            await self._wait_for_container(container_id)

            # Step 3: Publish (or schedule)
            reel_id = await self._publish_container(
                container_id,
                scheduled_publish_time=scheduled_publish_time,
            )

            # Step 4: Get permalink
            permalink = await self._get_permalink(reel_id)

            if scheduled_publish_time:
                logger.info(
                    "Instagram Reel scheduled: reel_id=%s permalink=%s publish_at=%s",
                    reel_id, permalink, scheduled_publish_time.isoformat(),
                )
            else:
                logger.info(
                    "Instagram Reel published: reel_id=%s permalink=%s",
                    reel_id, permalink,
                )
            return ReelResult(success=True, reel_id=reel_id, permalink=permalink)

        except Exception as exc:
            error_msg = f"Instagram Reel upload failed: {exc}"
            logger.error(error_msg)
            return ReelResult(success=False, error=error_msg)

    # ------------------------------------------------------------------
    # Internal Graph API calls
    # ------------------------------------------------------------------

    async def _create_container(
        self,
        video_url: str,
        caption: str,
        cover_url: Optional[str],
        share_to_feed: bool,
        scheduled_publish_time: Optional[datetime] = None,
        alt_text: Optional[str] = None,
    ) -> str:
        """Create a Reel media container and return its ID.

        If scheduled_publish_time is provided, the container includes a
        `published` field set to False so the Reel can be scheduled
        via media_publish with a publish time.

        Retries up to _UPLOAD_ATTEMPTS times with exponential backoff.
        """
        url = f"{_GRAPH_API_BASE}/{self._ig_account_id}/media"
        params: dict[str, Any] = {
            "media_type": "REELS",
            "video_url": video_url,
            "caption": caption,
            "share_to_feed": str(share_to_feed).lower(),
            "access_token": self._access_token,
        }
        if cover_url:
            params["cover_url"] = cover_url

        # Mirror Reel to connected Facebook Page automatically.
        # When facebook_page_id is set, the Graph API cross-posts the Reel
        # to the linked Facebook Page feed — no separate API call needed.
        # The page must be linked to the Instagram Business account in
        # Meta Business Suite (Settings → Linked Accounts).
        if self._facebook_page_id:
            import json as _json  # noqa: PLC0415
            params["facebook_reels_sync_data"] = _json.dumps({
                "fb_page_id": self._facebook_page_id,
            })
            logger.info(
                "InstagramReelsClient: will mirror Reel to Facebook Page %s",
                self._facebook_page_id,
            )

        # For scheduled publishing: Instagram Content Publishing API uses
        # the `published` field set to false at container creation, then
        # publishes at the specified time via media_publish endpoint.
        if scheduled_publish_time:
            params["published"] = "false"

        last_error: Optional[Exception] = None
        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(url, data=params)
                    resp.raise_for_status()
                    data = resp.json()
                    container_id = data["id"]
                    logger.info(
                        "Instagram Reel container created: %s (attempt %d)",
                        container_id, attempt,
                    )
                    return container_id
            except Exception as exc:
                last_error = exc
                # Log the response body for debugging 400/401 errors
                error_detail = ""
                if hasattr(exc, "response") and exc.response is not None:
                    try:
                        error_detail = f" Response: {exc.response.text[:500]}"
                    except Exception:
                        pass
                if attempt < _UPLOAD_ATTEMPTS:
                    delay = min(
                        _UPLOAD_BASE_DELAY_S * (2 ** (attempt - 1)),
                        _UPLOAD_MAX_DELAY_S,
                    )
                    logger.warning(
                        "Reel container creation failed (attempt %d/%d): %s%s — retrying in %.0fs",
                        attempt, _UPLOAD_ATTEMPTS, exc, error_detail, delay,
                    )
                    await asyncio.sleep(delay)

        raise InstagramReelsError(
            f"Failed to create Reel container after {_UPLOAD_ATTEMPTS} attempts: {last_error}"
        )

    async def _wait_for_container(self, container_id: str) -> None:
        """Poll container status until it reaches FINISHED or errors out."""
        url = f"{_GRAPH_API_BASE}/{container_id}"
        params = {
            "fields": "status_code,status",
            "access_token": self._access_token,
        }

        elapsed = 0.0
        while elapsed < _POLL_TIMEOUT_S:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            status_code = data.get("status_code", "").upper()

            if status_code == "FINISHED":
                logger.info("Reel container %s is FINISHED", container_id)
                return
            elif status_code == "ERROR":
                error_detail = data.get("status", "Unknown error")
                raise InstagramReelsError(
                    f"Reel container {container_id} failed: {error_detail}"
                )
            elif status_code == "EXPIRED":
                raise InstagramReelsError(
                    f"Reel container {container_id} expired before publishing"
                )

            # Still IN_PROGRESS — wait and poll again
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

        raise InstagramReelsError(
            f"Reel container {container_id} timed out after {_POLL_TIMEOUT_S}s"
        )

    async def _publish_container(
        self,
        container_id: str,
        scheduled_publish_time: Optional[datetime] = None,
    ) -> str:
        """Publish the finished container and return the published media ID.

        If scheduled_publish_time is set, passes the Unix timestamp so Instagram
        schedules the Reel for that time (must be 10 min to 75 days in the future).
        """
        url = f"{_GRAPH_API_BASE}/{self._ig_account_id}/media_publish"
        params: dict[str, Any] = {
            "creation_id": container_id,
            "access_token": self._access_token,
        }

        if scheduled_publish_time:
            # Instagram expects Unix timestamp for scheduled publishing
            unix_ts = int(scheduled_publish_time.timestamp())
            params["published"] = "false"
            params["scheduled_publish_time"] = str(unix_ts)
            logger.info(
                "Scheduling Reel container %s for %s (unix: %d)",
                container_id, scheduled_publish_time.isoformat(), unix_ts,
            )

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, data=params)
            resp.raise_for_status()
            data = resp.json()
            return data["id"]

    async def _get_permalink(self, media_id: str) -> Optional[str]:
        """Fetch the permalink for a published media item."""
        url = f"{_GRAPH_API_BASE}/{media_id}"
        params = {
            "fields": "permalink",
            "access_token": self._access_token,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                return data.get("permalink")
        except Exception as exc:
            logger.warning("Could not fetch Reel permalink: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Convenience function for pipeline integration
# ---------------------------------------------------------------------------


async def upload_reel_from_short(
    client: InstagramReelsClient,
    video_public_url: str,
    metadata: MetadataPackage,
    youtube_url: str,
    thumbnail_url: Optional[str] = None,
    extra_hashtags: Optional[list[str]] = None,
    scheduled_publish_time: Optional[datetime] = None,
) -> ReelResult:
    """High-level helper: build SEO caption and upload a Reel.

    This is the main entry point called by the Orchestrator after Shorts extraction.
    Uses the Instagram-optimized encoded clip (separate from YouTube Shorts).

    Args:
        client: Configured InstagramReelsClient.
        video_public_url: Public URL to the vertical MP4 (shared via Drive or CDN).
        metadata: Video MetadataPackage for caption generation.
        youtube_url: Full YouTube video URL for cross-promotion link.
        thumbnail_url: Optional cover image URL.
        extra_hashtags: Optional niche hashtags to boost discoverability.
        scheduled_publish_time: Optional UTC datetime to align Reel publishing
            with the YouTube video schedule. If None, publishes immediately.

    Returns:
        ReelResult indicating success/failure.
    """
    caption = build_reel_caption(
        metadata=metadata,
        youtube_url=youtube_url,
        extra_hashtags=extra_hashtags,
    )

    schedule_info = ""
    if scheduled_publish_time:
        schedule_info = f", scheduled for {scheduled_publish_time.isoformat()}"

    logger.info(
        "Uploading Instagram Reel for video: %s (caption length: %d chars, hashtags: %d%s)",
        metadata.video_id,
        len(caption),
        caption.count("#"),
        schedule_info,
    )

    return await client.upload_reel(
        video_url=video_public_url,
        caption=caption,
        cover_url=thumbnail_url,
        share_to_feed=True,
        scheduled_publish_time=scheduled_publish_time,
        alt_text=build_alt_text(metadata),
    )


__all__ = [
    "InstagramReelsClient",
    "InstagramReelsError",
    "ReelResult",
    "build_alt_text",
    "build_reel_caption",
    "upload_reel_from_short",
]
