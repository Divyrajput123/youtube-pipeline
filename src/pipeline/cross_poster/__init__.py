"""Cross_Poster subsystem — distribute published YouTube videos to social platforms.

Supports X (Twitter), LinkedIn, Instagram, and Facebook.  For each enabled platform
a platform-native caption is generated from the video's ``MetadataPackage`` and the
post is submitted with per-platform retry logic.  Failures on one platform are fully
isolated: the remaining platforms always receive their posts regardless.

SLA:
    ``Cross_Poster.post`` **must be called within 30 minutes of the video transitioning
    to the** ``Published`` **pipeline state**.  The caller (Orchestrator) is responsible
    for triggering this method promptly after the ``Published`` transition; this class
    does not enforce the 30-minute window itself but documents it here so implementors
    are aware of the constraint.

Retry policy (per platform):
    - 2 attempts total.
    - Fixed 60-second wait between attempt 1 and attempt 2.
    - If both attempts fail the platform is logged and Notifier is called with the
      platform name and failure reason.  No further attempts are made.

Caption building strategy (by platform):
    Core template: ``"{title}\\n\\n{video_url}\\n\\n{hashtags joined by space}"``

    LinkedIn / Instagram additionally include the first 200 characters of the video
    description between the title and the URL when they fit within the character budget.

    If the complete caption exceeds the platform character limit the builder
    progressively degrades:
    1. Remove description (LinkedIn / Instagram only).
    2. Reduce hashtags to the minimum required 3.
    3. The result is always within the character limit and always contains ≥ 3 hashtags.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from pipeline.models import (
    CrossPostingConfig,
    MetadataPackage,
    Platform,
    PlatformCrossPostConfig,
)
from pipeline.notifier import Notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Character limits per platform
# ---------------------------------------------------------------------------

_CHAR_LIMITS: dict[Platform, int] = {
    Platform.X: 280,
    Platform.LINKEDIN: 3_000,
    Platform.INSTAGRAM: 2_200,
    Platform.FACEBOOK: 500,
}

# Minimum hashtags that must appear in every caption.
_MIN_HASHTAGS = 3

# Number of description characters to include for LinkedIn / Instagram captions.
_DESCRIPTION_SNIPPET_LEN = 200

# Retry configuration
_MAX_ATTEMPTS = 2
_RETRY_DELAY_SECONDS = 60.0

# ---------------------------------------------------------------------------
# Platform client protocol and stub implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class PlatformClient(Protocol):
    """Protocol that every social-platform API client must satisfy.

    A ``PlatformClient`` posts a pre-formatted caption string to its platform
    and returns the post identifier or URL assigned by that platform.
    """

    async def post_update(self, caption: str) -> str:
        """Publish ``caption`` on the platform.

        Args:
            caption: Fully formatted, platform-native caption string.  Must
                respect the platform's character limit before being passed here.

        Returns:
            A platform-assigned post identifier or URL (non-empty string).

        Raises:
            NotImplementedError: Stub implementations always raise this.
            Exception: Any network or API error surfaces as a plain exception.
        """
        ...


class XAPIClient:
    """Stub X (Twitter) API client.

    Replace with a real implementation backed by the X API v2 endpoints
    (``POST /2/tweets``) and OAuth 2.0 / OAuth 1.0a credentials supplied
    via ``PlatformCrossPostConfig.api_key``.
    """

    def __init__(self, config: PlatformCrossPostConfig) -> None:
        self._config = config

    async def post_update(self, caption: str) -> str:  # noqa: ARG002
        raise NotImplementedError("XAPIClient.post_update is not yet implemented")


class LinkedInAPIClient:
    """Stub LinkedIn API client.

    Replace with a real implementation backed by the LinkedIn Share API v2
    (``POST /v2/ugcPosts``) and the ``access_token`` from
    ``PlatformCrossPostConfig``.
    """

    def __init__(self, config: PlatformCrossPostConfig) -> None:
        self._config = config

    async def post_update(self, caption: str) -> str:  # noqa: ARG002
        raise NotImplementedError("LinkedInAPIClient.post_update is not yet implemented")


class InstagramAPIClient:
    """Stub Instagram API client.

    Replace with a real implementation backed by the Instagram Graph API
    (``POST /{ig-user-id}/media`` + ``POST /{ig-user-id}/media_publish``) and
    the ``access_token`` from ``PlatformCrossPostConfig``.
    """

    def __init__(self, config: PlatformCrossPostConfig) -> None:
        self._config = config

    async def post_update(self, caption: str) -> str:  # noqa: ARG002
        raise NotImplementedError("InstagramAPIClient.post_update is not yet implemented")


class FacebookAPIClient:
    """Stub Facebook API client.

    Replace with a real implementation backed by the Facebook Graph API
    (``POST /{page-id}/feed``) and the ``page_access_token`` from
    ``PlatformCrossPostConfig``.
    """

    def __init__(self, config: PlatformCrossPostConfig) -> None:
        self._config = config

    async def post_update(self, caption: str) -> str:  # noqa: ARG002
        raise NotImplementedError("FacebookAPIClient.post_update is not yet implemented")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PostResult:
    """Outcome of a single platform cross-post attempt.

    Attributes:
        platform: The social platform this result corresponds to.
        success: ``True`` if the post was published successfully.
        post_url: Platform-assigned URL or post ID when ``success`` is ``True``;
            ``None`` otherwise.
        error: Human-readable error description when ``success`` is ``False``;
            ``None`` when ``success`` is ``True``.
    """

    platform: Platform
    success: bool
    post_url: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Caption builder
# ---------------------------------------------------------------------------


def _build_caption(
    platform: Platform,
    title: str,
    description: str,
    video_url: str,
    hashtags: list[str],
) -> str:
    """Build a platform-native caption that respects the character limit.

    Strategy:
        1. Always start with the core template:
           ``"{title}\\n\\n{video_url}\\n\\n{hashtags}"``
        2. For LinkedIn and Instagram: attempt to include the first
           ``_DESCRIPTION_SNIPPET_LEN`` characters of the description between
           the title and the URL.
        3. If the full caption exceeds the character limit, progressively degrade:
           a. Remove the description snippet.
           b. Reduce hashtags down to ``_MIN_HASHTAGS`` (keeping the first three).
        4. The returned caption always fits within the limit and always contains
           at least ``_MIN_HASHTAGS`` hashtags.

    Args:
        platform: Target platform (determines character limit and whether description
            is included).
        title: Video title from ``MetadataPackage``.
        description: Full video description from ``MetadataPackage``.
        video_url: Published YouTube URL for the video.
        hashtags: All hashtags from ``MetadataPackage`` (must have ≥ 3 entries).

    Returns:
        Formatted caption string within the platform character limit, containing
        ≥ ``_MIN_HASHTAGS`` hashtags.
    """
    limit = _CHAR_LIMITS[platform]
    include_description = platform in (Platform.LINKEDIN, Platform.INSTAGRAM)

    # Ensure we always have at least _MIN_HASHTAGS available.
    safe_hashtags = hashtags if len(hashtags) >= _MIN_HASHTAGS else hashtags + ["#video"] * (
        _MIN_HASHTAGS - len(hashtags)
    )

    description_snippet = description[:_DESCRIPTION_SNIPPET_LEN].rstrip()

    def _assemble(use_description: bool, tag_list: list[str]) -> str:
        hashtag_str = " ".join(tag_list)
        if use_description and description_snippet:
            return f"{title}\n\n{description_snippet}\n\n{video_url}\n\n{hashtag_str}"
        return f"{title}\n\n{video_url}\n\n{hashtag_str}"

    # Attempt 1: full caption with description (where applicable) and all hashtags.
    candidate = _assemble(include_description, safe_hashtags)
    if len(candidate) <= limit:
        return candidate

    # Attempt 2: drop description, keep all hashtags.
    candidate = _assemble(False, safe_hashtags)
    if len(candidate) <= limit:
        return candidate

    # Attempt 3: drop description, reduce to minimum 3 hashtags.
    min_tags = safe_hashtags[:_MIN_HASHTAGS]
    candidate = _assemble(False, min_tags)
    if len(candidate) <= limit:
        return candidate

    # Last resort: truncate the title to make it fit with core + 3 hashtags.
    # This should be unreachable in practice given the generous per-platform limits,
    # but we guard against extreme edge cases (e.g. very long title + URL + hashtags).
    hashtag_str = " ".join(min_tags)
    # Calculate how much room is left for the title.
    # Template without title: "\n\n{video_url}\n\n{hashtag_str}"
    suffix = f"\n\n{video_url}\n\n{hashtag_str}"
    title_budget = limit - len(suffix)
    if title_budget < 1:
        # Absolute minimum: just the first hashtag and URL, truncated URL if needed.
        return (suffix[:limit])
    truncated_title = title[:title_budget]
    return f"{truncated_title}{suffix}"


# ---------------------------------------------------------------------------
# Cross_Poster
# ---------------------------------------------------------------------------


class Cross_Poster:
    """Distribute a published YouTube video across configured social platforms.

    SLA: ``post`` must be invoked within 30 minutes of the video transitioning to
    the ``Published`` pipeline state.  The Orchestrator is responsible for calling
    this method promptly after that transition.

    Args:
        config: ``CrossPostingConfig`` describing which platforms are enabled and
            their API credentials.
        notifier: ``Notifier`` instance used to send failure alerts when a platform
            post fails after all retries.

    Usage::

        cross_poster = Cross_Poster(config=pipeline_config.cross_posting,
                                    notifier=notifier)
        results = await cross_poster.post(
            video_url="https://youtu.be/abc123",
            metadata=metadata_package,
            platforms=[Platform.X, Platform.LINKEDIN],
        )
    """

    def __init__(self, config: CrossPostingConfig, notifier: Notifier) -> None:
        self._config = config
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def post(
        self,
        video_url: str,
        metadata: MetadataPackage,
        platforms: list[Platform],
    ) -> list[PostResult]:
        """Post the video to all requested, enabled social platforms.

        Platforms that are disabled in ``CrossPostingConfig`` are silently skipped
        (no ``PostResult`` entry, no notification, no log entry).

        Each enabled platform is attempted independently.  A failure on one platform
        does not prevent posting to the remaining platforms.

        Per-platform retry:
            - Up to ``_MAX_ATTEMPTS`` (2) total attempts.
            - ``_RETRY_DELAY_SECONDS`` (60 s) fixed wait between attempts.
            - On exhaustion: log the error and call ``Notifier.send_failure_alert``.

        Args:
            video_url: Publicly accessible YouTube URL for the published video.
            metadata: ``MetadataPackage`` providing title, description, tags,
                and hashtags for caption construction.
            platforms: Subset of ``Platform`` values to attempt.  Duplicates are
                silently de-duplicated.

        Returns:
            A list of ``PostResult`` objects — one per *enabled* platform in
            ``platforms``.  Disabled platforms produce no entry.
        """
        results: list[PostResult] = []

        for platform in dict.fromkeys(platforms):  # preserve order, deduplicate
            platform_cfg = self._platform_config(platform)

            # Disabled platforms are skipped silently.
            if not platform_cfg.enabled:
                logger.debug("Skipping disabled platform: %s", platform.value)
                continue

            result = await self._post_to_platform(
                platform=platform,
                config=platform_cfg,
                video_url=video_url,
                metadata=metadata,
            )
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _platform_config(self, platform: Platform) -> PlatformCrossPostConfig:
        """Return the ``PlatformCrossPostConfig`` for the given platform."""
        return {
            Platform.X: self._config.x,
            Platform.LINKEDIN: self._config.linkedin,
            Platform.INSTAGRAM: self._config.instagram,
            Platform.FACEBOOK: self._config.facebook,
        }[platform]

    def _build_client(
        self, platform: Platform, config: PlatformCrossPostConfig
    ) -> PlatformClient:
        """Instantiate the API client stub for the given platform.

        In production, replace the stub constructors with real implementations
        that read credentials from ``config``.
        """
        return {
            Platform.X: XAPIClient,
            Platform.LINKEDIN: LinkedInAPIClient,
            Platform.INSTAGRAM: InstagramAPIClient,
            Platform.FACEBOOK: FacebookAPIClient,
        }[platform](config)

    async def _post_to_platform(
        self,
        platform: Platform,
        config: PlatformCrossPostConfig,
        video_url: str,
        metadata: MetadataPackage,
    ) -> PostResult:
        """Attempt to post to a single platform with retry logic.

        Args:
            platform: Target platform.
            config: Platform-specific credentials and toggle.
            video_url: YouTube URL to include in the caption.
            metadata: ``MetadataPackage`` for caption content.

        Returns:
            ``PostResult`` reflecting the final outcome (success or failure).
        """
        caption = _build_caption(
            platform=platform,
            title=metadata.title,
            description=metadata.description,
            video_url=video_url,
            hashtags=metadata.hashtags,
        )

        client = self._build_client(platform, config)
        last_error: Optional[str] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                logger.info(
                    "Cross-posting to %s (attempt %d/%d)",
                    platform.value,
                    attempt,
                    _MAX_ATTEMPTS,
                )
                post_url = await client.post_update(caption)
                logger.info(
                    "Successfully posted to %s: %s", platform.value, post_url
                )
                return PostResult(
                    platform=platform,
                    success=True,
                    post_url=post_url,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning(
                    "Attempt %d/%d failed for platform %s: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    platform.value,
                    last_error,
                )
                if attempt < _MAX_ATTEMPTS:
                    logger.info(
                        "Waiting %s s before retry for platform %s",
                        _RETRY_DELAY_SECONDS,
                        platform.value,
                    )
                    await asyncio.sleep(_RETRY_DELAY_SECONDS)

        # All attempts exhausted — log and notify.
        failure_reason = last_error or "Unknown error"
        logger.error(
            "All %d attempts failed for platform %s. Reason: %s",
            _MAX_ATTEMPTS,
            platform.value,
            failure_reason,
        )
        self._notifier.send_failure_alert(
            video_id="cross_poster",  # video_id is not threaded here; use stage name
            stage_name=f"cross_poster/{platform.value}",
            error_message=(
                f"Cross-posting to {platform.value} failed after "
                f"{_MAX_ATTEMPTS} attempts. Reason: {failure_reason}"
            ),
        )
        return PostResult(
            platform=platform,
            success=False,
            error=failure_reason,
        )


__all__ = [
    "Cross_Poster",
    "PostResult",
    "PlatformClient",
    "XAPIClient",
    "LinkedInAPIClient",
    "InstagramAPIClient",
    "FacebookAPIClient",
]
