"""Bilibili_Uploader subsystem — upload videos to Bilibili (B站).

Uses the ``bilitool`` library which wraps Bilibili's web upload API with
cookie-based authentication (SESSDATA + bili_jct).

Authentication setup:
    1. Log into bilibili.com in your browser
    2. Open DevTools → Application → Cookies → bilibili.com
    3. Copy the SESSDATA and bili_jct cookie values
    4. Set BILIBILI_SESSDATA and BILIBILI_BILI_JCT in your .env / GitHub Secrets

Category IDs (tid):
    17  = Short Film (影视)
    21  = Anime (番剧)
    122 = Science & Technology
    171 = Original Content
    201 = Science Fiction (科幻)  ← recommended for sci-fi channel
    253 = Science & Education

Note: Bilibili cookies expire periodically (~30–90 days). Refresh them
by logging in again and updating your secrets.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.models import BilibiliConfig, MetadataPackage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BilibiliUploadResult:
    """Outcome of a Bilibili video upload attempt.

    Attributes:
        success: True if the video was submitted successfully.
        bvid: Bilibili video ID (e.g. "BV1xx411c7mD") on success; None otherwise.
        error: Human-readable error description on failure; None on success.
    """

    success: bool
    bvid: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# BilibiliUploader
# ---------------------------------------------------------------------------


class BilibiliUploader:
    """Upload videos to Bilibili using cookie-based authentication.

    Uses the ``bilitool`` library internally. The uploader downloads the
    compiled MP4 from Google Drive (via asset_store), writes it to a temp
    file, uploads to Bilibili, then cleans up.

    Args:
        config: :class:`~pipeline.models.BilibiliConfig` with auth credentials
            and upload defaults (tid, tags, source).
    """

    def __init__(self, config: BilibiliConfig) -> None:
        self._config = config
        self._fallback_mode = False

        if not config.sessdata or not config.bili_jct:
            logger.warning(
                "BilibiliUploader: SESSDATA or bili_jct missing — uploads will be skipped. "
                "Set BILIBILI_SESSDATA and BILIBILI_BILI_JCT to enable."
            )
            self._fallback_mode = True

    async def upload(
        self,
        mp4_bytes: bytes,
        metadata: MetadataPackage,
        cover_bytes: Optional[bytes] = None,
    ) -> BilibiliUploadResult:
        """Upload an MP4 video to Bilibili.

        Downloads the compiled video, writes to a temp file, submits to
        Bilibili via bilitool, and returns the result.

        Args:
            mp4_bytes: Raw MP4 file bytes to upload.
            metadata: :class:`~pipeline.models.MetadataPackage` used for
                title, description, and tags.
            cover_bytes: Optional JPEG thumbnail bytes for the cover image.

        Returns:
            :class:`BilibiliUploadResult` indicating success/failure and bvid.
        """
        if self._fallback_mode:
            logger.info("BilibiliUploader: skipping upload (fallback mode — no credentials)")
            return BilibiliUploadResult(
                success=False,
                error="Bilibili credentials not configured",
            )

        try:
            from bilitool import UploadController  # noqa: PLC0415
        except ImportError:
            logger.error(
                "BilibiliUploader: bilitool not installed. "
                "Run: pip install bilitool"
            )
            return BilibiliUploadResult(
                success=False,
                error="bilitool package not installed",
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._upload_sync, mp4_bytes, metadata, cover_bytes)

    def _upload_sync(
        self,
        mp4_bytes: bytes,
        metadata: MetadataPackage,
        cover_bytes: Optional[bytes],
    ) -> BilibiliUploadResult:
        """Synchronous upload logic — runs in a thread pool executor."""
        import json as _json  # noqa: PLC0415
        from bilitool import UploadController  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="bili_upload_") as tmpdir:
            tmp = Path(tmpdir)

            # Write MP4 to temp file
            mp4_path = tmp / "video.mp4"
            mp4_path.write_bytes(mp4_bytes)

            # Write cover image if provided
            cover_path: Optional[str] = None
            if cover_bytes:
                cover_file = tmp / "cover.jpg"
                cover_file.write_bytes(cover_bytes)
                cover_path = str(cover_file)

            # Write cookies file for bilitool
            cookies_path = tmp / "cookies.json"
            cookies = {
                "SESSDATA": self._config.sessdata,
                "bili_jct": self._config.bili_jct,
                "DedeUserID": "",  # not required for upload
            }
            cookies_path.write_text(_json.dumps(cookies))

            # Set cookies path env so bilitool picks it up
            original_cookies_path = os.environ.get("BILITOOL_COOKIES_PATH")
            os.environ["BILITOOL_COOKIES_PATH"] = str(cookies_path)

            try:
                # Build tags: combine metadata tags with config default tags
                tags_from_metadata = [
                    t.replace(" ", "").replace("-", "") for t in metadata.tags[:6]
                ]
                tags_from_config = [t for t in self._config.tags if t]
                all_tags = list(dict.fromkeys(tags_from_metadata + tags_from_config))[:12]
                tag_str = ",".join(all_tags)

                # Build description: use metadata description, truncated to 250 chars
                # (Bilibili desc field is more limited than YouTube)
                desc = metadata.description[:250].rstrip() + "..." if len(metadata.description) > 250 else metadata.description

                logger.info(
                    "BilibiliUploader: uploading '%s' (tid=%d, tags=%s)",
                    metadata.title, self._config.tid, tag_str,
                )

                controller = UploadController()
                controller.upload_video_entry(
                    video_path=str(mp4_path),
                    yaml="",          # no YAML config — use explicit params
                    line="",          # auto-select fastest CDN
                    tid=self._config.tid,
                    title=metadata.title,
                    desc=desc,
                    tag=tag_str,
                    source=self._config.source or "",
                    cover=cover_path or "",
                    dynamic="",       # no dynamic (story) post
                    cdn="",           # auto CDN selection
                )

                # bilitool doesn't return bvid directly from upload_video_entry
                # It prints to stdout. We log success and return without bvid for now.
                logger.info(
                    "BilibiliUploader: upload submitted successfully for '%s'",
                    metadata.title,
                )
                return BilibiliUploadResult(success=True, bvid=None)

            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "BilibiliUploader: upload failed for '%s': %s",
                    metadata.title, exc,
                )
                return BilibiliUploadResult(success=False, error=str(exc))

            finally:
                # Restore original env var
                if original_cookies_path is not None:
                    os.environ["BILITOOL_COOKIES_PATH"] = original_cookies_path
                elif "BILITOOL_COOKIES_PATH" in os.environ:
                    del os.environ["BILITOOL_COOKIES_PATH"]


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "BilibiliUploader",
    "BilibiliUploadResult",
]
