"""Instagram Reel Manager — daily auto-posting with duplicate prevention.

Modes:
  daily   — Automatically post 1 Reel for the next unposted video (for cron)
  publish — Post a Reel for a specific video ID (manual)
  test    — Post a test Reel from any existing video

Usage:
  python scripts/instagram_reel_manager.py --mode daily
  python scripts/instagram_reel_manager.py --mode publish --video-id video-d10bd04d
  python scripts/instagram_reel_manager.py --mode test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime, date, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Tracking file stored on Drive — records which videos have been posted to Instagram
_TRACKING_FILENAME = "instagram_posted.json"
_TRACKING_FOLDER = "ai-youtube-pipeline"


# ---------------------------------------------------------------------------
# Tracking: which videos have been posted to Instagram
# ---------------------------------------------------------------------------


async def _load_tracking(asset_store) -> dict:
    """Load the Instagram tracking file from Drive.

    Returns dict like:
    {
        "posted": [
            {"video_id": "video-abc123", "posted_at": "2026-08-01", "reel_id": "123", "permalink": "..."},
            ...
        ]
    }
    """
    try:
        data = await asset_store._drive.download_file(_TRACKING_FOLDER, _TRACKING_FILENAME)
        return json.loads(data.decode("utf-8"))
    except Exception:
        # File doesn't exist yet — start fresh
        return {"posted": []}


async def _save_tracking(asset_store, tracking: dict) -> None:
    """Save the Instagram tracking file to Drive."""
    data = json.dumps(tracking, indent=2).encode("utf-8")
    await asset_store._drive.upload_file(_TRACKING_FOLDER, _TRACKING_FILENAME, data)


def _is_already_posted(tracking: dict, video_id: str) -> bool:
    """Check if a video has already been posted to Instagram."""
    return any(entry["video_id"] == video_id for entry in tracking.get("posted", []))


def _posted_today(tracking: dict) -> bool:
    """Check if any Reel was already posted today."""
    today = date.today().isoformat()
    return any(entry.get("posted_at", "").startswith(today) for entry in tracking.get("posted", []))


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def daily_mode():
    """Post 1 Reel for the next unposted video. Skips if already posted today."""
    from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
    from pipeline.content_calendar import Content_Calendar, NotionMCPClient
    from pipeline.models import SubFolder, PipelineStatus

    print("=" * 60)
    print("  MODE: DAILY — Post 1 Reel (next unposted video)")
    print("=" * 60)

    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    # Load tracking
    tracking = await _load_tracking(asset_store)
    posted_count = len(tracking.get("posted", []))
    print(f"\n   Tracking: {posted_count} videos already posted to Instagram")

    # Check if already posted today
    if _posted_today(tracking):
        print("   Already posted 1 Reel today — skipping. Will post again tomorrow.")
        return True

    # Query Notion for scheduled/published videos
    print("\n1. Querying Notion for videos with Reels ready...")
    notion_token = os.environ.get("NOTION_AUTH_TOKEN", "")
    notion_db_id = os.environ.get("NOTION_DATABASE_ID", "")
    notion_client = NotionMCPClient(auth_token=notion_token, database_id=notion_db_id)
    content_calendar = Content_Calendar(notion_client=notion_client, database_id=notion_db_id)

    statuses_to_check = [PipelineStatus.SCHEDULED, PipelineStatus.PUBLISHED, PipelineStatus.UNLISTED]
    all_videos = []
    for status in statuses_to_check:
        try:
            videos = await content_calendar.list_videos_by_status(status)
            all_videos.extend(videos)
        except Exception:
            continue

    print(f"   Found {len(all_videos)} total videos")

    # Filter out already-posted videos
    candidates = [
        v for v in all_videos
        if not _is_already_posted(tracking, v["video_id"])
    ]
    print(f"   {len(candidates)} not yet posted to Instagram")

    if not candidates:
        print("\n   No unposted videos remaining — all caught up!")
        return True

    # Pick the first candidate (oldest unposted)
    video = candidates[0]
    vid_id = video["video_id"]
    print(f"\n2. Selected: {vid_id}")

    # Check if reel.mp4 exists on Drive (pre-encoded by pipeline)
    reel_bytes = None
    try:
        reel_bytes = await asset_store.read(
            video_id=vid_id,
            subfolder=SubFolder.VIDEOS,
            filename="reel.mp4",
        )
        if reel_bytes and len(reel_bytes) > 50_000:
            print(f"   Found pre-encoded reel.mp4 ({len(reel_bytes)/(1024*1024):.1f} MB)")
    except Exception:
        reel_bytes = None

    # If no pre-encoded reel, encode from source video
    if not reel_bytes:
        print(f"   No pre-encoded reel — encoding from source video...")
        try:
            source_bytes = await asset_store.read(
                video_id=vid_id,
                subfolder=SubFolder.VIDEOS,
                filename=f"{vid_id}_v1.mp4",
            )
            if not source_bytes or len(source_bytes) < 100_000:
                print(f"   ERROR: Source video too small or missing — skipping {vid_id}")
                return False
        except Exception as exc:
            print(f"   ERROR: Could not fetch source video — {exc}")
            return False

        reel_bytes = await _encode_reel(source_bytes)
        if not reel_bytes:
            return False

        # Save encoded reel to Drive for future use
        await asset_store.write(
            video_id=vid_id,
            subfolder=SubFolder.VIDEOS,
            filename="reel.mp4",
            content=reel_bytes,
        )

    # Post to Instagram
    result = await _post_reel(asset_store, reel_bytes, vid_id)

    if result:
        # Update tracking
        reel_id, permalink = result
        tracking["posted"].append({
            "video_id": vid_id,
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "reel_id": reel_id,
            "permalink": permalink,
        })
        await _save_tracking(asset_store, tracking)
        print(f"\n   Tracking updated ({len(tracking['posted'])} total posted)")
        return True

    return False


async def publish_mode(video_id: str):
    """Post a Reel for a specific video ID."""
    from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
    from pipeline.models import SubFolder

    print("=" * 60)
    print(f"  MODE: PUBLISH — Reel for {video_id}")
    print("=" * 60)

    if not video_id:
        print("   ERROR: --video-id is required")
        return False

    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    # Load tracking to check if already posted
    tracking = await _load_tracking(asset_store)
    if _is_already_posted(tracking, video_id):
        print(f"   WARNING: {video_id} was already posted to Instagram")
        print(f"   Posting anyway (manual override)...")

    # Check for pre-encoded reel
    reel_bytes = None
    try:
        reel_bytes = await asset_store.read(
            video_id=video_id,
            subfolder=SubFolder.VIDEOS,
            filename="reel.mp4",
        )
        if reel_bytes and len(reel_bytes) > 50_000:
            print(f"   Found pre-encoded reel.mp4 ({len(reel_bytes)/(1024*1024):.1f} MB)")
    except Exception:
        reel_bytes = None

    # If no pre-encoded reel, encode from source
    if not reel_bytes:
        print(f"   No pre-encoded reel — encoding from source...")
        try:
            source_bytes = await asset_store.read(
                video_id=video_id,
                subfolder=SubFolder.VIDEOS,
                filename=f"{video_id}_v1.mp4",
            )
            if not source_bytes or len(source_bytes) < 100_000:
                print(f"   ERROR: Source video not found or too small")
                return False
            print(f"   Source: {len(source_bytes)/(1024*1024):.1f} MB")
        except Exception as exc:
            print(f"   ERROR: {exc}")
            return False

        reel_bytes = await _encode_reel(source_bytes)
        if not reel_bytes:
            return False

        # Save for future
        await asset_store.write(
            video_id=video_id,
            subfolder=SubFolder.VIDEOS,
            filename="reel.mp4",
            content=reel_bytes,
        )

    # Post
    result = await _post_reel(asset_store, reel_bytes, video_id)

    if result:
        reel_id, permalink = result
        # Update tracking
        tracking["posted"].append({
            "video_id": video_id,
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "reel_id": reel_id,
            "permalink": permalink,
        })
        await _save_tracking(asset_store, tracking)
        return True

    return False


async def test_mode():
    """Post a test Reel from any available video."""
    from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
    from pipeline.models import SubFolder

    print("=" * 60)
    print("  MODE: TEST — Single test Reel")
    print("=" * 60)

    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    # Find any real video
    video_ids = [
        "video-d10bd04d", "video-96bdb181r", "video-94a43bder",
        "video-ae51253er", "video-0e600383", "video-8afb7a5d",
    ]

    for vid_id in video_ids:
        try:
            reel_bytes = await asset_store.read(
                video_id=vid_id,
                subfolder=SubFolder.VIDEOS,
                filename="reel.mp4",
            )
            if reel_bytes and len(reel_bytes) > 50_000:
                print(f"   Using pre-encoded reel from {vid_id}")
                result = await _post_reel(asset_store, reel_bytes, vid_id)
                return bool(result)
        except Exception:
            continue

    print("   No pre-encoded reels found — trying source videos...")
    for vid_id in video_ids:
        try:
            source = await asset_store.read(
                video_id=vid_id,
                subfolder=SubFolder.VIDEOS,
                filename=f"{vid_id}_v1.mp4",
            )
            if source and len(source) > 100_000:
                print(f"   Encoding from {vid_id}...")
                reel_bytes = await _encode_reel(source)
                if reel_bytes:
                    result = await _post_reel(asset_store, reel_bytes, vid_id)
                    return bool(result)
        except Exception:
            continue

    print("   ERROR: No videos found")
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _encode_reel(source_bytes: bytes) -> bytes | None:
    """Encode source video bytes into Instagram Reel format."""
    tmpdir = tempfile.mkdtemp()
    full_path = Path(tmpdir) / "full.mp4"
    reel_path = Path(tmpdir) / "reel.mp4"
    full_path.write_bytes(source_bytes)

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(full_path),
            "-t", "60",
            "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,fps=30",
            "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.0",
            "-b:v", "2M", "-maxrate", "2.5M", "-bufsize", "4M",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
            "-movflags", "+faststart",
            str(reel_path),
        ], capture_output=True, timeout=180, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"   FFmpeg failed: {exc.stderr.decode()[:200]}")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    reel_bytes = reel_path.read_bytes()
    print(f"   Encoded: {len(reel_bytes)/(1024*1024):.2f} MB")

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if len(reel_bytes) < 50_000:
        print(f"   ERROR: Encoded file too small")
        return None

    return reel_bytes


async def _post_reel(asset_store, reel_bytes: bytes, video_id: str) -> tuple[str, str] | None:
    """Upload reel to Drive (if not already there) and post to Instagram.

    Returns (reel_id, permalink) on success, None on failure.
    """
    from pipeline.models import SubFolder, MetadataPackage
    from pipeline.instagram_reels import InstagramReelsClient, build_reel_caption
    import httpx

    # Upload to Drive (overwrites if exists)
    print(f"\n3. Uploading to Drive...")
    drive_url = await asset_store.write(
        video_id=video_id,
        subfolder=SubFolder.VIDEOS,
        filename="reel.mp4",
        content=reel_bytes,
    )

    # Build API URL
    drive_id_match = re.search(r"/file/d/([^/]+)", drive_url)
    gcloud_key = os.environ.get("GOOGLE_CLOUD_API_KEY", "")
    if not drive_id_match or not gcloud_key:
        print(f"   ERROR: Cannot build URL (file_id={bool(drive_id_match)}, key={bool(gcloud_key)})")
        return None

    file_id = drive_id_match.group(1)
    api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={gcloud_key}"

    # Wait for CDN propagation (poll until accessible)
    print(f"\n4. Verifying URL accessibility...")
    url_ready = False
    for attempt in range(40):  # 10 min max
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(api_url)
                content_length = int(resp.headers.get("content-length", "0"))
                if resp.status_code == 200 and content_length > 100_000:
                    print(f"   Ready (HTTP 200, {content_length} bytes) after {(attempt+1)*15}s")
                    url_ready = True
                    break
        except Exception:
            pass
        await asyncio.sleep(15)

    if not url_ready:
        print(f"   WARNING: URL not confirmed after 10 min — trying anyway")

    # Load metadata for caption
    metadata = None
    youtube_url = ""
    try:
        meta_bytes = await asset_store.read(
            video_id=video_id,
            subfolder=SubFolder.METADATA,
            filename=f"{video_id}.json",
        )
        if meta_bytes:
            meta_dict = json.loads(meta_bytes.decode("utf-8"))
            metadata = MetadataPackage(**meta_dict)
            youtube_url = f"https://www.youtube.com/watch?v={metadata.youtube_video_id}" if metadata.youtube_video_id else ""
            print(f"   Caption from: \"{metadata.title}\"")
    except Exception:
        print(f"   No metadata found — using fallback caption")

    # Build caption
    if metadata:
        caption = build_reel_caption(metadata=metadata, youtube_url=youtube_url)
    else:
        caption = "Full breakdown on YouTube (link in bio)\n\n#superhero #reels #marvel #dc #explorepage #viral #fyp"

    # Fetch Shorts thumbnail for Reel cover image (if available on Drive)
    cover_url = None
    try:
        thumb_bytes = await asset_store.read(
            video_id=video_id,
            subfolder=SubFolder.THUMBNAILS,
            filename="thumbnail_shorts.jpg",
        )
        if thumb_bytes and len(thumb_bytes) > 1000:
            # Upload/ensure it's on Drive and get a public URL
            thumb_drive_url = await asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.THUMBNAILS,
                filename="thumbnail_shorts.jpg",
                content=thumb_bytes,
            )
            # Build public API URL for the cover image
            thumb_id_match = re.search(r"/file/d/([^/]+)", thumb_drive_url)
            if thumb_id_match and gcloud_key:
                cover_url = (
                    f"https://www.googleapis.com/drive/v3/files/"
                    f"{thumb_id_match.group(1)}?alt=media&key={gcloud_key}"
                )
                print(f"   Cover image: thumbnail_shorts.jpg found on Drive")
            else:
                print(f"   Cover image: could not build public URL")
        else:
            print(f"   Cover image: no thumbnail_shorts.jpg found — posting without cover")
    except Exception as cover_exc:
        print(f"   Cover image: not available ({cover_exc}) — posting without cover")

    # Post to Instagram
    print(f"\n5. Posting to Instagram...")
    ig_client = InstagramReelsClient(
        access_token=os.environ["INSTAGRAM_ACCESS_TOKEN"],
        instagram_account_id=os.environ["INSTAGRAM_ACCOUNT_ID"],
    )

    result = await ig_client.upload_reel(
        video_url=api_url,
        caption=caption,
        cover_url=cover_url,
        share_to_feed=True,
    )

    if result.success:
        print(f"\n   ✅ REEL POSTED! ID: {result.reel_id}")
        print(f"   Permalink: {result.permalink}")
        return (result.reel_id, result.permalink)
    else:
        print(f"\n   ❌ FAILED: {result.error}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Instagram Reel Manager")
    parser.add_argument("--mode", choices=["daily", "publish", "test"], default="daily")
    parser.add_argument("--video-id", default="", help="Video ID for publish mode")
    args = parser.parse_args()

    if args.mode == "daily":
        success = asyncio.run(daily_mode())
    elif args.mode == "publish":
        if not args.video_id:
            print("ERROR: --video-id required for publish mode")
            exit(1)
        success = asyncio.run(publish_mode(args.video_id))
    elif args.mode == "test":
        success = asyncio.run(test_mode())
    else:
        success = False

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
