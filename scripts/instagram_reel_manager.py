"""Instagram Reel Manager — test uploads and backfill Reels for scheduled videos.

Modes:
  test     — Upload a single test Reel from an existing pipeline video
  backfill — Find all scheduled/published videos without Reels, encode and post them

Usage:
  python scripts/instagram_reel_manager.py --mode test
  python scripts/instagram_reel_manager.py --mode backfill
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


async def test_mode():
    """Upload a single test Reel from an existing pipeline video."""
    from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
    from pipeline.models import SubFolder
    from pipeline.instagram_reels import InstagramReelsClient

    print("=" * 60)
    print("  MODE: TEST — Single Reel upload")
    print("=" * 60)

    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    # Find a real video from Drive
    print("\n1. Finding a real video on Drive...")
    video_ids = [
        "video-d10bd04d", "video-96bdb181r", "video-94a43bder",
        "video-ae51253er", "video-0e600383", "video-8afb7a5d",
    ]

    video_bytes = None
    used_id = None
    for vid_id in video_ids:
        try:
            video_bytes = await asset_store.read(
                video_id=vid_id,
                subfolder=SubFolder.VIDEOS,
                filename=f"{vid_id}_v1.mp4",
            )
            if video_bytes and len(video_bytes) > 100_000:
                used_id = vid_id
                print(f"   Found {vid_id}: {len(video_bytes)/(1024*1024):.1f} MB")
                break
        except Exception:
            continue

    if not video_bytes:
        print("   ERROR: No real video found on Drive")
        return False

    # Encode, upload, and post
    return await _encode_and_post_reel(asset_store, video_bytes, used_id, None)


async def backfill_mode():
    """Find all scheduled/published videos and post Reels for ones that don't have them."""
    from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
    from pipeline.content_calendar import Content_Calendar, NotionMCPClient
    from pipeline.models import SubFolder, PipelineStatus
    from pipeline.instagram_reels import InstagramReelsClient, build_reel_caption
    from pipeline.models import MetadataPackage

    print("=" * 60)
    print("  MODE: BACKFILL — Post Reels for all scheduled videos")
    print("=" * 60)

    # Set up clients
    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    notion_token = os.environ.get("NOTION_AUTH_TOKEN", "")
    notion_db_id = os.environ.get("NOTION_DATABASE_ID", "")
    notion_client = NotionMCPClient(auth_token=notion_token, database_id=notion_db_id)
    content_calendar = Content_Calendar(notion_client=notion_client, database_id=notion_db_id)

    # Query Notion for scheduled and published videos
    print("\n1. Querying Notion for scheduled/published videos...")
    statuses_to_check = [PipelineStatus.SCHEDULED, PipelineStatus.PUBLISHED, PipelineStatus.UNLISTED]

    all_videos = []
    for status in statuses_to_check:
        try:
            videos = await content_calendar.list_videos_by_status(status)
            for v in videos:
                v["status"] = status.value
            all_videos.extend(videos)
            print(f"   {status.value}: {len(videos)} video(s)")
        except Exception as exc:
            print(f"   {status.value}: query failed ({exc})")

    if not all_videos:
        print("\n   No scheduled/published videos found.")
        return True

    print(f"\n   Total candidates: {len(all_videos)}")

    # For each video, check if a reel.mp4 already exists (meaning Reel was already posted)
    print("\n2. Checking which videos already have Reels...")
    videos_needing_reels = []
    for video in all_videos:
        vid_id = video["video_id"]
        try:
            # If reel.mp4 exists in Drive, we already posted it
            await asset_store.read(
                video_id=vid_id,
                subfolder=SubFolder.VIDEOS,
                filename="reel.mp4",
            )
            print(f"   {vid_id}: ✓ already has Reel")
        except Exception:
            # No reel.mp4 → needs one
            videos_needing_reels.append(video)
            print(f"   {vid_id}: ✗ needs Reel")

    if not videos_needing_reels:
        print("\n   All videos already have Reels!")
        return True

    print(f"\n3. Creating Reels for {len(videos_needing_reels)} video(s)...")

    success_count = 0
    fail_count = 0

    for video in videos_needing_reels:
        vid_id = video["video_id"]
        print(f"\n   --- Processing {vid_id} ({video.get('status', '?')}) ---")

        # Fetch the full video from Drive
        try:
            video_bytes = await asset_store.read(
                video_id=vid_id,
                subfolder=SubFolder.VIDEOS,
                filename=f"{vid_id}_v1.mp4",
            )
            if not video_bytes or len(video_bytes) < 100_000:
                print(f"   Skipping {vid_id}: video file too small or missing")
                fail_count += 1
                continue
            print(f"   Downloaded: {len(video_bytes)/(1024*1024):.1f} MB")
        except Exception as exc:
            print(f"   Skipping {vid_id}: could not fetch video ({exc})")
            fail_count += 1
            continue

        # Try to get the scheduled publish time from Notion
        scheduled_time = None
        try:
            # Query the specific video record for its scheduled datetime
            notion_filter = {"property": "video_id", "rich_text": {"equals": vid_id}}
            pages = await notion_client.query_database(notion_db_id, filter=notion_filter)
            if pages:
                props = pages[0].get("properties", {})
                sched_prop = props.get("scheduled_publish_datetime", {})
                date_val = sched_prop.get("date", {})
                if date_val and date_val.get("start"):
                    dt = datetime.fromisoformat(
                        date_val["start"].replace("Z", "+00:00")
                    )
                    # Only use scheduled time if it's in the future (>10 min from now)
                    # Instagram requires 10 min to 75 days in the future
                    from datetime import timedelta
                    now = datetime.now(timezone.utc)
                    if dt > now + timedelta(minutes=10):
                        scheduled_time = dt
                        print(f"   Scheduled for: {scheduled_time.strftime('%Y-%m-%d %H:%M UTC')}")
                    else:
                        print(f"   Schedule date is in the past ({dt.strftime('%Y-%m-%d')}) — posting immediately")
        except Exception as exc:
            logger.debug("Could not get schedule for %s: %s", vid_id, exc)

        # Encode and post
        result = await _encode_and_post_reel(
            asset_store, video_bytes, vid_id, scheduled_time
        )
        if result:
            success_count += 1
        else:
            fail_count += 1

        # Small delay between posts to avoid rate limiting
        await asyncio.sleep(5)

    print(f"\n{'=' * 60}")
    print(f"  BACKFILL COMPLETE: {success_count} posted, {fail_count} failed")
    print(f"{'=' * 60}")
    return fail_count == 0


async def _encode_and_post_reel(
    asset_store,
    video_bytes: bytes,
    video_id: str,
    scheduled_time: datetime | None,
) -> bool:
    """Encode a video as Reel, upload to Drive, and post to Instagram."""
    from pipeline.models import SubFolder, MetadataPackage
    from pipeline.instagram_reels import InstagramReelsClient, build_reel_caption
    import json as _json

    # Try to load the video's metadata for proper caption
    metadata = None
    youtube_url = ""
    try:
        meta_bytes = await asset_store.read(
            video_id=video_id,
            subfolder=SubFolder.METADATA,
            filename=f"{video_id}.json",
        )
        if meta_bytes:
            meta_dict = _json.loads(meta_bytes.decode("utf-8"))
            metadata = MetadataPackage(**meta_dict)
            youtube_url = f"https://www.youtube.com/watch?v={metadata.youtube_video_id}" if metadata.youtube_video_id else ""
            print(f"   Metadata loaded: \"{metadata.title}\"")
    except Exception as exc:
        logger.debug("Could not load metadata for %s: %s", video_id, exc)
        print(f"   Metadata not found — will use fallback caption")

    # Encode for Instagram Reels
    print(f"   Encoding for Instagram Reels...")
    tmpdir = tempfile.mkdtemp()
    full_path = Path(tmpdir) / "full.mp4"
    reel_path = Path(tmpdir) / "reel.mp4"
    full_path.write_bytes(video_bytes)

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
        print(f"   FFmpeg encoding failed: {exc.stderr.decode()[:200]}")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False

    reel_size = reel_path.stat().st_size
    print(f"   Encoded: {reel_size/(1024*1024):.2f} MB")

    if reel_size < 50_000:
        print(f"   ERROR: Encoded file too small ({reel_size} bytes) — likely bad source")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False

    # Upload to Drive
    print(f"   Uploading to Drive...")
    reel_bytes = reel_path.read_bytes()
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
        print(f"   ERROR: Cannot build API URL (file_id={bool(drive_id_match)}, key={bool(gcloud_key)})")
        return False

    file_id = drive_id_match.group(1)
    api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={gcloud_key}"

    # Wait for Drive CDN propagation
    print(f"   Waiting 60s for Drive CDN propagation...")
    await asyncio.sleep(60)

    # Verify accessibility
    import httpx
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.head(api_url)
        if resp.status_code != 200:
            print(f"   ERROR: URL not accessible (HTTP {resp.status_code})")
            return False

    # Post to Instagram
    print(f"   Posting to Instagram...")
    ig_client = InstagramReelsClient(
        access_token=os.environ["INSTAGRAM_ACCESS_TOKEN"],
        instagram_account_id=os.environ["INSTAGRAM_ACCOUNT_ID"],
    )

    # Build proper SEO caption from metadata (or fallback to generic)
    if metadata:
        caption = build_reel_caption(
            metadata=metadata,
            youtube_url=youtube_url,
        )
        print(f"   Caption: \"{metadata.title}\" ({len(caption)} chars, {caption.count('#')} hashtags)")
    else:
        caption = (
            f"Full breakdown on YouTube (link in bio)\n\n"
            f"#superhero #reels #marvel #dc #explorepage #viral #fyp"
        )
        print(f"   Using fallback caption (no metadata found)")

    result = await ig_client.upload_reel(
        video_url=api_url,
        caption=caption,
        share_to_feed=True,
        scheduled_publish_time=scheduled_time,
    )

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if result.success:
        schedule_note = ""
        if scheduled_time:
            schedule_note = f" (scheduled: {scheduled_time.strftime('%Y-%m-%d %H:%M')})"
        print(f"   ✅ Reel posted: {result.permalink}{schedule_note}")
        return True
    else:
        print(f"   ❌ Failed: {result.error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Instagram Reel Manager")
    parser.add_argument("--mode", choices=["test", "backfill"], default="test")
    args = parser.parse_args()

    if args.mode == "test":
        success = asyncio.run(test_mode())
    else:
        success = asyncio.run(backfill_mode())

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
