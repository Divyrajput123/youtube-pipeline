"""Instagram Reel Manager — test or publish a Reel for a specific video.

Modes:
  test    — Upload a single test Reel from any existing pipeline video
  publish — Create and publish a Reel for a specific video ID

Usage:
  python scripts/instagram_reel_manager.py --mode test
  python scripts/instagram_reel_manager.py --mode publish --video-id video-d10bd04d
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import subprocess
import tempfile
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

    return await _encode_and_post_reel(asset_store, video_bytes, used_id)


async def publish_mode(video_id: str):
    """Create and publish a Reel for a specific video ID."""
    from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
    from pipeline.models import SubFolder

    print("=" * 60)
    print(f"  MODE: PUBLISH — Reel for {video_id}")
    print("=" * 60)

    if not video_id:
        print("   ERROR: --video-id is required for publish mode")
        return False

    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    # Download the video
    print(f"\n1. Downloading {video_id} from Drive...")
    try:
        video_bytes = await asset_store.read(
            video_id=video_id,
            subfolder=SubFolder.VIDEOS,
            filename=f"{video_id}_v1.mp4",
        )
        if not video_bytes or len(video_bytes) < 100_000:
            print(f"   ERROR: Video file too small or missing ({len(video_bytes) if video_bytes else 0} bytes)")
            return False
        print(f"   Downloaded: {len(video_bytes)/(1024*1024):.1f} MB")
    except Exception as exc:
        print(f"   ERROR: Could not fetch video — {exc}")
        return False

    return await _encode_and_post_reel(asset_store, video_bytes, video_id)


async def _encode_and_post_reel(
    asset_store,
    video_bytes: bytes,
    video_id: str,
) -> bool:
    """Encode a video as Reel, upload to Drive, and publish to Instagram immediately."""
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
    print(f"\n2. Encoding for Instagram Reels...")
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
        print(f"   ERROR: Encoded file too small ({reel_size} bytes)")
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False

    # Upload to Drive
    print(f"\n3. Uploading Reel to Drive...")
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
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False

    file_id = drive_id_match.group(1)
    api_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={gcloud_key}"

    # Wait for Drive CDN propagation
    print(f"\n4. Waiting for Drive CDN propagation...")
    import httpx
    url_ready = False
    for attempt in range(40):  # 40 * 15s = 10 min max
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.head(api_url)
                content_length = int(resp.headers.get("content-length", "0"))
                if resp.status_code == 200 and content_length > 100_000:
                    print(f"   URL ready (HTTP {resp.status_code}, {content_length} bytes) after {(attempt+1)*15}s")
                    url_ready = True
                    break
                else:
                    print(f"   Not ready yet (HTTP {resp.status_code}, {content_length} bytes) — waiting...")
        except Exception as exc:
            print(f"   Check failed ({exc}) — waiting...")
        await asyncio.sleep(15)

    if not url_ready:
        print(f"   WARNING: URL not confirmed accessible after 10 min — trying anyway")

    # Post to Instagram
    print(f"\n5. Posting to Instagram...")
    ig_client = InstagramReelsClient(
        access_token=os.environ["INSTAGRAM_ACCESS_TOKEN"],
        instagram_account_id=os.environ["INSTAGRAM_ACCOUNT_ID"],
    )

    # Build caption
    if metadata:
        caption = build_reel_caption(metadata=metadata, youtube_url=youtube_url)
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
    )

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    if result.success:
        print(f"\n   ✅ REEL POSTED! ID: {result.reel_id}")
        print(f"   Permalink: {result.permalink}")
        return True
    else:
        print(f"\n   ❌ FAILED: {result.error}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Instagram Reel Manager")
    parser.add_argument("--mode", choices=["test", "publish"], default="test")
    parser.add_argument("--video-id", default="", help="Video ID for publish mode")
    args = parser.parse_args()

    if args.mode == "test":
        success = asyncio.run(test_mode())
    elif args.mode == "publish":
        if not args.video_id:
            print("ERROR: --video-id is required for publish mode")
            exit(1)
        success = asyncio.run(publish_mode(args.video_id))
    else:
        success = False

    if not success:
        exit(1)


if __name__ == "__main__":
    main()
