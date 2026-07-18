#!/usr/bin/env python3
"""Quick smoke test for the AI YouTube Content Pipeline with placeholder fallbacks.

This script tests that the pipeline can run end-to-end locally even when
external API services (ElevenLabs, Viewmax, YouTube) are unavailable.

The placeholder modes will:
- Generate silent MP3 files for narration
- Generate colored placeholder JPEGs for video clips
- Return fake YouTube video IDs and URLs for uploads

Run with:
    python test_pipeline_local.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)


async def test_narration_fallback():
    """Test that Narration_Generator works with placeholder MP3 files."""
    logger.info("=" * 70)
    logger.info("TEST 1: Narration_Generator fallback mode")
    logger.info("=" * 70)
    
    from pipeline.narration_generator import ElevenLabsMCPClient
    
    client = ElevenLabsMCPClient()
    
    # Try to synthesize a test phrase
    try:
        mp3_bytes = await client.synthesize(
            text="This is a test narration for the AI YouTube pipeline.",
            voice_id="test_voice_id",
            sample_rate=44_100,
            bitrate_kbps=128,
        )
        
        logger.info(f"✓ Generated MP3: {len(mp3_bytes)} bytes")
        
        # Verify it's not empty
        assert len(mp3_bytes) > 0, "MP3 bytes should not be empty"
        
        logger.info("✓ Narration_Generator fallback test PASSED")
        return True
        
    except Exception as exc:
        logger.error(f"✗ Narration_Generator fallback test FAILED: {exc}")
        return False


async def test_visual_fallback():
    """Test that Visual_Generator works with placeholder clip JPEGs."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 2: Visual_Generator fallback mode")
    logger.info("=" * 70)
    
    from pipeline.visual_generator import ViewmaxMCPClient
    
    client = ViewmaxMCPClient()
    
    # Try to generate a test clip
    try:
        clip_bytes = await client.generate_clip(
            prompt="A beautiful sunset over mountains with calm music playing",
            duration_seconds=5,
        )
        
        logger.info(f"✓ Generated clip: {len(clip_bytes)} bytes")
        
        # Verify it's not empty and looks like JPEG data
        assert len(clip_bytes) > 0, "Clip bytes should not be empty"
        assert clip_bytes[:2] == b'\xff\xd8', "Should start with JPEG magic bytes"
        
        logger.info("✓ Visual_Generator fallback test PASSED")
        return True
        
    except Exception as exc:
        logger.error(f"✗ Visual_Generator fallback test FAILED: {exc}")
        return False


async def test_publisher_fallback():
    """Test that Publisher works with placeholder YouTube IDs."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 3: Publisher fallback mode")
    logger.info("=" * 70)
    
    from pipeline.publisher import YouTubeDataAPIClient
    
    client = YouTubeDataAPIClient()
    
    # Try to upload a fake video
    try:
        result = await client.upload_video(
            mp4_path="/fake/path/test.mp4",
            title="Test Video Title",
            description="This is a test video description.",
            tags=["test", "ai", "youtube"],
            privacy="unlisted",
        )
        
        logger.info(f"✓ Upload result: {result}")
        
        # Verify we got a video ID and URL
        assert "id" in result, "Result should contain 'id'"
        assert "url" in result, "Result should contain 'url'"
        assert len(result["id"]) > 0, "Video ID should not be empty"
        
        logger.info("✓ Publisher fallback test PASSED")
        return True
        
    except Exception as exc:
        logger.error(f"✗ Publisher fallback test FAILED: {exc}")
        return False


async def test_metadata_generator():
    """Test that Metadata_Generator works with the Claude API."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST 4: Metadata_Generator with Claude API")
    logger.info("=" * 70)
    
    try:
        import os
        from pipeline.metadata_generator import Metadata_Generator
        from pipeline.models import Script, TopicEntry
        from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
        from pipeline.notifier import Notifier, NotifierConfig
        from pipeline.script_writer import AnthropicClaudeClient
        
        # Check if Claude API key is available
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or api_key.startswith("sk-ant-REPLACE"):
            logger.warning("Claude API key not configured — skipping metadata test")
            logger.info("✓ Metadata_Generator test SKIPPED (no API key)")
            return True
        
        # Create mock dependencies
        claude_client = AnthropicClaudeClient()
        
        # Use a simple mock for asset store and notifier
        class MockAssetStore:
            async def write(self, *args, **kwargs):
                return "https://drive.google.com/fake/metadata.json"
        
        class MockNotifier:
            def send_failure_alert(self, *args, **kwargs):
                pass
        
        asset_store = MockAssetStore()
        notifier = MockNotifier()
        
        generator = Metadata_Generator(
            claude_client=claude_client,
            asset_store=asset_store,  # type: ignore
            notifier=notifier,  # type: ignore
        )
        
        # Create a test script
        from datetime import datetime, timezone
        
        script = Script(
            video_id="test-video-123",
            content="""# Introduction to AI
            
Artificial Intelligence is transforming how we create content. In this video, 
we'll explore the latest developments in AI and how they impact content creators.

## Main Content

AI tools can now generate scripts, narration, and even video content automatically.
This opens up new possibilities for creators who want to scale their output.

## Conclusion

The future of content creation is AI-assisted but human-guided. Let's embrace
these new tools while maintaining our creative vision.
""",
            version=1,
            style_profile_doc_id="test-style-profile",
            word_count=100,  # Add required field
            created_at=datetime.now(timezone.utc),  # Add required field
        )
        
        # Create test topics
        from datetime import datetime, timezone
        
        topics = [
            TopicEntry(
                title="AI Content Creation",
                composite_score=0.95,
                recency_hours=24.0,
                source_query_timestamp=datetime.now(timezone.utc),
                search_volume_signal=1000.0,
                relevance_tags_matched=["AI", "content", "automation"],
            ),
            TopicEntry(
                title="YouTube Automation Tools",
                composite_score=0.85,
                recency_hours=48.0,
                source_query_timestamp=datetime.now(timezone.utc),
                search_volume_signal=800.0,
                relevance_tags_matched=["YouTube", "automation", "tools"],
            ),
        ]
        
        # Generate metadata
        logger.info("Generating metadata with Claude API (this may take 10-15 seconds)...")
        metadata = await generator.generate(
            script=script,
            topics=topics,
            video_id="test-video-123",
        )
        
        logger.info(f"✓ Generated metadata:")
        logger.info(f"  Title: {metadata.title}")
        logger.info(f"  Tags: {len(metadata.tags)} tags")
        logger.info(f"  Hashtags: {metadata.hashtags}")
        logger.info(f"  Chapters: {len(metadata.chapters)} chapters")
        logger.info(f"  Description length: {len(metadata.description)} chars")
        
        # Basic validations
        assert len(metadata.title) <= 60, "Title should be ≤ 60 characters"
        assert 10 <= len(metadata.tags) <= 15, "Should have 10-15 tags"
        assert 3 <= len(metadata.hashtags) <= 5, "Should have 3-5 hashtags"
        assert len(metadata.chapters) > 0, "Should have at least one chapter"
        
        logger.info("✓ Metadata_Generator test PASSED")
        return True
        
    except Exception as exc:
        logger.error(f"✗ Metadata_Generator test FAILED: {exc}", exc_info=True)
        return False


async def main():
    """Run all fallback mode tests."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("AI YouTube Content Pipeline - Local Fallback Mode Tests")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Testing that the pipeline works locally with placeholder data")
    logger.info("when external API services are unavailable...")
    logger.info("")
    
    results = []
    
    # Run all tests
    results.append(await test_narration_fallback())
    results.append(await test_visual_fallback())
    results.append(await test_publisher_fallback())
    results.append(await test_metadata_generator())
    
    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("✓ All tests PASSED - pipeline can run locally with fallbacks!")
        return 0
    else:
        logger.error(f"✗ {total - passed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
