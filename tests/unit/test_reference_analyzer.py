"""Unit tests for the Reference_Analyzer subsystem.

Tests cover:
- Transcript metric extraction helpers (_compute_sentiment_polarity, _compute_wpm, etc.)
- Segment annotation detection
- Dominant-color extraction
- StyleProfile aggregation
- Reference_Analyzer.analyze() — happy path and error scenarios (channel inaccessible,
  write failure, fewer than 10 videos warning)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.asset_store import Asset_Store
from pipeline.content_calendar import Content_Calendar
from pipeline.models import StyleProfile
from pipeline.notifier import Notifier
from pipeline.reference_analyzer import (
    BrowserClient,
    Reference_Analyzer,
    ReferenceAnalyzerError,
    VideoMetadata,
    _compute_avg_sentence_length,
    _compute_sentiment_polarity,
    _compute_wpm,
    _count_body_segments,
    _extract_dominant_colors,
    _extract_segment_annotations,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_video(
    url: str = "https://youtube.com/watch?v=test",
    title: str = "Test Video",
    duration_seconds: Optional[int] = 300,
) -> VideoMetadata:
    return VideoMetadata(
        url=url,
        title=title,
        published_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        duration_seconds=duration_seconds,
    )


def _make_analyzer(
    browser: BrowserClient,
    asset_store: Optional[Asset_Store] = None,
    content_calendar: Optional[Content_Calendar] = None,
    notifier: Optional[Notifier] = None,
) -> Reference_Analyzer:
    if asset_store is None:
        asset_store = MagicMock(spec=Asset_Store)
        asset_store.write = AsyncMock(return_value="https://drive.google.com/fake")
    if content_calendar is None:
        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = MagicMock()
        content_calendar._client.query_database = AsyncMock(return_value=[])
    if notifier is None:
        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()
    return Reference_Analyzer(
        browser_client=browser,
        asset_store=asset_store,
        content_calendar=content_calendar,
        notifier=notifier,
    )


_SENTINEL: list[VideoMetadata] = []  # sentinel to distinguish None from empty list


class _FakeBrowser:
    """Controllable BrowserClient test double."""

    def __init__(
        self,
        videos: list[VideoMetadata] | None = None,
        transcript: str = "Welcome to this video. Today we explore AI.",
        thumbnail: bytes = b"",
        fail_navigate: bool = False,
        fail_video_list: bool = False,
    ) -> None:
        # Allow callers to pass an explicit empty list; only fall back to a
        # default single video when videos is None.
        self._videos = [_make_video()] if videos is None else videos
        self._transcript = transcript
        self._thumbnail = thumbnail
        self._fail_navigate = fail_navigate
        self._fail_video_list = fail_video_list

    async def navigate(self, url: str) -> str:
        if self._fail_navigate:
            raise ConnectionError("Cannot reach channel")
        return "<html>channel page</html>"

    async def get_video_list(self, channel_url: str, days_back: int) -> list[VideoMetadata]:
        if self._fail_video_list:
            raise ConnectionError("Cannot list videos")
        return self._videos

    async def get_video_transcript(self, video_url: str) -> str:
        return self._transcript

    async def get_thumbnail_image(self, video_url: str) -> bytes:
        return self._thumbnail


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestComputeSentimentPolarity:
    def test_positive_text_returns_positive(self) -> None:
        score = _compute_sentiment_polarity("great amazing wonderful awesome love happy")
        assert score > 0.0

    def test_negative_text_returns_negative(self) -> None:
        score = _compute_sentiment_polarity("terrible awful horrible bad hate fail")
        assert score < 0.0

    def test_neutral_text_returns_near_zero(self) -> None:
        score = _compute_sentiment_polarity("the cat sat on the mat")
        assert -0.2 <= score <= 0.2

    def test_result_within_bounds(self) -> None:
        for text in ["", "good " * 100, "bad " * 100]:
            score = _compute_sentiment_polarity(text)
            assert -1.0 <= score <= 1.0


class TestComputeWpm:
    def test_with_known_duration(self) -> None:
        # 150 words in 60 seconds = 150 wpm
        text = " ".join(["word"] * 150)
        assert _compute_wpm(text, 60) == 150

    def test_no_duration_returns_default(self) -> None:
        assert _compute_wpm("some text", None) == 150

    def test_zero_duration_returns_default(self) -> None:
        assert _compute_wpm("some text", 0) == 150

    def test_result_at_least_one(self) -> None:
        # Very few words over long duration should still be >= 1
        assert _compute_wpm("hi", 36000) >= 1


class TestComputeAvgSentenceLength:
    def test_simple_two_sentences(self) -> None:
        # "Hello world" (2 words) + "How are you" (3 words) = avg 2.5
        result = _compute_avg_sentence_length("Hello world. How are you.")
        assert result == pytest.approx(2.5, abs=0.1)

    def test_empty_text_returns_one(self) -> None:
        assert _compute_avg_sentence_length("") == 1.0

    def test_no_sentence_boundary_counts_as_one_sentence(self) -> None:
        result = _compute_avg_sentence_length("no boundaries here at all")
        assert result == pytest.approx(5.0, abs=0.1)


class TestCountBodySegments:
    def test_h2_headings_counted(self) -> None:
        text = "## Introduction\n## Main Body\n## Conclusion"
        assert _count_body_segments(text) == 3.0

    def test_numbered_sections_counted(self) -> None:
        text = "1. First point\n2. Second point\n3. Third point"
        assert _count_body_segments(text) == 3.0

    def test_mixed_headings(self) -> None:
        text = "## Heading\n### Sub-heading\n2. Item"
        assert _count_body_segments(text) == 3.0

    def test_no_headings_returns_zero(self) -> None:
        assert _count_body_segments("Just some plain text here.") == 0.0


class TestExtractSegmentAnnotations:
    def test_intro_detected(self) -> None:
        result = _extract_segment_annotations("Welcome to this video. Today we explore Python.")
        assert result["intro_present"] is True

    def test_hook_detected(self) -> None:
        result = _extract_segment_annotations("Did you know that AI can write code?")
        assert result["hook_present"] is True

    def test_cta_detected(self) -> None:
        result = _extract_segment_annotations("Please subscribe and like this video!")
        assert result["cta_present"] is True

    def test_no_annotations_in_plain_text(self) -> None:
        result = _extract_segment_annotations("The algorithm processes data efficiently.")
        assert result["intro_present"] is False
        assert result["hook_present"] is False
        assert result["cta_present"] is False


class TestExtractDominantColors:
    def test_empty_bytes_returns_default(self) -> None:
        colors = _extract_dominant_colors(b"")
        assert colors == ["#000000"]

    def test_invalid_bytes_returns_default(self) -> None:
        colors = _extract_dominant_colors(b"not an image")
        assert colors == ["#000000"]

    def test_returns_at_most_five_colors(self) -> None:
        # Build a tiny valid JPEG (1x1 red pixel) using Pillow
        try:
            import io as _io
            from PIL import Image

            img = Image.new("RGB", (10, 10), color=(255, 0, 0))
            buf = _io.BytesIO()
            img.save(buf, format="JPEG")
            colors = _extract_dominant_colors(buf.getvalue())
            assert len(colors) <= 5
            assert all(c.startswith("#") for c in colors)
        except ImportError:
            pytest.skip("Pillow not available")


# ---------------------------------------------------------------------------
# Reference_Analyzer.analyze() tests
# ---------------------------------------------------------------------------


class TestReferenceAnalyzerAnalyze:
    """End-to-end tests for analyze() with injected browser doubles."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_style_profile(self) -> None:
        """analyze() returns a valid StyleProfile on a clean run."""
        videos = [_make_video(url=f"https://youtube.com/watch?v={i}") for i in range(12)]
        browser = _FakeBrowser(videos=videos, transcript="Welcome. Imagine this. Subscribe now.")
        analyzer = _make_analyzer(browser)

        profile = await analyzer.analyze("https://youtube.com/@testchannel")

        assert isinstance(profile, StyleProfile)
        assert profile.channel_url == "https://youtube.com/@testchannel"
        assert profile.version == 1
        assert len(profile.doc_id) == 36  # UUID4 format
        assert profile.thumbnail_composition.sample_count == 12
        assert profile.thumbnail_composition.lookback_days == 90

    @pytest.mark.asyncio
    async def test_fewer_than_10_videos_logs_warning(self) -> None:
        """analyze() proceeds and warns when fewer than 10 videos are found."""
        videos = [_make_video(url=f"https://youtube.com/watch?v={i}") for i in range(5)]
        browser = _FakeBrowser(videos=videos)
        analyzer = _make_analyzer(browser)

        with patch("pipeline.reference_analyzer.logger") as mock_logger:
            profile = await analyzer.analyze("https://youtube.com/@testchannel")
            # Warning should have been logged
            warning_calls = [
                call for call in mock_logger.warning.call_args_list
                if "only" in str(call).lower() or "5" in str(call)
            ]
            assert len(warning_calls) >= 1

        assert isinstance(profile, StyleProfile)
        assert profile.thumbnail_composition.sample_count == 5

    @pytest.mark.asyncio
    async def test_channel_inaccessible_raises_error(self) -> None:
        """analyze() raises ReferenceAnalyzerError when navigate fails all retries."""
        browser = _FakeBrowser(fail_navigate=True)
        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()

        asset_store = MagicMock(spec=Asset_Store)
        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = MagicMock()
        content_calendar._client.query_database = AsyncMock(return_value=[])

        analyzer = Reference_Analyzer(
            browser_client=browser,
            asset_store=asset_store,
            content_calendar=content_calendar,
            notifier=notifier,
        )

        # Patch sleep to avoid real delays in tests
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ReferenceAnalyzerError, match="inaccessible"):
                await analyzer.analyze("https://youtube.com/@badchannel")

        notifier.send_failure_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_video_list_failure_raises_error(self) -> None:
        """analyze() raises ReferenceAnalyzerError when video list fetch fails."""
        browser = _FakeBrowser(fail_video_list=True)
        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()

        asset_store = MagicMock(spec=Asset_Store)
        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = MagicMock()
        content_calendar._client.query_database = AsyncMock(return_value=[])

        analyzer = Reference_Analyzer(
            browser_client=browser,
            asset_store=asset_store,
            content_calendar=content_calendar,
            notifier=notifier,
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ReferenceAnalyzerError):
                await analyzer.analyze("https://youtube.com/@badchannel")

        notifier.send_failure_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_videos_raises_error(self) -> None:
        """analyze() raises ReferenceAnalyzerError when channel has zero uploads."""
        browser = _FakeBrowser(videos=[])
        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()
        asset_store = MagicMock(spec=Asset_Store)
        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = MagicMock()
        content_calendar._client.query_database = AsyncMock(return_value=[])

        analyzer = Reference_Analyzer(
            browser_client=browser,
            asset_store=asset_store,
            content_calendar=content_calendar,
            notifier=notifier,
        )

        with pytest.raises(ReferenceAnalyzerError, match="No uploads"):
            await analyzer.analyze("https://youtube.com/@emptychannel")

        notifier.send_failure_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_failure_retries_and_raises(self) -> None:
        """analyze() retries write 3× and raises ReferenceAnalyzerError on all-fail."""
        browser = _FakeBrowser(
            videos=[_make_video()],
            transcript="Welcome. Subscribe now.",
        )
        asset_store = MagicMock(spec=Asset_Store)
        asset_store.write = AsyncMock(side_effect=IOError("Drive unavailable"))

        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()

        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = MagicMock()
        content_calendar._client.query_database = AsyncMock(return_value=[])

        analyzer = Reference_Analyzer(
            browser_client=browser,
            asset_store=asset_store,
            content_calendar=content_calendar,
            notifier=notifier,
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ReferenceAnalyzerError, match="Failed to write"):
                await analyzer.analyze("https://youtube.com/@testchannel")

        # 3 write attempts expected
        assert asset_store.write.call_count == 3
        notifier.send_failure_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_id_updates_calendar(self) -> None:
        """analyze() calls Content_Calendar query when batch_id is provided."""
        videos = [_make_video()]
        browser = _FakeBrowser(videos=videos)

        asset_store = MagicMock(spec=Asset_Store)
        asset_store.write = AsyncMock(return_value="https://drive.google.com/fake")

        notion_client = MagicMock()
        notion_client.query_database = AsyncMock(return_value=[])

        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = notion_client

        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()

        analyzer = Reference_Analyzer(
            browser_client=browser,
            asset_store=asset_store,
            content_calendar=content_calendar,
            notifier=notifier,
        )

        profile = await analyzer.analyze("https://youtube.com/@ch", batch_id="batch-001")
        assert isinstance(profile, StyleProfile)
        # The calendar query should have been called
        notion_client.query_database.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_batch_id_skips_calendar(self) -> None:
        """analyze() skips calendar update when batch_id is None."""
        videos = [_make_video()]
        browser = _FakeBrowser(videos=videos)

        asset_store = MagicMock(spec=Asset_Store)
        asset_store.write = AsyncMock(return_value="https://drive.google.com/fake")

        notion_client = MagicMock()
        notion_client.query_database = AsyncMock(return_value=[])

        content_calendar = MagicMock(spec=Content_Calendar)
        content_calendar._db_id = "fake-db"
        content_calendar._client = notion_client

        notifier = MagicMock(spec=Notifier)
        notifier.send_failure_alert = MagicMock()

        analyzer = Reference_Analyzer(
            browser_client=browser,
            asset_store=asset_store,
            content_calendar=content_calendar,
            notifier=notifier,
        )

        await analyzer.analyze("https://youtube.com/@ch", batch_id=None)
        notion_client.query_database.assert_not_called()
