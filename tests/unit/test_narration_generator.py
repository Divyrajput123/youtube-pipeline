"""Unit tests for pipeline.narration_generator.

Covers:
- Pre-flight guard (empty/missing voice_id)
- Text segmentation helper (_split_into_segments)
- Version probing (_probe_next_version)
- Successful end-to-end generate() flow
- ElevenLabs API retry exhaustion → NarrationGeneratorError + notifier alert
- Asset_Store write failure → NarrationGeneratorError + notifier alert
- Content_Calendar update failure → NarrationGeneratorError + notifier alert
- Multi-segment concatenation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from pipeline.asset_store import Asset_Store, AssetStoreError
from pipeline.content_calendar import Content_Calendar
from pipeline.models import NarrationAsset, PipelineStatus, Script, SubFolder
from pipeline.narration_generator import (
    MAX_SEGMENT_CHARS,
    Narration_Generator,
    NarrationGeneratorError,
    _probe_next_version,
    _retry_delay_elevenlabs,
    _split_into_segments,
)
from pipeline.notifier import Notifier


# ---------------------------------------------------------------------------
# Factories / helpers
# ---------------------------------------------------------------------------


def _make_script(content: str = "Hello world. This is a test script.") -> Script:
    return Script(
        video_id="vid-001",
        version=1,
        content=content,
        word_count=len(content.split()),
        style_profile_doc_id="sp-doc-1",
        created_at=datetime.now(tz=timezone.utc),
    )


def _make_tts_client(return_bytes: bytes = b"FAKEMP3") -> AsyncMock:
    """Return an AsyncMock that satisfies the ElevenLabsClient protocol."""
    client = AsyncMock()
    client.synthesize = AsyncMock(return_value=return_bytes)
    return client


def _make_asset_store(drive_url: str = "https://drive.google.com/file/x") -> MagicMock:
    store = MagicMock(spec=Asset_Store)
    store.write = AsyncMock(return_value=drive_url)
    # By default, reading any file raises an exception (file doesn't exist yet).
    store.read = AsyncMock(side_effect=Exception("not found"))
    return store


def _make_content_calendar() -> MagicMock:
    cal = MagicMock(spec=Content_Calendar)
    cal.update_status = AsyncMock(return_value=None)
    return cal


def _make_notifier() -> MagicMock:
    notifier = MagicMock(spec=Notifier)
    notifier.send_failure_alert = MagicMock(return_value=None)
    return notifier


def _make_generator(
    tts=None,
    store=None,
    calendar=None,
    notifier=None,
) -> Narration_Generator:
    return Narration_Generator(
        elevenlabs_client=tts or _make_tts_client(),
        asset_store=store or _make_asset_store(),
        content_calendar=calendar or _make_content_calendar(),
        notifier=notifier or _make_notifier(),
    )


# ---------------------------------------------------------------------------
# _split_into_segments
# ---------------------------------------------------------------------------


class TestSplitIntoSegments:
    def test_empty_text_returns_empty_list(self):
        assert _split_into_segments("") == []

    def test_short_text_returned_as_single_segment(self):
        text = "Short script."
        result = _split_into_segments(text, max_chars=5_000)
        assert result == [text]

    def test_text_exactly_at_limit_is_single_segment(self):
        text = "a" * MAX_SEGMENT_CHARS
        result = _split_into_segments(text, max_chars=MAX_SEGMENT_CHARS)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_split_on_sentence_boundaries(self):
        # Build two clearly distinct sentences totalling > 20 chars
        sentence_a = "First sentence ends here. "
        sentence_b = "Second sentence starts here."
        text = sentence_a + sentence_b
        result = _split_into_segments(text, max_chars=len(sentence_a) - 1)
        assert len(result) >= 2
        # Reassembled text (normalized) should contain all the words
        rejoined = " ".join(result)
        for word in ["First", "sentence", "Second"]:
            assert word in rejoined

    def test_each_segment_within_max_chars(self):
        # Generate a script well over 5000 chars
        long_text = ("The quick brown fox jumps over the lazy dog. " * 200)
        segments = _split_into_segments(long_text, max_chars=500)
        for seg in segments:
            assert len(seg) <= 500, f"Segment too long: {len(seg)}"

    def test_no_segments_are_empty(self):
        text = "Sentence one. Sentence two. Sentence three."
        segments = _split_into_segments(text, max_chars=20)
        assert all(len(s) > 0 for s in segments)


# ---------------------------------------------------------------------------
# _retry_delay_elevenlabs
# ---------------------------------------------------------------------------


class TestRetryDelayElevenlabs:
    def test_attempt_1_is_5_seconds(self):
        assert _retry_delay_elevenlabs(1) == 5.0

    def test_attempt_2_is_10_seconds(self):
        assert _retry_delay_elevenlabs(2) == 10.0

    def test_attempt_3_is_20_seconds(self):
        assert _retry_delay_elevenlabs(3) == 20.0

    def test_large_attempt_capped_at_60_seconds(self):
        assert _retry_delay_elevenlabs(100) == 60.0


# ---------------------------------------------------------------------------
# _probe_next_version
# ---------------------------------------------------------------------------


class TestProbeNextVersion:
    @pytest.mark.asyncio
    async def test_returns_1_when_no_files_exist(self):
        store = _make_asset_store()
        store.read = AsyncMock(side_effect=Exception("not found"))
        version = await _probe_next_version("vid-001", store)
        assert version == 1

    @pytest.mark.asyncio
    async def test_returns_2_when_v1_exists(self):
        store = _make_asset_store()
        # v1 exists → read succeeds; v2 does not → raises
        store.read = AsyncMock(side_effect=[b"data", Exception("not found")])
        version = await _probe_next_version("vid-001", store)
        assert version == 2

    @pytest.mark.asyncio
    async def test_returns_3_when_v1_and_v2_exist(self):
        store = _make_asset_store()
        store.read = AsyncMock(side_effect=[b"d1", b"d2", Exception("not found")])
        version = await _probe_next_version("vid-001", store)
        assert version == 3


# ---------------------------------------------------------------------------
# Narration_Generator.generate — pre-flight checks
# ---------------------------------------------------------------------------


class TestPreFlight:
    @pytest.mark.asyncio
    async def test_empty_voice_id_raises_and_notifies(self):
        notifier = _make_notifier()
        gen = _make_generator(notifier=notifier)
        with pytest.raises(NarrationGeneratorError, match="voice_id is absent or empty"):
            await gen.generate(_make_script(), voice_id="", video_id="vid-001")
        notifier.send_failure_alert.assert_called_once()
        kwargs = notifier.send_failure_alert.call_args.kwargs
        assert kwargs["stage_name"] == "narration_generator"

    @pytest.mark.asyncio
    async def test_whitespace_only_voice_id_raises(self):
        notifier = _make_notifier()
        gen = _make_generator(notifier=notifier)
        with pytest.raises(NarrationGeneratorError, match="voice_id is absent or empty"):
            await gen.generate(_make_script(), voice_id="   ", video_id="vid-001")
        notifier.send_failure_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_tts_call_when_voice_id_empty(self):
        tts = _make_tts_client()
        gen = _make_generator(tts=tts)
        with pytest.raises(NarrationGeneratorError):
            await gen.generate(_make_script(), voice_id="", video_id="vid-001")
        tts.synthesize.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_script_content_raises(self):
        notifier = _make_notifier()
        gen = _make_generator(notifier=notifier)
        script = _make_script(content="")
        with pytest.raises(NarrationGeneratorError):
            await gen.generate(script, voice_id="voice-abc", video_id="vid-001")
        notifier.send_failure_alert.assert_called_once()


# ---------------------------------------------------------------------------
# Narration_Generator.generate — happy path
# ---------------------------------------------------------------------------


class TestGenerateSuccess:
    @pytest.mark.asyncio
    async def test_returns_narration_asset(self):
        gen = _make_generator()
        result = await gen.generate(_make_script(), voice_id="v123", video_id="vid-001")
        assert isinstance(result, NarrationAsset)

    @pytest.mark.asyncio
    async def test_mp3_path_contains_video_id_and_version(self):
        gen = _make_generator()
        result = await gen.generate(_make_script(), voice_id="v123", video_id="vid-001")
        assert "vid-001" in result.mp3_path
        assert "_v1.mp3" in result.mp3_path

    @pytest.mark.asyncio
    async def test_asset_url_set_from_drive(self):
        store = _make_asset_store(drive_url="https://drive.google.com/file/abc")
        gen = _make_generator(store=store)
        result = await gen.generate(_make_script(), voice_id="v123", video_id="vid-001")
        assert result.asset_url == "https://drive.google.com/file/abc"

    @pytest.mark.asyncio
    async def test_version_increments_when_v1_exists(self):
        store = _make_asset_store()
        # v1 exists; v2 does not
        store.read = AsyncMock(side_effect=[b"old_data", Exception("not found")])
        gen = _make_generator(store=store)
        result = await gen.generate(_make_script(), voice_id="v123", video_id="vid-001")
        assert "_v2.mp3" in result.mp3_path
        assert result.version == 2

    @pytest.mark.asyncio
    async def test_content_calendar_updated_to_narration_ready(self):
        calendar = _make_content_calendar()
        gen = _make_generator(calendar=calendar)
        await gen.generate(_make_script(), voice_id="v123", video_id="vid-001")
        calendar.update_status.assert_called_once_with(
            "vid-001", PipelineStatus.NARRATION_READY
        )

    @pytest.mark.asyncio
    async def test_synthesize_called_with_correct_params(self):
        tts = _make_tts_client()
        gen = _make_generator(tts=tts)
        await gen.generate(_make_script("Short text."), voice_id="voice-xyz", video_id="vid-001")
        tts.synthesize.assert_called_once()
        _, kwargs = tts.synthesize.call_args
        assert kwargs["voice_id"] == "voice-xyz"
        assert kwargs["sample_rate"] == 44_100
        assert kwargs["bitrate_kbps"] == 128

    @pytest.mark.asyncio
    async def test_asset_store_write_called_with_mp3_subfolder(self):
        store = _make_asset_store()
        gen = _make_generator(store=store)
        await gen.generate(_make_script(), voice_id="v123", video_id="vid-001")
        store.write.assert_called_once()
        _, kwargs = store.write.call_args
        assert kwargs["subfolder"] == SubFolder.NARRATION
        assert kwargs["video_id"] == "vid-001"


# ---------------------------------------------------------------------------
# Narration_Generator.generate — multi-segment concatenation
# ---------------------------------------------------------------------------


class TestMultiSegment:
    @pytest.mark.asyncio
    async def test_two_segments_concatenated(self):
        # Build a script that is just over one segment limit.
        chunk = "Word " * 1001  # ~5005 chars
        tts = _make_tts_client(return_bytes=b"CHUNK")
        store = _make_asset_store()
        gen = _make_generator(tts=tts, store=store)
        await gen.generate(_make_script(content=chunk), voice_id="v1", video_id="vid-002")
        # synthesize should have been called more than once
        assert tts.synthesize.call_count >= 2
        # The write call should receive the concatenated bytes
        _, kwargs = store.write.call_args
        # Each call returns b"CHUNK"; concatenated = b"CHUNK" * call_count
        expected = b"CHUNK" * tts.synthesize.call_count
        assert kwargs["content"] == expected


# ---------------------------------------------------------------------------
# Narration_Generator.generate — ElevenLabs retry exhaustion
# ---------------------------------------------------------------------------


class TestElevenLabsRetry:
    @pytest.mark.asyncio
    async def test_retries_three_times_then_raises(self):
        tts = AsyncMock()
        tts.synthesize = AsyncMock(side_effect=Exception("API error"))
        notifier = _make_notifier()
        gen = _make_generator(tts=tts, notifier=notifier)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NarrationGeneratorError, match="ElevenLabs synthesis failed"):
                await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

        assert tts.synthesize.call_count == 3

    @pytest.mark.asyncio
    async def test_notifier_called_on_retry_exhaustion(self):
        tts = AsyncMock()
        tts.synthesize = AsyncMock(side_effect=Exception("API 500"))
        notifier = _make_notifier()
        gen = _make_generator(tts=tts, notifier=notifier)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NarrationGeneratorError):
                await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

        notifier.send_failure_alert.assert_called_once()
        kwargs = notifier.send_failure_alert.call_args.kwargs
        assert kwargs["stage_name"] == "narration_generator"
        assert "vid-001" == kwargs["video_id"]

    @pytest.mark.asyncio
    async def test_succeeds_on_third_attempt(self):
        tts = AsyncMock()
        # Fail twice, succeed on the third attempt.
        tts.synthesize = AsyncMock(
            side_effect=[Exception("err1"), Exception("err2"), b"FAKEMP3"]
        )
        notifier = _make_notifier()
        gen = _make_generator(tts=tts, notifier=notifier)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

        assert isinstance(result, NarrationAsset)
        notifier.send_failure_alert.assert_not_called()
        assert tts.synthesize.call_count == 3

    @pytest.mark.asyncio
    async def test_asset_store_not_called_when_tts_fails(self):
        tts = AsyncMock()
        tts.synthesize = AsyncMock(side_effect=Exception("err"))
        store = _make_asset_store()
        gen = _make_generator(tts=tts, store=store)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NarrationGeneratorError):
                await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

        store.write.assert_not_called()


# ---------------------------------------------------------------------------
# Narration_Generator.generate — Asset_Store write failure
# ---------------------------------------------------------------------------


class TestAssetStoreFailure:
    @pytest.mark.asyncio
    async def test_asset_store_write_failure_raises_narration_error(self):
        store = _make_asset_store()
        store.write = AsyncMock(side_effect=AssetStoreError("Drive unavailable"))
        notifier = _make_notifier()
        gen = _make_generator(store=store, notifier=notifier)

        with pytest.raises(NarrationGeneratorError, match="Asset_Store write failed"):
            await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

    @pytest.mark.asyncio
    async def test_notifier_called_on_asset_store_write_failure(self):
        store = _make_asset_store()
        store.write = AsyncMock(side_effect=AssetStoreError("Drive down"))
        notifier = _make_notifier()
        gen = _make_generator(store=store, notifier=notifier)

        with pytest.raises(NarrationGeneratorError):
            await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

        notifier.send_failure_alert.assert_called_once()
        kwargs = notifier.send_failure_alert.call_args.kwargs
        assert kwargs["stage_name"] == "narration_generator"


# ---------------------------------------------------------------------------
# Narration_Generator.generate — Content_Calendar update failure
# ---------------------------------------------------------------------------


class TestContentCalendarFailure:
    @pytest.mark.asyncio
    async def test_content_calendar_failure_raises_narration_error(self):
        calendar = _make_content_calendar()
        calendar.update_status = AsyncMock(side_effect=Exception("Notion down"))
        notifier = _make_notifier()
        gen = _make_generator(calendar=calendar, notifier=notifier)

        with pytest.raises(NarrationGeneratorError, match="Content_Calendar status update failed"):
            await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

    @pytest.mark.asyncio
    async def test_notifier_called_on_calendar_failure(self):
        calendar = _make_content_calendar()
        calendar.update_status = AsyncMock(side_effect=Exception("Notion error"))
        notifier = _make_notifier()
        gen = _make_generator(calendar=calendar, notifier=notifier)

        with pytest.raises(NarrationGeneratorError):
            await gen.generate(_make_script(), voice_id="v1", video_id="vid-001")

        notifier.send_failure_alert.assert_called_once()
