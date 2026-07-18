"""Unit tests for pipeline.script_writer (Tasks 8.1, 8.2, 8.3).

Tests cover:
- Missing topic title triggers Notifier without calling Claude (8.1)
- None style_profile triggers Notifier without calling Claude (8.1)
- Claude API error applies exponential retry policy (8.1)
- Successful generation stores correct Script fields (8.1)
- Word count in bounds → no automatic revision (8.2)
- Word count out of bounds → one automatic revision (8.2)
- Word count still out of bounds after revision → ScriptGenerationError + Notifier (8.2)
- Script stored as script_v{n}.md; version increments correctly (8.2)
- revise with empty edits string raises EmptyEditError (8.3)
- revise with identical content raises EmptyEditError (8.3)
- revise applies edits and writes script_v{n+1}.md (8.3)
- revise preserves previous version (8.3)
- style_profile_doc_id is recorded in Script metadata (8.1)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from pipeline.asset_store import Asset_Store, AssetStoreError
from pipeline.models import (
    NarrationTone,
    Pacing,
    Script,
    SegmentStructure,
    StyleProfile,
    SubFolder,
    ThumbnailComposition,
    TopicEntry,
    VisualStyle,
)
from pipeline.notifier import Notifier, NotifierConfig
from pipeline.script_writer import (
    ClaudeClient,
    EmptyEditError,
    Script_Writer,
    ScriptGenerationError,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_VALID_CONTENT = " ".join(["word"] * 1000)  # 1000 words — within [800, 1500]
_SHORT_CONTENT = " ".join(["word"] * 400)   # 400 words — below 800
_LONG_CONTENT = " ".join(["word"] * 2000)   # 2000 words — above 1500


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_topic(title: str = "How AI is Changing Everything") -> TopicEntry:
    return TopicEntry(
        title=title,
        composite_score=0.85,
        recency_hours=12.0,
        source_query_timestamp=_utcnow(),
        search_volume_signal=1000.0,
        relevance_tags_matched=["large language model"],
    )


def _make_style_profile(doc_id: str = "sp-doc-001") -> StyleProfile:
    return StyleProfile(
        doc_id=doc_id,
        version=1,
        created_at=_utcnow(),
        channel_url="https://youtube.com/@testchannel",
        narration_tone=NarrationTone(sentiment_polarity=0.5),
        pacing=Pacing(avg_words_per_minute=140, avg_sentence_length_words=12.0),
        segment_structure=SegmentStructure(
            intro_present=True,
            hook_present=True,
            body_segment_count_avg=4.0,
            cta_present=True,
        ),
        visual_style=VisualStyle(composition_patterns=["close-up", "b-roll"]),
        thumbnail_composition=ThumbnailComposition(
            dominant_colors=["#FF0000", "#FFFFFF"],
            text_overlay_position="bottom-left",
            subject_framing="center",
            sample_count=15,
            lookback_days=90,
        ),
        rhetorical_patterns=["direct address", "rhetorical question"],
    )


def _make_script(version: int = 1, content: str = _VALID_CONTENT) -> Script:
    return Script(
        video_id="video-001",
        version=version,
        content=content,
        word_count=len(content.split()),
        style_profile_doc_id="sp-doc-001",
        asset_url="https://drive.google.com/file/v1",
        created_at=_utcnow(),
    )


def _make_notifier() -> Notifier:
    return Notifier(config=NotifierConfig())


def _make_mock_claude(response: str = _VALID_CONTENT) -> AsyncMock:
    """Return an AsyncMock that satisfies the ClaudeClient protocol."""
    mock = AsyncMock()
    mock.complete = AsyncMock(return_value=response)
    return mock


def _make_mock_store(*, url_raises: bool = True) -> MagicMock:
    """Return a MagicMock Asset_Store.

    By default ``url()`` raises AssetStoreError (file not found), meaning
    version 1 is always the next version.  Set ``url_raises=False`` to
    simulate an existing v1 file.
    """
    store = MagicMock(spec=Asset_Store)
    if url_raises:
        store.url = AsyncMock(side_effect=AssetStoreError("not found"))
    else:
        # First call returns successfully (v1 exists), second raises (v2 free)
        store.url = AsyncMock(
            side_effect=[
                "https://drive.google.com/v1",
                AssetStoreError("not found"),
            ]
        )
    store.write = AsyncMock(return_value="https://drive.google.com/new-file")
    return store


# ---------------------------------------------------------------------------
# Task 8.1 — generate: input validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_empty_topic_title_notifies_and_raises() -> None:
    """Missing topic title must notify Notifier and raise ScriptGenerationError."""
    notifier = MagicMock(spec=Notifier)
    claude = _make_mock_claude()
    store = _make_mock_store()
    writer = Script_Writer(claude, store, notifier)

    empty_topic = _make_topic(title="   ")  # whitespace-only
    with pytest.raises(ScriptGenerationError):
        await writer.generate(empty_topic, _make_style_profile(), "video-001")

    notifier.send_failure_alert.assert_called_once()
    claude.complete.assert_not_called()


@pytest.mark.asyncio
async def test_generate_none_style_profile_notifies_and_raises() -> None:
    """None style_profile must notify Notifier and raise ScriptGenerationError."""
    notifier = MagicMock(spec=Notifier)
    claude = _make_mock_claude()
    store = _make_mock_store()
    writer = Script_Writer(claude, store, notifier)

    with pytest.raises(ScriptGenerationError):
        await writer.generate(_make_topic(), None, "video-001")  # type: ignore[arg-type]

    notifier.send_failure_alert.assert_called_once()
    claude.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Task 8.1 — generate: Claude retry policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_claude_retries_on_error() -> None:
    """Claude API errors are retried up to 3 times with exponential back-off."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()

    claude = AsyncMock()
    # Fail twice then succeed
    claude.complete = AsyncMock(
        side_effect=[Exception("API error"), Exception("API error"), _VALID_CONTENT]
    )

    writer = Script_Writer(claude, store, notifier)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.word_count == len(_VALID_CONTENT.split())
    assert claude.complete.call_count == 3
    # Two sleep calls between attempts 1→2 and 2→3
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_generate_claude_all_retries_exhausted_raises() -> None:
    """All 3 Claude retries failing raises the original exception."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()

    claude = AsyncMock()
    claude.complete = AsyncMock(side_effect=Exception("permanent API error"))

    writer = Script_Writer(claude, store, notifier)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(Exception, match="permanent API error"):
            await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert claude.complete.call_count == 3


# ---------------------------------------------------------------------------
# Task 8.1 — generate: Script fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_script_records_style_profile_doc_id() -> None:
    """Generated Script must record the style_profile.doc_id."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()
    claude = _make_mock_claude(_VALID_CONTENT)

    style_profile = _make_style_profile(doc_id="sp-xyz-789")
    writer = Script_Writer(claude, store, notifier)

    script = await writer.generate(_make_topic(), style_profile, "video-001")

    assert script.style_profile_doc_id == "sp-xyz-789"


@pytest.mark.asyncio
async def test_generate_script_has_correct_word_count() -> None:
    """Script.word_count must equal len(content.split())."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()
    claude = _make_mock_claude(_VALID_CONTENT)
    writer = Script_Writer(claude, store, notifier)

    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.word_count == len(_VALID_CONTENT.split())


# ---------------------------------------------------------------------------
# Task 8.2 — word-count enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_word_count_in_bounds_no_revision() -> None:
    """Word count in [800, 1500] must not trigger a second Claude call."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()
    claude = _make_mock_claude(_VALID_CONTENT)
    writer = Script_Writer(claude, store, notifier)

    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert 800 <= script.word_count <= 1500
    # Only ONE Claude call needed when word count is already valid
    assert claude.complete.call_count == 1


@pytest.mark.asyncio
async def test_generate_word_count_too_short_triggers_revision() -> None:
    """Word count below 800 must trigger exactly one automatic revision call."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()

    claude = AsyncMock()
    # First call returns short content; second call (revision) returns valid content
    claude.complete = AsyncMock(side_effect=[_SHORT_CONTENT, _VALID_CONTENT])

    writer = Script_Writer(claude, store, notifier)
    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.word_count == len(_VALID_CONTENT.split())
    assert claude.complete.call_count == 2  # initial + one revision


@pytest.mark.asyncio
async def test_generate_word_count_too_long_triggers_revision() -> None:
    """Word count above 1500 must trigger exactly one automatic revision call."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()

    claude = AsyncMock()
    claude.complete = AsyncMock(side_effect=[_LONG_CONTENT, _VALID_CONTENT])

    writer = Script_Writer(claude, store, notifier)
    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.word_count == len(_VALID_CONTENT.split())
    assert claude.complete.call_count == 2


@pytest.mark.asyncio
async def test_generate_word_count_still_wrong_after_revision_raises_and_notifies() -> None:
    """Word count still out of bounds after revision must raise ScriptGenerationError and notify."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()

    claude = AsyncMock()
    # Both calls return out-of-bounds content
    claude.complete = AsyncMock(side_effect=[_SHORT_CONTENT, _SHORT_CONTENT])

    writer = Script_Writer(claude, store, notifier)

    with pytest.raises(ScriptGenerationError):
        await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    # ScriptGenerationError must have triggered a failure alert
    notifier.send_failure_alert.assert_called_once()
    # Two Claude calls: initial + one revision attempt
    assert claude.complete.call_count == 2
    # Asset_Store write must NOT have been called (failure before write)
    store.write.assert_not_called()


# ---------------------------------------------------------------------------
# Task 8.2 — versioned Asset_Store writes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_writes_script_v1_when_no_prior_versions() -> None:
    """First generation must write script_v1.md."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store(url_raises=True)  # url() always raises → v1 is next
    claude = _make_mock_claude(_VALID_CONTENT)
    writer = Script_Writer(claude, store, notifier)

    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.version == 1
    store.write.assert_called_once_with(
        video_id="video-001",
        subfolder=SubFolder.SCRIPTS,
        filename="script_v1.md",
        content=_VALID_CONTENT.encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_generate_writes_script_v2_when_v1_exists() -> None:
    """Second generation must write script_v2.md when v1 already exists."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store(url_raises=False)  # url() succeeds for v1, fails for v2
    claude = _make_mock_claude(_VALID_CONTENT)
    writer = Script_Writer(claude, store, notifier)

    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.version == 2
    store.write.assert_called_once_with(
        video_id="video-001",
        subfolder=SubFolder.SCRIPTS,
        filename="script_v2.md",
        content=_VALID_CONTENT.encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_generate_asset_url_set_from_store_write() -> None:
    """Script.asset_url must match the URL returned by Asset_Store.write."""
    notifier = MagicMock(spec=Notifier)
    expected_url = "https://drive.google.com/expected-url"
    store = _make_mock_store()
    store.write = AsyncMock(return_value=expected_url)
    claude = _make_mock_claude(_VALID_CONTENT)
    writer = Script_Writer(claude, store, notifier)

    script = await writer.generate(_make_topic(), _make_style_profile(), "video-001")

    assert script.asset_url == expected_url


# ---------------------------------------------------------------------------
# Task 8.3 — revise: validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revise_empty_edits_string_raises_empty_edit_error() -> None:
    """Empty edit string must raise EmptyEditError without calling Claude."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()
    claude = _make_mock_claude()
    writer = Script_Writer(claude, store, notifier)

    existing = _make_script(version=1)

    with pytest.raises(EmptyEditError):
        await writer.revise(existing, "", "video-001")

    claude.complete.assert_not_called()


@pytest.mark.asyncio
async def test_revise_whitespace_only_edits_raises_empty_edit_error() -> None:
    """Whitespace-only edit string must raise EmptyEditError."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()
    claude = _make_mock_claude()
    writer = Script_Writer(claude, store, notifier)

    existing = _make_script(version=1)

    with pytest.raises(EmptyEditError):
        await writer.revise(existing, "   \n\t  ", "video-001")

    claude.complete.assert_not_called()


@pytest.mark.asyncio
async def test_revise_identical_content_raises_empty_edit_error() -> None:
    """Edits identical to current content must raise EmptyEditError."""
    notifier = MagicMock(spec=Notifier)
    store = _make_mock_store()
    claude = _make_mock_claude()
    writer = Script_Writer(claude, store, notifier)

    existing = _make_script(version=1, content=_VALID_CONTENT)

    with pytest.raises(EmptyEditError):
        await writer.revise(existing, _VALID_CONTENT, "video-001")

    claude.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Task 8.3 — revise: correct revision behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revise_calls_claude_and_saves_next_version() -> None:
    """revise must call Claude and save script_v{n+1}.md."""
    notifier = MagicMock(spec=Notifier)
    store = MagicMock(spec=Asset_Store)
    store.write = AsyncMock(return_value="https://drive.google.com/revised")

    revised_text = " ".join(["revised"] * 900)
    claude = _make_mock_claude(revised_text)
    writer = Script_Writer(claude, store, notifier)

    existing = _make_script(version=2, content=_VALID_CONTENT)
    new_script = await writer.revise(existing, "Please make it shorter", "video-001")

    assert new_script.version == 3  # n+1
    assert new_script.content == revised_text
    assert new_script.word_count == len(revised_text.split())
    assert new_script.style_profile_doc_id == existing.style_profile_doc_id

    store.write.assert_called_once_with(
        video_id="video-001",
        subfolder=SubFolder.SCRIPTS,
        filename="script_v3.md",
        content=revised_text.encode("utf-8"),
    )


@pytest.mark.asyncio
async def test_revise_preserves_previous_version() -> None:
    """revise must only write the new version, not delete or overwrite the old one."""
    notifier = MagicMock(spec=Notifier)
    store = MagicMock(spec=Asset_Store)
    store.write = AsyncMock(return_value="https://drive.google.com/v2")

    revised_text = " ".join(["new"] * 900)
    claude = _make_mock_claude(revised_text)
    writer = Script_Writer(claude, store, notifier)

    existing = _make_script(version=1)
    await writer.revise(existing, "Add more detail about the main topic", "video-001")

    # write called exactly once — for the new version only
    store.write.assert_called_once()
    written_filename = store.write.call_args.kwargs["filename"]
    assert written_filename == "script_v2.md"


@pytest.mark.asyncio
async def test_revise_asset_url_set_from_store_write() -> None:
    """Revised Script.asset_url must match the URL returned by Asset_Store.write."""
    notifier = MagicMock(spec=Notifier)
    expected_url = "https://drive.google.com/revised-url"
    store = MagicMock(spec=Asset_Store)
    store.write = AsyncMock(return_value=expected_url)

    revised_text = " ".join(["word"] * 900)
    claude = _make_mock_claude(revised_text)
    writer = Script_Writer(claude, store, notifier)

    existing = _make_script(version=1)
    new_script = await writer.revise(existing, "Please revise this", "video-001")

    assert new_script.asset_url == expected_url
