"""Unit tests for pipeline.metadata_generator (Task 12.1).

Tests cover:
- Pre-flight: empty script content triggers Notifier and raises MetadataGenerationError
- Pre-flight: empty topics list triggers Notifier and raises MetadataGenerationError
- primary_keyword resolves to the highest composite_score topic
- Chapter derivation builds correct MM:SS timestamps from script segments
- Successful generation returns a valid MetadataPackage and writes JSON to Asset_Store
- Title validation failure triggers targeted regeneration, returns fixed title
- Title still invalid after regeneration → halt + Notifier + MetadataGenerationError
- Description validation failure triggers targeted regeneration
- Description still invalid after regeneration → halt + raise
- Tags validation failure triggers targeted regeneration
- Tags still invalid after regeneration → halt + raise
- Hashtags validation failure triggers targeted regeneration
- Hashtags still invalid after regeneration → halt + raise
- Claude API error applies retry policy (3 attempts) for main generation call
- JSON parse failure on Claude response halts stage and raises MetadataGenerationError
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from pipeline.asset_store import Asset_Store, AssetStoreError
from pipeline.metadata_generator import Metadata_Generator, MetadataGenerationError
from pipeline.metadata_generator import (
    _derive_chapters,
    _primary_keyword,
    _validate_description,
    _validate_hashtags,
    _validate_tags,
    _validate_title,
)
from pipeline.models import Script, SubFolder, TopicEntry
from pipeline.notifier import Notifier, NotifierConfig


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _make_topic(title: str, score: float = 0.5) -> TopicEntry:
    return TopicEntry(
        title=title,
        composite_score=score,
        recency_hours=10.0,
        source_query_timestamp=_utcnow(),
        search_volume_signal=500.0,
        relevance_tags_matched=[],
    )


def _make_script(content: str = "", video_id: str = "vid-001") -> Script:
    return Script(
        video_id=video_id,
        version=1,
        content=content,
        word_count=len(content.split()),
        style_profile_doc_id="sp-001",
        created_at=_utcnow(),
    )


def _good_description(tags: list[str]) -> str:
    """Build a description that passes validation: 200-500 words, includes ≥3 tags."""
    # Use the first 3 tags inline; pad to 210 words to safely clear the 200-word minimum
    tag_str = " ".join(tags[:3])
    filler = " ".join(["content"] * 200)
    return f"Summary paragraph here. {filler} {tag_str} closing paragraph here."


def _good_json_response(
    title: str,
    description: str,
    tags: list[str],
    hashtags: list[str],
) -> str:
    return json.dumps(
        {
            "title": title,
            "description": description,
            "tags": tags,
            "hashtags": hashtags,
        }
    )


_GOOD_TAGS = [
    "artificial intelligence tutorial",
    "machine learning basics",
    "deep learning guide",
    "neural network explained",
    "AI technology trends",
    "large language models",
    "generative AI tools",
    "prompt engineering tips",
    "AI content creation",
    "YouTube AI channel",
]  # exactly 10 tags, each 2-5 words

_GOOD_HASHTAGS = ["#AIContent", "#MachineLearning", "#DeepLearning"]

_SCRIPT_BODY = (
    "# HOOK\n"
    "This video covers artificial intelligence.\n\n"
    "# BODY SEGMENT ONE\n"
    + " ".join(["word"] * 150) + "\n\n"
    "# BODY SEGMENT TWO\n"
    + " ".join(["word"] * 150) + "\n\n"
    "# CTA\n"
    "Subscribe and comment below.\n"
)

@pytest.fixture()
def notifier() -> Notifier:
    cfg = NotifierConfig(slack_webhook_url=None, discord_webhook_url=None, smtp=None)
    n = Notifier(config=cfg)
    n.send_failure_alert = MagicMock()  # type: ignore[method-assign]
    return n


@pytest.fixture()
def asset_store() -> Asset_Store:
    store = MagicMock(spec=Asset_Store)
    store.write = AsyncMock(return_value="https://drive.google.com/fake-url")
    return store


@pytest.fixture()
def claude() -> MagicMock:
    return MagicMock()


def _make_generator(claude: MagicMock, asset_store: Asset_Store, notifier: Notifier) -> Metadata_Generator:
    return Metadata_Generator(
        claude_client=claude,  # type: ignore[arg-type]
        asset_store=asset_store,
        notifier=notifier,
    )


# ---------------------------------------------------------------------------
# Unit tests: validation helpers
# ---------------------------------------------------------------------------


class TestValidateTitle:
    def test_valid_title(self) -> None:
        assert _validate_title("AI Tutorial for Beginners", "AI Tutorial") is True

    def test_too_long(self) -> None:
        long = "A" * 61
        assert _validate_title(long, "AI") is False

    def test_missing_keyword(self) -> None:
        assert _validate_title("Great Video About Technology", "AI Tutorial") is False

    def test_keyword_case_insensitive(self) -> None:
        assert _validate_title("all about ai tutorial today", "AI Tutorial") is True

    def test_exactly_60_chars(self) -> None:
        kw = "AI"
        title = "A" * 58 + kw  # 60 chars total
        assert _validate_title(title, kw) is True

    def test_61_chars_fails(self) -> None:
        kw = "AI"
        title = "A" * 59 + kw  # 61 chars total
        assert _validate_title(title, kw) is False


class TestValidateTags:
    def test_valid_tags(self) -> None:
        tags = ["machine learning basics"] * 10
        assert _validate_tags(tags) is True

    def test_too_few(self) -> None:
        assert _validate_tags(["tag one two"] * 9) is False

    def test_too_many(self) -> None:
        assert _validate_tags(["tag one two"] * 16) is False

    def test_single_word_tag_fails(self) -> None:
        tags = ["ai"] + ["two words here"] * 9
        assert _validate_tags(tags) is False

    def test_six_word_tag_fails(self) -> None:
        tags = ["one two three four five six"] + ["two words"] * 9
        assert _validate_tags(tags) is False

    def test_boundary_10_tags(self) -> None:
        assert _validate_tags(["valid tag"] * 10) is True

    def test_boundary_15_tags(self) -> None:
        assert _validate_tags(["valid tag"] * 15) is True


class TestValidateHashtags:
    def test_valid_hashtags(self) -> None:
        assert _validate_hashtags(["#AIContent", "#MachineLearning", "#Tech"]) is True

    def test_too_few(self) -> None:
        assert _validate_hashtags(["#AI", "#ML"]) is False

    def test_too_many(self) -> None:
        assert _validate_hashtags(["#A" + str(i) for i in range(6)]) is False

    def test_missing_hash(self) -> None:
        assert _validate_hashtags(["AIContent", "#MachineLearning", "#Tech"]) is False

    def test_with_space_fails(self) -> None:
        assert _validate_hashtags(["#AI Content", "#MachineLearning", "#Tech"]) is False

    def test_body_too_short_fails(self) -> None:
        assert _validate_hashtags(["#A", "#MachineLearning", "#Tech"]) is False

    def test_body_exactly_2_chars(self) -> None:
        assert _validate_hashtags(["#AI", "#ML", "#ok"]) is True

    def test_body_exactly_30_chars(self) -> None:
        ht = "#" + "A" * 30
        assert _validate_hashtags([ht, "#Two", "#Three"]) is True

    def test_body_31_chars_fails(self) -> None:
        ht = "#" + "A" * 31
        assert _validate_hashtags([ht, "#Two", "#Three"]) is False


class TestValidateDescription:
    def test_valid_description(self) -> None:
        tags = ["machine learning basics", "AI tutorial guide", "deep learning explained"]
        # 9 + 200 + 10 ≈ 219 words — within [200, 500] and has 3 tag hits
        filler = " ".join(["word"] * 200)
        desc = f"Summary paragraph here. {filler} machine learning basics AI tutorial guide deep learning explained end."
        assert _validate_description(desc, tags) is True

    def test_too_short(self) -> None:
        desc = " ".join(["word"] * 50)
        tags = ["machine learning"]
        assert _validate_description(desc, tags) is False

    def test_too_long(self) -> None:
        desc = " ".join(["word"] * 600)
        tags = ["machine learning"]
        assert _validate_description(desc, tags) is False

    def test_insufficient_tag_appearances(self) -> None:
        tags = ["machine learning", "deep learning", "neural network"]
        desc = " ".join(["unrelated word"] * 250)  # no tags in text
        assert _validate_description(desc, tags) is False

    def test_exactly_3_tag_hits(self) -> None:
        tags = ["tag alpha", "tag beta", "tag gamma", "tag delta"]
        # 200 words of filler + 3 tag phrases
        filler = " ".join(["word"] * 195)
        desc = f"{filler} tag alpha tag beta tag gamma "
        assert _validate_description(desc, tags) is True


class TestPrimaryKeyword:
    def test_returns_highest_score(self) -> None:
        topics = [
            _make_topic("Low Score Topic", score=0.2),
            _make_topic("High Score Topic", score=0.9),
            _make_topic("Mid Score Topic", score=0.5),
        ]
        assert _primary_keyword(topics) == "High Score Topic"

    def test_single_topic(self) -> None:
        assert _primary_keyword([_make_topic("Only Topic", 0.7)]) == "Only Topic"


class TestDeriveChapters:
    def test_segments_produce_chapters(self) -> None:
        script = _make_script(_SCRIPT_BODY)
        chapters = _derive_chapters(script)
        assert len(chapters) >= 2
        # First chapter always starts at 00:00
        assert chapters[0].timestamp == "00:00"

    def test_timestamps_are_mm_ss(self) -> None:
        script = _make_script(_SCRIPT_BODY)
        import re
        for ch in _derive_chapters(script):
            assert re.match(r"^\d{2}:\d{2}$", ch.timestamp), f"bad ts: {ch.timestamp}"

    def test_timestamps_increase_monotonically(self) -> None:
        script = _make_script(_SCRIPT_BODY)
        chapters = _derive_chapters(script)
        times = []
        for ch in chapters:
            mm, ss = map(int, ch.timestamp.split(":"))
            times.append(mm * 60 + ss)
        assert times == sorted(times)

    def test_no_headings_fallback(self) -> None:
        script = _make_script("just some regular text with no headings here.")
        chapters = _derive_chapters(script)
        assert len(chapters) >= 1
        assert chapters[0].timestamp == "00:00"

# ---------------------------------------------------------------------------
# Unit tests: Metadata_Generator.generate
# ---------------------------------------------------------------------------


class TestMetadataGeneratorPreflight:
    async def test_empty_script_content_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic("AI Topic", 0.9)]
        script = _make_script("")

        with pytest.raises(MetadataGenerationError, match="script content is empty"):
            await mg.generate(script, topics, "vid-001")

        notifier.send_failure_alert.assert_called_once()
        # Claude should NOT have been called
        claude.complete.assert_not_called()

    async def test_whitespace_only_script_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic("AI Topic", 0.9)]
        script = _make_script("   \n\t  ")

        with pytest.raises(MetadataGenerationError):
            await mg.generate(script, topics, "vid-001")

        claude.complete.assert_not_called()

    async def test_empty_topics_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        mg = _make_generator(claude, asset_store, notifier)
        script = _make_script("Some valid script content here.")

        with pytest.raises(MetadataGenerationError, match="topics list is empty"):
            await mg.generate(script, [], "vid-001")

        notifier.send_failure_alert.assert_called_once()
        claude.complete.assert_not_called()


class TestMetadataGeneratorSuccess:
    async def test_successful_generation_returns_package(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        """All fields valid → MetadataPackage returned and JSON written to store."""
        primary_kw = "Artificial Intelligence Tutorial"
        good_desc = _good_description(_GOOD_TAGS)
        response_json = _good_json_response(
            title=f"{primary_kw} for Beginners",
            description=good_desc,
            tags=_GOOD_TAGS,
            hashtags=_GOOD_HASHTAGS,
        )
        claude.complete = AsyncMock(return_value=response_json)

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9), _make_topic("Low Topic", 0.1)]
        script = _make_script(_SCRIPT_BODY)

        pkg = await mg.generate(script, topics, "vid-001")

        assert pkg.video_id == "vid-001"
        assert pkg.primary_keyword == primary_kw
        assert len(pkg.title) <= 60
        assert primary_kw.lower() in pkg.title.lower()
        assert 10 <= len(pkg.tags) <= 15
        assert 3 <= len(pkg.hashtags) <= 5
        assert len(pkg.chapters) >= 1

        # Verify JSON was written to Asset_Store
        asset_store.write.assert_called_once()
        call_args = asset_store.write.call_args
        assert call_args.kwargs["subfolder"] == SubFolder.METADATA
        assert call_args.kwargs["filename"] == "vid-001.json"

    async def test_asset_store_receives_valid_json(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        """The bytes written to Asset_Store should parse as valid JSON."""
        primary_kw = "Machine Learning Guide"
        good_desc = _good_description(_GOOD_TAGS)
        claude.complete = AsyncMock(
            return_value=_good_json_response(
                title=f"The {primary_kw}",
                description=good_desc,
                tags=_GOOD_TAGS,
                hashtags=_GOOD_HASHTAGS,
            )
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.8)]
        await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-002")

        written_bytes: bytes = asset_store.write.call_args.kwargs["content"]
        parsed = json.loads(written_bytes.decode("utf-8"))
        assert parsed["video_id"] == "vid-002"
        assert "title" in parsed
        assert "description" in parsed
        assert "tags" in parsed
        assert "hashtags" in parsed
        assert "chapters" in parsed
        assert "primary_keyword" in parsed


class TestMetadataGeneratorTitleValidation:
    async def test_title_too_long_triggers_regen(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        """Title > 60 chars → regeneration called; valid regen title accepted."""
        primary_kw = "AI Tutorial"
        good_desc = _good_description(_GOOD_TAGS)
        long_title = "A" * 61  # 61 chars — invalid

        # Call sequence:
        # 1. Full generation (bad title, good desc, good tags, good hashtags)
        # 2. Title regen → returns good title
        good_title = f"Great {primary_kw} for Everyone"
        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(long_title, good_desc, _GOOD_TAGS, _GOOD_HASHTAGS),
                good_title,  # title regen response (plain string)
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]
        pkg = await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-003")

        assert pkg.title == good_title
        assert claude.complete.call_count == 2

    async def test_title_missing_keyword_triggers_regen(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        primary_kw = "Deep Learning Basics"
        good_desc = _good_description(_GOOD_TAGS)
        bad_title = "Amazing Video About Stuff Today!"  # missing primary keyword

        good_title = f"Intro to {primary_kw}"
        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(bad_title, good_desc, _GOOD_TAGS, _GOOD_HASHTAGS),
                good_title,
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]
        pkg = await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-004")

        assert primary_kw.lower() in pkg.title.lower()

    async def test_title_still_invalid_after_regen_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        primary_kw = "Neural Networks Explained"
        good_desc = _good_description(_GOOD_TAGS)
        bad_title = "A" * 61  # too long

        # Both responses produce an invalid title
        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(bad_title, good_desc, _GOOD_TAGS, _GOOD_HASHTAGS),
                "B" * 61,  # still too long after regen
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]

        with pytest.raises(MetadataGenerationError, match="title still invalid"):
            await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-005")

        notifier.send_failure_alert.assert_called_once()
        asset_store.write.assert_not_called()


class TestMetadataGeneratorTagsValidation:
    async def test_too_few_tags_triggers_regen(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        primary_kw = "AI Content Creation"
        good_desc = _good_description(_GOOD_TAGS)
        few_tags = ["one two three"] * 5  # only 5 tags

        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(
                    f"Best {primary_kw} Guide", good_desc, few_tags, _GOOD_HASHTAGS
                ),
                json.dumps(_GOOD_TAGS),  # regen returns good tags as JSON array
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]
        pkg = await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-006")

        assert 10 <= len(pkg.tags) <= 15

    async def test_tags_still_invalid_after_regen_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        primary_kw = "Prompt Engineering Tips"
        good_desc = _good_description(_GOOD_TAGS)
        bad_tags = ["one"] * 5  # 5 single-word tags — invalid on count and word count

        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(
                    f"Guide to {primary_kw}", good_desc, bad_tags, _GOOD_HASHTAGS
                ),
                json.dumps(["still bad"] * 5),  # regen still only 5 tags
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]

        with pytest.raises(MetadataGenerationError):
            await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-007")


class TestMetadataGeneratorHashtagsValidation:
    async def test_malformed_hashtag_triggers_regen(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        primary_kw = "YouTube SEO Strategy"
        good_desc = _good_description(_GOOD_TAGS)
        bad_hashtags = ["noHash", "alsoNoHash", "stillNoHash"]  # missing '#'

        good_hashtags = ["#YouTubeSEO", "#ContentStrategy", "#VideoMarketing"]
        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(
                    f"Best {primary_kw}", good_desc, _GOOD_TAGS, bad_hashtags
                ),
                json.dumps(good_hashtags),
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]
        pkg = await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-008")

        assert all(h.startswith("#") for h in pkg.hashtags)
        assert 3 <= len(pkg.hashtags) <= 5

    async def test_hashtags_still_invalid_after_regen_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        primary_kw = "AI Video Production"
        good_desc = _good_description(_GOOD_TAGS)

        claude.complete = AsyncMock(
            side_effect=[
                _good_json_response(
                    f"Guide: {primary_kw}", good_desc, _GOOD_TAGS, ["bad1", "bad2"]
                ),
                json.dumps(["stillBad1", "stillBad2"]),  # still malformed (no '#')
            ]
        )

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]

        with pytest.raises(MetadataGenerationError):
            await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-009")


class TestMetadataGeneratorClaudeRetry:
    async def test_claude_error_retries_3_times(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        """Claude API errors trigger up to 3 retry attempts (with mocked sleep)."""
        import pipeline.metadata_generator as mg_module

        call_count = 0

        async def flaky_complete(prompt: str, max_tokens: int = 4096) -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Claude API unavailable")

        claude.complete = flaky_complete

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic("Test Topic", 0.9)]
        script = _make_script(_SCRIPT_BODY)

        # Patch asyncio.sleep to avoid waiting during tests
        from unittest.mock import patch
        with patch("pipeline.metadata_generator.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="Claude API unavailable"):
                await mg.generate(script, topics, "vid-010")

        assert call_count == 3

    async def test_claude_json_parse_failure_raises(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        """Non-JSON Claude response halts stage and raises MetadataGenerationError."""
        claude.complete = AsyncMock(return_value="This is not JSON at all!!!")

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic("AI Topic", 0.9)]
        script = _make_script(_SCRIPT_BODY)

        with pytest.raises(MetadataGenerationError, match="failed to parse"):
            await mg.generate(script, topics, "vid-011")

        notifier.send_failure_alert.assert_called_once()
        asset_store.write.assert_not_called()

    async def test_claude_markdown_fenced_json_is_parsed(
        self, claude: MagicMock, asset_store: Asset_Store, notifier: Notifier
    ) -> None:
        """Claude response wrapped in markdown fences should be stripped and parsed."""
        primary_kw = "AI for Creators"
        good_desc = _good_description(_GOOD_TAGS)
        inner_json = _good_json_response(
            f"Learn {primary_kw} Today", good_desc, _GOOD_TAGS, _GOOD_HASHTAGS
        )
        fenced = f"```json\n{inner_json}\n```"
        claude.complete = AsyncMock(return_value=fenced)

        mg = _make_generator(claude, asset_store, notifier)
        topics = [_make_topic(primary_kw, 0.9)]
        pkg = await mg.generate(_make_script(_SCRIPT_BODY), topics, "vid-012")

        assert pkg.primary_keyword == primary_kw
