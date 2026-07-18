"""Hypothesis generator strategies for AI YouTube Content Pipeline property-based tests.

Each strategy generates valid Pydantic model instances that satisfy all field validators
and constraints defined in ``pipeline/models.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

from hypothesis import strategies as st

from pipeline.models import (
    Chapter,
    MetadataPackage,
    NarrationTone,
    Pacing,
    Platform,
    PipelineStatus,
    SegmentStructure,
    StyleProfile,
    ThumbnailComposition,
    TopicEntry,
    VideoRecord,
    VisualStyle,
    AssetLinks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Timezone-aware UTC datetime strategy used across strategies
_UTC = timezone.utc

_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=_UTC)
_ONE_YEAR = timedelta(days=365)
_FIFTEEN_MIN = timedelta(minutes=15)


def datetime_strategy() -> st.SearchStrategy[datetime]:
    """Generate timezone-aware UTC datetimes spanning past (up to 1 year ago),
    present +/- 15 minutes, and future (up to 1 year from now).
    """
    past_start = _NOW - _ONE_YEAR
    future_end = _NOW + _ONE_YEAR

    # Express as integer milliseconds to keep Hypothesis happy with floats
    past_ms = int(past_start.timestamp() * 1000)
    future_ms = int(future_end.timestamp() * 1000)

    return st.integers(min_value=past_ms, max_value=future_ms).map(
        lambda ms: datetime.fromtimestamp(ms / 1000.0, tz=_UTC)
    )


# ---------------------------------------------------------------------------
# Strategy 1 — TopicEntry
# ---------------------------------------------------------------------------


@st.composite
def topic_entry_strategy(draw: st.DrawFn) -> TopicEntry:
    """Generate a valid :class:`TopicEntry` instance."""
    title = draw(st.text(min_size=1, max_size=200))
    composite_score = draw(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    recency_hours = draw(
        st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
    )
    source_query_timestamp = draw(datetime_strategy())
    search_volume_signal = draw(
        st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False)
    )
    relevance_tags_matched = draw(
        st.lists(st.text(min_size=1, max_size=50), min_size=0, max_size=5)
    )

    return TopicEntry(
        title=title,
        composite_score=composite_score,
        recency_hours=recency_hours,
        source_query_timestamp=source_query_timestamp,
        search_volume_signal=search_volume_signal,
        relevance_tags_matched=relevance_tags_matched,
    )


# ---------------------------------------------------------------------------
# Strategy 2 — StyleProfile
# ---------------------------------------------------------------------------

# Valid CSS hex colour like "#AABBCC"
_HEX_CHAR = st.sampled_from("0123456789ABCDEF")


@st.composite
def _hex_color_strategy(draw: st.DrawFn) -> str:
    """Generate a valid six-digit hex color string e.g. '#1A2B3C'."""
    chars = draw(st.lists(_HEX_CHAR, min_size=6, max_size=6))
    return "#" + "".join(chars)


@st.composite
def style_profile_strategy(draw: st.DrawFn) -> StyleProfile:
    """Generate a valid :class:`StyleProfile` instance with all nested models."""
    doc_id = str(draw(st.uuids()))
    version = draw(st.integers(min_value=1, max_value=100))
    created_at = draw(datetime_strategy())
    channel_url = draw(st.text(min_size=1, max_size=200))

    sentiment_polarity = draw(
        st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    narration_tone = NarrationTone(sentiment_polarity=sentiment_polarity)

    avg_wpm = draw(st.integers(min_value=50, max_value=300))
    avg_sentence_len = draw(
        st.floats(min_value=5.0, max_value=30.0, allow_nan=False, allow_infinity=False)
    )
    pacing = Pacing(
        avg_words_per_minute=avg_wpm,
        avg_sentence_length_words=avg_sentence_len,
    )

    intro_present = draw(st.booleans())
    hook_present = draw(st.booleans())
    body_segment_count_avg = draw(
        st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    cta_present = draw(st.booleans())
    segment_structure = SegmentStructure(
        intro_present=intro_present,
        hook_present=hook_present,
        body_segment_count_avg=body_segment_count_avg,
        cta_present=cta_present,
    )

    visual_style = VisualStyle(composition_patterns=draw(st.lists(st.text(min_size=1, max_size=50), max_size=5)))

    dominant_colors = draw(st.lists(_hex_color_strategy(), min_size=1, max_size=5))
    text_overlay_position = draw(st.text(min_size=1, max_size=50))
    subject_framing = draw(st.text(min_size=1, max_size=50))
    sample_count = draw(st.integers(min_value=1, max_value=100))
    lookback_days = draw(st.integers(min_value=30, max_value=365))
    thumbnail_composition = ThumbnailComposition(
        dominant_colors=dominant_colors,
        text_overlay_position=text_overlay_position,
        subject_framing=subject_framing,
        sample_count=sample_count,
        lookback_days=lookback_days,
    )

    rhetorical_patterns = draw(st.lists(st.text(min_size=1, max_size=80), max_size=10))

    return StyleProfile(
        doc_id=doc_id,
        version=version,
        created_at=created_at,
        channel_url=channel_url,
        narration_tone=narration_tone,
        pacing=pacing,
        segment_structure=segment_structure,
        visual_style=visual_style,
        thumbnail_composition=thumbnail_composition,
        rhetorical_patterns=rhetorical_patterns,
    )


# ---------------------------------------------------------------------------
# Strategy 3 — MetadataPackage
# ---------------------------------------------------------------------------

# Safe printable ASCII text that avoids Pydantic/re edge cases
_WORD_CHARS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    min_size=2,
    max_size=10,
)


@st.composite
def _multi_word_tag_strategy(draw: st.DrawFn, min_words: int = 2, max_words: int = 5) -> str:
    """Generate a tag composed of 2–5 space-separated words."""
    word_count = draw(st.integers(min_value=min_words, max_value=max_words))
    words = [draw(_WORD_CHARS) for _ in range(word_count)]
    return " ".join(words)


@st.composite
def _hashtag_strategy(draw: st.DrawFn) -> str:
    """Generate a hashtag matching ``^#[^\\s]{2,30}$``."""
    body = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
            min_size=2,
            max_size=30,
        )
    )
    return "#" + body


@st.composite
def _chapter_strategy(draw: st.DrawFn) -> Chapter:
    """Generate a valid :class:`Chapter` with a MM:SS timestamp."""
    minutes = draw(st.integers(min_value=0, max_value=59))
    seconds = draw(st.integers(min_value=0, max_value=59))
    timestamp = f"{minutes:02d}:{seconds:02d}"
    label = draw(st.text(min_size=1, max_size=80))
    return Chapter(timestamp=timestamp, label=label)


@st.composite
def metadata_package_strategy(draw: st.DrawFn) -> MetadataPackage:
    """Generate a valid :class:`MetadataPackage` that passes all validators."""
    video_id = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_",
            min_size=1,
            max_size=64,
        )
    )

    # Title: ≤ 60 chars
    title = draw(st.text(min_size=1, max_size=60))

    # Description: build by repeating word units to hit 200–500 word count
    target_words = draw(st.integers(min_value=200, max_value=500))
    base_word = draw(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz",
            min_size=3,
            max_size=8,
        )
    )
    # Produce exactly target_words words by repeating the base word.
    words = [base_word] * target_words
    description = " ".join(words)

    # Tags: 10–15, each 2–5 words
    tag_count = draw(st.integers(min_value=10, max_value=15))
    tags = [draw(_multi_word_tag_strategy()) for _ in range(tag_count)]

    # Hashtags: 3–5, each matching ^#[^\s]{2,30}$
    hashtag_count = draw(st.integers(min_value=3, max_value=5))
    hashtags = [draw(_hashtag_strategy()) for _ in range(hashtag_count)]

    # Chapters: 0–10
    chapter_count = draw(st.integers(min_value=0, max_value=10))
    chapters = [draw(_chapter_strategy()) for _ in range(chapter_count)]

    primary_keyword = draw(st.text(min_size=1, max_size=50))
    generated_at = draw(datetime_strategy())

    return MetadataPackage(
        video_id=video_id,
        title=title,
        description=description,
        tags=tags,
        hashtags=hashtags,
        chapters=chapters,
        primary_keyword=primary_keyword,
        generated_at=generated_at,
    )


# ---------------------------------------------------------------------------
# Strategy 4 — VideoRecord
# ---------------------------------------------------------------------------

_VIDEO_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"


@st.composite
def video_record_strategy(draw: st.DrawFn) -> VideoRecord:
    """Generate a valid :class:`VideoRecord` instance."""
    video_id = draw(
        st.text(alphabet=_VIDEO_ID_ALPHABET, min_size=1, max_size=128)
    )
    batch_id: str | None = draw(st.one_of(st.none(), st.text(min_size=1, max_size=64)))
    title = draw(st.text(min_size=1, max_size=200))
    topic = draw(topic_entry_strategy())
    status = draw(st.sampled_from(list(PipelineStatus)))
    scheduled_publish_datetime: datetime | None = draw(
        st.one_of(st.none(), datetime_strategy())
    )
    style_profile_doc_id = draw(st.text(min_size=1, max_size=64))
    pipeline_run_timestamp = draw(datetime_strategy())
    created_at = draw(datetime_strategy())
    updated_at = draw(datetime_strategy())

    return VideoRecord(
        video_id=video_id,
        batch_id=batch_id,
        title=title,
        topic=topic,
        status=status,
        scheduled_publish_datetime=scheduled_publish_datetime,
        style_profile_doc_id=style_profile_doc_id,
        asset_links=AssetLinks(),
        pipeline_run_timestamp=pipeline_run_timestamp,
        created_at=created_at,
        updated_at=updated_at,
    )


# ---------------------------------------------------------------------------
# Strategy 6 — Platform caption (MetadataPackage, Platform) tuples
# ---------------------------------------------------------------------------


@st.composite
def platform_caption_strategy(draw: st.DrawFn) -> tuple[MetadataPackage, Platform]:
    """Generate a (MetadataPackage, Platform) tuple for cross-post caption testing."""
    metadata = draw(metadata_package_strategy())
    platform = draw(st.sampled_from(list(Platform)))
    return metadata, platform
