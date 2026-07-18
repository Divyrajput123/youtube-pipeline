"""Spot-check tests that verify each Hypothesis strategy produces valid model instances.

Five example-based tests confirm that the strategies can draw at least one valid object
without raising a Pydantic ``ValidationError``.
"""

from __future__ import annotations

import re
from datetime import timezone

from hypothesis import given, settings

from tests.property.strategies import (
    datetime_strategy,
    metadata_package_strategy,
    platform_caption_strategy,
    style_profile_strategy,
    topic_entry_strategy,
    video_record_strategy,
)
from pipeline.models import (
    MetadataPackage,
    Platform,
    StyleProfile,
    TopicEntry,
    VideoRecord,
)


# ---------------------------------------------------------------------------
# Test 1 — topic_entry_strategy produces a valid TopicEntry
# ---------------------------------------------------------------------------


@given(topic_entry_strategy())
@settings(max_examples=25)
def test_topic_entry_strategy_produces_valid_instances(entry: TopicEntry) -> None:
    """topic_entry_strategy must produce TopicEntry instances that satisfy all field constraints."""
    assert 1 <= len(entry.title) <= 200
    assert 0.0 <= entry.composite_score <= 1.0
    assert entry.recency_hours >= 0.0
    assert entry.source_query_timestamp.tzinfo is not None
    assert entry.search_volume_signal >= 0.0
    assert 0 <= len(entry.relevance_tags_matched) <= 5


# ---------------------------------------------------------------------------
# Test 2 — style_profile_strategy produces a valid StyleProfile
# ---------------------------------------------------------------------------


@given(style_profile_strategy())
@settings(max_examples=25)
def test_style_profile_strategy_produces_valid_instances(profile: StyleProfile) -> None:
    """style_profile_strategy must produce StyleProfile instances with all nested models valid."""
    assert -1.0 <= profile.narration_tone.sentiment_polarity <= 1.0
    assert 50 <= profile.pacing.avg_words_per_minute <= 300
    assert 5.0 <= profile.pacing.avg_sentence_length_words <= 30.0
    assert 1.0 <= profile.segment_structure.body_segment_count_avg <= 10.0
    colors = profile.thumbnail_composition.dominant_colors
    assert 1 <= len(colors) <= 5
    hex_re = re.compile(r"^#[0-9A-Fa-f]{6}$")
    for color in colors:
        assert hex_re.match(color), f"Invalid hex color: {color}"
    assert 1 <= profile.thumbnail_composition.sample_count <= 100
    assert 30 <= profile.thumbnail_composition.lookback_days <= 365
    assert profile.created_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Test 3 — metadata_package_strategy produces a valid MetadataPackage
# ---------------------------------------------------------------------------


@given(metadata_package_strategy())
@settings(max_examples=25)
def test_metadata_package_strategy_produces_valid_instances(pkg: MetadataPackage) -> None:
    """metadata_package_strategy must produce MetadataPackage instances that pass all validators."""
    assert len(pkg.title) <= 60
    word_count = len(pkg.description.split())
    assert 200 <= word_count <= 500, f"description word count {word_count} out of range"
    assert 10 <= len(pkg.tags) <= 15
    for tag in pkg.tags:
        words = tag.split()
        assert 2 <= len(words) <= 5, f"tag '{tag}' has {len(words)} words"
    assert 3 <= len(pkg.hashtags) <= 5
    hashtag_re = re.compile(r"^#[^\s]{2,30}$")
    for ht in pkg.hashtags:
        assert hashtag_re.match(ht), f"Invalid hashtag: {ht}"
    assert 0 <= len(pkg.chapters) <= 10
    ts_re = re.compile(r"^\d{2}:\d{2}$")
    for chapter in pkg.chapters:
        assert ts_re.match(chapter.timestamp), f"Invalid chapter timestamp: {chapter.timestamp}"
    assert pkg.generated_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Test 4 — video_record_strategy produces a valid VideoRecord
# ---------------------------------------------------------------------------


@given(video_record_strategy())
@settings(max_examples=25)
def test_video_record_strategy_produces_valid_instances(record: VideoRecord) -> None:
    """video_record_strategy must produce VideoRecord instances with valid video_id and datetimes."""
    video_id_re = re.compile(r"^[a-zA-Z0-9\-_]{1,128}$")
    assert video_id_re.match(record.video_id), f"Invalid video_id: {record.video_id}"
    assert record.pipeline_run_timestamp.tzinfo is not None
    assert record.created_at.tzinfo is not None
    assert record.updated_at.tzinfo is not None
    if record.scheduled_publish_datetime is not None:
        assert record.scheduled_publish_datetime.tzinfo is not None


# ---------------------------------------------------------------------------
# Test 5 — platform_caption_strategy produces valid (MetadataPackage, Platform) tuples
# ---------------------------------------------------------------------------


@given(platform_caption_strategy())
@settings(max_examples=25)
def test_platform_caption_strategy_produces_valid_tuples(
    pair: tuple[MetadataPackage, Platform],
) -> None:
    """platform_caption_strategy must produce (MetadataPackage, Platform) tuples."""
    metadata, platform = pair
    assert isinstance(metadata, MetadataPackage)
    assert isinstance(platform, Platform)
    assert platform in list(Platform)
