"""Cold-open hook: the first seconds are the payoff, not a setup."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from pipeline.models import (
    NarrationTone,
    Pacing,
    SegmentStructure,
    StyleProfile,
    ThumbnailComposition,
    TopicEntry,
    VisualStyle,
)
from pipeline.script_writer import _build_generation_prompt
from pipeline.visual_generator import _Segment, _generate_video_prompt_with_claude


def _topic() -> TopicEntry:
    return TopicEntry(
        title="What happens when the punch actually lands",
        composite_score=0.9,
        recency_hours=1.0,
        source_query_timestamp=datetime.now(tz=timezone.utc),
        search_volume_signal=100.0,
        relevance_tags_matched=["superhero"],
    )


def _profile(wpm: int = 120) -> StyleProfile:
    return StyleProfile(
        doc_id="sp-1",
        version=1,
        created_at=datetime.now(tz=timezone.utc),
        channel_url="https://youtube.com/@test",
        narration_tone=NarrationTone(sentiment_polarity=0.4),
        pacing=Pacing(avg_words_per_minute=wpm, avg_sentence_length_words=22.0),
        segment_structure=SegmentStructure(
            intro_present=True,
            hook_present=True,
            body_segment_count_avg=4.0,
            cta_present=True,
        ),
        visual_style=VisualStyle(composition_patterns=["wide"]),
        thumbnail_composition=ThumbnailComposition(
            dominant_colors=["#000000"],
            text_overlay_position="bottom-left",
            subject_framing="center",
            sample_count=10,
            lookback_days=30,
        ),
        rhetorical_patterns=["storytelling"],
    )


def test_cinematic_hook_is_capped_and_ignores_a_slow_reference() -> None:
    prompt = _build_generation_prompt(_topic(), _profile(), script_style="cinematic_storytelling")

    assert "HARD CAP 25 words" in prompt
    assert "[pause] and [slow] are forbidden in the hook" in prompt
    assert "IGNORE the reference channel" in prompt
    assert "Picture this" in prompt  # listed as a banned opening, not the model to copy


def test_kids_script_keeps_the_measured_pace() -> None:
    prompt = _build_generation_prompt(_topic(), _profile(), script_style="kids_rhyming")

    assert "HARD CAP 25 words" not in prompt
    assert "IGNORE the reference channel" not in prompt
    assert "120 words/minute" in prompt


@pytest.mark.asyncio
async def test_opening_video_prompt_starts_on_the_hit() -> None:
    client = AsyncMock()
    client.complete = AsyncMock(return_value="A shockwave tears the street apart.")

    with patch("pipeline.visual_generator.build_claude_client", return_value=client):
        result = await _generate_video_prompt_with_claude(
            segment=_Segment(title="HOOK", body="One punch levels six blocks."),
            clip_num=0,
            total_clips=4,
            scene_state={"global_clip_num": 0, "total_global_clips": 4},
            script_topic="a city-level punch",
        )

    sent = client.complete.await_args.args[0]
    assert "COLD OPEN" in sent
    assert "No establishing shot" in sent
    assert "no slow motion" in result.lower()
