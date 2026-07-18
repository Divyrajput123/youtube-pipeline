"""Unit tests for pipeline/topic_researcher/__init__.py

Covers Tasks 7.1 and 7.2:
- Composite scoring and sort order
- Partial-results path (1–4 topics → PartialResultsError, no store)
- Zero-results-after-retries path (Notifier alerted, TopicResearchError raised)
- Batch size validation (max 50)
- Deduplication against excluded_titles (case-insensitive)
- JSON persistence to Asset_Store under RESEARCH subfolder
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.asset_store import Asset_Store
from pipeline.models import SubFolder
from pipeline.notifier import Notifier
from pipeline.topic_researcher import (
    PartialResultsError,
    PerplexityMCPClient,
    RawTopicResult,
    TavilyMCPClient,
    Topic_Researcher,
    TopicResearchError,
    _compute_relevance,
    _min_max_normalize,
)


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------


def make_raw(
    title: str = "AI topic",
    search_volume: float = 1.0,
    hours_ago: float = 10.0,
) -> RawTopicResult:
    return RawTopicResult(
        title=title,
        search_volume_signal=search_volume,
        first_seen_hours_ago=hours_ago,
    )


def make_five_raw(base_volume: float = 10.0) -> list[RawTopicResult]:
    """Return 5 distinct RawTopicResult objects."""
    return [
        make_raw(f"Topic {i}", search_volume=base_volume + i, hours_ago=float(i))
        for i in range(5)
    ]


def make_researcher(
    raw_results: list[RawTopicResult] | None = None,
    *,
    side_effect: Exception | None = None,
    provider: str = "perplexity",
) -> tuple[Topic_Researcher, AsyncMock, AsyncMock]:
    """Build a Topic_Researcher with mocked SearchClient, Asset_Store, and Notifier.

    Returns:
        (researcher, mock_query_trending, mock_store_write)
    """
    mock_client = MagicMock()
    if side_effect is not None:
        mock_client.query_trending = AsyncMock(side_effect=side_effect)
    else:
        mock_client.query_trending = AsyncMock(return_value=raw_results or [])

    mock_store = MagicMock(spec=Asset_Store)
    mock_store.write = AsyncMock(return_value="https://drive.google.com/fake-url")

    mock_notifier = MagicMock(spec=Notifier)
    mock_notifier.send_failure_alert = MagicMock()

    researcher = Topic_Researcher(
        search_client=mock_client,
        asset_store=mock_store,
        notifier=mock_notifier,
        provider=provider,  # type: ignore[arg-type]
    )
    return researcher, mock_client.query_trending, mock_store.write


# ---------------------------------------------------------------------------
# Unit tests: normalisation helper
# ---------------------------------------------------------------------------


class TestMinMaxNormalize:
    def test_all_same_values_returns_half(self) -> None:
        result = _min_max_normalize([5.0, 5.0, 5.0])
        assert result == [0.5, 0.5, 0.5]

    def test_distinct_values_min_is_zero_max_is_one(self) -> None:
        result = _min_max_normalize([0.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)

    def test_empty_list_returns_empty(self) -> None:
        assert _min_max_normalize([]) == []

    def test_single_value_returns_half(self) -> None:
        assert _min_max_normalize([42.0]) == [0.5]

    def test_output_bounds(self) -> None:
        values = [1.0, 3.0, 7.0, 100.0]
        result = _min_max_normalize(values)
        assert all(0.0 <= v <= 1.0 for v in result)


# ---------------------------------------------------------------------------
# Unit tests: relevance helper
# ---------------------------------------------------------------------------


class TestComputeRelevance:
    def test_matches_machine_learning(self) -> None:
        score, tags = _compute_relevance("Advances in machine learning 2025")
        assert score == 1.0
        assert "machine learning" in tags

    def test_matches_generative_ai_case_insensitive(self) -> None:
        score, tags = _compute_relevance("GENERATIVE AI is changing everything")
        assert score == 1.0
        assert "generative AI" in tags

    def test_no_match_returns_zero(self) -> None:
        score, tags = _compute_relevance("The weather forecast for tomorrow")
        assert score == 0.0
        assert tags == []

    def test_multiple_tags_matched(self) -> None:
        score, tags = _compute_relevance(
            "Large language model safety and neural network research"
        )
        assert score == 1.0
        assert len(tags) >= 2

    def test_empty_title_returns_zero(self) -> None:
        score, tags = _compute_relevance("")
        assert score == 0.0
        assert tags == []


# ---------------------------------------------------------------------------
# Unit tests: composite scoring and sort order (Task 7.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestScoringAndSortOrder:
    async def test_results_sorted_descending_by_composite_score(self) -> None:
        raw = make_five_raw()
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5, excluded_titles=[], run_id="run-001"
        )

        scores = [e.composite_score for e in results]
        assert scores == sorted(scores, reverse=True)

    async def test_composite_score_in_valid_range(self) -> None:
        raw = make_five_raw()
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5, excluded_titles=[], run_id="run-002"
        )

        for entry in results:
            assert 0.0 <= entry.composite_score <= 1.0

    async def test_relevance_tag_matched_boosts_score(self) -> None:
        """A topic containing a relevance tag should have score contribution from relevance."""
        raw = [
            make_raw("Advances in machine learning", search_volume=0.0, hours_ago=0.0),
            make_raw("Unrelated cooking topic", search_volume=0.0, hours_ago=0.0),
            make_raw("Generic tech news", search_volume=0.0, hours_ago=0.0),
            make_raw("Cloud computing trends", search_volume=0.0, hours_ago=0.0),
            make_raw("Startup funding round", search_volume=0.0, hours_ago=0.0),
        ]
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=1, excluded_titles=[], run_id="run-003"
        )

        ml_entry = next(e for e in results if "machine learning" in e.title.lower())
        # Relevance contributes 1.0/3 = 0.333; others with identical sv+recency have 0
        assert ml_entry.composite_score > 0.0
        assert ml_entry.relevance_tags_matched == ["machine learning"]

    async def test_source_query_timestamp_truncated_to_minute(self) -> None:
        raw = make_five_raw()
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5, excluded_titles=[], run_id="run-004"
        )

        for entry in results:
            ts = entry.source_query_timestamp
            assert ts.second == 0
            assert ts.microsecond == 0
            assert ts.tzinfo is not None  # must be timezone-aware

    async def test_all_entries_share_same_timestamp(self) -> None:
        raw = make_five_raw()
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5, excluded_titles=[], run_id="run-005"
        )

        timestamps = {e.source_query_timestamp for e in results}
        assert len(timestamps) == 1, "All entries must share the same source_query_timestamp"

    async def test_recency_hours_matches_raw_first_seen(self) -> None:
        raw = [make_raw(f"Topic {i}", hours_ago=float(i * 5)) for i in range(5)]
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5, excluded_titles=[], run_id="run-006"
        )

        raw_map = {r.title: r.first_seen_hours_ago for r in raw}
        for entry in results:
            assert entry.recency_hours == raw_map[entry.title]


# ---------------------------------------------------------------------------
# Unit tests: JSON persistence (Task 7.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestJSONPersistence:
    async def test_write_called_once_on_success(self) -> None:
        raw = make_five_raw()
        researcher, _, mock_write = make_researcher(raw_results=raw)

        await researcher.research(batch_size=5, excluded_titles=[], run_id="abc123")

        mock_write.assert_called_once()

    async def test_write_uses_research_subfolder(self) -> None:
        raw = make_five_raw()
        researcher, _, mock_write = make_researcher(raw_results=raw)

        await researcher.research(batch_size=5, excluded_titles=[], run_id="abc123")

        _, kwargs = mock_write.call_args
        assert kwargs.get("subfolder") == SubFolder.RESEARCH

    async def test_write_filename_contains_run_id(self) -> None:
        raw = make_five_raw()
        researcher, _, mock_write = make_researcher(raw_results=raw)

        await researcher.research(batch_size=5, excluded_titles=[], run_id="myrun")

        _, kwargs = mock_write.call_args
        assert "myrun" in kwargs.get("filename", "")

    async def test_write_video_id_is_run_prefixed(self) -> None:
        raw = make_five_raw()
        researcher, _, mock_write = make_researcher(raw_results=raw)

        await researcher.research(batch_size=5, excluded_titles=[], run_id="run42")

        _, kwargs = mock_write.call_args
        assert kwargs.get("video_id") == "run-run42"

    async def test_stored_json_contains_required_fields(self) -> None:
        raw = make_five_raw()
        researcher, _, mock_write = make_researcher(raw_results=raw)

        await researcher.research(batch_size=5, excluded_titles=[], run_id="check")

        _, kwargs = mock_write.call_args
        content_bytes: bytes = kwargs.get("content", b"[]")
        records = json.loads(content_bytes.decode("utf-8"))

        assert len(records) >= 5
        for rec in records:
            assert "title" in rec
            assert "composite_score" in rec
            assert "recency_hours" in rec
            assert "source_query_timestamp" in rec

    async def test_stored_json_is_valid_utf8_json(self) -> None:
        raw = make_five_raw()
        researcher, _, mock_write = make_researcher(raw_results=raw)

        await researcher.research(batch_size=5, excluded_titles=[], run_id="utf8test")

        _, kwargs = mock_write.call_args
        content_bytes: bytes = kwargs.get("content", b"")
        # Should not raise
        parsed = json.loads(content_bytes.decode("utf-8"))
        assert isinstance(parsed, list)


# ---------------------------------------------------------------------------
# Unit tests: partial results error (Task 7.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPartialResults:
    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    async def test_partial_results_raises_immediately(self, count: int) -> None:
        raw = [make_raw(f"Topic {i}") for i in range(count)]
        researcher, mock_query, mock_write = make_researcher(raw_results=raw)

        with pytest.raises(PartialResultsError) as exc_info:
            await researcher.research(
                batch_size=1, excluded_titles=[], run_id="partial"
            )

        assert exc_info.value.count == count

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    async def test_partial_results_does_not_store_json(self, count: int) -> None:
        raw = [make_raw(f"Topic {i}") for i in range(count)]
        researcher, _, mock_write = make_researcher(raw_results=raw)

        with pytest.raises(PartialResultsError):
            await researcher.research(
                batch_size=1, excluded_titles=[], run_id="partial-nostore"
            )

        mock_write.assert_not_called()

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    async def test_partial_results_does_not_retry(self, count: int) -> None:
        """PartialResultsError must be raised on the first partial response — no retry."""
        raw = [make_raw(f"Topic {i}") for i in range(count)]
        researcher, mock_query, _ = make_researcher(raw_results=raw)

        with pytest.raises(PartialResultsError):
            await researcher.research(
                batch_size=1, excluded_titles=[], run_id="partial-noretry"
            )

        # Only one query attempt should have been made
        assert mock_query.call_count == 1


# ---------------------------------------------------------------------------
# Unit tests: zero-results path (Task 7.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestZeroResults:
    async def test_zero_results_notifies_and_raises(self) -> None:
        researcher, _, _ = make_researcher(raw_results=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TopicResearchError):
                await researcher.research(
                    batch_size=1, excluded_titles=[], run_id="zero"
                )

        researcher._notifier.send_failure_alert.assert_called_once()

    async def test_zero_results_failure_alert_contains_run_id(self) -> None:
        researcher, _, _ = make_researcher(raw_results=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TopicResearchError):
                await researcher.research(
                    batch_size=1, excluded_titles=[], run_id="special-run-99"
                )

        call_kwargs = researcher._notifier.send_failure_alert.call_args[1]
        assert "special-run-99" in call_kwargs.get("error_message", "")

    async def test_zero_results_retries_three_times(self) -> None:
        researcher, mock_query, _ = make_researcher(raw_results=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TopicResearchError):
                await researcher.research(
                    batch_size=1, excluded_titles=[], run_id="retry3"
                )

        assert mock_query.call_count == 3

    async def test_zero_results_does_not_store_json(self) -> None:
        researcher, _, mock_write = make_researcher(raw_results=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TopicResearchError):
                await researcher.research(
                    batch_size=1, excluded_titles=[], run_id="zero-nostore"
                )

        mock_write.assert_not_called()

    async def test_query_exception_treated_as_zero_result(self) -> None:
        """Provider exceptions should be treated like empty results and retried."""
        researcher, mock_query, _ = make_researcher(
            side_effect=RuntimeError("network error")
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TopicResearchError):
                await researcher.research(
                    batch_size=1, excluded_titles=[], run_id="exc-zero"
                )

        assert mock_query.call_count == 3


# ---------------------------------------------------------------------------
# Unit tests: deduplication (Task 7.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeduplication:
    async def test_excluded_titles_removed_case_insensitive(self) -> None:
        # Provide 7 raw results so that after excluding 2 we still have ≥ 5 unique entries
        raw = [
            make_raw("Machine Learning Trends"),
            make_raw("Deep Learning Advances"),
            make_raw("AI in Healthcare"),
            make_raw("Quantum Computing"),
            make_raw("Neural Network Breakthroughs"),
            make_raw("Robotics Update"),
            make_raw("Generative AI News"),
        ]
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5,
            excluded_titles=["machine learning trends", "DEEP LEARNING ADVANCES"],
            run_id="dedup-01",
        )

        titles = [e.title for e in results]
        assert "Machine Learning Trends" not in titles
        assert "Deep Learning Advances" not in titles

    async def test_intra_batch_duplicates_removed(self) -> None:
        """Same title appearing twice in raw results → only first occurrence kept."""
        # 6 raw entries but one is a duplicate; 5 unique titles survive deduplication.
        raw = [
            make_raw("Duplicate Topic"),
            make_raw("Duplicate Topic"),  # exact duplicate → removed
            make_raw("Another Topic A"),
            make_raw("Another Topic B"),
            make_raw("Another Topic C"),
            make_raw("Another Topic D"),
        ]
        researcher, _, _ = make_researcher(raw_results=raw)

        results = await researcher.research(
            batch_size=5, excluded_titles=[], run_id="dedup-02"
        )

        titles = [e.title for e in results]
        assert titles.count("Duplicate Topic") == 1
        assert len(titles) == 5

    async def test_insufficient_topics_after_dedup_raises(self) -> None:
        """If deduplication removes enough entries that < max(5, batch_size) remain,
        a TopicResearchError must be raised (and no JSON stored)."""
        raw = [make_raw(f"Topic {i}") for i in range(6)]
        # Exclude all but 2 of them
        excluded = [f"Topic {i}" for i in range(4)]

        researcher, _, mock_write = make_researcher(raw_results=raw)

        with pytest.raises(TopicResearchError):
            await researcher.research(
                batch_size=5, excluded_titles=excluded, run_id="dedup-insufficient"
            )

        mock_write.assert_not_called()


# ---------------------------------------------------------------------------
# Unit tests: batch size validation (Task 7.2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBatchSizeValidation:
    async def test_batch_size_over_50_raises(self) -> None:
        researcher, _, _ = make_researcher()

        with pytest.raises(TopicResearchError, match="50"):
            await researcher.research(
                batch_size=51, excluded_titles=[], run_id="big-batch"
            )

    async def test_batch_size_exactly_50_accepted(self) -> None:
        """batch_size=50 is the maximum allowed; must not raise immediately."""
        raw = [make_raw(f"Topic {i}") for i in range(50)]
        researcher, _, _ = make_researcher(raw_results=raw)

        # Should not raise a TopicResearchError about batch_size
        results = await researcher.research(
            batch_size=50, excluded_titles=[], run_id="batch50"
        )
        assert len(results) == 50


# ---------------------------------------------------------------------------
# Unit tests: stubs raise NotImplementedError
# ---------------------------------------------------------------------------


class TestClientStubs:
    @pytest.mark.asyncio
    async def test_perplexity_stub_raises(self) -> None:
        client = PerplexityMCPClient()
        with pytest.raises(NotImplementedError):
            await client.query_trending("query", 72)

    @pytest.mark.asyncio
    async def test_tavily_stub_raises(self) -> None:
        client = TavilyMCPClient()
        with pytest.raises(NotImplementedError):
            await client.query_trending("query", 72)
