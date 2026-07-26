"""Topic_Researcher subsystem — Perplexity/Tavily MCP + composite scoring.

Queries a configurable search provider (Perplexity or Tavily) for trending AI topics
published within the past 72 hours, computes a composite score from three normalised
signals (search volume, recency, relevance), deduplicates against previously-used
titles, and persists the ranked list to Asset_Store.

Design reference: §3 Topic_Researcher
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from pipeline.asset_store import Asset_Store
from pipeline.config import is_production_mode, require_production_config
from pipeline.models import SubFolder, TopicEntry
from pipeline.notifier import Notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Retry policy: 3 attempts, 30-second fixed interval between each attempt.
_MAX_ATTEMPTS = 3
_RETRY_INTERVAL_SECONDS = 30.0

# How far back (in hours) to query for trending topics.
_TRENDING_HOURS_BACK = 2160  # 3 months — wide window for evergreen + recent content

# The query string sent to the search provider (used for logging only — actual
# prompt is built inside query_trending).
_SEARCH_QUERY = (
    "trending superhero topics Marvel DC anime fights crossovers past 72 hours"
)

# Maximum batch size allowed by the design spec.
_MAX_BATCH_SIZE = 50

# Minimum number of topics that must be returned for a valid result.
_MIN_VALID_COUNT = 5

# Keyword tags used for the binary relevance signal — superhero content focus.
_RELEVANCE_TAGS: list[str] = [
    "marvel",
    "dc",
    "superhero",
    "spider-man",
    "batman",
    "superman",
    "avengers",
    "thor",
    "iron man",
    "goku",
    "naruto",
    "anime",
    "fight",
    "vs",
    "crossover",
    "origin",
    "power",
    "villain",
]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TopicResearchError(Exception):
    """Raised when Topic_Researcher cannot produce a valid ranked topic list."""


class PartialResultsError(TopicResearchError):
    """Raised immediately when 1–4 topics are returned before retries are exhausted.

    The caller must NOT store any JSON when this error is raised.
    """

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            f"Partial results: received {count} topic(s) (expected ≥ {_MIN_VALID_COUNT}). "
            "Halting without storing."
        )


# ---------------------------------------------------------------------------
# SearchClient Protocol and RawTopicResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class RawTopicResult:
    """A single raw topic entry returned by a search provider.

    Attributes:
        title: Human-readable topic title.
        search_volume_signal: Numeric proxy for search interest (raw, unnormalised).
        first_seen_hours_ago: How many hours ago this topic first appeared.
    """

    title: str
    search_volume_signal: float
    first_seen_hours_ago: float


@runtime_checkable
class SearchClient(Protocol):
    """Protocol for search provider clients used by Topic_Researcher.

    Any object implementing this single method is compatible.
    """

    async def query_trending(
        self,
        query: str,
        hours_back: int,
        excluded_titles: list[str] | None = None,
    ) -> list[RawTopicResult]:
        """Query the provider for trending topics.

        Args:
            query: Natural-language search query string.
            hours_back: Only topics that surfaced within this many hours should
                be returned.
            excluded_titles: Topics already used — the provider should avoid
                returning these or semantically similar ones.

        Returns:
            A list of :class:`RawTopicResult` entries (may be empty).
        """
        ...


# ---------------------------------------------------------------------------
# Concrete search client stubs
# ---------------------------------------------------------------------------


class PerplexityMCPClient:
    """Perplexity API search client for local development.

    Uses httpx to call Perplexity's API directly rather than going through MCP.
    Reads API key from PERPLEXITY_API_KEY environment variable.
    """

    def __init__(self) -> None:
        import os  # noqa: PLC0415
        self._api_key = os.environ.get("PERPLEXITY_API_KEY", "")
        if not self._api_key:
            logger.warning("PERPLEXITY_API_KEY not set — search will fail")

    async def query_trending(
        self,
        query: str,
        hours_back: int,
        excluded_titles: list[str] | None = None,
    ) -> list[RawTopicResult]:
        """Call Perplexity API to get trending topics.

        For local development, returns placeholder data if API key is missing or invalid.
        """
        import httpx  # noqa: PLC0415
        import re  # noqa: PLC0415
        from datetime import datetime, timezone  # noqa: PLC0415

        # If no API key, return placeholder data immediately (only in development mode)
        if not self._api_key or self._api_key.startswith("pplx-REPLACE"):
            if is_production_mode():
                raise ValueError(
                    "Production mode requires a valid Perplexity API key (PERPLEXITY_API_KEY). "
                    "Set PIPELINE_MODE=development to use placeholder topics."
                )
            logger.warning("Perplexity API key not configured — returning placeholder topics for local dev")
            return self._get_placeholder_topics()

        # Build exclusion block to inject into the prompt
        exclusion_block = ""
        if excluded_titles:
            # Send up to 50 most recent to keep prompt size reasonable
            titles_to_exclude = excluded_titles[:50]
            exclusion_list = "\n".join(f"- {t}" for t in titles_to_exclude)
            exclusion_block = (
                f"\n\nIMPORTANT: Do NOT suggest any of the following topics or anything "
                f"semantically similar (same characters, same matchup, same concept):\n"
                f"{exclusion_list}\n"
                f"These have already been covered. Suggest completely different topics."
            )

        # Build a prompt that asks for structured topic data
        prompt = (
            "List 10 popular superhero video topics suitable for a YouTube channel. "
            "Include topics about: Marvel, DC Comics, anime heroes (Dragon Ball, Naruto, One Piece), "
            "superhero fights and power comparisons, character vs character battles, "
            "origin stories, superhero movie/show analysis, comic book events, and hero rankings. "
            "These can be evergreen popular topics, not necessarily trending today. "
            "Format: numbered list 1-10, one topic per line, short catchy title (5-10 words max). "
            "Example topics: 'Thor vs Superman: Who Would Win', 'Goku vs Saitama Power Comparison', "
            "'Black Panther Powers Explained', 'Spider-Man vs Batman Fight Analysis', "
            "'Top 10 Strongest Marvel Heroes Ranked'. "
            "Write every title entirely in English using standard Latin characters. "
            "Do not use Chinese, Japanese, Korean, Cyrillic, or any other non-Latin characters. "
            "Output ONLY the numbered list, no explanations, no disclaimers."
            + exclusion_block
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "sonar",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a research assistant that identifies trending AI topics.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Parse the response — extract all topic titles (not just first 10 lines)
                results: list[RawTopicResult] = []
                import re as _re  # noqa: PLC0415

                lines = [l.strip() for l in content.split("\n") if l.strip()]
                for i, line in enumerate(lines):  # iterate ALL lines, not just first 10
                    # Strip numbering like "1.", "1)", "- ", "* "
                    cleaned = _re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", line).strip()
                    # Skip empty or very long lines
                    if not cleaned or len(cleaned) > 200:
                        continue
                    # Skip lines that look like refusals, headers, or boilerplate
                    if any(w in cleaned.lower() for w in [
                        "cannot provide", "unable to", "sorry,", "don't have",
                        "search results do not", "outdated", "unrelated",
                        "here are the", "following topics", "based on my",
                        "note:", "disclaimer:", "i don't have", "i cannot",
                        "unfortunately,", "as of my", "as of today",
                        "provided search", "generic youtube",
                    ]):
                        continue
                    results.append(
                        RawTopicResult(
                            title=cleaned,
                            search_volume_signal=float(90 - len(results) * 5),
                            first_seen_hours_ago=float(len(results) * 6),
                        )
                    )
                    if len(results) >= 10:
                        break  # stop after 10 good topics

                logger.info(f"PerplexityMCPClient: extracted {len(results)} topics from API response")
                return results

        except httpx.HTTPStatusError as exc:
            logger.error(f"Perplexity API HTTP error: {exc.response.status_code} — {exc.response.text[:200]}")
            raise Exception(
                f"Perplexity API error: {exc.response.status_code}. "
                f"Response: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Perplexity API call failed: {exc}")
            if is_production_mode():
                raise Exception(
                    f"Perplexity API call failed in production mode: {exc}. "
                    "Set PIPELINE_MODE=development to use placeholder topics."
                ) from exc
            logger.warning(f"Perplexity API call failed: {exc} - returning placeholder topics")
            return self._get_placeholder_topics()

    def _get_placeholder_topics(self) -> list[RawTopicResult]:
        """Return placeholder superhero trending topics for local development."""
        return [
            RawTopicResult(
                title="Spider-Man vs Batman: Who Would Win in a Real Fight",
                search_volume_signal=95.0,
                first_seen_hours_ago=2.0,
            ),
            RawTopicResult(
                title="Avengers vs Justice League: Ultimate Crossover Battle",
                search_volume_signal=88.0,
                first_seen_hours_ago=5.0,
            ),
            RawTopicResult(
                title="Top 10 Most Powerful Superhero Abilities Ranked",
                search_volume_signal=82.0,
                first_seen_hours_ago=8.0,
            ),
            RawTopicResult(
                title="New Marvel Phase 5 Character Powers Explained",
                search_volume_signal=76.0,
                first_seen_hours_ago=12.0,
            ),
            RawTopicResult(
                title="Superman vs Thor: Ultimate Power Comparison",
                search_volume_signal=71.0,
                first_seen_hours_ago=18.0,
            ),
            RawTopicResult(
                title="DC vs Marvel: Every Major Superhero Crossover Event",
                search_volume_signal=65.0,
                first_seen_hours_ago=24.0,
            ),
            RawTopicResult(
                title="Black Panther's Vibranium Powers: Full Breakdown",
                search_volume_signal=60.0,
                first_seen_hours_ago=36.0,
            ),
            RawTopicResult(
                title="Invincible vs Omni-Man: Father vs Son Fight Analysis",
                search_volume_signal=55.0,
                first_seen_hours_ago=48.0,
            ),
            RawTopicResult(
                title="Every Time a Hero Switched Sides in Comics History",
                search_volume_signal=50.0,
                first_seen_hours_ago=60.0,
            ),
            RawTopicResult(
                title="The Boys Homelander vs All Superheroes: Power Level",
                search_volume_signal=45.0,
                first_seen_hours_ago=70.0,
            ),
        ]


class TavilyMCPClient:
    """Tavily Search API client for trending topic discovery.

    Uses httpx to call Tavily's REST API directly. Reads the API key from the
    TAVILY_API_KEY environment variable.
    """

    def __init__(self) -> None:
        import os  # noqa: PLC0415
        self._api_key = os.environ.get("TAVILY_API_KEY", "")
        if not self._api_key:
            logger.warning("TAVILY_API_KEY not set — search will fail")

    async def query_trending(
        self,
        query: str,
        hours_back: int,
        excluded_titles: list[str] | None = None,
    ) -> list[RawTopicResult]:
        """Call the Tavily Search API to get trending superhero video topics.

        For local development, returns placeholder data if API key is missing or invalid.
        """
        import httpx  # noqa: PLC0415
        import re as _re  # noqa: PLC0415

        if not self._api_key or self._api_key.startswith("tvly-REPLACE"):
            if is_production_mode():
                raise ValueError(
                    "Production mode requires a valid Tavily API key (TAVILY_API_KEY). "
                    "Set PIPELINE_MODE=development to use placeholder topics."
                )
            logger.warning("Tavily API key not configured — returning placeholder topics for local dev")
            return self._get_placeholder_topics()

        # Map hours_back to Tavily's time_range enum (day/week/month/year).
        # hours_back defaults to a wide 3-month window for evergreen content.
        if hours_back <= 24:
            time_range = "day"
        elif hours_back <= 24 * 7:
            time_range = "week"
        elif hours_back <= 24 * 31:
            time_range = "month"
        else:
            time_range = "year"

        search_query = (
            '"who would win" OR "vs" superhero battle Marvel DC anime power '
            "comparison fight breakdown -trailer -review -merchandise -blu-ray"
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "query": search_query,
                        "search_depth": "basic",
                        "topic": "general",
                        "time_range": time_range,
                        "max_results": 20,
                        "include_answer": False,
                    },
                )
                response.raise_for_status()
                data = response.json()
                raw_results = data.get("results", [])

                results: list[RawTopicResult] = []
                for i, item in enumerate(raw_results):
                    title = (item.get("title") or "").strip()
                    content = (item.get("content") or "").strip()

                    # Prefer the page title if it looks like a usable video topic;
                    # otherwise fall back to the first sentence of the content.
                    candidate = title if title else content[:120]
                    cleaned = _re.sub(r"^[\d]+[.)]\s*|^[-*]\s*", "", candidate).strip()
                    cleaned = _re.sub(r"\s*[-|]\s*[^-|]{0,40}$", "", cleaned).strip()  # strip " - SiteName" suffixes

                    # Reject titles that are too short/long or not a real sentence-like phrase.
                    word_count = len(cleaned.split())
                    if not cleaned or len(cleaned) > 200 or word_count < 3:
                        continue

                    # Reject boilerplate / promotional / off-topic pages.
                    if any(w in cleaned.lower() for w in [
                        "cannot provide", "unable to", "sorry,", "don't have",
                        "404", "not found", "login", "sign in", "subscribe now",
                        "subscribe to", "blu-ray", "4kuhd", "unrated edition",
                        "oscars guide", "best picture", "earnings explained",
                        "trailer", "excited for", "bundle on",
                    ]):
                        continue

                    # Require the title to actually reference a hero/battle concept —
                    # otherwise it's likely an unrelated news/merch/review page.
                    if not any(tag in cleaned.lower() for tag in _RELEVANCE_TAGS):
                        continue

                    score = float(item.get("score", 0.5))
                    results.append(
                        RawTopicResult(
                            title=cleaned,
                            search_volume_signal=float(score * 100),
                            first_seen_hours_ago=float(i * 6),
                        )
                    )
                    if len(results) >= 10:
                        break

                logger.info(f"TavilyMCPClient: extracted {len(results)} topics from API response")
                return results

        except httpx.HTTPStatusError as exc:
            logger.error(f"Tavily API HTTP error: {exc.response.status_code} — {exc.response.text[:200]}")
            raise Exception(
                f"Tavily API error: {exc.response.status_code}. "
                f"Response: {exc.response.text[:200]}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tavily API call failed: {exc}")
            if is_production_mode():
                raise Exception(
                    f"Tavily API call failed in production mode: {exc}. "
                    "Set PIPELINE_MODE=development to use placeholder topics."
                ) from exc
            logger.warning(f"Tavily API call failed: {exc} - returning placeholder topics")
            return self._get_placeholder_topics()

    def _get_placeholder_topics(self) -> list[RawTopicResult]:
        """Return placeholder superhero trending topics for local development."""
        return [
            RawTopicResult(
                title="Spider-Man vs Batman: Who Would Win in a Real Fight",
                search_volume_signal=95.0,
                first_seen_hours_ago=2.0,
            ),
            RawTopicResult(
                title="Avengers vs Justice League: Ultimate Crossover Battle",
                search_volume_signal=88.0,
                first_seen_hours_ago=5.0,
            ),
            RawTopicResult(
                title="Top 10 Most Powerful Superhero Abilities Ranked",
                search_volume_signal=82.0,
                first_seen_hours_ago=8.0,
            ),
            RawTopicResult(
                title="New Marvel Phase 5 Character Powers Explained",
                search_volume_signal=76.0,
                first_seen_hours_ago=12.0,
            ),
            RawTopicResult(
                title="Superman vs Thor: Ultimate Power Comparison",
                search_volume_signal=71.0,
                first_seen_hours_ago=18.0,
            ),
            RawTopicResult(
                title="DC vs Marvel: Every Major Superhero Crossover Event",
                search_volume_signal=65.0,
                first_seen_hours_ago=24.0,
            ),
            RawTopicResult(
                title="Black Panther's Vibranium Powers: Full Breakdown",
                search_volume_signal=60.0,
                first_seen_hours_ago=36.0,
            ),
            RawTopicResult(
                title="Invincible vs Omni-Man: Father vs Son Fight Analysis",
                search_volume_signal=55.0,
                first_seen_hours_ago=48.0,
            ),
            RawTopicResult(
                title="Every Time a Hero Switched Sides in Comics History",
                search_volume_signal=50.0,
                first_seen_hours_ago=60.0,
            ),
            RawTopicResult(
                title="The Boys Homelander vs All Superheroes: Power Level",
                search_volume_signal=45.0,
                first_seen_hours_ago=70.0,
            ),
        ]


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _min_max_normalize(values: list[float]) -> list[float]:
    """Apply min-max normalisation to *values*.

    Formula:
        ``norm_x = (x - min_x) / (max_x - min_x)`` when ``max_x > min_x``,
        else ``0.5`` for every element (all values are identical).

    Args:
        values: Raw floating-point values (one per topic candidate).

    Returns:
        A new list of normalised floats in [0.0, 1.0], same length as *values*.
    """
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v > min_v:
        return [(v - min_v) / (max_v - min_v) for v in values]
    # All identical — assign a neutral mid-point
    return [0.5] * len(values)


def _compute_relevance(title: str) -> tuple[float, list[str]]:
    """Return the binary relevance score and matched tags for *title*.

    Relevance is ``1.0`` if any :data:`_RELEVANCE_TAGS` string appears as a
    case-insensitive substring of *title*, otherwise ``0.0``.

    Args:
        title: Topic title to test.

    Returns:
        A ``(score, matched_tags)`` tuple where *score* is 0.0 or 1.0 and
        *matched_tags* is the list of matching tag strings.
    """
    lower_title = title.lower()
    matched = [tag for tag in _RELEVANCE_TAGS if tag.lower() in lower_title]
    score = 1.0 if matched else 0.0
    return score, matched


def _truncate_to_minute(dt: datetime) -> datetime:
    """Return *dt* truncated to the nearest minute (seconds and microseconds zeroed).

    Args:
        dt: Any timezone-aware or naive datetime.

    Returns:
        New :class:`datetime` with seconds=0 and microsecond=0.
    """
    return dt.replace(second=0, microsecond=0)


# ---------------------------------------------------------------------------
# Topic_Researcher
# ---------------------------------------------------------------------------


class Topic_Researcher:
    """Queries a search provider for trending AI topics and produces a ranked list.

    Each call to :meth:`research` queries the configured provider, computes a
    three-signal composite score for every candidate, deduplicates against
    previously-used titles, and persists the result to Asset_Store.

    Args:
        search_client: Any :class:`SearchClient`-compatible object.
        asset_store: An :class:`~pipeline.asset_store.Asset_Store` instance used
            to persist the JSON research artefact.
        notifier: A :class:`~pipeline.notifier.Notifier` instance used to send
            failure alerts on zero-result exhaustion.
        provider: Informational label for the search provider (used in log messages
            and error details). Accepted values: ``"perplexity"`` or ``"tavily"``.
    """

    def __init__(
        self,
        search_client: SearchClient,
        asset_store: Asset_Store,
        notifier: Notifier,
        provider: Literal["perplexity", "tavily"],
    ) -> None:
        self._client = search_client
        self._store = asset_store
        self._notifier = notifier
        self._provider = provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def research(
        self,
        batch_size: int,
        excluded_titles: list[str],
        run_id: str,
    ) -> list[TopicEntry]:
        """Query the search provider, score, deduplicate, and return ranked topics.

        The method attempts up to :data:`_MAX_ATTEMPTS` queries.  Between each
        failed attempt it waits :data:`_RETRY_INTERVAL_SECONDS` seconds.

        Error semantics:
        - **Partial results (1–4)** before all retries are exhausted → raise
          :class:`PartialResultsError` immediately (no JSON stored, no further
          retries).
        - **Zero results** after all retries → call
          :py:meth:`~pipeline.notifier.Notifier.send_failure_alert` with the
          *run_id* embedded in the message, then raise :class:`TopicResearchError`.
        - **≥ 5 results**: compute composite scores, sort descending, deduplicate,
          persist JSON, and return.

        Args:
            batch_size: Minimum number of topics to return.  Must be ≤
                :data:`_MAX_BATCH_SIZE` (50).  The result list will contain at
                least ``max(5, batch_size)`` entries after deduplication.
            excluded_titles: Case-insensitive list of titles used in the past 30
                days.  Topics whose titles match any entry (case-insensitively) are
                removed before the final list is returned.
            run_id: Orchestrator-assigned pipeline run identifier.  Used to name
                the JSON artefact and in failure alert messages.

        Returns:
            A list of :class:`~pipeline.models.TopicEntry` objects sorted by
            ``composite_score`` descending, with duplicates removed.

        Raises:
            PartialResultsError: When 1–4 results are received (immediate halt,
                no store).
            TopicResearchError: When zero results are obtained after all retries,
                or the result count after deduplication falls below
                ``max(5, batch_size)``.
        """
        if batch_size > _MAX_BATCH_SIZE:
            raise TopicResearchError(
                f"batch_size {batch_size} exceeds maximum allowed value of {_MAX_BATCH_SIZE}."
            )

        # Normalise excluded titles to lowercase for fast O(1) lookup
        excluded_lower: set[str] = {t.lower() for t in excluded_titles}

        raw_results: list[RawTopicResult] = []
        last_error: Exception | None = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            logger.info(
                "Topic_Researcher[%s]: query attempt %d/%d (run_id=%s)",
                self._provider,
                attempt,
                _MAX_ATTEMPTS,
                run_id,
            )
            try:
                raw_results = await self._client.query_trending(
                    query=_SEARCH_QUERY,
                    hours_back=_TRENDING_HOURS_BACK,
                    excluded_titles=list(excluded_lower),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Topic_Researcher: query attempt %d/%d failed: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
                raw_results = []

            rejected_titles = [result.title for result in raw_results if not result.title.isascii()]
            if rejected_titles:
                logger.warning(
                    "Topic_Researcher: discarded %d non-English/Latin-character "
                    "topic title(s) on attempt %d.",
                    len(rejected_titles),
                    attempt,
                )
                raw_results = [result for result in raw_results if result.title.isascii()]

            count = len(raw_results)

            if count == 0:
                # Zero results — sleep and retry (unless this was the last attempt)
                if attempt < _MAX_ATTEMPTS:
                    logger.info(
                        "Topic_Researcher: zero results on attempt %d, "
                        "retrying in %.0f s…",
                        attempt,
                        _RETRY_INTERVAL_SECONDS,
                    )
                    await asyncio.sleep(_RETRY_INTERVAL_SECONDS)
                continue

            if 1 <= count <= 4:
                # Partial results — log and retry rather than halt immediately
                logger.warning(
                    "Topic_Researcher: only %d topic(s) on attempt %d — retrying with wider search…",
                    count, attempt,
                )
                raw_results = []  # treat as zero, will retry
                if attempt < _MAX_ATTEMPTS:
                    await asyncio.sleep(_RETRY_INTERVAL_SECONDS)
                continue

            # ≥ 5 results — proceed to scoring and deduplication
            logger.info(
                "Topic_Researcher: received %d raw results on attempt %d (run_id=%s).",
                count,
                attempt,
                run_id,
            )
            break  # exit retry loop
        else:
            # All attempts returned zero results
            failure_reason = (
                f"Zero topics returned after {_MAX_ATTEMPTS} attempts "
                f"using provider '{self._provider}'"
                + (f": {last_error}" if last_error else "")
            )
            logger.error(
                "Topic_Researcher: %s (run_id=%s)", failure_reason, run_id
            )
            self._notifier.send_failure_alert(
                video_id="",
                stage_name="topic_researcher",
                error_message=f"run_id={run_id}; {failure_reason}",
            )
            raise TopicResearchError(failure_reason)

        # ------------------------------------------------------------------
        # Score, sort, deduplicate
        # ------------------------------------------------------------------

        entries = self._score_and_sort(raw_results)
        entries = self._deduplicate(entries, excluded_lower)

        # Validate we have enough topics after deduplication
        required = max(_MIN_VALID_COUNT, batch_size)
        if len(entries) < required:
            msg = (
                f"Only {len(entries)} unique topic(s) remain after deduplication "
                f"(need ≥ {required}, run_id={run_id})."
            )
            logger.error("Topic_Researcher: %s", msg)
            raise TopicResearchError(msg)

        # ------------------------------------------------------------------
        # Persist JSON to Asset_Store
        # ------------------------------------------------------------------

        await self._store_results(entries, run_id)

        return entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score_and_sort(
        self,
        raw: list[RawTopicResult],
    ) -> list[TopicEntry]:
        """Compute composite scores for *raw* results and return sorted entries.

        Scoring pipeline:
        1. Collect raw signal vectors (search volume, recency via ``first_seen_hours_ago``).
        2. Normalise each vector with min-max normalisation.
        3. Compute binary relevance per title.
        4. ``composite_score = (norm_search_volume + norm_recency + norm_relevance) / 3``.
        5. Sort descending by ``composite_score``.

        Note: higher ``first_seen_hours_ago`` means *less* recent.  We invert the
        normalised value (``1 − norm``) so that a smaller hours-ago value maps to a
        higher recency signal.

        Args:
            raw: Non-empty list of raw search results.

        Returns:
            List of :class:`~pipeline.models.TopicEntry` sorted by
            ``composite_score`` descending.
        """
        # Capture the query timestamp (truncated to the nearest minute) once for the
        # whole batch so all entries share a consistent timestamp.
        query_ts = _truncate_to_minute(datetime.now(timezone.utc))

        search_volumes = [r.search_volume_signal for r in raw]
        hours_ago = [r.first_seen_hours_ago for r in raw]

        # Normalise search volume (higher → more popular)
        norm_sv = _min_max_normalize(search_volumes)

        # Normalise hours_ago then invert so that *more recent* → higher score
        norm_hours_raw = _min_max_normalize(hours_ago)
        norm_recency = [1.0 - v for v in norm_hours_raw]

        entries: list[TopicEntry] = []
        for i, result in enumerate(raw):
            relevance_score, matched_tags = _compute_relevance(result.title)
            composite = (norm_sv[i] + norm_recency[i] + relevance_score) / 3.0

            entries.append(
                TopicEntry(
                    title=result.title,
                    composite_score=round(composite, 6),
                    recency_hours=result.first_seen_hours_ago,
                    source_query_timestamp=query_ts,
                    search_volume_signal=result.search_volume_signal,
                    relevance_tags_matched=matched_tags,
                )
            )

        # Sort descending by composite score; use title as tiebreaker for stability
        entries.sort(key=lambda e: (-e.composite_score, e.title))
        return entries

    @staticmethod
    def _deduplicate(
        entries: list[TopicEntry],
        excluded_lower: set[str],
    ) -> list[TopicEntry]:
        """Remove entries whose titles appear in *excluded_lower* (case-insensitive).

        Also removes intra-list duplicates, keeping the first occurrence (highest score).

        Args:
            entries: Scored and sorted topic entries.
            excluded_lower: Set of lowercase title strings to filter out.

        Returns:
            Filtered list preserving the original sort order.
        """
        seen: set[str] = set()
        result: list[TopicEntry] = []
        for entry in entries:
            key = entry.title.lower()
            if key in excluded_lower:
                logger.debug(
                    "Topic_Researcher: skipping duplicate title %r (matches excluded list).",
                    entry.title,
                )
                continue
            if key in seen:
                logger.debug(
                    "Topic_Researcher: skipping intra-batch duplicate title %r.",
                    entry.title,
                )
                continue
            seen.add(key)
            result.append(entry)
        return result

    async def _store_results(
        self,
        entries: list[TopicEntry],
        run_id: str,
    ) -> None:
        """Serialise *entries* to JSON and write to Asset_Store under RESEARCH subfolder.

        File path pattern:
            ``ai-youtube-pipeline/run-{run_id}/research/{run_id}_topics.json``

        The JSON document is an array of objects; each object contains at minimum:
        ``title``, ``composite_score``, ``recency_hours``, ``source_query_timestamp``.

        Args:
            entries: Deduplicated, sorted :class:`~pipeline.models.TopicEntry` list.
            run_id: Pipeline run identifier; used for the video_id and filename.
        """
        # Build list of dicts guaranteed to include the four required fields.
        payload: list[dict] = []  # type: ignore[type-arg]
        for e in entries:
            payload.append(
                {
                    "title": e.title,
                    "composite_score": e.composite_score,
                    "recency_hours": e.recency_hours,
                    "source_query_timestamp": e.source_query_timestamp.isoformat(),
                    # Extra fields persisted for downstream consumers
                    "search_volume_signal": e.search_volume_signal,
                    "relevance_tags_matched": e.relevance_tags_matched,
                }
            )

        json_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
        filename = f"{run_id}_topics.json"
        video_id = f"run-{run_id}"

        logger.info(
            "Topic_Researcher: storing %d entries to %s/%s (run_id=%s).",
            len(entries),
            SubFolder.RESEARCH.value,
            filename,
            run_id,
        )

        await self._store.write(
            video_id=video_id,
            subfolder=SubFolder.RESEARCH,
            filename=filename,
            content=json_bytes,
        )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Topic_Researcher",
    "TopicResearchError",
    "PartialResultsError",
    "SearchClient",
    "RawTopicResult",
    "PerplexityMCPClient",
    "TavilyMCPClient",
]
