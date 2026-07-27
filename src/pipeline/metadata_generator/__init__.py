"""Metadata_Generator subsystem — Claude API + SEO metadata.

Generates SEO-optimised YouTube metadata (title, description, tags, hashtags,
chapters) from an approved script and the topic list.  Each field is validated
independently after generation; failing fields are regenerated once with a
targeted prompt.  If a field is still invalid after regeneration the stage is
halted, the Notifier is alerted, and ``MetadataGenerationError`` is raised.

Design reference: §7 Metadata_Generator
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from pipeline.asset_store import Asset_Store
from pipeline.models import Chapter, MetadataPackage, Script, SubFolder, TopicEntry
from pipeline.notifier import Notifier
from pipeline.script_writer import ClaudeClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reading-speed assumption for chapter timestamp derivation (design §7)
_WPM_READING = 150

# Claude retry policy — same as script_writer (design §1)
_CLAUDE_RETRY_ATTEMPTS = 3
_CLAUDE_RETRY_BASE_S = 5.0
_CLAUDE_RETRY_MAX_S = 20.0

# Description word-count bounds
_DESC_MIN_WORDS = 200
_DESC_MAX_WORDS = 500

# Tag count bounds
_TAG_MIN_COUNT = 10
_TAG_MAX_COUNT = 15

# Tag per-item word bounds
_TAG_WORD_MIN = 2
_TAG_WORD_MAX = 5

# Hashtag count bounds
_HASHTAG_MIN_COUNT = 3
_HASHTAG_MAX_COUNT = 5

# Hashtag body character bounds (excluding '#')
_HASHTAG_BODY_MIN = 2
_HASHTAG_BODY_MAX = 30

# Regex for valid hashtag: '#' followed by 2–30 non-whitespace characters
_HASHTAG_RE = re.compile(r"^#[^\s]{2,30}$")

# Minimum number of tag words that must appear in description
_DESC_TAG_APPEARANCES_MIN = 3

# Title character limit
_TITLE_MAX_CHARS = 60

# Max tokens to request from Claude for full-package generation
_FULL_GENERATION_MAX_TOKENS = 4096

# Max tokens for a single-field regeneration prompt
_REGEN_MAX_TOKENS = 2048

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class MetadataGenerationError(Exception):
    """Raised when a metadata field remains invalid after one regeneration attempt."""


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _retry_delay(attempt: int) -> float:
    """Exponential back-off delay for the given 1-based attempt number.

    Formula: ``min(_CLAUDE_RETRY_BASE_S * 2^(attempt-1), _CLAUDE_RETRY_MAX_S)``
    → 5 s, 10 s, 20 s (capped).
    """
    return min(_CLAUDE_RETRY_BASE_S * (2 ** (attempt - 1)), _CLAUDE_RETRY_MAX_S)


def _count_words(text: str) -> int:
    """Return the whitespace-split word count of *text*."""
    return len(text.split())


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from a Claude JSON response.

    Handles variants like:
    - ```json\\n{...}\\n```
    - ```\\n{...}\\n```
    - raw ``{...}``
    """
    text = text.strip()
    # Remove opening fence (```json or ```)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Remove closing fence
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _primary_keyword(topics: list[TopicEntry]) -> str:
    """Return the title of the highest-scoring topic, sanitised.
    
    Filters out Perplexity refusal/error messages that sometimes come back
    as topic titles (e.g. 'The search results include outdated items...')
    to ensure the primary keyword is a real topic phrase.

    Truncation is done at a word boundary so the keyword never ends mid-word
    or with dangling punctuation (open parenthesis, colon, etc.).
    """
    _REFUSAL_MARKERS = (
        "search results", "outdated", "unrelated", "cannot", "unable",
        "here are", "following", "based on", "note:", "disclaimer",
        "i don't", "i cannot", "unfortunately", "as of", "no specific",
    )
    sorted_topics = sorted(topics, key=lambda t: t.composite_score, reverse=True)
    for topic in sorted_topics:
        title_lower = topic.title.lower()
        if not any(marker in title_lower for marker in _REFUSAL_MARKERS):
            return _truncate_keyword(topic.title, max_chars=60)
    # All topics look like refusals — use a safe fallback from the first one
    fallback = sorted_topics[0].title
    # Take first 5 words as a keyword
    words = fallback.split()[:5]
    return _truncate_keyword(" ".join(words), max_chars=60)


def _truncate_keyword(text: str, max_chars: int = 60) -> str:
    """Truncate *text* to at most *max_chars* at a word boundary.

    Ensures the result does not end with dangling punctuation such as an
    open parenthesis, colon, dash, or comma. Also removes trailing words
    that leave unmatched opening brackets/parentheses.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    # Truncate at last space within limit
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]

    # Strip trailing dangling punctuation
    truncated = truncated.rstrip(" :,;-–—(\"'")

    # Remove unmatched opening parenthesis/bracket content
    # If there's an open paren without a close, drop everything from it
    if "(" in truncated and ")" not in truncated[truncated.rfind("("):]:
        truncated = truncated[:truncated.rfind("(")].rstrip(" :,;-–—")
    if "[" in truncated and "]" not in truncated[truncated.rfind("["):]:
        truncated = truncated[:truncated.rfind("[")].rstrip(" :,;-–—")

    return truncated.strip()


def _derive_chapters(script: Script) -> list[Chapter]:
    """Estimate per-segment timestamps at *_WPM_READING* WPM and build Chapter list.

    Strategy:
    1. Split the script content by lines that look like segment headings
       (lines starting with '#', or ALL-CAPS lines, or numbered headings).
    2. Count words in each segment.
    3. Accumulate elapsed seconds; convert to MM:SS.

    If fewer than 2 headings are found the whole script is treated as one
    chapter labelled "Introduction".

    Args:
        script: The approved :class:`~pipeline.models.Script`.

    Returns:
        A list of :class:`~pipeline.models.Chapter` objects.
    """
    lines = script.content.splitlines()

    # Detect heading lines: markdown headings or plain ALL-CAPS short lines
    heading_pattern = re.compile(
        r"^(#{1,4}\s+.+|[A-Z][A-Z\s\d/&:,\-]{3,}[A-Z\d])$"
    )

    segments: list[tuple[str, list[str]]] = []  # (heading_label, body_lines)
    current_heading: str = "Introduction"
    current_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and heading_pattern.match(stripped):
            # Save previous segment
            if current_body or not segments:
                segments.append((current_heading, current_body))
            # Strip leading '#' characters from markdown headings
            current_heading = re.sub(r"^#+\s*", "", stripped)
            current_body = []
        else:
            current_body.append(line)

    # Flush last segment
    if current_body or not segments:
        segments.append((current_heading, current_body))

    if len(segments) < 2:
        # Fallback: single chapter
        return [Chapter(timestamp="00:00", label=current_heading)]

    chapters: list[Chapter] = []
    elapsed_seconds = 0.0

    for heading, body_lines in segments:
        body_text = " ".join(body_lines)
        word_count = _count_words(body_text)

        # Format current elapsed time as MM:SS
        total_secs = int(elapsed_seconds)
        mm = total_secs // 60
        ss = total_secs % 60
        timestamp = f"{mm:02d}:{ss:02d}"

        chapters.append(Chapter(timestamp=timestamp, label=heading))

        # Advance elapsed time by this segment's reading duration
        segment_seconds = (word_count / _WPM_READING) * 60.0
        elapsed_seconds += segment_seconds

    return chapters

# ---------------------------------------------------------------------------
# Validation helpers — each returns True if the field is valid
# ---------------------------------------------------------------------------


def _validate_title(title: str, primary_kw: str) -> bool:
    """Validate the generated title.

    Rules:
    - ``len(title) <= 60``
    - At least 2 significant words from ``primary_kw`` appear in the title
      (full-phrase matching is too strict for long keywords)
    """
    if len(title) > _TITLE_MAX_CHARS:
        return False

    # Check for at least 2 significant words from the keyword in the title
    # (ignores stop words for more flexible matching)
    _STOP_WORDS = {"a", "an", "the", "of", "in", "on", "at", "to", "for",
                   "is", "are", "was", "were", "be", "been", "and", "or",
                   "but", "with", "from", "as", "by", "that", "this",
                   "it", "its", "such", "no", "not", "vs", "vs."}
    kw_words = [
        w.lower().strip(":'\".,!?")
        for w in primary_kw.split()
        if w.lower().strip(":'\".,!?") not in _STOP_WORDS and len(w) > 2
    ]
    title_lower = title.lower()
    matches = sum(1 for w in kw_words if w in title_lower)
    return matches >= min(2, len(kw_words))


def _validate_description(description: str, tags: list[str]) -> bool:
    """Validate the generated description.

    Rules:
    - word count in [200, 500]
    - at least 3 of the tag strings (case-insensitive) appear somewhere in the description
    """
    wc = _count_words(description)
    if not (_DESC_MIN_WORDS <= wc <= _DESC_MAX_WORDS):
        return False

    desc_lower = description.lower()
    tag_hits = sum(1 for tag in tags if tag.lower() in desc_lower)
    if tag_hits < _DESC_TAG_APPEARANCES_MIN:
        return False

    return True


def _validate_tags(tags: list[str]) -> bool:
    """Validate the generated tags list.

    Rules:
    - count in [10, 15]
    - each tag is 2–5 words
    """
    if not (_TAG_MIN_COUNT <= len(tags) <= _TAG_MAX_COUNT):
        return False
    for tag in tags:
        wc = _count_words(tag)
        if not (_TAG_WORD_MIN <= wc <= _TAG_WORD_MAX):
            return False
    return True


def _validate_hashtags(hashtags: list[str]) -> bool:
    """Validate the generated hashtags list.

    Rules:
    - count in [3, 5]
    - each matches ``^#[^\\s]{2,30}$``
    """
    if not (_HASHTAG_MIN_COUNT <= len(hashtags) <= _HASHTAG_MAX_COUNT):
        return False
    for ht in hashtags:
        if not _HASHTAG_RE.match(ht):
            return False
    return True


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_full_generation_prompt(
    script: Script,
    topics: list[TopicEntry],
    primary_kw: str,
    chapters: list[Chapter],
) -> str:
    """Build the all-fields Claude generation prompt.

    Returns a prompt that instructs Claude to output a single JSON object
    containing title, description, tags, hashtags, and chapters.

    Args:
        script: The approved script (content used for context).
        topics: Ranked topic list (used for keyword context).
        primary_kw: Highest-scoring topic title.
        chapters: Pre-derived chapter list (timestamps + labels to embed).

    Returns:
        A fully-formed prompt string.
    """
    # Format chapter timestamps for the prompt
    chapter_lines = "\n".join(
        f"  {ch.timestamp} - {ch.label}" for ch in chapters
    )

    # Top 5 topics for keyword context
    top_topics = sorted(topics, key=lambda t: t.composite_score, reverse=True)[:5]
    topic_keywords = ", ".join(t.title for t in top_topics)

    prompt = f"""You are an expert YouTube SEO specialist. Generate optimised metadata for the
following video script.

PRIMARY KEYWORD: {primary_kw}
RELATED KEYWORDS: {topic_keywords}

SCRIPT CONTENT (first 3000 characters for context):
{script.content[:3000]}

CHAPTER TIMESTAMPS TO INCLUDE IN DESCRIPTION:
{chapter_lines}

OUTPUT REQUIREMENTS — return ONLY a valid JSON object with these exact keys:

{{
  "title": "<string: ≤60 characters total, but the FIRST 50 characters must be a complete, compelling hook visible on mobile. Use this formula: [Power Word/Emotion] + [Specific Topic] + [Curiosity Gap]. Examples: 'UNSTOPPABLE: Why DC Heroes Destroy Marvel in a Fight', 'The REAL Reason Thor Beats Superman Every Time'. The hook before char 50 must make sense alone — do NOT cut mid-word at position 50. Must relate to the primary keyword '{primary_kw}'>",
  "description": "<string: 200-500 words. Must contain: (1) a one-paragraph summary of the video, (2) timestamped chapters using exactly the timestamps listed above in format 'MM:SS - label', (3) 3-5 CTAs or links such as 'Subscribe', 'Check the description', 'Comment below', (4) a closing paragraph that naturally includes at least 3 of the tags>",
  "tags": ["<2-5 word tag>", ...],
  "hashtags": ["#keyword", ...]
}}

CONSTRAINTS:
- title: at most 60 characters total. The first 50 characters MUST form a complete, compelling hook (YouTube truncates at ~50 chars on mobile). Use a power word (UNSTOPPABLE, INSANE, SHOCKING, ULTIMATE, BRUTAL) or curiosity gap (Why X Beats Y, The REAL Reason, What Nobody Tells You About). Must relate to '{primary_kw}'
- description: exactly 200-500 words (count carefully), include chapter markers, 3-5 CTAs, closing paragraph uses ≥3 tags
- tags: exactly 10-15 tags, each tag must be 2-5 words, cover primary topic, related subtopics, and channel brand terms
- hashtags: exactly 3-5 hashtags, each starts with '#', body (after '#') is 2-30 characters with NO spaces

Return ONLY the JSON object. No markdown fences, no explanation, no extra text.
"""
    return prompt


def _build_title_regen_prompt(
    primary_kw: str,
    bad_title: str,
    reason: str,
) -> str:
    """Targeted prompt to regenerate only the title field."""
    return f"""The previously generated YouTube title was invalid.

INVALID TITLE: "{bad_title}"
REASON: {reason}

REQUIREMENTS:
- At most 60 characters total
- First 50 characters must form a complete hook visible on mobile (YouTube truncates at ~50 on mobile)
- Use formula: [Power Word] + [Topic] + [Curiosity Gap]
- Power words: UNSTOPPABLE, INSANE, SHOCKING, ULTIMATE, BRUTAL, The REAL Reason, What Nobody Tells You
- Must relate to the primary keyword: '{primary_kw}'
- Engaging, click-worthy, accurate to the video content

Return ONLY the new title string — no quotes, no JSON, no explanation.
"""


def _build_description_regen_prompt(
    primary_kw: str,
    tags: list[str],
    chapters: list[Chapter],
    bad_desc: str,
    reason: str,
) -> str:
    """Targeted prompt to regenerate only the description field."""
    chapter_lines = "\n".join(f"  {ch.timestamp} - {ch.label}" for ch in chapters)
    tags_sample = ", ".join(tags[:8])

    return f"""The previously generated YouTube description was invalid.

REASON: {reason}

REQUIREMENTS:
- Word count: exactly 200-500 words
- One-paragraph summary at the start
- Timestamped chapters section:
{chapter_lines}
- 3-5 calls-to-action / links (e.g. "Subscribe", "Comment below", link placeholders)
- Closing paragraph that naturally includes at least 3 of these tags: {tags_sample}
- Primary keyword '{primary_kw}' should appear naturally

PREVIOUS INVALID DESCRIPTION (for reference):
{bad_desc[:1500]}

Return ONLY the new description text. No JSON, no markdown, no explanation.
"""


def _build_tags_regen_prompt(
    primary_kw: str,
    topics: list[TopicEntry],
    bad_tags: list[str],
    reason: str,
) -> str:
    """Targeted prompt to regenerate only the tags field."""
    top_topics = sorted(topics, key=lambda t: t.composite_score, reverse=True)[:5]
    topic_str = ", ".join(t.title for t in top_topics)

    return f"""The previously generated YouTube tags were invalid.

REASON: {reason}
INVALID TAGS: {bad_tags}

REQUIREMENTS:
- Exactly 10-15 tags (no more, no fewer)
- Each tag must be 2-5 words (not 1 word, not 6+ words)
- Cover: primary topic ('{primary_kw}'), related subtopics ({topic_str}), channel brand terms
- No duplicate tags

Return ONLY a JSON array of tag strings, e.g. ["tag one", "tag two", ...].
No markdown fences, no extra text.
"""


def _build_hashtags_regen_prompt(
    primary_kw: str,
    bad_hashtags: list[str],
    reason: str,
) -> str:
    """Targeted prompt to regenerate only the hashtags field."""
    return f"""The previously generated YouTube hashtags were invalid.

REASON: {reason}
INVALID HASHTAGS: {bad_hashtags}

REQUIREMENTS:
- Exactly 3-5 hashtags
- Each starts with '#'
- Body after '#' is 2-30 characters with NO spaces or punctuation (CamelCase or lowercase)
- Based on primary keyword: '{primary_kw}'

Examples of valid hashtags: #AIContent, #YouTubeSEO, #MachineLearning

Return ONLY a JSON array of hashtag strings, e.g. ["#HashOne", "#HashTwo", ...].
No markdown fences, no extra text.
"""

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_full_response(raw: str) -> dict[str, Any]:
    """Parse Claude's full-generation JSON response.

    Strips markdown code fences then calls ``json.loads``.

    Args:
        raw: Raw Claude response text.

    Returns:
        Parsed dictionary with metadata fields.

    Raises:
        ValueError: If JSON parsing fails.
    """
    cleaned = _strip_code_fences(raw)
    return json.loads(cleaned)  # type: ignore[no-any-return]


def _parse_json_array(raw: str) -> list[str]:
    """Parse a Claude response expected to be a JSON array of strings.

    Args:
        raw: Raw Claude response text (possibly with code fences).

    Returns:
        List of strings.

    Raises:
        ValueError: If JSON parsing fails or result is not a list.
    """
    cleaned = _strip_code_fences(raw)
    result = json.loads(cleaned)
    if not isinstance(result, list):
        raise ValueError(f"Expected JSON array, got: {type(result).__name__}")
    return [str(item) for item in result]


# ---------------------------------------------------------------------------
# Metadata_Generator
# ---------------------------------------------------------------------------


class Metadata_Generator:
    """Generates SEO-optimised YouTube metadata using the Claude API.

    Flow:
    1. Pre-flight validation: halt if script content is empty or topics list is empty.
    2. Derive chapters from script segments at 150 WPM.
    3. Call Claude to produce all fields in one JSON response.
    4. Validate each field independently.
    5. On validation failure: regenerate ONLY that field once with a targeted prompt.
    6. If still invalid: raise :class:`MetadataGenerationError`.
    7. Write complete :class:`~pipeline.models.MetadataPackage` JSON to
       ``metadata/{video_id}.json`` in the Asset_Store.

    Args:
        claude_client: Any object satisfying the
            :class:`~pipeline.script_writer.ClaudeClient` protocol.
        asset_store: The pipeline :class:`~pipeline.asset_store.Asset_Store` instance.
        notifier: The pipeline :class:`~pipeline.notifier.Notifier` instance.
    """

    def __init__(
        self,
        claude_client: ClaudeClient,
        asset_store: Asset_Store,
        notifier: Notifier,
    ) -> None:
        self._claude = claude_client
        self._store = asset_store
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        script: Script,
        topics: list[TopicEntry],
        video_id: str,
    ) -> MetadataPackage:
        """Generate and validate a complete :class:`~pipeline.models.MetadataPackage`.

        Args:
            script: The approved :class:`~pipeline.models.Script`.
            topics: Ranked topic list from Topic_Researcher.
            video_id: Pipeline video identifier (used for Asset_Store path).

        Returns:
            A validated :class:`~pipeline.models.MetadataPackage` whose JSON is
            persisted to ``metadata/{video_id}.json`` in the Asset_Store.

        Raises:
            MetadataGenerationError: On pre-flight failure or when any field
                remains invalid after one regeneration attempt.
        """
        # ---- 1. Pre-flight checks ----------------------------------------
        if not script.content or not script.content.strip():
            msg = (
                f"Metadata_Generator.generate: script content is empty "
                f"for video_id={video_id}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg)

        if not topics:
            msg = (
                f"Metadata_Generator.generate: topics list is empty "
                f"for video_id={video_id}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg)

        # ---- 2. Resolve primary keyword and derive chapters ----------------
        primary_kw = _primary_keyword(topics)
        chapters = _derive_chapters(script)
        logger.info(
            "Metadata_Generator: primary_keyword=%r, chapters=%d for video_id=%s",
            primary_kw,
            len(chapters),
            video_id,
        )

        # ---- 3. Call Claude for full metadata generation -------------------
        prompt = _build_full_generation_prompt(script, topics, primary_kw, chapters)
        raw_response = await self._call_claude_with_retry(prompt, video_id=video_id)

        try:
            data = _parse_full_response(raw_response)
        except (json.JSONDecodeError, ValueError) as exc:
            msg = (
                f"Metadata_Generator: failed to parse Claude JSON response "
                f"for video_id={video_id}: {exc}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg) from exc

        title: str = str(data.get("title", ""))
        description: str = str(data.get("description", ""))
        tags: list[str] = [str(t) for t in data.get("tags", [])]
        hashtags: list[str] = [str(h) for h in data.get("hashtags", [])]

        # ---- 4 & 5. Validate each field; regenerate once on failure --------
        title = await self._validate_and_maybe_regen_title(
            title, primary_kw, video_id
        )
        tags = await self._validate_and_maybe_regen_tags(
            tags, primary_kw, topics, video_id
        )
        description = await self._validate_and_maybe_regen_description(
            description, tags, chapters, primary_kw, video_id
        )
        hashtags = await self._validate_and_maybe_regen_hashtags(
            hashtags, primary_kw, video_id
        )

        # ---- 6. Assemble the MetadataPackage --------------------------------
        package = MetadataPackage(
            video_id=video_id,
            title=title,
            description=description,
            tags=tags,
            hashtags=hashtags,
            chapters=chapters,
            primary_keyword=primary_kw,
            generated_at=_utcnow(),
        )

        # ---- 7. Persist to Asset_Store as JSON ------------------------------
        filename = f"{video_id}.json"
        metadata_url = await self._store.write(
            video_id=video_id,
            subfolder=SubFolder.METADATA,
            filename=filename,
            content=package.model_dump_json(indent=2).encode("utf-8"),
        )
        package = package.model_copy(update={"asset_url": metadata_url})

        logger.info(
            "Metadata_Generator: wrote %s for video_id=%s", filename, video_id
        )
        return package

    async def patch_youtube_id(
        self,
        package: MetadataPackage,
        youtube_video_id: str,
    ) -> MetadataPackage:
        """Patch the metadata JSON in Drive with the YouTube video ID after upload.

        Re-writes ``metadata/{video_id}.json`` in the Asset_Store with
        ``youtube_video_id`` set, and returns an updated :class:`MetadataPackage`.

        Args:
            package: The existing :class:`MetadataPackage` returned by :meth:`generate`.
            youtube_video_id: The YouTube video ID returned after a successful upload.

        Returns:
            Updated :class:`MetadataPackage` with ``youtube_video_id`` set.
        """
        updated = package.model_copy(update={"youtube_video_id": youtube_video_id})
        filename = f"{package.video_id}.json"
        await self._store.write(
            video_id=package.video_id,
            subfolder=SubFolder.METADATA,
            filename=filename,
            content=updated.model_dump_json(indent=2).encode("utf-8"),
        )
        logger.info(
            "Metadata_Generator: patched youtube_video_id=%s into %s",
            youtube_video_id,
            filename,
        )
        return updated

    # ------------------------------------------------------------------
    # Per-field validate-and-regenerate helpers
    # ------------------------------------------------------------------

    async def _validate_and_maybe_regen_title(
        self,
        title: str,
        primary_kw: str,
        video_id: str,
    ) -> str:
        """Validate title; regenerate once if invalid; raise on second failure."""
        if _validate_title(title, primary_kw):
            return title

        reason = []
        if len(title) > _TITLE_MAX_CHARS:
            reason.append(f"length {len(title)} exceeds {_TITLE_MAX_CHARS} characters")
        if primary_kw.lower() not in title.lower():
            reason.append(f"primary keyword '{primary_kw}' not found in title")
        reason_str = "; ".join(reason)

        logger.warning(
            "Metadata_Generator: title validation failed (%s) — regenerating "
            "for video_id=%s",
            reason_str,
            video_id,
        )

        regen_prompt = _build_title_regen_prompt(primary_kw, title, reason_str)
        new_title = (
            await self._call_claude_with_retry(regen_prompt, video_id=video_id)
        ).strip().strip('"')

        if not _validate_title(new_title, primary_kw):
            msg = (
                f"Metadata_Generator: title still invalid after regeneration "
                f"({reason_str}) for video_id={video_id}. title={new_title!r}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg)

        logger.info(
            "Metadata_Generator: title regenerated successfully for video_id=%s",
            video_id,
        )
        return new_title

    async def _validate_and_maybe_regen_description(
        self,
        description: str,
        tags: list[str],
        chapters: list[Chapter],
        primary_kw: str,
        video_id: str,
    ) -> str:
        """Validate description; regenerate once if invalid; raise on second failure."""
        if _validate_description(description, tags):
            return description

        wc = _count_words(description)
        tag_hits = sum(1 for tag in tags if tag.lower() in description.lower())
        reason = []
        if not (_DESC_MIN_WORDS <= wc <= _DESC_MAX_WORDS):
            reason.append(f"word count {wc} not in [{_DESC_MIN_WORDS}, {_DESC_MAX_WORDS}]")
        if tag_hits < _DESC_TAG_APPEARANCES_MIN:
            reason.append(
                f"only {tag_hits}/{_DESC_TAG_APPEARANCES_MIN} required tags appear in description"
            )
        reason_str = "; ".join(reason)

        logger.warning(
            "Metadata_Generator: description validation failed (%s) — regenerating "
            "for video_id=%s",
            reason_str,
            video_id,
        )

        regen_prompt = _build_description_regen_prompt(
            primary_kw, tags, chapters, description, reason_str
        )
        new_desc = (
            await self._call_claude_with_retry(regen_prompt, video_id=video_id)
        ).strip()

        if not _validate_description(new_desc, tags):
            new_wc = _count_words(new_desc)
            new_tag_hits = sum(1 for tag in tags if tag.lower() in new_desc.lower())
            msg = (
                f"Metadata_Generator: description still invalid after regeneration "
                f"(wc={new_wc}, tag_hits={new_tag_hits}) for video_id={video_id}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg)

        logger.info(
            "Metadata_Generator: description regenerated successfully for video_id=%s",
            video_id,
        )
        return new_desc

    async def _validate_and_maybe_regen_tags(
        self,
        tags: list[str],
        primary_kw: str,
        topics: list[TopicEntry],
        video_id: str,
    ) -> list[str]:
        """Validate tags; regenerate once if invalid; raise on second failure."""
        if _validate_tags(tags):
            return tags

        reason = []
        if not (_TAG_MIN_COUNT <= len(tags) <= _TAG_MAX_COUNT):
            reason.append(f"tag count {len(tags)} not in [{_TAG_MIN_COUNT}, {_TAG_MAX_COUNT}]")
        bad_tags = [t for t in tags if not (_TAG_WORD_MIN <= _count_words(t) <= _TAG_WORD_MAX)]
        if bad_tags:
            reason.append(
                f"{len(bad_tags)} tag(s) have word count outside [{_TAG_WORD_MIN}, {_TAG_WORD_MAX}]: {bad_tags[:3]}"
            )
        reason_str = "; ".join(reason)

        logger.warning(
            "Metadata_Generator: tags validation failed (%s) — regenerating "
            "for video_id=%s",
            reason_str,
            video_id,
        )

        regen_prompt = _build_tags_regen_prompt(primary_kw, topics, tags, reason_str)
        raw = await self._call_claude_with_retry(regen_prompt, video_id=video_id)
        try:
            new_tags = _parse_json_array(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            msg = (
                f"Metadata_Generator: tags regeneration produced unparseable JSON "
                f"for video_id={video_id}: {exc}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg) from exc

        if not _validate_tags(new_tags):
            msg = (
                f"Metadata_Generator: tags still invalid after regeneration "
                f"(count={len(new_tags)}) for video_id={video_id}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg)

        logger.info(
            "Metadata_Generator: tags regenerated successfully for video_id=%s", video_id
        )
        return new_tags

    async def _validate_and_maybe_regen_hashtags(
        self,
        hashtags: list[str],
        primary_kw: str,
        video_id: str,
    ) -> list[str]:
        """Validate hashtags; regenerate once if invalid; raise on second failure."""
        if _validate_hashtags(hashtags):
            return hashtags

        reason = []
        if not (_HASHTAG_MIN_COUNT <= len(hashtags) <= _HASHTAG_MAX_COUNT):
            reason.append(
                f"hashtag count {len(hashtags)} not in [{_HASHTAG_MIN_COUNT}, {_HASHTAG_MAX_COUNT}]"
            )
        bad_ht = [h for h in hashtags if not _HASHTAG_RE.match(h)]
        if bad_ht:
            reason.append(f"malformed hashtags: {bad_ht[:3]}")
        reason_str = "; ".join(reason)

        logger.warning(
            "Metadata_Generator: hashtags validation failed (%s) — regenerating "
            "for video_id=%s",
            reason_str,
            video_id,
        )

        regen_prompt = _build_hashtags_regen_prompt(primary_kw, hashtags, reason_str)
        raw = await self._call_claude_with_retry(regen_prompt, video_id=video_id)
        try:
            new_hashtags = _parse_json_array(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            msg = (
                f"Metadata_Generator: hashtags regeneration produced unparseable JSON "
                f"for video_id={video_id}: {exc}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg) from exc

        if not _validate_hashtags(new_hashtags):
            msg = (
                f"Metadata_Generator: hashtags still invalid after regeneration "
                f"(count={len(new_hashtags)}) for video_id={video_id}"
            )
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="metadata_generation",
                error_message=msg,
            )
            raise MetadataGenerationError(msg)

        logger.info(
            "Metadata_Generator: hashtags regenerated successfully for video_id=%s",
            video_id,
        )
        return new_hashtags

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_claude_with_retry(self, prompt: str, video_id: str) -> str:
        """Call the Claude API with exponential back-off retry.

        Retry policy (design §1):
        - 3 attempts total.
        - Back-off: ``min(5 * 2^(attempt-1), 20)`` seconds → 5 s, 10 s, 20 s.

        Args:
            prompt: The prompt to submit to Claude.
            video_id: Used for log messages.

        Returns:
            The model's text response.

        Raises:
            Exception: If all retry attempts fail (re-raises the last exception).
        """
        last_exc: Exception = Exception("No attempts made")

        for attempt in range(1, _CLAUDE_RETRY_ATTEMPTS + 1):
            try:
                result = await self._claude.complete(
                    prompt, max_tokens=_FULL_GENERATION_MAX_TOKENS
                )
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = _retry_delay(attempt)
                logger.warning(
                    "Claude API attempt %d/%d failed for video_id=%s: %s — "
                    "retrying in %.0f s.",
                    attempt,
                    _CLAUDE_RETRY_ATTEMPTS,
                    video_id,
                    exc,
                    delay,
                )
                if attempt < _CLAUDE_RETRY_ATTEMPTS:
                    await asyncio.sleep(delay)

        raise last_exc


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Metadata_Generator",
    "MetadataGenerationError",
]
