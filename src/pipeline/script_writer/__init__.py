"""Script_Writer subsystem — Claude API + word-count enforcement.

Generates and revises YouTube video scripts using the Claude API with
Style_Profile injection.  Produces versioned Markdown files persisted to the
Asset_Store under ``scripts/script_v{n}.md``.

Design reference: §4 Script_Writer
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

import anthropic

from pipeline.asset_store import Asset_Store, AssetStoreError
from pipeline.config import is_production_mode, require_production_config
from pipeline.models import Script, StyleProfile, SubFolder, TopicEntry
from pipeline.notifier import Notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Word-count bounds — derived from script_duration_minutes in config.json
# Default 1 min if not configured. The generate() method receives the actual
# value from PipelineConfig at runtime.
_WPM = 150
_DEFAULT_DURATION_MIN = 1.0
_MIN_WORDS = int(_DEFAULT_DURATION_MIN * _WPM * 0.85)
_MAX_WORDS = int(_DEFAULT_DURATION_MIN * _WPM * 1.15)

# Timing constants used to derive max-word limits per section (at 150 WPM)
_WPM = 150
_HOOK_MAX_SECONDS = 60
_CTA_MAX_SECONDS = 30
_HOOK_MAX_WORDS = _WPM * _HOOK_MAX_SECONDS // 60  # 150 words
_CTA_MAX_WORDS = _WPM * _CTA_MAX_SECONDS // 60    # 75 words

# Claude retry policy (design §1 General Retry Policy)
_CLAUDE_RETRY_ATTEMPTS = 3
_CLAUDE_RETRY_BASE_S = 5.0
_CLAUDE_RETRY_MAX_S = 20.0

# Claude model identifier
_CLAUDE_MODEL = "claude-opus-4-5"
_DEFAULT_MAX_TOKENS = 4096

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ScriptGenerationError(Exception):
    """Raised when the generated script's word count remains out of bounds
    after one automatic revision attempt."""


class EmptyEditError(ValueError):
    """Raised when a user-submitted edit is empty or identical to the current
    script content — no revision is written and Claude is not called."""


# ---------------------------------------------------------------------------
# ClaudeClient Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ClaudeClient(Protocol):
    """Minimal async interface for calling a Claude language model.

    Concrete implementations: :class:`AnthropicClaudeClient` (production) or
    any test double that satisfies this protocol.
    """

    async def complete(self, prompt: str, max_tokens: int = _DEFAULT_MAX_TOKENS) -> str:
        """Send *prompt* to Claude and return the full text response.

        Args:
            prompt: The user-turn content to submit.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            The model's text response as a plain string.

        Raises:
            Exception: Any API-level error (handled externally with retries).
        """
        ...


# ---------------------------------------------------------------------------
# AnthropicClaudeClient — production concrete implementation
# ---------------------------------------------------------------------------


class AnthropicClaudeClient:
    """Production Claude client backed by the ``anthropic`` SDK.

    Reads ``ANTHROPIC_API_KEY`` from the environment at construction time.
    For local development, returns placeholder content when API key is invalid.

    Args:
        api_key: Override the default environment-variable key lookup.
            Defaults to ``os.environ["ANTHROPIC_API_KEY"]``.
        model: Claude model identifier.  Defaults to :data:`_CLAUDE_MODEL`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _CLAUDE_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._api_key = resolved_key
        self._model = model
        
        # In production mode, require a valid API key
        if is_production_mode():
            require_production_config(
                "Anthropic Claude API",
                resolved_key if resolved_key and not resolved_key.startswith("sk-ant-REPLACE") else None,
                "Production mode requires a valid ANTHROPIC_API_KEY. "
                "Set PIPELINE_MODE=development to use placeholder scripts."
            )
        
        # Only create client if key looks valid
        if resolved_key and not resolved_key.startswith("sk-ant-REPLACE"):
            self._client: Optional[anthropic.Anthropic] = anthropic.Anthropic(api_key=resolved_key)
        else:
            self._client = None
            logger.warning(
                "Anthropic API key not configured — will use placeholder scripts for local dev. "
                "Set PIPELINE_MODE=production to enforce real API."
            )

    async def complete(self, prompt: str, max_tokens: int = _DEFAULT_MAX_TOKENS) -> str:
        """Call the Anthropic Messages API (synchronous SDK; run in an executor).

        For local development, returns placeholder script content when API key is invalid.

        Args:
            prompt: The user-turn prompt text.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            The first text block from the model response, or placeholder content.
        """
        # If no valid client, return placeholder
        if self._client is None:
            if is_production_mode():
                raise ValueError(
                    "Production mode requires a valid Anthropic API key. "
                    "Set PIPELINE_MODE=development to use placeholder scripts."
                )
            logger.info("Anthropic client not configured — returning placeholder script")
            return self._get_placeholder_script(prompt)

        loop = asyncio.get_running_loop()

        def _call() -> str:
            try:
                message = self._client.messages.create(  # type: ignore[union-attr]
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text  # type: ignore[union-attr]
            except anthropic.AuthenticationError as exc:
                if is_production_mode():
                    raise ValueError(
                        f"Anthropic authentication failed in production mode: {exc}. "
                        "Check your ANTHROPIC_API_KEY."
                    ) from exc
                logger.warning(f"Anthropic authentication failed — returning placeholder script: {exc}")
                return self._get_placeholder_script(prompt)

        return await loop.run_in_executor(None, _call)

    def _get_placeholder_script(self, prompt: str) -> str:
        """Generate placeholder script content for local development."""
        # Extract topic from prompt if possible
        import re  # noqa: PLC0415
        topic_match = re.search(r'Topic:\s*"([^"]+)"', prompt)
        topic = topic_match.group(1) if topic_match else "AI and Machine Learning"
        
        # Generate a script with word count in the valid range (800-1500)
        # This placeholder is approximately 1100 words
        return f"""# {topic}

## Hook (First 60 seconds)

Have you ever wondered how artificial intelligence is transforming our world in ways we never imagined? In this video, we're diving deep into {topic.lower()}, one of the most exciting and groundbreaking developments in technology today. Whether you're a complete beginner or already familiar with AI concepts, you'll discover something new that could fundamentally change how you think about the future of technology and innovation.

## Introduction

Welcome back to the channel! I'm incredibly excited to talk about today's topic because it's not only timely and relevant, but it's also something that affects every single one of us. {topic} has been making headlines across the tech industry, dominating conversations at conferences, and driving massive investments from companies around the world. And for good reason—this represents a fundamental shift in how we approach technology and problem-solving.

We're going to break down exactly what makes this so important, explore the key concepts you need to understand to really grasp what's happening, and discuss in detail what it means for the future of technology, business, and society as a whole. By the end of this video, you'll have a comprehensive understanding that goes beyond the hype and gets to the real substance.

## Main Content

### Part 1: Understanding the Basics

Let's start with the fundamentals, because if we don't get this foundation right, everything else becomes confusing. {topic} represents a significant advancement in how machines process, understand, and work with information. At its core, this technology enables computers to perform complex tasks that traditionally required human intelligence, reasoning, and decision-making capabilities.

The key innovation here lies in how the system learns from data rather than following rigid, pre-programmed rules. Unlike traditional programming where every single rule and condition must be explicitly coded by a human programmer, these systems can identify patterns, make predictions, and generate insights based on examples. This approach has opened up possibilities that seemed like pure science fiction just a few years ago, and the pace of progress is accelerating.

What's particularly fascinating is how this mirrors certain aspects of human learning. We don't learn by memorizing every possible rule and scenario—we learn by exposure to examples, by trial and error, by building intuitions. These AI systems work in a similar way, though at a scale and speed that's impossible for biological intelligence.

### Part 2: Real-World Applications

Now let's talk about how this technology is being used in practice, because that's where things get really interesting and tangible. From healthcare to finance, from entertainment to transportation, from education to scientific research, {topic.lower()} is already making a significant and measurable impact on our daily lives in ways both visible and invisible.

In healthcare, these systems are helping doctors diagnose diseases more accurately and quickly, develop personalized treatment plans based on individual patient characteristics, and even predict health issues before they become serious problems. In finance, they're detecting fraudulent transactions in real-time, providing better investment advice, and helping banks assess credit risk more fairly and accurately.

But it goes even further. In transportation, this technology is powering autonomous vehicles and optimizing traffic flow in major cities. In entertainment, it's creating personalized recommendations and even generating new content. In education, it's enabling adaptive learning systems that adjust to each student's pace and style. The applications are virtually endless, and we're honestly only scratching the surface of what's possible as the technology continues to evolve and mature.

### Part 3: The Technology Behind It

To truly appreciate this innovation, we need to understand the technical foundations, at least at a high level. The breakthrough came from combining massive datasets with powerful computational resources and clever algorithmic approaches that researchers developed over years of experimentation and refinement.

Researchers discovered that by structuring the system in specific ways—using what we call neural networks inspired by biological brains—they could achieve remarkable results that exceeded expectations. The training process involves exposing the system to millions or even billions of examples, allowing it to gradually refine its understanding through a process of continuous adjustment and optimization.

This iterative learning process mirrors how humans learn in some ways, but operates at a scale and speed that's completely impossible for biological intelligence. Where a human might need years to become an expert in a domain, these systems can process vast amounts of information in hours or days.

### Part 4: Challenges and Limitations

Of course, no technology is perfect, and it's crucial we talk about this honestly. {topic} faces several important challenges that researchers, practitioners, and policymakers are actively working to address. Issues around bias and fairness remain significant concerns—these systems can inadvertently perpetuate or even amplify biases present in their training data.

There are also questions about interpretability and explainability. Unlike traditional software where you can trace exactly why a decision was made, these AI systems often operate as black boxes, making it difficult to understand their reasoning. This creates challenges for accountability and trust, especially in high-stakes applications.

Additionally, there are practical considerations around computational costs, which can be enormous, data requirements that may not always be available, and deployment complexity that requires specialized expertise. Understanding these limitations is just as important as appreciating the capabilities, especially as we think about responsible development and deployment of these technologies.

### Part 5: Future Directions

Looking ahead, the potential for {topic.lower()} is truly enormous and exciting. Researchers around the world are exploring new architectures, more efficient training techniques, and entirely new application areas we haven't even considered yet. Some of the most exciting work involves combining this technology with other emerging fields—like quantum computing, biotechnology, and materials science—to create even more powerful and capable systems.

As the technology matures, we can expect to see it become more accessible to smaller organizations and individuals, more efficient in terms of computational requirements, and more capable in terms of what problems it can solve. The next few years will likely bring breakthroughs and applications that we literally can't even imagine today, given how fast this field is moving.

## Conclusion

So there you have it—a comprehensive look at {topic} that covers everything from the basics to the cutting edge. We've covered the fundamental concepts, explored real-world applications across multiple industries, examined the underlying technology and how it works, discussed the current challenges and limitations that need to be addressed, and looked toward an exciting future full of possibilities.

This field is moving incredibly fast, with new breakthroughs announced almost weekly, and staying informed is more important than ever. Whether you're a developer, a business leader, a student, or just someone interested in technology, understanding these concepts will help you navigate the rapidly changing technological landscape.

## Call to Action (Last 30 seconds)

If you found this video helpful and informative, please hit that like button—it really helps the channel grow and reach more people. Subscribe for more content on AI, emerging technologies, and how they're shaping our future. Leave a comment below with your thoughts on {topic.lower()}—I read every single one and love hearing your perspectives and questions. And don't forget to click the bell icon so you never miss an upload when I post new content. Thanks so much for watching, and I'll see you in the next video!"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_words(text: str) -> int:
    """Return the word count of *text* using a simple whitespace split.

    Matches the specification: ``len(content.split())``.
    """
    return len(text.split())


def _retry_delay(attempt: int) -> float:
    """Return exponential back-off delay (seconds) for the given 1-based attempt.

    Formula: ``min(_CLAUDE_RETRY_BASE_S * 2^(attempt-1), _CLAUDE_RETRY_MAX_S)``
    → 5 s, 10 s, 20 s (capped).
    """
    return min(_CLAUDE_RETRY_BASE_S * (2 ** (attempt - 1)), _CLAUDE_RETRY_MAX_S)


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_generation_prompt(
    topic: TopicEntry,
    style_profile: StyleProfile,
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
) -> str:
    """Construct the Claude generation prompt for a new script.

    Injects Style_Profile data (tone, pacing, rhetorical patterns, segment
    structure) and enforces the required script structure with per-segment
    speaker-direction annotations.

    Args:
        topic: The selected TopicEntry whose title is the script subject.
        style_profile: The channel StyleProfile used to calibrate tone and structure.
        min_words: Minimum target word count.
        max_words: Maximum target word count.

    Returns:
        A fully-formed prompt string ready to be submitted to Claude.
    """
    sp = style_profile
    tone_desc = (
        "positive and enthusiastic" if sp.narration_tone.sentiment_polarity > 0.2
        else "negative and critical" if sp.narration_tone.sentiment_polarity < -0.2
        else "neutral and balanced"
    )
    pacing_desc = (
        "fast-paced" if sp.pacing.avg_words_per_minute > 160
        else "slow-paced" if sp.pacing.avg_words_per_minute < 110
        else "moderate-paced"
    )

    rhetorical = (
        ", ".join(sp.rhetorical_patterns) if sp.rhetorical_patterns
        else "storytelling, direct address"
    )

    body_segments = max(3, min(5, round(sp.segment_structure.body_segment_count_avg)))

    prompt = f"""You are writing a YouTube video script for a channel with the following style profile:

CHANNEL STYLE:
- Narration tone: {tone_desc} (sentiment polarity: {sp.narration_tone.sentiment_polarity:.2f})
- Pacing: {pacing_desc} ({sp.pacing.avg_words_per_minute} words/minute)
- Average sentence length: {sp.pacing.avg_sentence_length_words:.1f} words
- Rhetorical patterns used: {rhetorical}
- Segment structure: {"intro present, " if sp.segment_structure.intro_present else ""}{"hook present, " if sp.segment_structure.hook_present else ""}body ({body_segments} segments average){"," if sp.segment_structure.cta_present else ""}{"CTA present" if sp.segment_structure.cta_present else ""}

VIDEO TOPIC: {topic.title}

STORYTELLING STYLE (THIS IS THE #1 PRIORITY — OVERRIDE EVERYTHING ELSE):
You are a cinematic narrator telling an EPIC STORY — not explaining facts.
Your script should feel like a movie, not a textbook. Every sentence should make
the viewer feel something: awe, fear, excitement, curiosity.

RULES FOR STORYTELLING:
- NEVER say "Let's talk about..." or "In this video we'll explore..." — these are boring
- NEVER list facts like a Wikipedia article
- ALWAYS write in scenes: describe what's HAPPENING, not what something IS
- Open EVERY segment by dropping the viewer into the middle of action
- Use present tense for action scenes: "He raises his fist. The ground cracks."
- Short sentences for impact. Then a longer sentence to let the rhythm breathe and build.
- Create REVEALS and TWISTS: "But here's the thing nobody realizes..."
- End every segment on a cliffhanger that forces the viewer to keep watching
- Write like you're narrating a movie trailer combined with a fight commentary

BAD (boring factual — NEVER DO THIS):
"Superman has super strength and can fly at high speeds. He was born on Krypton and sent to Earth. His powers include heat vision, freeze breath, and invulnerability."

GOOD (cinematic storytelling — ALWAYS DO THIS):
"[pause] Picture this. A man floating above the clouds... fists clenched... cape tearing in the wind. Below him, an entire city holds its breath. Because the last time he hit something this hard... [emphasis]a mountain disappeared.[/emphasis] Born on a dying world, launched into the stars as an infant, he crash-landed on a planet where the sun itself makes him a god. Every photon that touches his skin becomes pure, unstoppable power."

REQUIRED SCRIPT STRUCTURE:
Write a complete YouTube video script told as a STORY with these sections:

1. HOOK (≤ {_HOOK_MAX_WORDS} words / ≤ {_HOOK_MAX_SECONDS} seconds at {_WPM} WPM)
   - Drop the viewer into a dramatic moment — mid-action, mid-crisis
   - Pose the central question as a life-or-death stakes scenario

2. BODY ({body_segments} segments — each told as a SCENE, not a lecture)
   - Each segment is a new SCENE in the story with rising stakes
   - Describe powers/abilities through ACTION, not description
   - Show don't tell: "His fist connects. Shockwave levels six city blocks." NOT "He has super strength."
   - Each segment ends with a twist or escalation that pulls into the next

3. CTA (≤ {_CTA_MAX_WORDS} words / ≤ {_CTA_MAX_SECONDS} seconds at {_WPM} WPM)
   - Wrap the story with a final dramatic statement
   - Then transition to call-to-action naturally

ANNOTATION REQUIREMENTS (MANDATORY):
- Add AT LEAST ONE speaker-direction annotation per section/segment using these tags:
  [pause], [emphasis], [slow], [fast], [loud], [quiet], [breath]
- Place annotations inline where the direction applies, e.g.: "This is [emphasis]critical[/emphasis]"
  or "[pause]" between sentences to signal a beat.
- These annotations guide the narrator and must reflect the channel pacing data.

TOTAL WORD COUNT TARGET: between {min_words} and {max_words} words (excluding annotation tags).

Match the channel's rhetorical patterns ({rhetorical}) throughout.
Output ONLY the script content — no preamble, no explanations, no metadata.
Start directly with the HOOK section heading.
"""
    return prompt


def _build_word_count_revision_prompt(
    content: str,
    current_word_count: int,
    topic_title: str,
    min_words: int = _MIN_WORDS,
    max_words: int = _MAX_WORDS,
) -> str:
    """Construct a prompt asking Claude to revise the script to fit word-count bounds.

    Args:
        content: The current script text.
        current_word_count: Word count of the current draft.
        topic_title: Topic title for context.
        min_words: Minimum target word count.
        max_words: Maximum target word count.

    Returns:
        A revision prompt string.
    """
    if current_word_count < min_words:
        direction = (
            f"The script is too short ({current_word_count} words). "
            f"Expand it to reach at least {min_words} words by adding more detail, "
            f"examples, and explanation in the body segments."
        )
    else:
        direction = (
            f"The script is too long ({current_word_count} words). "
            f"Shorten it to at most {max_words} words by tightening sentences "
            f"and removing redundant content, while keeping the HOOK, BODY segments, "
            f"and CTA structure intact."
        )

    return f"""The following YouTube video script about "{topic_title}" needs revision.

{direction}

REQUIREMENTS:
- Keep ALL section headings (HOOK, BODY segments, CTA) intact.
- Keep AT LEAST ONE [annotation] per section/segment.
- Final word count must be between {min_words} and {max_words} words (excluding annotations).
- Output ONLY the revised script, no explanations.

CURRENT SCRIPT:
{content}
"""


def _build_revision_prompt(script_content: str, edits: str) -> str:
    """Construct a prompt asking Claude to apply user edits to an existing script.

    Args:
        script_content: The current script Markdown text.
        edits: The creator's edit instructions or replacement text.

    Returns:
        A revision prompt string.
    """
    return f"""You are revising a YouTube video script based on creator feedback.

EDIT INSTRUCTIONS:
{edits}

REQUIREMENTS FOR THE REVISED SCRIPT:
- Apply the edit instructions faithfully.
- Preserve the original structure (HOOK, BODY segments, CTA).
- Keep AT LEAST ONE speaker-direction annotation ([pause], [emphasis], etc.) per section/segment.
- Output ONLY the revised script, no preamble, no explanations.

CURRENT SCRIPT:
{script_content}
"""


# ---------------------------------------------------------------------------
# Script_Writer
# ---------------------------------------------------------------------------


class Script_Writer:
    """Generates and revises YouTube video scripts using the Claude API.

    Scripts are persisted to the Asset_Store as versioned Markdown files
    at ``scripts/script_v{n}.md``.  The writer injects Style_Profile data
    into every generation prompt to match the reference channel's voice.

    Args:
        claude_client: Any object satisfying the :class:`ClaudeClient` protocol.
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
        topic: TopicEntry,
        style_profile: StyleProfile,
        video_id: str,
        script_duration_minutes: Optional[float] = None,
    ) -> Script:
        """Generate a new script for *topic* styled after *style_profile*.

        Steps:
        1. Validate inputs (non-empty topic title; non-None style_profile).
        2. Build a Claude prompt with full Style_Profile injection.
        3. Call Claude with retry (3 attempts, exponential 5 s base / 20 s max).
        4. Enforce word-count bounds; revise once automatically if out of range.
        5. Raise :class:`ScriptGenerationError` if still out of range after revision.
        6. Determine the next version number and write ``script_v{n}.md`` to Asset_Store.
        7. Return a fully-populated :class:`~pipeline.models.Script` object.

        Args:
            topic: Selected :class:`~pipeline.models.TopicEntry`.
            style_profile: Loaded :class:`~pipeline.models.StyleProfile`.
            video_id: Pipeline video identifier (used for Asset_Store path).
            script_duration_minutes: Target duration in minutes. If provided,
                overrides module-level _MIN_WORDS / _MAX_WORDS. Defaults to
                config.json or env var values.

        Returns:
            A :class:`~pipeline.models.Script` with ``asset_url`` set.

        Raises:
            ScriptGenerationError: When word count remains out of bounds after
                one automatic revision.
        """
        # ---- 1. Input validation -------------------------------------------
        if not topic or not topic.title.strip():
            msg = f"Script_Writer.generate: missing or empty topic title for video_id={video_id}"
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="script_generation",
                error_message=msg,
            )
            raise ScriptGenerationError(msg)

        if style_profile is None:
            msg = f"Script_Writer.generate: style_profile is None for video_id={video_id}"
            logger.error(msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="script_generation",
                error_message=msg,
            )
            raise ScriptGenerationError(msg)

        # ---- 2. Build generation prompt ------------------------------------
        # Compute word-count bounds from duration config
        if script_duration_minutes and script_duration_minutes > 0:
            min_words = int(script_duration_minutes * _WPM * 0.85)
            max_words = int(script_duration_minutes * _WPM * 1.15)
        else:
            min_words = _MIN_WORDS
            max_words = _MAX_WORDS

        prompt = _build_generation_prompt(topic, style_profile, min_words, max_words)

        # ---- 3. Call Claude with retry -------------------------------------
        content = await self._call_claude_with_retry(prompt, video_id=video_id)

        # ---- 4. Word-count enforcement -------------------------------------
        word_count = _count_words(content)
        logger.info(
            "Script_Writer.generate: initial word count=%d (target %d-%d) for video_id=%s",
            word_count,
            min_words,
            max_words,
            video_id,
        )

        if not (min_words <= word_count <= max_words):
            logger.warning(
                "Script_Writer.generate: word count %d out of [%d, %d] — "
                "running automatic revision for video_id=%s",
                word_count,
                min_words,
                max_words,
                video_id,
            )
            revision_prompt = _build_word_count_revision_prompt(
                content, word_count, topic.title, min_words, max_words
            )
            content = await self._call_claude_with_retry(revision_prompt, video_id=video_id)
            word_count = _count_words(content)

            # ---- 5. Check again after revision ----------------------------
            if not (min_words <= word_count <= max_words):
                error_msg = (
                    f"Script word count {word_count} still outside [{min_words}, {max_words}] "
                    f"after automatic revision for video_id={video_id}."
                )
                logger.error(error_msg)
                self._notifier.send_failure_alert(
                    video_id=video_id,
                    stage_name="script_generation",
                    error_message=error_msg,
                )
                raise ScriptGenerationError(error_msg)

        # ---- 6. Determine version and persist to Asset_Store ---------------
        version = await self._next_version(video_id)
        filename = f"script_v{version}.md"
        asset_url = await self._store.write(
            video_id=video_id,
            subfolder=SubFolder.SCRIPTS,
            filename=filename,
            content=content.encode("utf-8"),
        )

        logger.info(
            "Script_Writer.generate: wrote %s (words=%d, version=%d) for video_id=%s",
            filename,
            word_count,
            version,
            video_id,
        )

        # ---- 7. Return Script model ----------------------------------------
        return Script(
            video_id=video_id,
            version=version,
            content=content,
            word_count=word_count,
            style_profile_doc_id=style_profile.doc_id,
            asset_url=asset_url,
            created_at=_utcnow(),
        )

    async def revise(
        self,
        script: Script,
        edits: str,
        video_id: str,
    ) -> Script:
        """Apply creator edits to an existing script and save as the next version.

        Validation:
        - Raises :class:`EmptyEditError` if ``edits.strip()`` is empty.
        - Raises :class:`EmptyEditError` if ``edits == script.content`` (no change).

        The revised script is saved as ``script_v{n+1}.md``; the previous version
        is preserved (no deletion).

        Args:
            script: The current :class:`~pipeline.models.Script` to revise.
            edits: The creator's edit instructions or replacement text.
            video_id: Pipeline video identifier.

        Returns:
            A new :class:`~pipeline.models.Script` with ``version = script.version + 1``.

        Raises:
            EmptyEditError: When ``edits`` is empty or identical to the current content.
        """
        # ---- Validate edits ------------------------------------------------
        if not edits.strip():
            raise EmptyEditError(
                f"revise: edits string is empty for video_id={video_id}, "
                f"script version {script.version}."
            )
        if edits == script.content:
            raise EmptyEditError(
                f"revise: edits are identical to the current script content "
                f"for video_id={video_id}, script version {script.version}."
            )

        # ---- Build revision prompt and call Claude -------------------------
        prompt = _build_revision_prompt(script.content, edits)
        revised_content = await self._call_claude_with_retry(prompt, video_id=video_id)

        word_count = _count_words(revised_content)
        new_version = script.version + 1
        filename = f"script_v{new_version}.md"

        # ---- Persist revised script ----------------------------------------
        asset_url = await self._store.write(
            video_id=video_id,
            subfolder=SubFolder.SCRIPTS,
            filename=filename,
            content=revised_content.encode("utf-8"),
        )

        logger.info(
            "Script_Writer.revise: wrote %s (words=%d, version=%d) for video_id=%s",
            filename,
            word_count,
            new_version,
            video_id,
        )

        return Script(
            video_id=video_id,
            version=new_version,
            content=revised_content,
            word_count=word_count,
            style_profile_doc_id=script.style_profile_doc_id,
            asset_url=asset_url,
            created_at=_utcnow(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _call_claude_with_retry(self, prompt: str, video_id: str) -> str:
        """Call the Claude API with exponential back-off retry.

        Retry policy (design §1 General Retry Policy):
        - 3 attempts total.
        - Back-off: ``min(5 * 2^(attempt-1), 20)`` seconds → 5 s, 10 s, 20 s.

        Args:
            prompt: The prompt to submit.
            video_id: Used for log messages.

        Returns:
            The model's text response.

        Raises:
            Exception: If all retry attempts fail (re-raises the last exception).
        """
        last_exc: Exception = Exception("No attempts made")

        for attempt in range(1, _CLAUDE_RETRY_ATTEMPTS + 1):
            try:
                result = await self._claude.complete(prompt)
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

    async def _next_version(self, video_id: str) -> int:
        """Determine the next script version — always v1 on first call, 
        increments each subsequent call for the same video_id."""
        # Track in-memory: each pipeline run creates a fresh video_id,
        # so version probing Drive is unnecessary. Just count writes.
        if not hasattr(self, "_version_counters"):
            self._version_counters: dict[str, int] = {}
        current = self._version_counters.get(video_id, 0) + 1
        self._version_counters[video_id] = current
        return current


# ---------------------------------------------------------------------------
# GeminiClaudeClient — alternate production implementation using Google's
# Gemini API instead of Anthropic. Satisfies the same ClaudeClient protocol
# so it's a drop-in replacement wherever ClaudeClient is expected
# (Script_Writer, Metadata_Generator).
#
# Used as a fallback when ANTHROPIC_API_KEY is unavailable/expired or
# billing is blocked (e.g. RBI card restrictions in India) — Gemini accepts
# Indian billing methods and has a generous free tier.
# ---------------------------------------------------------------------------

_GEMINI_MODEL = "gemini-3-flash-preview"


class GeminiClaudeClient:
    """Production LLM client backed by Google's ``google-generativeai`` SDK.

    Implements the same :class:`ClaudeClient` protocol as
    :class:`AnthropicClaudeClient`, so it can be used as a drop-in
    replacement for Script_Writer and Metadata_Generator.

    Reads ``GEMINI_API_KEY`` from the environment at construction time.

    Args:
        api_key: Override the default environment-variable key lookup.
            Defaults to ``os.environ["GEMINI_API_KEY"]``.
        model: Gemini model identifier. Defaults to :data:`_GEMINI_MODEL`.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _GEMINI_MODEL,
    ) -> None:
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._api_key = resolved_key
        self._model_name = model

        if is_production_mode():
            require_production_config(
                "Google Gemini API",
                resolved_key if resolved_key and not resolved_key.startswith("REPLACE") else None,
                "Production mode requires a valid GEMINI_API_KEY. "
                "Set PIPELINE_MODE=development to use placeholder scripts."
            )

        if resolved_key and not resolved_key.startswith("REPLACE"):
            from google import genai  # noqa: PLC0415
            self._client: Optional["genai.Client"] = genai.Client(api_key=resolved_key)
        else:
            self._client = None
            logger.warning(
                "Gemini API key not configured — will use placeholder scripts for local dev. "
                "Set PIPELINE_MODE=production to enforce real API."
            )

    async def complete(self, prompt: str, max_tokens: int = _DEFAULT_MAX_TOKENS) -> str:
        """Call the Gemini generateContent API (synchronous SDK; run in an executor).

        For local development, returns placeholder script content when API key is invalid.

        Args:
            prompt: The user-turn prompt text.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            The model's text response, or placeholder content.
        """
        if self._client is None:
            if is_production_mode():
                raise ValueError(
                    "Production mode requires a valid Gemini API key. "
                    "Set PIPELINE_MODE=development to use placeholder scripts."
                )
            logger.info("Gemini client not configured — returning placeholder script")
            return self._get_placeholder_script(prompt)

        from google.genai import types as genai_types  # noqa: PLC0415

        loop = asyncio.get_running_loop()

        def _call() -> str:
            response = self._client.models.generate_content(  # type: ignore[union-attr]
                model=self._model_name,
                contents=prompt,
                # Disable "thinking" — otherwise the model can spend its entire
                # token budget reasoning and never emit the actual answer,
                # especially on longer/more constrained prompts.
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return response.text

        # Retry on transient errors (503 overloaded, 429 rate-limited) with
        # exponential backoff — these are common on preview/experimental models.
        last_exc: Optional[Exception] = None
        for attempt in range(1, _CLAUDE_RETRY_ATTEMPTS + 1):
            try:
                return await loop.run_in_executor(None, _call)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                exc_str = str(exc)
                is_transient = "503" in exc_str or "429" in exc_str or "UNAVAILABLE" in exc_str
                if is_transient and attempt < _CLAUDE_RETRY_ATTEMPTS:
                    delay = _retry_delay(attempt)
                    logger.warning(
                        "Gemini API attempt %d/%d failed (transient: %s) — retrying in %.0fs",
                        attempt, _CLAUDE_RETRY_ATTEMPTS, exc_str[:100], delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                break

        if is_production_mode():
            raise ValueError(
                f"Gemini API call failed in production mode: {last_exc}. "
                "Check your GEMINI_API_KEY."
            ) from last_exc
        logger.warning(f"Gemini API call failed — returning placeholder script: {last_exc}")
        return self._get_placeholder_script(prompt)

    def _get_placeholder_script(self, prompt: str) -> str:
        """Delegate to the same placeholder generator used by AnthropicClaudeClient."""
        return AnthropicClaudeClient._get_placeholder_script(self, prompt)  # type: ignore[arg-type]


def build_claude_client(api_key: Optional[str] = None) -> ClaudeClient:
    """Build the appropriate LLM client based on environment configuration.

    Selection order:
    1. If ``LLM_PROVIDER=gemini`` is set, use :class:`GeminiClaudeClient`.
    2. If ``ANTHROPIC_API_KEY`` is missing/placeholder but ``GEMINI_API_KEY``
       is present, automatically fall back to Gemini.
    3. Otherwise, use :class:`AnthropicClaudeClient` (default).

    This lets the pipeline keep working when Anthropic billing is blocked
    (e.g. RBI card restrictions) without any code changes — just set
    ``GEMINI_API_KEY`` in ``.env``.

    Args:
        api_key: Optional explicit API key override (passed to whichever
            client is selected).

    Returns:
        A :class:`ClaudeClient`-compatible instance.
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    anthropic_configured = bool(anthropic_key) and not anthropic_key.startswith("sk-ant-REPLACE")

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    gemini_configured = bool(gemini_key) and not gemini_key.startswith("REPLACE")

    if provider == "gemini" or (not anthropic_configured and gemini_configured):
        logger.info("build_claude_client: using GeminiClaudeClient")
        return GeminiClaudeClient(api_key=api_key)

    logger.info("build_claude_client: using AnthropicClaudeClient")
    return AnthropicClaudeClient(api_key=api_key)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "AnthropicClaudeClient",
    "ClaudeClient",
    "EmptyEditError",
    "GeminiClaudeClient",
    "Script_Writer",
    "ScriptGenerationError",
    "build_claude_client",
]
