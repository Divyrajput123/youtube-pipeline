"""Visual_Generator subsystem — Viewmax MCP + video compilation.

Generates per-segment video clips via the Viewmax MCP, compiles them into a
final MP4 (synchronized with narration audio), produces a thumbnail JPEG, and
writes both artefacts to the Asset_Store.

Key design points
-----------------
* **Scene prompt derivation** — the script is split on Markdown headings
  (``## `` or ``### ``).  If no headings are found the content is divided into
  equal thirds labelled hook / body / cta.
* **Per-clip retry** — each Viewmax clip is attempted up to 3 times with a
  random delay sampled from [5, 30] seconds via ``random.uniform``.
* **Static fallback** — if all retries for a clip are exhausted a 1920×1080
  JPEG is synthesised with Pillow (black background, white centred text) and
  fed to ffmpeg as a looped still image.
* **Video compilation** — ``subprocess`` calls ``ffmpeg`` to concatenate clips
  and mux with the narration MP3.  Clips originating from still-image
  fallbacks use ``-loop 1 -t 3`` to produce a 3-second clip.
* **Thumbnail** — Pillow renders a 1280×720 JPEG using the first dominant
  colour from the StyleProfile, white text overlay, and a contrasting
  placeholder rectangle.  The file is compressed to stay < 2 MB.

Documented constraint
---------------------
Generated frames are **not** pixel-for-pixel reproductions of reference
channel frames.  This is guaranteed at the Viewmax MCP prompt level — scene
prompts are unique per video — but no runtime pixel-comparison check is
performed (doing so would require downloading the reference channel frames at
generation time, which is outside this subsystem's scope).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import random
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol, runtime_checkable

from PIL import Image, ImageDraw, ImageFont

from pipeline.asset_store import Asset_Store
from pipeline.config import is_production_mode
from pipeline.content_calendar import Content_Calendar
from pipeline.models import (
    NarrationAsset,
    PipelineStatus,
    Script,
    StyleProfile,
    SubFolder,
    VisualAsset,
)
from pipeline.notifier import Notifier
from pipeline.script_writer import build_claude_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLIP_RETRY_ATTEMPTS: int = 3
_FALLBACK_WIDTH: int = 1920
_FALLBACK_HEIGHT: int = 1080
_THUMBNAIL_WIDTH: int = 1280
_THUMBNAIL_HEIGHT: int = 720
_THUMBNAIL_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB
_FFMPEG_FRAMERATE: int = 24
_FFMPEG_RESOLUTION: str = "1920x1080"
_FALLBACK_CLIP_DURATION_S: int = 3  # seconds for a still-image fallback clip
_CLIP_DEFAULT_DURATION_S: int = 5   # seconds requested from Viewmax per clip

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VisualGeneratorError(Exception):
    """Raised when the Visual_Generator cannot complete an operation."""


class KlingContentModerationError(Exception):
    """Raised when Kling rejects a prompt due to content moderation (risk control system).

    Signals that the prompt itself — not a transient network issue — caused
    the failure, so the caller should rephrase rather than retry as-is.
    """


# ---------------------------------------------------------------------------
# ViewmaxClient Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ViewmaxClient(Protocol):
    """Minimal async interface to the Viewmax video-generation service.

    Implementors wrap the Viewmax MCP server.  The stub
    :class:`ViewmaxMCPClient` raises ``NotImplementedError`` so that tests
    can substitute a mock without requiring a live Viewmax connection.
    """

    async def generate_clip(self, prompt: str, duration_seconds: int, seed: int = -1) -> bytes:
        """Generate a video clip for *prompt* of *duration_seconds* length.

        Args:
            prompt: Natural-language description of the desired visual scene.
            duration_seconds: Requested clip length in seconds.
            seed: Random seed for reproducibility. -1 = random.

        Returns:
            Raw MP4 bytes for the generated clip.

        Raises:
            Exception: Any Viewmax API error (retried by Visual_Generator).
        """
        ...


# ---------------------------------------------------------------------------
# Placeholder clip generator for fallback mode
# ---------------------------------------------------------------------------


def _generate_placeholder_clip_jpeg(prompt: str) -> bytes:
    """Generate a placeholder 1920×1080 JPEG with prompt text for testing.
    
    This allows the Visual_Generator to continue without a real Viewmax MCP server.
    The JPEG will be looped by ffmpeg to create a video clip.
    
    Args:
        prompt: Scene prompt text to display on the placeholder frame.
        
    Returns:
        Raw JPEG bytes for a 1920×1080 frame with the prompt text.
    """
    import hashlib
    
    # Pick a color based on the prompt hash for variety
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
    r = (prompt_hash >> 16) & 0xFF
    g = (prompt_hash >> 8) & 0xFF
    b = prompt_hash & 0xFF
    bg_color = (r, g, b)
    
    img = Image.new("RGB", (_FALLBACK_WIDTH, _FALLBACK_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", size=48)
        font_prompt = ImageFont.truetype("DejaVuSans.ttf", size=32)
    except (IOError, OSError):
        font_title = ImageFont.load_default()
        font_prompt = ImageFont.load_default()

    # Draw "PLACEHOLDER" at the top
    title = "PLACEHOLDER CLIP"
    try:
        bbox = draw.textbbox((0, 0), title, font=font_title)
        text_w = bbox[2] - bbox[0]
    except AttributeError:
        text_w, _ = draw.textsize(title, font=font_title)  # type: ignore[attr-defined]
    
    x_title = (_FALLBACK_WIDTH - text_w) // 2
    y_title = 100
    draw.text((x_title, y_title), title, fill=(255, 255, 255), font=font_title)
    
    # Draw the prompt text wrapped in the middle
    import textwrap
    wrapped_prompt = textwrap.fill(prompt, width=60)
    lines = wrapped_prompt.split('\n')
    
    y_offset = 400
    for line in lines[:6]:  # Limit to 6 lines
        try:
            bbox = draw.textbbox((0, 0), line, font=font_prompt)
            line_w = bbox[2] - bbox[0]
        except AttributeError:
            line_w, _ = draw.textsize(line, font=font_prompt)  # type: ignore[attr-defined]
        
        x_line = (_FALLBACK_WIDTH - line_w) // 2
        draw.text((x_line, y_offset), line, fill=(255, 255, 255), font=font_prompt)
        y_offset += 50

    # Convert to JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ViewmaxMCPClient — production stub
# ---------------------------------------------------------------------------


class ViewmaxMCPClient:
    """Video clip provider selected explicitly by ``visual_video_provider``.

    ``"kling"`` uses Kling's text-to-video API; ``"runpod"`` uses the
    configured LTX-Video RunPod server. Both credentials may remain configured,
    but this client calls only the selected provider.
    """

    CLIP_DURATION: int = 5
    CLIP_WIDTH: int = 1280
    CLIP_HEIGHT: int = 720

    def __init__(self, provider: Literal["kling", "runpod"] = "kling") -> None:
        if provider not in {"kling", "runpod"}:
            raise ValueError(
                "visual_video_provider must be either 'kling' or 'runpod', "
                f"got {provider!r}."
            )

        self._provider = provider
        self._api_key = os.environ.get("KLING_API_KEY", "").strip()
        self._model = os.environ.get("KLING_MODEL", "kling-v1-6")
        # Serverless endpoint credentials
        self._runpod_endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
        self._runpod_api_key = os.environ.get("RUNPOD_API_KEY", "").strip()

        if provider == "kling":
            if self._api_key:
                logger.info("ViewmaxMCPClient: selected Kling AI API (model=%s)", self._model)
            elif is_production_mode():
                raise ValueError(
                    "visual_video_provider='kling' requires KLING_API_KEY in production mode."
                )
            else:
                logger.warning(
                    "ViewmaxMCPClient: Kling selected but KLING_API_KEY is not configured — "
                    "using placeholder clips."
                )
        elif self._runpod_endpoint_id and self._runpod_api_key:
            logger.info(
                "ViewmaxMCPClient: selected RunPod Serverless endpoint=%s",
                self._runpod_endpoint_id,
            )
        elif is_production_mode():
            raise ValueError(
                "visual_video_provider='runpod' requires RUNPOD_ENDPOINT_ID and "
                "RUNPOD_API_KEY in production mode."
            )
        else:
            logger.warning(
                "ViewmaxMCPClient: RunPod selected but RUNPOD_ENDPOINT_ID or "
                "RUNPOD_API_KEY is not configured — using placeholder clips."
            )

    async def generate_clip(self, prompt: str, duration_seconds: int, seed: int = -1) -> bytes:
        """Generate a clip with only the explicitly selected provider."""
        if self._provider == "kling":
            if not self._api_key:
                return _generate_placeholder_clip_jpeg(prompt)
            try:
                return await self._call_kling(prompt)
            except KlingContentModerationError:
                raise  # let _generate_single_clip handle rephrasing and retry
            except Exception as exc:
                exc_str = str(exc)
                # Billing failures must reach the caller for notification/retry handling.
                if (
                    "402" in exc_str
                    or "401" in exc_str
                    or "payment" in exc_str.lower()
                    or "quota" in exc_str.lower()
                ):
                    raise
                logger.warning(
                    "ViewmaxMCPClient: Kling failed ('%s'): %s — using placeholder.",
                    prompt[:60],
                    exc,
                )
                return _generate_placeholder_clip_jpeg(prompt)

        if not self._runpod_endpoint_id or not self._runpod_api_key:
            return _generate_placeholder_clip_jpeg(prompt)
        try:
            return await self._call_ltx_server(prompt, seed=seed)
        except Exception as exc:
            logger.warning(
                "ViewmaxMCPClient: RunPod failed ('%s'): %s — using placeholder.",
                prompt[:60],
                exc,
            )
            return _generate_placeholder_clip_jpeg(prompt)

    async def _call_kling(self, prompt: str) -> bytes:
        """Generate video via Kling AI API and return MP4 bytes."""
        import httpx  # noqa: PLC0415
        import time  # noqa: PLC0415

        clean_prompt = self._clean_prompt(prompt)
        logger.info("ViewmaxMCPClient: Kling generating '%s'", clean_prompt[:80])

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Submit generation task
        submit_payload = {
            "model_name": self._model,
            "prompt": clean_prompt,
            "negative_prompt": "static, motionless, standing still, frozen, slow, boring, no movement, talking head, static background, low energy, lifeless, watermark, text overlay, subtitle",
            "cfg_scale": 0.5,
            "mode": "std",   # std works for both kling-v1-6 and kling-v2-master
            "duration": str(self.CLIP_DURATION),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.klingai.com/v1/videos/text2video",
                headers=headers,
                json=submit_payload,
            )
            resp.raise_for_status()
            data = resp.json()

        task_id = data["data"]["task_id"]
        logger.info("ViewmaxMCPClient: Kling task submitted %s", task_id)

        # Poll until done (max 10 minutes)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(60):
                await asyncio.sleep(10)
                try:
                    poll = await client.get(
                        f"https://api.klingai.com/v1/videos/text2video/{task_id}",
                        headers=headers,
                    )
                    poll.raise_for_status()
                    result = poll.json()
                    status = result["data"]["task_status"]
                    logger.info("ViewmaxMCPClient: Kling task %s status=%s (attempt %d)", task_id, status, attempt + 1)

                    if status == "succeed":
                        video_url = result["data"]["task_result"]["videos"][0]["url"]
                        logger.info("ViewmaxMCPClient: downloading from %s", video_url[:80])
                        video_resp = await client.get(video_url, timeout=60.0)
                        video_resp.raise_for_status()
                        logger.info("ViewmaxMCPClient: received %d bytes of MP4", len(video_resp.content))
                        return video_resp.content
                    elif status == "failed":
                        msg = result["data"].get("task_status_msg", "")
                        if "risk control" in msg.lower():
                            raise KlingContentModerationError(
                                f"Kling task {task_id} failed: {msg}"
                            )
                        raise RuntimeError(f"Kling task {task_id} failed: {msg}")
                except (KlingContentModerationError, RuntimeError):
                    # The task has reached a terminal failure state.  Do not
                    # keep polling the same task: callers can rephrase a
                    # moderated prompt or handle the provider failure.
                    raise
                except Exception as poll_exc:
                    logger.warning("ViewmaxMCPClient: poll attempt %d failed (%s) — retrying", attempt + 1, poll_exc)
                    continue

        raise RuntimeError(f"Kling task {task_id} timed out after 10 minutes")

    async def _call_ltx_server(self, prompt: str, seed: int = -1) -> bytes:
        """Submit a job to the RunPod Serverless endpoint and poll until done.

        Uses the RunPod REST API:
            POST /v2/{endpoint_id}/run        — submit, returns job_id
            GET  /v2/{endpoint_id}/status/{id} — poll; terminal statuses:
                                                  COMPLETED / FAILED / CANCELLED
        The handler returns {"mp4_b64": "<base64>"} which is decoded here.
        """
        import base64 as _b64  # noqa: PLC0415
        import httpx  # noqa: PLC0415

        clean_prompt = self._clean_prompt(prompt)
        logger.info("ViewmaxMCPClient: RunPod Serverless generating '%s' seed=%d", clean_prompt[:80], seed)

        base_url = f"https://api.runpod.ai/v2/{self._runpod_endpoint_id}"
        headers = {
            "Authorization": f"Bearer {self._runpod_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": {
                "prompt": clean_prompt,
                "negative_prompt": (
                    "worst quality, inconsistent motion, blurry, jittery, "
                    "distorted, watermark, text, static, low quality"
                ),
                "num_frames": 97,
                "width": 1280,
                "height": 720,
                "frame_rate": 25,
                "seed": seed,
            }
        }

        # Submit job (retry up to 6 times for transient network errors)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for submit_attempt in range(6):
                try:
                    resp = await client.post(
                        f"{base_url}/run",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    break
                except Exception as exc:
                    if submit_attempt < 5:
                        logger.warning(
                            "ViewmaxMCPClient: submit attempt %d failed (%s) — retrying in 15s",
                            submit_attempt + 1, exc,
                        )
                        await asyncio.sleep(15)
                    else:
                        raise

            job_id = resp.json()["id"]
            logger.info("ViewmaxMCPClient: RunPod Serverless job submitted %s", job_id)

            # Poll until terminal status (max 20 minutes = 120 × 10s)
            # Use a separate client with a longer timeout for polling — the COMPLETED
            # response includes the full base64 MP4 (~2MB) which can take >30s to receive.
            async with httpx.AsyncClient(timeout=180.0) as poll_client:
                for attempt in range(120):
                    await asyncio.sleep(10)
                    try:
                        poll = await poll_client.get(
                            f"{base_url}/status/{job_id}",
                            headers=headers,
                        )
                        poll.raise_for_status()
                    except Exception as poll_exc:
                        logger.warning(
                            "ViewmaxMCPClient: poll attempt %d failed (%s) — retrying",
                            attempt + 1, poll_exc,
                        )
                        continue

                    result = poll.json()
                    status = result.get("status", "")
                    logger.info(
                        "ViewmaxMCPClient: RunPod job %s status=%s (attempt %d)",
                        job_id, status, attempt + 1,
                    )

                    if status == "COMPLETED":
                        output = result.get("output", {})
                        mp4_b64 = output.get("mp4_b64", "")
                        if not mp4_b64:
                            raise RuntimeError(
                                f"RunPod job {job_id} completed but output missing mp4_b64"
                            )
                        mp4_bytes = _b64.b64decode(mp4_b64)
                        logger.info(
                            "ViewmaxMCPClient: RunPod received %d bytes of MP4", len(mp4_bytes)
                        )
                        return mp4_bytes

                    if status in {"FAILED", "CANCELLED"}:
                        error = result.get("error", "unknown error")
                        raise RuntimeError(
                            f"RunPod job {job_id} {status.lower()}: {error}"
                        )

                    # IN_QUEUE / IN_PROGRESS — keep polling

        raise RuntimeError(f"RunPod job {job_id} timed out after 20 minutes")

    @staticmethod
    def _clean_prompt(raw_prompt: str) -> str:
        """Clean the prompt for LTX-Video generation.

        The prompt already contains specific script content from _build_scene_prompt_variant.
        Just do light cleanup — strip annotations and truncate.
        """
        import re as _re  # noqa: PLC0415

        # Strip annotation tags like [pause], [emphasis]
        clean = _re.sub(r"\[/?[a-zA-Z]+\]", "", raw_prompt).strip()

        # Collapse multiple spaces
        clean = " ".join(clean.split())

        # Truncate to 350 chars — LTX-Video handles up to ~400 well.
        # Front-loaded character descriptions get full weight from the model.
        return clean[:350]


# ---------------------------------------------------------------------------
# ClipResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    """Result of generating (or substituting) a single video clip.

    Attributes:
        segment_index: Zero-based index of the script segment this clip
            corresponds to.
        mp4_bytes: Raw MP4 bytes *or* raw JPEG bytes if ``is_fallback`` is
            ``True`` (ffmpeg handles JPEG inputs specially).
        is_fallback: ``True`` when the clip was produced by the static-image
            fallback path (all Viewmax retries were exhausted).
    """

    segment_index: int
    mp4_bytes: bytes
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Internal helpers — script parsing
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """A single parsed script segment."""

    title: str
    body: str


def _parse_segments(content: str) -> list[_Segment]:
    """Split *content* into segments using Markdown headings.

    Splits on ``## `` or ``### `` headings.  If no headings are found, the
    content is divided into three equal thirds with labels ``hook``, ``body``,
    and ``cta``.

    Args:
        content: Full script Markdown content.

    Returns:
        Non-empty list of :class:`_Segment` objects.
    """
    # Match lines that start with ## or ### followed by space
    heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    if not matches:
        # No headings — split into equal thirds
        total = len(content)
        third = max(total // 3, 1)
        return [
            _Segment(title="hook", body=content[:third].strip()),
            _Segment(title="body", body=content[third : 2 * third].strip()),
            _Segment(title="cta", body=content[2 * third :].strip()),
        ]

    segments: list[_Segment] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        # Body runs from end of this heading line to start of next heading (or EOF)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        segments.append(_Segment(title=title, body=body))

    return segments


def _build_scene_prompt(segment: _Segment, style_profile: StyleProfile) -> str:
    """Derive a Viewmax scene prompt for *segment* (first clip)."""
    body_preview = segment.body[:100]
    patterns_str = ", ".join(style_profile.visual_style.composition_patterns)
    return f"{segment.title} visual scene: {body_preview}. Style: {patterns_str}"


async def _rephrase_prompt_for_moderation(original_prompt: str) -> str:
    """Ask Claude to rewrite a prompt that was blocked by Kling's risk control system.

    Strips copyrighted character names and replaces them with visual descriptions
    so the same scene idea passes Kling's content moderation.

    Args:
        original_prompt: The prompt that was rejected.

    Returns:
        A rephrased prompt, or the original if Claude is unavailable.
    """
    try:
        client = build_claude_client()
        prompt = (
            "You are an AI video prompt editor. Rewrite video prompts that were "
            "rejected by a video generator's content moderation system.\n\n"
            "RULES:\n"
            "1. Replace every named fictional character, franchise name, or copyrighted IP "
            "with a pure visual description. Examples:\n"
            "   - Superman / Man of Steel → 'a caped hero in a blue suit with golden solar energy'\n"
            "   - Thor / Odinson → 'a blond-bearded warrior in silver armor with a glowing war hammer'\n"
            "   - Batman / Dark Knight → 'a dark-armored figure in a black cape and cowl'\n"
            "   - Iron Man / Tony Stark → 'a hero in red-and-gold powered armor'\n"
            "   - Spider-Man → 'a hero in a red-and-blue web-patterned suit'\n"
            "   - Hulk / Bruce Banner → 'a massive green-skinned giant in torn shorts'\n"
            "   - Goku / Vegeta → 'a spiky-haired fighter in an orange uniform with golden aura'\n"
            "   - Captain America → 'a hero in a blue star-spangled suit with a round shield'\n"
            "   - Mjolnir → 'glowing war hammer'\n"
            "   - Vibranium → 'advanced alloy'\n"
            "   - Arc reactor → 'glowing chest device'\n"
            "   - Metropolis / Gotham → 'a futuristic city'\n"
            "2. Remove or soften any explicit violence — replace 'destroys', 'kills', 'attacks' "
            "with 'clashes with', 'confronts', 'faces off against'.\n"
            "3. Keep the camera direction, motion, and lighting descriptions exactly the same.\n"
            "4. Return only the rewritten prompt, no explanation.\n"
            "5. CRITICAL: the output must contain ZERO copyrighted names, character names, "
            "franchise names, or weapon names.\n\n"
            f"Rewrite this prompt removing all copyrighted names and softening "
            f"any explicit violence:\n\n{original_prompt}"
        )
        result = await client.complete(prompt, max_tokens=300)
        rephrased = result.strip().strip('"\'')
        logger.info("Prompt rephrased for moderation: %s", rephrased[:80])
        return rephrased[:400]

    except Exception as exc:
        logger.warning("Failed to rephrase prompt via Claude: %s — using original", exc)
        return original_prompt


async def _generate_character_descriptions(script_topic: str, script_body: str) -> dict:
    """Dynamically generate character visual descriptions from the script topic.

    Calls the LLM once per video to create proper, gender-correct, visually
    accurate character descriptions based on the actual characters in the script —
    not a hardcoded list.

    Returns:
        Dict with keys: hero1_name, hero1_desc, hero2_name, hero2_desc
    """
    try:
        client = build_claude_client()
        prompt = (
            f"Video topic: {script_topic}\n"
            f"Script excerpt: {script_body[:300]}\n\n"
            "Identify the TWO main characters in this video topic. For each character, "
            "write a detailed visual description that a video generator can use to "
            "render them accurately. NEVER use copyrighted names — describe only by appearance.\n\n"
            "IMPORTANT: Include gender, body type, hair, costume/armor details, "
            "signature weapon or power visuals, and any distinctive features.\n\n"
            "Format your response EXACTLY like this (4 lines only):\n"
            "HERO1_NAME: [short label like 'amazonian warrior' or 'armored titan']\n"
            "HERO1_DESC: [full visual description, 30-50 words]\n"
            "HERO2_NAME: [short label]\n"
            "HERO2_DESC: [full visual description, 30-50 words]\n\n"
            "Examples:\n"
            "HERO1_NAME: amazonian warrior princess\n"
            "HERO1_DESC: Tall athletic woman with long flowing black hair, golden tiara with red star, red and blue armored corset, silver bracelet gauntlets, golden lasso at hip, tanned skin, fierce determined expression\n"
            "HERO2_NAME: armored purple titan\n"
            "HERO2_DESC: Massive 8-foot tall purple-skinned muscular male titan, bald head with deep chin ridges, golden armored gauntlet on left hand with glowing gems, dark blue and gold battle armor\n\n"
            "Output ONLY the 4 lines, nothing else."
        )
        result = await client.complete(prompt, max_tokens=300)
        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]

        hero1_name = "hero 1"
        hero1_desc = "a powerful superhero in dramatic pose"
        hero2_name = "hero 2"
        hero2_desc = "a powerful warrior in battle stance"

        for line in lines:
            if line.upper().startswith("HERO1_NAME:"):
                hero1_name = line.split(":", 1)[1].strip()
            elif line.upper().startswith("HERO1_DESC:"):
                hero1_desc = line.split(":", 1)[1].strip()
            elif line.upper().startswith("HERO2_NAME:"):
                hero2_name = line.split(":", 1)[1].strip()
            elif line.upper().startswith("HERO2_DESC:"):
                hero2_desc = line.split(":", 1)[1].strip()

        logger.info(
            "Character descriptions generated: Hero1='%s' Hero2='%s'",
            hero1_name, hero2_name,
        )
        return {
            "hero1_name": hero1_name,
            "hero1_desc": hero1_desc,
            "hero2_name": hero2_name,
            "hero2_desc": hero2_desc,
        }
    except Exception as exc:
        logger.warning("Character description generation failed: %s — using generic fallback", exc)
        return {
            "hero1_name": "hero 1",
            "hero1_desc": "a powerful superhero in dramatic pose with distinctive costume",
            "hero2_name": "hero 2",
            "hero2_desc": "a powerful warrior in battle stance with unique armor",
        }


async def _generate_video_prompt_with_claude(
    segment: _Segment,
    clip_num: int,
    total_clips: int,
    scene_state: Optional[dict] = None,
    script_topic: str = "",
    character_descs: Optional[dict] = None,
) -> str:
    """Use Claude to generate an optimized video prompt with scene continuity.

    Tracks scene phase (opening/escalation/climax/resolution) across clips,
    varies character emotions, and uses proper filmmaking terminology.
    """
    import re as _re  # noqa: PLC0415

    body = _re.sub(r"\[/?[a-zA-Z]+\]", "", segment.body).strip()
    title = segment.title.strip()
    words = body.split()

    if total_clips > 1:
        chunk_size = max(1, len(words) // total_clips)
        start = clip_num * chunk_size
        end = start + chunk_size if clip_num < total_clips - 1 else len(words)
        body_chunk = " ".join(words[start:end])
    else:
        body_chunk = body

    if scene_state is None:
        scene_state = {}

    total_position = scene_state.get("global_clip_num", clip_num)
    total_global = scene_state.get("total_global_clips", total_clips)
    progress = total_position / max(1, total_global - 1)

    if progress < 0.25:
        scene_phase = "opening: storm clouds gathering, city intact, tension building, dusk"
        damage_level = "city intact, glass windows reflecting storm clouds"
        emotion_guidance = "characters sizing each other up — narrow eyes, battle stance, restrained power"
    elif progress < 0.5:
        scene_phase = "escalation: rain pouring, first impacts, blue lightning tears across sky"
        damage_level = "windows shattered on nearby buildings, cracks in pavement"
        emotion_guidance = "first contact — surprise, determination, adrenaline"
    elif progress < 0.75:
        scene_phase = "climax: heavy destruction, fires spreading, smoke filling air"
        damage_level = "buildings partially collapsed, debris everywhere, fires burning"
        emotion_guidance = "intense fury, gritted teeth, screaming with effort, pushing limits"
    else:
        scene_phase = "resolution: rubble-filled crater, embers floating, dust settling"
        damage_level = "massive crater, collapsed skyscrapers, embers floating"
        emotion_guidance = "exhausted but standing, determination in eyes, battle-worn"

    camera_options = [
        "epic aerial drone shot pushing forward slowly",
        "extreme close-up slow motion dolly-in",
        "drone camera rapidly pulling backward through explosion",
        "low-angle tracking shot circling the heroes",
        "handheld shaky cam rushing alongside the action",
        "crane shot slowly descending into the destruction",
        "orbit shot circling both warriors",
        "whip pan following the projectile",
    ]
    camera = camera_options[clip_num % len(camera_options)]

    # Master style block — injected once, keeps every clip stylistically consistent
    _MASTER_STYLE = (
        "Ultra realistic live-action Hollywood blockbuster IMAX HDR PBR "
        "ray-traced reflections volumetric lighting cinematic color grading "
        "24fps realistic destruction physics shallow depth of field dynamic "
        "camera movement film grain high temporal consistency."
    )

    # Use dynamic character descriptions if available, else generic fallback
    if character_descs is None:
        character_descs = {
            "hero1_name": "hero 1",
            "hero1_desc": "a powerful superhero in dramatic pose",
            "hero2_name": "hero 2",
            "hero2_desc": "a powerful warrior in battle stance",
        }

    h1_desc = character_descs["hero1_desc"]
    h2_desc = character_descs["hero2_desc"]
    h1_name = character_descs["hero1_name"]
    h2_name = character_descs["hero2_name"]

    system_prompt = f"""You are an expert AI video prompt engineer writing Hollywood storyboard prompts for a video generator.

VIDEO TOPIC: {script_topic if script_topic else "superhero battle"}

CHARACTER 1 ({h1_name}): {h1_desc}
CHARACTER 2 ({h2_name}): {h2_desc}

IMPORTANT: Every prompt MUST open by describing the character's full appearance from the description above. Never use names, pronouns, or vague references like "the hero." Always include gender, costume details, and distinctive features so the video generator renders the correct character.

ENVIRONMENT: Futuristic city. Dark storm clouds, rain-soaked streets, glass skyscrapers. Damage escalates: intact → windows crack → buildings shake → skyscrapers collapse → streets split → crater forms.

MASTER STYLE (append to every prompt unchanged):
{_MASTER_STYLE}

RULES:
1. ONE action per clip — one character doing one clear thing
2. ALTERNATE between Character 1 and Character 2 every clip
3. Vary actions — no two clips should show the same move
4. TRANSITION PHRASES (use each once in order): Clip2="The collision detonates..." Clip3="Before the smoke clears..." Clip4="Emerging from the dust cloud..." Clip5="In the battle's aftermath..." Clip6+="In the silence that follows..."
5. EMOTION WORDS — forbidden: "widen". Use: jaw tightens / gaze sharpens / expression hardens / eyes blaze / grimaces / grins with contempt / unwavering stare / refuses to yield
6. Append master style tag at end
7. Max 280 characters before style tag, present tense, no dialogue
8. CRITICAL: output must contain ZERO character names, franchise names, or copyrighted IP — describe ONLY by physical appearance"""

    # Alternate between Character 1 and Character 2
    if clip_num % 6 == 2 or clip_num % 6 == 4:
        clip_focus = "EQUAL CLASH — show both characters at the moment of impact"
        focus_instruction = f"Open with Character 1's full description ({h1_desc[:50]}...), mention Character 2 in the action"
    elif clip_num % 2 == 0:
        clip_focus = f"CHARACTER 1 ({h1_name}) — show their offensive/defensive action"
        focus_instruction = f"Open with: {h1_desc}"
    else:
        clip_focus = f"CHARACTER 2 ({h2_name}) — show their offensive/defensive action"
        focus_instruction = f"Open with: {h2_desc}"

    user_prompt = f"""VIDEO TOPIC: {script_topic if script_topic else "superhero battle"}
Segment: "{title}"
Script action (FOLLOW THIS CLOSELY): "{body_chunk}"
Scene phase: {scene_phase}
Damage: {damage_level}
Emotion (ONE word, never "widen"): {emotion_guidance}
Camera: {camera}
Focus this clip: {clip_focus}
Character description to use: {focus_instruction}
Clip {clip_num + 1}/{total_clips} (global {total_position + 1}/{total_global})
{"Transition: use clip " + str(min(clip_num, 5) + 2) + " phrase" if clip_num > 0 else "Opening clip — no transition"}

Start by describing the character's FULL appearance, then the action.
Write ONE prompt. End with the master style tag."""

    try:
        client = build_claude_client()
        result = await client.complete(
            f"{system_prompt}\n\n{user_prompt}",
            max_tokens=200,
        )
        prompt = result.strip().strip('"\'')
        logger.info("Claude prompt clip %d/%d: %s", clip_num + 1, total_clips, prompt[:80])
        return prompt[:400]

    except Exception as exc:
        logger.warning("Claude prompt failed for '%s' clip %d: %s — fallback", title[:30], clip_num, exc)
        return _build_scene_prompt_variant_fallback(segment, clip_num, total_clips)

def _build_scene_prompt_variant_fallback(
    segment: _Segment,
    clip_num: int,
    total_clips: int,
) -> str:
    """Fallback template prompt when Claude is unavailable."""
    import re as _re  # noqa: PLC0415
    body = _re.sub(r"\[/?[a-zA-Z]+\]", "", segment.body).strip()
    title = segment.title.strip()
    words = body.split()
    if total_clips > 1:
        chunk_size = max(1, len(words) // total_clips)
        start = clip_num * chunk_size
        end = start + chunk_size if clip_num < total_clips - 1 else len(words)
        body_chunk = " ".join(words[start:end])[:150]
    else:
        body_chunk = body[:150]

    templates = [
        f"Dynamic action: {title}. {body_chunk}. Fast motion, cinematic 4K.",
        f"Battle scene: {body_chunk}. Explosion on impact, debris flying. 4K.",
        f"Hero in motion: {body_chunk}. Camera tracking, dramatic lighting. 4K.",
        f"Power unleashed: {title}. {body_chunk}. Energy surging, cinematic.",
        f"Epic climax: {body_chunk}. Camera pulls back, golden light. IMAX 4K.",
    ]
    return templates[clip_num % len(templates)][:300]


# ---------------------------------------------------------------------------
# Internal helpers — fallback image
# ---------------------------------------------------------------------------


def _make_fallback_jpeg(title: str) -> bytes:
    """Create a 1920×1080 JPEG fallback image with *title* centred on a black
    background.

    Args:
        title: Text to render at the centre of the frame (the segment title).

    Returns:
        Raw JPEG bytes.
    """
    img = Image.new("RGB", (_FALLBACK_WIDTH, _FALLBACK_HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Use the default PIL bitmap font — no external font file required
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=72)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Calculate text bounding box for centring
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow versions
        text_w, text_h = draw.textsize(title, font=font)  # type: ignore[attr-defined]

    x = (_FALLBACK_WIDTH - text_w) // 2
    y = (_FALLBACK_HEIGHT - text_h) // 2
    draw.text((x, y), title, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers — thumbnail
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a CSS hex color string (``#RRGGBB``) to an RGB tuple.

    Falls back to black on any parse error.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return (0, 0, 0)


def _contrasting_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return a colour that contrasts with *rgb* (simple luminance flip)."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def _extract_thumbnail_from_clips(clip_results: list[ClipResult]) -> Optional[bytes]:
    """Extract the best frame from the generated clips to use as thumbnail.

    Picks the middle frame from the first non-fallback MP4 clip.
    Falls back to None if extraction fails (caller uses text overlay instead).

    Args:
        clip_results: List of generated clips.

    Returns:
        JPEG bytes of the extracted frame, or None if extraction fails.
    """
    import subprocess as _sp  # noqa: PLC0415
    import shutil as _sh  # noqa: PLC0415

    ffmpeg_bin = (
        _sh.which("ffmpeg")
        or "/opt/homebrew/bin/ffmpeg"
        or "/usr/local/bin/ffmpeg"
        or "ffmpeg"
    )

    # Find first real (non-fallback) clip
    real_clips = [c for c in clip_results if not c.is_fallback]
    if not real_clips:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            clip_path = tmp_path / "clip.mp4"
            clip_path.write_bytes(real_clips[0].mp4_bytes)
            thumb_path = tmp_path / "thumb.jpg"

            # Extract frame at 50% of the clip duration (middle frame)
            result = _sp.run(
                [
                    ffmpeg_bin, "-y",
                    "-i", str(clip_path),
                    "-vf", "thumbnail,scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                    "-frames:v", "1",
                    str(thumb_path),
                ],
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0 or not thumb_path.exists():
                # Try simpler extraction without thumbnail filter
                result2 = _sp.run(
                    [
                        ffmpeg_bin, "-y",
                        "-i", str(clip_path),
                        "-ss", "00:00:02",
                        "-frames:v", "1",
                        "-vf", "scale=1280:720",
                        str(thumb_path),
                    ],
                    capture_output=True,
                    timeout=30,
                )
                if result2.returncode != 0 or not thumb_path.exists():
                    logger.warning("Thumbnail extraction failed: %s", result2.stderr[-200:].decode("utf-8", errors="replace"))
                    return None

            thumb_bytes = thumb_path.read_bytes()
            logger.info("Thumbnail extracted from first clip (%d bytes)", len(thumb_bytes))

            # Ensure under 2MB
            if len(thumb_bytes) > _THUMBNAIL_MAX_BYTES:
                img = Image.open(io.BytesIO(thumb_bytes))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                thumb_bytes = buf.getvalue()

            return thumb_bytes

    except Exception as exc:
        logger.warning("Thumbnail extraction from clip failed: %s", exc)
        return None


async def _generate_ai_thumbnail(
    title: str,
    script_body: str,
) -> Optional[bytes]:
    """Generate a professional YouTube thumbnail using Pollinations.AI + Pillow text overlay.

    Steps:
    1. Claude generates a punchy 2-line thumbnail text (title + hook)
    2. Pollinations.AI (flux-pro, free) generates the background image
    3. Pillow overlays bold text with drop shadow on the image

    Args:
        title: Video title / segment title.
        script_body: Script body text for context.

    Returns:
        JPEG bytes of the thumbnail, or None if generation fails.
    """
    import httpx  # noqa: PLC0415

    # ---- Step 1: Claude generates thumbnail text -------------------------
    top_text = title.upper()[:30]  # fallback
    bottom_text = "WHO WINS?"       # fallback

    try:
        client = build_claude_client()
        prompt = (
            f"Video title: {title}\nScript: {script_body[:200]}\n\n"
            "Write 2 lines of YouTube thumbnail text that makes viewers want to click.\n"
            "Line 1: 3-4 word bold title (ALL CAPS, exciting, e.g. 'THOR VS SUPERMAN')\n"
            "Line 2: 3-5 word hook question (e.g. 'WHO ACTUALLY WINS?')\n"
            "Output ONLY the 2 lines, nothing else."
        )
        result = await client.complete(prompt, max_tokens=60)
        lines = result.strip().split("\n")
        if len(lines) >= 2:
            top_text = lines[0].strip().upper()[:35]
            bottom_text = lines[1].strip().upper()[:40]
        elif len(lines) == 1:
            top_text = lines[0].strip().upper()[:35]
    except Exception as exc:
        logger.warning("Claude thumbnail text generation failed: %s", exc)

    # ---- Step 2: Pollinations.AI generates cinematic background at native 1280x720
    # No upscaling needed — Pollinations renders at exact requested resolution.
    import urllib.parse  # noqa: PLC0415
    import random as _random  # noqa: PLC0415

    image_prompt = (
        f"Photorealistic cinematic movie still, ultra realistic, NOT cartoon, NOT animated, "
        f"YouTube thumbnail for: {title}, "
        f"split screen composition, two powerful characters facing off in dramatic poses, "
        f"dramatic orange and blue lighting, lens flare, rain drops, "
        f"destroyed city skyline background, IMAX cinematography, "
        f"shallow depth of field, film grain, 8K detail, "
        f"like a real Hollywood movie poster screenshot, no text anywhere"
    )
    encoded_prompt = urllib.parse.quote(image_prompt)
    thumb_seed = _random.randint(0, 99999)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1280&height=720&model=flux-pro&nologo=true&enhance=true&seed={thumb_seed}"
    )

    img_bytes = None
    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
            img_bytes = resp.content
        logger.info("Pollinations thumbnail generated (%d bytes)", len(img_bytes))
    except Exception as exc:
        logger.warning("Pollinations image generation failed: %s", exc)
        return None

    # ---- Step 3: Pillow overlays bold text with drop shadow --------------
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Pollinations delivers at native 1280x720 — no resize needed.
        # Only resize if the image isn't already the right dimensions.
        if img.size != (1280, 720):
            img = img.resize((1280, 720), Image.LANCZOS)
        draw = ImageDraw.Draw(img)

        # Try to load a bold font, fall back to default
        def _load_font(size: int) -> ImageFont.FreeTypeFont:
            for font_name in [
                "Impact.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
                "/System/Library/Fonts/Supplemental/Impact.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
            ]:
                try:
                    return ImageFont.truetype(font_name, size=size)
                except (IOError, OSError):
                    continue
            return ImageFont.load_default()

        font_top = _load_font(90)
        font_bottom = _load_font(60)

        def _draw_text_with_shadow(
            draw: ImageDraw.ImageDraw,
            text: str,
            font: ImageFont.FreeTypeFont,
            y: int,
            fill: tuple,
            shadow_offset: int = 4,
        ) -> None:
            """Draw text centered with drop shadow and black outline."""
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
            except AttributeError:
                text_w, _ = draw.textsize(text, font=font)  # type: ignore
            x = (1280 - text_w) // 2

            # Draw thick black outline
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))

            # Draw drop shadow
            draw.text((x + shadow_offset, y + shadow_offset), text, font=font, fill=(0, 0, 0, 180))

            # Draw main text
            draw.text((x, y), text, font=font, fill=fill)

        # Top text — bright yellow, upper portion
        _draw_text_with_shadow(draw, top_text, font_top, y=40, fill=(255, 220, 0))

        # Bottom text — white, lower portion
        _draw_text_with_shadow(draw, bottom_text, font_bottom, y=620, fill=(255, 255, 255))

        # Save as high quality JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        result = buf.getvalue()

        logger.info("AI thumbnail with text overlay generated (%d bytes)", len(result))
        return result

    except Exception as exc:
        logger.warning("Pillow text overlay failed: %s", exc)
        # Return the plain Pollinations image without text
        return img_bytes if len(img_bytes) < _THUMBNAIL_MAX_BYTES else None


def _make_thumbnail_jpeg(style_profile: StyleProfile, title: str) -> bytes:
    """Generate a 1280×720 JPEG thumbnail using the StyleProfile.

    Layout
    ------
    * Background filled with the first dominant color (black if none).
    * White text overlay at the ``text_overlay_position`` (top / center / bottom).
    * A placeholder rectangle in a contrasting color to represent subject framing.

    Args:
        style_profile: Source of dominant colors, text overlay position, and
            subject framing metadata.
        title: Text to render as the overlay (typically the video title).

    Returns:
        JPEG bytes guaranteed to be < 2 MB (compressed at quality 85, then 70
        if still too large).
    """
    tc = style_profile.thumbnail_composition

    # --- Background color ------------------------------------------------
    bg_rgb: tuple[int, int, int] = (0, 0, 0)
    if tc.dominant_colors:
        bg_rgb = _hex_to_rgb(tc.dominant_colors[0])

    img = Image.new("RGB", (_THUMBNAIL_WIDTH, _THUMBNAIL_HEIGHT), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    # --- Subject-framing placeholder rectangle ---------------------------
    contrast_rgb = _contrasting_color(bg_rgb)
    rect_margin = 80
    draw.rectangle(
        [
            (rect_margin, rect_margin),
            (_THUMBNAIL_WIDTH - rect_margin, _THUMBNAIL_HEIGHT - rect_margin),
        ],
        outline=contrast_rgb,
        width=6,
    )

    # --- Text overlay ----------------------------------------------------
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=60)
    except (IOError, OSError):
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(title, font=font)  # type: ignore[attr-defined]

    padding = 40
    position = tc.text_overlay_position.lower() if tc.text_overlay_position else "center"

    x = (_THUMBNAIL_WIDTH - text_w) // 2

    if position == "top":
        y = padding
    elif position == "bottom":
        y = _THUMBNAIL_HEIGHT - text_h - padding
    else:  # center (default)
        y = (_THUMBNAIL_HEIGHT - text_h) // 2

    draw.text((x, y), title, fill=(255, 255, 255), font=font)

    # --- Compress to stay < 2 MB -----------------------------------------
    for quality in (85, 70):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) < _THUMBNAIL_MAX_BYTES:
            return data

    # Should not happen for a 1280×720 JPEG at quality=70, but return anyway
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers — version scanning
# ---------------------------------------------------------------------------


async def _next_version(
    asset_store: Asset_Store,
    video_id: str,
    subfolder: SubFolder,
    base_name: str,
    extension: str,
) -> int:
    """Return version 1 — each pipeline run uses a fresh video_id, so v1 is always free."""
    return 1


# ---------------------------------------------------------------------------
# Visual_Generator
# ---------------------------------------------------------------------------


class Visual_Generator:
    """Generates per-segment video clips via Viewmax, compiles them into an
    MP4, creates a thumbnail, and writes everything to Asset_Store.

    Args:
        viewmax_client: A :class:`ViewmaxClient`-compatible object (production
            or test double).
        asset_store: :class:`~pipeline.asset_store.Asset_Store` instance.
        content_calendar: :class:`~pipeline.content_calendar.Content_Calendar`
            instance used to update pipeline status.
        notifier: :class:`~pipeline.notifier.Notifier` instance for failure
            alerts.
    """

    def __init__(
        self,
        viewmax_client: ViewmaxClient,
        asset_store: Asset_Store,
        content_calendar: Content_Calendar,
        notifier: Notifier,
    ) -> None:
        self._viewmax = viewmax_client
        self._asset_store = asset_store
        self._calendar = content_calendar
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        script: Script,
        narration: NarrationAsset,
        style_profile: StyleProfile,
        video_id: str,
    ) -> VisualAsset:
        """Generate a compiled MP4 and thumbnail JPEG for *video_id*.

        Steps
        -----
        1. Parse script segments.
        2. Derive one scene prompt per segment.
        3. Generate each clip via Viewmax (with retry + fallback).
        4. Write clips to a temp directory and compile via ffmpeg.
        5. Generate thumbnail JPEG from StyleProfile.
        6. Upload MP4 and thumbnail to Asset_Store.
        7. Update Content_Calendar status to ``Visuals Ready``.

        Args:
            script: Approved :class:`~pipeline.models.Script`.
            narration: :class:`~pipeline.models.NarrationAsset` with MP3 path.
            style_profile: :class:`~pipeline.models.StyleProfile` with visual
                composition and thumbnail data.
            video_id: Pipeline-assigned video identifier.

        Returns:
            :class:`~pipeline.models.VisualAsset` with Drive URLs set.

        Raises:
            VisualGeneratorError: On unrecoverable failures (ffmpeg error, all
                clips failed to upload, etc.).
        """
        logger.info("Visual_Generator.generate started for video_id=%s", video_id)

        # ---- 1. Parse segments ------------------------------------------
        segments = _parse_segments(script.content)
        logger.debug(
            "Parsed %d segments for video_id=%s: %s",
            len(segments),
            video_id,
            [s.title for s in segments],
        )

        # ---- 2. Get actual narration duration + calculate per-segment durations ----
        wpm = 150
        total_words = sum(len(s.body.split()) for s in segments)

        # Try to get actual narration duration from MP3
        actual_audio_seconds: float = 0.0
        try:
            mp3_bytes = await self._asset_store.read(
                video_id=video_id,
                subfolder=SubFolder.NARRATION,
                filename=narration.mp3_path,
            )
            # Get duration from MP3 header using mutagen or estimate from file size
            try:
                import mutagen.mp3 as _mp3  # noqa: PLC0415
                import io as _io  # noqa: PLC0415
                audio = _mp3.MP3(fileobj=_io.BytesIO(mp3_bytes))
                actual_audio_seconds = audio.info.length
                logger.info("Visual_Generator: actual narration duration=%.1fs", actual_audio_seconds)
            except Exception:
                # Fallback: estimate from file size (128kbps MP3 = 16KB/s)
                actual_audio_seconds = len(mp3_bytes) / 16000
                logger.info("Visual_Generator: estimated narration duration=%.1fs from file size", actual_audio_seconds)
        except Exception as exc:
            logger.warning("Visual_Generator: could not get narration duration (%s), using word count estimate", exc)

        # Use actual duration if available, otherwise estimate from word count
        if actual_audio_seconds > 0:
            total_audio_seconds = actual_audio_seconds
        else:
            total_audio_seconds = max(len(segments) * 5, (total_words / wpm) * 60)

        segment_durations = []
        for seg in segments:
            seg_words = len(seg.body.split())
            seg_duration = max(3.0, (seg_words / wpm) * 60)
            segment_durations.append(seg_duration)
        raw_sum = sum(segment_durations)
        segment_durations = [d * total_audio_seconds / raw_sum for d in segment_durations]
        logger.info(
            "Visual_Generator: %d segments, total duration=%.1fs, cutting every ~10s",
            len(segments), total_audio_seconds,
        )

        # ---- 3. Generate clips — one per segment ------
        # Extract topic from script: use first heading or first line as context
        # for Claude to generate relevant video prompts
        script_topic = ""
        for line in script.content.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                script_topic = line[:100]
                break

        clip_results = await self._generate_all_clips(
            segments, style_profile, video_id,
            segment_durations=segment_durations,
            script_topic=script_topic,
        )

        # ---- 4. Build per-clip durations (even split within each segment) --
        # Match cut interval to actual clip duration: 97 frames ÷ 25fps = 3.88s
        _CUT_EVERY_S: float = 3.88  # matches num_frames=97 at frame_rate=25
        per_clip_durations: list[float] = []
        for seg_dur in segment_durations:
            n = max(1, round(seg_dur / _CUT_EVERY_S))
            per_clip_durations.extend([seg_dur / n] * n)

        logger.info(
            "Visual_Generator: %d total clips, durations: %s",
            len(per_clip_durations),
            [f"{d:.1f}s" for d in per_clip_durations],
        )

        # ---- 5. Compile MP4 with ffmpeg ----------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mp4_bytes = await self._compile_video(
                clip_results=clip_results,
                segments=segments,
                segment_durations=per_clip_durations,  # one duration per clip
                narration_mp3_path=narration.mp3_path,
                tmp_path=tmp_path,
                video_id=video_id,
            )

            # ---- 5. Generate thumbnail — AI image + text overlay -----------
            video_title = segments[0].title if segments else video_id
            script_body = segments[0].body if segments else ""
            thumbnail_bytes = await _generate_ai_thumbnail(video_title, script_body) or \
                              _make_thumbnail_jpeg(style_profile, video_title)

            # ---- 6. Determine version and upload --------------------------
            mp4_version = await _next_version(
                self._asset_store, video_id, SubFolder.VIDEOS, video_id, "mp4"
            )
            thumb_version = await _next_version(
                self._asset_store, video_id, SubFolder.THUMBNAILS, video_id, "jpg"
            )

            mp4_filename = f"{video_id}_v{mp4_version}.mp4"
            thumb_filename = f"{video_id}_v{thumb_version}.jpg"

            logger.info(
                "Uploading %s and %s for video_id=%s",
                mp4_filename,
                thumb_filename,
                video_id,
            )

            mp4_url = await self._asset_store.write(
                video_id, SubFolder.VIDEOS, mp4_filename, mp4_bytes
            )
            thumbnail_url = await self._asset_store.write(
                video_id, SubFolder.THUMBNAILS, thumb_filename, thumbnail_bytes
            )

        # ---- 7. Update Content_Calendar status ---------------------------
        await self._calendar.update_status(video_id, PipelineStatus.VISUALS_READY)
        logger.info(
            "Visual_Generator.generate complete for video_id=%s "
            "(mp4=%s, thumbnail=%s)",
            video_id,
            mp4_filename,
            thumb_filename,
        )

        return VisualAsset(
            video_id=video_id,
            version=mp4_version,
            mp4_path=f"videos/{mp4_filename}",
            thumbnail_path=f"thumbnails/{thumb_filename}",
            mp4_url=mp4_url,
            thumbnail_url=thumbnail_url,
            created_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Task 11.1 — Per-clip generation with retry + fallback
    # ------------------------------------------------------------------

    async def _generate_all_clips(
        self,
        segments: list[_Segment],
        style_profile: StyleProfile,
        video_id: str,
        segment_durations: Optional[list[float]] = None,
        script_topic: str = "",
    ) -> list[ClipResult]:
        """Generate enough YouTube clips to cover every segment fully.

        Target cut rate: one new clip every ~10 seconds.
        For a 40-second segment that means 4 different YouTube clips,
        each 10 seconds, no repetition within the segment.

        Different search queries are used for each clip within a segment
        (varying keywords) so the visual variety matches the narration pacing.
        """
        _CUT_EVERY_S: float = 3.88  # matches num_frames=97 at frame_rate=25, no looping

        results: list[ClipResult] = []

        # Generate dynamic character descriptions from the script topic — called
        # once per video so all clips share consistent character descriptions.
        script_body = segments[0].body if segments else ""
        character_descs = await _generate_character_descriptions(script_topic, script_body)

        # Calculate total clips across all segments for scene state continuity
        total_clips_all_segments = sum(
            max(1, round((segment_durations[i] if segment_durations and i < len(segment_durations) else 30.0) / _CUT_EVERY_S))
            for i in range(len(segments))
        )
        global_clip_num = 0  # tracks position across all clips for scene phase

        for seg_idx, segment in enumerate(segments):
            seg_duration = (
                segment_durations[seg_idx]
                if segment_durations and seg_idx < len(segment_durations)
                else 30.0
            )
            n_clips = max(1, round(seg_duration / _CUT_EVERY_S))

            logger.info(
                "Visual_Generator: segment '%s' %.1fs → %d clips",
                segment.title, seg_duration, n_clips,
            )

            for clip_num in range(n_clips):
                # Pass scene state so Claude knows the global narrative position
                scene_state = {
                    "global_clip_num": global_clip_num,
                    "total_global_clips": total_clips_all_segments,
                }
                prompt = await _generate_video_prompt_with_claude(
                    segment, clip_num, n_clips, scene_state=scene_state,
                    script_topic=script_topic, character_descs=character_descs,
                )
                global_clip_num += 1

                clip_result = await self._generate_single_clip(
                    segment_index=seg_idx,
                    segment=segment,
                    prompt=prompt,
                    video_id=video_id,
                    seed=-1,  # random per clip for visual variety
                )
                results.append(clip_result)

        return results

    async def _generate_single_clip(
        self,
        segment_index: int,
        segment: _Segment,
        prompt: str,
        video_id: str,
        seed: int = -1,
    ) -> ClipResult:
        """Attempt to generate one clip via Viewmax; fall back to JPEG on exhaustion.

        Retry policy:
        * Up to 3 attempts.
        * On KlingContentModerationError, rephrases the prompt once via Claude
          (replacing copyrighted character names with visual descriptions) before
          retrying — no sleep needed since the error is deterministic.
        * Random delay between other failure attempts drawn from
          ``random.uniform(5, 30)`` seconds.

        Args:
            segment_index: Zero-based segment index (used in log messages).
            segment: The segment being rendered.
            prompt: Pre-built Viewmax scene prompt.
            video_id: Used in log messages.

        Returns:
            :class:`ClipResult` — either a real MP4 or a JPEG fallback.
        """
        last_exc: Optional[Exception] = None
        current_prompt = prompt
        moderation_rephrase_attempted = False

        for attempt in range(1, _CLIP_RETRY_ATTEMPTS + 1):
            try:
                logger.debug(
                    "Viewmax clip attempt %d/%d for segment '%s' (video_id=%s)",
                    attempt,
                    _CLIP_RETRY_ATTEMPTS,
                    segment.title,
                    video_id,
                )
                mp4_bytes = await self._viewmax.generate_clip(
                    prompt=current_prompt,
                    duration_seconds=_CLIP_DEFAULT_DURATION_S,
                    seed=seed,
                )
                logger.debug(
                    "Viewmax clip succeeded for segment '%s' (video_id=%s, attempt=%d)",
                    segment.title,
                    video_id,
                    attempt,
                )
                return ClipResult(
                    segment_index=segment_index,
                    mp4_bytes=mp4_bytes,
                    is_fallback=mp4_bytes.startswith(b"\xff\xd8"),
                )
            except KlingContentModerationError as exc:
                last_exc = exc
                if attempt >= _CLIP_RETRY_ATTEMPTS or moderation_rephrase_attempted:
                    logger.warning(
                        "Kling moderation still blocks segment '%s' (video_id=%s) after "
                        "one rephrase; using a static fallback.",
                        segment.title,
                        video_id,
                    )
                    break

                moderation_rephrase_attempted = True
                logger.warning(
                    "Kling content moderation blocked prompt (attempt %d/%d) for "
                    "segment '%s' (video_id=%s): %s. Rephrasing once and retrying.",
                    attempt,
                    _CLIP_RETRY_ATTEMPTS,
                    segment.title,
                    video_id,
                    exc,
                )
                rephrased_prompt = await _rephrase_prompt_for_moderation(current_prompt)
                if not rephrased_prompt or rephrased_prompt.strip() == current_prompt.strip():
                    logger.warning(
                        "Kling moderation rephrase did not change segment '%s' "
                        "(video_id=%s); using a static fallback instead of resubmitting "
                        "the blocked prompt.",
                        segment.title,
                        video_id,
                    )
                    break

                current_prompt = rephrased_prompt
                logger.info(
                    "Rephrased prompt for segment '%s': %s",
                    segment.title,
                    current_prompt[:100],
                )
                # No sleep — the next request is a new task with a new prompt.
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _CLIP_RETRY_ATTEMPTS:
                    delay = random.uniform(5, 30)
                    logger.warning(
                        "Viewmax clip failed (attempt %d/%d) for segment '%s' "
                        "(video_id=%s): %s. Retrying in %.1f s.",
                        attempt,
                        _CLIP_RETRY_ATTEMPTS,
                        segment.title,
                        video_id,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted — generate static JPEG fallback
        logger.warning(
            "All %d Viewmax retries exhausted for segment '%s' (segment_index=%d, "
            "video_id=%s): %s. Substituting static fallback JPEG.",
            _CLIP_RETRY_ATTEMPTS,
            segment.title,
            segment_index,
            video_id,
            last_exc,
        )
        fallback_bytes = _make_fallback_jpeg(segment.title)
        return ClipResult(
            segment_index=segment_index,
            mp4_bytes=fallback_bytes,
            is_fallback=True,
        )

    # ------------------------------------------------------------------
    # Task 11.2 — Video compilation via ffmpeg
    # ------------------------------------------------------------------

    async def _compile_video(
        self,
        clip_results: list[ClipResult],
        segments: list[_Segment],
        segment_durations: list[float],
        narration_mp3_path: str,
        tmp_path: Path,
        video_id: str,
    ) -> bytes:
        """Compile all clip results into a single MP4 using ffmpeg.

        Clips from real Viewmax calls are saved as ``.mp4`` files.
        Fallback clips (static JPEG) are saved as ``.jpg`` and given a
        looped ffmpeg input with ``-loop 1 -t {_FALLBACK_CLIP_DURATION_S}``.

        The narration MP3 is muxed in and the audio stream is taken with
        ``-shortest`` to ensure the output duration matches the audio track
        (audio sync tolerance ±100 ms).

        ffmpeg command pattern::

            ffmpeg
              [-loop 1] -i clip_N.{mp4,jpg} ...
              -i narration.mp3
              -filter_complex "concat=n={N}:v=1:a=0[v]"
              -map [v] -map {N}:a
              -c:v libx264 -r 24 -s 1920x1080
              -c:a aac -shortest
              output.mp4

        Args:
            clip_results: Ordered list of :class:`ClipResult` objects.
            segments: Matching segment list (used for logging only).
            narration_mp3_path: Filesystem path to the narration MP3.
            tmp_path: Writable temporary directory.
            video_id: Used in log messages.

        Returns:
            Raw MP4 bytes of the compiled video.

        Raises:
            VisualGeneratorError: If ffmpeg exits with a non-zero code.
        """
        # Write each clip to disk
        # For fallback clips, reuse the nearest real MP4 clip instead of a static JPEG.
        # First pass: write all real clips to disk so we have paths to reference.
        # Second pass: fill fallback slots — prefer the previous real clip, then
        # the next real clip (look-ahead), and only use the static JPEG if no real
        # clip exists anywhere in the list (i.e. every clip failed).
        import shutil as _shutil  # noqa: PLC0415

        # First pass — write real clips, record their paths by index.
        real_clip_paths: dict[int, Path] = {}
        for clip_idx, result in enumerate(clip_results):
            if not result.is_fallback:
                clip_file = tmp_path / f"clip_{clip_idx:04d}.mp4"
                clip_file.write_bytes(result.mp4_bytes)
                real_clip_paths[clip_idx] = clip_file

        # Helper: find the nearest real clip path (previous first, then next).
        def _nearest_real_clip(idx: int) -> Optional[Path]:
            # Search backwards
            for i in range(idx - 1, -1, -1):
                if i in real_clip_paths:
                    return real_clip_paths[i]
            # Search forwards
            for i in range(idx + 1, len(clip_results)):
                if i in real_clip_paths:
                    return real_clip_paths[i]
            return None

        # Second pass — build clip_paths with fallback slots resolved.
        clip_paths: list[tuple[Path, bool]] = []
        for clip_idx, result in enumerate(clip_results):
            if result.is_fallback:
                nearest = _nearest_real_clip(clip_idx)
                if nearest is not None:
                    clip_file = tmp_path / f"clip_{clip_idx:04d}.mp4"
                    _shutil.copy(nearest, clip_file)
                    clip_paths.append((clip_file, False))  # treat as real MP4
                    logger.info(
                        "Visual_Generator: fallback slot %d — reusing clip '%s'",
                        clip_idx, nearest.name,
                    )
                else:
                    # Every clip failed — last resort static JPEG
                    clip_file = tmp_path / f"clip_{clip_idx:04d}.jpg"
                    clip_file.write_bytes(result.mp4_bytes)
                    clip_paths.append((clip_file, True))
                    logger.warning(
                        "Visual_Generator: fallback slot %d — no real clip available, using static JPEG",
                        clip_idx,
                    )
            else:
                clip_paths.append((real_clip_paths[clip_idx], False))

        # Download the narration MP3 from Asset_Store into the temp directory
        # so ffmpeg can access it as a local file.
        # narration_mp3_path is the bare filename (e.g. "video-abc_v1.mp3")
        # stored in the NARRATION subfolder for this video_id.
        try:
            mp3_bytes = await self._asset_store.read(
                video_id=video_id,
                subfolder=SubFolder.NARRATION,
                filename=narration_mp3_path,
            )
            local_mp3_path = tmp_path / narration_mp3_path
            local_mp3_path.write_bytes(mp3_bytes)
            logger.debug("Narration MP3 downloaded to %s (%d bytes)", local_mp3_path, len(mp3_bytes))
        except Exception as exc:
            raise VisualGeneratorError(
                f"Failed to download narration MP3 '{narration_mp3_path}' "
                f"for video_id={video_id}: {exc}"
            ) from exc

        output_path = tmp_path / "output.mp4"
        n_clips = len(clip_paths)

        # Build ffmpeg argument list — try common install locations
        import shutil as _shutil  # noqa: PLC0415
        ffmpeg_bin = (
            _shutil.which("ffmpeg")
            or "/opt/homebrew/bin/ffmpeg"   # macOS Apple Silicon (brew)
            or "/usr/local/bin/ffmpeg"       # macOS Intel (brew)
            or "ffmpeg"                      # Linux / production server
        )
        cmd: list[str] = [ffmpeg_bin, "-y"]

        for i, (clip_path, is_fallback) in enumerate(clip_paths):
            seg_dur = segment_durations[i] if i < len(segment_durations) else _FALLBACK_CLIP_DURATION_S
            if is_fallback:
                # JPEG still: loop for the full segment duration
                cmd += ["-loop", "1", "-t", f"{seg_dur:.2f}", "-i", str(clip_path)]
            else:
                # Real MP4: play once, trimmed to seg_dur (no looping)
                cmd += ["-t", f"{seg_dur:.2f}", "-i", str(clip_path)]

        # Narration audio input (index = n_clips)
        cmd += ["-i", str(local_mp3_path)]

        # Filter graph: normalise every input to 1920x1080 @ 24fps, square pixels,
        # yuv420p — handles portrait/landscape/square and any SAR uniformly.
        # Black padding fills any letterbox/pillarbox from aspect ratio mismatch.
        filter_parts = []
        for i in range(n_clips):
            filter_parts.append(
                f"[{i}:v]"
                f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1:1,"
                f"fps={_FFMPEG_FRAMERATE},"
                f"format=yuv420p"
                f"[v{i}]"
            )
        concat_inputs = "".join(f"[v{i}]" for i in range(n_clips))
        filter_complex = (
            ";".join(filter_parts)
            + f";{concat_inputs}concat=n={n_clips}:v=1:a=0[vout]"
        )
        cmd += [
            "-filter_complex", filter_complex,
            "-map", "[vout]",
            "-map", f"{n_clips}:a",
            "-c:v", "libx264",
            "-r", str(_FFMPEG_FRAMERATE),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            str(output_path),
        ]

        logger.debug("ffmpeg command: %s", " ".join(cmd))

        loop = asyncio.get_running_loop()
        try:
            proc_result: subprocess.CompletedProcess[bytes] = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    check=False,  # we check returncode manually below
                ),
            )
        except FileNotFoundError as exc:
            raise VisualGeneratorError(
                "ffmpeg executable not found. Ensure ffmpeg is installed and on PATH."
            ) from exc

        if proc_result.returncode != 0:
            stderr_snippet = proc_result.stderr[-2000:].decode("utf-8", errors="replace")
            raise VisualGeneratorError(
                f"ffmpeg failed (exit code {proc_result.returncode}) for "
                f"video_id={video_id}:\n{stderr_snippet}"
            )

        logger.info(
            "ffmpeg compilation succeeded for video_id=%s (%d clips)",
            video_id,
            n_clips,
        )
        return output_path.read_bytes()


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ClipResult",
    "ViewmaxClient",
    "ViewmaxMCPClient",
    "VisualGeneratorError",
    "Visual_Generator",
]
