"""Thumbnail Generator — AI-generated YouTube thumbnails using Wanx (Alibaba).

Flow:
  1. Claude/Gemini generates a thumbnail prompt from the script content
  2. Wanx API (wan2.6-t2i) renders the image from the prompt
  3. Result is a high-quality PNG uploaded to Asset_Store as the video thumbnail

Requirements:
  - DashScope API key (DASHSCOPE_API_KEY env var)
  - Get from: https://dashscope.console.aliyun.com → API Keys
  - Free tier: several hundred images for new accounts
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

from pipeline.config import is_production_mode

logger = logging.getLogger(__name__)

# Wanx API endpoints (Beijing region — international also available via Singapore)
_WANX_SUBMIT_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
_WANX_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks"

# Thumbnail dimensions (16:9 for YouTube)
_THUMBNAIL_SIZE = "1472*1104"  # ~16:10, within wan2.6 limits

# Polling settings
_POLL_INTERVAL_S = 5.0
_POLL_TIMEOUT_S = 120.0  # 2 minutes max


class ThumbnailGeneratorError(Exception):
    """Raised when thumbnail generation fails."""


async def generate_thumbnail_prompt(
    script_content: str,
    title: str,
    primary_keyword: str,
) -> str:
    """Use Claude/Gemini to generate a thumbnail image prompt from the script.

    The prompt describes a cinematic thumbnail suitable for YouTube:
    - Dramatic composition with strong focal point
    - Characters in action/confrontation poses
    - Bold colors, high contrast, dramatic lighting
    - NO text (text overlays are added separately)

    Args:
        script_content: The full script text.
        title: Video title.
        primary_keyword: Primary keyword/topic.

    Returns:
        A detailed image generation prompt string.
    """
    from pipeline.script_writer import build_claude_client

    client = build_claude_client()

    prompt = f"""You are a YouTube thumbnail designer. Generate a DETAILED image prompt for an AI image generator (like DALL-E or Midjourney) that will create a clickable, dramatic YouTube thumbnail.

VIDEO TITLE: {title}
PRIMARY TOPIC: {primary_keyword}
SCRIPT EXCERPT: {script_content[:500]}

THUMBNAIL REQUIREMENTS:
- Cinematic, movie-poster quality composition
- Show the main characters/subjects in DRAMATIC ACTION POSES (not static)
- Epic lighting: volumetric light rays, dramatic shadows, lens flares
- Bold vibrant colors with high contrast (thumbnails must pop at small sizes)
- Close-up or medium shot (faces/upper bodies visible for emotional connection)
- Background shows destruction, energy blasts, or dramatic environment
- Photorealistic style, 8K quality, detailed textures
- NO TEXT, NO WORDS, NO LETTERS in the image (text is added separately)
- NO watermarks, NO borders, NO frames

IMPORTANT RULES FOR THE PROMPT:
- NEVER use character names (no "Superman", "Batman", etc.) — describe by appearance ONLY
- Describe costumes, body types, hair, skin color, powers visually
- Focus on ONE dramatic moment that makes someone WANT to click
- The image should tell a story in a single frame

Output ONLY the image generation prompt (200-300 words). No explanations."""

    result = await client.complete(prompt, max_tokens=400)
    thumbnail_prompt = result.strip().strip('"\'')
    logger.info("Thumbnail prompt generated: %s", thumbnail_prompt[:100])
    return thumbnail_prompt


async def generate_thumbnail_image(
    prompt: str,
    negative_prompt: str = "",
) -> bytes:
    """Generate a thumbnail image using Wanx (Alibaba DashScope) text-to-image API.

    Uses the async flow: submit task → poll until done → download image.

    Args:
        prompt: The image generation prompt.
        negative_prompt: What to avoid in the image.

    Returns:
        PNG image bytes.

    Raises:
        ThumbnailGeneratorError: If generation fails.
    """
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")

    if not api_key:
        if is_production_mode():
            raise ThumbnailGeneratorError(
                "DASHSCOPE_API_KEY not set. Get one at https://dashscope.console.aliyun.com"
            )
        logger.warning("DASHSCOPE_API_KEY not set — cannot generate AI thumbnail")
        raise ThumbnailGeneratorError("DASHSCOPE_API_KEY not set")

    if not negative_prompt:
        negative_prompt = (
            "text, words, letters, watermark, logo, low quality, blurry, "
            "distorted, deformed, ugly, amateur, cartoon style, anime, "
            "oversaturated, washed out colors"
        )

    # Step 1: Submit task
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
    }

    body = {
        "model": "wan2.5-t2i-preview",
        "input": {
            "prompt": prompt[:2000],
            "negative_prompt": negative_prompt[:500],
        },
        "parameters": {
            "size": _THUMBNAIL_SIZE,
            "n": 1,
            "prompt_extend": True,
            "watermark": False,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(_WANX_SUBMIT_URL, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()

    task_id = data.get("output", {}).get("task_id")
    if not task_id:
        error = data.get("message", "Unknown error")
        raise ThumbnailGeneratorError(f"Wanx task creation failed: {error}")

    logger.info("Wanx thumbnail task submitted: %s", task_id)

    # Step 2: Poll until done
    poll_headers = {"Authorization": f"Bearer {api_key}"}
    elapsed = 0.0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while elapsed < _POLL_TIMEOUT_S:
            await asyncio.sleep(_POLL_INTERVAL_S)
            elapsed += _POLL_INTERVAL_S

            poll_resp = await client.get(
                f"{_WANX_TASK_URL}/{task_id}",
                headers=poll_headers,
            )
            poll_resp.raise_for_status()
            result = poll_resp.json()

            status = result.get("output", {}).get("task_status", "")
            logger.debug("Wanx task %s status: %s", task_id, status)

            if status == "SUCCEEDED":
                results = result.get("output", {}).get("results", [])
                if results and results[0].get("url"):
                    image_url = results[0]["url"]
                    logger.info("Wanx thumbnail ready: %s", image_url[:80])

                    # Download the image
                    img_resp = await client.get(image_url, timeout=60.0)
                    img_resp.raise_for_status()
                    return img_resp.content

                raise ThumbnailGeneratorError("Wanx task succeeded but no image URL returned")

            elif status == "FAILED":
                error_msg = result.get("output", {}).get("message", "Unknown error")
                raise ThumbnailGeneratorError(f"Wanx thumbnail generation failed: {error_msg}")

    raise ThumbnailGeneratorError(f"Wanx thumbnail generation timed out after {_POLL_TIMEOUT_S}s")


async def generate_thumbnail(
    script_content: str,
    title: str,
    primary_keyword: str,
) -> bytes:
    """Full pipeline: generate prompt from script, then render thumbnail image.

    Args:
        script_content: The video script text.
        title: Video title.
        primary_keyword: Primary keyword/topic.

    Returns:
        PNG image bytes of the generated thumbnail.

    Raises:
        ThumbnailGeneratorError: If any step fails.
    """
    # Step 1: Generate prompt from script
    prompt = await generate_thumbnail_prompt(script_content, title, primary_keyword)

    # Step 2: Render image from prompt
    image_bytes = await generate_thumbnail_image(prompt)

    logger.info("AI thumbnail generated: %d bytes", len(image_bytes))
    return image_bytes


__all__ = [
    "ThumbnailGeneratorError",
    "generate_thumbnail",
    "generate_thumbnail_prompt",
    "generate_thumbnail_image",
]
