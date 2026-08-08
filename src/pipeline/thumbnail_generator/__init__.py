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


# ---------------------------------------------------------------------------
# Shorts / Reels thumbnail (9:16 portrait)
# ---------------------------------------------------------------------------

# Shorts thumbnail dimensions (9:16 for YouTube Shorts + Instagram Reels)
_SHORTS_THUMBNAIL_WIDTH: int = 1080
_SHORTS_THUMBNAIL_HEIGHT: int = 1920
_SHORTS_THUMBNAIL_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB (YouTube limit)


async def generate_shorts_thumbnail(
    title: str,
    script_content: str,
    existing_thumbnail_bytes: Optional[bytes] = None,
) -> bytes:
    """Generate a 1080×1920 (9:16) thumbnail for YouTube Shorts and Instagram Reels.

    Strategy (in priority order):
      1. If an existing 16:9 thumbnail is provided, crop/reframe it to 9:16
         with a text overlay optimized for vertical viewing.
      2. Otherwise, generate a fresh vertical image via Pollinations.AI
         with bold text overlay (same style as main thumbnails).

    The resulting image is also used as the Instagram Reel cover image.

    Args:
        title: Video title (used for text overlay).
        script_content: Script body for context when generating fresh images.
        existing_thumbnail_bytes: Optional JPEG/PNG bytes of the existing 16:9
            thumbnail. If provided, it's reframed to 9:16 instead of generating
            a new image from scratch.

    Returns:
        JPEG bytes of the 1080×1920 thumbnail (< 2 MB).

    Raises:
        ThumbnailGeneratorError: If generation fails from all sources.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    result_bytes: Optional[bytes] = None

    # --- Strategy 1: Reframe existing 16:9 thumbnail to 9:16 ---
    if existing_thumbnail_bytes and len(existing_thumbnail_bytes) > 1000:
        try:
            result_bytes = _reframe_to_vertical(existing_thumbnail_bytes, title)
            logger.info(
                "Shorts thumbnail: reframed existing thumbnail to 9:16 (%d bytes)",
                len(result_bytes),
            )
        except Exception as exc:
            logger.warning("Shorts thumbnail: reframe failed (%s), trying fresh generation", exc)
            result_bytes = None

    # --- Strategy 2: Generate fresh vertical image ---
    if not result_bytes:
        try:
            result_bytes = await _generate_fresh_vertical_thumbnail(title, script_content)
            logger.info(
                "Shorts thumbnail: generated fresh vertical image (%d bytes)",
                len(result_bytes),
            )
        except Exception as exc:
            logger.warning("Shorts thumbnail: fresh generation failed: %s", exc)
            result_bytes = None

    # --- Strategy 3: Simple text-on-gradient fallback ---
    if not result_bytes:
        result_bytes = _make_vertical_fallback_thumbnail(title)
        logger.info("Shorts thumbnail: using gradient fallback (%d bytes)", len(result_bytes))

    if not result_bytes:
        raise ThumbnailGeneratorError("All Shorts thumbnail generation strategies failed")

    return result_bytes


def _reframe_to_vertical(thumbnail_bytes: bytes, title: str) -> bytes:
    """Crop/reframe a 16:9 thumbnail into 9:16 with text overlay.

    Places the original image in the center (scaled to fill width),
    adds a gradient overlay at top and bottom, and renders the title text.

    Args:
        thumbnail_bytes: Original 16:9 JPEG/PNG bytes.
        title: Title text for overlay.

    Returns:
        JPEG bytes at 1080×1920.
    """
    from PIL import Image, ImageDraw, ImageFont, ImageFilter  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    src = Image.open(_io.BytesIO(thumbnail_bytes)).convert("RGB")

    # Create the vertical canvas
    canvas = Image.new("RGB", (_SHORTS_THUMBNAIL_WIDTH, _SHORTS_THUMBNAIL_HEIGHT), (0, 0, 0))

    # Scale the source to fill the width (1080px), keeping aspect ratio
    scale_factor = _SHORTS_THUMBNAIL_WIDTH / src.width
    new_w = _SHORTS_THUMBNAIL_WIDTH
    new_h = int(src.height * scale_factor)
    resized = src.resize((new_w, new_h), Image.LANCZOS)

    # Place in center vertically
    y_offset = (_SHORTS_THUMBNAIL_HEIGHT - new_h) // 2
    canvas.paste(resized, (0, y_offset))

    # Fill top and bottom gaps with blurred/darkened versions of the image
    if y_offset > 0:
        # Top fill: stretch a strip of the top of the image
        top_strip = resized.crop((0, 0, new_w, min(50, new_h)))
        top_fill = top_strip.resize((new_w, y_offset), Image.LANCZOS)
        top_fill = top_fill.filter(ImageFilter.GaussianBlur(radius=15))
        canvas.paste(top_fill, (0, 0))

        # Bottom fill: stretch a strip of the bottom of the image
        bottom_strip = resized.crop((0, max(0, new_h - 50), new_w, new_h))
        bottom_fill = bottom_strip.resize((new_w, _SHORTS_THUMBNAIL_HEIGHT - y_offset - new_h), Image.LANCZOS)
        bottom_fill = bottom_fill.filter(ImageFilter.GaussianBlur(radius=15))
        canvas.paste(bottom_fill, (0, y_offset + new_h))

    # Add gradient overlays for text readability
    draw = ImageDraw.Draw(canvas)

    # Top gradient (dark → transparent)
    for i in range(300):
        alpha = int(180 * (1 - i / 300))
        draw.line([(0, i), (_SHORTS_THUMBNAIL_WIDTH, i)], fill=(0, 0, 0, alpha) if canvas.mode == "RGBA" else (0, 0, 0))
        # Since we're in RGB mode, use decreasing opacity simulation
        pass

    # Overlay semi-transparent gradient at top and bottom
    overlay = Image.new("RGBA", (_SHORTS_THUMBNAIL_WIDTH, _SHORTS_THUMBNAIL_HEIGHT), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    for i in range(350):
        alpha = int(200 * (1 - i / 350))
        overlay_draw.line([(0, i), (_SHORTS_THUMBNAIL_WIDTH, i)], fill=(0, 0, 0, alpha))
    for i in range(_SHORTS_THUMBNAIL_HEIGHT - 300, _SHORTS_THUMBNAIL_HEIGHT):
        alpha = int(200 * ((i - (_SHORTS_THUMBNAIL_HEIGHT - 300)) / 300))
        overlay_draw.line([(0, i), (_SHORTS_THUMBNAIL_WIDTH, i)], fill=(0, 0, 0, alpha))

    canvas = canvas.convert("RGBA")
    canvas = Image.alpha_composite(canvas, overlay)
    canvas = canvas.convert("RGB")

    # Add title text at the top
    draw = ImageDraw.Draw(canvas)
    _draw_vertical_title(draw, title, _SHORTS_THUMBNAIL_WIDTH, top_y=80)

    # Save as JPEG
    buf = _io.BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    result = buf.getvalue()

    # Ensure under 2MB
    if len(result) > _SHORTS_THUMBNAIL_MAX_BYTES:
        buf = _io.BytesIO()
        canvas.save(buf, format="JPEG", quality=75)
        result = buf.getvalue()

    return result


async def _generate_fresh_vertical_thumbnail(title: str, script_content: str) -> Optional[bytes]:
    """Generate a fresh 9:16 image via Pollinations.AI with text overlay.

    Args:
        title: Video title for the image prompt and text overlay.
        script_content: Script body for context.

    Returns:
        JPEG bytes at 1080×1920, or None if generation fails.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    import io as _io  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415
    import random as _random  # noqa: PLC0415

    image_prompt = (
        f"Photorealistic cinematic vertical portrait shot, ultra realistic, "
        f"YouTube Shorts thumbnail for: {title}, "
        f"dramatic character in powerful pose, center frame, "
        f"dramatic orange and blue lighting, lens flare, volumetric light, "
        f"epic background with destruction or energy, "
        f"IMAX cinematography, shallow depth of field, film grain, 8K detail, "
        f"vertical 9:16 composition, no text anywhere"
    )

    encoded_prompt = urllib.parse.quote(image_prompt[:500])
    seed = _random.randint(0, 99999)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        f"?width=1080&height=1920&model=flux-pro&nologo=true&enhance=true&seed={seed}"
    )

    img_bytes = None

    # Try Ideogram first if key is available (best quality)
    ideogram_key = os.environ.get("IDEOGRAM_API_KEY", "")
    if ideogram_key:
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    "https://api.ideogram.ai/v1/ideogram-v4/generate",
                    headers={"Api-Key": ideogram_key, "Content-Type": "application/json"},
                    json={"text_prompt": image_prompt, "aspect_ratio": "9:16"},
                )
                resp.raise_for_status()
                data = resp.json()
                images = data.get("data", [])
                if images and images[0].get("url"):
                    img_resp = await client.get(images[0]["url"], timeout=30.0)
                    img_resp.raise_for_status()
                    img_bytes = img_resp.content
                    logger.info("Shorts thumbnail: Ideogram generated (%d bytes)", len(img_bytes))
        except Exception as exc:
            logger.warning("Shorts thumbnail: Ideogram failed (%s), trying Pollinations", exc)

    # Fallback: Pollinations.AI (free)
    if not img_bytes:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                img_bytes = resp.content
            logger.info("Shorts thumbnail: Pollinations generated (%d bytes)", len(img_bytes))
        except Exception as exc:
            logger.warning("Shorts thumbnail: Pollinations failed: %s", exc)
            return None

    if not img_bytes:
        return None

    # Add text overlay
    try:
        img = Image.open(_io.BytesIO(img_bytes)).convert("RGB")
        if img.size != (_SHORTS_THUMBNAIL_WIDTH, _SHORTS_THUMBNAIL_HEIGHT):
            img = img.resize((_SHORTS_THUMBNAIL_WIDTH, _SHORTS_THUMBNAIL_HEIGHT), Image.LANCZOS)

        draw = ImageDraw.Draw(img)
        _draw_vertical_title(draw, title, _SHORTS_THUMBNAIL_WIDTH, top_y=100)

        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        result = buf.getvalue()

        if len(result) > _SHORTS_THUMBNAIL_MAX_BYTES:
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            result = buf.getvalue()

        return result

    except Exception as exc:
        logger.warning("Shorts thumbnail: text overlay failed: %s", exc)
        # Return plain image without text
        return img_bytes if len(img_bytes) < _SHORTS_THUMBNAIL_MAX_BYTES else None


def _make_vertical_fallback_thumbnail(title: str) -> bytes:
    """Generate a simple 1080×1920 gradient thumbnail with bold text.

    Used as a last resort when image generation APIs are unavailable.

    Args:
        title: Title text to render.

    Returns:
        JPEG bytes at 1080×1920.
    """
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    import io as _io  # noqa: PLC0415

    # Dark gradient background (top-to-bottom, deep blue to black)
    img = Image.new("RGB", (_SHORTS_THUMBNAIL_WIDTH, _SHORTS_THUMBNAIL_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for y in range(_SHORTS_THUMBNAIL_HEIGHT):
        ratio = y / _SHORTS_THUMBNAIL_HEIGHT
        r = int(20 * (1 - ratio))
        g = int(10 * (1 - ratio))
        b = int(60 * (1 - ratio) + 10)
        draw.line([(0, y), (_SHORTS_THUMBNAIL_WIDTH, y)], fill=(r, g, b))

    # Draw title text
    _draw_vertical_title(draw, title, _SHORTS_THUMBNAIL_WIDTH, top_y=300)

    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _draw_vertical_title(
    draw: "ImageDraw.ImageDraw",
    title: str,
    canvas_width: int,
    top_y: int = 100,
) -> None:
    """Draw bold title text centered on a vertical canvas with outline.

    Splits the title into multiple lines if needed and renders with a
    thick black outline for readability on any background.

    Args:
        draw: Pillow ImageDraw instance.
        title: Title text to render.
        canvas_width: Width of the canvas (for centering).
        top_y: Y coordinate for the first line.
    """
    from PIL import ImageFont  # noqa: PLC0415
    import textwrap  # noqa: PLC0415

    # Load bold font
    font_size = 72
    font = None
    for font_name in [
        "Impact.ttf",
        "Arial Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_name, size=font_size)
            break
        except (IOError, OSError):
            continue

    if font is None:
        font = ImageFont.load_default()

    # Wrap text to fit width (roughly 15-18 chars per line for Impact at 72px on 1080px)
    short_title = title[:80].upper()
    lines = textwrap.wrap(short_title, width=16)

    y = top_y
    for line in lines[:4]:  # Max 4 lines
        try:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(line, font=font)  # type: ignore

        x = (canvas_width - text_w) // 2

        # Draw thick black outline
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0))

        # Draw main text in white/yellow
        draw.text((x, y), line, font=font, fill=(255, 220, 0))
        y += text_h + 15


__all__ = [
    "ThumbnailGeneratorError",
    "generate_thumbnail",
    "generate_thumbnail_prompt",
    "generate_thumbnail_image",
    "generate_shorts_thumbnail",
]
