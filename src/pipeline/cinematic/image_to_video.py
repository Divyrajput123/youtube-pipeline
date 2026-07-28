"""Image-to-Video pipeline — generates still frames via Flux Pro, then animates via Kling.

This is the high-quality approach used by top AI YouTube channels:
1. Generate a cinematic still frame per beat (Flux Pro via fal.ai)
2. Animate each frame into a 5-second video clip (Kling image-to-video)
3. Result: consistent characters, clear action, cinematic quality

Cost: ~$0.17/clip (Flux $0.03 + Kling $0.14) = ~$3.40 for 20 beats
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

from pipeline.cinematic.models import Beat, BeatType, CinematicScript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_FAL_API_KEY = os.environ.get("FAL_KEY", "")
_KLING_API_KEY = os.environ.get("KLING_API_KEY", "")
_KLING_MODEL = os.environ.get("KLING_MODEL", "kling-v1-6")


# ---------------------------------------------------------------------------
# Flux Pro image generation (via fal.ai)
# ---------------------------------------------------------------------------


async def generate_still_frame(prompt: str) -> str:
    """Generate a single high-quality still frame via Flux Pro on fal.ai.

    Args:
        prompt: Detailed scene description for the frame.

    Returns:
        URL of the generated image (hosted by fal.ai temporarily).

    Raises:
        RuntimeError: If generation fails.
    """
    api_key = _FAL_API_KEY or os.environ.get("FAL_KEY", "")
    if not api_key:
        raise RuntimeError("FAL_KEY not set — needed for Flux Pro image generation")

    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }

    # Submit to Flux Pro
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://fal.run/fal-ai/flux-pro/v1.1",
            headers=headers,
            json={
                "prompt": prompt,
                "image_size": {"width": 1280, "height": 720},
                "num_images": 1,
                "enable_safety_checker": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    # Extract image URL
    images = data.get("images", [])
    if not images:
        raise RuntimeError(f"Flux Pro returned no images: {data}")

    image_url = images[0].get("url", "")
    if not image_url:
        raise RuntimeError(f"Flux Pro image has no URL: {images[0]}")

    logger.info("Flux Pro: generated frame — %s", image_url[:60])
    return image_url


# ---------------------------------------------------------------------------
# Kling image-to-video
# ---------------------------------------------------------------------------


async def animate_frame(image_url: str, motion_prompt: str) -> bytes:
    """Animate a still frame into a 5-second video clip via Kling image-to-video.

    Args:
        image_url: URL of the source frame (from Flux Pro).
        motion_prompt: Description of what motion/action to add.

    Returns:
        MP4 bytes of the animated clip.

    Raises:
        RuntimeError: If animation fails or times out.
    """
    api_key = _KLING_API_KEY or os.environ.get("KLING_API_KEY", "")
    if not api_key:
        raise RuntimeError("KLING_API_KEY not set — needed for image-to-video")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Submit image-to-video task
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            "https://api.klingai.com/v1/videos/image2video",
            headers=headers,
            json={
                "model_name": _KLING_MODEL,
                "image": image_url,
                "prompt": motion_prompt[:200],
                "negative_prompt": (
                    "static, frozen, no movement, blurry, low quality, "
                    "watermark, text overlay, morphing face, extra limbs"
                ),
                "cfg_scale": 0.5,
                "mode": "std",
                "duration": "5",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    task_id = data["data"]["task_id"]
    logger.info("Kling image-to-video: task %s submitted", task_id)

    # Poll until done (max 5 minutes)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(60):
            await asyncio.sleep(5)
            poll = await client.get(
                f"https://api.klingai.com/v1/videos/image2video/{task_id}",
                headers=headers,
            )
            poll.raise_for_status()
            result = poll.json()
            status = result["data"]["task_status"]

            if status == "succeed":
                video_url = result["data"]["task_result"]["videos"][0]["url"]
                # Download the video
                vid_resp = await client.get(video_url)
                vid_resp.raise_for_status()
                logger.info("Kling image-to-video: task %s completed", task_id)
                return vid_resp.content
            elif status == "failed":
                error_msg = result["data"].get("task_status_msg", "unknown error")
                raise RuntimeError(f"Kling image-to-video failed: {error_msg}")
            # processing — keep polling

    raise RuntimeError(f"Kling image-to-video task {task_id} timed out")


# ---------------------------------------------------------------------------
# Combined: generate frame + animate
# ---------------------------------------------------------------------------


async def generate_beat_clip(
    beat: Beat,
    script: CinematicScript,
    beat_index: int,
    total_beats: int,
) -> bytes:
    """Generate a full video clip for one beat using image-to-video approach.

    Steps:
    1. Build a detailed frame prompt with character descriptions + scene context
    2. Generate a still frame via Flux Pro
    3. Animate the frame via Kling image-to-video

    Args:
        beat: The beat to generate a clip for.
        script: Full script for character/setting context.
        beat_index: Position in the sequence (for scene phase).
        total_beats: Total number of beats (for phase calculation).

    Returns:
        MP4 bytes of the animated clip.
    """
    # Build the still frame prompt — rich character + scene description
    frame_prompt = _build_frame_prompt(beat, script, beat_index, total_beats)

    # Build the motion prompt — what action/movement to add
    motion_prompt = _build_motion_prompt(beat)

    # Step 1: Generate still frame
    logger.info("Beat %d: generating still frame...", beat_index)
    image_url = await generate_still_frame(frame_prompt)

    # Step 2: Animate the frame
    logger.info("Beat %d: animating frame...", beat_index)
    mp4_bytes = await animate_frame(image_url, motion_prompt)

    return mp4_bytes


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_frame_prompt(
    beat: Beat,
    script: CinematicScript,
    beat_index: int,
    total_beats: int,
) -> str:
    """Build a detailed Flux Pro prompt for a single still frame.

    Includes character descriptions, environment state, camera angle,
    and cinematic style instructions.
    """
    progress = beat_index / max(1, total_beats - 1)

    # Scene phase determines environment destruction level
    if progress < 0.15:
        environment = "city rooftop at night, intact buildings, storm clouds gathering, rain starting"
        lighting = "dramatic backlighting, lightning in clouds, dark blue tones"
    elif progress < 0.4:
        environment = "cracked rooftop, shattered windows, rain pouring, debris in air"
        lighting = "harsh side lighting, orange fire glow, blue lightning flashes"
    elif progress < 0.7:
        environment = "heavily destroyed rooftop, fires burning, smoke, collapsed structures"
        lighting = "intense warm light from fires, dramatic shadows, god rays through smoke"
    else:
        environment = "crater in ruins, embers floating, dust settling, aftermath"
        lighting = "cold blue moonlight, faint embers, volumetric dust"

    # Determine which character to focus on
    action_desc = beat.video_prompt

    # Build the full prompt
    prompt = (
        f"Cinematic still frame, {beat.camera_angle} shot. "
        f"Character 1: {script.hero1_description}. "
        f"Character 2: {script.hero2_description}. "
        f"Scene: {action_desc}. "
        f"Environment: {environment}. "
        f"Lighting: {lighting}. "
        f"Style: ultra realistic, Hollywood blockbuster, IMAX quality, "
        f"volumetric rain droplets, photorealistic skin, detailed costume textures, "
        f"cinematic depth of field, professional color grading, 8K resolution."
    )

    return prompt[:1000]  # Flux Pro handles long prompts well


def _build_motion_prompt(beat: Beat) -> str:
    """Build the motion description for Kling image-to-video.

    Describes what movement to add to the still frame.
    """
    # Base motion from the beat
    motion = beat.video_prompt

    # Add camera movement based on beat type
    if beat.beat_type == BeatType.IMPACT:
        motion += ", camera shake on impact, debris flying outward, shockwave ripple"
    elif beat.beat_type == BeatType.ACTION:
        motion += ", dynamic motion, cape flowing, rain droplets splashing"
    elif beat.beat_type == BeatType.TENSION:
        motion += ", slow subtle movement, wind blowing capes, breathing visible"
    elif beat.beat_type == BeatType.TRANSITION:
        motion += ", slow camera pan, dust settling, embers floating upward"

    return motion[:200]
