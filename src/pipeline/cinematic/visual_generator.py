"""Cinematic Visual Generator — produces beat-synced video clips.

Generates one short video clip per beat with the exact duration specified,
then stitches them together with flash frames on impact beats.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pipeline.cinematic.models import Beat, BeatType, CinematicScript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WIDTH = 1920
_HEIGHT = 1080
_FPS = 24
_FLASH_FRAMES = 2  # Number of white flash frames on impact


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_fallback_clip(
    beat: Beat,
    duration_s: float,
    tmpdir: str,
    index: int,
) -> str:
    """Create a fallback still-image clip for a beat.

    Renders a dark gradient background with beat info text.
    Returns path to the generated MP4 clip.
    """
    # Create a dark cinematic background
    img = Image.new("RGB", (_WIDTH, _HEIGHT), (10, 10, 15))
    draw = ImageDraw.Draw(img)

    # Add a subtle gradient
    for y in range(_HEIGHT):
        intensity = int(15 + 10 * (y / _HEIGHT))
        draw.line([(0, y), (_WIDTH, y)], fill=(intensity, intensity, intensity + 5))

    # Add beat text for debugging (won't be visible in final with effects)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except Exception:
        font = ImageFont.load_default()

    text = beat.video_prompt[:80] + "..." if len(beat.video_prompt) > 80 else beat.video_prompt
    draw.text((_WIDTH // 2 - 300, _HEIGHT // 2 - 20), text, fill=(180, 180, 200), font=font)

    # Save frame
    frame_path = Path(tmpdir) / f"beat_{index:03d}_frame.jpg"
    img.save(str(frame_path), quality=90)

    # Generate clip from still image at the exact duration
    clip_path = Path(tmpdir) / f"beat_{index:03d}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(frame_path),
            "-t", str(duration_s),
            "-vf", f"scale={_WIDTH}:{_HEIGHT}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", str(_FPS),
            str(clip_path),
        ],
        capture_output=True, timeout=30,
    )
    return str(clip_path)


def _create_flash_frame(tmpdir: str, index: int) -> str:
    """Create a 2-frame white flash clip for impact moments."""
    img = Image.new("RGB", (_WIDTH, _HEIGHT), (255, 255, 255))
    frame_path = Path(tmpdir) / f"flash_{index:03d}.jpg"
    img.save(str(frame_path), quality=90)

    flash_duration = _FLASH_FRAMES / _FPS  # 2 frames at 24fps ≈ 0.083s
    clip_path = Path(tmpdir) / f"flash_{index:03d}.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(frame_path),
            "-t", str(flash_duration),
            "-c:v", "libx264", "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-r", str(_FPS),
            str(clip_path),
        ],
        capture_output=True, timeout=10,
    )
    return str(clip_path)


# ---------------------------------------------------------------------------
# CinematicVisualGenerator
# ---------------------------------------------------------------------------


class CinematicVisualGenerator:
    """Generates beat-synced video clips for the cinematic pipeline.

    For each beat, generates a video clip at the exact beat duration.
    Impact beats get a white flash frame prepended.
    Falls back to still-image clips when the video API fails.

    Args:
        video_client: Async callable that takes (prompt, width, height, num_frames, frame_rate)
                      and returns MP4 bytes. If None, uses fallback for all clips.
    """

    def __init__(self, video_client=None) -> None:
        self._video_client = video_client

    async def _build_cinematic_prompt(self, beat: Beat, script: CinematicScript) -> str:
        """Use Claude to generate an optimized video prompt for this beat.

        Same approach as the working narrated pipeline — gives Claude the
        character descriptions, scene context, and action, and gets back
        a filmmaking-quality prompt.
        """
        from pipeline.script_writer import build_claude_client  # noqa: PLC0415

        # Determine scene phase based on beat position
        total_beats = len(script.beats)
        progress = beat.beat_index / max(1, total_beats - 1)

        if progress < 0.15:
            scene_phase = "opening: storm clouds gathering, tension building, characters sizing each other up"
            damage_level = "environment intact, dark sky, rain starting"
        elif progress < 0.4:
            scene_phase = "escalation: first impacts, rain pouring, lightning flashes"
            damage_level = "windows cracking, ground fracturing, debris flying"
        elif progress < 0.7:
            scene_phase = "climax: heavy destruction, fires spreading, maximum intensity"
            damage_level = "buildings collapsing, massive craters, smoke and fire everywhere"
        else:
            scene_phase = "resolution: dust settling, embers floating, aftermath"
            damage_level = "rubble-filled crater, one standing, one defeated"

        # Camera angles
        camera_options = [
            "epic wide shot", "extreme close-up", "low angle hero shot",
            "aerial drone shot", "tracking shot", "over-the-shoulder",
            "slow-motion impact shot", "dutch angle", "point-of-view shot",
        ]
        camera = camera_options[beat.beat_index % len(camera_options)]

        # Master style
        master_style = (
            "Ultra realistic live-action Hollywood blockbuster, IMAX HDR, "
            "cinematic color grading, volumetric lighting, rain droplets visible, "
            "shallow depth of field, film grain, high temporal consistency."
        )

        try:
            client = build_claude_client()
            prompt = (
                f"Write a video generation prompt for a fight scene clip.\n\n"
                f"CHARACTER 1: {script.hero1_description}\n"
                f"CHARACTER 2: {script.hero2_description}\n"
                f"SETTING: {script.setting}\n"
                f"SCENE PHASE: {scene_phase}\n"
                f"DAMAGE LEVEL: {damage_level}\n"
                f"CAMERA: {camera}\n"
                f"ACTION IN THIS CLIP: {beat.video_prompt}\n\n"
                f"RULES:\n"
                f"1. Start by describing the character's FULL appearance (costume, hair, body type)\n"
                f"2. Then describe the action and environment\n"
                f"3. Include camera movement and lighting\n"
                f"4. NEVER use character names or franchise names\n"
                f"5. Max 280 characters\n"
                f"6. End with: {master_style[:80]}\n\n"
                f"Output ONLY the prompt, no explanation."
            )
            result = await client.complete(prompt, max_tokens=200)
            final_prompt = result.strip().strip('"\'')[:400]
            logger.info("Cinematic prompt for beat %d: %s", beat.beat_index, final_prompt[:60])
            return final_prompt
        except Exception as exc:
            logger.warning("Claude prompt generation failed for beat %d: %s", beat.beat_index, exc)
            # Fallback: use the raw beat prompt with character descriptions prepended
            fallback = (
                f"{script.hero1_description}, {script.hero2_description}, "
                f"{beat.video_prompt}, {master_style}"
            )
            return fallback[:400]

    async def generate_clip(
        self,
        beat: Beat,
        script: CinematicScript,
        tmpdir: str,
    ) -> str:
        """Generate a single video clip for a beat.

        Args:
            beat: The Beat to generate video for.
            script: Full script (for character descriptions in prompts).
            tmpdir: Temp directory for intermediate files.

        Returns:
            Path to the generated MP4 clip file.
        """
        # Use Claude to generate a cinematic-quality video prompt
        # (same approach as the working narrated pipeline)
        prompt = await self._build_cinematic_prompt(beat, script)
        # Calculate frames needed for this beat's duration
        num_frames = max(int(beat.duration_seconds * _FPS), _FPS)  # minimum 1 second

        if self._video_client:
            try:
                mp4_bytes = await self._video_client(
                    prompt=prompt[:350],
                    width=768,
                    height=512,
                    num_frames=min(num_frames, 97),  # LTX-Video max
                    frame_rate=_FPS,
                )
                # Save to file
                clip_path = Path(tmpdir) / f"beat_{beat.beat_index:03d}.mp4"
                clip_path.write_bytes(mp4_bytes)

                # Trim/extend to exact duration
                trimmed_path = Path(tmpdir) / f"beat_{beat.beat_index:03d}_trimmed.mp4"
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(clip_path),
                        "-t", str(beat.duration_seconds),
                        "-vf", f"scale={_WIDTH}:{_HEIGHT}",
                        "-c:v", "libx264", "-preset", "fast",
                        "-pix_fmt", "yuv420p",
                        "-an",  # no audio in individual clips
                        str(trimmed_path),
                    ],
                    capture_output=True, timeout=60,
                )
                return str(trimmed_path)
            except Exception as exc:
                logger.warning(
                    "CinematicVisualGenerator: API failed for beat %d: %s — using fallback",
                    beat.beat_index, exc,
                )

        # Fallback: still image clip
        return _create_fallback_clip(beat, beat.duration_seconds, tmpdir, beat.beat_index)

    async def generate_all(
        self,
        script: CinematicScript,
        audio_wav_path: str,
    ) -> bytes:
        """Generate all video clips, stitch with flash frames, mux with audio.

        Args:
            script: The full CinematicScript.
            audio_wav_path: Path to the mixed audio WAV file.

        Returns:
            Final MP4 bytes (video + audio).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths: list[str] = []

            # Generate clips for each beat
            for beat in script.beats:
                # Add flash frame before impact beats
                if beat.flash_frame and beat.beat_type == BeatType.IMPACT:
                    flash_path = _create_flash_frame(tmpdir, beat.beat_index)
                    clip_paths.append(flash_path)

                clip_path = await self.generate_clip(beat, script, tmpdir)
                clip_paths.append(clip_path)

            # Write concat list for ffmpeg
            concat_list = Path(tmpdir) / "concat.txt"
            with open(concat_list, "w") as f:
                for path in clip_paths:
                    f.write(f"file '{path}'\n")

            # Concatenate all clips
            concat_output = Path(tmpdir) / "concat.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c:v", "libx264", "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-r", str(_FPS),
                    str(concat_output),
                ],
                capture_output=True, timeout=120, check=True,
            )

            # Mux video with audio
            final_output = Path(tmpdir) / "final.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(concat_output),
                    "-i", audio_wav_path,
                    "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k",
                    "-shortest",
                    "-movflags", "+faststart",
                    str(final_output),
                ],
                capture_output=True, timeout=60, check=True,
            )

            final_bytes = final_output.read_bytes()
            logger.info(
                "CinematicVisualGenerator: produced final MP4 — %d clips, %.1f MB",
                len(clip_paths), len(final_bytes) / 1e6,
            )
            return final_bytes
