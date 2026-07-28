"""Cinematic Pipeline Runner — end-to-end cinematic fight video generation.

Orchestrates: CinematicScriptWriter → DialogueGenerator + SFXEngine + AudioMixer
→ CinematicVisualGenerator → final MP4 upload.

Usage:
    runner = CinematicRunner(config)
    mp4_bytes = await runner.run(topic, video_id)
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from pipeline.asset_store import Asset_Store
from pipeline.cinematic.audio_mixer import AudioMixer
from pipeline.cinematic.dialogue_generator import DialogueGenerator
from pipeline.cinematic.models import CinematicScript
from pipeline.cinematic.script_writer import CinematicScriptWriter
from pipeline.cinematic.sfx_engine import SFXEngine
from pipeline.cinematic.visual_generator import CinematicVisualGenerator
from pipeline.models import (
    NarrationAsset,
    Script,
    SubFolder,
    TopicEntry,
    VisualAsset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CinematicRunner
# ---------------------------------------------------------------------------


class CinematicRunner:
    """End-to-end cinematic fight pipeline runner.

    Replaces the narration → visual flow with:
    screenplay → dialogue + SFX + music → beat-synced video → final MP4.
    """

    def __init__(
        self,
        asset_store: Optional[Asset_Store] = None,
        video_client=None,
        duration_seconds: int = 120,
    ) -> None:
        """Initialize the cinematic runner.

        Args:
            asset_store: Asset_Store for persisting artifacts to Drive.
            video_client: Async callable for video generation (RunPod/Kling).
            duration_seconds: Target fight duration in seconds.
        """
        self._asset_store = asset_store
        self._duration = duration_seconds
        self._script_writer = CinematicScriptWriter()
        self._dialogue_gen = DialogueGenerator()
        self._sfx_engine = SFXEngine()
        self._audio_mixer = AudioMixer(sfx_engine=self._sfx_engine)
        self._visual_gen = CinematicVisualGenerator(video_client=video_client)

    async def run(
        self,
        topic: TopicEntry,
        video_id: str,
    ) -> tuple[VisualAsset, Script, NarrationAsset]:
        """Run the full cinematic pipeline for a topic.

        Steps:
        1. Generate cinematic screenplay (Claude)
        2. Generate character dialogue (ElevenLabs, multiple voices)
        3. Mix SFX + dialogue + music into final audio
        4. Generate beat-synced video clips + stitch + mux with audio
        5. Store all assets to Drive

        Args:
            topic: The fight matchup topic.
            video_id: Pipeline video identifier.

        Returns:
            Tuple of (VisualAsset, Script, NarrationAsset) — compatible with
            the existing pipeline publisher interface.
        """
        from datetime import datetime, timezone  # noqa: PLC0415

        logger.info("CinematicRunner: starting for video_id=%s topic='%s'", video_id, topic.title)

        # --- Step 1: Generate screenplay ---
        cinematic_script = await self._script_writer.generate(
            topic=topic,
            video_id=video_id,
            duration_seconds=self._duration,
        )
        logger.info(
            "CinematicRunner: screenplay ready — %d beats, %.1fs, %s vs %s",
            len(cinematic_script.beats),
            cinematic_script.total_duration_seconds,
            cinematic_script.hero1_name,
            cinematic_script.hero2_name,
        )

        # Persist the screenplay as the "script" artifact
        script_content = _format_screenplay_markdown(cinematic_script)
        script_url = None
        if self._asset_store:
            script_url = await self._asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.SCRIPTS,
                filename="script_v1.md",
                content=script_content.encode("utf-8"),
            )

        script = Script(
            video_id=video_id,
            version=1,
            content=script_content,
            word_count=len(script_content.split()),
            style_profile_doc_id="cinematic",
            asset_url=script_url,
            created_at=datetime.now(tz=timezone.utc),
        )

        # --- Step 2: Generate dialogue audio ---
        dialogue_audio = await self._dialogue_gen.generate_all_dialogue(cinematic_script)
        logger.info("CinematicRunner: dialogue generated — %d lines", len(dialogue_audio))

        # --- Step 3: Mix audio (SFX + dialogue + music) ---
        mixed_audio_wav = await self._audio_mixer.mix(cinematic_script, dialogue_audio)
        logger.info("CinematicRunner: audio mixed — %d bytes", len(mixed_audio_wav))

        # Save mixed audio to temp file for video muxing
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(mixed_audio_wav)
            audio_path = f.name

        # Also store as narration asset
        narration_url = None
        if self._asset_store:
            narration_url = await self._asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.NARRATION,
                filename=f"{video_id}_v1.mp3",
                content=mixed_audio_wav,  # WAV but stored as the narration artifact
            )

        narration = NarrationAsset(
            video_id=video_id,
            version=1,
            mp3_path=audio_path,
            asset_url=narration_url,
            created_at=datetime.now(tz=timezone.utc),
        )

        # --- Step 4: Generate beat-synced video ---
        final_mp4 = await self._visual_gen.generate_all(
            script=cinematic_script,
            audio_wav_path=audio_path,
        )
        logger.info("CinematicRunner: final MP4 ready — %.1f MB", len(final_mp4) / 1e6)

        # Store video and generate thumbnail
        mp4_url = None
        thumb_url = None
        mp4_path = ""
        thumb_path = ""

        if self._asset_store:
            mp4_url = await self._asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.VIDEOS,
                filename=f"{video_id}_v1.mp4",
                content=final_mp4,
            )
            # Generate a simple thumbnail from the script info
            thumb_bytes = _generate_thumbnail(cinematic_script)
            thumb_url = await self._asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.THUMBNAILS,
                filename=f"{video_id}_v1.jpg",
                content=thumb_bytes,
            )

        visual = VisualAsset(
            video_id=video_id,
            version=1,
            mp4_path=mp4_path,
            thumbnail_path=thumb_path,
            mp4_url=mp4_url,
            thumbnail_url=thumb_url,
            created_at=datetime.now(tz=timezone.utc),
        )

        # Cleanup temp audio file
        Path(audio_path).unlink(missing_ok=True)

        logger.info("CinematicRunner: complete for video_id=%s", video_id)
        return visual, script, narration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_screenplay_markdown(script: CinematicScript) -> str:
    """Format the CinematicScript as readable Markdown for storage/review."""
    lines = [
        f"# {script.title}",
        f"",
        f"**Setting:** {script.setting}",
        f"**{script.hero1_name}:** {script.hero1_description}",
        f"**{script.hero2_name}:** {script.hero2_description}",
        f"**Duration:** {script.total_duration_seconds:.0f}s ({len(script.beats)} beats)",
        f"",
        "---",
        "",
    ]

    for beat in script.beats:
        timestamp = _seconds_to_timestamp(
            sum(b.duration_seconds for b in script.beats[:beat.beat_index])
        )
        sfx_str = ", ".join(beat.sfx_cues) if beat.sfx_cues else "none"

        if beat.dialogue_text and beat.character_id:
            char_name = script.hero1_name if beat.character_id == "hero1" else script.hero2_name
            lines.append(f"**[{timestamp}] {beat.beat_type.value.upper()}** ({beat.duration_seconds}s)")
            lines.append(f"  {char_name}: \"{beat.dialogue_text}\"")
            lines.append(f"  SFX: {sfx_str} | Camera: {beat.camera_angle}")
        else:
            lines.append(f"**[{timestamp}] {beat.beat_type.value.upper()}** ({beat.duration_seconds}s)")
            lines.append(f"  {beat.video_prompt[:100]}")
            lines.append(f"  SFX: {sfx_str} | Camera: {beat.camera_angle}")

        lines.append("")

    return "\n".join(lines)


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _generate_thumbnail(script: CinematicScript) -> bytes:
    """Generate a simple cinematic thumbnail."""
    from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    import io  # noqa: PLC0415

    img = Image.new("RGB", (1280, 720), (15, 15, 20))
    draw = ImageDraw.Draw(img)

    # Dark gradient background
    for y in range(720):
        r = int(15 + 20 * (y / 720))
        draw.line([(0, y), (1280, y)], fill=(r, r, r + 10))

    # Title text
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 64)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large

    # "VS" in the center
    draw.text((580, 300), "VS", fill=(255, 50, 50), font=font_large)

    # Character names
    draw.text((100, 310), script.hero1_name.upper(), fill=(200, 200, 255), font=font_small)
    draw.text((800, 310), script.hero2_name.upper(), fill=(255, 200, 200), font=font_small)

    # Title at bottom
    draw.text((100, 600), script.title[:50], fill=(255, 255, 255), font=font_small)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
