#!/usr/bin/env python3
"""Standalone test for the cinematic fight pipeline.

Runs the full flow: screenplay → dialogue → SFX + music mix → video → final MP4.
Outputs the final video to ./pipeline_output/cinematic_test/

Usage:
    python test_cinematic.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv(".env", override=True)

sys.path.insert(0, "src")

from pipeline.cinematic.script_writer import CinematicScriptWriter
from pipeline.cinematic.sfx_engine import SFXEngine
from pipeline.cinematic.dialogue_generator import DialogueGenerator
from pipeline.cinematic.audio_mixer import AudioMixer
from pipeline.cinematic.visual_generator import CinematicVisualGenerator
from pipeline.cinematic.runner import CinematicRunner
from pipeline.models import TopicEntry


async def main():
    # Setup output directory
    out_dir = Path("pipeline_output/cinematic_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a test topic
    topic = TopicEntry(
        title="Superman vs Thor: Ultimate Power Battle",
        composite_score=0.95,
        recency_hours=1.0,
        source_query_timestamp=datetime.now(tz=timezone.utc),
        search_volume_signal=90.0,
        relevance_tags_matched=["superman", "thor", "battle"],
    )

    video_id = "cinematic-test-001"

    print("=" * 60)
    print("CINEMATIC PIPELINE STANDALONE TEST")
    print("=" * 60)
    print(f"Topic: {topic.title}")
    print(f"Video ID: {video_id}")
    print()

    # --- Step 1: Generate screenplay ---
    print("[1/5] Generating cinematic screenplay via Claude...")
    writer = CinematicScriptWriter()
    script = await writer.generate(
        topic=topic,
        video_id=video_id,
        duration_seconds=60,  # 1 minute for testing (shorter)
    )
    print(f"  ✓ Generated {len(script.beats)} beats, {script.total_duration_seconds:.0f}s total")
    print(f"  ✓ {script.hero1_name} vs {script.hero2_name}")
    print(f"  ✓ Setting: {script.setting[:60]}...")

    # Save screenplay
    from pipeline.cinematic.runner import _format_screenplay_markdown
    screenplay_md = _format_screenplay_markdown(script)
    (out_dir / "screenplay.md").write_text(screenplay_md)
    print(f"  ✓ Saved screenplay to {out_dir}/screenplay.md")
    print()

    # --- Step 2: Generate dialogue ---
    print("[2/5] Generating character dialogue via ElevenLabs...")
    dialogue_gen = DialogueGenerator()
    dialogue_audio = await dialogue_gen.generate_all_dialogue(script)
    dialogue_count = sum(1 for v in dialogue_audio.values() if v)
    print(f"  ✓ Generated {dialogue_count} dialogue lines")
    print()

    # --- Step 3: Mix audio ---
    print("[3/5] Mixing SFX + dialogue + music...")
    sfx_engine = SFXEngine()
    mixer = AudioMixer(sfx_engine=sfx_engine)
    mixed_wav = await mixer.mix(script, dialogue_audio)
    audio_path = out_dir / "mixed_audio.wav"
    audio_path.write_bytes(mixed_wav)
    print(f"  ✓ Mixed audio: {len(mixed_wav) / 1024:.0f} KB")
    print(f"  ✓ Saved to {audio_path}")
    print()

    # --- Step 4: Generate video (Image-to-Video: Flux Pro + Kling) ---
    print("[4/5] Generating video: Flux Pro frames → Kling animation...")

    import os
    import tempfile
    import subprocess
    from pathlib import Path as _Path

    from pipeline.cinematic.image_to_video import generate_beat_clip

    fal_key = os.environ.get("FAL_KEY", "")
    kling_key = os.environ.get("KLING_API_KEY", "")

    if fal_key and kling_key:
        print(f"  Using Flux Pro (fal.ai) + Kling image-to-video")
        print(f"  Estimated cost: ~${len(script.beats) * 0.17:.2f} ({len(script.beats)} beats × $0.17)")

        with tempfile.TemporaryDirectory() as tmpdir:
            clip_paths = []

            for i, beat in enumerate(script.beats):
                print(f"    Beat {i+1}/{len(script.beats)}: {beat.beat_type.value} ({beat.duration_seconds}s)...", end=" ", flush=True)
                try:
                    mp4_bytes = await generate_beat_clip(
                        beat=beat,
                        script=script,
                        beat_index=i,
                        total_beats=len(script.beats),
                    )
                    clip_path = _Path(tmpdir) / f"beat_{i:03d}.mp4"
                    clip_path.write_bytes(mp4_bytes)
                    clip_paths.append(str(clip_path))
                    print(f"✓ ({len(mp4_bytes)//1024}KB)")
                except Exception as exc:
                    print(f"✗ ({exc})")
                    # Fallback: black frame
                    from pipeline.cinematic.visual_generator import _create_fallback_clip
                    fallback = _create_fallback_clip(beat, beat.duration_seconds, tmpdir, i)
                    clip_paths.append(fallback)

            # Stitch all clips together
            concat_list = _Path(tmpdir) / "concat.txt"
            with open(concat_list, "w") as f:
                for path in clip_paths:
                    f.write(f"file '{path}'\n")

            concat_out = _Path(tmpdir) / "concat.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", str(concat_list), "-c:v", "libx264", "-preset", "fast",
                 "-pix_fmt", "yuv420p", "-r", "24", str(concat_out)],
                capture_output=True, timeout=120, check=True,
            )

            # Mux with audio
            final_out = _Path(tmpdir) / "final.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(concat_out), "-i", str(audio_path),
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                 "-shortest", "-movflags", "+faststart", str(final_out)],
                capture_output=True, timeout=60, check=True,
            )

            final_mp4 = final_out.read_bytes()
    else:
        print("  Missing FAL_KEY or KLING_API_KEY — using fallback stills")
        visual_gen = CinematicVisualGenerator(video_client=None)
        final_mp4 = await visual_gen.generate_all(
            script=script,
            audio_wav_path=str(audio_path),
        )
    mp4_path = out_dir / "cinematic_fight.mp4"
    mp4_path.write_bytes(final_mp4)
    print(f"  ✓ Final MP4: {len(final_mp4) / 1024 / 1024:.1f} MB")
    print(f"  ✓ Saved to {mp4_path}")
    print()

    # --- Step 5: Summary ---
    print("[5/5] Done!")
    print()
    print("=" * 60)
    print("OUTPUT FILES:")
    print(f"  📝 Screenplay:  {out_dir}/screenplay.md")
    print(f"  🔊 Mixed audio: {out_dir}/mixed_audio.wav")
    print(f"  🎬 Final video: {out_dir}/cinematic_fight.mp4")
    print("=" * 60)
    print()
    print("Play the video to hear the SFX + dialogue + music mix!")
    print("Note: Video uses fallback stills (no API calls). With RunPod")
    print("      enabled, each beat would get an AI-generated clip.")


if __name__ == "__main__":
    asyncio.run(main())
