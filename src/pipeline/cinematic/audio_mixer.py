"""Audio Mixer — combines dialogue, SFX, and music into a final audio track.

Uses ffmpeg for all audio operations (decoding, mixing, filtering).
Produces a single MP3 file with proper levels:
  - Music: -12dB base, ducked to -18dB during dialogue
  - SFX: 0dB (full volume)
  - Dialogue: -3dB
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pipeline.cinematic.models import Beat, BeatType, CinematicScript, SFXCue
from pipeline.cinematic.sfx_engine import SFXEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AudioMixer
# ---------------------------------------------------------------------------


class AudioMixer:
    """Mixes dialogue, SFX, and music using ffmpeg.

    Strategy: build each beat's audio as a separate file, then concatenate.
    This ensures perfect timing alignment between beats.
    """

    def __init__(self, sfx_engine: Optional[SFXEngine] = None) -> None:
        self._sfx = sfx_engine or SFXEngine()

    async def mix(
        self,
        script: CinematicScript,
        dialogue_audio: dict[int, bytes],
    ) -> bytes:
        """Mix all audio layers into a final WAV file.

        Args:
            script: The cinematic script with beat timing and SFX cues.
            dialogue_audio: Dict of beat_index -> MP3 bytes for dialogue.

        Returns:
            Final mixed audio as WAV bytes (16-bit stereo, 44100Hz).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            beat_audio_files: list[str] = []

            for beat in script.beats:
                beat_file = self._mix_single_beat(
                    beat=beat,
                    script=script,
                    dialogue_mp3=dialogue_audio.get(beat.beat_index),
                    tmpdir=tmpdir,
                )
                beat_audio_files.append(beat_file)

            # Concatenate all beat audio files
            concat_list = Path(tmpdir) / "concat.txt"
            with open(concat_list, "w") as f:
                for path in beat_audio_files:
                    f.write(f"file '{path}'\n")

            # Concat into one file
            concat_out = Path(tmpdir) / "concat.wav"
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_list),
                    "-c:a", "pcm_s16le",
                    "-ar", "44100", "-ac", "2",
                    str(concat_out),
                ],
                capture_output=True, timeout=60, check=True,
            )

            # Now mix in the music bed
            music_path = self._sfx.get_music_path()
            if music_path:
                final_out = Path(tmpdir) / "final.wav"
                # Music at -12dB, mixed content at 0dB
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-i", str(concat_out),
                        "-i", music_path,
                        "-filter_complex",
                        "[1]volume=0.25,afade=t=in:d=2,afade=t=out:st="
                        f"{script.total_duration_seconds - 3}:d=3[music];"
                        "[0][music]amix=inputs=2:duration=first:dropout_transition=2",
                        "-c:a", "pcm_s16le",
                        "-ar", "44100", "-ac", "2",
                        str(final_out),
                    ],
                    capture_output=True, timeout=60, check=True,
                )
                result = final_out.read_bytes()
            else:
                result = concat_out.read_bytes()

            logger.info(
                "AudioMixer: mixed %.1fs audio — %d beats, %.1f MB",
                script.total_duration_seconds,
                len(script.beats),
                len(result) / 1e6,
            )
            return result

    def _mix_single_beat(
        self,
        beat: Beat,
        script: CinematicScript,
        dialogue_mp3: Optional[bytes],
        tmpdir: str,
    ) -> str:
        """Mix audio for a single beat: SFX + optional dialogue.

        Returns path to the beat's WAV file (exact duration).
        """
        beat_path = Path(tmpdir) / f"beat_{beat.beat_index:03d}.wav"
        duration = beat.duration_seconds

        # Start with silence at the exact beat duration
        silence_path = Path(tmpdir) / f"silence_{beat.beat_index:03d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
                "-t", str(duration),
                "-c:a", "pcm_s16le",
                str(silence_path),
            ],
            capture_output=True, timeout=10,
        )

        # Collect audio inputs to mix
        inputs: list[str] = [str(silence_path)]
        volumes: list[str] = ["1.0"]  # silence is just the base

        # Determine if this beat has dialogue — if so, duck the SFX
        has_dialogue = bool(dialogue_mp3 and beat.dialogue_text)

        # Ambient SFX that should be very quiet when dialogue plays
        _AMBIENT_SFX = {"rain", "wind", "fire_crackle", "thunder_rumble", "electric_crackle"}

        # Add SFX
        for i, cue in enumerate(beat.sfx_cues):
            sfx_bytes = self._sfx.get(cue)
            sfx_path = Path(tmpdir) / f"sfx_{beat.beat_index:03d}_{i}.mp3"
            sfx_path.write_bytes(sfx_bytes)
            inputs.append(str(sfx_path))

            # Duck ambient SFX when dialogue is present
            if has_dialogue and cue in _AMBIENT_SFX:
                volumes.append("0.15")  # Very quiet behind dialogue
            elif has_dialogue:
                volumes.append("0.5")   # Impact SFX still audible but reduced
            else:
                volumes.append("1.0")   # Full volume when no dialogue

        # Add dialogue
        if has_dialogue:
            dlg_path = Path(tmpdir) / f"dlg_{beat.beat_index:03d}.mp3"
            dlg_path.write_bytes(dialogue_mp3)
            inputs.append(str(dlg_path))
            volumes.append("1.5")  # Dialogue significantly louder than everything

        # Mix everything together using ffmpeg
        if len(inputs) == 1:
            # Just silence — copy it
            beat_path = silence_path
        else:
            # Build filter complex for mixing
            # Use duration=longest so dialogue is NEVER cut off
            filter_parts = []
            for i in range(len(inputs)):
                vol = volumes[i]
                filter_parts.append(f"[{i}]volume={vol}[a{i}]")

            mix_inputs = "".join(f"[a{i}]" for i in range(len(inputs)))
            filter_parts.append(
                f"{mix_inputs}amix=inputs={len(inputs)}:duration=longest:normalize=0"
            )

            filter_complex = ";".join(filter_parts)

            cmd = ["ffmpeg", "-y"]
            for inp in inputs:
                cmd.extend(["-i", inp])
            cmd.extend([
                "-filter_complex", filter_complex,
                "-c:a", "pcm_s16le",
                "-ar", "44100", "-ac", "2",
                str(beat_path),
            ])

            result = subprocess.run(cmd, capture_output=True, timeout=30)
            if result.returncode != 0:
                logger.warning(
                    "AudioMixer: beat %d mix failed: %s — using silence",
                    beat.beat_index,
                    result.stderr.decode()[:200],
                )
                beat_path = silence_path

        return str(beat_path)
