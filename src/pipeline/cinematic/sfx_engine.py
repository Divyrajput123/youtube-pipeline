"""SFX Engine — manages a library of pre-generated fight sound effects.

Sound effects are stored as MP3 files in assets/sfx/. They are generated
once via download_sfx.py using ffmpeg's audio synthesis with proper EQ,
reverb, and compression for cinematic quality.

If assets don't exist, falls back to generating them on the fly.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pipeline.cinematic.models import SFXCue

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ASSETS_DIR = Path(__file__).parent / "assets" / "sfx"
_MUSIC_DIR = Path(__file__).parent / "assets" / "music"

# Map SFXCue enum → filename
_SFX_FILENAMES: dict[SFXCue, str] = {
    SFXCue.PUNCH_HEAVY: "punch_heavy.mp3",
    SFXCue.PUNCH_LIGHT: "punch_light.mp3",
    SFXCue.KICK: "kick.mp3",
    SFXCue.BODY_SLAM: "body_slam.mp3",
    SFXCue.EXPLOSION: "explosion.mp3",
    SFXCue.GLASS_SHATTER: "glass_shatter.mp3",
    SFXCue.CONCRETE_CRUMBLE: "concrete_crumble.mp3",
    SFXCue.METAL_IMPACT: "metal_impact.mp3",
    SFXCue.WHOOSH: "whoosh.mp3",
    SFXCue.SONIC_BOOM: "sonic_boom.mp3",
    SFXCue.LANDING_HEAVY: "landing_heavy.mp3",
    SFXCue.CAPE_FLUTTER: "cape_flutter.mp3",
    SFXCue.LIGHTNING: "lightning.mp3",
    SFXCue.ENERGY_BLAST: "energy_blast.mp3",
    SFXCue.POWER_CHARGE: "power_charge.mp3",
    SFXCue.ELECTRIC_CRACKLE: "electric_crackle.mp3",
    SFXCue.THUNDER_RUMBLE: "thunder_rumble.mp3",
    SFXCue.RAIN: "rain.mp3",
    SFXCue.WIND: "wind.mp3",
    SFXCue.FIRE_CRACKLE: "fire_crackle.mp3",
    SFXCue.SILENCE: "silence.mp3",
    SFXCue.BASS_DROP: "bass_drop.mp3",
    SFXCue.HEARTBEAT: "heartbeat.mp3",
    SFXCue.SHOCKWAVE: "shockwave.mp3",
}


# ---------------------------------------------------------------------------
# SFXEngine
# ---------------------------------------------------------------------------


class SFXEngine:
    """Manages the SFX library. Loads MP3 files from assets/sfx/.

    If asset files don't exist, runs download_sfx.py to generate them first.

    Usage:
        engine = SFXEngine()
        mp3_bytes = engine.get("punch_heavy")
        mp3_bytes = engine.get("bone_crunch")  # unknown cue — auto-generated
    """

    def __init__(self) -> None:
        self._cache: dict[str, bytes] = {}
        self._ensure_assets()

    def _ensure_assets(self) -> None:
        """Check if SFX assets exist. Generate them if not."""
        if not _ASSETS_DIR.exists() or not any(_ASSETS_DIR.glob("*.mp3")):
            logger.info("SFXEngine: assets not found, generating...")
            from pipeline.cinematic.download_sfx import generate_all_sfx
            generate_all_sfx()

    def get(self, cue: str) -> bytes:
        """Return MP3 bytes for the given SFX cue name.

        Known cues load from pre-generated files. Unknown cues are
        auto-generated using ffmpeg based on keyword analysis of the name.
        """
        if cue not in self._cache:
            filepath = _ASSETS_DIR / f"{cue}.mp3"
            if filepath.exists():
                self._cache[cue] = filepath.read_bytes()
            else:
                logger.info("SFXEngine: auto-generating unknown cue '%s'", cue)
                self._cache[cue] = self._auto_generate(cue)
        return self._cache[cue]

    def _auto_generate(self, cue_name: str) -> bytes:
        """Generate SFX on the fly based on keyword analysis of the cue name."""
        name = cue_name.lower()
        if any(w in name for w in ["punch", "hit", "slap", "crunch", "crack", "bone", "snap"]):
            cmd = ('ffmpeg -y -f lavfi -i "anoisesrc=d=0.4:c=pink:r=44100:a=0.8" '
                   '-f lavfi -i "sine=f=55:d=0.3" -filter_complex '
                   '"[0]lowpass=f=900,afade=t=in:d=0.001,afade=t=out:st=0.08:d=0.3[n];'
                   '[1]afade=t=in:d=0.001,afade=t=out:st=0.05:d=0.25[s];'
                   '[n][s]amix=inputs=2,aecho=0.6:0.3:35:0.4,bass=g=12:f=70,'
                   'compand=attacks=0:decays=0.1:points=-80/-80|-30/-15|0/-3:gain=6" '
                   '-t 0.5 -c:a libmp3lame -b:a 192k')
        elif any(w in name for w in ["boom", "explod", "blast", "detonate", "crash"]):
            cmd = ('ffmpeg -y -f lavfi -i "anoisesrc=d=2.0:c=brown:r=44100:a=0.9" '
                   '-f lavfi -i "sine=f=28:d=1.5" -filter_complex '
                   '"[0]lowpass=f=500,afade=t=in:d=0.01,afade=t=out:st=0.5:d=1.5[boom];'
                   '[1]afade=t=in:d=0.01,afade=t=out:st=0.3:d=1.2,volume=1.5[bass];'
                   '[boom][bass]amix=inputs=2,aecho=0.8:0.6:80:0.5,bass=g=15:f=35" '
                   '-t 2.0 -c:a libmp3lame -b:a 192k')
        elif any(w in name for w in ["swoosh", "swipe", "rush", "fly", "dash", "speed", "whoosh"]):
            cmd = ('ffmpeg -y -f lavfi -i "anoisesrc=d=0.5:c=pink:r=44100:a=0.7" '
                   '-filter_complex "bandpass=f=1200:w=2000,afade=t=in:d=0.05,'
                   'afade=t=out:st=0.15:d=0.3,aecho=0.3:0.2:10:0.3,volume=1.5" '
                   '-t 0.5 -c:a libmp3lame -b:a 192k')
        elif any(w in name for w in ["roar", "scream", "yell", "growl"]):
            cmd = ('ffmpeg -y -f lavfi -i "sine=f=120:d=1.5" '
                   '-f lavfi -i "anoisesrc=d=1.0:c=pink:r=44100:a=0.4" -filter_complex '
                   '"[0]tremolo=f=5:d=0.7,volume=0.8[tone];[1]lowpass=f=800,volume=0.4[noise];'
                   '[tone][noise]amix=inputs=2,afade=t=in:d=0.1,afade=t=out:st=0.8:d=0.7,'
                   'bass=g=10:f=100,aecho=0.7:0.5:50:0.4" -t 1.5 -c:a libmp3lame -b:a 192k')
        elif any(w in name for w in ["electric", "spark", "zap", "energy", "charge"]):
            cmd = ('ffmpeg -y -f lavfi -i "anoisesrc=d=0.8:c=white:r=44100:a=0.5" '
                   '-filter_complex "highpass=f=4000,tremolo=f=20:d=0.8,'
                   'afade=t=in:d=0.05,afade=t=out:st=0.4:d=0.4,volume=1.2" '
                   '-t 0.8 -c:a libmp3lame -b:a 192k')
        elif any(w in name for w in ["metal", "clang", "ring", "shield"]):
            cmd = ('ffmpeg -y -f lavfi -i "sine=f=800:d=0.8" '
                   '-f lavfi -i "anoisesrc=d=0.1:c=white:r=44100:a=0.7" -filter_complex '
                   '"[0]afade=t=in:d=0.001,afade=t=out:st=0.2:d=0.6[ring];'
                   '[1]afade=t=out:d=0.08[hit];'
                   '[ring][hit]amix=inputs=2,aecho=0.6:0.4:30:0.5,treble=g=3:f=3000" '
                   '-t 0.8 -c:a libmp3lame -b:a 192k')
        else:
            cmd = ('ffmpeg -y -f lavfi -i "anoisesrc=d=0.5:c=pink:r=44100:a=0.6" '
                   '-f lavfi -i "sine=f=80:d=0.4" -filter_complex '
                   '"[0]lowpass=f=1000,afade=t=out:st=0.1:d=0.4[n];'
                   '[1]afade=t=out:st=0.1:d=0.3[s];'
                   '[n][s]amix=inputs=2,aecho=0.5:0.3:30:0.3,volume=1.2" '
                   '-t 0.5 -c:a libmp3lame -b:a 192k')

        output_path = _ASSETS_DIR / f"{cue_name}.mp3"
        full_cmd = f'{cmd} "{output_path}"'
        result = subprocess.run(full_cmd, shell=True, capture_output=True, timeout=15)
        if result.returncode == 0 and output_path.exists():
            return output_path.read_bytes()
        fallback = _ASSETS_DIR / "punch_heavy.mp3"
        if fallback.exists():
            return fallback.read_bytes()
        return self._generate_silence()

    def get_music_path(self) -> Optional[str]:
        """Return path to the epic battle music MP3, or None."""
        music_path = _MUSIC_DIR / "epic_battle.mp3"
        if music_path.exists():
            return str(music_path)
        return None

    def get_all_for_beat(self, sfx_cues: list[str]) -> list[bytes]:
        """Return MP3 bytes for all SFX cues in a beat."""
        return [self.get(cue) for cue in sfx_cues]

    @staticmethod
    def _generate_silence() -> bytes:
        """Generate a 1-second silent MP3 via ffmpeg."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
             "-t", "1", "-c:a", "libmp3lame", "-b:a", "128k", path],
            capture_output=True, timeout=10,
        )
        data = Path(path).read_bytes()
        Path(path).unlink(missing_ok=True)
        return data
