#!/usr/bin/env python3
"""Download CC0 sound effects for the cinematic pipeline.

Downloads royalty-free SFX from Pixabay and other CC0 sources.
Run once: python -m pipeline.cinematic.download_sfx

Stores files in src/pipeline/cinematic/assets/sfx/
"""

import os
import subprocess
import tempfile
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets" / "sfx"
MUSIC_DIR = Path(__file__).parent / "assets" / "music"

# We'll generate high-quality SFX using ffmpeg's audio synthesis
# These sound MUCH better than raw numpy sine waves because ffmpeg
# applies proper filters (reverb, distortion, EQ, compression)


def _generate_with_ffmpeg(output_path: str, filter_complex: str, duration: float = 1.0):
    """Generate an audio file using ffmpeg's lavfi audio source."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo",
        "-f", "lavfi", "-i", f"sine=frequency=1:duration={duration}",
        "-filter_complex", filter_complex,
        "-t", str(duration),
        "-c:a", "libmp3lame", "-b:a", "192k",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, timeout=30)


def generate_all_sfx():
    """Generate all SFX using ffmpeg synthesis with proper audio processing."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating cinematic SFX library...")

    sfx_commands = {
        # Heavy punch: low thump + mid snap + short noise burst
        "punch_heavy.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.4:c=pink:r=44100:a=0.7' "
            "-f lavfi -i 'sine=f=60:d=0.3' "
            "-filter_complex "
            "'[0]lowpass=f=800,afade=t=in:d=0.005,afade=t=out:st=0.1:d=0.3[n];"
            "[1]afade=t=in:d=0.001,afade=t=out:st=0.05:d=0.25[s];"
            "[n][s]amix=inputs=2:duration=shortest,aecho=0.6:0.3:40:0.4,"
            "bass=g=10:f=80,compand=attacks=0:decays=0.1:points=-80/-80|-30/-15|0/-3:gain=5' "
            "-t 0.5 -c:a libmp3lame -b:a 192k"
        ),
        # Light punch: quick snap
        "punch_light.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.2:c=white:r=44100:a=0.6' "
            "-filter_complex "
            "'bandpass=f=1500:w=800,afade=t=in:d=0.001,afade=t=out:st=0.03:d=0.15,"
            "aecho=0.5:0.2:20:0.3,compand=attacks=0:decays=0.05:points=-80/-80|-20/-10|0/-3:gain=5' "
            "-t 0.25 -c:a libmp3lame -b:a 192k"
        ),
        # Kick
        "kick.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.4:c=pink:r=44100:a=0.5' "
            "-f lavfi -i 'sine=f=45:d=0.4' "
            "-filter_complex "
            "'[0]lowpass=f=600,afade=t=out:st=0.05:d=0.3[n];"
            "[1]afade=t=in:d=0.001,afade=t=out:st=0.1:d=0.3[s];"
            "[n][s]amix=inputs=2,bass=g=12:f=60,"
            "aecho=0.7:0.4:30:0.3,compand=attacks=0:decays=0.1:points=-80/-80|-30/-15|0/-3:gain=5' "
            "-t 0.5 -c:a libmp3lame -b:a 192k"
        ),
        # Body slam
        "body_slam.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.6:c=brown:r=44100:a=0.8' "
            "-f lavfi -i 'sine=f=35:d=0.5' "
            "-filter_complex "
            "'[0]lowpass=f=400,afade=t=in:d=0.01,afade=t=out:st=0.15:d=0.4[n];"
            "[1]afade=t=in:d=0.005,afade=t=out:st=0.1:d=0.4[s];"
            "[n][s]amix=inputs=2,bass=g=15:f=50,"
            "aecho=0.8:0.5:60:0.5,compand=attacks=0:decays=0.15:points=-80/-80|-30/-15|0/-3:gain=6' "
            "-t 0.7 -c:a libmp3lame -b:a 192k"
        ),
        # Explosion: layered noise + deep bass + reverb
        "explosion.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=2.0:c=brown:r=44100:a=1.0' "
            "-f lavfi -i 'sine=f=25:d=1.5' "
            "-f lavfi -i 'anoisesrc=d=1.5:c=pink:r=44100:a=0.5' "
            "-filter_complex "
            "'[0]lowpass=f=500,afade=t=in:d=0.01,afade=t=out:st=0.5:d=1.5[boom];"
            "[1]afade=t=in:d=0.01,afade=t=out:st=0.3:d=1.2[bass];"
            "[2]highpass=f=2000,afade=t=in:d=0.05,afade=t=out:st=0.3:d=1.0[crackle];"
            "[boom][bass][crackle]amix=inputs=3:duration=longest,"
            "aecho=0.8:0.6:100:0.5,bass=g=15:f=40,"
            "compand=attacks=0.01:decays=0.3:points=-80/-80|-30/-10|0/-3:gain=8' "
            "-t 2.5 -c:a libmp3lame -b:a 192k"
        ),
        # Glass shatter
        "glass_shatter.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.5:c=white:r=44100:a=0.8' "
            "-filter_complex "
            "'highpass=f=3000,afade=t=in:d=0.001,afade=t=out:st=0.1:d=0.4,"
            "aecho=0.4:0.3:15:0.5,treble=g=5:f=5000,"
            "compand=attacks=0:decays=0.05:points=-80/-80|-20/-10|0/-3:gain=5' "
            "-t 0.6 -c:a libmp3lame -b:a 192k"
        ),
        # Concrete crumble
        "concrete_crumble.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=1.5:c=brown:r=44100:a=0.6' "
            "-f lavfi -i 'sine=f=50:d=1.2' "
            "-filter_complex "
            "'[0]lowpass=f=800,afade=t=in:d=0.05,afade=t=out:st=0.5:d=1.0[rumble];"
            "[1]afade=t=out:st=0.3:d=0.9,volume=0.4[bass];"
            "[rumble][bass]amix=inputs=2,aecho=0.7:0.5:80:0.4,"
            "compand=attacks=0.02:decays=0.2:points=-80/-80|-30/-15|0/-3:gain=5' "
            "-t 1.8 -c:a libmp3lame -b:a 192k"
        ),
        # Metal impact (clang)
        "metal_impact.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'sine=f=800:d=0.8' "
            "-f lavfi -i 'sine=f=1200:d=0.5' "
            "-f lavfi -i 'anoisesrc=d=0.1:c=white:r=44100:a=0.7' "
            "-filter_complex "
            "'[0]afade=t=in:d=0.001,afade=t=out:st=0.2:d=0.6[ring1];"
            "[1]afade=t=in:d=0.001,afade=t=out:st=0.1:d=0.4,volume=0.6[ring2];"
            "[2]afade=t=out:d=0.08[hit];"
            "[ring1][ring2][hit]amix=inputs=3,"
            "aecho=0.6:0.4:30:0.5,treble=g=3:f=3000,"
            "compand=attacks=0:decays=0.1:points=-80/-80|-20/-10|0/-3:gain=5' "
            "-t 1.0 -c:a libmp3lame -b:a 192k"
        ),
        # Whoosh
        "whoosh.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.5:c=pink:r=44100:a=0.7' "
            "-filter_complex "
            "'bandpass=f=1000:w=2000,afade=t=in:d=0.05,afade=t=out:st=0.15:d=0.3,"
            "aecho=0.3:0.2:10:0.3,volume=1.5' "
            "-t 0.5 -c:a libmp3lame -b:a 192k"
        ),
        # Sonic boom
        "sonic_boom.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=1.0:c=white:r=44100:a=0.9' "
            "-f lavfi -i 'sine=f=20:d=0.8' "
            "-filter_complex "
            "'[0]bandpass=f=200:w=400,afade=t=in:d=0.001,afade=t=out:st=0.2:d=0.8[crack];"
            "[1]afade=t=in:d=0.001,afade=t=out:st=0.2:d=0.6,volume=2.0[sub];"
            "[crack][sub]amix=inputs=2,aecho=0.8:0.6:60:0.5,"
            "bass=g=18:f=30,compand=attacks=0:decays=0.2:points=-80/-80|-30/-10|0/-3:gain=8' "
            "-t 1.2 -c:a libmp3lame -b:a 192k"
        ),
        # Landing heavy
        "landing_heavy.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.5:c=brown:r=44100:a=0.6' "
            "-f lavfi -i 'sine=f=40:d=0.4' "
            "-filter_complex "
            "'[0]lowpass=f=500,afade=t=in:d=0.005,afade=t=out:st=0.1:d=0.4[thud];"
            "[1]afade=t=in:d=0.005,afade=t=out:st=0.05:d=0.35,volume=1.5[bass];"
            "[thud][bass]amix=inputs=2,aecho=0.7:0.4:50:0.4,"
            "bass=g=12:f=50,compand=attacks=0:decays=0.1:points=-80/-80|-30/-15|0/-3:gain=6' "
            "-t 0.6 -c:a libmp3lame -b:a 192k"
        ),
        # Cape flutter
        "cape_flutter.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.8:c=pink:r=44100:a=0.3' "
            "-filter_complex "
            "'bandpass=f=800:w=600,tremolo=f=10:d=0.7,"
            "afade=t=in:d=0.1,afade=t=out:st=0.4:d=0.4,volume=0.8' "
            "-t 0.8 -c:a libmp3lame -b:a 192k"
        ),
        # Lightning
        "lightning.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=1.2:c=white:r=44100:a=1.0' "
            "-f lavfi -i 'sine=f=80:d=1.0' "
            "-filter_complex "
            "'[0]highpass=f=2000,afade=t=in:d=0.001,afade=t=out:d=0.05[crack];"
            "[0]lowpass=f=300,afade=t=in:d=0.1,afade=t=out:st=0.3:d=0.9,volume=0.5[rumble];"
            "[1]afade=t=in:d=0.05,afade=t=out:st=0.3:d=0.7,volume=0.6[bass];"
            "[crack][rumble][bass]amix=inputs=3:duration=longest,"
            "aecho=0.7:0.5:80:0.5,compand=attacks=0:decays=0.2:points=-80/-80|-30/-10|0/-3:gain=6' "
            "-t 1.5 -c:a libmp3lame -b:a 192k"
        ),
        # Energy blast
        "energy_blast.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'sine=f=200:d=0.6' "
            "-f lavfi -i 'anoisesrc=d=0.6:c=pink:r=44100:a=0.4' "
            "-filter_complex "
            "'[0]asetrate=44100*1.5,afade=t=out:st=0.1:d=0.5[sweep];"
            "[1]highpass=f=1000,afade=t=out:st=0.2:d=0.4[hiss];"
            "[sweep][hiss]amix=inputs=2,"
            "aecho=0.5:0.3:20:0.4,volume=1.5' "
            "-t 0.8 -c:a libmp3lame -b:a 192k"
        ),
        # Power charge (rising tone)
        "power_charge.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'sine=f=80:d=1.2' "
            "-f lavfi -i 'anoisesrc=d=1.2:c=pink:r=44100:a=0.3' "
            "-filter_complex "
            "'[0]asetrate=44100*0.5,atempo=2.0,afade=t=in:d=0.1,afade=t=out:st=0.9:d=0.3[rise];"
            "[1]highpass=f=3000,tremolo=f=15:d=0.5,afade=t=in:d=0.3,volume=0.4[sizzle];"
            "[rise][sizzle]amix=inputs=2,volume=1.3' "
            "-t 1.2 -c:a libmp3lame -b:a 192k"
        ),
        # Electric crackle
        "electric_crackle.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=0.8:c=white:r=44100:a=0.5' "
            "-filter_complex "
            "'highpass=f=4000,agate=threshold=-30:range=-50:attack=1:release=10,"
            "aecho=0.3:0.2:5:0.4,volume=1.5' "
            "-t 0.8 -c:a libmp3lame -b:a 192k"
        ),
        # Thunder rumble
        "thunder_rumble.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=3.0:c=brown:r=44100:a=0.5' "
            "-f lavfi -i 'sine=f=30:d=2.5' "
            "-filter_complex "
            "'[0]lowpass=f=200,afade=t=in:d=0.5,afade=t=out:st=1.5:d=1.5[rumble];"
            "[1]afade=t=in:d=0.3,afade=t=out:st=1.0:d=1.5,volume=0.5[bass];"
            "[rumble][bass]amix=inputs=2,aecho=0.8:0.6:150:0.6,"
            "bass=g=10:f=40,volume=0.8' "
            "-t 3.5 -c:a libmp3lame -b:a 192k"
        ),
        # Rain
        "rain.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=3.0:c=white:r=44100:a=0.2' "
            "-filter_complex "
            "'highpass=f=2000,lowpass=f=8000,volume=0.5,"
            "afade=t=in:d=0.3,afade=t=out:st=2.5:d=0.5' "
            "-t 3.0 -c:a libmp3lame -b:a 192k"
        ),
        # Wind
        "wind.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=3.0:c=pink:r=44100:a=0.3' "
            "-filter_complex "
            "'lowpass=f=1000,tremolo=f=0.3:d=0.8,"
            "afade=t=in:d=0.5,afade=t=out:st=2.0:d=1.0,volume=0.6' "
            "-t 3.0 -c:a libmp3lame -b:a 192k"
        ),
        # Fire crackle
        "fire_crackle.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anoisesrc=d=3.0:c=brown:r=44100:a=0.4' "
            "-filter_complex "
            "'bandpass=f=2000:w=1500,agate=threshold=-25:range=-40:attack=2:release=20,"
            "afade=t=in:d=0.2,afade=t=out:st=2.5:d=0.5,volume=0.7' "
            "-t 3.0 -c:a libmp3lame -b:a 192k"
        ),
        # Silence
        "silence.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'anullsrc=r=44100:cl=stereo' "
            "-t 1.5 -c:a libmp3lame -b:a 192k"
        ),
        # Bass drop
        "bass_drop.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'sine=f=25:d=1.5' "
            "-filter_complex "
            "'afade=t=in:d=0.001,afade=t=out:st=0.3:d=1.2,"
            "bass=g=20:f=30,volume=2.0,"
            "aecho=0.8:0.5:40:0.4' "
            "-t 1.5 -c:a libmp3lame -b:a 192k"
        ),
        # Heartbeat
        "heartbeat.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'sine=f=50:d=0.1' "
            "-f lavfi -i 'sine=f=40:d=0.08' "
            "-filter_complex "
            "'[0]afade=t=in:d=0.001,afade=t=out:d=0.08,adelay=0|0[lub];"
            "[1]afade=t=in:d=0.001,afade=t=out:d=0.06,adelay=150|150[dub];"
            "[lub][dub]amix=inputs=2,bass=g=15:f=50,"
            "aecho=0.5:0.3:30:0.3,volume=1.5' "
            "-t 1.0 -c:a libmp3lame -b:a 192k"
        ),
        # Shockwave
        "shockwave.mp3": (
            "ffmpeg -y -f lavfi -i "
            "'sine=f=200:d=1.0' "
            "-f lavfi -i 'anoisesrc=d=0.5:c=brown:r=44100:a=0.7' "
            "-filter_complex "
            "'[0]asetrate=44100*2,atempo=0.5,afade=t=out:st=0.2:d=0.8[sweep];"
            "[1]lowpass=f=300,afade=t=in:d=0.001,afade=t=out:st=0.1:d=0.4[boom];"
            "[sweep][boom]amix=inputs=2,"
            "aecho=0.8:0.6:60:0.5,bass=g=18:f=35,"
            "compand=attacks=0:decays=0.2:points=-80/-80|-30/-10|0/-3:gain=8' "
            "-t 1.2 -c:a libmp3lame -b:a 192k"
        ),
    }

    for filename, cmd in sfx_commands.items():
        output_path = ASSETS_DIR / filename
        full_cmd = f"{cmd} '{output_path}'"
        print(f"  Generating {filename}...")
        result = subprocess.run(
            full_cmd, shell=True, capture_output=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"    WARNING: failed — {result.stderr.decode()[:100]}")
        elif output_path.exists():
            size_kb = output_path.stat().st_size / 1024
            print(f"    ✓ {size_kb:.0f} KB")
        else:
            print(f"    WARNING: file not created")

    # Generate epic music bed
    print("\nGenerating epic music bed...")
    music_path = MUSIC_DIR / "epic_battle.mp3"
    # Layered drone + rhythm + strings — 3 minutes long
    music_cmd = (
        "ffmpeg -y "
        "-f lavfi -i 'sine=f=55:d=180' "         # A1 bass drone
        "-f lavfi -i 'sine=f=82.5:d=180' "       # E2 fifth
        "-f lavfi -i 'sine=f=110:d=180' "        # A2 octave
        "-f lavfi -i 'anoisesrc=d=180:c=brown:r=44100:a=0.1' "  # texture
        "-filter_complex "
        "'[0]volume=0.4,tremolo=f=0.1:d=0.3[drone];"
        "[1]volume=0.2,tremolo=f=0.15:d=0.4[fifth];"
        "[2]volume=0.15,tremolo=f=2:d=0.5[pulse];"
        "[3]lowpass=f=500,volume=0.15[texture];"
        "[drone][fifth][pulse][texture]amix=inputs=4:duration=longest,"
        "aecho=0.8:0.7:100:0.4,bass=g=8:f=60,"
        "afade=t=in:d=3,afade=t=out:st=175:d=5' "
        f"-t 180 -c:a libmp3lame -b:a 192k '{music_path}'"
    )
    result = subprocess.run(music_cmd, shell=True, capture_output=True, timeout=60)
    if music_path.exists():
        size_mb = music_path.stat().st_size / 1024 / 1024
        print(f"  ✓ epic_battle.mp3 — {size_mb:.1f} MB")
    else:
        print(f"  WARNING: music generation failed")

    print("\nDone! SFX library ready.")


if __name__ == "__main__":
    generate_all_sfx()
