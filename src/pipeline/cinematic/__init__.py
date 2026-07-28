"""Cinematic fight pipeline — full cinematic mode with multi-character dialogue,
SFX layering, background music, and beat-synced video generation.

This module provides an alternative to the standard narration-based pipeline.
Instead of a narrator reading analysis over visuals, it produces a short film
with character dialogue, fight SFX, and epic music.
"""

from pipeline.cinematic.models import (
    Beat,
    BeatType,
    CinematicScript,
    CharacterVoice,
    SFXCue,
)
from pipeline.cinematic.script_writer import CinematicScriptWriter
from pipeline.cinematic.sfx_engine import SFXEngine
from pipeline.cinematic.audio_mixer import AudioMixer
from pipeline.cinematic.dialogue_generator import DialogueGenerator

__all__ = [
    "AudioMixer",
    "Beat",
    "BeatType",
    "CharacterVoice",
    "CinematicScript",
    "CinematicScriptWriter",
    "DialogueGenerator",
    "SFXCue",
    "SFXEngine",
]
