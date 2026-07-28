"""Dialogue Generator — produces character voice lines via ElevenLabs.

Generates short dialogue audio for each DIALOGUE beat using different
ElevenLabs voice IDs per character. Returns WAV/MP3 bytes for each line.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pipeline.cinematic.models import Beat, BeatType, CharacterVoice, CinematicScript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default voice IDs — two distinct voices for hero1/hero2
# These are ElevenLabs library voices that sound distinct from each other.
# hero1: deep/heroic, hero2: commanding/powerful
# ---------------------------------------------------------------------------

_DEFAULT_HERO1_VOICE = "SOYHLrjzK2X1ezoPC6cr"   # "Harry" — Fierce Warrior (hero)
_DEFAULT_HERO2_VOICE = "nPczCjzI2devNBz1zQrb"   # "Brian" — Deep, Resonant (god/villain)

# Fallback free voice if the above aren't available
_FALLBACK_VOICE = "CwhRBWXzGAHq8TQ4Fs17"  # "Roger" — free tier

# Character archetype → voice mapping
# The system picks the best voice based on keywords in the character description
_ARCHETYPE_VOICES: dict[str, str] = {
    # Gods / cosmic / Norse / divine characters → Deep, Resonant, booming
    "god": "nPczCjzI2devNBz1zQrb",       # Brian - godlike
    "thunder": "nPczCjzI2devNBz1zQrb",    # Brian
    "norse": "nPczCjzI2devNBz1zQrb",      # Brian
    "cosmic": "nPczCjzI2devNBz1zQrb",     # Brian
    "titan": "nPczCjzI2devNBz1zQrb",      # Brian
    "king": "nPczCjzI2devNBz1zQrb",       # Brian
    "ancient": "JBFqnCBsd6RMkjVDRZzb",    # George - wise/old
    "wizard": "JBFqnCBsd6RMkjVDRZzb",     # George
    "sorcerer": "JBFqnCBsd6RMkjVDRZzb",   # George

    # Warriors / fighters / aggressive → Fierce Warrior
    "warrior": "SOYHLrjzK2X1ezoPC6cr",    # Harry - fierce
    "soldier": "SOYHLrjzK2X1ezoPC6cr",    # Harry
    "knight": "SOYHLrjzK2X1ezoPC6cr",     # Harry
    "assassin": "SOYHLrjzK2X1ezoPC6cr",   # Harry
    "fighter": "SOYHLrjzK2X1ezoPC6cr",    # Harry
    "vigilante": "SOYHLrjzK2X1ezoPC6cr",  # Harry

    # Villains / tricksters / menacing → Husky Trickster
    "villain": "N2lVS1w4EtoT3dr4eOWO",    # Callum - trickster
    "trickster": "N2lVS1w4EtoT3dr4eOWO",  # Callum
    "evil": "N2lVS1w4EtoT3dr4eOWO",       # Callum
    "dark": "N2lVS1w4EtoT3dr4eOWO",       # Callum
    "demon": "N2lVS1w4EtoT3dr4eOWO",      # Callum

    # Leaders / authority / Superman-type → Dominant, Firm
    "leader": "pNInz6obpgDQGcFmaJgB",     # Adam - dominant
    "captain": "pNInz6obpgDQGcFmaJgB",    # Adam
    "commander": "pNInz6obpgDQGcFmaJgB",  # Adam
    "protector": "IKne3meq5aSn9XLyUdCD",  # Charlie - confident
    "hero": "IKne3meq5aSn9XLyUdCD",       # Charlie
    "super": "IKne3meq5aSn9XLyUdCD",      # Charlie

    # Speedsters / young / energetic
    "speed": "IKne3meq5aSn9XLyUdCD",      # Charlie - energetic
    "young": "IKne3meq5aSn9XLyUdCD",      # Charlie
    "spider": "IKne3meq5aSn9XLyUdCD",     # Charlie
}


# ---------------------------------------------------------------------------
# DialogueGenerator
# ---------------------------------------------------------------------------


class DialogueGenerator:
    """Generates character dialogue audio via ElevenLabs TTS.

    Uses different voice IDs per character to create distinct voices.
    Falls back to a single voice if specific voice IDs aren't available.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self._api_key:
            logger.warning("DialogueGenerator: ELEVENLABS_API_KEY not set")

    def _get_voice_id(self, character_id: str, script: CinematicScript) -> str:
        """Resolve the ElevenLabs voice ID for a character.

        Uses two clear, distinct voices:
        - hero1: Adam (Dominant, Firm) — strong authority, always clear
        - hero2: Brian (Deep, Resonant) — powerful, godlike

        Both are proven clear and loud voices on ElevenLabs.
        """
        # Check explicit assignment first
        for voice in script.voices:
            if voice.character_id == character_id:
                return voice.voice_id

        # Use the two clearest, most distinct voices
        if character_id == "hero1":
            return "pNInz6obpgDQGcFmaJgB"  # Adam — Dominant, Firm, always clear
        elif character_id == "hero2":
            return "nPczCjzI2devNBz1zQrb"  # Brian — Deep, Resonant, powerful
        return _FALLBACK_VOICE

    async def generate_line(
        self,
        text: str,
        character_id: str,
        script: CinematicScript,
    ) -> bytes:
        """Generate audio for a single dialogue line.

        Args:
            text: The dialogue text to speak.
            character_id: Which character is speaking.
            script: The full script (for voice lookup).

        Returns:
            MP3 bytes of the spoken line.

        Raises:
            RuntimeError: If ElevenLabs API fails.
        """
        import httpx  # noqa: PLC0415

        voice_id = self._get_voice_id(character_id, script)

        # Add emotion cues for more dramatic delivery:
        dramatic_text = self._add_emotion_cues(text, character_id)

        # Skip if nothing to speak after cleaning (was pure stage direction)
        if not dramatic_text:
            logger.info("DialogueGenerator: skipping empty dialogue after cleaning for %s", character_id)
            return b""

        logger.info(
            "DialogueGenerator: generating '%s' for %s (voice=%s)",
            dramatic_text[:30], character_id, voice_id,
        )

        # Use different settings per character for more contrast
        # hero2 (Thor/god) = lower stability for more dramatic variation
        # hero1 (Superman/hero) = slightly higher for controlled authority
        if character_id == "hero2":
            stability = 0.35
            similarity = 0.75
        else:
            stability = 0.50
            similarity = 0.80

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": dramatic_text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity,
                    },
                },
            )
            response.raise_for_status()
            return response.content

    @staticmethod
    def _add_emotion_cues(text: str, character_id: str) -> str:
        """Clean dialogue text for ElevenLabs delivery.

        Strips stage directions, parenthetical notes, and action cues
        that ElevenLabs would read literally (e.g. "(heavy breathing)",
        "*gasping*", "[screaming]").

        Keeps only the actual spoken words.
        """
        import re

        # Remove parenthetical stage directions: (heavy breathing), (through gritted teeth)
        text = re.sub(r"\([^)]*\)", "", text)

        # Remove bracketed directions: [screaming], [whispering]
        text = re.sub(r"\[[^\]]*\]", "", text)

        # Remove asterisk actions: *gasping*, *clenches fist*
        text = re.sub(r"\*[^*]*\*", "", text)

        # Remove common non-speech cues that Claude might include
        noise_words = [
            "heavy breathing", "breathing heavily", "gasping", "panting",
            "grunts", "groans", "screams", "whispers", "mutters",
            "coughs", "spits blood", "wipes blood",
        ]
        for noise in noise_words:
            text = re.sub(rf"\b{noise}\b", "", text, flags=re.IGNORECASE)

        # Clean up double spaces and trailing punctuation mess
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[,.\s]+", "", text)
        text = re.sub(r"[,.\s]+$", "", text)

        # Replace multiple ellipses with a single comma (keeps flow, no robot pauses)
        # "My world... is not... yours" → "My world, is not, yours"
        # But keep a single ellipsis at the end for trailing off
        # Actually — remove ALL ellipses. They always sound robotic in TTS.
        text = text.replace("...", ", ")
        text = text.replace(" ,", ",")
        # Clean double commas
        text = text.replace(",,", ",")

        # If nothing left after cleaning, return a short grunt sound
        if not text or len(text) < 2:
            return ""

        # Skip lines that are ONLY non-word screams (no actual words)
        # Keep short exclamations like "ENOUGH!", "COME ON!", "KNEEL!" — those are real dialogue
        import re as _re
        # Only skip if it's PURELY vowel screams with no real words
        if _re.match(r"^[AaHhUuOoEe]+[RrGgHh!.\s]*$", text.strip()):
            return ""

        # Ensure it ends with punctuation for clean delivery
        if not text.endswith((".", "!", "?", "—")):
            text = text + "."

        return text

    async def generate_all_dialogue(
        self,
        script: CinematicScript,
    ) -> dict[int, bytes]:
        """Generate audio for all dialogue beats in the script.

        Args:
            script: The full CinematicScript.

        Returns:
            Dict mapping beat_index → MP3 bytes for each dialogue beat.
        """
        results: dict[int, bytes] = {}

        for beat in script.beats:
            if beat.dialogue_text and beat.character_id:
                try:
                    audio = await self.generate_line(
                        text=beat.dialogue_text,
                        character_id=beat.character_id,
                        script=script,
                    )
                    results[beat.beat_index] = audio
                except Exception as exc:
                    logger.warning(
                        "DialogueGenerator: failed for beat %d ('%s'): %s",
                        beat.beat_index, beat.dialogue_text, exc,
                    )
                    # Return empty bytes — the mixer will handle silence gracefully
                    results[beat.beat_index] = b""

        logger.info(
            "DialogueGenerator: generated %d/%d dialogue lines",
            sum(1 for v in results.values() if v),
            len(results),
        )
        return results
