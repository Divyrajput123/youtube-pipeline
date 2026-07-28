"""CinematicScriptWriter — generates a structured fight screenplay via Claude.

Outputs a CinematicScript with timed beats, dialogue, SFX cues, and video
prompts suitable for the audio mixer and visual generator.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from pipeline.cinematic.models import (
    Beat,
    BeatType,
    CinematicScript,
    CharacterVoice,
    SFXCue,
)
from pipeline.models import TopicEntry
from pipeline.script_writer import ClaudeClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOKENS = 8192
_TARGET_DURATION_S = 120  # 2-minute fight by default


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_cinematic_prompt(
    topic: TopicEntry,
    duration_seconds: int = _TARGET_DURATION_S,
) -> str:
    """Build the Claude prompt for generating a cinematic fight script."""

    sfx_options = ", ".join(f'"{s.value}"' for s in SFXCue)
    beat_types = ", ".join(f'"{b.value}"' for b in BeatType)

    return f"""You are a Hollywood action movie screenwriter creating a short superhero fight sequence for an AI video generator.

TOPIC: {topic.title}

Write a {duration_seconds}-second fight sequence as a JSON object. The fight should be DRAMATIC, VISCERAL, and CINEMATIC — like a Marvel movie climax.

STRUCTURE RULES:
- Start with 1-2 TENSION beats (characters face off, the air is thick with hatred)
- Then CONTINUOUS ACTION with dialogue layered on top — characters talk WHILE fighting
- Most beats should be ACTION or IMPACT type WITH dialogue attached
- Characters REACT mid-fight: grunts of pain, screams of rage, breathless taunts
- End with a devastating final blow + victor's line delivered while standing over the defeated
- Total duration of all beats must sum to approximately {duration_seconds} seconds
- Each beat is 2-5 seconds (IMPACT beats should be 2-3s, ACTION 2-4s)
- DIALOGUE beats MUST be at least 3 seconds — give the voice room to breathe
- Aim for 20-30 beats total
- ACTION and IMPACT beats CAN have dialogue_text — this means the character speaks WHILE fighting
- Only use pure DIALOGUE beat_type for the opening face-off and final line

DIALOGUE QUALITY RULES (CRITICAL — this makes or breaks the scene):
- Write like a SCREENWRITER, not a video game. Think Dark Knight, Infinity War, Logan.
- Dialogue must show PERSONAL STAKES: why do they hate each other? What's at risk?
- Include PHYSICAL REACTIONS in the text for ElevenLabs to deliver with emotion:
  * Pain: "Aaaargh!" / "Gah—!" / a scream
  * Rage: text in CAPS = character is screaming
  * Breathless: "You think... you can stop me?" (ellipsis = gasping between words)
  * Quiet menace: short whispered threats before the explosion of violence
- Mix dialogue types across the fight:
  * Opening: cold, quiet threats (whispered menace)
  * Early fight: surprised reactions, "Hngh!", "That... actually hurt."
  * Mid fight: angry taunts through gritted teeth
  * Climax: SCREAMING, raw emotion, battle cries
  * End: breathless, exhausted, one final line
- NOT EVERY BEAT needs dialogue — some should be PURE SFX with no words (let the punches speak)
- Only 8-12 total dialogue lines in the whole fight (quality over quantity)
- NO generic lines like "You're strong" or "Is that all you've got?" — make every line SPECIFIC to these characters

DIALOGUE EXAMPLES — study these for TONE and QUALITY:

HERE IS THE EXACT EMOTIONAL ARC YOUR DIALOGUE MUST FOLLOW:

Beat 1-2 (OPENING): Cold, quiet, controlled threat. Spoken softly but deadly.
  Example: "You took everything from my world. Now I take everything from yours."
  Example: "I warned you. You didn't listen."

Beat 5-8 (EARLY FIGHT): Short reactions during combat — surprise, defiance
  Example: "That actually hurt!"
  Example: "COME ON!"
  Example: "Is THAT your thunder?!"

Beat 10-15 (MID FIGHT — ESCALATION): Raw rage, screaming, losing control
  Example: "YOU THINK LIGHTNING SCARES ME?!"
  Example: "ENOUGH!"
  Example: "I WILL BREAK YOU!"
  Example: "KNEEL!"

Beat 18-20 (CLIMAX + END): Exhausted, breathless, final statement of dominance
  Example: "Stay down."
  Example: "It's over."
  Example: "Go home."
  Example: "You never stood a chance."

CRITICAL RULES:
- Follow this emotional arc EXACTLY: cold → defiant → RAGING → quiet dominance
- Lines in the RAGING section should be ALL CAPS (character is screaming)
- Lines in the opening should be calm and quiet (controlled menace)
- Final line should be short, soft, and devastating
- Only 6-10 total dialogue lines in the whole fight — SILENCE IS POWERFUL
- Between dialogue lines, have 2-4 beats of PURE ACTION (no talking, just SFX)
- Every line must be something the audience would quote to their friends

BAD dialogue (NEVER write like this):
- "You represent a false peace built on fear." (essay, not dialogue)
- "I was invited. You are trespassing." (sounds like a lawyer)
- "Do you feel that? That is the weight of a world." (explaining kills the moment)
- Anything over 10 words
- Two sentences in one line (always split or cut one)

CHARACTER RULES:
- hero1 and hero2 are the two fighters
- Give them a REASON to fight (invent a personal conflict that fits the characters)
- Each character should have a distinct speaking style (one calm/controlled, one aggressive/emotional)
- hero1 style: SHORT, calm, matter-of-fact (like someone who knows they'll win)
- hero2 style: PASSIONATE, loud, emotional (like someone with something to prove)
- Alternate action between them — don't show the same character twice in a row
- Describe characters by APPEARANCE ONLY in video prompts (no names, no IP)
- ONLY include actual spoken words in dialogue_text — NO stage directions, NO "(breathing)", NO "*gasping*"
- Use at most ONE ellipsis (...) per line — multiple ellipses make TTS sound robotic
- Prefer dashes (—) over ellipses for mid-sentence interruptions
- NEVER write pure screams/grunts as dialogue ("AAARGH!", "Hngh!", "RAAAH!") — use SFX cue "battle_cry" or "pain_grunt" instead. TTS cannot do screams — they sound like a baby crying.
- dialogue_text should ONLY contain actual WORDS the character speaks

SOUND DESIGN RULES:
- Every ACTION beat must have at least 1 SFX cue
- Every IMPACT beat must have flash_frame=true and a heavy SFX (punch_heavy, explosion, etc.)
- TENSION beats use atmospheric SFX (thunder_rumble, wind, heartbeat)
- Beats with NO dialogue should have HEAVIER SFX (let sound design fill the space)
- Available SFX cues: {sfx_options}
- sfx_offset_ms: 0 for immediate, 500-1500 for delayed hits

VIDEO PROMPT RULES:
- NEVER use character names or franchise names — describe by appearance only
- Include: character appearance, action, camera angle, environment state
- Keep under 200 characters per prompt
- Include destruction level that escalates through the fight

MUSIC INTENSITY CURVE:
- Provide a float (0.0-1.0) per beat
- Start at 0.2-0.3 (tension), build to 0.5-0.7 (mid-fight), peak at 1.0 (climax), drop to 0.3 (resolution)

OUTPUT FORMAT — return ONLY valid JSON:
{{
  "title": "Superman vs Thor",
  "hero1_name": "Superman",
  "hero1_description": "Tall muscular male in blue suit with red cape, S symbol on chest, clean-shaven, black hair",
  "hero2_name": "Thor",
  "hero2_description": "Tall muscular male with long blonde hair, silver armor, red cape, wielding a glowing hammer",
  "setting": "Destroyed city rooftop at night, rain pouring, lightning in clouds",
  "beats": [
    {{
      "beat_type": "tension",
      "duration_seconds": 3.0,
      "video_prompt": "Wide shot of destroyed rooftop at night, rain pouring, two silhouettes facing each other 20 meters apart, lightning illuminates the sky",
      "camera_angle": "wide_shot",
      "flash_frame": false,
      "sfx_cues": ["thunder_rumble", "rain"],
      "sfx_offset_ms": 0,
      "character_id": null,
      "dialogue_text": null,
      "beat_index": 0
    }},
    {{
      "beat_type": "action",
      "duration_seconds": 3.0,
      "video_prompt": "Muscular male in blue suit charges forward with fist raised, smashing through debris, rain exploding around him",
      "camera_angle": "low_angle_tracking",
      "flash_frame": false,
      "sfx_cues": ["sonic_boom", "whoosh"],
      "sfx_offset_ms": 0,
      "character_id": "hero1",
      "dialogue_text": "This ends tonight!",
      "beat_index": 1
    }}
  ],
  "music_intensity_curve": [0.2, 0.5]
}}

IMPORTANT:
- beat_type must be one of: {beat_types}
- All beats must have beat_index set sequentially starting from 0
- total_duration_seconds = sum of all beat duration_seconds
- Dialogue lines: max 12 words, punchy action-movie style
- Generate 20-30 beats for a {duration_seconds}-second fight

Return ONLY the JSON. No markdown fences, no explanation.
"""


# ---------------------------------------------------------------------------
# CinematicScriptWriter
# ---------------------------------------------------------------------------


class CinematicScriptWriter:
    """Generates structured cinematic fight scripts via Claude.

    Args:
        claude_client: Async Claude API client.
    """

    def __init__(self, claude_client: Optional[ClaudeClient] = None) -> None:
        if claude_client is None:
            from pipeline.script_writer import build_claude_client  # noqa: PLC0415
            claude_client = build_claude_client()
        self._claude = claude_client

    async def generate(
        self,
        topic: TopicEntry,
        video_id: str,
        duration_seconds: int = _TARGET_DURATION_S,
    ) -> CinematicScript:
        """Generate a cinematic fight script for the given topic.

        Args:
            topic: TopicEntry with the fight matchup title.
            video_id: Pipeline video identifier.
            duration_seconds: Target total duration in seconds.

        Returns:
            A validated CinematicScript ready for audio/video generation.

        Raises:
            ValueError: If Claude's response cannot be parsed.
        """
        prompt = _build_cinematic_prompt(topic, duration_seconds)

        logger.info(
            "CinematicScriptWriter: generating %ds fight for '%s'",
            duration_seconds, topic.title,
        )

        raw = await self._claude.complete(prompt, max_tokens=_MAX_TOKENS)

        # Strip code fences if present
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```\s*$", "", raw)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("CinematicScriptWriter: JSON parse failed: %s", exc)
            raise ValueError(f"Claude returned invalid JSON: {exc}") from exc

        # Inject video_id and compute total duration
        data["video_id"] = video_id
        if "total_duration_seconds" not in data:
            data["total_duration_seconds"] = sum(
                b.get("duration_seconds", 2.0) for b in data.get("beats", [])
            )

        # Validate and return
        script = CinematicScript.model_validate(data)
        logger.info(
            "CinematicScriptWriter: generated %d beats, %.1fs total for video_id=%s",
            len(script.beats), script.total_duration_seconds, video_id,
        )
        return script
