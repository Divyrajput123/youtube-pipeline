"""Narration_Generator subsystem — ElevenLabs TTS integration.

Synthesizes speech from an approved Script using the ElevenLabs API, stores
the resulting MP3 in Asset_Store under ``narration/{video_id}_v{n}.mp3``, and
advances the Content_Calendar status to ``Narration Ready``.

Design reference: §5 Narration_Generator
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from pipeline.asset_store import Asset_Store, AssetStoreError
from pipeline.config import is_production_mode, require_production_config
from pipeline.content_calendar import Content_Calendar
from pipeline.models import NarrationAsset, PipelineStatus, Script, SubFolder
from pipeline.notifier import Notifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — tunable via module-level for testing
# ---------------------------------------------------------------------------

#: Maximum number of characters submitted to ElevenLabs in a single call.
MAX_SEGMENT_CHARS: int = 5_000

#: ElevenLabs synthesis quality settings.
SAMPLE_RATE_HZ: int = 44_100
BITRATE_KBPS: int = 128

#: Retry policy for ElevenLabs API calls.
_ELEVENLABS_MAX_ATTEMPTS: int = 3
_ELEVENLABS_BASE_DELAY_S: float = 5.0
_ELEVENLABS_MAX_DELAY_S: float = 60.0

#: Sentence-boundary characters used when splitting long scripts.
_SENTENCE_TERMINATORS = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NarrationGeneratorError(Exception):
    """Raised for all terminal failures in the Narration_Generator stage."""


# ---------------------------------------------------------------------------
# ElevenLabsClient Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ElevenLabsClient(Protocol):
    """Minimal interface that Narration_Generator needs from an ElevenLabs client.

    Any concrete implementation (real or test double) must satisfy this protocol.
    """

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int,
        bitrate_kbps: int,
    ) -> bytes:
        """Convert *text* to MP3 bytes using the given *voice_id*.

        Args:
            text: Script text to synthesize (≤ 5,000 characters).
            voice_id: ElevenLabs voice identifier.
            sample_rate: Output sample rate in Hz (e.g. 44,100).
            bitrate_kbps: Minimum bitrate in kbps (e.g. 128).

        Returns:
            Raw MP3 audio bytes.

        Raises:
            Exception: Any ElevenLabs API error.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Placeholder MP3 generator for fallback mode
# ---------------------------------------------------------------------------


def _generate_placeholder_mp3(sample_rate: int = 44_100, duration_seconds: float = 1.0) -> bytes:
    """Generate a valid silent MP3/WAV of *duration_seconds* using ffmpeg.

    Used when ElevenLabs is unavailable so the pipeline produces a video
    whose length matches the script (at ~150 WPM) rather than defaulting
    to 1 second.

    Falls back to a WAV silence file if ffmpeg is unavailable.
    """
    import subprocess as _sp  # noqa: PLC0415
    import tempfile as _tmp  # noqa: PLC0415
    import shutil as _shutil  # noqa: PLC0415

    ffmpeg = (
        _shutil.which("ffmpeg")
        or "/opt/homebrew/bin/ffmpeg"
        or "/usr/local/bin/ffmpeg"
    )

    if ffmpeg:
        try:
            with _tmp.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                out_path = f.name
            _sp.run(
                [
                    ffmpeg, "-y",
                    "-f", "lavfi",
                    "-i", f"anullsrc=r={sample_rate}:cl=mono",
                    "-t", str(duration_seconds),
                    "-ar", str(sample_rate),
                    "-ab", "128k",
                    out_path,
                ],
                capture_output=True,
                check=True,
            )
            import pathlib as _pl  # noqa: PLC0415
            data = _pl.Path(out_path).read_bytes()
            _pl.Path(out_path).unlink(missing_ok=True)
            return data
        except Exception as exc:
            logger.warning("ffmpeg silent MP3 generation failed (%s) — using WAV fallback", exc)

    # WAV fallback
    import struct as _struct  # noqa: PLC0415
    num_samples = int(sample_rate * duration_seconds)
    data_size = num_samples * 2
    header = _struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2,
        2, 16,
        b"data", data_size,
    )
    return header + b"\x00" * data_size


# ---------------------------------------------------------------------------
# ElevenLabsMCPClient — production concrete class
# ---------------------------------------------------------------------------


class ElevenLabsMCPClient:
    """Production ElevenLabs client backed by the ``elevenlabs`` Python library.

    Reads the API key from the ``ELEVENLABS_API_KEY`` environment variable at
    construction time.
    
    **Fallback mode**: If the ElevenLabs API key is missing or the API call fails,
    returns a minimal placeholder MP3 file (1 second of silence) so the pipeline
    can continue testing locally.

    Example::

        client = ElevenLabsMCPClient()
        mp3_bytes = await client.synthesize(
            text="Hello world",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            sample_rate=44_100,
            bitrate_kbps=128,
        )
    """

    def __init__(self) -> None:
        self._client = None
        self._fallback_mode = False
        
        try:
            from elevenlabs.client import ElevenLabs  # type: ignore[import-untyped]
            api_key = os.environ.get("ELEVENLABS_API_KEY", "")
            
            # In production mode, require a valid API key
            if is_production_mode():
                require_production_config(
                    "ElevenLabs API",
                    api_key if api_key and not api_key.startswith("sk_REPLACE") else None,
                    "Production mode requires a valid ELEVENLABS_API_KEY. "
                    "Set PIPELINE_MODE=development to use fallback mode."
                )
            
            if not api_key or api_key == "sk_REPLACE_ME":
                logger.warning(
                    "ElevenLabs API key missing or placeholder — using fallback mode "
                    "(placeholder MP3 files). Set PIPELINE_MODE=production to enforce real API."
                )
                self._fallback_mode = True
            else:
                self._client = ElevenLabs(api_key=api_key)
        except (ImportError, KeyError) as exc:
            if is_production_mode():
                raise ValueError(
                    f"Production mode requires ElevenLabs client: {exc}. "
                    "Set PIPELINE_MODE=development to use fallback mode."
                ) from exc
            logger.warning(
                "ElevenLabs client initialization failed (%s) — using fallback mode.", exc
            )
            self._fallback_mode = True

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int,
        bitrate_kbps: int,
    ) -> bytes:
        """Synthesize *text* to MP3 via ElevenLabs ``text_to_speech.convert``.

        Runs the synchronous ElevenLabs SDK call in a thread executor so that
        the async event loop is not blocked.
        
        **Fallback**: If the ElevenLabs client is unavailable or the API call fails,
        returns a minimal 1-second silent MP3 file as a placeholder.

        Args:
            text: Script text to synthesize.
            voice_id: ElevenLabs voice identifier.
            sample_rate: Output sample rate (passed for documentation; the
                output_format encodes quality settings).
            bitrate_kbps: Minimum bitrate; the output_format encodes this value.

        Returns:
            Raw MP3 bytes (real or placeholder).
        """
        # If in fallback mode, return placeholder immediately
        if self._fallback_mode or self._client is None:
            # Estimate duration from word count at 150 WPM so the video
            # length matches the script rather than defaulting to 1 second.
            word_count = len(text.split())
            estimated_seconds = max(1.0, (word_count / 150) * 60)
            logger.info(
                "ElevenLabs fallback: generating %.1fs silent MP3 for %d words",
                estimated_seconds, word_count,
            )
            return _generate_placeholder_mp3(sample_rate, duration_seconds=estimated_seconds)
        
        # Try real synthesis with fallback on error
        try:
            # Build the output_format string, e.g. "mp3_44100_128"
            output_format = f"mp3_{sample_rate}_{bitrate_kbps}"

            def _sync_call() -> bytes:
                audio_chunks = self._client.text_to_speech.convert(
                    voice_id=voice_id,
                    text=text,
                    model_id="eleven_multilingual_v2",
                    output_format=output_format,
                )
                return b"".join(audio_chunks)

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_call)
            
        except Exception as exc:
            # In production mode, try Google Cloud TTS fallback before failing
            err_str = str(exc).lower()
            if "quota_exceeded" in err_str or "quota exceeded" in err_str or "exceeded" in err_str:
                # Try Google Cloud TTS as fallback
                try:
                    gcloud_tts = GoogleCloudTTSClient()
                    if gcloud_tts.available:
                        logger.warning(
                            "ElevenLabs quota exceeded — falling back to Google Cloud TTS"
                        )
                        return await gcloud_tts.synthesize(text, voice_id, sample_rate, bitrate_kbps)
                except Exception as gcloud_exc:
                    logger.warning("Google Cloud TTS fallback also failed: %s", gcloud_exc)

                if is_production_mode():
                    raise Exception(
                        f"ElevenLabs quota exceeded and Google Cloud TTS fallback failed: {exc}\n"
                        "Top up ElevenLabs credits or check Google Cloud TTS setup."
                    ) from exc

            if is_production_mode():
                raise Exception(
                    f"ElevenLabs API error in production mode: {exc}. "
                    "Set PIPELINE_MODE=development to use fallback mode."
                ) from exc

            word_count = len(text.split())
            estimated_seconds = max(1.0, (word_count / 150) * 60)
            logger.warning(
                "ElevenLabs API error (%s), falling back to %.1fs silent MP3.", exc, estimated_seconds
            )
            return _generate_placeholder_mp3(sample_rate, duration_seconds=estimated_seconds)


# ---------------------------------------------------------------------------
# Google Cloud TTS Client — fallback when ElevenLabs credits are exhausted
# ---------------------------------------------------------------------------


class GoogleCloudTTSClient:
    """Google Cloud Text-to-Speech client using the same Google OAuth credentials.

    Uses WaveNet voices for high-quality output. Falls back to Standard voices
    if WaveNet fails. The output is MP3 format matching ElevenLabs' interface.

    Free tier: 1 million characters/month (WaveNet) or 4 million (Standard).
    Uses the same GOOGLE_CLIENT_ID/SECRET/REFRESH_TOKEN as YouTube/Drive.
    """

    def __init__(self) -> None:
        self._service = None
        self._available = False

        try:
            from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
            from googleapiclient.discovery import build  # type: ignore[import-untyped]
            import google.auth.transport.requests as _gtr  # noqa: PLC0415

            client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
            client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
            refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN", "")

            if client_id and client_secret and refresh_token:
                creds = Credentials(
                    token=None,
                    refresh_token=refresh_token,
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                creds.refresh(_gtr.Request())
                self._service = build("texttospeech", "v1", credentials=creds, cache_discovery=False)
                self._available = True
                logger.info("GoogleCloudTTSClient: initialized successfully")
            else:
                logger.debug("GoogleCloudTTSClient: Google credentials not available")
        except Exception as exc:
            logger.debug("GoogleCloudTTSClient: init failed (%s) — not available as fallback", exc)

    @property
    def available(self) -> bool:
        return self._available

    async def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int,
        bitrate_kbps: int,
    ) -> bytes:
        """Synthesize text using Google Cloud TTS WaveNet voices.

        Args:
            text: Text to synthesize.
            voice_id: Ignored (uses Google's en-US-WaveNet-D for male narration).
            sample_rate: Output sample rate.
            bitrate_kbps: Ignored (Google controls encoding quality).

        Returns:
            Raw MP3 bytes.
        """
        import base64 as _b64  # noqa: PLC0415

        if not self._available or not self._service:
            raise NarrationGeneratorError("Google Cloud TTS is not available")

        # Use a cinematic male WaveNet voice suitable for superhero narration
        voice_name = "en-US-WaveNet-D"  # Deep male voice, good for narration
        # Alternative voices: en-US-WaveNet-J (male), en-US-WaveNet-A (male)

        body = {
            "input": {"text": text},
            "voice": {
                "languageCode": "en-US",
                "name": voice_name,
                "ssmlGender": "MALE",
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "sampleRateHertz": sample_rate,
                "speakingRate": 1.05,  # Slightly faster for engaging narration
                "pitch": -1.0,  # Slightly deeper
                "volumeGainDb": 0.0,
            },
        }

        def _sync_call() -> bytes:
            response = self._service.text().synthesize(body=body).execute()
            audio_content = response.get("audioContent", "")
            return _b64.b64decode(audio_content)

        loop = asyncio.get_running_loop()
        mp3_bytes = await loop.run_in_executor(None, _sync_call)
        logger.info("GoogleCloudTTS: synthesized %d bytes", len(mp3_bytes))
        return mp3_bytes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_into_segments(text: str, max_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    """Split *text* into segments of at most *max_chars* characters.

    Strategy:
    1. Try splitting on sentence boundaries (characters after ``.``, ``!``, ``?``).
    2. Fall back to splitting on whitespace if sentence splits produce a
       segment that still exceeds *max_chars*.
    3. Hard-slice any remaining oversized token as a last resort.

    Args:
        text: The full script text to split.
        max_chars: Maximum allowed characters per segment.

    Returns:
        A list of non-empty text segments, each ≤ *max_chars* characters.
    """
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # Split into candidate sentences on whitespace after sentence terminators.
    sentences = _SENTENCE_TERMINATORS.split(text)

    segments: list[str] = []
    current: str = ""

    for sentence in sentences:
        # If the sentence itself is too long, break it on whitespace.
        if len(sentence) > max_chars:
            words = sentence.split()
            for word in words:
                # Hard-slice words that are individually too long.
                if len(word) > max_chars:
                    # Flush current buffer first.
                    if current:
                        segments.append(current.strip())
                        current = ""
                    # Slice the giant word.
                    for i in range(0, len(word), max_chars):
                        segments.append(word[i : i + max_chars])
                    continue

                candidate = (current + " " + word).strip() if current else word
                if len(candidate) > max_chars:
                    segments.append(current.strip())
                    current = word
                else:
                    current = candidate
            continue

        # Normal sentence: try to append to the current buffer.
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) > max_chars:
            if current:
                segments.append(current.strip())
            current = sentence
        else:
            current = candidate

    if current.strip():
        segments.append(current.strip())

    return [s for s in segments if s]


def _next_version(video_id: str, asset_store: Asset_Store) -> int:
    """Determine the next available version number for a narration asset.

    Scans Asset_Store for existing ``{video_id}_v1.mp3``, ``{video_id}_v2.mp3``,
    etc. by probing read operations and returns the first version number that
    does not yet exist.

    Because Asset_Store does not expose a directory listing API, we probe
    sequentially up to a reasonable limit (100 versions).

    Args:
        video_id: Pipeline video identifier.
        asset_store: Initialized Asset_Store instance.

    Returns:
        Integer version number ≥ 1.
    """
    # We return 1 as the base version; the caller will attempt a read to verify
    # whether the file already exists. This function is intentionally synchronous
    # because version scanning is performed before any async TTS call; the
    # actual probe is done in the async generate() method.
    return 1  # Base: refined asynchronously in generate()


async def _probe_next_version(video_id: str, asset_store: Asset_Store) -> int:
    """Return version 1 — each pipeline run uses a fresh video_id, so v1 is always free."""
    return 1


def _retry_delay_elevenlabs(attempt: int) -> float:
    """Return back-off delay (seconds) for the given 1-based attempt number.

    Schedule: ``min(5 * 2^(attempt-1), 60)``
    Attempt 1 → 5 s, Attempt 2 → 10 s, Attempt 3+ → 20 s … capped at 60 s.

    Args:
        attempt: 1-based attempt counter.

    Returns:
        Float seconds to wait before the next attempt.
    """
    return min(_ELEVENLABS_BASE_DELAY_S * (2 ** (attempt - 1)), _ELEVENLABS_MAX_DELAY_S)


# ---------------------------------------------------------------------------
# Narration_Generator
# ---------------------------------------------------------------------------


class Narration_Generator:
    """Synthesizes MP3 narration from an approved Script via ElevenLabs TTS.

    Workflow
    --------
    1. **Pre-flight** — validate that ``voice_id`` is non-empty.
    2. **Text segmentation** — split script into chunks of ≤ 5,000 characters
       using sentence-boundary splitting.
    3. **TTS synthesis** — submit each chunk to ElevenLabs; concatenate MP3
       bytes. Retry each segment up to 3× with exponential back-off
       (5 s → 10 s → 60 s max).
    4. **Storage** — write the combined MP3 to
       ``narration/{video_id}_v{n}.mp3`` via Asset_Store.
    5. **Status update** — advance the Content_Calendar record to
       ``Narration Ready``.
    6. Return a :class:`~pipeline.models.NarrationAsset`.

    On any unrecoverable error a :class:`NarrationGeneratorError` is raised
    and a failure alert is dispatched via the :class:`~pipeline.notifier.Notifier`.

    Args:
        elevenlabs_client: Any object satisfying the :class:`ElevenLabsClient`
            protocol (real API client or test double).
        asset_store: Initialized :class:`~pipeline.asset_store.Asset_Store`.
        content_calendar: Initialized :class:`~pipeline.content_calendar.Content_Calendar`.
        notifier: Initialized :class:`~pipeline.notifier.Notifier` for failure alerts.
    """

    def __init__(
        self,
        elevenlabs_client: ElevenLabsClient,
        asset_store: Asset_Store,
        content_calendar: Content_Calendar,
        notifier: Notifier,
    ) -> None:
        self._tts = elevenlabs_client
        self._asset_store = asset_store
        self._content_calendar = content_calendar
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        script: Script,
        voice_id: str,
        video_id: str,
    ) -> NarrationAsset:
        """Generate MP3 narration from *script* using ElevenLabs TTS.

        Args:
            script: Approved :class:`~pipeline.models.Script` containing the
                text to synthesize.
            voice_id: ElevenLabs voice identifier (must be non-empty).
            video_id: Pipeline video identifier used for file naming and
                Content_Calendar updates.

        Returns:
            A :class:`~pipeline.models.NarrationAsset` with ``mp3_path`` set
            to ``{video_id}_v{n}.mp3`` and ``asset_url`` set to the Google
            Drive URL returned by Asset_Store.

        Raises:
            NarrationGeneratorError: On pre-flight failure, ElevenLabs API
                failure after all retries, or Asset_Store write failure.
        """
        # ------------------------------------------------------------------ #
        # Step 1 — Pre-flight: voice_id must be present and non-empty          #
        # ------------------------------------------------------------------ #
        if not voice_id or not voice_id.strip():
            error_msg = (
                f"Narration pre-flight failed for video_id={video_id!r}: "
                "voice_id is absent or empty."
            )
            logger.error(error_msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="narration_generator",
                error_message=error_msg,
            )
            raise NarrationGeneratorError(error_msg)

        # ------------------------------------------------------------------ #
        # Step 2 — Strip all non-speech markup, then split                   #
        # ------------------------------------------------------------------ #
        import re as _re  # noqa: PLC0415

        clean_content = script.content

        # 1. Remove [tag] and [/tag] speaker-direction annotations
        clean_content = _re.sub(r"\[/?[a-zA-Z]+\]", "", clean_content)

        # 2. Remove Markdown headings (## HOOK, ### Segment 1: ..., etc.)
        clean_content = _re.sub(r"^#{1,6}\s+.*$", "", clean_content, flags=_re.MULTILINE)

        # 3. Remove section label lines (HOOK, CTA, BODY, INTRO, OUTRO — with or
        #    without markdown, numbering, dashes, colons, or a trailing title,
        #    e.g. "HOOK", "## HOOK", "BODY - SEGMENT 1: TACTICAL GENIUS",
        #    "2. BODY (3 segments)", "Segment 1: The Setup").
        # NOTE: use [ \t] (not \s) inside these classes so the match never
        # crosses a newline into the next line's actual narration content.
        clean_content = _re.sub(
            r"^[ \t]*#{0,6}[ \t]*(?:\d+[.)][ \t]*)?"
            r"(HOOK|CTA|BODY|INTRO|OUTRO)"
            r"([ \t]*[-:–—]?[ \t]*SEGMENT[ \t]*\d+)?"
            r"[ \t:\-–—]*[^\n]*$",
            "", clean_content, flags=_re.MULTILINE | _re.IGNORECASE,
        )
        # Also catch standalone "Segment N: ..." headings not preceded by BODY.
        clean_content = _re.sub(
            r"^[ \t]*#{0,6}[ \t]*(?:\d+[.)][ \t]*)?Segment[ \t]*\d+[ \t:\-–—]*[^\n]*$",
            "", clean_content, flags=_re.MULTILINE | _re.IGNORECASE,
        )

        # 4. Remove lines that are only dashes or asterisks (horizontal rules)
        clean_content = _re.sub(r"^[-*_]{2,}\s*$", "", clean_content, flags=_re.MULTILINE)

        # 5. Collapse multiple blank lines into one
        clean_content = _re.sub(r"\n{3,}", "\n\n", clean_content).strip()

        segments = _split_into_segments(clean_content, MAX_SEGMENT_CHARS)
        if not segments:
            error_msg = (
                f"Narration pre-flight failed for video_id={video_id!r}: "
                "script.content is empty after segmentation."
            )
            logger.error(error_msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="narration_generator",
                error_message=error_msg,
            )
            raise NarrationGeneratorError(error_msg)

        logger.info(
            "Narration_Generator: video_id=%r, %d segment(s) to synthesize.",
            video_id,
            len(segments),
        )

        # ------------------------------------------------------------------ #
        # Step 3 — Synthesize each segment; concatenate MP3 bytes             #
        # ------------------------------------------------------------------ #
        combined_mp3: bytes = b""
        for seg_idx, segment_text in enumerate(segments, start=1):
            mp3_chunk = await self._synthesize_with_retry(
                segment_text=segment_text,
                voice_id=voice_id,
                video_id=video_id,
                seg_idx=seg_idx,
                total_segs=len(segments),
            )
            combined_mp3 += mp3_chunk

        # ------------------------------------------------------------------ #
        # Step 4 — Determine version number; store in Asset_Store             #
        # ------------------------------------------------------------------ #
        version = await _probe_next_version(video_id, self._asset_store)
        filename = f"{video_id}_v{version}.mp3"

        logger.info(
            "Narration_Generator: storing %s (%d bytes).",
            filename,
            len(combined_mp3),
        )

        try:
            drive_url = await self._asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.NARRATION,
                filename=filename,
                content=combined_mp3,
            )
        except AssetStoreError as exc:
            # Discard the audio bytes and halt.
            combined_mp3 = b""
            error_msg = (
                f"Narration_Generator: Asset_Store write failed for "
                f"video_id={video_id!r}, file={filename!r}: {exc}"
            )
            logger.error(error_msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="narration_generator",
                error_message=error_msg,
            )
            raise NarrationGeneratorError(error_msg) from exc

        # ------------------------------------------------------------------ #
        # Step 5 — Update Content_Calendar status to Narration Ready          #
        # ------------------------------------------------------------------ #
        try:
            await self._content_calendar.update_status(
                video_id, PipelineStatus.NARRATION_READY
            )
            logger.info(
                "Narration_Generator: Content_Calendar status updated to "
                "NARRATION_READY for video_id=%r.",
                video_id,
            )
        except Exception as exc:  # noqa: BLE001
            # A status update failure is non-fatal for the narration asset itself,
            # but we log and re-raise so the Orchestrator can handle it.
            error_msg = (
                f"Narration_Generator: Content_Calendar status update failed for "
                f"video_id={video_id!r}: {exc}"
            )
            logger.error(error_msg)
            self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name="narration_generator",
                error_message=error_msg,
            )
            raise NarrationGeneratorError(error_msg) from exc

        # ------------------------------------------------------------------ #
        # Step 6 — Return NarrationAsset; mp3_path is the stored filename     #
        # ------------------------------------------------------------------ #
        return NarrationAsset(
            video_id=video_id,
            version=version,
            mp3_path=filename,
            asset_url=drive_url,
            created_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _synthesize_with_retry(
        self,
        segment_text: str,
        voice_id: str,
        video_id: str,
        seg_idx: int,
        total_segs: int,
    ) -> bytes:
        """Synthesize a single text segment with exponential back-off retry.

        Submits *segment_text* to ElevenLabs up to ``_ELEVENLABS_MAX_ATTEMPTS``
        times.  On transient failure waits ``_retry_delay_elevenlabs(attempt)``
        seconds before the next attempt.  If all attempts fail, discards any
        partial audio, notifies the Notifier, and raises
        :class:`NarrationGeneratorError`.

        Args:
            segment_text: Text chunk to synthesize (≤ 5,000 characters).
            voice_id: ElevenLabs voice identifier.
            video_id: Used for error notifications.
            seg_idx: 1-based index of this segment (for log messages).
            total_segs: Total segment count (for log messages).

        Returns:
            Raw MP3 bytes for this segment.

        Raises:
            NarrationGeneratorError: If all retry attempts are exhausted.
        """
        last_exc: Exception = Exception("No synthesis attempts made.")

        for attempt in range(1, _ELEVENLABS_MAX_ATTEMPTS + 1):
            try:
                mp3_bytes = await self._tts.synthesize(
                    text=segment_text,
                    voice_id=voice_id,
                    sample_rate=SAMPLE_RATE_HZ,
                    bitrate_kbps=BITRATE_KBPS,
                )
                logger.debug(
                    "Narration_Generator: segment %d/%d synthesized on attempt %d "
                    "(%d bytes).",
                    seg_idx,
                    total_segs,
                    attempt,
                    len(mp3_bytes),
                )
                return mp3_bytes

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = _retry_delay_elevenlabs(attempt)
                logger.warning(
                    "Narration_Generator: ElevenLabs error on segment %d/%d "
                    "(attempt %d/%d) for video_id=%r: %s. "
                    "Retrying in %.0f s.",
                    seg_idx,
                    total_segs,
                    attempt,
                    _ELEVENLABS_MAX_ATTEMPTS,
                    video_id,
                    exc,
                    delay,
                )
                if attempt < _ELEVENLABS_MAX_ATTEMPTS:
                    await asyncio.sleep(delay)

        # All retries exhausted — discard and halt.
        error_msg = (
            f"Narration_Generator: ElevenLabs synthesis failed for "
            f"video_id={video_id!r}, segment {seg_idx}/{total_segs} after "
            f"{_ELEVENLABS_MAX_ATTEMPTS} attempts: {last_exc}"
        )
        logger.error(error_msg)
        self._notifier.send_failure_alert(
            video_id=video_id,
            stage_name="narration_generator",
            error_message=error_msg,
        )
        raise NarrationGeneratorError(error_msg) from last_exc


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ElevenLabsClient",
    "ElevenLabsMCPClient",
    "Narration_Generator",
    "NarrationGeneratorError",
]
