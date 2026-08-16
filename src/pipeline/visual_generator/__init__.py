"""Visual_Generator subsystem — Viewmax MCP + video compilation.

Generates per-segment video clips via the Viewmax MCP, compiles them into a
final MP4 (synchronized with narration audio), produces a thumbnail JPEG, and
writes both artefacts to the Asset_Store.

Key design points
-----------------
* **Scene prompt derivation** — the script is split on Markdown headings
  (``## `` or ``### ``).  If no headings are found the content is divided into
  equal thirds labelled hook / body / cta.
* **Per-clip retry** — each Viewmax clip is attempted up to 3 times with a
  random delay sampled from [5, 30] seconds via ``random.uniform``.
* **Static fallback** — if all retries for a clip are exhausted a 1920×1080
  JPEG is synthesised with Pillow (black background, white centred text) and
  fed to ffmpeg as a looped still image.
* **Video compilation** — ``subprocess`` calls ``ffmpeg`` to concatenate clips
  and mux with the narration MP3.  Clips originating from still-image
  fallbacks use ``-loop 1 -t 3`` to produce a 3-second clip.
* **Thumbnail** — Pillow renders a 1280×720 JPEG using the first dominant
  colour from the StyleProfile, white text overlay, and a contrasting
  placeholder rectangle.  The file is compressed to stay < 2 MB.

Documented constraint
---------------------
Generated frames are **not** pixel-for-pixel reproductions of reference
channel frames.  This is guaranteed at the Viewmax MCP prompt level — scene
prompts are unique per video — but no runtime pixel-comparison check is
performed (doing so would require downloading the reference channel frames at
generation time, which is outside this subsystem's scope).
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import random
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Protocol, runtime_checkable

from PIL import Image, ImageDraw, ImageFont

from pipeline.asset_store import Asset_Store
from pipeline.config import is_production_mode
from pipeline.content_calendar import Content_Calendar
from pipeline.models import (
    NarrationAsset,
    PipelineStatus,
    Script,
    StyleProfile,
    SubFolder,
    VisualAsset,
)
from pipeline.notifier import Notifier
from pipeline.script_writer import build_claude_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CLIP_RETRY_ATTEMPTS: int = 3
_FALLBACK_WIDTH: int = 1920
_FALLBACK_HEIGHT: int = 1080
_THUMBNAIL_WIDTH: int = 1280
_THUMBNAIL_HEIGHT: int = 720
_THUMBNAIL_MAX_BYTES: int = 2 * 1024 * 1024  # 2 MB
_FFMPEG_FRAMERATE: int = 24
_FFMPEG_RESOLUTION: str = "1920x1080"
_FALLBACK_CLIP_DURATION_S: int = 3  # seconds for a still-image fallback clip
# NOTE: Do NOT add a _CLIP_DEFAULT_DURATION_S constant here.
# Clip duration is always read from ViewmaxMCPClient.CLIP_DURATION so there
# is exactly one source of truth per provider (kling=5, minimax_h3=15).
# Using a module-level constant causes silent duration mismatches when the
# provider is switched — the constant gets stale while CLIP_DURATION is updated.

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class VisualGeneratorError(Exception):
    """Raised when the Visual_Generator cannot complete an operation."""


class KlingContentModerationError(Exception):
    """Raised when Kling rejects a prompt due to content moderation (risk control system).

    Signals that the prompt itself — not a transient network issue — caused
    the failure, so the caller should rephrase rather than retry as-is.
    """


# ---------------------------------------------------------------------------
# ViewmaxClient Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ViewmaxClient(Protocol):
    """Minimal async interface to the Viewmax video-generation service.

    Implementors wrap the Viewmax MCP server.  The stub
    :class:`ViewmaxMCPClient` raises ``NotImplementedError`` so that tests
    can substitute a mock without requiring a live Viewmax connection.
    """

    async def generate_clip(self, prompt: str, duration_seconds: int, seed: int = -1) -> bytes:
        """Generate a video clip for *prompt* of *duration_seconds* length.

        Args:
            prompt: Natural-language description of the desired visual scene.
            duration_seconds: Requested clip length in seconds.
            seed: Random seed for reproducibility. -1 = random.

        Returns:
            Raw MP4 bytes for the generated clip.

        Raises:
            Exception: Any Viewmax API error (retried by Visual_Generator).
        """
        ...


# ---------------------------------------------------------------------------
# Placeholder clip generator for fallback mode
# ---------------------------------------------------------------------------


def _generate_placeholder_clip_jpeg(prompt: str) -> bytes:
    """Generate a placeholder 1920×1080 JPEG with prompt text for testing.
    
    This allows the Visual_Generator to continue without a real Viewmax MCP server.
    The JPEG will be looped by ffmpeg to create a video clip.
    
    Args:
        prompt: Scene prompt text to display on the placeholder frame.
        
    Returns:
        Raw JPEG bytes for a 1920×1080 frame with the prompt text.
    """
    import hashlib
    
    # Pick a color based on the prompt hash for variety
    prompt_hash = int(hashlib.md5(prompt.encode()).hexdigest()[:6], 16)
    r = (prompt_hash >> 16) & 0xFF
    g = (prompt_hash >> 8) & 0xFF
    b = prompt_hash & 0xFF
    bg_color = (r, g, b)
    
    img = Image.new("RGB", (_FALLBACK_WIDTH, _FALLBACK_HEIGHT), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", size=48)
        font_prompt = ImageFont.truetype("DejaVuSans.ttf", size=32)
    except (IOError, OSError):
        font_title = ImageFont.load_default()
        font_prompt = ImageFont.load_default()

    # Draw "PLACEHOLDER" at the top
    title = "PLACEHOLDER CLIP"
    try:
        bbox = draw.textbbox((0, 0), title, font=font_title)
        text_w = bbox[2] - bbox[0]
    except AttributeError:
        text_w, _ = draw.textsize(title, font=font_title)  # type: ignore[attr-defined]
    
    x_title = (_FALLBACK_WIDTH - text_w) // 2
    y_title = 100
    draw.text((x_title, y_title), title, fill=(255, 255, 255), font=font_title)
    
    # Draw the prompt text wrapped in the middle
    import textwrap
    wrapped_prompt = textwrap.fill(prompt, width=60)
    lines = wrapped_prompt.split('\n')
    
    y_offset = 400
    for line in lines[:6]:  # Limit to 6 lines
        try:
            bbox = draw.textbbox((0, 0), line, font=font_prompt)
            line_w = bbox[2] - bbox[0]
        except AttributeError:
            line_w, _ = draw.textsize(line, font=font_prompt)  # type: ignore[attr-defined]
        
        x_line = (_FALLBACK_WIDTH - line_w) // 2
        draw.text((x_line, y_offset), line, fill=(255, 255, 255), font=font_prompt)
        y_offset += 50

    # Convert to JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ViewmaxMCPClient — production stub
# ---------------------------------------------------------------------------


class ViewmaxMCPClient:
    """Video clip provider selected explicitly by ``visual_video_provider``.

    ``"kling"`` uses Kling's text-to-video API; ``"runpod"`` uses the
    configured LTX-Video RunPod server; ``"minimax_h3"`` uses the MiniMax H3
    model on a RunPod serverless endpoint (15s clips, 768p, native audio).
    Both credentials may remain configured, but this client calls only the
    selected provider.
    """

    CLIP_DURATION: int = 5
    CLIP_WIDTH: int = 1280
    CLIP_HEIGHT: int = 720

    def __init__(self, provider: Literal["kling", "minimax_h3"] = "minimax_h3") -> None:
        if provider not in {"kling", "minimax_h3"}:
            raise ValueError(
                "visual_video_provider must be 'kling' or 'minimax_h3', "
                f"got {provider!r}."
            )

        self._provider = provider
        self._api_key = os.environ.get("KLING_API_KEY", "").strip()
        self._model = os.environ.get("KLING_MODEL", "kling-v1-6")
        # MiniMax H3 endpoint (ComfyUI Pod)
        self._h3_endpoint_id = os.environ.get("RUNPOD_H3_ENDPOINT_ID", "").strip()
        self._h3_api_key = os.environ.get("RUNPOD_H3_API_KEY", os.environ.get("RUNPOD_API_KEY", "")).strip()

        if provider == "minimax_h3":
            # MiniMax H3: 15s clips at 768p with character consistency (max supported)
            self.CLIP_DURATION = 15
            self.CLIP_WIDTH = 1366  # 768p 16:9
            self.CLIP_HEIGHT = 768
            self._h3_pod_id = os.environ.get("RUNPOD_H3_POD_ID", "").strip()
            if self._h3_pod_id:
                logger.info(
                    "ViewmaxMCPClient: selected MiniMax H3 ComfyUI pod=%s (10s clips, 768p)",
                    self._h3_pod_id,
                )
            elif self._h3_endpoint_id and self._h3_api_key:
                logger.info(
                    "ViewmaxMCPClient: selected MiniMax H3 endpoint=%s (10s clips, 768p)",
                    self._h3_endpoint_id,
                )
            elif is_production_mode():
                raise ValueError(
                    "visual_video_provider='minimax_h3' requires RUNPOD_H3_POD_ID or "
                    "RUNPOD_H3_ENDPOINT_ID in production mode."
                )
            else:
                logger.warning(
                    "ViewmaxMCPClient: MiniMax H3 selected but not configured — "
                    "using placeholder clips."
                )
        elif provider == "kling":
            if self._api_key:
                logger.info("ViewmaxMCPClient: selected Kling AI API (model=%s)", self._model)
            elif is_production_mode():
                raise ValueError(
                    "visual_video_provider='kling' requires KLING_API_KEY in production mode."
                )
            else:
                logger.warning(
                    "ViewmaxMCPClient: Kling selected but KLING_API_KEY is not configured — "
                    "using placeholder clips."
                )

    async def generate_clip(self, prompt: str, duration_seconds: int, seed: int = -1) -> bytes:
        """Generate a clip with only the explicitly selected provider."""
        if self._provider == "kling":
            if not self._api_key:
                return _generate_placeholder_clip_jpeg(prompt)
            try:
                return await self._call_kling(prompt)
            except KlingContentModerationError:
                raise  # let _generate_single_clip handle rephrasing and retry
            except Exception as exc:
                exc_str = str(exc)
                # Billing failures must reach the caller for notification/retry handling.
                if (
                    "402" in exc_str
                    or "401" in exc_str
                    or "payment" in exc_str.lower()
                    or "quota" in exc_str.lower()
                ):
                    raise
                logger.warning(
                    "ViewmaxMCPClient: Kling failed ('%s'): %s — using placeholder.",
                    prompt[:60],
                    exc,
                )
                return _generate_placeholder_clip_jpeg(prompt)

        if self._provider == "minimax_h3":
            if not self._h3_pod_id and not (self._h3_endpoint_id and self._h3_api_key):
                return _generate_placeholder_clip_jpeg(prompt)
            try:
                # Pick up any first_frame set by _generate_clips_cinematic for chaining
                _first_frame = getattr(self, "_next_first_frame", None)
                self._next_first_frame = None  # consume it so it doesn't leak to next call
                return await self._call_minimax_h3(
                    prompt,
                    duration_seconds=duration_seconds,
                    seed=seed,
                    first_frame_filename=_first_frame,
                )
            except Exception as exc:
                logger.warning(
                    "ViewmaxMCPClient: MiniMax H3 failed ('%s'): %s — using placeholder.",
                    prompt[:60],
                    exc,
                )
                return _generate_placeholder_clip_jpeg(prompt)

        # Should not reach here — all valid providers handled above
        return _generate_placeholder_clip_jpeg(prompt)

    async def _call_kling(self, prompt: str) -> bytes:
        """Generate video via Kling AI API and return MP4 bytes."""
        import httpx  # noqa: PLC0415
        import time  # noqa: PLC0415

        clean_prompt = self._clean_prompt(prompt)
        logger.info("ViewmaxMCPClient: Kling generating '%s'", clean_prompt[:80])

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Submit generation task
        submit_payload = {
            "model_name": self._model,
            "prompt": clean_prompt,
            "negative_prompt": "static, motionless, standing still, frozen, slow, boring, no movement, talking head, static background, low energy, lifeless, watermark, text overlay, subtitle",
            "cfg_scale": 0.5,
            "mode": "std",   # std works for both kling-v1-6 and kling-v2-master
            "duration": str(self.CLIP_DURATION),
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.klingai.com/v1/videos/text2video",
                headers=headers,
                json=submit_payload,
            )
            resp.raise_for_status()
            data = resp.json()

        task_id = data["data"]["task_id"]
        logger.info("ViewmaxMCPClient: Kling task submitted %s", task_id)

        # Poll until done (max 10 minutes)
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(60):
                await asyncio.sleep(10)
                try:
                    poll = await client.get(
                        f"https://api.klingai.com/v1/videos/text2video/{task_id}",
                        headers=headers,
                    )
                    poll.raise_for_status()
                    result = poll.json()
                    status = result["data"]["task_status"]
                    logger.info("ViewmaxMCPClient: Kling task %s status=%s (attempt %d)", task_id, status, attempt + 1)

                    if status == "succeed":
                        video_url = result["data"]["task_result"]["videos"][0]["url"]
                        logger.info("ViewmaxMCPClient: downloading from %s", video_url[:80])
                        video_resp = await client.get(video_url, timeout=60.0)
                        video_resp.raise_for_status()
                        logger.info("ViewmaxMCPClient: received %d bytes of MP4", len(video_resp.content))
                        return video_resp.content
                    elif status == "failed":
                        msg = result["data"].get("task_status_msg", "")
                        if "risk control" in msg.lower():
                            raise KlingContentModerationError(
                                f"Kling task {task_id} failed: {msg}"
                            )
                        raise RuntimeError(f"Kling task {task_id} failed: {msg}")
                except (KlingContentModerationError, RuntimeError):
                    # The task has reached a terminal failure state.  Do not
                    # keep polling the same task: callers can rephrase a
                    # moderated prompt or handle the provider failure.
                    raise
                except Exception as poll_exc:
                    logger.warning("ViewmaxMCPClient: poll attempt %d failed (%s) — retrying", attempt + 1, poll_exc)
                    continue

        raise RuntimeError(f"Kling task {task_id} timed out after 10 minutes")

    async def _extract_last_frame(self, mp4_bytes: bytes) -> Optional[bytes]:
        """Extract the last frame of an MP4 clip as JPEG bytes using ffmpeg.

        Uses the second-to-last second rather than the very last frame to avoid
        motion blur or partial frames that could confuse H3's first_frame anchor.

        Args:
            mp4_bytes: Raw MP4 bytes of the generated clip.

        Returns:
            JPEG bytes of the extracted frame, or None if extraction fails.
        """
        import shutil as _sh  # noqa: PLC0415
        import subprocess as _sp  # noqa: PLC0415
        import tempfile as _tf  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415

        ffmpeg_bin = _sh.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg" or "ffmpeg"
        ffprobe_bin = _sh.which("ffprobe") or "/opt/homebrew/bin/ffprobe" or "ffprobe"

        try:
            with _tf.TemporaryDirectory() as tmpdir:
                tmp = _Path(tmpdir)
                clip_path = tmp / "clip.mp4"
                frame_path = tmp / "frame.jpg"
                clip_path.write_bytes(mp4_bytes)

                # Get clip duration
                probe = _sp.run(
                    [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
                    capture_output=True, timeout=15,
                )
                duration = float(probe.stdout.strip()) if probe.returncode == 0 else 15.0

                # Extract frame at 80% duration — well into the action but before final blur
                seek_time = max(0.0, duration * 0.80)

                result = _sp.run(
                    [ffmpeg_bin, "-y",
                     "-ss", f"{seek_time:.3f}",
                     "-i", str(clip_path),
                     "-frames:v", "1",
                     "-q:v", "2",
                     str(frame_path)],
                    capture_output=True, timeout=30,
                )

                if result.returncode != 0 or not frame_path.exists():
                    logger.warning(
                        "ViewmaxMCPClient: last-frame extraction failed: %s",
                        result.stderr[-200:].decode("utf-8", errors="replace"),
                    )
                    return None

                frame_bytes = frame_path.read_bytes()
                logger.info(
                    "ViewmaxMCPClient: extracted last frame at %.1fs (%d bytes JPEG)",
                    seek_time, len(frame_bytes),
                )
                return frame_bytes

        except Exception as exc:
            logger.warning("ViewmaxMCPClient: last-frame extraction error: %s", exc)
            return None

    async def _upload_frame_to_comfyui(
        self,
        frame_jpeg_bytes: bytes,
        comfyui_url: str,
        filename: str = "last_frame.jpg",
    ) -> Optional[str]:
        """Upload a JPEG frame to ComfyUI's /upload/image endpoint.

        ComfyUI stores uploaded images in its input folder and returns the
        actual filename (may be deduplicated). That filename is what the
        workflow JSON must reference.

        Args:
            frame_jpeg_bytes: Raw JPEG bytes to upload.
            comfyui_url: Base ComfyUI URL.
            filename: Suggested filename (ComfyUI may rename it).

        Returns:
            The filename as stored by ComfyUI (use in workflow JSON), or None on failure.
        """
        import httpx  # noqa: PLC0415

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{comfyui_url}/upload/image",
                    files={"image": (filename, frame_jpeg_bytes, "image/jpeg")},
                    data={"type": "input", "overwrite": "true"},
                )
                if resp.status_code != 200:
                    logger.warning(
                        "ViewmaxMCPClient: frame upload failed (HTTP %d): %s",
                        resp.status_code, resp.text[:200],
                    )
                    return None

                data = resp.json()
                stored_name = data.get("name", filename)
                subfolder = data.get("subfolder", "")
                logger.info(
                    "ViewmaxMCPClient: uploaded last frame as '%s' (subfolder='%s')",
                    stored_name, subfolder,
                )
                return stored_name

        except Exception as exc:
            logger.warning("ViewmaxMCPClient: frame upload error: %s", exc)
            return None

    async def _call_minimax_h3(self, prompt: str, duration_seconds: int = 10, seed: int = -1, first_frame_filename: Optional[str] = None) -> bytes:
        """Generate video via MiniMax H3 on a ComfyUI Pod.

        Submits the exact ComfyUI workflow (API format) to the ComfyUI HTTP API,
        waits for completion, and downloads the generated MP4.

        Args:
            prompt: Scene description for video generation.
            duration_seconds: Clip duration (5-15 seconds, default 10).
            seed: Random seed (-1 for random).
            first_frame_filename: Optional filename of a JPEG previously uploaded
                to ComfyUI's input folder via _upload_frame_to_comfyui. When set,
                the workflow uses MiniMaxH3ImageToVideo's first_frame input so the
                generated clip visually continues from that frame (last-frame chaining).
                When None, T2V mode is used (no visual anchor).

        Returns:
            MP4 bytes of the generated video clip.
        """
        import json as _json  # noqa: PLC0415
        import uuid as _uuid  # noqa: PLC0415
        import httpx  # noqa: PLC0415
        import random as _random  # noqa: PLC0415

        # Pass is_cinematic=True so _clean_prompt uses the dialogue-safe path
        # regardless of whether the T2VA brief header is present (e.g. parse fallback).
        clean_prompt = self._clean_prompt(prompt, is_cinematic=self._provider == "minimax_h3")
        duration = max(5, min(15, duration_seconds))

        if seed < 0:
            seed = _random.randint(0, 2**53 - 1)

        logger.info(
            "ViewmaxMCPClient: MiniMax H3 (ComfyUI) generating '%s' duration=%ds seed=%d%s",
            clean_prompt[:80], duration, seed,
            f" [chaining from '{first_frame_filename}']" if first_frame_filename else " [T2V mode]",
        )

        # Build ComfyUI Pod URL from pod ID
        pod_id = os.environ.get("RUNPOD_H3_POD_ID", "").strip()
        if not pod_id:
            raise RuntimeError(
                "RUNPOD_H3_POD_ID not set. Create an H3 ComfyUI pod and set this env var."
            )
        comfyui_url = f"https://{pod_id}-8188.proxy.runpod.net"

        # Calculate frame count snapped to H3's 17k+5 grid
        # Formula: max(5, round(duration * 24)) + (5 - (max(5, round(duration * 24)) % 17)) % 17
        raw_frames = max(5, round(duration * 24))
        length = raw_frames + (5 - (raw_frames % 17)) % 17

        client_id = _uuid.uuid4().hex[:8]

        # If chaining, add a LoadImage node to load the frame and wire it to
        # MiniMaxH3ImageToVideo's first_frame input. Without this, passing the
        # filename directly as a node link reference fails silently because
        # ComfyUI expects ["node_id", output_index] not a filename string.
        load_image_node: dict = {}
        first_frame_link = None
        if first_frame_filename:
            load_image_node = {
                "200": {
                    "inputs": {
                        "image": first_frame_filename,
                        "upload": "image",
                    },
                    "class_type": "LoadImage",
                }
            }
            first_frame_link = ["200", 0]  # output 0 of LoadImage node = IMAGE tensor

        # Exact workflow from the working ComfyUI H3 T2V template
        workflow = {
            "92": {
                "inputs": {
                    "filename_prefix": "video/h3_pipeline",
                    "format": "auto",
                    "codec": "auto",
                    "video": ["105:91", 0]
                },
                "class_type": "SaveVideo"
            },
            "115": {
                "inputs": {
                    "aspect_ratio": "16:9 (Widescreen)",
                    "megapixels": 0.4,
                    "multiple": 32
                },
                "class_type": "ResolutionSelector"
            },
            "105:11": {
                "inputs": {
                    "vae_name": "minimax_h3_video_vae_fp16.safetensors"
                },
                "class_type": "VAELoader"
            },
            "105:24": {
                "inputs": {
                    "vae_name": "minimax_h3_audio_vae_fp32.safetensors"
                },
                "class_type": "VAELoader"
            },
            "105:23": {
                "inputs": {
                    "samples": ["105:14", 0],
                    "vae": ["105:24", 0]
                },
                "class_type": "VAEDecodeAudio"
            },
            "105:10": {
                "inputs": {
                    "samples": ["105:14", 0],
                    "vae": ["105:11", 0]
                },
                "class_type": "VAEDecode"
            },
            "105:17": {
                "inputs": {
                    "sampler_name": "res_multistep"
                },
                "class_type": "KSamplerSelect"
            },
            "105:9": {
                "inputs": {
                    "scheduler": "simple",
                    "steps": 20,
                    "denoise": 1,
                    "model": ["105:6", 0]
                },
                "class_type": "BasicScheduler"
            },
            "105:14": {
                "inputs": {
                    "noise": ["105:15", 0],
                    "guider": ["105:16", 0],
                    "sampler": ["105:17", 0],
                    "sigmas": ["105:9", 0],
                    "latent_image": ["105:104", 1]
                },
                "class_type": "SamplerCustomAdvanced"
            },
            "105:16": {
                "inputs": {
                    "model": ["105:6", 0],
                    "conditioning": ["105:104", 0]
                },
                "class_type": "BasicGuider"
            },
            "105:6": {
                "inputs": {
                    "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader"
            },
            "105:13": {
                "inputs": {
                    "clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                    "type": "minimax",
                    "device": "default"
                },
                "class_type": "CLIPLoader"
            },
            "105:15": {
                "inputs": {
                    "noise_seed": seed
                },
                "class_type": "RandomNoise"
            },
            "105:91": {
                "inputs": {
                    "fps": 24,
                    "bit_depth": 8,
                    "images": ["105:10", 0],
                    "audio": ["105:23", 0]
                },
                "class_type": "CreateVideo"
            },
            "105:104": {
                "inputs": {
                    "prompt": clean_prompt,
                    "width": ["115", 0],
                    "height": ["115", 1],
                    "length": length,
                    "clip": ["105:13", 0],
                    "vae": ["105:11", 0],
                    **({"first_frame": first_frame_link} if first_frame_link else {}),
                },
                "class_type": "MiniMaxH3ImageToVideo"
            },
            **load_image_node,  # empty dict if no chaining, LoadImage node if chaining
        }

        # Submit workflow to ComfyUI
        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Check if ComfyUI is running
            try:
                health = await client.get(f"{comfyui_url}/system_stats")
                if health.status_code != 200:
                    raise RuntimeError(
                        f"ComfyUI not responding at {comfyui_url} (status={health.status_code}). "
                        "Is the H3 pod running? Start it with: python pod_manager.py start"
                    )
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot connect to ComfyUI at {comfyui_url}. "
                    "Is the H3 pod running? Start it with: python pod_manager.py start"
                )

            # Queue the prompt
            resp = await client.post(
                f"{comfyui_url}/prompt",
                json=payload,
            )
            if resp.status_code != 200:
                error_detail = resp.text[:500]
                raise RuntimeError(
                    f"ComfyUI rejected workflow (HTTP {resp.status_code}): {error_detail}"
                )
            result = resp.json()
            prompt_id = result["prompt_id"]
            logger.info("ViewmaxMCPClient: H3 ComfyUI prompt queued: %s", prompt_id)

        # Poll /history until the prompt is done.
        # With serial generation (_MAX_CONCURRENT=1) each clip is processed
        # one at a time, so 15 min per clip is sufficient.
        # Give 30 min to be safe for large 15s clips on slower GPUs.
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(180):  # 180 * 10s = 30 min
                await asyncio.sleep(10)

                try:
                    hist_resp = await client.get(f"{comfyui_url}/history/{prompt_id}")
                    if hist_resp.status_code == 200:
                        history = hist_resp.json()
                        if prompt_id in history:
                            prompt_result = history[prompt_id]
                            status = prompt_result.get("status", {})

                            if status.get("completed", False):
                                # Find the output video file
                                outputs = prompt_result.get("outputs", {})
                                for node_id, node_output in outputs.items():
                                    videos = (
                                        node_output.get("videos", [])
                                        or node_output.get("gifs", [])
                                        or node_output.get("images", [])
                                    )
                                    if videos:
                                        video_info = videos[0]
                                        filename = video_info["filename"]
                                        subfolder = video_info.get("subfolder", "")
                                        file_type = video_info.get("type", "output")

                                        # Download the video file
                                        params = {
                                            "filename": filename,
                                            "subfolder": subfolder,
                                            "type": file_type,
                                        }
                                        dl_resp = await client.get(
                                            f"{comfyui_url}/view",
                                            params=params,
                                            timeout=120.0,
                                        )
                                        dl_resp.raise_for_status()
                                        mp4_bytes = dl_resp.content

                                        logger.info(
                                            "ViewmaxMCPClient: H3 received %d bytes of MP4 (%ds clip)",
                                            len(mp4_bytes), duration,
                                        )
                                        return mp4_bytes

                                raise RuntimeError(
                                    f"H3 prompt {prompt_id} completed but no video found in outputs"
                                )

                            if status.get("status_str") == "error":
                                messages = status.get("messages", [])
                                error_msg = str(messages) if messages else "unknown error"
                                raise RuntimeError(
                                    f"H3 ComfyUI generation failed: {error_msg}"
                                )
                except httpx.HTTPStatusError as exc:
                    logger.warning(
                        "ViewmaxMCPClient: H3 history poll %d failed (HTTP %d) — retrying",
                        attempt + 1, exc.response.status_code,
                    )
                except RuntimeError:
                    raise
                except Exception as exc:
                    logger.debug(
                        "ViewmaxMCPClient: H3 history poll %d: %s — retrying",
                        attempt + 1, exc,
                    )

        raise RuntimeError(f"H3 ComfyUI prompt {prompt_id} timed out after 30 minutes")

    @staticmethod
    def _clean_prompt(raw_prompt: str, is_cinematic: bool = False) -> str:
        """Clean the prompt before sending to the video provider.

        For MiniMax H3 T2VA briefs the prompt contains structured fields
        (integrated_multimodal_description / overall_soundscape /
        non_diegetic_music) with <d>[Language] dialogue.</d> tags that the
        model parses for lipsync.  We must NOT strip [Language] tags from
        inside <d>…</d> blocks, or H3 will ignore the dialogue entirely.

        Args:
            raw_prompt: The raw prompt string.
            is_cinematic: True when called in cinematic mode — uses the
                dialogue-safe cleaning path regardless of prompt content.
                Avoids fragile content-sniffing when the T2VA brief is
                partially formed (e.g. BEAT parse fallback).

        Safe cleanup:
          - Strip pipeline annotation tags like [pause] / [emphasis] that
            appear OUTSIDE <d> blocks.
          - Collapse multiple spaces.
          - For H3 T2VA briefs skip truncation — the full brief is required.
          - For all other providers truncate to 350 chars.
        """
        import re as _re  # noqa: PLC0415

        # Use is_cinematic as the authoritative signal; fall back to content
        # sniffing only when the flag is not set (backwards compat for tests).
        is_h3_brief = is_cinematic or ("integrated_multimodal_description" in raw_prompt)

        if is_h3_brief:
            # Only strip [pause]/[emphasis]-style tags that are NOT inside <d> tags.
            # Strategy: split on <d>...</d> blocks, clean outside blocks only.
            parts = _re.split(r"(<d>.*?</d>)", raw_prompt, flags=_re.DOTALL)
            cleaned_parts = []
            for part in parts:
                if part.startswith("<d>") and part.endswith("</d>"):
                    # Inside a dialogue block — preserve exactly as-is
                    cleaned_parts.append(part)
                else:
                    # Outside dialogue — strip pipeline annotation tags only
                    cleaned_parts.append(
                        _re.sub(r"\[/?(?:pause|emphasis|break|breath)\]", "", part)
                    )
            clean = "".join(cleaned_parts).strip()
            # Collapse multiple spaces but preserve newlines (field separators)
            clean = _re.sub(r"[ \t]+", " ", clean)
            return clean  # no truncation for H3 briefs
        else:
            # Legacy path for Kling / LTX-Video — strip all bracket annotations
            clean = _re.sub(r"\[/?[a-zA-Z]+\]", "", raw_prompt).strip()
            clean = " ".join(clean.split())
            return clean[:350]


# ---------------------------------------------------------------------------
# ClipResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    """Result of generating (or substituting) a single video clip.

    Attributes:
        segment_index: Zero-based index of the script segment this clip
            corresponds to.
        mp4_bytes: Raw MP4 bytes *or* raw JPEG bytes if ``is_fallback`` is
            ``True`` (ffmpeg handles JPEG inputs specially).
        is_fallback: ``True`` when the clip was produced by the static-image
            fallback path (all Viewmax retries were exhausted).
    """

    segment_index: int
    mp4_bytes: bytes
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Internal helpers — script parsing
# ---------------------------------------------------------------------------


@dataclass
class _Segment:
    """A single parsed script segment."""

    title: str
    body: str


def _parse_segments(content: str) -> list[_Segment]:
    """Split *content* into segments using Markdown headings.

    Splits on ``## `` or ``### `` headings.  If no headings are found, the
    content is divided into three equal thirds with labels ``hook``, ``body``,
    and ``cta``.

    Args:
        content: Full script Markdown content.

    Returns:
        Non-empty list of :class:`_Segment` objects.
    """
    # Match lines that start with ## or ### followed by space
    heading_pattern = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(content))

    if not matches:
        # No headings — split into equal thirds
        total = len(content)
        third = max(total // 3, 1)
        return [
            _Segment(title="hook", body=content[:third].strip()),
            _Segment(title="body", body=content[third : 2 * third].strip()),
            _Segment(title="cta", body=content[2 * third :].strip()),
        ]

    segments: list[_Segment] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        # Body runs from end of this heading line to start of next heading (or EOF)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        segments.append(_Segment(title=title, body=body))

    return segments


def _build_scene_prompt(segment: _Segment, style_profile: StyleProfile) -> str:
    """Derive a Viewmax scene prompt for *segment* (first clip)."""
    body_preview = segment.body[:100]
    patterns_str = ", ".join(style_profile.visual_style.composition_patterns)
    return f"{segment.title} visual scene: {body_preview}. Style: {patterns_str}"


async def _rephrase_prompt_for_moderation(original_prompt: str) -> str:
    """Ask Claude to rewrite a prompt that was blocked by Kling's risk control system.

    Strips copyrighted character names and replaces them with visual descriptions
    so the same scene idea passes Kling's content moderation.

    Args:
        original_prompt: The prompt that was rejected.

    Returns:
        A rephrased prompt, or the original if Claude is unavailable.
    """
    try:
        client = build_claude_client()
        prompt = (
            "You are an AI video prompt editor. Rewrite video prompts that were "
            "rejected by a video generator's content moderation system.\n\n"
            "RULES:\n"
            "1. Replace every named fictional character, franchise name, or copyrighted IP "
            "with a pure visual description. Examples:\n"
            "   - Superman / Man of Steel → 'a caped hero in a blue suit with golden solar energy'\n"
            "   - Thor / Odinson → 'a blond-bearded warrior in silver armor with a glowing war hammer'\n"
            "   - Batman / Dark Knight → 'a dark-armored figure in a black cape and cowl'\n"
            "   - Iron Man / Tony Stark → 'a hero in red-and-gold powered armor'\n"
            "   - Spider-Man → 'a hero in a red-and-blue web-patterned suit'\n"
            "   - Hulk / Bruce Banner → 'a massive green-skinned giant in torn shorts'\n"
            "   - Goku / Vegeta → 'a spiky-haired fighter in an orange uniform with golden aura'\n"
            "   - Captain America → 'a hero in a blue star-spangled suit with a round shield'\n"
            "   - Mjolnir → 'glowing war hammer'\n"
            "   - Vibranium → 'advanced alloy'\n"
            "   - Arc reactor → 'glowing chest device'\n"
            "   - Metropolis / Gotham → 'a futuristic city'\n"
            "2. Remove or soften any explicit violence — replace 'destroys', 'kills', 'attacks' "
            "with 'clashes with', 'confronts', 'faces off against'.\n"
            "3. Keep the camera direction, motion, and lighting descriptions exactly the same.\n"
            "4. Return only the rewritten prompt, no explanation.\n"
            "5. CRITICAL: the output must contain ZERO copyrighted names, character names, "
            "franchise names, or weapon names.\n\n"
            f"Rewrite this prompt removing all copyrighted names and softening "
            f"any explicit violence:\n\n{original_prompt}"
        )
        result = await client.complete(prompt, max_tokens=300)
        rephrased = result.strip().strip('"\'')
        logger.info("Prompt rephrased for moderation: %s", rephrased[:80])
        return rephrased[:400]

    except Exception as exc:
        logger.warning("Failed to rephrase prompt via Claude: %s — using original", exc)
        return original_prompt


async def _generate_character_descriptions(script_topic: str, script_body: str) -> dict:
    """Dynamically generate character visual descriptions from the script topic.

    Calls the LLM once per video to identify ALL characters in the script and
    create proper, gender-correct, visually accurate descriptions for each.
    Supports 2-10 characters (e.g., team battles, power rankings, crossovers).

    Returns:
        Dict with keys:
          - "characters": list of dicts, each with "name" and "desc"
          - Legacy keys (hero1_name, hero1_desc, hero2_name, hero2_desc) for backwards compat
    """
    try:
        client = build_claude_client()
        prompt = (
            f"Video topic: {script_topic}\n"
            f"Script excerpt: {script_body[:500]}\n\n"
            "Identify ALL characters mentioned or implied in this video topic and script. "
            "There may be 2, 3, 4, or more characters. For EACH character, write a detailed "
            "visual description that a video generator can use to render them accurately.\n\n"
            "NEVER use copyrighted names — describe only by physical appearance.\n\n"
            "IMPORTANT: Include gender, body type, hair, costume/armor details, "
            "signature weapon or power visuals, and any distinctive features.\n\n"
            "Format your response as numbered entries:\n"
            "CHAR1_NAME: [short label like 'amazonian warrior' or 'armored titan']\n"
            "CHAR1_DESC: [full visual description, 30-50 words]\n"
            "CHAR2_NAME: [short label]\n"
            "CHAR2_DESC: [full visual description, 30-50 words]\n"
            "CHAR3_NAME: [short label] (if applicable)\n"
            "CHAR3_DESC: [full visual description] (if applicable)\n"
            "... continue for all characters\n\n"
            "Examples:\n"
            "CHAR1_NAME: amazonian warrior princess\n"
            "CHAR1_DESC: Tall athletic woman with long flowing black hair, golden tiara with red star, "
            "red and blue armored corset, silver bracelet gauntlets, golden lasso at hip, fierce expression\n"
            "CHAR2_NAME: thunder god\n"
            "CHAR2_DESC: Muscular tall man with long blonde hair, silver winged helmet, red cape flowing, "
            "silver chainmail armor, wielding a short-handled war hammer crackling with blue lightning\n"
            "CHAR3_NAME: scarlet speedster\n"
            "CHAR3_DESC: Lean athletic man in skin-tight crimson red suit with golden lightning bolt "
            "emblem on chest, cowl with small golden lightning bolt ears, yellow boots, motion blur trails\n\n"
            "Output ONLY the numbered character lines, nothing else. Minimum 2, maximum 10 characters."
        )
        result = await client.complete(prompt, max_tokens=600)
        lines = [l.strip() for l in result.strip().split("\n") if l.strip()]

        characters: list[dict[str, str]] = []
        current_name = ""
        current_desc = ""

        for line in lines:
            upper = line.upper()
            # Match patterns like CHAR1_NAME:, CHAR2_NAME:, etc.
            if "_NAME:" in upper:
                if current_name and current_desc:
                    characters.append({"name": current_name, "desc": current_desc})
                current_name = line.split(":", 1)[1].strip()
                current_desc = ""
            elif "_DESC:" in upper:
                current_desc = line.split(":", 1)[1].strip()

        # Don't forget the last character
        if current_name and current_desc:
            characters.append({"name": current_name, "desc": current_desc})

        # Fallback if parsing failed
        if len(characters) < 2:
            characters = [
                {"name": "hero 1", "desc": "a powerful superhero in dramatic pose with distinctive costume"},
                {"name": "hero 2", "desc": "a powerful warrior in battle stance with unique armor"},
            ]

        logger.info(
            "Character descriptions generated: %d characters — %s",
            len(characters),
            ", ".join(c["name"] for c in characters),
        )

        # Build result dict with both new format and legacy keys for backwards compat
        result_dict: dict = {"characters": characters}
        # Legacy keys (for any code still using hero1/hero2 format)
        result_dict["hero1_name"] = characters[0]["name"]
        result_dict["hero1_desc"] = characters[0]["desc"]
        result_dict["hero2_name"] = characters[1]["name"] if len(characters) > 1 else characters[0]["name"]
        result_dict["hero2_desc"] = characters[1]["desc"] if len(characters) > 1 else characters[0]["desc"]

        return result_dict
    except Exception as exc:
        logger.warning("Character description generation failed: %s — using generic fallback", exc)
        fallback_chars = [
            {"name": "hero 1", "desc": "a powerful superhero in dramatic pose with distinctive costume"},
            {"name": "hero 2", "desc": "a powerful warrior in battle stance with unique armor"},
        ]
        return {
            "characters": fallback_chars,
            "hero1_name": "hero 1",
            "hero1_desc": "a powerful superhero in dramatic pose with distinctive costume",
            "hero2_name": "hero 2",
            "hero2_desc": "a powerful warrior in battle stance with unique armor",
        }


async def _generate_kids_t2va_brief(
    segment: _Segment,
    clip_idx: int,
    total_clips: int,
    script_topic: str = "",
) -> str:
    """Convert a kids rhyming verse into an H3 T2VA brief with 3D animation style.

    Uses Claude to generate a scene description that matches the actual verse
    content — so a farm song gets farm scenes, a colors song gets colorful
    settings, a Spiderman song gets city scenes etc.

    Falls back to a generic cheerful brief if Claude is unavailable.
    """
    import re as _re  # noqa: PLC0415

    title = segment.title.strip()
    body = segment.body.strip()

    # Strip annotation tags for visual description, keep for voiceover
    visual_body = _re.sub(r"\[/?[a-zA-Z\s]+\]", "", body).strip()

    # Scene energy based on position in the video
    progress = clip_idx / max(1, total_clips - 1)
    if progress < 0.25:
        scene_energy = "gentle opening, calm and inviting"
        time_of_day = "bright sunny morning"
    elif progress < 0.6:
        scene_energy = "playful and energetic, full of movement"
        time_of_day = "cheerful midday"
    elif progress < 0.85:
        scene_energy = "exciting and joyful, peak energy"
        time_of_day = "golden afternoon light"
    else:
        scene_energy = "warm and satisfying, winding down"
        time_of_day = "cozy warm sunset"

    camera_moves = [
        "the camera slowly pushes in with small amplitude",
        "the camera gently pulls out with small amplitude",
        "the camera holds a static warm wide shot",
        "the camera slowly pans right at slow speed",
        "the camera gently tilts up at slow speed",
        "the camera arcs around the scene at slow speed",
    ]
    camera = camera_moves[clip_idx % len(camera_moves)]

    # Extract singable lines for voiceover
    lines = [l.strip() for l in body.split("\n") if l.strip()][:4]
    sung_lines = " ".join(lines)[:300]

    # Ask Claude to generate a scene description that fits the actual content
    scene_description = await _generate_kids_scene_description(
        script_topic=script_topic,
        verse_title=title,
        verse_body=visual_body[:200],
        clip_idx=clip_idx,
        total_clips=total_clips,
    )

    brief = f"""integrated_multimodal_description: [Shot 1] Bright cheerful 3D animated style, Pixar-quality rendering, vivid saturated colors, {time_of_day}, {scene_energy}. {scene_description}. {camera} as the scene unfolds warmly. A warm friendly narrator voice, clear and melodic, speaking in a bouncy sing-song rhythm (S1) says in an off-screen voiceover: <d>[English] {sung_lines}</d> S1's lips remain completely closed throughout. The characters and animals react with big happy expressions and gentle movements in sync with the song. The scene is full of bright colors, sparkles, and gentle motion.

overall_soundscape: Cheerful ambient sounds matching the scene — gentle breeze, birds singing, happy animal sounds, children laughing. Everything is warm, safe, and joyful with no sudden loud noises.

non_diegetic_music: Light xylophone melody at a bouncy moderate tempo, ukulele chords, gentle hand percussion on every beat, bright major key, cheerful and singable throughout the entire clip."""

    logger.info(
        "Kids T2VA brief clip %d/%d: %s", clip_idx + 1, total_clips, brief[:80]
    )
    return brief


async def _generate_kids_scene_description(
    script_topic: str,
    verse_title: str,
    verse_body: str,
    clip_idx: int,
    total_clips: int,
) -> str:
    """Ask Claude to describe a visually appropriate 3D animated scene for this verse.

    Returns a 1-2 sentence scene description that H3 uses as the visual context.
    Falls back to a generic cheerful scene if Claude fails.
    """
    try:
        client = build_claude_client()
        prompt = (
            f"Song topic: {script_topic}\n"
            f"Verse title: {verse_title}\n"
            f"Verse content: {verse_body}\n"
            f"Clip {clip_idx + 1} of {total_clips}\n\n"
            "Write a 1-2 sentence description of a bright cheerful 3D animated scene "
            "(Pixar/CoComelon style) that visually matches this verse. "
            "The scene should show what the song is ABOUT — if it's about a farm, show farm animals; "
            "if it's about colors, show colorful objects; if it's about Spiderman, show a friendly "
            "cartoon Spiderman in a kids-appropriate city scene. "
            "Each clip should show a DIFFERENT aspect or angle of the same world — "
            f"clip {clip_idx + 1} should look visually distinct from clip {clip_idx}. "
            "Keep it child-friendly, warm, and full of movement. "
            "Output ONLY the scene description, no explanations."
        )
        result = await client.complete(prompt, max_tokens=100)
        scene = result.strip().strip('"\'')
        logger.debug("Kids scene description clip %d: %s", clip_idx + 1, scene[:60])
        return scene[:250]
    except Exception as exc:
        logger.debug("Kids scene description fallback (clip %d): %s", clip_idx + 1, exc)
        # Fallback: generic but content-aware using the verse body
        return (
            f"A warm colorful 3D animated world showing {script_topic or verse_title}, "
            f"with happy characters, bright colors, and gentle playful movement"
        )


async def _generate_cinematic_t2va_brief(
    segment: _Segment,
    clip_num: int,
    total_clips: int,
    scene_state: dict,
    script_topic: str,
    character_descs: Optional[dict],
) -> str:
    """Convert a cinematic screenplay segment into an H3 T2VA production brief.

    Parses the BEAT / ACTION / LINE / CHAR / SOUND / MUSIC / DAMAGE fields
    written by _build_cinematic_generation_prompt and assembles the three
    required T2VA sections:
      - integrated_multimodal_description
      - overall_soundscape
      - non_diegetic_music

    Falls back to a generic action brief when the segment has no structured
    fields (e.g. plain narration accidentally piped through cinematic mode).
    """
    import re as _re  # noqa: PLC0415

    title = segment.title.strip()
    body = segment.body.strip()

    # ---- Parse structured screenplay fields --------------------------------
    def _field(key: str) -> str:
        m = _re.search(rf"^{key}:\s*(.+)$", body, _re.MULTILINE | _re.IGNORECASE)
        return m.group(1).strip() if m else ""

    beat_type = _field("BEAT").lower()  # "action" or "dialogue"
    action_desc = _field("ACTION")
    sound_desc = _field("SOUND")
    music_desc = _field("MUSIC")
    damage_desc = _field("DAMAGE") or "environment shows battle damage"

    char1 = _field("CHAR1")
    voice1 = _field("VOICE1")
    line1 = _field("LINE1")
    char2 = _field("CHAR2")
    voice2 = _field("VOICE2")
    line2 = _field("LINE2")
    reaction = _field("REACTION")

    if beat_type not in ("action", "dialogue", ""):
        logger.warning(
            "Cinematic brief clip %d/%d: unrecognized BEAT type %r — "
            "falling back to action brief (no dialogue will be generated)",
            clip_num + 1, total_clips, beat_type,
        )

    # If BEAT is completely absent AND no ACTION text was written either,
    # the segment is unparseable — raise rather than silently produce a
    # brief with no meaningful content that H3 will render as mute action.
    if beat_type == "" and not action_desc and not line1:
        raise ValueError(
            f"Unreadable cinematic segment '{title}': BEAT field missing or "
            f"unparseable and no ACTION/LINE1 fallback available. "
            f"Segment body: {body[:200]!r}"
        )

    # Scene slug from title (e.g. "SCENE 3 — EXT. ROOFTOP - DUSK")
    slug_match = _re.search(r"—\s*(.+)$", title)
    slug = slug_match.group(1).strip() if slug_match else title

    # Clip position context
    global_pos = scene_state.get("global_clip_num", clip_num)
    total_global = scene_state.get("total_global_clips", total_clips)
    progress = global_pos / max(1, total_global - 1)

    # Choose camera movement based on position
    camera_moves = [
        "the camera pushes in with small amplitude at slow speed",
        "the camera pulls out with large amplitude at fast speed",
        "the camera arcs around the subject at fast speed",
        "the camera holds a static shot, no pan, no push-in",
        "the camera tracks alongside the action at fast speed",
        "the camera tilts up with small amplitude at slow speed",
        "the camera shakes strongly during the impact",
    ]
    camera = camera_moves[clip_num % len(camera_moves)]

    # Style prefix
    style = (
        "Ultra-realistic live-action cinematic style, Hollywood blockbuster IMAX, "
        "photorealistic 8K, shallow depth of field, film grain, dynamic lighting"
    )

    # ---- Build T2VA brief based on beat type --------------------------------
    if beat_type == "dialogue" and (line1 or line2):
        # Dialogue beat — assign speaker IDs
        s1_desc = char1 or (character_descs.get("hero1_desc", "a powerful warrior") if character_descs else "a powerful warrior")
        s1_voice = voice1 or "deep commanding voice"
        s2_desc = char2 or (character_descs.get("hero2_desc", "a fierce opponent") if character_descs else "a fierce opponent")
        s2_voice = voice2 or "cold calm voice"

        # Build shot description
        shot_body = (
            f"[Shot 1] {style}. {slug}. {damage_desc}. "
            f"{s1_desc} stands across from {s2_desc}. {camera}. "
        )

        # Speaker 1 line
        if line1:
            shot_body += (
                f"The {s1_desc.split(',')[0].strip()} with a {s1_voice} (S1) says: "
                f"<d>[English] {line1}</d> "
                f"Their lips close fully after the last word. "
            )

        # Cut to reaction / speaker 2
        if line2 and s2_desc:
            shot_body += (
                f"[Shot 2] At 00:05.000, the camera cuts to {s2_desc}. "
                f"The {s2_desc.split(',')[0].strip()} with a {s2_voice} (S2) replies: "
                f"<d>[English] {line2}</d> "
                f"Their lips close fully. "
            )

        if reaction:
            shot_body += f"{reaction} "

        # Add action continuation
        if action_desc:
            shot_body += (
                f"[Shot 3] At 00:10.000, the camera cuts to a wide shot as {action_desc}"
            )

        soundscape = sound_desc or (
            "Crackling supernatural energy, distant thunder, debris shifting, "
            "heavy breathing between combatants"
        )
        music = music_desc or (
            "Orchestral score with low brass drones building to a sharp brass accent "
            "at the moment of confrontation"
        )

    else:
        # Action beat — pure visual, no dialogue
        action = action_desc or f"{title}: intense combat and destruction"

        shot_body = (
            f"[Shot 1] {style}. {slug}. {damage_desc}. "
            f"{action}. {camera}. "
            f"[Shot 2] At 00:06.000, the camera cuts to a wide shot as the shockwave "
            f"tears outward, debris flying in all directions, energy crackling. "
            f"[Shot 3] At 00:11.000, the camera cuts to a close-up of the aftermath — "
            f"smoke rising, ground cracked and scorched, combatants reassessing."
        )

        soundscape = sound_desc or (
            "Thunderous impacts, energy explosions, debris raining down, "
            "shockwave roar echoing across the environment"
        )
        music = music_desc or (
            "Full orchestral and electronic hybrid, driving percussion, "
            "brass hits synchronized to each impact, building relentlessly"
        )

    brief = (
        f"integrated_multimodal_description: {shot_body}\n\n"
        f"overall_soundscape: {soundscape}\n\n"
        f"non_diegetic_music: {music}"
    )

    logger.info(
        "Cinematic T2VA brief clip %d/%d (%s): %s",
        clip_num + 1, total_clips, beat_type, brief[:80],
    )
    return brief


async def _generate_video_prompt_with_claude(
    segment: _Segment,
    clip_num: int,
    total_clips: int,
    scene_state: Optional[dict] = None,
    script_topic: str = "",
    character_descs: Optional[dict] = None,
    visual_prompt_mode: str = "narration",
) -> str:
    """Use Claude to generate an optimized video prompt with scene continuity.

    In ``narration`` mode: generic cinematic scene description (current behaviour).
    In ``cinematic`` mode: full H3 T2VA production brief with timed shots,
    speaker IDs, ``<d>[English]...</d>`` dialogue tags, soundscape, and score —
    parsed from the screenplay segment produced by _build_cinematic_generation_prompt.
    """
    import re as _re  # noqa: PLC0415

    # ------------------------------------------------------------------
    # Cinematic mode: parse screenplay segment → T2VA brief
    # ------------------------------------------------------------------
    if visual_prompt_mode == "cinematic":
        return await _generate_cinematic_t2va_brief(
            segment=segment,
            clip_num=clip_num,
            total_clips=total_clips,
            scene_state=scene_state or {},
            script_topic=script_topic,
            character_descs=character_descs,
        )

    body = _re.sub(r"\[/?[a-zA-Z]+\]", "", segment.body).strip()
    title = segment.title.strip()
    words = body.split()

    if total_clips > 1:
        chunk_size = max(1, len(words) // total_clips)
        start = clip_num * chunk_size
        end = start + chunk_size if clip_num < total_clips - 1 else len(words)
        body_chunk = " ".join(words[start:end])
    else:
        body_chunk = body

    if scene_state is None:
        scene_state = {}

    total_position = scene_state.get("global_clip_num", clip_num)
    total_global = scene_state.get("total_global_clips", total_clips)
    progress = total_position / max(1, total_global - 1)

    if progress < 0.25:
        scene_phase = "opening: storm clouds gathering, city intact, tension building, dusk"
        damage_level = "city intact, glass windows reflecting storm clouds"
        emotion_guidance = "characters sizing each other up — narrow eyes, battle stance, restrained power"
    elif progress < 0.5:
        scene_phase = "escalation: rain pouring, first impacts, blue lightning tears across sky"
        damage_level = "windows shattered on nearby buildings, cracks in pavement"
        emotion_guidance = "first contact — surprise, determination, adrenaline"
    elif progress < 0.75:
        scene_phase = "climax: heavy destruction, fires spreading, smoke filling air"
        damage_level = "buildings partially collapsed, debris everywhere, fires burning"
        emotion_guidance = "intense fury, gritted teeth, screaming with effort, pushing limits"
    else:
        scene_phase = "resolution: rubble-filled crater, embers floating, dust settling"
        damage_level = "massive crater, collapsed skyscrapers, embers floating"
        emotion_guidance = "exhausted but standing, determination in eyes, battle-worn"

    camera_options = [
        "epic aerial drone shot pushing forward slowly",
        "extreme close-up slow motion dolly-in",
        "drone camera rapidly pulling backward through explosion",
        "low-angle tracking shot circling the heroes",
        "handheld shaky cam rushing alongside the action",
        "crane shot slowly descending into the destruction",
        "orbit shot circling both warriors",
        "whip pan following the projectile",
    ]
    camera = camera_options[clip_num % len(camera_options)]

    # Master style block — injected once, keeps every clip stylistically consistent
    _MASTER_STYLE = (
        "Ultra realistic live-action Hollywood blockbuster IMAX HDR PBR "
        "ray-traced reflections volumetric lighting cinematic color grading "
        "24fps realistic destruction physics shallow depth of field dynamic "
        "camera movement film grain high temporal consistency."
    )

    # Use dynamic character descriptions if available, else generic fallback
    if character_descs is None:
        character_descs = {
            "characters": [
                {"name": "hero 1", "desc": "a powerful superhero in dramatic pose"},
                {"name": "hero 2", "desc": "a powerful warrior in battle stance"},
            ],
            "hero1_name": "hero 1",
            "hero1_desc": "a powerful superhero in dramatic pose",
            "hero2_name": "hero 2",
            "hero2_desc": "a powerful warrior in battle stance",
        }

    characters = character_descs.get("characters", [])
    if len(characters) < 2:
        characters = [
            {"name": character_descs.get("hero1_name", "hero 1"), "desc": character_descs.get("hero1_desc", "a powerful superhero")},
            {"name": character_descs.get("hero2_name", "hero 2"), "desc": character_descs.get("hero2_desc", "a powerful warrior")},
        ]

    # Build character roster for the system prompt
    char_roster_lines = []
    for i, char in enumerate(characters, 1):
        char_roster_lines.append(f"CHARACTER {i} ({char['name']}): {char['desc']}")
    char_roster = "\n".join(char_roster_lines)

    # For the focus logic, pick which characters are active in this clip
    # Rotate through all characters, not just 2
    num_chars = len(characters)
    primary_idx = clip_num % num_chars
    secondary_idx = (clip_num + 1) % num_chars

    h1_name = characters[primary_idx]["name"]
    h1_desc = characters[primary_idx]["desc"]
    h2_name = characters[secondary_idx]["name"]
    h2_desc = characters[secondary_idx]["desc"]

    system_prompt = f"""You are an expert AI video prompt engineer writing Hollywood storyboard prompts for a video generator.

VIDEO TOPIC: {script_topic if script_topic else "superhero battle"}

ALL CHARACTERS IN THIS VIDEO:
{char_roster}

IMPORTANT: Every prompt MUST describe the characters' full appearance from the descriptions above. Never use names, pronouns, or vague references like "the hero." Always include gender, costume details, and distinctive features so the video generator renders the correct characters.

ENVIRONMENT: Futuristic city. Dark storm clouds, rain-soaked streets, glass skyscrapers. Damage escalates: intact → windows crack → buildings shake → skyscrapers collapse → streets split → crater forms.

MASTER STYLE (append to every prompt unchanged):
{_MASTER_STYLE}

RULES:
1. Read the script action — show EXACTLY the characters mentioned in it
2. BOTH/ALL characters in the script action must be FULLY DESCRIBED by appearance in the prompt
3. Vary actions — no two clips should show the same move
4. TRANSITION PHRASES (use each once in order): Clip2="The collision detonates..." Clip3="Before the smoke clears..." Clip4="Emerging from the dust cloud..." Clip5="In the battle's aftermath..." Clip6+="In the silence that follows..."
5. EMOTION WORDS — forbidden: "widen". Use: jaw tightens / gaze sharpens / expression hardens / eyes blaze / grimaces / grins with contempt / unwavering stare / refuses to yield
6. Append master style tag at end
7. Max 350 characters before style tag, present tense, no dialogue
8. CRITICAL: output must contain ZERO character names, franchise names, or copyrighted IP — describe ONLY by physical appearance
9. Match the character descriptions from the roster above EXACTLY — use the costume details, body type, and features listed. The video generator only renders what you describe."""

    # Alternate focus across ALL characters, always showing at least 2 in frame
    # Let Claude decide which characters are active based on the script chunk
    # (handled in the user_prompt below — Claude picks from the full roster)
    focus_instruction = (
        "Based on the script action below, identify which characters from the roster "
        "are involved in THIS specific moment and describe them in the prompt. "
        "Always describe at least 2 characters with their FULL appearance."
    )
    clip_focus = "Script-driven — show the characters mentioned in the script action"

    # If there are 3+ characters, remind Claude about extras
    extra_chars_in_scene = ""
    if num_chars > 2:
        extra_chars_in_scene = (
            f"\nThis video has {num_chars} characters total. Show ONLY the ones "
            "relevant to the current script action. Others may appear in background if the scene calls for it."
        )

    user_prompt = f"""VIDEO TOPIC: {script_topic if script_topic else "superhero battle"}
Segment: "{title}"
Script action (FOLLOW THIS CLOSELY): "{body_chunk}"
Scene phase: {scene_phase}
Damage: {damage_level}
Emotion (ONE word, never "widen"): {emotion_guidance}
Camera: {camera}
{extra_chars_in_scene}
Clip {clip_num + 1}/{total_clips} (global {total_position + 1}/{total_global})
{"Transition: use clip " + str(min(clip_num, 5) + 2) + " phrase" if clip_num > 0 else "Opening clip — no transition"}

INSTRUCTIONS:
1. Read the "Script action" above carefully — it tells you EXACTLY which characters are doing what.
2. From the CHARACTER ROSTER in the system prompt, identify which characters match the ones in the script action.
3. Describe those specific characters by their FULL APPEARANCE (costume, features, body type) — not by name.
4. If the script mentions 2 characters interacting, BOTH must be fully described and visible in the prompt.
5. If the script mentions only 1 character, show them as the focus with another character reacting in background.
6. The video generator ONLY renders what you explicitly describe — unnamed or briefly-mentioned characters will NOT appear.

Write ONE prompt. End with the master style tag."""

    try:
        client = build_claude_client()
        result = await client.complete(
            f"{system_prompt}\n\n{user_prompt}",
            max_tokens=350,
        )
        prompt = result.strip().strip('"\'')
        logger.info("Claude prompt clip %d/%d: %s", clip_num + 1, total_clips, prompt[:80])
        return prompt[:500]

    except Exception as exc:
        logger.warning("Claude prompt failed for '%s' clip %d: %s — fallback", title[:30], clip_num, exc)
        return _build_scene_prompt_variant_fallback(segment, clip_num, total_clips)

def _build_scene_prompt_variant_fallback(
    segment: _Segment,
    clip_num: int,
    total_clips: int,
) -> str:
    """Fallback template prompt when Claude is unavailable."""
    import re as _re  # noqa: PLC0415
    body = _re.sub(r"\[/?[a-zA-Z]+\]", "", segment.body).strip()
    title = segment.title.strip()
    words = body.split()
    if total_clips > 1:
        chunk_size = max(1, len(words) // total_clips)
        start = clip_num * chunk_size
        end = start + chunk_size if clip_num < total_clips - 1 else len(words)
        body_chunk = " ".join(words[start:end])[:150]
    else:
        body_chunk = body[:150]

    templates = [
        f"Dynamic action: {title}. {body_chunk}. Fast motion, cinematic 4K.",
        f"Battle scene: {body_chunk}. Explosion on impact, debris flying. 4K.",
        f"Hero in motion: {body_chunk}. Camera tracking, dramatic lighting. 4K.",
        f"Power unleashed: {title}. {body_chunk}. Energy surging, cinematic.",
        f"Epic climax: {body_chunk}. Camera pulls back, golden light. IMAX 4K.",
    ]
    return templates[clip_num % len(templates)][:300]


# ---------------------------------------------------------------------------
# Internal helpers — fallback image
# ---------------------------------------------------------------------------


def _make_fallback_jpeg(title: str) -> bytes:
    """Create a 1920×1080 JPEG fallback image with *title* centred on a black
    background.

    Args:
        title: Text to render at the centre of the frame (the segment title).

    Returns:
        Raw JPEG bytes.
    """
    img = Image.new("RGB", (_FALLBACK_WIDTH, _FALLBACK_HEIGHT), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Use the default PIL bitmap font — no external font file required
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=72)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Calculate text bounding box for centring
    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow versions
        text_w, text_h = draw.textsize(title, font=font)  # type: ignore[attr-defined]

    x = (_FALLBACK_WIDTH - text_w) // 2
    y = (_FALLBACK_HEIGHT - text_h) // 2
    draw.text((x, y), title, fill=(255, 255, 255), font=font)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers — thumbnail
# ---------------------------------------------------------------------------


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert a CSS hex color string (``#RRGGBB``) to an RGB tuple.

    Falls back to black on any parse error.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (0, 0, 0)
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except ValueError:
        return (0, 0, 0)


def _contrasting_color(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return a colour that contrasts with *rgb* (simple luminance flip)."""
    r, g, b = rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (255, 255, 255) if luminance < 128 else (0, 0, 0)


def _extract_thumbnail_from_clips(clip_results: list[ClipResult]) -> Optional[bytes]:
    """Extract the best frame from the generated clips to use as thumbnail.

    Picks the middle frame from the first non-fallback MP4 clip.
    Falls back to None if extraction fails (caller uses text overlay instead).

    Args:
        clip_results: List of generated clips.

    Returns:
        JPEG bytes of the extracted frame, or None if extraction fails.
    """
    import subprocess as _sp  # noqa: PLC0415
    import shutil as _sh  # noqa: PLC0415

    ffmpeg_bin = (
        _sh.which("ffmpeg")
        or "/opt/homebrew/bin/ffmpeg"
        or "/usr/local/bin/ffmpeg"
        or "ffmpeg"
    )

    # Find first real (non-fallback) clip
    real_clips = [c for c in clip_results if not c.is_fallback]
    if not real_clips:
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            clip_path = tmp_path / "clip.mp4"
            clip_path.write_bytes(real_clips[0].mp4_bytes)
            thumb_path = tmp_path / "thumb.jpg"

            # Extract frame at 50% of the clip duration (middle frame)
            result = _sp.run(
                [
                    ffmpeg_bin, "-y",
                    "-i", str(clip_path),
                    "-vf", "thumbnail,scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720",
                    "-frames:v", "1",
                    str(thumb_path),
                ],
                capture_output=True,
                timeout=30,
            )

            if result.returncode != 0 or not thumb_path.exists():
                # Try simpler extraction without thumbnail filter
                result2 = _sp.run(
                    [
                        ffmpeg_bin, "-y",
                        "-i", str(clip_path),
                        "-ss", "00:00:02",
                        "-frames:v", "1",
                        "-vf", "scale=1280:720",
                        str(thumb_path),
                    ],
                    capture_output=True,
                    timeout=30,
                )
                if result2.returncode != 0 or not thumb_path.exists():
                    logger.warning("Thumbnail extraction failed: %s", result2.stderr[-200:].decode("utf-8", errors="replace"))
                    return None

            thumb_bytes = thumb_path.read_bytes()
            logger.info("Thumbnail extracted from first clip (%d bytes)", len(thumb_bytes))

            # Ensure under 2MB
            if len(thumb_bytes) > _THUMBNAIL_MAX_BYTES:
                img = Image.open(io.BytesIO(thumb_bytes))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=70)
                thumb_bytes = buf.getvalue()

            return thumb_bytes

    except Exception as exc:
        logger.warning("Thumbnail extraction from clip failed: %s", exc)
        return None


async def _generate_ai_thumbnail(
    title: str,
    script_body: str,
) -> Optional[bytes]:
    """Generate a professional YouTube thumbnail using Claude + Ideogram/Pollinations.

    Steps:
    1. Claude analyzes the script and generates a DETAILED, topic-specific
       image generation prompt (characters, composition, colors, text placement)
    2. Ideogram (best) or Pollinations.AI renders the thumbnail image
    3. If the image API doesn't render text well, Pillow adds text overlay

    The key difference from a generic prompt: Claude reads the actual script
    and describes the SPECIFIC characters, scene, and conflict — not a generic
    "two characters facing off" every time.

    Args:
        title: Video title / segment title.
        script_body: Full script body text for context.

    Returns:
        JPEG bytes of the thumbnail, or None if generation fails.
    """
    import httpx  # noqa: PLC0415

    # ---- Step 1: Claude generates full thumbnail image prompt ---------------
    # This is the core improvement: Claude reads the script and outputs a
    # detailed, YouTube-CTR-optimized image generation prompt tailored to
    # the specific video content.

    try:
        client = build_claude_client()

        meta_prompt = f"""You are a YouTube thumbnail designer who creates prompts for AI image generators (Ideogram).

VIDEO TITLE: {title}
FULL SCRIPT:
{script_body}

Generate a SINGLE detailed image generation prompt for a 1280×720 YouTube thumbnail that will maximize CTR (click-through rate).

REQUIREMENTS:
1. DESCRIBE THE ACTUAL CHARACTERS from the script by their appearance (costume, colors, powers, body type) — NEVER use copyrighted character names
2. COMPOSITION: characters on left and right sides, facing each other, filling 80-90% of frame, low-angle or eye-level dramatic perspective
3. STYLE: Match the content — for anime/manga/superhero topics use high-quality anime key visual style (like Jujutsu Kaisen, Demon Slayer, Attack on Titan). For live-action topics use cinematic photorealistic style. NEVER generic.
4. BACKGROUND: Destroyed/dramatic environment matching the script (ruined shrine, destroyed city, space, etc.), NOT plain gradients
5. LIGHTING: Dramatic rim lighting, energy glow from characters illuminating the scene, deep shadows, high contrast
6. ENERGY EFFECTS: Glowing auras, energy blasts, particle effects, sparks, motion blur — make it feel explosive
7. TEXT: Specify large bold ALL-CAPS text in the upper portion of the image that does NOT cover character faces:
   - Main text (huge, white with thick dark outline): a 2-4 word punchy question or statement from the video topic
   - Sub-text (smaller, red or yellow): a 2-4 word hook
8. Full-bleed edge-to-edge, no borders, no watermarks, no logos, designed for mobile visibility

TEXT RULES:
- Derive the text from the actual VIDEO TITLE — make it feel like a direct teaser for this specific video
- Text must be EXTREMELY large, Impact-font style, bold, with thick dark outline and slight glow
- Upper-left or upper area placement so character faces are visible

Output ONLY the image generation prompt (300-450 words). No explanations, no markdown."""

        result = await client.complete(meta_prompt, max_tokens=800)
        image_prompt = result.strip().strip('"\'')
        logger.info("Claude thumbnail prompt generated: %s", image_prompt[:120])

    except Exception as exc:
        logger.warning("Claude thumbnail prompt generation failed: %s — using fallback", exc)
        # Fallback: generic but better than before
        image_prompt = (
            f"Ultra-wide 16:9 YouTube thumbnail, 1280x720, full-bleed edge-to-edge, "
            f"no borders, no white space. Dramatic cinematic scene for: {title}. "
            f"Two powerful characters in dramatic confrontation poses filling 90% of frame. "
            f"Dramatic volumetric lighting, lens flare, cinematic depth, "
            f"destroyed environment background, high contrast, "
            f"photorealistic, Hollywood movie poster quality, 8K detail. "
            f"Large bold text 'WHO WINS?' with thick dark outline and glow. "
            f"Designed for maximum YouTube CTR and mobile visibility."
        )

    # ---- Step 2: Generate the thumbnail image --------------------------------
    img_bytes = None

    # Try Ideogram first (best quality, supports text rendering)
    ideogram_key = os.environ.get("IDEOGRAM_API_KEY", "")
    if not ideogram_key:
        logger.warning(
            "IDEOGRAM_API_KEY not set — falling back to Pollinations for thumbnail. "
            "Add IDEOGRAM_API_KEY to GitHub Secrets for better quality thumbnails."
        )
    if ideogram_key:
        try:
            async with httpx.AsyncClient(timeout=90.0) as http_client:
                resp = await http_client.post(
                    "https://api.ideogram.ai/v1/ideogram-v4/generate",
                    headers={"Api-Key": ideogram_key, "Content-Type": "application/json"},
                    json={"text_prompt": image_prompt, "aspect_ratio": "16:9"},
                )
                resp.raise_for_status()
                data = resp.json()
                images = data.get("data", [])
                if images and images[0].get("url"):
                    img_url = images[0]["url"]
                    img_resp = await http_client.get(img_url, timeout=30.0)
                    img_resp.raise_for_status()
                    img_bytes = img_resp.content
                    logger.info("Ideogram thumbnail generated (%d bytes)", len(img_bytes))
        except Exception as exc:
            logger.warning("Ideogram thumbnail failed: %s — trying Pollinations", exc)
            img_bytes = None

    # Fallback: Pollinations.AI (free, no key needed)
    if not img_bytes:
        import urllib.parse  # noqa: PLC0415
        import random as _random  # noqa: PLC0415

        encoded_prompt = urllib.parse.quote(image_prompt[:500])
        thumb_seed = _random.randint(0, 99999)
        url = (
            f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            f"?width=1280&height=720&model=flux-pro&nologo=true&enhance=true&seed={thumb_seed}"
        )

        try:
            async with httpx.AsyncClient(timeout=60.0) as http_client:
                resp = await http_client.get(url)
                resp.raise_for_status()
                img_bytes = resp.content
            logger.info("Pollinations thumbnail generated (%d bytes)", len(img_bytes))
        except Exception as exc:
            logger.warning("Pollinations image generation failed: %s", exc)
            return None

    if not img_bytes:
        logger.warning("No thumbnail image generated from any source")
        return None

    # ---- Step 3: Final processing ----------------------------------------
    # If Ideogram generated the image, it likely rendered the text already
    # (Ideogram v4 has excellent text rendering). Just validate size.
    # If Pollinations was used, text won't be in the image — but since the
    # prompt includes text instructions, we trust the AI rendered it.
    # Only add Pillow text overlay if the image is from Pollinations AND
    # looks like it has no text (heuristic: skip overlay for Ideogram).
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if img.size != (1280, 720):
            img = img.resize((1280, 720), Image.LANCZOS)

        # Save as high quality JPEG
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        result = buf.getvalue()

        logger.info("AI thumbnail generated (%d bytes)", len(result))
        return result

    except Exception as exc:
        logger.warning("Thumbnail post-processing failed: %s", exc)
        # Return the raw image bytes if processing fails
        return img_bytes if len(img_bytes) < _THUMBNAIL_MAX_BYTES else None


def _make_thumbnail_jpeg(style_profile: StyleProfile, title: str) -> bytes:
    """Generate a 1280×720 JPEG thumbnail using the StyleProfile.

    Layout
    ------
    * Background filled with the first dominant color (black if none).
    * White text overlay at the ``text_overlay_position`` (top / center / bottom).
    * A placeholder rectangle in a contrasting color to represent subject framing.

    Args:
        style_profile: Source of dominant colors, text overlay position, and
            subject framing metadata.
        title: Text to render as the overlay (typically the video title).

    Returns:
        JPEG bytes guaranteed to be < 2 MB (compressed at quality 85, then 70
        if still too large).
    """
    tc = style_profile.thumbnail_composition

    # --- Background color ------------------------------------------------
    bg_rgb: tuple[int, int, int] = (0, 0, 0)
    if tc.dominant_colors:
        bg_rgb = _hex_to_rgb(tc.dominant_colors[0])

    img = Image.new("RGB", (_THUMBNAIL_WIDTH, _THUMBNAIL_HEIGHT), color=bg_rgb)
    draw = ImageDraw.Draw(img)

    # --- Subject-framing placeholder rectangle ---------------------------
    contrast_rgb = _contrasting_color(bg_rgb)
    rect_margin = 80
    draw.rectangle(
        [
            (rect_margin, rect_margin),
            (_THUMBNAIL_WIDTH - rect_margin, _THUMBNAIL_HEIGHT - rect_margin),
        ],
        outline=contrast_rgb,
        width=6,
    )

    # --- Text overlay ----------------------------------------------------
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=60)
    except (IOError, OSError):
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(title, font=font)  # type: ignore[attr-defined]

    padding = 40
    position = tc.text_overlay_position.lower() if tc.text_overlay_position else "center"

    x = (_THUMBNAIL_WIDTH - text_w) // 2

    if position == "top":
        y = padding
    elif position == "bottom":
        y = _THUMBNAIL_HEIGHT - text_h - padding
    else:  # center (default)
        y = (_THUMBNAIL_HEIGHT - text_h) // 2

    draw.text((x, y), title, fill=(255, 255, 255), font=font)

    # --- Compress to stay < 2 MB -----------------------------------------
    for quality in (85, 70):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) < _THUMBNAIL_MAX_BYTES:
            return data

    # Should not happen for a 1280×720 JPEG at quality=70, but return anyway
    return data  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers — version scanning
# ---------------------------------------------------------------------------


async def _next_version(
    asset_store: Asset_Store,
    video_id: str,
    subfolder: SubFolder,
    base_name: str,
    extension: str,
) -> int:
    """Return version 1 — each pipeline run uses a fresh video_id, so v1 is always free."""
    return 1


# ---------------------------------------------------------------------------
# Visual_Generator
# ---------------------------------------------------------------------------


class Visual_Generator:
    """Generates per-segment video clips via Viewmax, compiles them into an
    MP4, creates a thumbnail, and writes everything to Asset_Store.

    Args:
        viewmax_client: A :class:`ViewmaxClient`-compatible object (production
            or test double).
        asset_store: :class:`~pipeline.asset_store.Asset_Store` instance.
        content_calendar: :class:`~pipeline.content_calendar.Content_Calendar`
            instance used to update pipeline status.
        notifier: :class:`~pipeline.notifier.Notifier` instance for failure
            alerts.
    """

    def __init__(
        self,
        viewmax_client: ViewmaxClient,
        asset_store: Asset_Store,
        content_calendar: Content_Calendar,
        notifier: Notifier,
        visual_prompt_mode: str = "narration",
    ) -> None:
        self._viewmax = viewmax_client
        self._asset_store = asset_store
        self._calendar = content_calendar
        self._notifier = notifier
        self._visual_prompt_mode = visual_prompt_mode

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(
        self,
        script: Script,
        narration: Optional[NarrationAsset],
        style_profile: StyleProfile,
        video_id: str,
    ) -> VisualAsset:
        """Generate a compiled MP4 and thumbnail JPEG for *video_id*.

        In ``narration`` mode: clips are generated as cinematic b-roll and
        muxed with the TTS narration MP3.
        In ``cinematic`` mode: H3 T2VA briefs drive character dialogue and
        native audio — no external narration is muxed in.

        Steps
        -----
        1. Parse script segments.
        2. Derive one scene prompt per segment.
        3. Generate each clip via Viewmax (with retry + fallback).
        4. Write clips to a temp directory and compile via ffmpeg.
        5. Generate thumbnail JPEG from StyleProfile.
        6. Upload MP4 and thumbnail to Asset_Store.
        7. Update Content_Calendar status to ``Visuals Ready``.

        Args:
            script: Approved :class:`~pipeline.models.Script`.
            narration: :class:`~pipeline.models.NarrationAsset` with MP3 path.
            style_profile: :class:`~pipeline.models.StyleProfile` with visual
                composition and thumbnail data.
            video_id: Pipeline-assigned video identifier.

        Returns:
            :class:`~pipeline.models.VisualAsset` with Drive URLs set.

        Raises:
            VisualGeneratorError: On unrecoverable failures (ffmpeg error, all
                clips failed to upload, etc.).
        """
        logger.info("Visual_Generator.generate started for video_id=%s", video_id)

        # ---- 1. Parse segments ------------------------------------------
        segments = _parse_segments(script.content)
        logger.debug(
            "Parsed %d segments for video_id=%s: %s",
            len(segments),
            video_id,
            [s.title for s in segments],
        )

        # ---- 2. Get actual narration duration + calculate per-segment durations ----
        wpm = 150
        total_words = sum(len(s.body.split()) for s in segments)

        # Try to get actual narration duration from MP3 (narration mode only)
        actual_audio_seconds: float = 0.0
        if narration is not None:
            try:
                mp3_bytes = await self._asset_store.read(
                    video_id=video_id,
                    subfolder=SubFolder.NARRATION,
                    filename=narration.mp3_path,
                )
                # Get duration from MP3 header using mutagen
                try:
                    import mutagen.mp3 as _mp3  # noqa: PLC0415
                    import io as _io  # noqa: PLC0415
                    audio = _mp3.MP3(fileobj=_io.BytesIO(mp3_bytes))
                    actual_audio_seconds = audio.info.length
                    logger.info(
                        "Visual_Generator: actual narration duration=%.1fs "
                        "(bitrate=%d kbps, size=%d bytes)",
                        actual_audio_seconds,
                        getattr(audio.info, "bitrate", 0) // 1000,
                        len(mp3_bytes),
                    )
                except Exception as mutagen_exc:
                    # Fallback: use ffprobe if available, else estimate from word count
                    # Do NOT use file-size estimate — bitrate varies 64-320kbps
                    logger.warning(
                        "Visual_Generator: mutagen failed (%s) — trying ffprobe", mutagen_exc
                    )
                    try:
                        import subprocess as _sp  # noqa: PLC0415
                        import shutil as _sh  # noqa: PLC0415
                        import tempfile as _tf  # noqa: PLC0415
                        ffprobe = _sh.which("ffprobe") or "/opt/homebrew/bin/ffprobe" or "ffprobe"
                        with _tf.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                            tmp_mp3.write(mp3_bytes)
                            tmp_mp3_path = tmp_mp3.name
                        probe = _sp.run(
                            [ffprobe, "-v", "error", "-show_entries",
                             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                             tmp_mp3_path],
                            capture_output=True, timeout=15,
                        )
                        import os as _os  # noqa: PLC0415
                        _os.unlink(tmp_mp3_path)
                        if probe.returncode == 0 and probe.stdout.strip():
                            actual_audio_seconds = float(probe.stdout.strip())
                            logger.info(
                                "Visual_Generator: ffprobe narration duration=%.1fs",
                                actual_audio_seconds,
                            )
                    except Exception as probe_exc:
                        logger.warning(
                            "Visual_Generator: ffprobe also failed (%s) — "
                            "will use word-count estimate", probe_exc
                        )
            except Exception as exc:
                logger.warning(
                    "Visual_Generator: could not read narration MP3 (%s), "
                    "using word count estimate", exc
                )

        # Use actual duration if available, otherwise estimate from word count
        if actual_audio_seconds > 0:
            total_audio_seconds = actual_audio_seconds
        else:
            total_audio_seconds = max(len(segments) * 5, (total_words / wpm) * 60)
            logger.info(
                "Visual_Generator: using word-count estimate for duration=%.1fs "
                "(%d words / %d wpm × 60)",
                total_audio_seconds, total_words, wpm,
            )

        segment_durations = []
        for seg in segments:
            seg_words = len(seg.body.split())
            seg_duration = max(3.0, (seg_words / wpm) * 60)
            segment_durations.append(seg_duration)
        raw_sum = sum(segment_durations)
        segment_durations = [d * total_audio_seconds / raw_sum for d in segment_durations]
        logger.info(
            "Visual_Generator: %d segments, total duration=%.1fs, cutting every ~10s",
            len(segments), total_audio_seconds,
        )

        # ---- 3. Generate clips — one per segment ------
        # Extract topic from script: use first heading or first line as context
        # for Claude to generate relevant video prompts
        script_topic = ""
        for line in script.content.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                script_topic = line[:100]
                break

        clip_results = await self._generate_all_clips(
            segments, style_profile, video_id,
            segment_durations=segment_durations,
            script_topic=script_topic,
            visual_prompt_mode=self._visual_prompt_mode,
        )

        # ---- 4. Build per-clip durations ----------------------------------------
        # Cinematic mode: 1 clip per scene, each exactly CLIP_DURATION seconds.
        # Narration mode: split each segment into clips of CLIP_DURATION each.
        _CLIP_DUR: float = float(getattr(self._viewmax, "CLIP_DURATION", 15))
        per_clip_durations: list[float] = []
        if self._visual_prompt_mode == "cinematic":
            # One clip per scene — screenplay already defines the exact scene count
            per_clip_durations = [_CLIP_DUR] * len(segments)
        elif self._visual_prompt_mode == "kids_rhyming":
            # Kids: split segments by duration so total clips cover the full audio length.
            # A 3-min kids script may only have 3-4 ## headings but needs ~12 clips of 15s.
            _CUT_EVERY_S: float = _CLIP_DUR
            for seg_dur in segment_durations:
                n = max(1, round(seg_dur / _CUT_EVERY_S))
                per_clip_durations.extend([seg_dur / n] * n)
        else:
            _CUT_EVERY_S: float = _CLIP_DUR
            for seg_dur in segment_durations:
                n = max(1, round(seg_dur / _CUT_EVERY_S))
                per_clip_durations.extend([seg_dur / n] * n)

        logger.info(
            "Visual_Generator: %d total clips, durations: %s",
            len(per_clip_durations),
            [f"{d:.1f}s" for d in per_clip_durations],
        )

        # ---- 5. Compile MP4 with ffmpeg ----------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mp4_bytes = await self._compile_video(
                clip_results=clip_results,
                segments=segments,
                segment_durations=per_clip_durations,  # one duration per clip
                narration_mp3_path=narration.mp3_path if narration else None,
                tmp_path=tmp_path,
                video_id=video_id,
            )

            # ---- 5. Generate thumbnail — AI image + text overlay -----------
            video_title = segments[0].title if segments else video_id
            thumbnail_bytes = await _generate_ai_thumbnail(video_title, script.content) or \
                              _make_thumbnail_jpeg(style_profile, video_title)

            # ---- 6. Determine version and upload --------------------------
            mp4_version = await _next_version(
                self._asset_store, video_id, SubFolder.VIDEOS, video_id, "mp4"
            )
            thumb_version = await _next_version(
                self._asset_store, video_id, SubFolder.THUMBNAILS, video_id, "jpg"
            )

            mp4_filename = f"{video_id}_v{mp4_version}.mp4"
            thumb_filename = f"{video_id}_v{thumb_version}.jpg"

            logger.info(
                "Uploading %s and %s for video_id=%s",
                mp4_filename,
                thumb_filename,
                video_id,
            )

            mp4_url = await self._asset_store.write(
                video_id, SubFolder.VIDEOS, mp4_filename, mp4_bytes
            )
            thumbnail_url = await self._asset_store.write(
                video_id, SubFolder.THUMBNAILS, thumb_filename, thumbnail_bytes
            )

        # ---- 7. Update Content_Calendar status ---------------------------
        await self._calendar.update_status(video_id, PipelineStatus.VISUALS_READY)
        logger.info(
            "Visual_Generator.generate complete for video_id=%s "
            "(mp4=%s, thumbnail=%s)",
            video_id,
            mp4_filename,
            thumb_filename,
        )

        return VisualAsset(
            video_id=video_id,
            version=mp4_version,
            mp4_path=f"videos/{mp4_filename}",
            thumbnail_path=f"thumbnails/{thumb_filename}",
            mp4_url=mp4_url,
            thumbnail_url=thumbnail_url,
            created_at=datetime.now(tz=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Task 11.1 — Per-clip generation with retry + fallback
    # ------------------------------------------------------------------

    async def _generate_all_clips(
        self,
        segments: list[_Segment],
        style_profile: StyleProfile,
        video_id: str,
        segment_durations: Optional[list[float]] = None,
        script_topic: str = "",
        visual_prompt_mode: str = "narration",
    ) -> list[ClipResult]:
        """Dispatch clip generation to the mode-specific implementation.

        Each mode has its own independent logic so changes to one cannot
        accidentally break the other:
          - narration: splits segments into multiple clips by duration,
            parallel generation (up to 5 concurrent for cloud APIs, 1 for pod).
          - cinematic: one clip per screenplay scene, always serial on the pod,
            T2VA brief prompts with dialogue tags.
        """
        if visual_prompt_mode == "cinematic":
            return await self._generate_clips_cinematic(
                segments=segments,
                video_id=video_id,
                script_topic=script_topic,
            )
        if visual_prompt_mode == "kids_rhyming":
            return await self._generate_clips_kids(
                segments=segments,
                video_id=video_id,
                script_topic=script_topic,
                segment_durations=segment_durations,
            )
        return await self._generate_clips_narration(
            segments=segments,
            video_id=video_id,
            segment_durations=segment_durations,
            script_topic=script_topic,
        )

    # ------------------------------------------------------------------
    # Cinematic clip generation — one clip per screenplay scene
    # ------------------------------------------------------------------

    async def _generate_clips_cinematic(
        self,
        segments: list[_Segment],
        video_id: str,
        script_topic: str = "",
    ) -> list[ClipResult]:
        """Generate one clip per screenplay scene using T2VA briefs.

        Rules:
        - 1 clip per ## SCENE segment — the screenplay already wrote the
          correct number of scenes; over-splitting repeats dialogue fields.
        - Always serial (MAX_CONCURRENT=1) — single ComfyUI pod, one GPU.
        - No stagger delay needed since generation is already sequential.
        - Character descriptions generated once and shared across all clips.
        """
        total_clips = len(segments)

        # Character descriptions — generated once, shared across all scenes
        script_body = segments[0].body if segments else ""
        character_descs = await _generate_character_descriptions(script_topic, script_body)

        logger.info(
            "Visual_Generator [cinematic]: generating %d clips (1 per scene, serial)",
            total_clips,
        )

        # Build T2VA prompts sequentially (ordering matters for scene continuity)
        prompts: list[str] = []
        for clip_idx, segment in enumerate(segments):
            scene_state = {
                "global_clip_num": clip_idx,
                "total_global_clips": total_clips,
            }
            prompt = await _generate_video_prompt_with_claude(
                segment, clip_num=0, total_clips=1,
                scene_state=scene_state,
                script_topic=script_topic,
                character_descs=character_descs,
                visual_prompt_mode="cinematic",
            )
            prompts.append(prompt)

        # Generate clips one at a time — ComfyUI pod has one GPU.
        # Last-frame chaining: extract the last frame of clip N and pass it
        # as first_frame to clip N+1 so H3 anchors visually to the previous shot.
        results: list[ClipResult] = []
        last_frame_filename: Optional[str] = None  # None for clip 1 (pure T2V)

        # Need the ComfyUI URL to upload frames — derive it here
        _pod_id = os.environ.get("RUNPOD_H3_POD_ID", "").strip()
        _comfyui_url = f"https://{_pod_id}-8188.proxy.runpod.net" if _pod_id else ""

        for clip_idx, (segment, prompt) in enumerate(zip(segments, prompts)):
            logger.info(
                "Visual_Generator [cinematic]: clip %d/%d — %s%s",
                clip_idx + 1, total_clips, segment.title[:50],
                f" [chaining from clip {clip_idx}]" if last_frame_filename else " [T2V anchor]",
            )

            # Inject first_frame into the viewmax client call for this clip
            # We call _generate_single_clip which calls self._viewmax.generate_clip
            # which calls _call_minimax_h3 — we need to pass first_frame_filename through.
            # _generate_single_clip → generate_clip → _call_minimax_h3
            # The cleanest path: temporarily set first_frame on the client before calling.
            # Use a context-safe approach: pass via a per-call attribute.
            if hasattr(self._viewmax, "_call_minimax_h3"):
                self._viewmax._next_first_frame = last_frame_filename
            else:
                self._viewmax._next_first_frame = None

            result = await self._generate_single_clip(
                segment_index=clip_idx,
                segment=segment,
                prompt=prompt,
                video_id=video_id,
                seed=-1,
            )
            results.append(result)

            # If clip succeeded and we have a ComfyUI URL, extract last frame for next clip
            last_frame_filename = None  # reset — don't chain from a failed clip
            if (
                not result.is_fallback
                and len(result.mp4_bytes) > 10_000
                and _comfyui_url
                and clip_idx < total_clips - 1  # no need to chain after last clip
            ):
                frame_jpeg = await self._viewmax._extract_last_frame(result.mp4_bytes)
                if frame_jpeg:
                    stored_name = await self._viewmax._upload_frame_to_comfyui(
                        frame_jpeg,
                        _comfyui_url,
                        filename=f"chain_frame_{clip_idx:02d}.jpg",
                    )
                    if stored_name:
                        last_frame_filename = stored_name
                        logger.info(
                            "Visual_Generator [cinematic]: clip %d last frame uploaded as '%s' "
                            "→ will anchor clip %d",
                            clip_idx + 1, stored_name, clip_idx + 2,
                        )
                    else:
                        logger.warning(
                            "Visual_Generator [cinematic]: clip %d frame upload failed "
                            "— clip %d will use T2V mode",
                            clip_idx + 1, clip_idx + 2,
                        )
                else:
                    logger.warning(
                        "Visual_Generator [cinematic]: clip %d frame extraction failed "
                        "— clip %d will use T2V mode",
                        clip_idx + 1, clip_idx + 2,
                    )

            # Brief cooldown between clips to allow H3 pod GPU to fully reset.
            # Prevents alternating clip failures caused by VRAM not being freed
            # before the next generation request arrives.
            if clip_idx < total_clips - 1:
                logger.info(
                    "Visual_Generator [cinematic]: clip %d done — waiting 15s for pod cooldown",
                    clip_idx + 1,
                )
                await asyncio.sleep(15.0)

        return results

    # ------------------------------------------------------------------
    # Narration clip generation — multiple clips per segment by duration
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Kids rhyming clip generation — one clip per verse, 3D animation style
    # ------------------------------------------------------------------

    async def _generate_clips_kids(
        self,
        segments: list[_Segment],
        video_id: str,
        script_topic: str = "",
        segment_durations: Optional[list[float]] = None,
    ) -> list[ClipResult]:
        """Generate clips for kids rhyming mode using bright 3D animation style.

        Rules:
        - Duration-based splitting: each segment gets n clips based on its duration
          and CLIP_DURATION (15s). A 3-min script with 3 segments → ~12 clips total.
        - Always serial — single ComfyUI pod.
        - T2VA brief uses cheerful 3D animation style and narrator voiceover format.
        - Last-frame chaining for environment continuity between clips.
        - H3 native audio kept — no external TTS.
        """
        _CLIP_DUR: float = float(getattr(self._viewmax, "CLIP_DURATION", 15))

        # Build expanded clip list based on duration (same logic as narration)
        clip_tasks: list[tuple[int, _Segment, float]] = []  # (clip_idx_in_seg, segment, seg_dur)
        for seg_idx, segment in enumerate(segments):
            seg_dur = (
                segment_durations[seg_idx]
                if segment_durations and seg_idx < len(segment_durations)
                else 15.0
            )
            n = max(1, round(seg_dur / _CLIP_DUR))
            for _ in range(n):
                clip_tasks.append((seg_idx, segment, seg_dur))

        total_clips = len(clip_tasks)
        logger.info(
            "Visual_Generator [kids_rhyming]: generating %d clips (%d segments × ~%ds each, serial)",
            total_clips, len(segments), _CLIP_DUR,
        )

        # Build T2VA prompts for each clip
        prompts: list[str] = []
        for clip_idx, (seg_idx, segment, _) in enumerate(clip_tasks):
            prompt = await _generate_kids_t2va_brief(
                segment=segment,
                clip_idx=clip_idx,
                total_clips=total_clips,
                script_topic=script_topic,
            )
            prompts.append(prompt)

        # Generate clips serially with last-frame chaining
        results: list[ClipResult] = []
        last_frame_filename: Optional[str] = None

        _pod_id = os.environ.get("RUNPOD_H3_POD_ID", "").strip()
        _comfyui_url = f"https://{_pod_id}-8188.proxy.runpod.net" if _pod_id else ""

        for clip_idx, ((seg_idx, segment, _), prompt) in enumerate(zip(clip_tasks, prompts)):
            logger.info(
                "Visual_Generator [kids_rhyming]: clip %d/%d — %s%s",
                clip_idx + 1, total_clips, segment.title[:50],
                f" [chaining from clip {clip_idx}]" if last_frame_filename else " [first clip]",
            )

            if hasattr(self._viewmax, "_call_minimax_h3"):
                self._viewmax._next_first_frame = last_frame_filename
            else:
                self._viewmax._next_first_frame = None

            result = await self._generate_single_clip(
                segment_index=seg_idx,
                segment=segment,
                prompt=prompt,
                video_id=video_id,
                seed=-1,
            )
            results.append(result)

            # Brief cooldown between clips to allow H3 pod GPU to fully reset.
            # Without this, alternating clips fail because the pod hasn't freed
            # VRAM from the previous generation before the next request arrives.
            if clip_idx < total_clips - 1:
                logger.info(
                    "Visual_Generator [kids_rhyming]: clip %d done — waiting 15s for pod cooldown",
                    clip_idx + 1,
                )
                await asyncio.sleep(15.0)

            # Chain last frame to next clip for environment continuity
            last_frame_filename = None
            if (
                not result.is_fallback
                and len(result.mp4_bytes) > 10_000
                and _comfyui_url
                and clip_idx < total_clips - 1
            ):
                frame_jpeg = await self._viewmax._extract_last_frame(result.mp4_bytes)
                if frame_jpeg:
                    stored_name = await self._viewmax._upload_frame_to_comfyui(
                        frame_jpeg,
                        _comfyui_url,
                        filename=f"kids_chain_{clip_idx:02d}.jpg",
                    )
                    if stored_name:
                        last_frame_filename = stored_name

        return results

    # ------------------------------------------------------------------
    # Narration clip generation — multiple clips per segment by duration
    # ------------------------------------------------------------------

    async def _generate_clips_narration(
        self,
        segments: list[_Segment],
        video_id: str,
        segment_durations: Optional[list[float]] = None,
        script_topic: str = "",
    ) -> list[ClipResult]:
        """Generate b-roll clips covering the full narration duration.

        Rules:
        - Split each segment into clips of CLIP_DURATION each based on duration.
        - Parallel generation: 1 concurrent for minimax_h3 pod, 5 for cloud APIs.
        - 2s stagger between launches to avoid thundering-herd on the pod.
        - Character descriptions generated once, shared across all clips for
          visual consistency.
        """
        _CLIP_DUR: float = float(getattr(self._viewmax, "CLIP_DURATION", 15))
        provider = getattr(self._viewmax, "_provider", "kling")
        _MAX_CONCURRENT: int = 1 if provider == "minimax_h3" else 5

        # Character descriptions — generated once, shared across all clips
        script_body = segments[0].body if segments else ""
        character_descs = await _generate_character_descriptions(script_topic, script_body)

        # Build clip task list — one entry per clip to generate
        clip_tasks: list[dict] = []
        global_clip_num = 0

        total_clips_all_segments = sum(
            max(1, round(
                (segment_durations[i] if segment_durations and i < len(segment_durations) else 30.0)
                / _CLIP_DUR
            ))
            for i in range(len(segments))
        )

        for seg_idx, segment in enumerate(segments):
            seg_duration = (
                segment_durations[seg_idx]
                if segment_durations and seg_idx < len(segment_durations)
                else 30.0
            )
            n_clips = max(1, round(seg_duration / _CLIP_DUR))
            logger.info(
                "Visual_Generator [narration]: segment '%s' %.1fs → %d clips",
                segment.title, seg_duration, n_clips,
            )
            for clip_num in range(n_clips):
                clip_tasks.append({
                    "seg_idx": seg_idx,
                    "segment": segment,
                    "clip_num": clip_num,
                    "n_clips": n_clips,
                    "global_clip_num": global_clip_num,
                    "total_global_clips": total_clips_all_segments,
                })
                global_clip_num += 1

        logger.info(
            "Visual_Generator [narration]: generating %d clips (max %d concurrent)",
            len(clip_tasks), _MAX_CONCURRENT,
        )

        # Build prompts sequentially (Claude needs ordering context)
        prompts: list[str] = []
        for task in clip_tasks:
            scene_state = {
                "global_clip_num": task["global_clip_num"],
                "total_global_clips": task["total_global_clips"],
            }
            prompt = await _generate_video_prompt_with_claude(
                task["segment"], task["clip_num"], task["n_clips"],
                scene_state=scene_state,
                script_topic=script_topic,
                character_descs=character_descs,
                visual_prompt_mode="narration",
            )
            prompts.append(prompt)

        # Generate clips in parallel batches using semaphore
        semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

        async def _gen_with_limit(idx: int) -> ClipResult:
            await asyncio.sleep(idx * 2.0)  # stagger to avoid thundering herd
            async with semaphore:
                return await self._generate_single_clip(
                    segment_index=clip_tasks[idx]["seg_idx"],
                    segment=clip_tasks[idx]["segment"],
                    prompt=prompts[idx],
                    video_id=video_id,
                    seed=-1,
                )

        clip_results = await asyncio.gather(
            *[_gen_with_limit(i) for i in range(len(clip_tasks))]
        )
        return list(clip_results)

    async def _generate_single_clip(
        self,
        segment_index: int,
        segment: _Segment,
        prompt: str,
        video_id: str,
        seed: int = -1,
    ) -> ClipResult:
        """Attempt to generate one clip via Viewmax; fall back to JPEG on exhaustion.

        Retry policy:
        * Up to 3 attempts.
        * On KlingContentModerationError, rephrases the prompt once via Claude
          (replacing copyrighted character names with visual descriptions) before
          retrying — no sleep needed since the error is deterministic.
        * Random delay between other failure attempts drawn from
          ``random.uniform(5, 30)`` seconds.

        Args:
            segment_index: Zero-based segment index (used in log messages).
            segment: The segment being rendered.
            prompt: Pre-built Viewmax scene prompt.
            video_id: Used in log messages.

        Returns:
            :class:`ClipResult` — either a real MP4 or a JPEG fallback.
        """
        last_exc: Optional[Exception] = None
        current_prompt = prompt
        moderation_rephrase_attempted = False

        for attempt in range(1, _CLIP_RETRY_ATTEMPTS + 1):
            try:
                logger.debug(
                    "Viewmax clip attempt %d/%d for segment '%s' (video_id=%s)",
                    attempt,
                    _CLIP_RETRY_ATTEMPTS,
                    segment.title,
                    video_id,
                )
                mp4_bytes = await self._viewmax.generate_clip(
                    prompt=current_prompt,
                    duration_seconds=getattr(self._viewmax, "CLIP_DURATION", 5),
                    seed=seed,
                )
                logger.debug(
                    "Viewmax clip succeeded for segment '%s' (video_id=%s, attempt=%d)",
                    segment.title,
                    video_id,
                    attempt,
                )
                return ClipResult(
                    segment_index=segment_index,
                    mp4_bytes=mp4_bytes,
                    is_fallback=mp4_bytes.startswith(b"\xff\xd8"),
                )
            except KlingContentModerationError as exc:
                last_exc = exc
                if attempt >= _CLIP_RETRY_ATTEMPTS or moderation_rephrase_attempted:
                    logger.warning(
                        "Kling moderation still blocks segment '%s' (video_id=%s) after "
                        "one rephrase; using a static fallback.",
                        segment.title,
                        video_id,
                    )
                    break

                moderation_rephrase_attempted = True
                logger.warning(
                    "Kling content moderation blocked prompt (attempt %d/%d) for "
                    "segment '%s' (video_id=%s): %s. Rephrasing once and retrying.",
                    attempt,
                    _CLIP_RETRY_ATTEMPTS,
                    segment.title,
                    video_id,
                    exc,
                )
                rephrased_prompt = await _rephrase_prompt_for_moderation(current_prompt)
                if not rephrased_prompt or rephrased_prompt.strip() == current_prompt.strip():
                    logger.warning(
                        "Kling moderation rephrase did not change segment '%s' "
                        "(video_id=%s); using a static fallback instead of resubmitting "
                        "the blocked prompt.",
                        segment.title,
                        video_id,
                    )
                    break

                current_prompt = rephrased_prompt
                logger.info(
                    "Rephrased prompt for segment '%s': %s",
                    segment.title,
                    current_prompt[:100],
                )
                # No sleep — the next request is a new task with a new prompt.
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < _CLIP_RETRY_ATTEMPTS:
                    delay = random.uniform(5, 30)
                    logger.warning(
                        "Viewmax clip failed (attempt %d/%d) for segment '%s' "
                        "(video_id=%s): %s. Retrying in %.1f s.",
                        attempt,
                        _CLIP_RETRY_ATTEMPTS,
                        segment.title,
                        video_id,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted — generate static JPEG fallback
        logger.warning(
            "All %d Viewmax retries exhausted for segment '%s' (segment_index=%d, "
            "video_id=%s): %s. Substituting static fallback JPEG.",
            _CLIP_RETRY_ATTEMPTS,
            segment.title,
            segment_index,
            video_id,
            last_exc,
        )
        fallback_bytes = _make_fallback_jpeg(segment.title)
        return ClipResult(
            segment_index=segment_index,
            mp4_bytes=fallback_bytes,
            is_fallback=True,
        )

    # ------------------------------------------------------------------
    # Task 11.2 — Video compilation via ffmpeg
    # ------------------------------------------------------------------

    async def _compile_video(
        self,
        clip_results: list[ClipResult],
        segments: list[_Segment],
        segment_durations: list[float],
        narration_mp3_path: Optional[str],
        tmp_path: Path,
        video_id: str,
    ) -> bytes:
        """Compile all clip results into a single MP4 using ffmpeg.

        Clips from real Viewmax calls are saved as ``.mp4`` files.
        Fallback clips (static JPEG) are saved as ``.jpg`` and given a
        looped ffmpeg input with ``-loop 1 -t {_FALLBACK_CLIP_DURATION_S}``.

        The narration MP3 is muxed in and the audio stream is taken with
        ``-shortest`` to ensure the output duration matches the audio track
        (audio sync tolerance ±100 ms).

        ffmpeg command pattern::

            ffmpeg
              [-loop 1] -i clip_N.{mp4,jpg} ...
              -i narration.mp3
              -filter_complex "concat=n={N}:v=1:a=0[v]"
              -map [v] -map {N}:a
              -c:v libx264 -r 24 -s 1920x1080
              -c:a aac -shortest
              output.mp4

        Args:
            clip_results: Ordered list of :class:`ClipResult` objects.
            segments: Matching segment list (used for logging only).
            narration_mp3_path: Filesystem path to the narration MP3.
            tmp_path: Writable temporary directory.
            video_id: Used in log messages.

        Returns:
            Raw MP4 bytes of the compiled video.

        Raises:
            VisualGeneratorError: If ffmpeg exits with a non-zero code.
        """
        # Write each clip to disk
        # For fallback clips, reuse the nearest real MP4 clip instead of a static JPEG.
        # First pass: write all real clips to disk so we have paths to reference.
        # Second pass: fill fallback slots — prefer the previous real clip, then
        # the next real clip (look-ahead), and only use the static JPEG if no real
        # clip exists anywhere in the list (i.e. every clip failed).
        import shutil as _shutil  # noqa: PLC0415

        # First pass — write real clips, record their paths by index.
        real_clip_paths: dict[int, Path] = {}
        for clip_idx, result in enumerate(clip_results):
            if not result.is_fallback:
                clip_file = tmp_path / f"clip_{clip_idx:04d}.mp4"
                clip_file.write_bytes(result.mp4_bytes)
                real_clip_paths[clip_idx] = clip_file

        # Helper: find the nearest real clip path (previous first, then next).
        def _nearest_real_clip(idx: int) -> Optional[Path]:
            # Search backwards
            for i in range(idx - 1, -1, -1):
                if i in real_clip_paths:
                    return real_clip_paths[i]
            # Search forwards
            for i in range(idx + 1, len(clip_results)):
                if i in real_clip_paths:
                    return real_clip_paths[i]
            return None

        # Second pass — build clip_paths with fallback slots resolved.
        clip_paths: list[tuple[Path, bool]] = []
        for clip_idx, result in enumerate(clip_results):
            if result.is_fallback:
                nearest = _nearest_real_clip(clip_idx)
                if nearest is not None:
                    clip_file = tmp_path / f"clip_{clip_idx:04d}.mp4"
                    _shutil.copy(nearest, clip_file)
                    clip_paths.append((clip_file, False))  # treat as real MP4
                    logger.info(
                        "Visual_Generator: fallback slot %d — reusing clip '%s'",
                        clip_idx, nearest.name,
                    )
                else:
                    # Every clip failed — last resort static JPEG
                    clip_file = tmp_path / f"clip_{clip_idx:04d}.jpg"
                    clip_file.write_bytes(result.mp4_bytes)
                    clip_paths.append((clip_file, True))
                    logger.warning(
                        "Visual_Generator: fallback slot %d — no real clip available, using static JPEG",
                        clip_idx,
                    )
            else:
                clip_paths.append((real_clip_paths[clip_idx], False))

        # ---------------------------------------------------------------------------
        # Audio handling — branches on visual_prompt_mode:
        #   narration: download TTS MP3 and mux as the single audio track
        #   cinematic:  keep H3 native audio from each clip; concat with video
        # ---------------------------------------------------------------------------
        cinematic_mode = self._visual_prompt_mode in ("cinematic", "kids_rhyming")

        local_mp3_path: Optional[Path] = None
        if not cinematic_mode:
            # narration mode — download MP3 from asset store
            if not narration_mp3_path:
                raise VisualGeneratorError(
                    f"narration_mp3_path is required in narration mode for video_id={video_id}"
                )
            try:
                mp3_bytes = await self._asset_store.read(
                    video_id=video_id,
                    subfolder=SubFolder.NARRATION,
                    filename=narration_mp3_path,
                )
                local_mp3_path = tmp_path / narration_mp3_path
                local_mp3_path.write_bytes(mp3_bytes)
                logger.debug(
                    "Narration MP3 downloaded to %s (%d bytes)",
                    local_mp3_path, len(mp3_bytes),
                )
            except Exception as exc:
                raise VisualGeneratorError(
                    f"Failed to download narration MP3 '{narration_mp3_path}' "
                    f"for video_id={video_id}: {exc}"
                ) from exc

        output_path = tmp_path / "output.mp4"
        n_clips = len(clip_paths)

        # Guard: duration list must match clip list exactly.
        # A mismatch causes silent 3-second fallback clips in ffmpeg — hard to debug.
        if len(segment_durations) != n_clips:
            raise VisualGeneratorError(
                f"Duration list length {len(segment_durations)} != clip list length {n_clips} "
                f"for video_id={video_id}. This is a bug in clip/duration calculation."
            )

        # Build ffmpeg argument list — try common install locations
        import shutil as _shutil  # noqa: PLC0415
        ffmpeg_bin = (
            _shutil.which("ffmpeg")
            or "/opt/homebrew/bin/ffmpeg"
            or "/usr/local/bin/ffmpeg"
            or "ffmpeg"
        )
        cmd: list[str] = [ffmpeg_bin, "-y"]

        for i, (clip_path, is_fallback) in enumerate(clip_paths):
            seg_dur = segment_durations[i] if i < len(segment_durations) else _FALLBACK_CLIP_DURATION_S
            if is_fallback:
                cmd += ["-loop", "1", "-t", f"{seg_dur:.2f}", "-i", str(clip_path)]
            else:
                cmd += ["-t", f"{seg_dur:.2f}", "-i", str(clip_path)]

        if not cinematic_mode:
            # Narration audio input (index = n_clips)
            cmd += ["-i", str(local_mp3_path)]

        # Filter graph — video normalisation (same for both modes)
        filter_parts = []
        for i in range(n_clips):
            filter_parts.append(
                f"[{i}:v]"
                f"scale=1920:1080:force_original_aspect_ratio=decrease,"
                f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1:1,"
                f"fps={_FFMPEG_FRAMERATE},"
                f"format=yuv420p"
                f"[v{i}]"
            )

        concat_inputs = "".join(f"[v{i}]" for i in range(n_clips))

        if cinematic_mode:
            # Cinematic mode: concat H3 native audio streams from each clip.
            # Clips without an audio stream get silent padding via aevalsrc.
            import subprocess as _sp  # noqa: PLC0415
            import shutil as _sh  # noqa: PLC0415
            ffprobe_bin = _sh.which("ffprobe") or "/opt/homebrew/bin/ffprobe" or "ffprobe"

            def _has_audio(path: Path) -> bool:
                r = _sp.run(
                    [ffprobe_bin, "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
                    capture_output=True,
                )
                return bool(r.stdout.strip())

            for i, (clip_path, _) in enumerate(clip_paths):
                if _has_audio(clip_path):
                    filter_parts.append(
                        f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]"
                    )
                else:
                    seg_dur = segment_durations[i] if i < len(segment_durations) else _FALLBACK_CLIP_DURATION_S
                    filter_parts.append(
                        f"aevalsrc=0:channel_layout=stereo:sample_rate=44100"
                        f":duration={seg_dur:.2f}[a{i}]"
                    )

            a_inputs = "".join(f"[a{i}]" for i in range(n_clips))
            filter_complex = (
                ";".join(filter_parts)
                + f";{concat_inputs}concat=n={n_clips}:v=1:a=0[vout]"
                + f";{a_inputs}concat=n={n_clips}:v=0:a=1[aout]"
            )
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", "[aout]",
                "-c:v", "libx264",
                "-r", str(_FFMPEG_FRAMERATE),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(output_path),
            ]
        else:
            # Narration mode: mux external MP3 as audio.
            # The concatenated video plays once, then the last frame loops
            # for any remaining audio duration via the tpad filter.
            # -shortest then trims both streams to the MP3 length.
            filter_complex = (
                ";".join(filter_parts)
                + f";{concat_inputs}concat=n={n_clips}:v=1:a=0[vconcat]"
                # tpad: hold last frame indefinitely so -shortest can trim to MP3 length
                + f";[vconcat]tpad=stop_mode=clone:stop_duration=600[vout]"
            )
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", f"{n_clips}:a",
                "-c:v", "libx264",
                "-r", str(_FFMPEG_FRAMERATE),
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                str(output_path),
            ]

        logger.debug("ffmpeg command: %s", " ".join(cmd))

        loop = asyncio.get_running_loop()
        try:
            proc_result: subprocess.CompletedProcess[bytes] = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    check=False,  # we check returncode manually below
                ),
            )
        except FileNotFoundError as exc:
            raise VisualGeneratorError(
                "ffmpeg executable not found. Ensure ffmpeg is installed and on PATH."
            ) from exc

        if proc_result.returncode != 0:
            stderr_snippet = proc_result.stderr[-2000:].decode("utf-8", errors="replace")
            raise VisualGeneratorError(
                f"ffmpeg failed (exit code {proc_result.returncode}) for "
                f"video_id={video_id}:\n{stderr_snippet}"
            )

        logger.info(
            "ffmpeg compilation succeeded for video_id=%s (%d clips)",
            video_id,
            n_clips,
        )
        return output_path.read_bytes()


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "ClipResult",
    "ViewmaxClient",
    "ViewmaxMCPClient",
    "VisualGeneratorError",
    "Visual_Generator",
]
