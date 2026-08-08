"""MiniMax H3 RunPod Serverless handler using diffusers.

Generates video clips via MiniMax H3 model loaded from a network volume.
Returns base64-encoded MP4 bytes.

H3 generates 4-15 second clips at 768p/24fps with optional native stereo audio.
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import sys
import tempfile

import runpod
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_VOLUME_ROOT = os.environ.get("VOLUME_ROOT", "/runpod-volume")
_MODEL_PATH = os.environ.get("H3_MODEL_PATH", f"{_VOLUME_ROOT}/MiniMax-H3")

# ---------------------------------------------------------------------------
# Lazy model loading
# ---------------------------------------------------------------------------
_pipeline = None


def _load_pipeline():
    """Load the H3 pipeline from local weights on first call."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    logger.info("Loading MiniMax H3 pipeline from %s ...", _MODEL_PATH)

    try:
        from diffusers import ModularPipeline  # noqa: PLC0415

        _pipeline = ModularPipeline.from_pretrained(
            _MODEL_PATH,
            variant="fl2va",
            torch_dtype=torch.bfloat16,
        )
        _pipeline.to("cuda")
        logger.info("MiniMax H3 pipeline loaded successfully")

    except ImportError:
        # Fallback: try loading via the H3-specific pipeline class
        logger.info("ModularPipeline not available, trying MiniMaxH3Pipeline...")
        from diffusers import DiffusionPipeline  # noqa: PLC0415

        _pipeline = DiffusionPipeline.from_pretrained(
            os.path.join(_MODEL_PATH, "FL2VA"),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _pipeline.to("cuda")
        logger.info("MiniMax H3 pipeline loaded via DiffusionPipeline")

    return _pipeline


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _generate_video(
    prompt: str,
    duration: int = 10,
    aspect_ratio: str = "16:9",
    seed: int = -1,
    audio_enabled: bool = False,
) -> bytes:
    """Generate a video clip using MiniMax H3.

    Args:
        prompt: Scene description.
        duration: Duration in seconds (4-15).
        aspect_ratio: Output aspect ratio.
        seed: Random seed (-1 = random).
        audio_enabled: Generate native audio.

    Returns:
        MP4 bytes.
    """
    import random  # noqa: PLC0415

    pipe = _load_pipeline()

    duration = max(4, min(15, duration))
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)

    logger.info(
        "Generating: prompt='%s...' duration=%ds seed=%d",
        prompt[:60], duration, seed,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Map aspect ratio to resolution (shorter side = 768)
    ar_map = {
        "16:9": (1360, 768),
        "9:16": (768, 1360),
        "4:3": (1024, 768),
        "3:4": (768, 1024),
        "1:1": (768, 768),
    }
    width, height = ar_map.get(aspect_ratio, (1360, 768))

    # Number of frames at 24fps
    num_frames = duration * 24

    with torch.inference_mode():
        result = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            generator=generator,
        )

    # Export to MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    try:
        # diffusers typically returns frames as a list or tensor
        if hasattr(result, "frames"):
            frames = result.frames
        elif hasattr(result, "videos"):
            frames = result.videos
        else:
            frames = result[0] if isinstance(result, (list, tuple)) else result

        # Export using diffusers export utility or imageio
        try:
            from diffusers.utils import export_to_video  # noqa: PLC0415
            export_to_video(frames[0] if len(frames) > 0 and hasattr(frames[0], '__len__') else frames, tmp_path, fps=24)
        except Exception:
            import imageio  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            if torch.is_tensor(frames):
                frames = frames.cpu().numpy()
            if hasattr(frames, 'shape') and len(frames.shape) == 5:
                frames = frames[0]  # batch dim
            # Convert to uint8 if float
            if hasattr(frames, 'dtype') and frames.dtype != np.uint8:
                frames = (frames * 255).clip(0, 255).astype(np.uint8)
            writer = imageio.get_writer(tmp_path, fps=24, codec="libx264")
            for frame in frames:
                writer.append_data(frame)
            writer.close()

        mp4_bytes = open(tmp_path, "rb").read()
        logger.info("Generated video: %d bytes (%.1f MB)", len(mp4_bytes), len(mp4_bytes) / 1e6)

    finally:
        os.unlink(tmp_path)
        gc.collect()
        torch.cuda.empty_cache()

    return mp4_bytes


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """RunPod serverless handler.

    Input:
    {
        "prompt": "A cinematic scene...",
        "duration": 10,
        "aspect_ratio": "16:9",
        "seed": -1,
        "audio_enabled": false
    }

    Output:
    {
        "mp4_b64": "<base64 MP4>"
    }
    """
    inputs = job.get("input", {})
    prompt = inputs.get("prompt", "")
    if not prompt:
        raise ValueError("'prompt' is required")

    mp4_bytes = _generate_video(
        prompt=prompt,
        duration=int(inputs.get("duration", 10)),
        aspect_ratio=inputs.get("aspect_ratio", "16:9"),
        seed=int(inputs.get("seed", -1)),
        audio_enabled=bool(inputs.get("audio_enabled", False)),
    )

    return {"mp4_b64": base64.b64encode(mp4_bytes).decode("utf-8")}


if __name__ == "__main__":
    logger.info("MiniMax H3 worker starting — model: %s", _MODEL_PATH)
    runpod.serverless.start({"handler": handler})
