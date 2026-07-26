"""LTX-Video RunPod Serverless handler.

All heavy imports (torch, ltx_video) are deferred until the first job arrives.
This ensures the worker starts cleanly even before the volume is mounted and
lets RunPod confirm the handler is registered before model loading begins.
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import pathlib
import sys
import tempfile

import runpod

# ---------------------------------------------------------------------------
# Path / env — set before any ltx_video import
# ---------------------------------------------------------------------------
_VOLUME_ROOT = os.environ.get("VOLUME_ROOT", "/runpod-volume")
_LTX_REPO    = os.environ.get("LTX_REPO", "/opt/ltx-video")

if _LTX_REPO not in sys.path:
    sys.path.insert(0, _LTX_REPO)

os.environ.setdefault("HF_HOME",                f"{_VOLUME_ROOT}/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE",  f"{_VOLUME_ROOT}/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE",      f"{_VOLUME_ROOT}/huggingface")
os.environ.setdefault("HF_DATASETS_CACHE",       f"{_VOLUME_ROOT}/huggingface")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fix yaml config paths at startup — rewrites any stale /workspace/ or
# /runpod-volume/ prefix to the actual VOLUME_ROOT so the server works
# regardless of where the volume was originally set up.
# ---------------------------------------------------------------------------
def _fix_config_yaml() -> str:
    """Return path to a corrected yaml config, rewriting stale volume paths
    and disabling the prompt enhancer."""
    import re
    src_path = pathlib.Path(os.environ.get(
        "LTX_CONFIG_PATH",
        f"{_VOLUME_ROOT}/ltxv-13b-0.9.8-dev-local.yaml",
    ))

    if not src_path.exists():
        raise FileNotFoundError(
            f"LTX config not found at {src_path}. "
            f"Is the network volume mounted at {_VOLUME_ROOT}?"
        )

    text = src_path.read_text()

    # Replace any absolute volume path prefix with current VOLUME_ROOT
    fixed = re.sub(r"(/workspace|/runpod-volume)/", f"{_VOLUME_ROOT}/", text)

    # Write fixed version to /tmp so we don't modify the volume
    dst_path = pathlib.Path("/tmp/ltxv-config-fixed.yaml")
    dst_path.write_text(fixed)
    logger.info("Config written to %s (VOLUME_ROOT=%s)", dst_path, _VOLUME_ROOT)
    return str(dst_path)


_CONFIG_PATH = _fix_config_yaml()

# ---------------------------------------------------------------------------
# Lazy globals — populated on first job
# ---------------------------------------------------------------------------
_pipeline        = None
_infer           = None
_InferenceConfig = None
_patches_done    = False


def _ensure_ready() -> None:
    """Import torch + ltx_video and apply patches on first call."""
    global _pipeline, _infer, _InferenceConfig, _patches_done

    if _patches_done:
        return

    import torch  # noqa: PLC0415

    # ---- patch pipeline caching ----
    import ltx_video.inference as _inf  # noqa: PLC0415
    _orig_create = _inf.create_ltx_video_pipeline

    def _cached_create(*args, **kwargs):
        global _pipeline
        if _pipeline is not None:
            logger.info("Returning cached pipeline")
            return _pipeline
        logger.info("Loading pipeline for the first time…")
        result = _orig_create(*args, **kwargs)
        _pipeline = result
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        logger.info("Pipeline cached — VRAM %.1f GB", vram)
        return result

    _inf.create_ltx_video_pipeline = _cached_create

    # ---- patch spatial upscaler lookup ----
    import huggingface_hub.file_download as _fd  # noqa: PLC0415
    _orig_dl = _fd.hf_hub_download

    def _patched_dl(repo_id=None, filename=None, *args, **kwargs):
        if filename and "spatial-upscaler" in filename:
            local = f"{_VOLUME_ROOT}/{filename}"
            if os.path.isfile(local):
                logger.info("Using cached spatial upscaler: %s", local)
                return local
        return _orig_dl(repo_id=repo_id, filename=filename, *args, **kwargs)

    _fd.hf_hub_download = _patched_dl
    _inf.hf_hub_download = _patched_dl

    from ltx_video.inference import infer, InferenceConfig  # noqa: PLC0415
    _infer = infer
    _InferenceConfig = InferenceConfig

    _patches_done = True
    logger.info("LTX-Video ready — config: %s", _CONFIG_PATH)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

import random as _random

def _run_inference(
    prompt: str,
    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "watermark, text, static, low quality"
    ),
    height: int = 512,
    width: int = 768,
    num_frames: int = 97,
    frame_rate: int = 25,
    seed: int = -1,  # -1 = random
) -> bytes:
    import torch  # noqa: PLC0415

    _ensure_ready()

    # Randomize seed if not explicitly provided
    if seed < 0:
        seed = _random.randint(0, 2**31 - 1)
        logger.info("Using random seed: %d", seed)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Generating '%s…' (%d frames %dx%d)", prompt[:60], num_frames, width, height)

    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            config = _InferenceConfig(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                seed=seed,
                pipeline_config=_CONFIG_PATH,
                offload_to_cpu=False,
                output_path=tmpdir,
            )
            _infer(config=config)

            mp4_files = list(pathlib.Path(tmpdir).glob("*.mp4"))
            if not mp4_files:
                raise RuntimeError(f"No MP4 generated in {tmpdir}")

            data = mp4_files[0].read_bytes()
            logger.info("Done — %d bytes (%.1f MB)", len(data), len(data) / 1e6)
            return data
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# RunPod Serverless handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """RunPod calls this for every submitted job."""
    inputs = job.get("input", {})
    prompt = inputs.get("prompt", "")
    if not prompt:
        raise ValueError("'prompt' is required in job input")

    mp4_bytes = _run_inference(
        prompt=prompt,
        negative_prompt=inputs.get(
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted, "
            "watermark, text, static, low quality",
        ),
        height=int(inputs.get("height", 512)),
        width=int(inputs.get("width", 768)),
        num_frames=int(inputs.get("num_frames", 97)),
        frame_rate=int(inputs.get("frame_rate", 25)),
        seed=int(inputs.get("seed", -1)),  # -1 = random
    )

    return {"mp4_b64": base64.b64encode(mp4_bytes).decode("utf-8")}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("RunPod Serverless worker starting — VOLUME_ROOT: %s", _VOLUME_ROOT)
    runpod.serverless.start({"handler": handler})
