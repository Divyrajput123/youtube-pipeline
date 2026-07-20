"""LTX-Video RunPod Serverless handler.

Loads the LTX-Video pipeline ONCE per worker lifetime (model stays cached in
memory across multiple jobs on the same worker).  RunPod manages worker
lifecycle — workers spin up on demand and terminate when idle, so there is no
persistent pod to start, stop, or pay for between runs.

Deploy
------
1. Build and push a Docker image containing this file and the LTX-Video repo.
2. In the RunPod dashboard create a Serverless endpoint pointing to that image.
3. Copy the endpoint ID into RUNPOD_ENDPOINT_ID in your .env / GitHub Secrets.
4. Set RUNPOD_API_KEY in your .env / GitHub Secrets.

The pipeline client calls:
    POST https://api.runpod.ai/v2/{endpoint_id}/run
    GET  https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
"""

from __future__ import annotations

import base64
import gc
import logging
import os
import pathlib
import sys
import tempfile

import torch
import runpod

# ---------------------------------------------------------------------------
# Path / env
# ---------------------------------------------------------------------------
_LTX_REPO = "/workspace/LTX-Video"
if _LTX_REPO not in sys.path:
    sys.path.insert(0, _LTX_REPO)

os.environ.setdefault("HF_HOME", "/workspace/huggingface")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/huggingface")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_CONFIG_PATH = os.environ.get(
    "LTX_CONFIG_PATH",
    "/tmp/ltxv-13b-0.9.8-dev-local.yaml",
)

# ---------------------------------------------------------------------------
# Global pipeline cache (persists across jobs on the same worker)
# ---------------------------------------------------------------------------
_pipeline = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Monkey-patch: cache pipeline + disable prompt enhancer
# ---------------------------------------------------------------------------

def _apply_patches() -> None:
    """Patch create_ltx_video_pipeline to cache after first load and disable
    the prompt enhancer (avoids downloading Florence-2 + Llama)."""
    import ltx_video.inference as _inf  # noqa: PLC0415

    _orig_create = _inf.create_ltx_video_pipeline

    def _cached_create(*args, **kwargs):
        global _pipeline
        if _pipeline is not None:
            logger.info("Returning cached pipeline (skipping reload)")
            return _pipeline

        logger.info("create_ltx_video_pipeline: loading for the first time…")

        for a in args:
            if isinstance(a, dict):
                a["prompt_enhancement_words_threshold"] = 99999
                a["prompt_enhancer_image_caption_model_name_or_path"] = None
                a["prompt_enhancer_llm_model_name_or_path"] = None

        kwargs.setdefault("prompt_enhancer_image_caption_model_name_or_path", None)
        kwargs.setdefault("prompt_enhancer_llm_model_name_or_path", None)

        result = _orig_create(*args, **kwargs)
        _pipeline = result
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        logger.info("Pipeline cached — VRAM: %.1f GB used", vram)
        return result

    _inf.create_ltx_video_pipeline = _cached_create
    logger.info("Pipeline caching patch applied")


def _patch_hf() -> None:
    """Patch hf_hub_download to use a locally cached spatial upscaler."""
    import huggingface_hub.file_download as _fd  # noqa: PLC0415
    import ltx_video.inference as _ltx_inf  # noqa: PLC0415

    _orig = _fd.hf_hub_download

    def _patched(repo_id=None, filename=None, *args, **kwargs):
        if filename and "spatial-upscaler" in filename:
            local = f"/tmp/{filename}"
            if os.path.isfile(local):
                logger.info("Using cached spatial upscaler: %s", local)
                return local
        return _orig(repo_id=repo_id, filename=filename, *args, **kwargs)

    _fd.hf_hub_download = _patched
    _ltx_inf.hf_hub_download = _patched
    logger.info("hf_hub_download patched for spatial upscaler")


_apply_patches()
_patch_hf()

from ltx_video.inference import infer, InferenceConfig  # noqa: E402, PLC0415


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

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
    seed: int = 42,
) -> bytes:
    """Run LTX-Video inference and return raw MP4 bytes."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Generating '%s…' (%d frames %dx%d)",
        prompt[:60], num_frames, width, height,
    )

    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            config = InferenceConfig(
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
            infer(config=config)

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
            logger.info(
                "VRAM after inference: %.1f GB / %.1f GB",
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )


# ---------------------------------------------------------------------------
# RunPod Serverless handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """RunPod calls this function for every submitted job.

    Expected input fields (all optional except ``prompt``):
        prompt          str   — scene description
        negative_prompt str   — things to avoid
        height          int   — frame height in pixels (default 512)
        width           int   — frame width in pixels (default 768)
        num_frames      int   — total frames to generate (default 97)
        frame_rate      int   — output FPS (default 25)
        seed            int   — RNG seed (default 42)

    Returns:
        {"mp4_b64": "<base64-encoded MP4 bytes>"}
        or raises an exception which RunPod converts to an error response.
    """
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
        seed=int(inputs.get("seed", 42)),
    )

    return {"mp4_b64": base64.b64encode(mp4_bytes).decode("utf-8")}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting RunPod Serverless worker — config: %s", _CONFIG_PATH)
    runpod.serverless.start({"handler": handler})
