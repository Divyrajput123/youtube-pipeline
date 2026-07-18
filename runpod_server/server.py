"""LTX-Video generation server — pipeline cached at startup.

Loads the LTX-Video pipeline ONCE at startup and reuses it for all requests.
This means first startup takes 2-3 minutes (model loading) but each subsequent
clip generates in ~20-30s on H100.

Run inside the venv:
    cd /workspace/LTX-Video
    source .venv/bin/activate
    HF_HOME=/workspace/huggingface PORT=9000 python /workspace/server_new.py

API:
    POST /jobs      — submit async generation job, returns job_id immediately
    GET  /jobs/{id} — poll job status, returns MP4 bytes when done
    POST /generate  — synchronous (blocks until done, hits proxy timeout for long jobs)
    GET  /health    — liveness + VRAM stats
"""

from __future__ import annotations

import gc
import logging
import os
import pathlib
import sys
import tempfile

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

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

# Config path — set by start.sh with local checkpoint paths
_CONFIG_PATH = os.environ.get(
    "LTX_CONFIG_PATH",
    "/tmp/ltxv-13b-0.9.8-dev-local.yaml"
)

# ---------------------------------------------------------------------------
# Global pipeline cache
# ---------------------------------------------------------------------------
_pipeline = None          # cached pipeline dict
_device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Monkey-patch: cache pipeline + keep text encoder on CPU
# ---------------------------------------------------------------------------

def _apply_patches():
    """Patch create_ltx_video_pipeline to:
    1. Cache the pipeline after first load (avoids reloading on every request)
    2. Keep text encoder on CPU (saves VRAM — though H100 80GB doesn't need it)
    3. Disable prompt enhancer (avoids downloading Florence-2 + Llama)
    """
    import ltx_video.inference as _inf  # noqa: PLC0415
    _orig_create = _inf.create_ltx_video_pipeline

    def _cached_create(*args, **kwargs):
        global _pipeline
        if _pipeline is not None:
            logger.info("Returning cached pipeline (skipping reload)")
            return _pipeline

        logger.info("create_ltx_video_pipeline: loading for the first time...")

        # Disable prompt enhancer
        for i, a in enumerate(args):
            if isinstance(a, dict):
                a["prompt_enhancement_words_threshold"] = 99999
                a["prompt_enhancer_image_caption_model_name_or_path"] = None
                a["prompt_enhancer_llm_model_name_or_path"] = None
        if "prompt_enhancer_image_caption_model_name_or_path" in kwargs:
            kwargs["prompt_enhancer_image_caption_model_name_or_path"] = None
            kwargs["prompt_enhancer_llm_model_name_or_path"] = None

        result = _orig_create(*args, **kwargs)
        _pipeline = result
        logger.info("Pipeline cached! VRAM: %.1f GB used",
                    torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
        return result

    _inf.create_ltx_video_pipeline = _cached_create
    logger.info("Pipeline caching patch applied")


def _patch_inference_py():
    """Patch inference.py in memory to disable enhance_prompt and fix generator."""
    import ltx_video.inference as _inf  # noqa: PLC0415
    import inspect, types  # noqa: PLC0415

    src = inspect.getsource(_inf.infer)
    # These are already patched on disk by start.sh — just verify
    logger.info("infer() function loaded OK")


_apply_patches()

# Patch hf_hub_download to use local spatial upscaler
def _patch_hf():
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

_patch_hf()

from ltx_video.inference import infer, InferenceConfig  # noqa: E402, PLC0415


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="LTX-Video Generation Server")
_pipeline_lock = None

# Job store
_jobs: dict = {}

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = (
        "worst quality, inconsistent motion, blurry, jittery, distorted, "
        "watermark, text, static, low quality"
    )
    height: int = 512
    width: int = 768
    num_frames: int = 97
    frame_rate: int = 25
    seed: int = 42


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    import asyncio  # noqa: PLC0415
    global _pipeline_lock
    _pipeline_lock = asyncio.Lock()
    logger.info("Server ready on port %s — pipeline loads on first request (cached after)", 
                os.environ.get("PORT", 8000))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    total = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    return {
        "status": "ok",
        "model": "LTX-Video 13B (pipeline cached)",
        "pipeline_loaded": _pipeline is not None,
        "vram_used_gb": round(used, 2),
        "vram_total_gb": round(total, 2),
        "active_jobs": len([j for j in _jobs.values() if j["status"] == "pending"]),
    }


@app.post("/generate")
async def generate(req: GenerateRequest) -> Response:
    """Synchronous — blocks until done. Use /jobs for async."""
    import asyncio  # noqa: PLC0415
    async with _pipeline_lock:
        loop = asyncio.get_running_loop()
        try:
            mp4_bytes = await loop.run_in_executor(None, _run_inference, req)
        except Exception as exc:
            logger.error("Generation failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return Response(content=mp4_bytes, media_type="video/mp4")


@app.post("/jobs")
async def submit_job(req: GenerateRequest):
    """Async job submission — returns job_id immediately."""
    import asyncio, uuid  # noqa: PLC0415
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "data": None, "error": None}
    logger.info("Job %s submitted", job_id)

    async def _run():
        async with _pipeline_lock:
            loop = asyncio.get_running_loop()
            try:
                data = await loop.run_in_executor(None, _run_inference, req)
                _jobs[job_id]["data"] = data
                _jobs[job_id]["status"] = "done"
                logger.info("Job %s done (%d bytes)", job_id, len(data))
            except Exception as exc:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(exc)
                logger.error("Job %s failed: %s", job_id, exc, exc_info=True)

    asyncio.create_task(_run())
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _jobs[job_id]
    if job["status"] == "pending":
        return {"job_id": job_id, "status": "pending"}
    if job["status"] == "error":
        _jobs.pop(job_id)
        raise HTTPException(status_code=500, detail=job["error"])
    data = job["data"]
    _jobs.pop(job_id)
    return Response(content=data, media_type="video/mp4")


# ---------------------------------------------------------------------------
# Inference — uses cached pipeline
# ---------------------------------------------------------------------------
def _run_inference(req: GenerateRequest) -> bytes:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Generating '%s…' (%d frames %dx%d)", req.prompt[:60], req.num_frames, req.width, req.height)

    try:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmpdir:
            config = InferenceConfig(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                height=req.height,
                width=req.width,
                num_frames=req.num_frames,
                frame_rate=req.frame_rate,
                seed=req.seed,
                pipeline_config=_CONFIG_PATH,
                offload_to_cpu=False,  # H100 80GB has plenty of VRAM
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
            logger.info("VRAM: %.1f GB used / %.1f GB total",
                        torch.cuda.memory_allocated() / 1e9,
                        torch.cuda.get_device_properties(0).total_memory / 1e9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting on port %d — config: %s", port, _CONFIG_PATH)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
