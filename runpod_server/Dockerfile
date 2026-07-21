# LTX-Video RunPod Serverless worker image
#
# Build:
#   docker build -t yourusername/ltx-video-serverless:latest ./runpod_server
#
# Push:
#   docker push yourusername/ltx-video-serverless:latest
#
# NOTE: Model weights (~30GB total) are NOT baked into this image.
#       They are loaded from a RunPod Network Volume mounted at /workspace.
#       Run setup_volume.sh once on a temporary pod to populate the volume.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        git wget curl ffmpeg libgl1 libglib2.0-0 \
        python3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# LTX-Video repo (code only, no weights)
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/Lightricks/LTX-Video.git /workspace/LTX-Video && \
    cd /workspace/LTX-Video && \
    pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Apply inference patches at build time
# (disables prompt enhancer + fixes generator device mismatch)
# ---------------------------------------------------------------------------
RUN python3 - << 'PYEOF'
import pathlib

src = pathlib.Path("/workspace/LTX-Video/ltx_video/inference.py").read_text()

# Disable enhance_prompt
src = src.replace(
    "enhance_prompt = (",
    "enhance_prompt = False  # patched: disabled\nif False and (",
)
# Fix generator device mismatch for CPU offload
src = src.replace(
    'generator = torch.Generator(device=device).manual_seed(config.seed)',
    'generator = torch.Generator(device="cpu").manual_seed(config.seed)',
)

pathlib.Path("/workspace/LTX-Video/ltx_video/inference.py").write_text(src)
print("inference.py patched OK")
PYEOF

# ---------------------------------------------------------------------------
# Copy handler
# ---------------------------------------------------------------------------
COPY server.py /workspace/server.py

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/workspace/LTX-Video
ENV HF_HOME=/workspace/huggingface
ENV HUGGINGFACE_HUB_CACHE=/workspace/huggingface
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Config path written by setup_volume.sh and present on the network volume
ENV LTX_CONFIG_PATH=/workspace/ltxv-13b-0.9.8-dev-local.yaml

# ---------------------------------------------------------------------------
# Entry point — RunPod calls server.py which starts runpod.serverless
# ---------------------------------------------------------------------------
CMD ["python3", "/workspace/server.py"]
