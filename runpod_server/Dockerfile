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

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

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
# Install to /opt/ltx-video so it's never overwritten by the network volume
# which mounts at /runpod-volume
# ---------------------------------------------------------------------------
RUN git clone --depth 1 https://github.com/Lightricks/LTX-Video.git /opt/ltx-video && \
    cd /opt/ltx-video && \
    pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Apply inference patches at build time
# Only fix the generator device mismatch — safe single-line replacement.
# The prompt enhancer is disabled via config (prompt_enhancement_words_threshold).
# ---------------------------------------------------------------------------
RUN python3 - << 'PYEOF'
import pathlib, py_compile, sys

inf_path = pathlib.Path("/opt/ltx-video/ltx_video/inference.py")
src = inf_path.read_text()

# Fix generator device mismatch for CPU offload
patched = src.replace(
    'generator = torch.Generator(device=device).manual_seed(config.seed)',
    'generator = torch.Generator(device="cpu").manual_seed(config.seed)',
)

if patched == src:
    print("WARNING: generator patch target not found — skipping")
else:
    inf_path.write_text(patched)
    print("Generator device patch applied")

# Verify syntax
try:
    py_compile.compile(str(inf_path), doraise=True)
    print("inference.py syntax OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ---------------------------------------------------------------------------
# Copy handler
# ---------------------------------------------------------------------------
COPY server.py /opt/server.py

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV PYTHONPATH=/opt/ltx-video
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VOLUME_ROOT is set to /runpod-volume where the network volume is mounted
ENV VOLUME_ROOT=/runpod-volume

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
CMD ["python3", "/opt/server.py"]
