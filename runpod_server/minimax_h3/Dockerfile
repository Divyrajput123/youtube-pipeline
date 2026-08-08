# MiniMax H3 RunPod Serverless worker
#
# Uses RunPod's PyTorch base image (has CUDA + PyTorch pre-installed)
# Model weights are loaded from a Network Volume at /runpod-volume
#
# Build & push:
#   docker buildx build --platform linux/amd64 -t yourusername/minimax-h3-serverless:latest --push .

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# System deps
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# Python deps — diffusers + H3 requirements
RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.33 \
    transformers>=4.45 \
    accelerate>=0.34 \
    torch>=2.8 \
    httpx \
    sentencepiece \
    protobuf \
    soundfile \
    imageio[ffmpeg]

# Copy handler
COPY server.py /opt/server.py

# Environment
ENV VOLUME_ROOT=/runpod-volume
ENV H3_MODEL_PATH=/runpod-volume/MiniMax-H3
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["python3", "/opt/server.py"]
