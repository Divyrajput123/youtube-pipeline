#!/bin/bash
# =============================================================================
# setup_volume.sh — One-time setup script for the RunPod Network Volume
#
# Run this ONCE on a temporary RunPod pod that has the network volume mounted
# at /workspace. After it completes, terminate the pod. The serverless workers
# will reuse everything stored on the volume on every subsequent cold start.
#
# Steps to run this:
#   1. In RunPod dashboard → Storage → create a Network Volume (50GB minimum)
#      in the same region you plan to deploy the serverless endpoint.
#   2. Deploy a temporary pod (any GPU, even CPU works for downloading):
#        Template: RunPod PyTorch
#        Attach the network volume at mount path: /workspace
#   3. SSH into the pod and run:
#        bash /workspace/setup_volume.sh
#   4. Wait for completion (~15-30 min depending on connection speed)
#   5. Terminate the temporary pod — volume contents are preserved.
#   6. Create the serverless endpoint and attach the same volume.
# =============================================================================

set -euo pipefail

WORKSPACE=/workspace
HF_CACHE="$WORKSPACE/huggingface"

echo "======================================================"
echo " LTX-Video Network Volume Setup"
echo " Workspace: $WORKSPACE"
echo "======================================================"

# ---------------------------------------------------------------------------
# 1. Install Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[1/5] Installing Python dependencies..."
pip install --quiet runpod diffusers accelerate transformers \
    "imageio[ffmpeg]" torch huggingface_hub pyyaml
echo "  Done."

# ---------------------------------------------------------------------------
# 2. Clone LTX-Video repo
# ---------------------------------------------------------------------------
echo ""
echo "[2/5] Cloning LTX-Video repository..."
if [ -d "$WORKSPACE/LTX-Video" ]; then
    echo "  Already cloned — pulling latest..."
    git -C "$WORKSPACE/LTX-Video" pull --quiet
else
    git clone --depth 1 https://github.com/Lightricks/LTX-Video.git \
        "$WORKSPACE/LTX-Video"
fi
cd "$WORKSPACE/LTX-Video"
pip install --quiet -e .
echo "  Done."

# ---------------------------------------------------------------------------
# 3. Download model weights
# ---------------------------------------------------------------------------
echo ""
echo "[3/5] Downloading LTX-Video 13B model weights..."
mkdir -p "$HF_CACHE"

# 13B main checkpoint (~12GB)
CKPT="$WORKSPACE/ltxv-13b-0.9.8-dev.safetensors"
if [ ! -f "$CKPT" ]; then
    echo "  Downloading 13B checkpoint (~12GB)..."
    wget -q --show-progress \
        "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-dev.safetensors" \
        -O "$CKPT"
    echo "  13B checkpoint downloaded ($(du -sh "$CKPT" | cut -f1))"
else
    echo "  13B checkpoint already present ($(du -sh "$CKPT" | cut -f1))"
fi

# Spatial upscaler (~481MB)
UPSCALER="$WORKSPACE/ltxv-spatial-upscaler-0.9.8.safetensors"
if [ ! -f "$UPSCALER" ]; then
    echo "  Downloading spatial upscaler (~481MB)..."
    wget -q --show-progress \
        "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-spatial-upscaler-0.9.8.safetensors" \
        -O "$UPSCALER"
    echo "  Spatial upscaler downloaded"
else
    echo "  Spatial upscaler already present"
fi

# PixArt-XL text encoder (~18GB) — used by LTX-Video pipeline
echo "  Downloading PixArt-XL text encoder (~18GB, this takes a while)..."
python3 - << 'PYEOF'
import os
from huggingface_hub import snapshot_download
cache_dir = os.environ.get("HF_HOME", "/workspace/huggingface")
if not os.path.isdir(f"{cache_dir}/models--PixArt-alpha--PixArt-XL-2-1024-MS"):
    print("  Downloading PixArt-XL-2-1024-MS...")
    snapshot_download(
        "PixArt-alpha/PixArt-XL-2-1024-MS",
        cache_dir=cache_dir,
    )
    print("  PixArt text encoder downloaded")
else:
    print("  PixArt text encoder already cached")
PYEOF

echo "  All weights downloaded."

# ---------------------------------------------------------------------------
# 4. Write local YAML config pointing to volume paths
# ---------------------------------------------------------------------------
echo ""
echo "[4/5] Writing local model config..."
python3 - << 'PYEOF'
import yaml, pathlib

workspace = "/workspace"
base_cfg = pathlib.Path(f"{workspace}/LTX-Video/configs/ltxv-13b-0.9.8-dev.yaml")

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg["checkpoint_path"] = f"{workspace}/ltxv-13b-0.9.8-dev.safetensors"
cfg["spatial_upscaler_model_path"] = f"{workspace}/ltxv-spatial-upscaler-0.9.8.safetensors"

out = pathlib.Path(f"{workspace}/ltxv-13b-0.9.8-dev-local.yaml")
with open(out, "w") as f:
    yaml.dump(cfg, f)

print(f"  Config written to: {out}")
print(f"  checkpoint_path: {cfg['checkpoint_path']}")
print(f"  spatial_upscaler_model_path: {cfg['spatial_upscaler_model_path']}")
PYEOF

# ---------------------------------------------------------------------------
# 5. Patch inference.py in the volume copy
# ---------------------------------------------------------------------------
echo ""
echo "[5/5] Patching inference.py..."
python3 - << 'PYEOF'
import pathlib

inf = pathlib.Path("/workspace/LTX-Video/ltx_video/inference.py")
src = inf.read_text()

# Disable enhance_prompt
if "patched: disabled" not in src:
    src = src.replace(
        "enhance_prompt = (",
        "enhance_prompt = False  # patched: disabled\nif False and (",
    )
    # Fix generator device mismatch
    src = src.replace(
        'generator = torch.Generator(device=device).manual_seed(config.seed)',
        'generator = torch.Generator(device="cpu").manual_seed(config.seed)',
    )
    inf.write_text(src)
    print("  inference.py patched OK")
else:
    print("  inference.py already patched")
PYEOF

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo " Setup complete! Volume contents:"
echo "======================================================"
du -sh "$WORKSPACE"/* 2>/dev/null || true
echo ""
echo " Next steps:"
echo "   1. Terminate this temporary pod."
echo "   2. In RunPod Serverless dashboard, create a new endpoint:"
echo "      - Container image: yourusername/ltx-video-serverless:latest"
echo "      - Attach this network volume at mount path: /workspace"
echo "      - GPU: RTX 4090 or A100 (24GB+ VRAM)"
echo "      - Min workers: 0  (scales to zero when idle)"
echo "      - Max workers: 1"
echo "      - Idle timeout: 5 seconds"
echo "   3. Copy the endpoint ID → add as GitHub Secret RUNPOD_ENDPOINT_ID"
echo "======================================================"
