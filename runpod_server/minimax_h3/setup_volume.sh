#!/bin/bash
# =============================================================================
# setup_volume.sh — One-time setup for MiniMax H3 on RunPod Network Volume
#
# Run this ONCE on a temporary RunPod pod that has the network volume mounted.
# After completion, terminate the pod and create the serverless endpoint.
#
# Steps:
#   1. Create a Network Volume (100GB minimum) in your RunPod dashboard.
#      IMPORTANT: Choose a data center OUTSIDE US/EU/UK/South Korea
#      (MiniMax H3 license restriction). Asia/Mumbai or similar is fine.
#   2. Deploy a temporary pod (any GPU, even CPU works for downloading):
#        Template: RunPod PyTorch
#        Attach the network volume at mount path: /workspace
#   3. SSH into the pod and run:
#        bash /workspace/setup_volume.sh
#   4. Wait for completion (~30-60 min depending on connection speed)
#   5. Terminate the temporary pod — volume contents are preserved.
#   6. Create the serverless endpoint:
#      - Container image: yourusername/minimax-h3-serverless:latest
#      - Attach this network volume at mount path: /runpod-volume
#      - GPU: 2x A100 80GB (recommended) or 4x A6000 48GB
#      - Min workers: 0  (scales to zero when idle)
#      - Max workers: 1
#      - Idle timeout: 30 seconds
# =============================================================================

set -euo pipefail

WORKSPACE="${VOLUME_ROOT:-/workspace}"

echo "======================================================"
echo " MiniMax H3 Network Volume Setup"
echo " Workspace: $WORKSPACE"
echo "======================================================"

# ---------------------------------------------------------------------------
# 1. Install Python dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] Installing Python dependencies..."
pip install --quiet runpod httpx huggingface_hub torch

# Install SGLang (the recommended inference framework for H3)
pip install --quiet "sglang[all]>=0.4"

echo "  Done."

# ---------------------------------------------------------------------------
# 2. Authenticate with Hugging Face (required for faster downloads)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] Hugging Face authentication..."
if [ -z "${HF_TOKEN:-}" ]; then
    echo "  WARNING: HF_TOKEN not set. Downloads will be slower."
    echo "  Set HF_TOKEN env var with your Hugging Face access token for faster downloads."
else
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    echo "  Authenticated with Hugging Face."
fi

# ---------------------------------------------------------------------------
# 3. Download MiniMax H3 model weights (FL2VA — text/image to video)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Downloading MiniMax H3 FL2VA model weights (~42GB)..."
echo "  This will take 30-60 minutes depending on connection speed."

MODEL_DIR="$WORKSPACE/MiniMax-H3"
mkdir -p "$MODEL_DIR"

# Download FL2VA checkpoint (text-to-video + first/last frame to video)
# This includes: processor, tokenizer, text_encoder, transformer, visual_vae, audio_vae
huggingface-cli download MiniMaxAI/MiniMax-H3 \
    --include "model_index.json" "FL2VA/*" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "  Model download complete!"
echo "  Size: $(du -sh "$MODEL_DIR" | cut -f1)"

# ---------------------------------------------------------------------------
# 4. Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Verifying installation..."

# Check critical files exist
if [ -f "$MODEL_DIR/FL2VA/model_index.json" ]; then
    echo "  FL2VA model_index.json: OK"
else
    echo "  ERROR: FL2VA/model_index.json not found!"
    exit 1
fi

if [ -d "$MODEL_DIR/FL2VA/transformer" ]; then
    echo "  FL2VA transformer weights: OK"
else
    echo "  ERROR: FL2VA/transformer not found!"
    exit 1
fi

if [ -d "$MODEL_DIR/FL2VA/text_encoder" ]; then
    echo "  FL2VA text_encoder (Qwen3-VL-32B): OK"
else
    echo "  ERROR: FL2VA/text_encoder not found!"
    exit 1
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "======================================================"
echo " Setup complete! Volume contents:"
echo "======================================================"
du -sh "$WORKSPACE"/* 2>/dev/null || true
echo ""
echo " Model specs:"
echo "   - MiniMax H3 FL2VA (text/image to video)"
echo "   - 33B params (dense), BF16 precision"
echo "   - Output: 768p, 4-15 seconds, 24fps, native stereo audio"
echo "   - Aspect ratios: 16:9, 9:16, 1:1, 4:3, 3:4"
echo ""
echo " Next steps:"
echo "   1. Terminate this temporary pod."
echo "   2. Build and push the Docker image:"
echo "      docker build -t yourusername/minimax-h3-serverless:latest ./runpod_server/minimax_h3"
echo "      docker push yourusername/minimax-h3-serverless:latest"
echo "   3. Create a serverless endpoint in RunPod:"
echo "      - Container image: yourusername/minimax-h3-serverless:latest"
echo "      - Attach this network volume at: /runpod-volume"
echo "      - GPU: 2x A100 80GB"
echo "      - Min workers: 0"
echo "      - Max workers: 1"
echo "      - Idle timeout: 30 seconds"
echo "   4. Copy endpoint ID → set RUNPOD_H3_ENDPOINT_ID in .env / GitHub Secrets"
echo "======================================================"
