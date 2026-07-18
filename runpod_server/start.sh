#!/bin/bash
# LTX-Video server startup script
# Run this after every pod restart:
#   bash /tmp/start.sh

set -e

# Activate venv first
source /workspace/LTX-Video/.venv/bin/activate
echo "Venv activated: $(which python3)"

echo "=== LTX-Video Server Startup ==="

# 1. Patch inference.py (apply every time since /tmp is wiped on restart)
echo "[1/4] Patching inference.py..."
git -C /workspace/LTX-Video show HEAD:ltx_video/inference.py > /tmp/inference_original.py

python3 << 'PYEOF'
with open('/tmp/inference_original.py', 'r') as f:
    lines = f.readlines()

# Disable prompt enhancer (avoids downloading Florence-2 + Llama models)
lines[479] = '    enhance_prompt = False\n'
lines[480] = '\n'
lines[481] = '\n'
lines[482] = '\n'
lines[505] = '        enhance_prompt=False,\n'
lines[587] = '        enhance_prompt=False,\n'

# Fix generator device mismatch (CPU offload requires CPU generator)
lines[566] = '    generator = torch.Generator(device="cpu").manual_seed(config.seed)\n'

with open('/tmp/inference_patched.py', 'w') as f:
    f.writelines(lines)
print(f"  Patched {len(lines)} lines OK")
PYEOF

ln -sf /tmp/inference_patched.py /workspace/LTX-Video/ltx_video/inference.py
python3 -c "import ltx_video.inference; assert hasattr(ltx_video.inference, 'create_ltx_video_pipeline'), 'import failed'" && echo "  Import OK"

# 2. Download 13B model weights and spatial upscaler
echo "[2/4] Checking model weights..."

# 13B checkpoint (~12GB) — download to /workspace to persist across restarts
if [ ! -f /workspace/ltxv-13b-0.9.8-dev.safetensors ]; then
    echo "  Downloading LTX-Video 13B weights (~12GB)..."
    wget -q --show-progress \
        "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-13b-0.9.8-dev.safetensors" \
        -O /workspace/ltxv-13b-0.9.8-dev.safetensors
    echo "  13B weights downloaded OK"
else
    echo "  13B weights already present ($(du -sh /workspace/ltxv-13b-0.9.8-dev.safetensors | cut -f1))"
fi

# Spatial upscaler (~481MB) — keep in /tmp (fast SSD, re-download on restart)
if [ ! -f /tmp/ltxv-spatial-upscaler-0.9.8.safetensors ]; then
    echo "  Downloading spatial upscaler (481MB)..."
    wget -q --show-progress \
        "https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-spatial-upscaler-0.9.8.safetensors" \
        -O /tmp/ltxv-spatial-upscaler-0.9.8.safetensors
    echo "  Spatial upscaler downloaded OK"
else
    echo "  Spatial upscaler already present"
fi

# Patch the 13B config to use local checkpoint path (avoid HF download)
python3 << 'PYEOF'
import yaml, os
config_path = "/workspace/LTX-Video/configs/ltxv-13b-0.9.8-dev.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)
# Point to local file instead of HF download
cfg["checkpoint_path"] = "/workspace/ltxv-13b-0.9.8-dev.safetensors"
cfg["spatial_upscaler_model_path"] = "/tmp/ltxv-spatial-upscaler-0.9.8.safetensors"
with open("/tmp/ltxv-13b-0.9.8-dev-local.yaml", "w") as f:
    yaml.dump(cfg, f)
print(f"  Config patched — checkpoint: {cfg['checkpoint_path']}")
PYEOF

# 3. Set environment
echo "[3/4] Setting environment..."
# Use workspace for HF cache — persistent across restarts, no symlink needed
export HF_HOME=/workspace/huggingface
export HUGGINGFACE_HUB_CACHE=/workspace/huggingface/hub
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PORT=9000
mkdir -p /workspace/huggingface/hub

# Pre-download PixArt text encoder if not already cached
if [ ! -d /workspace/huggingface/hub/models--PixArt-alpha--PixArt-XL-2-1024-MS ]; then
    echo "  Downloading PixArt-XL text encoder (~18GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('PixArt-alpha/PixArt-XL-2-1024-MS', cache_dir='/workspace/huggingface')
print('  PixArt text encoder downloaded OK')
"
else
    echo "  PixArt text encoder already cached"
fi

# 4. Start server in auto-restart loop
echo "[4/4] Starting server on port $PORT (auto-restart loop)..."
pkill -9 python3 2>/dev/null || true
sleep 2

# Auto-restart wrapper — restarts server if it crashes
while true; do
    echo "  [$(date)] Starting server..."
    python3 /tmp/server_new.py >> /tmp/server.log 2>&1
    EXIT_CODE=$?
    echo "  [$(date)] Server exited with code $EXIT_CODE — restarting in 3s..."
    sleep 3
done &
LOOP_PID=$!
echo "  Loop PID: $LOOP_PID"

# Wait for server to be ready
echo "  Waiting for server..."
for i in $(seq 1 30); do
    sleep 2
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "  Server ready on port $PORT!"
        curl -s http://localhost:$PORT/health
        echo ""
        break
    fi
    echo -n "."
done

echo ""
echo "=== Startup complete ==="
echo "Test with:"
echo "  curl -X POST http://localhost:$PORT/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Thor cinematic\", \"num_frames\": 49, \"width\": 512, \"height\": 352}' --output /tmp/test.mp4 -w '\nHTTP %{http_code}\n'"
