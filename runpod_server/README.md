# LTX-Video RunPod Server

## Deploy on RunPod

1. Go to [runpod.io](https://runpod.io) → Pods → Deploy
2. Choose template: **RunPod PyTorch** (CUDA 12.1)
3. GPU: **RTX 4090** (24GB VRAM) — $0.69/hr
4. Click **Deploy**

## Setup on the pod

SSH into the pod, then run:

```bash
# Install dependencies
pip install fastapi uvicorn diffusers accelerate transformers imageio[ffmpeg] torch

# Upload server.py (or clone your repo)
# Then start the server:
python server.py
```

## Get your pod URL

In RunPod dashboard → your pod → **Connect** → copy the **HTTP port 8000** URL.
It looks like: `https://abc123-8000.proxy.runpod.net`

## Configure the pipeline

Add to your `.env`:
```
RUNPOD_SERVER_URL=https://abc123-8000.proxy.runpod.net
```

## Test it

```bash
curl -X POST https://abc123-8000.proxy.runpod.net/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Thor warrior lightning hammer storm epic cinematic"}' \
  --output test_clip.mp4

open test_clip.mp4
```

## Cost estimate

- RTX 4090: $0.69/hr
- 5-second clip: ~1-2 min generation
- 20 clips per video: ~30-40 min = **~$0.40 per video**

Remember to **stop the pod** when not generating to avoid charges.
