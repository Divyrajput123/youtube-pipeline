# LTX-Video RunPod Serverless Worker

Runs LTX-Video 13B as a RunPod Serverless endpoint. Workers spin up on demand
and terminate when idle — no persistent pod to manage or pay for between runs.

---

## Prerequisites

- Docker installed locally (for building the image)
- Docker Hub account (or any container registry)
- RunPod account with a Network Volume

---

## Step 1 — Create a Network Volume

1. RunPod dashboard → **Storage** → **+ Network Volume**
2. Name: `ltx-video-weights`
3. Size: **50 GB minimum** (weights alone are ~32GB)
4. Region: choose the same region you want the serverless endpoint in
5. Click **Create**

---

## Step 2 — Populate the volume (one-time)

1. Deploy a **temporary pod** with the volume attached:
   - Template: **RunPod PyTorch** (any GPU or CPU-only)
   - Attach your network volume at mount path `/workspace`
   - Click Deploy

2. SSH into the pod and run:
   ```bash
   bash /workspace/setup_volume.sh
   ```
   This downloads ~32GB of model weights and patches the LTX-Video code.
   Takes 15–30 minutes depending on connection speed.

3. **Terminate the temporary pod** when the script finishes.
   Volume contents are preserved permanently.

---

## Step 3 — Build and push the Docker image

```bash
# from the repo root
docker build -t yourusername/ltx-video-serverless:latest ./runpod_server
docker push yourusername/ltx-video-serverless:latest
```

Replace `yourusername` with your Docker Hub username.

---

## Step 4 — Create the Serverless Endpoint

1. RunPod dashboard → **Serverless** → **+ New Endpoint**
2. Select **Custom** (not a template)
3. Settings:

   | Field | Value |
   |---|---|
   | Container image | `yourusername/ltx-video-serverless:latest` |
   | Network volume | attach `ltx-video-weights` at `/workspace` |
   | GPU | RTX 4090 or A100 (24GB+ VRAM required) |
   | Min workers | `0` (scales to zero — no idle cost) |
   | Max workers | `1` |
   | Idle timeout | `5` seconds |
   | Container disk | `10 GB` (image + tmp space) |

4. Click **Deploy**
5. Copy the **Endpoint ID** (looks like `abc123xyz`)

---

## Step 5 — Add GitHub Secrets

In your GitHub repo → **Settings** → **Secrets and variables** → **Actions**:

| Secret | Value |
|---|---|
| `RUNPOD_ENDPOINT_ID` | Endpoint ID from Step 4 |
| `RUNPOD_API_KEY` | RunPod account API key (Settings → API Keys) |

---

## Step 6 — Switch to RunPod in config.json

```json
{
  "visual_video_provider": "runpod"
}
```

---

## Testing the endpoint manually

```bash
curl -s -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Thor warrior lightning hammer storm epic cinematic",
      "num_frames": 49,
      "width": 512,
      "height": 352
    }
  }' | python3 -m json.tool
```

Poll the returned `id` for completion:
```bash
curl -s "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/status/JOB_ID" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" | python3 -m json.tool
```

When `status` is `COMPLETED`, the `output.mp4_b64` field contains the
base64-encoded MP4. Decode it:
```bash
# using Python
python3 -c "
import base64, json, sys
d = json.load(sys.stdin)
open('test.mp4','wb').write(base64.b64decode(d['output']['mp4_b64']))
print('Saved test.mp4')
"
```

---

## Cost estimate

- RTX 4090 serverless: ~\$0.00069/second
- 5-second clip: ~30–60s generation = **~\$0.02–\$0.04 per clip**
- 20 clips per video: **~\$0.40–\$0.80 per video**
- No idle charges between runs (Min workers = 0)
