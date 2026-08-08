"""RunPod Pod Manager — start, stop, and query MiniMax H3 ComfyUI pods.

Provides utility functions to manage the lifecycle of a RunPod GPU Pod
running ComfyUI with MiniMax H3. The pod ID is stored in env var
RUNPOD_H3_POD_ID. The pod is started before batch generation and stopped
after to avoid idle charges.

Usage:
    # Start the pod and wait for ComfyUI to be ready
    python pod_manager.py start

    # Check if the pod is running and ComfyUI is responsive
    python pod_manager.py status

    # Stop the pod (saves money when not generating)
    python pod_manager.py stop

Environment variables:
    RUNPOD_API_KEY      - Your RunPod API key
    RUNPOD_H3_POD_ID   - The Pod ID for the MiniMax H3 ComfyUI pod
"""

from __future__ import annotations

import logging
import os
import sys
import time

import httpx

logger = logging.getLogger(__name__)

_RUNPOD_API_BASE = "https://api.runpod.io/graphql"
_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
_POD_ID = os.environ.get("RUNPOD_H3_POD_ID", "")

# Timeout for waiting for pod to be ready (10 minutes)
_START_TIMEOUT_S = 600
_POLL_INTERVAL_S = 10


def _graphql(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against the RunPod API."""
    headers = {"Authorization": f"Bearer {_API_KEY}"}
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    resp = httpx.post(_RUNPOD_API_BASE, json=payload, headers=headers, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    if "errors" in data:
        raise RuntimeError(f"RunPod API error: {data['errors']}")
    return data.get("data", {})


def get_pod_status() -> dict:
    """Get current pod status and runtime info."""
    query = """
    query Pod($podId: String!) {
        pod(input: { podId: $podId }) {
            id
            name
            desiredStatus
            runtime {
                uptimeInSeconds
                ports {
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
        }
    }
    """
    result = _graphql(query, {"podId": _POD_ID})
    return result.get("pod", {})


def get_pod_url() -> str | None:
    """Get the public URL for the ComfyUI API on the running pod.

    RunPod exposes pods via proxy URLs like:
        https://{POD_ID}-{PORT}.proxy.runpod.net

    ComfyUI runs on port 8188 by default.
    """
    pod = get_pod_status()
    if not pod or pod.get("desiredStatus") != "RUNNING":
        return None

    # RunPod proxy URL format
    return f"https://{_POD_ID}-8188.proxy.runpod.net"


def start_pod() -> str:
    """Start the pod and wait for ComfyUI to be ready.

    Retries with exponential backoff if GPU is unavailable (up to 30 minutes).
    This handles the common case where community cloud GPUs are temporarily
    out of capacity but become available within a few minutes.

    Returns:
        The ComfyUI API URL once ready.

    Raises:
        RuntimeError: If the pod fails to start after all retries.
    """
    logger.info("Starting pod %s...", _POD_ID)

    query = """
    mutation ResumePod($podId: String!) {
        podResume(input: { podId: $podId }) {
            id
            desiredStatus
        }
    }
    """

    # Retry resume if GPU not available (up to 30 minutes with backoff)
    max_resume_attempts = 10
    resume_delay = 60  # Start with 1 minute between retries

    for attempt in range(1, max_resume_attempts + 1):
        try:
            result = _graphql(query, {"podId": _POD_ID})
            logger.info("Pod resume requested (attempt %d): %s", attempt, result)
            break  # Success — move on to wait for ready
        except RuntimeError as exc:
            error_msg = str(exc)
            if "not enough free GPUs" in error_msg.lower() or "capacity" in error_msg.lower():
                if attempt < max_resume_attempts:
                    wait_time = min(resume_delay * (1.5 ** (attempt - 1)), 300)  # Max 5 min
                    logger.warning(
                        "GPU not available (attempt %d/%d). Retrying in %.0fs...",
                        attempt, max_resume_attempts, wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(
                        f"GPU not available after {max_resume_attempts} attempts (~30 min). "
                        f"Try again later or check RunPod dashboard for availability. "
                        f"Last error: {error_msg}"
                    ) from exc
            else:
                raise  # Non-capacity error — don't retry

    # Wait for pod to be running and ComfyUI to respond
    comfyui_url = f"https://{_POD_ID}-8188.proxy.runpod.net"
    start_time = time.time()

    while time.time() - start_time < _START_TIMEOUT_S:
        time.sleep(_POLL_INTERVAL_S)

        # Check pod status
        pod = get_pod_status()
        status = pod.get("desiredStatus", "UNKNOWN")
        logger.info("Pod status: %s (%.0fs elapsed)", status, time.time() - start_time)

        if status != "RUNNING":
            continue

        # Pod is running — check if ComfyUI is responding
        try:
            resp = httpx.get(f"{comfyui_url}/system_stats", timeout=10.0)
            if resp.status_code == 200:
                logger.info("ComfyUI is ready at %s", comfyui_url)
                return comfyui_url
        except Exception:
            pass  # Not ready yet

    raise RuntimeError(
        f"Pod {_POD_ID} failed to start within {_START_TIMEOUT_S}s"
    )


def stop_pod() -> None:
    """Stop the pod to save money."""
    logger.info("Stopping pod %s...", _POD_ID)

    query = """
    mutation StopPod($podId: String!) {
        podStop(input: { podId: $podId }) {
            id
            desiredStatus
        }
    }
    """
    result = _graphql(query, {"podId": _POD_ID})
    logger.info("Pod stop requested: %s", result)


def is_ready() -> bool:
    """Check if the pod is running and ComfyUI is responsive."""
    url = get_pod_url()
    if not url:
        return False
    try:
        resp = httpx.get(f"{url}/system_stats", timeout=10.0)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not _API_KEY:
        print("ERROR: RUNPOD_API_KEY not set")
        sys.exit(1)
    if not _POD_ID:
        print("ERROR: RUNPOD_H3_POD_ID not set")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python pod_manager.py [start|stop|status]")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "start":
        url = start_pod()
        print(f"ComfyUI ready at: {url}")
    elif cmd == "stop":
        stop_pod()
        print("Pod stopped.")
    elif cmd == "status":
        pod = get_pod_status()
        print(f"Pod: {pod.get('name', 'unknown')}")
        print(f"Status: {pod.get('desiredStatus', 'unknown')}")
        print(f"URL: {get_pod_url() or 'not running'}")
        print(f"Ready: {is_ready()}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
