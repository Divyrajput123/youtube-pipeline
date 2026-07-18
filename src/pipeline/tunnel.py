"""Public URL tunnel — makes the review webhook reachable from anywhere.

When you're running the pipeline on your laptop (or any machine behind a router),
the review approve/edit links in emails point to a local IP that's unreachable
from outside your network.

This module auto-starts an ngrok tunnel so the review server gets a public
HTTPS URL that works from anywhere — your phone, another network, etc.

Setup (one-time):
    pip install pyngrok
    ngrok authtoken <your-token>       # free at https://dashboard.ngrok.com
    # OR set NGROK_AUTH_TOKEN in .env

Usage:
    The tunnel is started automatically by the pipeline CLI before the review
    server begins accepting connections.  ``PIPELINE_PUBLIC_URL`` is updated
    in-process so all review tokens use the real public URL.

    You can also disable tunneling by setting:
        NGROK_ENABLED=false   in .env

Configuration (.env):
    NGROK_ENABLED=true           # set to "false" to disable (default: true)
    NGROK_AUTH_TOKEN=<token>     # optional if already configured via ngrok CLI
    REVIEW_SERVER_PORT=8742      # must match the review server port
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def is_tunnel_enabled() -> bool:
    """Return True unless NGROK_ENABLED is explicitly set to 'false'."""
    return os.environ.get("NGROK_ENABLED", "true").lower().strip() not in ("false", "0", "no")


def start_tunnel(port: Optional[int] = None) -> Optional[str]:
    """Start an ngrok tunnel and return the public HTTPS URL.

    Also updates ``os.environ["PIPELINE_PUBLIC_URL"]`` so all downstream
    code (review_server.get_review_urls, etc.) automatically uses the tunnel.

    Args:
        port: Local port to tunnel. Defaults to ``REVIEW_SERVER_PORT`` env var
              or 8742.

    Returns:
        The public HTTPS URL (e.g. ``"https://abc123.ngrok-free.app"``) or
        ``None`` if tunneling is disabled or ngrok is unavailable.
    """
    if not is_tunnel_enabled():
        logger.info("Tunnel disabled via NGROK_ENABLED=false")
        return None

    resolved_port = port or int(os.environ.get("REVIEW_SERVER_PORT", "8742"))

    try:
        from pyngrok import ngrok, conf  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "pyngrok not installed — review links will use PIPELINE_PUBLIC_URL as-is.\n"
            "  Install it:  pip install pyngrok\n"
            "  Then set:    NGROK_AUTH_TOKEN=<your-token> in .env"
        )
        return None

    # Apply auth token if provided in env (takes precedence over ngrok CLI config)
    auth_token = os.environ.get("NGROK_AUTH_TOKEN")
    if auth_token:
        conf.get_default().auth_token = auth_token

    try:
        tunnel = ngrok.connect(resolved_port, "http")
        public_url: str = tunnel.public_url  # type: ignore[attr-defined]

        # ngrok free tier may give http:// — upgrade to https://
        if public_url.startswith("http://"):
            public_url = "https://" + public_url[len("http://"):]

        # Update the env var so review_server.get_review_urls() picks it up
        os.environ["PIPELINE_PUBLIC_URL"] = public_url

        logger.info(
            "ngrok tunnel active: localhost:%d → %s",
            resolved_port,
            public_url,
        )
        print(f"🌐 Review links will use public URL: {public_url}")
        return public_url

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to start ngrok tunnel: %s\n"
            "  Review links will use PIPELINE_PUBLIC_URL=%s (may be unreachable remotely).\n"
            "  To fix: check your NGROK_AUTH_TOKEN and internet connection.",
            exc,
            os.environ.get("PIPELINE_PUBLIC_URL", "http://localhost:8742"),
        )
        return None


def stop_tunnel() -> None:
    """Disconnect all active ngrok tunnels (called at pipeline shutdown)."""
    try:
        from pyngrok import ngrok  # type: ignore[import-untyped]
        ngrok.kill()
        logger.info("ngrok tunnel stopped")
    except Exception:  # noqa: BLE001
        pass  # Already stopped or never started


__all__ = ["start_tunnel", "stop_tunnel", "is_tunnel_enabled"]
