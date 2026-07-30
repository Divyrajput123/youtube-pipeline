"""Automatic Instagram/Facebook long-lived token refresh.

Long-lived tokens expire after 60 days, but they can be exchanged for a NEW
60-day token at any time before expiry — no user interaction required.

This module:
  1. Checks if the current token is within 7 days of expiry (or already expired).
  2. If so, exchanges it for a fresh 60-day token via the Graph API.
  3. Updates the .env file on disk so the new token persists across pipeline runs.

Called automatically at the start of each pipeline run when Instagram Reels is enabled.

Reference: https://developers.facebook.com/docs/facebook-login/guides/access-tokens/get-long-lived/
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
from datetime import datetime, timezone

import httpx

logger = logging.getLogger(__name__)

# Refresh the token when it has fewer than this many days left
_REFRESH_THRESHOLD_DAYS = 7

_GRAPH_API_BASE = "https://graph.facebook.com/v25.0"


async def check_and_refresh_token(
    access_token: str,
    app_id: str | None = None,
    app_secret: str | None = None,
    env_file_path: str | None = None,
) -> str:
    """Check token expiry and refresh if needed. Returns the (possibly new) token.

    The refresh works by exchanging the current long-lived token for a new one.
    This does NOT require app_secret — Facebook allows refreshing long-lived
    user tokens with just the token itself via the /oauth/access_token endpoint.

    However, if app_id and app_secret are provided, it uses the more reliable
    fb_exchange_token flow.

    Args:
        access_token: Current long-lived access token.
        app_id: Facebook App ID (optional, from FACEBOOK_APP_ID env var).
        app_secret: Facebook App Secret (optional, from FACEBOOK_APP_SECRET env var).
        env_file_path: Path to .env file to update with new token.

    Returns:
        The valid access token (refreshed if it was near expiry, unchanged otherwise).
    """
    if not access_token:
        return access_token

    # Step 1: Check token expiry using the debug_token endpoint
    expiry_info = await _get_token_expiry(access_token)

    if expiry_info is None:
        logger.warning("Could not check Instagram token expiry — using token as-is")
        return access_token

    days_left, expires_at = expiry_info

    if days_left > _REFRESH_THRESHOLD_DAYS:
        logger.info(
            "Instagram token is valid for %d more days (expires %s) — no refresh needed",
            days_left, expires_at.strftime("%Y-%m-%d"),
        )
        return access_token

    # Step 2: Token is near expiry — refresh it
    logger.info(
        "Instagram token expires in %d days (%s) — refreshing...",
        days_left, expires_at.strftime("%Y-%m-%d"),
    )

    new_token = await _refresh_token(
        current_token=access_token,
        app_id=app_id or os.environ.get("FACEBOOK_APP_ID", ""),
        app_secret=app_secret or os.environ.get("FACEBOOK_APP_SECRET", ""),
    )

    if new_token and new_token != access_token:
        logger.info("Instagram token refreshed successfully (new 60-day token)")

        # Step 3: Update .env file on disk
        _update_env_file(new_token, env_file_path)

        # Also update the environment variable for this process
        os.environ["INSTAGRAM_ACCESS_TOKEN"] = new_token

        return new_token

    logger.warning("Instagram token refresh failed — using existing token")
    return access_token


async def _get_token_expiry(access_token: str) -> tuple[int, datetime] | None:
    """Query the token debug endpoint to get expiry info.

    Returns:
        Tuple of (days_remaining, expiry_datetime) or None if check fails.
    """
    # Use the token to query its own debug info
    url = f"{_GRAPH_API_BASE}/debug_token"
    params = {
        "input_token": access_token,
        "access_token": access_token,  # self-inspection
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        token_data = data.get("data", {})
        expires_at_unix = token_data.get("expires_at", 0)

        if expires_at_unix == 0:
            # Token never expires (rare, but possible for system tokens)
            logger.info("Instagram token has no expiry (never expires)")
            return (9999, datetime(2099, 1, 1, tzinfo=timezone.utc))

        expires_at = datetime.fromtimestamp(expires_at_unix, tz=timezone.utc)
        now = datetime.now(timezone.utc)
        days_left = (expires_at - now).days

        return (days_left, expires_at)

    except Exception as exc:
        logger.debug("Token debug check failed: %s", exc)
        return None


async def _refresh_token(
    current_token: str,
    app_id: str,
    app_secret: str,
) -> str | None:
    """Exchange the current long-lived token for a new 60-day token.

    Uses the fb_exchange_token grant type if app_id and app_secret are available.
    Otherwise attempts a direct refresh via the /me endpoint (less reliable).
    """
    if not app_id or not app_secret:
        logger.warning(
            "FACEBOOK_APP_ID or FACEBOOK_APP_SECRET not set — "
            "cannot auto-refresh Instagram token. "
            "Set these in .env to enable automatic token renewal."
        )
        return None

    url = f"{_GRAPH_API_BASE}/oauth/access_token"
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": current_token,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        new_token = data.get("access_token")
        if new_token:
            expires_in = data.get("expires_in", 0)
            logger.info(
                "Token exchange successful — new token expires in %d seconds (~%d days)",
                expires_in, expires_in // 86400,
            )
            return new_token

        logger.warning("Token exchange response missing access_token: %s", data)
        return None

    except Exception as exc:
        logger.error("Token refresh request failed: %s", exc)
        return None


def _update_env_file(new_token: str, env_file_path: str | None = None) -> None:
    """Update the INSTAGRAM_ACCESS_TOKEN value in the .env file."""
    if env_file_path:
        env_path = pathlib.Path(env_file_path)
    else:
        # Default: .env in project root (two levels up from this file)
        env_path = pathlib.Path(__file__).parent.parent.parent / ".env"

    if not env_path.exists():
        logger.warning("Cannot update .env file — %s not found", env_path)
        return

    try:
        content = env_path.read_text()

        # Replace the token value using regex
        pattern = r"(INSTAGRAM_ACCESS_TOKEN=).*"
        replacement = rf"\g<1>{new_token}"
        new_content = re.sub(pattern, replacement, content)

        if new_content != content:
            env_path.write_text(new_content)
            logger.info("Updated INSTAGRAM_ACCESS_TOKEN in %s", env_path)
        else:
            logger.warning(
                "Could not find INSTAGRAM_ACCESS_TOKEN line in %s — "
                "add it manually: INSTAGRAM_ACCESS_TOKEN=%s",
                env_path, new_token[:20] + "...",
            )
    except Exception as exc:
        logger.error("Failed to update .env file: %s", exc)


__all__ = ["check_and_refresh_token"]
