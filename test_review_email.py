#!/usr/bin/env python3
"""Test the full review gate flow with real email delivery.

What this does:
  1. Starts the webhook server on localhost:8742
  2. Opens a review gate (simulating script being ready)
  3. Sends a REAL email to divysingh178@gmail.com with approve/edit links
  4. Waits for you to tap a link (up to 5 minutes)
  5. Reports what action was received

Run:
    PYTHONPATH=src python test_review_email.py
"""

import asyncio
import json
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from dotenv import load_dotenv
load_dotenv(".env", override=True)

os.environ["PIPELINE_PUBLIC_URL"] = "http://localhost:8742"
os.environ["REVIEW_SERVER_PORT"]  = "8742"

from pipeline.review_server import start_review_server, _pending
from pipeline.orchestrator.review_gate import ReviewGate
from pipeline.notifier import Notifier, NotifierConfig
from pipeline.models import PipelineStatus, SmtpConfig


class _FakeCalendar:
    def __init__(self):
        self._statuses = {}

    async def update_status(self, video_id, status):
        self._statuses[video_id] = status

    async def _get_status(self, video_id):
        return self._statuses.get(video_id, PipelineStatus.PENDING)

    async def update_asset_link(self, *a, **kw):
        pass


async def main():
    print()
    print("=" * 60)
    print("  Review Gate — Real Email Test")
    print("=" * 60)

    # Load SMTP config from config.json
    with open("config.json") as f:
        cfg = json.load(f)
    smtp_raw = cfg["notification_channels"]["smtp"]

    if not smtp_raw or smtp_raw.get("username", "").startswith("REPLACE"):
        print("✗ SMTP not configured in config.json")
        return 1

    smtp = SmtpConfig(**smtp_raw)
    notifier = Notifier(config=NotifierConfig(smtp=smtp))

    # Start webhook server
    server_task = asyncio.create_task(start_review_server(port=8742))
    await asyncio.sleep(0.4)

    gate = ReviewGate(
        gate_type="script",
        video_id="video-email-test",
        content_calendar=_FakeCalendar(),
        notifier=notifier,
    )

    print(f"\nSending review email to {smtp.to_address}...")
    await gate.trigger(asset_links=[
        "https://drive.google.com/file/d/EXAMPLE_SCRIPT_URL/view"
    ])

    # Show URLs in terminal too
    from pipeline.review_server import get_review_urls
    token = next(iter(_pending), None)
    if token:
        approve_url, edit_url = get_review_urls(token)
        print(f"\n✓ Email sent!")
        print(f"\nOr tap these links directly in your browser:")
        print(f"  Approve: {approve_url}")
        print(f"  Edit:    {edit_url}")

    print(f"\nWaiting up to 5 minutes for your response...")
    print("(tap a link in the email or browser above)\n")

    # Wait up to 5 minutes
    import pipeline.orchestrator.review_gate as rg
    rg._POLL_INTERVAL_S = 5.0  # faster polling for this test

    try:
        result = await asyncio.wait_for(gate.poll_until_action(), timeout=300)
        print(f"\n✓ Response received: action = '{result['action']}'")
        if result["action"] == "approve":
            print("  → Pipeline would continue to narration generation.")
        elif result["action"] == "edit":
            print("  → Pipeline would regenerate the script.")
    except asyncio.TimeoutError:
        print("\n✗ Timed out after 5 minutes. No response received.")

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
