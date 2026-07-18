#!/usr/bin/env python3
"""End-to-end test for the review gate + webhook server.

Tests:
  1. Webhook tap — Approve
  2. Webhook tap — Request Edits
  3. Notion fallback — status change picked up by polling
  4. Security — invalid token returns 404

Run with:
    PYTHONPATH=src python test_review_gate.py
"""

import asyncio
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

import httpx
from pipeline.review_server import create_review_token, get_review_urls, start_review_server, _pending
from pipeline.orchestrator.review_gate import ReviewGate
from pipeline.notifier import Notifier, NotifierConfig
from pipeline.models import PipelineStatus


class _FakeCalendar:
    def __init__(self):
        self._statuses = {}

    async def update_status(self, video_id, status):
        self._statuses[video_id] = status
        print(f"  [calendar] {video_id} → {status.value}")

    async def _get_status(self, video_id):
        return self._statuses.get(video_id, PipelineStatus.PENDING)

    async def update_asset_link(self, *a, **kw):
        pass


async def test_webhook_approve():
    print("\n" + "=" * 60)
    print("TEST 1: Webhook — tap Approve link")
    print("=" * 60)

    gate = ReviewGate(
        gate_type="script",
        video_id="video-test-001",
        content_calendar=_FakeCalendar(),
        notifier=Notifier(config=NotifierConfig()),
    )
    await gate.trigger(asset_links=["https://drive.google.com/fake-script"])

    token = next(iter(_pending))
    approve_url, edit_url = get_review_urls(token)
    print(f"  Approve URL: {approve_url}")
    print(f"  Edit URL:    {edit_url}")

    async def _tap():
        await asyncio.sleep(0.3)
        print("  → Simulating phone tap on Approve...")
        async with httpx.AsyncClient() as c:
            r = await c.get(approve_url)
            print(f"  → HTTP {r.status_code}")
            assert r.status_code == 200
            assert "Approved" in r.text

    asyncio.create_task(_tap())
    result = await asyncio.wait_for(gate.poll_until_action(), timeout=10)
    assert result["action"] == "approve"
    print("  ✓ PASSED")


async def test_webhook_edit():
    print("\n" + "=" * 60)
    print("TEST 2: Webhook — tap Request Edits link")
    print("=" * 60)

    gate = ReviewGate(
        gate_type="script",
        video_id="video-test-002",
        content_calendar=_FakeCalendar(),
        notifier=Notifier(config=NotifierConfig()),
    )
    await gate.trigger(asset_links=["https://drive.google.com/fake-script"])

    token = next(iter(_pending))
    _, edit_url = get_review_urls(token)

    async def _tap():
        await asyncio.sleep(0.3)
        print("  → Simulating phone tap on Request Edits...")
        async with httpx.AsyncClient() as c:
            r = await c.get(edit_url)
            print(f"  → HTTP {r.status_code}")
            assert r.status_code == 200

    asyncio.create_task(_tap())
    result = await asyncio.wait_for(gate.poll_until_action(), timeout=10)
    assert result["action"] == "edit"
    print("  ✓ PASSED")


async def test_notion_fallback():
    print("\n" + "=" * 60)
    print("TEST 3: Notion fallback — status change triggers approval")
    print("=" * 60)

    import pipeline.orchestrator.review_gate as rg
    original = rg._POLL_INTERVAL_S
    rg._POLL_INTERVAL_S = 1.0  # speed up for test

    calendar = _FakeCalendar()
    gate = ReviewGate(
        gate_type="script",
        video_id="video-test-003",
        content_calendar=calendar,
        notifier=Notifier(config=NotifierConfig()),
    )
    await gate.trigger(asset_links=[])
    print("  Gate open. Waiting 2s then simulating Notion status change...")

    async def _update():
        await asyncio.sleep(2)
        print("  → Notion status: Script Approved")
        calendar._statuses["video-test-003"] = PipelineStatus.SCRIPT_APPROVED

    asyncio.create_task(_update())
    result = await asyncio.wait_for(gate.poll_until_action(), timeout=15)
    assert result["action"] == "approve"
    print("  ✓ PASSED")

    rg._POLL_INTERVAL_S = original


async def test_invalid_token():
    print("\n" + "=" * 60)
    print("TEST 4: Security — invalid token returns 404")
    print("=" * 60)

    async with httpx.AsyncClient() as c:
        r = await c.get("http://localhost:8742/review/approve?token=INVALID_TOKEN")
        print(f"  → HTTP {r.status_code}")
        assert r.status_code == 404
    print("  ✓ PASSED")


async def main():
    print("\n" + "=" * 60)
    print("  Review Gate + Webhook Server — End-to-End Tests")
    print("=" * 60)

    # Single server for all tests
    server_task = asyncio.create_task(start_review_server(port=8742))
    await asyncio.sleep(0.4)

    results = []
    for name, fn in [
        ("Webhook Approve",  test_webhook_approve),
        ("Webhook Edit",     test_webhook_edit),
        ("Notion Fallback",  test_notion_fallback),
        ("Invalid Token",    test_invalid_token),
    ]:
        try:
            await fn()
            results.append((name, True, None))
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append((name, False, str(e)))
        await asyncio.sleep(0.1)

    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for name, ok, err in results:
        print(f"  {'✓' if ok else '✗'}  {name}" + (f"  — {err}" if err else ""))

    passed = all(ok for _, ok, _ in results)
    print()
    print("  All tests passed! ✓" if passed else "  Some tests FAILED ✗")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
