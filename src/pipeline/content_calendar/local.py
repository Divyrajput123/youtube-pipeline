"""Local JSON-based Content Calendar for development/testing.

Stores all video records in a local JSON file instead of Notion.
Implements the same interface as Content_Calendar so it's a drop-in replacement.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from pipeline.models import PipelineStatus


class LocalContentCalendar:
    """File-based content calendar for local development.

    Stores records in ./pipeline_output/calendar.json
    """

    def __init__(self, db_path: str = "./pipeline_output/calendar.json") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, dict[str, Any]] = {}
        if self._path.exists():
            try:
                self._records = json.loads(self._path.read_text())
            except Exception:
                self._records = {}

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._records, indent=2, default=str))

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    async def create_record(self, video_id: str, batch_id: Optional[str] = None) -> str:
        self._records[video_id] = {
            "video_id": video_id,
            "title": "",
            "topic": "",
            "status": PipelineStatus.PENDING.value,
            "scheduled_publish_datetime": None,
            "script_url": None,
            "narration_url": None,
            "video_url": None,
            "thumbnail_url": None,
            "metadata_url": None,
            "pipeline_run_timestamp": self._now().isoformat(),
            "batch_id": batch_id,
            "style_profile_doc_id": "",
        }
        self._save()
        print(f"[calendar] Created record for {video_id}")
        return video_id

    async def update_status(self, video_id: str, status: PipelineStatus) -> None:
        if video_id not in self._records:
            raise Exception(f"No record found for video_id='{video_id}'")
        self._records[video_id]["status"] = status.value
        self._save()
        print(f"[calendar] {video_id} → {status.value}")

    async def update_asset_link(self, video_id: str, asset_type: str, url: str) -> None:
        field_map = {
            "script": "script_url", "narration": "narration_url",
            "video": "video_url", "thumbnail": "thumbnail_url", "metadata": "metadata_url",
        }
        field = field_map.get(asset_type)
        if not field:
            raise ValueError(f"Unknown asset_type '{asset_type}'")
        if video_id not in self._records:
            raise Exception(f"No record found for video_id='{video_id}'")
        self._records[video_id][field] = url
        self._save()

    async def set_publish_datetime(self, video_id: str, dt: datetime) -> None:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = self._now()
        if dt <= now:
            raise ValueError(f"Publish datetime {dt.isoformat()} is in the past")
        rec = self._records.get(video_id, {})
        if rec.get("status") == PipelineStatus.PUBLISHED.value:
            raise ValueError(f"Video {video_id} is already Published")
        if video_id not in self._records:
            raise Exception(f"No record found for video_id='{video_id}'")
        self._records[video_id]["scheduled_publish_datetime"] = dt.isoformat()
        self._save()

    async def get_batch_topics(self, batch_id: str, lookback_days: int) -> list[str]:
        cutoff = self._now() - timedelta(days=lookback_days)
        topics = []
        for rec in self._records.values():
            if rec.get("batch_id") != batch_id:
                continue
            ts_str = rec.get("pipeline_run_timestamp")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts >= cutoff and rec.get("topic"):
                        topics.append(rec["topic"])
                except Exception:
                    pass
        return topics

    async def get_batch_completion(self, batch_id: str) -> int:
        batch_records = [r for r in self._records.values() if r.get("batch_id") == batch_id]
        if not batch_records:
            return 0
        published = sum(1 for r in batch_records if r.get("status") == PipelineStatus.PUBLISHED.value)
        return math.floor(published / len(batch_records) * 100)

    async def _get_status(self, video_id: str) -> PipelineStatus:
        rec = self._records.get(video_id)
        if not rec:
            raise Exception(f"No record found for video_id='{video_id}'")
        return PipelineStatus(rec["status"])

    @staticmethod
    def detect_conflicts(scheduled_datetimes: dict[str, datetime]) -> list[tuple[str, str]]:
        from collections import defaultdict
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for vid, dt in scheduled_datetimes.items():
            buckets[(dt.year, dt.month, dt.day, dt.hour, dt.minute)].append(vid)
        conflicts = []
        for vids in buckets.values():
            if len(vids) >= 2:
                vids_sorted = sorted(vids)
                for i in range(len(vids_sorted)):
                    for j in range(i + 1, len(vids_sorted)):
                        conflicts.append((vids_sorted[i], vids_sorted[j]))
        return sorted(conflicts)

    @staticmethod
    def detect_gaps(scheduled_datetimes: dict[str, datetime]) -> list[tuple[date, date]]:
        if len(scheduled_datetimes) < 2:
            return []
        dates = sorted({dt.date() for dt in scheduled_datetimes.values()})
        gaps = []
        for i in range(len(dates) - 1):
            gap_start = dates[i] + timedelta(days=1)
            gap_end = dates[i + 1] - timedelta(days=1)
            if (gap_end - gap_start).days + 1 >= 7:
                gaps.append((gap_start, gap_end))
        return gaps
