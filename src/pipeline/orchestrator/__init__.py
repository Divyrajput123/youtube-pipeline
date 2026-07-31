"""Orchestrator subsystem — Claude-based pipeline coordinator.

The Orchestrator sequences subsystem calls for the AI YouTube Content Pipeline,
handles retry logic with HTTP error classification, manages Review Gates,
emits structured JSON logs to Asset_Store, and supports resume-from-error.

Tasks implemented here
----------------------
* 16.1 — ``start_pipeline``: stage sequencing, status updates, structured logging.
* 16.2 — General retry policy: transient vs non-transient HTTP error classification,
         ``_call_with_retry`` helper with 5 s → 10 s → 20 s exponential back-off.
* 16.3 — ``resume_pipeline``: load last good outputs from Asset_Store, restart
         from the first missing/unreadable stage, otherwise resume from failed stage.
* 16.4 — ``ReviewGate``: durable review checkpoints (Gate 1 — Script, Gate 2 — Final)
         with reminder schedule, auto-approve, and edit validation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Literal, Optional

from pipeline.asset_store import Asset_Store, AssetStoreError
from pipeline.content_calendar import Content_Calendar
from pipeline.cross_poster import Cross_Poster
from pipeline.instagram_reels import InstagramReelsClient, upload_reel_from_short
from pipeline.metadata_generator import Metadata_Generator
from pipeline.models import (
    BatchRecord,
    LogEntry,
    MetadataPackage,
    NarrationAsset,
    PipelineConfig,
    PipelineStatus,
    Script,
    StyleProfile,
    SubFolder,
    TopicEntry,
    VisualAsset,
    YouTubeVideoRef,
)
from pipeline.orchestrator.batch import (
    compute_batch_completion,
    generate_batch_slots,
    validate_batch_size,
)
from pipeline.narration_generator import Narration_Generator
from pipeline.notifier import Notifier
from pipeline.orchestrator.review_gate import ReviewGate, ReviewGateError
from pipeline.publisher import Publisher
from pipeline.reference_analyzer import Reference_Analyzer
from pipeline.script_writer import Script_Writer
from pipeline.topic_researcher import Topic_Researcher
from pipeline.visual_generator import Visual_Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — orchestrator retry policy (design §1, task 16.2)
# ---------------------------------------------------------------------------

# Transient HTTP status codes — retry 3× with exponential back-off.
_TRANSIENT_HTTP_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Non-transient HTTP status codes — fail immediately, no retry.
_NONTRANSIENT_HTTP_CODES: frozenset[int] = frozenset({400, 401, 403, 404})

# General retry schedule: delays for attempt 1, 2, 3 (seconds).
_RETRY_DELAYS: tuple[float, float, float] = (5.0, 10.0, 20.0)

# Maximum time (seconds) to update Content_Calendar after a stage transition.
_CALENDAR_UPDATE_DEADLINE_S: float = 30.0

# Maximum time (seconds) to notify the Notifier after a stage failure.
_FAILURE_NOTIFY_DEADLINE_S: float = 60.0


# ---------------------------------------------------------------------------
# PipelineRun dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineRun:
    """In-memory state for a single pipeline execution.

    Attributes:
        run_id: UUID4 string that uniquely identifies this run.
        video_ids: Ordered list of video IDs processed in this run.
        status: High-level status string (e.g. ``"running"``, ``"completed"``,
            ``"error"``).
    """

    run_id: str
    video_ids: list[str]
    status: str = "running"


# ---------------------------------------------------------------------------
# HTTP error classification helpers (task 16.2)
# ---------------------------------------------------------------------------


def _extract_http_code(exc: BaseException) -> Optional[int]:
    """Try to extract an HTTP status code from *exc*.

    Inspection order:
    1. ``exc.http_response_code`` attribute (int).
    2. ``exc.status_code`` attribute (int).
    3. Scan ``str(exc)`` for ``status_code=NNN`` or a bare 3-digit HTTP code.

    Returns:
        Integer HTTP status code, or ``None`` if none can be determined.
    """
    for attr in ("http_response_code", "status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val < 600:
            return val

    # Scan the string representation for common patterns.
    msg = str(exc)
    m = re.search(r"status[_\s]?code[=:\s]+(\d{3})", msg, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Bare 3-digit code at word boundary (e.g. "HTTP 429")
    m = re.search(r"\b([45]\d{2})\b", msg)
    if m:
        return int(m.group(1))

    return None


def _is_transient_http_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* represents a transient HTTP error (429, 500, 502, 503, 504)."""
    code = _extract_http_code(exc)
    return code is not None and code in _TRANSIENT_HTTP_CODES


def _is_nontransient_http_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* represents a non-transient HTTP error (400, 401, 403, 404)."""
    code = _extract_http_code(exc)
    return code is not None and code in _NONTRANSIENT_HTTP_CODES


# ---------------------------------------------------------------------------
# LogEntry helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def _make_log_entry(
    *,
    event_type: Literal["api_call", "stage_transition", "retry_attempt", "error", "warning"],
    stage_name: str,
    message: str,
    video_id: Optional[str] = None,
    batch_id: Optional[str] = None,
    http_response_code: Optional[int] = None,
    retry_attempt: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> LogEntry:
    """Construct a :class:`~pipeline.models.LogEntry` with the current UTC timestamp."""
    return LogEntry(
        timestamp=_utcnow(),
        event_type=event_type,
        stage_name=stage_name,
        video_id=video_id,
        batch_id=batch_id,
        http_response_code=http_response_code,
        retry_attempt=retry_attempt,
        message=message,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Sequences subsystem calls for the AI YouTube Content Pipeline.

    The Orchestrator is the central coordinator: it reads ``PipelineConfig``,
    resolves which stage to execute next for each video, calls subsystems in
    the canonical order, applies the general retry policy for transient HTTP
    errors, updates the Content_Calendar on every state transition, and emits
    a structured JSON log to Asset_Store.

    **Stage sequence** (single-video pipeline):
      Reference_Analyzer → Topic_Researcher → Script_Writer (→ Review Gate 1)
      → Narration_Generator → Visual_Generator → Metadata_Generator
      (→ Review Gate 2) → Publisher → Cross_Poster

    **Retry policy (task 16.2)**:
      - Transient HTTP errors (429, 500, 502, 503, 504): up to 3 attempts with
        delays 5 s → 10 s → 20 s.
      - Non-transient errors (400, 401, 403, 404): fail immediately, no retry.

    Args:
        config: Pipeline configuration (:class:`~pipeline.models.PipelineConfig`).
        reference_analyzer: :class:`~pipeline.reference_analyzer.Reference_Analyzer`.
        topic_researcher: :class:`~pipeline.topic_researcher.Topic_Researcher`.
        script_writer: :class:`~pipeline.script_writer.Script_Writer`.
        narration_generator: :class:`~pipeline.narration_generator.Narration_Generator`.
        visual_generator: :class:`~pipeline.visual_generator.Visual_Generator`.
        metadata_generator: :class:`~pipeline.metadata_generator.Metadata_Generator`.
        publisher: :class:`~pipeline.publisher.Publisher`.
        cross_poster: :class:`~pipeline.cross_poster.Cross_Poster`.
        asset_store: :class:`~pipeline.asset_store.Asset_Store`.
        content_calendar: :class:`~pipeline.content_calendar.Content_Calendar`.
        notifier: :class:`~pipeline.notifier.Notifier`.
    """

    def __init__(
        self,
        config: PipelineConfig,
        reference_analyzer: Reference_Analyzer,
        topic_researcher: Topic_Researcher,
        script_writer: Script_Writer,
        narration_generator: Narration_Generator,
        visual_generator: Visual_Generator,
        metadata_generator: Metadata_Generator,
        publisher: Publisher,
        cross_poster: Cross_Poster,
        instagram_reels_client: Optional[InstagramReelsClient],
        asset_store: Asset_Store,
        content_calendar: Content_Calendar,
        notifier: Notifier,
    ) -> None:
        self._config = config
        self._reference_analyzer = reference_analyzer
        self._topic_researcher = topic_researcher
        self._script_writer = script_writer
        self._narration_generator = narration_generator
        self._visual_generator = visual_generator
        self._metadata_generator = metadata_generator
        self._publisher = publisher
        self._cross_poster = cross_poster
        self._instagram_reels_client = instagram_reels_client
        self._asset_store = asset_store
        self._content_calendar = content_calendar
        self._notifier = notifier

        # In-memory log accumulator: list of LogEntry objects for the current run.
        # Flushed to Asset_Store after each stage via _flush_log().
        self._log_entries: list[LogEntry] = []

    # ------------------------------------------------------------------
    # Task 16.1 — start_pipeline
    # ------------------------------------------------------------------

    async def start_pipeline(self) -> str:
        """Start a new pipeline run for a single video.

        Steps
        -----
        1. Generate a UUID4 ``run_id``; derive ``video_id`` as
           ``f"video-{run_id[:8]}"``.
        2. Create a Content_Calendar record for the video.
        3. Execute the ordered stage sequence, updating Content_Calendar
           status and emitting ``LogEntry`` objects at every step.
        4. On any unrecoverable stage failure: log the error, update
           Content_Calendar to ``Pipeline Error — {stage_name}``, and notify
           the Notifier within 60 seconds.

        Returns:
            The ``run_id`` (UUID4 string) for this pipeline run.
        """
        run_id: str = str(uuid.uuid4())
        video_id: str = f"video-{run_id[:8]}"

        self._log_entries = []  # reset log accumulator for this run
        logger.info("Orchestrator.start_pipeline: run_id=%s video_id=%s", run_id, video_id)

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Pipeline started: run_id={run_id}, video_id={video_id}",
            video_id=video_id,
        )

        # Create Content_Calendar record
        try:
            await self._content_calendar.create_record(video_id=video_id, batch_id=None)
        except Exception as exc:  # noqa: BLE001
            self._emit_log(
                event_type="error",
                stage_name="orchestrator",
                message=f"Failed to create Content_Calendar record: {exc}",
                video_id=video_id,
            )
            await self._flush_log(run_id, video_id)
            raise

        # ------------------------------------------------------------------ #
        # Stage 1: Reference_Analyzer → StyleProfile                          #
        # ------------------------------------------------------------------ #
        style_profile: StyleProfile = await self._run_stage(
            stage_name="reference_analyzer",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.RESEARCHING,
            coro_factory=lambda: self._reference_analyzer.analyze(
                channel_url=self._config.reference_channel_url
            ),
        )

        # ------------------------------------------------------------------ #
        # Stage 2: Topic_Researcher → list[TopicEntry]                        #
        # ------------------------------------------------------------------ #
        # Fetch topics used in past 90 days to avoid repetition
        past_topics: list[str] = []
        try:
            past_topics = await self._content_calendar.get_batch_topics(
                batch_id="", lookback_days=180
            )
        except Exception:
            pass  # if it fails, proceed without exclusions

        # Also pull titles of already-uploaded YouTube videos to avoid repeating
        # topics that exist on the channel but predate the Notion calendar
        try:
            yt_titles = await self._publisher._yt.list_uploaded_titles()  # noqa: SLF001
            past_topics = list({*past_topics, *yt_titles})  # merge, deduplicate
            logger.info(
                "start_pipeline: built exclusion list with %d titles (%d from Notion, %d from YouTube)",
                len(past_topics), len(past_topics) - len(yt_titles), len(yt_titles),
            )
        except Exception as exc:
            logger.warning("start_pipeline: could not fetch YouTube titles for exclusion: %s", exc)

        topics: list[TopicEntry] = await self._run_stage(
            stage_name="topic_researcher",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.RESEARCHING,
            coro_factory=lambda: self._topic_researcher.research(
                batch_size=1,
                excluded_titles=past_topics,
                run_id=run_id,
            ),
        )

        await self._update_calendar_status(video_id, PipelineStatus.SCRIPTING, run_id)

        # ------------------------------------------------------------------ #
        # Stage 3: Script_Writer → Script                                     #
        # ------------------------------------------------------------------ #
        # Use the top-ranked topic.
        top_topic: TopicEntry = topics[0]

        # Persist chosen topic to Notion so future runs can exclude it
        try:
            await self._content_calendar.update_topic(video_id, top_topic.title)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save topic to Notion (dedup may miss it): %s", exc)
        script: Script = await self._run_stage(
            stage_name="script_writer",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.SCRIPTING,
            coro_factory=lambda: self._script_writer.generate(
                topic=top_topic,
                style_profile=style_profile,
                video_id=video_id,
                script_duration_minutes=self._config.script_duration_minutes,
            ),
        )

        # Update Notion with script Drive URL
        if script.asset_url:
            try:
                await self._content_calendar.update_asset_link(video_id, "script", script.asset_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update script_url in Notion: %s", exc)
        await self._update_calendar_status(
            video_id, PipelineStatus.AWAITING_SCRIPT_REVIEW, run_id
        )
        self._emit_log(
            event_type="stage_transition",
            stage_name="review_gate_1",
            message="Review Gate 1 open — awaiting script approval",
            video_id=video_id,
        )
        # Gate 1: trigger and enter WAIT state.
        gate1 = self.create_review_gate(gate_type="script", video_id=video_id)
        await gate1.trigger(
            asset_links=[script.asset_url] if script.asset_url else [],
        )
        gate1_result = await gate1.poll_until_action()
        gate1_action = gate1_result.get("action")
        self._emit_log(
            event_type="stage_transition",
            stage_name="review_gate_1",
            message=f"Review Gate 1 closed: action={gate1_action}",
            video_id=video_id,
        )
        await self._flush_log(run_id, video_id)

        # Handle edit: revise the script with the supplied instructions then
        # re-open Gate 1 for another round of review.
        if gate1_action == "edit":
            edit_instructions = gate1_result.get("payload", "").strip()
            if edit_instructions:
                logger.info(
                    "Gate 1 edit: revising script for video_id=%s instructions=%s",
                    video_id, edit_instructions[:80],
                )
                script = await self._run_stage(
                    stage_name="script_writer_revision",
                    video_id=video_id,
                    run_id=run_id,
                    pre_status=PipelineStatus.SCRIPTING,
                    coro_factory=lambda: self._script_writer.revise(
                        script=script,
                        edits=edit_instructions,
                        video_id=video_id,
                    ),
                )
                if script.asset_url:
                    try:
                        await self._content_calendar.update_asset_link(video_id, "script", script.asset_url)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Could not update revised script_url in Notion: %s", exc)
                # Re-open Gate 1 for the revised script
                await self._update_calendar_status(video_id, PipelineStatus.AWAITING_SCRIPT_REVIEW, run_id)
                gate1b = self.create_review_gate(gate_type="script", video_id=video_id)
                await gate1b.trigger(asset_links=[script.asset_url] if script.asset_url else [])
                gate1_result = await gate1b.poll_until_action()
                gate1_action = gate1_result.get("action")
                self._emit_log(
                    event_type="stage_transition",
                    stage_name="review_gate_1",
                    message=f"Review Gate 1 (revision) closed: action={gate1_action}",
                    video_id=video_id,
                )
                await self._flush_log(run_id, video_id)

        # Handle reject: keep regenerating new topics + scripts until approved
        while gate1_action == "reject":
            logger.info("Gate 1 rejected: restarting from topic research for video_id=%s", video_id)

            # Fetch fresh exclusion list before marking as rejected
            past_topics_reject: list[str] = []
            try:
                past_topics_reject = await self._content_calendar.get_batch_topics(
                    batch_id="", lookback_days=180
                )
            except Exception:
                pass

            # Create a NEW Notion record, so topic research updates the new one
            new_video_id = f"video-{run_id[:8]}r"
            try:
                await self._content_calendar.create_record(video_id=new_video_id, batch_id=None)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create new Notion record after reject: %s — reusing video_id", exc)
                new_video_id = video_id

            # Mark the original/previous record as rejected
            if new_video_id != video_id:
                await self._update_calendar_status(video_id, PipelineStatus.SCRIPT_REJECTED, run_id)

            # Switch to the new video_id
            video_id = new_video_id

            new_topics: list[TopicEntry] = await self._run_stage(
                stage_name="topic_researcher_regen",
                video_id=video_id,
                run_id=run_id,
                pre_status=PipelineStatus.RESEARCHING,
                coro_factory=lambda: self._topic_researcher.research(
                    batch_size=1,
                    excluded_titles=past_topics_reject,
                    run_id=run_id,
                ),
            )
            top_topic = new_topics[0]

            try:
                await self._content_calendar.update_topic(video_id, top_topic.title)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update topic in Notion after reject: %s", exc)

            script = await self._run_stage(
                stage_name="script_writer_regen",
                video_id=video_id,
                run_id=run_id,
                pre_status=PipelineStatus.SCRIPTING,
                coro_factory=lambda: self._script_writer.generate(
                    topic=top_topic,
                    style_profile=style_profile,
                    video_id=video_id,
                    script_duration_minutes=self._config.script_duration_minutes,
                ),
            )
            if script.asset_url:
                try:
                    await self._content_calendar.update_asset_link(video_id, "script", script.asset_url)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not update regenerated script_url in Notion: %s", exc)

            # Re-open Gate 1 for the new script
            await self._update_calendar_status(video_id, PipelineStatus.AWAITING_SCRIPT_REVIEW, run_id)
            gate_regen = self.create_review_gate(gate_type="script", video_id=video_id)
            await gate_regen.trigger(asset_links=[script.asset_url] if script.asset_url else [])
            gate1_result = await gate_regen.poll_until_action()
            gate1_action = gate1_result.get("action")
            self._emit_log(
                event_type="stage_transition",
                stage_name="review_gate_1",
                message=f"Review Gate 1 (regeneration) closed: action={gate1_action}",
                video_id=video_id,
            )
            await self._flush_log(run_id, video_id)

        # ------------------------------------------------------------------ #
        # Stage 4: Narration_Generator → NarrationAsset                       #
        # ------------------------------------------------------------------ #
        await self._update_calendar_status(video_id, PipelineStatus.SCRIPT_APPROVED, run_id)

        narration: NarrationAsset = await self._run_stage(
            stage_name="narration_generator",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.NARRATION_READY,
            coro_factory=lambda: self._narration_generator.generate(
                script=script,
                voice_id=self._config.voice_id,
                video_id=video_id,
            ),
        )

        # Update Notion with narration Drive URL
        if narration.asset_url:
            try:
                await self._content_calendar.update_asset_link(video_id, "narration", narration.asset_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update narration_url in Notion: %s", exc)

        # ------------------------------------------------------------------ #
        # Stage 5: Visual_Generator → VisualAsset                             #
        # ------------------------------------------------------------------ #
        visual: VisualAsset = await self._run_stage(
            stage_name="visual_generator",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.GENERATING_VISUALS,
            coro_factory=lambda: self._visual_generator.generate(
                script=script,
                narration=narration,
                style_profile=style_profile,
                video_id=video_id,
            ),
        )

        await self._update_calendar_status(video_id, PipelineStatus.VISUALS_READY, run_id)

        # Update Notion with video and thumbnail Drive URLs
        try:
            updates = {}
            if visual.mp4_url:
                updates["video"] = visual.mp4_url
            if visual.thumbnail_url:
                updates["thumbnail"] = visual.thumbnail_url
            for asset_type, url in updates.items():
                await self._content_calendar.update_asset_link(video_id, asset_type, url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update video/thumbnail URLs in Notion: %s", exc)

        # ------------------------------------------------------------------ #
        # Stage 6: Metadata_Generator → MetadataPackage                       #
        # ------------------------------------------------------------------ #
        metadata: MetadataPackage = await self._run_stage(
            stage_name="metadata_generator",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.GENERATING_METADATA,
            coro_factory=lambda: self._metadata_generator.generate(
                script=script,
                topics=topics,
                video_id=video_id,
            ),
        )

        # Update Notion with metadata Drive URL
        if getattr(metadata, "asset_url", None):
            try:
                await self._content_calendar.update_asset_link(video_id, "metadata", metadata.asset_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update metadata_url in Notion: %s", exc)

        # Advance to Awaiting Final Review (Review Gate 2)
        # Record pipeline end time BEFORE the gate — all generation work is done
        try:
            await self._content_calendar.set_pipeline_end_time(video_id)
        except Exception as exc:
            logger.warning("Could not set pipeline_end_time for %s: %s", video_id, exc)

        await self._update_calendar_status(
            video_id, PipelineStatus.AWAITING_FINAL_REVIEW, run_id
        )
        self._emit_log(
            event_type="stage_transition",
            stage_name="review_gate_2",
            message="Review Gate 2 open — awaiting final asset approval",
            video_id=video_id,
        )
        asset_links = [
            lnk for lnk in [
                visual.mp4_url, visual.thumbnail_url,
                script.asset_url,
                narration.asset_url,
            ] if lnk
        ]
        # Gate 2: trigger and enter WAIT state.
        gate2 = self.create_review_gate(gate_type="final", video_id=video_id)
        await gate2.trigger(asset_links=asset_links)
        gate2_result = await gate2.poll_until_action()
        self._emit_log(
            event_type="stage_transition",
            stage_name="review_gate_2",
            message=f"Review Gate 2 closed: action={gate2_result.get('action')}",
            video_id=video_id,
        )
        await self._flush_log(run_id, video_id)

        # Handle Gate 2 reject — mark as rejected and stop pipeline
        if gate2_result.get("action") == "reject":
            await self._update_calendar_status(video_id, PipelineStatus.SCRIPT_REJECTED, run_id)
            logger.info(
                "start_pipeline: Gate 2 rejected by creator — video_id=%s marked as Script Rejected",
                video_id,
            )
            self._notifier.send_review_gate(
                video_id=video_id,
                gate_type="final",
                asset_links=[],
                action_prompt="❌ Video rejected at final review. Pipeline stopped.",
            )
            return run_id

        # ------------------------------------------------------------------ #
        # Stage 7: Publisher → YouTubeVideoRef                                #
        # (Gate 2 handled externally via handle_review_response)               #
        # ------------------------------------------------------------------ #
        await self._update_calendar_status(
            video_id, PipelineStatus.APPROVED_FOR_UPLOAD, run_id
        )

        yt_ref: YouTubeVideoRef = await self._run_stage(
            stage_name="publisher",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.UPLOADING,
            coro_factory=lambda: self._publisher.upload(
                video_id=video_id,
                assets=visual,
                metadata=metadata,
            ),
        )

        await self._update_calendar_status(video_id, PipelineStatus.UNLISTED, run_id)

        # Store the YouTube video ID in the calendar so the batch scheduler can find it
        try:
            await self._content_calendar.set_youtube_url(
                video_id=video_id,
                youtube_video_id=yt_ref.youtube_video_id,
                unlisted_url=yt_ref.unlisted_url,
            )
        except Exception as exc:
            logger.warning(
                "start_pipeline: could not store YouTube video ID for %s: %s",
                video_id, exc,
            )

        # Patch the metadata JSON in Drive with the YouTube video ID
        try:
            metadata = await self._metadata_generator.patch_youtube_id(
                package=metadata,
                youtube_video_id=yt_ref.youtube_video_id,
            )
        except Exception as exc:
            logger.warning(
                "start_pipeline: could not patch youtube_video_id into metadata JSON for %s: %s",
                video_id, exc,
            )

        # Auto-schedule this single video in the next free slot on YouTube
        scheduled_publish_dt = None  # Track for Instagram Reels sync
        try:
            slots = await self._find_next_free_slot(n_slots=1)
            publish_dt = slots[0]
            await self._content_calendar.set_publish_datetime(
                video_id=video_id,
                dt=publish_dt,
            )
            await self._publisher.schedule(
                video_id=video_id,
                youtube_video_id=yt_ref.youtube_video_id,
                publish_datetime=publish_dt,
            )
            scheduled_publish_dt = publish_dt
            logger.info(
                "start_pipeline: auto-scheduled video_id=%s for %s",
                video_id, publish_dt.strftime("%Y-%m-%d %H:%M UTC"),
            )
        except Exception as exc:
            logger.warning(
                "start_pipeline: could not auto-schedule video_id=%s (%s) — staying Unlisted",
                video_id, exc,
            )

        # ------------------------------------------------------------------ #
        # Post-upload enhancements: Shorts extraction + end-screen            #
        # ------------------------------------------------------------------ #

        # Extract a Short clip and upload it (best-effort, non-blocking)
        try:
            mp4_source = visual.mp4_url or visual.mp4_path
            short_id = await self._publisher.extract_and_upload_short(
                video_id=video_id,
                mp4_url=mp4_source,
                metadata=metadata,
                full_video_id=yt_ref.youtube_video_id,
                publish_at=scheduled_publish_dt,
            )
            if short_id:
                schedule_note = ""
                if scheduled_publish_dt:
                    schedule_note = f" (scheduled: {scheduled_publish_dt.strftime('%Y-%m-%d %H:%M UTC')})"
                logger.info("start_pipeline: Short uploaded — %s%s", short_id, schedule_note)
        except Exception as exc:
            logger.warning("start_pipeline: Shorts extraction failed (non-fatal): %s", exc)

        # Upload Instagram Reel (separate encoding from Shorts, synced schedule)
        if self._instagram_reels_client and self._config.instagram_reels.enabled:
            try:
                # 1. Encode video for Instagram Reels using asset_store.read()
                #    (avoids the flaky _fetch_bytes path that has SSL issues)
                import pathlib as _pl  # noqa: PLC0415
                import subprocess as _sp  # noqa: PLC0415
                import tempfile as _tmp  # noqa: PLC0415
                import re as _re  # noqa: PLC0415

                logger.info("Instagram Reel: downloading source video from Drive...")
                source_bytes = await self._asset_store.read(
                    video_id=video_id,
                    subfolder=SubFolder.VIDEOS,
                    filename=f"{video_id}_v1.mp4",
                )

                if not source_bytes or len(source_bytes) < 100_000:
                    logger.warning(
                        "Instagram Reel: source video too small (%d bytes) — skipping",
                        len(source_bytes) if source_bytes else 0,
                    )
                    raise Exception("Source video too small or missing")

                logger.info(
                    "Instagram Reel: source video downloaded (%.2f MB)",
                    len(source_bytes) / (1024 * 1024),
                )

                # Encode for Reels (60s, 1080x1920, 2Mbps)
                tmpdir = _tmp.mkdtemp(prefix="ig_reel_")
                full_path = _pl.Path(tmpdir) / "full.mp4"
                reel_path = _pl.Path(tmpdir) / "reel.mp4"
                full_path.write_bytes(source_bytes)

                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(full_path),
                    "-t", "60",
                    "-vf", (
                        "scale=1080:1920:force_original_aspect_ratio=increase,"
                        "crop=1080:1920,"
                        "setsar=1,"
                        "fps=30"
                    ),
                    "-c:v", "libx264",
                    "-profile:v", "high",
                    "-level:v", "4.0",
                    "-b:v", "2M",
                    "-maxrate", "2.5M",
                    "-bufsize", "4M",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "44100",
                    "-ac", "2",
                    "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p",
                    str(reel_path),
                ]

                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: _sp.run(ffmpeg_cmd, capture_output=True, timeout=180, check=True),
                )

                reel_bytes = reel_path.read_bytes()
                logger.info(
                    "Instagram Reel: encoded file size = %.2f MB",
                    len(reel_bytes) / (1024 * 1024),
                )

                # 2. Upload to Drive
                reel_drive_url = await self._asset_store.write(
                    video_id=video_id,
                    subfolder=SubFolder.VIDEOS,
                    filename="reel.mp4",
                    content=reel_bytes,
                )
                logger.info("Instagram Reel: uploaded to Drive → %s", reel_drive_url)

                # Build Drive API v3 direct media URL
                drive_id_match = _re.search(r"/file/d/([^/]+)", reel_drive_url)
                gcloud_key = os.environ.get("GOOGLE_CLOUD_API_KEY", "")
                logger.info(
                    "Instagram Reel: Drive file_id=%s, GOOGLE_CLOUD_API_KEY set=%s (len=%d)",
                    drive_id_match.group(1) if drive_id_match else "NOT_FOUND",
                    bool(gcloud_key),
                    len(gcloud_key),
                )

                if drive_id_match and gcloud_key:
                    file_id = drive_id_match.group(1)
                    reel_public_url = (
                        f"https://www.googleapis.com/drive/v3/files/{file_id}"
                        f"?alt=media&key={gcloud_key}"
                    )
                    logger.info("Instagram Reel: using Drive API v3 URL (direct media)")
                elif drive_id_match:
                    file_id = drive_id_match.group(1)
                    reel_public_url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                    logger.warning("Instagram Reel: GOOGLE_CLOUD_API_KEY not set! Falling back")
                else:
                    reel_public_url = reel_drive_url
                    logger.warning("Instagram Reel: could not extract file_id from Drive URL")

                logger.info("Instagram Reel: final URL = %s", reel_public_url[:100])

                # Wait for Drive CDN propagation
                logger.info("Instagram Reel: waiting 60s for Drive CDN propagation...")
                await asyncio.sleep(60)

                youtube_full_url = f"https://www.youtube.com/watch?v={yt_ref.youtube_video_id}"
                thumbnail_url = visual.thumbnail_url or None

                # 3. Upload/schedule Reel at the same time as YouTube publish
                reel_result = await upload_reel_from_short(
                    client=self._instagram_reels_client,
                    video_public_url=reel_public_url,
                    metadata=metadata,
                    youtube_url=youtube_full_url,
                    thumbnail_url=thumbnail_url,
                    extra_hashtags=self._config.instagram_reels.extra_hashtags,
                    scheduled_publish_time=scheduled_publish_dt,
                )

                if reel_result.success:
                    schedule_note = ""
                    if scheduled_publish_dt:
                        schedule_note = f" (scheduled for {scheduled_publish_dt.strftime('%Y-%m-%d %H:%M UTC')})"
                    logger.info(
                        "start_pipeline: Instagram Reel ready — %s (%s)%s",
                        reel_result.reel_id, reel_result.permalink, schedule_note,
                    )
                else:
                    logger.warning(
                        "start_pipeline: Instagram Reel failed — %s", reel_result.error,
                    )

                # 4. Cleanup temp files
                import shutil as _shutil  # noqa: PLC0415
                _shutil.rmtree(tmpdir, ignore_errors=True)

            except Exception as exc:
                logger.warning("start_pipeline: Instagram Reels upload failed (non-fatal): %s", exc)

        # Add end-screen linking to best-performing video (best-effort)
        try:
            await self._publisher.add_end_screen(yt_ref.youtube_video_id)
        except Exception as exc:
            logger.warning("start_pipeline: end-screen addition failed (non-fatal): %s", exc)

        # Send post-upload notification with YouTube unlisted URL
        self._notifier.send_review_gate(
            video_id=video_id,
            gate_type="final",
            asset_links=[yt_ref.unlisted_url],
            action_prompt=(
                f"✅ Video uploaded successfully!\n\n"
                f"Title: {metadata.title}\n"
                f"YouTube URL: {yt_ref.unlisted_url}\n\n"
                f"The video is Unlisted and will auto-publish on its scheduled date.\n"
                f"Review it at the link above before it goes public."
            ),
        )

        # ------------------------------------------------------------------ #
        # Stage 8: Cross_Poster                                               #
        # ------------------------------------------------------------------ #
        await self._run_stage(
            stage_name="cross_poster",
            video_id=video_id,
            run_id=run_id,
            pre_status=None,  # status updated inside Cross_Poster / Publisher
            coro_factory=lambda: self._cross_poster.post(
                video_url=yt_ref.unlisted_url,
                metadata=metadata,
                platforms=[],  # populated from config in full implementation
            ),
        )

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Pipeline completed: run_id={run_id}, video_id={video_id}",
            video_id=video_id,
        )
        await self._flush_log(run_id, video_id)

        logger.info("Orchestrator.start_pipeline completed: run_id=%s", run_id)
        return run_id

    # ------------------------------------------------------------------
    # Task 16.3 — resume_pipeline
    # ------------------------------------------------------------------

    async def resume_pipeline(self, run_id: str, video_id: str) -> None:
        """Resume a previously-failed pipeline run from the last good stage.

        Algorithm
        ---------
        For each stage (in sequence), check whether its expected output artefact
        is present and readable in Asset_Store:

        * If the artefact **exists and is readable** → the stage has already
          completed; skip it and advance the resume cursor.
        * If the artefact **is missing or unreadable** → resume (restart) from
          this stage and execute all subsequent stages normally.

        Expected output artefacts per stage
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ============  ================================  ====================
        Stage         Subfolder                         Filename pattern
        ============  ================================  ====================
        script_writer scripts                           ``script_v1.md``
        narration     narration                         ``{video_id}_v1.mp3``
        visual        videos                            ``{video_id}_v1.mp4``
        metadata      metadata                          ``{video_id}.json``
        ============  ================================  ====================

        ``reference_analyzer`` and ``topic_researcher`` do not produce
        video-scoped artefacts that can be easily probed by ``video_id``, so
        they are always re-run when the resume cursor reaches them.

        Args:
            run_id: The pipeline run identifier returned by ``start_pipeline``.
            video_id: The video identifier associated with this run.
        """
        logger.info(
            "Orchestrator.resume_pipeline: run_id=%s video_id=%s", run_id, video_id
        )
        self._log_entries = []

        # Guard: don't resume videos that were intentionally rejected or already done
        try:
            current_status = await self._content_calendar._get_status(video_id)
            skip_statuses = {
                PipelineStatus.SCRIPT_REJECTED,
                PipelineStatus.PUBLISHED,
                PipelineStatus.SCHEDULED,
                PipelineStatus.PIPELINE_ERROR,
            }
            if current_status in skip_statuses:
                logger.info(
                    "resume_pipeline: skipping video_id=%s — status is %s",
                    video_id, current_status.value,
                )
                return
        except Exception as exc:
            logger.warning("resume_pipeline: could not check status for %s: %s", video_id, exc)

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Resuming pipeline: run_id={run_id}, video_id={video_id}",
            video_id=video_id,
        )

        # ------------------------------------------------------------------ #
        # Probe each restorable stage artefact.                               #
        # ------------------------------------------------------------------ #

        # Stage: script_writer — probe script_v1.md
        script_bytes = await self._probe_asset(
            video_id, SubFolder.SCRIPTS, "script_v1.md"
        )
        # Stage: narration — probe {video_id}_v1.mp3
        narration_bytes = await self._probe_asset(
            video_id, SubFolder.NARRATION, f"{video_id}_v1.mp3"
        )
        # Stage: visual_generator — probe {video_id}_v1.mp4
        visual_bytes = await self._probe_asset(
            video_id, SubFolder.VIDEOS, f"{video_id}_v1.mp4"
        )
        # Stage: metadata_generator — probe {video_id}.json
        metadata_bytes = await self._probe_asset(
            video_id, SubFolder.METADATA, f"{video_id}.json"
        )

        # Determine resume cursor: the first stage whose artefact is absent.
        if script_bytes is None:
            resume_from = "script_writer"
        elif narration_bytes is None:
            resume_from = "narration_generator"
        elif visual_bytes is None:
            resume_from = "visual_generator"
        elif metadata_bytes is None:
            resume_from = "metadata_generator"
        else:
            # All generation artefacts are present; resume at publisher.
            resume_from = "publisher"

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Resume cursor: restarting from stage '{resume_from}'",
            video_id=video_id,
        )
        logger.info(
            "Orchestrator.resume_pipeline: resuming from stage=%s (run_id=%s)",
            resume_from,
            run_id,
        )

        # ------------------------------------------------------------------ #
        # Re-run stages from the resume cursor.                                #
        # We always re-fetch the style_profile and topics (no video-scoped    #
        # artefact probe available for these stages).                         #
        # ------------------------------------------------------------------ #

        # Stages before script_writer are always re-executed when we need them.
        style_profile: Optional[StyleProfile] = None
        topics: Optional[list[TopicEntry]] = None
        script: Optional[Script] = None
        narration: Optional[NarrationAsset] = None
        visual: Optional[VisualAsset] = None
        metadata: Optional[MetadataPackage] = None

        stages_that_need_ra_tr = {
            "script_writer", "narration_generator",
            "visual_generator", "metadata_generator", "publisher",
        }

        if resume_from in stages_that_need_ra_tr:
            # Re-run reference analysis (cheap, always needed for style context).
            style_profile = await self._run_stage(
                stage_name="reference_analyzer",
                video_id=video_id,
                run_id=run_id,
                pre_status=PipelineStatus.RESEARCHING,
                coro_factory=lambda: self._reference_analyzer.analyze(
                    channel_url=self._config.reference_channel_url
                ),
            )

            # Only re-run topic research when we actually need to generate a
            # new script.  For later stages the script already exists and its
            # content is the source of truth — generating fresh topics risks
            # producing metadata that doesn't match the already-written script.
            if resume_from == "script_writer":
                topics = await self._run_stage(
                    stage_name="topic_researcher",
                    video_id=video_id,
                    run_id=run_id,
                    pre_status=PipelineStatus.RESEARCHING,
                    coro_factory=lambda: self._topic_researcher.research(
                        batch_size=1, excluded_titles=[], run_id=run_id
                    ),
                )
            else:
                # Derive a synthetic TopicEntry from the stored script content
                # so metadata generation stays consistent with the video.
                assert script_bytes is not None
                _script = _reconstruct_script(video_id, script_bytes)
                topics = [_derive_topic_from_script(_script)]
                logger.info(
                    "resume_pipeline: derived topic from script: '%s'",
                    topics[0].title,
                )

        # Stage: script_writer
        if resume_from in {"script_writer", "narration_generator",
                           "visual_generator", "metadata_generator", "publisher"}:
            if resume_from == "script_writer":
                # Re-generate the script.
                script = await self._run_stage(
                    stage_name="script_writer",
                    video_id=video_id,
                    run_id=run_id,
                    pre_status=PipelineStatus.SCRIPTING,
                    coro_factory=lambda: self._script_writer.generate(
                        topic=topics[0],  # type: ignore[index]
                        style_profile=style_profile,  # type: ignore[arg-type]
                        video_id=video_id,
                        script_duration_minutes=self._config.script_duration_minutes,
                    ),
                )
                await self._update_calendar_status(
                    video_id, PipelineStatus.SCRIPT_APPROVED, run_id
                )
            else:
                # Reconstruct Script object from the stored bytes.
                assert script_bytes is not None
                script = _reconstruct_script(video_id, script_bytes)

        # Stage: narration_generator
        if resume_from in {"narration_generator", "visual_generator",
                           "metadata_generator", "publisher"}:
            if resume_from == "narration_generator":
                narration = await self._run_stage(
                    stage_name="narration_generator",
                    video_id=video_id,
                    run_id=run_id,
                    pre_status=PipelineStatus.NARRATION_READY,
                    coro_factory=lambda: self._narration_generator.generate(
                        script=script,  # type: ignore[arg-type]
                        voice_id=self._config.voice_id,
                        video_id=video_id,
                    ),
                )
                # Update Notion with narration Drive URL
                if getattr(narration, "asset_url", None):
                    try:
                        await self._content_calendar.update_asset_link(video_id, "narration", narration.asset_url)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("resume: could not update narration_url for %s: %s", video_id, exc)
            else:
                assert narration_bytes is not None
                narration = _reconstruct_narration(video_id)

        # Stage: visual_generator
        if resume_from in {"visual_generator", "metadata_generator", "publisher"}:
            if resume_from == "visual_generator":
                visual = await self._run_stage(
                    stage_name="visual_generator",
                    video_id=video_id,
                    run_id=run_id,
                    pre_status=PipelineStatus.GENERATING_VISUALS,
                    coro_factory=lambda: self._visual_generator.generate(
                        script=script,  # type: ignore[arg-type]
                        narration=narration,  # type: ignore[arg-type]
                        style_profile=style_profile,  # type: ignore[arg-type]
                        video_id=video_id,
                    ),
                )
                await self._update_calendar_status(
                    video_id, PipelineStatus.VISUALS_READY, run_id
                )
                # Update Notion with video and thumbnail Drive URLs
                try:
                    if getattr(visual, "mp4_url", None):
                        await self._content_calendar.update_asset_link(video_id, "video", visual.mp4_url)
                    if getattr(visual, "thumbnail_url", None):
                        await self._content_calendar.update_asset_link(video_id, "thumbnail", visual.thumbnail_url)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("resume: could not update video/thumbnail URLs for %s: %s", video_id, exc)
            else:
                assert visual_bytes is not None
                visual = _reconstruct_visual(video_id)
                # Resolve Drive URLs so the publisher can upload correctly
                try:
                    mp4_url = await self._asset_store.url(
                        video_id=video_id,
                        subfolder=SubFolder.VIDEOS,
                        filename=f"{video_id}_v1.mp4",
                    )
                    thumb_url = await self._asset_store.url(
                        video_id=video_id,
                        subfolder=SubFolder.THUMBNAILS,
                        filename=f"{video_id}_v1.jpg",
                    )
                    visual = visual.model_copy(update={"mp4_url": mp4_url, "thumbnail_url": thumb_url})
                    logger.info("resume_pipeline: resolved Drive URLs for video_id=%s", video_id)
                except Exception as url_exc:
                    logger.warning("resume_pipeline: could not resolve Drive URLs for %s: %s", video_id, url_exc)
        # Stage: metadata_generator
        if resume_from in {"metadata_generator", "publisher"}:
            if resume_from == "metadata_generator":
                metadata = await self._run_stage(
                    stage_name="metadata_generator",
                    video_id=video_id,
                    run_id=run_id,
                    pre_status=PipelineStatus.GENERATING_METADATA,
                    coro_factory=lambda: self._metadata_generator.generate(
                        script=script,  # type: ignore[arg-type]
                        topics=topics,  # type: ignore[arg-type]
                        video_id=video_id,
                    ),
                )
                # Update Notion with metadata Drive URL
                if getattr(metadata, "asset_url", None):
                    try:
                        await self._content_calendar.update_asset_link(video_id, "metadata", metadata.asset_url)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("resume: could not update metadata_url for %s: %s", video_id, exc)
            else:
                assert metadata_bytes is not None
                metadata = _reconstruct_metadata(video_id, metadata_bytes)

        # Stage: publisher
        if resume_from == "publisher":
            await self._update_calendar_status(
                video_id, PipelineStatus.APPROVED_FOR_UPLOAD, run_id
            )
            yt_ref: YouTubeVideoRef = await self._run_stage(
                stage_name="publisher",
                video_id=video_id,
                run_id=run_id,
                pre_status=PipelineStatus.UPLOADING,
                coro_factory=lambda: self._publisher.upload(
                    video_id=video_id,
                    assets=visual,  # type: ignore[arg-type]
                    metadata=metadata,  # type: ignore[arg-type]
                ),
            )
            await self._update_calendar_status(video_id, PipelineStatus.UNLISTED, run_id)

            # Store the YouTube video ID so the batch scheduler can find it
            try:
                await self._content_calendar.set_youtube_url(
                    video_id=video_id,
                    youtube_video_id=yt_ref.youtube_video_id,
                    unlisted_url=yt_ref.unlisted_url,
                )
            except Exception as exc:
                logger.warning(
                    "resume_pipeline: could not store YouTube video ID for %s: %s",
                    video_id, exc,
                )

            # Patch the metadata JSON in Drive with the YouTube video ID
            try:
                metadata = await self._metadata_generator.patch_youtube_id(
                    package=metadata,  # type: ignore[arg-type]
                    youtube_video_id=yt_ref.youtube_video_id,
                )
            except Exception as exc:
                logger.warning(
                    "resume_pipeline: could not patch youtube_video_id into metadata JSON for %s: %s",
                    video_id, exc,
                )

            # Auto-schedule in the next free slot (same as start_pipeline)
            try:
                slots = await self._find_next_free_slot(n_slots=1)
                publish_dt = slots[0]
                await self._content_calendar.set_publish_datetime(
                    video_id=video_id,
                    dt=publish_dt,
                )
                await self._publisher.schedule(
                    video_id=video_id,
                    youtube_video_id=yt_ref.youtube_video_id,
                    publish_datetime=publish_dt,
                )
                logger.info(
                    "resume_pipeline: auto-scheduled video_id=%s for %s",
                    video_id, publish_dt.strftime("%Y-%m-%d %H:%M UTC"),
                )
            except Exception as exc:
                logger.warning(
                    "resume_pipeline: could not auto-schedule video_id=%s (%s) — staying Unlisted",
                    video_id, exc,
                )

            # Cross_Poster
            await self._run_stage(
                stage_name="cross_poster",
                video_id=video_id,
                run_id=run_id,
                pre_status=None,
                coro_factory=lambda: self._cross_poster.post(
                    video_url=yt_ref.unlisted_url,
                    metadata=metadata,  # type: ignore[arg-type]
                    platforms=[],
                ),
            )

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Resume completed: run_id={run_id}, video_id={video_id}",
            video_id=video_id,
        )
        await self._flush_log(run_id, video_id)

        # Record pipeline end time in Notion
        try:
            await self._content_calendar.set_pipeline_end_time(video_id)
        except Exception as exc:
            logger.warning("Could not set pipeline_end_time for %s: %s", video_id, exc)

        logger.info("Orchestrator.resume_pipeline completed: run_id=%s", run_id)

    # ------------------------------------------------------------------
    # handle_review_response — Review Gate handler
    # ------------------------------------------------------------------

    async def handle_review_response(
        self,
        video_id: str,
        gate: Literal["script", "final"],
        action: Literal["approve", "edit", "regenerate"],
        payload: Optional[str] = None,
    ) -> None:
        """Handle a creator's review response for Gate 1 (script) or Gate 2 (final).

        Simplified implementation: reads the gate type and action, calls the
        appropriate subsystem method, and updates the Content_Calendar status.

        Gate 1 — Script:
          - ``approve``: advance status to ``Script Approved``.
          - ``edit``: call ``Script_Writer.revise`` with *payload* as the edit
            instructions; update status to ``Scripting``.
          - ``regenerate``: re-run ``Script_Writer.generate`` for the current
            topic; update status to ``Scripting``.

        Gate 2 — Final:
          - ``approve``: advance status to ``Approved for Upload``.
          - ``regenerate`` (``payload`` names the asset: ``"script"``,
            ``"narration"``, ``"video"``, ``"thumbnail"``, ``"metadata"``):
            update status to the corresponding ``Generating …`` status so the
            Orchestrator re-enters that stage on the next poll cycle.

        Args:
            video_id: Pipeline video identifier.
            gate: ``"script"`` for Gate 1, ``"final"`` for Gate 2.
            action: Creator's chosen action.
            payload: Action-specific data (edit text, or asset name to regenerate).
        """
        logger.info(
            "handle_review_response: video_id=%s gate=%s action=%s",
            video_id, gate, action,
        )

        if gate == "script":
            if action == "approve":
                await self._content_calendar.update_status(
                    video_id, PipelineStatus.SCRIPT_APPROVED
                )
                logger.info("Gate 1 approved for video_id=%s", video_id)

            elif action == "edit":
                if not payload or not payload.strip():
                    raise ValueError(
                        f"handle_review_response: edit payload is empty for video_id={video_id}"
                    )
                # Re-enter Scripting; script revision is handled by the caller
                # providing the edits — we update status and leave the revision
                # to the orchestrator's next stage execution.
                await self._content_calendar.update_status(
                    video_id, PipelineStatus.SCRIPTING
                )
                logger.info("Gate 1 edit submitted for video_id=%s", video_id)

            elif action == "regenerate":
                # Re-enter Scripting for full re-generation.
                await self._content_calendar.update_status(
                    video_id, PipelineStatus.SCRIPTING
                )
                logger.info("Gate 1 regenerate requested for video_id=%s", video_id)

        elif gate == "final":
            if action == "approve":
                await self._content_calendar.update_status(
                    video_id, PipelineStatus.APPROVED_FOR_UPLOAD
                )
                logger.info("Gate 2 approved for video_id=%s", video_id)

            elif action == "regenerate":
                # Map asset name to re-entry status.
                _regen_status_map: dict[str, PipelineStatus] = {
                    "script": PipelineStatus.SCRIPTING,
                    "narration": PipelineStatus.SCRIPT_APPROVED,
                    "video": PipelineStatus.GENERATING_VISUALS,
                    "thumbnail": PipelineStatus.GENERATING_VISUALS,
                    "metadata": PipelineStatus.GENERATING_METADATA,
                }
                asset_name = (payload or "").strip().lower()
                new_status = _regen_status_map.get(asset_name)
                if new_status is None:
                    raise ValueError(
                        f"handle_review_response: unknown regenerate asset "
                        f"'{asset_name}' for video_id={video_id}. "
                        f"Valid assets: {list(_regen_status_map.keys())}"
                    )
                await self._content_calendar.update_status(video_id, new_status)
                logger.info(
                    "Gate 2 regenerate asset='%s' → status=%s for video_id=%s",
                    asset_name,
                    new_status.value,
                    video_id,
                )

    # ------------------------------------------------------------------
    # Task 16.4 — create_review_gate factory
    # ------------------------------------------------------------------

    def create_review_gate(
        self,
        gate_type: Literal["script", "final"],
        video_id: str,
    ) -> ReviewGate:
        """Create a :class:`ReviewGate` wired to this Orchestrator's dependencies.

        This factory method is the canonical way to instantiate ``ReviewGate``
        objects inside the pipeline.  It injects ``content_calendar`` and
        ``notifier`` from the Orchestrator's own attributes so callers don't
        need to manage those dependencies directly.

        Args:
            gate_type: ``"script"`` for Gate 1, ``"final"`` for Gate 2.
            video_id: Pipeline video identifier this gate belongs to.

        Returns:
            A new :class:`~pipeline.orchestrator.review_gate.ReviewGate`
            instance ready to be ``trigger()``-ed.
        """
        return ReviewGate(
            gate_type=gate_type,
            video_id=video_id,
            content_calendar=self._content_calendar,
            notifier=self._notifier,
        )

    # ------------------------------------------------------------------
    # schedule_video
    # ------------------------------------------------------------------

    async def schedule_video(self, video_id: str, publish_datetime: datetime) -> None:
        """Confirm a publish datetime for an unlisted video and schedule it.

        Delegates to ``Publisher.schedule`` after resolving the YouTube video
        ID from the Content_Calendar.  Updates Content_Calendar to ``Scheduled``
        on success.

        Args:
            video_id: Pipeline video identifier.
            publish_datetime: UTC datetime at which the video should go public.
        """
        # Publisher.schedule handles datetime validation and calendar update.
        # We pass a placeholder youtube_video_id here; the real implementation
        # would resolve it from the Content_Calendar or a local cache.
        logger.info(
            "schedule_video: video_id=%s publish_datetime=%s",
            video_id,
            publish_datetime.isoformat(),
        )
        await self._publisher.schedule(
            video_id=video_id,
            youtube_video_id=video_id,  # resolved from calendar in production
            publish_datetime=publish_datetime,
        )

    # ------------------------------------------------------------------
    # Tasks 17.1 / 17.2 — Batch processing
    # ------------------------------------------------------------------

    async def start_batch(self, n: int) -> str:
        """Start a batch pipeline run for *n* videos processed sequentially.

        Steps
        -----
        1. Validate *n* ∈ [2, 10]; raise :class:`ValueError` and create zero
           records if out of range.
        2. Generate a UUID4 ``batch_id``.
        3. Create *n* ``VideoRecord`` entries in Content_Calendar under
           ``batch_id`` using deterministic ``video_id`` values of the form
           ``video-{batch_id[:8]}-{i}`` (0-indexed).
        4. Call ``Topic_Researcher.research(batch_size=n, excluded=past_30_days)``
           to obtain a ranked topic list.
        5. Process videos one at a time: run all generation stages for video *i*
           through ``Awaiting Final Review`` before starting video *i+1*.
           Review Gate responses do not block subsequent video generation.
        6. Stage failure for video *i*: mark ``Pipeline Error — {stage_name}``,
           continue with video *i+1*, and notify the Notifier.

        Args:
            n: Number of videos in the batch (must be 2–10 inclusive).

        Returns:
            The ``batch_id`` (UUID4 string) for this batch run.

        Raises:
            ValueError: When *n* is outside [2, 10].
        """
        # Step 1: validate
        validate_batch_size(n)

        # Step 2: generate batch_id
        batch_id: str = str(uuid.uuid4())
        logger.info("Orchestrator.start_batch: batch_id=%s n=%d", batch_id, n)

        self._log_entries = []

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Batch started: batch_id={batch_id}, n={n}",
            batch_id=batch_id,
        )

        # Step 3: derive video IDs and create Content_Calendar records
        video_ids: list[str] = [
            f"video-{batch_id[:8]}-{i}" for i in range(n)
        ]
        for video_id in video_ids:
            try:
                await self._content_calendar.create_record(
                    video_id=video_id, batch_id=batch_id
                )
            except Exception as exc:  # noqa: BLE001
                self._emit_log(
                    event_type="error",
                    stage_name="orchestrator",
                    message=(
                        f"Failed to create Content_Calendar record for "
                        f"video_id={video_id}: {exc}"
                    ),
                    batch_id=batch_id,
                )
                # Non-fatal at this point — log and continue creating others.
                logger.error(
                    "start_batch: could not create record for video_id=%s: %s",
                    video_id,
                    exc,
                )

        # Step 4: research topics for the full batch (exclude past-30-day topics)
        run_id = batch_id  # reuse batch_id as the research run identifier
        excluded_titles: list[str] = await self._content_calendar.get_batch_topics(
            batch_id=batch_id, lookback_days=180  # exclude topics used in past 90 days
        )

        # Also include already-uploaded YouTube video titles
        try:
            yt_titles = await self._publisher._yt.list_uploaded_titles()  # noqa: SLF001
            excluded_titles = list({*excluded_titles, *yt_titles})
            logger.info(
                "start_batch: exclusion list has %d titles (%d from YouTube)",
                len(excluded_titles), len(yt_titles),
            )
        except Exception as exc:
            logger.warning("start_batch: could not fetch YouTube titles for exclusion: %s", exc)

        style_profile: StyleProfile = await self._run_stage(
            stage_name="reference_analyzer",
            video_id=video_ids[0],
            run_id=run_id,
            pre_status=PipelineStatus.RESEARCHING,
            coro_factory=lambda: self._reference_analyzer.analyze(
                channel_url=self._config.reference_channel_url
            ),
        )

        topics: list[TopicEntry] = await self._run_stage(
            stage_name="topic_researcher",
            video_id=video_ids[0],
            run_id=run_id,
            pre_status=PipelineStatus.RESEARCHING,
            coro_factory=lambda: self._topic_researcher.research(
                batch_size=n,
                excluded_titles=excluded_titles,
                run_id=run_id,
            ),
        )

        # Step 5 & 6: sequential video processing with failure isolation
        for i, video_id in enumerate(video_ids):
            topic = topics[i] if i < len(topics) else topics[-1]
            self._emit_log(
                event_type="stage_transition",
                stage_name="orchestrator",
                message=(
                    f"Batch video {i + 1}/{n} starting: video_id={video_id}"
                ),
                video_id=video_id,
                batch_id=batch_id,
            )
            try:
                await self._start_pipeline_for_video(
                    video_id=video_id,
                    topic=topic,
                    style_profile=style_profile,
                    batch_id=batch_id,
                    run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001
                # Failure isolation: log, notify, and continue to video i+1.
                logger.error(
                    "start_batch: video_id=%s failed — continuing with next video: %s",
                    video_id,
                    exc,
                )
                self._emit_log(
                    event_type="error",
                    stage_name="orchestrator",
                    message=(
                        f"Batch video {i + 1}/{n} failed for video_id={video_id}: {exc}. "
                        "Continuing with next video."
                    ),
                    video_id=video_id,
                    batch_id=batch_id,
                )
                # Notifier already called by _run_stage; flush log and move on.
                await self._flush_log(run_id, video_id)
                continue

        self._emit_log(
            event_type="stage_transition",
            stage_name="orchestrator",
            message=f"Batch completed: batch_id={batch_id}, n={n}",
            batch_id=batch_id,
        )
        logger.info("Orchestrator.start_batch completed: batch_id=%s", batch_id)

        # Auto-schedule all videos at configured publish time
        if n >= 1:
            await self._auto_schedule_weekly_batch(batch_id, video_ids)

        return batch_id

    async def get_batch_schedule(self, batch_id: str) -> list[datetime]:
        """Return evenly-spaced publish datetimes for all videos in *batch_id*.

        When all videos in the batch have passed their Final Review Gate the
        Orchestrator proposes a schedule spanning 14 days.

        Formula (design §11.4):
            ``slot[i] = now + i * timedelta(days=14) / (N − 1)``

        Args:
            batch_id: The batch identifier (UUID4 string).

        Returns:
            A list of :class:`datetime` objects, one per video in the batch,
            sorted in ascending order.  Returns an empty list when *batch_id*
            has no associated video records.
        """
        # Determine batch size from Content_Calendar records.
        all_topics = await self._content_calendar.get_batch_topics(
            batch_id=batch_id, lookback_days=365  # wide window to find all records
        )
        n = len(all_topics)
        if n == 0:
            logger.warning(
                "get_batch_schedule: no records found for batch_id=%s", batch_id
            )
            return []

        now = _utcnow()
        slots = generate_batch_slots(n, now)
        logger.info(
            "get_batch_schedule: batch_id=%s n=%d first=%s last=%s",
            batch_id,
            n,
            slots[0].isoformat() if slots else "—",
            slots[-1].isoformat() if slots else "—",
        )
        return slots

    async def _start_pipeline_for_video(
        self,
        video_id: str,
        topic: TopicEntry,
        style_profile: StyleProfile,
        batch_id: str,
        run_id: str,
    ) -> None:
        """Run all generation stages for a single video in the batch.

        Executes the ordered stage sequence from Script_Writer through
        ``Awaiting Final Review``.  Review Gate responses do not block this
        coroutine — the gate is triggered and the method returns so the next
        video can begin generation.

        After each stage completes the Content_Calendar status is updated and
        a structured log entry is emitted.  Any unrecovered stage failure
        propagates to the caller (``start_batch``), which applies failure
        isolation.

        Args:
            video_id: Pipeline video identifier for this specific video.
            topic: The :class:`~pipeline.models.TopicEntry` assigned to this video.
            style_profile: Shared :class:`~pipeline.models.StyleProfile` for the batch.
            batch_id: Parent batch identifier (for log entries).
            run_id: Pipeline run identifier (reused from batch_id).
        """
        # Persist chosen topic to Notion so future runs can exclude it
        try:
            await self._content_calendar.update_topic(video_id, topic.title)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save topic to Notion for video_id=%s: %s", video_id, exc)

        # Stage: Script_Writer
        script: Script = await self._run_stage(
            stage_name="script_writer",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.SCRIPTING,
            coro_factory=lambda: self._script_writer.generate(
                topic=topic,
                style_profile=style_profile,
                video_id=video_id,
                script_duration_minutes=self._config.script_duration_minutes,
            ),
        )

        # Update Notion with script Drive URL
        if getattr(script, "asset_url", None):
            try:
                await self._content_calendar.update_asset_link(video_id, "script", script.asset_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update script_url in Notion for %s: %s", video_id, exc)

        # Trigger Review Gate 1 (non-blocking — does not halt generation)
        await self._update_calendar_status(
            video_id, PipelineStatus.AWAITING_SCRIPT_REVIEW, run_id
        )
        self._emit_log(
            event_type="stage_transition",
            stage_name="review_gate_1",
            message="Review Gate 1 open — awaiting script approval (batch)",
            video_id=video_id,
            batch_id=batch_id,
        )
        gate1 = self.create_review_gate(gate_type="script", video_id=video_id)
        await gate1.trigger(
            asset_links=[script.asset_url] if getattr(script, "asset_url", None) else [],
        )
        # Batch mode: do NOT await gate1.poll_until_action() here — that would
        # block the next video from starting.  The gate notification is sent;
        # the creator can approve via webhook or Notion at any point.

        # Advance past gate (batch mode: gates do not block generation).
        await self._update_calendar_status(video_id, PipelineStatus.SCRIPT_APPROVED, run_id)

        # Stage: Narration_Generator
        narration: NarrationAsset = await self._run_stage(
            stage_name="narration_generator",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.NARRATION_READY,
            coro_factory=lambda: self._narration_generator.generate(
                script=script,
                voice_id=self._config.voice_id,
                video_id=video_id,
            ),
        )

        # Update Notion with narration Drive URL
        if getattr(narration, "asset_url", None):
            try:
                await self._content_calendar.update_asset_link(video_id, "narration", narration.asset_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update narration_url in Notion for %s: %s", video_id, exc)

        # Stage: Visual_Generator
        visual: VisualAsset = await self._run_stage(
            stage_name="visual_generator",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.GENERATING_VISUALS,
            coro_factory=lambda: self._visual_generator.generate(
                script=script,
                narration=narration,
                style_profile=style_profile,
                video_id=video_id,
            ),
        )
        await self._update_calendar_status(video_id, PipelineStatus.VISUALS_READY, run_id)

        # Update Notion with video and thumbnail Drive URLs
        try:
            for asset_type, url in [("video", getattr(visual, "mp4_url", None)),
                                     ("thumbnail", getattr(visual, "thumbnail_url", None))]:
                if url:
                    await self._content_calendar.update_asset_link(video_id, asset_type, url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update video/thumbnail URLs in Notion for %s: %s", video_id, exc)

        # Stage: Metadata_Generator
        metadata: MetadataPackage = await self._run_stage(
            stage_name="metadata_generator",
            video_id=video_id,
            run_id=run_id,
            pre_status=PipelineStatus.GENERATING_METADATA,
            coro_factory=lambda: self._metadata_generator.generate(
                script=script,
                topics=[topic],
                video_id=video_id,
            ),
        )

        # Update Notion with metadata Drive URL
        if getattr(metadata, "asset_url", None):
            try:
                await self._content_calendar.update_asset_link(video_id, "metadata", metadata.asset_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not update metadata_url in Notion for %s: %s", video_id, exc)

        # Trigger Review Gate 2 (non-blocking)
        await self._update_calendar_status(
            video_id, PipelineStatus.AWAITING_FINAL_REVIEW, run_id
        )
        self._emit_log(
            event_type="stage_transition",
            stage_name="review_gate_2",
            message="Review Gate 2 open — awaiting final asset approval (batch)",
            video_id=video_id,
            batch_id=batch_id,
        )
        asset_links = [
            lnk for lnk in [
                getattr(visual, "mp4_url", None),
                getattr(visual, "thumbnail_url", None),
                getattr(script, "asset_url", None),
                getattr(narration, "asset_url", None),
            ] if lnk
        ]
        gate2 = self.create_review_gate(gate_type="final", video_id=video_id)
        await gate2.trigger(asset_links=asset_links)
        # Batch mode: do NOT await poll_until_action() — non-blocking.
        await self._flush_log(run_id, video_id)

        # Generation through Awaiting Final Review is complete for this video.
        # The caller (start_batch) may now start the next video's generation.
        logger.info(
            "_start_pipeline_for_video: video_id=%s reached Awaiting Final Review "
            "(batch_id=%s)",
            video_id,
            batch_id,
        )

    async def _find_next_free_slot(self, n_slots: int = 1) -> list[datetime]:
        """Return *n_slots* consecutive daily publish datetimes starting after
        the latest already-scheduled video on the channel.

        Steps:
        1. Query YouTube for all scheduled (upcoming) video datetimes.
        2. Find the latest one. If none exist, use tomorrow.
        3. Return n_slots datetimes, each one day apart, at the configured
           publish time (converted to UTC).

        Args:
            n_slots: Number of consecutive daily slots to return.

        Returns:
            List of UTC datetimes, length == n_slots, each at publish time.
        """
        from datetime import timedelta  # noqa: PLC0415

        # Try to determine optimal publish hour from YouTube Analytics (peak audience hour).
        # Falls back to configured local publish time if analytics unavailable.
        peak_hour_utc: Optional[int] = None
        try:
            peak_hour_utc = await self._publisher._yt.get_audience_peak_hour()  # noqa: SLF001
        except Exception as exc:
            logger.warning("_find_next_free_slot: analytics query failed: %s", exc)

        if peak_hour_utc is not None:
            publish_hour_utc = peak_hour_utc
            publish_minute_utc = 0  # publish on the hour during peak
            logger.info(
                "_find_next_free_slot: using analytics-derived peak hour %d UTC",
                publish_hour_utc,
            )
        else:
            # Fallback: convert configured local publish time to UTC
            publish_hour_local = self._config.weekly_publish_time_hour
            publish_minute_local = self._config.weekly_publish_time_minute
            tz_offset = self._config.timezone_offset_hours
            total_minutes = publish_hour_local * 60 + publish_minute_local - int(tz_offset * 60)
            publish_hour_utc = (total_minutes // 60) % 24
            publish_minute_utc = total_minutes % 60
            logger.info(
                "_find_next_free_slot: using configured publish time %02d:%02d UTC",
                publish_hour_utc, publish_minute_utc,
            )

        # Query YouTube for already-scheduled videos
        try:
            scheduled_dts = await self._publisher._yt.list_scheduled_videos()  # noqa: SLF001
        except Exception as exc:
            logger.warning("_find_next_free_slot: could not query YouTube schedules: %s", exc)
            scheduled_dts = []

        # Also check the local calendar for already-assigned slots — this catches
        # cases where YouTube hasn't yet reflected a slot assigned moments ago.
        calendar_dts: list[datetime] = []
        try:
            calendar_dts = await self._content_calendar.get_scheduled_datetimes()
            logger.info(
                "_find_next_free_slot: calendar returned %d scheduled dates: %s",
                len(calendar_dts),
                [dt.strftime("%Y-%m-%d") for dt in sorted(calendar_dts)],
            )
            scheduled_dts = list({*scheduled_dts, *calendar_dts})
        except Exception as exc:
            logger.warning("_find_next_free_slot: could not query calendar schedules: %s", exc)

        now_utc = _utcnow()

        if scheduled_dts:
            # Start the day after the latest scheduled video
            latest = max(scheduled_dts)
            base_date = latest.date()
            logger.info(
                "_find_next_free_slot: latest scheduled video is %s, starting from %s",
                latest.strftime("%Y-%m-%d"), (base_date + timedelta(days=1)).isoformat(),
            )
        else:
            # No videos scheduled yet — start from tomorrow
            base_date = now_utc.date()
            logger.info("_find_next_free_slot: no scheduled videos found, starting from tomorrow")

        # Build a set of already-taken dates (date only, ignoring time)
        taken_dates = {dt.date() for dt in scheduled_dts}
        logger.info(
            "_find_next_free_slot: taken_dates=%s",
            sorted(taken_dates),
        )

        slots: list[datetime] = []
        candidate_date = base_date + timedelta(days=1)
        while len(slots) < n_slots:
            # Skip dates that already have a video scheduled
            if candidate_date in taken_dates:
                candidate_date += timedelta(days=1)
                continue
            slot_dt = datetime(
                candidate_date.year, candidate_date.month, candidate_date.day,
                publish_hour_utc, publish_minute_utc, 0,
                tzinfo=timezone.utc,
            )
            # Must be at least 15 minutes in the future (YouTube requirement)
            if slot_dt <= now_utc + timedelta(minutes=15):
                candidate_date += timedelta(days=1)
                continue
            slots.append(slot_dt)
            taken_dates.add(candidate_date)  # mark as taken for subsequent slots
            candidate_date += timedelta(days=1)

        return slots

    async def _auto_schedule_weekly_batch(
        self, batch_id: str, video_ids: list[str]
    ) -> None:
        """Auto-schedule 7 videos to publish Mon-Sun at the configured time.

        1. Sets Notion calendar publish datetime for each video
        2. Calls Publisher.schedule() to set YouTube publishAt so videos
           automatically go Public at the scheduled time.
        """
        from datetime import timedelta  # noqa: PLC0415

        # Find n consecutive free slots starting after the last scheduled video
        slots = await self._find_next_free_slot(n_slots=len(video_ids))
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        logger.info(
            "_auto_schedule_weekly_batch: scheduling %d videos starting %s",
            len(video_ids), slots[0].strftime("%Y-%m-%d") if slots else "unknown",
        )

        for i, video_id in enumerate(video_ids):
            publish_dt = slots[i]
            day_name = day_names[publish_dt.weekday()]
            logger.info(
                "_auto_schedule_weekly_batch: %s → %s (%s)",
                video_id, publish_dt.isoformat(), day_name,
            )

            try:
                # 1. Set Notion calendar datetime
                await self._content_calendar.set_publish_datetime(
                    video_id=video_id,
                    dt=publish_dt,
                )

                # 2. Call YouTube API to set publishAt (video goes Public automatically)
                # Look up the YouTube video ID from Notion calendar
                try:
                    yt_video_id = await self._get_youtube_video_id(video_id)
                    if yt_video_id:
                        await self._publisher.schedule(
                            video_id=video_id,
                            youtube_video_id=yt_video_id,
                            publish_datetime=publish_dt,
                        )
                        logger.info(
                            "_auto_schedule_weekly_batch: YouTube scheduled %s → %s (%s)",
                            yt_video_id, publish_dt.strftime('%Y-%m-%d %H:%M UTC'), day_name,
                        )
                    else:
                        logger.warning(
                            "_auto_schedule_weekly_batch: no YouTube video ID found for %s — "
                            "video will stay Unlisted until manually scheduled",
                            video_id,
                        )
                except Exception as yt_exc:
                    logger.warning(
                        "_auto_schedule_weekly_batch: YouTube schedule failed for %s: %s — "
                        "Notion calendar updated but YouTube publishAt not set",
                        video_id, yt_exc,
                    )

                self._emit_log(
                    event_type="stage_transition",
                    stage_name="orchestrator",
                    message=(
                        f"Auto-scheduled {video_id} for {day_name} "
                        f"{publish_dt.strftime('%Y-%m-%d %H:%M UTC')}"
                    ),
                    video_id=video_id,
                    batch_id=batch_id,
                )
            except Exception as exc:
                logger.warning(
                    "_auto_schedule_weekly_batch: failed to schedule %s: %s",
                    video_id, exc,
                )

    async def _get_youtube_video_id(self, video_id: str) -> Optional[str]:
        """Look up the YouTube video ID for a pipeline video_id from the Content Calendar."""
        try:
            return await self._content_calendar.get_youtube_video_id(video_id)
        except Exception as exc:
            logger.debug("_get_youtube_video_id failed for %s: %s", video_id, exc)
            return None

    async def _update_batch_completion(
        self, batch_id: str, video_ids: list[str]
    ) -> None:
        """Recalculate and persist ``BatchRecord.completion_percentage``.

        Called after each video transitions to ``Published``.  Queries
        Content_Calendar for the published count across all *video_ids* in
        the batch and stores the updated floor percentage.

        Args:
            batch_id: Parent batch identifier.
            video_ids: Ordered list of all video IDs in the batch.
        """
        try:
            pct = await self._content_calendar.get_batch_completion(batch_id)
            logger.info(
                "_update_batch_completion: batch_id=%s completion=%d%%",
                batch_id,
                pct,
            )
            self._emit_log(
                event_type="stage_transition",
                stage_name="orchestrator",
                message=(
                    f"Batch completion updated: batch_id={batch_id}, "
                    f"completion_percentage={pct}"
                ),
                batch_id=batch_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "_update_batch_completion: failed for batch_id=%s: %s",
                batch_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Task 16.2 — _call_with_retry (general retry helper)
    # ------------------------------------------------------------------

    async def _call_with_retry(
        self,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
        stage_name: str,
        video_id: str,
    ) -> Any:
        """Execute *coro_factory()* with the general orchestrator retry policy.

        Retry policy (task 16.2):
        - Transient HTTP errors (429, 500, 502, 503, 504): retry up to 3 times
          with delays 5 s → 10 s → 20 s.
        - Non-transient HTTP errors (400, 401, 403, 404): fail immediately with
          no retry.
        - All other exceptions: treated as transient; retry up to 3 times.

        A ``LogEntry`` with ``event_type="retry_attempt"`` is emitted before
        each retry and a ``LogEntry`` with ``event_type="api_call"`` is emitted
        on the first successful call.

        Args:
            coro_factory: Zero-argument callable returning an awaitable.
            stage_name: Human-readable stage label (used in log entries).
            video_id: Pipeline video identifier (used in log entries).

        Returns:
            The result of the first successful call.

        Raises:
            Exception: The last exception after all retries are exhausted, or
                immediately for non-transient HTTP errors.
        """
        last_exc: Exception = Exception("No attempts made")

        for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
            try:
                result = await coro_factory()
                # Log successful API call
                self._emit_log(
                    event_type="api_call",
                    stage_name=stage_name,
                    message=f"API call succeeded on attempt {attempt}",
                    video_id=video_id,
                    retry_attempt=attempt if attempt > 1 else None,
                )
                return result

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                http_code = _extract_http_code(exc)

                # Non-transient errors: fail immediately.
                if _is_nontransient_http_error(exc):
                    self._emit_log(
                        event_type="error",
                        stage_name=stage_name,
                        message=(
                            f"Non-transient HTTP error {http_code} on stage "
                            f"'{stage_name}' — failing immediately: {exc}"
                        ),
                        video_id=video_id,
                        http_response_code=http_code,
                        retry_attempt=attempt,
                    )
                    logger.error(
                        "_call_with_retry: non-transient HTTP %s on stage '%s' "
                        "for video_id=%s — no retry",
                        http_code,
                        stage_name,
                        video_id,
                    )
                    raise

                # Exhausted all retries.
                if attempt == len(_RETRY_DELAYS):
                    self._emit_log(
                        event_type="error",
                        stage_name=stage_name,
                        message=(
                            f"Stage '{stage_name}' failed after {attempt} attempts: {exc}"
                        ),
                        video_id=video_id,
                        http_response_code=http_code,
                        retry_attempt=attempt,
                    )
                    logger.error(
                        "_call_with_retry: stage '%s' exhausted %d attempts "
                        "for video_id=%s: %s",
                        stage_name,
                        attempt,
                        video_id,
                        exc,
                    )
                    break

                # Transient / unknown error: log and retry.
                self._emit_log(
                    event_type="retry_attempt",
                    stage_name=stage_name,
                    message=(
                        f"Attempt {attempt}/{len(_RETRY_DELAYS)} failed for "
                        f"stage '{stage_name}': {exc}. "
                        f"Retrying in {delay:.0f} s."
                    ),
                    video_id=video_id,
                    http_response_code=http_code,
                    retry_attempt=attempt,
                )
                logger.warning(
                    "_call_with_retry: stage '%s' attempt %d/%d failed "
                    "for video_id=%s: %s — retrying in %.0f s",
                    stage_name,
                    attempt,
                    len(_RETRY_DELAYS),
                    video_id,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        raise last_exc

    # ------------------------------------------------------------------
    # _run_stage — wraps a subsystem call with logging and error handling
    # ------------------------------------------------------------------

    async def _run_stage(
        self,
        stage_name: str,
        video_id: str,
        run_id: str,
        pre_status: Optional[PipelineStatus],
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> Any:
        """Execute one pipeline stage with full observability and error handling.

        Steps
        -----
        1. If *pre_status* is not ``None``, update Content_Calendar to that
           status (within :data:`_CALENDAR_UPDATE_DEADLINE_S`).
        2. Emit a ``stage_transition`` log entry on entry.
        3. Call ``_call_with_retry(coro_factory, stage_name, video_id)``.
        4. On success: emit a ``stage_transition`` log entry on exit; flush log
           to Asset_Store.
        5. On failure: log the error, update Content_Calendar to
           ``Pipeline Error — {stage_name}``, notify the Notifier within
           :data:`_FAILURE_NOTIFY_DEADLINE_S`, flush log, re-raise.

        Args:
            stage_name: Human-readable stage identifier.
            video_id: Pipeline video identifier.
            run_id: Pipeline run identifier (used for log flushing).
            pre_status: Optional status to set in Content_Calendar before
                the stage executes.  ``None`` means no pre-status update.
            coro_factory: Zero-argument callable returning an awaitable.

        Returns:
            The result returned by *coro_factory*.

        Raises:
            Exception: Any exception raised by the subsystem after retries.
        """
        # Step 1: pre-status update
        if pre_status is not None:
            await self._update_calendar_status(video_id, pre_status, run_id)

        # Step 2: entry log
        self._emit_log(
            event_type="stage_transition",
            stage_name=stage_name,
            message=f"Stage '{stage_name}' starting",
            video_id=video_id,
        )

        try:
            # Step 3: execute with retry
            result = await self._call_with_retry(coro_factory, stage_name, video_id)

            # Step 4: success log + flush
            self._emit_log(
                event_type="stage_transition",
                stage_name=stage_name,
                message=f"Stage '{stage_name}' completed successfully",
                video_id=video_id,
            )
            await self._flush_log(run_id, video_id)
            return result

        except Exception as exc:  # noqa: BLE001
            # Step 5: failure handling
            error_status_value = f"Pipeline Error — {stage_name}"
            error_msg = str(exc)

            self._emit_log(
                event_type="error",
                stage_name=stage_name,
                message=(
                    f"Stage '{stage_name}' failed: {error_msg}"
                ),
                video_id=video_id,
                http_response_code=_extract_http_code(exc),
            )

            # Update Content_Calendar to Pipeline Error within 30 s.
            try:
                await asyncio.wait_for(
                    self._set_pipeline_error_status(video_id, stage_name),
                    timeout=_CALENDAR_UPDATE_DEADLINE_S,
                )
            except Exception as cal_exc:  # noqa: BLE001
                logger.error(
                    "_run_stage: failed to set Pipeline Error status for "
                    "video_id=%s stage=%s: %s",
                    video_id,
                    stage_name,
                    cal_exc,
                )

            # Notify within 60 s.
            try:
                await asyncio.wait_for(
                    self._async_notify_failure(video_id, stage_name, error_msg),
                    timeout=_FAILURE_NOTIFY_DEADLINE_S,
                )
            except Exception as notify_exc:  # noqa: BLE001
                logger.error(
                    "_run_stage: notifier failed for video_id=%s stage=%s: %s",
                    video_id,
                    stage_name,
                    notify_exc,
                )

            await self._flush_log(run_id, video_id)
            raise

    # ------------------------------------------------------------------
    # Status update helpers
    # ------------------------------------------------------------------

    async def _update_calendar_status(
        self,
        video_id: str,
        status: PipelineStatus,
        run_id: str,
    ) -> None:
        """Update Content_Calendar status within 30 seconds; log the transition."""
        self._emit_log(
            event_type="stage_transition",
            stage_name="content_calendar",
            message=f"Updating status → {status.value}",
            video_id=video_id,
        )
        try:
            await asyncio.wait_for(
                self._content_calendar.update_status(video_id, status),
                timeout=_CALENDAR_UPDATE_DEADLINE_S,
            )
        except asyncio.TimeoutError:
            logger.error(
                "_update_calendar_status: timed out after %.0f s for "
                "video_id=%s status=%s",
                _CALENDAR_UPDATE_DEADLINE_S,
                video_id,
                status.value,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "_update_calendar_status: failed for video_id=%s status=%s: %s",
                video_id,
                status.value,
                exc,
            )

    async def _set_pipeline_error_status(
        self, video_id: str, stage_name: str
    ) -> None:
        """Set a ``Pipeline Error — {stage_name}`` status in Content_Calendar.

        Uses the generic ``PIPELINE_ERROR`` enum value; the stage-specific
        message is appended for observability in the log.
        """
        await self._content_calendar.update_status(
            video_id, PipelineStatus.PIPELINE_ERROR
        )
        logger.info(
            "Content_Calendar set to Pipeline Error for video_id=%s stage=%s",
            video_id,
            stage_name,
        )

    async def _async_notify_failure(
        self, video_id: str, stage_name: str, error_msg: str
    ) -> None:
        """Send a failure alert via Notifier (async-friendly wrapper)."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self._notifier.send_failure_alert,
            video_id,
            stage_name,
            error_msg,
            None,  # publish_datetime
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _emit_log(
        self,
        *,
        event_type: Literal["api_call", "stage_transition", "retry_attempt", "error", "warning"],
        stage_name: str,
        message: str,
        video_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        http_response_code: Optional[int] = None,
        retry_attempt: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append a :class:`~pipeline.models.LogEntry` to the in-memory accumulator.

        The Python ``logger`` is also called for immediate visibility.

        Args:
            event_type: One of the five canonical event type strings.
            stage_name: Stage that emitted this entry.
            message: Human-readable description.
            video_id: Optional pipeline video identifier.
            batch_id: Optional batch identifier.
            http_response_code: Optional HTTP status code from the subsystem.
            retry_attempt: Optional retry counter (1-based).
            metadata: Optional free-form key/value metadata dict.
        """
        entry = _make_log_entry(
            event_type=event_type,
            stage_name=stage_name,
            message=message,
            video_id=video_id,
            batch_id=batch_id,
            http_response_code=http_response_code,
            retry_attempt=retry_attempt,
            metadata=metadata,
        )
        self._log_entries.append(entry)

        # Mirror to Python logging at an appropriate level.
        if event_type == "error":
            logger.error("[%s] %s", stage_name, message)
        elif event_type in ("retry_attempt", "warning"):
            logger.warning("[%s] %s", stage_name, message)
        else:
            logger.info("[%s] %s", stage_name, message)

    async def _flush_log(self, run_id: str, video_id: str) -> None:
        """Serialize all accumulated :class:`~pipeline.models.LogEntry` objects to
        ``logs/pipeline_run_{run_id}.json`` in Asset_Store.

        Entries already written on a prior flush are re-included (the full log
        is always overwritten atomically) so that the file always contains the
        complete run history.

        Args:
            run_id: Pipeline run identifier (used in the filename).
            video_id: Used as the ``video_id`` scope for Asset_Store.write.
        """
        if not self._log_entries:
            return

        filename = f"pipeline_run_{run_id}.json"
        try:
            # Serialise all entries to a JSON array.
            payload = json.dumps(
                [
                    entry.model_dump(mode="json")
                    for entry in self._log_entries
                ],
                indent=2,
                ensure_ascii=False,
                default=str,  # fallback for datetime and other non-serialisable types
            ).encode("utf-8")

            await self._asset_store.write(
                video_id=video_id,
                subfolder=SubFolder.LOGS,
                filename=filename,
                content=payload,
            )
            logger.debug(
                "_flush_log: wrote %d entries to %s for video_id=%s",
                len(self._log_entries),
                filename,
                video_id,
            )
        except Exception as exc:  # noqa: BLE001
            # Log flush failures are non-fatal; we log but do not re-raise.
            logger.error(
                "_flush_log: failed to write %s for video_id=%s: %s",
                filename,
                video_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Asset probing helper (task 16.3)
    # ------------------------------------------------------------------

    async def _probe_asset(
        self,
        video_id: str,
        subfolder: SubFolder,
        filename: str,
    ) -> Optional[bytes]:
        """Attempt to read *filename* from Asset_Store.

        Returns:
            The raw bytes if the file is present and readable; ``None`` if the
            file is absent or an :class:`~pipeline.asset_store.AssetStoreError`
            is raised.
        """
        try:
            data = await self._asset_store.read(video_id, subfolder, filename)
            logger.debug(
                "_probe_asset: found %s/%s for video_id=%s (%d bytes)",
                subfolder.value,
                filename,
                video_id,
                len(data),
            )
            return data
        except Exception:  # noqa: BLE001
            logger.debug(
                "_probe_asset: missing or unreadable %s/%s for video_id=%s",
                subfolder.value,
                filename,
                video_id,
            )
            return None


# ---------------------------------------------------------------------------
# Reconstruction helpers for resume_pipeline (task 16.3)
# ---------------------------------------------------------------------------


def _derive_topic_from_script(script: "Script") -> "TopicEntry":
    """Derive a synthetic :class:`~pipeline.models.TopicEntry` from a script.

    Extracts a meaningful topic title from the script content by looking for:
    1. A Markdown heading (``# Title``)
    2. The first substantial non-heading line (the hook/intro paragraph)

    Skips section markers (short ALL-CAPS lines like "HOOK", "INTRODUCTION")
    and speaker-direction annotations (``[pause]``, ``[emphasis]``).

    This is used during pipeline resume so the metadata generator uses a topic
    consistent with the already-written script, rather than fresh (possibly
    unrelated) search results.

    Args:
        script: The reconstructed Script object.

    Returns:
        A single :class:`~pipeline.models.TopicEntry` with a synthetic score.
    """
    import re as _re  # noqa: PLC0415
    from pipeline.models import TopicEntry  # local import

    lines = script.content.splitlines()

    # Strategy 1: Find a markdown heading like "# Topic Title"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if len(heading) > 10:
                title = heading[:200]
                break
    else:
        # Strategy 2: Find first substantial content line that isn't a section marker
        # Section markers are short ALL-CAPS lines (HOOK, INTRODUCTION, etc.)
        title = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Skip ALL-CAPS section headers (under ~40 chars typically)
            if stripped == stripped.upper() and len(stripped) < 50:
                continue
            # Skip lines that start with annotations
            if stripped.startswith("["):
                continue
            # Clean speaker-direction annotations for a readable title
            clean = _re.sub(r"\[.*?\]", "", stripped).strip()
            if len(clean) > 15:
                # Extract the core question/statement — take the first sentence
                sentences = _re.split(r"[.!?]", clean)
                title = sentences[0].strip()[:200] if sentences else clean[:200]
                break

        if not title:
            words = [w for w in script.content.split() if w.strip()][:10]
            title = " ".join(words)[:200]

    return TopicEntry(
        title=title,
        composite_score=1.0,
        recency_hours=0.0,
        source_query_timestamp=_utcnow(),
        search_volume_signal=100.0,
        relevance_tags_matched=[],
    )


def _reconstruct_script(video_id: str, content_bytes: bytes) -> Script:
    """Reconstruct a minimal :class:`~pipeline.models.Script` from stored bytes.

    Used by ``resume_pipeline`` when the script artefact exists in Asset_Store
    and does not need to be regenerated.

    Args:
        video_id: Pipeline video identifier.
        content_bytes: Raw UTF-8 bytes of the stored script Markdown file.

    Returns:
        A :class:`~pipeline.models.Script` with ``content`` populated and
        ``version=1``.
    """
    from pipeline.models import Script  # local import to avoid circularity

    content = content_bytes.decode("utf-8", errors="replace")
    return Script(
        video_id=video_id,
        version=1,
        content=content,
        word_count=len(content.split()),
        style_profile_doc_id="",  # doc_id not stored in the file itself
        asset_url=None,
        created_at=_utcnow(),
    )


def _reconstruct_narration(video_id: str) -> NarrationAsset:
    """Reconstruct a minimal :class:`~pipeline.models.NarrationAsset` for resume.

    The MP3 bytes are not re-read into memory; only the path is restored so
    the Visual_Generator can reference the file.

    Args:
        video_id: Pipeline video identifier.

    Returns:
        A :class:`~pipeline.models.NarrationAsset` with ``mp3_path`` pointing
        to the version-1 MP3 filename.
    """
    from pipeline.models import NarrationAsset  # local import

    filename = f"{video_id}_v1.mp3"
    return NarrationAsset(
        video_id=video_id,
        version=1,
        mp3_path=filename,
        asset_url=None,
        created_at=_utcnow(),
    )


def _reconstruct_visual(video_id: str) -> VisualAsset:
    """Reconstruct a minimal :class:`~pipeline.models.VisualAsset` for resume.

    Note: mp4_url and thumbnail_url are set to None here — the publisher
    is responsible for fetching the Drive URL via asset_store when these are None.
    """
    from pipeline.models import VisualAsset  # local import

    return VisualAsset(
        video_id=video_id,
        version=1,
        mp4_path=f"videos/{video_id}_v1.mp4",
        thumbnail_path=f"thumbnails/{video_id}_v1.jpg",
        mp4_url=None,
        thumbnail_url=None,
        created_at=_utcnow(),
    )


def _reconstruct_metadata(video_id: str, content_bytes: bytes) -> MetadataPackage:
    """Reconstruct a :class:`~pipeline.models.MetadataPackage` from stored JSON bytes.

    Args:
        video_id: Pipeline video identifier.
        content_bytes: Raw UTF-8 JSON bytes of the stored metadata file.

    Returns:
        A :class:`~pipeline.models.MetadataPackage` parsed from the JSON.

    Raises:
        ValueError: If the JSON cannot be parsed or is missing required fields.
    """
    from pipeline.models import MetadataPackage  # local import

    raw = json.loads(content_bytes.decode("utf-8"))
    return MetadataPackage.model_validate(raw)


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Orchestrator",
    "PipelineRun",
    "ReviewGate",
    "ReviewGateError",
    "_is_transient_http_error",
    "_is_nontransient_http_error",
    "_extract_http_code",
    "compute_batch_completion",
    "generate_batch_slots",
    "validate_batch_size",
]
