# Implementation Plan: AI YouTube Content Pipeline

## Overview

Implement the AI YouTube Content Pipeline — an end-to-end automated system that takes a creator
from topic discovery to a scheduled, cross-posted YouTube video. The pipeline is built in Python
and structured as twelve subsystems orchestrated by a Claude-based agent. Every stage is
idempotent, observable via structured JSON logs, and gated by two mandatory human-review
checkpoints. Testing uses Hypothesis for 30 property-based tests complemented by unit, integration,
and smoke tests.

---

## Tasks

- [x] 1. Project scaffolding and configuration schema
  - Create the package directory structure under `pipeline/` with `__init__.py` files for each
    subsystem module: `asset_store`, `content_calendar`, `notifier`, `reference_analyzer`,
    `topic_researcher`, `script_writer`, `narration_generator`, `visual_generator`,
    `metadata_generator`, `publisher`, `cross_poster`, `orchestrator`.
  - Define `PipelineConfig` Pydantic model (all fields from design §Data Models) with validators
    for `voice_id` non-empty, `batch_mode.target_count` ∈ [2, 10], and
    `topic_research_provider` ∈ {"perplexity", "tavily"}.
  - Define all shared data models as Pydantic models: `StyleProfile`, `TopicEntry`, `Script`,
    `NarrationAsset`, `VisualAsset`, `MetadataPackage`, `VideoRecord`, `BatchRecord`,
    `NotificationEvent`, `LogEntry`, `YouTubeVideoRef`, `PipelineStatus` enum (18 values),
    `SubFolder` enum, `Platform` enum.
  - Add `pyproject.toml` dependencies: `pydantic`, `hypothesis`, `pytest`, `pytest-asyncio`,
    `httpx`, `mutagen`, `pillow`, `anthropic`, `elevenlabs`, `google-api-python-client`,
    `notion-client`, `slack_sdk`.
  - Create `tests/` directory tree matching design §Test Organisation with `__init__.py` files.
  - _Requirements: 1.1, 2.1, 3.1, 5.1, 6.1, 7.1, 9.1, 10.1, 11.1, 12.1, 13.1, 14.1, 15.1_


- [x] 2. Implement Asset_Store (Google Drive MCP wrapper)
  - [x] 2.1 Implement `Asset_Store` class with `write`, `read`, and `url` methods
    - Wrap Google Drive MCP calls; enforce folder hierarchy
      `ai-youtube-pipeline/{video_id}/{subfolder}/{filename}`.
    - Validate `video_id` against pattern `[a-zA-Z0-9\-_]{1,128}` before every call.
    - `write` must return a `DriveURL` within 10 seconds.
    - Apply exponential back-off retry (3 attempts, 10 s base, 80 s max) on any Google Drive
      API failure; on final failure log the error, notify Notifier, and raise `AssetStoreError`.
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

  - [ ]* 2.2 Write property test for Asset_Store path format invariant
    - **Property 2: Asset_Store path format invariant**
    - **Validates: Requirements 1.2, 3.4, 5.3, 6.4, 12.1**

  - [ ]* 2.3 Write unit tests for Asset_Store
    - Test retry exhaustion triggers Notifier and raises `AssetStoreError`.
    - Test `video_id` pattern validation rejects invalid characters.
    - Test `write` returns a URL within simulated 10-second timeout.
    - _Requirements: 12.2, 12.4_


- [x] 3. Implement Content_Calendar (Notion MCP wrapper)
  - [x] 3.1 Implement `Content_Calendar` class with all six interface methods
    - `create_record` — create Notion page with all 13 required schema fields; associate with
      `batch_id` when provided.
    - `update_status` — update status field within 30-second SLA; apply Notion API retry
      (3 attempts, 5 s base, 20 s max exponential).
    - `update_asset_link`, `set_publish_datetime`, `get_batch_topics`, `get_batch_completion`.
    - Reject `set_publish_datetime` calls when datetime is in the past or video is `Published`.
    - `get_batch_completion` returns `floor(published_count / n * 100)`.
    - _Requirements: 10.1, 10.2, 10.3, 10.5, 10.6, 11.6_

  - [x] 3.2 Implement calendar view conflict and gap detection
    - `detect_conflicts` — return pairs of `video_id`s sharing the same scheduled datetime
      (to the minute).
    - `detect_gaps` — return date intervals of 7 or more consecutive days without a scheduled
      video.
    - _Requirements: 10.4_

  - [ ]* 3.3 Write property test for VideoRecord schema invariant
    - **Property 20: VideoRecord schema invariant**
    - **Validates: Requirements 10.1**

  - [ ]* 3.4 Write property test for calendar conflict and gap detection
    - **Property 21: Calendar conflict and gap detection**
    - **Validates: Requirements 10.4**

  - [ ]* 3.5 Write property test for calendar rejects invalid datetime updates
    - **Property 22: Calendar rejects invalid datetime updates**
    - **Validates: Requirements 10.6**

  - [ ]* 3.6 Write unit tests for Content_Calendar
    - Test `update_status` SLA assertion with mock Notion API.
    - Test `get_batch_completion` returns correct floor values for boundary inputs.
    - Test Notion API retry and failure handling.
    - _Requirements: 10.2, 10.6, 11.6_


- [x] 4. Implement Notifier
  - [x] 4.1 Implement `Notifier` class with all four interface methods
    - Support Slack webhook, Discord webhook, and SMTP email as configurable channels.
    - `send_review_gate` — dispatch within 60 seconds of gate trigger.
    - `send_failure_alert` — dispatch within 60 seconds of failure detection.
    - `send_batch_summary` — include exactly N entries with title, status, and
      `scheduled_publish_datetime`.
    - If no channel configured: suppress all notifications and log a warning to Asset_Store.
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5_

  - [x] 4.2 Implement deduplication and fallback chain
    - Deduplication key: `(video_id, notification_type, stage_name)`; suppress duplicate within
      10-minute window.
    - On delivery failure after 2 attempts: log failure and try next configured channel.
    - _Requirements: 14.6, 14.7_

  - [ ]* 4.3 Write property test for notification deduplication invariant
    - **Property 28: Notification deduplication invariant**
    - **Validates: Requirements 14.6**

  - [ ]* 4.4 Write property test for batch completion notification completeness
    - **Property 27: Batch completion notification completeness**
    - **Validates: Requirements 14.4**

  - [ ]* 4.5 Write unit tests for Notifier
    - Test fallback chain: primary Slack fails → Discord used.
    - Test no-channel-configured suppression and Asset_Store warning log.
    - Test deduplication suppresses second call within 10-minute window.
    - _Requirements: 14.2, 14.6, 14.7_


- [x] 5. Checkpoint — Core infrastructure complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Implement Reference_Analyzer (Browser MCP + StyleProfile)
  - [x] 6.1 Implement `Reference_Analyzer.analyze` method
    - Navigate to channel URL via Browser MCP.
    - Collect uploads within last 90 days; use all available if fewer than 10 (minimum 1),
      log a warning when count < 10.
    - Extract per-upload: transcript (sentiment polarity, WPM, avg sentence length), segment
      annotations (intro/hook/body/CTA), thumbnail metadata (dominant colors ≤ 5 hex values,
      text overlay position, subject framing).
    - Aggregate into a `StyleProfile` document; apply channel-access retry (3 attempts,
      5 s → 10 s → 20 s); on failure log, notify Notifier, halt without writing JSON.
    - Write StyleProfile JSON to `Asset_Store/style-profiles/`; on write failure retry 3× at
      10 s fixed intervals; on final failure notify Notifier and halt.
    - Update Content_Calendar batch record with `style_profile_doc_id`.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 6.2 Write property test for StyleProfile required fields and value ranges
    - **Property 1: StyleProfile contains all required fields with valid values**
    - **Validates: Requirements 1.1, 1.5**

  - [ ]* 6.3 Write unit tests for Reference_Analyzer
    - Test inaccessible URL halts without writing JSON and notifies Notifier.
    - Test fewer-than-10-uploads path logs warning and proceeds.
    - Test Asset_Store write retry exhaustion triggers Notifier and halt.
    - _Requirements: 1.4, 1.6, 1.7_


- [x] 7. Implement Topic_Researcher (Perplexity/Tavily MCP + scoring)
  - [x] 7.1 Implement `Topic_Researcher.research` with composite scoring
    - Query Perplexity or Tavily (configurable) for AI topics trending within past 72 hours.
    - Compute composite score: `(norm_search_volume + norm_recency + norm_relevance) / 3` using
      min-max normalization; relevance = 1.0 if any subject tag matches, else 0.0.
    - Sort result descending by `composite_score`; require ≥ 5 entries (≥ batch_size).
    - Store JSON to `Asset_Store/research/` timestamped to nearest minute.
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 7.2 Implement deduplication and error handling for Topic_Researcher
    - Deduplicate against `excluded_titles` (case-insensitive) before returning results.
    - Partial results (1–4 topics) before retries exhausted → halt, log partial count, do not
      store.
    - Zero results after 3 retries at 30 s fixed intervals → notify Notifier with run ID and
      failure reason, halt pipeline.
    - Batch mode: produce N topics (max 50), no title repeated from past 30 days in calendar.
    - _Requirements: 2.4, 2.5, 2.6_

  - [ ]* 7.3 Write property test for topic composite score correctness
    - **Property 3: Topic composite score correctness**
    - **Validates: Requirements 2.2**

  - [ ]* 7.4 Write property test for topic list serialization round-trip
    - **Property 4: Topic list serialization round-trip**
    - **Validates: Requirements 2.3**

  - [ ]* 7.5 Write property test for batch topic deduplication
    - **Property 5: Batch topic deduplication**
    - **Validates: Requirements 2.6**

  - [ ]* 7.6 Write unit tests for Topic_Researcher
    - Test partial-results path does not store and halts.
    - Test zero-results-after-retries notifies Notifier and halts.
    - Test batch size validation (max 50, ≥ 5 topics).
    - _Requirements: 2.4, 2.5, 2.6_


- [x] 8. Implement Script_Writer (Claude API + word-count enforcement)
  - [x] 8.1 Implement `Script_Writer.generate` with Claude API and Style_Profile injection
    - Validate topic and style_profile are non-empty; if either is absent reject and notify
      Notifier without calling Claude.
    - Build Claude prompt incorporating style_profile narration tone, pacing, rhetorical patterns,
      and segment structure.
    - Require script structure: hook (≤ 60 s at 150 WPM), body (3–5 segments), CTA (≤ 30 s at
      150 WPM).
    - Embed ≥ 1 speaker-direction annotation per segment (`[pause]`, `[emphasis]`, etc.).
    - Record `style_profile.doc_id` in script metadata.
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 8.2 Implement word-count enforcement and versioned Asset_Store writes
    - After generation count words; if outside [800, 1500] revise once automatically.
    - If still outside range after revision: halt, set status `Pipeline Error — script_generation`,
      notify Notifier.
    - Write script to `scripts/script_v{n}.md`; increment version per generation/edit event;
      preserve all prior versions.
    - _Requirements: 3.4, 3.5_

  - [x] 8.3 Implement `Script_Writer.revise` for user-submitted edits
    - Accept edit string; reject if diff against current version is empty (return validation error).
    - Save revised script as `script_v{n+1}.md`; preserve previous version.
    - _Requirements: 4.4, 4.7_

  - [ ]* 8.4 Write property test for script word count invariant
    - **Property 6: Script word count invariant**
    - **Validates: Requirements 3.1, 3.5**

  - [ ]* 8.5 Write property test for script always references its style profile
    - **Property 7: Script always references its style profile**
    - **Validates: Requirements 3.2**

  - [ ]* 8.6 Write property test for script annotation density invariant
    - **Property 8: Script annotation density invariant**
    - **Validates: Requirements 3.3**

  - [ ]* 8.7 Write property test for script version monotonicity
    - **Property 9: Script version monotonicity**
    - **Validates: Requirements 3.4, 4.4**

  - [ ]* 8.8 Write unit tests for Script_Writer
    - Test empty-diff edit rejection returns validation error without advancing gate.
    - Test Claude API error applies general retry policy.
    - Test missing topic or style_profile triggers Notifier without Claude call.
    - _Requirements: 3.1, 3.5, 4.7_


- [x] 9. Checkpoint — Generation subsystems (research + script) complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement Narration_Generator (ElevenLabs API)
  - [x] 10.1 Implement `Narration_Generator.generate` with pre-flight and retry logic
    - Pre-flight: if `voice_id` is absent or empty notify Notifier and halt before any API call.
    - Submit script text (≤ 5,000 characters per segment) to ElevenLabs; request 44,100 Hz /
      ≥ 128 kbps MP3.
    - Retry on ElevenLabs API error: 3 attempts with exponential back-off 5 s / 60 s max.
    - On all retries exhausted: discard audio, notify Notifier, halt narration stage.
    - Write MP3 to `narration/{video_id}_v{n}.mp3`; on Asset_Store write failure: discard audio,
      notify Notifier, halt.
    - On success update Content_Calendar status to `Narration Ready` and pass MP3 path to
      Visual_Generator.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8_

  - [ ]* 10.2 Write property test for narration audio specification invariant
    - **Property 10: Narration audio specification invariant**
    - Use `mutagen` to verify sample rate = 44,100 Hz and bitrate ≥ 128 kbps on generated files.
    - **Validates: Requirements 5.2**

  - [ ]* 10.3 Write unit tests for Narration_Generator
    - Test empty `voice_id` halts without API call.
    - Test ElevenLabs API error retry exhaustion discards audio and notifies Notifier.
    - Test Asset_Store write failure discards audio and notifies Notifier.
    - _Requirements: 5.4, 5.7, 5.8_


- [x] 11. Implement Visual_Generator (Viewmax MCP + video compilation)
  - [x] 11.1 Implement scene prompt derivation and per-clip generation
    - Derive one scene prompt per script segment (hook, each body segment, CTA) incorporating
      Style_Profile visual composition patterns.
    - Submit each prompt to Viewmax MCP; retry each clip 3× with random delay in [5, 30] s.
    - On all retries for a clip exhausted: substitute a static JPEG fallback (1920×1080) and
      log substitution with segment identifier and failure reason.
    - Guarantee generated frames are not pixel-for-pixel reproductions of reference channel frames.
    - _Requirements: 6.1, 6.5, 6.6_

  - [x] 11.2 Implement video compilation, thumbnail generation, and Asset_Store writes
    - Compile clips into a single MP4 synchronized with narration audio within ±100 ms.
    - Output MP4: 1920×1080, H.264, ≥ 24 fps.
    - Generate thumbnail JPEG (1280×720, < 2 MB) using Style_Profile dominant colors, text overlay
      position, and subject framing.
    - Write MP4 to `videos/{video_id}_v{n}.mp4` and JPEG to `thumbnails/{video_id}_v{n}.jpg`.
    - Update Content_Calendar status to `Visuals Ready` and trigger metadata stage.
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.7_

  - [ ]* 11.3 Write property test for video output specification invariant
    - **Property 12: Video output specification invariant**
    - Use `pillow`/`ffprobe` to verify resolution, codec, frame rate, and audio sync offset.
    - **Validates: Requirements 6.1**

  - [ ]* 11.4 Write property test for thumbnail specification invariant
    - **Property 13: Thumbnail specification invariant**
    - **Validates: Requirements 6.3**

  - [ ]* 11.5 Write property test for clip fallback after all retries exhausted
    - **Property 14: Clip fallback after all retries exhausted**
    - **Validates: Requirements 6.6**

  - [ ]* 11.6 Write unit tests for Visual_Generator
    - Test clip retry exhaustion substitutes static fallback and logs substitution.
    - Test thumbnail file size < 2 MB constraint enforced before Asset_Store write.
    - _Requirements: 6.3, 6.6_


- [x] 12. Implement Metadata_Generator (Claude API + validation)
  - [x] 12.1 Implement `Metadata_Generator.generate` with field validation and regeneration
    - Pre-flight: if script or topic data is absent notify Notifier and do not attempt generation.
    - Call Claude API to produce: title (≤ 60 chars, includes primary keyword), description
      (200–500 words with summary, chapter markers aligned to script segments, 3–5 CTAs/links,
      closing paragraph with ≥ 3 tags), tag list (10–15 tags, each 2–5 words), hashtags (3–5,
      prefixed `#`, body 2–30 chars, no spaces).
    - Validate each field independently; on failure regenerate that field once.
    - If field still invalid after one regeneration: halt stage, notify Notifier.
    - Write `MetadataPackage` JSON to `metadata/{video_id}.json` in Asset_Store.
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.7, 7.8_

  - [ ]* 12.2 Write property test for MetadataPackage structural invariant
    - **Property 15: MetadataPackage structural invariant**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

  - [ ]* 12.3 Write property test for MetadataPackage serialization round-trip
    - **Property 16: MetadataPackage serialization round-trip**
    - **Validates: Requirements 7.5**

  - [ ]* 12.4 Write unit tests for Metadata_Generator
    - Test missing script input skips generation and notifies Notifier.
    - Test field-level regeneration when initial generation fails validation.
    - Test field still failing after regeneration halts stage and notifies Notifier.
    - _Requirements: 7.7, 7.8_


- [x] 13. Checkpoint — Media generation subsystems complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. Implement Publisher (YouTube Data API + scheduling + rollback)
  - [x] 14.1 Implement `Publisher.upload` with retry and Notifier integration
    - Upload MP4 + thumbnail + metadata to YouTube Data API; set privacy to `Unlisted`.
    - Make `youtube_video_id` and `unlisted_url` available to Notifier within 10 minutes.
    - Retry upload: 3 attempts, exponential back-off 60 s / 300 s max.
    - On all retries exhausted: notify Notifier with error details, halt upload stage.
    - _Requirements: 9.1, 9.2, 9.3, 9.5_

  - [x] 14.2 Implement `Publisher.schedule` with datetime validation and rollback
    - Validate `publish_datetime > now + 15 minutes`; reject and re-prompt if invalid.
    - Call YouTube API `videos.update` with `publishAt` on valid datetime.
    - No schedule confirmation within 7 days → set status `Awaiting Schedule`, leave `Unlisted`.
    - On `Published` transition: update Content_Calendar status; if update fails after 3 retries
      revert YouTube video to `Unlisted` and notify Notifier with rollback details.
    - _Requirements: 9.4, 9.6, 9.7_

  - [x] 14.3 Implement `Publisher.reschedule` for calendar-driven rescheduling
    - Read updated datetime from Content_Calendar; reschedule YouTube video within 5 minutes.
    - Guard: skip reschedule if video is already `Published` or datetime is in the past.
    - _Requirements: 10.5, 10.6_

  - [ ]* 14.4 Write property test for publish datetime validation
    - **Property 18: Publish datetime validation**
    - **Validates: Requirements 9.4**

  - [ ]* 14.5 Write property test for calendar rollback on persistent update failure
    - **Property 19: Calendar rollback on persistent update failure**
    - **Validates: Requirements 9.6**

  - [ ]* 14.6 Write unit tests for Publisher
    - Test past datetime and near-future (< 15 min) datetimes are rejected.
    - Test 7-day no-schedule timeout sets `Awaiting Schedule`.
    - Test three consecutive calendar update failures trigger rollback to `Unlisted`.
    - _Requirements: 9.4, 9.6, 9.7_


- [x] 15. Implement Cross_Poster (X, LinkedIn, Instagram, Facebook)
  - [x] 15.1 Implement `Cross_Poster.post` with platform caption generation
    - Generate platform-native captions from title + description + tags within character budgets:
      X ≤ 280, LinkedIn ≤ 3,000, Instagram ≤ 2,200, Facebook ≤ 500.
    - Each caption must include the YouTube video URL and ≥ 3 hashtags within the character limit.
    - Disabled platforms are skipped silently without error.
    - _Requirements: 15.1, 15.2, 15.3, 15.5_

  - [x] 15.2 Implement retry, failure isolation, and Notifier integration for Cross_Poster
    - Per-platform retry: 2 attempts at 60 s fixed intervals.
    - On platform failure after retries: log and notify Notifier with platform name and reason.
    - Failure on one platform must not prevent posting to remaining enabled platforms.
    - Must execute within 30 minutes of video transitioning to `Published`.
    - _Requirements: 15.1, 15.4, 15.6_

  - [ ]* 15.3 Write property test for cross-post caption fits platform character limit
    - **Property 29: Cross-post caption fits platform character limit**
    - **Validates: Requirements 15.2, 15.3**

  - [ ]* 15.4 Write property test for cross-post platform failure isolation
    - **Property 30: Cross-post platform failure isolation**
    - **Validates: Requirements 15.6**

  - [ ]* 15.5 Write unit tests for Cross_Poster
    - Test disabled platform is skipped without error or notification.
    - Test one platform failure does not prevent posting to other platforms.
    - _Requirements: 15.5, 15.6_


- [x] 16. Implement Orchestrator (pipeline coordination, retry logic, batch processing, Review Gates)
  - [x] 16.1 Implement `Orchestrator.start_pipeline` and stage sequencing
    - Read `PipelineConfig`; resolve which stage to execute next per video record.
    - Sequence calls: Reference_Analyzer → Topic_Researcher → Script_Writer → Narration_Generator
      → Visual_Generator → Metadata_Generator → Publisher → Cross_Poster.
    - Update Content_Calendar status within 30 seconds of every state transition.
    - Emit a structured `LogEntry` for every API call and state transition; write to
      `logs/pipeline_run_{run_id}.json` in Asset_Store.
    - _Requirements: 10.2, 13.3, 13.6_

  - [x] 16.2 Implement general retry policy and error classification
    - Transient HTTP errors (429, 500, 502, 503, 504): retry 3× with 5 s → 10 s → 20 s.
    - Non-transient errors (400, 401, 403, 404): fail immediately, no retry.
    - On stage failure: log, update Content_Calendar to `Pipeline Error — {stage_name}`, preserve
      prior outputs, notify Notifier within 60 seconds.
    - _Requirements: 13.1, 13.2, 13.3, 13.4_

  - [x] 16.3 Implement `Orchestrator.resume_pipeline` for retry-from-error
    - Load last known good outputs from Asset_Store.
    - If any expected prior-stage output is missing or unreadable: restart from that stage.
    - Otherwise resume from the failed stage.
    - _Requirements: 13.5, 13.7_

  - [x] 16.4 Implement Review Gate mechanism (Gate 1 — Script and Gate 2 — Final)
    - `trigger_gate`: update Content_Calendar status, send Notifier alert within 60 s, enter WAIT
      state.
    - `poll_gate`: check Content_Calendar for user action every 60 s.
    - Gate 1 (Script): on approve → advance to narration; on edit → increment version, re-enter
      Scripting; reminder at 48 h, 72 h, 96 h (max 3 reminders).
    - Gate 2 (Final): on approve → advance to upload; on regenerate individual asset → re-run
      that stage only, reset 72 h timer; auto-approve at 72 h timeout.
    - Validate edit submissions: reject empty diff without advancing gate.
    - Notify user within 60 seconds of gate trigger with video title, review type, asset links,
      and required action.
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 16.5 Write property test for selective regeneration isolation
    - **Property 17: Selective regeneration isolation**
    - **Validates: Requirements 8.3**

  - [ ]* 16.6 Write unit tests for Orchestrator Review Gates and error handling
    - Test Gate 1 approve advances state to `Script Approved`.
    - Test Gate 1 edit with empty diff returns validation error.
    - Test Gate 2 72-hour auto-approve transitions to `Auto-Approved for Upload`.
    - Test selective regeneration preserves unrequested asset Drive URLs.
    - Test non-transient error (401) fails immediately with no retry.
    - Test resume loads prior outputs and restarts missing stage.
    - _Requirements: 4.5, 4.7, 8.3, 8.5, 13.2, 13.5_


- [x] 17. Implement batch processing in Orchestrator
  - [x] 17.1 Implement `start_batch` with validation and sequential video processing
    - Validate N ∈ [2, 10]; reject with validation error and create zero records if out of range.
    - Generate `batch_id`; create N `VideoRecord` entries in Content_Calendar under that batch_id.
    - Call `Topic_Researcher.research(batch_size=N, excluded=past_30_days)`.
    - Process videos one at a time: do not start video i+1 generation until video i completes all
      generation stages through `Awaiting Final Review`.
    - Review Gate responses do not block subsequent video generation.
    - _Requirements: 11.1, 11.2, 11.3_

  - [x] 17.2 Implement batch scheduling interface and failure isolation
    - When all batch videos pass Final Review Gate: propose evenly-spaced publish datetimes across
      14 days using formula `slot[i] = now + i * (14 days / (N−1))`.
    - Stage failure for video i: mark `Pipeline Error — {stage_name}`, continue with video i+1,
      notify Notifier.
    - After each `Published` transition recalculate and update `BatchRecord.completion_percentage`
      with `floor(published_count / N * 100)`.
    - _Requirements: 11.4, 11.5, 11.6_

  - [ ]* 17.3 Write property test for batch count invariant
    - **Property 23: Batch count invariant**
    - **Validates: Requirements 11.1, 11.2**

  - [ ]* 17.4 Write property test for batch schedule even spacing
    - **Property 24: Batch schedule even spacing**
    - **Validates: Requirements 11.4**

  - [ ]* 17.5 Write property test for batch completion percentage formula
    - **Property 25: Batch completion percentage formula**
    - **Validates: Requirements 11.6**

  - [ ]* 17.6 Write unit tests for batch processing
    - Test N=1 and N=11 are rejected with validation error and zero records created.
    - Test stage failure for one batch video continues processing remaining videos.
    - Test sequential processing order (video i+1 does not start until video i reaches
      `Awaiting Final Review`).
    - _Requirements: 11.2, 11.3, 11.5_


- [x] 18. Checkpoint — Orchestrator and batch processing complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 19. Write property-based tests — Properties 1–10 (`tests/property/test_properties_1_to_10.py`)
  - [x] 19.1 Implement Hypothesis generator strategies
    - Write `topic_entry_strategy()`, `style_profile_strategy()`, `metadata_package_strategy()`,
      `video_record_strategy()`, `datetime_strategy()`, and `platform_caption_strategy()` in
      `tests/property/strategies.py`.
    - Each strategy must generate valid instances across full field value ranges as defined in the
      design §Data Models.
    - _Requirements: design §Testing Strategy_

  - [ ]* 19.2 Write property test: Property 1 — StyleProfile required fields
    - **Property 1: StyleProfile contains all required fields with valid values**
    - `@given(style_profile_strategy())` with `@settings(max_examples=100)`.
    - Assert all six fields present and within valid ranges.
    - **Validates: Requirements 1.1, 1.5**

  - [ ]* 19.3 Write property test: Property 2 — Asset_Store path format
    - **Property 2: Asset_Store path format invariant**
    - **Validates: Requirements 1.2, 3.4, 5.3, 6.4, 12.1**

  - [ ]* 19.4 Write property test: Property 3 — Topic composite score correctness
    - **Property 3: Topic composite score correctness**
    - **Validates: Requirements 2.2**

  - [ ]* 19.5 Write property test: Property 4 — Topic list serialization round-trip
    - **Property 4: Topic list serialization round-trip**
    - **Validates: Requirements 2.3**

  - [ ]* 19.6 Write property test: Property 5 — Batch topic deduplication
    - **Property 5: Batch topic deduplication**
    - **Validates: Requirements 2.6**

  - [ ]* 19.7 Write property test: Property 6 — Script word count invariant
    - **Property 6: Script word count invariant**
    - **Validates: Requirements 3.1, 3.5**

  - [ ]* 19.8 Write property test: Property 7 — Script always references its style profile
    - **Property 7: Script always references its style profile**
    - **Validates: Requirements 3.2**

  - [ ]* 19.9 Write property test: Property 8 — Script annotation density invariant
    - **Property 8: Script annotation density invariant**
    - **Validates: Requirements 3.3**

  - [ ]* 19.10 Write property test: Property 9 — Script version monotonicity
    - **Property 9: Script version monotonicity**
    - **Validates: Requirements 3.4, 4.4**

  - [ ]* 19.11 Write property test: Property 10 — Narration audio specification invariant
    - **Property 10: Narration audio specification invariant**
    - **Validates: Requirements 5.2**


- [ ] 20. Write property-based tests — Properties 11–20 (`tests/property/test_properties_11_to_20.py`)
  - [ ]* 20.1 Write property test: Property 11 — Retry count and delay bounds
    - **Property 11: Retry count and delay bounds**
    - **Validates: Requirements 1.7, 5.4, 9.5, 12.4, 13.1, 13.2**

  - [ ]* 20.2 Write property test: Property 12 — Video output specification invariant
    - **Property 12: Video output specification invariant**
    - **Validates: Requirements 6.1**

  - [ ]* 20.3 Write property test: Property 13 — Thumbnail specification invariant
    - **Property 13: Thumbnail specification invariant**
    - **Validates: Requirements 6.3**

  - [ ]* 20.4 Write property test: Property 14 — Clip fallback after all retries exhausted
    - **Property 14: Clip fallback after all retries exhausted**
    - **Validates: Requirements 6.6**

  - [ ]* 20.5 Write property test: Property 15 — MetadataPackage structural invariant
    - **Property 15: MetadataPackage structural invariant**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4**

  - [ ]* 20.6 Write property test: Property 16 — MetadataPackage serialization round-trip
    - **Property 16: MetadataPackage serialization round-trip**
    - **Validates: Requirements 7.5**

  - [ ]* 20.7 Write property test: Property 17 — Selective regeneration isolation
    - **Property 17: Selective regeneration isolation**
    - **Validates: Requirements 8.3**

  - [ ]* 20.8 Write property test: Property 18 — Publish datetime validation
    - **Property 18: Publish datetime validation**
    - **Validates: Requirements 9.4**

  - [ ]* 20.9 Write property test: Property 19 — Calendar rollback on persistent update failure
    - **Property 19: Calendar rollback on persistent update failure**
    - **Validates: Requirements 9.6**

  - [ ]* 20.10 Write property test: Property 20 — VideoRecord schema invariant
    - **Property 20: VideoRecord schema invariant**
    - **Validates: Requirements 10.1**


- [ ] 21. Write property-based tests — Properties 21–30 (`tests/property/test_properties_21_to_30.py`)
  - [ ]* 21.1 Write property test: Property 21 — Calendar conflict and gap detection
    - **Property 21: Calendar conflict and gap detection**
    - **Validates: Requirements 10.4**

  - [ ]* 21.2 Write property test: Property 22 — Calendar rejects invalid datetime updates
    - **Property 22: Calendar rejects invalid datetime updates**
    - **Validates: Requirements 10.6**

  - [ ]* 21.3 Write property test: Property 23 — Batch count invariant
    - **Property 23: Batch count invariant**
    - **Validates: Requirements 11.1, 11.2**

  - [ ]* 21.4 Write property test: Property 24 — Batch schedule even spacing
    - **Property 24: Batch schedule even spacing**
    - **Validates: Requirements 11.4**

  - [ ]* 21.5 Write property test: Property 25 — Batch completion percentage formula
    - **Property 25: Batch completion percentage formula**
    - **Validates: Requirements 11.6**

  - [ ]* 21.6 Write property test: Property 26 — Structured log entry completeness
    - **Property 26: Structured log entry completeness**
    - **Validates: Requirements 13.6**

  - [ ]* 21.7 Write property test: Property 27 — Batch completion notification completeness
    - **Property 27: Batch completion notification completeness**
    - **Validates: Requirements 14.4**

  - [ ]* 21.8 Write property test: Property 28 — Notification deduplication invariant
    - **Property 28: Notification deduplication invariant**
    - **Validates: Requirements 14.6**

  - [ ]* 21.9 Write property test: Property 29 — Cross-post caption fits platform character limit
    - **Property 29: Cross-post caption fits platform character limit**
    - **Validates: Requirements 15.2, 15.3**

  - [ ]* 21.10 Write property test: Property 30 — Cross-post platform failure isolation
    - **Property 30: Cross-post platform failure isolation**
    - **Validates: Requirements 15.6**


- [ ] 22. Write integration tests (`tests/integration/`)
  - [ ]* 22.1 Write ElevenLabs TTS integration test
    - Call ElevenLabs API with a valid short script → assert MP3 returned, sample rate = 44,100 Hz,
      bitrate ≥ 128 kbps.
    - _Requirements: 5.1, 5.2_

  - [ ]* 22.2 Write YouTube Data API integration test
    - Upload a minimal test MP4 with dummy metadata → assert YouTube video ID returned and privacy
      state is `Unlisted`.
    - _Requirements: 9.1, 9.2_

  - [ ]* 22.3 Write Google Drive MCP integration test
    - Write a test file to `Asset_Store` → assert Drive URL returned within 10 seconds.
    - _Requirements: 12.2_

  - [ ]* 22.4 Write Notion MCP integration test
    - Create a `VideoRecord` in Content_Calendar → update status → assert record reflects new
      status.
    - _Requirements: 10.1, 10.2_

  - [ ]* 22.5 Write Slack/Discord/SMTP notification delivery integration test
    - Send a `review_gate` notification event → assert delivery success response from channel.
    - _Requirements: 14.1, 14.3_

  - [ ]* 22.6 Write Perplexity/Tavily query integration test
    - Query for AI topics → assert ≥ 5 `TopicEntry` objects with required fields returned.
    - _Requirements: 2.1, 2.3_

- [ ] 23. Write smoke tests (`tests/smoke/test_configuration_smoke.py`)
  - [ ]* 23.1 Implement configuration smoke tests
    - Assert all required API credentials present and non-empty at pipeline start.
    - Assert Google Drive folder hierarchy `ai-youtube-pipeline/` exists or can be created.
    - Assert Notion database has all required columns from `VideoRecord` schema.
    - Assert ElevenLabs `voice_id` resolves to a valid voice.
    - _Requirements: 5.7, 12.1, 10.1_


- [x] 24. Final integration and wiring
  - [x] 24.1 Wire all subsystems into Orchestrator entry points
    - Connect `Orchestrator.start_pipeline` to the complete subsystem chain using dependency
      injection for all subsystem instances.
    - Wire `Orchestrator.handle_review_response` to Gate 1 and Gate 2 handlers.
    - Wire `Orchestrator.schedule_video` to `Publisher.schedule`.
    - Expose a CLI entry point (`pipeline/cli.py`) accepting `--config`, `--batch-size`, and
      `--resume-run-id` flags.
    - _Requirements: 1–15 (end-to-end integration)_

  - [ ]* 24.2 Write end-to-end integration smoke test
    - Run a single-video pipeline in dry-run mode (all external calls mocked) asserting correct
      state machine progression from `Pending` through to `Unlisted` with all Content_Calendar
      status updates and Asset_Store writes verified.
    - _Requirements: 10.2, 13.3, 13.6_

- [x] 25. Final checkpoint — All tests pass
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; all 30 correctness
  properties and integration tests are marked optional.
- Each task references specific requirements for full traceability to the requirements document.
- Hypothesis property tests run a minimum of 100 iterations per test; configure via
  `@settings(max_examples=100)`.
- The design uses Python throughout; all code must be type-annotated and pass `mypy --strict`.
- Subsystem classes are designed for dependency injection to simplify unit testing with mocks.
- Structured JSON logs (`LogEntry`) are written for every API call and state transition; required
  fields are `timestamp`, `event_type`, `stage_name`, and `http_response_code`.
- All external API credentials are read from environment variables or a secrets manager — never
  hardcoded.


## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "3.1", "3.2", "4.1", "4.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.3", "3.4", "3.5", "3.6", "4.3", "4.4", "4.5"] },
    { "id": 3, "tasks": ["6.1", "7.1", "7.2"] },
    { "id": 4, "tasks": ["6.2", "6.3", "7.3", "7.4", "7.5", "7.6"] },
    { "id": 5, "tasks": ["8.1", "8.2", "8.3"] },
    { "id": 6, "tasks": ["8.4", "8.5", "8.6", "8.7", "8.8"] },
    { "id": 7, "tasks": ["10.1", "11.1"] },
    { "id": 8, "tasks": ["10.2", "10.3", "11.2"] },
    { "id": 9, "tasks": ["11.3", "11.4", "11.5", "11.6"] },
    { "id": 10, "tasks": ["12.1"] },
    { "id": 11, "tasks": ["12.2", "12.3", "12.4"] },
    { "id": 12, "tasks": ["14.1", "14.2", "14.3", "15.1", "15.2"] },
    { "id": 13, "tasks": ["14.4", "14.5", "14.6", "15.3", "15.4", "15.5"] },
    { "id": 14, "tasks": ["16.1", "16.2", "16.3", "16.4"] },
    { "id": 15, "tasks": ["16.5", "16.6", "17.1", "17.2"] },
    { "id": 16, "tasks": ["17.3", "17.4", "17.5", "17.6"] },
    { "id": 17, "tasks": ["19.1"] },
    { "id": 18, "tasks": ["19.2", "19.3", "19.4", "19.5", "19.6", "19.7", "19.8", "19.9", "19.10", "19.11"] },
    { "id": 19, "tasks": ["20.1", "20.2", "20.3", "20.4", "20.5", "20.6", "20.7", "20.8", "20.9", "20.10"] },
    { "id": 20, "tasks": ["21.1", "21.2", "21.3", "21.4", "21.5", "21.6", "21.7", "21.8", "21.9", "21.10"] },
    { "id": 21, "tasks": ["22.1", "22.2", "22.3", "22.4", "22.5", "22.6", "23.1"] },
    { "id": 22, "tasks": ["24.1"] },
    { "id": 23, "tasks": ["24.2"] }
  ]
}
```
