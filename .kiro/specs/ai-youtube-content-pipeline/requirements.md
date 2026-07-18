# Requirements Document

## Introduction

An automated AI-powered content pipeline that produces YouTube videos from topic discovery through
publishing with minimal manual effort. The system analyzes a reference channel to replicate its
storytelling style, researches trending AI topics, generates scripts, AI narration, visuals, and
thumbnails, then manages the full publishing lifecycle including review gating, scheduling, and
cross-platform distribution. A content calendar in Notion provides scheduling and batch management,
while Google Drive stores all generated assets. The user is notified via Slack, Discord, or email
at each human-review checkpoint.

---

## Glossary

- **Pipeline**: The end-to-end automated workflow from topic discovery to published video.
- **Orchestrator**: The Claude-based agent that coordinates all pipeline stages and tool calls.
- **Reference_Analyzer**: The subsystem that extracts style, pacing, structure, and thumbnail
  patterns from a reference YouTube channel using Browser MCP.
- **Topic_Researcher**: The subsystem that discovers trending AI topics using Perplexity or
  Tavily MCP.
- **Script_Writer**: The subsystem that generates a video script using Claude, modeled on the
  reference channel's storytelling style profile.
- **Narration_Generator**: The subsystem that converts a script into audio using ElevenLabs.
- **Visual_Generator**: The subsystem that creates video footage and thumbnails using Viewmax MCP.
- **Metadata_Generator**: The subsystem that produces SEO-optimized title, description, tags,
  chapters, and hashtags using Claude.
- **Publisher**: The subsystem that uploads content to YouTube via the YouTube Data API or
  YouTube MCP and manages publish state transitions.
- **Cross_Poster**: The subsystem that distributes published content to X, LinkedIn, Instagram,
  and Facebook.
- **Asset_Store**: Google Drive folders managed by the Google Drive MCP that archive all
  pipeline-generated files.
- **Content_Calendar**: A Notion database managed by the Notion MCP that tracks video status,
  scheduled publish dates, and batch queues.
- **Notifier**: The subsystem that sends alerts to the user via Slack, Discord, or email at
  review checkpoints and failure events.
- **Style_Profile**: A structured document produced by the Reference_Analyzer that captures the
  reference channel's narration tone (sentiment polarity −1 to +1), pacing, segment structure,
  visual style, and thumbnail composition.
- **Review_Gate**: A mandatory human-approval checkpoint that pauses the pipeline and awaits
  explicit user confirmation before proceeding.
- **Batch**: A set of videos queued for creation and scheduling in a single pipeline run.
- **Unlisted**: A YouTube privacy state in which a video is accessible via direct link but does
  not appear in search or channel listings.

---

## Requirements

### Requirement 1: Reference Channel Style Analysis

**User Story:** As a content creator, I want the pipeline to analyze a reference YouTube channel,
so that generated scripts and visuals consistently match the channel's proven storytelling style.

#### Acceptance Criteria

1. WHEN a reference channel URL is provided, THE Reference_Analyzer SHALL extract the Style_Profile
   covering narration tone (expressed as sentiment polarity on a scale of −1 to +1), average pacing
   (words per minute), average sentence length (in words), segment structure (intro, hook, body,
   CTA), visual composition patterns, and thumbnail layout within 5 minutes.
2. THE Reference_Analyzer SHALL store the Style_Profile as a versioned JSON document in the
   Asset_Store under a `style-profiles/` folder.
3. WHEN a new Style_Profile is generated, THE Reference_Analyzer SHALL update the Content_Calendar
   record for the active batch to reference the Style_Profile document ID.
4. IF the Reference_Analyzer fails to access the reference channel URL, THEN THE Reference_Analyzer
   SHALL log the error, notify the Notifier, and halt the current pipeline run without writing a
   Style_Profile JSON file to the Asset_Store.
5. WHEN thumbnail composition data is required, THE Reference_Analyzer SHALL extract dominant color
   palette (up to 5 hex values), text overlay position, and subject framing from a minimum of 10
   uploads published within the last 90 days on the reference channel.
6. IF the reference channel has fewer than 10 qualifying uploads published within the last 90 days,
   THEN THE Reference_Analyzer SHALL use all available uploads (minimum 1), log a warning, and
   proceed with extraction.
7. IF the Asset_Store write for the Style_Profile fails, THEN THE Reference_Analyzer SHALL retry
   up to 3 times with a 10-second back-off interval between attempts before notifying the Notifier
   and halting the current pipeline run.

---

### Requirement 2: Trending Topic Research

**User Story:** As a content creator, I want the pipeline to surface trending AI topics automatically,
so that every video targets high-relevance subjects without manual research.

#### Acceptance Criteria

1. WHEN a pipeline run is initiated, THE Topic_Researcher SHALL query Perplexity or Tavily for AI
   topics trending within the past 72 hours and return a ranked list of at least 5 candidate topics,
   where each topic entry includes: title (1–200 characters), recency in hours since first
   appearance, and composite score (0.00–1.00).
2. THE Topic_Researcher SHALL rank topics by a composite score derived from three equally weighted
   dimensions (each weighted 1/3): search volume signal, recency (hours since first appearance),
   and relevance to AI/ML — where relevance is determined by matching at least one of the following
   subject tags: "machine learning", "large language model", "neural network", "AI safety", or
   "generative AI". Each dimension SHALL be normalized using min-max normalization before computing
   the composite score.
3. THE Topic_Researcher SHALL store the ranked topic list as a JSON document in the Asset_Store
   under a `research/` folder, timestamped to the nearest minute. The JSON document SHALL include
   at minimum the following fields per entry: `title`, `composite_score`, `recency_hours`, and
   `source_query_timestamp`.
4. IF the Topic_Researcher returns between 1 and 4 topics (partial results) without having
   exhausted all retry attempts, THEN THE Topic_Researcher SHALL halt the pipeline, log the partial
   count, and not store the incomplete list in the Asset_Store.
5. IF the Topic_Researcher receives no results after 3 consecutive retry attempts with 30-second
   intervals, THEN THE Topic_Researcher SHALL notify the Notifier with a message containing the
   pipeline run identifier and the failure reason string, and halt the current pipeline run.
6. WHERE batch mode is enabled, THE Topic_Researcher SHALL produce one candidate topic per queued
   video slot in the batch, with a maximum batch size of 50 topics, without repeating topics used
   in the preceding 30 days as recorded in the Content_Calendar. Deduplication SHALL be performed
   by case-insensitive title matching.

---

### Requirement 3: Script Generation

**User Story:** As a content creator, I want the pipeline to write original video scripts that match
the reference channel's style, so that the output feels authentic and on-brand.

#### Acceptance Criteria

1. WHEN a topic and Style_Profile are both available (successfully loaded and non-empty), THE
   Script_Writer SHALL generate a script between 800 and 1,500 words structured as: hook (≤60
   seconds spoken at 150 words per minute), body (3–5 segments), and CTA (≤30 seconds spoken at
   150 words per minute).
2. WHEN a topic and Style_Profile are loaded, THE Script_Writer SHALL apply the narration tone,
   pacing, and rhetorical patterns defined in the Style_Profile when generating the script. The
   script metadata SHALL reference the Style_Profile document ID used during generation.
3. WHEN a topic and Style_Profile are loaded, THE Script_Writer SHALL include speaker-direction
   annotations (e.g., `[pause]`, `[emphasis]`) aligned with the pacing data from the Style_Profile,
   with a minimum annotation density of at least 1 annotation per script segment.
4. WHEN a script is generated, THE Script_Writer SHALL store the generated script as a Markdown
   file in the Asset_Store under a `scripts/` folder, using a versioning scheme of sequential
   integer suffixes (e.g., `script_v1.md`, `script_v2.md`).
5. IF the Script_Writer produces a script that exceeds 1,500 words or falls below 800 words, THEN
   THE Script_Writer SHALL automatically revise the script once to fall within the specified range;
   IF the revised script still falls outside the 800–1,500 word range, THEN THE Pipeline SHALL
   halt the script stage, update the Content_Calendar status to `Pipeline Error — script_generation`,
   and notify the Notifier.
6. WHEN a script word count is verified to be within the 800–1,500 word range, THE Pipeline SHALL
   trigger a Review_Gate, upload the script to the Content_Calendar entry, and notify the user via
   the Notifier with a direct link to the script.

---

### Requirement 4: Human Review Gate — Script

**User Story:** As a content creator, I want to review and edit the script before narration is
generated, so that I can correct factual errors or adjust tone before committing to audio production.

#### Acceptance Criteria

1. WHEN the Script Review_Gate is triggered, THE Pipeline SHALL pause all downstream stages for
   that video until the user provides explicit approval or submits edits.
2. WHEN the Script Review_Gate is triggered, THE Notifier SHALL send the user a notification within
   60 seconds containing the script link and the available actions (approve or submit edits).
3. WHEN the Script Review_Gate is open, THE Content_Calendar SHALL reflect a status of
   `Awaiting Script Review`.
4. WHEN the user submits edits to the script, THE Script_Writer SHALL save the revised script as a
   new version in the Asset_Store, preserving the previous version.
5. WHEN the user approves the script, THE Pipeline SHALL advance the video to the narration stage
   and update the Content_Calendar status to `Script Approved`.
6. IF the Script Review_Gate remains open for more than 48 hours without user action, THEN THE
   Notifier SHALL send a follow-up reminder to the user once every 24 hours, up to a maximum of
   3 reminders.
7. IF the user submits an edit with no content changes, THEN THE Pipeline SHALL reject the
   submission and return a validation error without advancing the gate.

---

### Requirement 5: AI Narration Generation

**User Story:** As a content creator, I want the pipeline to generate natural-sounding AI narration
from the approved script, so that I do not need to record voiceovers manually.

#### Acceptance Criteria

1. WHEN a script is approved, THE Narration_Generator SHALL submit the script text (1–5,000
   characters) to ElevenLabs using the configured voice ID and return an MP3 audio file.
2. THE Narration_Generator SHALL request audio at a sample rate of 44,100 Hz and a bitrate of at
   least 128 kbps.
3. WHEN narration audio is stored, THE Narration_Generator SHALL save the generated MP3 in the
   Asset_Store under a `narration/` folder, named with the video ID and version number in the
   format `v{n}` where `n` is a positive integer incremented per generation attempt per video ID.
4. IF the ElevenLabs API returns an error, THEN THE Narration_Generator SHALL retry up to 3 times
   with exponential back-off starting at 5 seconds, with a maximum back-off delay of 60 seconds
   per retry interval, before notifying the Notifier and halting the narration stage.
5. WHEN narration is successfully generated, THE Pipeline SHALL update the Content_Calendar status
   to `Narration Ready`.
6. WHEN narration is successfully generated, THE Pipeline SHALL pass the Asset_Store MP3 file path
   to the Visual_Generator.
7. IF the configured voice ID is absent or empty at pipeline start, THEN THE Narration_Generator
   SHALL notify the Notifier and halt before making any API call.
8. IF the Asset_Store write for the narration file fails after all retries are exhausted, THEN THE
   Narration_Generator SHALL discard the audio file, notify the Notifier, and halt the narration
   stage.

---

### Requirement 6: AI Visual and Video Generation

**User Story:** As a content creator, I want the pipeline to generate relevant video footage and a
styled thumbnail automatically, so that I do not need to source or edit media manually.

#### Acceptance Criteria

1. WHEN narration audio and the approved script are available, THE Visual_Generator SHALL submit
   scene prompts derived from each script segment to Viewmax MCP and compile the resulting clips
   into a single MP4 video synchronized with the narration audio within ±100ms audio sync
   tolerance. The output MP4 SHALL meet a minimum specification of 1920×1080 resolution, H.264
   codec, and 24 frames per second minimum.
2. WHEN the Style_Profile thumbnail composition data is available, THE Visual_Generator SHALL
   generate a thumbnail image using the dominant color palette, text overlay position, and subject
   framing from the Style_Profile.
3. WHEN a thumbnail is generated, THE Visual_Generator SHALL produce the thumbnail as a JPEG image
   with dimensions of 1,280 × 720 pixels and a file size under 2 MB.
4. WHEN video and thumbnail generation are complete, THE Visual_Generator SHALL store the compiled
   MP4 and thumbnail JPEG in the Asset_Store under `videos/` and `thumbnails/` folders
   respectively, named with the video ID and version number.
5. THE Visual_Generator SHALL generate only original artwork such that no generated frame is a
   pixel-for-pixel reproduction of any frame from the reference channel.
6. IF Viewmax MCP returns an error for any scene clip, THEN THE Visual_Generator SHALL retry that
   clip up to 3 times with a retry delay of 5–30 seconds before substituting a fallback static
   image (JPEG, 1920×1080) and logging the substitution.
7. WHEN video and thumbnail generation are complete, THE Pipeline SHALL update the Content_Calendar
   status to `Visuals Ready` and trigger the metadata generation stage.

---

### Requirement 7: SEO Metadata Generation

**User Story:** As a content creator, I want the pipeline to generate fully optimized YouTube
metadata automatically, so that every upload is search-ready without manual copywriting.

#### Acceptance Criteria

1. WHEN a video script and topic research data are available, THE Metadata_Generator SHALL produce
   a YouTube title of 60 characters or fewer that includes the primary keyword, defined as the
   highest-scoring topic from the Topic_Researcher's ranked list.
2. WHEN a video script and topic research data are available, THE Metadata_Generator SHALL produce
   a video description of 200–500 words containing: a one-paragraph summary, timestamped chapter
   markers aligned to the script segments, 3–5 relevant links or CTAs, and a closing paragraph
   containing at least 3 of the tags from the tag list.
3. WHEN a video script and topic research data are available, THE Metadata_Generator SHALL produce
   a tag list of 10–15 tags, each tag 2–5 words, covering the primary topic, related subtopics,
   and channel brand terms.
4. WHEN a video script and topic research data are available, THE Metadata_Generator SHALL produce
   3–5 hashtags for the YouTube description footer, where each hashtag is prefixed with `#`, contains
   no spaces, and is 2–30 characters excluding the `#`. Hashtags SHALL be sourced from the tag list
   or primary keyword.
5. WHEN metadata fields are validated, THE Metadata_Generator SHALL store the complete metadata
   package as a JSON document in the Asset_Store under a `metadata/` folder, named with the
   video ID.
6. WHEN metadata is ready, THE Pipeline SHALL trigger a Review_Gate, attach the metadata to the
   Content_Calendar entry, and notify the user via the Notifier with a direct link.
7. IF any metadata field fails validation, THEN THE Metadata_Generator SHALL regenerate only that
   field once; IF the regenerated field still fails validation, THEN THE Pipeline SHALL halt and
   notify the Notifier.
8. IF the script or topic research data is absent when metadata generation is triggered, THEN THE
   Metadata_Generator SHALL not attempt generation and SHALL notify the Notifier.

---

### Requirement 8: Human Review Gate — Final Video and Metadata

**User Story:** As a content creator, I want to review the final video, thumbnail, and metadata
before upload, so that I can catch issues before the video is visible on YouTube.

#### Acceptance Criteria

1. WHEN the Final Review_Gate is triggered, THE Pipeline SHALL present the user with links to the
   MP4, thumbnail JPEG, and metadata JSON stored in the Asset_Store within 5 minutes of the gate
   being triggered.
2. WHEN the Final Review_Gate is open, THE Content_Calendar SHALL reflect a status of
   `Awaiting Final Review`.
3. WHEN the user requests a re-generation of any individual asset (video, thumbnail, or metadata),
   THE Pipeline SHALL regenerate only the requested asset and re-trigger the Final Review_Gate
   without regenerating the others.
4. WHEN the user approves all assets, THE Pipeline SHALL advance the video to the upload stage and
   update the Content_Calendar status to `Approved for Upload`.
5. IF the Final Review_Gate remains open for more than 72 hours from the moment the gate was
   triggered without user action, THEN THE Pipeline SHALL automatically approve all assets, advance
   the video to the upload stage, update the Content_Calendar status to `Auto-Approved for Upload`,
   and notify the user via the Notifier. The 72-hour timer SHALL reset to 72 hours after each
   user-requested re-generation.
6. IF any of the three asset links (MP4, thumbnail, metadata) is unavailable when the Final
   Review_Gate is triggered, THEN THE Pipeline SHALL halt, log the missing asset, and notify the
   Notifier without opening the gate.

---

### Requirement 9: YouTube Upload and Scheduling

**User Story:** As a content creator, I want the pipeline to upload the video to YouTube as
unlisted and then schedule it for publishing, so that I can verify the upload before it goes live.

#### Acceptance Criteria

1. WHEN a video is approved for upload, THE Publisher SHALL upload the MP4, thumbnail JPEG, title,
   description, tags, and chapters to YouTube via the YouTube Data API and set the privacy state to
   `Unlisted`.
2. WHEN the upload is initiated, THE Publisher SHALL make the YouTube video ID and the unlisted
   watch URL available to the Notifier within 10 minutes of initiating the upload.
3. WHEN the upload succeeds, THE Notifier SHALL send the user the unlisted watch URL and prompt
   the user to confirm the scheduled publish date and time.
4. WHEN the user confirms a publish datetime, THE Publisher SHALL validate that the confirmed
   publish datetime is not in the past and is not within 15 minutes of the current time; IF the
   datetime fails this validation, THEN THE Publisher SHALL reject it, notify the user with an
   error message, and re-prompt for a valid datetime. WHEN a valid datetime is confirmed, THE
   Publisher SHALL schedule the video to transition from `Unlisted` to `Public` at that datetime
   using the YouTube Data API.
5. IF the upload fails, THEN THE Publisher SHALL retry up to 3 times with exponential back-off
   starting at 60 seconds, with a maximum back-off delay of 300 seconds per retry interval, before
   notifying the Notifier with the error details and halting the upload stage.
6. WHEN the video transitions to `Public`, THE Publisher SHALL update the Content_Calendar status
   to `Published` and record the actual publish timestamp; IF the Content_Calendar update fails
   after 3 retry attempts, THEN THE Publisher SHALL revert the video privacy state to `Unlisted`
   and notify the Notifier with the rollback details.
7. IF the user does not confirm a publish datetime within 7 days of the upload success notification,
   THEN THE Publisher SHALL leave the video as `Unlisted`, update the Content_Calendar status to
   `Awaiting Schedule`, and notify the Notifier.

---

### Requirement 10: Content Calendar Management

**User Story:** As a content creator, I want a Notion-based content calendar that tracks every
video's status and schedule, so that I can manage my publishing pipeline at a glance.

#### Acceptance Criteria

1. THE Content_Calendar SHALL maintain a Notion database record for each video containing: video
   ID, title, topic, current status, scheduled publish datetime (in UTC), asset links (script,
   audio, video, thumbnail, metadata), and pipeline run timestamp. Valid status values are:
   `Pending`, `Researching`, `Scripting`, `Awaiting Script Review`, `Script Approved`,
   `Narration Ready`, `Generating Visuals`, `Visuals Ready`, `Generating Metadata`,
   `Awaiting Final Review`, `Approved for Upload`, `Uploading`, `Unlisted`, `Awaiting Schedule`,
   `Scheduled`, `Published`, `Auto-Approved for Upload`, and `Pipeline Error — {stage_name}`.
2. WHEN any pipeline stage transitions, THE Orchestrator SHALL update the corresponding
   Content_Calendar record status field to the new stage value within 30 seconds of the transition.
3. WHERE batch mode is enabled, THE Content_Calendar SHALL display all videos in the current batch
   grouped under a shared batch ID with individual per-video status fields.
4. WHEN the user opens the calendar view, THE Content_Calendar SHALL render a date-based calendar
   view of scheduled publish datetimes, highlighting conflicts (defined as two videos sharing the
   same scheduled publish datetime) and gaps (defined as 7 or more consecutive days without a
   scheduled video).
5. WHEN a user updates the scheduled publish datetime directly in the Content_Calendar, THE
   Publisher SHALL read the updated datetime and reschedule the corresponding YouTube video
   accordingly within 5 minutes, provided the new datetime is in the future and the video has not
   yet transitioned to `Published`.
6. IF the user sets a scheduled publish datetime that is in the past, or if the video is already
   `Published`, THEN THE Content_Calendar SHALL reject the update, display a validation error, and
   leave the existing datetime unchanged.

---

### Requirement 11: Batch Video Creation and Scheduling

**User Story:** As a content creator, I want to queue and process multiple videos in a single
pipeline run, so that I can maintain a consistent publishing cadence without triggering individual
runs.

#### Acceptance Criteria

1. WHEN batch mode is initiated with a target count N (where N is between 2 and 10 inclusive), THE
   Pipeline SHALL create N Content_Calendar entries and execute the research, script, narration,
   and visual stages for each video sequentially.
2. IF N is outside the range 2–10 inclusive, THEN THE Pipeline SHALL reject the batch initiation,
   return a validation error, and not create any Content_Calendar entries.
3. WHILE batch processing is active, THE Pipeline SHALL process batch videos one at a time, waiting
   for the prior video's generation stages to complete before starting the next.
4. WHEN all videos in a batch have passed their Final Review_Gate, THE Orchestrator SHALL present
   the user with a batch scheduling interface that suggests evenly spaced publish datetimes across
   the next 14 days, where "evenly spaced" means equal time intervals between publish datetimes
   anchored to the current date and time, with one slot per video.
5. IF a pipeline stage fails for a video in the batch after all retries are exhausted, THEN THE
   Pipeline SHALL mark that video's status as `Pipeline Error — {stage_name}`, continue processing
   the remaining batch videos, and notify the Notifier.
6. WHEN a video in the batch reaches `Published` status, THE Content_Calendar SHALL update the
   batch completion percentage, calculated as a whole number rounded down using the formula
   `floor(completed_videos / N * 100)`.

---

### Requirement 12: Asset Organization in Google Drive

**User Story:** As a content creator, I want all pipeline-generated files stored in an organized
Google Drive structure, so that I can find any asset quickly without searching.

#### Acceptance Criteria

1. THE Asset_Store SHALL organize files in Google Drive using the folder hierarchy
   `ai-youtube-pipeline/{video_id}/` with the following six sub-folders nested under it:
   `scripts/`, `narration/`, `videos/`, `thumbnails/`, `metadata/`, and `research/`. The
   `video_id` SHALL be a pipeline-assigned identifier consisting of 1–128 alphanumeric characters,
   hyphens, or underscores.
2. WHEN a file is saved to the Asset_Store, THE Asset_Store SHALL return a Google Drive URL
   accessible to any authenticated Google account holder granted viewer access within 10 seconds.
3. THE Asset_Store SHALL retain all files written during a pipeline run for a minimum of 90 days
   from the pipeline run date, with no early deletion permitted.
4. IF a Google Drive API call fails, THEN THE Asset_Store SHALL retry up to 3 times with
   exponential back-off starting at 10 seconds, with a maximum back-off delay of 80 seconds per
   retry interval, before logging the failure, notifying the Notifier, and marking the pipeline
   run as failed.

---

### Requirement 13: Failure Handling and Automatic Retry

**User Story:** As a content creator, I want the pipeline to recover from transient failures
automatically, so that minor API errors do not require me to restart the entire process.

#### Acceptance Criteria

1. WHEN any external API call fails with a transient HTTP error (429, 500, 502, 503, or 504),
   THE Orchestrator SHALL retry the call using exponential back-off: 5 s, 10 s, 20 s (maximum 3
   retries) before marking the stage as failed.
2. IF an external API call fails with a non-transient HTTP error (400, 401, 403, or 404), THEN THE
   Orchestrator SHALL immediately mark the stage as failed without retrying.
3. WHEN a stage is marked as failed after exhausting retries, THE Orchestrator SHALL preserve all
   successfully completed stage outputs in the Asset_Store and update the Content_Calendar status
   to `Pipeline Error — {stage_name}`.
4. IF a pipeline stage fails, THEN THE Notifier SHALL send the user an alert containing the failed
   stage name, an error message indicating the cause of the failure, and a link to the
   Content_Calendar entry within 60 seconds of the failure.
5. WHEN the user selects the retry action on a failed Content_Calendar entry, THE Pipeline SHALL
   resume from the failed stage, reusing all previously completed stage outputs stored in the
   Asset_Store.
6. WHEN any external API call fails with a transient HTTP error, THE Orchestrator SHALL log every
   API call, response code, retry attempt, and stage transition to a structured JSON log file
   stored in the Asset_Store under a `logs/` folder. Required log fields are: `timestamp` (UTC
   ISO-8601), `event_type`, `stage_name`, and `http_response_code`.
7. IF a previously completed stage output is missing or unreadable from the Asset_Store at retry
   time, THEN THE Pipeline SHALL restart from that stage rather than skipping it.

---

### Requirement 14: User Notifications

**User Story:** As a content creator, I want to receive timely notifications at each review
checkpoint and failure event, so that I can take action without constantly monitoring the pipeline.

#### Acceptance Criteria

1. THE Notifier SHALL support Slack webhook, Discord webhook, or email via SMTP as delivery
   channels, configurable per user.
2. IF no delivery channel is configured, THEN THE Notifier SHALL suppress all notification
   attempts and log a warning to the structured log in the Asset_Store.
3. WHEN a Review_Gate is triggered, THE Notifier SHALL send a notification within 60 seconds
   containing: the video title, the review type (Script or Final), a direct link to the relevant
   asset(s), and the action required from the user.
4. WHEN a pipeline run completes successfully for all videos in a batch, THE Notifier SHALL send
   a summary notification listing each video title, its Content_Calendar status, and its scheduled
   publish datetime.
5. WHEN a pipeline stage fails after all retries are exhausted, IF the failure alert has not
   already been sent for this event, THEN THE Notifier SHALL send an alert within 60 seconds
   containing the video ID, failed stage name, error message, and scheduled publish datetime in
   UTC.
6. WHILE the deduplication window is active, THE Notifier SHALL not send duplicate notifications
   for the same event within a 10-minute window, where the deduplication key is defined as
   (`video_id` + `notification_type` + `stage_name`).
7. IF a configured delivery channel fails to deliver a notification after 2 attempts, THEN THE
   Notifier SHALL log the delivery failure and attempt delivery on the next configured channel if
   available.

---

### Requirement 15: Optional Cross-Platform Distribution

**User Story:** As a content creator, I want the pipeline to optionally post published videos to
X, LinkedIn, Instagram, and Facebook, so that I can reach broader audiences without manual
social media work.

#### Acceptance Criteria

1. WHERE cross-posting is enabled for a specific platform, THE Cross_Poster SHALL post a
   platform-formatted update within 30 minutes of the YouTube video transitioning to `Public`.
2. WHERE cross-posting is enabled, THE Cross_Poster SHALL generate a platform-native caption for
   each enabled platform derived from the video title, description, and tags, with hashtags
   counting within the following character limits: X (≤280 characters), LinkedIn (≤3,000
   characters), Instagram (≤2,200 characters with hashtags), Facebook (≤500 characters).
3. WHEN a cross-platform post is dispatched to an enabled platform, THE Cross_Poster SHALL include
   the YouTube video URL and at least 3 hashtags in that post, within the platform's character
   limit defined in criterion 2.
4. IF a cross-platform post fails, THEN THE Cross_Poster SHALL retry up to 2 times with a
   60-second interval before logging the failure and notifying the Notifier with the platform name
   and failure reason.
5. WHERE cross-posting is disabled for a platform, THE Cross_Poster SHALL skip that platform
   entirely without raising an error or notification.
6. IF a cross-platform post fails for one enabled platform, THEN THE Cross_Poster SHALL continue
   posting to the remaining enabled platforms without halting.
