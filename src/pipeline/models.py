"""Shared Pydantic data models for the AI YouTube Content Pipeline.

All models defined here correspond to the Data Models section of the design document.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PipelineStatus(str, Enum):
    """18 pipeline status values driven by the Orchestrator state machine."""

    PENDING = "Pending"
    RESEARCHING = "Researching"
    SCRIPTING = "Scripting"
    AWAITING_SCRIPT_REVIEW = "Awaiting Script Review"
    SCRIPT_APPROVED = "Script Approved"
    NARRATION_READY = "Narration Ready"
    GENERATING_VISUALS = "Generating Visuals"
    VISUALS_READY = "Visuals Ready"
    GENERATING_METADATA = "Generating Metadata"
    AWAITING_FINAL_REVIEW = "Awaiting Final Review"
    APPROVED_FOR_UPLOAD = "Approved for Upload"
    AUTO_APPROVED_FOR_UPLOAD = "Auto-Approved for Upload"
    UPLOADING = "Uploading"
    UNLISTED = "Unlisted"
    SCHEDULED = "Scheduled"
    PUBLISHED = "Published"
    PIPELINE_ERROR = "Pipeline Error"
    SCRIPT_REJECTED = "Script Rejected"
    VIDEO_REJECTED = "Video Rejected"


class SubFolder(str, Enum):
    """Asset_Store sub-folder names under each video_id directory."""

    SCRIPTS = "scripts"
    NARRATION = "narration"
    VIDEOS = "videos"
    THUMBNAILS = "thumbnails"
    METADATA = "metadata"
    RESEARCH = "research"
    LOGS = "logs"
    STYLE_PROFILES = "style-profiles"


class Platform(str, Enum):
    """Supported cross-posting platforms."""

    X = "x"
    LINKEDIN = "linkedin"
    INSTAGRAM = "instagram"
    FACEBOOK = "facebook"


# ---------------------------------------------------------------------------
# StyleProfile
# ---------------------------------------------------------------------------


class NarrationTone(BaseModel):
    """Sentiment polarity extracted from reference channel transcripts."""

    sentiment_polarity: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity on a scale of -1.0 (negative) to +1.0 (positive).",
    )


class Pacing(BaseModel):
    """Narration pacing metrics."""

    avg_words_per_minute: int = Field(..., gt=0, description="Average words per minute.")
    avg_sentence_length_words: float = Field(..., gt=0.0, description="Average sentence length in words.")


class SegmentStructure(BaseModel):
    """Segment structure flags from reference channel analysis."""

    intro_present: bool
    hook_present: bool
    body_segment_count_avg: float = Field(..., ge=0.0, description="Average number of body segments.")
    cta_present: bool


class VisualStyle(BaseModel):
    """Visual composition patterns identified in reference channel videos."""

    composition_patterns: list[str] = Field(default_factory=list)


class ThumbnailComposition(BaseModel):
    """Thumbnail layout data extracted from reference channel uploads."""

    dominant_colors: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Up to 5 dominant hex color values (e.g., '#FF5733').",
    )
    text_overlay_position: str
    subject_framing: str
    sample_count: int = Field(..., ge=0)
    lookback_days: int = Field(..., ge=0)


class StyleProfile(BaseModel):
    """Versioned style profile produced by the Reference_Analyzer."""

    doc_id: str
    version: int = Field(..., ge=1)
    created_at: datetime
    channel_url: str
    narration_tone: NarrationTone
    pacing: Pacing
    segment_structure: SegmentStructure
    visual_style: VisualStyle
    thumbnail_composition: ThumbnailComposition
    rhetorical_patterns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# TopicEntry
# ---------------------------------------------------------------------------


class TopicEntry(BaseModel):
    """A single ranked research topic produced by the Topic_Researcher."""

    title: str = Field(..., min_length=1, max_length=200)
    composite_score: float = Field(..., ge=0.0, le=1.0)
    recency_hours: float = Field(..., ge=0.0)
    source_query_timestamp: datetime
    search_volume_signal: float
    relevance_tags_matched: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Script / NarrationAsset / VisualAsset
# ---------------------------------------------------------------------------


class Script(BaseModel):
    """Generated or revised script document."""

    video_id: str
    version: int = Field(..., ge=1)
    content: str
    word_count: int = Field(..., ge=0)
    style_profile_doc_id: str
    asset_url: Optional[str] = None
    created_at: datetime


class NarrationAsset(BaseModel):
    """ElevenLabs-generated MP3 narration asset."""

    video_id: str
    version: int = Field(..., ge=1)
    mp3_path: str
    asset_url: Optional[str] = None
    created_at: datetime


class VisualAsset(BaseModel):
    """Compiled MP4 video and thumbnail produced by the Visual_Generator."""

    video_id: str
    version: int = Field(..., ge=1)
    mp4_path: str
    thumbnail_path: str
    mp4_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    created_at: datetime


# ---------------------------------------------------------------------------
# MetadataPackage
# ---------------------------------------------------------------------------


class Chapter(BaseModel):
    """A timestamped chapter marker for the YouTube description."""

    timestamp: str = Field(..., description="Chapter timestamp in MM:SS format.")
    label: str

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_format(cls, v: str) -> str:
        if not re.match(r"^\d{2}:\d{2}$", v):
            raise ValueError("timestamp must be in MM:SS format")
        return v


class MetadataPackage(BaseModel):
    """SEO-optimized YouTube metadata produced by the Metadata_Generator."""

    video_id: str
    title: str = Field(..., max_length=60, description="YouTube title, at most 60 characters.")
    description: str = Field(..., description="Video description, 200–500 words.")
    tags: list[str] = Field(..., min_length=10, max_length=15, description="10–15 tags, each 2–5 words.")
    hashtags: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="3–5 hashtags prefixed with '#', body 2–30 chars, no spaces.",
    )
    chapters: list[Chapter] = Field(default_factory=list)
    primary_keyword: str
    asset_url: Optional[str] = None
    youtube_video_id: Optional[str] = None
    generated_at: datetime

    @field_validator("hashtags")
    @classmethod
    def validate_hashtags(cls, v: list[str]) -> list[str]:
        pattern = re.compile(r"^#[^\s]{2,30}$")
        for tag in v:
            if not pattern.match(tag):
                raise ValueError(
                    f"Invalid hashtag '{tag}': must start with '#', contain no spaces, "
                    "and have a body of 2–30 characters."
                )
        return v


# ---------------------------------------------------------------------------
# VideoRecord
# ---------------------------------------------------------------------------


_VIDEO_ID_RE = re.compile(r"^[a-zA-Z0-9\-_]{1,128}$")


class AssetLinks(BaseModel):
    """Drive URLs for each generated asset type."""

    script: Optional[str] = None
    narration: Optional[str] = None
    video: Optional[str] = None
    thumbnail: Optional[str] = None
    metadata: Optional[str] = None


class VideoRecord(BaseModel):
    """Per-video record in the Content_Calendar (Notion database)."""

    video_id: str = Field(..., description="1–128 alphanumeric characters, hyphens, or underscores.")
    batch_id: Optional[str] = None
    title: str
    topic: TopicEntry
    status: PipelineStatus
    scheduled_publish_datetime: Optional[datetime] = None
    style_profile_doc_id: str
    asset_links: AssetLinks = Field(default_factory=AssetLinks)
    youtube_video_id: Optional[str] = None
    unlisted_url: Optional[str] = None
    pipeline_run_timestamp: datetime
    pipeline_end_time: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, v: str) -> str:
        if not _VIDEO_ID_RE.match(v):
            raise ValueError(
                "video_id must be 1–128 characters consisting only of "
                "alphanumeric characters, hyphens, or underscores."
            )
        return v


# ---------------------------------------------------------------------------
# BatchRecord
# ---------------------------------------------------------------------------


class BatchRecord(BaseModel):
    """Batch-level tracking record in the Content_Calendar."""

    batch_id: str
    pipeline_run_id: str
    target_count: int = Field(..., ge=2, le=10, description="Number of videos in the batch (2–10).")
    video_ids: list[str]
    status: Literal["active", "completed", "partial_error"]
    completion_percentage: int = Field(..., ge=0, le=100)
    created_at: datetime
    style_profile_doc_id: str


# ---------------------------------------------------------------------------
# NotificationEvent
# ---------------------------------------------------------------------------


class NotificationPayload(BaseModel):
    """Content payload for a notification message."""

    title: str
    body: str
    asset_links: list[str] = Field(default_factory=list)
    action_prompt: Optional[str] = None


class NotificationEvent(BaseModel):
    """A single notification dispatched by the Notifier."""

    event_id: UUID
    video_id: str
    notification_type: Literal[
        "review_gate", "failure_alert", "batch_summary", "reminder", "upload_success"
    ]
    stage_name: Optional[str] = None
    channel: Literal["slack", "discord", "email"]
    payload: NotificationPayload
    dedup_key: str
    dispatched_at: Optional[datetime] = None
    status: Literal["pending", "sent", "failed", "suppressed"]


# ---------------------------------------------------------------------------
# LogEntry
# ---------------------------------------------------------------------------


class LogEntry(BaseModel):
    """Structured JSON log entry written to Asset_Store logs."""

    timestamp: datetime
    event_type: Literal["api_call", "stage_transition", "retry_attempt", "error", "warning"]
    stage_name: str
    video_id: Optional[str] = None
    batch_id: Optional[str] = None
    http_response_code: Optional[int] = None
    retry_attempt: Optional[int] = None
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# YouTubeVideoRef
# ---------------------------------------------------------------------------


class YouTubeVideoRef(BaseModel):
    """Reference returned after a successful YouTube upload."""

    youtube_video_id: str
    unlisted_url: str


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class SmtpConfig(BaseModel):
    """SMTP server settings for email notifications."""

    host: str
    port: int = Field(..., gt=0, le=65535)
    username: str
    password: str
    from_address: str
    to_address: str


class NotificationChannels(BaseModel):
    """Configured notification delivery channels."""

    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None
    smtp: Optional[SmtpConfig] = None


class PlatformCrossPostConfig(BaseModel):
    """Credentials and toggle for a single cross-posting platform."""

    enabled: bool
    api_key: Optional[str] = None        # X
    access_token: Optional[str] = None   # LinkedIn / Instagram
    page_access_token: Optional[str] = None  # Facebook


class CrossPostingConfig(BaseModel):
    """Cross-posting configuration for all supported platforms."""

    x: PlatformCrossPostConfig = Field(default_factory=lambda: PlatformCrossPostConfig(enabled=False))
    linkedin: PlatformCrossPostConfig = Field(default_factory=lambda: PlatformCrossPostConfig(enabled=False))
    instagram: PlatformCrossPostConfig = Field(default_factory=lambda: PlatformCrossPostConfig(enabled=False))
    facebook: PlatformCrossPostConfig = Field(default_factory=lambda: PlatformCrossPostConfig(enabled=False))


class InstagramReelsConfig(BaseModel):
    """Configuration for Instagram Reels auto-posting alongside YouTube Shorts.

    Requires a Facebook/Instagram long-lived access token and the numeric
    Instagram Business/Creator account ID.
    """

    enabled: bool = False
    access_token: Optional[str] = None
    instagram_account_id: Optional[str] = None
    share_to_feed: bool = True
    extra_hashtags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Additional niche hashtags to include in Reel captions (without # prefix).",
    )


class BatchModeConfig(BaseModel):
    """Batch processing settings."""

    enabled: bool = False
    target_count: int = Field(default=2, ge=2, le=10, description="Batch size (2–10).")


class PipelineConfig(BaseModel):
    """Root configuration model for the AI YouTube Content Pipeline."""

    reference_channel_url: str
    voice_id: str
    notification_channels: NotificationChannels = Field(default_factory=NotificationChannels)
    cross_posting: CrossPostingConfig = Field(default_factory=CrossPostingConfig)
    instagram_reels: InstagramReelsConfig = Field(default_factory=InstagramReelsConfig)
    batch_mode: BatchModeConfig = Field(default_factory=BatchModeConfig)
    topic_research_provider: Literal["perplexity", "tavily"]
    visual_video_provider: Literal["kling", "runpod", "minimax_h3"] = "minimax_h3"
    visual_prompt_mode: Literal["narration", "cinematic", "kids_rhyming"] = "narration"
    narration_provider: Literal["elevenlabs", "google_tts"] = "elevenlabs"
    style_profile_cache_days: int = Field(..., ge=0)
    script_duration_minutes: float = Field(default=1.0, gt=0, le=30)
    script_style: str = Field(
        default="cinematic_storytelling",
        description="Writing style for scripts: cinematic_storytelling, kids_rhyming, educational, documentary",
    )

    # Weekly schedule settings — used when batch_size=7
    # Videos publish Mon-Sun at this time (local time, converted to UTC)
    weekly_publish_time_hour: int = Field(default=17, ge=0, le=23)   # 5pm default
    weekly_publish_time_minute: int = Field(default=0, ge=0, le=59)
    timezone_offset_hours: float = Field(default=0.0)  # e.g. 5.5 for IST, -5 for EST

    @field_validator("voice_id")
    @classmethod
    def voice_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("voice_id must not be empty or whitespace-only.")
        return v

    @model_validator(mode="after")
    def validate_batch_target_count(self) -> "PipelineConfig":
        """Explicit cross-field check; Pydantic's ge/le on BatchModeConfig already covers this,
        but we re-validate here for clear PipelineConfig-level error messages."""
        count = self.batch_mode.target_count
        if not (2 <= count <= 10):
            raise ValueError(
                f"batch_mode.target_count must be between 2 and 10 inclusive, got {count}."
            )
        return self
