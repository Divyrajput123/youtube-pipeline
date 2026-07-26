"""Dependency-injection factory for the AI YouTube Content Pipeline."""

from __future__ import annotations

import os

from pipeline.asset_store import Asset_Store, GoogleDriveMCPClient
from pipeline.config import is_production_mode
from pipeline.content_calendar import Content_Calendar, NotionMCPClient
from pipeline.cross_poster import Cross_Poster
from pipeline.metadata_generator import Metadata_Generator
from pipeline.models import NotificationChannels, PipelineConfig, SmtpConfig
from pipeline.narration_generator import ElevenLabsMCPClient, Narration_Generator
from pipeline.notifier import Notifier, NotifierConfig
from pipeline.orchestrator import Orchestrator
from pipeline.publisher import Publisher, YouTubeDataAPIClient
from pipeline.reference_analyzer import BrowserMCPClient, Reference_Analyzer
from pipeline.script_writer import Script_Writer, build_claude_client
from pipeline.topic_researcher import (
    PerplexityMCPClient,
    TavilyMCPClient,
    Topic_Researcher,
)
from pipeline.visual_generator import ViewmaxMCPClient, Visual_Generator


def build_orchestrator(config: PipelineConfig) -> Orchestrator:
    """Wire all subsystems and return a fully configured Orchestrator."""

    # Ensure .env is loaded before reading credentials
    from dotenv import load_dotenv  # noqa: PLC0415
    import pathlib  # noqa: PLC0415
    env_path = pathlib.Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)

    # ------------------------------------------------------------------
    # 1. Asset_Store
    #    Production: real Google Drive API
    #    Development: local filesystem under ./pipeline_output/
    # ------------------------------------------------------------------
    drive_client = GoogleDriveMCPClient()
    asset_store = Asset_Store(drive_client=drive_client)

    # ------------------------------------------------------------------
    # 2. Content_Calendar
    #    Production: real Notion API (requires NOTION_AUTH_TOKEN + NOTION_DATABASE_ID)
    #    Development: local JSON file (./pipeline_output/calendar.json)
    # ------------------------------------------------------------------
    notion_token = os.environ.get("NOTION_AUTH_TOKEN", "")
    notion_db_id = os.environ.get("NOTION_DATABASE_ID", "")

    if is_production_mode():
        if not notion_token or notion_token.startswith("secret_REPLACE"):
            raise ValueError(
                "Production mode requires a valid NOTION_AUTH_TOKEN. "
                "Set PIPELINE_MODE=development to use local calendar."
            )
        if not notion_db_id or notion_db_id == "REPLACE_ME":
            raise ValueError(
                "Production mode requires a valid NOTION_DATABASE_ID. "
                "Set PIPELINE_MODE=development to use local calendar."
            )
        notion_client = NotionMCPClient(
            auth_token=notion_token,
            database_id=notion_db_id,
        )
        content_calendar: Content_Calendar = Content_Calendar(
            notion_client=notion_client,
            database_id=notion_db_id,
        )
    elif notion_token and not notion_token.startswith("secret_REPLACE") \
            and notion_db_id and notion_db_id != "REPLACE_ME":
        # Development mode but credentials are present — use real Notion.
        notion_client = NotionMCPClient(
            auth_token=notion_token,
            database_id=notion_db_id,
        )
        import logging as _logging  # noqa: PLC0415
        _log = _logging.getLogger(__name__)
        _log.info(
            "Notion credentials present — using real Notion calendar. "
            "If you see 'database not shared' errors, go to Notion → "
            "database ⋯ menu → Connections and add your integration."
        )
        content_calendar = Content_Calendar(
            notion_client=notion_client,
            database_id=notion_db_id,
        )
    else:
        # Development mode, no credentials — fall back to local JSON.
        import logging as _logging  # noqa: PLC0415
        _logging.getLogger(__name__).warning(
            "Notion credentials missing — using local JSON calendar "
            "(./pipeline_output/calendar.json). "
            "Set NOTION_AUTH_TOKEN and NOTION_DATABASE_ID to use real Notion."
        )
        from pipeline.content_calendar.local import LocalContentCalendar  # noqa: PLC0415
        content_calendar = LocalContentCalendar("./pipeline_output/calendar.json")  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # 3. Notifier
    # ------------------------------------------------------------------
    channels: NotificationChannels = config.notification_channels
    smtp_cfg: SmtpConfig | None = channels.smtp

    # Resolve ${SMTP_PASSWORD} placeholder from environment if present
    if smtp_cfg is not None and smtp_cfg.password.startswith("${") and smtp_cfg.password.endswith("}"):
        env_var = smtp_cfg.password[2:-1]
        resolved = os.environ.get(env_var, "")
        if resolved:
            smtp_cfg = smtp_cfg.model_copy(update={"password": resolved})
        else:
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger(__name__).warning(
                "SMTP password placeholder '%s' not resolved — %s not set in environment. "
                "Email notifications will be disabled.",
                smtp_cfg.password, env_var,
            )
            smtp_cfg = None  # disable email rather than send with wrong password

    notifier_config = NotifierConfig(
        slack_webhook_url=channels.slack_webhook_url,
        discord_webhook_url=channels.discord_webhook_url,
        smtp=smtp_cfg,
    )
    notifier = Notifier(config=notifier_config)

    # ------------------------------------------------------------------
    # 4. Reference_Analyzer
    # ------------------------------------------------------------------
    browser_client = BrowserMCPClient()
    reference_analyzer = Reference_Analyzer(
        browser_client=browser_client,
        asset_store=asset_store,
        content_calendar=content_calendar,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 5. Topic_Researcher
    # ------------------------------------------------------------------
    provider = config.topic_research_provider
    if provider == "tavily":
        search_client = TavilyMCPClient()
    else:
        search_client = PerplexityMCPClient()

    topic_researcher = Topic_Researcher(
        search_client=search_client,
        asset_store=asset_store,
        notifier=notifier,
        provider=provider,
    )

    # ------------------------------------------------------------------
    # 6. Script_Writer
    # ------------------------------------------------------------------
    claude_client = build_claude_client()
    script_writer = Script_Writer(
        claude_client=claude_client,
        asset_store=asset_store,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 7. Narration_Generator
    # ------------------------------------------------------------------
    elevenlabs_client = ElevenLabsMCPClient()
    narration_generator = Narration_Generator(
        elevenlabs_client=elevenlabs_client,
        asset_store=asset_store,
        content_calendar=content_calendar,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 8. Visual_Generator
    # ------------------------------------------------------------------
    viewmax_client = ViewmaxMCPClient(provider=config.visual_video_provider)
    visual_generator = Visual_Generator(
        viewmax_client=viewmax_client,
        asset_store=asset_store,
        content_calendar=content_calendar,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 9. Metadata_Generator
    # ------------------------------------------------------------------
    metadata_generator = Metadata_Generator(
        claude_client=claude_client,
        asset_store=asset_store,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 10. Publisher
    # ------------------------------------------------------------------
    youtube_client = YouTubeDataAPIClient()
    publisher = Publisher(
        youtube_client=youtube_client,
        content_calendar=content_calendar,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 11. Cross_Poster
    # ------------------------------------------------------------------
    cross_poster = Cross_Poster(
        config=config.cross_posting,
        notifier=notifier,
    )

    # ------------------------------------------------------------------
    # 12. Orchestrator
    # ------------------------------------------------------------------
    return Orchestrator(
        config=config,
        reference_analyzer=reference_analyzer,
        topic_researcher=topic_researcher,
        script_writer=script_writer,
        narration_generator=narration_generator,
        visual_generator=visual_generator,
        metadata_generator=metadata_generator,
        publisher=publisher,
        cross_poster=cross_poster,
        asset_store=asset_store,
        content_calendar=content_calendar,
        notifier=notifier,
    )


__all__ = ["build_orchestrator"]
