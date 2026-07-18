# AI YouTube Content Pipeline

[![ci](https://github.com/yourusername/ai-youtube-pipeline/workflows/ci/badge.svg)](https://github.com/yourusername/ai-youtube-pipeline/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

An end-to-end automated video production system for creating AI-powered YouTube content.

## Project Structure

```
ai-youtube-pipeline/
├── src/
│   └── pipeline/              # Main application package
│       ├── __init__.py
│       ├── __main__.py        # Module entry point
│       ├── cli.py             # Command-line interface
│       ├── factory.py         # Dependency injection
│       ├── models.py          # Data models
│       ├── asset_store/       # Asset management
│       ├── content_calendar/  # Content scheduling
│       ├── cross_poster/      # Multi-platform posting
│       ├── metadata_generator/ # Video metadata
│       ├── narration_generator/ # Voice generation
│       ├── notifier/          # Notifications
│       ├── orchestrator/      # Pipeline orchestration
│       ├── publisher/         # YouTube publishing
│       ├── reference_analyzer/ # Channel analysis
│       ├── script_writer/     # Script generation
│       ├── topic_researcher/  # Topic research
│       └── visual_generator/  # Video generation
├── scripts/
│   ├── get_youtube_token.py  # OAuth token generation
│   └── setup_notion_db.py    # Notion database setup
├── config/
│   ├── config.example.json   # Example configuration
│   └── .env.example          # Example environment variables
├── data/
│   └── output/               # Pipeline output directory
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   ├── property/             # Property-based tests
│   └── smoke/                # Smoke tests
├── docs/                     # Documentation
└── .github/                  # GitHub workflows
```

## Features

- **Automated Content Research**: AI-powered topic research using Perplexity or other providers
- **Script Writing**: Generate engaging video scripts based on reference channel analysis
- **Narration**: Text-to-speech using ElevenLabs
- **Visual Generation**: Automated video creation with visuals
- **Metadata Generation**: SEO-optimized titles, descriptions, and tags
- **YouTube Publishing**: Direct upload to YouTube
- **Cross-Platform Posting**: Share to X (Twitter), LinkedIn, Instagram, and Facebook
- **Content Calendar**: Integration with Notion for planning
- **Batch Processing**: Generate multiple videos in one run
- **Notifications**: Slack, Discord, or email alerts

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [Hatch](https://hatch.pypa.io/latest/install/) for environment management

### Installation

1. Clone the repository:
```sh
git clone <your-repo-url>
cd ai-youtube-pipeline
```

2. Install dependencies:
```sh
hatch env create
```

3. Configure the pipeline:
```sh
# Copy example files
cp config/config.example.json config.json
cp config/.env.example .env

# Edit config.json and .env with your API keys and settings
```

### Configuration

#### Required API Keys (in `.env`):
- `ANTHROPIC_API_KEY` - For AI script generation
- `ELEVENLABS_API_KEY` - For voice narration
- `YOUTUBE_CLIENT_ID` & `YOUTUBE_CLIENT_SECRET` - For YouTube uploads
- `PERPLEXITY_API_KEY` - For topic research (if using Perplexity)

#### Configuration File (`config.json`):
```json
{
  "reference_channel_url": "https://www.youtube.com/@yourchannel",
  "voice_id": "your-elevenlabs-voice-id",
  "topic_research_provider": "perplexity",
  "notification_channels": {
    "slack_webhook_url": "https://hooks.slack.com/...",
    "discord_webhook_url": null,
    "smtp": null
  },
  "batch_mode": {
    "enabled": false,
    "target_count": 2
  }
}
```

### Usage

#### Single Video Generation
```sh
# Run the pipeline
hatch run python -m pipeline --config config.json
```

#### Batch Processing
```sh
# Generate 3 videos
hatch run python -m pipeline --config config.json --batch-size 3
```

#### Resume Failed Run
```sh
# Resume a previously failed pipeline run
hatch run python -m pipeline --config config.json --resume-run-id <UUID>
```

### Utility Scripts

#### Get YouTube OAuth Token
```sh
hatch run python scripts/get_youtube_token.py
```

#### Setup Notion Database
```sh
hatch run python scripts/setup_notion_db.py
```

## Development

### Run Tests
```sh
# Run all tests with coverage
hatch run coverage run

# Run specific test types
hatch run pytest tests/unit/
hatch run pytest tests/integration/
hatch run pytest tests/property/
```

### Code Quality
```sh
# Run all checks (format, lint, type check)
hatch run check

# Auto-format code
hatch run format
```

### Project Commands
```sh
# Format code
hatch run ruff format

# Lint code
hatch run ruff check

# Type check
hatch run mypy

# Run tests with coverage
hatch run coverage run
```

## Architecture

The pipeline follows a modular architecture with clear separation of concerns:

1. **CLI Layer** (`cli.py`) - Command-line interface and argument parsing
2. **Factory Pattern** (`factory.py`) - Dependency injection and component creation
3. **Orchestrator** (`orchestrator/`) - Pipeline workflow coordination
4. **Components** - Independent modules for each pipeline stage
5. **Models** (`models.py`) - Shared data structures using Pydantic

## Output Structure

Generated content is stored in `data/output/`:
```
data/output/
└── video-{id}/
    ├── script.txt
    ├── narration.mp3
    ├── video.mp4
    ├── thumbnail.png
    └── metadata.json
```

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

For issues and questions, please use the GitHub issue tracker.
