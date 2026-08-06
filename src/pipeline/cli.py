"""Command-line entry point for the AI YouTube Content Pipeline.

Usage examples::

    # Single-video run
    python -m pipeline.cli --config config.json

    # Batch run (3 videos)
    python -m pipeline.cli --config config.json --batch-size 3

    # Resume a previously-failed run
    python -m pipeline.cli --config config.json --resume-run-id <UUID>

Exit codes:
    0 — pipeline started (or resumed) successfully.
    1 — configuration error or pipeline runtime error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Load .env automatically if present — must happen before any other imports
# so that os.environ is populated before factory.py reads credentials.
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]  # noqa: PLC0415
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"Loaded environment from {env_path}")
        else:
            # Also try current working directory
            cwd_env = Path(".env")
            if cwd_env.exists():
                load_dotenv(dotenv_path=cwd_env, override=True)
                print(f"Loaded environment from {cwd_env.resolve()}")
    except ImportError:
        pass  # python-dotenv not installed; rely on shell env vars

_load_dotenv()

# Configure logging to stdout so all pipeline logs appear in GitHub Actions / terminal
import logging as _logging
_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="AI YouTube Content Pipeline — start, batch, or resume a pipeline run.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        required=True,
        help="Path to the JSON pipeline configuration file.",
    )
    parser.add_argument(
        "--env-file",
        metavar="PATH",
        default=None,
        help="Path to a custom .env file (for multi-channel support). Overrides the default .env.",
    )
    parser.add_argument(
        "--batch-size",
        metavar="N",
        type=int,
        default=None,
        help="Run in batch mode, creating N videos (must be between 2 and 10).",
    )
    parser.add_argument(
        "--resume-run-id",
        metavar="RUN_ID",
        default=None,
        help="Resume a previously-failed pipeline run identified by this UUID.",
    )
    parser.add_argument(
        "--resume-all",
        action="store_true",
        default=False,
        help="Query Notion for all stuck/in-progress videos and resume them.",
    )
    parser.add_argument(
        "--resume-and-generate",
        action="store_true",
        default=False,
        help="Resume all stuck videos first, then generate a new one.",
    )
    return parser


def _load_config(config_path: str):  # type: ignore[return]
    """Load and validate a :class:`~pipeline.models.PipelineConfig` from *config_path*.

    Args:
        config_path: Path to a JSON file containing the pipeline configuration.

    Returns:
        A validated :class:`~pipeline.models.PipelineConfig` instance.

    Raises:
        SystemExit: With exit code 1 when the file cannot be read or the
            configuration fails validation.
    """
    # Defer heavy imports so that --help is fast.
    from pipeline.models import PipelineConfig  # noqa: PLC0415
    from pydantic import ValidationError  # noqa: PLC0415

    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"Error: config file is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        return PipelineConfig(**raw)
    except ValidationError as exc:
        print(f"Error: invalid pipeline configuration:\n{exc}", file=sys.stderr)
        sys.exit(1)


def _validate_batch_size(n: int) -> None:
    """Validate that *n* is within the allowed [2, 10] range.

    Raises:
        SystemExit: With exit code 1 when *n* is out of range.
    """
    if not (2 <= n <= 10):
        print(
            f"Error: --batch-size must be between 2 and 10 (got {n}).",
            file=sys.stderr,
        )
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    """Async core: start review webhook server, build orchestrator, run pipeline."""
    # Load custom .env file if specified (for multi-channel support)
    if args.env_file:
        from dotenv import load_dotenv  # noqa: PLC0415
        env_path = Path(args.env_file)
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"Loaded custom environment from {env_path}")
        else:
            print(f"Warning: --env-file {args.env_file} not found", file=sys.stderr)

    from pipeline.factory import build_orchestrator  # noqa: PLC0415
    from pipeline.review_server import start_review_server  # noqa: PLC0415
    from pipeline.tunnel import start_tunnel, stop_tunnel  # noqa: PLC0415

    config = _load_config(args.config)
    orchestrator = build_orchestrator(config)

    # Start the review webhook server as a background task.
    # Give it a moment to bind to the port before the pipeline starts.
    review_server_task = asyncio.create_task(
        start_review_server(),
        name="review_webhook_server",
    )
    await asyncio.sleep(0.5)  # wait for server to bind before pipeline runs

    # Start public tunnel so review email links work from any network/device.
    # This updates PIPELINE_PUBLIC_URL in-process before any tokens are issued.
    start_tunnel()

    try:
        if args.resume_run_id:
            run_id: str = args.resume_run_id
            video_id: str = f"video-{run_id[:8]}"
            await orchestrator.resume_pipeline(run_id=run_id, video_id=video_id)
            print(f"Resumed run: {run_id}")
        elif args.resume_all:
            # Query Notion for all videos not yet published and resume them
            from pipeline.models import PipelineStatus  # noqa: PLC0415

            resumable_statuses = [
                PipelineStatus.SCRIPTING,
                PipelineStatus.AWAITING_SCRIPT_REVIEW,
                PipelineStatus.SCRIPT_APPROVED,
                PipelineStatus.NARRATION_READY,
                PipelineStatus.GENERATING_VISUALS,
                PipelineStatus.VISUALS_READY,
                PipelineStatus.GENERATING_METADATA,
                PipelineStatus.AWAITING_FINAL_REVIEW,
                PipelineStatus.APPROVED_FOR_UPLOAD,
                PipelineStatus.AUTO_APPROVED_FOR_UPLOAD,
                PipelineStatus.UPLOADING,
            ]

            print("Querying Notion for stuck/in-progress videos...")
            resumed_count = 0
            for status in resumable_statuses:
                try:
                    videos = await orchestrator._content_calendar.list_videos_by_status(status)
                    for video in videos:
                        video_id = video.get("video_id") or video.get("id", "")
                        if not video_id:
                            continue
                        # Derive run_id from video_id (video_id = "video-XXXXXXXX")
                        run_id = video_id.replace("video-", "") if video_id.startswith("video-") else video_id
                        print(f"  Resuming {video_id} (status: {status.value})...")
                        try:
                            await orchestrator.resume_pipeline(run_id=run_id, video_id=video_id)
                            print(f"  ✓ {video_id} resumed successfully")
                            resumed_count += 1
                        except Exception as exc:
                            print(f"  ✗ {video_id} failed: {exc}")
                except Exception as exc:
                    print(f"  Could not query status '{status.value}': {exc}")

            print(f"Resume complete: {resumed_count} video(s) processed")
        elif args.resume_and_generate:
            # Resume all stuck videos first, then generate a new one
            from pipeline.models import PipelineStatus  # noqa: PLC0415
            resumable_statuses = [
                PipelineStatus.SCRIPTING, PipelineStatus.AWAITING_SCRIPT_REVIEW,
                PipelineStatus.SCRIPT_APPROVED, PipelineStatus.NARRATION_READY,
                PipelineStatus.GENERATING_VISUALS, PipelineStatus.VISUALS_READY,
                PipelineStatus.GENERATING_METADATA, PipelineStatus.AWAITING_FINAL_REVIEW,
                PipelineStatus.APPROVED_FOR_UPLOAD, PipelineStatus.AUTO_APPROVED_FOR_UPLOAD,
            ]
            print("Step 1/2: Resuming stuck videos...")
            resumed_count = 0
            for status in resumable_statuses:
                try:
                    videos = await orchestrator._content_calendar.list_videos_by_status(status)
                    for video in videos:
                        video_id = video.get("video_id") or video.get("id", "")
                        if not video_id:
                            continue
                        run_id = video_id.replace("video-", "") if video_id.startswith("video-") else video_id
                        print(f"  Resuming {video_id} (status: {status.value})...")
                        try:
                            await orchestrator.resume_pipeline(run_id=run_id, video_id=video_id)
                            print(f"  ✓ {video_id} resumed")
                            resumed_count += 1
                        except Exception as exc:
                            print(f"  ✗ {video_id} failed: {exc}")
                except Exception as exc:
                    print(f"  Could not query status '{status.value}': {exc}")
            print(f"  Resumed {resumed_count} video(s)")
            print("Step 2/2: Generating new video...")
            run_id = await orchestrator.start_pipeline()
            print(f"  New pipeline started: {run_id}")
        elif args.batch_size is not None:
            _validate_batch_size(args.batch_size)
            batch_id = await orchestrator.start_batch(n=args.batch_size)
            print(f"Batch started: {batch_id}")
        else:
            run_id = await orchestrator.start_pipeline()
            print(f"Pipeline started: {run_id}")
    finally:
        stop_tunnel()
        review_server_task.cancel()
        try:
            await review_server_task
        except asyncio.CancelledError:
            pass


def main() -> None:
    """CLI entry point — parse arguments and run the pipeline."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
