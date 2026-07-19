"""Asset_Store subsystem — Google Drive MCP wrapper.

Provides structured file storage for pipeline artifacts using Google Drive,
organized under the folder hierarchy: ai-youtube-pipeline/{video_id}/{subfolder}/{filename}.

All methods are async. Exponential back-off retry (3 attempts, 10 s base, 80 s max) is
applied to every Drive API call. On final failure, the error is logged, the optional
Notifier is alerted, and AssetStoreError is raised.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from pipeline.config import is_production_mode
from pipeline.models import SubFolder

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

DriveURL = str  # A Google Drive shareable URL string

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIDEO_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]{1,128}$")
_ROOT_FOLDER = "ai-youtube-pipeline"

# Retry policy: 3 attempts, exponential back-off: min(10 * 2^(attempt-1), 80)
_RETRY_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 10.0
_RETRY_MAX_SECONDS = 80.0

# write() must return a DriveURL within this many seconds
# Large MP4 files (100MB+) need more time than the default 30s
_WRITE_TIMEOUT_SECONDS = 300.0  # 5 minutes — covers large video uploads

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AssetStoreError(Exception):
    """Raised when the Asset_Store cannot complete an operation after all retries."""


class InvalidVideoIDError(ValueError):
    """Raised when a video_id does not match the required pattern."""


# ---------------------------------------------------------------------------
# Notifier protocol (minimal subset used by Asset_Store)
# ---------------------------------------------------------------------------


class NotifierProtocol(Protocol):
    """Minimal interface Asset_Store needs from a Notifier."""

    async def send_failure_alert(
        self,
        video_id: str,
        stage_name: str,
        error_message: str,
        publish_datetime: datetime,
    ) -> None:
        """Send a failure alert notification."""
        ...


# ---------------------------------------------------------------------------
# DriveClient protocol — abstracts the Google Drive MCP transport layer
# ---------------------------------------------------------------------------


class DriveClient(ABC):
    """Abstract interface for a Google Drive client.

    A concrete implementation wraps the Google Drive MCP server calls.
    This abstraction keeps Asset_Store fully testable without a real Drive
    connection.
    """

    @abstractmethod
    async def upload_file(
        self,
        folder_path: str,
        filename: str,
        content: bytes,
    ) -> DriveURL:
        """Upload *content* to *folder_path/filename* and return a shareable URL.

        Args:
            folder_path: Full slash-separated path of the parent folder,
                         e.g. ``ai-youtube-pipeline/vid-001/scripts``.
            filename: The bare filename, e.g. ``script_v1.md``.
            content: Raw bytes to upload.

        Returns:
            A Google Drive shareable URL for the uploaded file.

        Raises:
            Exception: Any Drive API error (will be retried by Asset_Store).
        """

    @abstractmethod
    async def download_file(
        self,
        folder_path: str,
        filename: str,
    ) -> bytes:
        """Download the file at *folder_path/filename* and return its raw bytes.

        Args:
            folder_path: Full slash-separated path of the parent folder.
            filename: The bare filename.

        Returns:
            Raw file content as bytes.

        Raises:
            Exception: Any Drive API error (will be retried by Asset_Store).
        """

    @abstractmethod
    async def get_file_url(
        self,
        folder_path: str,
        filename: str,
    ) -> DriveURL:
        """Return the shareable URL for an already-uploaded file.

        Args:
            folder_path: Full slash-separated path of the parent folder.
            filename: The bare filename.

        Returns:
            A Google Drive shareable URL.

        Raises:
            Exception: Any Drive API error (will be retried by Asset_Store).
        """


# ---------------------------------------------------------------------------
# Concrete stub — GoogleDriveMCPClient
# ---------------------------------------------------------------------------


class GoogleDriveMCPClient(DriveClient):
    """Google Drive client.

    Production: uses the Google Drive API v3 via ``google-api-python-client``.
    Development fallback: stores files under ``./pipeline_output/`` when
    credentials are absent and PIPELINE_MODE is not production.

    Credentials are read from environment variables:
        GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN

    All files are uploaded into a shared Drive folder whose ID is read from
    GOOGLE_DRIVE_FOLDER_ID (optional — falls back to searching/creating a
    root folder named ``ai-youtube-pipeline``).
    """

    def __init__(self, base_dir: str = "./pipeline_output") -> None:
        import pathlib  # noqa: PLC0415
        self._base = pathlib.Path(base_dir)
        self._service: Any = None
        self._root_folder_id: Optional[str] = None

        client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN", "")
        self._drive_folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "")

        # Store credentials for per-call service rebuilds (avoids stale SSL connections)
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token

        # Detect placeholder / missing credentials
        _creds_present = (
            client_id and client_secret and refresh_token
            and not any(v.startswith("REPLACE") for v in [client_id, client_secret, refresh_token])
        )

        if is_production_mode() and not _creds_present:
            raise ValueError(
                "Production mode requires Google Drive credentials: "
                "GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN. "
                "Set PIPELINE_MODE=development to use local filesystem."
            )

        if _creds_present:
            try:
                self._service = self._build_drive_service(
                    client_id, client_secret, refresh_token
                )
                # Eagerly validate the token has Drive scope by making a cheap
                # API call — this surfaces invalid_scope immediately at startup
                # instead of silently retrying on the first file upload.
                import google.auth.transport.requests as _gtr  # noqa: PLC0415
                req = _gtr.Request()
                self._service._http.credentials.refresh(req)  # type: ignore[union-attr]
                logger.info("GoogleDriveMCPClient: connected to real Google Drive API.")
            except Exception as exc:
                raise ValueError(
                    f"Google Drive auth failed: {exc}\n\n"
                    "Your GOOGLE_REFRESH_TOKEN does not have Drive scope.\n"
                    "It was probably created for YouTube only.\n\n"
                    "Fix: run this to get a new token with both scopes:\n\n"
                    "    python get_refresh_token.py\n\n"
                    "Then update GOOGLE_REFRESH_TOKEN in your .env file."
                ) from exc
        else:
            raise ValueError(
                "Google Drive credentials missing. "
                "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN in .env.\n\n"
                "Run:  python get_refresh_token.py"
            )

    # ------------------------------------------------------------------
    # Internal: build the Drive service
    # ------------------------------------------------------------------

    @staticmethod
    def _build_drive_service(
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> Any:
        """Build and return an authenticated Google Drive v3 service resource."""
        from google.oauth2.credentials import Credentials  # type: ignore[import-untyped]
        from googleapiclient.discovery import build  # type: ignore[import-untyped]

        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=[
                "https://www.googleapis.com/auth/drive",
            ],
        )
        return build("drive", "v3", credentials=creds, cache_discovery=False)

    # ------------------------------------------------------------------
    # Internal: folder management
    # ------------------------------------------------------------------

    async def _get_or_create_folder(self, folder_path: str) -> str:
        """Return the Drive folder ID for *folder_path*, creating subfolders as needed.

        Folder path format: ``ai-youtube-pipeline/{video_id}/{subfolder}``

        If GOOGLE_DRIVE_FOLDER_ID is set, that folder is used as the root and
        the pipeline creates ``{video_id}/{subfolder}`` inside it.
        Otherwise it finds/creates an ``ai-youtube-pipeline`` folder in My Drive.
        """
        import asyncio as _asyncio  # noqa: PLC0415

        loop = _asyncio.get_running_loop()

        def _sync_get_or_create() -> str:
            parts = [p for p in folder_path.split("/") if p]
            # parts = ["ai-youtube-pipeline", "{video_id}", "{subfolder}"]

            if self._drive_folder_id:
                # Root is the user-specified folder — skip the first part
                parent_id = self._drive_folder_id
                subparts = parts[1:]   # e.g. ["video-abc123", "scripts"]
            else:
                # No root folder set — find/create "ai-youtube-pipeline" in My Drive
                parent_id = self._ensure_root_folder()
                subparts = parts[1:]   # skip "ai-youtube-pipeline" (already the root)

            for part in subparts:
                parent_id = self._find_or_create_subfolder(part, parent_id)

            return parent_id

        return await loop.run_in_executor(None, _sync_get_or_create)

    def _ensure_root_folder(self) -> str:
        """Find or create the ``ai-youtube-pipeline`` root folder in My Drive."""
        result = self._service.files().list(
            q="name='ai-youtube-pipeline' and mimeType='application/vnd.google-apps.folder' "
              "and 'root' in parents and trashed=false",
            fields="files(id, name)",
            spaces="drive",
        ).execute()

        files = result.get("files", [])
        if files:
            return files[0]["id"]

        # Create it
        folder_meta = {
            "name": "ai-youtube-pipeline",
            "mimeType": "application/vnd.google-apps.folder",
        }
        folder = self._service.files().create(
            body=folder_meta, fields="id"
        ).execute()
        return folder["id"]

    def _find_or_create_subfolder(self, name: str, parent_id: str) -> str:
        """Find or create a subfolder *name* under *parent_id*."""
        result = self._service.files().list(
            q=f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
              f"and '{parent_id}' in parents and trashed=false",
            fields="files(id, name)",
            spaces="drive",
        ).execute()

        files = result.get("files", [])
        if files:
            return files[0]["id"]

        folder_meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [parent_id],
        }
        folder = self._service.files().create(
            body=folder_meta, fields="id"
        ).execute()
        return folder["id"]

    # ------------------------------------------------------------------
    # DriveClient interface
    # ------------------------------------------------------------------

    async def upload_file(self, folder_path: str, filename: str, content: bytes) -> DriveURL:
        import asyncio as _asyncio  # noqa: PLC0415
        import io as _io  # noqa: PLC0415
        from googleapiclient.http import MediaIoBaseUpload  # type: ignore[import-untyped]

        folder_id = await self._get_or_create_folder(folder_path)

        def _sync_upload() -> DriveURL:
            existing = self._service.files().list(
                q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
                fields="files(id)",
                spaces="drive",
            ).execute().get("files", [])

            media = MediaIoBaseUpload(
                _io.BytesIO(content),
                mimetype="application/octet-stream",
                resumable=True,
            )

            if existing:
                file_id = existing[0]["id"]
                self._service.files().update(
                    fileId=file_id,
                    media_body=media,
                    fields="id",
                ).execute()
            else:
                file_meta = {"name": filename, "parents": [folder_id]}
                result = self._service.files().create(
                    body=file_meta,
                    media_body=media,
                    fields="id",
                ).execute()
                file_id = result["id"]

            self._service.permissions().create(
                fileId=file_id,
                body={"type": "anyone", "role": "reader"},
            ).execute()

            return f"https://drive.google.com/file/d/{file_id}/view"

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_upload)

    async def download_file(self, folder_path: str, filename: str) -> bytes:
        import asyncio as _asyncio  # noqa: PLC0415
        import io as _io  # noqa: PLC0415
        from googleapiclient.http import MediaIoBaseDownload  # type: ignore[import-untyped]

        folder_id = await self._get_or_create_folder(folder_path)

        def _sync_download() -> bytes:
            # Rebuild the Drive service each call to avoid stale SSL connections.
            # httplib2 (used by google-api-python-client) keeps connections alive
            # and they go stale, causing "EOF in violation of protocol" on reuse.
            import google.auth.transport.requests as _gtr  # noqa: PLC0415
            from google.oauth2.credentials import Credentials  # noqa: PLC0415
            from googleapiclient.discovery import build  # noqa: PLC0415

            creds = Credentials(
                token=None,
                refresh_token=self._refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self._client_id,
                client_secret=self._client_secret,
                scopes=["https://www.googleapis.com/auth/drive"],
            )
            creds.refresh(_gtr.Request())
            service = build("drive", "v3", credentials=creds, cache_discovery=False)

            result = service.files().list(
                q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
                fields="files(id)",
                spaces="drive",
            ).execute()
            files = result.get("files", [])
            if not files:
                raise FileNotFoundError(
                    f"File not found in Drive: {folder_path}/{filename}"
                )
            file_id = files[0]["id"]
            request = service.files().get_media(fileId=file_id)
            buf = _io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue()

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_download)

    async def get_file_url(self, folder_path: str, filename: str) -> DriveURL:
        import asyncio as _asyncio  # noqa: PLC0415

        folder_id = await self._get_or_create_folder(folder_path)

        def _sync_get_url() -> DriveURL:
            result = self._service.files().list(
                q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
                fields="files(id)",
                spaces="drive",
            ).execute()
            files = result.get("files", [])
            if not files:
                return f"https://drive.google.com/file/d/pending-{folder_path}-{filename}/view"
            file_id = files[0]["id"]
            return f"https://drive.google.com/file/d/{file_id}/view"

        loop = _asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_get_url)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_video_id(video_id: str) -> None:
    """Raise InvalidVideoIDError if *video_id* does not match the required pattern."""
    if not _VIDEO_ID_PATTERN.match(video_id):
        raise InvalidVideoIDError(
            f"video_id must be 1–128 alphanumeric characters, hyphens, or underscores; "
            f"got: {video_id!r}"
        )


def _folder_path(video_id: str, subfolder: SubFolder) -> str:
    """Return the canonical forward-slash folder path for a given video and subfolder.

    Example: ``ai-youtube-pipeline/my-video-01/scripts``
    """
    return f"{_ROOT_FOLDER}/{video_id}/{subfolder.value}"


def _retry_delay(attempt: int) -> float:
    """Return back-off delay (seconds) for the given 1-based attempt number.

    Formula: min(10 * 2^(attempt-1), 80)
    Attempt 1 → 10 s, Attempt 2 → 20 s, Attempt 3+ → 40 s … capped at 80 s.
    """
    return min(_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), _RETRY_MAX_SECONDS)


# ---------------------------------------------------------------------------
# Asset_Store
# ---------------------------------------------------------------------------


class Asset_Store:
    """Structured file storage backed by Google Drive.

    Organises every artifact produced by the pipeline under:
    ``ai-youtube-pipeline/{video_id}/{subfolder}/{filename}``

    Args:
        drive_client: A :class:`DriveClient` implementation. Defaults to
            :class:`GoogleDriveMCPClient` (the MCP stub) when not provided.
        notifier: Optional :class:`NotifierProtocol` instance used to send
            failure alerts when all retries are exhausted.
    """

    def __init__(
        self,
        drive_client: Optional[DriveClient] = None,
        notifier: Optional[NotifierProtocol] = None,
    ) -> None:
        self._drive = drive_client if drive_client is not None else GoogleDriveMCPClient()
        self._notifier = notifier

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def write(
        self,
        video_id: str,
        subfolder: SubFolder,
        filename: str,
        content: bytes,
    ) -> DriveURL:
        """Upload *content* to Drive and return a shareable URL within 10 seconds.

        Args:
            video_id: Pipeline-assigned identifier (1–128 alphanumeric / hyphens / underscores).
            subfolder: One of the canonical :class:`~pipeline.models.SubFolder` values.
            filename: The bare filename to store (e.g. ``script_v1.md``).
            content: Raw bytes to upload.

        Returns:
            A Google Drive shareable URL (:data:`DriveURL`).

        Raises:
            InvalidVideoIDError: If *video_id* fails pattern validation.
            AssetStoreError: If all retries are exhausted or the 10-second timeout is exceeded.
        """
        _validate_video_id(video_id)
        folder = _folder_path(video_id, subfolder)

        async def _do_upload() -> DriveURL:
            return await self._drive.upload_file(folder, filename, content)

        try:
            url: DriveURL = await asyncio.wait_for(
                self._retry_with_backoff(
                    operation=_do_upload,
                    video_id=video_id,
                    operation_name=f"write({folder}/{filename})",
                ),
                timeout=_WRITE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            msg = (
                f"Asset_Store.write timed out after {_WRITE_TIMEOUT_SECONDS}s "
                f"for {folder}/{filename}"
            )
            logger.error(msg)
            await self._notify_failure(video_id, "asset_store.write", msg)
            raise AssetStoreError(msg) from exc

        return url

    async def read(
        self,
        video_id: str,
        subfolder: SubFolder,
        filename: str,
    ) -> bytes:
        """Download and return the raw bytes of a stored file.

        Args:
            video_id: Pipeline-assigned identifier.
            subfolder: Target sub-folder.
            filename: The bare filename to retrieve.

        Returns:
            Raw file content as :class:`bytes`.

        Raises:
            InvalidVideoIDError: If *video_id* fails pattern validation.
            AssetStoreError: If all retries are exhausted.
        """
        _validate_video_id(video_id)
        folder = _folder_path(video_id, subfolder)

        async def _do_download() -> bytes:
            return await self._drive.download_file(folder, filename)

        return await self._retry_with_backoff(
            operation=_do_download,
            video_id=video_id,
            operation_name=f"read({folder}/{filename})",
        )

    async def url(
        self,
        video_id: str,
        subfolder: SubFolder,
        filename: str,
    ) -> DriveURL:
        """Return the shareable Drive URL for an already-uploaded file.

        Args:
            video_id: Pipeline-assigned identifier.
            subfolder: Target sub-folder.
            filename: The bare filename.

        Returns:
            A Google Drive shareable URL (:data:`DriveURL`).

        Raises:
            InvalidVideoIDError: If *video_id* fails pattern validation.
            AssetStoreError: If all retries are exhausted.
        """
        _validate_video_id(video_id)
        folder = _folder_path(video_id, subfolder)

        async def _do_get_url() -> DriveURL:
            return await self._drive.get_file_url(folder, filename)

        return await self._retry_with_backoff(
            operation=_do_get_url,
            video_id=video_id,
            operation_name=f"url({folder}/{filename})",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _retry_with_backoff(
        self,
        operation: Any,  # Callable[[], Awaitable[T]]
        video_id: str,
        operation_name: str,
    ) -> Any:
        """Execute *operation* with exponential back-off retry.

        Attempts up to :data:`_RETRY_ATTEMPTS` times. On each failure, waits
        ``min(10 * 2^(attempt-1), 80)`` seconds before retrying. If all
        attempts fail, logs the error, optionally notifies the Notifier, and
        raises :class:`AssetStoreError`.

        Args:
            operation: An async callable (no arguments) that performs the Drive
                API call and returns a result.
            video_id: Used in log / notification messages.
            operation_name: Human-readable label for log messages.

        Returns:
            The result of a successful *operation* call.

        Raises:
            AssetStoreError: After all retry attempts are exhausted.
        """
        last_exc: Exception = Exception("No attempts made")

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                result = await operation()
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                delay = _retry_delay(attempt)

                logger.warning(
                    "Asset_Store %s failed on attempt %d/%d: %s. "
                    "Retrying in %.0f s.",
                    operation_name,
                    attempt,
                    _RETRY_ATTEMPTS,
                    exc,
                    delay,
                )

                if attempt < _RETRY_ATTEMPTS:
                    await asyncio.sleep(delay)

        # All retries exhausted
        error_message = (
            f"Asset_Store {operation_name} failed after {_RETRY_ATTEMPTS} attempts: {last_exc}"
        )
        logger.error(error_message)
        await self._notify_failure(video_id, "asset_store", error_message)
        raise AssetStoreError(error_message) from last_exc

    async def _notify_failure(
        self,
        video_id: str,
        stage_name: str,
        error_message: str,
    ) -> None:
        """Send a failure alert via the Notifier (if one is configured).

        Silently ignores any error raised by the Notifier itself to avoid
        masking the original AssetStoreError.
        """
        if self._notifier is None:
            return
        try:
            await self._notifier.send_failure_alert(
                video_id=video_id,
                stage_name=stage_name,
                error_message=error_message,
                publish_datetime=datetime.now(tz=timezone.utc),
            )
        except Exception as notify_exc:  # noqa: BLE001
            logger.error(
                "Asset_Store: Notifier.send_failure_alert itself failed: %s",
                notify_exc,
            )


# ---------------------------------------------------------------------------
# Public re-exports
# ---------------------------------------------------------------------------

__all__ = [
    "Asset_Store",
    "AssetStoreError",
    "DriveClient",
    "DriveURL",
    "GoogleDriveMCPClient",
    "InvalidVideoIDError",
    "NotifierProtocol",
]
