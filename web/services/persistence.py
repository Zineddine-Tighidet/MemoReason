"""Durable state sync for Cloud Run deployments.

This module keeps the annotation state outside the container filesystem:
- SQLite DB snapshot
- human annotation working files

When configured with a GCS bucket, state is restored on startup and synced
after write operations.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from web.services.perf_logging import PerfTimer
from web.settings import DB_PATH, WORK_DIR, remote_state_enabled

logger = logging.getLogger(__name__)

_BUCKET_ENV = "ANNOTATION_STATE_GCS_BUCKET"
_PREFIX_ENV = "ANNOTATION_STATE_GCS_PREFIX"
_DEFAULT_PREFIX = "annotation_state"

_lock = threading.Lock()
_client: Any = None
_db_generation: int | None = None
_last_db_restore_monotonic: float | None = None
_last_worktree_restore_monotonic: float | None = None
_work_file_restore_monotonic: dict[str, float] = {}


def _sqlite_integrity_is_ok(path: Path) -> bool:
    """Return True when a SQLite file passes integrity_check."""
    conn = None
    try:
        conn = sqlite3.connect(str(path))
        row = conn.execute("PRAGMA integrity_check").fetchone()
        return bool(row) and str(row[0]).lower() == "ok"
    except sqlite3.DatabaseError:
        return False
    finally:
        if conn is not None:
            conn.close()


def _read_local_sessions_snapshot(path: Path) -> list[tuple[str, int, str, str]]:
    """Capture locally-issued sessions so DB refreshes do not log users out mid-request."""
    if not path.exists():
        return []

    conn = None
    try:
        conn = sqlite3.connect(str(path))
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions' LIMIT 1"
        ).fetchone()
        if not table_exists:
            return []

        rows = conn.execute(
            "SELECT token, user_id, created_at, expires_at FROM sessions"
        ).fetchall()
        return [
            (str(token), int(user_id), str(created_at), str(expires_at))
            for token, user_id, created_at, expires_at in rows
            if token
        ]
    except sqlite3.DatabaseError:
        return []
    finally:
        if conn is not None:
            conn.close()


def _merge_local_sessions_snapshot(path: Path, sessions: list[tuple[str, int, str, str]]) -> None:
    """Re-apply locally-issued sessions after replacing the SQLite snapshot."""
    if not sessions:
        return

    conn = None
    try:
        conn = sqlite3.connect(str(path))
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions' LIMIT 1"
        ).fetchone()
        if not table_exists:
            return

        conn.executemany(
            """
            INSERT OR REPLACE INTO sessions (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            sessions,
        )
        conn.commit()
    finally:
        if conn is not None:
            conn.close()


def _bucket_name() -> str:
    return os.getenv(_BUCKET_ENV, "").strip()


def _prefix() -> str:
    raw = os.getenv(_PREFIX_ENV, _DEFAULT_PREFIX).strip().strip("/")
    return raw or _DEFAULT_PREFIX


def is_enabled() -> bool:
    return remote_state_enabled()


def _storage_client():
    global _client
    if _client is None:
        from google.cloud import storage
        _client = storage.Client()
    return _client


def _bucket():
    return _storage_client().bucket(_bucket_name())


def _is_gcs_not_found_error(exc: Exception) -> bool:
    """Best-effort detection for missing GCS object errors without hard dependency at import time."""
    try:
        from google.api_core import exceptions as gcs_exceptions  # type: ignore

        if isinstance(exc, gcs_exceptions.NotFound):
            return True
    except Exception:
        pass

    message = str(exc or "").lower()
    return "no such object" in message or ("404" in message and "storage" in message)


def _is_gcs_precondition_error(exc: Exception) -> bool:
    """Detect conditional upload failures (HTTP 412) from GCS."""
    try:
        from google.api_core import exceptions as gcs_exceptions  # type: ignore

        if isinstance(exc, gcs_exceptions.PreconditionFailed):
            return True
    except Exception:
        pass

    message = str(exc or "").lower()
    return "412" in message and ("condition" in message or "precondition" in message)


def _db_blob_name() -> str:
    return f"{_prefix()}/db/annotation.db"


def _work_blob_name(local_path: Path) -> str:
    rel = local_path.relative_to(WORK_DIR).as_posix()
    return f"{_prefix()}/work/{rel}"


def restore_state_from_gcs() -> None:
    """Restore DB + annotation files from GCS if configured."""
    if not is_enabled():
        return
    global _last_db_restore_monotonic, _last_worktree_restore_monotonic
    with _lock:
        _restore_db_from_gcs()
        _restore_worktree_from_gcs()
        now = time.monotonic()
        _last_db_restore_monotonic = now
        _last_worktree_restore_monotonic = now


def restore_db_from_gcs(
    *,
    max_age_seconds: float | None = None,
    force_download: bool = False,
) -> None:
    """Restore only the SQLite snapshot from GCS.

    Dashboard requests only need database state; reloading the full worktree on
    every API call is much more expensive and can add multi-second latency.
    """
    if not is_enabled():
        return

    global _last_db_restore_monotonic
    timer = PerfTimer("restore_db_from_gcs", force_download=force_download, max_age=max_age_seconds)
    with _lock:
        try:
            now = time.monotonic()
            cache_age = (
                None if _last_db_restore_monotonic is None else now - _last_db_restore_monotonic
            )
            if cache_age is not None:
                timer.add_field("cache_age_s", f"{cache_age:.1f}")
            if (
                max_age_seconds is not None
                and _last_db_restore_monotonic is not None
                and (now - _last_db_restore_monotonic) < max_age_seconds
            ):
                timer.add_field("result", "memory_cache_skip")
                return
            with timer.step("restore_db_impl"):
                restored = _restore_db_from_gcs(force_download=force_download)
            timer.add_field("result", "downloaded" if restored else "generation_skip")
            _last_db_restore_monotonic = time.monotonic()
        finally:
            timer.emit()


def restore_worktree_from_gcs() -> None:
    """Refresh only the annotation workspace from GCS."""
    if not is_enabled():
        return
    global _last_worktree_restore_monotonic
    timer = PerfTimer("restore_worktree_from_gcs")
    with _lock:
        try:
            with timer.step("restore_worktree_impl"):
                restored = _restore_worktree_from_gcs()
            timer.add_field("files", restored)
            _last_worktree_restore_monotonic = time.monotonic()
        finally:
            timer.emit()


def restore_work_file_from_gcs(local_path: Path, *, max_age_seconds: float | None = None) -> bool:
    """Refresh one annotation workspace file from GCS when it exists remotely."""
    if not is_enabled():
        return False

    try:
        blob_name = _work_blob_name(local_path)
    except Exception:
        return False

    cache_key = str(local_path.resolve())
    with _lock:
        now = time.monotonic()
        if max_age_seconds is not None:
            last_restore = _work_file_restore_monotonic.get(cache_key)
            if last_restore is not None and (now - last_restore) < max_age_seconds:
                return local_path.exists()
        blob = _bucket().blob(blob_name)
        if not blob.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob.download_to_filename(str(local_path))
            _work_file_restore_monotonic[cache_key] = time.monotonic()
            return True
        except Exception as exc:
            if _is_gcs_not_found_error(exc):
                logger.warning("Skipped missing remote work file: gs://%s/%s", _bucket_name(), blob_name)
                return False
            raise


def sync_db_to_gcs() -> None:
    """Upload a consistent SQLite snapshot to GCS."""
    if not is_enabled():
        return
    global _db_generation, _last_db_restore_monotonic
    timer = PerfTimer("sync_db_to_gcs")
    with _lock:
        try:
            if not DB_PATH.exists():
                timer.add_field("result", "missing_local_db")
                return

            blob = _bucket().blob(_db_blob_name())
            if _db_generation is None:
                with timer.step("load_remote_generation"):
                    if blob.exists():
                        blob.reload()
                        _db_generation = int(blob.generation or 0)
                    else:
                        _db_generation = 0

            tmp_dir = Path(tempfile.mkdtemp(prefix="annotation-db-sync-"))
            tmp_db = tmp_dir / "annotation.snapshot.db"
            src = None
            dst = None
            try:
                with timer.step("sqlite_backup"):
                    src = sqlite3.connect(str(DB_PATH), check_same_thread=False)
                    dst = sqlite3.connect(str(tmp_db))
                    src.backup(dst)
                    dst.commit()
                with timer.step("integrity_check"):
                    if not _sqlite_integrity_is_ok(tmp_db):
                        raise RuntimeError("Refusing to upload malformed SQLite snapshot to GCS.")
                try:
                    with timer.step("upload"):
                        blob.upload_from_filename(str(tmp_db), if_generation_match=int(_db_generation))
                except Exception as exc:
                    if _is_gcs_precondition_error(exc):
                        raise RuntimeError(
                            "Remote DB changed since the last pull. Pull latest state (restore_state_from_gcs) "
                            "before pushing to avoid overwriting newer annotations."
                        ) from exc
                    raise
                with timer.step("reload_generation"):
                    blob.reload()
                    _db_generation = int(blob.generation or 0)
                _last_db_restore_monotonic = time.monotonic()
                timer.add_field("result", "uploaded")
            finally:
                if dst is not None:
                    dst.close()
                if src is not None:
                    src.close()
                shutil.rmtree(tmp_dir, ignore_errors=True)
        finally:
            timer.emit()


def sync_work_file_to_gcs(local_path: Path) -> None:
    """Upload one annotation YAML file to GCS."""
    if not is_enabled():
        return
    if not local_path.exists() or not local_path.is_file():
        return
    with _lock:
        _bucket().blob(_work_blob_name(local_path)).upload_from_filename(str(local_path))


def delete_work_prefix_from_gcs(relative_prefix: str | Path) -> None:
    """Delete a subtree under the synced annotation workspace in GCS."""
    if not is_enabled():
        return

    normalized = str(relative_prefix or "").strip().strip("/")
    if not normalized:
        return

    blob_prefix = f"{_prefix()}/work/{normalized}/"
    with _lock:
        for blob in _bucket().list_blobs(prefix=blob_prefix):
            blob.delete()


def _restore_db_from_gcs(*, force_download: bool = False) -> bool:
    global _db_generation
    blob = _bucket().blob(_db_blob_name())
    local_sessions = _read_local_sessions_snapshot(DB_PATH)
    try:
        blob.reload()
    except Exception as exc:
        if _is_gcs_not_found_error(exc):
            logger.info("No remote DB snapshot found at gs://%s/%s", _bucket_name(), _db_blob_name())
            _db_generation = 0
            return False
        raise

    remote_generation = int(blob.generation or 0)
    if (
        not force_download
        and DB_PATH.exists()
        and _db_generation is not None
        and remote_generation == _db_generation
    ):
        logger.debug(
            "Skipped DB restore; remote generation %s already matches local snapshot.",
            remote_generation,
        )
        return False

    if force_download:
        logger.info(
            "Forcing DB restore from gs://%s/%s at generation %s",
            _bucket_name(),
            _db_blob_name(),
            remote_generation,
        )
    else:
        logger.info(
            "Restoring DB from gs://%s/%s at generation %s",
            _bucket_name(),
            _db_blob_name(),
            remote_generation,
        )

    if remote_generation == 0 and not DB_PATH.exists():
        logger.info("No remote DB snapshot found at gs://%s/%s", _bucket_name(), _db_blob_name())
        _db_generation = 0
        return False

    # SQLite sidecar files from a previous local WAL session can make the
    # freshly downloaded primary DB appear malformed if they no longer match.
    for sidecar in (DB_PATH.with_name(f"{DB_PATH.name}-wal"), DB_PATH.with_name(f"{DB_PATH.name}-shm")):
        try:
            if sidecar.exists():
                sidecar.unlink()
        except FileNotFoundError:
            pass

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = DB_PATH.with_suffix(".restore.tmp")
    blob.download_to_filename(str(tmp_path))
    if not _sqlite_integrity_is_ok(tmp_path):
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise RuntimeError(
            f"Refusing to replace local DB with malformed remote snapshot from "
            f"gs://{_bucket_name()}/{_db_blob_name()}."
        )
    tmp_path.replace(DB_PATH)
    _merge_local_sessions_snapshot(DB_PATH, local_sessions)

    # Drop stale sidecar files from previous local runs.
    wal = DB_PATH.with_name(DB_PATH.name + "-wal")
    shm = DB_PATH.with_name(DB_PATH.name + "-shm")
    if wal.exists():
        wal.unlink()
    if shm.exists():
        shm.unlink()

    _db_generation = remote_generation
    logger.info("Restored DB from gs://%s/%s", _bucket_name(), _db_blob_name())
    return True


def _restore_worktree_from_gcs() -> int:
    prefix = f"{_prefix()}/work/"
    blobs = list(_bucket().list_blobs(prefix=prefix))
    if not blobs:
        logger.info("No remote annotation worktree found at gs://%s/%s", _bucket_name(), prefix)
        return 0

    restored = 0
    skipped_missing = 0
    for blob in blobs:
        name = blob.name
        if name.endswith("/"):
            continue
        rel = name[len(prefix):]
        if not rel:
            continue
        local_path = WORK_DIR / rel
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob.download_to_filename(str(local_path))
            restored += 1
        except Exception as exc:
            if _is_gcs_not_found_error(exc):
                skipped_missing += 1
                logger.warning("Skipped missing remote blob during worktree restore: gs://%s/%s", _bucket_name(), name)
                continue
            raise

    logger.info(
        "Restored %d annotation files from gs://%s/%s (skipped missing: %d)",
        restored,
        _bucket_name(),
        prefix,
        skipped_missing,
    )
    return restored
