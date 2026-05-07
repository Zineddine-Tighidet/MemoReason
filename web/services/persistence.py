"""Local-only persistence hooks for the public annotation UI.

The public release runs the interface from local files and a local SQLite
database. These functions keep the same call sites as the internal deployment
code, but they intentionally do not sync state to any remote service.
"""

from __future__ import annotations

from pathlib import Path


def is_enabled() -> bool:
    """Return whether remote state sync is enabled."""
    return False


def restore_state_from_gcs() -> None:
    """No-op compatibility hook for local-only runs."""
    return None


def restore_db_from_gcs(
    *,
    max_age_seconds: float | None = None,
    force_download: bool = False,
) -> None:
    """No-op compatibility hook for local-only runs."""
    _ = (max_age_seconds, force_download)
    return None


def restore_worktree_from_gcs() -> None:
    """No-op compatibility hook for local-only runs."""
    return None


def restore_work_file_from_gcs(local_path: Path, *, max_age_seconds: float | None = None) -> bool:
    """Return whether the local file already exists."""
    _ = max_age_seconds
    return Path(local_path).exists()


def sync_db_to_gcs() -> None:
    """No-op compatibility hook for local-only runs."""
    return None


def sync_work_file_to_gcs(local_path: Path) -> None:
    """No-op compatibility hook for local-only runs."""
    _ = local_path
    return None


def delete_work_prefix_from_gcs(relative_prefix: str | Path) -> None:
    """No-op compatibility hook for local-only runs."""
    _ = relative_prefix
    return None
