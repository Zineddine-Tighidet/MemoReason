"""Authentication service: password hashing, session management, user CRUD."""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import bcrypt

from web.services.db import close_db, get_db
from web.services.persistence import restore_db_from_gcs, sync_db_to_gcs

logger = logging.getLogger(__name__)

_SESSION_DURATION_RAW = os.getenv("SESSION_DURATION_HOURS", "24").strip().lower()
SESSION_UNLIMITED = _SESSION_DURATION_RAW in {
    "0",
    "unlimited",
    "infinite",
    "inf",
    "none",
    "no_expiry",
    "no-expiry",
}

if SESSION_UNLIMITED:
    SESSION_DURATION_HOURS = 0
    SESSION_DURATION_SECONDS = 0
else:
    SESSION_DURATION_HOURS = int(_SESSION_DURATION_RAW or "24")
    SESSION_DURATION_SECONDS = SESSION_DURATION_HOURS * 60 * 60

SESSION_EXPIRES_AT_UNLIMITED = "9999-12-31 23:59:59"
UNLIMITED_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365 * 10


def _utc_now_sql() -> str:
    """Return UTC timestamp in SQLite-friendly format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def authenticate(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify credentials and return user dict or None."""
    db = get_db()
    row = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row and verify_password(password, row["password_hash"]):
        return dict(row)
    return None


def _sync_auth_db_change(operation: str) -> None:
    """Persist auth DB changes and retry once against the latest remote snapshot."""
    try:
        sync_db_to_gcs()
        return
    except RuntimeError:
        logger.warning(
            "Auth DB sync drift detected during %s; refreshing remote DB and retrying once.",
            operation,
            exc_info=True,
        )
    except Exception:
        logger.warning("Failed to sync auth DB change during %s.", operation, exc_info=True)
        return

    try:
        close_db()
        restore_db_from_gcs(max_age_seconds=0, force_download=True)
        sync_db_to_gcs()
    except Exception:
        logger.warning("Failed to resync auth DB change during %s after refresh.", operation, exc_info=True)


def create_session(user_id: int) -> str:
    """Create a session token and store in database. Returns the token."""
    db = get_db()
    token = secrets.token_urlsafe(32)
    if SESSION_UNLIMITED:
        expires = SESSION_EXPIRES_AT_UNLIMITED
    else:
        expires = (datetime.now(timezone.utc) + timedelta(hours=SESSION_DURATION_HOURS)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    db.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
        (token, user_id, expires),
    )
    db.commit()
    _sync_auth_db_change("login")
    _ensure_session_replicated(token, user_id, expires)
    return token


def _ensure_session_replicated(token: str, user_id: int, expires_at: str) -> None:
    """Best-effort guarantee that a newly issued session is visible cross-instance.

    Cloud Run requests for the redirected dashboard can race onto a worker that only
    sees the last synced DB snapshot. If the login write wasn't replicated yet,
    authenticated HTML can still load while API fan-out returns 401.
    """
    if not token:
        return

    try:
        close_db()
        restore_db_from_gcs(max_age_seconds=0, force_download=True)
        if validate_session(token):
            return
    except Exception:
        logger.warning(
            "Failed to verify freshly-issued session against remote snapshot.",
            exc_info=True,
        )

    try:
        db = get_db()
        db.execute(
            "INSERT OR REPLACE INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, int(user_id), str(expires_at)),
        )
        db.commit()
        sync_db_to_gcs()
        close_db()
        restore_db_from_gcs(max_age_seconds=0, force_download=True)
    except Exception:
        logger.warning(
            "Failed to force-replicate freshly-issued session token.",
            exc_info=True,
        )


def validate_session(token: str) -> Optional[Dict[str, Any]]:
    """Check if a session token is valid. Returns user dict or None."""
    if not token:
        return None
    db = get_db()
    if SESSION_UNLIMITED:
        row = db.execute(
            """SELECT u.id, u.username, u.role, u.created_at
               FROM sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token = ?""",
            (token,),
        ).fetchone()
    else:
        # Clean up expired sessions
        db.execute("DELETE FROM sessions WHERE datetime(expires_at) <= datetime(?)", (_utc_now_sql(),))
        db.commit()

        row = db.execute(
            """SELECT u.id, u.username, u.role, u.created_at
               FROM sessions s JOIN users u ON s.user_id = u.id
               WHERE s.token = ? AND datetime(s.expires_at) > datetime(?)""",
            (token, _utc_now_sql()),
        ).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    """Delete a session (logout)."""
    db = get_db()
    db.execute("DELETE FROM sessions WHERE token = ?", (token,))
    db.commit()
    _sync_auth_db_change("logout")


def create_user(username: str, password: str, role: str = "regular_user") -> int:
    """Create a new user. Returns user id. Raises on duplicate username."""
    db = get_db()
    pw_hash = hash_password(password)
    cursor = db.execute(
        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
        (username, pw_hash, role),
    )
    db.commit()
    sync_db_to_gcs()
    return cursor.lastrowid


def list_users() -> List[Dict[str, Any]]:
    """List all users (without password hashes)."""
    db = get_db()
    rows = db.execute("SELECT id, username, role, created_at FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get a user by id."""
    db = get_db()
    row = db.execute("SELECT id, username, role, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None
