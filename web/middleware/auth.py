"""FastAPI authentication dependencies."""

import time
from typing import Any, Dict

from fastapi import Cookie, HTTPException, Request

from web.services.db import close_db
from web.services.auth_service import validate_session
from web.services.persistence import is_enabled, restore_db_from_gcs


def _resolve_session_user(session_token: str) -> Dict[str, Any] | None:
    user = validate_session(session_token)
    if user or not session_token or not is_enabled():
        return user

    # Cloud Run can serve the HTML login redirect on one instance while the
    # first dashboard API fan-out lands on another instance that has not yet
    # pulled the freshly synced session row. Retry the refresh path once more
    # with a short delay before treating the session as truly expired.
    for attempt in range(2):
        close_db()
        restore_db_from_gcs(max_age_seconds=0, force_download=True)
        user = validate_session(session_token)
        if user:
            return user
        if attempt == 0:
            time.sleep(0.25)
    return None


def get_current_user(request: Request, session_token: str = Cookie(default="")) -> Dict[str, Any]:
    """Dependency: extract the current user from the session cookie.
    Raises 401 if not authenticated."""
    user = _resolve_session_user(session_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def require_power_user(request: Request, session_token: str = Cookie(default="")) -> Dict[str, Any]:
    """Dependency: require a power_user role. Raises 403 if not authorized."""
    user = get_current_user(request, session_token)
    if user["role"] != "power_user":
        raise HTTPException(status_code=403, detail="Power user access required")
    return user
