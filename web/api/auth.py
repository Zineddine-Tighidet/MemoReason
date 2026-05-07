"""Authentication API endpoints."""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Response

from web.middleware.auth import get_current_user, require_power_user
from web.services import auth_service
from web.services import workflow_service

router = APIRouter(prefix="/api/v1/auth")


@router.post("/login")
def login(body: Dict[str, str], response: Response) -> Dict[str, Any]:
    """Login with username and password. Sets session cookie."""
    username = body.get("username", "")
    password = body.get("password", "")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")

    user = auth_service.authenticate(username, password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = auth_service.create_session(user["id"])
    if auth_service.SESSION_UNLIMITED:
        cookie_expires = datetime.now(timezone.utc) + timedelta(
            seconds=auth_service.UNLIMITED_COOKIE_MAX_AGE_SECONDS
        )
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            samesite="strict",
            max_age=auth_service.UNLIMITED_COOKIE_MAX_AGE_SECONDS,
            expires=cookie_expires,
            path="/",
        )
    else:
        cookie_expires = datetime.now(timezone.utc) + timedelta(
            seconds=auth_service.SESSION_DURATION_SECONDS
        )
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            samesite="strict",
            max_age=auth_service.SESSION_DURATION_SECONDS,
            expires=cookie_expires,
            path="/",
        )
    return {
        "user": {"id": user["id"], "username": user["username"], "role": user["role"]},
    }


@router.post("/logout")
def logout(response: Response, user: Dict = Depends(get_current_user)) -> Dict[str, str]:
    """Logout: clear session."""
    response.delete_cookie("session_token", path="/")
    return {"status": "logged_out"}


@router.get("/me")
def me(user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get current user info."""
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


@router.post("/users")
def create_user(body: Dict[str, str], user: Dict = Depends(require_power_user)) -> Dict[str, Any]:
    """Create a new user (power_user only)."""
    username = body.get("username", "")
    password = body.get("password", "")
    role = body.get("role", "regular_user")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password required")
    if role not in ("power_user", "regular_user"):
        raise HTTPException(status_code=400, detail="Invalid role")
    try:
        user_id = auth_service.create_user(username, password, role)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already exists")

    workflow_assignment: Dict[str, Any] | None = None
    if role == "regular_user":
        try:
            workflow_assignment = workflow_service.add_annotator_to_active_run(int(user_id))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"User created but failed to add to active run: {exc}",
            ) from exc

    return {
        "id": user_id,
        "username": username,
        "role": role,
        "workflow_assignment": workflow_assignment,
    }


@router.get("/users")
def list_users(user: Dict = Depends(require_power_user)) -> List[Dict[str, Any]]:
    """List all users (power_user only)."""
    return auth_service.list_users()
