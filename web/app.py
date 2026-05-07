"""FastAPI application for the annotation web interface v2."""

import hashlib
import html
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import Cookie, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from web.api.auth import router as auth_router
from web.api.documents import router as documents_router
from web.api.history import router as history_router
from web.api.taxonomy import router as taxonomy_router
from web.api.workflow import router as workflow_router
from web.services import auth_service, yaml_service
from web.services.auth_service import validate_session
from web.services.db import close_db, init_db
from web.services.persistence import is_enabled, restore_db_from_gcs


@asynccontextmanager
async def lifespan(app: FastAPI):
    yaml_service.ensure_workspace_dirs()
    # Cold-start optimization:
    # restoring the full workspace can take tens of seconds when instances wake
    # up after idle. We only need the DB snapshot at startup; YAML files are
    # restored lazily per document through restore_work_file_from_gcs().
    restore_db_from_gcs(max_age_seconds=0, force_download=True)
    init_db()
    yield
    close_db()


app = FastAPI(title="Parametric Annotation Tool", lifespan=lifespan)
CANONICAL_HOST = os.getenv("APP_CANONICAL_HOST", "").strip().lower()
MAINTENANCE_MODE = os.getenv("ANNOTATION_MAINTENANCE_MODE", "off").strip().lower()
MAINTENANCE_TZ_NAME = os.getenv("ANNOTATION_MAINTENANCE_TZ", "Europe/Paris").strip() or "Europe/Paris"
try:
    MAINTENANCE_AUTO_UNTIL_HOUR = int(os.getenv("ANNOTATION_MAINTENANCE_AUTO_UNTIL_HOUR", "11"))
except ValueError:
    MAINTENANCE_AUTO_UNTIL_HOUR = 11


@app.middleware("http")
async def enforce_canonical_host(request: Request, call_next):
    """Redirect alternate service URLs to one canonical host for stable cookies."""
    if not CANONICAL_HOST:
        return await call_next(request)

    host_header = request.headers.get("host", "")
    request_host = host_header.split(":", 1)[0].strip().lower()
    if request_host and request_host != CANONICAL_HOST:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip()
        scheme = forwarded_proto if forwarded_proto in {"http", "https"} else request.url.scheme
        if scheme not in {"http", "https"}:
            scheme = "https"
        redirect_url = request.url.replace(scheme=scheme, netloc=CANONICAL_HOST)
        return RedirectResponse(url=str(redirect_url), status_code=308)

    return await call_next(request)


@app.middleware("http")
async def disable_cache_for_dynamic_routes(request: Request, call_next):
    """Prevent browsers and intermediaries from caching dynamic app state.

    The annotation UI depends on frequently changing shared state coming from
    the synchronized SQLite snapshot. If browsers reuse cached dashboard or
    queue responses, users can see stale campaign memberships or stale task
    availability even after the backend state has already been fixed.
    """
    response = await call_next(request)
    path = request.url.path or ""
    if path.startswith("/static/"):
        return response
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    existing_vary = response.headers.get("Vary", "").strip()
    if existing_vary:
        if "Cookie" not in {item.strip() for item in existing_vary.split(",") if item.strip()}:
            response.headers["Vary"] = f"{existing_vary}, Cookie"
    else:
        response.headers["Vary"] = "Cookie"
    return response

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
TEMPLATES_DIR = Path(__file__).parent / "templates"
RULE_ANNOTATION_GUIDE_RELATIVE_PATH = Path("docs") / "RULE_REVIEW_GUIDE.md"
RULE_ANNOTATION_GUIDE_PATH = PROJECT_ROOT / RULE_ANNOTATION_GUIDE_RELATIVE_PATH
RULE_ANNOTATION_GUIDE_BUNDLED_RELATIVE_PATH = Path("web") / "docs" / "rule_annotation_manual.md"
RULE_ANNOTATION_GUIDE_BUNDLED_PATH = PROJECT_ROOT / RULE_ANNOTATION_GUIDE_BUNDLED_RELATIVE_PATH


def _resolve_rule_annotation_guide_path() -> Path | None:
    """Return the first existing rule-guide path from common runtime roots."""
    configured_path = os.getenv("RULE_ANNOTATION_GUIDE_PATH", "").strip()
    candidate_paths: list[Path] = []
    if configured_path:
        configured = Path(configured_path).expanduser()
        if not configured.is_absolute():
            configured = Path.cwd() / configured
        candidate_paths.append(configured)

    candidate_paths.extend(
        [
            RULE_ANNOTATION_GUIDE_PATH,
            RULE_ANNOTATION_GUIDE_BUNDLED_PATH,
            Path.cwd() / RULE_ANNOTATION_GUIDE_RELATIVE_PATH,
            Path.cwd() / RULE_ANNOTATION_GUIDE_BUNDLED_RELATIVE_PATH,
        ]
    )

    seen: set[Path] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _inline_markdown_to_html(text: str) -> str:
    """Render minimal inline markdown safely."""
    rendered = html.escape(text)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", rendered)
    rendered = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener noreferrer">\1</a>', rendered)
    return rendered


def _markdown_to_html(markdown_text: str) -> str:
    """Render markdown for guide display without external dependencies."""
    lines = markdown_text.splitlines()
    parts: list[str] = []
    paragraph_lines: list[str] = []
    in_ul = False
    in_ol = False
    in_code_block = False

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        merged = " ".join(line.strip() for line in paragraph_lines).strip()
        if merged:
            parts.append(f"<p>{_inline_markdown_to_html(merged)}</p>")
        paragraph_lines = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            parts.append("</ul>")
            in_ul = False
        if in_ol:
            parts.append("</ol>")
            in_ol = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if in_code_block:
            if stripped.startswith("```"):
                parts.append("</code></pre>")
                in_code_block = False
            else:
                parts.append(html.escape(raw_line))
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            close_lists()
            parts.append("<pre><code>")
            in_code_block = True
            continue

        if not stripped:
            flush_paragraph()
            close_lists()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            close_lists()
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            parts.append(f"<h{level}>{_inline_markdown_to_html(heading_text)}</h{level}>")
            continue

        ordered_match = re.match(r"^\d+\.\s+(.*)$", stripped)
        if ordered_match:
            flush_paragraph()
            if in_ul:
                parts.append("</ul>")
                in_ul = False
            if not in_ol:
                parts.append("<ol>")
                in_ol = True
            parts.append(f"<li>{_inline_markdown_to_html(ordered_match.group(1).strip())}</li>")
            continue

        unordered_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if unordered_match:
            flush_paragraph()
            if in_ol:
                parts.append("</ol>")
                in_ol = False
            if not in_ul:
                parts.append("<ul>")
                in_ul = True
            parts.append(f"<li>{_inline_markdown_to_html(unordered_match.group(1).strip())}</li>")
            continue

        if in_ul or in_ol:
            close_lists()
        paragraph_lines.append(stripped)

    flush_paragraph()
    close_lists()
    if in_code_block:
        parts.append("</code></pre>")

    return "\n".join(parts)


def _load_rule_annotation_guide() -> tuple[str, str, str]:
    """Return rule-guide markdown, rendered HTML, and source path label."""
    guide_path = _resolve_rule_annotation_guide_path()
    if guide_path is None:
        fallback = (
            "Rule annotation guide not found.\n"
            "Checked:\n"
            f"- {RULE_ANNOTATION_GUIDE_PATH}\n"
            f"- {RULE_ANNOTATION_GUIDE_BUNDLED_PATH}\n"
            f"- {Path.cwd() / RULE_ANNOTATION_GUIDE_RELATIVE_PATH}\n"
            f"- {Path.cwd() / RULE_ANNOTATION_GUIDE_BUNDLED_RELATIVE_PATH}"
        )
        return fallback, _markdown_to_html(fallback), str(RULE_ANNOTATION_GUIDE_RELATIVE_PATH)

    guide_markdown = guide_path.read_text(encoding="utf-8")
    source_label = str(guide_path)
    for root in (PROJECT_ROOT, Path.cwd()):
        try:
            source_label = str(guide_path.relative_to(root))
            break
        except ValueError:
            continue
    return guide_markdown, _markdown_to_html(guide_markdown), source_label


def _template_common_context(_request: Request) -> dict[str, str]:
    guide_markdown, guide_html, guide_source = _load_rule_annotation_guide()
    return {
        "rule_annotation_guide_text": guide_markdown,
        "rule_annotation_guide_html": guide_html,
        "rule_annotation_guide_source_path": guide_source,
    }


templates = Jinja2Templates(
    directory=str(TEMPLATES_DIR),
    context_processors=[_template_common_context],
)


def asset_url(relative_path: str) -> str:
    """Return a static asset URL with a content-hash cache-busting suffix."""
    cleaned = str(relative_path or "").lstrip("/")
    asset_path = STATIC_DIR / cleaned
    version = "0"
    if asset_path.exists():
        version = hashlib.sha256(asset_path.read_bytes()).hexdigest()[:12]
    return f"/static/{cleaned}?v={version}"


templates.env.globals["asset_url"] = asset_url

# Register API routes
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(history_router)
app.include_router(taxonomy_router)
app.include_router(workflow_router)


# --- Helper to check auth for HTML pages ---

def _get_user_or_none(session_token: str):
    """Return user dict if session is valid, else None.

    HTML routes need the same remote-refresh fallback as API auth dependencies,
    otherwise a login handled by one Cloud Run instance can look like a failed
    login when the redirected page lands on another instance with a stale DB
    snapshot.
    """
    if not session_token:
        return None
    user = validate_session(session_token)
    if not user and is_enabled():
        close_db()
        restore_db_from_gcs(max_age_seconds=0, force_download=True)
        user = validate_session(session_token)
    return user


def _set_session_cookie(response, token: str) -> None:
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
        return

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


def _maintenance_context() -> dict[str, str] | None:
    mode = MAINTENANCE_MODE
    if mode in {"", "off", "0", "false", "disabled"}:
        return None

    try:
        tz = ZoneInfo(MAINTENANCE_TZ_NAME)
    except Exception:
        tz = ZoneInfo("Europe/Paris")

    now = datetime.now(tz)
    auto_until = now.replace(
        hour=max(0, min(MAINTENANCE_AUTO_UNTIL_HOUR, 23)),
        minute=0,
        second=0,
        microsecond=0,
    )

    if mode == "auto" and now >= auto_until:
        return None

    until_text = auto_until.strftime("%-I:%M %p") if mode == "auto" else os.getenv(
        "ANNOTATION_MAINTENANCE_UNTIL_TEXT",
        "11:00 AM",
    ).strip() or "11:00 AM"
    return {
        "headline": "Scheduled maintenance in progress",
        "message": f"The annotation interface is temporarily unavailable while we ship fixes. Please check back at {until_text}.",
        "until_text": until_text,
    }


def _maintenance_response(request: Request, user: dict | None):
    maintenance = _maintenance_context()
    if not maintenance:
        return None
    return templates.TemplateResponse(
        "maintenance.html",
        {
            "request": request,
            "user": user,
            "maintenance": maintenance,
        },
    )


# --- HTML page routes ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    if user:
        return RedirectResponse("/", status_code=302)
    initial_error = str(request.query_params.get("error") or "").strip()
    return templates.TemplateResponse(
        "login.html",
        {
            "request": request,
            "initial_error": initial_error,
        },
    )


@app.post("/login", response_class=HTMLResponse)
async def login_submit(request: Request, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    if user:
        return RedirectResponse("/", status_code=302)

    raw_body = (await request.body()).decode("utf-8", errors="ignore")
    form_data = parse_qs(raw_body, keep_blank_values=True)
    username = str((form_data.get("username") or [""])[0]).strip()
    password = str((form_data.get("password") or [""])[0])

    if not username or not password:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "initial_error": "Username and password required",
            },
            status_code=400,
        )

    authenticated = auth_service.authenticate(username, password)
    if not authenticated:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "initial_error": "Invalid credentials",
            },
            status_code=401,
        )

    token = auth_service.create_session(int(authenticated["id"]))
    response = RedirectResponse("/", status_code=302)
    _set_session_cookie(response, token)
    return response


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    maintenance_response = _maintenance_response(request, user)
    if maintenance_response is not None:
        return maintenance_response
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})


@app.get("/editor/{theme}/{doc_id}", response_class=HTMLResponse)
async def editor(request: Request, theme: str, doc_id: str, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    maintenance_response = _maintenance_response(request, user)
    if maintenance_response is not None:
        return maintenance_response
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("editor.html", {
        "request": request,
        "user": user,
        "theme": theme,
        "doc_id": doc_id,
    })


@app.get("/review/{theme}/{doc_id}", response_class=HTMLResponse)
async def review(request: Request, theme: str, doc_id: str, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    maintenance_response = _maintenance_response(request, user)
    if maintenance_response is not None:
        return maintenance_response
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("review.html", {
        "request": request,
        "user": user,
        "theme": theme,
        "doc_id": doc_id,
    })


@app.get("/taxonomy", response_class=HTMLResponse)
async def taxonomy_editor(request: Request, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    maintenance_response = _maintenance_response(request, user)
    if maintenance_response is not None:
        return maintenance_response
    if not user or user.get("role") != "power_user":
        return RedirectResponse("/", status_code=302)
    return templates.TemplateResponse("taxonomy.html", {"request": request, "user": user})


@app.get("/help/rule-annotation", response_class=HTMLResponse)
async def rule_annotation_guide(request: Request, session_token: str = Cookie(default="")):
    user = _get_user_or_none(session_token)
    maintenance_response = _maintenance_response(request, user)
    if maintenance_response is not None:
        return maintenance_response
    if not user:
        return RedirectResponse("/login", status_code=302)

    guide_markdown, guide_markdown_html, guide_source_path = _load_rule_annotation_guide()

    return templates.TemplateResponse(
        "rule_annotation_guide.html",
        {
            "request": request,
            "user": user,
            "guide_markdown": guide_markdown,
            "guide_markdown_html": guide_markdown_html,
            "guide_source_path": guide_source_path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=True)
