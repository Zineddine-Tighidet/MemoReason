"""SQLite database connection manager and schema initialization."""

import os
import sqlite3
import threading
from pathlib import Path

import bcrypt

from web.services.persistence import sync_db_to_gcs

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "annotation.db"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD_ENV = "DEFAULT_ADMIN_PASSWORD"
FALLBACK_ADMIN_PASSWORD = "admin"
_BIO_THEME = "biographies_of_famous_personalities"
_BIO_DOC_ID_REMAP = {
    "awards_01": "bio_12",
    "awards_03": "bio_13",
    "awards_11": "bio_14",
}

_local = threading.local()
_initialized = False
_init_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('power_user', 'regular_user')),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_path TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id),
    action TEXT NOT NULL CHECK(action IN ('edit', 'review', 'validate', 'migrate')),
    status TEXT NOT NULL CHECK(status IN ('draft', 'in_progress', 'completed', 'validated')),
    timestamp TEXT NOT NULL DEFAULT (datetime('now')),
    details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_history_document ON document_history(document_path);
CREATE INDEX IF NOT EXISTS idx_history_user ON document_history(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);

CREATE TABLE IF NOT EXISTS workflow_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'completed')),
    created_by_user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workflow_run_annotators (
    run_id INTEGER NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    PRIMARY KEY (run_id, user_id)
);

CREATE TABLE IF NOT EXISTS workflow_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    stage INTEGER NOT NULL CHECK(stage IN (1, 2)),
    assignee_user_id INTEGER NOT NULL REFERENCES users(id),
    status TEXT NOT NULL CHECK(status IN ('blocked', 'available', 'in_progress', 'completed')),
    predecessor_task_id INTEGER REFERENCES workflow_tasks(id),
    input_snapshot_path TEXT,
    output_snapshot_path TEXT,
    blocked_reason TEXT,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    available_at TEXT,
    started_at TEXT,
    completed_at TEXT,
    state_entered_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, theme, doc_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_run ON workflow_tasks(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_assignee ON workflow_tasks(assignee_user_id);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_doc ON workflow_tasks(theme, doc_id);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_status ON workflow_tasks(status);
CREATE INDEX IF NOT EXISTS idx_workflow_run_annotators_user ON workflow_run_annotators(user_id);

CREATE TABLE IF NOT EXISTS workflow_agreements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'resolved')) DEFAULT 'pending',
    packet_path TEXT,
    reviewer_a_task_id INTEGER REFERENCES workflow_tasks(id),
    reviewer_b_task_id INTEGER REFERENCES workflow_tasks(id),
    resolved_by_user_id INTEGER REFERENCES users(id),
    resolved_at TEXT,
    final_snapshot_path TEXT,
    final_source_label TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(run_id, theme, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_agreements_run ON workflow_agreements(run_id);
CREATE INDEX IF NOT EXISTS idx_workflow_agreements_status ON workflow_agreements(status);

CREATE TABLE IF NOT EXISTS review_campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
    seed INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('active', 'paused', 'completed')),
    created_by_user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS review_campaign_reviewers (
    campaign_id INTEGER NOT NULL REFERENCES review_campaigns(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    PRIMARY KEY (campaign_id, user_id)
);

CREATE TABLE IF NOT EXISTS review_campaign_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES review_campaigns(id) ON DELETE CASCADE,
    review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    assignee_user_id INTEGER NOT NULL REFERENCES users(id),
    initial_assignee_user_id INTEGER REFERENCES users(id),
    qa_group TEXT,
    qa_group_order INTEGER,
    status TEXT NOT NULL CHECK(status IN ('available', 'in_progress', 'completed')),
    input_snapshot_path TEXT NOT NULL,
    output_snapshot_path TEXT,
    assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(campaign_id, theme, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_review_campaigns_status ON review_campaigns(status);
CREATE INDEX IF NOT EXISTS idx_review_campaigns_type_status ON review_campaigns(review_type, status);
CREATE INDEX IF NOT EXISTS idx_review_campaign_reviewers_user ON review_campaign_reviewers(user_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_campaign ON review_campaign_tasks(campaign_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_assignee ON review_campaign_tasks(assignee_user_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_doc ON review_campaign_tasks(review_type, theme, doc_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_status ON review_campaign_tasks(review_type, status);

CREATE TABLE IF NOT EXISTS review_campaign_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES review_campaigns(id) ON DELETE CASCADE,
    review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    reviewer_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    task_id INTEGER REFERENCES review_campaign_tasks(id) ON DELETE SET NULL,
    snapshot_path TEXT NOT NULL,
    submitted_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(campaign_id, theme, doc_id, reviewer_user_id)
);

CREATE INDEX IF NOT EXISTS idx_review_campaign_submissions_campaign ON review_campaign_submissions(campaign_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_submissions_doc ON review_campaign_submissions(campaign_id, theme, doc_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_submissions_reviewer ON review_campaign_submissions(campaign_id, reviewer_user_id);

CREATE TABLE IF NOT EXISTS review_campaign_agreements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES review_campaigns(id) ON DELETE CASCADE,
    review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'ready', 'resolved', 'awaiting_reviewer_acceptance')) DEFAULT 'pending',
    reviewer_a_submission_id INTEGER REFERENCES review_campaign_submissions(id) ON DELETE SET NULL,
    reviewer_b_submission_id INTEGER REFERENCES review_campaign_submissions(id) ON DELETE SET NULL,
    resolved_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    resolved_at TEXT,
    final_snapshot_path TEXT,
    requires_reviewer_acceptance INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(campaign_id, theme, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_review_campaign_agreements_campaign ON review_campaign_agreements(campaign_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_agreements_doc ON review_campaign_agreements(campaign_id, theme, doc_id);
CREATE INDEX IF NOT EXISTS idx_review_campaign_agreements_status ON review_campaign_agreements(campaign_id, status);

CREATE TABLE IF NOT EXISTS review_artifact_statuses (
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
    status TEXT NOT NULL CHECK(status IN ('draft', 'in_progress', 'completed')),
    reviewed_by_user_id INTEGER REFERENCES users(id),
    reviewed_at TEXT,
    latest_task_id INTEGER REFERENCES review_campaign_tasks(id),
    latest_snapshot_path TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (theme, doc_id, review_type)
);

CREATE INDEX IF NOT EXISTS idx_review_artifact_statuses_type ON review_artifact_statuses(review_type, status);

CREATE TABLE IF NOT EXISTS review_campaign_resolution_responses (
    campaign_id INTEGER NOT NULL REFERENCES review_campaigns(id) ON DELETE CASCADE,
    review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    reviewer_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    reviewer_role TEXT NOT NULL CHECK(reviewer_role IN ('reviewer_a', 'reviewer_b')),
    response_status TEXT NOT NULL CHECK(response_status IN ('accepted', 'contest_requested')),
    responded_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (campaign_id, theme, doc_id, reviewer_user_id)
);

CREATE INDEX IF NOT EXISTS idx_review_campaign_resolution_responses_status
ON review_campaign_resolution_responses(response_status);
CREATE INDEX IF NOT EXISTS idx_review_campaign_resolution_responses_doc
ON review_campaign_resolution_responses(campaign_id, theme, doc_id);

CREATE TABLE IF NOT EXISTS workflow_resolution_responses (
    run_id INTEGER NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
    theme TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    reviewer_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    reviewer_role TEXT NOT NULL CHECK(reviewer_role IN ('reviewer_a', 'reviewer_b')),
    response_status TEXT NOT NULL CHECK(response_status IN ('accepted', 'contest_requested')),
    responded_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (run_id, theme, doc_id, reviewer_user_id)
);

CREATE INDEX IF NOT EXISTS idx_workflow_resolution_responses_status ON workflow_resolution_responses(response_status);
CREATE INDEX IF NOT EXISTS idx_workflow_resolution_responses_doc ON workflow_resolution_responses(run_id, theme, doc_id);
"""


def _open_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, colspec: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["name"] == column for row in cols):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {colspec}")


def _migrate_review_campaign_tasks_uniqueness(conn: sqlite3.Connection) -> bool:
    """Drop the legacy one-row-per-doc uniqueness on review campaign tasks.

    The QA preannotation experiment needs one independent task row per reviewer
    for the same shared document, so `(campaign_id, theme, doc_id)` can no
    longer be globally unique.
    """
    row = conn.execute(
        """
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table' AND name = 'review_campaign_tasks'
        """
    ).fetchone()
    create_sql = str(row["sql"] or "") if row else ""
    legacy_constraint = "UNIQUE(campaign_id, theme, doc_id)"
    if legacy_constraint not in create_sql:
        return False

    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.executescript(
            """
            ALTER TABLE review_campaign_tasks RENAME TO review_campaign_tasks_legacy;

            CREATE TABLE review_campaign_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id INTEGER NOT NULL REFERENCES review_campaigns(id) ON DELETE CASCADE,
                review_type TEXT NOT NULL CHECK(review_type IN ('rules', 'questions')),
                theme TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                assignee_user_id INTEGER NOT NULL REFERENCES users(id),
                initial_assignee_user_id INTEGER REFERENCES users(id),
                qa_group TEXT,
                qa_group_order INTEGER,
                status TEXT NOT NULL CHECK(status IN ('available', 'in_progress', 'completed')),
                input_snapshot_path TEXT NOT NULL,
                output_snapshot_path TEXT,
                assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            INSERT INTO review_campaign_tasks (
                id,
                campaign_id,
                review_type,
                theme,
                doc_id,
                assignee_user_id,
                initial_assignee_user_id,
                qa_group,
                qa_group_order,
                status,
                input_snapshot_path,
                output_snapshot_path,
                assigned_at,
                started_at,
                completed_at,
                updated_at
            )
            SELECT
                id,
                campaign_id,
                review_type,
                theme,
                doc_id,
                assignee_user_id,
                initial_assignee_user_id,
                qa_group,
                qa_group_order,
                status,
                input_snapshot_path,
                output_snapshot_path,
                assigned_at,
                started_at,
                completed_at,
                updated_at
            FROM review_campaign_tasks_legacy;

            DROP TABLE review_campaign_tasks_legacy;

            CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_campaign ON review_campaign_tasks(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_assignee ON review_campaign_tasks(assignee_user_id);
            CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_doc ON review_campaign_tasks(review_type, theme, doc_id);
            CREATE INDEX IF NOT EXISTS idx_review_campaign_tasks_status ON review_campaign_tasks(review_type, status);
            """
        )
    finally:
        conn.execute("PRAGMA foreign_keys=ON")
    return True


def _configured_admin_password() -> str | None:
    password = os.getenv(DEFAULT_ADMIN_PASSWORD_ENV, "")
    password = password.strip()
    return password or None


def _migrate_bio_doc_ids(conn: sqlite3.Connection) -> bool:
    """One-off id remap for docs moved from awards to biographies theme.

    Idempotent and collision-safe:
    - no-op when old ids are absent
    - no-op when target ids already exist (avoid unique collisions)
    """
    old_ids = tuple(_BIO_DOC_ID_REMAP.keys())
    new_ids = tuple(_BIO_DOC_ID_REMAP.values())

    placeholders_old = ",".join("?" for _ in old_ids)
    placeholders_new = ",".join("?" for _ in new_ids)

    old_task_count = int(
        conn.execute(
            f"""
            SELECT COUNT(*) AS c
            FROM workflow_tasks
            WHERE theme = ?
              AND doc_id IN ({placeholders_old})
            """,
            (_BIO_THEME, *old_ids),
        ).fetchone()["c"]
    )
    if old_task_count == 0:
        return False

    task_collision = int(
        conn.execute(
            f"""
            SELECT COUNT(*) AS c
            FROM workflow_tasks
            WHERE theme = ?
              AND doc_id IN ({placeholders_new})
            """,
            (_BIO_THEME, *new_ids),
        ).fetchone()["c"]
    )
    agreement_collision = int(
        conn.execute(
            f"""
            SELECT COUNT(*) AS c
            FROM workflow_agreements
            WHERE theme = ?
              AND doc_id IN ({placeholders_new})
            """,
            (_BIO_THEME, *new_ids),
        ).fetchone()["c"]
    )
    history_collision = int(
        conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM document_history
            WHERE document_path IN (?, ?, ?)
            """,
            (
                f"{_BIO_THEME}/{new_ids[0]}",
                f"{_BIO_THEME}/{new_ids[1]}",
                f"{_BIO_THEME}/{new_ids[2]}",
            ),
        ).fetchone()["c"]
    )
    if task_collision > 0 or agreement_collision > 0 or history_collision > 0:
        return False

    conn.execute(
        """
        UPDATE workflow_tasks
        SET doc_id = CASE doc_id
            WHEN 'awards_01' THEN 'bio_12'
            WHEN 'awards_03' THEN 'bio_13'
            WHEN 'awards_11' THEN 'bio_14'
            ELSE doc_id
        END,
            input_snapshot_path = REPLACE(REPLACE(REPLACE(COALESCE(input_snapshot_path, ''), '/awards_01.', '/bio_12.'), '/awards_03.', '/bio_13.'), '/awards_11.', '/bio_14.'),
            output_snapshot_path = REPLACE(REPLACE(REPLACE(COALESCE(output_snapshot_path, ''), '/awards_01.', '/bio_12.'), '/awards_03.', '/bio_13.'), '/awards_11.', '/bio_14.'),
            updated_at = datetime('now')
        WHERE theme = 'biographies_of_famous_personalities'
          AND doc_id IN ('awards_01', 'awards_03', 'awards_11')
        """
    )

    conn.execute(
        """
        UPDATE workflow_agreements
        SET doc_id = CASE doc_id
            WHEN 'awards_01' THEN 'bio_12'
            WHEN 'awards_03' THEN 'bio_13'
            WHEN 'awards_11' THEN 'bio_14'
            ELSE doc_id
        END,
            packet_path = REPLACE(REPLACE(REPLACE(COALESCE(packet_path, ''), '/awards_01.', '/bio_12.'), '/awards_03.', '/bio_13.'), '/awards_11.', '/bio_14.'),
            final_snapshot_path = REPLACE(REPLACE(REPLACE(COALESCE(final_snapshot_path, ''), '/awards_01.', '/bio_12.'), '/awards_03.', '/bio_13.'), '/awards_11.', '/bio_14.'),
            updated_at = datetime('now')
        WHERE theme = 'biographies_of_famous_personalities'
          AND doc_id IN ('awards_01', 'awards_03', 'awards_11')
        """
    )

    conn.execute(
        """
        UPDATE document_history
        SET document_path = CASE document_path
            WHEN 'biographies_of_famous_personalities/awards_01' THEN 'biographies_of_famous_personalities/bio_12'
            WHEN 'biographies_of_famous_personalities/awards_03' THEN 'biographies_of_famous_personalities/bio_13'
            WHEN 'biographies_of_famous_personalities/awards_11' THEN 'biographies_of_famous_personalities/bio_14'
            ELSE document_path
        END
        WHERE document_path IN (
            'biographies_of_famous_personalities/awards_01',
            'biographies_of_famous_personalities/awards_03',
            'biographies_of_famous_personalities/awards_11'
        )
        """
    )
    return True


def get_db() -> sqlite3.Connection:
    """Get a thread-local database connection, auto-initializing schema if needed."""
    global _initialized
    if not _initialized:
        init_db()
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = _open_connection()
    return _local.conn


def init_db() -> None:
    """Initialize the database schema and seed default admin user."""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        conn = _open_connection()
        try:
            conn.executescript(SCHEMA)
            _ensure_column(conn, "workflow_agreements", "final_snapshot_path", "TEXT")
            _ensure_column(conn, "workflow_agreements", "final_source_label", "TEXT")
            _ensure_column(conn, "review_campaign_agreements", "requires_reviewer_acceptance", "INTEGER NOT NULL DEFAULT 0")
            _ensure_column(conn, "review_campaign_tasks", "initial_assignee_user_id", "INTEGER REFERENCES users(id)")
            _ensure_column(conn, "review_campaign_tasks", "qa_group", "TEXT")
            _ensure_column(conn, "review_campaign_tasks", "qa_group_order", "INTEGER")
            needs_sync = False
            if _migrate_review_campaign_tasks_uniqueness(conn):
                needs_sync = True
            total_users = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()["c"]
            admin_row = conn.execute(
                "SELECT id, password_hash FROM users WHERE username = ?",
                (DEFAULT_ADMIN_USERNAME,),
            ).fetchone()
            configured_admin_password = _configured_admin_password()

            if admin_row is None and (total_users == 0 or configured_admin_password is not None):
                admin_password = configured_admin_password or FALLBACK_ADMIN_PASSWORD
                pw_hash = bcrypt.hashpw(admin_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (DEFAULT_ADMIN_USERNAME, pw_hash, "power_user"),
                )
                needs_sync = True
            elif admin_row is not None and configured_admin_password is not None:
                current_hash = str(admin_row["password_hash"] or "")
                password_matches = False
                if current_hash:
                    try:
                        password_matches = bcrypt.checkpw(
                            configured_admin_password.encode("utf-8"),
                            current_hash.encode("utf-8"),
                        )
                    except ValueError:
                        password_matches = False
                if not password_matches:
                    pw_hash = bcrypt.hashpw(configured_admin_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
                    conn.execute(
                        "UPDATE users SET password_hash = ?, role = ? WHERE id = ?",
                        (pw_hash, "power_user", admin_row["id"]),
                    )
                    needs_sync = True

            if _migrate_bio_doc_ids(conn):
                needs_sync = True

            if needs_sync:
                conn.commit()
                sync_db_to_gcs()
            _initialized = True
        finally:
            conn.close()


def close_db() -> None:
    """Close the thread-local connection."""
    conn = getattr(_local, "conn", None)
    if conn:
        conn.close()
        _local.conn = None


def reset_db() -> None:
    """Reset state for testing."""
    global _initialized
    close_db()
    _initialized = False
