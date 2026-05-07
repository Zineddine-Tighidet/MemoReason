"""Annotation history / audit trail API endpoints."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query

from web.middleware.auth import get_current_user
from web.services.db import close_db, get_db
from web.services.persistence import restore_db_from_gcs, sync_db_to_gcs
from web.services.yaml_service import load_document, load_source_document
from web.services import review_campaign_service, workflow_service, yaml_service

router = APIRouter(prefix="/api/v1/history")


def _refresh_dashboard_db_snapshot() -> None:
    close_db()
    restore_db_from_gcs(max_age_seconds=5.0)


def _theme_doc_paths(theme: str, doc_id: str) -> List[str]:
    canonical = yaml_service.canonical_theme_id(theme)
    paths = [f"{canonical}/{doc_id}"]
    if canonical == yaml_service.PUBLIC_ATTACKS_THEME:
        legacy = f"{yaml_service.LEGACY_THEME}/{doc_id}"
        if legacy not in paths:
            paths.append(legacy)
    return paths


def _safe_parse_snapshot(details_json: str | None) -> Dict[str, Any]:
    if not details_json:
        return {}
    try:
        parsed = json.loads(details_json)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def build_document_history_payload(
    theme: str,
    doc_id: str,
    review_type: str | None = None,
    *,
    db=None,
) -> Dict[str, Any]:
    """Return the history payload used by the editor and review pages."""
    if db is None:
        db = get_db()
    canonical_theme = yaml_service.canonical_theme_id(theme)
    normalized_review_type = str(review_type or "").strip().lower()
    if normalized_review_type in {"rules", "questions"}:
        annotation_versions = _build_review_versions(db, normalized_review_type, canonical_theme, doc_id)
        review_statuses = review_campaign_service.get_document_review_statuses(canonical_theme, doc_id, db=db)
        review_bucket = review_statuses.get(normalized_review_type, {})
        return {
            "entries": [],
            "current_status": str(review_bucket.get("status") or "draft"),
            "last_editor": review_bucket.get("last_edited_by"),
            "annotation_versions": annotation_versions,
        }

    doc_paths = _theme_doc_paths(canonical_theme, doc_id)
    placeholders = ",".join("?" for _ in doc_paths)
    rows = db.execute(
        f"""SELECT h.id, u.username, h.action, h.status, h.timestamp, h.details_json
           FROM document_history h JOIN users u ON h.user_id = u.id
           WHERE h.document_path IN ({placeholders})
           ORDER BY h.timestamp DESC, h.id DESC""",
        tuple(doc_paths),
    ).fetchall()

    entries = [dict(r) for r in rows]
    annotation_versions = _build_workflow_versions(db, canonical_theme, doc_id, entries)

    current_status = entries[0]["status"] if entries else "draft"
    last_editor = entries[0]["username"] if entries else None
    dashboard_status = _resolve_dashboard_status(canonical_theme, doc_id)
    if dashboard_status:
        dashboard_value = str(dashboard_status.get("status") or "").strip()
        if dashboard_value:
            current_status = dashboard_value
        dashboard_editor = dashboard_status.get("last_edited_by")
        if dashboard_editor:
            last_editor = dashboard_editor

    return {
        "entries": entries,
        "current_status": current_status,
        "last_editor": last_editor,
        "annotation_versions": annotation_versions,
    }


def build_annotations_metadata_payload(theme: str, doc_id: str, *, db=None) -> Dict[str, Any]:
    """Return per-annotation provenance metadata for power-user editor views."""
    if db is None:
        db = get_db()
    canonical_theme = yaml_service.canonical_theme_id(theme)
    doc_paths = _theme_doc_paths(canonical_theme, doc_id)
    placeholders = ",".join("?" for _ in doc_paths)

    rows = db.execute(
        f"""SELECT h.id, u.username, h.timestamp, h.details_json
           FROM document_history h JOIN users u ON h.user_id = u.id
           WHERE h.document_path IN ({placeholders})
           ORDER BY h.timestamp ASC, h.id ASC""",
        tuple(doc_paths),
    ).fetchall()

    import re

    annotation_metadata = {}
    question_metadata = {}
    rule_metadata = {}

    for row in rows:
        if not row["details_json"]:
            continue

        snapshot = _safe_parse_snapshot(row["details_json"])
        if not snapshot:
            continue
        username = row["username"]
        timestamp = row["timestamp"]

        doc_text = snapshot.get("document", "")
        if doc_text:
            pattern = r'\[([^\]]+);\s*([^\]]+)\]'
            for match in re.finditer(pattern, doc_text):
                entity_ref = match.group(2).strip()
                if entity_ref not in annotation_metadata:
                    annotation_metadata[entity_ref] = {
                        "username": username,
                        "timestamp": timestamp,
                        "first_seen": timestamp,
                    }
                annotation_metadata[entity_ref]["last_modified"] = timestamp
                annotation_metadata[entity_ref]["last_editor"] = username

        for q in snapshot.get("questions", []):
            qid = q.get("question_id")
            if not qid:
                continue
            qid = str(qid)
            if qid not in question_metadata:
                question_metadata[qid] = {
                    "username": username,
                    "timestamp": timestamp,
                    "first_seen": timestamp,
                }
            question_metadata[qid]["last_modified"] = timestamp
            question_metadata[qid]["last_editor"] = username

        for rule in snapshot.get("rules", []):
            if isinstance(rule, dict):
                rule_text = str(rule.get("rule") or "").strip()
            else:
                rule_text = str(rule or "").strip()
            if not rule_text:
                continue
            if rule_text not in rule_metadata:
                rule_metadata[rule_text] = {
                    "username": username,
                    "timestamp": timestamp,
                    "first_seen": timestamp,
                }
            rule_metadata[rule_text]["last_modified"] = timestamp
            rule_metadata[rule_text]["last_editor"] = username

    return {
        "annotations": annotation_metadata,
        "questions": question_metadata,
        "rules": rule_metadata,
        "has_history": len(rows) > 0,
    }


def build_recent_activity_payload(scope: str = "documents", *, db=None) -> List[Dict[str, Any]]:
    """Return recent dashboard activity for the selected scope."""
    if db is None:
        db = get_db()
    normalized_scope = str(scope or "documents").strip().lower()
    if normalized_scope in {"rules", "questions"}:
        return review_campaign_service.get_recent_review_activity(normalized_scope, db=db)
    rows = db.execute(
        """SELECT h.id, h.document_path, u.username, h.action, h.status, h.timestamp
           FROM document_history h JOIN users u ON h.user_id = u.id
           ORDER BY h.timestamp DESC
           LIMIT 50""",
    ).fetchall()
    return [dict(r) for r in rows]


def _snapshot_has_content(snapshot: Dict[str, Any]) -> bool:
    if not isinstance(snapshot, dict):
        return False
    document_text = snapshot.get("document", "")
    questions = snapshot.get("questions", []) or []
    rules = snapshot.get("rules", []) or []
    return bool(str(document_text).strip()) or bool(questions) or bool(rules)


def _resolve_snapshot_data(db, doc_path: str, history_id: int, details_json: str | None) -> Dict[str, Any]:
    """Resolve snapshot data for an entry, falling back to previous populated snapshot."""
    snapshot_data = _safe_parse_snapshot(details_json)
    if _snapshot_has_content(snapshot_data):
        return snapshot_data

    fallback_rows = db.execute(
        """SELECT details_json
           FROM document_history
           WHERE document_path = ? AND id <= ? AND details_json IS NOT NULL AND details_json != ''
           ORDER BY timestamp DESC, id DESC
           LIMIT 200""",
        (doc_path, history_id),
    ).fetchall()
    for row in fallback_rows:
        candidate = _safe_parse_snapshot(row["details_json"])
        if _snapshot_has_content(candidate):
            return candidate
    return snapshot_data


def _resolve_document_fallback(user: Dict[str, Any], theme: str, doc_id: str) -> Dict[str, Any]:
    """Best-effort fallback when no snapshot payload exists for a history entry."""
    try:
        doc_data = load_document(user["username"], theme, doc_id)
    except Exception:
        try:
            doc_data = load_source_document(theme, doc_id)
        except Exception:
            return {}
    return {
        "document": doc_data.get("document_to_annotate", "") or "",
        "questions": doc_data.get("questions", []) or [],
        "rules": doc_data.get("rules", []) or [],
    }


def _safe_parse_yaml_document(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    if isinstance(loaded, dict) and isinstance(loaded.get("document"), dict):
        doc = loaded["document"]
    elif isinstance(loaded, dict):
        doc = loaded
    else:
        doc = {}
    if not isinstance(doc, dict):
        return {}
    questions = doc.get("questions", []) or []
    rules = doc.get("rules", []) or []
    return {
        "document": doc.get("document_to_annotate", "") or "",
        "questions": questions,
        "rules": rules,
        "questions_count": len(questions),
        "rules_count": len(rules),
    }


def _resolve_source_snapshot(theme: str, doc_id: str, source_agent: str) -> Dict[str, Any]:
    ai_candidate = workflow_service.WORK_DIR / source_agent / theme / f"{doc_id}.yaml"
    if ai_candidate.exists():
        return _safe_parse_yaml_document(ai_candidate)
    try:
        source_doc = load_source_document(theme, doc_id)
    except Exception:
        return {}
    questions = source_doc.get("questions", []) or []
    rules = source_doc.get("rules", []) or []
    return {
        "document": source_doc.get("document_to_annotate", "") or "",
        "questions": questions,
        "rules": rules,
        "questions_count": len(questions),
        "rules_count": len(rules),
    }


def _resolve_dashboard_status(theme: str, doc_id: str) -> Dict[str, Any] | None:
    """Return workflow-aware status exactly as shown on dashboard cards."""
    payload = None
    try:
        # Canonical dashboard source.
        payload = yaml_service.get_theme_progress()
    except Exception:
        # Backward compatibility: older deployments exposed get_documents().
        getter = getattr(yaml_service, "get_documents", None)
        if callable(getter):
            try:
                payload = getter()
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return None

    for theme_block in payload.get("themes", []) or []:
        if yaml_service.canonical_theme_id(str(theme_block.get("theme_id") or "")) != yaml_service.canonical_theme_id(str(theme)):
            continue
        for document in theme_block.get("documents", []) or []:
            if str(document.get("doc_id") or "") == str(doc_id):
                return {
                    "status": str(document.get("status") or ""),
                    "last_edited_by": document.get("last_edited_by"),
                }
    return None


def _workflow_context(db, theme: str, doc_id: str) -> Dict[str, Any] | None:
    canonical = yaml_service.canonical_theme_id(theme)
    theme_variants = [canonical]
    if canonical == yaml_service.PUBLIC_ATTACKS_THEME:
        theme_variants.append(yaml_service.LEGACY_THEME)
    placeholders = ",".join("?" for _ in theme_variants)
    try:
        row = db.execute(
            f"""
            SELECT
                a.run_id,
                r.source_agent,
                r.created_at AS run_created_at,
                a.status AS agreement_status,
                a.packet_path,
                a.final_snapshot_path,
                a.final_source_label,
                a.resolved_at,
                ua.username AS resolved_by,
                t1.output_snapshot_path AS reviewer_a_output,
                t1.status AS reviewer_a_status,
                t1.completed_at AS reviewer_a_completed_at,
                u1.username AS reviewer_a_username,
                t2.output_snapshot_path AS reviewer_b_output,
                t2.status AS reviewer_b_status,
                t2.completed_at AS reviewer_b_completed_at,
                u2.username AS reviewer_b_username
            FROM workflow_agreements a
            JOIN workflow_runs r ON r.id = a.run_id
            LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
            LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
            LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
            LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
            LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
            WHERE a.theme IN ({placeholders})
              AND a.doc_id = ?
            ORDER BY a.run_id DESC
            LIMIT 1
            """,
            (*theme_variants, doc_id),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return dict(row) if row else None


def _entry_summary(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": entry.get("id"),
        "username": entry.get("username"),
        "action": entry.get("action"),
        "status": entry.get("status"),
        "timestamp": entry.get("timestamp"),
        "details_json": entry.get("details_json"),
    }


def _history_entries_for_user(entries: List[Dict[str, Any]], username: str | None) -> List[Dict[str, Any]]:
    if not username:
        return []
    return [_entry_summary(entry) for entry in entries if entry.get("username") == username]


def _build_workflow_versions(
    db,
    theme: str,
    doc_id: str,
    entries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    ctx = _workflow_context(db, theme, doc_id)
    if not ctx:
        return []

    source_snapshot = _resolve_source_snapshot(theme, doc_id, str(ctx.get("source_agent") or ""))
    reviewer_a_entries = _history_entries_for_user(entries, ctx.get("reviewer_a_username"))
    reviewer_b_entries = _history_entries_for_user(entries, ctx.get("reviewer_b_username"))

    reviewer_a_output = workflow_service.resolve_workflow_storage_path(ctx.get("reviewer_a_output"))
    reviewer_b_output = workflow_service.resolve_workflow_storage_path(ctx.get("reviewer_b_output"))
    final_snapshot = workflow_service.resolve_workflow_storage_path(ctx.get("final_snapshot_path"))
    packet_path = workflow_service.resolve_workflow_storage_path(ctx.get("packet_path"))

    versions = [
        {
            "key": "opus",
            "label": "Opus",
            "username": str(ctx.get("source_agent") or ""),
            "status": "completed",
            "timestamp": ctx.get("run_created_at"),
            "available": bool(source_snapshot),
            "history_entries": [],
        },
        {
            "key": "annotator_1",
            "label": "Annotator 1",
            "username": ctx.get("reviewer_a_username"),
            "status": str(ctx.get("reviewer_a_status") or "pending"),
            "timestamp": ctx.get("reviewer_a_completed_at"),
            "available": (reviewer_a_output is not None and reviewer_a_output.exists()) or bool(reviewer_a_entries),
            "history_entries": reviewer_a_entries,
        },
        {
            "key": "annotator_2",
            "label": "Annotator 2",
            "username": ctx.get("reviewer_b_username"),
            "status": str(ctx.get("reviewer_b_status") or "pending"),
            "timestamp": ctx.get("reviewer_b_completed_at"),
            "available": (reviewer_b_output is not None and reviewer_b_output.exists()) or bool(reviewer_b_entries),
            "history_entries": reviewer_b_entries,
        },
        {
            "key": "agreement",
            "label": "Agreement",
            "username": ctx.get("resolved_by"),
            "status": str(ctx.get("agreement_status") or "pending"),
            "timestamp": ctx.get("resolved_at"),
            "available": (final_snapshot is not None and final_snapshot.exists()) or (packet_path is not None and packet_path.exists()),
            "history_entries": [],
        },
    ]
    return versions


def _build_review_versions(
    db,
    review_type: str,
    theme: str,
    doc_id: str,
) -> list[Dict[str, Any]]:
    try:
        payload = review_campaign_service.get_admin_review_submission_for_document(
            review_type,
            theme,
            doc_id,
            db=db,
        )
    except FileNotFoundError:
        return []

    submission = payload.get("submission") or {}
    reviewer_a = submission.get("reviewer_a") or {}
    reviewer_b = submission.get("reviewer_b") or {}
    agreement_status = str(submission.get("agreement_status") or "pending")

    return [
        {
            "key": "opus",
            "label": "Opus",
            "username": "Opus",
            "status": "completed",
            "timestamp": None,
            "available": bool(submission.get("source_path")),
            "history_entries": [],
        },
        {
            "key": "annotator_1",
            "label": "First Annotator",
            "username": str(reviewer_a.get("username") or ""),
            "status": "completed" if reviewer_a.get("output_snapshot_path") else "pending",
            "timestamp": reviewer_a.get("submitted_at"),
            "available": bool(reviewer_a.get("output_snapshot_path")),
            "history_entries": [],
        },
        {
            "key": "annotator_2",
            "label": "Second Annotator",
            "username": str(reviewer_b.get("username") or ""),
            "status": "completed" if reviewer_b.get("output_snapshot_path") else "pending",
            "timestamp": reviewer_b.get("submitted_at"),
            "available": bool(reviewer_b.get("output_snapshot_path")),
            "history_entries": [],
        },
        {
            "key": "agreement",
            "label": "Agreement",
            "username": str(submission.get("resolved_by") or ""),
            "status": agreement_status,
            "timestamp": submission.get("resolved_at"),
            "available": bool(submission.get("has_final_snapshot")),
            "history_entries": [],
        },
    ]


def _latest_user_history_snapshot(db, doc_paths: List[str], username: str) -> Dict[str, Any] | None:
    placeholders = ",".join("?" for _ in doc_paths)
    row = db.execute(
        f"""
        SELECT h.id, h.details_json, h.timestamp, h.action, h.status, u.username, h.document_path
        FROM document_history h
        JOIN users u ON u.id = h.user_id
        WHERE h.document_path IN ({placeholders})
          AND u.username = ?
        ORDER BY h.timestamp DESC, h.id DESC
        LIMIT 1
        """,
        (*doc_paths, username),
    ).fetchone()
    if row is None:
        return None
    snapshot_data = _resolve_snapshot_data(db, str(row["document_path"]), int(row["id"]), row["details_json"])
    if not snapshot_data:
        return None
    questions = snapshot_data.get("questions", []) or []
    rules = snapshot_data.get("rules", []) or []
    return {
        "document": snapshot_data.get("document", "") or "",
        "questions": questions,
        "rules": rules,
        "username": row["username"],
        "timestamp": row["timestamp"],
        "action": row["action"],
        "status": row["status"],
        "questions_count": snapshot_data.get("questions_count", len(questions)),
        "rules_count": snapshot_data.get("rules_count", len(rules)),
        "variant_key": None,
        "variant_label": None,
        "content_format": "snapshot",
    }


@router.get("/{theme}/{doc_id}")
def get_document_history(
    theme: str,
    doc_id: str,
    review_type: str | None = Query(default=None),
    user: Dict = Depends(get_current_user),
) -> Dict[str, Any]:
    """Get the history timeline for a document."""
    return build_document_history_payload(theme, doc_id, review_type, db=get_db())


@router.get("/{theme}/{doc_id}/snapshot/{history_id}")
def get_history_snapshot(theme: str, doc_id: str, history_id: int, user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get the document snapshot from a specific history entry."""
    db = get_db()
    canonical_theme = yaml_service.canonical_theme_id(theme)
    doc_paths = _theme_doc_paths(canonical_theme, doc_id)
    placeholders = ",".join("?" for _ in doc_paths)
    row = db.execute(
        f"""SELECT h.details_json, u.username, h.timestamp, h.action, h.status, h.document_path
           FROM document_history h JOIN users u ON h.user_id = u.id
           WHERE h.id = ? AND h.document_path IN ({placeholders})""",
        (history_id, *doc_paths),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="History entry not found")

    snapshot_data = _resolve_snapshot_data(db, str(row["document_path"]), history_id, row["details_json"])
    if not snapshot_data:
        snapshot_data = _resolve_document_fallback(user, canonical_theme, doc_id)
    questions = snapshot_data.get("questions", []) or []
    rules = snapshot_data.get("rules", []) or []

    return {
        "document": snapshot_data.get("document", "") or "",
        "questions": questions,
        "rules": rules,
        "username": row["username"],
        "timestamp": row["timestamp"],
        "action": row["action"],
        "status": row["status"],
        "questions_count": snapshot_data.get("questions_count", len(questions)),
        "rules_count": snapshot_data.get("rules_count", len(rules)),
    }


@router.get("/{theme}/{doc_id}/versions/{version_key}")
def get_annotation_version_snapshot(
    theme: str,
    doc_id: str,
    version_key: str,
    review_type: str | None = Query(default=None),
    user: Dict = Depends(get_current_user),
) -> Dict[str, Any]:
    db = get_db()
    canonical_theme = yaml_service.canonical_theme_id(theme)
    normalized_review_type = str(review_type or "").strip().lower()
    if normalized_review_type in {"rules", "questions"}:
        key_map = {
            "opus": "source",
            "annotator_1": "reviewer_a",
            "annotator_2": "reviewer_b",
            "agreement": "final",
        }
        normalized_key = str(version_key or "").strip().lower()
        variant = key_map.get(normalized_key)
        if not variant:
            raise HTTPException(status_code=400, detail="Unknown version key")
        try:
            payload = review_campaign_service.get_admin_review_submission_content(
                normalized_review_type,
                canonical_theme,
                doc_id,
                variant,
                db=db,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        editable_document = payload.get("editable_document")
        structured = payload.get("structured")
        source = editable_document if isinstance(editable_document, dict) else structured
        if not isinstance(source, dict):
            raise HTTPException(status_code=404, detail="Version not found")

        questions = list(source.get("questions") or [])
        rules = list(source.get("rules") or [])
        labels = {
            "opus": "Opus",
            "annotator_1": "First Annotator",
            "annotator_2": "Second Annotator",
            "agreement": "Agreement",
        }
        history_payload = get_document_history(
            canonical_theme,
            doc_id,
            review_type=normalized_review_type,
            user=user,
        )
        version_meta = next(
            (item for item in history_payload.get("annotation_versions", []) if str(item.get("key")) == normalized_key),
            {},
        )
        return {
            "document": source.get("document_to_annotate", "") or "",
            "questions": questions,
            "rules": rules,
            "username": version_meta.get("username"),
            "timestamp": version_meta.get("timestamp"),
            "action": "review",
            "status": version_meta.get("status") or "pending",
            "questions_count": source.get("num_questions", len(questions)),
            "rules_count": len(rules),
            "variant_key": normalized_key,
            "variant_label": labels.get(normalized_key, normalized_key.title()),
            "content_format": "snapshot",
        }

    ctx = _workflow_context(db, canonical_theme, doc_id)
    if not ctx:
        raise HTTPException(status_code=404, detail="No workflow versions found for this document")

    normalized = str(version_key or "").strip().lower()
    doc_paths = _theme_doc_paths(canonical_theme, doc_id)

    if normalized == "agreement" and user.get("role") and user.get("role") != "power_user":
        raise HTTPException(status_code=403, detail="Agreement packet access is admin-only")

    if normalized == "opus":
        snapshot = _resolve_source_snapshot(canonical_theme, doc_id, str(ctx.get("source_agent") or ""))
        if not snapshot:
            raise HTTPException(status_code=404, detail="Opus version not found")
        return {
            "document": snapshot.get("document", "") or "",
            "questions": snapshot.get("questions", []) or [],
            "rules": snapshot.get("rules", []) or [],
            "username": str(ctx.get("source_agent") or "Opus"),
            "timestamp": ctx.get("run_created_at"),
            "action": "source",
            "status": "completed",
            "questions_count": snapshot.get("questions_count", 0),
            "rules_count": snapshot.get("rules_count", 0),
            "variant_key": "opus",
            "variant_label": "Opus",
            "content_format": "snapshot",
        }

    if normalized in {"annotator_1", "annotator_2"}:
        output_key = "reviewer_a_output" if normalized == "annotator_1" else "reviewer_b_output"
        username_key = "reviewer_a_username" if normalized == "annotator_1" else "reviewer_b_username"
        status_key = "reviewer_a_status" if normalized == "annotator_1" else "reviewer_b_status"
        completed_key = "reviewer_a_completed_at" if normalized == "annotator_1" else "reviewer_b_completed_at"
        label = "Annotator 1" if normalized == "annotator_1" else "Annotator 2"

        output_path = workflow_service.resolve_workflow_storage_path(ctx.get(output_key))
        if output_path is not None and output_path.exists():
            snapshot = _safe_parse_yaml_document(output_path)
            if snapshot:
                return {
                    "document": snapshot.get("document", "") or "",
                    "questions": snapshot.get("questions", []) or [],
                    "rules": snapshot.get("rules", []) or [],
                    "username": ctx.get(username_key),
                    "timestamp": ctx.get(completed_key),
                    "action": "review",
                    "status": str(ctx.get(status_key) or "in_progress"),
                    "questions_count": snapshot.get("questions_count", 0),
                    "rules_count": snapshot.get("rules_count", 0),
                    "variant_key": normalized,
                    "variant_label": label,
                    "content_format": "snapshot",
                }

        fallback_username = str(ctx.get(username_key) or "")
        fallback = _latest_user_history_snapshot(db, doc_paths, fallback_username) if fallback_username else None
        if fallback:
            fallback["variant_key"] = normalized
            fallback["variant_label"] = label
            return fallback
        raise HTTPException(status_code=404, detail=f"{label} version not found")

    if normalized == "agreement":
        final_snapshot_path = workflow_service.resolve_workflow_storage_path(ctx.get("final_snapshot_path"))
        if final_snapshot_path is not None and final_snapshot_path.exists():
            snapshot = _safe_parse_yaml_document(final_snapshot_path)
            if snapshot:
                return {
                    "document": snapshot.get("document", "") or "",
                    "questions": snapshot.get("questions", []) or [],
                    "rules": snapshot.get("rules", []) or [],
                    "username": ctx.get("resolved_by"),
                    "timestamp": ctx.get("resolved_at"),
                    "action": "agreement",
                    "status": str(ctx.get("agreement_status") or "resolved"),
                    "questions_count": snapshot.get("questions_count", 0),
                    "rules_count": snapshot.get("rules_count", 0),
                    "variant_key": "agreement",
                    "variant_label": "Agreement",
                    "content_format": "snapshot",
                }

        packet_content = ""
        packet_path = workflow_service.resolve_workflow_storage_path(ctx.get("packet_path"))
        if packet_path is not None and packet_path.exists():
            packet_content = packet_path.read_text(encoding="utf-8")
        else:
            try:
                packet = workflow_service.get_agreement_packet(int(ctx["run_id"]), canonical_theme, doc_id)
                packet_content = str(packet.get("content") or "")
                packet_path = workflow_service.resolve_workflow_storage_path(packet.get("packet_path")) or packet_path
            except Exception:
                packet_content = ""

        if not packet_content:
            raise HTTPException(status_code=404, detail="Agreement version not found")
        return {
            "document": "",
            "questions": [],
            "rules": [],
            "username": ctx.get("resolved_by"),
            "timestamp": ctx.get("resolved_at"),
            "action": "agreement",
            "status": str(ctx.get("agreement_status") or "pending"),
            "questions_count": 0,
            "rules_count": 0,
            "variant_key": "agreement",
            "variant_label": "Agreement",
            "content_format": "markdown",
            "content": packet_content,
            "packet_path": str(packet_path) if packet_path is not None else "",
        }

    raise HTTPException(status_code=400, detail="Unknown version key")


@router.get("/{theme}/{doc_id}/annotations/metadata")
def get_annotations_metadata(theme: str, doc_id: str, user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get metadata about who created/edited each annotation, question, and rule."""
    return build_annotations_metadata_payload(theme, doc_id, db=get_db())


@router.get("/recent")
def get_recent_activity(
    scope: str = Query(default="documents"),
    user: Dict = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Get recent annotation activity for the selected dashboard scope."""
    _refresh_dashboard_db_snapshot()
    return build_recent_activity_payload(scope, db=get_db())


def record_history(doc_path: str, user_id: int, action: str, status: str, details: str = None) -> int:
    """Record a history entry. Returns the entry id."""
    db = get_db()
    cursor = db.execute(
        """INSERT INTO document_history (document_path, user_id, action, status, details_json)
           VALUES (?, ?, ?, ?, ?)""",
        (doc_path, user_id, action, status, details),
    )
    db.commit()
    sync_db_to_gcs()
    return cursor.lastrowid
