"""Workflow orchestration for blinded parallel dual review + agreement packets."""

from __future__ import annotations

import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from web.services.db import get_db
from web.services.persistence import (
    restore_work_file_from_gcs,
    sync_db_to_gcs,
    sync_work_file_to_gcs,
)
from web.services import yaml_service

WORK_DIR = yaml_service.WORK_DIR
WORKFLOW_ROOT = WORK_DIR / "_workflow_tasks"


def _configured_feedback_resolver_usernames() -> tuple[str, ...]:
    raw = os.environ.get("MEMOREASON_FEEDBACK_RESOLVER_USERNAMES", "")
    return tuple(
        username.strip().lower()
        for username in raw.split(",")
        if username.strip()
    )


ALLOWED_FEEDBACK_RESOLVER_USERNAMES = _configured_feedback_resolver_usernames()
ALLOWED_FEEDBACK_RESPONSE_STATUSES = (
    "accepted",
    "contest_requested",
)


def _normalize_username_key(value: Any) -> str:
    return str(value or "").strip().lower()


def resolver_requires_reviewer_acceptance(resolved_by_username: Any) -> bool:
    return _normalize_username_key(resolved_by_username) in ALLOWED_FEEDBACK_RESOLVER_USERNAMES


def _build_feedback_acceptance_state_map(run_id: int, db) -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows = db.execute(
        """
        SELECT
            a.theme,
            a.doc_id,
            ua.username AS resolved_by,
            t1.assignee_user_id AS reviewer_a_user_id,
            t2.assignee_user_id AS reviewer_b_user_id,
            u1.username AS reviewer_a_username,
            u2.username AS reviewer_b_username,
            ra.response_status AS reviewer_a_response_status,
            rb.response_status AS reviewer_b_response_status
        FROM workflow_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        LEFT JOIN workflow_resolution_responses ra
               ON ra.run_id = a.run_id
              AND ra.theme = a.theme
              AND ra.doc_id = a.doc_id
              AND ra.reviewer_user_id = t1.assignee_user_id
        LEFT JOIN workflow_resolution_responses rb
               ON rb.run_id = a.run_id
              AND rb.theme = a.theme
              AND rb.doc_id = a.doc_id
              AND rb.reviewer_user_id = t2.assignee_user_id
        WHERE a.run_id = ?
          AND a.status = 'resolved'
        """,
        (int(run_id),),
    ).fetchall()

    states: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        theme = yaml_service.canonical_theme_id(str(row["theme"]))
        doc_id = str(row["doc_id"])
        reviewer_a_response = str(row["reviewer_a_response_status"] or "").strip().lower()
        reviewer_b_response = str(row["reviewer_b_response_status"] or "").strip().lower()
        requires_acceptance = resolver_requires_reviewer_acceptance(row["resolved_by"])
        both_accepted = reviewer_a_response == "accepted" and reviewer_b_response == "accepted"
        awaiting_acceptance = bool(requires_acceptance and not both_accepted)
        contested_by: List[str] = []
        if reviewer_a_response == "contest_requested" and row["reviewer_a_username"]:
            contested_by.append(str(row["reviewer_a_username"]))
        if reviewer_b_response == "contest_requested" and row["reviewer_b_username"]:
            contested_by.append(str(row["reviewer_b_username"]))
        states[(theme, doc_id)] = {
            "theme": theme,
            "doc_id": doc_id,
            "resolved_by": str(row["resolved_by"] or ""),
            "requires_reviewer_acceptance": requires_acceptance,
            "reviewer_a_user_id": int(row["reviewer_a_user_id"]) if row["reviewer_a_user_id"] is not None else None,
            "reviewer_b_user_id": int(row["reviewer_b_user_id"]) if row["reviewer_b_user_id"] is not None else None,
            "reviewer_a_username": str(row["reviewer_a_username"] or ""),
            "reviewer_b_username": str(row["reviewer_b_username"] or ""),
            "reviewer_a_response_status": reviewer_a_response or "pending",
            "reviewer_b_response_status": reviewer_b_response or "pending",
            "accepted_count": int(reviewer_a_response == "accepted") + int(reviewer_b_response == "accepted"),
            "is_finalized": not awaiting_acceptance,
            "awaiting_reviewer_acceptance": awaiting_acceptance,
            "contested_by": contested_by,
        }
    return states


def get_run_feedback_acceptance_state_map(run_id: int, db=None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if db is None:
        db = get_db()
    return _build_feedback_acceptance_state_map(int(run_id), db)


def is_agreement_finalized_for_completion(run_id: int, theme: str, doc_id: str, db=None) -> bool:
    states = get_run_feedback_acceptance_state_map(int(run_id), db=db)
    key = (yaml_service.canonical_theme_id(theme), str(doc_id))
    state = states.get(key)
    if state is None:
        return True
    return bool(state.get("is_finalized", True))


def get_active_run_completed_doc_keys(db=None) -> set[tuple[str, str]] | None:
    """Return the document keys considered completed by the Document Annotation dashboard.

    This mirrors the admin monitor logic so downstream campaigns can align on the
    same definition of a "finished" document.
    """
    if db is None:
        db = get_db()

    run = get_active_run(db)
    if run is None:
        return None

    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)
    feedback_state_by_doc = get_run_feedback_acceptance_state_map(run_id, db=db)

    rows = db.execute(
        """
        SELECT theme, doc_id, status
        FROM workflow_agreements
        WHERE run_id = ?
        ORDER BY theme ASC, doc_id ASC
        """,
        (run_id,),
    ).fetchall()

    completed: set[tuple[str, str]] = set()
    for row in rows:
        theme = str(row["theme"] or "")
        doc_id = str(row["doc_id"] or "")
        canonical_key = (yaml_service.canonical_theme_id(theme), doc_id)
        if not _doc_in_catalog(theme, doc_id, catalog):
            continue
        if str(row["status"] or "").strip().lower() != "resolved":
            continue
        feedback_state = feedback_state_by_doc.get(canonical_key)
        if feedback_state and bool(feedback_state.get("awaiting_reviewer_acceptance")):
            continue
        completed.add(canonical_key)
    return completed


def get_active_run_catalog_doc_keys(db=None) -> set[tuple[str, str]] | None:
    """Return the active run catalog as canonical document keys."""
    if db is None:
        db = get_db()

    run = get_active_run(db)
    if run is None:
        return None

    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)
    return {
        (yaml_service.canonical_theme_id(str(theme)), str(doc_id))
        for theme, doc_id in catalog
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_doc_data(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"document": {}}
    if "document" in raw and isinstance(raw["document"], dict):
        return raw
    return {"document": raw}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return _normalize_doc_data(data)


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True, width=10000)
    sync_work_file_to_gcs(path)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    sync_work_file_to_gcs(dst)


def _storage_path_ref(path: Path | str) -> str:
    raw = Path(path)
    try:
        return f"/_workflow_tasks/{raw.relative_to(WORKFLOW_ROOT).as_posix()}"
    except ValueError:
        pass
    try:
        return f"/annotation_workspace/{raw.relative_to(WORK_DIR).as_posix()}"
    except ValueError:
        pass
    return str(raw)


def resolve_workflow_storage_path(raw_path: Any) -> Optional[Path]:
    raw = str(raw_path or "").strip()
    if not raw:
        return None

    workflow_marker = "/_workflow_tasks/"
    if workflow_marker in raw:
        suffix = raw.split(workflow_marker, 1)[1].lstrip("/")
        return WORKFLOW_ROOT / suffix

    workspace_marker = "/annotation_workspace/"
    if workspace_marker in raw:
        suffix = raw.split(workspace_marker, 1)[1].lstrip("/")
        return WORK_DIR / suffix

    legacy_workspace_prefix = "/app/data/diversified_theme_annotations/"
    if raw.startswith(legacy_workspace_prefix):
        suffix = raw[len(legacy_workspace_prefix):].lstrip("/")
        return WORK_DIR / suffix

    direct = Path(raw)
    if direct.exists():
        return direct

    return direct


def _doc_catalog(
    *,
    include_legacy: bool = False,
    include_public_attacks: bool = False,
) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for theme in yaml_service.THEMES:
        theme_id = yaml_service.canonical_theme_id(theme)
        if theme_id == yaml_service.PUBLIC_ATTACKS_THEME and not include_public_attacks:
            continue
        for doc_id in yaml_service.list_theme_doc_ids(theme):
            if doc_id:
                docs.append((theme_id, str(doc_id)))
    if include_legacy and yaml_service._list_legacy_docs():  # type: ignore[attr-defined]
        for doc_id in yaml_service.list_theme_doc_ids(yaml_service.PUBLIC_ATTACKS_THEME):
            if doc_id:
                docs.append((yaml_service.PUBLIC_ATTACKS_THEME, str(doc_id)))
    return sorted(set(docs), key=lambda item: (item[0], item[1]))


def _run_includes_legacy_docs(run_id: int, db) -> bool:
    row = db.execute(
        """
        SELECT 1
        FROM workflow_agreements
        WHERE run_id = ?
          AND theme IN (?, ?)
        LIMIT 1
        """,
        (int(run_id), yaml_service.PUBLIC_ATTACKS_THEME, yaml_service.LEGACY_THEME),
    ).fetchone()
    return row is not None


def _catalog_set_for_run(run_id: int, db) -> set[tuple[str, str]]:
    include_public_attacks = _run_includes_legacy_docs(int(run_id), db)
    return set(
        _doc_catalog(
            include_legacy=include_public_attacks,
            include_public_attacks=include_public_attacks,
        )
    )


def _doc_in_catalog(theme: str, doc_id: str, catalog: set[tuple[str, str]]) -> bool:
    theme_raw = str(theme)
    doc_raw = str(doc_id)
    if (theme_raw, doc_raw) in catalog:
        return True
    return (yaml_service.canonical_theme_id(theme_raw), doc_raw) in catalog


def _coerce_theme_to_theme_id(raw_theme: str) -> str:
    raw = str(raw_theme or "").strip()
    if not raw:
        return ""
    canonical = yaml_service.canonical_theme_id(raw)
    if canonical in yaml_service.ALL_THEMES:
        return canonical
    raw_lower = raw.lower()
    for theme_id, label in yaml_service.THEME_LABELS.items():
        if raw_lower == str(label).lower():
            return str(theme_id)
    return canonical


def _backfill_run_agreements_from_catalog(run_id: int, db) -> int:
    """Insert-only sync: add missing catalog docs to active-run agreements."""
    include_public_attacks = _run_includes_legacy_docs(int(run_id), db)
    catalog = _doc_catalog(
        include_legacy=include_public_attacks,
        include_public_attacks=include_public_attacks,
    )
    if not catalog:
        return 0

    existing_rows = db.execute(
        """
        SELECT theme, doc_id
        FROM workflow_agreements
        WHERE run_id = ?
        """,
        (int(run_id),),
    ).fetchall()
    existing = {
        (yaml_service.canonical_theme_id(str(row["theme"])), str(row["doc_id"]))
        for row in existing_rows
    }

    missing = [(theme, doc_id) for theme, doc_id in catalog if (theme, doc_id) not in existing]
    if not missing:
        return 0

    now = _utc_now()
    inserted = 0
    for theme, doc_id in missing:
        cursor = db.execute(
            """
            INSERT OR IGNORE INTO workflow_agreements (
                run_id, theme, doc_id, status, packet_path,
                reviewer_a_task_id, reviewer_b_task_id, updated_at
            )
            VALUES (?, ?, ?, 'pending', NULL, NULL, NULL, ?)
            """,
            (
                int(run_id),
                theme,
                doc_id,
                now,
            ),
        )
        inserted += int(cursor.rowcount or 0)
    return inserted


def _repair_run_doc_key_drift(run_id: int, db) -> int:
    """Relink stale workflow keys to current catalog keys when snapshot metadata matches.

    This handles safe migrations after theme/doc-id renames by:
    1) reading the stage-1 input snapshot for out-of-catalog agreement rows,
    2) deriving canonical target (theme, doc_id) from snapshot metadata,
    3) moving tasks + agreement row to target key, and
    4) deleting a target placeholder agreement if it is still empty/pending.
    """
    catalog = _catalog_set_for_run(run_id, db)
    if not catalog:
        return 0

    agreement_rows = db.execute(
        """
        SELECT id, theme, doc_id, status, reviewer_a_task_id, reviewer_b_task_id
        FROM workflow_agreements
        WHERE run_id = ?
        ORDER BY id ASC
        """,
        (int(run_id),),
    ).fetchall()

    repaired = 0
    now = _utc_now()

    for row in agreement_rows:
        source_theme = str(row["theme"])
        source_doc_id = str(row["doc_id"])
        source_key = (yaml_service.canonical_theme_id(source_theme), source_doc_id)
        if source_key in catalog:
            continue

        source_snapshot = _stage_input_snapshot_path(int(run_id), 1, source_theme, source_doc_id)
        if not source_snapshot.exists():
            continue
        try:
            source_payload = _load_yaml(source_snapshot)
        except Exception:
            continue

        doc = source_payload.get("document") if isinstance(source_payload, dict) else None
        if not isinstance(doc, dict):
            continue
        target_doc_id = str(doc.get("document_id") or "").strip()
        target_theme_raw = str(doc.get("document_theme") or "").strip()
        if not target_doc_id or not target_theme_raw:
            continue
        target_theme = _coerce_theme_to_theme_id(target_theme_raw)
        target_key = (target_theme, target_doc_id)
        if target_key not in catalog or target_key == source_key:
            continue

        existing_task_count = db.execute(
            """
            SELECT COUNT(*) AS c
            FROM workflow_tasks
            WHERE run_id = ? AND theme = ? AND doc_id = ?
            """,
            (int(run_id), target_theme, target_doc_id),
        ).fetchone()
        if int(existing_task_count["c"] or 0) > 0:
            continue

        target_agreement = db.execute(
            """
            SELECT id, status, reviewer_a_task_id, reviewer_b_task_id
            FROM workflow_agreements
            WHERE run_id = ? AND theme = ? AND doc_id = ?
            LIMIT 1
            """,
            (int(run_id), target_theme, target_doc_id),
        ).fetchone()
        if target_agreement is not None and int(target_agreement["id"]) != int(row["id"]):
            target_has_tasks = bool(
                target_agreement["reviewer_a_task_id"] or target_agreement["reviewer_b_task_id"]
            )
            target_status = str(target_agreement["status"] or "")
            if target_has_tasks or target_status != "pending":
                continue
            db.execute(
                "DELETE FROM workflow_agreements WHERE id = ?",
                (int(target_agreement["id"]),),
            )

        db.execute(
            """
            UPDATE workflow_tasks
            SET theme = ?, doc_id = ?, updated_at = ?
            WHERE run_id = ? AND theme = ? AND doc_id = ?
            """,
            (target_theme, target_doc_id, now, int(run_id), source_theme, source_doc_id),
        )
        db.execute(
            """
            UPDATE workflow_agreements
            SET theme = ?, doc_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (target_theme, target_doc_id, now, int(row["id"])),
        )
        repaired += 1

    return repaired


def _source_doc_path(theme: str, doc_id: str, source_agent: str) -> Path:
    ai_candidate = yaml_service._find_source_agent_copy(theme, doc_id, source_agent)  # type: ignore[attr-defined]
    if ai_candidate is not None:
        return ai_candidate
    source = yaml_service._find_source_doc(theme, doc_id)  # type: ignore[attr-defined]
    if source is None:
        raise FileNotFoundError(f"Source document not found for {theme}/{doc_id}")
    return source


@dataclass(frozen=True)
class Assignment:
    theme: str
    doc_id: str
    stage1_assignee: int
    stage2_assignee: int


def build_assignment_plan(
    docs: Sequence[Tuple[str, str]],
    annotator_user_ids: Sequence[int],
    seed: int,
) -> List[Assignment]:
    """Build deterministic balanced stage assignments for each document."""
    if len(annotator_user_ids) < 2:
        raise ValueError("At least two annotators are required")

    rng = random.Random(seed)
    ordered_docs = list(docs)
    rng.shuffle(ordered_docs)

    annotators = sorted(set(int(uid) for uid in annotator_user_ids))
    stage1_load = {uid: 0 for uid in annotators}
    stage2_load = {uid: 0 for uid in annotators}

    assignments: List[Assignment] = []
    for theme, doc_id in ordered_docs:
        min_stage1 = min(stage1_load.values())
        s1_candidates = [uid for uid in annotators if stage1_load[uid] == min_stage1]
        s1 = rng.choice(s1_candidates)
        stage1_load[s1] += 1

        stage2_candidates_pool = [uid for uid in annotators if uid != s1]
        min_stage2 = min(stage2_load[uid] for uid in stage2_candidates_pool)
        s2_candidates = [uid for uid in stage2_candidates_pool if stage2_load[uid] == min_stage2]
        s2 = rng.choice(s2_candidates)
        stage2_load[s2] += 1

        assignments.append(
            Assignment(
                theme=theme,
                doc_id=doc_id,
                stage1_assignee=s1,
                stage2_assignee=s2,
            )
        )

    assignments.sort(key=lambda item: (item.theme, item.doc_id))
    return assignments


def get_active_run(db=None) -> Optional[Dict[str, Any]]:
    if db is None:
        db = get_db()
    row = db.execute(
        "SELECT * FROM workflow_runs WHERE status = 'active' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is not None:
        return dict(row)

    completed_rows = db.execute(
        """
        SELECT *
        FROM workflow_runs
        WHERE status = 'completed'
        ORDER BY id DESC
        LIMIT 10
        """
    ).fetchall()
    for completed in completed_rows:
        run_id = int(completed["id"])
        feedback_states = _build_feedback_acceptance_state_map(run_id, db)
        has_waiting_acceptance = any(
            bool(state.get("awaiting_reviewer_acceptance"))
            for state in feedback_states.values()
        )
        if not has_waiting_acceptance:
            continue
        db.execute("UPDATE workflow_runs SET status = 'active' WHERE id = ?", (run_id,))
        db.commit()
        sync_db_to_gcs()
        refreshed = db.execute(
            "SELECT * FROM workflow_runs WHERE id = ? LIMIT 1",
            (run_id,),
        ).fetchone()
        return dict(refreshed) if refreshed else None
    return None


def _resolve_user_ids(usernames: Optional[Sequence[str]], db) -> List[int]:
    if usernames:
        placeholders = ",".join("?" for _ in usernames)
        rows = db.execute(
            f"SELECT id, username FROM users WHERE username IN ({placeholders}) AND role = 'regular_user'",
            tuple(usernames),
        ).fetchall()
        found = {row["username"]: row["id"] for row in rows}
        missing = [u for u in usernames if u not in found]
        if missing:
            raise ValueError(f"Unknown regular_user annotators: {', '.join(missing)}")
        return [int(found[u]) for u in usernames]

    rows = db.execute("SELECT id FROM users WHERE role = 'regular_user' ORDER BY id").fetchall()
    return [int(row["id"]) for row in rows]


def _is_user_allowed_in_run(run_id: int, user_id: int, db) -> bool:
    row = db.execute(
        """
        SELECT 1
        FROM workflow_run_annotators
        WHERE run_id = ? AND user_id = ?
        LIMIT 1
        """,
        (int(run_id), int(user_id)),
    ).fetchone()
    return row is not None


def _stage_input_snapshot_path(run_id: int, stage: int, theme: str, doc_id: str) -> Path:
    return WORKFLOW_ROOT / f"run_{run_id}" / f"stage_{stage}" / "input" / theme / f"{doc_id}.yaml"


def _stage_output_snapshot_path(run_id: int, stage: int, assignee_user_id: int, theme: str, doc_id: str) -> Path:
    return (
        WORKFLOW_ROOT
        / f"run_{run_id}"
        / f"stage_{stage}"
        / "work"
        / f"user_{assignee_user_id}"
        / theme
        / f"{doc_id}.yaml"
    )


def _agreement_packet_path(run_id: int, theme: str, doc_id: str) -> Path:
    return WORKFLOW_ROOT / f"run_{run_id}" / "agreements" / theme / f"{doc_id}.md"


def _agreement_final_path(run_id: int, theme: str, doc_id: str) -> Path:
    return WORKFLOW_ROOT / f"run_{run_id}" / "agreements" / "final" / theme / f"{doc_id}.yaml"


def _extract_inline_refs(text: str) -> List[str]:
    pattern = re.compile(r"\[[^\]]+;\s*([a-zA-Z0-9_]+\.[a-zA-Z0-9_\.]+)\]")
    return [match.group(1).strip() for match in pattern.finditer(text or "")]


def _normalize_doc_payload(path: Path) -> Dict[str, Any]:
    loaded = _load_yaml(path).get("document", {})
    if not isinstance(loaded, dict):
        return {}
    return yaml_service.normalize_document_taxonomy(loaded)


def _resolver_username_is_feedback_eligible(username: Any) -> bool:
    return resolver_requires_reviewer_acceptance(username)


def _strip_inline_annotation_markup(text: Any) -> str:
    return re.sub(r"\[([^;\]]+);\s*[^\]]+\]", r"\1", str(text or ""))


def _normalize_rule_list(raw_rules: Any) -> List[str]:
    if not isinstance(raw_rules, list):
        return []
    cleaned: List[str] = []
    for raw_rule in raw_rules:
        value = str(raw_rule or "").strip()
        if value:
            cleaned.append(value)
    return cleaned


def _normalize_question_map(raw_questions: Any) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_questions, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in raw_questions:
        if not isinstance(item, dict):
            continue
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            continue
        raw_answer_type = str(item.get("answer_type") or "").strip().lower()
        if raw_answer_type not in {"variant", "invariant", "refusal"}:
            raw_is_invariant = item.get("is_answer_invariant")
            if raw_is_invariant is True:
                raw_answer_type = "invariant"
            elif raw_is_invariant is False:
                raw_answer_type = "variant"
            else:
                raw_answer_type = "variant"
        out[question_id] = {
            "question_id": question_id,
            "question_type": str(item.get("question_type") or "").strip(),
            "answer_type": raw_answer_type,
            "question": str(item.get("question") or "").strip(),
            "answer": str(item.get("answer") or "").strip(),
        }
    return out


def _build_reviewer_to_final_diff(initial_doc: Dict[str, Any], final_doc: Dict[str, Any]) -> Dict[str, Any]:
    initial_text = str(initial_doc.get("document_to_annotate") or "")
    final_text = str(final_doc.get("document_to_annotate") or "")
    initial_plain_text = _strip_inline_annotation_markup(initial_text).strip()
    final_plain_text = _strip_inline_annotation_markup(final_text).strip()

    initial_refs = sorted(set(_extract_inline_refs(initial_text)))
    final_refs = sorted(set(_extract_inline_refs(final_text)))
    initial_ref_set = set(initial_refs)
    final_ref_set = set(final_refs)
    refs_added = sorted(final_ref_set - initial_ref_set)
    refs_removed = sorted(initial_ref_set - final_ref_set)

    initial_questions = _normalize_question_map(initial_doc.get("questions", []))
    final_questions = _normalize_question_map(final_doc.get("questions", []))
    initial_question_ids = set(initial_questions.keys())
    final_question_ids = set(final_questions.keys())
    question_ids_added = sorted(final_question_ids - initial_question_ids)
    question_ids_removed = sorted(initial_question_ids - final_question_ids)
    questions_added_details = [dict(final_questions[question_id]) for question_id in question_ids_added]
    questions_removed_details = [dict(initial_questions[question_id]) for question_id in question_ids_removed]
    shared_question_ids = sorted(initial_question_ids & final_question_ids)
    questions_changed: List[Dict[str, Any]] = []
    for question_id in shared_question_ids:
        if initial_questions[question_id] == final_questions[question_id]:
            continue
        questions_changed.append(
            {
                "question_id": question_id,
                "initial": initial_questions[question_id],
                "final": final_questions[question_id],
            }
        )

    initial_rules = sorted(set(_normalize_rule_list(initial_doc.get("rules", []))))
    final_rules = sorted(set(_normalize_rule_list(final_doc.get("rules", []))))
    initial_rule_set = set(initial_rules)
    final_rule_set = set(final_rules)
    rules_added = sorted(final_rule_set - initial_rule_set)
    rules_removed = sorted(initial_rule_set - final_rule_set)

    return {
        "document_text_changed": initial_plain_text != final_plain_text,
        "initial_plain_text_preview": initial_plain_text[:400],
        "final_plain_text_preview": final_plain_text[:400],
        "initial_ref_count": len(initial_refs),
        "final_ref_count": len(final_refs),
        "refs_added": refs_added,
        "refs_removed": refs_removed,
        "questions_added": question_ids_added,
        "questions_removed": question_ids_removed,
        "questions_added_details": questions_added_details,
        "questions_removed_details": questions_removed_details,
        "questions_changed": questions_changed,
        "rules_added": rules_added,
        "rules_removed": rules_removed,
        "summary": {
            "refs_added_count": len(refs_added),
            "refs_removed_count": len(refs_removed),
            "questions_added_count": len(question_ids_added),
            "questions_removed_count": len(question_ids_removed),
            "questions_changed_count": len(questions_changed),
            "rules_added_count": len(rules_added),
            "rules_removed_count": len(rules_removed),
        },
    }


def _build_agreement_markdown(
    *,
    run_id: int,
    theme: str,
    doc_id: str,
    reviewer_a: str,
    reviewer_b: str,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    reviewer_a_doc: Dict[str, Any],
    reviewer_b_doc: Dict[str, Any],
) -> str:
    a_text = str(reviewer_a_doc.get("document_to_annotate", "") or "")
    b_text = str(reviewer_b_doc.get("document_to_annotate", "") or "")
    a_refs = sorted(set(_extract_inline_refs(a_text)))
    b_refs = sorted(set(_extract_inline_refs(b_text)))
    only_a = [ref for ref in a_refs if ref not in set(b_refs)]
    only_b = [ref for ref in b_refs if ref not in set(a_refs)]

    a_questions = {
        str(q.get("question_id")): q
        for q in (reviewer_a_doc.get("questions", []) or [])
        if isinstance(q, dict) and q.get("question_id")
    }
    b_questions = {
        str(q.get("question_id")): q
        for q in (reviewer_b_doc.get("questions", []) or [])
        if isinstance(q, dict) and q.get("question_id")
    }
    only_a_q = sorted([qid for qid in a_questions.keys() if qid not in b_questions])
    only_b_q = sorted([qid for qid in b_questions.keys() if qid not in a_questions])
    shared_q = sorted(set(a_questions.keys()) & set(b_questions.keys()))
    changed_q = [
        qid
        for qid in shared_q
        if (a_questions[qid].get("question"), a_questions[qid].get("answer"))
        != (b_questions[qid].get("question"), b_questions[qid].get("answer"))
    ]

    a_rules = [str(r).strip() for r in (reviewer_a_doc.get("rules", []) or []) if str(r).strip()]
    b_rules = [str(r).strip() for r in (reviewer_b_doc.get("rules", []) or []) if str(r).strip()]
    only_a_rules = sorted([rule for rule in set(a_rules) if rule not in set(b_rules)])
    only_b_rules = sorted([rule for rule in set(b_rules) if rule not in set(a_rules)])

    lines: List[str] = []
    lines.append(f"# Agreement Packet: `{theme}/{doc_id}`")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Generated at (UTC): `{_utc_now()}`")
    lines.append(f"- Reviewer A: `{reviewer_a}`")
    lines.append(f"- Reviewer B: `{reviewer_b}`")
    lines.append("")
    lines.append("## Submission Paths")
    lines.append(f"- Reviewer A file: `{reviewer_a_path}`")
    lines.append(f"- Reviewer B file: `{reviewer_b_path}`")
    lines.append("")
    lines.append("## Inline Annotation Summary")
    lines.append(f"- Reviewer A unique refs: **{len(a_refs)}**")
    lines.append(f"- Reviewer B unique refs: **{len(b_refs)}**")
    lines.append(f"- Refs only in Reviewer A: **{len(only_a)}**")
    lines.append(f"- Refs only in Reviewer B: **{len(only_b)}**")
    if only_a:
        lines.append("- Only A refs:")
        for ref in only_a:
            lines.append(f"  - `{ref}`")
    if only_b:
        lines.append("- Only B refs:")
        for ref in only_b:
            lines.append(f"  - `{ref}`")
    lines.append("")
    lines.append("## Questions Summary")
    lines.append(f"- Reviewer A questions: **{len(a_questions)}**")
    lines.append(f"- Reviewer B questions: **{len(b_questions)}**")
    lines.append(f"- Only A question IDs: **{len(only_a_q)}**")
    lines.append(f"- Only B question IDs: **{len(only_b_q)}**")
    lines.append(f"- Shared IDs with different content: **{len(changed_q)}**")
    if only_a_q:
        lines.append("- Only A question IDs:")
        for qid in only_a_q:
            lines.append(f"  - `{qid}`")
    if only_b_q:
        lines.append("- Only B question IDs:")
        for qid in only_b_q:
            lines.append(f"  - `{qid}`")
    if changed_q:
        lines.append("- Shared IDs with changed content:")
        for qid in changed_q:
            lines.append(f"  - `{qid}`")
    lines.append("")
    lines.append("## Rules Summary")
    lines.append(f"- Reviewer A rules: **{len(a_rules)}**")
    lines.append(f"- Reviewer B rules: **{len(b_rules)}**")
    lines.append(f"- Only A rules: **{len(only_a_rules)}**")
    lines.append(f"- Only B rules: **{len(only_b_rules)}**")
    if only_a_rules:
        lines.append("- Only A rules:")
        for rule in only_a_rules:
            lines.append(f"  - `{rule}`")
    if only_b_rules:
        lines.append("- Only B rules:")
        for rule in only_b_rules:
            lines.append(f"  - `{rule}`")
    lines.append("")
    lines.append("## Adjudication Checklist")
    lines.append("- Confirm final inline annotation spans and refs.")
    lines.append("- Confirm final questions/answers and rules.")
    lines.append("- Resolve disagreements in a synchronous reviewer meeting.")
    lines.append("- Mark this packet as `resolved` in the admin panel once consensus is reached.")
    lines.append("")

    return "\n".join(lines)


def _refresh_run_status(run_id: int, db) -> None:
    remaining_tasks = db.execute(
        """
        SELECT COUNT(*) AS c
        FROM workflow_tasks
        WHERE run_id = ?
          AND status != 'completed'
        """,
        (int(run_id),),
    ).fetchone()
    if not remaining_tasks or int(remaining_tasks["c"]) > 0:
        return

    agreement_rows = db.execute(
        """
        SELECT
            SUM(CASE WHEN status = 'resolved' THEN 1 ELSE 0 END) AS resolved_count,
            COUNT(*) AS total_count
        FROM workflow_agreements
        WHERE run_id = ?
        """,
        (int(run_id),),
    ).fetchone()
    if agreement_rows and int(agreement_rows["total_count"] or 0) > 0:
        if int(agreement_rows["resolved_count"] or 0) != int(agreement_rows["total_count"] or 0):
            return

    feedback_states = _build_feedback_acceptance_state_map(int(run_id), db)
    if any(bool(state.get("awaiting_reviewer_acceptance")) for state in feedback_states.values()):
        return

    db.execute("UPDATE workflow_runs SET status = 'completed' WHERE id = ?", (int(run_id),))


def _maybe_generate_agreement_packet(run_id: int, theme: str, doc_id: str, db) -> Optional[Path]:
    task_rows = db.execute(
        """
        SELECT t.id, t.stage, t.status, t.output_snapshot_path, u.username
        FROM workflow_tasks t
        JOIN users u ON u.id = t.assignee_user_id
        WHERE t.run_id = ?
          AND t.theme = ?
          AND t.doc_id = ?
        ORDER BY t.stage ASC, t.id ASC
        """,
        (int(run_id), theme, doc_id),
    ).fetchall()
    if len(task_rows) < 2:
        return None
    if any(row["status"] != "completed" for row in task_rows):
        return None

    row_a = task_rows[0]
    row_b = task_rows[1]
    path_a = resolve_workflow_storage_path(row_a["output_snapshot_path"])
    path_b = resolve_workflow_storage_path(row_b["output_snapshot_path"])
    if path_a is None or path_b is None or not path_a.exists() or not path_b.exists():
        return None

    doc_a = _normalize_doc_payload(path_a)
    doc_b = _normalize_doc_payload(path_b)
    packet_text = _build_agreement_markdown(
        run_id=int(run_id),
        theme=theme,
        doc_id=doc_id,
        reviewer_a=str(row_a["username"]),
        reviewer_b=str(row_b["username"]),
        reviewer_a_path=path_a,
        reviewer_b_path=path_b,
        reviewer_a_doc=doc_a,
        reviewer_b_doc=doc_b,
    )

    packet_path = _agreement_packet_path(int(run_id), theme, doc_id)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(packet_text, encoding="utf-8")
    sync_work_file_to_gcs(packet_path)
    return packet_path


def _force_generate_agreement_packet(run_id: int, theme: str, doc_id: str, db) -> Optional[Path]:
    """Best-effort packet generation even when task statuses are stale.

    This is used as a recovery path for already-ready agreements when packet
    content cannot be loaded.
    """
    task_rows = db.execute(
        """
        SELECT t.id, t.stage, t.output_snapshot_path, u.username
        FROM workflow_tasks t
        JOIN users u ON u.id = t.assignee_user_id
        WHERE t.run_id = ?
          AND t.theme = ?
          AND t.doc_id = ?
        ORDER BY t.stage ASC, t.id ASC
        """,
        (int(run_id), theme, doc_id),
    ).fetchall()
    if len(task_rows) < 2:
        return None

    valid_rows = []
    for row in task_rows:
        raw = row["output_snapshot_path"]
        if isinstance(raw, str) and raw:
            path = resolve_workflow_storage_path(raw)
            if path is not None and path.exists():
                valid_rows.append((row, path))
        if len(valid_rows) == 2:
            break
    if len(valid_rows) < 2:
        return None

    row_a, path_a = valid_rows[0]
    row_b, path_b = valid_rows[1]
    doc_a = _normalize_doc_payload(path_a)
    doc_b = _normalize_doc_payload(path_b)
    packet_text = _build_agreement_markdown(
        run_id=int(run_id),
        theme=theme,
        doc_id=doc_id,
        reviewer_a=str(row_a["username"]),
        reviewer_b=str(row_b["username"]),
        reviewer_a_path=path_a,
        reviewer_b_path=path_b,
        reviewer_a_doc=doc_a,
        reviewer_b_doc=doc_b,
    )

    packet_path = _agreement_packet_path(int(run_id), theme, doc_id)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.write_text(packet_text, encoding="utf-8")
    sync_work_file_to_gcs(packet_path)
    return packet_path


def add_annotator_to_active_run(user_id: int) -> Dict[str, Any]:
    """Add a regular annotator to the active run participant pool."""
    db = get_db()
    run = get_active_run(db)
    if run is None:
        return {"has_active_run": False, "assigned_count": 0}

    user_row = db.execute(
        "SELECT id, username, role FROM users WHERE id = ?",
        (int(user_id),),
    ).fetchone()
    if user_row is None:
        raise ValueError("User not found")
    if user_row["role"] != "regular_user":
        return {
            "has_active_run": True,
            "run_id": int(run["id"]),
            "assigned_count": 0,
            "reason": "non_regular_user",
        }

    existing_member = db.execute(
        """
        SELECT 1
        FROM workflow_run_annotators
        WHERE run_id = ?
          AND user_id = ?
        LIMIT 1
        """,
        (int(run["id"]), int(user_id)),
    ).fetchone()
    if existing_member is None:
        db.execute(
            """
            INSERT INTO workflow_run_annotators (run_id, user_id)
            VALUES (?, ?)
            """,
            (int(run["id"]), int(user_id)),
        )
        db.commit()
        sync_db_to_gcs()

    existing_assignments = db.execute(
        """
        SELECT COUNT(*) AS c
        FROM workflow_tasks
        WHERE run_id = ?
          AND assignee_user_id = ?
        """,
        (int(run["id"]), int(user_id)),
    ).fetchone()
    return {
        "has_active_run": True,
        "run_id": int(run["id"]),
        "assigned_count": int(existing_assignments["c"] if existing_assignments else 0),
    }


def _assign_random_task_for_user(user_id: int, db) -> Optional[Dict[str, Any]]:
    run = get_active_run(db)
    if run is None:
        return None
    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)
    if not catalog:
        return None

    if not _is_user_allowed_in_run(run_id, int(user_id), db):
        return None

    existing_rows = db.execute(
        """
        SELECT t.*, r.status AS run_status
        FROM workflow_tasks t
        JOIN workflow_runs r ON r.id = t.run_id
        WHERE t.run_id = ?
          AND t.assignee_user_id = ?
          AND t.status IN ('available', 'in_progress')
        ORDER BY t.status = 'in_progress' DESC, t.state_entered_at ASC, t.id ASC
        LIMIT 20
        """,
        (run_id, int(user_id)),
    ).fetchall()
    for row in existing_rows:
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog):
            return dict(row)

    now = _utc_now()
    backfilled_agreements = 0
    try:
        db.execute("BEGIN IMMEDIATE")

        existing_again_rows = db.execute(
            """
            SELECT t.*, r.status AS run_status
            FROM workflow_tasks t
            JOIN workflow_runs r ON r.id = t.run_id
            WHERE t.run_id = ?
              AND t.assignee_user_id = ?
              AND t.status IN ('available', 'in_progress')
            ORDER BY t.status = 'in_progress' DESC, t.state_entered_at ASC, t.id ASC
            LIMIT 20
            """,
            (run_id, int(user_id)),
        ).fetchall()
        for row in existing_again_rows:
            if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog):
                db.commit()
                return dict(row)

        backfilled_agreements = _backfill_run_agreements_from_catalog(run_id, db)

        candidates = db.execute(
            """
            SELECT
                a.theme,
                a.doc_id,
                COUNT(t.id) AS assigned_count,
                SUM(CASE WHEN t.assignee_user_id = ? THEN 1 ELSE 0 END) AS already_assigned
            FROM workflow_agreements a
            LEFT JOIN workflow_tasks t
                   ON t.run_id = a.run_id
                  AND t.theme = a.theme
                  AND t.doc_id = a.doc_id
            WHERE a.run_id = ?
              AND a.status = 'pending'
            GROUP BY a.theme, a.doc_id
            HAVING assigned_count < 2
               AND already_assigned = 0
            """,
            (int(user_id), run_id),
        ).fetchall()

        filtered_candidates = [
            row for row in candidates
            if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
        ]
        if not filtered_candidates:
            db.commit()
            if backfilled_agreements > 0:
                sync_db_to_gcs()
            return None

        chosen = random.SystemRandom().choice(filtered_candidates)
        theme = str(chosen["theme"])
        doc_id = str(chosen["doc_id"])

        used_stages = {
            int(row["stage"])
            for row in db.execute(
                """
                SELECT stage
                FROM workflow_tasks
                WHERE run_id = ?
                  AND theme = ?
                  AND doc_id = ?
                """,
                (run_id, theme, doc_id),
            ).fetchall()
        }
        stage = 1 if 1 not in used_stages else 2
        source_path = _source_doc_path(theme, doc_id, str(run["source_agent"]))
        stage_input = _stage_input_snapshot_path(run_id, stage, theme, doc_id)
        _copy_file(source_path, stage_input)

        inserted = db.execute(
            """
            INSERT INTO workflow_tasks (
                run_id, theme, doc_id, stage, assignee_user_id, status,
                predecessor_task_id, input_snapshot_path, output_snapshot_path,
                blocked_reason, assigned_at, available_at, started_at, completed_at,
                state_entered_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'available', NULL, ?, NULL, NULL, ?, ?, NULL, NULL, ?, ?)
            """,
            (
                run_id,
                theme,
                doc_id,
                int(stage),
                int(user_id),
                _storage_path_ref(stage_input),
                now,
                now,
                now,
                now,
            ),
        )
        task_id = int(inserted.lastrowid)

        if int(stage) == 1:
            db.execute(
                """
                UPDATE workflow_agreements
                SET reviewer_a_task_id = COALESCE(reviewer_a_task_id, ?),
                    updated_at = ?
                WHERE run_id = ?
                  AND theme = ?
                  AND doc_id = ?
                """,
                (task_id, now, run_id, theme, doc_id),
            )
        else:
            db.execute(
                """
                UPDATE workflow_agreements
                SET reviewer_b_task_id = COALESCE(reviewer_b_task_id, ?),
                    updated_at = ?
                WHERE run_id = ?
                  AND theme = ?
                  AND doc_id = ?
                """,
                (task_id, now, run_id, theme, doc_id),
            )

        row = db.execute(
            """
            SELECT t.*, r.status AS run_status
            FROM workflow_tasks t
            JOIN workflow_runs r ON r.id = t.run_id
            WHERE t.id = ?
            LIMIT 1
            """,
            (task_id,),
        ).fetchone()
        db.commit()
        sync_db_to_gcs()
        return dict(row) if row else None
    except Exception:
        db.rollback()
        raise


def create_run(
    *,
    name: str,
    source_agent: str,
    seed: int,
    created_by_user_id: int,
    annotator_usernames: Optional[Sequence[str]] = None,
    include_legacy: bool = False,
) -> Dict[str, Any]:
    """Create a blinded independent dual-review workflow run across source documents."""
    db = get_db()
    active = get_active_run(db)
    if active:
        raise ValueError(
            f"Active run already exists (id={active['id']}, name={active['name']}). "
            "Complete or close it before creating a new run."
        )

    annotator_ids = _resolve_user_ids(annotator_usernames, db)
    if len(annotator_ids) < 2:
        raise ValueError("Need at least two regular_user annotators")

    docs = _doc_catalog(
        include_legacy=include_legacy,
        include_public_attacks=include_legacy,
    )
    if not docs:
        raise ValueError("No documents found for workflow creation")

    now = _utc_now()
    cursor = db.execute(
        """
        INSERT INTO workflow_runs (name, source_agent, seed, status, created_by_user_id, created_at)
        VALUES (?, ?, ?, 'active', ?, ?)
        """,
        (name, source_agent, int(seed), int(created_by_user_id), now),
    )
    run_id = int(cursor.lastrowid)

    try:
        for annotator_id in sorted(set(int(uid) for uid in annotator_ids)):
            db.execute(
                """
                INSERT INTO workflow_run_annotators (run_id, user_id)
                VALUES (?, ?)
                """,
                (run_id, annotator_id),
            )

        for theme, doc_id in docs:
            db.execute(
                """
                INSERT INTO workflow_agreements (
                    run_id, theme, doc_id, status, packet_path,
                    reviewer_a_task_id, reviewer_b_task_id, updated_at
                )
                VALUES (?, ?, ?, 'pending', NULL, NULL, NULL, ?)
                """,
                (
                    run_id,
                    theme,
                    doc_id,
                    now,
                ),
            )

        db.commit()
        sync_db_to_gcs()
    except Exception:
        db.rollback()
        raise

    return {
        "run_id": run_id,
        "name": name,
        "source_agent": source_agent,
        "seed": int(seed),
        "documents": len(docs),
        "annotators": len(set(int(uid) for uid in annotator_ids)),
    }


def _active_task_row_for_user_doc(user_id: int, theme: str, doc_id: str, db=None):
    if db is None:
        db = get_db()
    row = db.execute(
        """
        SELECT t.*, r.status AS run_status
        FROM workflow_tasks t
        JOIN workflow_runs r ON r.id = t.run_id
        WHERE r.status = 'active'
          AND t.assignee_user_id = ?
          AND t.theme = ?
          AND t.doc_id = ?
          AND t.status IN ('available', 'in_progress')
        ORDER BY t.stage ASC, t.id ASC
        LIMIT 1
        """,
        (int(user_id), theme, doc_id),
    ).fetchone()
    if row is None:
        return None
    catalog = _catalog_set_for_run(int(row["run_id"]), db)
    if not _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog):
        return None
    return row


def get_task_for_user_document(user_id: int, theme: str, doc_id: str, db=None) -> Optional[Dict[str, Any]]:
    row = _active_task_row_for_user_doc(user_id, theme, doc_id, db=db)
    return dict(row) if row else None


def get_assigned_task_for_user_document(user_id: int, theme: str, doc_id: str, db=None) -> Optional[Dict[str, Any]]:
    if db is None:
        db = get_db()
    row = db.execute(
        """
        SELECT t.*, r.status AS run_status
        FROM workflow_tasks t
        JOIN workflow_runs r ON r.id = t.run_id
        WHERE r.status = 'active'
          AND t.assignee_user_id = ?
          AND t.theme = ?
          AND t.doc_id = ?
        ORDER BY t.stage ASC, t.id ASC
        LIMIT 1
        """,
        (int(user_id), theme, doc_id),
    ).fetchone()
    if row is None:
        return None
    catalog = _catalog_set_for_run(int(row["run_id"]), db)
    if not _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog):
        return None
    return dict(row)


def _ensure_output_snapshot(task: Dict[str, Any], db) -> Path:
    output_path_raw = task.get("output_snapshot_path")
    if isinstance(output_path_raw, str) and output_path_raw:
        output_path = resolve_workflow_storage_path(output_path_raw)
        if output_path is not None and output_path.exists():
            return output_path

    input_snapshot = task.get("input_snapshot_path")
    if not input_snapshot:
        raise FileNotFoundError("Task has no input snapshot")
    input_path = resolve_workflow_storage_path(input_snapshot)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError("Task input snapshot is unavailable")

    output_path = _stage_output_snapshot_path(
        int(task["run_id"]),
        int(task["stage"]),
        int(task["assignee_user_id"]),
        str(task["theme"]),
        str(task["doc_id"]),
    )
    _copy_file(input_path, output_path)

    now = _utc_now()
    db.execute(
        """
        UPDATE workflow_tasks
        SET output_snapshot_path = ?, updated_at = ?
        WHERE id = ?
        """,
        (_storage_path_ref(output_path), now, int(task["id"])),
    )
    db.commit()
    sync_db_to_gcs()
    return output_path


def load_task_document(user_id: int, theme: str, doc_id: str) -> Dict[str, Any]:
    db = get_db()
    row = _active_task_row_for_user_doc(user_id, theme, doc_id, db=db)
    if row is None:
        raise FileNotFoundError("No active workflow task for this document")

    task = dict(row)
    now = _utc_now()
    if task["status"] == "available":
        db.execute(
            """
            UPDATE workflow_tasks
            SET status = 'in_progress',
                started_at = COALESCE(started_at, ?),
                state_entered_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, now, int(task["id"])),
        )
        db.commit()
        sync_db_to_gcs()
        task["status"] = "in_progress"

    output_path = _ensure_output_snapshot(task, db)
    data = _load_yaml(output_path)
    return data.get("document", {})


def save_task_document(user_id: int, theme: str, doc_id: str, document: Dict[str, Any]) -> None:
    db = get_db()
    row = _active_task_row_for_user_doc(user_id, theme, doc_id, db=db)
    if row is None:
        raise FileNotFoundError("No active workflow task for this document")
    task = dict(row)

    output_path = _ensure_output_snapshot(task, db)
    _write_yaml(output_path, {"document": document})

    now = _utc_now()
    db.execute(
        """
        UPDATE workflow_tasks
        SET status = 'in_progress',
            started_at = COALESCE(started_at, ?),
            state_entered_at = CASE WHEN status = 'available' THEN ? ELSE state_entered_at END,
            updated_at = ?
        WHERE id = ?
        """,
        (now, now, now, int(task["id"])),
    )
    db.commit()
    sync_db_to_gcs()


def finish_task(user_id: int, theme: str, doc_id: str) -> Dict[str, Any]:
    db = get_db()
    row = _active_task_row_for_user_doc(user_id, theme, doc_id, db=db)
    if row is None:
        raise FileNotFoundError("No active workflow task for this document")
    task = dict(row)
    _ensure_output_snapshot(task, db)

    now = _utc_now()
    db.execute(
        """
        UPDATE workflow_tasks
        SET status = 'completed',
            completed_at = COALESCE(completed_at, ?),
            state_entered_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (now, now, now, int(task["id"])),
    )

    agreement_ready = False
    packet_path: Optional[Path] = _maybe_generate_agreement_packet(
        int(task["run_id"]),
        str(task["theme"]),
        str(task["doc_id"]),
        db,
    )
    if packet_path is not None:
        db.execute(
            """
            UPDATE workflow_agreements
            SET status = 'ready',
                packet_path = ?,
                updated_at = ?
            WHERE run_id = ?
              AND theme = ?
              AND doc_id = ?
            """,
            (
                _storage_path_ref(packet_path),
                now,
                int(task["run_id"]),
                str(task["theme"]),
                str(task["doc_id"]),
            ),
        )
        agreement_ready = True

    _refresh_run_status(int(task["run_id"]), db)

    db.commit()
    sync_db_to_gcs()
    return {
        "task_id": int(task["id"]),
        "agreement_ready": agreement_ready,
        "packet_path": _storage_path_ref(packet_path) if packet_path else None,
    }


def _set_final_snapshot_from_variant(run_id: int, theme: str, doc_id: str, variant: str, db) -> Optional[str]:
    if variant not in {"reviewer_a", "reviewer_b"}:
        return None
    row = db.execute(
        """
        SELECT
            t1.output_snapshot_path AS reviewer_a_output,
            t2.output_snapshot_path AS reviewer_b_output
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        LIMIT 1
        """,
        (int(run_id), theme, doc_id),
    ).fetchone()
    if row is None:
        return None
    source_path_raw = row["reviewer_a_output"] if variant == "reviewer_a" else row["reviewer_b_output"]
    if not isinstance(source_path_raw, str) or not source_path_raw:
        return None
    source_path = resolve_workflow_storage_path(source_path_raw)
    if source_path is None or not source_path.exists():
        return None
    final_path = _agreement_final_path(int(run_id), theme, doc_id)
    _copy_file(source_path, final_path)
    return _storage_path_ref(final_path)


def get_latest_agreement_record(theme: str, doc_id: str, db=None) -> Optional[Dict[str, Any]]:
    """Return latest agreement row for a document across runs."""
    if db is None:
        db = get_db()
    row = db.execute(
        """
        SELECT run_id, theme, doc_id, status, final_snapshot_path, final_source_label, updated_at
        FROM workflow_agreements
        WHERE theme = ? AND doc_id = ?
        ORDER BY run_id DESC
        LIMIT 1
        """,
        (theme, doc_id),
    ).fetchone()
    return dict(row) if row else None


def load_latest_final_snapshot_document(theme: str, doc_id: str, db=None) -> Optional[Dict[str, Any]]:
    """Load latest agreement final snapshot document payload if available."""
    record = get_latest_agreement_record(theme, doc_id, db=db)
    if not record:
        return None
    raw = record.get("final_snapshot_path")
    if not isinstance(raw, str) or not raw:
        return None
    path = resolve_workflow_storage_path(raw)
    if path is None or not path.exists():
        return None
    restore_work_file_from_gcs(path)
    loaded = _normalize_doc_payload(path)
    return loaded if isinstance(loaded, dict) else None


def save_latest_final_snapshot_from_document(
    theme: str,
    doc_id: str,
    document: Dict[str, Any],
    source_label: str = "admin",
    db=None,
) -> Dict[str, Any]:
    """Persist a document as latest agreement final snapshot (any run status)."""
    owns_db = db is None
    if db is None:
        db = get_db()

    row = db.execute(
        """
        SELECT run_id
        FROM workflow_agreements
        WHERE theme = ? AND doc_id = ?
        ORDER BY run_id DESC
        LIMIT 1
        """,
        (theme, doc_id),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Agreement not found for this document")

    run_id = int(row["run_id"])
    final_path = _agreement_final_path(run_id, theme, doc_id)
    payload = document if isinstance(document, dict) else {}
    _write_yaml(final_path, {"document": payload})

    now = _utc_now()
    label = str(source_label or "admin").strip() or "admin"
    db.execute(
        """
        UPDATE workflow_agreements
        SET final_snapshot_path = ?,
            final_source_label = ?,
            updated_at = ?
        WHERE run_id = ? AND theme = ? AND doc_id = ?
        """,
        (_storage_path_ref(final_path), label, now, run_id, theme, doc_id),
    )
    if owns_db:
        db.commit()
        sync_db_to_gcs()
    return {
        "run_id": run_id,
        "theme": theme,
        "doc_id": doc_id,
        "final_snapshot_path": _storage_path_ref(final_path),
        "final_source_label": label,
    }


def resolve_agreement(
    run_id: int,
    theme: str,
    doc_id: str,
    resolved_by_user_id: int,
    final_variant: Optional[str] = None,
) -> Dict[str, Any]:
    db = get_db()
    row = db.execute(
        """
        SELECT
            a.id,
            a.status,
            a.packet_path,
            a.final_snapshot_path,
            t1.assignee_user_id AS reviewer_a_user_id,
            t2.assignee_user_id AS reviewer_b_user_id
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        """,
        (int(run_id), theme, doc_id),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Agreement packet not found")

    chosen_variant = final_variant
    if chosen_variant not in {"reviewer_a", "reviewer_b"}:
        chosen_variant = "reviewer_a"
    final_snapshot_path = row["final_snapshot_path"]
    if not final_snapshot_path:
        candidate = _set_final_snapshot_from_variant(int(run_id), theme, doc_id, str(chosen_variant), db)
        if candidate:
            final_snapshot_path = candidate

    existing_feedback_rows = db.execute(
        """
        SELECT reviewer_user_id, reviewer_role, response_status
        FROM workflow_resolution_responses
        WHERE run_id = ?
          AND theme = ?
          AND doc_id = ?
        """,
        (int(run_id), theme, doc_id),
    ).fetchall()
    had_contest_request = any(
        str(item["response_status"] or "").strip().lower() == "contest_requested"
        for item in existing_feedback_rows
    )

    now = _utc_now()
    db.execute(
        """
        UPDATE workflow_agreements
        SET status = 'resolved',
            resolved_by_user_id = ?,
            resolved_at = ?,
            final_snapshot_path = COALESCE(final_snapshot_path, ?),
            final_source_label = COALESCE(final_source_label, ?),
            updated_at = ?
        WHERE id = ?
        """,
        (
            int(resolved_by_user_id),
            now,
            final_snapshot_path,
            str(chosen_variant),
            now,
            int(row["id"]),
        ),
    )
    db.execute(
        """
        DELETE FROM workflow_resolution_responses
        WHERE run_id = ? AND theme = ? AND doc_id = ?
        """,
        (int(run_id), theme, doc_id),
    )
    if had_contest_request:
        reviewer_targets = [
            ("reviewer_a", row["reviewer_a_user_id"]),
            ("reviewer_b", row["reviewer_b_user_id"]),
        ]
        for reviewer_role, reviewer_user_id in reviewer_targets:
            if reviewer_user_id is None:
                continue
            db.execute(
                """
                INSERT INTO workflow_resolution_responses (
                    run_id,
                    theme,
                    doc_id,
                    reviewer_user_id,
                    reviewer_role,
                    response_status,
                    responded_at,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, 'accepted', ?, ?, ?)
                ON CONFLICT(run_id, theme, doc_id, reviewer_user_id)
                DO UPDATE SET
                    reviewer_role = excluded.reviewer_role,
                    response_status = excluded.response_status,
                    responded_at = excluded.responded_at,
                    updated_at = excluded.updated_at
                """,
                (
                    int(run_id),
                    theme,
                    doc_id,
                    int(reviewer_user_id),
                    reviewer_role,
                    now,
                    now,
                    now,
                ),
            )

    _refresh_run_status(int(run_id), db)
    feedback_state = get_run_feedback_acceptance_state_map(int(run_id), db=db).get(
        (yaml_service.canonical_theme_id(theme), str(doc_id)),
        {},
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "run_id": int(run_id),
        "theme": theme,
        "doc_id": doc_id,
        "status": "resolved",
        "packet_path": row["packet_path"],
        "final_snapshot_path": final_snapshot_path,
        "final_source_label": str(chosen_variant),
        "requires_reviewer_acceptance": bool(feedback_state.get("requires_reviewer_acceptance", False)),
        "is_finalized": bool(feedback_state.get("is_finalized", True)),
        "awaiting_reviewer_acceptance": bool(feedback_state.get("awaiting_reviewer_acceptance", False)),
        "resolved_after_contestation": bool(had_contest_request),
    }


def get_agreement_packet(run_id: int, theme: str, doc_id: str) -> Dict[str, Any]:
    db = get_db()
    row = db.execute(
        """
        SELECT
            a.status,
            a.packet_path,
            a.resolved_at,
            a.final_snapshot_path,
            a.final_source_label,
            ua.username AS resolved_by
        FROM workflow_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        """,
        (int(run_id), theme, doc_id),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Agreement packet not found")

    packet_path_raw = row["packet_path"]
    content = ""
    if isinstance(packet_path_raw, str) and packet_path_raw:
        packet_path = resolve_workflow_storage_path(packet_path_raw)
        if packet_path is not None and packet_path.exists():
            content = packet_path.read_text(encoding="utf-8")
            packet_path_raw = _storage_path_ref(packet_path)

    if not content:
        regenerated = _maybe_generate_agreement_packet(int(run_id), theme, doc_id, db)
        if regenerated is not None and regenerated.exists():
            content = regenerated.read_text(encoding="utf-8")
            packet_path_raw = _storage_path_ref(regenerated)
            now = _utc_now()
            db.execute(
                """
                UPDATE workflow_agreements
                SET packet_path = ?, updated_at = ?
                WHERE run_id = ? AND theme = ? AND doc_id = ?
                """,
                (packet_path_raw, now, int(run_id), theme, doc_id),
            )
            db.commit()
            sync_db_to_gcs()

    if not content and str(row["status"] or "") in {"ready", "resolved"}:
        forced = _force_generate_agreement_packet(int(run_id), theme, doc_id, db)
        if forced is not None and forced.exists():
            content = forced.read_text(encoding="utf-8")
            packet_path_raw = _storage_path_ref(forced)
            now = _utc_now()
            db.execute(
                """
                UPDATE workflow_agreements
                SET packet_path = ?, updated_at = ?
                WHERE run_id = ? AND theme = ? AND doc_id = ?
                """,
                (packet_path_raw, now, int(run_id), theme, doc_id),
            )
            db.commit()
            sync_db_to_gcs()

    return {
        "run_id": int(run_id),
        "theme": theme,
        "doc_id": doc_id,
        "status": row["status"],
        "packet_path": packet_path_raw,
        "resolved_by": row["resolved_by"],
        "resolved_at": row["resolved_at"],
        "final_snapshot_path": row["final_snapshot_path"],
        "final_source_label": row["final_source_label"],
        "content": content,
    }


def get_user_agreements(user_id: int) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        return {"has_active_run": False, "agreements": []}

    if not _is_user_allowed_in_run(int(run["id"]), int(user_id), db):
        return {
            "has_active_run": True,
            "run": {"id": int(run["id"]), "name": run["name"], "source_agent": run["source_agent"]},
            "agreements": [],
        }

    rows = db.execute(
        """
        SELECT
            a.theme,
            a.doc_id,
            a.status,
            a.packet_path,
            t1.assignee_user_id AS reviewer_a_user_id,
            t2.assignee_user_id AS reviewer_b_user_id,
            u1.username AS reviewer_a,
            u2.username AS reviewer_b,
            ua.username AS resolved_by,
            a.resolved_at
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        WHERE a.run_id = ?
          AND (t1.assignee_user_id = ? OR t2.assignee_user_id = ?)
        ORDER BY
          CASE a.status WHEN 'ready' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
          a.theme ASC,
          a.doc_id ASC
        """,
        (int(run["id"]), int(user_id), int(user_id)),
    ).fetchall()

    agreements = []
    for row in rows:
        my_role = "reviewer_a" if int(row["reviewer_a_user_id"] or -1) == int(user_id) else "reviewer_b"
        agreements.append(
            {
                "theme": row["theme"],
                "doc_id": row["doc_id"],
                "status": row["status"],
                "packet_path": row["packet_path"],
                "reviewer_a": row["reviewer_a"],
                "reviewer_b": row["reviewer_b"],
                "resolved_by": row["resolved_by"],
                "resolved_at": row["resolved_at"],
                "my_role": my_role,
            }
        )

    return {
        "has_active_run": True,
        "run": {"id": int(run["id"]), "name": run["name"], "source_agent": run["source_agent"]},
        "agreements": agreements,
    }


def get_user_resolution_feedback(user_id: int) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        return {"has_active_run": False, "items": []}

    if not _is_user_allowed_in_run(int(run["id"]), int(user_id), db):
        return {
            "has_active_run": True,
            "run": {"id": int(run["id"]), "name": run["name"], "source_agent": run["source_agent"]},
            "items": [],
            "is_participant": False,
        }

    resolver_placeholders = ",".join("?" for _ in ALLOWED_FEEDBACK_RESOLVER_USERNAMES)
    rows = db.execute(
        f"""
        SELECT
            a.run_id,
            a.theme,
            a.doc_id,
            a.resolved_at,
            a.final_snapshot_path,
            ua.username AS resolved_by,
            t1.assignee_user_id AS reviewer_a_user_id,
            t2.assignee_user_id AS reviewer_b_user_id,
            t1.output_snapshot_path AS reviewer_a_output_snapshot_path,
            t2.output_snapshot_path AS reviewer_b_output_snapshot_path,
            u1.username AS reviewer_a_username,
            u2.username AS reviewer_b_username,
            r.response_status,
            r.responded_at
        FROM workflow_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        LEFT JOIN workflow_resolution_responses r
               ON r.run_id = a.run_id
              AND r.theme = a.theme
              AND r.doc_id = a.doc_id
              AND r.reviewer_user_id = ?
        WHERE a.run_id = ?
          AND a.status = 'resolved'
          AND a.final_snapshot_path IS NOT NULL
          AND LOWER(COALESCE(ua.username, '')) IN ({resolver_placeholders})
          AND (t1.assignee_user_id = ? OR t2.assignee_user_id = ?)
        ORDER BY COALESCE(a.resolved_at, '') DESC, a.theme ASC, a.doc_id ASC
        """,
        (
            int(user_id),
            int(run["id"]),
            *ALLOWED_FEEDBACK_RESOLVER_USERNAMES,
            int(user_id),
            int(user_id),
        ),
    ).fetchall()

    items: List[Dict[str, Any]] = []
    feedback_state_by_doc = get_run_feedback_acceptance_state_map(int(run["id"]), db=db)
    for row in rows:
        is_reviewer_a = int(row["reviewer_a_user_id"] or -1) == int(user_id)
        reviewer_role = "reviewer_a" if is_reviewer_a else "reviewer_b"
        reviewer_username = row["reviewer_a_username"] if is_reviewer_a else row["reviewer_b_username"]
        initial_snapshot_raw = (
            row["reviewer_a_output_snapshot_path"] if is_reviewer_a else row["reviewer_b_output_snapshot_path"]
        )
        initial_snapshot_path = resolve_workflow_storage_path(initial_snapshot_raw)
        final_snapshot_path = resolve_workflow_storage_path(row["final_snapshot_path"])

        initial_doc: Dict[str, Any] = {}
        final_doc: Dict[str, Any] = {}
        if initial_snapshot_path is not None and initial_snapshot_path.exists():
            initial_doc = _normalize_doc_payload(initial_snapshot_path)
        if final_snapshot_path is not None and final_snapshot_path.exists():
            final_doc = _normalize_doc_payload(final_snapshot_path)

        diff = _build_reviewer_to_final_diff(initial_doc, final_doc)
        response_status = str(row["response_status"] or "").strip()
        if response_status not in ALLOWED_FEEDBACK_RESPONSE_STATUSES:
            response_status = "pending"
        feedback_state = feedback_state_by_doc.get(
            (yaml_service.canonical_theme_id(str(row["theme"])), str(row["doc_id"])),
            {},
        )

        items.append(
            {
                "run_id": int(row["run_id"]),
                "theme": str(row["theme"]),
                "doc_id": str(row["doc_id"]),
                "reviewer_role": reviewer_role,
                "reviewer_username": str(reviewer_username or ""),
                "reviewer_initial_snapshot_path": (
                    str(initial_snapshot_path) if initial_snapshot_path is not None else str(initial_snapshot_raw or "")
                ),
                "final_snapshot_path": (
                    str(final_snapshot_path) if final_snapshot_path is not None else str(row["final_snapshot_path"] or "")
                ),
                "resolved_by": str(row["resolved_by"] or ""),
                "resolved_at": row["resolved_at"],
                "response_status": response_status,
                "responded_at": row["responded_at"],
                "can_submit_decision": response_status != "accepted",
                "requires_reviewer_acceptance": bool(feedback_state.get("requires_reviewer_acceptance", False)),
                "is_finalized": bool(feedback_state.get("is_finalized", True)),
                "awaiting_reviewer_acceptance": bool(feedback_state.get("awaiting_reviewer_acceptance", False)),
                "initial_document_to_annotate": str(initial_doc.get("document_to_annotate") or ""),
                "final_document_to_annotate": str(final_doc.get("document_to_annotate") or ""),
                "diff": diff,
            }
        )

    return {
        "has_active_run": True,
        "run": {"id": int(run["id"]), "name": run["name"], "source_agent": run["source_agent"]},
        "is_participant": True,
        "items": items,
    }


def submit_user_resolution_feedback_response(
    user_id: int,
    theme: str,
    doc_id: str,
    response_status: str,
) -> Dict[str, Any]:
    decision = str(response_status or "").strip().lower()
    if decision not in ALLOWED_FEEDBACK_RESPONSE_STATUSES:
        raise ValueError("response_status must be one of: accepted, contest_requested")

    db = get_db()
    run = get_active_run(db)
    if run is None:
        raise FileNotFoundError("No active run")
    if not _is_user_allowed_in_run(int(run["id"]), int(user_id), db):
        raise PermissionError("User is not part of the active run")

    resolver_placeholders = ",".join("?" for _ in ALLOWED_FEEDBACK_RESOLVER_USERNAMES)
    row = db.execute(
        f"""
        SELECT
            a.run_id,
            a.theme,
            a.doc_id,
            a.resolved_at,
            t1.assignee_user_id AS reviewer_a_user_id,
            t2.assignee_user_id AS reviewer_b_user_id
        FROM workflow_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
          AND a.status = 'resolved'
          AND a.final_snapshot_path IS NOT NULL
          AND LOWER(COALESCE(ua.username, '')) IN ({resolver_placeholders})
          AND (t1.assignee_user_id = ? OR t2.assignee_user_id = ?)
        LIMIT 1
        """,
        (
            int(run["id"]),
            theme,
            doc_id,
            *ALLOWED_FEEDBACK_RESOLVER_USERNAMES,
            int(user_id),
            int(user_id),
        ),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Resolved feedback item not found for this user/document")

    reviewer_role = "reviewer_a" if int(row["reviewer_a_user_id"] or -1) == int(user_id) else "reviewer_b"
    now = _utc_now()
    db.execute(
        """
        INSERT INTO workflow_resolution_responses (
            run_id,
            theme,
            doc_id,
            reviewer_user_id,
            reviewer_role,
            response_status,
            responded_at,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, theme, doc_id, reviewer_user_id)
        DO UPDATE SET
            reviewer_role = excluded.reviewer_role,
            response_status = excluded.response_status,
            responded_at = excluded.responded_at,
            updated_at = excluded.updated_at
        """,
        (
            int(run["id"]),
            theme,
            doc_id,
            int(user_id),
            reviewer_role,
            decision,
            now,
            now,
            now,
        ),
    )
    _refresh_run_status(int(run["id"]), db)

    state = get_run_feedback_acceptance_state_map(int(run["id"]), db=db).get(
        (yaml_service.canonical_theme_id(theme), str(doc_id)),
        {},
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "run_id": int(run["id"]),
        "theme": theme,
        "doc_id": doc_id,
        "reviewer_user_id": int(user_id),
        "reviewer_role": reviewer_role,
        "response_status": decision,
        "responded_at": now,
        "requires_reviewer_acceptance": bool(state.get("requires_reviewer_acceptance", False)),
        "is_finalized": bool(state.get("is_finalized", True)),
        "awaiting_reviewer_acceptance": bool(state.get("awaiting_reviewer_acceptance", False)),
    }


def resolve_agreement_for_user(
    user_id: int,
    theme: str,
    doc_id: str,
    final_variant: Optional[str] = None,
) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        raise FileNotFoundError("No active run")
    row = db.execute(
        """
        SELECT a.status
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
          AND (t1.assignee_user_id = ? OR t2.assignee_user_id = ?)
        LIMIT 1
        """,
        (int(run["id"]), theme, doc_id, int(user_id), int(user_id)),
    ).fetchone()
    if row is None:
        raise PermissionError("Agreement is not assigned to this user")
    return resolve_agreement(
        int(run["id"]),
        theme,
        doc_id,
        int(user_id),
        final_variant=final_variant,
    )


def get_admin_submissions() -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        return {"has_active_run": False, "submissions": []}
    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)

    inserted = _backfill_run_agreements_from_catalog(run_id, db)
    if inserted > 0:
        db.commit()
        sync_db_to_gcs()

    rows = db.execute(
        """
        SELECT
            a.theme,
            a.doc_id,
            a.status AS agreement_status,
            a.packet_path,
            a.final_snapshot_path,
            a.final_source_label,
            a.resolved_at,
            ua.username AS resolved_by,
            t1.status AS reviewer_a_status,
            t1.output_snapshot_path AS reviewer_a_output,
            t1.completed_at AS reviewer_a_completed_at,
            u1.username AS reviewer_a_username,
            t2.status AS reviewer_b_status,
            t2.output_snapshot_path AS reviewer_b_output,
            t2.completed_at AS reviewer_b_completed_at,
            u2.username AS reviewer_b_username
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        WHERE a.run_id = ?
        ORDER BY
            CASE a.status WHEN 'ready' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
            a.theme ASC,
            a.doc_id ASC
        """,
        (run_id,),
    ).fetchall()
    rows = [
        row for row in rows
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    ]

    submissions = []
    for row in rows:
        source_path = None
        reviewer_a_path = resolve_workflow_storage_path(row["reviewer_a_output"])
        reviewer_b_path = resolve_workflow_storage_path(row["reviewer_b_output"])
        packet_path = resolve_workflow_storage_path(row["packet_path"])
        final_snapshot_path = resolve_workflow_storage_path(row["final_snapshot_path"])
        try:
            source_path = str(_source_doc_path(str(row["theme"]), str(row["doc_id"]), str(run["source_agent"])))
        except Exception:
            source_path = None
        submissions.append(
            {
                "theme": row["theme"],
                "doc_id": row["doc_id"],
                "source_agent": run["source_agent"],
                "source_path": source_path,
                "agreement_status": row["agreement_status"],
                "packet_path": str(packet_path) if packet_path is not None else row["packet_path"],
                "final_snapshot_path": str(final_snapshot_path) if final_snapshot_path is not None else row["final_snapshot_path"],
                "final_source_label": row["final_source_label"],
                "resolved_by": row["resolved_by"],
                "resolved_at": row["resolved_at"],
                "reviewer_a": {
                    "username": row["reviewer_a_username"],
                    "status": row["reviewer_a_status"],
                    "output_snapshot_path": str(reviewer_a_path) if reviewer_a_path is not None else row["reviewer_a_output"],
                    "completed_at": row["reviewer_a_completed_at"],
                },
                "reviewer_b": {
                    "username": row["reviewer_b_username"],
                    "status": row["reviewer_b_status"],
                    "output_snapshot_path": str(reviewer_b_path) if reviewer_b_path is not None else row["reviewer_b_output"],
                    "completed_at": row["reviewer_b_completed_at"],
                },
            }
        )

    return {
        "has_active_run": True,
        "run": {"id": run_id, "name": run["name"], "source_agent": run["source_agent"]},
        "submissions": submissions,
    }


def get_admin_submission_for_document(theme: str, doc_id: str) -> Dict[str, Any]:
    """Return workflow agreement metadata for one document in the active run."""
    db = get_db()
    run = get_active_run(db)
    if run is None:
        raise FileNotFoundError("No active run")
    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)

    inserted = _backfill_run_agreements_from_catalog(run_id, db)
    if inserted > 0:
        db.commit()
        sync_db_to_gcs()

    row = db.execute(
        """
        SELECT
            a.theme,
            a.doc_id,
            a.status AS agreement_status,
            a.packet_path,
            a.final_snapshot_path,
            a.final_source_label,
            a.resolved_at,
            ua.username AS resolved_by,
            t1.status AS reviewer_a_status,
            t1.output_snapshot_path AS reviewer_a_output,
            t1.completed_at AS reviewer_a_completed_at,
            u1.username AS reviewer_a_username,
            t2.status AS reviewer_b_status,
            t2.output_snapshot_path AS reviewer_b_output,
            t2.completed_at AS reviewer_b_completed_at,
            u2.username AS reviewer_b_username
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        LIMIT 1
        """,
        (run_id, theme, doc_id),
    ).fetchone()
    if row is None or not _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog):
        raise FileNotFoundError("Submission not found for this document")

    source_path = None
    reviewer_a_path = resolve_workflow_storage_path(row["reviewer_a_output"])
    reviewer_b_path = resolve_workflow_storage_path(row["reviewer_b_output"])
    packet_path = resolve_workflow_storage_path(row["packet_path"])
    final_snapshot_path = resolve_workflow_storage_path(row["final_snapshot_path"])
    try:
        source_path = str(_source_doc_path(str(row["theme"]), str(row["doc_id"]), str(run["source_agent"])))
    except Exception:
        source_path = None

    submission = {
        "theme": row["theme"],
        "doc_id": row["doc_id"],
        "source_agent": run["source_agent"],
        "source_path": source_path,
        "agreement_status": row["agreement_status"],
        "packet_path": str(packet_path) if packet_path is not None else row["packet_path"],
        "final_snapshot_path": str(final_snapshot_path) if final_snapshot_path is not None else row["final_snapshot_path"],
        "final_source_label": row["final_source_label"],
        "resolved_by": row["resolved_by"],
        "resolved_at": row["resolved_at"],
        "reviewer_a": {
            "username": row["reviewer_a_username"],
            "status": row["reviewer_a_status"],
            "output_snapshot_path": str(reviewer_a_path) if reviewer_a_path is not None else row["reviewer_a_output"],
            "completed_at": row["reviewer_a_completed_at"],
        },
        "reviewer_b": {
            "username": row["reviewer_b_username"],
            "status": row["reviewer_b_status"],
            "output_snapshot_path": str(reviewer_b_path) if reviewer_b_path is not None else row["reviewer_b_output"],
            "completed_at": row["reviewer_b_completed_at"],
        },
    }
    feedback_state = get_run_feedback_acceptance_state_map(run_id, db=db).get(
        (yaml_service.canonical_theme_id(str(theme)), str(doc_id)),
        {},
    )
    submission["requires_reviewer_acceptance"] = bool(feedback_state.get("requires_reviewer_acceptance", False))
    submission["is_finalized"] = bool(feedback_state.get("is_finalized", True))
    submission["awaiting_reviewer_acceptance"] = bool(feedback_state.get("awaiting_reviewer_acceptance", False))
    submission["contested_by"] = list(feedback_state.get("contested_by", []) or [])

    return {
        "has_active_run": True,
        "run": {"id": run_id, "name": run["name"], "source_agent": run["source_agent"]},
        "submission": submission,
    }


def get_admin_submission_content(theme: str, doc_id: str, variant: str) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        raise FileNotFoundError("No active run")
    run_id = int(run["id"])

    meta = db.execute(
        """
        SELECT
            a.packet_path,
            a.final_snapshot_path,
            t1.output_snapshot_path AS reviewer_a_output,
            t2.output_snapshot_path AS reviewer_b_output
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.run_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        LIMIT 1
        """,
        (run_id, theme, doc_id),
    ).fetchone()
    if meta is None:
        raise FileNotFoundError("Submission not found for this document")

    normalized = str(variant or "").strip().lower()
    if normalized == "agreement":
        packet = get_agreement_packet(run_id, theme, doc_id)
        return {
            "variant": "agreement",
            "content": packet.get("content", ""),
            "path": packet.get("packet_path"),
            "structured": None,
            "editable_document": None,
        }

    path: Optional[Path] = None
    if normalized == "source":
        path = _source_doc_path(theme, doc_id, str(run["source_agent"]))
    elif normalized == "reviewer_a":
        path = resolve_workflow_storage_path(meta["reviewer_a_output"])
    elif normalized == "reviewer_b":
        path = resolve_workflow_storage_path(meta["reviewer_b_output"])
    elif normalized == "final":
        path = resolve_workflow_storage_path(meta["final_snapshot_path"])
    else:
        raise ValueError("Unknown submission variant")

    if path is None or not path.exists():
        return {
            "variant": normalized,
            "content": "",
            "path": str(path) if path else None,
            "structured": None,
            "editable_document": None,
        }

    content = path.read_text(encoding="utf-8")
    structured: Optional[Dict[str, Any]] = None
    editable_document: Optional[Dict[str, Any]] = None
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            loaded = yaml.safe_load(content) or {}
            payload = loaded.get("document") if isinstance(loaded, dict) and isinstance(loaded.get("document"), dict) else loaded
            if isinstance(payload, dict):
                normalized_payload = yaml_service.normalize_document_taxonomy(payload)
                editable_document = normalized_payload
                questions = normalized_payload.get("questions", []) or []
                rules = normalized_payload.get("rules", []) or []
                structured = {
                    "document_id": normalized_payload.get("document_id"),
                    "document_theme": normalized_payload.get("document_theme"),
                    "document_to_annotate": normalized_payload.get("document_to_annotate", "") or "",
                    "num_questions": normalized_payload.get("num_questions", len(questions)),
                    "questions": questions,
                    "rules": rules,
                    "annotation_status": normalized_payload.get("annotation_status"),
                    "annotation_errors": normalized_payload.get("annotation_errors", []) or [],
                    "annotated_by": normalized_payload.get("annotated_by"),
                    "annotated_at": normalized_payload.get("annotated_at"),
                }
        except Exception:
            structured = None
            editable_document = None

    return {
        "variant": normalized,
        "content": content,
        "path": str(path),
        "structured": structured,
        "editable_document": editable_document,
    }


def set_admin_final_snapshot(theme: str, doc_id: str, source_variant: str) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        raise FileNotFoundError("No active run")
    run_id = int(run["id"])
    variant = str(source_variant or "").strip().lower()
    if variant not in {"reviewer_a", "reviewer_b"}:
        raise ValueError("source_variant must be 'reviewer_a' or 'reviewer_b'")

    final_snapshot = _set_final_snapshot_from_variant(run_id, theme, doc_id, variant, db)
    if not final_snapshot:
        raise FileNotFoundError("Chosen reviewer submission is not available")

    now = _utc_now()
    db.execute(
        """
        UPDATE workflow_agreements
        SET final_snapshot_path = ?,
            final_source_label = ?,
            updated_at = ?
        WHERE run_id = ? AND theme = ? AND doc_id = ?
        """,
        (final_snapshot, variant, now, run_id, theme, doc_id),
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "run_id": run_id,
        "theme": theme,
        "doc_id": doc_id,
        "final_snapshot_path": final_snapshot,
        "final_source_label": variant,
    }


def set_admin_final_snapshot_from_document(
    theme: str,
    doc_id: str,
    document: Dict[str, Any],
    source_label: str = "admin",
) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        raise FileNotFoundError("No active run")
    run_id = int(run["id"])

    row = db.execute(
        """
        SELECT id
        FROM workflow_agreements
        WHERE run_id = ? AND theme = ? AND doc_id = ?
        LIMIT 1
        """,
        (run_id, theme, doc_id),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Agreement not found for this document")

    final_path = _agreement_final_path(run_id, theme, doc_id)
    payload = document if isinstance(document, dict) else {}
    _write_yaml(final_path, {"document": payload})

    now = _utc_now()
    label = str(source_label or "admin").strip() or "admin"
    db.execute(
        """
        UPDATE workflow_agreements
        SET final_snapshot_path = ?,
            final_source_label = ?,
            updated_at = ?
        WHERE run_id = ? AND theme = ? AND doc_id = ?
        """,
        (_storage_path_ref(final_path), label, now, run_id, theme, doc_id),
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "run_id": run_id,
        "theme": theme,
        "doc_id": doc_id,
        "final_snapshot_path": _storage_path_ref(final_path),
        "final_source_label": label,
    }


def assign_random_task_to_user(user_id: int) -> Dict[str, Any]:
    """Assign one random pending document to a user on explicit request."""
    db = get_db()
    run = get_active_run(db)
    if run is None:
        return {
            "has_active_run": False,
            "is_participant": False,
            "assigned": False,
            "reason": "no_active_run",
            "task": None,
        }

    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)
    user_id = int(user_id)
    if not _is_user_allowed_in_run(run_id, user_id, db):
        return {
            "has_active_run": True,
            "is_participant": False,
            "assigned": False,
            "reason": "not_participant",
            "task": None,
        }

    existing_rows = db.execute(
        """
        SELECT id, theme, doc_id, status, stage
        FROM workflow_tasks
        WHERE run_id = ?
          AND assignee_user_id = ?
          AND status IN ('available', 'in_progress')
        ORDER BY status = 'in_progress' DESC, state_entered_at ASC, id ASC
        LIMIT 20
        """,
        (run_id, user_id),
    ).fetchall()
    existing = next(
        (
            row for row in existing_rows
            if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
        ),
        None,
    )
    if existing is not None:
        return {
            "has_active_run": True,
            "is_participant": True,
            "assigned": False,
            "reason": "already_has_open_task",
            "task": {
                "id": int(existing["id"]),
                "theme": str(existing["theme"]),
                "doc_id": str(existing["doc_id"]),
                "status": str(existing["status"]),
                "stage": int(existing["stage"]),
            },
        }

    task = _assign_random_task_for_user(user_id, db)
    if not task:
        return {
            "has_active_run": True,
            "is_participant": True,
            "assigned": False,
            "reason": "no_documents_available",
            "task": None,
        }

    return {
        "has_active_run": True,
        "is_participant": True,
        "assigned": True,
        "reason": "assigned",
        "task": {
            "id": int(task["id"]),
            "theme": str(task["theme"]),
            "doc_id": str(task["doc_id"]),
            "status": str(task["status"]),
            "stage": int(task["stage"]),
        },
    }


def get_user_queue(user_id: int, *, auto_assign: bool = False) -> Dict[str, Any]:
    db = get_db()
    run = get_active_run(db)
    if run is None:
        return {
            "has_active_run": False,
            "current_tasks": [],
            "assigned_count": 0,
            "remaining_count": 0,
            "upcoming_count": 0,
            "completed_count": 0,
            "agreement_ready_count": 0,
        }

    if not _is_user_allowed_in_run(int(run["id"]), int(user_id), db):
        return {
            "has_active_run": True,
            "run": {
                "id": int(run["id"]),
                "name": run["name"],
                "source_agent": run["source_agent"],
            },
            "current_tasks": [],
            "assigned_count": 0,
            "remaining_count": 0,
            "upcoming_count": 0,
            "completed_count": 0,
            "agreement_ready_count": 0,
            "is_participant": False,
        }

    catalog = _catalog_set_for_run(int(run["id"]), db)

    if auto_assign:
        _assign_random_task_for_user(int(user_id), db)

    current_rows_all = db.execute(
        """
        SELECT t.theme, t.doc_id, t.status, t.stage, t.updated_at
        FROM workflow_tasks t
        WHERE t.run_id = ?
          AND t.assignee_user_id = ?
          AND t.status IN ('available', 'in_progress')
        ORDER BY t.status = 'in_progress' DESC, t.updated_at ASC, t.id ASC
        LIMIT 20
        """,
        (int(run["id"]), int(user_id)),
    ).fetchall()
    current_rows = [
        row for row in current_rows_all
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    ][:1]

    task_rows = db.execute(
        """
        SELECT theme, doc_id, status
        FROM workflow_tasks
        WHERE run_id = ?
          AND assignee_user_id = ?
        """,
        (int(run["id"]), int(user_id)),
    ).fetchall()
    filtered_task_rows = [
        row for row in task_rows
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    ]
    assigned_count = len(filtered_task_rows)
    completed_count = sum(1 for row in filtered_task_rows if str(row["status"]) == "completed")

    agreement_ready_rows = db.execute(
        """
        SELECT a.theme, a.doc_id
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.run_id = ?
          AND a.status = 'ready'
          AND (t1.assignee_user_id = ? OR t2.assignee_user_id = ?)
        """,
        (int(run["id"]), int(user_id), int(user_id)),
    ).fetchall()
    agreement_ready_count = sum(
        1
        for row in agreement_ready_rows
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    )

    remaining_count = max(0, int(assigned_count) - int(completed_count))

    current_tasks = [
        {
            "theme": row["theme"],
            "doc_id": row["doc_id"],
            "status": row["status"],
        }
        for row in current_rows
    ]

    return {
        "has_active_run": True,
        "run": {
            "id": int(run["id"]),
            "name": run["name"],
            "source_agent": run["source_agent"],
        },
        "current_tasks": current_tasks,
        "assigned_count": int(assigned_count),
        "remaining_count": int(remaining_count),
        "upcoming_count": 0,
        "completed_count": int(completed_count),
        "agreement_ready_count": int(agreement_ready_count),
        "is_participant": True,
    }


def get_admin_monitor(db=None) -> Dict[str, Any]:
    if db is None:
        db = get_db()
    run = get_active_run(db)
    if run is None:
        return {"has_active_run": False}

    run_id = int(run["id"])
    catalog = _catalog_set_for_run(run_id, db)
    inserted = _backfill_run_agreements_from_catalog(run_id, db)
    repaired = _repair_run_doc_key_drift(run_id, db)
    if inserted > 0 or repaired > 0:
        db.commit()
        sync_db_to_gcs()

    task_rows = db.execute(
        """
        SELECT assignee_user_id, theme, doc_id, status
        FROM workflow_tasks
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchall()
    filtered_task_rows = [
        row for row in task_rows
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    ]
    status_counts: Dict[str, int] = {}
    for row in filtered_task_rows:
        key = str(row["status"] or "unknown")
        status_counts[key] = status_counts.get(key, 0) + 1

    annotator_rows = db.execute(
        """
        SELECT u.id AS user_id, u.username
        FROM workflow_run_annotators ra
        JOIN users u ON u.id = ra.user_id
        WHERE ra.run_id = ?
        ORDER BY u.username ASC
        """,
        (run_id,),
    ).fetchall()
    progress_by_user: Dict[int, Dict[str, Any]] = {
        int(row["user_id"]): {
            "username": str(row["username"]),
            "assigned_count": 0,
            "submitted_count": 0,
            "active_count": 0,
        }
        for row in annotator_rows
    }
    for row in filtered_task_rows:
        uid = int(row["assignee_user_id"])
        bucket = progress_by_user.get(uid)
        if bucket is None:
            continue
        bucket["assigned_count"] += 1
        if str(row["status"]) == "completed":
            bucket["submitted_count"] += 1
        if str(row["status"]) in {"available", "in_progress"}:
            bucket["active_count"] += 1

    agreement_rows = db.execute(
        """
        SELECT
            a.theme,
            a.doc_id,
            a.status,
            a.packet_path,
            ua.username AS resolved_by,
            a.resolved_at,
            t1.assignee_user_id AS reviewer_a_id,
            t2.assignee_user_id AS reviewer_b_id,
            u1.username AS reviewer_a,
            u2.username AS reviewer_b,
            COALESCE(COUNT(t.id), 0) AS assigned_reviews,
            COALESCE(SUM(CASE WHEN t.status = 'completed' THEN 1 ELSE 0 END), 0) AS completed_reviews
        FROM workflow_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        LEFT JOIN workflow_tasks t
               ON t.run_id = a.run_id
              AND t.theme = a.theme
              AND t.doc_id = a.doc_id
        WHERE a.run_id = ?
        GROUP BY
            a.id, a.theme, a.doc_id, a.status, a.packet_path, ua.username, a.resolved_at,
            t1.assignee_user_id, t2.assignee_user_id, u1.username, u2.username
        ORDER BY a.theme ASC, a.doc_id ASC
        """,
        (run_id,),
    ).fetchall()
    filtered_agreement_rows = [
        row for row in agreement_rows
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    ]

    docs_no_submission = 0
    docs_one_submission = 0
    docs_two_submissions = 0
    for row in filtered_agreement_rows:
        completed_reviews = int(row["completed_reviews"] or 0)
        if completed_reviews <= 0:
            docs_no_submission += 1
        elif completed_reviews == 1:
            docs_one_submission += 1
        else:
            docs_two_submissions += 1

    ready_agreements = [row for row in filtered_agreement_rows if row["status"] == "ready"]
    resolved_agreements = [row for row in filtered_agreement_rows if row["status"] == "resolved"]
    feedback_state_by_doc = get_run_feedback_acceptance_state_map(run_id, db=db)

    resolved_finalized: List[Any] = []
    resolved_awaiting_reviewer_acceptance: List[Any] = []
    for row in resolved_agreements:
        key = (yaml_service.canonical_theme_id(str(row["theme"])), str(row["doc_id"]))
        feedback_state = feedback_state_by_doc.get(key)
        if feedback_state and bool(feedback_state.get("awaiting_reviewer_acceptance")):
            resolved_awaiting_reviewer_acceptance.append(row)
        else:
            resolved_finalized.append(row)

    resolver_placeholders = ",".join("?" for _ in ALLOWED_FEEDBACK_RESOLVER_USERNAMES)
    contest_rows = db.execute(
        f"""
        SELECT
            r.theme,
            r.doc_id,
            r.reviewer_role,
            r.responded_at,
            ru.username AS requester_username,
            ua.username AS resolved_by,
            t1.assignee_user_id AS reviewer_a_id,
            t2.assignee_user_id AS reviewer_b_id,
            u1.username AS reviewer_a,
            u2.username AS reviewer_b
        FROM workflow_resolution_responses r
        JOIN workflow_agreements a
          ON a.run_id = r.run_id
         AND a.theme = r.theme
         AND a.doc_id = r.doc_id
        JOIN users ru ON ru.id = r.reviewer_user_id
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        LEFT JOIN users u1 ON u1.id = t1.assignee_user_id
        LEFT JOIN users u2 ON u2.id = t2.assignee_user_id
        WHERE r.run_id = ?
          AND r.response_status = 'contest_requested'
          AND a.status = 'resolved'
          AND LOWER(COALESCE(ua.username, '')) IN ({resolver_placeholders})
        ORDER BY COALESCE(r.responded_at, '') DESC, r.theme ASC, r.doc_id ASC, ru.username ASC
        """,
        (run_id, *ALLOWED_FEEDBACK_RESOLVER_USERNAMES),
    ).fetchall()
    resolution_contest_requests = [
        {
            "theme": str(row["theme"]),
            "doc_id": str(row["doc_id"]),
            "reviewer_role": str(row["reviewer_role"] or ""),
            "requester_username": str(row["requester_username"] or ""),
            "responded_at": row["responded_at"],
            "resolved_by": str(row["resolved_by"] or ""),
            "reviewer_a": str(row["reviewer_a"] or ""),
            "reviewer_b": str(row["reviewer_b"] or ""),
            "contest_variant": (
                "reviewer_a"
                if str(row["reviewer_role"] or "").strip().lower() == "reviewer_a"
                else (
                    "reviewer_b"
                    if str(row["reviewer_role"] or "").strip().lower() == "reviewer_b"
                    else ""
                )
            ),
            "contestation_note": (
                f"Contested by {str(row['requester_username'] or '')}, "
                f"to be discussed with {str(row['resolved_by'] or 'power user')}"
            ),
        }
        for row in contest_rows
        if _doc_in_catalog(str(row["theme"]), str(row["doc_id"]), catalog)
    ]

    return {
        "has_active_run": True,
        "run": {
            "id": run_id,
            "name": run["name"],
            "source_agent": run["source_agent"],
            "seed": int(run["seed"]),
            "created_at": run["created_at"],
        },
        "status_counts": status_counts,
        "submission_counts": {
            "no_submission": docs_no_submission,
            "one_submission": docs_one_submission,
            "two_submissions": docs_two_submissions,
        },
        "agreements": {
            "pending_count": int(len([row for row in filtered_agreement_rows if row["status"] == "pending"])),
            "ready_count": int(len(ready_agreements)),
            "resolved_count": int(len(resolved_finalized)),
            "awaiting_reviewer_acceptance_count": int(len(resolved_awaiting_reviewer_acceptance)),
        },
        "resolution_feedback": {
            "contest_requested_count": int(len(resolution_contest_requests)),
            "awaiting_reviewer_acceptance_count": int(len(resolved_awaiting_reviewer_acceptance)),
        },
        "annotator_progress": [
            {
                "username": values["username"],
                "assigned_count": int(values["assigned_count"]),
                "submitted_count": int(values["submitted_count"] or 0),
                "remaining_count": int(max(0, int(values["assigned_count"]) - int(values["submitted_count"] or 0))),
                "active_count": int(values["active_count"] or 0),
            }
            for _, values in sorted(progress_by_user.items(), key=lambda item: item[1]["username"].lower())
        ],
        "agreement_ready_details": [
            {
                "theme": row["theme"],
                "doc_id": row["doc_id"],
                "reviewer_a": row["reviewer_a"],
                "reviewer_b": row["reviewer_b"],
                "packet_path": row["packet_path"],
            }
            for row in ready_agreements
        ],
        "agreement_resolved_details": [
            {
                "theme": row["theme"],
                "doc_id": row["doc_id"],
                "reviewer_a": row["reviewer_a"],
                "reviewer_b": row["reviewer_b"],
                "packet_path": row["packet_path"],
                "resolved_by": row["resolved_by"],
                "resolved_at": row["resolved_at"],
            }
            for row in resolved_finalized
        ],
        "agreement_pending_reviewer_acceptance_details": [
            {
                "theme": row["theme"],
                "doc_id": row["doc_id"],
                "reviewer_a": row["reviewer_a"],
                "reviewer_b": row["reviewer_b"],
                "resolved_by": row["resolved_by"],
                "resolved_at": row["resolved_at"],
            }
            for row in resolved_awaiting_reviewer_acceptance
        ],
        "resolution_contest_requests": resolution_contest_requests,
    }
