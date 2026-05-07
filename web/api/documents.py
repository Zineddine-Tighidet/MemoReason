"""Document API endpoints - theme-based with per-user working copies."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Query

from src.core.annotation_runtime import normalize_entity_ref
from web.api.history import (
    build_annotations_metadata_payload,
    build_document_history_payload,
    record_history,
)
from web.middleware.auth import get_current_user, require_power_user
from web.services.db import close_db
from web.services.persistence import restore_db_from_gcs
from web.services import review_campaign_service, workflow_service, yaml_service
from web.services.generation_service import (
    generate_fictional_preview,
    generate_fictional_previews_batch,
)
from web.services.groq_playground_service import (
    groq_is_configured,
    list_groq_models,
    run_groq_document_questions,
)
from web.services.yaml_service import (
    extract_entities,
    get_taxonomy,
    get_theme_progress,
    list_theme_documents,
    list_themes,
    load_document,
    load_document_fresh,
    load_source_document,
    migrate_document,
    normalize_document_taxonomy,
    save_document,
    validate_document,
)

router = APIRouter(prefix="/api/v1")
INLINE_ANNOTATION_PATTERN = re.compile(r"\[[^\]]+;\s*[^\]]+\]")
INLINE_ANNOTATION_CAPTURE_PATTERN = re.compile(r"\[([^\]]+);\s*([^\]]+)\]")
logger = logging.getLogger(__name__)


def _refresh_dashboard_db_snapshot() -> None:
    close_db()
    restore_db_from_gcs(max_age_seconds=5.0)


def _save_latest_final_snapshot_with_refresh_retry(
    theme: str,
    doc_id: str,
    document: dict[str, Any],
    *,
    source_label: str,
) -> dict[str, Any]:
    """Persist the latest final snapshot, retrying once after remote DB drift."""
    try:
        return workflow_service.save_latest_final_snapshot_from_document(
            theme,
            doc_id,
            document,
            source_label=source_label,
        )
    except RuntimeError as exc:
        message = str(exc or "")
        if "Remote DB changed since the last pull" not in message:
            raise
        logger.warning(
            "Final snapshot sync drift detected during document save for %s/%s; refreshing DB and retrying once.",
            theme,
            doc_id,
            exc_info=True,
        )
        close_db()
        restore_db_from_gcs(max_age_seconds=0, force_download=True)
        return workflow_service.save_latest_final_snapshot_from_document(
            theme,
            doc_id,
            document,
            source_label=source_label,
        )


def _sanitize_for_blind_review(doc_data: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Hide provenance markers for regular annotators."""
    if user.get("role") == "power_user":
        return doc_data
    sanitized = dict(doc_data)
    for key in ("annotated_by", "annotated_at", "decision_logs", "decision_log"):
        sanitized.pop(key, None)
    return sanitized


def _require_workflow_access_if_needed(
    user: dict[str, Any],
    theme: str,
    doc_id: str,
    review_target: str | None = None,
) -> dict[str, Any] | None:
    """When a workflow run is active, regular users can only access assigned tasks."""
    if user.get("role") != "regular_user":
        return None
    normalized_review_target = str(review_target or "").strip().lower()
    if normalized_review_target in {"rules", "questions"}:
        # Review workspaces are controlled by review-campaign task assignment, not
        # document-annotation workflow task state.
        return None
    active_run = workflow_service.get_active_run()
    if not active_run:
        return None
    assigned = workflow_service.get_assigned_task_for_user_document(int(user["id"]), theme, doc_id)
    if assigned is None:
        raise HTTPException(status_code=403, detail="This document is not in your current queue")
    status = assigned.get("status")
    if status == "blocked":
        raise HTTPException(status_code=423, detail="This task is not yet available")
    if status == "completed":
        raise HTTPException(status_code=409, detail="This task is already completed")
    return assigned


def _build_history_snapshot(doc_data: dict[str, Any], username: str) -> str:
    """Build a serialized history snapshot from current document state."""
    questions = doc_data.get("questions", []) or []
    rules = doc_data.get("rules", []) or []
    snapshot = {
        "document": doc_data.get("document_to_annotate", "") or "",
        "questions": questions,
        "rules": rules,
        "questions_count": len(questions),
        "rules_count": len(rules),
        "username": username,
        "timestamp": None,  # Set by database
    }
    return json.dumps(snapshot)


def _strip_runtime_fields(doc_data: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(doc_data or {})
    for key in (
        "review_statuses",
        "active_review_target",
        "active_review_task_status",
        "active_review_campaign_name",
    ):
        cleaned.pop(key, None)
    return cleaned


def _attach_review_statuses(doc_data: dict[str, Any], theme: str, doc_id: str) -> dict[str, Any]:
    enriched = dict(doc_data or {})
    enriched["review_statuses"] = review_campaign_service.get_document_review_statuses(theme, doc_id)
    return enriched


def _load_resolved_final_document_if_available(theme: str, doc_id: str) -> dict[str, Any] | None:
    agreement = workflow_service.get_latest_agreement_record(theme, doc_id)
    if not agreement or str(agreement.get("status") or "").strip().lower() != "resolved":
        return None
    final_doc = workflow_service.load_latest_final_snapshot_document(theme, doc_id)
    if not isinstance(final_doc, dict) or not final_doc:
        return None
    return final_doc


def _load_reference_document(
    theme: str,
    doc_id: str,
    review_target: str | None,
    reference_reviewer: str | None,
    user: dict[str, Any],
) -> dict[str, Any]:
    """Load a read-only reference document that bypasses assignment checks.

    This is used for documentation/example links where regular users should be
    able to inspect a fully prepared document without entering an annotation
    queue or modifying the underlying review artifacts.
    """
    normalized_review_target = str(review_target or "").strip().lower()
    normalized_reference_reviewer = str(reference_reviewer or "").strip()

    if normalized_review_target in {"rules", "questions"}:
        try:
            if normalized_review_target == "questions" and normalized_reference_reviewer:
                doc_data = review_campaign_service.load_review_submission_document(
                    normalized_review_target,
                    theme,
                    doc_id,
                    normalized_reference_reviewer,
                )
            else:
                doc_data = review_campaign_service.load_admin_review_document(
                    normalized_review_target,
                    theme,
                    doc_id,
                )
        except FileNotFoundError:
            final_doc = _load_resolved_final_document_if_available(theme, doc_id)
            if final_doc is not None:
                doc_data = final_doc
            else:
                doc_data = load_document("anonymous_reference", theme, doc_id)
    else:
        final_doc = _load_resolved_final_document_if_available(theme, doc_id)
        if final_doc is not None:
            doc_data = final_doc
        else:
            try:
                doc_data = load_document("anonymous_reference", theme, doc_id)
            except FileNotFoundError:
                doc_data = load_source_document(theme, doc_id)

    return _sanitize_for_blind_review(_attach_review_statuses(doc_data, theme, doc_id), user)


def _load_canonical_factual_source_document(theme: str, doc_id: str) -> dict[str, Any] | None:
    """Best-effort factual source lookup that bypasses ANNOTATION_SOURCE_DIR overrides.

    This protects dashboard/power-user views when the active source directory was
    accidentally populated with fictionalized `document_to_annotate` text.
    """
    canonical_theme = yaml_service.canonical_theme_id(theme)
    candidate_paths: list[Path] = []

    if canonical_theme == yaml_service.PUBLIC_ATTACKS_THEME:
        candidate_paths.append(
            yaml_service.PROJECT_ROOT
            / "data"
            / "WikiEvent"
            / "public_attacks_news_articles"
            / f"{doc_id}.yaml"
        )
        candidate_paths.append(
            yaml_service.PROJECT_ROOT
            / "data"
            / "Wikipedia"
            / yaml_service.PUBLIC_ATTACKS_THEME
            / f"{doc_id}.yaml"
        )
        candidate_paths.append(
            yaml_service.PROJECT_ROOT
            / "data"
            / "Wikipedia"
            / yaml_service.LEGACY_THEME
            / f"{doc_id}.yaml"
        )
    else:
        candidate_paths.append(
            yaml_service.PROJECT_ROOT
            / "data"
            / "Wikipedia"
            / canonical_theme
            / f"{doc_id}.yaml"
        )

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        document = payload.get("document")
        if isinstance(document, dict):
            return document
    return None


def _coerce_factual_document_surface(doc_data: dict[str, Any], theme: str, doc_id: str) -> dict[str, Any]:
    """Ensure non-review dashboard/editor views render factual annotated text.

    Important: do not clobber persisted editable admin/final snapshots.
    Only coerce when the currently loaded surface is clearly a fallback
    (fictionalized surface or raw original plain text).
    """
    if not isinstance(doc_data, dict):
        return doc_data
    normalized = dict(doc_data)
    current_text = str(normalized.get("document_to_annotate") or "")
    current_stripped = current_text.strip()
    fictionalized = str(normalized.get("fictionalized_annotated_template_document") or "").strip()
    original_plain = str(normalized.get("original_document") or "").strip()

    def _annotation_surface_map(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for match in INLINE_ANNOTATION_CAPTURE_PATTERN.finditer(text or ""):
            surface = str(match.group(1) or "").strip()
            raw_ref = str(match.group(2) or "").strip()
            normalized_ref = normalize_entity_ref(raw_ref)
            if not normalized_ref or not surface:
                continue
            if normalized_ref not in out:
                out[normalized_ref] = surface
        return out

    def _rewrite_annotation_surfaces(text: str, factual_surface_by_ref: dict[str, str]) -> str:
        if not text or not factual_surface_by_ref:
            return text

        def _replace(match: re.Match[str]) -> str:
            original_surface = str(match.group(1) or "").strip()
            raw_ref = str(match.group(2) or "").strip()
            normalized_ref = normalize_entity_ref(raw_ref)
            factual_surface = factual_surface_by_ref.get(normalized_ref, original_surface)
            return f"[{factual_surface}; {raw_ref}]"

        return INLINE_ANNOTATION_CAPTURE_PATTERN.sub(_replace, text)

    should_coerce = False
    if not current_stripped:
        should_coerce = True
    elif fictionalized and current_stripped == fictionalized:
        # Loaded surface is the fictionalized one: switch to factual.
        should_coerce = True
    elif original_plain and current_stripped == original_plain:
        # Loaded surface is raw original text (non-annotated): switch to factual annotated.
        should_coerce = True
    elif not INLINE_ANNOTATION_PATTERN.search(current_text):
        # No inline annotations is almost always a fallback/plain payload.
        should_coerce = True

    def _is_semantic_ref(entity_ref: str) -> bool:
        ref = str(entity_ref or "").strip().lower()
        if not ref:
            return False
        if ref.startswith(("number_", "temporal_")):
            return False
        if ref.startswith("person_") and (
            ref.endswith(".subj_pronoun")
            or ref.endswith(".obj_pronoun")
            or ref.endswith(".poss_det_pronoun")
            or ref.endswith(".poss_pronoun")
            or ref.endswith(".reflex_pronoun")
        ):
            return False
        return True

    def _factual_surface_score(surface_by_ref: dict[str, str], base_text: str) -> tuple[int, int]:
        if not surface_by_ref or not base_text:
            return (0, 0)
        semantic_refs = [ref for ref in surface_by_ref if _is_semantic_ref(ref)]
        if not semantic_refs:
            semantic_refs = list(surface_by_ref.keys())
        in_text = sum(
            1
            for ref in semantic_refs
            if str(surface_by_ref.get(ref) or "").strip()
            and str(surface_by_ref.get(ref) or "").strip() in base_text
        )
        return (in_text, len(semantic_refs))

    # For editor rendering we must keep inline annotation spans. Prefer a factual
    # source candidate selected from:
    # 1) configured source dir (may be overridden in production)
    # 2) canonical data/Wikipedia fallback (always factual when present)
    factual_text = ""
    source_surface_by_ref: dict[str, str] = {}
    source_candidates: list[dict[str, Any]] = []
    try:
        source_doc = load_source_document(theme, doc_id)
        if isinstance(source_doc, dict):
            source_candidates.append(source_doc)
    except Exception:
        pass
    try:
        canonical_source_doc = _load_canonical_factual_source_document(theme, doc_id)
        if isinstance(canonical_source_doc, dict):
            source_candidates.append(canonical_source_doc)
    except Exception:
        pass

    ranked_candidates: list[tuple[tuple[int, int, int], str, dict[str, str]]] = []
    for candidate in source_candidates:
        candidate_text = str(candidate.get("document_to_annotate") or "").strip()
        if not candidate_text:
            continue
        candidate_surface_by_ref = _annotation_surface_map(candidate_text)
        score_in_original, semantic_ref_count = _factual_surface_score(candidate_surface_by_ref, original_plain)
        # Penalize candidates that match the known fictionalized variant.
        non_fictional_bonus = 0 if (fictionalized and candidate_text == fictionalized) else 1
        ranked_candidates.append(
            ((score_in_original, semantic_ref_count, non_fictional_bonus), candidate_text, candidate_surface_by_ref)
        )

    if ranked_candidates:
        ranked_candidates.sort(key=lambda item: item[0], reverse=True)
        _rank, factual_text, source_surface_by_ref = ranked_candidates[0]
    else:
        factual_text = ""
        source_surface_by_ref = {}

    current_has_annotations = bool(INLINE_ANNOTATION_PATTERN.search(current_text))
    if source_surface_by_ref and current_has_annotations:
        current_surface_by_ref = _annotation_surface_map(current_text)
        shared_refs = set(current_surface_by_ref).intersection(source_surface_by_ref)
        has_surface_mismatch = any(
            (current_surface_by_ref.get(ref) or "").strip() != (source_surface_by_ref.get(ref) or "").strip()
            for ref in shared_refs
        )
        # Always remap inline annotation surfaces to factual values in
        # dashboard/editor (non-review) views so power users never see
        # fictionalized entity values there.
        if should_coerce or has_surface_mismatch:
            normalized["document_to_annotate"] = _rewrite_annotation_surfaces(
                current_text,
                source_surface_by_ref,
            )
            return normalized

    if not factual_text and source_candidates:
        fallback_plain = str(source_candidates[0].get("original_document") or "").strip()
        factual_text = fallback_plain

    if not should_coerce:
        return normalized

    if not factual_text:
        # Keep current payload as a final fallback.
        factual_text = current_stripped

    if factual_text:
        normalized["document_to_annotate"] = factual_text
    return normalized


# --- Themes ---

@router.get("/themes")
def api_list_themes(user: dict = Depends(get_current_user)) -> list[dict[str, Any]]:
    return list_themes()


@router.get("/themes/{theme}/documents")
def api_list_theme_documents(theme: str, user: dict = Depends(get_current_user)) -> list[dict[str, Any]]:
    return list_theme_documents(theme)


# --- Documents ---

@router.get("/documents/{theme}/{doc_id}")
def api_load_document(
    theme: str,
    doc_id: str,
    review_target: str | None = Query(default=None),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Load a user's working copy (auto-creates from source if needed)."""
    _require_workflow_access_if_needed(user, theme, doc_id, review_target)
    try:
        normalized_review_target = str(review_target or "").strip().lower()
        if user.get("role") == "regular_user" and normalized_review_target in {"rules", "questions"}:
            doc_data = review_campaign_service.load_review_task_document(
                normalized_review_target,
                int(user["id"]),
                theme,
                doc_id,
            )
            return _sanitize_for_blind_review(_attach_review_statuses(doc_data, theme, doc_id), user)
        if user.get("role") == "power_user" and normalized_review_target in {"rules", "questions"}:
            doc_data = review_campaign_service.load_admin_review_document(normalized_review_target, theme, doc_id)
            return _sanitize_for_blind_review(_attach_review_statuses(doc_data, theme, doc_id), user)
        workflow_task = None
        if user.get("role") == "regular_user":
            workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            prefer_power_user_work_copy = (
                user.get("role") == "power_user"
                and normalized_review_target not in {"rules", "questions"}
            )
            if prefer_power_user_work_copy:
                doc_data = load_document_fresh(user["username"], theme, doc_id)
            else:
                final_doc = _load_resolved_final_document_if_available(theme, doc_id)
                if final_doc is not None:
                    doc_data = final_doc
                else:
                    doc_data = load_document(user["username"], theme, doc_id)
        if user.get("role") == "power_user" and normalized_review_target not in {"rules", "questions"}:
            doc_data = _coerce_factual_document_surface(doc_data, theme, doc_id)
        return _sanitize_for_blind_review(_attach_review_statuses(doc_data, theme, doc_id), user)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/documents/{theme}/{doc_id}/bootstrap")
def api_load_document_bootstrap(
    theme: str,
    doc_id: str,
    review_target: str | None = Query(default=None),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Load the editor bootstrap payload in one round trip."""
    document = api_load_document(theme, doc_id, review_target, user)
    normalized_review_target = str(review_target or "").strip().lower()
    is_power_user = user.get("role") == "power_user"

    history_payload = (
        build_document_history_payload(theme, doc_id, normalized_review_target or None)
        if is_power_user
        else {"entries": [], "annotation_versions": [], "current_status": "draft", "last_editor": None}
    )
    metadata_payload = (
        build_annotations_metadata_payload(theme, doc_id)
        if is_power_user and not normalized_review_target
        else {"annotations": {}, "questions": {}, "rules": {}, "has_history": False}
    )

    return {
        "document": document,
        "taxonomy": get_taxonomy(),
        "history": history_payload,
        "metadata": metadata_payload,
    }


@router.get("/documents/{theme}/{doc_id}/reference-bootstrap")
def api_load_reference_document_bootstrap(
    theme: str,
    doc_id: str,
    review_target: str | None = Query(default=None),
    reference_reviewer: str | None = Query(default=None),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    """Load a read-only reference document for documentation/examples."""
    try:
        document = _load_reference_document(theme, doc_id, review_target, reference_reviewer, user)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "document": document,
        "taxonomy": get_taxonomy(),
        "history": {"entries": [], "annotation_versions": [], "current_status": "draft", "last_editor": None},
        "metadata": {"annotations": {}, "questions": {}, "rules": {}, "has_history": False},
        "reference_mode": True,
    }


@router.get("/documents/{theme}/{doc_id}/source")
def api_load_source(theme: str, doc_id: str, user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Load the original source document (read-only)."""
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        doc_data = load_source_document(theme, doc_id)
        return _sanitize_for_blind_review(_attach_review_statuses(doc_data, theme, doc_id), user)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/documents/{theme}/{doc_id}")
def api_save_document(
    theme: str,
    doc_id: str,
    doc_data: dict[str, Any],
    review_target: str | None = Query(default=None),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    _require_workflow_access_if_needed(user, theme, doc_id, review_target)
    try:
        is_power_user = user.get("role") == "power_user"
        normalized_review_target = str(review_target or "").strip().lower()
        if user.get("role") == "regular_user" and normalized_review_target in {"rules", "questions"}:
            review_campaign_service.save_review_task_document(
                normalized_review_target,
                int(user["id"]),
                theme,
                doc_id,
                _strip_runtime_fields(doc_data),
            )
            return {"status": "saved"}
        workflow_task = None
        if user.get("role") == "regular_user":
            workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        agreement = workflow_service.get_latest_agreement_record(theme, doc_id) if is_power_user else None
        agreement_is_finalized = bool(
            agreement
            and int(agreement.get("run_id") or 0) > 0
            and workflow_service.is_agreement_finalized_for_completion(
                int(agreement.get("run_id")),
                theme,
                doc_id,
            )
        )
        agreement_resolved = bool(
            agreement
            and str(agreement.get("status") or "").lower() == "resolved"
            and agreement_is_finalized
        )
        agreement_in_progress = bool(
            agreement and (
                str(agreement.get("status") or "").lower() in {"pending", "ready"}
                or (
                    str(agreement.get("status") or "").lower() == "resolved"
                    and not agreement_is_finalized
                )
            )
        )
        clean_doc_data = review_campaign_service._canonicalize_mergeable_equality_rules(
            _strip_runtime_fields(doc_data)
        )
        if is_power_user and normalized_review_target in {"rules", "questions"}:
            return review_campaign_service.save_admin_review_document(
                normalized_review_target,
                theme,
                doc_id,
                clean_doc_data,
            )
        review_source_sync: dict[str, Any] | None = None
        save_warnings: list[str] = []
        if workflow_task is not None:
            workflow_service.save_task_document(int(user["id"]), theme, doc_id, clean_doc_data)
        else:
            if is_power_user and agreement_in_progress:
                _save_latest_final_snapshot_with_refresh_retry(
                    theme,
                    doc_id,
                    clean_doc_data,
                    source_label="admin_draft",
                )
            elif is_power_user and agreement_resolved:
                _save_latest_final_snapshot_with_refresh_retry(
                    theme,
                    doc_id,
                    clean_doc_data,
                    source_label="admin_editor",
                )
            # In power-user Document/Rules editing, keep annotation validation but do not
            # block save on legacy QA scope drift unrelated to the current edit target.
            save_document(
                user["username"],
                theme,
                doc_id,
                clean_doc_data,
                validate_question_scope=not (
                    is_power_user and normalized_review_target != "questions"
                ),
            )
            if is_power_user and not normalized_review_target:
                try:
                    review_source_sync = review_campaign_service.sync_document_annotations_to_review_sources(
                        theme=theme,
                        doc_id=doc_id,
                        document=clean_doc_data,
                    )
                except Exception:
                    logger.exception(
                        "Document save succeeded but review-source sync failed for %s/%s",
                        theme,
                        doc_id,
                    )
                    save_warnings.append("review_source_sync_failed")
        # Record edit in history with complete snapshot
        doc_path = f"{theme}/{doc_id}"
        history_status = "completed" if agreement_resolved else "in_progress"
        try:
            record_history(
                doc_path,
                user["id"],
                "edit",
                history_status,
                _build_history_snapshot(clean_doc_data, user["username"]),
            )
        except Exception:
            logger.exception(
                "Document save succeeded but history recording failed for %s/%s",
                theme,
                doc_id,
            )
            save_warnings.append("history_record_failed")
        response: dict[str, Any] = {"status": "saved"}
        if isinstance(review_source_sync, dict):
            response["review_source_sync"] = review_source_sync
        if save_warnings:
            response["warnings"] = save_warnings
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/documents/{theme}/{doc_id}/implicit-rules")
def api_regenerate_implicit_rules(
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        raw_doc_data = body.get("doc_data")
        if isinstance(raw_doc_data, dict):
            preview_doc_data = _strip_runtime_fields(raw_doc_data)
        else:
            workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
            if workflow_task is not None:
                preview_doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
            else:
                preview_doc_data = load_document(user["username"], theme, doc_id)

        if bool(body.get("reset_exclusions", True)):
            preview_doc_data.pop("implicit_rule_exclusions", None)

        normalized_doc_data = normalize_document_taxonomy(preview_doc_data)
        implicit_rules = normalized_doc_data.get("implicit_rules", []) or []
        implicit_rule_exclusions = normalized_doc_data.get("implicit_rule_exclusions", []) or []
        return {
            "implicit_rules": implicit_rules,
            "implicit_rule_exclusions": implicit_rule_exclusions,
            "count": len(implicit_rules),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/review-campaigns/{review_type}/documents/{theme}/{doc_id}")
def api_load_review_document(
    review_type: str,
    theme: str,
    doc_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    if user.get("role") != "regular_user":
        raise HTTPException(status_code=403, detail="Review campaign documents are for assigned reviewers only")
    try:
        doc_data = review_campaign_service.load_review_task_document(review_type, int(user["id"]), theme, doc_id)
        return _sanitize_for_blind_review(doc_data, user)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/review-campaigns/{review_type}/documents/{theme}/{doc_id}")
def api_save_review_document(
    review_type: str,
    theme: str,
    doc_id: str,
    doc_data: dict[str, Any],
    user: dict = Depends(get_current_user),
) -> dict[str, str]:
    if user.get("role") != "regular_user":
        raise HTTPException(status_code=403, detail="Review campaign documents are for assigned reviewers only")
    try:
        review_campaign_service.save_review_task_document(
            review_type,
            int(user["id"]),
            theme,
            doc_id,
            _strip_runtime_fields(doc_data),
        )
        return {"status": "saved"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/review-campaigns/{review_type}/documents/{theme}/{doc_id}/finish")
def api_finish_review_document(
    review_type: str,
    theme: str,
    doc_id: str,
    user: dict = Depends(get_current_user),
) -> dict[str, Any]:
    if user.get("role") != "regular_user":
        raise HTTPException(status_code=403, detail="Review campaign documents are for assigned reviewers only")
    try:
        return review_campaign_service.finish_review_task(review_type, int(user["id"]), theme, doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/documents/{theme}/{doc_id}/validate")
def api_validate_document(theme: str, doc_id: str,
                          user: dict = Depends(get_current_user)) -> dict[str, Any]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    errors = validate_document(doc_data)
    return {"valid": len(errors) == 0, "errors": errors}


@router.get("/documents/{theme}/{doc_id}/entities")
def api_extract_entities(theme: str, doc_id: str,
                         user: dict = Depends(get_current_user)) -> dict[str, Any]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return extract_entities(doc_data)


@router.post("/documents/{theme}/{doc_id}/finish")
def api_finish_document(theme: str, doc_id: str,
                        user: dict = Depends(get_current_user)) -> dict[str, str]:
    """Mark a document as finished (records history entry)."""
    _require_workflow_access_if_needed(user, theme, doc_id)
    workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
    try:
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    doc_path = f"{theme}/{doc_id}"
    history_status = "completed"
    if workflow_task is not None:
        history_status = "in_progress"
    record_history(doc_path, user["id"], "edit", history_status, _build_history_snapshot(doc_data, user["username"]))
    if workflow_task is not None:
        workflow_service.finish_task(int(user["id"]), theme, doc_id)
    return {"status": "finished"}


@router.post("/documents/{theme}/{doc_id}/review")
def api_review_document(theme: str, doc_id: str,
                        user: dict = Depends(get_current_user)) -> dict[str, str]:
    """Mark a document as reviewed."""
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    doc_path = f"{theme}/{doc_id}"
    record_history(doc_path, user["id"], "review", "completed", _build_history_snapshot(doc_data, user["username"]))
    return {"status": "reviewed"}


@router.post("/documents/{theme}/{doc_id}/validate-status")
def api_validate_status(theme: str, doc_id: str,
                        user: dict = Depends(require_power_user)) -> dict[str, str]:
    """Validate a document (power_user only)."""
    try:
        doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    doc_path = f"{theme}/{doc_id}"
    record_history(doc_path, user["id"], "validate", "validated", _build_history_snapshot(doc_data, user["username"]))
    return {"status": "validated"}


@router.post("/documents/{theme}/{doc_id}/unvalidate")
def api_unvalidate_status(theme: str, doc_id: str,
                          user: dict = Depends(require_power_user)) -> dict[str, str]:
    """Reopen a document for editing (power_user only)."""
    try:
        doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    doc_path = f"{theme}/{doc_id}"
    record_history(doc_path, user["id"], "edit", "in_progress", _build_history_snapshot(doc_data, user["username"]))
    return {"status": "in_progress"}


@router.post("/documents/{theme}/{doc_id}/migrate")
def api_migrate_document(theme: str, doc_id: str, body: dict[str, str],
                         user: dict = Depends(require_power_user)) -> dict[str, str]:
    """Migrate a document to completed annotations (power_user only)."""
    from_user = body.get("from_user", user["username"])
    try:
        migrate_document(from_user, doc_id)
        doc_path = f"{theme}/{doc_id}"
        try:
            doc_data = load_document(user["username"], theme, doc_id)
            details = _build_history_snapshot(doc_data, user["username"])
        except FileNotFoundError:
            details = None
        record_history(doc_path, user["id"], "migrate", "validated", details)
        return {"status": "migrated"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Fictional Generation ---

@router.post("/documents/{theme}/{doc_id}/generate-fictional")
def api_generate_fictional(theme: str, doc_id: str,
                           body: dict[str, Any] = Body(default={}),
                           user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Generate a fictional document preview from an annotated template."""
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    seed = body.get("seed", 42)
    try:
        return generate_fictional_preview(doc_data, seed=seed, theme_id=theme)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@router.post("/documents/{theme}/{doc_id}/generate-fictional-batch")
def api_generate_fictional_batch(theme: str, doc_id: str,
                                 body: dict[str, Any] = Body(default={}),
                                 user: dict = Depends(get_current_user)) -> dict[str, Any]:
    """Generate multiple unique fictional previews in one backend call."""
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    seed = body.get("seed", 23)
    target_version_count = body.get("target_version_count", 10)
    max_attempts = body.get("max_attempts", 60)
    seed_stride = body.get("seed_stride", 1_000_003)
    try:
        return generate_fictional_previews_batch(
            doc_data,
            seed=seed,
            target_version_count=target_version_count,
            max_attempts=max_attempts,
            seed_stride=seed_stride,
            theme_id=theme,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")


@router.get("/playground/groq/models")
def api_list_groq_models(user: dict = Depends(require_power_user)) -> dict[str, Any]:
    """List Groq models available to the playground."""
    if not groq_is_configured():
        return {"configured": False, "models": []}
    try:
        return {"configured": True, "models": list_groq_models()}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to load Groq models: {exc}") from exc


@router.post("/documents/{theme}/{doc_id}/playground/groq/run")
def api_run_groq_playground(
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict = Depends(require_power_user),
) -> dict[str, Any]:
    """Run freeform document questions against selected Groq models."""
    if not groq_is_configured():
        raise HTTPException(status_code=503, detail="Groq is not configured on this deployment.")

    raw_document_text = str(body.get("document_text") or "")
    document_source = str(body.get("document_source") or "current").strip().lower()
    try:
        if raw_document_text:
            document_text = raw_document_text
        elif document_source == "original":
            source_doc = load_source_document(theme, doc_id)
            document_text = str(source_doc.get("original_document") or source_doc.get("document_to_annotate") or "")
        else:
            current_doc = load_document(user["username"], theme, doc_id)
            document_text = str(current_doc.get("document_to_annotate") or current_doc.get("original_document") or "")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        return run_groq_document_questions(
            document_text=document_text,
            questions=body.get("questions") or [],
            models=body.get("models") or [],
            system_prompt=body.get("system_prompt"),
            temperature=body.get("temperature", 0),
            seed=body.get("seed", 23),
            max_tokens=body.get("max_tokens", 512),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# --- Taxonomy ---

@router.get("/taxonomy")
def api_taxonomy(user: dict = Depends(get_current_user)) -> dict[str, list[str]]:
    return get_taxonomy()


# --- Progress ---

@router.get("/progress")
def api_progress(user: dict = Depends(get_current_user)) -> dict[str, Any]:
    _refresh_dashboard_db_snapshot()
    return get_theme_progress()


# --- Question CRUD ---

@router.post("/documents/{theme}/{doc_id}/questions")
def api_add_question(theme: str, doc_id: str, question: dict[str, Any],
                     user: dict = Depends(get_current_user)) -> dict[str, str]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    if "questions" not in doc_data:
        doc_data["questions"] = []
    doc_data["questions"].append(question)
    doc_data["num_questions"] = len(doc_data["questions"])
    clean_doc_data = _strip_runtime_fields(doc_data)
    workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
    if workflow_task is not None:
        workflow_service.save_task_document(int(user["id"]), theme, doc_id, clean_doc_data)
    else:
        try:
            save_document(user["username"], theme, doc_id, clean_doc_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"status": "added", "question_id": question.get("question_id", "")}


@router.put("/documents/{theme}/{doc_id}/questions/{question_id}")
def api_update_question(theme: str, doc_id: str, question_id: str,
                        question: dict[str, Any],
                        user: dict = Depends(get_current_user)) -> dict[str, str]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    questions = doc_data.get("questions", [])
    for i, q in enumerate(questions):
        if q.get("question_id") == question_id:
            questions[i] = question
            clean_doc_data = _strip_runtime_fields(doc_data)
            if workflow_task is not None:
                workflow_service.save_task_document(int(user["id"]), theme, doc_id, clean_doc_data)
            else:
                try:
                    save_document(user["username"], theme, doc_id, clean_doc_data)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
            return {"status": "updated"}
    raise HTTPException(status_code=404, detail=f"Question {question_id} not found")


@router.delete("/documents/{theme}/{doc_id}/questions/{question_id}")
def api_delete_question(theme: str, doc_id: str, question_id: str,
                        user: dict = Depends(get_current_user)) -> dict[str, str]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    questions = doc_data.get("questions", [])
    original_len = len(questions)
    doc_data["questions"] = [q for q in questions if q.get("question_id") != question_id]
    if len(doc_data["questions"]) == original_len:
        raise HTTPException(status_code=404, detail=f"Question {question_id} not found")
    doc_data["num_questions"] = len(doc_data["questions"])
    clean_doc_data = _strip_runtime_fields(doc_data)
    if workflow_task is not None:
        workflow_service.save_task_document(int(user["id"]), theme, doc_id, clean_doc_data)
    else:
        try:
            save_document(user["username"], theme, doc_id, clean_doc_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted"}


# --- Rules ---

@router.put("/documents/{theme}/{doc_id}/rules")
def api_update_rules(theme: str, doc_id: str, rules: list[str],
                     user: dict = Depends(get_current_user)) -> dict[str, str]:
    _require_workflow_access_if_needed(user, theme, doc_id)
    try:
        workflow_task = workflow_service.get_task_for_user_document(int(user["id"]), theme, doc_id)
        if workflow_task is not None:
            doc_data = workflow_service.load_task_document(int(user["id"]), theme, doc_id)
        else:
            doc_data = load_document(user["username"], theme, doc_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    doc_data["rules"] = rules
    clean_doc_data = _strip_runtime_fields(doc_data)
    if workflow_task is not None:
        workflow_service.save_task_document(int(user["id"]), theme, doc_id, clean_doc_data)
    else:
        try:
            save_document(user["username"], theme, doc_id, clean_doc_data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return {"status": "updated"}
