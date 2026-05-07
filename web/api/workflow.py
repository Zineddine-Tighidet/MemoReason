"""Workflow queue and monitoring API endpoints."""

import sqlite3
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from web.api.history import build_recent_activity_payload
from web.middleware.auth import get_current_user, require_power_user
from web.services import review_campaign_service, workflow_service
from web.services import auth_service, yaml_service
from web.services.db import close_db, get_db
from web.services.persistence import restore_db_from_gcs

router = APIRouter(prefix="/api/v1/workflow")


def _refresh_synced_dashboard_state(*, force_download: bool = False) -> None:
    """Refresh only the shared DB snapshot for read-only dashboard calls.

    The dashboard does not need the full annotation workspace, and reloading the
    whole worktree per request is expensive on Cloud Run.
    """
    close_db()
    if force_download:
        # Admin dashboard numbers must reflect the latest shared snapshot.
        # Force-refresh avoids stale per-instance drift when a local SQLite file
        # diverges from GCS between requests.
        restore_db_from_gcs(max_age_seconds=0.0, force_download=True)
    else:
        restore_db_from_gcs(max_age_seconds=30.0)


def _refresh_synced_write_db_state() -> None:
    """Refresh the shared DB snapshot before mutating review/workflow state.

    Cloud Run instances can hold a slightly stale local SQLite copy. For write
    endpoints that depend on recent foreign-keyed rows created by another
    request/instance, pull the latest shared snapshot first.
    """
    close_db()
    restore_db_from_gcs(force_download=True)


@router.get("/my-queue")
def api_my_queue(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    _refresh_synced_dashboard_state()
    return workflow_service.get_user_queue(int(user["id"]), auto_assign=False)


@router.get("/dashboard-bootstrap")
def api_dashboard_bootstrap(
    scope: str = "documents",
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Load the dashboard's initial data in one request."""
    normalized_scope = str(scope or "documents").strip().lower()
    # For admin rules/questions dashboards, always refresh from the shared DB so
    # campaign counters are consistent across instances.
    force_refresh = user.get("role") == "power_user" and normalized_scope in {"rules", "questions"}
    _refresh_synced_dashboard_state(force_download=force_refresh)
    db = get_db()
    if user.get("role") == "power_user":
        review_campaign_monitor_payload = {"campaigns": []}
        if normalized_scope in {"rules", "questions"}:
            review_campaign_monitor_payload = review_campaign_service.get_review_campaign_monitor(
                normalized_scope,
                db=db,
            )
        return {
            "role": "power_user",
            "progress": yaml_service.get_theme_progress(db=db),
            "recent_activity": build_recent_activity_payload(normalized_scope, db=db),
            "monitor": workflow_service.get_admin_monitor(db=db),
            "review_campaign_monitor": review_campaign_monitor_payload,
            "users": auth_service.list_users(),
        }

    user_id = int(user["id"])
    return {
        "role": "regular_user",
        "queue": workflow_service.get_user_queue(user_id, auto_assign=False),
        "review_queues": {
            "rules": review_campaign_service.get_user_review_queue(user_id, "rules"),
            "questions": review_campaign_service.get_user_review_queue(user_id, "questions"),
        },
        "resolution_feedback": workflow_service.get_user_resolution_feedback(user_id),
        "review_resolution_feedback": {
            "rules": review_campaign_service.get_user_review_resolution_feedback(user_id, "rules"),
            "questions": review_campaign_service.get_user_review_resolution_feedback(user_id, "questions"),
        },
    }


@router.post("/my-queue/assign-random")
def api_my_queue_assign_random(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    assignment = workflow_service.assign_random_task_to_user(int(user["id"]))
    queue = workflow_service.get_user_queue(int(user["id"]), auto_assign=False)
    queue["assignment"] = assignment
    return queue


@router.get("/my-review-queues/{review_type}")
def api_my_review_queue(review_type: str, user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    try:
        _refresh_synced_dashboard_state()
        return review_campaign_service.get_user_review_queue(int(user["id"]), review_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/my-review-queues/{review_type}/assign-random")
def api_my_review_queue_assign_random(
    review_type: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    try:
        _refresh_synced_write_db_state()
        assignment = review_campaign_service.assign_random_review_task_to_user(int(user["id"]), review_type)
        queue = review_campaign_service.get_user_review_queue(int(user["id"]), review_type)
        queue["assignment"] = assignment
        return queue
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/my-agreements")
def api_my_agreements(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    _refresh_synced_dashboard_state()
    return workflow_service.get_user_agreements(int(user["id"]))


@router.get("/my-agreements/{theme}/{doc_id}/packet")
def api_my_agreement_packet(
    theme: str,
    doc_id: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    raise HTTPException(status_code=403, detail="Agreement packet access is admin-only")


@router.post("/my-agreements/{theme}/{doc_id}/resolve")
def api_my_resolve_agreement(
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    raise HTTPException(status_code=403, detail="Agreement resolution is admin-only")


@router.get("/my-resolution-feedback")
def api_my_resolution_feedback(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    _refresh_synced_dashboard_state()
    return workflow_service.get_user_resolution_feedback(int(user["id"]))


@router.get("/my-review-resolution-feedback/{review_type}")
def api_my_review_resolution_feedback(
    review_type: str,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _refresh_synced_dashboard_state()
    try:
        return review_campaign_service.get_user_review_resolution_feedback(int(user["id"]), review_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/my-resolution-feedback/{theme}/{doc_id}/decision")
def api_submit_my_resolution_feedback_decision(
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    status = str(body.get("response_status", "")).strip().lower()
    try:
        return workflow_service.submit_user_resolution_feedback_response(
            int(user["id"]),
            theme,
            doc_id,
            status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/my-review-resolution-feedback/{review_type}/{theme}/{doc_id}/decision")
def api_submit_my_review_resolution_feedback_decision(
    review_type: str,
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    status = str(body.get("response_status", "")).strip().lower()
    try:
        return review_campaign_service.submit_user_review_resolution_feedback_response(
            int(user["id"]),
            review_type,
            theme,
            doc_id,
            status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/admin/monitor")
def api_workflow_monitor(user: dict[str, Any] = Depends(require_power_user)) -> dict[str, Any]:
    _refresh_synced_dashboard_state()
    return workflow_service.get_admin_monitor()


@router.get("/admin/submissions")
def api_admin_submissions(user: dict[str, Any] = Depends(require_power_user)) -> dict[str, Any]:
    return workflow_service.get_admin_submissions()


@router.get("/admin/submissions/{theme}/{doc_id}/summary")
def api_admin_submission_summary(
    theme: str,
    doc_id: str,
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    try:
        return workflow_service.get_admin_submission_for_document(theme, doc_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/admin/review-campaigns/{review_type}/submissions/{theme}/{doc_id}/summary")
def api_admin_review_submission_summary(
    review_type: str,
    theme: str,
    doc_id: str,
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    try:
        return review_campaign_service.get_admin_review_submission_for_document(review_type, theme, doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/admin/submissions/{theme}/{doc_id}")
def api_admin_submission_content(
    theme: str,
    doc_id: str,
    variant: str,
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    try:
        return workflow_service.get_admin_submission_content(theme, doc_id, variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/admin/review-campaigns/{review_type}/submissions/{theme}/{doc_id}")
def api_admin_review_submission_content(
    review_type: str,
    theme: str,
    doc_id: str,
    variant: str,
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    try:
        return review_campaign_service.get_admin_review_submission_content(review_type, theme, doc_id, variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/submissions/{theme}/{doc_id}/final")
def api_admin_set_final_submission(
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    source_variant = str(body.get("source_variant", "")).strip().lower()
    try:
        return workflow_service.set_admin_final_snapshot(theme, doc_id, source_variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/submissions/{theme}/{doc_id}/final-from-editor")
def api_admin_set_final_from_editor(
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    raw_document = body.get("document")
    if isinstance(raw_document, dict):
        document = raw_document
    else:
        document = body if isinstance(body, dict) else {}
    source_label = str(body.get("source_label", "admin")).strip() or "admin"
    try:
        return workflow_service.set_admin_final_snapshot_from_document(
            theme,
            doc_id,
            document=document,
            source_label=source_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/review-campaigns/{review_type}/submissions/{theme}/{doc_id}/final-from-editor")
def api_admin_set_review_final_from_editor(
    review_type: str,
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    raw_document = body.get("document")
    if isinstance(raw_document, dict):
        document = raw_document
    else:
        document = body if isinstance(body, dict) else {}
    source_label = str(body.get("source_label", "admin")).strip() or "admin"
    try:
        _refresh_synced_write_db_state()
        return review_campaign_service.set_admin_review_final_snapshot_from_document(
            review_type,
            theme,
            doc_id,
            document=document,
            source_label=source_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/review-campaigns/{review_type}/documents/{theme}/{doc_id}/complete-from-editor")
def api_admin_complete_review_from_editor(
    review_type: str,
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    raw_document = body.get("document")
    if isinstance(raw_document, dict):
        document = raw_document
    else:
        document = body if isinstance(body, dict) else {}
    source_label = str(body.get("source_label", "admin_completed")).strip() or "admin_completed"
    try:
        _refresh_synced_write_db_state()
        return review_campaign_service.complete_admin_review_from_document(
            review_type,
            theme,
            doc_id,
            document=document,
            resolved_by_user_id=int(user["id"]),
            source_label=source_label,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/runs")
def api_create_run(
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    name = str(body.get("name", "")).strip() or "Blind Review Run"
    source_agent = str(body.get("source_agent", "")).strip() or "claude-opus-4-6"
    seed = int(body.get("seed", 20260226))
    annotator_usernames_raw = body.get("annotator_usernames")
    include_legacy = bool(body.get("include_legacy", False))

    annotator_usernames: list[str] | None = None
    if isinstance(annotator_usernames_raw, list):
        annotator_usernames = [str(item).strip() for item in annotator_usernames_raw if str(item).strip()]
        if not annotator_usernames:
            annotator_usernames = None

    try:
        return workflow_service.create_run(
            name=name,
            source_agent=source_agent,
            seed=seed,
            created_by_user_id=int(user["id"]),
            annotator_usernames=annotator_usernames,
            include_legacy=include_legacy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/review-campaigns/{review_type}/agreements/{campaign_id}/{theme}/{doc_id}/resolve")
def api_resolve_review_agreement(
    review_type: str,
    campaign_id: int,
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    final_variant_raw = body.get("final_variant")
    final_variant = str(final_variant_raw).strip().lower() if isinstance(final_variant_raw, str) else None
    completion_mode = str(body.get("completion_mode", "")).strip().lower()
    require_reviewer_acceptance = completion_mode != "complete"
    try:
        _refresh_synced_write_db_state()
        try:
            return review_campaign_service.resolve_review_agreement(
                campaign_id=campaign_id,
                review_type=review_type,
                theme=theme,
                doc_id=doc_id,
                resolved_by_user_id=int(user["id"]),
                final_variant=final_variant,
                require_reviewer_acceptance=require_reviewer_acceptance,
            )
        except sqlite3.IntegrityError:
            _refresh_synced_write_db_state()
            return review_campaign_service.resolve_review_agreement(
                campaign_id=campaign_id,
                review_type=review_type,
                theme=theme,
                doc_id=doc_id,
                resolved_by_user_id=int(user["id"]),
                final_variant=final_variant,
                require_reviewer_acceptance=require_reviewer_acceptance,
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/admin/review-campaigns")
def api_get_review_campaigns(user: dict[str, Any] = Depends(require_power_user)) -> dict[str, Any]:
    _refresh_synced_dashboard_state()
    return review_campaign_service.get_review_campaign_monitor()


@router.post("/admin/review-campaigns")
def api_create_review_campaign(
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    name = str(body.get("name", "")).strip() or "Review Campaign"
    review_type = str(body.get("review_type", "")).strip().lower()
    seed = int(body.get("seed", 20260309))

    try:
        return review_campaign_service.create_review_campaign(
            name=name,
            review_type=review_type,
            seed=seed,
            created_by_user_id=int(user["id"]),
            reviewer_usernames=None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/review-campaigns/{campaign_id}/complete")
def api_complete_review_campaign(
    campaign_id: int,
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    try:
        return review_campaign_service.complete_review_campaign(campaign_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/runs/{run_id}/complete")
def api_mark_run_completed(run_id: int, user: dict[str, Any] = Depends(require_power_user)) -> dict[str, str]:
    db = get_db()
    row = db.execute("SELECT id, status FROM workflow_runs WHERE id = ?", (int(run_id),)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Workflow run not found")
    db.execute("UPDATE workflow_runs SET status = 'completed' WHERE id = ?", (int(run_id),))
    db.commit()
    from web.services.persistence import sync_db_to_gcs

    sync_db_to_gcs()
    return {"status": "completed"}


@router.get("/admin/runs/{run_id}/agreements/{theme}/{doc_id}")
def api_get_agreement_packet(
    run_id: int,
    theme: str,
    doc_id: str,
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    try:
        return workflow_service.get_agreement_packet(int(run_id), theme, doc_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/admin/runs/{run_id}/agreements/{theme}/{doc_id}/resolve")
def api_resolve_agreement_packet(
    run_id: int,
    theme: str,
    doc_id: str,
    body: dict[str, Any] = Body(default={}),
    user: dict[str, Any] = Depends(require_power_user),
) -> dict[str, Any]:
    final_variant = body.get("final_variant")
    if final_variant is not None:
        final_variant = str(final_variant).strip().lower()
    try:
        return workflow_service.resolve_agreement(
            int(run_id),
            theme,
            doc_id,
            int(user["id"]),
            final_variant=final_variant if isinstance(final_variant, str) else None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
