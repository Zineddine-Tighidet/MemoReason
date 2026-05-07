"""Rule/question review campaigns built from completed annotated templates."""

from __future__ import annotations

import random
import re
import shutil
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.dataset_export.dataset_paths import HUMAN_ANNOTATED_TEMPLATES_DIR, iter_template_paths, resolve_template_identity
from src.core.annotation_runtime import (
    AnnotationParser,
    compose_rule_text,
    normalize_document_taxonomy,
    normalize_entity_ref,
    normalize_text_entity_refs,
    split_rule_text_and_comment,
)
from web.services import yaml_service
from web.services.db import get_db
from web.services.persistence import (
    delete_work_prefix_from_gcs,
    restore_state_from_gcs,
    restore_work_file_from_gcs,
    restore_worktree_from_gcs,
    sync_db_to_gcs,
    sync_work_file_to_gcs,
)

REVIEW_TYPES: tuple[str, ...] = ("rules", "questions")
REVIEW_SOURCE_ROOT = yaml_service.WORK_DIR / "_review_sources"
REVIEW_CAMPAIGN_ROOT = yaml_service.WORK_DIR / "_review_campaigns"
RULE_REVIEW_EXCLUDED_THEMES: frozenset[str] = frozenset({"public_attacks_news_articles"})
RULE_REVIEW_AUTO_RESOLVED_THEMES: frozenset[str] = frozenset({"public_attacks_news_articles"})
QUESTION_REVIEW_EXCLUDED_THEMES: frozenset[str] = frozenset({"public_attacks_news_articles"})
ALLOWED_REVIEW_FEEDBACK_RESPONSE_STATUSES: tuple[str, ...] = ("accepted", "contest_requested")
QUESTION_EXPERIMENT_GROUP_LABELS: dict[str, str] = {
    "ai_drafted_qas": "AI drafted QAs",
    "no_qas": "No QAs",
}
QUESTION_EXPERIMENT_GROUP_SORT_ORDER: dict[str, int] = {
    "ai_drafted_qas": 0,
    "no_qas": 1,
}
QUESTION_REVIEW_REQUIRED_TYPES: tuple[str, ...] = ("extractive", "arithmetic", "inference", "temporal")
QUESTION_REVIEW_REQUIRED_ANSWER_TYPES: tuple[str, ...] = ("variant", "invariant", "refusal")
QUESTION_REVIEW_REFUSAL_ANSWER_LITERAL = "Cannot be determined"
QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD = "qa_coverage_exemptions"
INLINE_ANNOTATION_PATTERN = re.compile(r"\[([^\]]+);\s*([^\]]+)\]")
MERGEABLE_EQUALITY_RULE_PATTERN = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*)\s*==\s*([A-Za-z][A-Za-z0-9_]*\.[A-Za-z][A-Za-z0-9_]*)\s*$"
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _validate_review_type(review_type: str) -> str:
    normalized = str(review_type or "").strip().lower()
    if normalized not in REVIEW_TYPES:
        raise ValueError(f"Unsupported review_type: {review_type!r}")
    return normalized


def _required_review_submissions(review_type: str) -> int:
    """Number of reviewer submissions needed before admin agreement can start."""
    normalized = _validate_review_type(review_type)
    # QA review now follows a sequential flow:
    # 1 regular reviewer submission -> 1 power-user agreement/review pass.
    if normalized == "questions":
        return 1
    return 2


def _normalize_question_experiment_group(raw_group: Any) -> str:
    normalized = str(raw_group or "").strip().lower()
    return normalized if normalized in QUESTION_EXPERIMENT_GROUP_LABELS else ""


def _normalize_question_type(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in QUESTION_REVIEW_REQUIRED_TYPES:
        return normalized
    return "extractive"


def _normalize_answer_type(raw_value: Any, raw_invariant: Any = None) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in QUESTION_REVIEW_REQUIRED_ANSWER_TYPES:
        return normalized
    if raw_invariant is True:
        return "invariant"
    if raw_invariant is False:
        return "variant"
    return "variant"


def _normalize_question_review_questions(raw_questions: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_questions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw_questions, start=1):
        if not isinstance(item, dict):
            continue
        answer_type = _normalize_answer_type(item.get("answer_type"), item.get("is_answer_invariant"))
        answer_text = str(item.get("answer") or "").strip()
        if answer_type == "refusal":
            answer_text = QUESTION_REVIEW_REFUSAL_ANSWER_LITERAL
        reasoning_chain = item.get("reasoning_chain")
        normalized_reasoning_chain = [
            str(step or "").strip()
            for step in (reasoning_chain if isinstance(reasoning_chain, list) else [])
            if str(step or "").strip()
        ]
        normalized.append(
            {
                "question_id": str(item.get("question_id") or f"q_{index}").strip(),
                "question": str(item.get("question") or "").strip(),
                "question_type": _normalize_question_type(item.get("question_type")),
                "answer": answer_text,
                "answer_type": answer_type,
                "reasoning_chain": normalized_reasoning_chain,
            }
        )
    return normalized


def _document_annotation_surface_map(document_text: Any) -> dict[str, str]:
    text = str(document_text or "")
    if not text.strip():
        return {}
    surfaces_by_ref: dict[str, str] = {}
    for annotation in AnnotationParser.parse_annotations(text):
        entity_id = str(annotation.entity_id or "").strip()
        attribute = str(annotation.attribute or "").strip()
        surface = str(annotation.original_text or "").strip()
        if not entity_id or not surface:
            continue
        raw_ref = f"{entity_id}.{attribute}" if attribute else entity_id
        normalized_ref = normalize_entity_ref(raw_ref)
        if normalized_ref and normalized_ref not in surfaces_by_ref:
            surfaces_by_ref[normalized_ref] = surface
    return surfaces_by_ref


def _document_unique_ref_by_surface(document_text: Any) -> dict[str, str]:
    surfaces_by_ref = _document_annotation_surface_map(document_text)
    refs_by_surface: dict[str, str] = {}
    ambiguous_surfaces: set[str] = set()
    for ref, surface in surfaces_by_ref.items():
        surface_key = str(surface or "").strip().casefold()
        if not surface_key:
            continue
        existing_ref = refs_by_surface.get(surface_key)
        if existing_ref and existing_ref != ref:
            ambiguous_surfaces.add(surface_key)
            refs_by_surface.pop(surface_key, None)
            continue
        if surface_key not in ambiguous_surfaces:
            refs_by_surface[surface_key] = ref
    return refs_by_surface


def _question_annotation_texts(document: dict[str, Any]) -> list[str]:
    question_texts: list[str] = []
    raw_questions = document.get("questions")
    if not isinstance(raw_questions, list):
        return question_texts
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        for field_name in ("question", "answer", "reasoning_chain_text"):
            value = raw_question.get(field_name)
            if isinstance(value, str) and value.strip():
                question_texts.append(value)
            elif isinstance(value, list):
                question_texts.extend(str(item) for item in value if isinstance(item, str) and item.strip())
        reasoning_chain = raw_question.get("reasoning_chain")
        if isinstance(reasoning_chain, list):
            question_texts.extend(str(step) for step in reasoning_chain if isinstance(step, str) and step.strip())
    return question_texts


def _infer_document_entity_id_remap(document: dict[str, Any]) -> dict[str, str]:
    if not isinstance(document, dict):
        return {}
    unique_ref_by_surface = _document_unique_ref_by_surface(document.get("document_to_annotate"))
    if not unique_ref_by_surface:
        return {}

    entity_id_remap: dict[str, str] = {}
    for text in _question_annotation_texts(document):
        for annotation in AnnotationParser.parse_annotations(text):
            entity_id = str(annotation.entity_id or "").strip()
            attribute = str(annotation.attribute or "").strip()
            raw_ref = f"{entity_id}.{attribute}" if attribute else entity_id
            normalized_ref = normalize_entity_ref(raw_ref)
            if not normalized_ref:
                continue
            normalized_entity_id = normalized_ref.split(".", 1)[0]
            if normalized_ref in unique_ref_by_surface.values():
                continue
            surface_key = str(annotation.original_text or "").strip().casefold()
            target_ref = unique_ref_by_surface.get(surface_key)
            if not target_ref:
                continue
            target_entity_id = target_ref.split(".", 1)[0]
            previous_target = entity_id_remap.get(normalized_entity_id)
            if previous_target and previous_target != target_entity_id:
                continue
            entity_id_remap[normalized_entity_id] = target_entity_id
    return entity_id_remap


def _entity_family(entity_id: str) -> str:
    normalized = str(entity_id or "").strip()
    if not normalized:
        return ""
    return re.sub(r"_\d+$", "", normalized)


def _entity_index(entity_id: str) -> int | None:
    match = re.search(r"_(\d+)$", str(entity_id or "").strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _document_entity_first_positions(document_text: Any) -> dict[str, int]:
    text = str(document_text or "")
    positions: dict[str, int] = {}
    if not text.strip():
        return positions
    for match in INLINE_ANNOTATION_PATTERN.finditer(text):
        raw_ref = str(match.group(2) or "").strip()
        normalized_ref = normalize_entity_ref(raw_ref)
        if not normalized_ref:
            continue
        normalized_entity_id = normalized_ref.split(".", 1)[0]
        start = int(match.start())
        positions.setdefault(normalized_entity_id, start)
    return positions


def _parse_mergeable_equality_rule(rule_text: Any) -> tuple[str, str, str, str] | None:
    expression, _ = split_rule_text_and_comment(str(rule_text or ""))
    if not expression:
        return None
    match = MERGEABLE_EQUALITY_RULE_PATTERN.match(expression)
    if not match:
        return None
    left_ref = normalize_entity_ref(match.group(1))
    right_ref = normalize_entity_ref(match.group(2))
    if not left_ref or not right_ref or "." not in left_ref or "." not in right_ref:
        return None
    left_entity_id, left_attr = left_ref.split(".", 1)
    right_entity_id, right_attr = right_ref.split(".", 1)
    if left_attr != right_attr:
        return None
    if _entity_family(left_entity_id) not in {"temporal", "number"}:
        return None
    if _entity_family(left_entity_id) != _entity_family(right_entity_id):
        return None
    return left_ref, right_ref, left_entity_id, right_entity_id


def _resolve_entity_id_remap(entity_id: str, entity_id_remap: dict[str, str]) -> str:
    current = str(entity_id or "").strip()
    seen: set[str] = set()
    while current and current in entity_id_remap and current not in seen:
        seen.add(current)
        current = str(entity_id_remap.get(current) or "").strip()
    return current


def _preferred_canonical_entity_id(
    left_entity_id: str,
    right_entity_id: str,
    *,
    first_positions: dict[str, int],
) -> str:
    left = str(left_entity_id or "").strip()
    right = str(right_entity_id or "").strip()
    left_pos = first_positions.get(left)
    right_pos = first_positions.get(right)
    if left_pos is not None and right_pos is not None and left_pos != right_pos:
        return left if left_pos < right_pos else right
    if left_pos is not None:
        return left
    if right_pos is not None:
        return right

    left_index = _entity_index(left)
    right_index = _entity_index(right)
    if left_index is not None and right_index is not None and left_index != right_index:
        return left if left_index < right_index else right
    return min(left, right) if left and right else (left or right)


def _mergeable_equality_entity_id_remap_for_document(document: dict[str, Any]) -> dict[str, str]:
    if not isinstance(document, dict):
        return {}

    normalized_document = normalize_document_taxonomy(document)
    rules = normalized_document.get("rules")
    if not isinstance(rules, list) or not rules:
        return {}

    entity_id_remap: dict[str, str] = {}
    first_positions = _document_entity_first_positions(normalized_document.get("document_to_annotate"))
    for raw_rule in rules:
        parsed = _parse_mergeable_equality_rule(raw_rule)
        if not parsed:
            continue
        _, _, left_entity_id, right_entity_id = parsed
        left_root = _resolve_entity_id_remap(left_entity_id, entity_id_remap)
        right_root = _resolve_entity_id_remap(right_entity_id, entity_id_remap)
        if not left_root or not right_root or left_root == right_root:
            continue
        canonical_root = _preferred_canonical_entity_id(
            left_root,
            right_root,
            first_positions=first_positions,
        )
        alias_root = right_root if canonical_root == left_root else left_root
        entity_id_remap[alias_root] = canonical_root

    compressed_remap: dict[str, str] = {}
    for source_entity_id in list(entity_id_remap):
        target_entity_id = _resolve_entity_id_remap(source_entity_id, entity_id_remap)
        if target_entity_id and target_entity_id != source_entity_id:
            compressed_remap[source_entity_id] = target_entity_id
    return compressed_remap


def _apply_entity_id_remap_to_document(document: dict[str, Any], entity_id_remap: dict[str, str]) -> dict[str, Any]:
    if not isinstance(document, dict):
        return {}
    normalized_document = normalize_document_taxonomy(document)
    if not entity_id_remap:
        return normalized_document

    rewritten = dict(normalized_document)
    for field_name in ("document_to_annotate", "fictionalized_annotated_template_document"):
        if field_name in rewritten:
            rewritten[field_name] = normalize_text_entity_refs(
                str(rewritten.get(field_name) or ""),
                entity_id_remap=entity_id_remap,
            )

    rewritten_rules: list[str] = []
    for raw_rule in rewritten.get("rules") or []:
        expression, comment = split_rule_text_and_comment(str(raw_rule or ""))
        if not expression:
            continue
        normalized_expression = normalize_text_entity_refs(expression, entity_id_remap=entity_id_remap)
        rendered_rule = compose_rule_text(normalized_expression, comment)
        if rendered_rule:
            rewritten_rules.append(rendered_rule)
    rewritten["rules"] = rewritten_rules

    rewritten_questions: list[dict[str, Any]] = []
    for raw_question in rewritten.get("questions") or []:
        if not isinstance(raw_question, dict):
            continue
        question = dict(raw_question)
        for field_name in ("question", "answer", "reasoning_chain_text"):
            value = question.get(field_name)
            if isinstance(value, str):
                question[field_name] = normalize_text_entity_refs(value, entity_id_remap=entity_id_remap)
            elif isinstance(value, list):
                question[field_name] = [
                    normalize_text_entity_refs(str(item), entity_id_remap=entity_id_remap)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
        reasoning_chain = question.get("reasoning_chain")
        if isinstance(reasoning_chain, list):
            normalized_reasoning_chain = [
                normalize_text_entity_refs(str(step), entity_id_remap=entity_id_remap)
                if isinstance(step, str)
                else step
                for step in reasoning_chain
            ]
            question["reasoning_chain"] = normalized_reasoning_chain
            question["reasoning_chain_text"] = "\n".join(
                str(step or "").strip()
                for step in normalized_reasoning_chain
                if str(step or "").strip()
            )
        rewritten_questions.append(question)
    rewritten["questions"] = rewritten_questions

    _align_question_annotation_surfaces_with_document(rewritten)
    return normalize_document_taxonomy(rewritten)


def _canonicalize_mergeable_equality_rules(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        return document

    normalized_document = normalize_document_taxonomy(document)
    rules = normalized_document.get("rules")
    if not isinstance(rules, list) or not rules:
        return normalized_document

    compressed_remap = _mergeable_equality_entity_id_remap_for_document(normalized_document)
    rewritten = (
        _apply_entity_id_remap_to_document(normalized_document, compressed_remap)
        if compressed_remap
        else yaml.safe_load(yaml.safe_dump(normalized_document, sort_keys=False, allow_unicode=True, width=10000)) or {}
    )

    merge_rule_indexes: set[int] = set()
    for index, raw_rule in enumerate(rules):
        if _parse_mergeable_equality_rule(raw_rule):
            merge_rule_indexes.add(index)
    rewritten_rules: list[str] = []
    for index, raw_rule in enumerate(rules):
        expression, comment = split_rule_text_and_comment(str(raw_rule or ""))
        if not expression:
            continue
        normalized_expression = normalize_text_entity_refs(expression, entity_id_remap=compressed_remap)
        equality_match = MERGEABLE_EQUALITY_RULE_PATTERN.match(normalized_expression)
        if equality_match:
            left_ref = normalize_entity_ref(equality_match.group(1))
            right_ref = normalize_entity_ref(equality_match.group(2))
            if left_ref and right_ref and left_ref == right_ref:
                continue
        parsed_after_rewrite = _parse_mergeable_equality_rule(normalized_expression)
        if index in merge_rule_indexes:
            continue
        if parsed_after_rewrite is not None:
            left_ref, right_ref, _, _ = parsed_after_rewrite
            if left_ref == right_ref:
                continue
        rendered_rule = compose_rule_text(normalized_expression, comment)
        if rendered_rule:
            rewritten_rules.append(rendered_rule)
    rewritten["rules"] = rewritten_rules
    return normalize_document_taxonomy(rewritten)


def _historical_rules_entity_id_remap(theme: str, doc_id: str, db=None) -> dict[str, str]:
    canonical_theme = yaml_service.canonical_theme_id(str(theme))
    canonical_doc_id = str(doc_id)
    candidate_paths: list[Path] = [review_source_path("rules", canonical_theme, canonical_doc_id)]
    candidate_paths.extend(
        sorted(REVIEW_CAMPAIGN_ROOT.glob(f"rules/campaign_*/agreements/final/{canonical_theme}/{canonical_doc_id}.yaml"))
    )
    if db is None:
        db = get_db()
    snapshot_row = db.execute(
        """
        SELECT latest_snapshot_path
        FROM review_artifact_statuses
        WHERE review_type = 'rules'
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (canonical_theme, canonical_doc_id),
    ).fetchone()
    if snapshot_row is not None:
        snapshot_path = _resolve_review_storage_path(snapshot_row["latest_snapshot_path"])
        if snapshot_path is not None:
            candidate_paths.insert(1, snapshot_path)

    seen: set[Path] = set()
    for candidate_path in candidate_paths:
        if candidate_path in seen or candidate_path is None:
            continue
        seen.add(candidate_path)
        if not candidate_path.exists():
            restore_work_file_from_gcs(candidate_path)
        if not candidate_path.exists():
            continue
        payload = _load_yaml(candidate_path)
        document = payload.get("document", payload if isinstance(payload, dict) else {})
        if not isinstance(document, dict):
            continue
        remap = _mergeable_equality_entity_id_remap_for_document(document)
        if remap:
            return remap
    return {}


def _rewrite_cross_section_entity_refs_to_document(document: dict[str, Any]) -> None:
    if not isinstance(document, dict):
        return
    entity_id_remap = _infer_document_entity_id_remap(document)
    if not entity_id_remap:
        return

    rules = document.get("rules")
    if isinstance(rules, list):
        document["rules"] = [
            normalize_text_entity_refs(str(rule), entity_id_remap=entity_id_remap)
            for rule in rules
        ]

    raw_questions = document.get("questions")
    if not isinstance(raw_questions, list):
        return

    rewritten_questions: list[dict[str, Any]] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        question = dict(raw_question)
        for field_name in ("question", "answer", "reasoning_chain_text"):
            value = question.get(field_name)
            if isinstance(value, str):
                question[field_name] = normalize_text_entity_refs(value, entity_id_remap=entity_id_remap)
            elif isinstance(value, list):
                question[field_name] = [
                    normalize_text_entity_refs(str(item), entity_id_remap=entity_id_remap)
                    if isinstance(item, str)
                    else item
                    for item in value
                ]
        reasoning_chain = question.get("reasoning_chain")
        if isinstance(reasoning_chain, list):
            question["reasoning_chain"] = [
                normalize_text_entity_refs(step, entity_id_remap=entity_id_remap)
                if isinstance(step, str)
                else step
                for step in reasoning_chain
            ]
        rewritten_questions.append(question)
    document["questions"] = rewritten_questions


def _rewrite_inline_annotation_surfaces(
    text: Any,
    surfaces_by_ref: dict[str, str],
    unique_ref_by_surface: dict[str, str] | None = None,
) -> str:
    source_text = str(text or "")
    if not source_text or not surfaces_by_ref:
        return source_text

    def _replace(match: re.Match[str]) -> str:
        original_surface = str(match.group(1) or "").strip()
        raw_ref = str(match.group(2) or "").strip()
        normalized_ref = normalize_entity_ref(raw_ref)
        if not normalized_ref:
            return match.group(0)
        replacement_surface = surfaces_by_ref.get(normalized_ref)
        if not replacement_surface and unique_ref_by_surface:
            replacement_ref = unique_ref_by_surface.get(original_surface.casefold())
            if replacement_ref:
                normalized_ref = replacement_ref
                replacement_surface = surfaces_by_ref.get(normalized_ref)
        if not replacement_surface:
            replacement_surface = original_surface
        return f"[{replacement_surface}; {normalized_ref}]"

    return INLINE_ANNOTATION_PATTERN.sub(_replace, source_text)


def _align_question_annotation_surfaces_with_document(document: dict[str, Any]) -> None:
    if not isinstance(document, dict):
        return
    raw_questions = document.get("questions")
    if not isinstance(raw_questions, list) or not raw_questions:
        return
    surfaces_by_ref = _document_annotation_surface_map(document.get("document_to_annotate"))
    if not surfaces_by_ref:
        return
    refs_by_surface: dict[str, str] = {}
    ambiguous_surfaces: set[str] = set()
    for ref, surface in surfaces_by_ref.items():
        surface_key = str(surface or "").strip().casefold()
        if not surface_key:
            continue
        existing_ref = refs_by_surface.get(surface_key)
        if existing_ref and existing_ref != ref:
            ambiguous_surfaces.add(surface_key)
            refs_by_surface.pop(surface_key, None)
            continue
        if surface_key not in ambiguous_surfaces:
            refs_by_surface[surface_key] = ref

    normalized_questions: list[dict[str, Any]] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        question = dict(raw_question)
        question["question"] = _rewrite_inline_annotation_surfaces(
            question.get("question"), surfaces_by_ref, refs_by_surface
        )

        raw_reasoning_chain = question.get("reasoning_chain")
        if isinstance(raw_reasoning_chain, list):
            rewritten_reasoning_chain = [
                _rewrite_inline_annotation_surfaces(step, surfaces_by_ref, refs_by_surface)
                if isinstance(step, str)
                else step
                for step in raw_reasoning_chain
            ]
            question["reasoning_chain"] = rewritten_reasoning_chain
            normalized_steps = [str(step or "").strip() for step in rewritten_reasoning_chain if str(step or "").strip()]
            question["reasoning_chain_text"] = "\n".join(normalized_steps)
        elif isinstance(question.get("reasoning_chain_text"), str):
            question["reasoning_chain_text"] = _rewrite_inline_annotation_surfaces(
                question.get("reasoning_chain_text"), surfaces_by_ref, refs_by_surface
            )

        raw_answer = question.get("answer")
        if isinstance(raw_answer, str):
            question["answer"] = _rewrite_inline_annotation_surfaces(raw_answer, surfaces_by_ref, refs_by_surface)
        elif isinstance(raw_answer, list):
            question["answer"] = [
                _rewrite_inline_annotation_surfaces(item, surfaces_by_ref, refs_by_surface)
                if isinstance(item, str)
                else item
                for item in raw_answer
            ]

        normalized_questions.append(question)

    document["questions"] = normalized_questions
    document["num_questions"] = len(normalized_questions)


def _normalize_question_review_coverage_exemptions(raw_exemptions: Any) -> list[dict[str, str]]:
    if not isinstance(raw_exemptions, list):
        return []
    normalized_rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_exemptions:
        if not isinstance(item, dict):
            continue
        question_type = _normalize_question_type(item.get("question_type"))
        answer_type = _normalize_answer_type(item.get("answer_type"))
        if (question_type, answer_type) in seen:
            continue
        justification = str(item.get("justification") or "").strip()
        if not justification:
            continue
        seen.add((question_type, answer_type))
        normalized_rows.append(
            {
                "question_type": question_type,
                "answer_type": answer_type,
                "justification": justification,
            }
        )
    return normalized_rows


def _required_question_review_pairs() -> set[tuple[str, str]]:
    return {
        (question_type, answer_type)
        for question_type in QUESTION_REVIEW_REQUIRED_TYPES
        for answer_type in QUESTION_REVIEW_REQUIRED_ANSWER_TYPES
    }


def _is_question_review_draft_source_label(source_label: Any) -> bool:
    normalized = str(source_label or "").strip().lower()
    if not normalized:
        return False
    return normalized == "draft" or normalized.endswith("_draft")


def _question_review_submission_contract_errors(document: dict[str, Any]) -> list[str]:
    questions = _normalize_question_review_questions(document.get("questions"))
    exemptions = _normalize_question_review_coverage_exemptions(
        document.get(QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD)
    )
    required_pairs = _required_question_review_pairs()
    max_question_count = len(required_pairs)
    errors: list[str] = []

    if len(questions) > max_question_count:
        errors.append(
            f"Questions submission contains {len(questions)} questions; maximum is {max_question_count}."
        )

    question_id_counts: dict[str, int] = {}
    for question in questions:
        question_id = str(question.get("question_id") or "").strip()
        if not question_id:
            continue
        question_id_counts[question_id] = question_id_counts.get(question_id, 0) + 1
    duplicate_question_ids = sorted(
        question_id for question_id, count in question_id_counts.items() if count > 1
    )
    if duplicate_question_ids:
        labels = ", ".join(duplicate_question_ids)
        errors.append(
            f"Questions submission contains duplicate question IDs: {labels}. Keep question IDs unique."
        )

    pair_counts: dict[tuple[str, str], int] = {}
    for question in questions:
        pair = (
            str(question.get("question_type") or "").strip().lower(),
            str(question.get("answer_type") or "").strip().lower(),
        )
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    duplicate_pairs = sorted(pair for pair, count in pair_counts.items() if count > 1)
    if duplicate_pairs:
        labels = ", ".join(f"{question_type}/{answer_type}" for question_type, answer_type in duplicate_pairs)
        errors.append(
            "Questions submission contains duplicate question/answer-type combinations: "
            f"{labels}. Keep at most one QA per combination."
        )

    exemption_pairs = {(row["question_type"], row["answer_type"]) for row in exemptions}
    covered_pairs = {
        (str(question.get("question_type") or "").strip().lower(), str(question.get("answer_type") or "").strip().lower())
        for question in questions
        if str(question.get("question") or "").strip()
        and (
            str(question.get("answer") or "").strip()
            or str(question.get("answer_type") or "").strip().lower() == "refusal"
        )
    }
    missing_pairs = sorted(required_pairs - covered_pairs - exemption_pairs)
    if missing_pairs:
        labels = ", ".join(f"{question_type}/{answer_type}" for question_type, answer_type in missing_pairs)
        errors.append(
            "Questions submission is missing required question/answer-type combinations: "
            f"{labels}. Add the missing questions or add coverage exemptions with a justification."
        )
    return errors


def _validate_question_review_submission_contract(document: dict[str, Any]) -> None:
    errors = _question_review_submission_contract_errors(document)
    if errors:
        raise ValueError("Invalid questions submission. " + " ".join(errors))


def _normalize_question_review_document(document: dict[str, Any]) -> dict[str, Any]:
    normalized_document = normalize_document_taxonomy(dict(document or {}))
    normalized_document["questions"] = _normalize_question_review_questions(normalized_document.get("questions"))
    normalized_document["num_questions"] = len(normalized_document["questions"])
    normalized_document[QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD] = _normalize_question_review_coverage_exemptions(
        normalized_document.get(QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD)
    )
    return normalized_document


def _question_review_document_has_sampleable_questions(document: dict[str, Any] | None) -> bool:
    if not isinstance(document, dict):
        return False
    normalized_document = _normalize_question_review_document(document)
    questions = _normalize_question_review_questions(normalized_document.get("questions"))
    return len(questions) > 0


def _question_review_snapshot_has_sampleable_questions(path_like: Any) -> bool:
    path = _resolve_review_storage_path(path_like)
    if path is None or not path.exists():
        return False
    try:
        payload = _load_yaml(path)
    except Exception:
        return False
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    return _question_review_document_has_sampleable_questions(document)


@lru_cache(maxsize=512)
def _cached_question_review_snapshot_validation_error(resolved_path: str, mtime_ns: int) -> str | None:
    _ = mtime_ns
    path = Path(resolved_path)
    payload = _load_yaml(path)
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    if not isinstance(document, dict):
        document = {}
    try:
        _validate_question_review_submission_contract(_normalize_question_review_document(document))
    except Exception as exc:
        return str(exc)
    return None


@lru_cache(maxsize=512)
def _cached_question_review_snapshot_ignored_count(resolved_path: str, mtime_ns: int) -> int:
    _ = mtime_ns
    path = Path(resolved_path)
    payload = _load_yaml(path)
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    if not isinstance(document, dict):
        document = {}
    normalized_document = _normalize_question_review_document(document)
    exemptions = normalized_document.get(QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD)
    if not isinstance(exemptions, list):
        return 0
    return len(exemptions)


def _question_review_snapshot_validation_error(raw_path: Any) -> str | None:
    resolved_path = _resolve_review_storage_path(raw_path)
    if resolved_path is None or not resolved_path.exists():
        return "Questions submission final snapshot is missing."
    try:
        stat = resolved_path.stat()
    except OSError:
        return "Questions submission final snapshot is unreadable."
    return _cached_question_review_snapshot_validation_error(str(resolved_path), int(stat.st_mtime_ns))


def _question_review_snapshot_is_valid(raw_path: Any) -> bool:
    return _question_review_snapshot_validation_error(raw_path) is None


def _load_fictionalized_question_document_text(theme: str, doc_id: str) -> str:
    canonical_theme = yaml_service.canonical_theme_id(str(theme))
    candidate_paths: list[Path] = []

    template_candidates = {
        canonical_theme,
        str(theme).strip(),
    }
    if canonical_theme == yaml_service.PUBLIC_ATTACKS_THEME:
        template_candidates.add(yaml_service.LEGACY_THEME)
    if canonical_theme == yaml_service.LEGACY_THEME:
        template_candidates.add(yaml_service.PUBLIC_ATTACKS_THEME)

    for theme_candidate in sorted(template_candidates):
        if not theme_candidate:
            continue
        candidate_paths.append(HUMAN_ANNOTATED_TEMPLATES_DIR / theme_candidate / f"{doc_id}.yaml")
    # Prefer canonical template payloads first (they consistently carry the
    # fictionalized surface), then fall back to synced review/admin files.
    candidate_paths.append(review_source_path("questions", canonical_theme, str(doc_id)))
    candidate_paths.append(yaml_service._user_doc_path("admin", canonical_theme, str(doc_id)))

    for path in candidate_paths:
        resolved = _resolve_review_storage_path(path)
        if resolved is None:
            continue
        if not resolved.exists():
            try:
                restore_work_file_from_gcs(resolved)
            except Exception:
                pass
        if not resolved.exists():
            continue
        try:
            payload = _load_yaml(resolved)
        except Exception:
            continue
        document = payload.get("document", payload if isinstance(payload, dict) else {})
        if not isinstance(document, dict):
            continue
        fictionalized_text = str(document.get("fictionalized_annotated_template_document") or "").strip()
        if fictionalized_text:
            return fictionalized_text
    return ""


def _load_factual_question_document_text(theme: str, doc_id: str) -> str:
    canonical_theme = yaml_service.canonical_theme_id(str(theme))
    candidate_paths: list[Path] = []

    template_candidates = {
        canonical_theme,
        str(theme).strip(),
    }
    if canonical_theme == yaml_service.PUBLIC_ATTACKS_THEME:
        template_candidates.add(yaml_service.LEGACY_THEME)
    if canonical_theme == yaml_service.LEGACY_THEME:
        template_candidates.add(yaml_service.PUBLIC_ATTACKS_THEME)

    for theme_candidate in sorted(template_candidates):
        if not theme_candidate:
            continue
        candidate_paths.append(HUMAN_ANNOTATED_TEMPLATES_DIR / theme_candidate / f"{doc_id}.yaml")

    for path in candidate_paths:
        resolved = _resolve_review_storage_path(path)
        if resolved is None or not resolved.exists():
            continue
        try:
            payload = _load_yaml(resolved)
        except Exception:
            continue
        document = payload.get("document", payload if isinstance(payload, dict) else {})
        if not isinstance(document, dict):
            continue
        factual_text = str(document.get("document_to_annotate") or "").strip()
        if factual_text:
            return factual_text
    return ""


def _question_review_snapshot_ignored_count(raw_path: Any) -> int:
    resolved_path = _resolve_review_storage_path(raw_path)
    if resolved_path is None:
        return 0
    if not resolved_path.exists():
        try:
            restore_work_file_from_gcs(resolved_path)
        except Exception:
            pass
    if not resolved_path.exists():
        return 0
    try:
        stat = resolved_path.stat()
    except OSError:
        return 0
    try:
        return int(_cached_question_review_snapshot_ignored_count(str(resolved_path), int(stat.st_mtime_ns)) or 0)
    except Exception:
        return 0


def _is_excluded_from_review_campaign(review_type: str, theme: str) -> bool:
    normalized_type = _validate_review_type(review_type)
    canonical_theme = yaml_service.canonical_theme_id(theme)
    if normalized_type == "rules":
        return canonical_theme in RULE_REVIEW_EXCLUDED_THEMES
    if normalized_type == "questions":
        return canonical_theme in QUESTION_REVIEW_EXCLUDED_THEMES
    return False


def _is_excluded_from_review_campaign_doc(review_type: str, theme: str, doc_id: str) -> bool:
    if _is_excluded_from_review_campaign(review_type, theme):
        return True
    return yaml_service.is_excluded_document(theme, doc_id)


def _review_campaign_completed_excluded_task_count(
    review_type: str,
    task_rows: list[Any],
    agreement_map: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> int:
    """Count completed tasks from excluded themes that should still count in dashboard progress.

    QA excludes public attacks from the agreement workflow, but those documents can still be
    legitimately completed and should contribute to the dashboard's completed-doc count.
    """
    normalized_type = _validate_review_type(review_type)
    if not task_rows:
        return 0

    agreement_map = agreement_map or {}
    completed_keys: set[tuple[str, str]] = set()
    for row in task_rows:
        theme = str(row["theme"] or "").strip()
        doc_id = str(row["doc_id"] or "").strip()
        status = str(row["status"] or "").strip().lower()
        if not theme or not doc_id or status != "completed":
            continue
        if not _is_excluded_from_review_campaign(normalized_type, theme):
            continue
        key = (theme, doc_id)
        agreement = agreement_map.get(key) or {}
        agreement_status = str(agreement.get("status") or "").strip().lower()
        if agreement_status in {"resolved", "awaiting_reviewer_acceptance"}:
            continue
        completed_keys.add((yaml_service.canonical_theme_id(theme), doc_id))
    return len(completed_keys)


def _legacy_rules_workflow_resolution(theme: str, doc_id: str, db) -> dict[str, Any] | None:
    row = db.execute(
        """
        SELECT
            a.id,
            a.final_snapshot_path,
            a.resolved_at,
            a.resolved_by_user_id,
            t1.assignee_user_id AS reviewer_a_user_id,
            t1.output_snapshot_path AS reviewer_a_output_snapshot_path,
            t1.completed_at AS reviewer_a_completed_at,
            t2.assignee_user_id AS reviewer_b_user_id,
            t2.output_snapshot_path AS reviewer_b_output_snapshot_path,
            t2.completed_at AS reviewer_b_completed_at
        FROM workflow_agreements a
        LEFT JOIN workflow_tasks t1 ON t1.id = a.reviewer_a_task_id
        LEFT JOIN workflow_tasks t2 ON t2.id = a.reviewer_b_task_id
        WHERE a.theme = ?
          AND a.doc_id = ?
          AND a.status = 'resolved'
        ORDER BY COALESCE(a.resolved_at, a.updated_at) DESC, a.id DESC
        LIMIT 1
        """,
        (str(theme), str(doc_id)),
    ).fetchone()
    if row is None:
        return None

    from web.services import workflow_service

    row_dict = dict(row)
    return {
        "final_snapshot_path": workflow_service.resolve_workflow_storage_path(row_dict.get("final_snapshot_path")),
        "resolved_at": str(row_dict.get("resolved_at") or "").strip() or None,
        "resolved_by_user_id": (
            int(row_dict["resolved_by_user_id"])
            if row_dict.get("resolved_by_user_id") is not None
            else None
        ),
        "reviewer_a_user_id": (
            int(row_dict["reviewer_a_user_id"])
            if row_dict.get("reviewer_a_user_id") is not None
            else None
        ),
        "reviewer_a_snapshot_path": workflow_service.resolve_workflow_storage_path(
            row_dict.get("reviewer_a_output_snapshot_path")
        ),
        "reviewer_a_completed_at": str(row_dict.get("reviewer_a_completed_at") or "").strip() or None,
        "reviewer_b_user_id": (
            int(row_dict["reviewer_b_user_id"])
            if row_dict.get("reviewer_b_user_id") is not None
            else None
        ),
        "reviewer_b_snapshot_path": workflow_service.resolve_workflow_storage_path(
            row_dict.get("reviewer_b_output_snapshot_path")
        ),
        "reviewer_b_completed_at": str(row_dict.get("reviewer_b_completed_at") or "").strip() or None,
    }


def _restore_rules_review_from_legacy_workflow(
    *,
    campaign_id: int,
    theme: str,
    doc_id: str,
    task_id: int,
    db,
) -> dict[str, Any] | None:
    legacy = _legacy_rules_workflow_resolution(theme, doc_id, db)
    if legacy is None:
        return None

    now = _utc_now()
    inserted = 0
    existing_submissions = _get_review_campaign_submissions(int(campaign_id), str(theme), str(doc_id), db)
    existing_user_ids = {
        int(submission["reviewer_user_id"])
        for submission in existing_submissions
        if submission.get("reviewer_user_id") is not None
    }
    for reviewer_key in ("a", "b"):
        reviewer_user_id = legacy.get(f"reviewer_{reviewer_key}_user_id")
        snapshot_path = legacy.get(f"reviewer_{reviewer_key}_snapshot_path")
        submitted_at = legacy.get(f"reviewer_{reviewer_key}_completed_at") or legacy.get("resolved_at") or now
        if reviewer_user_id is None or snapshot_path is None:
            continue
        if int(reviewer_user_id) in existing_user_ids or not Path(snapshot_path).exists():
            continue
        db.execute(
            """
            INSERT INTO review_campaign_submissions (
                campaign_id,
                review_type,
                theme,
                doc_id,
                reviewer_user_id,
                task_id,
                snapshot_path,
                submitted_at,
                updated_at
            )
            VALUES (?, 'rules', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(campaign_id),
                str(theme),
                str(doc_id),
                int(reviewer_user_id),
                int(task_id),
                str(snapshot_path),
                str(submitted_at),
                now,
            ),
        )
        existing_user_ids.add(int(reviewer_user_id))
        inserted += 1

    submissions = _get_review_campaign_submissions(int(campaign_id), str(theme), str(doc_id), db)
    if len(submissions) < _required_review_submissions("rules"):
        return {"backfilled_submissions": inserted, "restored": False}

    agreement = _upsert_review_campaign_agreement_state(
        campaign_id=int(campaign_id),
        review_type="rules",
        theme=str(theme),
        doc_id=str(doc_id),
        db=db,
    )
    final_snapshot_path = legacy.get("final_snapshot_path")
    latest_snapshot_path = str(final_snapshot_path or submissions[-1]["snapshot_path"])
    db.execute(
        """
        UPDATE review_campaign_agreements
        SET status = 'resolved',
            final_snapshot_path = ?,
            resolved_by_user_id = COALESCE(resolved_by_user_id, ?),
            resolved_at = COALESCE(resolved_at, ?),
            requires_reviewer_acceptance = 0,
            updated_at = ?
        WHERE campaign_id = ?
          AND review_type = 'rules'
          AND theme = ?
          AND doc_id = ?
        """,
        (
            str(final_snapshot_path) if final_snapshot_path is not None else None,
            legacy.get("resolved_by_user_id"),
            legacy.get("resolved_at") or now,
            now,
            int(campaign_id),
            str(theme),
            str(doc_id),
        ),
    )
    db.execute(
        """
        UPDATE review_campaign_tasks
        SET status = 'completed',
            output_snapshot_path = NULL,
            completed_at = COALESCE(completed_at, ?),
            updated_at = ?
        WHERE id = ?
        """,
        (legacy.get("resolved_at") or now, now, int(task_id)),
    )
    _upsert_artifact_status(
        review_type="rules",
        theme=str(theme),
        doc_id=str(doc_id),
        status="completed",
        latest_task_id=int(task_id),
        latest_snapshot_path=latest_snapshot_path,
        db=db,
    )
    return {"backfilled_submissions": inserted, "restored": True}


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True, width=10000)
    _cached_question_review_snapshot_validation_error.cache_clear()
    sync_work_file_to_gcs(path)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    _cached_question_review_snapshot_validation_error.cache_clear()
    sync_work_file_to_gcs(dst)


def _storage_path_ref(path: Path | str) -> str:
    raw = Path(path)
    try:
        return f"/annotation_workspace/{raw.relative_to(yaml_service.WORK_DIR).as_posix()}"
    except ValueError:
        return str(raw)


def _resolve_review_storage_path(raw_path: Any) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None

    direct = Path(raw)
    if direct.exists():
        return direct

    workspace_marker = "/annotation_workspace/"
    if workspace_marker in raw:
        suffix = raw.split(workspace_marker, 1)[1].lstrip("/")
        return yaml_service.WORK_DIR / suffix

    legacy_workspace_prefix = "/app/data/diversified_theme_annotations/"
    if raw.startswith(legacy_workspace_prefix):
        suffix = raw[len(legacy_workspace_prefix):].lstrip("/")
        return yaml_service.WORK_DIR / suffix

    return direct


def _publish_review_fields_to_admin_workspace(
    review_type: str,
    *,
    theme: str,
    doc_id: str,
    template_payload: dict[str, Any],
) -> bool:
    normalized_type = _validate_review_type(review_type)
    template_document = template_payload.get("document")
    if not isinstance(template_document, dict):
        return False

    admin_path = yaml_service._user_doc_path("admin", yaml_service.canonical_theme_id(theme), doc_id)
    if admin_path.exists():
        admin_payload = _load_yaml(admin_path)
    else:
        admin_payload = {}

    admin_document = admin_payload.get("document")
    if not isinstance(admin_document, dict):
        admin_payload = yaml.safe_load(yaml.safe_dump(template_payload, sort_keys=False, allow_unicode=True, width=10000)) or {}
        admin_document = admin_payload.get("document")
        if not isinstance(admin_document, dict):
            return False

    if normalized_type == "rules":
        for field_name in (
            "document_id",
            "document_theme",
            "original_document",
            "document_to_annotate",
            "fictionalized_annotated_template_document",
        ):
            if field_name in template_document:
                admin_document[field_name] = template_document.get(field_name)
        if "relations" in template_document:
            admin_document["relations"] = list(template_document.get("relations") or [])
        admin_document["rules"] = list(template_document.get("rules") or [])
        admin_document["implicit_rules"] = list(template_document.get("implicit_rules") or [])
        if "implicit_rule_exclusions" in template_document:
            admin_document["implicit_rule_exclusions"] = list(template_document.get("implicit_rule_exclusions") or [])
        else:
            admin_document.pop("implicit_rule_exclusions", None)
    else:
        questions = list(template_document.get("questions") or [])
        admin_document["questions"] = questions
        admin_document["num_questions"] = len(questions)

    _write_yaml(admin_path, admin_payload)
    return True


def _publish_review_fields_to_latest_final_snapshot(
    review_type: str,
    *,
    theme: str,
    doc_id: str,
    template_payload: dict[str, Any],
) -> bool:
    normalized_type = _validate_review_type(review_type)
    template_document = template_payload.get("document")
    if not isinstance(template_document, dict):
        return False

    # Import lazily to avoid circular imports at module load.
    from web.services import workflow_service

    final_document: dict[str, Any] | None = None
    persisted_via_workflow = False
    fallback_snapshot_path: Path | None = None
    fallback_snapshot_payload: dict[str, Any] | None = None

    agreement = workflow_service.get_latest_agreement_record(theme, doc_id)
    if agreement and str(agreement.get("status") or "").strip().lower() == "resolved":
        latest_final = workflow_service.load_latest_final_snapshot_document(theme, doc_id)
        if isinstance(latest_final, dict) and latest_final:
            final_document = yaml.safe_load(
                yaml.safe_dump(latest_final, sort_keys=False, allow_unicode=True, width=10000)
            ) or {}
            persisted_via_workflow = True

    if final_document is None:
        db = get_db()
        row = db.execute(
            """
            SELECT latest_snapshot_path
            FROM review_artifact_statuses
            WHERE review_type = ?
              AND theme = ?
              AND doc_id = ?
              AND status = 'completed'
            ORDER BY
                CASE WHEN reviewed_at IS NULL THEN 1 ELSE 0 END ASC,
                reviewed_at DESC
            LIMIT 1
            """,
            (normalized_type, yaml_service.canonical_theme_id(theme), str(doc_id)),
        ).fetchone()
        raw_snapshot_path = str(row["latest_snapshot_path"] or "").strip() if row else ""
        resolved_snapshot_path = _resolve_review_storage_path(raw_snapshot_path) if raw_snapshot_path else None
        if resolved_snapshot_path and not resolved_snapshot_path.exists():
            restore_work_file_from_gcs(resolved_snapshot_path)
        if not resolved_snapshot_path or not resolved_snapshot_path.exists():
            return False
        payload = _load_yaml(resolved_snapshot_path)
        snapshot_document = payload.get("document")
        if not isinstance(snapshot_document, dict):
            return False
        final_document = yaml.safe_load(
            yaml.safe_dump(snapshot_document, sort_keys=False, allow_unicode=True, width=10000)
        ) or {}
        fallback_snapshot_path = resolved_snapshot_path
        fallback_snapshot_payload = payload

    for field_name in (
        "document_id",
        "document_theme",
        "original_document",
        "document_to_annotate",
        "fictionalized_annotated_template_document",
    ):
        if field_name in template_document:
            final_document[field_name] = template_document.get(field_name)
    if "relations" in template_document:
        final_document["relations"] = list(template_document.get("relations") or [])

    if normalized_type == "rules":
        final_document["rules"] = list(template_document.get("rules") or [])
        final_document["implicit_rules"] = list(template_document.get("implicit_rules") or [])
        if "implicit_rule_exclusions" in template_document:
            final_document["implicit_rule_exclusions"] = list(template_document.get("implicit_rule_exclusions") or [])
        else:
            final_document.pop("implicit_rule_exclusions", None)
    else:
        if "questions" in template_document:
            questions = list(template_document.get("questions") or [])
            final_document["questions"] = questions
            final_document["num_questions"] = len(questions)
        elif isinstance(final_document.get("questions"), list):
            final_document["num_questions"] = len(final_document.get("questions") or [])
        _align_question_annotation_surfaces_with_document(final_document)

    if persisted_via_workflow:
        workflow_service.save_latest_final_snapshot_from_document(
            theme,
            doc_id,
            final_document,
            source_label=f"review_publish_{normalized_type}",
        )
    else:
        if fallback_snapshot_path is None:
            return False
        payload_to_write = fallback_snapshot_payload if isinstance(fallback_snapshot_payload, dict) else {}
        payload_to_write["document"] = final_document
        _write_yaml(fallback_snapshot_path, payload_to_write)
    return True


def _refresh_active_available_review_task_inputs(
    review_type: str,
    *,
    theme: str,
    doc_id: str,
    template_payload: dict[str, Any],
) -> int:
    """Refresh active review-campaign input snapshots for zero-submission tasks.

    This keeps newly pushed rules/questions visible to reviewers without overriding
    active reviewer outputs or invalidating an already-submitted first pass.
    """
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    rows = db.execute(
        """
        SELECT t.id, t.campaign_id, t.assignee_user_id
        FROM review_campaign_tasks t
        JOIN review_campaigns c ON c.id = t.campaign_id
        WHERE c.review_type = ?
          AND c.status = 'active'
          AND t.review_type = ?
          AND t.theme = ?
          AND t.doc_id = ?
          AND t.status = 'available'
          AND NOT EXISTS (
              SELECT 1
              FROM review_campaign_submissions s
              WHERE s.campaign_id = t.campaign_id
                AND s.theme = t.theme
                AND s.doc_id = t.doc_id
          )
        """,
        (normalized_type, normalized_type, str(theme), str(doc_id)),
    ).fetchall()

    if not rows:
        return 0

    now = _utc_now()
    updated = 0
    for row in rows:
        campaign_id = int(row["campaign_id"])
        user_id = int(row["assignee_user_id"])
        input_path = _campaign_input_path(campaign_id, normalized_type, str(theme), str(doc_id))
        _write_yaml(input_path, template_payload)

        output_path = _campaign_output_path(campaign_id, normalized_type, user_id, str(theme), str(doc_id))
        if output_path.exists():
            _write_yaml(output_path, template_payload)
            output_snapshot_path: str | None = str(output_path)
        else:
            output_snapshot_path = None

        db.execute(
            """
            UPDATE review_campaign_tasks
            SET input_snapshot_path = ?,
                output_snapshot_path = COALESCE(?, output_snapshot_path),
                updated_at = ?
            WHERE id = ?
            """,
            (str(input_path), output_snapshot_path, now, int(row["id"])),
        )
        updated += 1

    db.commit()
    sync_db_to_gcs()
    return updated


def sync_document_annotations_to_review_sources(
    *,
    theme: str,
    doc_id: str,
    document: dict[str, Any],
) -> dict[str, Any]:
    """Propagate document-annotation edits into Rules/QA review source snapshots.

    The document section is the authoritative place for inline entity refs.
    This helper keeps review source YAMLs aligned so opening Rules/QA sections
    reflects the latest document-level taxonomy edits immediately.
    """
    if not isinstance(document, dict):
        return {
            "theme": yaml_service.canonical_theme_id(str(theme)),
            "doc_id": str(doc_id),
            "updated_review_sources": 0,
            "updated_review_types": [],
            "refreshed_campaign_inputs": 0,
        }

    canonical_theme = yaml_service.canonical_theme_id(str(theme))
    canonical_doc_id = str(doc_id)
    authoritative_document = _canonicalize_mergeable_equality_rules(dict(document))

    updated_review_types: list[str] = []
    refreshed_campaign_inputs = 0
    published_final_snapshots = 0

    for review_type in REVIEW_TYPES:
        source_path = review_source_path(review_type, canonical_theme, canonical_doc_id)
        source_missing = False
        if not source_path.exists():
            restore_work_file_from_gcs(source_path)
        if not source_path.exists():
            source_missing = True

        if source_missing:
            source_payload: dict[str, Any] = {}
            latest_final_document: dict[str, Any] | None = None
            try:
                # Import lazily to avoid circular imports at module load.
                from web.services import workflow_service

                latest_final_raw = workflow_service.load_latest_final_snapshot_document(canonical_theme, canonical_doc_id)
                if isinstance(latest_final_raw, dict) and latest_final_raw:
                    latest_final_document = yaml.safe_load(
                        yaml.safe_dump(latest_final_raw, sort_keys=False, allow_unicode=True, width=10000)
                    ) or {}
            except Exception:
                latest_final_document = None
            source_document = latest_final_document if isinstance(latest_final_document, dict) else {}
            if source_document:
                source_payload["document"] = source_document
        else:
            source_payload = _load_yaml(source_path)
            source_document = source_payload.get("document", source_payload if isinstance(source_payload, dict) else {})
        if not isinstance(source_document, dict):
            source_document = {}

        merged_document = dict(source_document)
        for field_name in (
            "document_id",
            "document_theme",
            "original_document",
            "document_to_annotate",
            "fictionalized_annotated_template_document",
        ):
            if field_name in authoritative_document:
                merged_document[field_name] = authoritative_document.get(field_name)
        if "relations" in authoritative_document:
            merged_document["relations"] = list(authoritative_document.get("relations") or [])
        if review_type == "rules":
            authoritative_rules = authoritative_document.get("rules")
            existing_rules = source_document.get("rules")
            if isinstance(authoritative_rules, list) and authoritative_rules:
                merged_document["rules"] = list(authoritative_rules)
            elif isinstance(existing_rules, list):
                merged_document["rules"] = list(existing_rules)
            if "implicit_rules" in authoritative_document:
                merged_document["implicit_rules"] = list(authoritative_document.get("implicit_rules") or [])
            if "implicit_rule_exclusions" in authoritative_document:
                merged_document["implicit_rule_exclusions"] = list(
                    authoritative_document.get("implicit_rule_exclusions") or []
                )
            else:
                merged_document.pop("implicit_rule_exclusions", None)
        if review_type == "questions":
            authoritative_questions = authoritative_document.get("questions")
            existing_questions = source_document.get("questions")
            if isinstance(authoritative_questions, list) and len(authoritative_questions) > 0:
                merged_document["questions"] = list(authoritative_questions)
                merged_document["num_questions"] = len(merged_document.get("questions") or [])
            elif isinstance(existing_questions, list):
                # Document-annotation saves should not wipe an existing QA set.
                # Keep the synced question review source authoritative for QAs
                # unless the incoming document explicitly carries a non-empty list.
                merged_document["questions"] = list(existing_questions)
                merged_document["num_questions"] = len(merged_document.get("questions") or [])
            existing_exemptions = source_document.get(QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD)
            if isinstance(existing_exemptions, list):
                merged_document[QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD] = list(existing_exemptions)
        _rewrite_cross_section_entity_refs_to_document(merged_document)
        if review_type == "questions":
            _align_question_annotation_surfaces_with_document(merged_document)

        normalized_merged_document = _canonicalize_mergeable_equality_rules(merged_document)
        if review_type == "questions":
            _align_question_annotation_surfaces_with_document(normalized_merged_document)
        candidate_payload = yaml.safe_load(
            yaml.safe_dump(source_payload, sort_keys=False, allow_unicode=True, width=10000)
        ) or {}
        if not isinstance(candidate_payload, dict):
            candidate_payload = {}
        candidate_payload["document"] = normalized_merged_document
        prepared_payload = _prepare_template_payload_for_review(review_type, candidate_payload)

        if _payloads_equal(source_payload, prepared_payload):
            try:
                if _publish_review_fields_to_latest_final_snapshot(
                    review_type,
                    theme=canonical_theme,
                    doc_id=canonical_doc_id,
                    template_payload=prepared_payload,
                ):
                    published_final_snapshots += 1
            except Exception:
                # Keep document saves resilient if workflow/final-snapshot linkage is unavailable.
                pass
            continue

        _write_yaml(source_path, prepared_payload)
        updated_review_types.append(review_type)
        try:
            refreshed_campaign_inputs += _refresh_active_available_review_task_inputs(
                review_type,
                theme=canonical_theme,
                doc_id=canonical_doc_id,
                template_payload=prepared_payload,
            )
        except Exception:
            # Keep document saves resilient even when campaign refresh metadata is stale.
            pass
        try:
            if _publish_review_fields_to_latest_final_snapshot(
                review_type,
                theme=canonical_theme,
                doc_id=canonical_doc_id,
                template_payload=prepared_payload,
            ):
                published_final_snapshots += 1
        except Exception:
            # Keep document saves resilient if workflow/final-snapshot linkage is unavailable.
            pass

    return {
        "theme": canonical_theme,
        "doc_id": canonical_doc_id,
        "updated_review_sources": len(updated_review_types),
        "updated_review_types": updated_review_types,
        "refreshed_campaign_inputs": int(refreshed_campaign_inputs),
        "published_final_snapshots": int(published_final_snapshots),
    }


def _payloads_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare task payloads with stable YAML normalization."""
    left_norm = yaml.safe_load(yaml.safe_dump(left or {}, sort_keys=True, allow_unicode=True, width=10000)) or {}
    right_norm = yaml.safe_load(yaml.safe_dump(right or {}, sort_keys=True, allow_unicode=True, width=10000)) or {}
    return left_norm == right_norm


def _refresh_active_untouched_in_progress_review_task_inputs(
    review_type: str,
    *,
    theme: str,
    doc_id: str,
    template_payload: dict[str, Any],
) -> dict[str, int]:
    """Refresh untouched in-progress tasks before any review submission exists.

    A task is considered untouched when its current output snapshot equals its input snapshot.
    This avoids overwriting any reviewer edits while still propagating updated campaign sources.
    """
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    rows = db.execute(
        """
        SELECT t.id, t.campaign_id, t.assignee_user_id, t.input_snapshot_path, t.output_snapshot_path
        FROM review_campaign_tasks t
        JOIN review_campaigns c ON c.id = t.campaign_id
        WHERE c.review_type = ?
          AND c.status = 'active'
          AND t.review_type = ?
          AND t.theme = ?
          AND t.doc_id = ?
          AND t.status = 'in_progress'
          AND NOT EXISTS (
              SELECT 1
              FROM review_campaign_submissions s
              WHERE s.campaign_id = t.campaign_id
                AND s.theme = t.theme
                AND s.doc_id = t.doc_id
          )
        """,
        (normalized_type, normalized_type, str(theme), str(doc_id)),
    ).fetchall()

    if not rows:
        return {"refreshed": 0, "skipped_protected": 0}

    now = _utc_now()
    refreshed = 0
    skipped_protected = 0
    for row in rows:
        task_id = int(row["id"])
        campaign_id = int(row["campaign_id"])
        user_id = int(row["assignee_user_id"])

        input_path = _campaign_input_path(campaign_id, normalized_type, str(theme), str(doc_id))
        output_path = _campaign_output_path(campaign_id, normalized_type, user_id, str(theme), str(doc_id))

        current_input_path = _resolve_review_storage_path(row["input_snapshot_path"]) or input_path
        current_output_path = _resolve_review_storage_path(row["output_snapshot_path"]) or output_path

        current_input_payload = _load_yaml(current_input_path) if current_input_path.exists() else {}
        current_output_payload = _load_yaml(current_output_path) if current_output_path.exists() else {}

        # If reviewer modified output, protect it and skip refresh.
        if current_output_payload and not _payloads_equal(current_output_payload, current_input_payload):
            skipped_protected += 1
            continue

        _write_yaml(input_path, template_payload)
        _write_yaml(output_path, template_payload)
        db.execute(
            """
            UPDATE review_campaign_tasks
            SET input_snapshot_path = ?,
                output_snapshot_path = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (str(input_path), str(output_path), now, task_id),
        )
        refreshed += 1

    if refreshed > 0:
        db.commit()
        sync_db_to_gcs()
    return {"refreshed": refreshed, "skipped_protected": skipped_protected}


def _add_document_to_active_review_campaign_if_missing(
    review_type: str,
    *,
    theme: str,
    doc_id: str,
    template_payload: dict[str, Any],
) -> int:
    """Add a new review task to the active campaign when this document is not yet queued."""
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return 0

    existing = db.execute(
        """
        SELECT id
        FROM review_campaign_tasks t
        WHERE campaign_id = ?
          AND review_type = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (int(campaign["id"]), normalized_type, str(theme), str(doc_id)),
    ).fetchone()
    if existing is not None:
        return 0

    now = _utc_now()
    input_path = _campaign_input_path(int(campaign["id"]), normalized_type, str(theme), str(doc_id))
    _write_yaml(input_path, template_payload)

    task_cursor = db.execute(
        """
        INSERT INTO review_campaign_tasks (
            campaign_id,
            review_type,
            theme,
            doc_id,
            assignee_user_id,
            status,
            input_snapshot_path,
            output_snapshot_path,
            assigned_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, 'available', ?, NULL, ?, ?)
        """,
        (
            int(campaign["id"]),
            normalized_type,
            str(theme),
            str(doc_id),
            int(campaign["created_by_user_id"]),
            str(input_path),
            now,
            now,
        ),
    )
    _upsert_artifact_status(
        review_type=normalized_type,
        theme=str(theme),
        doc_id=str(doc_id),
        status="draft",
        latest_task_id=int(task_cursor.lastrowid),
        latest_snapshot_path=str(input_path),
        db=db,
    )
    db.commit()
    sync_db_to_gcs()
    return 1


def review_source_path(review_type: str, theme: str, doc_id: str) -> Path:
    normalized_type = _validate_review_type(review_type)
    return REVIEW_SOURCE_ROOT / normalized_type / str(theme) / f"{doc_id}.yaml"


def _campaign_input_path(campaign_id: int, review_type: str, theme: str, doc_id: str) -> Path:
    normalized_type = _validate_review_type(review_type)
    return REVIEW_CAMPAIGN_ROOT / normalized_type / f"campaign_{campaign_id}" / "input" / theme / f"{doc_id}.yaml"


def _campaign_output_path(campaign_id: int, review_type: str, user_id: int, theme: str, doc_id: str) -> Path:
    normalized_type = _validate_review_type(review_type)
    return (
        REVIEW_CAMPAIGN_ROOT
        / normalized_type
        / f"campaign_{campaign_id}"
        / "work"
        / f"user_{user_id}"
        / theme
        / f"{doc_id}.yaml"
    )


def _review_agreement_final_path(campaign_id: int, review_type: str, theme: str, doc_id: str) -> Path:
    normalized_type = _validate_review_type(review_type)
    return (
        REVIEW_CAMPAIGN_ROOT
        / normalized_type
        / f"campaign_{campaign_id}"
        / "agreements"
        / "final"
        / str(theme)
        / f"{doc_id}.yaml"
    )


def _question_experiment_task_rows_for_user(campaign_id: int, user_id: int, db) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT
            t.id,
            t.theme,
            t.doc_id,
            t.status,
            t.assignee_user_id,
            t.initial_assignee_user_id,
            t.input_snapshot_path,
            t.output_snapshot_path,
            t.qa_group,
            t.qa_group_order,
            CASE WHEN s.id IS NULL THEN 0 ELSE 1 END AS reviewer_submitted,
            s.snapshot_path AS reviewer_snapshot_path,
            COALESCE(submissions.submission_count, 0) AS submission_count
        FROM review_campaign_tasks t
        LEFT JOIN review_campaign_submissions s
               ON s.campaign_id = t.campaign_id
              AND s.theme = t.theme
              AND s.doc_id = t.doc_id
              AND s.reviewer_user_id = ?
        LEFT JOIN (
            SELECT campaign_id, theme, doc_id, COUNT(*) AS submission_count
            FROM review_campaign_submissions
            GROUP BY campaign_id, theme, doc_id
        ) submissions
               ON submissions.campaign_id = t.campaign_id
              AND submissions.theme = t.theme
              AND submissions.doc_id = t.doc_id
        WHERE t.campaign_id = ?
          AND t.review_type = 'questions'
          AND t.qa_group IS NOT NULL
          AND t.initial_assignee_user_id = ?
        ORDER BY
            CASE t.qa_group
                WHEN 'ai_drafted_qas' THEN 0
                WHEN 'no_qas' THEN 1
                ELSE 2
            END ASC,
            COALESCE(t.qa_group_order, 999999) ASC,
            t.id ASC
        """,
        (int(user_id), int(campaign_id), int(user_id)),
    ).fetchall()
    return [
        dict(row)
        for row in rows
        if not _is_excluded_from_review_campaign_doc(
            "questions",
            str(row["theme"] or ""),
            str(row["doc_id"] or ""),
        )
    ]


def _build_question_experiment_groups(campaign_id: int, user_id: int, db) -> list[dict[str, Any]]:
    rows = _question_experiment_task_rows_for_user(int(campaign_id), int(user_id), db)
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        group_key = _normalize_question_experiment_group(row.get("qa_group"))
        if not group_key:
            continue
        bucket = grouped.setdefault(
            group_key,
            {
                "key": group_key,
                "label": QUESTION_EXPERIMENT_GROUP_LABELS[group_key],
                "items": [],
            },
        )
        reviewer_submitted = bool(int(row.get("reviewer_submitted") or 0))
        live_status = str(row.get("status") or "").strip().lower()
        current_assignee_user_id = int(row.get("assignee_user_id") or 0)
        if reviewer_submitted:
            display_status = "submitted"
        elif live_status == "in_progress" and current_assignee_user_id == int(user_id):
            display_status = "in_progress"
        elif live_status == "available" and current_assignee_user_id == int(user_id):
            display_status = "available"
        else:
            display_status = "waiting"
        ignored_qa_count = _question_review_snapshot_ignored_count(
            row.get("reviewer_snapshot_path")
            or row.get("output_snapshot_path")
            or row.get("input_snapshot_path")
        )
        bucket["items"].append(
            {
                "theme": str(row.get("theme") or ""),
                "doc_id": str(row.get("doc_id") or ""),
                "status": display_status,
                "qa_group": group_key,
                "qa_group_label": QUESTION_EXPERIMENT_GROUP_LABELS[group_key],
                "qa_group_order": int(row.get("qa_group_order") or 0),
                "can_open": display_status in {"available", "in_progress"},
                "ignored_qa_count": ignored_qa_count,
                "has_ignored_qas": ignored_qa_count > 0,
            }
        )

    groups = list(grouped.values())
    groups.sort(key=lambda item: QUESTION_EXPERIMENT_GROUP_SORT_ORDER.get(str(item.get("key") or ""), 999))
    for group in groups:
        group["total_count"] = len(group["items"])
        group["completed_count"] = sum(1 for item in group["items"] if str(item.get("status")) == "submitted")
        group["remaining_count"] = max(0, int(group["total_count"]) - int(group["completed_count"]))
    return groups


def _is_reviewable_template(review_type: str, template_payload: dict[str, Any]) -> bool:
    normalized_type = _validate_review_type(review_type)
    metadata = template_payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    raw_status = str(metadata.get("status") or "").strip().lower()
    if raw_status not in {"completed", "validated"}:
        return False

    document = template_payload.get("document")
    if not isinstance(document, dict):
        return False

    if normalized_type == "rules":
        rules = document.get("rules")
        # A completed document can still be valid for rules review even when
        # the source model produced no explicit rules.
        return isinstance(rules, list)

    return _question_review_document_has_sampleable_questions(document)


def _prepare_template_payload_for_review(review_type: str, template_payload: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized payload ready for review-source publication."""
    normalized_type = _validate_review_type(review_type)
    cloned = yaml.safe_load(yaml.safe_dump(template_payload, sort_keys=False, allow_unicode=True, width=10000)) or {}
    if not isinstance(cloned, dict):
        return template_payload

    document = cloned.get("document")
    if not isinstance(document, dict):
        return cloned

    normalized_document = normalize_document_taxonomy(document)
    if normalized_type == "questions":
        fictionalized_text = str(
            normalized_document.get("fictionalized_annotated_template_document") or ""
        ).strip()
        if fictionalized_text:
            # Question review should use the same fictionalized annotated text that
            # the QA generator used, so the questions and the displayed document stay aligned.
            normalized_document["document_to_annotate"] = fictionalized_text
    cloned["document"] = normalized_document
    return cloned


def _eligible_review_doc_keys(review_type: str, db=None) -> set[tuple[str, str]] | None:
    normalized_type = _validate_review_type(review_type)
    if normalized_type == "questions":
        eligible: set[tuple[str, str]] = set()
        restore_worktree_from_gcs()
        for path in _iter_review_source_files(normalized_type):
            theme = yaml_service.canonical_theme_id(path.parent.name)
            doc_id = path.stem
            if not theme or not doc_id:
                continue
            if _is_excluded_from_review_campaign_doc(normalized_type, theme, doc_id):
                continue
            if _question_review_snapshot_has_sampleable_questions(path):
                eligible.add((theme, doc_id))
        if db is None:
            db = get_db()
        active_campaign = get_active_review_campaign(normalized_type, db=db)
        if active_campaign is not None:
            task_rows = db.execute(
                """
                SELECT theme, doc_id, input_snapshot_path, output_snapshot_path
                FROM review_campaign_tasks
                WHERE campaign_id = ?
                  AND review_type = ?
                  AND status IN ('available', 'in_progress')
                ORDER BY id ASC
                """,
                (int(active_campaign["id"]), normalized_type),
            ).fetchall()
            for row in task_rows:
                theme = yaml_service.canonical_theme_id(str(row["theme"] or ""))
                doc_id = str(row["doc_id"] or "").strip()
                if not theme or not doc_id:
                    continue
                if _is_excluded_from_review_campaign_doc(normalized_type, theme, doc_id):
                    continue
                candidate_path = row["output_snapshot_path"] or row["input_snapshot_path"]
                if _question_review_snapshot_has_sampleable_questions(candidate_path):
                    eligible.add((theme, doc_id))
        return eligible
    if normalized_type != "rules":
        return None
    if db is None:
        db = get_db()

    # Align rules eligibility with the Document Annotation dashboard. Prefer
    # completed docs whenever they can be inferred from progress state; only
    # fall back to permissive behavior when no active run exists and no
    # completed-doc signal is available (isolated local staging/tests).
    from web.services import workflow_service

    has_active_run = workflow_service.get_active_run(db) is not None

    progress = yaml_service.get_theme_progress(db=db)
    themes = progress.get("themes") if isinstance(progress, dict) else None
    if not isinstance(themes, list):
        return None

    completed: set[tuple[str, str]] = set()
    for theme_entry in themes:
        if not isinstance(theme_entry, dict):
            continue
        theme_id = yaml_service.canonical_theme_id(str(theme_entry.get("theme_id") or ""))
        documents = theme_entry.get("documents")
        if not theme_id or not isinstance(documents, list):
            continue
        for document_entry in documents:
            if not isinstance(document_entry, dict):
                continue
            if str(document_entry.get("status") or "").strip().lower() != "completed":
                continue
            doc_id = str(document_entry.get("doc_id") or "").strip()
            if not doc_id:
                continue
            completed.add((theme_id, doc_id))
    if completed:
        return completed

    if not has_active_run:
        return None
    return completed


def _prune_available_review_tasks_outside_eligibility(
    review_type: str,
    eligible_keys: set[tuple[str, str]] | None,
    db=None,
) -> int:
    normalized_type = _validate_review_type(review_type)
    if not eligible_keys:
        return 0
    if db is None:
        db = get_db()

    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return 0

    rows = db.execute(
        """
        SELECT id, theme, doc_id
        FROM review_campaign_tasks
        WHERE campaign_id = ?
          AND review_type = ?
          AND status = 'available'
          AND NOT EXISTS (
              SELECT 1
              FROM review_campaign_submissions s
              WHERE s.campaign_id = review_campaign_tasks.campaign_id
                AND s.theme = review_campaign_tasks.theme
                AND s.doc_id = review_campaign_tasks.doc_id
          )
        """,
        (int(campaign["id"]), normalized_type),
    ).fetchall()

    removed = 0
    now = _utc_now()
    for row in rows:
        key = (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
        if key in eligible_keys:
            continue
        db.execute(
            """
            UPDATE review_artifact_statuses
            SET latest_task_id = NULL,
                status = CASE WHEN status = 'completed' THEN status ELSE 'draft' END,
                updated_at = ?
            WHERE review_type = ?
              AND theme = ?
              AND doc_id = ?
              AND latest_task_id = ?
            """,
            (now, normalized_type, str(row["theme"] or ""), str(row["doc_id"] or ""), int(row["id"])),
        )
        db.execute("DELETE FROM review_campaign_tasks WHERE id = ?", (int(row["id"]),))
        removed += 1

    if removed > 0:
        db.commit()
        sync_db_to_gcs()
    return removed


def prepare_review_source_documents(
    review_type: str,
    *,
    themes: list[str] | None = None,
    doc_ids: list[str] | None = None,
    overwrite: bool = False,
    replace_existing: bool = False,
    respect_workflow_eligibility: bool = True,
) -> dict[str, int]:
    """Copy completed templates into the synced review-source workspace."""
    normalized_type = _validate_review_type(review_type)
    selected = 0
    published = 0
    published_final_snapshots = 0
    refreshed_campaign_inputs = 0
    refreshed_in_progress_campaign_inputs = 0
    protected_in_progress_tasks = 0
    added_to_active_campaign = 0
    pruned_ineligible_available_tasks = 0
    copied = 0
    skipped = 0
    review_root = REVIEW_SOURCE_ROOT / normalized_type

    restore_worktree_from_gcs()
    db = get_db()
    eligible_doc_keys = (
        _eligible_review_doc_keys(normalized_type, db=db)
        if normalized_type != "questions"
        else None
        if respect_workflow_eligibility
        else None
    )
    template_paths = list(iter_template_paths(themes=themes, document_ids=doc_ids))

    if replace_existing:
        if review_root.exists():
            shutil.rmtree(review_root)
        delete_work_prefix_from_gcs(Path("_review_sources") / normalized_type)

    for template_path in template_paths:
        selected += 1
        payload = _load_yaml(template_path)
        theme, document_id = resolve_template_identity(template_path)
        if _is_excluded_from_review_campaign_doc(normalized_type, theme, str(document_id)):
            skipped += 1
            continue
        canonical_key = (yaml_service.canonical_theme_id(theme), str(document_id))
        if eligible_doc_keys is not None and canonical_key not in eligible_doc_keys:
            skipped += 1
            continue
        if not _is_reviewable_template(normalized_type, payload):
            skipped += 1
            continue
        payload = _prepare_template_payload_for_review(normalized_type, payload)
        if _publish_review_fields_to_admin_workspace(
            normalized_type,
            theme=theme,
            doc_id=document_id,
            template_payload=payload,
        ):
            published += 1
        try:
            if _publish_review_fields_to_latest_final_snapshot(
                normalized_type,
                theme=theme,
                doc_id=document_id,
                template_payload=payload,
            ):
                published_final_snapshots += 1
        except Exception:
            # Do not block review-source syncing if workflow snapshot sync fails.
            pass
        try:
            refreshed_campaign_inputs += _refresh_active_available_review_task_inputs(
                normalized_type,
                theme=theme,
                doc_id=document_id,
                template_payload=payload,
            )
        except Exception:
            # Do not block review-source syncing if campaign snapshot refresh fails.
            pass
        try:
            in_progress_refresh = _refresh_active_untouched_in_progress_review_task_inputs(
                normalized_type,
                theme=theme,
                doc_id=document_id,
                template_payload=payload,
            )
            refreshed_in_progress_campaign_inputs += int(in_progress_refresh.get("refreshed", 0))
            protected_in_progress_tasks += int(in_progress_refresh.get("skipped_protected", 0))
        except Exception:
            # Do not block review-source syncing if in-progress task refresh fails.
            pass
        added_to_active_campaign += _add_document_to_active_review_campaign_if_missing(
            normalized_type,
            theme=theme,
            doc_id=document_id,
            template_payload=payload,
        )
        dst_path = review_source_path(normalized_type, theme, document_id)
        if dst_path.exists() and not overwrite:
            skipped += 1
            continue
        _write_yaml(dst_path, payload)
        copied += 1

    # Keep active campaign aligned with current review-source set by pruning
    # stale AVAILABLE tasks that no longer have a review-source file.
    try:
        stale_pruned = _prune_available_tasks_missing_review_sources(normalized_type)
    except Exception:
        stale_pruned = 0
    try:
        pruned_ineligible_available_tasks = _prune_available_review_tasks_outside_eligibility(
            normalized_type,
            eligible_doc_keys,
            db=db,
        )
    except Exception:
        pruned_ineligible_available_tasks = 0

    return {
        "selected": selected,
        "published": published,
        "published_final_snapshots": published_final_snapshots,
        "refreshed_campaign_inputs": refreshed_campaign_inputs,
        "refreshed_in_progress_campaign_inputs": refreshed_in_progress_campaign_inputs,
        "protected_in_progress_tasks": protected_in_progress_tasks,
        "added_to_active_campaign": added_to_active_campaign,
        "stale_available_tasks_pruned": stale_pruned,
        "ineligible_available_tasks_pruned": pruned_ineligible_available_tasks,
        "copied": copied,
        "skipped": skipped,
    }


def _prune_available_tasks_missing_review_sources(review_type: str) -> int:
    """Delete stale AVAILABLE tasks whose review-source file no longer exists.

    Safety: only prunes AVAILABLE tasks to avoid touching active reviewer work.
    """
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return 0

    rows = db.execute(
        """
        SELECT id, theme, doc_id
        FROM review_campaign_tasks
        WHERE campaign_id = ?
          AND review_type = ?
          AND status = 'available'
          AND NOT EXISTS (
              SELECT 1
              FROM review_campaign_submissions s
              WHERE s.campaign_id = review_campaign_tasks.campaign_id
                AND s.theme = review_campaign_tasks.theme
                AND s.doc_id = review_campaign_tasks.doc_id
          )
        """,
        (int(campaign["id"]), normalized_type),
    ).fetchall()
    if not rows:
        return 0

    pruned = 0
    for row in rows:
        theme = str(row["theme"] or "")
        doc_id = str(row["doc_id"] or "")
        source_path = review_source_path(normalized_type, theme, doc_id)
        if source_path.exists():
            continue
        task_id = int(row["id"])
        db.execute(
            """
            UPDATE review_artifact_statuses
            SET latest_task_id = NULL,
                status = CASE WHEN status = 'completed' THEN status ELSE 'draft' END,
                updated_at = ?
            WHERE review_type = ?
              AND theme = ?
              AND doc_id = ?
              AND latest_task_id = ?
            """,
            (_utc_now(), normalized_type, theme, doc_id, task_id),
        )
        db.execute("DELETE FROM review_campaign_tasks WHERE id = ?", (task_id,))
        pruned += 1

    if pruned > 0:
        db.commit()
        sync_db_to_gcs()
    return pruned


def _iter_review_source_files(review_type: str) -> list[Path]:
    normalized_type = _validate_review_type(review_type)
    root = REVIEW_SOURCE_ROOT / normalized_type
    if not root.exists():
        return []
    files: list[Path] = []
    for theme_dir in sorted(root.iterdir()):
        if not theme_dir.is_dir():
            continue
        if _is_excluded_from_review_campaign(normalized_type, theme_dir.name):
            continue
        for yaml_path in sorted(theme_dir.glob("*.yaml")):
            if _is_excluded_from_review_campaign_doc(
                normalized_type,
                theme_dir.name,
                yaml_path.stem,
            ):
                continue
            files.append(yaml_path)
    return files


def _finalize_excluded_review_tasks(review_type: str, db=None) -> int:
    normalized_type = _validate_review_type(review_type)
    if normalized_type != "rules":
        return 0
    if db is None:
        db = get_db()

    rows = db.execute(
        """
        SELECT
            t.id,
            t.campaign_id,
            t.theme,
            t.doc_id,
            t.input_snapshot_path,
            t.output_snapshot_path
        FROM review_campaign_tasks t
        JOIN review_campaigns c ON c.id = t.campaign_id
        WHERE c.review_type = ?
          AND c.status = 'active'
          AND t.status != 'completed'
        """,
        (normalized_type,),
    ).fetchall()

    now = _utc_now()
    updated = 0
    affected_campaign_ids: set[int] = set()
    for row in rows:
        theme = str(row["theme"] or "")
        doc_id = str(row["doc_id"] or "")
        if not _is_excluded_from_review_campaign_doc(normalized_type, theme, doc_id):
            continue

        task_id = int(row["id"])
        campaign_id = int(row["campaign_id"])
        latest_snapshot_path = str(
            row["output_snapshot_path"] or row["input_snapshot_path"] or ""
        ).strip()
        _upsert_artifact_status(
            review_type=normalized_type,
            theme=theme,
            doc_id=doc_id,
            status="completed",
            latest_task_id=None,
            latest_snapshot_path=latest_snapshot_path,
            reviewed_at=now,
            db=db,
        )
        db.execute("DELETE FROM review_campaign_tasks WHERE id = ?", (task_id,))
        affected_campaign_ids.add(campaign_id)
        updated += 1

    if updated == 0:
        return 0

    db.commit()
    sync_db_to_gcs()
    for campaign_id in sorted(affected_campaign_ids):
        _maybe_complete_review_campaign(campaign_id, db)
    return updated


def _resolve_reviewer_user_ids(usernames: list[str] | None, db) -> list[int]:
    if usernames:
        placeholders = ",".join("?" for _ in usernames)
        rows = db.execute(
            f"SELECT id, username FROM users WHERE username IN ({placeholders}) AND role = 'regular_user'",
            tuple(usernames),
        ).fetchall()
        found = {str(row["username"]): int(row["id"]) for row in rows}
        missing = [username for username in usernames if username not in found]
        if missing:
            raise ValueError(f"Unknown regular_user reviewers: {', '.join(missing)}")
        return [found[username] for username in usernames]

    rows = db.execute("SELECT id FROM users WHERE role = 'regular_user' ORDER BY id").fetchall()
    return [int(row["id"]) for row in rows]


def get_active_review_campaign(review_type: str, db=None) -> dict[str, Any] | None:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    _finalize_excluded_review_tasks(normalized_type, db=db)
    row = db.execute(
        """
        SELECT *
        FROM review_campaigns
        WHERE review_type = ?
          AND status = 'active'
        ORDER BY id DESC
        LIMIT 1
        """,
        (normalized_type,),
    ).fetchone()
    return dict(row) if row else None


def _preferred_review_campaign_for_document(
    review_type: str,
    theme: str,
    doc_id: str,
    *,
    db=None,
) -> dict[str, Any] | None:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()

    row = db.execute(
        """
        SELECT DISTINCT
            c.id,
            c.name,
            c.review_type,
            c.status
        FROM review_campaigns c
        JOIN review_campaign_tasks t ON t.campaign_id = c.id
        WHERE c.review_type = ?
          AND t.review_type = ?
          AND t.theme = ?
          AND t.doc_id = ?
        ORDER BY
            CASE c.status
                WHEN 'active' THEN 0
                WHEN 'paused' THEN 1
                ELSE 2
            END ASC,
            c.id DESC
        LIMIT 1
        """,
        (normalized_type, normalized_type, str(theme), str(doc_id)),
    ).fetchone()
    return dict(row) if row else None


def _is_user_allowed_in_campaign(campaign_id: int, user_id: int, db) -> bool:
    row = db.execute(
        """
        SELECT 1
        FROM review_campaign_reviewers r
        JOIN users u ON u.id = r.user_id
        WHERE r.campaign_id = ?
          AND r.user_id = ?
          AND u.role = 'regular_user'
        LIMIT 1
        """,
        (int(campaign_id), int(user_id)),
    ).fetchone()
    return row is not None


def create_review_campaign(
    *,
    name: str,
    review_type: str,
    seed: int,
    created_by_user_id: int,
    reviewer_usernames: list[str] | None = None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    db = get_db()

    existing = get_active_review_campaign(normalized_type, db=db)
    if existing is not None:
        raise ValueError(f"An active {normalized_type} review campaign already exists.")

    participant_user_ids = _resolve_reviewer_user_ids(reviewer_usernames, db)
    if normalized_type == "rules" and len(participant_user_ids) < 2:
        # Rules review resolves disagreements between two independent reviewers.
        # When a local staging run passes an underspecified reviewer list, fall
        # back to the full regular-user pool so the agreement workflow remains viable.
        all_regular_user_ids = _resolve_reviewer_user_ids(None, db)
        if len(all_regular_user_ids) >= 2:
            participant_user_ids = all_regular_user_ids
    restore_worktree_from_gcs()
    source_files = _iter_review_source_files(normalized_type)
    if not source_files:
        raise FileNotFoundError(
            f"No synced {normalized_type} review sources found under {REVIEW_SOURCE_ROOT / normalized_type}. "
            f"Push completed templates into the review workspace first."
        )

    docs = [(path.parent.name, path.stem) for path in source_files]
    if not participant_user_ids:
        raise ValueError("At least one regular_user reviewer is required.")
    now = _utc_now()

    cursor = db.execute(
        """
        INSERT INTO review_campaigns (
            name, review_type, seed, status, created_by_user_id, created_at
        )
        VALUES (?, ?, ?, 'active', ?, ?)
        """,
        (name, normalized_type, int(seed), int(created_by_user_id), now),
    )
    campaign_id = int(cursor.lastrowid)

    for reviewer_user_id in participant_user_ids:
        db.execute(
            """
            INSERT INTO review_campaign_reviewers (campaign_id, user_id)
            VALUES (?, ?)
            """,
            (campaign_id, int(reviewer_user_id)),
        )

    for theme, doc_id in docs:
        src_path = review_source_path(normalized_type, theme, doc_id)
        input_path = _campaign_input_path(campaign_id, normalized_type, theme, doc_id)
        _copy_file(src_path, input_path)
        task_cursor = db.execute(
            """
            INSERT INTO review_campaign_tasks (
                campaign_id,
                review_type,
                theme,
                doc_id,
                assignee_user_id,
                status,
                input_snapshot_path,
                output_snapshot_path,
                assigned_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'available', ?, NULL, ?, ?)
            """,
            (
                campaign_id,
                normalized_type,
                str(theme),
                str(doc_id),
                int(created_by_user_id),
                str(input_path),
                now,
                now,
            ),
        )
        _upsert_artifact_status(
            review_type=normalized_type,
            theme=str(theme),
            doc_id=str(doc_id),
            status="draft",
            latest_task_id=int(task_cursor.lastrowid),
            latest_snapshot_path=str(input_path),
            db=db,
        )

    db.commit()
    sync_db_to_gcs()

    return {
        "id": campaign_id,
        "name": name,
        "review_type": normalized_type,
        "status": "active",
        "seed": int(seed),
        "reviewer_count": len(participant_user_ids),
        "document_count": len(docs),
    }


def create_question_preannotation_experiment_campaign(
    *,
    name: str,
    seed: int,
    created_by_user_id: int,
    reviewer_usernames: list[str],
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a dedicated QA experiment campaign on one shared 6-document set.

    Each reviewer receives an independent task row for the same six documents so
    both reviewers can work on the identical AI/no-QA split without blocking one another.
    """
    normalized_type = "questions"
    db = get_db()

    existing = get_active_review_campaign(normalized_type, db=db)
    if existing is not None:
        raise ValueError(f"An active {normalized_type} review campaign already exists.")
    if not reviewer_usernames:
        raise ValueError("At least one reviewer is required.")
    if not documents:
        raise ValueError("At least one experiment document is required.")

    restore_worktree_from_gcs()
    now = _utc_now()

    reviewer_user_ids = _resolve_reviewer_user_ids(list(reviewer_usernames), db)
    reviewers_by_username = {
        str(username): int(user_id)
        for username, user_id in zip(reviewer_usernames, reviewer_user_ids, strict=True)
    }
    seen_docs: set[tuple[str, str]] = set()
    counts_by_group: dict[str, int] = {}
    normalized_documents: list[dict[str, Any]] = []

    for raw_document in documents:
        if not isinstance(raw_document, dict):
            raise ValueError("Invalid experiment document payload.")
        theme = yaml_service.canonical_theme_id(str(raw_document.get("theme") or "").strip())
        doc_id = str(raw_document.get("doc_id") or "").strip()
        qa_group = _normalize_question_experiment_group(raw_document.get("qa_group"))
        qa_group_order = int(raw_document.get("qa_group_order") or 0)
        initial_username = str(raw_document.get("initial_username") or "").strip()
        if not theme or not doc_id or not qa_group:
            raise ValueError(f"Incomplete experiment document: {raw_document!r}")
        doc_key = (theme, doc_id)
        if doc_key in seen_docs:
            raise ValueError(f"Document assigned more than once in the experiment: {theme}/{doc_id}")
        seen_docs.add(doc_key)
        if initial_username and initial_username not in reviewers_by_username:
            raise ValueError(f"Unknown initial reviewer {initial_username!r} for {theme}/{doc_id}")

        template_path = HUMAN_ANNOTATED_TEMPLATES_DIR / theme / f"{doc_id}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        payload = _load_yaml(template_path)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"Template metadata missing for {theme}/{doc_id}")
        status = str(metadata.get("status") or "").strip().lower()
        if status not in {"completed", "validated"}:
            raise ValueError(f"Template is not completed: {theme}/{doc_id}")

        document = payload.get("document")
        if not isinstance(document, dict):
            raise ValueError(f"Template document missing for {theme}/{doc_id}")
        if not str(document.get("document_to_annotate") or "").strip():
            raise ValueError(f"Template text missing for {theme}/{doc_id}")
        if not isinstance(document.get("rules"), list):
            raise ValueError(f"Template rules missing for {theme}/{doc_id}")

        questions = document.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"Template questions missing for {theme}/{doc_id}")
        has_questions = len(questions) > 0
        if qa_group == "ai_drafted_qas" and not has_questions:
            raise ValueError(f"AI-drafted QA arm requires existing questions: {theme}/{doc_id}")
        if qa_group == "no_qas" and has_questions:
            raise ValueError(f"No-QA arm requires an empty question list: {theme}/{doc_id}")

        counts_by_group[qa_group] = counts_by_group.get(qa_group, 0) + 1
        normalized_documents.append(
            {
                "theme": theme,
                "doc_id": doc_id,
                "qa_group": qa_group,
                "qa_group_order": qa_group_order,
                "initial_username": initial_username,
                "payload": _prepare_template_payload_for_review(normalized_type, payload),
            }
        )

    ai_count = counts_by_group.get("ai_drafted_qas", 0)
    no_qas_count = counts_by_group.get("no_qas", 0)
    if ai_count != 3 or no_qas_count != 3:
        raise ValueError(
            f"The shared experiment set must contain exactly 3 AI drafted QA docs and 3 no-QA docs "
            f"(got ai={ai_count}, no_qas={no_qas_count})."
        )

    cursor = db.execute(
        """
        INSERT INTO review_campaigns (
            name, review_type, seed, status, created_by_user_id, created_at
        )
        VALUES (?, ?, ?, 'active', ?, ?)
        """,
        (name, normalized_type, int(seed), int(created_by_user_id), now),
    )
    campaign_id = int(cursor.lastrowid)

    for user_id in sorted(set(reviewer_user_ids)):
        db.execute(
            """
            INSERT INTO review_campaign_reviewers (campaign_id, user_id)
            VALUES (?, ?)
            """,
            (campaign_id, int(user_id)),
        )

    for assignment in sorted(
        normalized_documents,
        key=lambda item: (
            QUESTION_EXPERIMENT_GROUP_SORT_ORDER.get(str(item["qa_group"]), 999),
            int(item["qa_group_order"]),
            str(item["theme"]),
            str(item["doc_id"]),
        ),
    ):
        theme = str(assignment["theme"])
        doc_id = str(assignment["doc_id"])
        payload = dict(assignment["payload"])

        source_path = review_source_path(normalized_type, theme, doc_id)
        _write_yaml(source_path, payload)

        input_path = _campaign_input_path(campaign_id, normalized_type, theme, doc_id)
        _write_yaml(input_path, payload)

        for reviewer_user_id in reviewer_user_ids:
            task_cursor = db.execute(
                """
                INSERT INTO review_campaign_tasks (
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
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'available', ?, NULL, ?, ?)
                """,
                (
                    int(campaign_id),
                    normalized_type,
                    theme,
                    doc_id,
                    int(reviewer_user_id),
                    int(reviewer_user_id),
                    str(assignment["qa_group"]),
                    int(assignment["qa_group_order"]),
                    str(input_path),
                    now,
                    now,
                ),
            )
            _upsert_artifact_status(
                review_type=normalized_type,
                theme=theme,
                doc_id=doc_id,
                status="draft",
                latest_task_id=int(task_cursor.lastrowid),
                latest_snapshot_path=str(input_path),
                db=db,
            )

    db.commit()
    sync_db_to_gcs()
    return {
        "id": campaign_id,
        "name": name,
        "review_type": normalized_type,
        "status": "active",
        "seed": int(seed),
        "reviewer_count": len(reviewer_user_ids),
        "document_count": len(normalized_documents),
    }


def _upsert_artifact_status(
    *,
    review_type: str,
    theme: str,
    doc_id: str,
    status: str,
    latest_task_id: int | None,
    latest_snapshot_path: str | None,
    db,
    reviewed_by_user_id: int | None = None,
    reviewed_at: str | None = None,
) -> None:
    normalized_type = _validate_review_type(review_type)
    db.execute(
        """
        INSERT INTO review_artifact_statuses (
            theme,
            doc_id,
            review_type,
            status,
            reviewed_by_user_id,
            reviewed_at,
            latest_task_id,
            latest_snapshot_path,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(theme, doc_id, review_type) DO UPDATE SET
            status = excluded.status,
            reviewed_by_user_id = excluded.reviewed_by_user_id,
            reviewed_at = excluded.reviewed_at,
            latest_task_id = excluded.latest_task_id,
            latest_snapshot_path = excluded.latest_snapshot_path,
            updated_at = excluded.updated_at
        """,
        (
            str(theme),
            str(doc_id),
            normalized_type,
            str(status),
            reviewed_by_user_id,
            reviewed_at,
            latest_task_id,
            latest_snapshot_path,
            _utc_now(),
        ),
    )


def _get_review_campaign_submissions(
    campaign_id: int,
    theme: str,
    doc_id: str,
    db,
) -> list[dict[str, Any]]:
    rows = db.execute(
        """
        SELECT
            s.*,
            u.username AS reviewer_username
        FROM review_campaign_submissions s
        LEFT JOIN users u ON u.id = s.reviewer_user_id
        WHERE s.campaign_id = ?
          AND s.theme = ?
          AND s.doc_id = ?
        ORDER BY s.submitted_at ASC, s.id ASC
        """,
        (int(campaign_id), str(theme), str(doc_id)),
    ).fetchall()
    return [dict(row) for row in rows]


def _get_latest_review_campaign_submission(
    campaign_id: int,
    theme: str,
    doc_id: str,
    db,
) -> dict[str, Any] | None:
    row = db.execute(
        """
        SELECT
            s.*,
            u.username AS reviewer_username
        FROM review_campaign_submissions s
        LEFT JOIN users u ON u.id = s.reviewer_user_id
        WHERE s.campaign_id = ?
          AND s.theme = ?
          AND s.doc_id = ?
        ORDER BY s.submitted_at DESC, s.id DESC
        LIMIT 1
        """,
        (int(campaign_id), str(theme), str(doc_id)),
    ).fetchone()
    return dict(row) if row is not None else None


def _review_campaign_submission_count(
    campaign_id: int,
    theme: str,
    doc_id: str,
    db,
) -> int:
    row = db.execute(
        """
        SELECT COUNT(*) AS c
        FROM review_campaign_submissions
        WHERE campaign_id = ?
          AND theme = ?
          AND doc_id = ?
        """,
        (int(campaign_id), str(theme), str(doc_id)),
    ).fetchone()
    return int(row["c"] or 0) if row else 0


def _reviewer_has_review_submission(
    campaign_id: int,
    reviewer_user_id: int,
    theme: str,
    doc_id: str,
    db,
) -> bool:
    row = db.execute(
        """
        SELECT 1
        FROM review_campaign_submissions
        WHERE campaign_id = ?
          AND reviewer_user_id = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (int(campaign_id), int(reviewer_user_id), str(theme), str(doc_id)),
    ).fetchone()
    return row is not None


def _upsert_review_campaign_agreement_state(
    *,
    campaign_id: int,
    review_type: str,
    theme: str,
    doc_id: str,
    db,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    submissions = _get_review_campaign_submissions(campaign_id, theme, doc_id, db)
    required_submissions = _required_review_submissions(normalized_type)
    if normalized_type == "questions":
        latest_submission = submissions[-1] if submissions else None
        reviewer_a_submission_id = int(latest_submission["id"]) if latest_submission is not None else None
        reviewer_b_submission_id = None
    else:
        reviewer_a_submission_id = int(submissions[0]["id"]) if len(submissions) >= 1 else None
        reviewer_b_submission_id = int(submissions[1]["id"]) if len(submissions) >= 2 else None
    next_status = "ready" if len(submissions) >= required_submissions else "pending"

    existing = db.execute(
        """
        SELECT *
        FROM review_campaign_agreements
        WHERE campaign_id = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (int(campaign_id), str(theme), str(doc_id)),
    ).fetchone()
    now = _utc_now()

    if existing is None:
        cursor = db.execute(
            """
            INSERT INTO review_campaign_agreements (
                campaign_id,
                review_type,
                theme,
                doc_id,
                status,
                reviewer_a_submission_id,
                reviewer_b_submission_id,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(campaign_id),
                normalized_type,
                str(theme),
                str(doc_id),
                next_status,
                reviewer_a_submission_id,
                reviewer_b_submission_id,
                now,
            ),
        )
        row = db.execute(
            "SELECT * FROM review_campaign_agreements WHERE id = ?",
            (int(cursor.lastrowid),),
        ).fetchone()
        return dict(row) if row else {}

    existing_dict = dict(existing)
    preserved_status = str(existing_dict.get("status") or "").strip().lower()
    existing_submission_pair = (
        int(existing_dict["reviewer_a_submission_id"]) if existing_dict.get("reviewer_a_submission_id") is not None else None,
        int(existing_dict["reviewer_b_submission_id"]) if existing_dict.get("reviewer_b_submission_id") is not None else None,
    )
    next_submission_pair = (reviewer_a_submission_id, reviewer_b_submission_id)
    has_new_submission_material = existing_submission_pair != next_submission_pair
    resolved_at_existing = str(existing_dict.get("resolved_at") or "").strip()
    latest_submission_at = str(submissions[-1]["submitted_at"] or "").strip() if submissions else ""
    if (
        preserved_status in {"resolved", "awaiting_reviewer_acceptance"}
        and resolved_at_existing
        and latest_submission_at
        and latest_submission_at > resolved_at_existing
    ):
        has_new_submission_material = True
    final_status = (
        preserved_status
        if preserved_status in {"resolved", "awaiting_reviewer_acceptance"} and not has_new_submission_material
        else next_status
    )
    resolved_by_user_id = existing_dict.get("resolved_by_user_id")
    resolved_at = existing_dict.get("resolved_at")
    final_snapshot_path = existing_dict.get("final_snapshot_path")
    requires_reviewer_acceptance = int(existing_dict.get("requires_reviewer_acceptance") or 0)
    if has_new_submission_material and final_status in {"pending", "ready"}:
        resolved_by_user_id = None
        resolved_at = None
        final_snapshot_path = None
        requires_reviewer_acceptance = 0
    db.execute(
        """
        UPDATE review_campaign_agreements
        SET review_type = ?,
            status = ?,
            reviewer_a_submission_id = ?,
            reviewer_b_submission_id = ?,
            resolved_by_user_id = ?,
            resolved_at = ?,
            final_snapshot_path = ?,
            requires_reviewer_acceptance = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            normalized_type,
            final_status,
            reviewer_a_submission_id,
            reviewer_b_submission_id,
            resolved_by_user_id,
            resolved_at,
            final_snapshot_path,
            requires_reviewer_acceptance,
            now,
            int(existing_dict["id"]),
        ),
    )
    row = db.execute(
        "SELECT * FROM review_campaign_agreements WHERE id = ?",
        (int(existing_dict["id"]),),
    ).fetchone()
    return dict(row) if row else {}


def reconcile_active_review_campaign_agreements(review_type: str, db=None) -> dict[str, int]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return {"updated": 0}

    rows = db.execute(
        """
        SELECT DISTINCT theme, doc_id
        FROM review_campaign_tasks
        WHERE campaign_id = ?
        ORDER BY theme ASC, doc_id ASC
        """,
        (int(campaign["id"]),),
    ).fetchall()
    updated = 0
    for row in rows:
        agreement_before = db.execute(
            """
            SELECT status, reviewer_a_submission_id, reviewer_b_submission_id, resolved_at
            FROM review_campaign_agreements
            WHERE campaign_id = ? AND theme = ? AND doc_id = ?
            """,
            (int(campaign["id"]), str(row["theme"]), str(row["doc_id"])),
        ).fetchone()
        before_tuple = tuple(agreement_before) if agreement_before is not None else None
        _upsert_review_campaign_agreement_state(
            campaign_id=int(campaign["id"]),
            review_type=normalized_type,
            theme=str(row["theme"]),
            doc_id=str(row["doc_id"]),
            db=db,
        )
        agreement_after = db.execute(
            """
            SELECT status, reviewer_a_submission_id, reviewer_b_submission_id, resolved_at
            FROM review_campaign_agreements
            WHERE campaign_id = ? AND theme = ? AND doc_id = ?
            """,
            (int(campaign["id"]), str(row["theme"]), str(row["doc_id"])),
        ).fetchone()
        after_tuple = tuple(agreement_after) if agreement_after is not None else None
        if before_tuple != after_tuple:
            updated += 1
    if updated > 0:
        db.commit()
        sync_db_to_gcs()
    return {"updated": updated}


def _load_review_feedback_document(raw_path: Any) -> dict[str, Any]:
    path = _resolve_review_storage_path(raw_path)
    if path is None or not path.exists():
        return {}
    payload = _load_yaml(path)
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    return document if isinstance(document, dict) else {}


def _build_review_feedback_diff(
    review_type: str,
    initial_doc: dict[str, Any],
    final_doc: dict[str, Any],
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)

    # Import lazily to avoid module cycles at import time.
    from web.services import workflow_service

    diff = workflow_service._build_reviewer_to_final_diff(initial_doc or {}, final_doc or {})  # type: ignore[attr-defined]
    summary = dict(diff.get("summary") or {})
    if normalized_type == "rules":
        summary["questions_added_count"] = 0
        summary["questions_removed_count"] = 0
        summary["questions_changed_count"] = 0
    diff["summary"] = summary
    return diff


def _review_feedback_diff_requires_acceptance(review_type: str, diff: dict[str, Any]) -> bool:
    _validate_review_type(review_type)
    # Review-campaign agreements finalize immediately once the power user resolves
    # them. We keep the acceptance-state machinery for backward compatibility with
    # older records, but no longer require accept/contest feedback for any review type.
    return False


def _normalize_review_feedback_state_from_row(
    review_type: str,
    row_dict: dict[str, Any],
    *,
    force_reviewer_acceptance: bool | None = None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    final_doc = _load_review_feedback_document(row_dict.get("final_snapshot_path"))
    stored_requires_acceptance = bool(int(row_dict.get("requires_reviewer_acceptance") or 0))

    reviewer_states: dict[str, dict[str, Any]] = {}
    pending_reviewers: list[str] = []
    contested_by: list[str] = []
    accepted_count = 0
    required_acceptance_count = 0

    for reviewer_role in ("reviewer_a", "reviewer_b"):
        reviewer_user_id = row_dict.get(f"{reviewer_role}_user_id")
        reviewer_username = str(row_dict.get(f"{reviewer_role}_username") or "").strip()
        reviewer_snapshot_path = row_dict.get(f"{reviewer_role}_snapshot_path")
        if reviewer_user_id is None or not reviewer_snapshot_path:
            continue

        initial_doc = _load_review_feedback_document(reviewer_snapshot_path)
        diff = _build_review_feedback_diff(normalized_type, initial_doc, final_doc)
        requires_acceptance = _review_feedback_diff_requires_acceptance(normalized_type, diff)
        if force_reviewer_acceptance is False:
            requires_acceptance = False
        elif force_reviewer_acceptance is None and not stored_requires_acceptance:
            requires_acceptance = False
        raw_status = str(row_dict.get(f"{reviewer_role}_response_status") or "").strip().lower()
        responded_at = row_dict.get(f"{reviewer_role}_responded_at")
        if raw_status not in ALLOWED_REVIEW_FEEDBACK_RESPONSE_STATUSES:
            raw_status = "pending"
            responded_at = None
        elif not requires_acceptance:
            raw_status = "accepted"
            responded_at = None

        if requires_acceptance:
            required_acceptance_count += 1
            if raw_status == "accepted":
                accepted_count += 1
            else:
                pending_reviewers.append(reviewer_username or reviewer_role)
                if raw_status == "contest_requested" and reviewer_username:
                    contested_by.append(reviewer_username)

        reviewer_states[reviewer_role] = {
            "reviewer_role": reviewer_role,
            "reviewer_user_id": int(reviewer_user_id),
            "reviewer_username": reviewer_username,
            "snapshot_path": reviewer_snapshot_path,
            "response_status": raw_status,
            "responded_at": responded_at,
            "requires_acceptance": requires_acceptance,
            "can_submit_decision": requires_acceptance and raw_status != "accepted",
            "diff": diff,
        }

    awaiting_acceptance = required_acceptance_count > 0 and accepted_count < required_acceptance_count
    agreement_status = "awaiting_reviewer_acceptance" if awaiting_acceptance else "resolved"
    return {
        "agreement_id": int(row_dict["id"]),
        "campaign_id": int(row_dict["campaign_id"]),
        "review_type": normalized_type,
        "theme": str(row_dict["theme"]),
        "doc_id": str(row_dict["doc_id"]),
        "resolved_by_user_id": int(row_dict["resolved_by_user_id"]) if row_dict.get("resolved_by_user_id") is not None else None,
        "resolved_by": str(row_dict.get("resolved_by") or ""),
        "resolved_at": row_dict.get("resolved_at"),
        "final_snapshot_path": row_dict.get("final_snapshot_path"),
        "reviewer_states": reviewer_states,
        "requires_reviewer_acceptance": required_acceptance_count > 0,
        "awaiting_reviewer_acceptance": awaiting_acceptance,
        "is_finalized": not awaiting_acceptance,
        "agreement_status": agreement_status,
        "accepted_count": accepted_count,
        "required_acceptance_count": required_acceptance_count,
        "pending_reviewers": pending_reviewers,
        "contested_by": contested_by,
    }


def _get_review_feedback_acceptance_state_map(review_type: str | None = None, db=None) -> dict[tuple[str, str, str], dict[str, Any]]:
    if db is None:
        db = get_db()

    query = """
        SELECT
            a.id,
            a.campaign_id,
            a.review_type,
            a.theme,
            a.doc_id,
            a.status,
            a.final_snapshot_path,
            a.requires_reviewer_acceptance,
            a.resolved_by_user_id,
            a.resolved_at,
            ua.username AS resolved_by,
            sa.snapshot_path AS reviewer_a_snapshot_path,
            sa.reviewer_user_id AS reviewer_a_user_id,
            u1.username AS reviewer_a_username,
            ra.response_status AS reviewer_a_response_status,
            ra.responded_at AS reviewer_a_responded_at,
            sb.snapshot_path AS reviewer_b_snapshot_path,
            sb.reviewer_user_id AS reviewer_b_user_id,
            u2.username AS reviewer_b_username,
            rb.response_status AS reviewer_b_response_status,
            rb.responded_at AS reviewer_b_responded_at
        FROM review_campaign_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN review_campaign_submissions sa ON sa.id = a.reviewer_a_submission_id
        LEFT JOIN review_campaign_submissions sb ON sb.id = a.reviewer_b_submission_id
        LEFT JOIN users u1 ON u1.id = sa.reviewer_user_id
        LEFT JOIN users u2 ON u2.id = sb.reviewer_user_id
        LEFT JOIN review_campaign_resolution_responses ra
               ON ra.campaign_id = a.campaign_id
              AND ra.theme = a.theme
              AND ra.doc_id = a.doc_id
              AND ra.reviewer_user_id = sa.reviewer_user_id
        LEFT JOIN review_campaign_resolution_responses rb
               ON rb.campaign_id = a.campaign_id
              AND rb.theme = a.theme
              AND rb.doc_id = a.doc_id
              AND rb.reviewer_user_id = sb.reviewer_user_id
        WHERE a.final_snapshot_path IS NOT NULL
          AND a.status IN ('resolved', 'awaiting_reviewer_acceptance')
    """
    params: list[Any] = []
    if review_type:
        normalized_type = _validate_review_type(review_type)
        query += " AND a.review_type = ?"
        params.append(normalized_type)
    query += " ORDER BY a.campaign_id ASC, a.theme ASC, a.doc_id ASC"

    rows = db.execute(query, params).fetchall()
    states: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        row_dict = dict(row)
        key = (
            str(row_dict["review_type"]),
            yaml_service.canonical_theme_id(str(row_dict["theme"])),
            str(row_dict["doc_id"]),
        )
        states[key] = _normalize_review_feedback_state_from_row(str(row_dict["review_type"]), row_dict)
    return states


def _publish_finalized_review_document_to_workflow(
    review_type: str,
    theme: str,
    doc_id: str,
    final_snapshot_path: Any,
    db,
    *,
    source_label: str = "review_agreement",
) -> None:
    normalized_type = _validate_review_type(review_type)
    if normalized_type != "rules":
        return

    final_document = _load_review_feedback_document(final_snapshot_path)
    if not final_document:
        return

    from web.services import workflow_service

    try:
        workflow_service.save_latest_final_snapshot_from_document(
            str(theme),
            str(doc_id),
            final_document,
            source_label=source_label,
            db=db,
        )
    except FileNotFoundError:
        # Isolated review-campaign test fixtures may not have a document-agreement run.
        return


def _review_task_id_for_document(campaign_id: int, review_type: str, theme: str, doc_id: str, db) -> int | None:
    row = db.execute(
        """
        SELECT id
        FROM review_campaign_tasks
        WHERE campaign_id = ?
          AND review_type = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (int(campaign_id), _validate_review_type(review_type), str(theme), str(doc_id)),
    ).fetchone()
    return int(row["id"]) if row else None


def _persist_review_feedback_acceptance_state(state: dict[str, Any], db) -> dict[str, Any]:
    agreement_status = str(state.get("agreement_status") or "resolved")
    now = _utc_now()
    db.execute(
        """
        UPDATE review_campaign_agreements
        SET status = ?,
            requires_reviewer_acceptance = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            agreement_status,
            1 if bool(state.get("requires_reviewer_acceptance")) else 0,
            now,
            int(state["agreement_id"]),
        ),
    )

    latest_task_id = _review_task_id_for_document(
        int(state["campaign_id"]),
        str(state["review_type"]),
        str(state["theme"]),
        str(state["doc_id"]),
        db,
    )
    if agreement_status == "awaiting_reviewer_acceptance":
        _upsert_artifact_status(
            review_type=str(state["review_type"]),
            theme=str(state["theme"]),
            doc_id=str(state["doc_id"]),
            status="in_progress",
            latest_task_id=latest_task_id,
            latest_snapshot_path=str(state.get("final_snapshot_path") or ""),
            db=db,
            reviewed_by_user_id=None,
            reviewed_at=None,
        )
    else:
        _upsert_artifact_status(
            review_type=str(state["review_type"]),
            theme=str(state["theme"]),
            doc_id=str(state["doc_id"]),
            status="completed",
            latest_task_id=latest_task_id,
            latest_snapshot_path=str(state.get("final_snapshot_path") or ""),
            db=db,
            reviewed_by_user_id=state.get("resolved_by_user_id"),
            reviewed_at=state.get("resolved_at"),
        )
        _publish_finalized_review_document_to_workflow(
            str(state["review_type"]),
            str(state["theme"]),
            str(state["doc_id"]),
            state.get("final_snapshot_path"),
            db,
            source_label="review_agreement",
        )
    return state


def reconcile_review_feedback_acceptance_states(review_type: str, db=None) -> dict[str, int]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()

    rows = db.execute(
        """
        SELECT
            a.id,
            a.campaign_id,
            a.review_type,
            a.theme,
            a.doc_id,
            a.status,
            a.final_snapshot_path,
            a.requires_reviewer_acceptance,
            a.resolved_by_user_id,
            a.resolved_at,
            ua.username AS resolved_by,
            sa.snapshot_path AS reviewer_a_snapshot_path,
            sa.reviewer_user_id AS reviewer_a_user_id,
            u1.username AS reviewer_a_username,
            ra.response_status AS reviewer_a_response_status,
            ra.responded_at AS reviewer_a_responded_at,
            sb.snapshot_path AS reviewer_b_snapshot_path,
            sb.reviewer_user_id AS reviewer_b_user_id,
            u2.username AS reviewer_b_username,
            rb.response_status AS reviewer_b_response_status,
            rb.responded_at AS reviewer_b_responded_at
        FROM review_campaign_agreements a
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN review_campaign_submissions sa ON sa.id = a.reviewer_a_submission_id
        LEFT JOIN review_campaign_submissions sb ON sb.id = a.reviewer_b_submission_id
        LEFT JOIN users u1 ON u1.id = sa.reviewer_user_id
        LEFT JOIN users u2 ON u2.id = sb.reviewer_user_id
        LEFT JOIN review_campaign_resolution_responses ra
               ON ra.campaign_id = a.campaign_id
              AND ra.theme = a.theme
              AND ra.doc_id = a.doc_id
              AND ra.reviewer_user_id = sa.reviewer_user_id
        LEFT JOIN review_campaign_resolution_responses rb
               ON rb.campaign_id = a.campaign_id
              AND rb.theme = a.theme
              AND rb.doc_id = a.doc_id
              AND rb.reviewer_user_id = sb.reviewer_user_id
        WHERE a.review_type = ?
          AND a.final_snapshot_path IS NOT NULL
          AND a.status IN ('resolved', 'awaiting_reviewer_acceptance')
        ORDER BY a.campaign_id ASC, a.theme ASC, a.doc_id ASC
        """,
        (normalized_type,),
    ).fetchall()
    states = [
        _normalize_review_feedback_state_from_row(
            normalized_type,
            dict(row),
            force_reviewer_acceptance=True,
        )
        for row in rows
    ]
    updated = 0
    awaiting = 0
    resolved = 0
    for state in states:
        current = db.execute(
            """
            SELECT status
            FROM review_campaign_agreements
            WHERE id = ?
            """,
            (int(state["agreement_id"]),),
        ).fetchone()
        current_status = str(current["status"] or "") if current else ""
        if current_status != str(state["agreement_status"]):
            updated += 1
        _persist_review_feedback_acceptance_state(state, db)
        if state.get("awaiting_reviewer_acceptance"):
            awaiting += 1
        else:
            resolved += 1

    if states:
        db.commit()
        sync_db_to_gcs()
    return {
        "updated": updated,
        "awaiting_reviewer_acceptance": awaiting,
        "resolved": resolved,
    }


def get_admin_review_submission_for_document(
    review_type: str,
    theme: str,
    doc_id: str,
    db=None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    campaign = _preferred_review_campaign_for_document(normalized_type, theme, doc_id, db=db)
    if campaign is None:
        raise FileNotFoundError(f"No {normalized_type} review campaign found for this document")

    row = db.execute(
        """
        SELECT
            a.*,
            ua.username AS resolved_by_username,
            sa.snapshot_path AS reviewer_a_snapshot_path,
            sa.submitted_at AS reviewer_a_submitted_at,
            sb.snapshot_path AS reviewer_b_snapshot_path,
            sb.submitted_at AS reviewer_b_submitted_at,
            u1.username AS reviewer_a_username,
            u2.username AS reviewer_b_username
        FROM review_campaign_agreements a
        LEFT JOIN review_campaign_submissions sa ON sa.id = a.reviewer_a_submission_id
        LEFT JOIN review_campaign_submissions sb ON sb.id = a.reviewer_b_submission_id
        LEFT JOIN users u1 ON u1.id = sa.reviewer_user_id
        LEFT JOIN users u2 ON u2.id = sb.reviewer_user_id
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        WHERE a.campaign_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        LIMIT 1
        """,
        (int(campaign["id"]), str(theme), str(doc_id)),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Submission not found for this document")

    row_dict = dict(row)
    latest_question_submission = (
        _get_latest_review_campaign_submission(int(campaign["id"]), str(theme), str(doc_id), db)
        if normalized_type == "questions"
        else None
    )
    reviewer_a_username = str(
        (
            latest_question_submission.get("reviewer_username")
            if isinstance(latest_question_submission, dict)
            else row_dict.get("reviewer_a_username")
        )
        or ""
    )
    reviewer_a_snapshot_path = (
        latest_question_submission.get("snapshot_path")
        if isinstance(latest_question_submission, dict)
        else row_dict.get("reviewer_a_snapshot_path")
    )
    reviewer_a_submitted_at = (
        latest_question_submission.get("submitted_at")
        if isinstance(latest_question_submission, dict)
        else row_dict.get("reviewer_a_submitted_at")
    )
    reviewer_b_username = "" if normalized_type == "questions" else str(row_dict.get("reviewer_b_username") or "")
    reviewer_b_snapshot_path = None if normalized_type == "questions" else row_dict.get("reviewer_b_snapshot_path")
    reviewer_b_submitted_at = None if normalized_type == "questions" else row_dict.get("reviewer_b_submitted_at")

    final_path = _resolve_review_storage_path(row_dict.get("final_snapshot_path"))
    source_path = review_source_path(normalized_type, str(theme), str(doc_id))
    feedback_state = _get_review_feedback_acceptance_state_map(normalized_type, db=db).get(
        (normalized_type, yaml_service.canonical_theme_id(str(theme)), str(doc_id)),
        {},
    )
    submission = {
        "theme": str(row_dict.get("theme") or theme),
        "doc_id": str(row_dict.get("doc_id") or doc_id),
        "agreement_status": str(feedback_state.get("agreement_status") or row_dict.get("status") or "pending"),
        "source_path": str(source_path) if source_path.exists() else None,
        "final_snapshot_path": _storage_path_ref(final_path) if final_path is not None else row_dict.get("final_snapshot_path"),
        "resolved_by": str(row_dict.get("resolved_by_username") or ""),
        "resolved_at": row_dict.get("resolved_at"),
        "reviewer_a": {
            "username": reviewer_a_username,
            "output_snapshot_path": reviewer_a_snapshot_path,
            "submitted_at": reviewer_a_submitted_at,
        },
        "reviewer_b": {
            "username": reviewer_b_username,
            "output_snapshot_path": reviewer_b_snapshot_path,
            "submitted_at": reviewer_b_submitted_at,
        },
        "has_final_snapshot": bool(final_path and final_path.exists()),
        "awaiting_reviewer_acceptance": bool(feedback_state.get("awaiting_reviewer_acceptance", False)),
        "is_finalized": bool(feedback_state.get("is_finalized", str(row_dict.get("status") or "").strip().lower() == "resolved")),
        "requires_reviewer_acceptance": bool(feedback_state.get("requires_reviewer_acceptance", False)),
        "contested_by": list(feedback_state.get("contested_by") or []),
        "pending_reviewers": list(feedback_state.get("pending_reviewers") or []),
    }
    return {
        "has_active_campaign": True,
        "campaign": {
            "id": int(campaign["id"]),
            "name": str(campaign["name"]),
            "review_type": normalized_type,
            "status": str(campaign.get("status") or ""),
        },
        "submission": submission,
    }


def get_admin_review_submission_content(
    review_type: str,
    theme: str,
    doc_id: str,
    variant: str,
    db=None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    campaign = _preferred_review_campaign_for_document(normalized_type, theme, doc_id, db=db)
    if campaign is None:
        raise FileNotFoundError(f"No {normalized_type} review campaign found for this document")

    meta = db.execute(
        """
        SELECT
            a.final_snapshot_path,
            sa.snapshot_path AS reviewer_a_snapshot_path,
            sb.snapshot_path AS reviewer_b_snapshot_path
        FROM review_campaign_agreements a
        LEFT JOIN review_campaign_submissions sa ON sa.id = a.reviewer_a_submission_id
        LEFT JOIN review_campaign_submissions sb ON sb.id = a.reviewer_b_submission_id
        WHERE a.campaign_id = ?
          AND a.theme = ?
          AND a.doc_id = ?
        LIMIT 1
        """,
        (int(campaign["id"]), str(theme), str(doc_id)),
    ).fetchone()
    if meta is None:
        raise FileNotFoundError("Submission not found for this document")

    latest_question_submission = (
        _get_latest_review_campaign_submission(int(campaign["id"]), str(theme), str(doc_id), db)
        if normalized_type == "questions"
        else None
    )

    normalized_variant = str(variant or "").strip().lower()
    path: Path | None = None
    if normalized_variant == "source":
        path = review_source_path(normalized_type, theme, doc_id)
    elif normalized_variant == "reviewer_a":
        if isinstance(latest_question_submission, dict) and latest_question_submission.get("snapshot_path"):
            path = _resolve_review_storage_path(latest_question_submission.get("snapshot_path"))
        else:
            path = _resolve_review_storage_path(meta["reviewer_a_snapshot_path"])
    elif normalized_variant == "reviewer_b":
        path = None if normalized_type == "questions" else _resolve_review_storage_path(meta["reviewer_b_snapshot_path"])
    elif normalized_variant == "final":
        path = _resolve_review_storage_path(meta["final_snapshot_path"])
    else:
        raise ValueError("Unknown submission variant")

    if path is None or not path.exists():
        return {
            "variant": normalized_variant,
            "content": "",
            "path": _storage_path_ref(path) if path is not None else None,
            "structured": None,
            "editable_document": None,
        }

    content = path.read_text(encoding="utf-8")
    structured: dict[str, Any] | None = None
    editable_document: dict[str, Any] | None = None
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            loaded = yaml.safe_load(content) or {}
            payload = loaded.get("document") if isinstance(loaded, dict) and isinstance(loaded.get("document"), dict) else loaded
            if isinstance(payload, dict):
                normalized_source_payload = dict(payload)
                if normalized_type == "questions":
                    fictionalized_text = str(
                        normalized_source_payload.get("fictionalized_annotated_template_document") or ""
                    ).strip()
                    if not fictionalized_text:
                        fictionalized_text = _load_fictionalized_question_document_text(str(theme), str(doc_id))
                    if (
                        not fictionalized_text
                        and isinstance(latest_question_submission, dict)
                        and latest_question_submission.get("snapshot_path")
                    ):
                        latest_snapshot_path = _resolve_review_storage_path(latest_question_submission.get("snapshot_path"))
                        if latest_snapshot_path is not None and latest_snapshot_path.exists():
                            try:
                                latest_payload = _load_yaml(latest_snapshot_path)
                                latest_document = latest_payload.get(
                                    "document",
                                    latest_payload if isinstance(latest_payload, dict) else {},
                                )
                                if isinstance(latest_document, dict):
                                    fictionalized_text = str(
                                        latest_document.get("fictionalized_annotated_template_document") or ""
                                    ).strip()
                                    if not fictionalized_text:
                                        fictionalized_text = str(latest_document.get("document_to_annotate") or "").strip()
                            except Exception:
                                fictionalized_text = ""
                    if fictionalized_text:
                        normalized_source_payload["fictionalized_annotated_template_document"] = fictionalized_text
                        normalized_source_payload["document_to_annotate"] = fictionalized_text
                    historical_rules_remap = _historical_rules_entity_id_remap(str(theme), str(doc_id), db=db)
                    if historical_rules_remap:
                        normalized_source_payload = _apply_entity_id_remap_to_document(
                            normalized_source_payload,
                            historical_rules_remap,
                        )

                normalized_payload = normalize_document_taxonomy(normalized_source_payload)
                if normalized_type == "questions":
                    _align_question_annotation_surfaces_with_document(normalized_payload)
                editable_document = normalized_payload
                questions = list(normalized_payload.get("questions") or [])
                rules = list(normalized_payload.get("rules") or [])
                structured = {
                    "document_id": normalized_payload.get("document_id"),
                    "document_theme": normalized_payload.get("document_theme"),
                    "document_to_annotate": normalized_payload.get("document_to_annotate", "") or "",
                    "fictionalized_annotated_template_document": normalized_payload.get(
                        "fictionalized_annotated_template_document", ""
                    )
                    or "",
                    "num_questions": normalized_payload.get("num_questions", len(questions)),
                    "questions": questions,
                    "rules": rules,
                    "implicit_rules": list(normalized_payload.get("implicit_rules") or []),
                    "annotation_status": normalized_payload.get("annotation_status"),
                    "annotation_errors": list(normalized_payload.get("annotation_errors") or []),
                    "annotated_by": normalized_payload.get("annotated_by"),
                    "annotated_at": normalized_payload.get("annotated_at"),
                }
        except Exception:
            structured = None
            editable_document = None

    return {
        "variant": normalized_variant,
        "content": content,
        "path": _storage_path_ref(path),
        "structured": structured,
        "editable_document": editable_document,
    }


def load_admin_review_document(review_type: str, theme: str, doc_id: str, db=None) -> dict[str, Any]:
    """Load the power-user review view for a document.

    Rules view should show the latest resolved rules when available, otherwise
    the original Opus/source rules. Questions view follows the same pattern for
    the questions payload.
    """
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()

    source_path = review_source_path(normalized_type, theme, doc_id)
    if not source_path.exists():
        restore_work_file_from_gcs(source_path)
    source_available = source_path.exists()

    snapshot_row = db.execute(
        """
        SELECT
            status,
            latest_snapshot_path
        FROM review_artifact_statuses
        WHERE review_type = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (normalized_type, str(theme), str(doc_id)),
    ).fetchone()

    active_status = str(snapshot_row["status"] or "").strip().lower() if snapshot_row else "draft"
    snapshot_path = _resolve_review_storage_path(snapshot_row["latest_snapshot_path"]) if snapshot_row else None
    if snapshot_path is not None and not snapshot_path.exists():
        restore_work_file_from_gcs(snapshot_path)
    snapshot_available = snapshot_path is not None and snapshot_path.exists()

    preferred_path: Path | None = source_path if source_available else None
    if normalized_type == "rules":
        # Rules dashboard/editor should reflect the latest synced source state by
        # default. Completed campaign snapshots remain available as explicit
        # "final" variants, but using them as the main surface can reintroduce
        # stale pre-merge equality aliases after the source/admin copy has moved on.
        if preferred_path is None and snapshot_available:
            preferred_path = snapshot_path
    else:
        if active_status == "completed" and snapshot_available:
            preferred_path = snapshot_path
        elif preferred_path is None and snapshot_available:
            preferred_path = snapshot_path

    if preferred_path is None:
        # Bootstrap a synced review source from the admin workspace so the
        # dashboard can still open/save this document.
        admin_path = yaml_service._user_doc_path("admin", yaml_service.canonical_theme_id(str(theme)), str(doc_id))
        if not admin_path.exists():
            restore_work_file_from_gcs(admin_path)
        if not admin_path.exists():
            raise FileNotFoundError(f"No synced {normalized_type} review source found for this document")
        admin_payload = _load_yaml(admin_path)
        admin_document = admin_payload.get("document", admin_payload if isinstance(admin_payload, dict) else {})
        if not isinstance(admin_document, dict):
            raise FileNotFoundError(f"No synced {normalized_type} review source found for this document")
        bootstrap_payload = _prepare_template_payload_for_review(normalized_type, {"document": admin_document})
        _write_yaml(source_path, bootstrap_payload)
        preferred_path = source_path

    payload = _load_yaml(preferred_path)
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    if not isinstance(document, dict):
        document = {}

    merged_document = dict(document)
    authoritative_document: dict[str, Any] = {}
    admin_workspace_document: dict[str, Any] = {}
    try:
        from web.services import workflow_service  # Local import to avoid module cycles.

        final_document = workflow_service.load_latest_final_snapshot_document(str(theme), str(doc_id))
        if isinstance(final_document, dict) and final_document:
            authoritative_document = dict(final_document)
    except Exception:
        pass

    try:
        admin_path = yaml_service._user_doc_path("admin", yaml_service.canonical_theme_id(str(theme)), str(doc_id))
        if not admin_path.exists():
            restore_work_file_from_gcs(admin_path)
        if admin_path.exists():
            admin_payload = _load_yaml(admin_path)
            admin_document = admin_payload.get("document", admin_payload if isinstance(admin_payload, dict) else {})
            if isinstance(admin_document, dict) and admin_document:
                admin_workspace_document = dict(admin_document)
    except Exception:
        admin_workspace_document = {}

    if normalized_type == "questions" and not authoritative_document and admin_workspace_document:
        authoritative_document = dict(admin_workspace_document)

    if authoritative_document:
        authoritative_text = str(authoritative_document.get("document_to_annotate", "") or "")
        if authoritative_text.strip():
            merged_document["document_to_annotate"] = authoritative_text
        if "original_document" in authoritative_document:
            merged_document["original_document"] = authoritative_document.get("original_document")
        if "relations" in authoritative_document:
            merged_document["relations"] = list(authoritative_document.get("relations") or [])
        if "fictionalized_annotated_template_document" in authoritative_document:
            merged_document["fictionalized_annotated_template_document"] = str(
                authoritative_document.get("fictionalized_annotated_template_document") or ""
            )

        if normalized_type == "questions":
            merged_document["rules"] = list(authoritative_document.get("rules") or [])
            merged_document["implicit_rules"] = list(authoritative_document.get("implicit_rules") or [])
            if "implicit_rule_exclusions" in authoritative_document:
                merged_document["implicit_rule_exclusions"] = list(
                    authoritative_document.get("implicit_rule_exclusions") or []
                )
            else:
                merged_document.pop("implicit_rule_exclusions", None)
        elif normalized_type == "rules":
            authoritative_questions = authoritative_document.get("questions")
            if isinstance(authoritative_questions, list):
                merged_document["questions"] = list(authoritative_questions)
                merged_document["num_questions"] = len(merged_document["questions"])
            if QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD in authoritative_document:
                merged_document[QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD] = list(
                    authoritative_document.get(QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD) or []
                )
    if normalized_type == "questions" and admin_workspace_document:
        admin_text = str(admin_workspace_document.get("document_to_annotate", "") or "")
        if admin_text.strip():
            merged_document["document_to_annotate"] = admin_text
        if "original_document" in admin_workspace_document:
            merged_document["original_document"] = admin_workspace_document.get("original_document")
        if "relations" in admin_workspace_document:
            merged_document["relations"] = list(admin_workspace_document.get("relations") or [])
        if "fictionalized_annotated_template_document" in admin_workspace_document:
            merged_document["fictionalized_annotated_template_document"] = str(
                admin_workspace_document.get("fictionalized_annotated_template_document") or ""
            )
    if normalized_type == "questions":
        fictionalized_text = str(merged_document.get("fictionalized_annotated_template_document") or "").strip()
        if not fictionalized_text:
            fictionalized_text = _load_fictionalized_question_document_text(str(theme), str(doc_id))
        if fictionalized_text:
            merged_document["fictionalized_annotated_template_document"] = fictionalized_text

        # Power-user dashboard QA review must stay factual, but preserve the
        # latest saved admin/final payload when it is already factual/edited.
        # Only coerce when the current surface is empty or clearly the
        # fictionalized text.
        current_text = str(merged_document.get("document_to_annotate") or "").strip()
        should_coerce_to_factual = (not current_text) or (
            bool(fictionalized_text) and current_text == fictionalized_text
        )
        if should_coerce_to_factual:
            factual_text = _load_factual_question_document_text(str(theme), str(doc_id))
            if factual_text:
                merged_document["document_to_annotate"] = factual_text
        historical_rules_remap = _historical_rules_entity_id_remap(str(theme), str(doc_id), db=db)
        if historical_rules_remap:
            merged_document = _apply_entity_id_remap_to_document(merged_document, historical_rules_remap)
    _rewrite_cross_section_entity_refs_to_document(merged_document)
    if normalized_type == "questions":
        _align_question_annotation_surfaces_with_document(merged_document)

    normalized_document = normalize_document_taxonomy(merged_document)
    if normalized_type == "questions":
        _align_question_annotation_surfaces_with_document(normalized_document)
    normalized_document["review_statuses"] = get_document_review_statuses(str(theme), str(doc_id), db=db)
    normalized_document["active_review_target"] = normalized_type
    normalized_document["active_review_task_status"] = active_status or "draft"

    campaign = _preferred_review_campaign_for_document(normalized_type, theme, doc_id, db=db)
    normalized_document["active_review_campaign_name"] = str(campaign.get("name") or "") if campaign else ""
    return normalized_document


def load_review_submission_document(
    review_type: str,
    theme: str,
    doc_id: str,
    reviewer_username: str,
    db=None,
) -> dict[str, Any]:
    """Load a specific review submission snapshot by reviewer username."""
    normalized_type = _validate_review_type(review_type)
    reviewer = str(reviewer_username or "").strip()
    if not reviewer:
        raise FileNotFoundError("No reviewer username provided for the reference submission.")
    if db is None:
        db = get_db()

    row = db.execute(
        """
        SELECT
            s.snapshot_path,
            s.submitted_at,
            c.name AS campaign_name
        FROM review_campaign_submissions s
        JOIN users u ON u.id = s.reviewer_user_id
        JOIN review_campaigns c ON c.id = s.campaign_id
        WHERE s.review_type = ?
          AND s.theme = ?
          AND s.doc_id = ?
          AND u.username = ?
        ORDER BY s.submitted_at DESC, s.id DESC
        LIMIT 1
        """,
        (normalized_type, str(theme), str(doc_id), reviewer),
    ).fetchone()
    if row is None:
        raise FileNotFoundError(
            f"No {normalized_type} submission by {reviewer} was found for this document."
        )

    snapshot_path = _resolve_review_storage_path(row["snapshot_path"])
    if snapshot_path is None or not snapshot_path.exists():
        raise FileNotFoundError(
            f"The {normalized_type} submission snapshot by {reviewer} is not available."
        )

    payload = _load_yaml(snapshot_path)
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    if not isinstance(document, dict):
        document = {}

    normalized_document = normalize_document_taxonomy(document)
    normalized_document["review_statuses"] = get_document_review_statuses(str(theme), str(doc_id), db=db)
    normalized_document["active_review_target"] = normalized_type
    normalized_document["active_review_task_status"] = "completed"
    normalized_document["active_review_campaign_name"] = str(row["campaign_name"] or "")
    normalized_document["reference_submission_reviewer"] = reviewer
    normalized_document["reference_submission_submitted_at"] = str(row["submitted_at"] or "")
    return normalized_document


def save_admin_review_document(
    review_type: str,
    theme: str,
    doc_id: str,
    document: dict[str, Any],
    db=None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()

    source_path = review_source_path(normalized_type, theme, doc_id)
    if not source_path.exists():
        restore_work_file_from_gcs(source_path)

    payload = {"document": _canonicalize_mergeable_equality_rules(document if isinstance(document, dict) else {})}
    payload = _prepare_template_payload_for_review(normalized_type, payload)

    snapshot_row = db.execute(
        """
        SELECT
            status,
            latest_snapshot_path
        FROM review_artifact_statuses
        WHERE review_type = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (normalized_type, str(theme), str(doc_id)),
    ).fetchone()
    active_status = str(snapshot_row["status"] or "").strip().lower() if snapshot_row else "draft"
    snapshot_path = _resolve_review_storage_path(snapshot_row["latest_snapshot_path"]) if snapshot_row else None

    target_path = source_path
    if active_status == "completed" and snapshot_path is not None and snapshot_path.exists():
        target_path = snapshot_path

    # Keep review-source snapshots aligned with admin edits even after a review
    # artifact reaches "completed". This avoids stale _review_sources drift.
    _write_yaml(source_path, payload)
    if target_path != source_path:
        _write_yaml(target_path, payload)

    try:
        _publish_review_fields_to_admin_workspace(
            normalized_type,
            theme=theme,
            doc_id=doc_id,
            template_payload=payload,
        )
    except Exception:
        pass

    refreshed_campaign_inputs = 0
    published_final_snapshot = False
    try:
        refreshed_campaign_inputs = _refresh_active_available_review_task_inputs(
            normalized_type,
            theme=theme,
            doc_id=doc_id,
            template_payload=payload,
        )
    except Exception:
        refreshed_campaign_inputs = 0
    if target_path != source_path:
        try:
            _publish_finalized_review_document_to_workflow(
                normalized_type,
                str(theme),
                str(doc_id),
                target_path,
                db,
                source_label="admin_editor",
            )
            published_final_snapshot = True
        except Exception:
            published_final_snapshot = False
    else:
        try:
            published_final_snapshot = bool(
                _publish_review_fields_to_latest_final_snapshot(
                    normalized_type,
                    theme=theme,
                    doc_id=doc_id,
                    template_payload=payload,
                )
            )
        except Exception:
            published_final_snapshot = False

    return {
        "status": "saved",
        "review_type": normalized_type,
        "theme": str(theme),
        "doc_id": str(doc_id),
        "active_status": active_status or "draft",
        "target_path": _storage_path_ref(target_path),
        "refreshed_campaign_inputs": int(refreshed_campaign_inputs),
        "published_final_snapshot": bool(published_final_snapshot),
    }


def set_admin_review_final_snapshot_from_document(
    review_type: str,
    theme: str,
    doc_id: str,
    document: dict[str, Any],
    source_label: str = "admin",
    db=None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        raise FileNotFoundError(f"No active {normalized_type} review campaign")

    agreement = db.execute(
        """
        SELECT id
        FROM review_campaign_agreements
        WHERE campaign_id = ?
          AND theme = ?
          AND doc_id = ?
        LIMIT 1
        """,
        (int(campaign["id"]), str(theme), str(doc_id)),
    ).fetchone()
    if agreement is None:
        raise FileNotFoundError("Agreement not found for this document")

    final_path = _review_agreement_final_path(int(campaign["id"]), normalized_type, theme, doc_id)
    payload = _canonicalize_mergeable_equality_rules(document if isinstance(document, dict) else {})
    if normalized_type == "questions":
        payload = _normalize_question_review_document(payload)
        if not _is_question_review_draft_source_label(source_label):
            _validate_question_review_submission_contract(payload)
    _write_yaml(final_path, {"document": payload})

    now = _utc_now()
    db.execute(
        """
        UPDATE review_campaign_agreements
        SET final_snapshot_path = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (_storage_path_ref(final_path), now, int(agreement["id"])),
    )
    _upsert_artifact_status(
        review_type=normalized_type,
        theme=str(theme),
        doc_id=str(doc_id),
        status="in_progress",
        latest_task_id=None,
        latest_snapshot_path=_storage_path_ref(final_path),
        db=db,
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "campaign_id": int(campaign["id"]),
        "review_type": normalized_type,
        "theme": str(theme),
        "doc_id": str(doc_id),
        "final_snapshot_path": _storage_path_ref(final_path),
        "final_source_label": str(source_label or "admin").strip() or "admin",
    }


def complete_admin_review_from_document(
    review_type: str,
    theme: str,
    doc_id: str,
    document: dict[str, Any],
    resolved_by_user_id: int,
    *,
    source_label: str = "admin_completed",
    db=None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        raise FileNotFoundError(f"No active {normalized_type} review campaign")

    agreement = _upsert_review_campaign_agreement_state(
        campaign_id=int(campaign["id"]),
        review_type=normalized_type,
        theme=str(theme),
        doc_id=str(doc_id),
        db=db,
    )
    if not agreement:
        raise FileNotFoundError("Agreement not found for this document")

    final_path = _review_agreement_final_path(int(campaign["id"]), normalized_type, theme, doc_id)
    payload = _canonicalize_mergeable_equality_rules(document if isinstance(document, dict) else {})
    if normalized_type == "questions":
        payload = _normalize_question_review_document(payload)
        _validate_question_review_submission_contract(payload)
    _write_yaml(final_path, {"document": payload})

    now = _utc_now()
    db.execute(
        """
        DELETE FROM review_campaign_resolution_responses
        WHERE campaign_id = ? AND theme = ? AND doc_id = ?
        """,
        (int(campaign["id"]), str(theme), str(doc_id)),
    )
    db.execute(
        """
        UPDATE review_campaign_agreements
        SET status = 'resolved',
            resolved_by_user_id = ?,
            resolved_at = ?,
            final_snapshot_path = ?,
            requires_reviewer_acceptance = 0,
            updated_at = ?
        WHERE id = ?
        """,
        (
            int(resolved_by_user_id),
            now,
            _storage_path_ref(final_path),
            now,
            int(agreement["id"]),
        ),
    )
    db.execute(
        """
        UPDATE review_campaign_tasks
        SET status = 'completed',
            completed_at = COALESCE(completed_at, ?),
            output_snapshot_path = COALESCE(output_snapshot_path, ?),
            updated_at = ?
        WHERE campaign_id = ?
          AND review_type = ?
          AND theme = ?
          AND doc_id = ?
        """,
        (
            now,
            _storage_path_ref(final_path),
            now,
            int(campaign["id"]),
            normalized_type,
            str(theme),
            str(doc_id),
        ),
    )

    agreement_row = {
        **dict(agreement),
        "campaign_id": int(campaign["id"]),
        "review_type": normalized_type,
        "theme": str(theme),
        "doc_id": str(doc_id),
        "resolved_by_user_id": int(resolved_by_user_id),
        "resolved_at": now,
        "resolved_by": "",
        "status": "resolved",
        "final_snapshot_path": _storage_path_ref(final_path),
        "reviewer_a_response_status": None,
        "reviewer_a_responded_at": None,
        "reviewer_b_response_status": None,
        "reviewer_b_responded_at": None,
        "requires_reviewer_acceptance": 0,
    }
    state = _normalize_review_feedback_state_from_row(
        normalized_type,
        agreement_row,
        force_reviewer_acceptance=False,
    )
    _persist_review_feedback_acceptance_state(state, db)
    latest_task_id = _review_task_id_for_document(int(campaign["id"]), normalized_type, str(theme), str(doc_id), db)
    _upsert_artifact_status(
        review_type=normalized_type,
        theme=str(theme),
        doc_id=str(doc_id),
        status="completed",
        latest_task_id=latest_task_id,
        latest_snapshot_path=_storage_path_ref(final_path),
        db=db,
        reviewed_by_user_id=int(resolved_by_user_id),
        reviewed_at=now,
    )
    _publish_finalized_review_document_to_workflow(
        normalized_type,
        str(theme),
        str(doc_id),
        _storage_path_ref(final_path),
        db,
        source_label=str(source_label or "admin_completed").strip() or "admin_completed",
    )

    db.commit()
    sync_db_to_gcs()
    return {
        "campaign_id": int(campaign["id"]),
        "review_type": normalized_type,
        "theme": str(theme),
        "doc_id": str(doc_id),
        "agreement_status": "resolved",
        "final_snapshot_path": _storage_path_ref(final_path),
        "is_finalized": True,
        "final_source_label": str(source_label or "admin_completed").strip() or "admin_completed",
    }


def resolve_review_agreement(
    campaign_id: int,
    review_type: str,
    theme: str,
    doc_id: str,
    resolved_by_user_id: int,
    final_variant: str | None = None,
    require_reviewer_acceptance: bool = True,
    db=None,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()

    row = db.execute(
        """
        SELECT
            a.id,
            a.status,
            a.campaign_id,
            a.theme,
            a.doc_id,
            a.resolved_by_user_id,
            a.resolved_at,
            a.final_snapshot_path,
            a.reviewer_a_submission_id,
            a.reviewer_b_submission_id,
            sa.snapshot_path AS reviewer_a_snapshot_path,
            sa.task_id AS reviewer_a_task_id,
            sa.reviewer_user_id AS reviewer_a_user_id,
            u1.username AS reviewer_a_username,
            sb.snapshot_path AS reviewer_b_snapshot_path,
            sb.task_id AS reviewer_b_task_id,
            sb.reviewer_user_id AS reviewer_b_user_id,
            u2.username AS reviewer_b_username,
            ua.username AS resolved_by,
            ra.response_status AS reviewer_a_response_status,
            ra.responded_at AS reviewer_a_responded_at,
            rb.response_status AS reviewer_b_response_status,
            rb.responded_at AS reviewer_b_responded_at
        FROM review_campaign_agreements a
        LEFT JOIN review_campaign_submissions sa ON sa.id = a.reviewer_a_submission_id
        LEFT JOIN review_campaign_submissions sb ON sb.id = a.reviewer_b_submission_id
        LEFT JOIN users u1 ON u1.id = sa.reviewer_user_id
        LEFT JOIN users u2 ON u2.id = sb.reviewer_user_id
        LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
        LEFT JOIN review_campaign_resolution_responses ra
               ON ra.campaign_id = a.campaign_id
              AND ra.theme = a.theme
              AND ra.doc_id = a.doc_id
              AND ra.reviewer_user_id = sa.reviewer_user_id
        LEFT JOIN review_campaign_resolution_responses rb
               ON rb.campaign_id = a.campaign_id
              AND rb.theme = a.theme
              AND rb.doc_id = a.doc_id
              AND rb.reviewer_user_id = sb.reviewer_user_id
        WHERE a.campaign_id = ?
          AND a.review_type = ?
          AND a.theme = ?
          AND a.doc_id = ?
        LIMIT 1
        """,
        (int(campaign_id), normalized_type, str(theme), str(doc_id)),
    ).fetchone()
    if row is None:
        raise FileNotFoundError("Agreement packet not found")

    row_dict = dict(row)
    chosen_variant = str(final_variant or "").strip().lower()
    if normalized_type == "questions":
        chosen_variant = "reviewer_a"
    elif chosen_variant not in {"reviewer_a", "reviewer_b"}:
        chosen_variant = "reviewer_a"

    final_snapshot_path = _resolve_review_storage_path(row_dict.get("final_snapshot_path"))
    if final_snapshot_path is None or not final_snapshot_path.exists():
        source_path = _resolve_review_storage_path(row_dict.get(f"{chosen_variant}_snapshot_path"))
        if (source_path is None or not source_path.exists()) and normalized_type == "questions":
            latest_question_submission = _get_latest_review_campaign_submission(
                int(campaign_id),
                str(theme),
                str(doc_id),
                db,
            )
            if isinstance(latest_question_submission, dict):
                source_path = _resolve_review_storage_path(latest_question_submission.get("snapshot_path"))
        if source_path is None or not source_path.exists():
            raise FileNotFoundError("Chosen reviewer submission is not available")
        final_snapshot_path = _review_agreement_final_path(int(campaign_id), normalized_type, theme, doc_id)
        _copy_file(source_path, final_snapshot_path)

    now = _utc_now()
    db.execute(
        """
        DELETE FROM review_campaign_resolution_responses
        WHERE campaign_id = ? AND theme = ? AND doc_id = ?
        """,
        (int(campaign_id), str(theme), str(doc_id)),
    )
    db.execute(
        """
        UPDATE review_campaign_agreements
        SET status = 'resolved',
            resolved_by_user_id = ?,
            resolved_at = ?,
            final_snapshot_path = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (
            int(resolved_by_user_id),
            now,
            _storage_path_ref(final_snapshot_path),
            now,
            int(row_dict["id"]),
        ),
    )

    state_row = {
        **row_dict,
        "campaign_id": int(campaign_id),
        "theme": str(theme),
        "doc_id": str(doc_id),
        "resolved_by_user_id": int(resolved_by_user_id),
        "resolved_at": now,
        "resolved_by": "",
        "final_snapshot_path": _storage_path_ref(final_snapshot_path),
        "reviewer_a_response_status": None,
        "reviewer_a_responded_at": None,
        "reviewer_b_response_status": None,
        "reviewer_b_responded_at": None,
        "requires_reviewer_acceptance": 1 if require_reviewer_acceptance else 0,
    }
    state = _normalize_review_feedback_state_from_row(
        normalized_type,
        state_row,
        force_reviewer_acceptance=True if require_reviewer_acceptance else False,
    )

    _persist_review_feedback_acceptance_state(state, db)
    _publish_finalized_review_document_to_workflow(
        normalized_type,
        str(theme),
        str(doc_id),
        _storage_path_ref(final_snapshot_path),
        db,
        source_label="review_agreement",
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "campaign_id": int(campaign_id),
        "review_type": normalized_type,
        "theme": str(theme),
        "doc_id": str(doc_id),
        "agreement_status": str(state.get("agreement_status") or "resolved"),
        "final_snapshot_path": _storage_path_ref(final_snapshot_path),
        "awaiting_reviewer_acceptance": bool(state.get("awaiting_reviewer_acceptance", False)),
        "is_finalized": bool(state.get("is_finalized", True)),
        "pending_reviewers": list(state.get("pending_reviewers") or []),
    }


def get_user_review_resolution_feedback(user_id: int, review_type: str) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return {"has_active_campaign": False, "is_participant": False, "campaign": None, "items": []}
    if not _is_user_allowed_in_campaign(int(campaign["id"]), int(user_id), db):
        return {
            "has_active_campaign": True,
            "is_participant": False,
            "campaign": {
                "id": int(campaign["id"]),
                "name": str(campaign["name"]),
                "review_type": normalized_type,
                "status": str(campaign["status"]),
            },
            "items": [],
        }

    states = _get_review_feedback_acceptance_state_map(normalized_type, db=db)
    items: list[dict[str, Any]] = []
    for key, state in states.items():
        _, theme, doc_id = key
        if int(state["campaign_id"]) != int(campaign["id"]):
            continue
        reviewer_states = state.get("reviewer_states") or {}
        selected: dict[str, Any] | None = None
        for role_state in reviewer_states.values():
            if int(role_state.get("reviewer_user_id") or -1) != int(user_id):
                continue
            if not bool(role_state.get("requires_acceptance")):
                continue
            selected = dict(role_state)
            break
        if selected is None:
            continue
        initial_doc = _load_review_feedback_document(selected.get("snapshot_path"))
        final_doc = _load_review_feedback_document(state.get("final_snapshot_path"))

        items.append(
            {
                "campaign_id": int(campaign["id"]),
                "review_type": normalized_type,
                "theme": theme,
                "doc_id": doc_id,
                "reviewer_role": str(selected.get("reviewer_role") or ""),
                "reviewer_username": str(selected.get("reviewer_username") or ""),
                "reviewer_initial_snapshot_path": str(selected.get("snapshot_path") or ""),
                "final_snapshot_path": str(state.get("final_snapshot_path") or ""),
                "resolved_by": str(state.get("resolved_by") or ""),
                "resolved_at": state.get("resolved_at"),
                "response_status": str(selected.get("response_status") or "pending"),
                "responded_at": selected.get("responded_at"),
                "can_submit_decision": bool(selected.get("can_submit_decision", False)),
                "requires_reviewer_acceptance": True,
                "is_finalized": bool(state.get("is_finalized", False)),
                "awaiting_reviewer_acceptance": bool(state.get("awaiting_reviewer_acceptance", False)),
                "initial_document_markup": str(initial_doc.get("document_to_annotate") or ""),
                "initial_document_to_annotate": str(
                    ((selected.get("diff") or {}).get("initial_plain_text_preview") or "")
                ),
                "final_document_markup": str(final_doc.get("document_to_annotate") or ""),
                "final_document_to_annotate": str(
                    ((selected.get("diff") or {}).get("final_plain_text_preview") or "")
                ),
                "initial_rules": list(initial_doc.get("rules") or []),
                "final_rules": list(final_doc.get("rules") or []),
                "diff": dict(selected.get("diff") or {}),
            }
        )

    items.sort(key=lambda item: (str(item.get("resolved_at") or ""), str(item.get("theme") or ""), str(item.get("doc_id") or "")), reverse=True)
    return {
        "has_active_campaign": True,
        "is_participant": True,
        "campaign": {
            "id": int(campaign["id"]),
            "name": str(campaign["name"]),
            "review_type": normalized_type,
            "status": str(campaign["status"]),
        },
        "items": items,
    }


def submit_user_review_resolution_feedback_response(
    user_id: int,
    review_type: str,
    theme: str,
    doc_id: str,
    response_status: str,
) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    decision = str(response_status or "").strip().lower()
    if decision not in ALLOWED_REVIEW_FEEDBACK_RESPONSE_STATUSES:
        raise ValueError("response_status must be one of: accepted, contest_requested")

    db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        raise FileNotFoundError(f"No active {normalized_type} review campaign")
    if not _is_user_allowed_in_campaign(int(campaign["id"]), int(user_id), db):
        raise PermissionError("User is not part of the active review campaign")

    state = _get_review_feedback_acceptance_state_map(normalized_type, db=db).get(
        (normalized_type, yaml_service.canonical_theme_id(str(theme)), str(doc_id)),
    )
    if state is None or int(state["campaign_id"]) != int(campaign["id"]):
        raise FileNotFoundError("Resolved feedback item not found for this user/document")

    reviewer_state: dict[str, Any] | None = None
    for role_state in (state.get("reviewer_states") or {}).values():
        if int(role_state.get("reviewer_user_id") or -1) != int(user_id):
            continue
        if not bool(role_state.get("requires_acceptance")):
            continue
        reviewer_state = dict(role_state)
        break
    if reviewer_state is None:
        raise FileNotFoundError("Resolved feedback item not found for this user/document")

    reviewer_role = str(reviewer_state.get("reviewer_role") or "")
    now = _utc_now()
    db.execute(
        """
        INSERT INTO review_campaign_resolution_responses (
            campaign_id,
            review_type,
            theme,
            doc_id,
            reviewer_user_id,
            reviewer_role,
            response_status,
            responded_at,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(campaign_id, theme, doc_id, reviewer_user_id)
        DO UPDATE SET
            reviewer_role = excluded.reviewer_role,
            response_status = excluded.response_status,
            responded_at = excluded.responded_at,
            updated_at = excluded.updated_at
        """,
        (
            int(campaign["id"]),
            normalized_type,
            str(theme),
            str(doc_id),
            int(user_id),
            reviewer_role,
            decision,
            now,
            now,
            now,
        ),
    )

    refreshed_state = _get_review_feedback_acceptance_state_map(normalized_type, db=db).get(
        (normalized_type, yaml_service.canonical_theme_id(str(theme)), str(doc_id)),
    )
    if refreshed_state is None:
        raise FileNotFoundError("Resolved feedback item not found for this user/document")
    _persist_review_feedback_acceptance_state(refreshed_state, db)
    db.commit()
    sync_db_to_gcs()
    return {
        "campaign_id": int(campaign["id"]),
        "review_type": normalized_type,
        "theme": str(theme),
        "doc_id": str(doc_id),
        "reviewer_user_id": int(user_id),
        "reviewer_role": reviewer_role,
        "response_status": decision,
        "responded_at": now,
        "requires_reviewer_acceptance": bool(refreshed_state.get("requires_reviewer_acceptance", False)),
        "is_finalized": bool(refreshed_state.get("is_finalized", True)),
        "awaiting_reviewer_acceptance": bool(refreshed_state.get("awaiting_reviewer_acceptance", False)),
    }


def _review_campaign_status_priority(status: Any) -> int:
    normalized = str(status or "").strip().lower()
    if normalized == "active":
        return 0
    if normalized == "paused":
        return 1
    return 2


def _unique_usernames(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        username = str(raw or "").strip()
        if not username or username in seen:
            continue
        seen.add(username)
        ordered.append(username)
    return ordered


def get_review_document_activity_map(db=None) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    if db is None:
        db = get_db()

    task_rows = db.execute(
        """
        SELECT
            t.campaign_id,
            t.review_type,
            t.theme,
            t.doc_id,
            t.status AS task_status,
            t.assigned_at,
            t.started_at,
            t.updated_at,
            c.status AS campaign_status,
            u.username AS assignee_username
        FROM review_campaign_tasks t
        JOIN review_campaigns c ON c.id = t.campaign_id
        LEFT JOIN users u ON u.id = t.assignee_user_id
        ORDER BY t.campaign_id DESC, t.id ASC
        """
    ).fetchall()
    if not task_rows:
        return {}

    submission_rows = db.execute(
        """
        SELECT
            s.campaign_id,
            s.review_type,
            s.theme,
            s.doc_id,
            s.submitted_at,
            u.username
        FROM review_campaign_submissions s
        LEFT JOIN users u ON u.id = s.reviewer_user_id
        ORDER BY s.campaign_id DESC, s.submitted_at ASC, s.id ASC
        """
    ).fetchall()
    agreement_rows = db.execute(
        """
        SELECT
            a.campaign_id,
            a.review_type,
            a.theme,
            a.doc_id,
            a.status,
            a.resolved_at,
            u.username AS resolved_by
        FROM review_campaign_agreements a
        LEFT JOIN users u ON u.id = a.resolved_by_user_id
        ORDER BY a.campaign_id DESC, a.id ASC
        """
    ).fetchall()

    preferred_campaign_for_doc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in task_rows:
        review_type = str(row["review_type"] or "").strip().lower()
        theme = yaml_service.canonical_theme_id(str(row["theme"] or ""))
        doc_id = str(row["doc_id"] or "")
        if not review_type or not theme or not doc_id:
            continue
        key = (review_type, theme, doc_id)
        candidate = (
            _review_campaign_status_priority(row["campaign_status"]),
            -int(row["campaign_id"]),
        )
        existing = preferred_campaign_for_doc.get(key)
        if existing is None or candidate < existing["rank"]:
            preferred_campaign_for_doc[key] = {
                "campaign_id": int(row["campaign_id"]),
                "rank": candidate,
            }

    task_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in task_rows:
        review_type = str(row["review_type"] or "").strip().lower()
        theme = yaml_service.canonical_theme_id(str(row["theme"] or ""))
        doc_id = str(row["doc_id"] or "")
        key = (review_type, theme, doc_id)
        preferred = preferred_campaign_for_doc.get(key)
        if preferred is None or int(row["campaign_id"]) != int(preferred["campaign_id"]):
            continue
        task_map.setdefault(key, []).append(dict(row))

    submission_map: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in submission_rows:
        review_type = str(row["review_type"] or "").strip().lower()
        theme = yaml_service.canonical_theme_id(str(row["theme"] or ""))
        doc_id = str(row["doc_id"] or "")
        key = (review_type, theme, doc_id)
        preferred = preferred_campaign_for_doc.get(key)
        if preferred is None or int(row["campaign_id"]) != int(preferred["campaign_id"]):
            continue
        submission_map.setdefault(key, []).append(dict(row))

    agreement_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in agreement_rows:
        review_type = str(row["review_type"] or "").strip().lower()
        theme = yaml_service.canonical_theme_id(str(row["theme"] or ""))
        doc_id = str(row["doc_id"] or "")
        key = (review_type, theme, doc_id)
        preferred = preferred_campaign_for_doc.get(key)
        if preferred is None or int(row["campaign_id"]) != int(preferred["campaign_id"]):
            continue
        agreement_map[key] = dict(row)

    result: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for key, preferred in preferred_campaign_for_doc.items():
        review_type, theme, doc_id = key
        tasks_for_doc = task_map.get(key, [])
        submissions_for_doc = submission_map.get(key, [])
        agreement_for_doc = agreement_map.get(key) or {}

        submitted_users = _unique_usernames([str(row.get("username") or "") for row in submissions_for_doc])
        active_users = _unique_usernames(
            [
                str(row.get("assignee_username") or "")
                for row in tasks_for_doc
                if str(row.get("task_status") or "").strip().lower() == "in_progress"
            ]
        )
        display_users = _unique_usernames(submitted_users + active_users)

        events: list[dict[str, Any]] = []
        for row in submissions_for_doc:
            username = str(row.get("username") or "").strip()
            timestamp = str(row.get("submitted_at") or "").strip()
            if username and timestamp:
                events.append({"timestamp": timestamp, "username": username, "action": "review"})
        for row in tasks_for_doc:
            if str(row.get("task_status") or "").strip().lower() != "in_progress":
                continue
            username = str(row.get("assignee_username") or "").strip()
            timestamp = str(row.get("updated_at") or row.get("started_at") or row.get("assigned_at") or "").strip()
            if username and timestamp:
                events.append({"timestamp": timestamp, "username": username, "action": "edit"})
        resolved_by = str(agreement_for_doc.get("resolved_by") or "").strip()
        resolved_at = str(agreement_for_doc.get("resolved_at") or "").strip()
        if resolved_by and resolved_at:
            events.append({"timestamp": resolved_at, "username": resolved_by, "action": "validate"})
        events.sort(key=lambda entry: (str(entry["timestamp"]), str(entry["username"])), reverse=True)
        latest_event = events[0] if events else {}

        bucket = result.setdefault((theme, doc_id), {})
        bucket[review_type] = {
            "campaign_id": int(preferred["campaign_id"]),
            "last_edited_by": ", ".join(display_users) if display_users else None,
            "activity_users": display_users,
            "last_activity_at": latest_event.get("timestamp"),
            "last_activity_user": latest_event.get("username"),
            "last_activity_action": latest_event.get("action"),
        }
    return result


def get_recent_review_activity(review_type: str, *, limit: int = 50, db=None) -> list[dict[str, Any]]:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()

    events: list[dict[str, Any]] = []

    submission_rows = db.execute(
        """
        SELECT
            s.id,
            s.theme,
            s.doc_id,
            s.submitted_at,
            u.username
        FROM review_campaign_submissions s
        LEFT JOIN users u ON u.id = s.reviewer_user_id
        WHERE s.review_type = ?
        ORDER BY s.submitted_at DESC, s.id DESC
        LIMIT 100
        """,
        (normalized_type,),
    ).fetchall()
    for row in submission_rows:
        if _is_excluded_from_review_campaign_doc(
            normalized_type,
            str(row["theme"] or ""),
            str(row["doc_id"] or ""),
        ):
            continue
        username = str(row["username"] or "").strip()
        timestamp = str(row["submitted_at"] or "").strip()
        if not username or not timestamp:
            continue
        events.append(
            {
                "id": f"{normalized_type}/submission/{int(row['id'])}",
                "document_path": f"{yaml_service.canonical_theme_id(str(row['theme']))}/{str(row['doc_id'])}",
                "username": username,
                "action": "review",
                "status": "completed",
                "timestamp": timestamp,
            }
        )

    task_rows = db.execute(
        """
        SELECT
            t.id,
            t.theme,
            t.doc_id,
            t.updated_at,
            t.started_at,
            t.assigned_at,
            u.username
        FROM review_campaign_tasks t
        LEFT JOIN users u ON u.id = t.assignee_user_id
        JOIN review_campaigns c ON c.id = t.campaign_id
        WHERE t.review_type = ?
          AND t.status = 'in_progress'
          AND c.status = 'active'
        ORDER BY t.updated_at DESC, t.id DESC
        LIMIT 100
        """,
        (normalized_type,),
    ).fetchall()
    for row in task_rows:
        if _is_excluded_from_review_campaign_doc(
            normalized_type,
            str(row["theme"] or ""),
            str(row["doc_id"] or ""),
        ):
            continue
        username = str(row["username"] or "").strip()
        timestamp = str(row["updated_at"] or row["started_at"] or row["assigned_at"] or "").strip()
        if not username or not timestamp:
            continue
        events.append(
            {
                "id": f"{normalized_type}/task/{int(row['id'])}",
                "document_path": f"{yaml_service.canonical_theme_id(str(row['theme']))}/{str(row['doc_id'])}",
                "username": username,
                "action": "edit",
                "status": "in_progress",
                "timestamp": timestamp,
            }
        )

    agreement_rows = db.execute(
        """
        SELECT
            a.id,
            a.theme,
            a.doc_id,
            a.status,
            a.resolved_at,
            u.username
        FROM review_campaign_agreements a
        LEFT JOIN users u ON u.id = a.resolved_by_user_id
        WHERE a.review_type = ?
          AND a.resolved_at IS NOT NULL
        ORDER BY a.resolved_at DESC, a.id DESC
        LIMIT 100
        """,
        (normalized_type,),
    ).fetchall()
    for row in agreement_rows:
        if _is_excluded_from_review_campaign_doc(
            normalized_type,
            str(row["theme"] or ""),
            str(row["doc_id"] or ""),
        ):
            continue
        username = str(row["username"] or "").strip()
        timestamp = str(row["resolved_at"] or "").strip()
        if not username or not timestamp:
            continue
        events.append(
            {
                "id": f"{normalized_type}/agreement/{int(row['id'])}",
                "document_path": f"{yaml_service.canonical_theme_id(str(row['theme']))}/{str(row['doc_id'])}",
                "username": username,
                "action": "validate",
                "status": str(row["status"] or "completed"),
                "timestamp": timestamp,
            }
        )

    events.sort(key=lambda entry: (str(entry.get("timestamp") or ""), str(entry.get("id") or "")), reverse=True)
    return events[: max(int(limit), 0)]


def get_review_status_map(db=None) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    if db is None:
        db = get_db()
    activity_map = get_review_document_activity_map(db=db)

    under_reviewed_doc_keys: set[tuple[str, str, str]] = set()
    try:
        submission_rows = db.execute(
            """
            SELECT
                c.review_type,
                t.theme,
                t.doc_id,
                COUNT(s.id) AS submission_count
            FROM review_campaign_tasks t
            JOIN review_campaigns c ON c.id = t.campaign_id
            LEFT JOIN review_campaign_submissions s
                   ON s.campaign_id = t.campaign_id
                  AND s.theme = t.theme
                  AND s.doc_id = t.doc_id
            WHERE c.status = 'active'
            GROUP BY c.review_type, t.theme, t.doc_id
            """
        ).fetchall()
    except Exception:
        submission_rows = []
    for row in submission_rows:
        review_type = str(row["review_type"] or "").strip().lower()
        if review_type not in REVIEW_TYPES:
            continue
        theme = yaml_service.canonical_theme_id(str(row["theme"]))
        doc_id = str(row["doc_id"] or "")
        if _is_excluded_from_review_campaign_doc(review_type, theme, doc_id):
            continue
        required_submissions = _required_review_submissions(review_type)
        submission_count = int(row["submission_count"] or 0)
        if submission_count >= required_submissions:
            continue
        under_reviewed_doc_keys.add(
            (
                review_type,
                theme,
                doc_id,
            )
        )
    try:
        rows = db.execute(
            """
            SELECT
                ras.theme,
                ras.doc_id,
                ras.review_type,
                ras.status,
                ras.reviewed_at,
                ras.latest_snapshot_path,
                u.username AS reviewed_by
            FROM review_artifact_statuses ras
            LEFT JOIN users u ON u.id = ras.reviewed_by_user_id
            """
        ).fetchall()
    except Exception:
        rows = []

    result: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for key, review_types in activity_map.items():
        bucket = result.setdefault(key, {})
        for review_type, activity in review_types.items():
            bucket[review_type] = {
                "status": "draft",
                "reviewed": False,
                "reviewed_at": None,
                "reviewed_by": None,
                "last_edited_by": activity.get("last_edited_by"),
                "activity_users": list(activity.get("activity_users") or []),
                "last_activity_at": activity.get("last_activity_at"),
                "last_activity_user": activity.get("last_activity_user"),
                "last_activity_action": activity.get("last_activity_action"),
                "ignored_qa_count": 0,
                "has_ignored_qas": False,
            }
    for row in rows:
        key = (yaml_service.canonical_theme_id(str(row["theme"])), str(row["doc_id"]))
        bucket = result.setdefault(key, {})
        review_type = str(row["review_type"] or "").strip().lower()
        if review_type in REVIEW_TYPES and _is_excluded_from_review_campaign_doc(review_type, key[0], key[1]):
            continue
        status = str(row["status"] or "draft")
        ignored_qa_count = 0
        if review_type == "questions":
            ignored_qa_count = _question_review_snapshot_ignored_count(row["latest_snapshot_path"])
        # Guardrail: a document cannot be considered completed while required
        # reviewer submissions are still missing in the active campaign.
        if (
            status == "completed"
            and (review_type, key[0], key[1]) in under_reviewed_doc_keys
        ):
            status = "in_progress"
        existing = bucket.get(review_type, {})
        bucket[review_type] = {
            "status": status,
            "reviewed": status == "completed",
            "reviewed_at": row["reviewed_at"],
            "reviewed_by": row["reviewed_by"],
            "last_edited_by": existing.get("last_edited_by"),
            "activity_users": list(existing.get("activity_users") or []),
            "last_activity_at": existing.get("last_activity_at"),
            "last_activity_user": existing.get("last_activity_user"),
            "last_activity_action": existing.get("last_activity_action"),
            "ignored_qa_count": ignored_qa_count,
            "has_ignored_qas": ignored_qa_count > 0,
        }
    return result


def get_document_review_statuses(theme: str, doc_id: str, db=None) -> dict[str, dict[str, Any]]:
    status_map = get_review_status_map(db=db)
    statuses = status_map.get((yaml_service.canonical_theme_id(str(theme)), str(doc_id)), {})
    return {
        "rules": statuses.get(
            "rules",
            {
                "status": "draft",
                "reviewed": False,
                "reviewed_at": None,
                "reviewed_by": None,
                "last_edited_by": None,
                "activity_users": [],
                "last_activity_at": None,
                "last_activity_user": None,
                "last_activity_action": None,
                "ignored_qa_count": 0,
                "has_ignored_qas": False,
            },
        ),
        "questions": statuses.get(
            "questions",
            {
                "status": "draft",
                "reviewed": False,
                "reviewed_at": None,
                "reviewed_by": None,
                "last_edited_by": None,
                "activity_users": [],
                "last_activity_at": None,
                "last_activity_user": None,
                "last_activity_action": None,
                "ignored_qa_count": 0,
                "has_ignored_qas": False,
            },
        ),
    }


def _active_task_row_for_user_doc(review_type: str, user_id: int, theme: str, doc_id: str, db=None):
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    row = db.execute(
        """
        SELECT t.*, c.name AS campaign_name, c.status AS campaign_status
        FROM review_campaign_tasks t
        JOIN review_campaigns c ON c.id = t.campaign_id
        WHERE c.review_type = ?
          AND c.status = 'active'
          AND t.assignee_user_id = ?
          AND t.theme = ?
          AND t.doc_id = ?
          AND t.status IN ('available', 'in_progress')
        ORDER BY CASE t.status WHEN 'in_progress' THEN 0 ELSE 1 END ASC, t.id ASC
        LIMIT 1
        """,
        (normalized_type, int(user_id), str(theme), str(doc_id)),
    ).fetchone()
    if row and _is_excluded_from_review_campaign_doc(
        normalized_type,
        str(row["theme"] or ""),
        str(row["doc_id"] or ""),
    ):
        return None
    return row


def get_review_task_for_user_document(
    review_type: str,
    user_id: int,
    theme: str,
    doc_id: str,
    db=None,
) -> dict[str, Any] | None:
    row = _active_task_row_for_user_doc(review_type, user_id, theme, doc_id, db=db)
    return dict(row) if row else None


def get_assigned_review_task_for_user_document(
    review_type: str,
    user_id: int,
    theme: str,
    doc_id: str,
    db=None,
) -> dict[str, Any] | None:
    normalized_type = _validate_review_type(review_type)
    if db is None:
        db = get_db()
    row = db.execute(
        """
        SELECT t.*, c.name AS campaign_name, c.status AS campaign_status
        FROM review_campaign_tasks t
        JOIN review_campaigns c ON c.id = t.campaign_id
        WHERE c.review_type = ?
          AND c.status = 'active'
          AND t.assignee_user_id = ?
          AND t.theme = ?
          AND t.doc_id = ?
        ORDER BY CASE t.status WHEN 'in_progress' THEN 0 WHEN 'available' THEN 1 ELSE 2 END, t.id ASC
        LIMIT 1
        """,
        (normalized_type, int(user_id), str(theme), str(doc_id)),
    ).fetchone()
    if row and _is_excluded_from_review_campaign_doc(
        normalized_type,
        str(row["theme"] or ""),
        str(row["doc_id"] or ""),
    ):
        return None
    return dict(row) if row else None


def _ensure_review_output_snapshot(task: dict[str, Any], db) -> Path:
    output_path_raw = task.get("output_snapshot_path")
    if isinstance(output_path_raw, str) and output_path_raw:
        output_path = _resolve_review_storage_path(output_path_raw)
        if output_path is not None and not output_path.exists():
            restore_work_file_from_gcs(output_path)
        if output_path is not None and output_path.exists():
            if str(output_path) != str(output_path_raw):
                db.execute(
                    """
                    UPDATE review_campaign_tasks
                    SET output_snapshot_path = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (str(output_path), _utc_now(), int(task["id"])),
                )
                db.commit()
                sync_db_to_gcs()
            return output_path

    input_path_raw = task.get("input_snapshot_path")
    input_path = _resolve_review_storage_path(input_path_raw)
    if input_path is not None and not input_path.exists():
        restore_work_file_from_gcs(input_path)
    if input_path is None or not input_path.exists():
        canonical_campaign_input = _campaign_input_path(
            int(task["campaign_id"]),
            str(task["review_type"]),
            str(task["theme"]),
            str(task["doc_id"]),
        )
        if not canonical_campaign_input.exists():
            restore_work_file_from_gcs(canonical_campaign_input)
        if canonical_campaign_input.exists():
            input_path = canonical_campaign_input
        else:
            source_path = review_source_path(
                str(task["review_type"]),
                str(task["theme"]),
                str(task["doc_id"]),
            )
            if not source_path.exists():
                restore_work_file_from_gcs(source_path)
            if source_path.exists():
                _copy_file(source_path, canonical_campaign_input)
                input_path = canonical_campaign_input
                db.execute(
                    """
                    UPDATE review_campaign_tasks
                    SET input_snapshot_path = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (str(canonical_campaign_input), _utc_now(), int(task["id"])),
                )
                db.commit()
                sync_db_to_gcs()
            else:
                raise FileNotFoundError("Review task input snapshot is unavailable.")

    output_path = _campaign_output_path(
        int(task["campaign_id"]),
        str(task["review_type"]),
        int(task["assignee_user_id"]),
        str(task["theme"]),
        str(task["doc_id"]),
    )
    _copy_file(input_path, output_path)
    db.execute(
        """
        UPDATE review_campaign_tasks
        SET output_snapshot_path = ?, updated_at = ?
        WHERE id = ?
        """,
        (str(output_path), _utc_now(), int(task["id"])),
    )
    db.commit()
    sync_db_to_gcs()
    return output_path


def load_review_task_document(review_type: str, user_id: int, theme: str, doc_id: str) -> dict[str, Any]:
    db = get_db()
    row = _active_task_row_for_user_doc(review_type, user_id, theme, doc_id, db=db)
    if row is None:
        raise FileNotFoundError("No active review task for this document.")

    task = dict(row)
    now = _utc_now()
    if str(task["status"]) == "available":
        db.execute(
            """
            UPDATE review_campaign_tasks
            SET status = 'in_progress',
                started_at = COALESCE(started_at, ?),
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, int(task["id"])),
        )
        _upsert_artifact_status(
            review_type=str(task["review_type"]),
            theme=str(task["theme"]),
            doc_id=str(task["doc_id"]),
            status="in_progress",
            latest_task_id=int(task["id"]),
            latest_snapshot_path=str(task.get("output_snapshot_path") or task.get("input_snapshot_path") or ""),
            db=db,
        )
        db.commit()
        sync_db_to_gcs()
        task["status"] = "in_progress"

    output_path = _ensure_review_output_snapshot(task, db)
    payload = _load_yaml(output_path)
    document = payload.get("document", payload if isinstance(payload, dict) else {})
    if not isinstance(document, dict):
        document = {}
    if str(task.get("review_type") or "").strip().lower() == "questions":
        fictionalized_text = _load_fictionalized_question_document_text(str(task["theme"]), str(task["doc_id"]))
        if fictionalized_text:
            document["fictionalized_annotated_template_document"] = fictionalized_text
            # Regular QA reviewers should always work on the fictionalized context.
            document["document_to_annotate"] = fictionalized_text
        if not _question_review_document_has_sampleable_questions(document):
            raise FileNotFoundError("This document has no synced questions available for QA review.")
        _align_question_annotation_surfaces_with_document(document)
    document = normalize_document_taxonomy(document)
    if str(task.get("review_type") or "").strip().lower() == "questions":
        _align_question_annotation_surfaces_with_document(document)
    document["review_statuses"] = get_document_review_statuses(str(task["theme"]), str(task["doc_id"]), db=db)
    document["active_review_target"] = str(task["review_type"])
    document["active_review_task_status"] = str(task["status"])
    document["active_review_campaign_name"] = str(task.get("campaign_name") or "")
    return document


def save_review_task_document(review_type: str, user_id: int, theme: str, doc_id: str, document: dict[str, Any]) -> None:
    db = get_db()
    row = _active_task_row_for_user_doc(review_type, user_id, theme, doc_id, db=db)
    if row is None:
        raise FileNotFoundError("No active review task for this document.")

    task = dict(row)
    output_path = _ensure_review_output_snapshot(task, db)
    prepared_document = normalize_document_taxonomy(dict(document or {}))
    if str(task.get("review_type") or "").strip().lower() == "questions":
        prepared_document["questions"] = _normalize_question_review_questions(prepared_document.get("questions"))
        prepared_document["num_questions"] = len(prepared_document["questions"])
        prepared_document[QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD] = _normalize_question_review_coverage_exemptions(
            prepared_document.get(QUESTION_REVIEW_COVERAGE_EXEMPTIONS_FIELD)
        )
    payload = {"document": prepared_document}
    payload["document"].pop("review_statuses", None)
    payload["document"].pop("active_review_target", None)
    payload["document"].pop("active_review_task_status", None)
    payload["document"].pop("active_review_campaign_name", None)
    _write_yaml(output_path, payload)

    now = _utc_now()
    db.execute(
        """
        UPDATE review_campaign_tasks
        SET status = 'in_progress',
            started_at = COALESCE(started_at, ?),
            output_snapshot_path = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (now, str(output_path), now, int(task["id"])),
    )
    _upsert_artifact_status(
        review_type=str(task["review_type"]),
        theme=str(task["theme"]),
        doc_id=str(task["doc_id"]),
        status="in_progress",
        latest_task_id=int(task["id"]),
        latest_snapshot_path=str(output_path),
        db=db,
    )
    db.commit()
    sync_db_to_gcs()


def finish_review_task(review_type: str, user_id: int, theme: str, doc_id: str) -> dict[str, Any]:
    db = get_db()
    row = _active_task_row_for_user_doc(review_type, user_id, theme, doc_id, db=db)
    if row is None:
        raise FileNotFoundError("No active review task for this document.")

    task = dict(row)
    output_path = _ensure_review_output_snapshot(task, db)
    now = _utc_now()

    if str(task.get("review_type") or "").strip().lower() == "questions":
        payload = _load_yaml(output_path)
        raw_document = payload.get("document", payload if isinstance(payload, dict) else {})
        if not isinstance(raw_document, dict):
            raw_document = {}
        normalized_document = _normalize_question_review_document(raw_document)
        _validate_question_review_submission_contract(normalized_document)
        _write_yaml(output_path, {"document": normalized_document})

    if _reviewer_has_review_submission(int(task["campaign_id"]), int(user_id), str(theme), str(doc_id), db):
        raise ValueError("This reviewer already submitted a review for this document.")

    db.execute(
        """
        INSERT INTO review_campaign_submissions (
            campaign_id,
            review_type,
            theme,
            doc_id,
            reviewer_user_id,
            task_id,
            snapshot_path,
            submitted_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(task["campaign_id"]),
            str(task["review_type"]),
            str(task["theme"]),
            str(task["doc_id"]),
            int(user_id),
            int(task["id"]),
            str(output_path),
            now,
            now,
        ),
    )

    agreement = _upsert_review_campaign_agreement_state(
        campaign_id=int(task["campaign_id"]),
        review_type=str(task["review_type"]),
        theme=str(task["theme"]),
        doc_id=str(task["doc_id"]),
        db=db,
    )
    required_submissions = _required_review_submissions(str(task["review_type"]))
    submission_count = _review_campaign_submission_count(
        int(task["campaign_id"]),
        str(task["theme"]),
        str(task["doc_id"]),
        db,
    )
    parallel_task_count = int(
        db.execute(
            """
            SELECT COUNT(*)
            FROM review_campaign_tasks
            WHERE campaign_id = ?
              AND review_type = ?
              AND theme = ?
              AND doc_id = ?
              AND qa_group IS NOT NULL
            """,
            (
                int(task["campaign_id"]),
                str(task["review_type"]),
                str(task["theme"]),
                str(task["doc_id"]),
            ),
        ).fetchone()[0]
    )

    if submission_count < required_submissions:
        if parallel_task_count > 1:
            db.execute(
                """
                UPDATE review_campaign_tasks
                SET status = 'completed',
                    completed_at = COALESCE(completed_at, ?),
                    output_snapshot_path = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, str(output_path), now, int(task["id"])),
            )
            _upsert_artifact_status(
                review_type=str(task["review_type"]),
                theme=str(task["theme"]),
                doc_id=str(task["doc_id"]),
                status="in_progress",
                latest_task_id=int(task["id"]),
                latest_snapshot_path=str(output_path),
                db=db,
            )
            db.commit()
            sync_db_to_gcs()
            return {
                "status": "submitted",
                "review_type": str(task["review_type"]),
                "theme": str(task["theme"]),
                "doc_id": str(task["doc_id"]),
                "submission_count": submission_count,
                "agreement_status": str(agreement.get("status") or "pending"),
            }

        next_assignee_user_id = int(task["assignee_user_id"])
        if task.get("initial_assignee_user_id") is not None:
            campaign_row = db.execute(
                "SELECT created_by_user_id FROM review_campaigns WHERE id = ?",
                (int(task["campaign_id"]),),
            ).fetchone()
            if campaign_row is not None:
                next_assignee_user_id = int(campaign_row["created_by_user_id"])
        db.execute(
            """
            UPDATE review_campaign_tasks
            SET status = 'available',
                assignee_user_id = ?,
                output_snapshot_path = NULL,
                started_at = NULL,
                completed_at = NULL,
                updated_at = ?
            WHERE id = ?
            """,
            (int(next_assignee_user_id), now, int(task["id"])),
        )
        _upsert_artifact_status(
            review_type=str(task["review_type"]),
            theme=str(task["theme"]),
            doc_id=str(task["doc_id"]),
            status="in_progress",
            latest_task_id=int(task["id"]),
            latest_snapshot_path=str(output_path),
            db=db,
        )
        db.commit()
        sync_db_to_gcs()
        return {
            "status": "submitted",
            "review_type": str(task["review_type"]),
            "theme": str(task["theme"]),
            "doc_id": str(task["doc_id"]),
            "submission_count": submission_count,
            "agreement_status": str(agreement.get("status") or "pending"),
        }

    db.execute(
        """
        UPDATE review_campaign_tasks
        SET status = 'completed',
            completed_at = COALESCE(completed_at, ?),
            output_snapshot_path = NULL,
            updated_at = ?
        WHERE id = ?
        """,
        (now, now, int(task["id"])),
    )
    _upsert_artifact_status(
        review_type=str(task["review_type"]),
        theme=str(task["theme"]),
        doc_id=str(task["doc_id"]),
        status="in_progress",
        latest_task_id=int(task["id"]),
        latest_snapshot_path=str(output_path),
        db=db,
    )
    db.commit()
    sync_db_to_gcs()

    return {
        "status": "agreement_ready",
        "review_type": str(task["review_type"]),
        "theme": str(task["theme"]),
        "doc_id": str(task["doc_id"]),
        "submission_count": submission_count,
        "agreement_status": str(agreement.get("status") or "ready"),
    }


def _maybe_complete_review_campaign(campaign_id: int, db) -> None:
    # Review campaigns are closed explicitly by the admin.
    return None


def _resolve_legacy_review_submission_snapshot(task: dict[str, Any]) -> Path | None:
    output_path = _resolve_review_storage_path(task.get("output_snapshot_path"))
    if output_path is not None and output_path.exists():
        return output_path
    candidate = _campaign_output_path(
        int(task["campaign_id"]),
        str(task["review_type"]),
        int(task["assignee_user_id"]),
        str(task["theme"]),
        str(task["doc_id"]),
    )
    if candidate.exists():
        return candidate
    return None


def repair_two_pass_review_campaign(campaign_id: int) -> dict[str, Any]:
    db = get_db()
    campaign_row = db.execute(
        "SELECT * FROM review_campaigns WHERE id = ?",
        (int(campaign_id),),
    ).fetchone()
    if campaign_row is None:
        raise FileNotFoundError("Review campaign not found.")

    campaign = dict(campaign_row)
    normalized_type = _validate_review_type(str(campaign["review_type"]))
    active = get_active_review_campaign(normalized_type, db=db)
    if active is not None and int(active["id"]) != int(campaign_id):
        raise ValueError(
            f"Cannot reopen campaign {campaign_id}: active {normalized_type} campaign {active['id']} already exists."
        )

    tasks = db.execute(
        """
        SELECT *
        FROM review_campaign_tasks
        WHERE campaign_id = ?
        ORDER BY id ASC
        """,
        (int(campaign_id),),
    ).fetchall()

    now = _utc_now()
    shared_assignee_user_id = int(campaign.get("created_by_user_id") or 0)
    backfilled_submissions = 0
    reopened_tasks = 0
    ready_for_agreement = 0
    untouched_tasks = 0
    missing_submission_snapshots = 0
    required_submissions = _required_review_submissions(normalized_type)

    for row in tasks:
        task = dict(row)
        theme = str(task["theme"])
        doc_id = str(task["doc_id"])
        reviewer_user_id = int(task["assignee_user_id"])
        submission_path = _resolve_legacy_review_submission_snapshot(task)
        if submission_path is not None and not _reviewer_has_review_submission(
            int(campaign_id),
            reviewer_user_id,
            theme,
            doc_id,
            db,
        ):
            submitted_at = (
                str(task.get("completed_at") or "").strip()
                or str(task.get("updated_at") or "").strip()
                or str(task.get("started_at") or "").strip()
                or str(task.get("assigned_at") or "").strip()
                or now
            )
            cursor = db.execute(
                """
                INSERT INTO review_campaign_submissions (
                    campaign_id,
                    review_type,
                    theme,
                    doc_id,
                    reviewer_user_id,
                    task_id,
                    snapshot_path,
                    submitted_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id, theme, doc_id, reviewer_user_id) DO NOTHING
                """,
                (
                    int(campaign_id),
                    normalized_type,
                    theme,
                    doc_id,
                    reviewer_user_id,
                    int(task["id"]),
                    str(submission_path),
                    submitted_at,
                    now,
                ),
            )
            if int(cursor.rowcount or 0) > 0:
                backfilled_submissions += 1
        elif str(task.get("status") or "").strip().lower() == "completed":
            missing_submission_snapshots += 1

        agreement = _upsert_review_campaign_agreement_state(
            campaign_id=int(campaign_id),
            review_type=normalized_type,
            theme=theme,
            doc_id=doc_id,
            db=db,
        )
        submissions = _get_review_campaign_submissions(int(campaign_id), theme, doc_id, db)
        submission_count = len(submissions)
        latest_snapshot_path = str(
            submissions[-1]["snapshot_path"]
            if submissions
            else task.get("input_snapshot_path") or ""
        )

        if submission_count >= required_submissions:
            db.execute(
                """
                UPDATE review_campaign_tasks
                SET status = 'completed',
                    output_snapshot_path = NULL,
                    completed_at = COALESCE(completed_at, ?),
                    updated_at = ?
                WHERE id = ?
                """,
                (now, now, int(task["id"])),
            )
            agreement_status = str(agreement.get("status") or "ready").strip().lower()
            artifact_status = (
                "completed"
                if agreement_status in {"resolved", "awaiting_reviewer_acceptance"}
                else "in_progress"
            )
            _upsert_artifact_status(
                review_type=normalized_type,
                theme=theme,
                doc_id=doc_id,
                status=artifact_status,
                latest_task_id=int(task["id"]),
                latest_snapshot_path=latest_snapshot_path,
                db=db,
            )
            if agreement_status == "ready":
                ready_for_agreement += 1
        elif submission_count > 0:
            if normalized_type == "rules":
                restored = _restore_rules_review_from_legacy_workflow(
                    campaign_id=int(campaign_id),
                    theme=theme,
                    doc_id=doc_id,
                    task_id=int(task["id"]),
                    db=db,
                )
                if restored and bool(restored.get("restored")):
                    backfilled_submissions += int(restored.get("backfilled_submissions") or 0)
                    continue
            next_assignee_user_id = shared_assignee_user_id or int(task.get("assignee_user_id") or 0)
            db.execute(
                """
                UPDATE review_campaign_tasks
                SET status = 'available',
                    assignee_user_id = ?,
                    output_snapshot_path = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (next_assignee_user_id, now, int(task["id"])),
            )
            _upsert_artifact_status(
                review_type=normalized_type,
                theme=theme,
                doc_id=doc_id,
                status="in_progress",
                latest_task_id=int(task["id"]),
                latest_snapshot_path=latest_snapshot_path,
                db=db,
            )
            reopened_tasks += 1
        else:
            db.execute(
                """
                UPDATE review_campaign_tasks
                SET status = 'available',
                    output_snapshot_path = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (now, int(task["id"])),
            )
            _upsert_artifact_status(
                review_type=normalized_type,
                theme=theme,
                doc_id=doc_id,
                status="draft",
                latest_task_id=int(task["id"]),
                latest_snapshot_path=str(task.get("input_snapshot_path") or ""),
                db=db,
            )
            untouched_tasks += 1

    db.execute(
        "UPDATE review_campaigns SET status = 'active' WHERE id = ?",
        (int(campaign_id),),
    )
    db.commit()
    sync_db_to_gcs()
    return {
        "campaign_id": int(campaign_id),
        "review_type": normalized_type,
        "campaign_status": "active",
        "backfilled_submissions": backfilled_submissions,
        "reopened_tasks": reopened_tasks,
        "ready_for_agreement": ready_for_agreement,
        "untouched_tasks": untouched_tasks,
        "missing_submission_snapshots": missing_submission_snapshots,
    }


def _assign_random_review_task_for_user(review_type: str, user_id: int, db) -> dict[str, Any] | None:
    active_campaign = get_active_review_campaign(review_type, db=db)
    if active_campaign is None:
        return None
    if not _is_user_allowed_in_campaign(int(active_campaign["id"]), int(user_id), db):
        return None
    eligible_doc_keys = _eligible_review_doc_keys(review_type, db=db)

    rows = db.execute(
        """
        SELECT
            t.*,
            COALESCE(submissions.submission_count, 0) AS submission_count
        FROM review_campaign_tasks t
        LEFT JOIN (
            SELECT campaign_id, theme, doc_id, COUNT(*) AS submission_count
            FROM review_campaign_submissions
            GROUP BY campaign_id, theme, doc_id
        ) submissions
          ON submissions.campaign_id = t.campaign_id
         AND submissions.theme = t.theme
         AND submissions.doc_id = t.doc_id
        WHERE t.campaign_id = ?
          AND t.status = 'available'
          AND NOT EXISTS (
              SELECT 1
              FROM review_campaign_submissions s
              WHERE s.campaign_id = t.campaign_id
                AND s.theme = t.theme
                AND s.doc_id = t.doc_id
                AND s.reviewer_user_id = ?
          )
        ORDER BY t.id ASC
        """,
        (int(active_campaign["id"]), int(user_id)),
    ).fetchall()
    if eligible_doc_keys is not None:
        rows = [
            row for row in rows
            if (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or "")) in eligible_doc_keys
        ]
    rows = [
        row for row in rows
        if not _is_excluded_from_review_campaign_doc(
            review_type,
            str(row["theme"] or ""),
            str(row["doc_id"] or ""),
        )
    ]
    if not rows:
        return None

    shared_assignee_id = int(active_campaign.get("created_by_user_id") or 0)
    prioritized_rows = []
    fallback_rows = []
    for row in rows:
        initial_user_id = row["initial_assignee_user_id"]
        submission_count = int(row["submission_count"] or 0)
        if initial_user_id is None:
            fallback_rows.append(row)
            continue
        if submission_count <= 0:
            if int(initial_user_id) == int(user_id):
                prioritized_rows.append(row)
            continue
        fallback_rows.append(row)

    if prioritized_rows:
        prioritized_rows.sort(
            key=lambda row: (
                QUESTION_EXPERIMENT_GROUP_SORT_ORDER.get(
                    _normalize_question_experiment_group(row["qa_group"]),
                    999,
                ),
                int(row["qa_group_order"] or 0),
                int(row["id"]),
            )
        )
        chosen = prioritized_rows[0]
    else:
        fallback_rows = [
            row
            for row in fallback_rows
            if row["initial_assignee_user_id"] is None
            or int(row["assignee_user_id"] or 0) in {0, int(user_id), shared_assignee_id}
        ]
        if not fallback_rows:
            return None
        chosen = random.choice(fallback_rows)

    now = _utc_now()
    db.execute(
        """
        UPDATE review_campaign_tasks
        SET assignee_user_id = ?,
            status = 'in_progress',
            started_at = COALESCE(started_at, ?),
            updated_at = ?
        WHERE id = ?
        """,
        (int(user_id), now, now, int(chosen["id"])),
    )
    chosen_dict = dict(chosen)
    chosen_dict["assignee_user_id"] = int(user_id)
    chosen_dict["status"] = "in_progress"
    _upsert_artifact_status(
        review_type=str(chosen_dict["review_type"]),
        theme=str(chosen_dict["theme"]),
        doc_id=str(chosen_dict["doc_id"]),
        status="in_progress",
        latest_task_id=int(chosen_dict["id"]),
        latest_snapshot_path=str(chosen_dict["input_snapshot_path"]),
        db=db,
    )
    db.commit()
    sync_db_to_gcs()
    return chosen_dict


def assign_random_review_task_to_user(user_id: int, review_type: str) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return {
            "has_active_campaign": False,
            "assigned": False,
            "reason": "no_active_campaign",
            "task": None,
        }

    if not _is_user_allowed_in_campaign(int(campaign["id"]), int(user_id), db):
        return {
            "has_active_campaign": True,
            "is_participant": False,
            "assigned": False,
            "reason": "not_participant",
            "task": None,
        }

    existing_rows = db.execute(
        """
        SELECT id, theme, doc_id, status, input_snapshot_path, output_snapshot_path
        FROM review_campaign_tasks
        WHERE campaign_id = ?
          AND assignee_user_id = ?
          AND status = 'in_progress'
        ORDER BY id ASC
        LIMIT 20
        """,
        (int(campaign["id"]), int(user_id)),
    ).fetchall()
    if existing_rows:
        existing_rows = [
            row
            for row in existing_rows
            if not _is_excluded_from_review_campaign_doc(
                normalized_type,
                str(row["theme"] or ""),
                str(row["doc_id"] or ""),
            )
        ]
    if existing_rows:
        # Prefer an open task with sampleable seeded QAs when available, but
        # never hide/block an existing in-progress assignment from the user.
        row = existing_rows[0]
        if normalized_type == "questions":
            for candidate in existing_rows:
                candidate_path = candidate["output_snapshot_path"] or candidate["input_snapshot_path"]
                if _question_review_snapshot_has_sampleable_questions(candidate_path):
                    row = candidate
                    break
        return {
            "has_active_campaign": True,
            "is_participant": True,
            "assigned": False,
            "reason": "already_has_open_task",
            "task": {
                "id": int(row["id"]),
                "theme": str(row["theme"]),
                "doc_id": str(row["doc_id"]),
                "status": str(row["status"]),
            },
        }

    task = _assign_random_review_task_for_user(normalized_type, int(user_id), db)
    if task is None and _required_review_submissions(normalized_type) > 1:
        repair_two_pass_review_campaign(int(campaign["id"]))
        task = _assign_random_review_task_for_user(normalized_type, int(user_id), db)
    if task is None:
        return {
            "has_active_campaign": True,
            "is_participant": True,
            "assigned": False,
            "reason": "no_documents_available",
            "task": None,
        }

    return {
        "has_active_campaign": True,
        "is_participant": True,
        "assigned": True,
        "reason": "assigned",
        "task": {
            "id": int(task["id"]),
            "theme": str(task["theme"]),
            "doc_id": str(task["doc_id"]),
            "status": "in_progress",
        },
    }


def get_user_review_queue(user_id: int, review_type: str) -> dict[str, Any]:
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    campaign = get_active_review_campaign(normalized_type, db=db)
    if campaign is None:
        return {
            "review_type": normalized_type,
            "has_active_campaign": False,
            "current_tasks": [],
            "assigned_count": 0,
            "completed_count": 0,
            "remaining_count": 0,
            "question_experiment_groups": [],
        }

    if not _is_user_allowed_in_campaign(int(campaign["id"]), int(user_id), db):
        return {
            "review_type": normalized_type,
            "has_active_campaign": True,
            "campaign": {
                "id": int(campaign["id"]),
                "name": str(campaign["name"]),
            },
            "current_tasks": [],
            "assigned_count": 0,
            "completed_count": 0,
            "remaining_count": 0,
            "is_participant": False,
            "question_experiment_groups": [],
        }

    current_rows = db.execute(
        """
        SELECT theme, doc_id, status, qa_group, input_snapshot_path, output_snapshot_path
        FROM review_campaign_tasks
        WHERE campaign_id = ?
          AND assignee_user_id = ?
          AND status = 'in_progress'
        ORDER BY id ASC
        LIMIT 20
        """,
        (int(campaign["id"]), int(user_id)),
    ).fetchall()
    current_rows = [
        row
        for row in current_rows
        if not _is_excluded_from_review_campaign_doc(
            normalized_type,
            str(row["theme"] or ""),
            str(row["doc_id"] or ""),
        )
    ]
    current_tasks = []
    if current_rows:
        selected_row = current_rows[0]
        selected_path = selected_row["output_snapshot_path"] or selected_row["input_snapshot_path"]
        if normalized_type == "questions":
            # Keep prefilled-QA tasks preferred when possible, but if a task is
            # already in progress with no seeded QAs we still surface it so the
            # reviewer can open/edit instead of being blocked.
            for candidate in current_rows:
                candidate_path = candidate["output_snapshot_path"] or candidate["input_snapshot_path"]
                if _question_review_snapshot_has_sampleable_questions(candidate_path):
                    selected_row = candidate
                    selected_path = candidate_path
                    break
        ignored_qa_count = (
            _question_review_snapshot_ignored_count(selected_path)
            if normalized_type == "questions"
            else 0
        )
        current_tasks.append(
            {
                "theme": str(selected_row["theme"]),
                "doc_id": str(selected_row["doc_id"]),
                "status": str(selected_row["status"]),
                "qa_group": _normalize_question_experiment_group(selected_row["qa_group"]),
                "ignored_qa_count": ignored_qa_count,
                "has_ignored_qas": ignored_qa_count > 0,
            }
        )

    submitted_row = db.execute(
        """
        SELECT COUNT(*) AS c
        FROM review_campaign_submissions
        WHERE campaign_id = ?
          AND reviewer_user_id = ?
        """,
        (int(campaign["id"]), int(user_id)),
    ).fetchone()
    completed_count = int(submitted_row["c"] or 0) if submitted_row else 0
    question_experiment_groups: list[dict[str, Any]] = []
    if normalized_type == "questions":
        question_experiment_groups = _build_question_experiment_groups(int(campaign["id"]), int(user_id), db)

    if question_experiment_groups:
        assigned_count = sum(int(group.get("total_count") or 0) for group in question_experiment_groups)
        completed_count = sum(int(group.get("completed_count") or 0) for group in question_experiment_groups)
        remaining_count = max(0, int(assigned_count) - int(completed_count))
    else:
        assigned_rows = db.execute(
            """
            SELECT DISTINCT theme, doc_id
            FROM (
                SELECT theme, doc_id
                FROM review_campaign_submissions
                WHERE campaign_id = ?
                  AND reviewer_user_id = ?
                UNION
                SELECT theme, doc_id
                FROM review_campaign_tasks
                WHERE campaign_id = ?
                  AND assignee_user_id = ?
            ) assigned_docs
            """,
            (int(campaign["id"]), int(user_id), int(campaign["id"]), int(user_id)),
        ).fetchall()
        assigned_count = sum(
            1
            for row in assigned_rows
            if not _is_excluded_from_review_campaign_doc(
                normalized_type,
                str(row["theme"] or ""),
                str(row["doc_id"] or ""),
            )
        )
        if normalized_type == "questions":
            assigned_count = sum(
                1
                for row in db.execute(
                    """
                    SELECT DISTINCT theme, doc_id, input_snapshot_path
                    FROM review_campaign_tasks
                    WHERE campaign_id = ?
                      AND assignee_user_id = ?
                    """,
                    (int(campaign["id"]), int(user_id)),
                ).fetchall()
                if not _is_excluded_from_review_campaign_doc(
                    normalized_type,
                    str(row["theme"] or ""),
                    str(row["doc_id"] or ""),
                )
                if _question_review_snapshot_has_sampleable_questions(row["input_snapshot_path"])
            )
        submitted_rows = db.execute(
            """
            SELECT DISTINCT theme, doc_id
            FROM review_campaign_submissions
            WHERE campaign_id = ?
              AND reviewer_user_id = ?
            """,
            (int(campaign["id"]), int(user_id)),
        ).fetchall()
        completed_count = sum(
            1
            for row in submitted_rows
            if not _is_excluded_from_review_campaign_doc(
                normalized_type,
                str(row["theme"] or ""),
                str(row["doc_id"] or ""),
            )
        )
        remaining_count = max(0, int(assigned_count) - int(completed_count))

    return {
        "review_type": normalized_type,
        "has_active_campaign": True,
        "campaign": {
            "id": int(campaign["id"]),
            "name": str(campaign["name"]),
        },
        "current_tasks": current_tasks,
        "assigned_count": assigned_count,
        "completed_count": completed_count,
        "remaining_count": remaining_count,
        "is_participant": True,
        "question_experiment_groups": question_experiment_groups,
    }


def complete_review_campaign(campaign_id: int) -> dict[str, Any]:
    def _mark_completed() -> dict[str, Any]:
        db = get_db()
        row_local = db.execute(
            "SELECT id, review_type FROM review_campaigns WHERE id = ?",
            (int(campaign_id),),
        ).fetchone()
        if row_local is None:
            raise FileNotFoundError("Review campaign not found.")
        db.execute(
            "UPDATE review_campaigns SET status = 'completed' WHERE id = ?",
            (int(campaign_id),),
        )
        db.commit()
        return {"id": int(row_local["id"]), "review_type": str(row_local["review_type"]), "status": "completed"}

    def _is_sync_conflict(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return (
            "remote db changed" in text
            or "pull latest state" in text
            or "refusing to sync db to gcs without pulling latest remote state first" in text
        )

    result = _mark_completed()
    try:
        sync_db_to_gcs()
    except Exception as exc:
        if not _is_sync_conflict(exc):
            raise
        # Recover from stale local generation by rebasing on GCS and re-applying the same idempotent update.
        restore_state_from_gcs()
        result = _mark_completed()
        sync_db_to_gcs()
    return result


def get_review_campaign_monitor(review_type: str | None = None, db=None) -> dict[str, Any]:
    if db is None:
        db = get_db()
    types = [review_type] if review_type else list(REVIEW_TYPES)

    campaigns: list[dict[str, Any]] = []
    for raw_type in types:
        normalized_type = _validate_review_type(raw_type)
        campaign = get_active_review_campaign(normalized_type, db=db)
        if campaign is None:
            campaigns.append(
                {
                    "review_type": normalized_type,
                    "has_active_campaign": False,
                }
            )
            continue

        task_rows = db.execute(
            """
            SELECT
                t.status,
                t.theme,
                t.doc_id,
                u.username
            FROM review_campaign_tasks t
            LEFT JOIN users u ON u.id = t.assignee_user_id
            WHERE t.campaign_id = ?
            ORDER BY t.id ASC
            """,
            (int(campaign["id"]),),
        ).fetchall()
        reviewer_rows = db.execute(
            """
            SELECT u.username
            FROM review_campaign_reviewers r
            JOIN users u ON u.id = r.user_id
            WHERE r.campaign_id = ?
            ORDER BY u.username ASC
            """,
            (int(campaign["id"]),),
        ).fetchall()
        submission_rows = db.execute(
            """
            SELECT
                s.id,
                s.theme,
                s.doc_id,
                s.reviewer_user_id,
                s.snapshot_path,
                s.submitted_at,
                u.username
            FROM review_campaign_submissions s
            LEFT JOIN users u ON u.id = s.reviewer_user_id
            WHERE s.campaign_id = ?
            ORDER BY s.submitted_at ASC, s.id ASC
            """,
            (int(campaign["id"]),),
        ).fetchall()
        agreement_rows = db.execute(
            """
            SELECT *
            FROM review_campaign_agreements
            WHERE campaign_id = ?
            ORDER BY theme ASC, doc_id ASC
            """,
            (int(campaign["id"]),),
        ).fetchall()
        eligible_doc_keys = (
            _eligible_review_doc_keys(normalized_type, db=db)
            if normalized_type != "questions"
            else None
        )
        # Rules monitor should stay aligned to the current dataset/workflow
        # scope, while QA keeps its broader campaign totals behavior.
        if eligible_doc_keys is not None and normalized_type != "questions":
            preserved_task_keys = {
                (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                for row in task_rows
                if str(row["status"] or "").strip().lower() == "completed"
            }
            preserved_agreement_keys = {
                (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                for row in agreement_rows
                if str(row["status"] or "").strip().lower() in {"resolved", "awaiting_reviewer_acceptance", "ready"}
            }
            task_rows = [
                row for row in task_rows
                if (
                    (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in eligible_doc_keys
                    or (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in preserved_task_keys
                )
            ]
            submission_rows = [
                row for row in submission_rows
                if (
                    (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in eligible_doc_keys
                    or (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in preserved_task_keys
                    or (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in preserved_agreement_keys
                )
            ]
            agreement_rows = [
                row for row in agreement_rows
                if (
                    (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in eligible_doc_keys
                    or (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
                    in preserved_agreement_keys
                )
            ]
        # Apply the same doc/theme exclusion logic the assignment layer uses so
        # stale legacy agreement rows cannot inflate campaign counts.
        task_rows = [
            row
            for row in task_rows
            if (
                not _is_excluded_from_review_campaign_doc(
                    normalized_type,
                    str(row["theme"] or ""),
                    str(row["doc_id"] or ""),
                )
                or (
                    str(row["status"] or "").strip().lower() == "completed"
                )
            )
        ]
        submission_rows = [
            row
            for row in submission_rows
            if not _is_excluded_from_review_campaign_doc(
                normalized_type,
                str(row["theme"] or ""),
                str(row["doc_id"] or ""),
            )
        ]
        agreement_rows = [
            row
            for row in agreement_rows
            if (
                not _is_excluded_from_review_campaign_doc(
                    normalized_type,
                    str(row["theme"] or ""),
                    str(row["doc_id"] or ""),
                )
                or str(row["status"] or "").strip().lower() in {"resolved", "awaiting_reviewer_acceptance", "ready"}
            )
        ]
        feedback_state_map = {
            key: value
            for key, value in _get_review_feedback_acceptance_state_map(normalized_type, db=db).items()
            if int(value.get("campaign_id") or 0) == int(campaign["id"])
        }
        task_status_by_key: dict[tuple[str, str], str] = {}
        for row in task_rows:
            key = (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
            status = str(row["status"] or "").strip().lower()
            previous = task_status_by_key.get(key, "")
            if previous == "completed":
                continue
            if status == "completed" or not previous:
                task_status_by_key[key] = status
        agreement_status_by_key: dict[tuple[str, str], str] = {}
        for row in agreement_rows:
            key = (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
            status = str(row["status"] or "").strip().lower()
            previous = agreement_status_by_key.get(key, "")
            if previous in {"resolved", "awaiting_reviewer_acceptance"}:
                continue
            if status in {"resolved", "awaiting_reviewer_acceptance"} or not previous:
                agreement_status_by_key[key] = status
        document_keys = {
            (yaml_service.canonical_theme_id(str(row["theme"] or "")), str(row["doc_id"] or ""))
            for row in task_rows
            if str(row["theme"] or "").strip() and str(row["doc_id"] or "").strip()
        }
        document_keys.update(
            (
                yaml_service.canonical_theme_id(str(row["theme"] or "")),
                str(row["doc_id"] or ""),
            )
            for row in submission_rows
            if str(row["theme"] or "").strip() and str(row["doc_id"] or "").strip()
        )
        document_keys.update(
            (
                yaml_service.canonical_theme_id(str(row["theme"] or "")),
                str(row["doc_id"] or ""),
            )
            for row in agreement_rows
            if str(row["theme"] or "").strip() and str(row["doc_id"] or "").strip()
        )
        # Final safety filter on the deduplicated campaign scope too.
        document_keys = {
            key
            for key in document_keys
            if (
                not _is_excluded_from_review_campaign_doc(normalized_type, key[0], key[1])
                or (
                    str(task_status_by_key.get(key) or "").strip().lower() == "completed"
                    or str(agreement_status_by_key.get(key) or "").strip().lower() in {"resolved", "awaiting_reviewer_acceptance", "ready"}
                )
            )
        }

        counts = {
            "available": sum(1 for row in task_rows if str(row["status"]) == "available"),
            "in_progress": sum(1 for row in task_rows if str(row["status"]) == "in_progress"),
            "completed": sum(1 for row in task_rows if str(row["status"]) == "completed"),
        }
        reviewer_names = [str(row["username"]) for row in reviewer_rows]
        per_user: dict[str, dict[str, int]] = {
            username: {"submitted_count": 0, "active_count": 0}
            for username in reviewer_names
            if username
        }
        for row in submission_rows:
            username = str(row["username"] or "").strip()
            if not username:
                continue
            bucket = per_user.setdefault(username, {"submitted_count": 0, "active_count": 0})
            bucket["submitted_count"] += 1
        for row in task_rows:
            status = str(row["status"] or "").strip().lower()
            if status != "in_progress":
                continue
            username = str(row["username"] or "").strip()
            if not username:
                continue
            bucket = per_user.setdefault(username, {"submitted_count": 0, "active_count": 0})
            bucket["active_count"] += 1

        annotator_progress = []
        for username in sorted(per_user):
            values = per_user[username]
            annotator_progress.append(
                {
                    "username": username,
                    "submitted_count": int(values["submitted_count"]),
                    "active_count": int(values["active_count"]),
                }
            )

        open_tasks: list[dict[str, Any]] = []
        for row in task_rows:
            status = str(row["status"] or "").strip().lower()
            if status not in {"available", "in_progress"}:
                continue
            assignee = str(row["username"] or "").strip()
            if status == "available":
                assignee = ""
            open_tasks.append(
                {
                    "theme": str(row["theme"]),
                    "doc_id": str(row["doc_id"]),
                    "status": status,
                    "assignee": assignee,
                }
            )

        task_keys = sorted(document_keys)
        submission_map: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in submission_rows:
            key = (str(row["theme"]), str(row["doc_id"]))
            submission_map.setdefault(key, []).append(dict(row))
        agreement_map: dict[tuple[str, str], dict[str, Any]] = {}
        for row in agreement_rows:
            agreement_map[(str(row["theme"]), str(row["doc_id"]))] = dict(row)

        ignored_question_count_by_key: dict[tuple[str, str], int] = {}
        if normalized_type == "questions":
            for key, agreement in agreement_map.items():
                agreement_status = str(agreement.get("status") or "").strip().lower()
                final_snapshot_path = agreement.get("final_snapshot_path")
                final_ignored_count = _question_review_snapshot_ignored_count(final_snapshot_path)
                if final_ignored_count > 0:
                    ignored_question_count_by_key[key] = final_ignored_count
                if agreement_status not in {"resolved", "awaiting_reviewer_acceptance"}:
                    continue

        completed_excluded_task_count = _review_campaign_completed_excluded_task_count(
            normalized_type,
            task_rows,
            agreement_map=agreement_map,
        )

        no_submission = 0
        one_submission = 0
        two_submissions = 0
        pending_count = 0
        ready_count = 0
        resolved_count = 0
        awaiting_reviewer_acceptance_count = 0
        agreement_ready_details: list[dict[str, Any]] = []
        resolution_contest_requests: list[dict[str, Any]] = []
        required_submissions = _required_review_submissions(normalized_type)
        for theme_key, doc_id_key in task_keys:
            key = (theme_key, doc_id_key)
            submissions_for_doc = submission_map.get(key, [])
            submission_count = len(submissions_for_doc)
            canonical_theme_key = yaml_service.canonical_theme_id(theme_key)
            agreement = agreement_map.get(key) or {}
            state = feedback_state_map.get((normalized_type, yaml_service.canonical_theme_id(theme_key), doc_id_key)) or {}
            agreement_status = str(state.get("agreement_status") or agreement.get("status") or "ready").strip().lower()
            task_status = str(task_status_by_key.get(key) or "").strip().lower()
            auto_resolved_without_submission = (
                normalized_type == "rules"
                and canonical_theme_key in RULE_REVIEW_AUTO_RESOLVED_THEMES
                and submission_count <= 0
            )
            if auto_resolved_without_submission:
                two_submissions += 1
                resolved_count += 1
                continue
            if (
                normalized_type == "questions"
                and _is_excluded_from_review_campaign(normalized_type, theme_key)
                and task_status == "completed"
            ):
                # Public-attacks docs are intentionally excluded from QA assignment,
                # but when they are already completed we still count them as completed
                # in dashboard totals.
                two_submissions += 1
                resolved_count += 1
                continue
            excluded_theme_agreement_resolved = (
                _is_excluded_from_review_campaign(normalized_type, theme_key)
                and agreement_status in {"resolved", "awaiting_reviewer_acceptance"}
            )
            if excluded_theme_agreement_resolved:
                two_submissions += 1
                if agreement_status == "resolved":
                    resolved_count += 1
                else:
                    awaiting_reviewer_acceptance_count += 1
                continue
            if submission_count <= 0:
                no_submission += 1
            elif submission_count == 1:
                one_submission += 1
            else:
                two_submissions += 1

            if submission_count < required_submissions:
                pending_count += 1
                continue

            if agreement_status == "resolved":
                resolved_count += 1
            elif agreement_status == "awaiting_reviewer_acceptance":
                awaiting_reviewer_acceptance_count += 1
            else:
                ready_count += 1
                reviewer_a_username = str(submissions_for_doc[0].get("username") or "") if submissions_for_doc else ""
                reviewer_b_username = str(submissions_for_doc[1].get("username") or "") if len(submissions_for_doc) > 1 else ""
                ignored_qa_count = int(ignored_question_count_by_key.get(key) or 0)
                if normalized_type == "questions" and ignored_qa_count <= 0:
                    ignored_qa_count = max(
                        (
                            _question_review_snapshot_ignored_count(submission.get("snapshot_path"))
                            for submission in submissions_for_doc
                        ),
                        default=0,
                    )
                agreement_ready_details.append(
                    {
                        "theme": theme_key,
                        "doc_id": doc_id_key,
                        "reviewer_a_username": reviewer_a_username,
                        "reviewer_b_username": reviewer_b_username,
                        "reviewer_a": reviewer_a_username,
                        "reviewer_b": reviewer_b_username,
                        "submission_count": submission_count,
                        "ignored_qa_count": ignored_qa_count,
                        "has_ignored_qas": ignored_qa_count > 0,
                    }
                )
        for state in feedback_state_map.values():
            for reviewer_state in (state.get("reviewer_states") or {}).values():
                if str(reviewer_state.get("response_status") or "").strip().lower() != "contest_requested":
                    continue
                theme = str(state.get("theme") or "")
                doc_id = str(state.get("doc_id") or "")
                key = (theme, doc_id)
                ignored_qa_count = int(ignored_question_count_by_key.get(key) or 0)
                if normalized_type == "questions" and ignored_qa_count <= 0:
                    submissions_for_doc = submission_map.get(key, [])
                    ignored_qa_count = max(
                        (
                            _question_review_snapshot_ignored_count(submission.get("snapshot_path"))
                            for submission in submissions_for_doc
                        ),
                        default=0,
                    )
                resolution_contest_requests.append(
                    {
                        "theme": theme,
                        "doc_id": doc_id,
                        "requester_username": str(reviewer_state.get("reviewer_username") or ""),
                        "contest_variant": str(reviewer_state.get("reviewer_role") or ""),
                        "contestation_note": "Annotator requested a follow-up agreement review.",
                        "ignored_qa_count": ignored_qa_count,
                        "has_ignored_qas": ignored_qa_count > 0,
                    }
                )

        display_completed_count = resolved_count + awaiting_reviewer_acceptance_count
        if normalized_type != "questions":
            display_completed_count += completed_excluded_task_count

        # Use the preserved deduplicated campaign scope for dashboard totals so
        # completed/resolved docs that remain intentionally visible do not make
        # "completed" exceed "total".
        document_count = len(document_keys)

        campaigns.append(
            {
                "review_type": normalized_type,
                "has_active_campaign": True,
                "campaign": {
                    "id": int(campaign["id"]),
                    "name": str(campaign["name"]),
                    "status": str(campaign["status"]),
                    "seed": int(campaign["seed"]),
                },
                "document_count": document_count,
                "reviewers": reviewer_names,
                "status_counts": counts,
                "annotator_progress": annotator_progress,
                "submission_counts": {
                    "no_submission": no_submission,
                    "one_submission": one_submission,
                    "two_submissions": two_submissions,
                },
                "agreements": {
                    "pending_count": pending_count,
                    "ready_count": ready_count,
                    "resolved_count": resolved_count,
                    "display_completed_count": display_completed_count,
                    "completed_excluded_task_count": completed_excluded_task_count,
                    "awaiting_reviewer_acceptance_count": awaiting_reviewer_acceptance_count,
                },
                "agreement_ready_details": agreement_ready_details,
                "resolution_contest_requests": resolution_contest_requests,
                "open_tasks": open_tasks[:30],
            }
        )
    return {"campaigns": campaigns}


def export_reviewed_template_fields(
    review_type: str,
    *,
    themes: list[str] | None = None,
    doc_ids: list[str] | None = None,
    campaign_id: int | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Write reviewed rules/questions back into completed template YAML files."""
    normalized_type = _validate_review_type(review_type)
    artifacts = list_completed_review_artifacts(
        normalized_type,
        themes=themes,
        doc_ids=doc_ids,
        campaign_id=campaign_id,
        limit=limit,
    )
    updated = 0
    skipped = 0

    for artifact in artifacts:
        theme = str(artifact["theme"])
        doc_id = str(artifact["doc_id"])
        snapshot_path = artifact.get("snapshot_path")
        if isinstance(snapshot_path, Path) and not snapshot_path.exists():
            restore_work_file_from_gcs(snapshot_path)
        if not isinstance(snapshot_path, Path) or not snapshot_path.exists():
            skipped += 1
            continue

        template_path = HUMAN_ANNOTATED_TEMPLATES_DIR / theme / f"{doc_id}.yaml"
        if not template_path.exists():
            skipped += 1
            continue

        template_payload = _load_yaml(template_path)
        snapshot_payload = _load_yaml(snapshot_path)
        template_document = template_payload.get("document")
        snapshot_document = snapshot_payload.get("document")
        if not isinstance(template_document, dict) or not isinstance(snapshot_document, dict):
            skipped += 1
            continue

        if normalized_type == "rules":
            template_document["rules"] = list(snapshot_document.get("rules") or [])
        else:
            questions = snapshot_document.get("questions") or []
            template_document["questions"] = questions
            template_document["num_questions"] = len(questions)
            _align_question_annotation_surfaces_with_document(template_document)

        with open(template_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(template_payload, handle, sort_keys=False, allow_unicode=True, width=10000)
        updated += 1

    return {"updated": updated, "skipped": skipped}


def list_completed_review_artifacts(
    review_type: str,
    *,
    themes: list[str] | None = None,
    doc_ids: list[str] | None = None,
    campaign_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return completed review artifacts with resolved snapshot paths."""
    normalized_type = _validate_review_type(review_type)
    db = get_db()
    rows = db.execute(
        """
        SELECT
            ras.theme,
            ras.doc_id,
            ras.latest_snapshot_path,
            ras.status,
            ras.reviewed_at
        FROM review_artifact_statuses ras
        WHERE ras.review_type = ?
          AND ras.status = 'completed'
          AND (
              ? IS NULL
              OR EXISTS (
                  SELECT 1
                  FROM review_campaign_agreements a
                  WHERE a.review_type = ras.review_type
                    AND a.theme = ras.theme
                    AND a.doc_id = ras.doc_id
                    AND a.campaign_id = ?
                    AND a.status = 'resolved'
              )
          )
        ORDER BY
            CASE WHEN ras.reviewed_at IS NULL THEN 1 ELSE 0 END ASC,
            ras.reviewed_at DESC,
            ras.theme ASC,
            ras.doc_id ASC
        """,
        (
            normalized_type,
            int(campaign_id) if campaign_id is not None else None,
            int(campaign_id) if campaign_id is not None else None,
        ),
    ).fetchall()

    theme_filter = {
        yaml_service.canonical_theme_id(str(theme).strip())
        for theme in (themes or [])
        if str(theme).strip()
    }
    doc_filter = {str(doc_id).strip() for doc_id in (doc_ids or []) if str(doc_id).strip()}
    artifacts: list[dict[str, Any]] = []

    for row in rows:
        theme = yaml_service.canonical_theme_id(str(row["theme"] or ""))
        doc_id = str(row["doc_id"] or "")
        if theme_filter and theme not in theme_filter:
            continue
        if doc_filter and doc_id not in doc_filter:
            continue

        raw_snapshot_path = str(row["latest_snapshot_path"] or "").strip()
        snapshot_path = _resolve_review_storage_path(raw_snapshot_path)
        artifacts.append(
            {
                "theme": theme,
                "doc_id": doc_id,
                "status": str(row["status"] or "completed"),
                "reviewed_at": row["reviewed_at"],
                "snapshot_path_raw": raw_snapshot_path,
                "snapshot_path": snapshot_path,
                "snapshot_exists": bool(snapshot_path and snapshot_path.exists()),
            }
        )
        if limit is not None and len(artifacts) >= int(limit):
            break

    return artifacts
