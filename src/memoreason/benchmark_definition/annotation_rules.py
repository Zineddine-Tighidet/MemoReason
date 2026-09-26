"""Normalize and partition the explicit rules attached to MemoReason documents."""

from __future__ import annotations

import re
from typing import Any

from .document_schema import AnnotatedDocument
from .implicit_numeric_rules import ensure_document_implicit_rules
from .annotation_parsing import AnnotationParser
from .annotation_references import (
    _infer_document_organization_id_map,
    _normalize_annotation_surface_key,
    find_entity_refs,
    normalize_entity_ref,
    normalize_text_entity_refs,
)
from .entity_taxonomy import parse_entity_id
from .organization_types import ORG_ENTITY_TYPES
from .rule_engine import RuleEngine


def _infer_document_organization_id_map_from_surface_matches(
    document_text: str,
    supplemental_texts: list[str],
) -> dict[str, str]:
    """Infer org id remaps by matching annotation surface text against the document body."""
    if not isinstance(document_text, str) or not document_text:
        return {}

    document_surface_to_org_ids: dict[str, set[str]] = {}
    document_org_ids: set[str] = set()
    for annotation in AnnotationParser.parse_annotations(document_text):
        normalized_entity_id = normalize_entity_ref(annotation.entity_id)
        entity_type, _ = parse_entity_id(normalized_entity_id)
        if entity_type not in ORG_ENTITY_TYPES:
            continue
        surface_key = _normalize_annotation_surface_key(annotation.original_text)
        if not surface_key:
            continue
        document_org_ids.add(normalized_entity_id)
        document_surface_to_org_ids.setdefault(surface_key, set()).add(normalized_entity_id)

    if not document_surface_to_org_ids:
        return {}

    remap: dict[str, str] = {}
    for text in supplemental_texts:
        if not isinstance(text, str) or not text:
            continue
        for annotation in AnnotationParser.parse_annotations(text):
            normalized_entity_id = normalize_entity_ref(annotation.entity_id)
            entity_type, _ = parse_entity_id(normalized_entity_id)
            if entity_type not in ORG_ENTITY_TYPES:
                continue
            if normalized_entity_id in document_org_ids:
                continue

            surface_key = _normalize_annotation_surface_key(annotation.original_text)
            if not surface_key:
                continue

            target_candidates = document_surface_to_org_ids.get(surface_key)
            if not target_candidates or len(target_candidates) != 1:
                continue
            target_entity_id = next(iter(target_candidates))

            previous_target = remap.get(normalized_entity_id)
            if previous_target and previous_target != target_entity_id:
                # Ambiguous remap request for the same stale id; skip to stay conservative.
                continue
            remap[normalized_entity_id] = target_entity_id
    return remap


RULE_TEXT_KEYS: tuple[str, ...] = ("rule", "expression", "constraint", "text")
RULE_COMMENT_KEYS: tuple[str, ...] = ("comment", "rationale", "reason", "note", "explanation", "why")
RULE_INLINE_COMMENT_SEPARATORS: tuple[str, ...] = (" // ", " # ", " -- ")
RULE_STORAGE_COMMENT_SEPARATOR = " # "


def split_rule_text_and_comment(rule_text: str) -> tuple[str, str]:
    """Split one stored rule string into ``(expression, comment)``.

    Accepted inline separators are `` // ``, `` # ``, and `` -- ``.
    """
    cleaned = str(rule_text or "").strip()
    if not cleaned:
        return "", ""
    for separator in RULE_INLINE_COMMENT_SEPARATORS:
        if separator not in cleaned:
            continue
        expression, comment = cleaned.split(separator, 1)
        expression = expression.strip()
        comment = comment.strip()
        if expression:
            return expression, comment
    return cleaned, ""


def compose_rule_text(expression: str, comment: str = "") -> str:
    """Compose one stored rule string from expression and optional comment."""
    rule_expression = str(expression or "").strip()
    if not rule_expression:
        return ""
    rule_comment = str(comment or "").strip()
    if not rule_comment:
        return rule_expression
    return f"{rule_expression}{RULE_STORAGE_COMMENT_SEPARATOR}{rule_comment}"


def _normalize_raw_rule_entry(raw_entry: Any) -> tuple[str, str]:
    if isinstance(raw_entry, str):
        return split_rule_text_and_comment(raw_entry)
    if isinstance(raw_entry, dict):
        expression_value = ""
        for key in RULE_TEXT_KEYS:
            value = raw_entry.get(key)
            if isinstance(value, str) and value.strip():
                expression_value = value
                break
        if not expression_value:
            return "", ""
        expression, inline_comment = split_rule_text_and_comment(expression_value)

        comment_value = ""
        for key in RULE_COMMENT_KEYS:
            value = raw_entry.get(key)
            if isinstance(value, str) and value.strip():
                comment_value = value.strip()
                break
        return expression, comment_value or inline_comment
    return "", ""


def normalize_rules_for_storage(raw_rules: Any) -> list[str] | None:
    """Normalize raw rules to the stored list[str] format with optional inline comments."""
    if raw_rules is None:
        return None
    if isinstance(raw_rules, (str, dict)):
        expression, comment = _normalize_raw_rule_entry(raw_rules)
        rendered = compose_rule_text(expression, comment)
        return [rendered] if rendered else []
    if isinstance(raw_rules, list):
        normalized: list[str] = []
        for raw_entry in raw_rules:
            expression, comment = _normalize_raw_rule_entry(raw_entry)
            rendered = compose_rule_text(expression, comment)
            if rendered:
                normalized.append(rendered)
        return normalized
    return None


def normalize_rule_expressions(raw_rules: Any) -> list[str] | None:
    """Normalize raw rules to expression-only strings for runtime rule evaluation."""
    normalized_storage = normalize_rules_for_storage(raw_rules)
    if normalized_storage is None:
        return None
    expressions: list[str] = []
    for stored_rule in normalized_storage:
        expression, _ = split_rule_text_and_comment(stored_rule)
        if expression:
            expressions.append(expression)
    return expressions


def normalize_document_taxonomy(doc_data: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw document payload to the current entity taxonomy."""
    if not isinstance(doc_data, dict):
        return doc_data

    cleaned = dict(doc_data)
    raw_questions = cleaned.get("questions", []) or []
    raw_rules = cleaned.get("rules", []) or []
    normalized_stored_rules = normalize_rules_for_storage(raw_rules) or []
    normalized_rule_expressions = normalize_rule_expressions(raw_rules) or []

    answer_texts: list[str] = []
    for question in raw_questions:
        if not isinstance(question, dict):
            continue
        raw_reasoning_chain = question.get("reasoning_chain", [])
        if isinstance(raw_reasoning_chain, list):
            answer_texts.extend(str(item) for item in raw_reasoning_chain if isinstance(item, str))
        raw_answer = question.get("answer")
        if isinstance(raw_answer, str):
            answer_texts.append(raw_answer)
        elif isinstance(raw_answer, list):
            answer_texts.extend(str(item) for item in raw_answer if isinstance(item, str))

    document_text = str(cleaned.get("document_to_annotate") or "")
    supplemental_texts = [
        *normalized_rule_expressions,
        *(str(question.get("question") or "") for question in raw_questions if isinstance(question, dict)),
        *answer_texts,
    ]
    texts_for_mapping = [document_text, *supplemental_texts]
    entity_id_remap = _infer_document_organization_id_map(texts_for_mapping)
    entity_id_remap.update(
        _infer_document_organization_id_map_from_surface_matches(
            document_text,
            supplemental_texts,
        )
    )

    cleaned["document_to_annotate"] = normalize_text_entity_refs(
        document_text,
        entity_id_remap=entity_id_remap,
    )
    normalized_rules_with_comments: list[str] = []
    for stored_rule in normalized_stored_rules:
        expression, comment = split_rule_text_and_comment(stored_rule)
        if not expression:
            continue
        normalized_expression = normalize_text_entity_refs(expression, entity_id_remap=entity_id_remap)
        rendered_rule = compose_rule_text(normalized_expression, comment)
        if rendered_rule:
            normalized_rules_with_comments.append(rendered_rule)
    cleaned["rules"] = normalized_rules_with_comments

    normalized_questions: list[dict[str, Any]] = []
    for raw_question in raw_questions:
        if not isinstance(raw_question, dict):
            continue
        question = dict(raw_question)
        question["question"] = normalize_text_entity_refs(
            str(question.get("question") or ""),
            entity_id_remap=entity_id_remap,
        )
        raw_reasoning_chain = question.get("reasoning_chain")
        if isinstance(raw_reasoning_chain, list):
            question["reasoning_chain"] = [
                normalize_text_entity_refs(str(item), entity_id_remap=entity_id_remap)
                if isinstance(item, str)
                else item
                for item in raw_reasoning_chain
            ]

        raw_answer = question.get("answer")
        if isinstance(raw_answer, str):
            question["answer"] = normalize_text_entity_refs(raw_answer, entity_id_remap=entity_id_remap)
        elif isinstance(raw_answer, list):
            question["answer"] = [
                normalize_text_entity_refs(str(item), entity_id_remap=entity_id_remap)
                if isinstance(item, str)
                else item
                for item in raw_answer
            ]

        normalized_questions.append(question)

    cleaned["questions"] = normalized_questions
    return ensure_document_implicit_rules(cleaned)


_PLACE_HIERARCHY_ATTRS = {"city", "region", "state", "country", "continent"}
_PLACE_HIERARCHY_EQUALITY_PATTERN = re.compile(
    r"^\s*(place_\d+\.(?:city|region|state|country|continent))\s*(==|=)\s*"
    r"(place_\d+\.(?:city|region|state|country|continent))\s*$"
)


def find_rule_sanity_errors(rules: list[str]) -> list[str]:
    """Return static rule errors that are invalid regardless of sampled values."""
    errors: list[str] = []
    for index, raw_rule in enumerate(rules or [], start=1):
        rule_text = str(raw_rule or "")
        cleaned = rule_text.split("#", 1)[0].strip()
        if not cleaned:
            continue
        match = _PLACE_HIERARCHY_EQUALITY_PATTERN.fullmatch(cleaned)
        if not match:
            continue
        left_ref, _, right_ref = match.groups()
        left_attr = left_ref.split(".", 1)[1]
        right_attr = right_ref.split(".", 1)[1]
        if left_attr == right_attr:
            continue
        if left_attr in _PLACE_HIERARCHY_ATTRS and right_attr in _PLACE_HIERARCHY_ATTRS:
            errors.append(
                "Rule "
                f"{index} compares incompatible place levels with equality: `{cleaned}`. "
                "Do not equate place `.city`, `.region`, `.state`, `.country`, or `.continent` "
                "to a different place level in a single equality rule."
            )
    return errors


def partition_generation_rules(
    doc: AnnotatedDocument,
    *,
    include_questions: bool = True,
) -> tuple[list[str], list[str]]:
    """Split document rules into factual-valid and factual-invalid subsets.

    Reviewed rules occasionally contain annotation mistakes that do not hold on
    the factual source document itself. Those rules are unsafe to enforce during
    fictional generation because they can make the sampler chase an impossible
    constraint set. We keep only the rules that validate on the factual source
    entities and surface the rest to callers for logging or QA.
    """
    rules = [str(rule) for rule in (doc.rules or []) if str(rule).split("#", 1)[0].strip()]
    if not rules:
        return [], []

    factual_entities = AnnotationParser.extract_factual_entities(doc, include_questions=include_questions)
    kept_rules: list[str] = []
    dropped_rules: list[str] = []
    factual_validity_inputs: list[str] = []
    for rule_text in rules:
        cleaned_rule = str(rule_text or "").split("#", 1)[0].strip()
        refs = set(find_entity_refs(cleaned_rule))
        if refs and all(not ref.startswith(("number_", "temporal_")) for ref in refs):
            kept_rules.append(rule_text)
            continue
        factual_validity_inputs.append(rule_text)

    results = RuleEngine.validate_all_rules(factual_validity_inputs, factual_entities)
    for rule_text, is_valid in results:
        if is_valid:
            kept_rules.append(rule_text)
        else:
            dropped_rules.append(rule_text)
    return kept_rules, dropped_rules
