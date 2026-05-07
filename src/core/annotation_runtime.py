"""
Shared utilities: run dirs, document/pool loaders, annotation parsing, expression rules.

Public API:
  - create_run_dir: create traceable run directory with run_params.json
  - load_annotated_document, load_entity_pool: load YAML docs and entity pools
  - Annotation, AnnotationParser: parse [text; entity_id.attr] annotations
  - RuleEngine: evaluate entity expressions (e.g. person_1.age)
  - find_entity_refs, is_valid_entity_ref: entity reference helpers
  - map_relationship_for_gender, get_appropriate_relationship: relationship wording
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from functools import lru_cache
from typing import Any

import yaml

from .century_expressions import century_end, century_of, century_start
from .implicit_numeric_rules import (
    ensure_document_implicit_rules,
    normalize_implicit_rule_exclusions,
    normalize_implicit_rules_for_storage,
)
from .document_schema import (
    AnnotatedDocument,
    AwardEntity,
    EntityCollection,
    EventEntity,
    FictionalDocument,
    LegalEntity,
    NumberEntity,
    OrganizationEntity,
    PersonEntity,
    PlaceEntity,
    ProductEntity,
    Question,
    TemporalEntity,
    ImplicitRule,
)
from .organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND,
    LEGACY_ORGANIZATION_TYPE_ALIASES,
    ORG_ENTITY_TYPES,
    ORGANIZATION_POOL_BUCKETS,
    canonicalize_organization_type,
    get_organization_name,
    infer_organization_kind,
    normalize_organization_pool_entry,
    organization_attribute_value,
    organization_pool_bucket,
)
from .entity_taxonomy import (
    _WEEKDAY_ALIASES,
    _WEEKDAY_TO_INDEX,
    ENTITY_TAXONOMY,
    FULL_REPLACE_ENTITY_TYPES,
    LEGACY_ENTITY_ATTRIBUTES,
    ORDINAL_WORD_TO_NUMBER,
    PARTIAL_REPLACE_ATTRIBUTES,
    REPLACE_MODE_ALL,
    REPLACE_MODE_NON_NUMERICAL,
    REPLACE_MODE_NUMERICAL,
    REPLACE_MODE_TEMPORAL,
    VALID_ENTITY_TYPES,
    VALID_REPLACE_MODES,
    WEEKDAYS,
    WORD_TO_NUMBER,
    infer_int_surface_format,
    infer_str_surface_format,
    parse_entity_id,
    parse_integer_surface_number,
    parse_word_number,
    replace_mode_label,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc


class AnnotationValidationError(Exception):
    """Raised when an annotation violates the entity taxonomy."""

    pass


def _split_entity_ref(entity_ref: str) -> tuple[str, str | None]:
    cleaned = str(entity_ref or "").strip()
    if "." in cleaned:
        return cleaned.split(".", 1)[0], cleaned.split(".", 1)[1]
    return cleaned, None


def _canonical_organization_attribute(attribute: str | None) -> str | None:
    if attribute is None:
        return None
    cleaned = str(attribute).strip()
    if not cleaned:
        return None
    if cleaned == "name":
        return "name"
    if cleaned in {"organization_kind", "kind"}:
        return cleaned
    if cleaned in LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND:
        return "name"
    if cleaned in LEGACY_ORGANIZATION_TYPE_ALIASES:
        return "name"
    if cleaned in CANONICAL_ORGANIZATION_TYPES:
        return "name"
    return cleaned


def _prefer_more_specific_organization_kind(current_kind: str | None, candidate_kind: str | None) -> str | None:
    if candidate_kind is None:
        return current_kind
    if current_kind in {None, "organization"}:
        return candidate_kind
    return current_kind


def _infer_document_organization_id_map(texts: list[str]) -> dict[str, str]:
    inferred_kinds: dict[str, str | None] = {}
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        for raw_ref in find_entity_refs(text):
            entity_id, attribute = _split_entity_ref(raw_ref)
            entity_type, entity_index = parse_entity_id(entity_id)
            if entity_type is None or entity_index is None:
                continue

            normalized_entity_id = entity_id.replace("organisation_", "organization_", 1)
            entity_type, entity_index = parse_entity_id(normalized_entity_id)
            if entity_type is None or entity_index is None:
                continue

            candidate_kind = infer_organization_kind(entity_type=entity_type, attribute=attribute)
            if candidate_kind is None:
                continue

            inferred_kinds[normalized_entity_id] = _prefer_more_specific_organization_kind(
                inferred_kinds.get(normalized_entity_id),
                candidate_kind,
            )

    remap: dict[str, str] = {}
    for original_entity_id, candidate_kind in inferred_kinds.items():
        entity_type, entity_index = parse_entity_id(original_entity_id)
        if entity_type is None or entity_index is None:
            continue
        canonical_type = candidate_kind or canonicalize_organization_type(entity_type) or entity_type
        canonical_entity_id = f"{canonical_type}_{entity_index}"
        remap[original_entity_id] = canonical_entity_id
    return remap


def _normalize_annotation_surface_key(raw_text: str) -> str:
    return re.sub(r"\s+", " ", str(raw_text or "").strip()).casefold()


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


def normalize_text_entity_refs(text: str, *, entity_id_remap: dict[str, str] | None = None) -> str:
    """Rewrite legacy entity references in one text field to the current taxonomy."""
    if not isinstance(text, str) or not text:
        return text
    remap = entity_id_remap or {}

    def _replace(match: re.Match[str]) -> str:
        return normalize_entity_ref(match.group(0), entity_id_remap=remap)

    return ENTITY_REF_PATTERN.sub(_replace, text)


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


_TEMPORAL_DECADE_SURFACE_PATTERN = re.compile(r"^\s*\d{4}s\s*$", re.IGNORECASE)
_TEMPORAL_CENTURY_SURFACE_PATTERN = re.compile(r"\bcentur(?:y|ies)\b", re.IGNORECASE)


def _validate_temporal_year_annotation_surface(
    *,
    original_text: str,
    source_label: str,
    entity_ref: str,
) -> None:
    text = str(original_text or "").strip()
    if not text:
        return
    if _TEMPORAL_DECADE_SURFACE_PATTERN.fullmatch(text):
        raise AnnotationValidationError(
            f"[{source_label}] Invalid temporal year surface '{text}' in annotation [{original_text}; {entity_ref}].\n"
            "Use the year annotation without trailing 's' (e.g., '[1960; temporal_1.year]s')."
        )
    if _TEMPORAL_CENTURY_SURFACE_PATTERN.search(text):
        raise AnnotationValidationError(
            f"[{source_label}] Invalid temporal year surface '{text}' in annotation [{original_text}; {entity_ref}].\n"
            "Century mentions must be annotated as numbers (e.g., '[17th; number_1.int] century')."
        )


def _collect_referenced_entity_refs(text: str) -> set[str]:
    referenced_entity_refs: set[str] = set()
    for raw_ref in find_entity_refs(text):
        normalized_ref = normalize_entity_ref(raw_ref).strip()
        if normalized_ref:
            referenced_entity_refs.add(normalized_ref)
    return referenced_entity_refs


def _entity_id_from_entity_ref(entity_ref: str) -> str:
    entity_id, _ = _split_entity_ref(normalize_entity_ref(entity_ref).strip())
    return entity_id.strip()


def validate_question_and_answer_entity_scope(
    document_text: str,
    questions: list[dict[str, Any]],
    *,
    source_label: str = "document",
) -> None:
    """Require question/answer/reasoning-chain refs to reuse entities annotated in the document body.

    Matching is entity-level (`entity_type_N`) rather than attribute-level so that
    references such as `number_3`, `number_3.int`, and `number_3.str` are treated
    as in-scope as long as `number_3` appears in `document_to_annotate`.
    """
    allowed_entity_refs = _collect_referenced_entity_refs(document_text or "")
    allowed_entity_ids = {
        _entity_id_from_entity_ref(ref)
        for ref in allowed_entity_refs
    }

    out_of_scope_references: list[str] = []
    for question_data in questions or []:
        if not isinstance(question_data, dict):
            continue
        question_id = str(question_data.get("question_id", "?"))

        question_text = str(question_data.get("question", "") or "")
        for entity_ref in sorted(_collect_referenced_entity_refs(question_text)):
            if (
                entity_ref not in allowed_entity_refs
                and _entity_id_from_entity_ref(entity_ref) not in allowed_entity_ids
            ):
                out_of_scope_references.append(f"{question_id}/question -> {entity_ref}")

        raw_reasoning_chain = question_data.get("reasoning_chain", [])
        if isinstance(raw_reasoning_chain, list):
            for step_index, reasoning_step in enumerate(raw_reasoning_chain, start=1):
                for entity_ref in sorted(_collect_referenced_entity_refs(str(reasoning_step or ""))):
                    if (
                        entity_ref not in allowed_entity_refs
                        and _entity_id_from_entity_ref(entity_ref) not in allowed_entity_ids
                    ):
                        out_of_scope_references.append(
                            f"{question_id}/reasoning_chain[{step_index}] -> {entity_ref}"
                        )

        raw_answer = question_data.get("answer", "")
        answer_texts: list[str] = []
        if isinstance(raw_answer, str):
            answer_texts.append(raw_answer)
        elif isinstance(raw_answer, list):
            answer_texts.extend(str(item) for item in raw_answer if isinstance(item, str))

        for answer_text in answer_texts:
            for entity_ref in sorted(_collect_referenced_entity_refs(answer_text)):
                if (
                    entity_ref not in allowed_entity_refs
                    and _entity_id_from_entity_ref(entity_ref) not in allowed_entity_ids
                ):
                    out_of_scope_references.append(f"{question_id}/answer -> {entity_ref}")

    if out_of_scope_references:
        rendered = ", ".join(sorted(set(out_of_scope_references)))
        raise AnnotationValidationError(
            f"[{source_label}] Questions/answers reference annotations not present in document_to_annotate: {rendered}."
        )


def validate_annotations(text: str, source_label: str = "document") -> None:
    """Validate all [text; entity_id.attribute] annotations against the taxonomy.

    Raises ``AnnotationValidationError`` if any annotation uses an entity type
    or attribute that is not part of the defined taxonomy.

    Args:
        text: The annotated text containing ``[text; entity_ref]`` patterns.
        source_label: Label for error messages (e.g. file name or "question").
    """
    if not text:
        return

    import re as _re

    pattern = r"\[([^\]]+);\s*([^\]]+)\]"

    for match in _re.finditer(pattern, text):
        original_text = match.group(1).strip()
        raw_entity_ref = match.group(2).strip()
        raw_entity_id, raw_attribute = _split_entity_ref(raw_entity_ref)
        if raw_entity_id.startswith("organisation_"):
            raise AnnotationValidationError(
                f"[{source_label}] Legacy entity ID spelling '{raw_entity_id}' "
                f"is not accepted in annotation [{original_text}; {raw_entity_ref}].\n"
                "Use 'organization_' spelling."
            )
        raw_entity_type, _ = parse_entity_id(raw_entity_id.replace("organisation_", "organization_", 1))
        if raw_entity_type == "organization":
            raise AnnotationValidationError(
                f"[{source_label}] Generic organization entity type '{raw_entity_type}' "
                f"is not accepted in annotation [{original_text}; {raw_entity_ref}].\n"
                "Use explicit organization entity types only: military_org, entreprise_org, ngo, "
                "government_org, educational_org, or media_org."
            )
        if raw_entity_type in LEGACY_ORGANIZATION_TYPE_ALIASES:
            raise AnnotationValidationError(
                f"[{source_label}] Legacy organization entity type '{raw_entity_type}' "
                f"is not accepted in annotation [{original_text}; {raw_entity_ref}].\n"
                "Use canonical entity types only: military_org, entreprise_org, ngo, "
                "government_org, educational_org, or media_org."
            )
        if raw_attribute in LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND:
            raise AnnotationValidationError(
                f"[{source_label}] Legacy organization attribute '{raw_attribute}' "
                f"is not accepted in annotation [{original_text}; {raw_entity_ref}].\n"
                "Use explicit organization entity types instead "
                "(e.g., government_org_1.name, media_org_2.name)."
            )

        entity_ref = normalize_entity_ref(raw_entity_ref)

        # Parse entity_id and attribute
        parts = entity_ref.split(".", 1)
        entity_id = parts[0].strip()
        attribute = parts[1].strip() if len(parts) > 1 else None

        # Extract entity type from entity_id (e.g. "person" from "person_1")
        entity_type, _ = parse_entity_id(entity_id)
        if not entity_type:
            raise AnnotationValidationError(
                f"[{source_label}] Invalid entity ID format: '{entity_id}' "
                f"in annotation [{original_text}; {entity_ref}].\n"
                f"Expected format: type_N (e.g. person_1, place_2).\n"
                f"Valid entity types: {sorted(ENTITY_TAXONOMY.keys())}"
            )
        # Validate entity type
        if entity_type not in ENTITY_TAXONOMY:
            raise AnnotationValidationError(
                f"[{source_label}] Invalid entity type '{entity_type}' "
                f"in annotation [{original_text}; {entity_ref}].\n"
                f"Valid entity types: {sorted(ENTITY_TAXONOMY.keys())}"
            )

        # Validate attribute (if provided)
        if attribute:
            valid_attrs = ENTITY_TAXONOMY[entity_type]
            # relationship.person_Y is a special pattern for person entities
            if attribute.startswith("relationship."):
                if entity_type != "person":
                    raise AnnotationValidationError(
                        f"[{source_label}] Invalid attribute 'relationship' "
                        f"on entity type '{entity_type}' "
                        f"in annotation [{original_text}; {entity_ref}].\n"
                        f"'relationship.person_Y' is only valid for person entities."
                    )
            elif attribute not in valid_attrs:
                legacy_attrs = LEGACY_ENTITY_ATTRIBUTES.get(entity_type, frozenset())
                if attribute in legacy_attrs:
                    continue
                raise AnnotationValidationError(
                    f"[{source_label}] Invalid attribute '{attribute}' "
                    f"for entity type '{entity_type}' "
                    f"in annotation [{original_text}; {entity_ref}].\n"
                    f"Valid attributes for '{entity_type}': {sorted(valid_attrs)}"
                )
            if entity_type == "temporal" and attribute == "year":
                _validate_temporal_year_annotation_surface(
                    original_text=original_text,
                    source_label=source_label,
                    entity_ref=entity_ref,
                )


def _sanitize(s: str) -> str:
    """Replace characters unsafe for directory names."""
    return s.replace("/", "-").replace(" ", "_")[:64]


def create_run_dir(
    provider: str,
    base_dir: Path | None = None,
    *,
    seed: int | None = None,
    models: list[str] | None = None,
    documents: list[str] | None = None,
    proportions: list[float] | None = None,
    num_versions: int | None = None,
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    **kwargs: Any,
) -> Path:
    """Create a traceable run directory and write run_params.json.

    Directory name format: {timestamp}_{provider}_seed_{seed}_models_{...}_docs_{...}
    This ensures each run is uniquely identifiable by when it ran, which
    provider served the models, and all relevant hyperparameters.
    """
    if base_dir is None:
        base_dir = Path("results") / _sanitize(provider)
    base_dir = Path(base_dir)
    now = datetime.now(UTC)
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    parts = [timestamp, _sanitize(provider)]
    if seed is not None:
        parts.append(f"seed_{seed}")
    if documents:
        parts.append("docs_" + "_".join(_sanitize(d) for d in documents[:5]))
    if models:
        parts.append("models_" + "_".join(_sanitize(m) for m in models[:3]))
    dir_name = "_".join(parts)
    run_dir = base_dir / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    params = {"provider": provider, "datetime_iso": now.isoformat(), "timestamp": timestamp}
    if seed is not None:
        params["seed"] = seed
    if models is not None:
        params["models"] = models
    if documents is not None:
        params["documents"] = documents
    if proportions is not None:
        params["proportions"] = proportions
    if num_versions is not None:
        params["num_versions"] = num_versions
    if skip_generation:
        params["skip_generation"] = skip_generation
    if skip_evaluation:
        params["skip_evaluation"] = skip_evaluation
    params.update(kwargs)
    with open(run_dir / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    return run_dir


def load_annotated_document(yaml_path: str, *, validate_question_scope: bool = True) -> AnnotatedDocument:
    """
    Load an annotated document from a YAML file.

    Expected YAML structure:
    document:
      document_id: ...
      document_theme: ...
      document_to_annotate: ...
      rules: [...]
      questions: [...]
      evaluated_answers: [...]  # Optional
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "document" not in data:
        raise ValueError(f"YAML file {yaml_path} does not contain 'document' key")

    doc_data = data["document"]
    if not isinstance(doc_data, dict):
        raise ValueError(f"YAML file {yaml_path} does not contain a document mapping")

    # Validate raw annotations before any normalization so legacy/invalid refs fail loudly.
    doc_id = doc_data.get("document_id", yaml_path)
    validate_annotations(
        doc_data.get("document_to_annotate", ""),
        source_label=f"{doc_id}/document_to_annotate",
    )
    validate_annotations(
        doc_data.get("fictionalized_annotated_template_document", ""),
        source_label=f"{doc_id}/fictionalized_annotated_template_document",
    )
    for q_data in doc_data.get("questions", []):
        qid = q_data.get("question_id", "?")
        q_label = f"{doc_id}/question/{qid}"
        validate_annotations(
            q_data.get("question", ""),
            source_label=f"{q_label}/question",
        )
        for step_index, reasoning_step in enumerate(q_data.get("reasoning_chain", []) or [], start=1):
            validate_annotations(
                str(reasoning_step),
                source_label=f"{q_label}/reasoning_chain[{step_index}]",
            )
        raw_answer = q_data.get("answer", "")
        if isinstance(raw_answer, list) and len(raw_answer) == 1:
            answer_text = str(raw_answer[0])
        elif isinstance(raw_answer, bool):
            answer_text = ""  # Yes/No answers have no entity refs
        else:
            answer_text = str(raw_answer)
        validate_annotations(
            answer_text,
            source_label=f"{q_label}/answer",
        )

    if validate_question_scope:
        validate_question_and_answer_entity_scope(
            doc_data.get("document_to_annotate", ""),
            doc_data.get("questions", []),
            source_label=str(doc_id),
        )

    doc_data = normalize_document_taxonomy(doc_data)

    # Parse questions
    questions = []
    for q_data in doc_data.get("questions", []):
        raw_answer = q_data.get("answer", "")
        if raw_answer is True:
            answer_str = "Yes"
        elif raw_answer is False:
            answer_str = "No"
        elif isinstance(raw_answer, list) and len(raw_answer) == 1 and isinstance(raw_answer[0], str):
            # Preserve inner string for list-wrapped composite answers (e.g. ['Boston Marathon; "place_1.city event_1.type"'])
            answer_str = raw_answer[0]
        else:
            answer_str = str(raw_answer)
        raw_answer_type = q_data.get("answer_type")
        if raw_answer_type is None:
            invariant_flag = q_data.get("is_answer_invariant")
            if invariant_flag is True:
                raw_answer_type = "invariant"
            elif invariant_flag is False:
                raw_answer_type = "variant"
        raw_answer_overrides = q_data.get("accepted_answer_overrides") or []
        if isinstance(raw_answer_overrides, str):
            accepted_answer_overrides = [raw_answer_overrides.strip()] if raw_answer_overrides.strip() else []
        else:
            accepted_answer_overrides = [
                str(item).strip()
                for item in raw_answer_overrides
                if str(item).strip()
            ]
        question = Question(
            question_id=q_data["question_id"],
            question=q_data["question"],
            answer=answer_str,
            question_type=q_data.get("question_type"),
            answer_type=raw_answer_type,
            reasoning_chain=[
                str(step).strip()
                for step in (q_data.get("reasoning_chain") or [])
                if str(step).strip()
            ],
            accepted_answer_overrides=accepted_answer_overrides,
        )
        questions.append(question)

    # Parse evaluated_answers if present
    evaluated_answers = None
    if "evaluated_answers" in doc_data:
        evaluated_answers = doc_data["evaluated_answers"]
    runtime_rule_expressions = normalize_rule_expressions(doc_data.get("rules", [])) or []
    implicit_rules_data = normalize_implicit_rules_for_storage(doc_data.get("implicit_rules", [])) or []

    # Create document
    doc = AnnotatedDocument(
        document_id=doc_data["document_id"],
        document_theme=doc_data.get("document_theme", ""),
        original_document=doc_data.get("original_document", ""),
        document_to_annotate=doc_data.get("document_to_annotate", ""),
        fictionalized_annotated_template_document=doc_data.get(
            "fictionalized_annotated_template_document", ""
        ),
        questions=questions,
        rules=runtime_rule_expressions,
        implicit_rules=[ImplicitRule(**rule_data) for rule_data in implicit_rules_data],
        implicit_rule_exclusions=normalize_implicit_rule_exclusions(
            doc_data.get("implicit_rule_exclusions")
        ),
        evaluated_answers=evaluated_answers,
    )

    return doc


def load_entity_pool(yaml_path: str) -> dict[str, Any]:
    """
    Load an entity pool from a YAML file.

    Expected YAML structure:
    persons: [...]
    places: [...]
    events: [...]
    organizations: [...]                 # generic organization_X only
    military_orgs: [...]
    entreprise_orgs: [...]
    ngos: [...]
    government_orgs: [...]
    educational_orgs: [...]
    media_orgs: [...]
    awards: [...]
    legals: [...]
    products: [...]
    numbers: [...]
    temporals: [...]
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Entity pool file {yaml_path} does not contain a mapping at the top level.")
    bucket_names = (
        "persons",
        "places",
        "events",
        "organizations",
        "military_orgs",
        "entreprise_orgs",
        "ngos",
        "government_orgs",
        "educational_orgs",
        "media_orgs",
        "awards",
        "legals",
        "products",
    )
    normalized_pool: dict[str, Any] = {
        **{bucket: [] for bucket in bucket_names},
        "_reference_pools": {bucket: {} for bucket in bucket_names},
        "_coverage": {bucket: {} for bucket in bucket_names},
    }
    pool_variant_metadata_keys = {"old_name"}

    def _dedupe_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for entry in entries:
            entry_key = tuple(sorted(entry.items()))
            if entry_key in seen:
                continue
            seen.add(entry_key)
            deduped.append(entry)
        return deduped

    def _normalize_person_entries(
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        allowed = set(required_attrs or [])
        for raw_person in raw_entries:
            if not isinstance(raw_person, dict):
                continue
            person = {key: str(value).strip() for key, value in raw_person.items() if str(value).strip()}
            provided_keys = set(person.keys())
            full_name = person.get("full_name", "")
            first_name = person.get("first_name", "")
            last_name = person.get("last_name", "")
            if not full_name and first_name and last_name:
                full_name = f"{first_name} {last_name}"
            if full_name and (not first_name or not last_name):
                parts = full_name.split()
                if len(parts) >= 2:
                    first_name = first_name or parts[0]
                    last_name = last_name or parts[-1]
            cleaned: dict[str, str] = {}
            def _keep(attr: str, value: str) -> None:
                if not value:
                    return
                if allowed:
                    if attr in allowed:
                        cleaned[attr] = value
                    return
                if attr in provided_keys:
                    cleaned[attr] = value

            _keep("full_name", full_name)
            if not allowed and "full_name" not in provided_keys and full_name and {"first_name", "last_name"} <= provided_keys:
                cleaned["full_name"] = full_name
            _keep("first_name", first_name)
            _keep("last_name", last_name)
            _keep("middle_name", person.get("middle_name", ""))
            _keep("nationality", person.get("nationality", ""))
            _keep("ethnicity", person.get("ethnicity", ""))
            if cleaned:
                normalized.append(cleaned)
        return _dedupe_entries(normalized)

    def _normalize_simple_entries(
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
        require_name: bool = False,
        keep_type: bool = False,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        allowed = set(required_attrs or [])
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            cleaned = {key: str(value).strip() for key, value in raw_entry.items() if str(value).strip()}
            if not cleaned:
                continue
            if "nationality" in cleaned and "demonym" not in cleaned:
                cleaned["demonym"] = cleaned["nationality"]
            if allowed:
                cleaned = {
                    key: value
                    for key, value in cleaned.items()
                    if key in allowed or key in pool_variant_metadata_keys or (keep_type and key == "type")
                }
            if require_name and not cleaned.get("name"):
                continue
            normalized.append(cleaned)
        return _dedupe_entries(normalized)

    def _normalize_legal_entries(
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
    ) -> list[dict[str, str]]:
        normalized = _normalize_simple_entries(raw_entries, required_attrs=required_attrs)
        return [entry for entry in normalized if entry.get("name") or entry.get("reference_code")]

    def _normalize_organization_entries(raw_entries: list[Any], *, bucket_name: str) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        expected_type = None
        if bucket_name != "organizations":
            expected_type = canonicalize_organization_type(bucket_name[:-1] if bucket_name.endswith("s") else bucket_name)
        if bucket_name == "ngos":
            expected_type = "ngo"
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            try:
                normalized_entry = normalize_organization_pool_entry(raw_entry, expected_entity_type=expected_type)
            except ValueError:
                continue
            normalized.append({"name": normalized_entry["name"]})
        return _dedupe_entries(normalized)

    def _normalize_bucket_entries(
        bucket_name: str,
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
    ) -> list[dict[str, str]]:
        if bucket_name == "persons":
            return _normalize_person_entries(raw_entries, required_attrs=required_attrs)
        if bucket_name == "places":
            return _normalize_simple_entries(raw_entries, required_attrs=required_attrs)
        if bucket_name == "events":
            require_event_name = "name" in set(required_attrs or [])
            return _normalize_simple_entries(
                raw_entries,
                required_attrs=required_attrs,
                require_name=require_event_name,
                keep_type=True,
            )
        if bucket_name in {"awards", "products"}:
            return _normalize_simple_entries(raw_entries, required_attrs=required_attrs, require_name=True)
        if bucket_name == "legals":
            return _normalize_legal_entries(raw_entries, required_attrs=required_attrs)
        if bucket_name in ORGANIZATION_POOL_BUCKETS:
            return _normalize_organization_entries(raw_entries, bucket_name=bucket_name)
        return []

    def _flatten_reference_bucket(bucket_name: str) -> list[dict[str, str]]:
        flattened: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        bucket_refs = normalized_pool["_reference_pools"].get(bucket_name, {})
        if not isinstance(bucket_refs, dict):
            return []
        for ref_payload in bucket_refs.values():
            if not isinstance(ref_payload, dict):
                continue
            for entry in ref_payload.get("variants", []) or []:
                if not isinstance(entry, dict):
                    continue
                entry_key = tuple(sorted((key, str(value).strip()) for key, value in entry.items() if str(value).strip()))
                if not entry_key or entry_key in seen:
                    continue
                seen.add(entry_key)
                flattened.append({key: str(value).strip() for key, value in entry.items() if str(value).strip()})
        return flattened

    seen_organization_entries: dict[str, set[tuple[tuple[str, str], ...]]] = {
        bucket_name: set() for bucket_name in ORGANIZATION_POOL_BUCKETS
    }
    for bucket_name in bucket_names:
        raw_bucket = data.get(bucket_name)
        if not isinstance(raw_bucket, dict):
            continue
        for entity_id, raw_value in raw_bucket.items():
            raw_variants: list[Any] = []
            required_attrs: list[str] = []
            if isinstance(raw_value, dict):
                raw_variants = list(raw_value.get("variants", []) or [])
                required_attrs = [str(attr).strip() for attr in raw_value.get("required_attributes", []) or [] if str(attr).strip()]
            elif isinstance(raw_value, list):
                raw_variants = list(raw_value)
            variants = _normalize_bucket_entries(bucket_name, raw_variants, required_attrs=required_attrs or None)
            normalized_pool["_reference_pools"][bucket_name][str(entity_id)] = {
                "required_attributes": required_attrs,
                "count": len(variants),
                "variants": variants,
            }
            normalized_pool["_coverage"][bucket_name][str(entity_id)] = len(variants)
        normalized_pool[bucket_name] = _flatten_reference_bucket(bucket_name)

    for bucket_name in ORGANIZATION_POOL_BUCKETS:
        if normalized_pool["_reference_pools"].get(bucket_name):
            continue
        expected_type = None
        if bucket_name != "organizations":
            expected_type = canonicalize_organization_type(bucket_name[:-1] if bucket_name.endswith("s") else bucket_name)
        if bucket_name == "ngos":
            expected_type = "ngo"
        for raw_entry in data.get(bucket_name, []) or []:
            if not isinstance(raw_entry, dict):
                continue
            try:
                normalized_entry = normalize_organization_pool_entry(raw_entry, expected_entity_type=expected_type)
            except ValueError:
                continue
            target_bucket = organization_pool_bucket(normalized_entry["organization_kind"]) or bucket_name
            entry_payload = {"name": normalized_entry["name"]}
            entry_key = tuple(sorted(entry_payload.items()))
            if entry_key in seen_organization_entries[target_bucket]:
                continue
            seen_organization_entries[target_bucket].add(entry_key)
            normalized_pool[target_bucket].append(entry_payload)

    if not normalized_pool["_reference_pools"]["places"]:
        normalized_pool["places"] = _normalize_bucket_entries("places", list(data.get("places", []) or []))
    if not normalized_pool["_reference_pools"]["awards"]:
        normalized_pool["awards"] = _normalize_bucket_entries("awards", list(data.get("awards", []) or []))
    if not normalized_pool["_reference_pools"]["legals"]:
        normalized_pool["legals"] = _normalize_bucket_entries("legals", list(data.get("legals", []) or []))
    if not normalized_pool["_reference_pools"]["products"]:
        normalized_pool["products"] = _normalize_bucket_entries("products", list(data.get("products", []) or []))
    if not normalized_pool["_reference_pools"]["persons"]:
        normalized_pool["persons"] = _normalize_bucket_entries("persons", list(data.get("persons", []) or []))
    if not normalized_pool["_reference_pools"]["events"]:
        normalized_pool["events"] = _normalize_bucket_entries("events", list(data.get("events", []) or []))
    return normalized_pool


def fictional_document_to_yaml_dict(
    generated_doc: FictionalDocument,
    question_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the dict used for YAML serialization of a generated fictional document.
    question_entries should come from AnswerEvaluator.build_question_entries_with_answers.
    """
    return {
        "document_id": generated_doc.document_id,
        "document_theme": generated_doc.document_theme,
        "generated_document": generated_doc.generated_document,
        "questions": question_entries,
        "entities_used": {
            "persons": {k: v.model_dump() for k, v in generated_doc.entities_used.persons.items()},
            "places": {k: v.model_dump() for k, v in generated_doc.entities_used.places.items()},
            "events": {k: v.model_dump() for k, v in generated_doc.entities_used.events.items()},
            "organizations": {k: v.model_dump() for k, v in generated_doc.entities_used.organizations.items()},
            "awards": {k: v.model_dump() for k, v in generated_doc.entities_used.awards.items()},
            "legals": {k: v.model_dump() for k, v in generated_doc.entities_used.legals.items()},
            "products": {k: v.model_dump() for k, v in generated_doc.entities_used.products.items()},
            "temporals": {k: v.model_dump() for k, v in generated_doc.entities_used.temporals.items()},
            "numbers": {k: v.model_dump() for k, v in generated_doc.entities_used.numbers.items()},
        },
    }


# --- Entity references and annotation parsing ---
_ENTITY_REF_TYPES = tuple(sorted(set(VALID_ENTITY_TYPES) | {"organisation"}, key=len, reverse=True))
_ENTITY_REF_TYPES_RE = "|".join(re.escape(t) for t in _ENTITY_REF_TYPES)
_PERSON_RELATIONSHIP_REF_RE = r"person_\d+\.relationship\.person_\d+"
_SIMPLE_ENTITY_REF_RE = rf"(?:{_ENTITY_REF_TYPES_RE})_\d+(?:\.[a-z_]+)?"
ENTITY_REF_PATTERN = re.compile(rf"\b(?:{_PERSON_RELATIONSHIP_REF_RE}|{_SIMPLE_ENTITY_REF_RE})\b")
ENTITY_REF_VALIDATION_PATTERN = re.compile(rf"^(?:{_PERSON_RELATIONSHIP_REF_RE}|{_SIMPLE_ENTITY_REF_RE})$")
PERSON_RELATIONSHIP_REF_PATTERN = re.compile(rf"^{_PERSON_RELATIONSHIP_REF_RE}$")


def normalize_entity_ref(ref: str, *, entity_id_remap: dict[str, str] | None = None) -> str:
    """Normalize one entity reference to the current taxonomy."""
    if not ref:
        return ref
    entity_id, attribute = _split_entity_ref(ref)
    normalized_entity_id = entity_id.replace("organisation_", "organization_", 1)
    entity_type, entity_index = parse_entity_id(normalized_entity_id)

    canonical_entity_id = normalized_entity_id
    if entity_type is not None and entity_index is not None:
        # Canonicalize zero-padded ids (e.g., number_09 -> number_9) so rule refs
        # resolve to annotation ids consistently across the pipeline.
        canonical_entity_id = f"{entity_type}_{entity_index}"
    canonical_attribute = attribute
    remap = entity_id_remap or {}
    if normalized_entity_id in remap:
        canonical_entity_id = remap[normalized_entity_id]
    elif canonical_entity_id in remap:
        canonical_entity_id = remap[canonical_entity_id]

    if entity_type is not None and entity_index is not None:
        canonical_organization_type = canonicalize_organization_type(entity_type)
        if canonical_organization_type is not None:
            canonical_organization_id = f"{canonical_organization_type}_{entity_index}"
            canonical_entity_id = remap.get(normalized_entity_id, remap.get(canonical_organization_id, canonical_organization_id))
            canonical_attribute = _canonical_organization_attribute(attribute)

    if canonical_attribute:
        return f"{canonical_entity_id}.{canonical_attribute}"
    return canonical_entity_id


def find_entity_refs(text: str) -> list[str]:
    """Return all entity references in text (e.g. number_3, place_2.city)."""
    if not text:
        return []
    return [match.group(0) for match in ENTITY_REF_PATTERN.finditer(str(text))]


def is_valid_entity_ref(s: str) -> bool:
    """True if s is a valid single entity reference."""
    return bool(s and ENTITY_REF_VALIDATION_PATTERN.match(s.strip()))


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


@dataclass
class Annotation:
    """One annotation in text: [text; entity_id.attribute]."""

    start_pos: int
    end_pos: int
    original_text: str
    entity_id: str
    attribute: str | None = None


_SPECIAL_NUMBER_WORDS: dict[str, int] = {
    "once": 1,
    "twice": 2,
    "thrice": 3,
}
_NUMBER_SCALE_WORDS: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_NUMBER_WORD_CONNECTORS: frozenset[str] = frozenset({"and", "a", "an", "of"})
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_YEAR_ONLY_PATTERN = re.compile(r"^(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})$")
_APPROX_EQUAL_REL_TOL = 0.02
_APPROX_EQUAL_ABS_TOL = 0.05


def _parse_word_number_extended(text: str) -> int | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None

    canonical = raw.replace(" ", "-")
    parsed_basic = parse_word_number(canonical)
    if parsed_basic is not None:
        return parsed_basic
    if raw in _SPECIAL_NUMBER_WORDS:
        return _SPECIAL_NUMBER_WORDS[raw]

    normalized = re.sub(r"[,\u2013\u2014]", " ", raw).replace("-", " ")
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return None

    total = 0
    current = 0
    seen_number_token = False
    for token in tokens:
        if token in _NUMBER_WORD_CONNECTORS:
            continue
        if token in _SPECIAL_NUMBER_WORDS:
            current += _SPECIAL_NUMBER_WORDS[token]
            seen_number_token = True
            continue
        if token in WORD_TO_NUMBER:
            current += WORD_TO_NUMBER[token]
            seen_number_token = True
            continue
        if token in _NUMBER_SCALE_WORDS:
            seen_number_token = True
            scale = _NUMBER_SCALE_WORDS[token]
            if scale == 100:
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * scale
                current = 0
            continue
        singular = token[:-1] if token.endswith("s") else token
        if singular in _NUMBER_SCALE_WORDS:
            seen_number_token = True
            scale = _NUMBER_SCALE_WORDS[singular]
            if scale == 100:
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * scale
                current = 0
            continue
        if token in ORDINAL_WORD_TO_NUMBER:
            current += ORDINAL_WORD_TO_NUMBER[token]
            seen_number_token = True
            continue
        return None

    if not seen_number_token:
        return None
    return total + current


def _coerce_numeric_surface(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value

    raw = str(value).strip()
    if not raw:
        return None
    compact = raw.replace(",", "").strip()
    if not compact:
        return None

    parsed_int_surface = parse_integer_surface_number(compact)
    if parsed_int_surface is not None:
        return parsed_int_surface

    if compact.endswith("%"):
        compact = compact[:-1].strip()
        if not compact:
            return None

    fraction_match = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", compact)
    if fraction_match:
        denominator = int(fraction_match.group(2))
        if denominator != 0:
            return int(fraction_match.group(1)) / denominator

    if _FLOAT_PATTERN.fullmatch(compact):
        try:
            as_float = float(compact)
        except ValueError:
            as_float = None
        if as_float is not None:
            if as_float.is_integer():
                return int(as_float)
            return as_float

    digit_scale_match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
        r"(hundreds?|thousands?|millions?|billions?|trillions?)",
        compact,
        flags=re.IGNORECASE,
    )
    if digit_scale_match:
        try:
            amount = float(digit_scale_match.group(1))
            scale_label = digit_scale_match.group(2).lower()
            if scale_label.endswith("s"):
                scale_label = scale_label[:-1]
            scale_value = _NUMBER_SCALE_WORDS.get(scale_label)
            if scale_value is not None:
                scaled = amount * scale_value
                if scaled.is_integer():
                    return int(scaled)
                return scaled
        except ValueError:
            pass

    parsed_word = _parse_word_number_extended(compact)
    if parsed_word is not None:
        return parsed_word

    return None


@lru_cache(maxsize=8192)
def _parse_date_surface_cached(raw: str) -> date | None:
    normalized = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    candidates = [normalized, normalized.replace(",", ""), normalized.replace("  ", " ").strip()]

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    )
    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue

    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", normalized)
    if year_match:
        try:
            return date(int(year_match.group(1)), 1, 1)
        except ValueError:
            return None
    return None


@lru_cache(maxsize=8192)
def _parse_date_surface_components_cached(raw: str) -> dict[str, Any]:
    normalized = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    candidates = [normalized, normalized.replace(",", ""), normalized.replace("  ", " ").strip()]

    component_formats: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("%Y-%m-%d", ("year", "month", "day_of_month")),
        ("%Y/%m/%d", ("year", "month", "day_of_month")),
        ("%d/%m/%Y", ("year", "month", "day_of_month")),
        ("%m/%d/%Y", ("year", "month", "day_of_month")),
        ("%B %d, %Y", ("year", "month", "day_of_month")),
        ("%b %d, %Y", ("year", "month", "day_of_month")),
        ("%B %d %Y", ("year", "month", "day_of_month")),
        ("%b %d %Y", ("year", "month", "day_of_month")),
        ("%d %B %Y", ("year", "month", "day_of_month")),
        ("%d %b %Y", ("year", "month", "day_of_month")),
        ("%B %Y", ("year", "month")),
        ("%b %Y", ("year", "month")),
        ("%B %d", ("month", "day_of_month")),
        ("%b %d", ("month", "day_of_month")),
        ("%Y", ("year",)),
    )

    for candidate in candidates:
        for fmt, fields in component_formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            components: dict[str, Any] = {}
            if "year" in fields:
                components["year"] = int(parsed.year)
            if "month" in fields:
                components["month"] = parsed.strftime("%B")
            if "day_of_month" in fields:
                components["day_of_month"] = int(parsed.day)
            return components

    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", normalized)
    if year_match:
        try:
            return {"year": int(year_match.group(1))}
        except ValueError:
            return {}
    return {}


def _parse_date_surface(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)):
        year = int(value)
        if 1000 <= year <= 9999:
            try:
                return date(year, 1, 1)
            except ValueError:
                return None
        return None

    raw = str(value).strip()
    if not raw:
        return None
    return _parse_date_surface_cached(raw)


def _parse_date_surface_components(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return {
            "year": int(value.year),
            "month": value.strftime("%B"),
            "day_of_month": int(value.day),
        }
    raw = str(value).strip()
    if not raw:
        return {}
    return dict(_parse_date_surface_components_cached(raw))


def _parse_timestamp_surface(value: Any) -> int | None:
    """Parse a time-of-day surface into minutes since midnight.

    Supports common document formats such as ``5:34 p.m. CDT`` and ``6:12 p.m``.
    Timezone suffixes are ignored because generation rules only rely on local
    same-document differences/orderings, not absolute timezone conversion.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    normalized = raw.casefold().strip()
    normalized = normalized.replace("\u202f", " ").replace("\xa0", " ")
    normalized = re.sub(r"\b(a|p)\.(m)\.\b", r"\1m", normalized)
    normalized = re.sub(r"\b(a|p)\.m\b", r"\1m", normalized)
    normalized = re.sub(r"\b(am|pm)\b.*$", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\s*(am|pm))?", normalized)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3)

    if minute >= 60:
        return None
    if ampm is None:
        if hour >= 24:
            return None
        return hour * 60 + minute
    if hour < 1 or hour > 12:
        return None
    if hour == 12:
        hour = 0
    if ampm == "pm":
        hour += 12
    return hour * 60 + minute


def _add_years_safe(base_date: date, years: int) -> date:
    target_year = int(base_date.year) + int(years)
    if target_year < 1:
        target_year = 1
    if target_year > 9999:
        target_year = 9999
    try:
        return base_date.replace(year=target_year)
    except ValueError:
        # Leap-day fallback for non-leap target years.
        return base_date.replace(year=target_year, month=2, day=28)


class AnnotationParser:
    """Parse annotations from document text and extract factual entities."""

    @staticmethod
    def parse_annotations(text: str) -> list[Annotation]:
        """Parse [text; entity_id.attribute] or [text; entity_id] from text."""
        if not text:
            return []
        annotations = []
        pattern = r"\[([^\]]+);\s*([^\]]+)\]"
        for match in re.finditer(pattern, text):
            original_text = match.group(1).strip()
            entity_ref = match.group(2).strip()
            entity_id, attribute = (
                (entity_ref.split(".", 1)[0], entity_ref.split(".", 1)[1]) if "." in entity_ref else (entity_ref, None)
            )
            annotations.append(
                Annotation(
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=original_text,
                    entity_id=entity_id,
                    attribute=attribute,
                )
            )
        return annotations

    @staticmethod
    def _parse_temporal_year_surface(year_text: str) -> int | None:
        raw = str(year_text or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass

        year_range_match = re.fullmatch(
            r"(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\s*[-\u2013\u2014]\s*(\d{2}|\d{4})",
            raw,
        )
        if year_range_match:
            return int(year_range_match.group(1))
        embedded_year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", raw)
        if embedded_year_match:
            return int(embedded_year_match.group(1))
        return None

    @staticmethod
    def extract_factual_entities(
        doc: AnnotatedDocument,
        *,
        include_questions: bool = False,
    ) -> EntityCollection:
        """Extract factual entities from document text (and optionally questions)."""
        entities = EntityCollection()
        all_annotations = list(AnnotationParser.parse_annotations(doc.document_to_annotate))
        if include_questions:
            for q in doc.questions:
                all_annotations.extend(AnnotationParser.parse_annotations(q.question))
                for ref in find_entity_refs(q.question):
                    entity_id, attr = (ref.split(".", 1)[0], ref.split(".", 1)[1]) if "." in ref else (ref, None)
                    all_annotations.append(Annotation(0, 0, "", entity_id, attr))
                if q.answer:
                    for ref in find_entity_refs(q.answer):
                        entity_id, attr = (ref.split(".", 1)[0], ref.split(".", 1)[1]) if "." in ref else (ref, None)
                        all_annotations.append(Annotation(0, 0, "", entity_id, attr))
        entity_data: dict[str, dict[str, str]] = {}
        for ann in all_annotations:
            # Normalize organisation -> organization for consistent entity keys
            eid = normalize_entity_ref(ann.entity_id)
            if eid not in entity_data:
                entity_data[eid] = {}
            if ann.attribute:
                if ann.original_text:
                    entity_data[eid].setdefault(ann.attribute, ann.original_text)
                else:
                    entity_data[eid].setdefault(ann.attribute, "")
            elif ann.original_text:
                entity_data[eid].setdefault("_default", ann.original_text)
        for entity_id, attrs in entity_data.items():
            entity_type, _ = parse_entity_id(entity_id)
            if not entity_type:
                continue
            if entity_type == "number":
                entity = NumberEntity()
                if "int" in attrs:
                    entity.int_surface_format = infer_int_surface_format(attrs["int"])
                    parsed_int = _coerce_numeric_surface(attrs["int"])
                    if isinstance(parsed_int, (int, float)):
                        entity.int = int(parsed_int)
                if "str" in attrs:
                    entity.str = attrs["str"]
                    entity.str_surface_format = infer_str_surface_format(attrs["str"])
                    if entity.int is None:
                        parsed = _coerce_numeric_surface(attrs["str"])
                        if isinstance(parsed, int):
                            entity.int = parsed
                        elif isinstance(parsed, float) and parsed.is_integer():
                            entity.int = int(parsed)
                for key in ("float", "percent", "proportion"):
                    if key in attrs:
                        parsed_numeric = _coerce_numeric_surface(attrs[key])
                        if parsed_numeric is not None:
                            try:
                                setattr(entity, key, float(parsed_numeric))
                            except (ValueError, TypeError):
                                pass
                if "fraction" in attrs:
                    entity.fraction = attrs["fraction"]
                entities.numbers[entity_id] = entity
            elif entity_type == "person":
                entity = PersonEntity()
                if "full_name" in attrs:
                    entity.full_name = attrs["full_name"]
                if "first_name" in attrs:
                    entity.first_name = attrs["first_name"]
                if "last_name" in attrs:
                    entity.last_name = attrs["last_name"]
                if "middle_name" in attrs:
                    entity.middle_name = attrs["middle_name"]
                if "age" in attrs:
                    try:
                        entity.age = int(attrs["age"])
                    except (ValueError, TypeError):
                        entity.age = attrs["age"]
                for k in (
                    "gender",
                    "ethnicity",
                    "nationality",
                    "honorific",
                    "relationship",
                    "subj_pronoun",
                    "obj_pronoun",
                    "poss_det_pronoun",
                    "poss_pro_pronoun",
                    "refl_pronoun",
                ):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                if entity.relationships is None:
                    entity.relationships = {}
                for k, v in attrs.items():
                    if k.startswith("relationship.") and v:
                        other_id = k.split(".", 1)[1]
                        entity.relationships[other_id] = v
                entities.persons[entity_id] = entity
            elif entity_type == "place":
                entity = PlaceEntity()
                for k in ("city", "street", "region", "country", "state", "natural_site", "continent", "demonym"):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                if "nationality" in attrs and entity.demonym is None:
                    entity.demonym = attrs["nationality"]
                    entity.nationality = attrs["nationality"]
                entities.places[entity_id] = entity
            elif entity_type == "temporal":
                entity = TemporalEntity()
                if "year" in attrs:
                    parsed_year = AnnotationParser._parse_temporal_year_surface(attrs["year"])
                    if parsed_year is not None:
                        entity.year = parsed_year
                if "day_of_month" in attrs:
                    try:
                        entity.day_of_month = int(attrs["day_of_month"])
                    except (ValueError, TypeError):
                        pass
                for k in ("day", "date", "month", "timestamp"):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                if "date" in attrs:
                    # Treat `.date` as the authoritative source for derived temporal
                    # components so stale question annotations cannot override the
                    # factual month/day extracted from the document body.
                    date_components = _parse_date_surface_components(attrs["date"])
                    parsed_year = date_components.get("year")
                    parsed_month = date_components.get("month")
                    parsed_day_of_month = date_components.get("day_of_month")
                    if parsed_year is not None:
                        entity.year = int(parsed_year)
                    if parsed_month is not None:
                        entity.month = parsed_month
                    if parsed_day_of_month is not None:
                        entity.day_of_month = int(parsed_day_of_month)
                    if entity.year is None:
                        m = re.search(r"\b(19|20)\d{2}\b", attrs["date"])
                        if m:
                            entity.year = int(m.group(0))
                entities.temporals[entity_id] = entity
            elif entity_type == "event":
                entity = EventEntity()
                for k in ("name", "type"):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                entities.events[entity_id] = entity
            elif entity_type == "award":
                entities.awards[entity_id] = AwardEntity(name=attrs.get("name") or attrs.get("_default"))
            elif entity_type == "legal":
                entities.legals[entity_id] = LegalEntity(
                    name=attrs.get("name") or attrs.get("_default"),
                    reference_code=attrs.get("reference_code"),
                )
            elif entity_type == "product":
                entities.products[entity_id] = ProductEntity(name=attrs.get("name") or attrs.get("_default"))
            elif entity_type in ORG_ENTITY_TYPES:
                organization_kind = infer_organization_kind(
                    entity_type=entity_type,
                    attribute=next((k for k in attrs if k != "_default"), None),
                )
                entity = OrganizationEntity(
                    name=attrs.get("name") or attrs.get("_default"),
                    organization_kind=organization_kind,
                )
                if entity.name is None:
                    for attribute_name, attribute_value in attrs.items():
                        if attribute_name == "_default":
                            continue
                        inferred_kind = infer_organization_kind(entity_type=entity_type, attribute=attribute_name)
                        if inferred_kind is not None:
                            entity.organization_kind = entity.organization_kind or inferred_kind
                            entity.name = attribute_value
                            break
                entity.organization_kind = entity.organization_kind or canonicalize_organization_type(entity_type)
                entities.organizations[entity_id] = entity
        return entities


class RuleEngine:
    """Evaluate expressions with entity references (arithmetic, conditionals, entity refs)."""

    @staticmethod
    def _normalize_weekday(value: Any) -> str | None:
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key:
            return None
        key = _WEEKDAY_ALIASES.get(key, key)
        idx = _WEEKDAY_TO_INDEX.get(key)
        if idx is None:
            return None
        return WEEKDAYS[idx]

    @staticmethod
    def _evaluate_weekday_shift(expression: str, entities: EntityCollection) -> str | None:
        """Evaluate expressions like temporal_1.day - 2 (cyclic on weekdays)."""
        m = re.fullmatch(r"(temporal_\d+\.day)\s*([+-])\s*(.+)", expression.strip(), re.IGNORECASE)
        if not m:
            return None
        ref, op, offset_expr = m.groups()
        base_value = RuleEngine._get_entity_value(entities, ref)
        base_day = RuleEngine._normalize_weekday(base_value)
        if base_day is None:
            return None
        offset_value = RuleEngine._evaluate_arithmetic(offset_expr.strip(), entities)
        offset_numeric = _coerce_numeric_surface(offset_value)
        if offset_numeric is None:
            return None
        base_idx = _WEEKDAY_TO_INDEX[base_day.lower()]
        offset = int(offset_numeric) % len(WEEKDAYS)
        if op == "-":
            offset = -offset
        return WEEKDAYS[(base_idx + offset) % len(WEEKDAYS)]

    @staticmethod
    def _calendar_year_difference(left_date: date, right_date: date) -> int:
        """Return signed full-year difference between two calendar dates."""
        if left_date >= right_date:
            years = left_date.year - right_date.year
            if (left_date.month, left_date.day) < (right_date.month, right_date.day):
                years -= 1
            return years
        years = right_date.year - left_date.year
        if (right_date.month, right_date.day) < (left_date.month, left_date.day):
            years -= 1
        return -years

    @staticmethod
    def _evaluate_year_function_expression(expression: str, entities: EntityCollection) -> int | None:
        """Evaluate helper expressions such as year(temporal_2.date - temporal_1.date)."""
        expr = str(expression or "").strip()
        match = re.fullmatch(r"year\s*\(\s*(.+?)\s*\)", expr, flags=re.IGNORECASE)
        if not match:
            return None
        inner_expr = match.group(1).strip()
        if not inner_expr:
            return None

        diff_match = re.fullmatch(r"(.+?)\s*-\s*(.+)", inner_expr)
        if diff_match:
            left_value = RuleEngine._evaluate_arithmetic(diff_match.group(1).strip(), entities)
            right_value = RuleEngine._evaluate_arithmetic(diff_match.group(2).strip(), entities)
            left_date = _parse_date_surface(left_value)
            right_date = _parse_date_surface(right_value)
            if left_date is not None and right_date is not None:
                return RuleEngine._calendar_year_difference(left_date, right_date)

        value = RuleEngine._evaluate_arithmetic(inner_expr, entities)
        parsed_date = _parse_date_surface(value)
        if parsed_date is not None:
            return int(parsed_date.year)
        numeric = _coerce_numeric_surface(value)
        if numeric is not None:
            return int(numeric)
        return None

    @staticmethod
    def _normalize_person_age_rules(rule: str) -> str:
        """Normalize bare person references in numeric comparisons to use .age.

        Converts e.g. 'person_2 < 18' -> 'person_2.age < 18' so that
        numeric comparison rules work even when the entity has a full_name.
        """
        # Pattern: person_N (without .attr) followed/preceded by a comparison operator and a number
        rule = re.sub(
            r"\b(person_\d+)\s*([<>=!]+)\s*(\d+)\b",
            lambda m: f"{m.group(1)}.age {m.group(2)} {m.group(3)}" if "." not in m.group(1) else m.group(0),
            rule,
        )
        rule = re.sub(
            r"\b(\d+)\s*([<>=!]+)\s*(person_\d+)\b",
            lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}.age" if "." not in m.group(3) else m.group(0),
            rule,
        )
        return rule

    @staticmethod
    def _normalize_comparison_expression(expression: str) -> str:
        # Accept author-provided single "=" equality in rule text.
        return re.sub(r"(?<![<>=!])=(?!=)", "==", str(expression or ""))

    @staticmethod
    def _contains_comparison_operator(expression: str) -> bool:
        return bool(re.search(r"(>=|<=|==|!=|>|<)", str(expression or "")))

    @staticmethod
    def _evaluate_temporal_offset_expression(expression: str, entities: EntityCollection) -> Any | None:
        expr = str(expression or "").strip()
        if not expr:
            return None

        if not re.search(r"[+-]", expr):
            unit_only_match = re.fullmatch(r"(.+?)\s+(years?|days?|hours?|minutes?)", expr, flags=re.IGNORECASE)
            if unit_only_match:
                value = RuleEngine._evaluate_arithmetic(unit_only_match.group(1).strip(), entities)
                numeric = _coerce_numeric_surface(value)
                return int(numeric) if numeric is not None else None

        term_pattern = re.compile(r"([+-])\s*([^+-]+?)\s*(years?|days?)\b", flags=re.IGNORECASE)
        terms = list(term_pattern.finditer(expr))
        if not terms:
            # Legacy shorthand: treat `<date_expr> +/- N` as year offsets.
            # We intentionally gate this to non-year date surfaces so arithmetic
            # such as `temporal_2.year - temporal_1.year` remains numeric.
            bare_offset_match = re.fullmatch(r"(.+?)\s*([+-])\s*(.+)", expr)
            if bare_offset_match:
                base_expr = bare_offset_match.group(1).strip()
                amount_expr = bare_offset_match.group(3).strip()
                if base_expr and amount_expr:
                    base_value = RuleEngine._evaluate_arithmetic(base_expr, entities)
                    amount_value = RuleEngine._evaluate_arithmetic(amount_expr, entities)
                    amount = _coerce_numeric_surface(amount_value)

                    is_year_like = False
                    if isinstance(base_value, (int, float)):
                        is_year_like = True
                    elif isinstance(base_value, str) and _YEAR_ONLY_PATTERN.fullmatch(base_value.strip()):
                        is_year_like = True

                    base_date = _parse_date_surface(base_value)
                    if amount is not None and base_date is not None and not is_year_like:
                        delta = int(amount)
                        if bare_offset_match.group(2) == "-":
                            delta = -delta
                        return _add_years_safe(base_date, delta)
            return None

        base_expr = expr[: terms[0].start()].strip()
        if not base_expr:
            return None
        if term_pattern.sub("", expr).strip() != base_expr:
            return None

        base_value = RuleEngine._evaluate_arithmetic(base_expr, entities)
        if base_value is None:
            return None

        mode: str
        current_numeric: float | int | None = None
        current_date: date | None = None
        if isinstance(base_value, (int, float)) or (
            isinstance(base_value, str) and _YEAR_ONLY_PATTERN.fullmatch(base_value.strip())
        ):
            parsed_year = _coerce_numeric_surface(base_value)
            if parsed_year is None:
                return None
            try:
                current_date = date(int(parsed_year), 1, 1)
            except ValueError:
                return None
            mode = "year"
        else:
            parsed_date = _parse_date_surface(base_value)
            if parsed_date is not None:
                current_date = parsed_date
                mode = "date"
            else:
                parsed_numeric = _coerce_numeric_surface(base_value)
                if parsed_numeric is None:
                    return None
                current_numeric = parsed_numeric
                mode = "numeric"

        for match in terms:
            sign = -1 if match.group(1) == "-" else 1
            term_expr = match.group(2).strip()
            term_unit = match.group(3).strip().lower()
            term_value = RuleEngine._evaluate_arithmetic(term_expr, entities)
            amount = _coerce_numeric_surface(term_value)
            if amount is None:
                return None
            delta = int(amount) * sign

            if mode in {"year", "date"} and current_date is not None:
                if term_unit.startswith("year"):
                    current_date = _add_years_safe(current_date, delta)
                else:
                    current_date = current_date + timedelta(days=delta)
            elif current_numeric is not None:
                current_numeric = float(current_numeric) + float(delta)
            else:
                return None

        if mode == "year" and current_date is not None:
            return int(current_date.year)
        if mode == "date":
            return current_date
        if current_numeric is None:
            return None
        if float(current_numeric).is_integer():
            return int(current_numeric)
        return current_numeric

    @staticmethod
    def _evaluate_date_difference_expression(expression: str, entities: EntityCollection) -> Any | None:
        expr = str(expression or "").strip()
        if not expr:
            return None
        match = re.fullmatch(r"(.+?)\s*-\s*(.+)", expr)
        if not match:
            return None
        left_expr = match.group(1).strip()
        right_expr = match.group(2).strip()
        if not left_expr or not right_expr:
            return None

        left_value = RuleEngine._evaluate_arithmetic(left_expr, entities)
        right_value = RuleEngine._evaluate_arithmetic(right_expr, entities)
        left_date = _parse_date_surface(left_value)
        right_date = _parse_date_surface(right_value)
        if left_date is None or right_date is None:
            return None

        def _is_year_like(value: Any) -> bool:
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str) and _YEAR_ONLY_PATTERN.fullmatch(value.strip()):
                return True
            return False

        left_year_like = _is_year_like(left_value)
        right_year_like = _is_year_like(right_value)
        if left_year_like and right_year_like:
            # Let pure year arithmetic run through normal numeric eval.
            return None
        if left_year_like or right_year_like:
            # Mixed year/date arithmetic compares calendar-year offsets.
            return int(left_date.year - right_date.year)

        return int((left_date - right_date).days)

    @staticmethod
    def _evaluate_timestamp_difference_expression(expression: str, entities: EntityCollection) -> Any | None:
        expr = str(expression or "").strip()
        if not expr:
            return None
        match = re.fullmatch(r"(.+?)\s*-\s*(.+)", expr)
        if not match:
            return None
        left_expr = match.group(1).strip()
        right_expr = match.group(2).strip()
        if not left_expr or not right_expr:
            return None

        left_value = RuleEngine._evaluate_arithmetic(left_expr, entities)
        right_value = RuleEngine._evaluate_arithmetic(right_expr, entities)
        left_minutes = _parse_timestamp_surface(left_value)
        right_minutes = _parse_timestamp_surface(right_value)
        if left_minutes is None or right_minutes is None:
            return None
        return int(left_minutes - right_minutes)

    @staticmethod
    def _resolve_temporal_date_component(entity_id: str, component: str, entities: EntityCollection) -> Any:
        date_value = RuleEngine._get_entity_value(entities, f"{entity_id}.date")
        parsed_date = _parse_date_surface(date_value)
        if parsed_date is None:
            return None
        if component == "year":
            return int(parsed_date.year)
        if component == "month":
            return parsed_date.strftime("%B")
        if component == "day":
            return int(parsed_date.day)
        return None

    @staticmethod
    def validate_all_rules(rules: list[str], entities: EntityCollection) -> list[tuple[str, bool]]:
        result = []
        for rule in rules:
            try:
                cleaned_rule = str(rule or "").split("#", 1)[0].strip()
                if not cleaned_rule:
                    continue
                # Normalize bare person refs in numeric comparisons: person_2 < 18 -> person_2.age < 18
                normalized = RuleEngine._normalize_person_age_rules(cleaned_rule)
                eval_rule = (
                    normalized
                    if any(x in normalized for x in ("!=", ">=", "<=", "=="))
                    else normalized.replace(" = ", " == ")
                )
                res = RuleEngine.evaluate_expression(eval_rule, entities)
                result.append((rule, bool(res) if isinstance(res, (bool, int, float)) else False))
            except Exception as e:
                logger.debug("Rule validation failed for %r: %s", rule, e)
                result.append((rule, False))
        return result

    @staticmethod
    def evaluate_expression(expression: str, entities: EntityCollection) -> Any:
        if not expression or not isinstance(expression, str):
            return expression
        expression = expression.strip()
        cond = re.match(r"^(.+?)\s+if\s+(.+?)\s+else\s+(.+)$", expression)
        if cond:
            cond_result = RuleEngine._evaluate_condition(cond.group(2).strip(), entities)
            return RuleEngine._resolve_value((cond.group(1) if cond_result else cond.group(3)).strip(), entities)
        if expression.lower() in ("true", "false"):
            return expression.lower() == "true"
        normalized = RuleEngine._normalize_comparison_expression(expression)
        if RuleEngine._contains_comparison_operator(normalized):
            return RuleEngine._evaluate_condition(normalized, entities)
        return RuleEngine._evaluate_arithmetic(normalized, entities)

    @staticmethod
    def _evaluate_condition(condition: str, entities: EntityCollection) -> bool:
        normalized = RuleEngine._normalize_comparison_expression(condition)
        for op in (">=", "<=", "==", "!=", ">", "<"):
            if op in normalized:
                parts = normalized.split(op, 1)
                if len(parts) == 2:
                    left = RuleEngine._evaluate_arithmetic(parts[0].strip(), entities)
                    right = RuleEngine._evaluate_arithmetic(parts[1].strip(), entities)
                    return RuleEngine._compare(op, left, right)
        return bool(RuleEngine._evaluate_arithmetic(normalized, entities))

    @staticmethod
    def _compare(op: str, left: Any, right: Any) -> bool:
        """Compare left and right with op, coercing to numbers when possible to avoid str vs int TypeError."""
        left_date = _parse_date_surface(left)
        right_date = _parse_date_surface(right)
        left_is_year_like = bool(
            isinstance(left, (int, float))
            or (isinstance(left, str) and _YEAR_ONLY_PATTERN.fullmatch(left.strip() or ""))
        )
        right_is_year_like = bool(
            isinstance(right, (int, float))
            or (isinstance(right, str) and _YEAR_ONLY_PATTERN.fullmatch(right.strip() or ""))
        )
        if left_date is not None and right_date is not None and left_is_year_like == right_is_year_like:
            if op == ">":
                return left_date > right_date
            if op == "<":
                return left_date < right_date
            if op == ">=":
                return left_date >= right_date
            if op == "<=":
                return left_date <= right_date
            if op == "==":
                return left_date == right_date
            if op == "!=":
                return left_date != right_date

        # Allow year-vs-date comparisons such as `temporal_2.year == temporal_4.date`.
        left_year_only = None
        right_year_only = None
        if isinstance(left, (int, float)) and float(left).is_integer() and 1000 <= int(float(left)) <= 9999:
            left_year_only = int(float(left))
        elif isinstance(left, str) and _YEAR_ONLY_PATTERN.fullmatch(left.strip() or ""):
            left_year_only = int(left.strip())
        if isinstance(right, (int, float)) and float(right).is_integer() and 1000 <= int(float(right)) <= 9999:
            right_year_only = int(float(right))
        elif isinstance(right, str) and _YEAR_ONLY_PATTERN.fullmatch(right.strip() or ""):
            right_year_only = int(right.strip())

        if left_year_only is not None and right_date is not None:
            right_year = int(right_date.year)
            left_year = int(left_year_only)
            if op == ">":
                return left_year > right_year
            if op == "<":
                return left_year < right_year
            if op == ">=":
                return left_year >= right_year
            if op == "<=":
                return left_year <= right_year
            if op == "==":
                return left_year == right_year
            if op == "!=":
                return left_year != right_year
        if right_year_only is not None and left_date is not None:
            left_year = int(left_date.year)
            right_year = int(right_year_only)
            if op == ">":
                return left_year > right_year
            if op == "<":
                return left_year < right_year
            if op == ">=":
                return left_year >= right_year
            if op == "<=":
                return left_year <= right_year
            if op == "==":
                return left_year == right_year
            if op == "!=":
                return left_year != right_year

        try:
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
        except TypeError:
            pass

        left_n = _coerce_numeric_surface(left)
        right_n = _coerce_numeric_surface(right)
        if left_n is not None and right_n is not None:
            left_f = float(left_n)
            right_f = float(right_n)
            if op == ">":
                return left_f > right_f
            if op == "<":
                return left_f < right_f
            if op == ">=":
                return left_f >= right_f
            if op == "<=":
                return left_f <= right_f
            both_integral = left_f.is_integer() and right_f.is_integer()
            approx_equal = math.isclose(
                left_f,
                right_f,
                rel_tol=_APPROX_EQUAL_REL_TOL,
                abs_tol=_APPROX_EQUAL_ABS_TOL,
            )
            if op == "==":
                return (left_f == right_f) if both_integral else approx_equal
            if op == "!=":
                return (left_f != right_f) if both_integral else (not approx_equal)

        if op in ("==", "!="):
            left_s = str(left or "").strip().lower()
            right_s = str(right or "").strip().lower()
            return left_s == right_s if op == "==" else left_s != right_s
        return False

    @staticmethod
    def _evaluate_arithmetic(expression: str, entities: EntityCollection) -> Any:
        expr = str(expression or "").strip()
        abs_match = re.fullmatch(r"abs\s*\(\s*(.+?)\s*\)", expr, flags=re.IGNORECASE)
        if abs_match:
            inner_value = RuleEngine._evaluate_arithmetic(abs_match.group(1).strip(), entities)
            numeric_value = _coerce_numeric_surface(inner_value)
            if numeric_value is not None:
                absolute_value = abs(float(numeric_value))
                return int(absolute_value) if absolute_value.is_integer() else absolute_value

        shifted_day = RuleEngine._evaluate_weekday_shift(expr, entities)
        if shifted_day is not None:
            return shifted_day
        year_result = RuleEngine._evaluate_year_function_expression(expr, entities)
        if year_result is not None:
            return year_result
        if PERSON_RELATIONSHIP_REF_PATTERN.fullmatch(expr):
            return RuleEngine._get_entity_value(entities, expr)

        temporal_offset = RuleEngine._evaluate_temporal_offset_expression(expr, entities)
        if temporal_offset is not None:
            return temporal_offset

        date_difference = RuleEngine._evaluate_date_difference_expression(expr, entities)
        if date_difference is not None:
            return date_difference

        timestamp_difference = RuleEngine._evaluate_timestamp_difference_expression(expr, entities)
        if timestamp_difference is not None:
            return timestamp_difference

        expr = re.sub(
            r"\b(temporal_\d+)\.date\.(year|month|day)\b",
            lambda m: (
                "None"
                if (value := RuleEngine._resolve_temporal_date_component(m.group(1), m.group(2), entities)) is None
                else (repr(value) if isinstance(value, str) else str(value))
            ),
            expr,
            flags=re.IGNORECASE,
        )

        m = ENTITY_REF_PATTERN.match(expr)
        if m and m.group(0) == expr:
            return RuleEngine._get_entity_value(entities, m.group(0))
        refs = find_entity_refs(expression)
        for ref in refs:
            if RuleEngine._get_entity_value(entities, ref) is None:
                return None
        # Plain space-separated refs only: return values joined by space.
        if refs and not ENTITY_REF_PATTERN.sub("", expr).strip():
            values = [RuleEngine._get_entity_value(entities, ref) for ref in refs]
            if all(v is not None for v in values):
                return " ".join(str(v) for v in values)

        def replacement_for_ref(entity_ref: str) -> str:
            v = RuleEngine._get_entity_value(entities, entity_ref)
            if v is None:
                return "None"
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                parsed_numeric = _coerce_numeric_surface(v)
                if parsed_numeric is not None:
                    return str(parsed_numeric)
                return json.dumps(v)
            return str(v)

        for ref in sorted(refs, key=len, reverse=True):
            expr = expr.replace(ref, replacement_for_ref(ref))
        try:
            return eval(
                expr,
                {
                    "__builtins__": {},
                    "abs": abs,
                    "century_of": century_of,
                    "century_start": century_start,
                    "century_end": century_end,
                },
                {},
            )
        except Exception as e:
            logger.debug("Arithmetic eval failed for %r: %s", expression, e)
            return expression

    @staticmethod
    def _get_entity_value(entities: EntityCollection, entity_ref: str) -> Any | None:
        if not entity_ref:
            return None
        entity_ref = normalize_entity_ref(entity_ref)
        entity_id, attribute = (
            (entity_ref.split(".", 1)[0], entity_ref.split(".", 1)[1]) if "." in entity_ref else (entity_ref, None)
        )
        entity_type, _ = parse_entity_id(entity_id)
        if not entity_type:
            return None
        canonical_entity_type = canonicalize_organization_type(entity_type) or entity_type
        coll = {
            "number": entities.numbers,
            "person": entities.persons,
            "place": entities.places,
            "temporal": entities.temporals,
            "event": entities.events,
            "award": entities.awards,
            "legal": entities.legals,
            "product": entities.products,
            "organization": entities.organizations,
        }
        for organization_type in ORG_ENTITY_TYPES:
            coll[organization_type] = entities.organizations
        target = coll.get(canonical_entity_type, {}) if canonical_entity_type in coll else {}
        entity = target.get(entity_id)
        if entity is None:
            parsed_type, parsed_index = parse_entity_id(entity_id)
            if parsed_type is not None and parsed_index is not None:
                canonical_id = f"{parsed_type}_{parsed_index}"
                if canonical_id != entity_id:
                    entity = target.get(canonical_id)
        if entity is None and entity_id.startswith(("organization_", "organisation_")):
            alternate_ids = []
            if entity_id.startswith("organization_"):
                alternate_ids.append(entity_id.replace("organization_", "organisation_", 1))
            if entity_id.startswith("organisation_"):
                alternate_ids.append(entity_id.replace("organisation_", "organization_", 1))
            for alternate_id in alternate_ids:
                entity = target.get(alternate_id)
                if entity is not None:
                    break
        if entity is None and canonical_entity_type in ORG_ENTITY_TYPES:
            _, entity_index = parse_entity_id(entity_id)
            if entity_index is not None:
                entity = entities.organizations.get(f"organization_{entity_index}")
        if entity is None:
            return None
        if attribute:
            if canonical_entity_type == "person" and attribute.startswith("relationship."):
                other_person_id = attribute.split(".", 1)[1].strip()
                relationship_map = getattr(entity, "relationships", None) or {}
                if isinstance(relationship_map, dict):
                    value = relationship_map.get(other_person_id)
                    if value is not None:
                        return value
                return getattr(entity, "relationship", None)
            if canonical_entity_type in ORG_ENTITY_TYPES:
                val = organization_attribute_value(entity, attribute)
            else:
                val = getattr(entity, attribute, None) if hasattr(entity, attribute) else None
            if canonical_entity_type == "temporal" and val is None:
                parsed_date = _parse_date_surface(getattr(entity, "date", None))
                if parsed_date is not None:
                    if attribute == "year":
                        val = int(parsed_date.year)
                    elif attribute == "month":
                        # Keep month textual to align with annotation surfaces.
                        val = parsed_date.strftime("%B")
                    elif attribute == "day_of_month":
                        val = int(parsed_date.day)
            if attribute == "int" and val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return val
        if entity_type == "number":
            for attr in ("int", "str", "float", "percent", "proportion", "fraction"):
                value = getattr(entity, attr, None)
                if value is not None:
                    return value
            return None
        # Person-specific cascade: try name first, then age (so bare `person_X`
        # resolves to age when the entity has no name, enabling rules like `person_2 < 18`).
        if canonical_entity_type == "person":
            for attr in ("full_name", "first_name", "last_name", "age"):
                v = getattr(entity, attr, None)
                if v is not None:
                    return v
            return None
        if canonical_entity_type == "place":
            for attr in ("city", "country", "state", "region", "natural_site", "street", "continent", "demonym"):
                value = getattr(entity, attr, None)
                if value is not None:
                    return value
            return None
        if canonical_entity_type == "legal":
            return getattr(entity, "name", None) or getattr(entity, "reference_code", None)
        if canonical_entity_type in {"award", "product"}:
            return getattr(entity, "name", None)
        if canonical_entity_type in ORG_ENTITY_TYPES:
            return get_organization_name(entity)
        for attr in ("full_name", "city", "date", "name"):
            v = getattr(entity, attr, None)
            if v is not None:
                return v
        return None

    @staticmethod
    def _resolve_value(value: str, entities: EntityCollection) -> Any:
        value = value.strip()
        if ENTITY_REF_VALIDATION_PATTERN.match(value):
            return RuleEngine._get_entity_value(entities, value)
        try:
            return RuleEngine._evaluate_arithmetic(value, entities)
        except Exception as e:
            logger.debug("Resolve value failed for %r: %s", value, e)
            return value


# --- Relationship mapper (gender-appropriate terms) ---
_GENDERED_RELATIONSHIPS = {
    "sibling": ("brother", "sister", "sibling"),
    "parent": ("father", "mother", "parent"),
    "child": ("son", "daughter", "child"),
    "spouse": ("husband", "wife", "spouse"),
    "grandparent": ("grandfather", "grandmother", "grandparent"),
    "grandchild": ("grandson", "granddaughter", "grandchild"),
    "parent_sibling": ("uncle", "aunt", "parent's sibling"),
    "sibling_child": ("nephew", "niece", "sibling's child"),
}
_RELATIONSHIP_TO_BASE = {}
for _base, (_m, _f, _n) in _GENDERED_RELATIONSHIPS.items():
    _RELATIONSHIP_TO_BASE[_m.lower()] = (_base, "male")
    _RELATIONSHIP_TO_BASE[_f.lower()] = (_base, "female")
    _RELATIONSHIP_TO_BASE[_n.lower()] = (_base, "neutral")
_NON_GENDERED = {"cousin", "friend", "partner", "colleague", "ally", "rival", "enemy", "neighbor"}


def map_relationship_for_gender(original_relationship: str, target_gender: str) -> str:
    """Map relationship to gender-appropriate term (target_gender: 'male'|'female'|'neutral')."""
    orig = original_relationship.lower()
    if orig in _NON_GENDERED:
        return original_relationship
    if orig in _RELATIONSHIP_TO_BASE:
        base, _ = _RELATIONSHIP_TO_BASE[orig]
        male, female, neutral = _GENDERED_RELATIONSHIPS[base]
        return male if target_gender == "male" else female if target_gender == "female" else neutral
    return original_relationship


def get_appropriate_relationship(original_relationship: str, person_entity: Any) -> str:
    """Map relationship to gender-appropriate term from person pronouns."""

    def infer_gender():
        for attr in ("subj_pronoun", "obj_pronoun", "poss_det_pronoun"):
            p = getattr(person_entity, attr, None)
            if not p:
                continue
            p = str(p).lower()
            if p in ("he", "him"):
                return "male"
            if p in ("she", "her", "hers"):
                return "female"
            if p in ("they", "them", "their", "theirs"):
                return "neutral"
        return "neutral"

    return map_relationship_for_gender(original_relationship, infer_gender())


__all__ = [
    "ENTITY_TAXONOMY",
    "FULL_REPLACE_ENTITY_TYPES",
    "PARTIAL_REPLACE_ATTRIBUTES",
    "REPLACE_MODE_ALL",
    "REPLACE_MODE_NON_NUMERICAL",
    "REPLACE_MODE_NUMERICAL",
    "REPLACE_MODE_TEMPORAL",
    "VALID_ENTITY_TYPES",
    "VALID_REPLACE_MODES",
    "WORD_TO_NUMBER",
    "Annotation",
    "AnnotationParser",
    "AnnotationValidationError",
    "RuleEngine",
    "create_run_dir",
    "compose_rule_text",
    "fictional_document_to_yaml_dict",
    "find_entity_refs",
    "get_appropriate_relationship",
    "is_valid_entity_ref",
    "load_annotated_document",
    "load_entity_pool",
    "normalize_rule_expressions",
    "normalize_rules_for_storage",
    "map_relationship_for_gender",
    "normalize_entity_ref",
    "parse_word_number",
    "partition_generation_rules",
    "replace_mode_label",
    "split_rule_text_and_comment",
    "validate_annotations",
    "validate_question_and_answer_entity_scope",
]
