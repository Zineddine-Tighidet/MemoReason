"""Validate annotation syntax and question scope against the MemoReason taxonomy."""

from __future__ import annotations

import re
from typing import Any

from .entity_taxonomy import (
    ENTITY_TAXONOMY,
    LEGACY_ENTITY_ATTRIBUTES,
    parse_entity_id,
)
from .organization_types import (
    LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND,
    LEGACY_ORGANIZATION_TYPE_ALIASES,
)
from .annotation_references import _split_entity_ref, find_entity_refs, normalize_entity_ref


class AnnotationValidationError(Exception):
    """Raised when an annotation violates the entity taxonomy."""

    pass


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
    allowed_entity_ids = {_entity_id_from_entity_ref(ref) for ref in allowed_entity_refs}

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
                        out_of_scope_references.append(f"{question_id}/reasoning_chain[{step_index}] -> {entity_ref}")

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
