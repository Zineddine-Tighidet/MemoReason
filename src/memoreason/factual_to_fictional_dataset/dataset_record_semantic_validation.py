"""Semantic lint checks for rendered fictional dataset records."""

from __future__ import annotations

import re
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import AnnotationParser
from memoreason.benchmark_definition.annotation_values import _weekday_from_date_surface
from memoreason.benchmark_definition.document_schema import AnnotatedDocument

from .dataset_record_constants import (
    AGE_YEAR_PATTERN,
    ALIAS_TAUTOLOGY_PATTERN,
    BIRTH_YEAR_PATTERN,
    BLANKED_NUMTEMP_RENDER_PATTERNS,
    DOUBLE_YEAR_DATE_PATTERN,
    FACTUAL_LEAK_ATTRIBUTES,
    GENERIC_SINGLE_TOKEN_NAME_LITERALS,
    PAREN_TAUTOLOGY_PATTERN,
    SHORT_YEAR_DATE_PATTERN,
)
from .dataset_record_core import _deserialize_entities


def _iter_replaced_factual_literals(
    replaced_factual_entities: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    literals: list[tuple[str, str]] = []
    for entity_entries in (replaced_factual_entities or {}).values():
        if not isinstance(entity_entries, dict):
            continue
        for attr_map in entity_entries.values():
            if not isinstance(attr_map, dict):
                continue
            for attr, value in attr_map.items():
                literal = _factual_leak_literal(attr, value)
                if literal:
                    literals.append((attr, literal))
    return literals


def _factual_leak_literal(attr: str, value: Any) -> str | None:
    if attr not in FACTUAL_LEAK_ATTRIBUTES or value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) < 4 or not re.search(r"[A-Za-z]", text):
        return None
    if attr == "name" and " " not in text and len(text) < 6:
        return None
    if attr == "name" and " " not in text and text == text.lower():
        return None
    if attr == "name" and " " not in text and text.casefold() in GENERIC_SINGLE_TOKEN_NAME_LITERALS:
        return None
    return text


def _literal_regex_source(literal: str) -> str:
    escaped = re.escape(literal)
    leading_boundary = r"(?<!\w)" if literal[:1].isalnum() or literal.startswith("_") else ""
    trailing_boundary = r"(?!\w)" if literal[-1:].isalnum() or literal.endswith("_") else ""
    return f"{leading_boundary}{escaped}{trailing_boundary}"


def _literal_occurs_in_text(literal: str, text: str) -> bool:
    return bool(re.search(_literal_regex_source(literal), text, flags=re.IGNORECASE))


def _literal_pattern(literal: str) -> re.Pattern[str]:
    return re.compile(_literal_regex_source(literal), flags=re.IGNORECASE)


def _replaced_attributes_by_entity(replaced_factual_entities: dict[str, Any] | None) -> dict[str, set[str]]:
    replaced: dict[str, set[str]] = {}
    for entity_entries in (replaced_factual_entities or {}).values():
        if not isinstance(entity_entries, dict):
            continue
        for entity_id, attr_map in entity_entries.items():
            if isinstance(attr_map, dict):
                replaced.setdefault(str(entity_id), set()).update(str(attr) for attr in attr_map)
    return replaced


def _plain_text_with_preserved_source_mask(
    annotated_text: str,
    *,
    replaced_attributes: dict[str, set[str]],
) -> tuple[str, list[bool]]:
    plain_parts: list[str] = []
    preserved_mask: list[bool] = []
    cursor = 0
    for annotation in sorted(AnnotationParser.parse_annotations(annotated_text), key=lambda item: item.start_pos):
        prefix = annotated_text[cursor : annotation.start_pos]
        plain_parts.append(prefix)
        # Text outside an annotation is copied verbatim by the renderer.  A
        # factual literal occurring there is therefore a legitimate preserved
        # source occurrence, not evidence that its annotated occurrence leaked.
        preserved_mask.extend([True] * len(prefix))

        surface = str(annotation.original_text or "")
        replaced_attrs = replaced_attributes.get(annotation.entity_id, set())
        annotation_is_unreplaced = annotation.entity_id not in replaced_attributes or (
            annotation.attribute not in replaced_attrs
        )
        plain_parts.append(surface)
        preserved_mask.extend([annotation_is_unreplaced] * len(surface))
        cursor = annotation.end_pos

    suffix = annotated_text[cursor:]
    plain_parts.append(suffix)
    preserved_mask.extend([True] * len(suffix))
    return "".join(plain_parts), preserved_mask


def _allowed_source_overlap_count(
    *,
    source_document: AnnotatedDocument | None,
    literal: str,
    replaced_factual_entities: dict[str, Any] | None,
) -> int:
    if source_document is None:
        return 0
    replaced_attributes = _replaced_attributes_by_entity(replaced_factual_entities)
    source_texts = [
        source_document.document_to_annotate,
        *(question.question for question in source_document.questions),
    ]
    pattern = _literal_pattern(literal)
    allowed_count = 0
    for annotated_text in source_texts:
        plain_text, preserved_mask = _plain_text_with_preserved_source_mask(
            annotated_text,
            replaced_attributes=replaced_attributes,
        )
        for match in pattern.finditer(plain_text):
            if any(preserved_mask[match.start() : match.end()]):
                allowed_count += 1
    return allowed_count


def _allowed_unreplaced_factual_literals(generated_payload: dict[str, Any]) -> list[str]:
    entities_used = generated_payload.get("entities_used") or {}
    replaced_entities = generated_payload.get("replaced_factual_entities") or {}
    allowed_literals: list[str] = []
    if not isinstance(entities_used, dict) or not isinstance(replaced_entities, dict):
        return allowed_literals

    for plural_key, entity_entries in entities_used.items():
        if not isinstance(entity_entries, dict):
            continue
        replaced_for_group = replaced_entities.get(plural_key) or {}
        if not isinstance(replaced_for_group, dict):
            replaced_for_group = {}
        for entity_id, attr_map in entity_entries.items():
            if not isinstance(attr_map, dict):
                continue
            replaced_attrs = replaced_for_group.get(entity_id) or {}
            if not isinstance(replaced_attrs, dict):
                replaced_attrs = {}
            for attr, value in attr_map.items():
                if attr in replaced_attrs:
                    continue
                literal = _factual_leak_literal(attr, value)
                if literal is None and isinstance(value, str):
                    candidate = " ".join(value.split())
                    literal = candidate if len(candidate) >= 4 and re.search(r"[A-Za-z]", candidate) else None
                if literal:
                    allowed_literals.append(literal)
    return allowed_literals


def _semantic_payload_issues(
    generated_payload: dict[str, Any],
    *,
    source_document: AnnotatedDocument | None = None,
) -> list[str]:
    document_text = str(generated_payload.get("generated_document") or "")
    entities = _deserialize_entities(generated_payload.get("entities_used") or {})
    replaced_entities = generated_payload.get("replaced_factual_entities") or {}
    question_texts = [
        str(question_entry.get("question") or "") for question_entry in generated_payload.get("questions") or []
    ]
    combined_text = "\n".join([document_text, *question_texts])
    issues: list[str] = []

    if match := DOUBLE_YEAR_DATE_PATTERN.search(document_text):
        issues.append(f"malformed date rendering: {match.group(0)!r}")
    if match := SHORT_YEAR_DATE_PATTERN.search(document_text):
        issues.append(f"short-year date rendering: {match.group(0)!r}")
    if match := ALIAS_TAUTOLOGY_PATTERN.search(document_text):
        issues.append(f"tautological aliasing: {match.group(0)!r}")
    if match := PAREN_TAUTOLOGY_PATTERN.search(document_text):
        issues.append(f"parenthetical tautology: {match.group(0)!r}")

    birth_match = BIRTH_YEAR_PATTERN.search(document_text)
    if birth_match:
        birth_year = int(birth_match.group(1))
        for age_text, year_text in AGE_YEAR_PATTERN.findall(document_text):
            age = int(age_text)
            event_year = int(year_text)
            if abs((event_year - birth_year) - age) > 1:
                issues.append(f"birth-year chronology contradiction: birth={birth_year}, age={age}, year={event_year}")
                break

    leaked_literals: list[str] = []
    allowed_unreplaced_literals = _allowed_unreplaced_factual_literals(generated_payload)
    for _attr, literal in _iter_replaced_factual_literals(generated_payload.get("replaced_factual_entities")):
        if any(_literal_occurs_in_text(literal, allowed_literal) for allowed_literal in allowed_unreplaced_literals):
            continue
        output_occurrence_count = len(_literal_pattern(literal).findall(combined_text))
        allowed_source_overlap_count = _allowed_source_overlap_count(
            source_document=source_document,
            literal=literal,
            replaced_factual_entities=generated_payload.get("replaced_factual_entities"),
        )
        if output_occurrence_count > allowed_source_overlap_count:
            leaked_literals.append(literal)
    if leaked_literals:
        issues.append(
            "replaced factual literals still appear in output: " + ", ".join(sorted(set(leaked_literals))[:10])
        )

    if (replaced_entities.get("numbers") or replaced_entities.get("temporals")) and any(
        pattern.search(document_text) for pattern in BLANKED_NUMTEMP_RENDER_PATTERNS
    ):
        issues.append("blank numeric/temporal render detected in generated document text")

    for temporal_id, temporal in entities.temporals.items():
        actual_weekday = str(getattr(temporal, "day", "") or "").strip()
        expected_weekday = _weekday_from_date_surface(getattr(temporal, "date", None))
        if actual_weekday and expected_weekday is not None and actual_weekday.casefold() != expected_weekday.casefold():
            issues.append(
                f"temporal weekday/date contradiction: {temporal_id}.day={actual_weekday!r}, "
                f"date implies {expected_weekday!r}"
            )

    primary_org = entities.organizations.get("entreprise_org_1")
    primary_org_name = str(getattr(primary_org, "name", "") or "").strip()
    if primary_org_name:
        replaced_attributes = _replaced_attributes_by_entity(replaced_entities)
        primary_org_name_was_replaced = "name" in replaced_attributes.get("entreprise_org_1", set())
        normalized_org_name = re.sub(r"[^a-z0-9]+", " ", primary_org_name.lower()).strip()
        org_tokens = {token for token in normalized_org_name.split() if token}
        for place_id, place in entities.places.items():
            for attr in ("country", "city", "region", "state", "natural_site"):
                value = str(getattr(place, attr, "") or "").strip()
                normalized_place = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
                place_attribute_was_replaced = attr in replaced_attributes.get(place_id, set())
                if (
                    value
                    and normalized_place
                    and (primary_org_name_was_replaced or place_attribute_was_replaced)
                    and (normalized_org_name == normalized_place or normalized_place in org_tokens)
                    and re.search(rf"\b{re.escape(value)}\b", document_text)
                ):
                    issues.append(
                        f"place/self-reference collision: organization {primary_org_name!r} overlaps with place {value!r}"
                    )
                    break
            if issues and issues[-1].startswith("place/self-reference collision:"):
                break

    return issues
