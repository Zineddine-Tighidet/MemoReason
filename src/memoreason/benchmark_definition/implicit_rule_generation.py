"""Generate default implicit numeric and temporal constraints from annotations."""

# Preserve the exact integer coercions used by the frozen dataset generator.
# ruff: noqa: RUF046

from __future__ import annotations

import re
from typing import Any

from .entity_taxonomy import parse_integer_surface_number, parse_word_number
from .implicit_rule_formatting import (
    AGE_RANGE_PERCENT,
    CENTURY_RANGE_PERCENT,
    IMPLICIT_RULE_PRECISION,
    NUMBER_RANGE_PERCENT,
    SMALL_NUMBER_FIXED_WINDOW_DELTA,
    SMALL_NUMBER_FIXED_WINDOW_THRESHOLD,
    SMALL_NUMBER_MIN_VALUE,
    TEMPORAL_RANGE_PERCENT,
    _ANNOTATION_PATTERN,
    _DMY_DATE_PATTERN,
    _FRACTION_PATTERN,
    _LEADING_NUMBER_PATTERN,
    _MDY_DATE_PATTERN,
    _YEAR_PATTERN,
    _AnnotationSpan,
    _apply_implicit_year_upper_cap,
    _normalize_implicit_bound,
    _normalize_implicit_numeric_value,
    _relative_small_integer_window,
    implicit_rule_uses_integer_bounds,
    implicit_rule_uses_small_number_fixed_window,
)


_NUMBER_SCALE_PATTERN = re.compile(
    r"\b(hundreds?|thousands?|millions?|billions?|trillions?)\b",
    re.IGNORECASE,
)
_NUMBER_SCALE_VALUES: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}


def generate_implicit_rules_for_document(
    doc_data: dict[str, Any],
    *,
    use_small_number_fixed_window: bool = True,
    small_number_fixed_window_delta: int = SMALL_NUMBER_FIXED_WINDOW_DELTA,
) -> list[dict[str, Any]]:
    """Generate default implicit rules from one annotated document payload."""
    document_text = str(doc_data.get("document_to_annotate") or "")
    if not document_text:
        return []

    ordered_rules: list[dict[str, Any]] = []
    seen_entity_refs: set[str] = set()

    for annotation in _parse_annotations(document_text):
        attribute = str(annotation.attribute or "").strip()
        if not attribute:
            continue

        for entity_ref in _implicit_rule_entity_refs(annotation):
            if not entity_ref or entity_ref in seen_entity_refs:
                continue
            implicit_rule = _build_rule_for_annotation(
                document_text,
                annotation,
                entity_ref,
                use_small_number_fixed_window=use_small_number_fixed_window,
                small_number_fixed_window_delta=small_number_fixed_window_delta,
            )
            if implicit_rule is None:
                continue
            ordered_rules.append(implicit_rule)
            seen_entity_refs.add(entity_ref)

    return ordered_rules


def _parse_annotations(text: str) -> list[_AnnotationSpan]:
    annotations: list[_AnnotationSpan] = []
    for match in _ANNOTATION_PATTERN.finditer(text or ""):
        entity_ref = str(match.group(2) or "").strip()
        entity_id, attribute = entity_ref.split(".", 1) if "." in entity_ref else (entity_ref, None)
        annotations.append(
            _AnnotationSpan(
                start_pos=match.start(),
                end_pos=match.end(),
                original_text=str(match.group(1) or "").strip(),
                entity_id=str(entity_id or "").strip(),
                attribute=str(attribute or "").strip() or None,
            )
        )
    return annotations


def _implicit_rule_entity_refs(annotation: _AnnotationSpan) -> list[str]:
    attribute = str(annotation.attribute or "").strip()
    if attribute == "age":
        return [annotation.entity_ref]
    if annotation.entity_id.startswith("number_") and attribute in {
        "int",
        "str",
        "float",
        "percent",
        "proportion",
        "fraction",
    }:
        return [annotation.entity_ref]
    if annotation.entity_id.startswith("temporal_"):
        if attribute == "year":
            return [annotation.entity_ref]
        if attribute == "date":
            return [f"{annotation.entity_id}.year"]
    return []


def _build_rule_for_annotation(
    document_text: str,
    annotation: _AnnotationSpan,
    entity_ref: str,
    *,
    use_small_number_fixed_window: bool = True,
    small_number_fixed_window_delta: int = SMALL_NUMBER_FIXED_WINDOW_DELTA,
) -> dict[str, Any] | None:
    if entity_ref.endswith(".age"):
        factual_value = _extract_numeric_value(annotation.original_text)
        if factual_value is None:
            return None
        return _build_rule(
            entity_ref,
            factual_value,
            AGE_RANGE_PERCENT,
            "age_range",
            use_small_number_fixed_window=use_small_number_fixed_window,
            small_number_fixed_window_delta=small_number_fixed_window_delta,
        )

    if annotation.entity_id.startswith("number_"):
        factual_value = _extract_number_value(annotation.original_text, str(annotation.attribute or ""))
        if factual_value is None:
            return None
        percentage = (
            CENTURY_RANGE_PERCENT if _is_century_annotation(document_text, annotation) else NUMBER_RANGE_PERCENT
        )
        rule_kind = "century_range" if percentage == CENTURY_RANGE_PERCENT else "number_range"
        return _build_rule(
            entity_ref,
            factual_value,
            percentage,
            rule_kind,
            use_small_number_fixed_window=use_small_number_fixed_window,
            small_number_fixed_window_delta=small_number_fixed_window_delta,
        )

    if annotation.entity_id.startswith("temporal_"):
        if entity_ref.endswith(".year"):
            factual_year = _extract_temporal_year(
                annotation.original_text,
                allow_numeric_fallback=annotation.attribute == "year",
            )
            if factual_year is None:
                return None
            return _build_rule(
                entity_ref,
                float(factual_year),
                TEMPORAL_RANGE_PERCENT,
                "temporal_year_range",
                use_small_number_fixed_window=use_small_number_fixed_window,
                small_number_fixed_window_delta=small_number_fixed_window_delta,
            )
    return None


def _build_rule(
    entity_ref: str,
    factual_value: float,
    percentage: float,
    rule_kind: str,
    *,
    use_small_number_fixed_window: bool = True,
    small_number_fixed_window_delta: int = SMALL_NUMBER_FIXED_WINDOW_DELTA,
) -> dict[str, Any]:
    integer_like = implicit_rule_uses_integer_bounds(entity_ref=entity_ref, rule_kind=rule_kind)
    factual_numeric = _normalize_implicit_numeric_value(factual_value, integer_like=integer_like)
    if use_small_number_fixed_window and implicit_rule_uses_small_number_fixed_window(
        entity_ref=entity_ref,
        rule_kind=rule_kind,
        factual_value=factual_numeric,
    ):
        factual_integer = int(round(float(factual_numeric)))
        lower_bound, upper_bound = _relative_small_integer_window(
            factual_integer,
            ratio=float(NUMBER_RANGE_PERCENT) / 100.0,
            min_delta=1,
            small_value_threshold=SMALL_NUMBER_FIXED_WINDOW_THRESHOLD,
            small_value_delta=small_number_fixed_window_delta,
            min_value=0 if factual_integer == 0 else SMALL_NUMBER_MIN_VALUE,
        )
    else:
        delta = abs(float(factual_value)) * (float(percentage) / 100.0)
        lower_bound = _normalize_implicit_bound(
            float(factual_value) - delta,
            integer_like=integer_like,
            bound_kind="lower_bound",
        )
        upper_bound = _normalize_implicit_bound(
            float(factual_value) + delta,
            integer_like=integer_like,
            bound_kind="upper_bound",
        )
    lower_bound, upper_bound = _apply_implicit_year_upper_cap(
        entity_ref,
        rule_kind,
        lower_bound,
        upper_bound,
        factual_numeric,
    )
    if lower_bound > upper_bound:
        lower_bound = upper_bound = factual_numeric
    return {
        "entity_ref": entity_ref,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "factual_value": factual_numeric,
        "percentage": round(float(percentage), IMPLICIT_RULE_PRECISION),
        "rule_kind": rule_kind,
    }


def _is_century_annotation(document_text: str, annotation: _AnnotationSpan) -> bool:
    sentence_start = annotation.start_pos
    sentence_end = annotation.end_pos
    while sentence_start > 0 and document_text[sentence_start - 1] not in ".!?\n":
        sentence_start -= 1
    while sentence_end < len(document_text) and document_text[sentence_end] not in ".!?\n":
        sentence_end += 1
    sentence_window = document_text[sentence_start:sentence_end].lower()
    if "century" in sentence_window or "centuries" in sentence_window:
        return True
    window_start = max(0, annotation.start_pos - 96)
    window_end = min(len(document_text), annotation.end_pos + 96)
    local_window = document_text[window_start:window_end].lower()
    return "century" in local_window or "centuries" in local_window


def _extract_number_value(text: str, attribute: str) -> float | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    scale_match = _NUMBER_SCALE_PATTERN.search(cleaned)
    if scale_match:
        scale_label = scale_match.group(1).lower().rstrip("s")
        scale = _NUMBER_SCALE_VALUES[scale_label]
        numeric_value = _extract_numeric_value(cleaned)
        if numeric_value is None:
            word_prefix = cleaned[: scale_match.start()].strip().lower().replace("-", " ")
            numeric_value = parse_word_number(word_prefix)
        if numeric_value is not None:
            return float(numeric_value) * float(scale)
    if attribute in {"int", "str"}:
        parsed_int = parse_integer_surface_number(cleaned)
        if parsed_int is not None:
            return float(parsed_int)
        parsed_word = parse_word_number(cleaned)
        if parsed_word is not None:
            return float(parsed_word)
    if attribute == "fraction":
        parsed_fraction = _extract_fraction_value(cleaned)
        if parsed_fraction is not None:
            return parsed_fraction
    return _extract_numeric_value(cleaned)


def _extract_fraction_value(text: str) -> float | None:
    cleaned = str(text or "").strip().lower().replace("-", " ")
    if not cleaned:
        return None
    fraction_match = _FRACTION_PATTERN.fullmatch(cleaned)
    if fraction_match:
        numerator = int(fraction_match.group(1))
        denominator = int(fraction_match.group(2))
        if denominator != 0:
            return numerator / denominator
    parts = cleaned.split()
    if len(parts) == 2:
        numerator = parse_word_number(parts[0])
        denominator_word = parts[1].rstrip("s")
        denominator = parse_word_number(denominator_word)
        if numerator is not None and denominator:
            return float(numerator) / float(denominator)
    return None


def _extract_numeric_value(text: str) -> float | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    match = _LEADING_NUMBER_PATTERN.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _extract_temporal_year(text: str, *, allow_numeric_fallback: bool = True) -> int | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    direct_year = _YEAR_PATTERN.search(cleaned)
    if direct_year:
        try:
            return int(direct_year.group(1))
        except ValueError:
            return None
    if not allow_numeric_fallback:
        return None
    numeric = _extract_numeric_value(cleaned)
    if numeric is None:
        return None
    return int(round(numeric))


def _extract_temporal_day_of_month(text: str) -> int | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    direct_numeric = _extract_numeric_value(cleaned)
    if (
        direct_numeric is not None
        and 1 <= int(round(direct_numeric)) <= 31
        and _YEAR_PATTERN.fullmatch(cleaned) is None
    ):
        return int(round(direct_numeric))
    dmy_match = _DMY_DATE_PATTERN.search(cleaned)
    if dmy_match:
        return int(dmy_match.group(1))
    mdy_match = _MDY_DATE_PATTERN.search(cleaned)
    if mdy_match:
        return int(mdy_match.group(1))
    return None
