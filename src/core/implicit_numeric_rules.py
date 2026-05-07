"""Auto-generated editable range rules for numeric and temporal sampling."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .document_schema import ImplicitRule
from .entity_taxonomy import parse_integer_surface_number, parse_word_number

NUMBER_RANGE_PERCENT = 20.0
AGE_RANGE_PERCENT = 20.0
TEMPORAL_RANGE_PERCENT = 1.0
CENTURY_RANGE_PERCENT = 10.0
SMALL_NUMBER_FIXED_WINDOW_THRESHOLD = 10
SMALL_NUMBER_FIXED_WINDOW_DELTA = 10
PREVIOUS_SMALL_NUMBER_FIXED_WINDOW_DELTAS: tuple[int, ...] = (3,)
SMALL_NUMBER_MIN_VALUE = 1
IMPLICIT_RULE_PRECISION = 2
IMPLICIT_RULE_YEAR_UPPER_BOUND = 2026

_ANNOTATION_PATTERN = re.compile(r"\[([^\]]+);\s*([^\]]+)\]")
_YEAR_PATTERN = re.compile(r"\b(\d{4})\b")
_DMY_DATE_PATTERN = re.compile(r"\b(\d{1,2})\s+[A-Za-z]+\s+\d{4}\b")
_MDY_DATE_PATTERN = re.compile(r"\b[A-Za-z]+\s+(\d{1,2}),?\s+\d{4}\b")
_LEADING_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FRACTION_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


@dataclass(frozen=True)
class _AnnotationSpan:
    start_pos: int
    end_pos: int
    original_text: str
    entity_id: str
    attribute: str | None

    @property
    def entity_ref(self) -> str:
        if self.attribute:
            return f"{self.entity_id}.{self.attribute}"
        return self.entity_id


def _format_percentage(percentage: float) -> str:
    if float(percentage).is_integer():
        return f"{int(percentage)}%"
    return f"{percentage:.2f}%"


def _entity_ref_parts(entity_ref: str) -> tuple[str, str]:
    entity_id, _, attribute = str(entity_ref or "").strip().partition(".")
    return entity_id, attribute


def _relative_small_integer_window(
    center: int,
    *,
    ratio: float,
    min_delta: int,
    small_value_threshold: int,
    small_value_delta: int,
    min_value: int,
) -> tuple[int, int]:
    delta = max(int(min_delta), int(math.ceil(abs(center) * ratio)))
    if abs(center) < int(small_value_threshold):
        delta = max(delta, int(small_value_delta))
    low = max(int(min_value), int(center) - delta)
    high = max(int(min_value), int(center) + delta)
    if low > high:
        low = high = max(int(min_value), int(center))
    if low == high:
        high += 1
    return int(low), int(high)


def implicit_rule_uses_integer_bounds(
    rule: ImplicitRule | dict[str, Any] | None = None,
    *,
    entity_ref: str | None = None,
    rule_kind: str | None = None,
) -> bool:
    payload: dict[str, Any] = {}
    if isinstance(rule, ImplicitRule):
        payload = rule.model_dump()
    elif isinstance(rule, dict):
        payload = rule

    resolved_entity_ref = str(payload.get("entity_ref") or entity_ref or "").strip()
    resolved_rule_kind = str(payload.get("rule_kind") or rule_kind or "").strip()
    entity_id, _, attribute = resolved_entity_ref.partition(".")

    if attribute in {"age", "year", "day_of_month"}:
        return True
    if entity_id.startswith("number_") and attribute in {"int", "str"}:
        return True
    if resolved_rule_kind == "century_range":
        return True
    return False


def implicit_rule_uses_small_number_fixed_window(
    rule: ImplicitRule | dict[str, Any] | None = None,
    *,
    entity_ref: str | None = None,
    rule_kind: str | None = None,
    factual_value: Any = None,
) -> bool:
    payload: dict[str, Any] = {}
    if isinstance(rule, ImplicitRule):
        payload = rule.model_dump()
    elif isinstance(rule, dict):
        payload = rule

    resolved_entity_ref = str(payload.get("entity_ref") or entity_ref or "").strip()
    resolved_rule_kind = str(payload.get("rule_kind") or rule_kind or "").strip()
    raw_factual_value = payload.get("factual_value") if "factual_value" in payload else factual_value
    entity_id, attribute = _entity_ref_parts(resolved_entity_ref)
    if resolved_rule_kind != "number_range":
        return False
    if not entity_id.startswith("number_") or attribute not in {"int", "str", "float", "percent", "proportion"}:
        return False
    try:
        numeric_value = abs(float(raw_factual_value))
    except (TypeError, ValueError):
        return False
    return numeric_value < float(SMALL_NUMBER_FIXED_WINDOW_THRESHOLD)


def implicit_rule_has_year_cap(
    rule: ImplicitRule | dict[str, Any] | None = None,
    *,
    entity_ref: str | None = None,
    rule_kind: str | None = None,
) -> bool:
    payload: dict[str, Any] = {}
    if isinstance(rule, ImplicitRule):
        payload = rule.model_dump()
    elif isinstance(rule, dict):
        payload = rule

    resolved_entity_ref = str(payload.get("entity_ref") or entity_ref or "").strip()
    resolved_rule_kind = str(payload.get("rule_kind") or rule_kind or "").strip()
    return resolved_entity_ref.endswith(".year") or resolved_rule_kind == "temporal_year_range"


def _format_implicit_bound(value: float, *, integer_like: bool) -> str:
    if integer_like:
        return str(int(round(float(value))))
    return f"{float(value):.{IMPLICIT_RULE_PRECISION}f}"


def _normalize_implicit_numeric_value(value: Any, *, integer_like: bool) -> int | float:
    numeric_value = float(value)
    if integer_like:
        return int(round(numeric_value))
    return round(numeric_value, IMPLICIT_RULE_PRECISION)


def _normalize_implicit_bound(
    value: Any,
    *,
    integer_like: bool,
    bound_kind: str,
) -> int | float:
    numeric_value = float(value)
    if integer_like:
        if bound_kind == "lower_bound":
            return int(math.ceil(numeric_value))
        return int(math.floor(numeric_value))
    return round(numeric_value, IMPLICIT_RULE_PRECISION)


def format_implicit_rule_expression(rule: ImplicitRule | dict[str, Any]) -> str:
    payload = rule if isinstance(rule, dict) else rule.model_dump()
    entity_ref = str(payload.get("entity_ref") or "").strip()
    integer_like = implicit_rule_uses_integer_bounds(payload)
    lower = _normalize_implicit_bound(
        payload.get("lower_bound") or 0.0,
        integer_like=integer_like,
        bound_kind="lower_bound",
    )
    upper = _normalize_implicit_bound(
        payload.get("upper_bound") or 0.0,
        integer_like=integer_like,
        bound_kind="upper_bound",
    )
    ordered_lower = min(lower, upper)
    ordered_upper = max(lower, upper)
    return (
        f"{entity_ref} ∈ ["
        f"{_format_implicit_bound(ordered_lower, integer_like=integer_like)}, "
        f"{_format_implicit_bound(ordered_upper, integer_like=integer_like)}]"
    )


def format_implicit_rule_explanation(rule: ImplicitRule | dict[str, Any]) -> str:
    payload = rule if isinstance(rule, dict) else rule.model_dump()
    percentage = float(payload.get("percentage") or 0.0)
    if implicit_rule_uses_small_number_fixed_window(payload):
        return (
            "This rule was generated following the interval of factual value +/- "
            f"{SMALL_NUMBER_FIXED_WINDOW_DELTA}, "
            "clamped to the valid domain when needed."
        )
    explanation = (
        f"This rule was generated following the range of "
        f"{_format_percentage(percentage)} around factual value."
    )
    if implicit_rule_has_year_cap(payload):
        explanation = (
            f"This rule was generated following the range of "
            f"{_format_percentage(percentage)} around factual value with an upper bound at "
            f"{IMPLICIT_RULE_YEAR_UPPER_BOUND}."
        )
    return explanation


def _apply_implicit_year_upper_cap(
    entity_ref: str,
    rule_kind: str,
    lower_bound: int | float,
    upper_bound: int | float,
    factual_value: int | float,
) -> tuple[int | float, int | float]:
    if not implicit_rule_has_year_cap(entity_ref=entity_ref, rule_kind=rule_kind):
        return lower_bound, upper_bound
    capped_upper = min(float(upper_bound), float(IMPLICIT_RULE_YEAR_UPPER_BOUND))
    integer_like = implicit_rule_uses_integer_bounds(entity_ref=entity_ref, rule_kind=rule_kind)
    normalized_upper = _normalize_implicit_bound(
        capped_upper,
        integer_like=integer_like,
        bound_kind="upper_bound",
    )
    if lower_bound > normalized_upper:
        factual_bound = _normalize_implicit_numeric_value(factual_value, integer_like=integer_like)
        capped_factual = min(float(factual_bound), float(IMPLICIT_RULE_YEAR_UPPER_BOUND))
        fallback = _normalize_implicit_bound(
            capped_factual,
            integer_like=integer_like,
            bound_kind="upper_bound",
        )
        return fallback, fallback
    return lower_bound, normalized_upper


def normalize_implicit_rules_for_storage(raw_rules: Any) -> list[dict[str, Any]] | None:
    """Normalize persisted implicit rules to a stable list-of-dicts format."""
    if raw_rules is None:
        return None
    if not isinstance(raw_rules, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_entry in raw_rules:
        if isinstance(raw_entry, ImplicitRule):
            raw_entry = raw_entry.model_dump()
        if not isinstance(raw_entry, dict):
            continue
        entity_ref = str(raw_entry.get("entity_ref") or "").strip()
        if not entity_ref:
            continue
        rule_kind = str(raw_entry.get("rule_kind") or "").strip() or "number_range"
        integer_like = implicit_rule_uses_integer_bounds(
            entity_ref=entity_ref,
            rule_kind=rule_kind,
        )
        try:
            raw_lower = float(raw_entry.get("lower_bound"))
            raw_upper = float(raw_entry.get("upper_bound"))
            ordered_lower = min(raw_lower, raw_upper)
            ordered_upper = max(raw_lower, raw_upper)
            lower_bound = _normalize_implicit_bound(
                ordered_lower,
                integer_like=integer_like,
                bound_kind="lower_bound",
            )
            upper_bound = _normalize_implicit_bound(
                ordered_upper,
                integer_like=integer_like,
                bound_kind="upper_bound",
            )
            factual_value = _normalize_implicit_numeric_value(
                raw_entry.get("factual_value"),
                integer_like=integer_like,
            )
            percentage = round(float(raw_entry.get("percentage")), IMPLICIT_RULE_PRECISION)
        except (TypeError, ValueError):
            continue
        lower_bound, upper_bound = _apply_implicit_year_upper_cap(
            entity_ref,
            rule_kind,
            lower_bound,
            upper_bound,
            factual_value,
        )
        if lower_bound > upper_bound:
            lower_bound = upper_bound = factual_value
        normalized.append(
            {
                "entity_ref": entity_ref,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "factual_value": factual_value,
                "percentage": percentage,
                "rule_kind": rule_kind,
            }
        )
    return normalized


def normalize_implicit_rule_exclusions(raw_exclusions: Any) -> list[str]:
    """Normalize persisted implicit-rule exclusions to a stable unique list."""
    if raw_exclusions is None:
        return []
    if not isinstance(raw_exclusions, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_entry in raw_exclusions:
        entity_ref = str(raw_entry or "").strip()
        if not entity_ref or entity_ref in seen:
            continue
        normalized.append(entity_ref)
        seen.add(entity_ref)
    return normalized


def ensure_document_implicit_rules(doc_data: dict[str, Any]) -> dict[str, Any]:
    """Return a document payload with generated/normalized implicit rules."""
    if not isinstance(doc_data, dict):
        return doc_data
    merged = dict(doc_data)
    defaults = normalize_implicit_rules_for_storage(generate_implicit_rules_for_document(merged)) or []
    legacy_defaults = normalize_implicit_rules_for_storage(
        generate_implicit_rules_for_document(merged, use_small_number_fixed_window=False)
    ) or []
    prior_small_window_defaults = [
        normalize_implicit_rules_for_storage(
            generate_implicit_rules_for_document(
                merged,
                small_number_fixed_window_delta=previous_delta,
            )
        ) or []
        for previous_delta in PREVIOUS_SMALL_NUMBER_FIXED_WINDOW_DELTAS
    ]
    existing = normalize_implicit_rules_for_storage(merged.get("implicit_rules")) or []
    default_entity_refs = {
        str(entry.get("entity_ref") or "").strip()
        for entry in defaults
        if str(entry.get("entity_ref") or "").strip()
    }
    excluded_entity_refs = [
        entity_ref
        for entity_ref in normalize_implicit_rule_exclusions(merged.get("implicit_rule_exclusions"))
        if entity_ref in default_entity_refs
    ]
    excluded_entity_ref_set = set(excluded_entity_refs)
    existing_by_ref = {
        str(entry.get("entity_ref") or "").strip(): entry
        for entry in existing
        if str(entry.get("entity_ref") or "").strip()
    }
    legacy_by_ref = {
        str(entry.get("entity_ref") or "").strip(): entry
        for entry in legacy_defaults
        if str(entry.get("entity_ref") or "").strip()
    }
    prior_small_window_by_ref = [
        {
            str(entry.get("entity_ref") or "").strip(): entry
            for entry in historical_defaults
            if str(entry.get("entity_ref") or "").strip()
        }
        for historical_defaults in prior_small_window_defaults
    ]

    merged_rules: list[dict[str, Any]] = []
    for default_rule in defaults:
        entity_ref = str(default_rule["entity_ref"])
        if entity_ref in excluded_entity_ref_set:
            continue
        existing_rule = existing_by_ref.get(entity_ref)
        if existing_rule is None:
            merged_rules.append(default_rule)
            continue
        legacy_rule = legacy_by_ref.get(entity_ref)
        if _should_refresh_implicit_rule_from_legacy_default(existing_rule, legacy_rule):
            merged_rules.append(default_rule)
            continue
        if any(
            _should_refresh_implicit_rule_from_legacy_default(
                existing_rule,
                historical_by_ref.get(entity_ref),
            )
            for historical_by_ref in prior_small_window_by_ref
        ):
            merged_rules.append(default_rule)
            continue
        merged_rules.append(
            normalize_implicit_rules_for_storage(
                [
                    {
                        **default_rule,
                        "lower_bound": existing_rule.get("lower_bound", default_rule["lower_bound"]),
                        "upper_bound": existing_rule.get("upper_bound", default_rule["upper_bound"]),
                    }
                ]
            )[0]
        )

    merged["implicit_rules"] = merged_rules
    if excluded_entity_refs:
        merged["implicit_rule_exclusions"] = excluded_entity_refs
    else:
        merged.pop("implicit_rule_exclusions", None)
    return merged


def implicit_rule_bounds_lookup(raw_rules: list[ImplicitRule] | list[dict[str, Any]] | None) -> dict[str, ImplicitRule]:
    """Build a normalized entity-ref keyed lookup."""
    lookup: dict[str, ImplicitRule] = {}
    for entry in normalize_implicit_rules_for_storage(raw_rules) or []:
        rule = ImplicitRule(**entry)
        lookup[rule.entity_ref] = rule
    return lookup


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
        entity_id, attribute = (
            entity_ref.split(".", 1) if "." in entity_ref else (entity_ref, None)
        )
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
    if annotation.entity_id.startswith("number_") and attribute in {"int", "str", "float", "percent", "proportion", "fraction"}:
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
        percentage = CENTURY_RANGE_PERCENT if _is_century_annotation(document_text, annotation) else NUMBER_RANGE_PERCENT
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
            factual_year = _extract_temporal_year(annotation.original_text)
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


def _same_implicit_numeric_value(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), abs_tol=10 ** (-IMPLICIT_RULE_PRECISION))
    except (TypeError, ValueError):
        return False


def _should_refresh_implicit_rule_from_legacy_default(
    existing_rule: dict[str, Any],
    legacy_rule: dict[str, Any] | None,
) -> bool:
    if legacy_rule is None:
        return False
    return (
        str(existing_rule.get("entity_ref") or "").strip() == str(legacy_rule.get("entity_ref") or "").strip()
        and str(existing_rule.get("rule_kind") or "").strip() == str(legacy_rule.get("rule_kind") or "").strip()
        and _same_implicit_numeric_value(existing_rule.get("factual_value"), legacy_rule.get("factual_value"))
        and _same_implicit_numeric_value(existing_rule.get("lower_bound"), legacy_rule.get("lower_bound"))
        and _same_implicit_numeric_value(existing_rule.get("upper_bound"), legacy_rule.get("upper_bound"))
    )


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


def _extract_temporal_year(text: str) -> int | None:
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    direct_year = _YEAR_PATTERN.search(cleaned)
    if direct_year:
        try:
            return int(direct_year.group(1))
        except ValueError:
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
    if direct_numeric is not None and 1 <= int(round(direct_numeric)) <= 31 and _YEAR_PATTERN.fullmatch(cleaned) is None:
        return int(round(direct_numeric))
    dmy_match = _DMY_DATE_PATTERN.search(cleaned)
    if dmy_match:
        return int(dmy_match.group(1))
    mdy_match = _MDY_DATE_PATTERN.search(cleaned)
    if mdy_match:
        return int(mdy_match.group(1))
    return None


__all__ = [
    "AGE_RANGE_PERCENT",
    "CENTURY_RANGE_PERCENT",
    "IMPLICIT_RULE_PRECISION",
    "NUMBER_RANGE_PERCENT",
    "TEMPORAL_RANGE_PERCENT",
    "ensure_document_implicit_rules",
    "format_implicit_rule_explanation",
    "format_implicit_rule_expression",
    "generate_implicit_rules_for_document",
    "implicit_rule_bounds_lookup",
    "implicit_rule_has_year_cap",
    "implicit_rule_uses_integer_bounds",
    "IMPLICIT_RULE_YEAR_UPPER_BOUND",
    "normalize_implicit_rule_exclusions",
    "normalize_implicit_rules_for_storage",
]
