"""Auto-generated editable range rules for numeric and temporal sampling."""

# Preserve the exact integer coercions used by the frozen dataset generator.
# ruff: noqa: RUF046

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .document_schema import ImplicitRule

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
    is_year_rule = resolved_entity_ref.endswith(".year") or resolved_rule_kind == "temporal_year_range"
    if not is_year_rule:
        return False
    if payload:
        try:
            if float(payload.get("factual_value")) > float(IMPLICIT_RULE_YEAR_UPPER_BOUND):
                return False
        except (TypeError, ValueError):
            pass
    return True


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
        f"This rule was generated following the range of {_format_percentage(percentage)} around factual value."
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
    if float(factual_value) > float(IMPLICIT_RULE_YEAR_UPPER_BOUND):
        # Future factual dates need a future sampling window. This mirrors the
        # temporal generator, which caps only non-future or missing years.
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
