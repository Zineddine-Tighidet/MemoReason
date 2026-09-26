"""Normalize and merge stored implicit rules with generated defaults."""

from __future__ import annotations

import math
from typing import Any

from .document_schema import ImplicitRule
from .implicit_rule_formatting import (
    IMPLICIT_RULE_PRECISION,
    PREVIOUS_SMALL_NUMBER_FIXED_WINDOW_DELTAS,
    _apply_implicit_year_upper_cap,
    _normalize_implicit_bound,
    _normalize_implicit_numeric_value,
    implicit_rule_uses_integer_bounds,
)
from .implicit_rule_generation import generate_implicit_rules_for_document


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
    legacy_defaults = (
        normalize_implicit_rules_for_storage(
            generate_implicit_rules_for_document(merged, use_small_number_fixed_window=False)
        )
        or []
    )
    prior_small_window_defaults = [
        normalize_implicit_rules_for_storage(
            generate_implicit_rules_for_document(
                merged,
                small_number_fixed_window_delta=previous_delta,
            )
        )
        or []
        for previous_delta in PREVIOUS_SMALL_NUMBER_FIXED_WINDOW_DELTAS
    ]
    existing = normalize_implicit_rules_for_storage(merged.get("implicit_rules")) or []
    default_entity_refs = {
        str(entry.get("entity_ref") or "").strip() for entry in defaults if str(entry.get("entity_ref") or "").strip()
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
