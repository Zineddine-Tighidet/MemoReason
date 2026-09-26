"""Public API for MemoReason implicit numeric and temporal constraints.

Formatting, persistence normalization, and annotation-driven generation remain
in responsibility-specific modules behind this cohesive import boundary.
"""

from .implicit_rule_formatting import (
    format_implicit_rule_explanation,
    format_implicit_rule_expression,
    implicit_rule_has_year_cap,
    implicit_rule_uses_integer_bounds,
    implicit_rule_uses_small_number_fixed_window,
)
from .implicit_rule_generation import generate_implicit_rules_for_document
from .implicit_rule_normalization import (
    ensure_document_implicit_rules,
    implicit_rule_bounds_lookup,
    normalize_implicit_rule_exclusions,
    normalize_implicit_rules_for_storage,
)

__all__ = [
    "ensure_document_implicit_rules",
    "format_implicit_rule_explanation",
    "format_implicit_rule_expression",
    "generate_implicit_rules_for_document",
    "implicit_rule_bounds_lookup",
    "implicit_rule_has_year_cap",
    "implicit_rule_uses_integer_bounds",
    "implicit_rule_uses_small_number_fixed_window",
    "normalize_implicit_rule_exclusions",
    "normalize_implicit_rules_for_storage",
]
