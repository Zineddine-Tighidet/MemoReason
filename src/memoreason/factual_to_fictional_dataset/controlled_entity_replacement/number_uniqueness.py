"""Canonical numeric value used for inter-variant avoidance."""

from __future__ import annotations

from typing import Any

from .number_generation.value_strategy import NumberValueStrategyMixin


def number_entity_uniqueness_value(entity: Any) -> int | float | None:
    """Match the exported number-uniqueness normalization.

    Non-integer surfaces take precedence over their internal integer sampling
    coordinate.  Fractions use the denominator coordinate because that is what
    generation can avoid deterministically.
    """

    def field_value(field: str) -> Any:
        if isinstance(entity, dict):
            return entity.get(field)
        return getattr(entity, field, None)

    for field in ("percent", "proportion", "float"):
        value = field_value(field)
        if value is not None:
            try:
                return round(float(value), 6)
            except (TypeError, ValueError, OverflowError):
                return None
    if field_value("fraction") not in (None, ""):
        return NumberValueStrategyMixin._number_entity_int_value(entity)
    value = field_value("int")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    return NumberValueStrategyMixin._number_entity_int_value(entity)


def number_entity_uniqueness_payload(entity: Any) -> dict[str, Any] | None:
    """Return the canonical serialized identity used by the batch audit."""

    def field_value(field: str) -> Any:
        if isinstance(entity, dict):
            return entity.get(field)
        return getattr(entity, field, None)

    for field in ("percent", "proportion", "float"):
        value = field_value(field)
        if value is not None:
            try:
                return {"kind": field, "value": round(float(value), 6)}
            except (TypeError, ValueError, OverflowError):
                return None
    if field_value("fraction") not in (None, ""):
        denominator = NumberValueStrategyMixin._number_entity_int_value(entity)
        return None if denominator is None else {"kind": "fraction", "value": int(denominator)}
    value = field_value("int")
    if value is not None:
        try:
            return {"kind": "int", "value": int(value)}
        except (TypeError, ValueError, OverflowError):
            return None
    value = number_entity_uniqueness_value(entity)
    return None if value is None else {"kind": "str", "value": value}


__all__ = ["number_entity_uniqueness_payload", "number_entity_uniqueness_value"]
