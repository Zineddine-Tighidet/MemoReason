"""Build deterministic temporal exclusion maps for variant sampling."""

from __future__ import annotations

from typing import Any

from memoreason.benchmark_definition.document_schema import EntityCollection


def build_temporal_exclusions(
    *,
    factual_entities: EntityCollection | None,
    used_years_by_id: dict[str, set[int]],
    used_values_by_id: dict[str, dict[str, set[Any]]],
) -> dict[str, Any]:
    exclusions: dict[str, Any] = {}
    if factual_entities and factual_entities.temporals:
        temporals = factual_entities.temporals
        exclusions = {
            "days": {item.day for item in temporals.values() if item.day},
            "months": {item.month for item in temporals.values() if item.month},
            "years": {item.year for item in temporals.values() if item.year is not None},
            "day_of_months": {int(item.day_of_month) for item in temporals.values() if item.day_of_month is not None},
            "days_by_id": {key: {item.day} for key, item in temporals.items() if item.day},
            "months_by_id": {key: {item.month} for key, item in temporals.items() if item.month},
            "day_of_months_by_id": {
                key: {int(item.day_of_month)} for key, item in temporals.items() if item.day_of_month is not None
            },
        }
    if used_years_by_id:
        exclusions["years_by_id"] = {
            str(key): {int(year) for year in years if year is not None}
            for key, years in used_years_by_id.items()
            if years
        }
    for attr, exclusion_key in (
        ("year", "years_by_id"),
        ("month", "months_by_id"),
        ("day", "days_by_id"),
        ("day_of_month", "day_of_months_by_id"),
        ("timestamp", "timestamps_by_id"),
    ):
        values = {
            str(key): {value for value in attrs.get(attr, set()) if value not in (None, "")}
            for key, attrs in used_values_by_id.items()
            if isinstance(attrs, dict)
        }
        for temporal_id, observed in values.items():
            if observed:
                exclusions.setdefault(exclusion_key, {}).setdefault(temporal_id, set()).update(observed)
    return exclusions


__all__ = ["build_temporal_exclusions"]
