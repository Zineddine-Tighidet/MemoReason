"""Fallback temporal-year shifting for constraints the year solver cannot model."""

from __future__ import annotations

import random

from memoreason.benchmark_definition.document_schema import EntityCollection


class TemporalShiftFallbackMixin:
    """Preserve factual chronology when explicit temporal solving is unavailable."""

    def _generate_shifted_required_years(
        self,
        required_temporals: list[tuple],
        existing_entities: EntityCollection | None,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
    ) -> dict[str, int] | None:
        factual_years: dict[str, int] = {}
        offset_low: int | None = None
        offset_high: int | None = None

        for temporal_id, attrs in required_temporals:
            if not any(attr in attrs for attr in ("year", "date")):
                continue
            factual_year = self._factual_temporal_year(temporal_id)
            if factual_year is None:
                continue
            factual_years[temporal_id] = factual_year
            base_lo, base_hi = self._temporal_year_base_range(temporal_id)
            local_low = base_lo - factual_year
            local_high = base_hi - factual_year
            offset_low = local_low if offset_low is None else max(offset_low, local_low)
            offset_high = local_high if offset_high is None else min(offset_high, local_high)

        if not factual_years or offset_low is None or offset_high is None or offset_low > offset_high:
            return None

        candidate_offsets = [offset for offset in range(offset_low, offset_high + 1) if offset != 0]
        random.shuffle(candidate_offsets)

        existing_temporal_years: dict[str, int] = {}
        if existing_entities and existing_entities.temporals:
            for existing_id, existing_temporal in existing_entities.temporals.items():
                if existing_id in factual_years:
                    continue
                existing_year = self._temporal_year_from_entity(existing_temporal)
                if existing_year is not None:
                    existing_temporal_years[existing_id] = existing_year

        valid_assignments: list[dict[str, int]] = []
        for offset in candidate_offsets:
            assignments: dict[str, int] = {}
            valid = True

            for temporal_id, factual_year in factual_years.items():
                shifted_year = factual_year + offset
                if not self._temporal_year_is_available(temporal_id, shifted_year, excluded_years):
                    valid = False
                    break
                if temporal_id in decade_year_temporal_ids and shifted_year % 10 != 0:
                    valid = False
                    break
                assignments[temporal_id] = shifted_year

            if not valid:
                continue

            for temporal_id, factual_year in factual_years.items():
                shifted_year = assignments[temporal_id]
                for existing_year in existing_temporal_years.values():
                    if factual_year < existing_year and not (shifted_year < existing_year):
                        valid = False
                        break
                    if factual_year > existing_year and not (shifted_year > existing_year):
                        valid = False
                        break
                    if factual_year == existing_year and shifted_year != existing_year:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                valid_assignments.append(assignments)

        if not valid_assignments:
            return None
        return random.choice(valid_assignments)


__all__ = ["TemporalShiftFallbackMixin"]
