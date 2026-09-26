"""Parse, bound, and construct temporal date and year values."""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from memoreason.benchmark_definition.document_schema import EntityCollection, TemporalEntity
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

from ..generation_limits import (
    _CURRENT_YEAR_TEMPORAL_CAP,
    _FUTURE_DATE_SAFETY_MAX_YEAR,
    _MIN_YEAR,
    _TEMPORAL_RELATIVE_RANGE_MIN_DELTA,
    _TEMPORAL_RELATIVE_RANGE_RATIO,
    _TEMPORAL_YEAR_RELATIVE_RANGE_HARD_CAP,
    _relative_int_window,
)

_MIN_DISTINCT_YEARS_PER_SERIES = 20


class TemporalDateValueMixin:
    """Convert and sample concrete dates while respecting excluded values."""

    def _factual_temporal_day_of_month(self, temporal_id: str) -> int | None:
        if not self.factual_entities or not self.factual_entities.temporals:
            return None
        factual_temporal = self.factual_entities.temporals.get(temporal_id)
        if factual_temporal is None:
            return None
        raw_day = (
            getattr(factual_temporal, "day_of_month", None)
            if not isinstance(factual_temporal, dict)
            else factual_temporal.get("day_of_month")
        )
        if raw_day is not None:
            try:
                day = int(raw_day)
                if 1 <= day <= 31:
                    return day
            except (TypeError, ValueError):
                pass
        date_value = (
            getattr(factual_temporal, "date", None)
            if not isinstance(factual_temporal, dict)
            else factual_temporal.get("date")
        )
        if not isinstance(date_value, str):
            return None
        iso_match = re.search(r"\b\d{4}-(\d{1,2})-(\d{1,2})\b", date_value)
        if iso_match:
            return int(iso_match.group(2))
        dmy_match = re.search(r"\b(\d{1,2})\s+[A-Za-z]+\s+\d{4}\b", date_value)
        if dmy_match:
            return int(dmy_match.group(1))
        mdy_match = re.search(r"\b[A-Za-z]+\s+(\d{1,2}),?\s+\d{4}\b", date_value)
        if mdy_match:
            return int(mdy_match.group(1))
        return None

    def _temporal_day_of_month_base_range(self, temporal_id: str) -> tuple[int, int]:
        del temporal_id
        return 1, 28

    def _month_name_to_number(self, month_name: str | None) -> int | None:
        if not isinstance(month_name, str):
            return None
        normalized = month_name.strip()
        if not normalized:
            return None
        for idx, candidate in enumerate(self._MONTHS, start=1):
            if candidate.lower() == normalized.lower():
                return idx
        return None

    def _month_number_to_name(self, month_number: int) -> str:
        return self._MONTHS[month_number - 1]

    def _temporal_entity_to_date(self, temporal: Any) -> date | None:
        if temporal is None:
            return None
        getter = (
            temporal.get if isinstance(temporal, dict) else lambda attr, default=None: getattr(temporal, attr, default)
        )

        raw_year = getter("year", None)
        raw_month = getter("month", None)
        raw_day = getter("day_of_month", None)
        try:
            year = int(raw_year) if raw_year is not None else None
            day_of_month = int(raw_day) if raw_day is not None else None
        except (TypeError, ValueError):
            year = None
            day_of_month = None
        month_number = self._month_name_to_number(raw_month)
        if year is not None and month_number is not None and day_of_month is not None:
            try:
                return date(year, month_number, day_of_month)
            except ValueError:
                return None

        raw_date = getter("date", None)
        if not isinstance(raw_date, str):
            return None

        iso_match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw_date.strip())
        if iso_match:
            try:
                return date(int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)))
            except ValueError:
                return None

        dmy_match = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", raw_date.strip())
        if dmy_match:
            month_number = self._month_name_to_number(dmy_match.group(2))
            if month_number is None:
                return None
            try:
                return date(int(dmy_match.group(3)), month_number, int(dmy_match.group(1)))
            except ValueError:
                return None
        return None

    def _set_temporal_entity_from_date(self, temporal: TemporalEntity, value: date) -> None:
        had_weekday = temporal.day is not None
        temporal.year = value.year
        temporal.month = self._month_number_to_name(value.month)
        temporal.day_of_month = value.day
        temporal.date = f"{value.day} {temporal.month} {value.year}"
        if had_weekday:
            temporal.day = value.strftime("%A")

    def _excluded_day_of_month_values(self, temporal_id: str) -> set[int]:
        excluded = {int(value) for value in (self.exclude_temporals.get("day_of_months") or set()) if value is not None}
        by_id = self.exclude_temporals.get("day_of_months_by_id") or {}
        excluded.update(int(value) for value in (by_id.get(temporal_id) or set()) if value is not None)
        return {value for value in excluded if 1 <= value <= 31}

    def _date_avoids_excluded_components(
        self,
        temporal_id: str,
        value: date,
        *,
        allow_soft_global: bool,
    ) -> bool:
        month_name = self._month_number_to_name(value.month)
        weekday = value.strftime("%A")
        hard_months = set((self.exclude_temporals.get("months_by_id") or {}).get(temporal_id) or set())
        hard_days = set((self.exclude_temporals.get("days_by_id") or {}).get(temporal_id) or set())
        hard_day_numbers = {
            int(candidate)
            for candidate in ((self.exclude_temporals.get("day_of_months_by_id") or {}).get(temporal_id) or set())
            if candidate is not None
        }
        if month_name in hard_months or weekday in hard_days or int(value.day) in hard_day_numbers:
            return False
        if allow_soft_global:
            return True
        soft_months = set(self.exclude_temporals.get("months") or set())
        soft_days = set(self.exclude_temporals.get("days") or set())
        soft_day_numbers = {
            int(candidate)
            for candidate in (self.exclude_temporals.get("day_of_months") or set())
            if candidate is not None
        }
        return month_name not in soft_months and weekday not in soft_days and int(value.day) not in soft_day_numbers

    def _date_avoids_excluded_day(self, temporal_id: str, value: date) -> bool:
        """Compatibility wrapper using the complete hard per-ID contract."""
        return self._date_avoids_excluded_components(temporal_id, value, allow_soft_global=True)

    def _date_pair_with_delta_avoiding_excluded_days(
        self,
        *,
        left_id: str,
        right_id: str,
        right_date: date,
        days_delta: int,
        linked_bounds: list[tuple[str, int, int, int, int]] | None = None,
    ) -> tuple[date, date]:
        target_date = right_date + timedelta(days=days_delta)
        right_year_start = date(right_date.year, 1, 1)
        right_year_end = date(right_date.year + 1, 1, 1)
        span = (right_year_end - right_year_start).days

        # Search the complete finite year, preserving both years selected by
        # the temporal solver.  Prefer a shared month: this keeps per-ID month
        # domains aligned across a multi-variant batch instead of greedily
        # consuming incompatible month permutations.
        for allow_soft_global in (False, True):
            feasible_pairs: list[tuple[date, date, tuple[date, ...]]] = []
            for day_offset in range(span):
                candidate_right = right_year_start + timedelta(days=day_offset)
                candidate_left = candidate_right + timedelta(days=days_delta)
                if candidate_left.year != target_date.year:
                    continue
                if not self._date_avoids_excluded_components(
                    right_id,
                    candidate_right,
                    allow_soft_global=allow_soft_global,
                ):
                    continue
                if not self._date_avoids_excluded_components(
                    left_id,
                    candidate_left,
                    allow_soft_global=allow_soft_global,
                ):
                    continue
                linked_candidates = tuple(
                    self._bounded_date_candidate_for_year(
                        temporal_id=bound_id,
                        right_date=candidate_right,
                        target_year=target_year,
                        preferred_delta=preferred_delta,
                        low=low,
                        high=high,
                        allow_soft_global=allow_soft_global,
                    )
                    for bound_id, target_year, preferred_delta, low, high in (linked_bounds or [])
                )
                if any(candidate is None for candidate in linked_candidates):
                    continue
                feasible_pairs.append(
                    (
                        candidate_right,
                        candidate_left,
                        tuple(candidate for candidate in linked_candidates if candidate is not None),
                    )
                )
            if feasible_pairs:
                variant_slot = int(getattr(self, "reference_variant_index", 0) or 0)
                preferred_month = (variant_slot % 12) + 1

                def cyclic_month_distance(month: int, target_month: int = preferred_month) -> int:
                    direct = abs(int(month) - target_month)
                    return min(direct, 12 - direct)

                best_right, best_left, _linked = min(
                    feasible_pairs,
                    key=lambda pair: (
                        cyclic_month_distance(pair[0].month),
                        any(bound_date.month != pair[0].month for bound_date in pair[2]),
                        pair[0].month != pair[1].month,
                        abs((pair[0] - right_date).days),
                        pair[0],
                    ),
                )
                return best_right, best_left

        return right_date, target_date

    def _bounded_date_candidate_for_year(
        self,
        *,
        temporal_id: str,
        right_date: date,
        target_year: int,
        preferred_delta: int,
        low: int,
        high: int,
        allow_soft_global: bool,
    ) -> date | None:
        year_start = date(int(target_year), 1, 1)
        year_end = date(int(target_year) + 1, 1, 1)
        candidates: list[tuple[int, date]] = []
        for day_offset in range((year_end - year_start).days):
            candidate_date = year_start + timedelta(days=day_offset)
            candidate_delta = (candidate_date - right_date).days
            if candidate_delta < low or candidate_delta > high:
                continue
            if not self._date_avoids_excluded_components(
                temporal_id,
                candidate_date,
                allow_soft_global=allow_soft_global,
            ):
                continue
            candidates.append((candidate_delta, candidate_date))
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (
                item[1].month != right_date.month,
                item[1].month == right_date.month and item[1].day >= right_date.day,
                abs(item[0] - preferred_delta),
                item[1],
            ),
        )[1]

    def _bounded_date_delta_avoiding_excluded_day(
        self,
        *,
        temporal_id: str,
        right_date: date,
        preferred_delta: int,
        low: int,
        high: int,
    ) -> int:
        preferred_delta = max(low, min(high, preferred_delta))
        preferred_date = right_date + timedelta(days=preferred_delta)

        # First preserve the year chosen by the exact temporal solver and align
        # the month with the right-hand date.  For a positive cross-year bound
        # below one year, an earlier day in that same month is the constructive
        # ordering-preserving choice.
        for allow_soft_global in (False, True):
            candidate_date = self._bounded_date_candidate_for_year(
                temporal_id=temporal_id,
                right_date=right_date,
                target_year=preferred_date.year,
                preferred_delta=preferred_delta,
                low=low,
                high=high,
                allow_soft_global=allow_soft_global,
            )
            if candidate_date is not None:
                return (candidate_date - right_date).days

        max_radius = max(abs(preferred_delta - low), abs(high - preferred_delta), 366)
        for preserve_target_year in (True, False):
            for allow_soft_global in (False, True):
                for radius in range(max_radius + 1):
                    candidates = (
                        (preferred_delta,)
                        if radius == 0
                        else (preferred_delta - radius, preferred_delta + radius)
                    )
                    for candidate_delta in candidates:
                        if candidate_delta < low or candidate_delta > high:
                            continue
                        candidate_date = right_date + timedelta(days=candidate_delta)
                        if preserve_target_year and candidate_date.year != preferred_date.year:
                            continue
                        if self._date_avoids_excluded_components(
                            temporal_id,
                            candidate_date,
                            allow_soft_global=allow_soft_global,
                        ):
                            return candidate_delta
        return preferred_delta

    def _number_days_value(
        self, existing_entities: EntityCollection | None, number_id: str, attribute: str
    ) -> int | None:
        if existing_entities is None:
            return None
        number = existing_entities.numbers.get(number_id)
        if number is None:
            return None
        if attribute == "int":
            raw_value = number.int
            if raw_value is None:
                return None
            try:
                return int(raw_value)
            except (TypeError, ValueError):
                return None
        if attribute == "str":
            if number.int is not None:
                try:
                    return int(number.int)
                except (TypeError, ValueError):
                    return None
            if isinstance(number.str, str):
                parsed = parse_word_number(number.str)
                if parsed is not None:
                    return int(parsed)
                parsed = parse_integer_surface_number(number.str)
                if parsed is not None:
                    return int(parsed)
        return None


class TemporalYearDomainMixin:
    """Build feasible temporal-year domains around factual values."""

    def _temporal_year_base_range(self, temporal_id: str) -> tuple[int, int]:
        _, max_year = self._temporal_year_sampling_bounds(temporal_id)
        implicit_range = getattr(self, "_implicit_temporal_range", lambda _temporal_id, _attribute: None)(
            temporal_id,
            "year",
        )
        if implicit_range is not None:
            low, high = implicit_range
            low = max(_MIN_YEAR, low)
            high = min(high, max_year)
            if low > high:
                low = high = max(_MIN_YEAR, min(high, max_year))
            return int(low), int(high)
        factual_year = self._factual_temporal_year(temporal_id)
        if factual_year is None:
            return _MIN_YEAR, max_year
        low, high = _relative_int_window(
            factual_year,
            ratio=_TEMPORAL_RELATIVE_RANGE_RATIO,
            min_delta=_TEMPORAL_RELATIVE_RANGE_MIN_DELTA,
            max_delta=_TEMPORAL_YEAR_RELATIVE_RANGE_HARD_CAP,
            min_value=_MIN_YEAR,
            max_value=max_year,
        )
        return self._ensure_minimum_year_domain_width(low, high, max_year=max_year)

    def _temporal_year_sampling_bounds(self, temporal_id: str) -> tuple[int, int]:
        factual_year = self._factual_temporal_year(temporal_id)
        if factual_year is not None and factual_year > _CURRENT_YEAR_TEMPORAL_CAP:
            return _MIN_YEAR, _FUTURE_DATE_SAFETY_MAX_YEAR
        return _MIN_YEAR, _CURRENT_YEAR_TEMPORAL_CAP

    def _temporal_year_is_available(
        self,
        temporal_id: str,
        year: int,
        excluded_years: set[int],
    ) -> bool:
        years_by_id = self.exclude_temporals.get("years_by_id", {})
        forbidden_years = {
            int(candidate) for candidate in (years_by_id.get(temporal_id) or set()) if candidate is not None
        }
        factual_year = self._factual_temporal_year(temporal_id)
        if factual_year is not None:
            return int(year) != int(factual_year) and int(year) not in forbidden_years
        return int(year) not in excluded_years and int(year) not in forbidden_years

    def _temporal_year_domain(
        self,
        temporal_id: str,
        low: int,
        high: int,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
    ) -> list[int]:
        values = [
            year for year in range(low, high + 1) if self._temporal_year_is_available(temporal_id, year, excluded_years)
        ]
        if temporal_id in decade_year_temporal_ids:
            values = [year for year in values if year % 10 == 0]
        return values

    def _expand_temporal_year_domain(
        self,
        temporal_id: str,
        low: int,
        high: int,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
    ) -> list[int]:
        values = self._temporal_year_domain(
            temporal_id,
            low,
            high,
            excluded_years,
            decade_year_temporal_ids,
        )
        if values:
            return values

        min_year, max_year = self._temporal_year_sampling_bounds(temporal_id)
        new_low = max(min_year, int(low))
        new_high = min(max_year, int(high))
        while not values and (new_low > min_year or new_high < max_year):
            if new_low > min_year:
                new_low -= 1
            if new_high < max_year:
                new_high += 1
            values = self._temporal_year_domain(
                temporal_id,
                new_low,
                new_high,
                excluded_years,
                decade_year_temporal_ids,
            )
        return values

    def _ensure_minimum_year_domain_width(
        self,
        low: int,
        high: int,
        *,
        max_year: int,
        target_size: int = _MIN_DISTINCT_YEARS_PER_SERIES,
    ) -> tuple[int, int]:
        new_low = max(_MIN_YEAR, int(low))
        new_high = min(int(high), max_year)
        while (new_high - new_low + 1) < target_size and (new_low > _MIN_YEAR or new_high < max_year):
            if new_low > _MIN_YEAR:
                new_low -= 1
            if (new_high - new_low + 1) >= target_size:
                break
            if new_high < max_year:
                new_high += 1
        return new_low, new_high


__all__ = ["TemporalDateValueMixin", "TemporalYearDomainMixin"]
