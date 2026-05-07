"""Temporal surface parsing and sampling-bound helpers."""

from __future__ import annotations

import random
import re
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.document_schema import EntityCollection, TemporalEntity
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number

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


class TemporalValueStrategyMixin:
    """Shared temporal value parsing, surface rendering, and domain helpers."""

    _MONTHS = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    _FICTIONAL_TIMEZONE_SUFFIXES = (
        "local standard time",
        "regional standard time",
        "civil time",
        "local time",
    )

    @staticmethod
    def _fictional_timestamp_suffix(suffix: str) -> str:
        if not str(suffix or "").strip():
            return ""
        return random.choice(TemporalValueStrategyMixin._FICTIONAL_TIMEZONE_SUFFIXES)

    @staticmethod
    def _add_years_safe(value: date, years: int) -> date:
        target_year = int(value.year) + int(years)
        try:
            return value.replace(year=target_year)
        except ValueError:
            # Leap-day fallback: clamp to Feb 28 when the target year is not leap.
            return value.replace(year=target_year, day=28)

    @staticmethod
    def _parse_timestamp_minutes_and_style(value: Any) -> tuple[int | None, dict[str, Any] | None]:
        raw = str(value or "").strip()
        if not raw:
            return None, None

        meridiem_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm|a\.m|p\.m)"
            r"(?:\s+(?P<suffix>.+))?",
            raw,
            flags=re.IGNORECASE,
        )
        if meridiem_match:
            hour = int(meridiem_match.group("hour"))
            minute = int(meridiem_match.group("minute"))
            second = meridiem_match.group("second")
            meridiem_token = meridiem_match.group("meridiem")
            suffix = meridiem_match.group("suffix") or ""
            meridiem = "am" if meridiem_token.casefold().startswith("a") else "pm"
            if meridiem == "am":
                hour24 = 0 if hour == 12 else hour
            else:
                hour24 = 12 if hour == 12 else hour + 12
            return (
                (hour24 * 60) + minute,
                {
                    "kind": "12h",
                    "second": second,
                    "meridiem_token": meridiem_token,
                    "suffix": suffix,
                },
            )

        clock_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"(?:\s+(?P<suffix>.+))?",
            raw,
        )
        if clock_match:
            hour = int(clock_match.group("hour")) % 24
            minute = int(clock_match.group("minute"))
            return (
                (hour * 60) + minute,
                {
                    "kind": "24h",
                    "second": clock_match.group("second"),
                    "suffix": clock_match.group("suffix") or "",
                },
            )

        return None, None

    @staticmethod
    def _format_timestamp_like(template: str | None, total_minutes: int) -> str | None:
        minutes, style = TemporalValueStrategyMixin._parse_timestamp_minutes_and_style(template)
        if style is None:
            return template
        del minutes
        total_minutes %= 24 * 60
        hour24, minute = divmod(total_minutes, 60)
        second = style.get("second")
        suffix = TemporalValueStrategyMixin._fictional_timestamp_suffix(style.get("suffix") or "")

        if style.get("kind") == "12h":
            meridiem = "a.m." if hour24 < 12 else "p.m."
            original_token = str(style.get("meridiem_token") or "").casefold()
            if "." not in original_token:
                meridiem = "am" if hour24 < 12 else "pm"
            elif original_token.endswith(".m") and not original_token.endswith(".m."):
                meridiem = "a.m" if hour24 < 12 else "p.m"
            hour12 = hour24 % 12
            if hour12 == 0:
                hour12 = 12
            rendered = f"{hour12}:{minute:02d}"
            if second is not None:
                rendered += f":{int(second):02d}"
            rendered = f"{rendered} {meridiem}"
            if suffix:
                rendered = f"{rendered} {suffix}"
            return rendered

        rendered = f"{hour24:02d}:{minute:02d}"
        if second is not None:
            rendered += f":{int(second):02d}"
        if suffix:
            rendered = f"{rendered} {suffix}"
        return rendered

    def _factual_temporal_year(self, temporal_id: str) -> Optional[int]:
        if not self.factual_entities or not self.factual_entities.temporals:
            return None
        factual_temporal = self.factual_entities.temporals.get(temporal_id)
        if factual_temporal is None:
            return None
        return self._temporal_year_from_entity(factual_temporal)

    def _factual_temporal_timestamp(self, temporal_id: str) -> Optional[str]:
        if not self.factual_entities or not self.factual_entities.temporals:
            return None
        factual_temporal = self.factual_entities.temporals.get(temporal_id)
        if factual_temporal is None:
            return None
        timestamp = (
            getattr(factual_temporal, "timestamp", None)
            if not isinstance(factual_temporal, dict)
            else factual_temporal.get("timestamp")
        )
        return str(timestamp) if timestamp else None

    def _fictionalize_timestamp_surface(self, timestamp: str) -> str:
        original = str(timestamp or "").strip()
        if not original:
            return original

        meridiem_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm|a\.m|p\.m)"
            r"(?:\s+(?P<suffix>.+))?",
            original,
            flags=re.IGNORECASE,
        )
        if meridiem_match:
            hour = int(meridiem_match.group("hour"))
            minute = int(meridiem_match.group("minute"))
            second = meridiem_match.group("second")
            meridiem = meridiem_match.group("meridiem")
            suffix = self._fictional_timestamp_suffix(meridiem_match.group("suffix") or "")
            shifted_hour = ((hour - 1 + random.randint(1, 5)) % 12) + 1
            shifted_minute = (minute + random.choice((7, 11, 13, 17, 23))) % 60
            pieces = [f"{shifted_hour}:{shifted_minute:02d}"]
            if second is not None:
                shifted_second = (int(second) + random.choice((5, 9, 13, 17))) % 60
                pieces[0] += f":{shifted_second:02d}"
            pieces.append(meridiem)
            if suffix:
                pieces.append(suffix)
            return " ".join(pieces)

        clock_match = re.fullmatch(
            r"(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
            r"(?:\s+(?P<suffix>.+))?",
            original,
        )
        if clock_match:
            hour = int(clock_match.group("hour"))
            minute = int(clock_match.group("minute"))
            second = clock_match.group("second")
            suffix = self._fictional_timestamp_suffix(clock_match.group("suffix") or "")
            shifted_hour = (hour + random.randint(1, 11)) % 24
            shifted_minute = (minute + random.choice((7, 11, 13, 17, 23))) % 60
            pieces = [f"{shifted_hour:02d}:{shifted_minute:02d}"]
            if second is not None:
                shifted_second = (int(second) + random.choice((5, 9, 13, 17))) % 60
                pieces[0] += f":{shifted_second:02d}"
            if suffix:
                pieces.append(suffix)
            return " ".join(pieces)

        return f"{original} local time"

    def _temporal_year_base_range(self, temporal_id: str) -> Tuple[int, int]:
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
            return self._ensure_minimum_year_domain_width(low, high, max_year=max_year)
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

    def _temporal_year_sampling_bounds(self, temporal_id: str) -> Tuple[int, int]:
        factual_year = self._factual_temporal_year(temporal_id)
        if factual_year is not None and factual_year > _CURRENT_YEAR_TEMPORAL_CAP:
            return _MIN_YEAR, _FUTURE_DATE_SAFETY_MAX_YEAR
        return _MIN_YEAR, _CURRENT_YEAR_TEMPORAL_CAP

    def _temporal_year_is_available(
        self,
        temporal_id: str,
        year: int,
        excluded_years: Set[int],
    ) -> bool:
        years_by_id = self.exclude_temporals.get("years_by_id", {})
        forbidden_years = {
            int(candidate)
            for candidate in (years_by_id.get(temporal_id) or set())
            if candidate is not None
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
        excluded_years: Set[int],
        decade_year_temporal_ids: Set[str],
    ) -> List[int]:
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
        excluded_years: Set[int],
        decade_year_temporal_ids: Set[str],
    ) -> List[int]:
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
    ) -> Tuple[int, int]:
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

    def _factual_temporal_day_of_month(self, temporal_id: str) -> Optional[int]:
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

    def _temporal_day_of_month_base_range(self, temporal_id: str) -> Tuple[int, int]:
        del temporal_id
        return 1, 28

    def _month_name_to_number(self, month_name: Optional[str]) -> Optional[int]:
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

    def _temporal_entity_to_date(self, temporal: Any) -> Optional[date]:
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
        temporal.year = value.year
        temporal.month = self._month_number_to_name(value.month)
        temporal.day_of_month = value.day
        temporal.date = f"{value.day} {temporal.month} {value.year}"

    def _excluded_day_of_month_values(self, temporal_id: str) -> set[int]:
        excluded = {
            int(value)
            for value in (self.exclude_temporals.get("day_of_months") or set())
            if value is not None
        }
        by_id = self.exclude_temporals.get("day_of_months_by_id") or {}
        excluded.update(
            int(value)
            for value in (by_id.get(temporal_id) or set())
            if value is not None
        )
        return {value for value in excluded if 1 <= value <= 31}

    def _date_avoids_excluded_day(self, temporal_id: str, value: date) -> bool:
        return int(value.day) not in self._excluded_day_of_month_values(temporal_id)

    def _date_pair_with_delta_avoiding_excluded_days(
        self,
        *,
        left_id: str,
        right_id: str,
        right_date: date,
        days_delta: int,
    ) -> tuple[date, date]:
        target_date = right_date + timedelta(days=days_delta)
        if self._date_avoids_excluded_day(right_id, right_date) and self._date_avoids_excluded_day(left_id, target_date):
            return right_date, target_date

        # Shift the pair together so exact date-difference rules still hold.
        for offset in range(1, 29):
            for signed_offset in (offset, -offset):
                candidate_right = right_date + timedelta(days=signed_offset)
                candidate_left = candidate_right + timedelta(days=days_delta)
                if candidate_right.year != right_date.year or candidate_left.year != target_date.year:
                    continue
                if not self._date_avoids_excluded_day(right_id, candidate_right):
                    continue
                if not self._date_avoids_excluded_day(left_id, candidate_left):
                    continue
                return candidate_right, candidate_left

        return right_date, target_date

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
        if self._date_avoids_excluded_day(temporal_id, right_date + timedelta(days=preferred_delta)):
            return preferred_delta

        max_radius = max(abs(preferred_delta - low), abs(high - preferred_delta), 366)
        for radius in range(1, max_radius + 1):
            for candidate_delta in (preferred_delta - radius, preferred_delta + radius):
                if candidate_delta < low or candidate_delta > high:
                    continue
                candidate_date = right_date + timedelta(days=candidate_delta)
                if self._date_avoids_excluded_day(temporal_id, candidate_date):
                    return candidate_delta
        return preferred_delta

    def _number_days_value(
        self, existing_entities: Optional[EntityCollection], number_id: str, attribute: str
    ) -> Optional[int]:
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

    def _apply_date_difference_rules(
        self,
        temporals: Dict[str, TemporalEntity],
        rules: List[str],
        existing_entities: Optional[EntityCollection],
    ) -> None:
        date_year_offset_pattern = re.compile(
            r"^\s*(temporal_\d+)\.date\s*(?:==|=)\s*(temporal_\d+)\.date\s*([+-])\s*(\d+)\s*$"
        )
        date_diff_pattern = re.compile(
            r"^\s*(temporal_\d+)\.date\s*-\s*(temporal_\d+)\.date\s*(?:==|=)\s*"
            r"(number_\d+)\.(int|str)\s+days\s*$"
        )
        date_diff_literal_pattern = re.compile(
            r"^\s*(temporal_\d+)\.date\s*-\s*(temporal_\d+)\.date\s*(?:==|=)\s*(-?\d+)\s+days\s*$"
        )
        date_diff_bound_pattern = re.compile(
            r"^\s*(temporal_\d+)\.date\s*-\s*(temporal_\d+)\.date\s*(<|<=|>|>=)\s*(-?\d+)\s*$"
        )
        day_diff_pattern = re.compile(
            r"^\s*(temporal_\d+)\.day_of_month\s*-\s*(temporal_\d+)\.day_of_month\s*(?:==|=)\s*"
            r"(number_\d+)\.(int|str)\s+days\s*$"
        )
        day_gt_pattern = re.compile(
            r"^\s*(temporal_\d+)\.day_of_month\s*>\s*(temporal_\d+)\.day_of_month\s*$"
        )
        timestamp_diff_pattern = re.compile(
            r"^\s*(temporal_\d+)\.timestamp\s*-\s*(temporal_\d+)\.timestamp\s*(?:==|=)\s*"
            r"(number_\d+)\.(int|str)\s+minutes\s*$"
        )
        timestamp_gt_pattern = re.compile(
            r"^\s*(temporal_\d+)\.timestamp\s*>\s*(temporal_\d+)\.timestamp\s*$"
        )
        date_diff_bounds: dict[tuple[str, str], dict[str, int | None]] = {}
        for raw_rule in rules:
            cleaned = self._strip_rule_comment(str(raw_rule))
            if not cleaned:
                continue

            match = date_year_offset_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id, sign, raw_delta = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                right_date = self._temporal_entity_to_date(temporals[right_id])
                if right_date is None:
                    continue
                delta_years = int(raw_delta)
                if sign == "-":
                    delta_years = -delta_years
                target_date = self._add_years_safe(right_date, delta_years)
                self._set_temporal_entity_from_date(temporals[left_id], target_date)
                continue

            match = date_diff_bound_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id, op, raw_bound = match.groups()
                bound = int(raw_bound)
                pair_bounds = date_diff_bounds.setdefault((left_id, right_id), {"low": None, "high": None})
                if op == ">":
                    lower = bound + 1
                    pair_bounds["low"] = lower if pair_bounds["low"] is None else max(pair_bounds["low"], lower)
                elif op == ">=":
                    pair_bounds["low"] = bound if pair_bounds["low"] is None else max(pair_bounds["low"], bound)
                elif op == "<":
                    upper = bound - 1
                    pair_bounds["high"] = upper if pair_bounds["high"] is None else min(pair_bounds["high"], upper)
                elif op == "<=":
                    pair_bounds["high"] = bound if pair_bounds["high"] is None else min(pair_bounds["high"], bound)
                continue

            match = date_diff_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id, number_id, number_attr = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                days_delta = self._number_days_value(existing_entities, number_id, number_attr)
                if days_delta is None:
                    continue
                right_date = self._temporal_entity_to_date(temporals[right_id])
                if right_date is None:
                    continue
                adjusted_right_date, target_date = self._date_pair_with_delta_avoiding_excluded_days(
                    left_id=left_id,
                    right_id=right_id,
                    right_date=right_date,
                    days_delta=int(days_delta),
                )
                if adjusted_right_date != right_date:
                    self._set_temporal_entity_from_date(temporals[right_id], adjusted_right_date)
                self._set_temporal_entity_from_date(temporals[left_id], target_date)
                continue

            match = date_diff_literal_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id, raw_days_delta = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                right_date = self._temporal_entity_to_date(temporals[right_id])
                if right_date is None:
                    continue
                adjusted_right_date, target_date = self._date_pair_with_delta_avoiding_excluded_days(
                    left_id=left_id,
                    right_id=right_id,
                    right_date=right_date,
                    days_delta=int(raw_days_delta),
                )
                if adjusted_right_date != right_date:
                    self._set_temporal_entity_from_date(temporals[right_id], adjusted_right_date)
                self._set_temporal_entity_from_date(temporals[left_id], target_date)
                continue

            match = day_diff_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id, number_id, number_attr = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                days_delta = self._number_days_value(existing_entities, number_id, number_attr)
                right_day = getattr(temporals[right_id], "day_of_month", None)
                if days_delta is None or right_day is None:
                    continue
                right_day_int = int(right_day)
                days_delta_int = int(days_delta)
                max_right_day = 28 - days_delta_int
                if max_right_day < 1:
                    continue
                adjusted_right_day = max(1, min(right_day_int, max_right_day))
                temporals[right_id].day_of_month = adjusted_right_day
                temporals[left_id].day_of_month = adjusted_right_day + days_delta_int
                continue

            match = day_gt_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                right_day = getattr(temporals[right_id], "day_of_month", None)
                left_day = getattr(temporals[left_id], "day_of_month", None)
                if right_day is None:
                    continue
                if left_day is None or int(left_day) <= int(right_day):
                    temporals[left_id].day_of_month = max(1, min(28, int(right_day) + 1))
                continue

            match = timestamp_diff_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id, number_id, number_attr = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                minutes_delta = self._number_days_value(existing_entities, number_id, number_attr)
                right_timestamp = getattr(temporals[right_id], "timestamp", None)
                right_minutes, _style = self._parse_timestamp_minutes_and_style(right_timestamp)
                if minutes_delta is None or right_minutes is None:
                    continue
                minutes_delta_int = int(minutes_delta)
                latest_right_minutes = (24 * 60 - 1) - minutes_delta_int
                if latest_right_minutes < 0:
                    continue
                adjusted_right_minutes = max(0, min(int(right_minutes), latest_right_minutes))
                temporals[right_id].timestamp = self._format_timestamp_like(
                    right_timestamp,
                    adjusted_right_minutes,
                )
                template = getattr(temporals[left_id], "timestamp", None) or right_timestamp
                temporals[left_id].timestamp = self._format_timestamp_like(
                    template,
                    adjusted_right_minutes + minutes_delta_int,
                )
                continue

            match = timestamp_gt_pattern.fullmatch(cleaned)
            if match:
                left_id, right_id = match.groups()
                if left_id not in temporals or right_id not in temporals:
                    continue
                left_timestamp = getattr(temporals[left_id], "timestamp", None)
                right_timestamp = getattr(temporals[right_id], "timestamp", None)
                left_minutes, _ = self._parse_timestamp_minutes_and_style(left_timestamp)
                right_minutes, _ = self._parse_timestamp_minutes_and_style(right_timestamp)
                if right_minutes is None:
                    continue
                if left_minutes is None or left_minutes <= right_minutes:
                    template = left_timestamp or right_timestamp
                    temporals[left_id].timestamp = self._format_timestamp_like(
                        template,
                        int(right_minutes) + 1,
                    )

        for (left_id, right_id), pair_bounds in date_diff_bounds.items():
            if left_id not in temporals or right_id not in temporals:
                continue
            right_date = self._temporal_entity_to_date(temporals[right_id])
            if right_date is None:
                continue
            left_date = self._temporal_entity_to_date(temporals[left_id])
            low = pair_bounds.get("low")
            high = pair_bounds.get("high")
            if low is None:
                low = -36500
            if high is None:
                high = 36500
            if int(low) > int(high):
                continue
            if left_date is None:
                target_delta = int(high) if int(high) < 36500 else int(low)
                if target_delta < int(low):
                    target_delta = int(low)
            else:
                current_delta = (left_date - right_date).days
                if (
                    int(low) <= current_delta <= int(high)
                    and self._date_avoids_excluded_day(left_id, left_date)
                ):
                    continue
                if current_delta < int(low):
                    target_delta = int(high) if int(high) < 36500 else int(low)
                else:
                    target_delta = int(high)
            target_delta = self._bounded_date_delta_avoiding_excluded_day(
                temporal_id=left_id,
                right_date=right_date,
                preferred_delta=int(target_delta),
                low=int(low),
                high=int(high),
            )
            self._set_temporal_entity_from_date(temporals[left_id], right_date + timedelta(days=target_delta))

    def generate_temporals(self, required_temporals: List[tuple]) -> Dict[str, TemporalEntity]:
        return self.generate_temporals_with_rules(required_temporals, rules=None, existing_entities=None)

    def _temporal_year_from_entity(self, temporal_entity: Any) -> Optional[int]:
        if temporal_entity is None:
            return None
        year = (
            getattr(temporal_entity, "year", None)
            if not isinstance(temporal_entity, dict)
            else temporal_entity.get("year")
        )
        if year is not None:
            try:
                return int(year)
            except (TypeError, ValueError):
                return None
        date_val = (
            getattr(temporal_entity, "date", None)
            if not isinstance(temporal_entity, dict)
            else temporal_entity.get("date")
        )
        if isinstance(date_val, str):
            m = re.search(r"\b(\d{4})\b", date_val)
            if m:
                return int(m.group(1))
        return None
