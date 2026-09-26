"""Temporal surface parsing and sampling-bound helpers."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from memoreason.benchmark_definition.document_schema import EntityCollection, TemporalEntity


_MIN_DISTINCT_YEARS_PER_SERIES = 20


class TemporalDateRuleApplicationMixin:
    """Apply date-difference rules and expose temporal generation helpers."""

    def _apply_date_difference_rules(
        self,
        temporals: dict[str, TemporalEntity],
        rules: list[str],
        existing_entities: EntityCollection | None,
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
            r"|^\s*(temporal_\d+)\.date\s*-\s*(temporal_\d+)\.date\s*(?:==|=)\s*(-?\d+)\s*$"
        )
        date_diff_bound_pattern = re.compile(
            r"^\s*(temporal_\d+)\.date\s*-\s*(temporal_\d+)\.date\s*(<|<=|>|>=)\s*(-?\d+)\s*$"
        )
        day_diff_pattern = re.compile(
            r"^\s*(temporal_\d+)\.day_of_month\s*-\s*(temporal_\d+)\.day_of_month\s*(?:==|=)\s*"
            r"(number_\d+)\.(int|str)\s+days\s*$"
        )
        day_gt_pattern = re.compile(r"^\s*(temporal_\d+)\.day_of_month\s*>\s*(temporal_\d+)\.day_of_month\s*$")
        timestamp_diff_pattern = re.compile(
            r"^\s*(temporal_\d+)\.timestamp\s*-\s*(temporal_\d+)\.timestamp\s*(?:==|=)\s*"
            r"(number_\d+)\.(int|str)\s+minutes\s*$"
        )
        timestamp_gt_pattern = re.compile(r"^\s*(temporal_\d+)\.timestamp\s*>\s*(temporal_\d+)\.timestamp\s*$")
        date_diff_bounds: dict[tuple[str, str], dict[str, int | None]] = {}
        # Collect bounds before applying equalities so an exact linked pair can
        # choose a right-hand date that still leaves a feasible per-ID surface
        # for every bounded dependent date.
        for raw_rule in rules:
            cleaned = self._strip_rule_comment(str(raw_rule))
            match = date_diff_bound_pattern.fullmatch(cleaned)
            if match is None:
                continue
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
                linked_bounds: list[tuple[str, int, int, int, int]] = []
                for (bound_left_id, bound_right_id), pair_bounds in date_diff_bounds.items():
                    if bound_right_id != right_id or bound_left_id not in temporals:
                        continue
                    bound_left_date = self._temporal_entity_to_date(temporals[bound_left_id])
                    if bound_left_date is None:
                        continue
                    low = int(pair_bounds.get("low") if pair_bounds.get("low") is not None else -36500)
                    high = int(pair_bounds.get("high") if pair_bounds.get("high") is not None else 36500)
                    if low > high:
                        continue
                    preferred_delta = max(low, min(high, (bound_left_date - right_date).days))
                    linked_bounds.append(
                        (bound_left_id, int(bound_left_date.year), int(preferred_delta), low, high)
                    )
                adjusted_right_date, target_date = self._date_pair_with_delta_avoiding_excluded_days(
                    left_id=left_id,
                    right_id=right_id,
                    right_date=right_date,
                    days_delta=int(days_delta),
                    linked_bounds=linked_bounds,
                )
                if adjusted_right_date != right_date:
                    self._set_temporal_entity_from_date(temporals[right_id], adjusted_right_date)
                self._set_temporal_entity_from_date(temporals[left_id], target_date)
                continue

            match = date_diff_literal_pattern.fullmatch(cleaned)
            if match:
                (
                    first_left_id,
                    first_right_id,
                    first_days_delta,
                    second_left_id,
                    second_right_id,
                    second_days_delta,
                ) = match.groups()
                left_id = first_left_id or second_left_id
                right_id = first_right_id or second_right_id
                raw_days_delta = first_days_delta or second_days_delta
                if left_id not in temporals or right_id not in temporals:
                    continue
                right_date = self._temporal_entity_to_date(temporals[right_id])
                if right_date is None:
                    continue
                linked_bounds = []
                for (bound_left_id, bound_right_id), pair_bounds in date_diff_bounds.items():
                    if bound_right_id != right_id or bound_left_id not in temporals:
                        continue
                    bound_left_date = self._temporal_entity_to_date(temporals[bound_left_id])
                    if bound_left_date is None:
                        continue
                    low = int(pair_bounds.get("low") if pair_bounds.get("low") is not None else -36500)
                    high = int(pair_bounds.get("high") if pair_bounds.get("high") is not None else 36500)
                    if low > high:
                        continue
                    preferred_delta = max(low, min(high, (bound_left_date - right_date).days))
                    linked_bounds.append(
                        (bound_left_id, int(bound_left_date.year), int(preferred_delta), low, high)
                    )
                adjusted_right_date, target_date = self._date_pair_with_delta_avoiding_excluded_days(
                    left_id=left_id,
                    right_id=right_id,
                    right_date=right_date,
                    days_delta=int(raw_days_delta),
                    linked_bounds=linked_bounds,
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
                if int(low) <= current_delta <= int(high):
                    target_delta = current_delta
                elif current_delta < int(low):
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

    def generate_temporals(self, required_temporals: list[tuple]) -> dict[str, TemporalEntity]:
        return self.generate_temporals_with_rules(required_temporals, rules=None, existing_entities=None)

    def _temporal_year_from_entity(self, temporal_entity: Any) -> int | None:
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


__all__ = ["TemporalDateRuleApplicationMixin"]
