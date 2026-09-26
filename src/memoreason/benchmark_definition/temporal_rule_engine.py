"""Temporal and date-expression operations for the MemoReason rule engine."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

from .document_schema import EntityCollection
from .entity_taxonomy import _WEEKDAY_ALIASES, _WEEKDAY_TO_INDEX, WEEKDAYS
from .annotation_values import (
    _YEAR_ONLY_PATTERN,
    _add_years_safe,
    _coerce_numeric_surface,
    _parse_date_surface,
    _parse_timestamp_surface,
)

logger = logging.getLogger(__name__)

# Bound to the composed class by ``rule_engine`` after module import.
RuleEngine: Any = None


class TemporalRuleEngineMixin:
    def _normalize_weekday(value: Any) -> str | None:
        if value is None:
            return None
        key = str(value).strip().lower()
        if not key:
            return None
        key = _WEEKDAY_ALIASES.get(key, key)
        idx = _WEEKDAY_TO_INDEX.get(key)
        if idx is None:
            return None
        return WEEKDAYS[idx]

    @staticmethod
    def _evaluate_weekday_shift(expression: str, entities: EntityCollection) -> str | None:
        """Evaluate expressions like temporal_1.day - 2 (cyclic on weekdays)."""
        m = re.fullmatch(r"(temporal_\d+\.day)\s*([+-])\s*(.+)", expression.strip(), re.IGNORECASE)
        if not m:
            return None
        ref, op, offset_expr = m.groups()
        base_value = RuleEngine._get_entity_value(entities, ref)
        base_day = RuleEngine._normalize_weekday(base_value)
        if base_day is None:
            return None
        offset_value = RuleEngine._evaluate_arithmetic(offset_expr.strip(), entities)
        offset_numeric = _coerce_numeric_surface(offset_value)
        if offset_numeric is None:
            return None
        base_idx = _WEEKDAY_TO_INDEX[base_day.lower()]
        offset = int(offset_numeric) % len(WEEKDAYS)
        if op == "-":
            offset = -offset
        return WEEKDAYS[(base_idx + offset) % len(WEEKDAYS)]

    @staticmethod
    def _evaluate_weekday_difference(expression: str, entities: EntityCollection) -> int | None:
        """Evaluate elapsed days such as ``Tuesday - Monday`` on a weekly cycle."""
        match = re.fullmatch(
            r"(temporal_\d+\.day)\s*-\s*(temporal_\d+\.day)",
            expression.strip(),
            re.IGNORECASE,
        )
        if not match:
            return None
        left_ref, right_ref = match.groups()
        left_day = RuleEngine._normalize_weekday(RuleEngine._get_entity_value(entities, left_ref))
        right_day = RuleEngine._normalize_weekday(RuleEngine._get_entity_value(entities, right_ref))
        if left_day is None or right_day is None:
            return None
        return (_WEEKDAY_TO_INDEX[left_day.lower()] - _WEEKDAY_TO_INDEX[right_day.lower()]) % len(WEEKDAYS)

    @staticmethod
    def _calendar_year_difference(left_date: date, right_date: date) -> int:
        """Return signed full-year difference between two calendar dates."""
        if left_date >= right_date:
            years = left_date.year - right_date.year
            if (left_date.month, left_date.day) < (right_date.month, right_date.day):
                years -= 1
            return years
        years = right_date.year - left_date.year
        if (right_date.month, right_date.day) < (left_date.month, left_date.day):
            years -= 1
        return -years

    @staticmethod
    def _evaluate_year_function_expression(expression: str, entities: EntityCollection) -> int | None:
        """Evaluate helper expressions such as year(temporal_2.date - temporal_1.date)."""
        expr = str(expression or "").strip()
        match = re.fullmatch(r"year\s*\(\s*(.+?)\s*\)", expr, flags=re.IGNORECASE)
        if not match:
            return None
        inner_expr = match.group(1).strip()
        if not inner_expr:
            return None

        diff_match = re.fullmatch(r"(.+?)\s*-\s*(.+)", inner_expr)
        if diff_match:
            left_value = RuleEngine._evaluate_arithmetic(diff_match.group(1).strip(), entities)
            right_value = RuleEngine._evaluate_arithmetic(diff_match.group(2).strip(), entities)
            left_date = _parse_date_surface(left_value)
            right_date = _parse_date_surface(right_value)
            if left_date is not None and right_date is not None:
                return RuleEngine._calendar_year_difference(left_date, right_date)

        value = RuleEngine._evaluate_arithmetic(inner_expr, entities)
        parsed_date = _parse_date_surface(value)
        if parsed_date is not None:
            return int(parsed_date.year)
        numeric = _coerce_numeric_surface(value)
        if numeric is not None:
            return int(numeric)
        return None

    @staticmethod
    def _normalize_person_age_rules(rule: str) -> str:
        """Normalize bare person references in numeric comparisons to use .age.

        Converts e.g. 'person_2 < 18' -> 'person_2.age < 18' so that
        numeric comparison rules work even when the entity has a full_name.
        """
        # Pattern: person_N (without .attr) followed/preceded by a comparison operator and a number
        rule = re.sub(
            r"\b(person_\d+)\s*([<>=!]+)\s*(\d+)\b",
            lambda m: f"{m.group(1)}.age {m.group(2)} {m.group(3)}" if "." not in m.group(1) else m.group(0),
            rule,
        )
        rule = re.sub(
            r"\b(\d+)\s*([<>=!]+)\s*(person_\d+)\b",
            lambda m: f"{m.group(1)} {m.group(2)} {m.group(3)}.age" if "." not in m.group(3) else m.group(0),
            rule,
        )
        return rule

    @staticmethod
    def _normalize_comparison_expression(expression: str) -> str:
        # Accept author-provided single "=" equality in rule text.
        return re.sub(r"(?<![<>=!])=(?!=)", "==", str(expression or ""))

    @staticmethod
    def _contains_comparison_operator(expression: str) -> bool:
        return bool(re.search(r"(>=|<=|==|!=|>|<)", str(expression or "")))

    @staticmethod
    def _evaluate_temporal_offset_expression(expression: str, entities: EntityCollection) -> Any | None:
        expr = str(expression or "").strip()
        if not expr:
            return None

        if not re.search(r"[+-]", expr):
            unit_only_match = re.fullmatch(r"(.+?)\s+(years?|days?|hours?|minutes?)", expr, flags=re.IGNORECASE)
            if unit_only_match:
                value = RuleEngine._evaluate_arithmetic(unit_only_match.group(1).strip(), entities)
                numeric = _coerce_numeric_surface(value)
                return int(numeric) if numeric is not None else None

        term_pattern = re.compile(r"([+-])\s*([^+-]+?)\s*(years?|days?)\b", flags=re.IGNORECASE)
        terms = list(term_pattern.finditer(expr))
        if not terms:
            # Legacy shorthand: treat `<date_expr> +/- N` as year offsets.
            # We intentionally gate this to non-year date surfaces so arithmetic
            # such as `temporal_2.year - temporal_1.year` remains numeric.
            bare_offset_match = re.fullmatch(r"(.+?)\s*([+-])\s*(.+)", expr)
            if bare_offset_match:
                base_expr = bare_offset_match.group(1).strip()
                amount_expr = bare_offset_match.group(3).strip()
                if base_expr and amount_expr:
                    base_value = RuleEngine._evaluate_arithmetic(base_expr, entities)
                    amount_value = RuleEngine._evaluate_arithmetic(amount_expr, entities)
                    amount = _coerce_numeric_surface(amount_value)

                    is_year_like = False
                    if isinstance(base_value, (int, float)):
                        is_year_like = True
                    elif isinstance(base_value, str) and _YEAR_ONLY_PATTERN.fullmatch(base_value.strip()):
                        is_year_like = True

                    base_date = _parse_date_surface(base_value)
                    if amount is not None and base_date is not None and not is_year_like:
                        delta = int(amount)
                        if bare_offset_match.group(2) == "-":
                            delta = -delta
                        return _add_years_safe(base_date, delta)
            return None

        base_expr = expr[: terms[0].start()].strip()
        if not base_expr:
            return None
        if term_pattern.sub("", expr).strip() != base_expr:
            return None

        base_value = RuleEngine._evaluate_arithmetic(base_expr, entities)
        if base_value is None:
            return None

        mode: str
        current_numeric: float | int | None = None
        current_date: date | None = None
        if isinstance(base_value, (int, float)) or (
            isinstance(base_value, str) and _YEAR_ONLY_PATTERN.fullmatch(base_value.strip())
        ):
            parsed_year = _coerce_numeric_surface(base_value)
            if parsed_year is None:
                return None
            try:
                current_date = date(int(parsed_year), 1, 1)
            except ValueError:
                return None
            mode = "year"
        else:
            parsed_date = _parse_date_surface(base_value)
            if parsed_date is not None:
                current_date = parsed_date
                mode = "date"
            else:
                parsed_numeric = _coerce_numeric_surface(base_value)
                if parsed_numeric is None:
                    return None
                current_numeric = parsed_numeric
                mode = "numeric"

        for match in terms:
            sign = -1 if match.group(1) == "-" else 1
            term_expr = match.group(2).strip()
            term_unit = match.group(3).strip().lower()
            term_value = RuleEngine._evaluate_arithmetic(term_expr, entities)
            amount = _coerce_numeric_surface(term_value)
            if amount is None:
                return None
            delta = int(amount) * sign

            if mode in {"year", "date"} and current_date is not None:
                if term_unit.startswith("year"):
                    current_date = _add_years_safe(current_date, delta)
                else:
                    current_date = current_date + timedelta(days=delta)
            elif current_numeric is not None:
                current_numeric = float(current_numeric) + float(delta)
            else:
                return None

        if mode == "year" and current_date is not None:
            return int(current_date.year)
        if mode == "date":
            return current_date
        if current_numeric is None:
            return None
        if float(current_numeric).is_integer():
            return int(current_numeric)
        return current_numeric

    @staticmethod
    def _evaluate_date_difference_expression(expression: str, entities: EntityCollection) -> Any | None:
        expr = str(expression or "").strip()
        if not expr:
            return None
        match = re.fullmatch(r"(.+?)\s*-\s*(.+)", expr)
        if not match:
            return None
        left_expr = match.group(1).strip()
        right_expr = match.group(2).strip()
        if not left_expr or not right_expr:
            return None

        left_value = RuleEngine._evaluate_arithmetic(left_expr, entities)
        right_value = RuleEngine._evaluate_arithmetic(right_expr, entities)
        left_date = _parse_date_surface(left_value)
        right_date = _parse_date_surface(right_value)
        if left_date is None or right_date is None:
            return None

        def _is_year_like(value: Any) -> bool:
            if isinstance(value, (int, float)):
                return True
            if isinstance(value, str) and _YEAR_ONLY_PATTERN.fullmatch(value.strip()):
                return True
            return False

        left_year_like = _is_year_like(left_value)
        right_year_like = _is_year_like(right_value)
        if left_year_like and right_year_like:
            # Let pure year arithmetic run through normal numeric eval.
            return None
        if left_year_like or right_year_like:
            # Mixed year/date arithmetic compares calendar-year offsets.
            return int(left_date.year - right_date.year)

        return int((left_date - right_date).days)

    @staticmethod
    def _evaluate_timestamp_difference_expression(expression: str, entities: EntityCollection) -> Any | None:
        expr = str(expression or "").strip()
        if not expr:
            return None
        match = re.fullmatch(r"(.+?)\s*-\s*(.+)", expr)
        if not match:
            return None
        left_expr = match.group(1).strip()
        right_expr = match.group(2).strip()
        if not left_expr or not right_expr:
            return None

        left_value = RuleEngine._evaluate_arithmetic(left_expr, entities)
        right_value = RuleEngine._evaluate_arithmetic(right_expr, entities)
        left_minutes = _parse_timestamp_surface(left_value)
        right_minutes = _parse_timestamp_surface(right_value)
        if left_minutes is None or right_minutes is None:
            return None
        return int(left_minutes - right_minutes)

    @staticmethod
    def _resolve_temporal_date_component(entity_id: str, component: str, entities: EntityCollection) -> Any:
        date_value = RuleEngine._get_entity_value(entities, f"{entity_id}.date")
        parsed_date = _parse_date_surface(date_value)
        if parsed_date is None:
            return None
        if component == "year":
            return int(parsed_date.year)
        if component == "month":
            return parsed_date.strftime("%B")
        if component == "day":
            return int(parsed_date.day)
        return None

    @staticmethod
    def validate_all_rules(rules: list[str], entities: EntityCollection) -> list[tuple[str, bool]]:
        result = []
        for rule in rules:
            try:
                cleaned_rule = str(rule or "").split("#", 1)[0].strip()
                if not cleaned_rule:
                    continue
                # Normalize bare person refs in numeric comparisons: person_2 < 18 -> person_2.age < 18
                normalized = RuleEngine._normalize_person_age_rules(cleaned_rule)
                eval_rule = (
                    normalized
                    if any(x in normalized for x in ("!=", ">=", "<=", "=="))
                    else normalized.replace(" = ", " == ")
                )
                res = RuleEngine.evaluate_expression(eval_rule, entities)
                result.append((rule, bool(res) if isinstance(res, (bool, int, float)) else False))
            except Exception as e:
                logger.debug("Rule validation failed for %r: %s", rule, e)
                result.append((rule, False))
        return result
