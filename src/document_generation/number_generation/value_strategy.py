"""Number-entity representation, conversion, and sampling-bound logic."""

import math
import random
import re
from typing import Iterable, List, Optional, Set, Tuple

from src.core.document_schema import NumberEntity
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number, render_word_surface_number
from ..generation_limits import (
    _CSP_EXACT_DOMAIN_LIMIT,
    _CSP_MAX_SAMPLE_CANDIDATES,
    _DEFAULT_NUMBER_MAX,
    _DEFAULT_NUMBER_MIN,
    _relative_int_window,
)

_MIN_DISTINCT_VALUES_PER_SERIES = 20


class NumberValueStrategyMixin:
    """Shared number-entity value logic used across generation stages."""

    _FRACTION_ORDINAL_WORDS = {
        2: ("half", "halves"),
        3: ("third", "thirds"),
        4: ("fourth", "fourths"),
        5: ("fifth", "fifths"),
        6: ("sixth", "sixths"),
        7: ("seventh", "sevenths"),
        8: ("eighth", "eighths"),
        9: ("ninth", "ninths"),
        10: ("tenth", "tenths"),
        11: ("eleventh", "elevenths"),
        12: ("twelfth", "twelfths"),
    }
    _FRACTION_ORDINAL_ALIASES = {
        "half": 2,
        "halves": 2,
        "third": 3,
        "thirds": 3,
        "quarter": 4,
        "quarters": 4,
        "fourth": 4,
        "fourths": 4,
        "fifth": 5,
        "fifths": 5,
        "sixth": 6,
        "sixths": 6,
        "seventh": 7,
        "sevenths": 7,
        "eighth": 8,
        "eighths": 8,
        "ninth": 9,
        "ninths": 9,
        "tenth": 10,
        "tenths": 10,
        "eleventh": 11,
        "elevenths": 11,
        "twelfth": 12,
        "twelfths": 12,
    }

    @staticmethod
    def _round_non_integer_surface_value(value: float) -> float:
        return round(float(value), 2)

    @staticmethod
    def _coerce_forbidden_number_values(value: int | Iterable[int] | None) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, (set, frozenset, list, tuple)):
            raw_values = value
        else:
            raw_values = [value]
        forbidden: set[int] = set()
        for item in raw_values:
            if isinstance(item, str):
                parsed_fraction = NumberValueStrategyMixin._parse_fraction_surface(item)
                if parsed_fraction is not None:
                    _numerator, denominator = parsed_fraction
                    forbidden.add(int(denominator))
                    continue
            try:
                forbidden.add(int(item))
            except (TypeError, ValueError):
                continue
        return forbidden

    @staticmethod
    def _coerce_forbidden_actual_values(value: int | float | Iterable[int | float] | None) -> set[float]:
        if value is None:
            return set()
        if isinstance(value, (set, frozenset, list, tuple)):
            raw_values = value
        else:
            raw_values = [value]
        forbidden: set[float] = set()
        for item in raw_values:
            try:
                forbidden.add(float(item))
            except (TypeError, ValueError):
                continue
        return forbidden

    @staticmethod
    def _number_field_value(number_entity: NumberEntity | dict | None, field: str):
        if number_entity is None:
            return None
        if isinstance(number_entity, dict):
            return number_entity.get(field)
        return getattr(number_entity, field, None)

    def _factual_number_entity(self, number_id: str) -> NumberEntity | dict | None:
        if not self.factual_entities or not self.factual_entities.numbers:
            return None
        return self.factual_entities.numbers.get(number_id)

    def _factual_number_kind(self, number_id: str) -> str:
        factual_number = self._factual_number_entity(number_id)
        if factual_number is None:
            return "int"
        for field in ("percent", "proportion", "float", "fraction", "int"):
            if self._number_field_value(factual_number, field) is not None:
                return field
        return "int"

    @classmethod
    def _number_entity_int_value(cls, number_entity: NumberEntity | dict | None) -> Optional[int]:
        for field in ("int", "percent", "proportion", "float"):
            value = cls._number_field_value(number_entity, field)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        fraction_value = cls._number_field_value(number_entity, "fraction")
        parsed_fraction = cls._parse_fraction_surface(fraction_value)
        if parsed_fraction is not None:
            _numerator, denominator = parsed_fraction
            return denominator
        str_value = cls._number_field_value(number_entity, "str")
        if str_value is not None:
            parsed_integer = parse_integer_surface_number(str_value)
            if parsed_integer is None:
                parsed_integer = parse_word_number(str_value)
            if parsed_integer is not None:
                return int(parsed_integer)
        return None

    @staticmethod
    def _parse_fraction_surface(value: str | None) -> Optional[tuple[int, int]]:
        if value is None:
            return None
        cleaned = " ".join(str(value).strip().lower().replace("-", " ").split())
        if not cleaned:
            return None
        if cleaned in {"half", "a half"}:
            return 1, 2
        slash_match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", cleaned)
        if slash_match:
            numerator = int(slash_match.group(1))
            denominator = int(slash_match.group(2))
            if numerator > 0 and denominator > numerator:
                return numerator, denominator
            return None
        parts = cleaned.split()
        if len(parts) == 1:
            denominator = NumberValueStrategyMixin._FRACTION_ORDINAL_ALIASES.get(parts[0])
            if denominator is not None:
                return 1, denominator
            return None
        if len(parts) != 2:
            return None
        numerator_text, denominator_text = parts
        denominator = NumberValueStrategyMixin._FRACTION_ORDINAL_ALIASES.get(denominator_text)
        if denominator is None:
            return None
        try:
            numerator = int(numerator_text)
        except ValueError:
            word_to_int = {
                "one": 1,
                "two": 2,
                "three": 3,
                "four": 4,
                "five": 5,
                "six": 6,
                "seven": 7,
                "eight": 8,
                "nine": 9,
                "ten": 10,
                "eleven": 11,
                "twelve": 12,
            }
            numerator = word_to_int.get(numerator_text, 0)
        if numerator <= 0 or denominator <= numerator:
            return None
        return numerator, denominator

    @classmethod
    def _render_fraction_surface(
        cls,
        *,
        numerator: int,
        denominator: int,
        factual_fraction: str | None,
    ) -> str:
        numerator = max(1, int(numerator))
        denominator = max(numerator + 1, int(denominator))
        factual_text = str(factual_fraction or "").strip()
        if "/" in factual_text:
            return f"{numerator}/{denominator}"
        singular, plural = cls._FRACTION_ORDINAL_WORDS.get(
            denominator,
            (
                f"{render_word_surface_number(denominator, 'ordinal_words')}",
                f"{render_word_surface_number(denominator, 'ordinal_words')}s",
            ),
        )
        denominator_word = singular if numerator == 1 else plural
        numerator_word = render_word_surface_number(numerator, "cardinal_words")
        if numerator == 1 and factual_text.lower() == "half":
            return "half"
        return f"{numerator_word} {denominator_word}"

    def _sample_int_in_range(self, low: int, high: int, avoid: Optional[int | Iterable[int]] = None) -> int:
        if low > high:
            raise ValueError(f"Invalid integer range: [{low}, {high}]")
        low, high = self._expand_int_domain_to_escape_forbidden(low, high, avoid=avoid)
        forbidden = {value for value in self._coerce_forbidden_number_values(avoid) if low <= value <= high}
        if not forbidden or low == high:
            return random.randint(low, high)
        domain_size = high - low + 1
        if len(forbidden) >= domain_size:
            return random.randint(low, high)
        if domain_size <= 32:
            available = [value for value in range(low, high + 1) if value not in forbidden]
            if available:
                return random.choice(available)
            return random.randint(low, high)
        for _ in range(64):
            draw = random.randint(low, high)
            if draw not in forbidden:
                return draw
        anchor = random.randint(low, high)
        for offset in range(domain_size):
            candidate = low + ((anchor - low + offset) % domain_size)
            if candidate not in forbidden:
                return candidate
        return anchor

    def _candidate_values(self, low: int, high: int, avoid: Optional[int | Iterable[int]] = None) -> List[int]:
        low, high = self._expand_int_domain_to_escape_forbidden(low, high, avoid=avoid)
        span = high - low + 1
        forbidden = {value for value in self._coerce_forbidden_number_values(avoid) if low <= value <= high}
        if span <= _CSP_EXACT_DOMAIN_LIMIT:
            values = list(range(low, high + 1))
            random.shuffle(values)
        else:
            target = min(_CSP_MAX_SAMPLE_CANDIDATES, span)
            sampled = {low, high, (low + high) // 2}
            while len(sampled) < target:
                sampled.add(random.randint(low, high))
            values = list(sampled)
            random.shuffle(values)
        available = [value for value in values if value not in forbidden]
        if available:
            return available
        return values

    def _expand_int_domain_to_escape_forbidden(
        self,
        low: int,
        high: int,
        *,
        avoid: Optional[int | Iterable[int]] = None,
        min_value: int = _DEFAULT_NUMBER_MIN,
    ) -> Tuple[int, int]:
        low = int(low)
        high = int(high)
        if low > high:
            return low, high
        forbidden = self._coerce_forbidden_number_values(avoid)
        if not forbidden:
            return low, high

        def has_available(domain_low: int, domain_high: int) -> bool:
            for candidate in range(domain_low, domain_high + 1):
                if candidate not in forbidden:
                    return True
            return False

        new_low = low
        new_high = high
        if has_available(new_low, new_high):
            return new_low, new_high

        # Expand symmetrically until at least one legal value remains.
        # This keeps the domain as tight as possible while allowing later
        # fictional variants to avoid reusing previously sampled values.
        safety_limit = len(forbidden) + 64
        for _ in range(safety_limit):
            expanded = False
            if new_low > min_value:
                new_low -= 1
                expanded = True
            new_high += 1
            expanded = True
            if has_available(new_low, new_high):
                return new_low, new_high
            if not expanded:
                break
        return new_low, new_high

    def _ensure_minimum_int_domain_width(
        self,
        low: int,
        high: int,
        *,
        min_value: int = _DEFAULT_NUMBER_MIN,
        target_size: int = _MIN_DISTINCT_VALUES_PER_SERIES,
    ) -> Tuple[int, int]:
        if low > high:
            return low, high
        new_low = int(low)
        new_high = int(high)
        while (new_high - new_low + 1) < target_size:
            if new_low > min_value:
                new_low -= 1
            new_high += 1
        return new_low, new_high

    def _factual_number_int(self, number_id: str) -> Optional[int]:
        return self._number_entity_int_value(self._factual_number_entity(number_id))

    def _number_base_range(self, number_id: str) -> Tuple[int, int]:
        implicit_range = getattr(self, "_implicit_number_range", lambda _number_id: None)(number_id)
        number_kind = self._number_kind_for_generation(number_id)
        if implicit_range is not None:
            low, high = implicit_range
            low = int(low)
            high = int(high)
        else:
            factual_value = self._factual_number_int(number_id)
            if factual_value is None:
                low, high = _DEFAULT_NUMBER_MIN, _DEFAULT_NUMBER_MAX
            else:
                low, high = _relative_int_window(
                    factual_value,
                    small_value_threshold=10,
                    small_value_delta=10,
                    min_value=0 if factual_value == 0 else _DEFAULT_NUMBER_MIN,
                )
                low, high = self._ensure_minimum_int_domain_width(
                    low,
                    high,
                    min_value=0 if factual_value == 0 else _DEFAULT_NUMBER_MIN,
                )
        factual_value = self._factual_number_int(number_id)
        avoid_values = getattr(self, "_current_number_avoid_values", {}) or {}
        if factual_value is None:
            min_value = _DEFAULT_NUMBER_MIN
        else:
            min_value = 0 if factual_value == 0 else _DEFAULT_NUMBER_MIN
        if number_kind == "fraction":
            factual_fraction = self._number_field_value(self._factual_number_entity(number_id), "fraction")
            parsed_fraction = self._parse_fraction_surface(factual_fraction)
            numerator = parsed_fraction[0] if parsed_fraction is not None else 1
            min_value = max(min_value, int(numerator) + 1)
        low = max(int(low), int(min_value))
        if number_id in avoid_values:
            low, high = self._expand_int_domain_to_escape_forbidden(
                low,
                high,
                avoid=avoid_values.get(number_id),
                min_value=min_value,
            )
        extra_padding = int((getattr(self, "_current_number_avoid_expansion", {}) or {}).get(number_id, 0) or 0)
        if extra_padding > 0:
            low = max(min_value, int(low) - extra_padding)
            high = int(high) + extra_padding
        return int(low), int(high)

    def _factual_number_surface_formats(self, number_id: str) -> tuple[Optional[str], Optional[str]]:
        factual_number = self._factual_number_entity(number_id)
        if factual_number is None:
            return None, None
        return (
            self._number_field_value(factual_number, "int_surface_format"),
            self._number_field_value(factual_number, "str_surface_format"),
        )

    def _number_to_string(self, number_id: str, num: int) -> str:
        _, str_surface_format = self._factual_number_surface_formats(number_id)
        if str_surface_format is not None:
            return render_word_surface_number(num, str_surface_format)
        if 1 <= num <= 9:
            return ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"][num - 1]
        return str(num)

    def _number_kind_for_generation(self, number_id: str, required_attrs: Optional[Set[str]] = None) -> str:
        factual_number = self._factual_number_entity(number_id)
        if factual_number is not None:
            for field in ("percent", "proportion", "float", "fraction", "int"):
                if self._number_field_value(factual_number, field) is not None:
                    return field
        for field in ("percent", "proportion", "float", "fraction", "int"):
            if required_attrs and field in required_attrs:
                return field
        return "int"

    def _build_number_entity(
        self,
        number_id: str,
        value: int,
        required_attrs: Optional[Set[str]] = None,
        *,
        allow_non_integer_adjustment: bool = False,
    ) -> NumberEntity:
        int_surface_format, str_surface_format = self._factual_number_surface_formats(number_id)
        number_kind = self._number_kind_for_generation(number_id, required_attrs)
        if number_kind == "percent":
            adjusted = (
                self._non_integer_number_value(number_id, float(value), kind="percent")
                if allow_non_integer_adjustment
                else float(value)
            )
            adjusted = self._round_non_integer_surface_value(adjusted)
            return NumberEntity(int=int(round(adjusted)), percent=float(adjusted))
        if number_kind == "proportion":
            adjusted = (
                self._non_integer_number_value(number_id, float(value), kind="proportion")
                if allow_non_integer_adjustment
                else float(value)
            )
            adjusted = self._round_non_integer_surface_value(adjusted)
            return NumberEntity(int=int(round(adjusted)), proportion=float(adjusted))
        if number_kind == "float":
            adjusted = (
                self._non_integer_number_value(number_id, float(value), kind="float")
                if allow_non_integer_adjustment
                else float(value)
            )
            adjusted = self._round_non_integer_surface_value(adjusted)
            return NumberEntity(int=int(round(adjusted)), float=float(adjusted))
        if number_kind == "fraction":
            factual_fraction = self._number_field_value(self._factual_number_entity(number_id), "fraction")
            parsed_fraction = self._parse_fraction_surface(factual_fraction)
            numerator = parsed_fraction[0] if parsed_fraction is not None else 1
            denominator = max(numerator + 1, int(value))
            return NumberEntity(
                int=denominator,
                fraction=self._render_fraction_surface(
                    numerator=numerator,
                    denominator=denominator,
                    factual_fraction=factual_fraction,
                ),
            )
        return NumberEntity(
            int=value,
            str=self._number_to_string(number_id, value),
            int_surface_format=int_surface_format,
            str_surface_format=str_surface_format,
        )

    def _non_integer_number_value(self, number_id: str, value: float, *, kind: str) -> float:
        adjusted = float(value)
        factual_number = self._factual_number_entity(number_id)
        factual_value = self._number_field_value(factual_number, kind)
        try:
            factual_float = float(factual_value) if factual_value is not None else None
        except (TypeError, ValueError):
            factual_float = None
        if factual_float is not None:
            adjusted += factual_float - math.floor(factual_float)

        bounds_rule = getattr(self, "_implicit_rule_lookup", {}).get(f"{number_id}.{kind}")
        if bounds_rule is not None:
            low = float(bounds_rule.lower_bound)
            high = float(bounds_rule.upper_bound)
        elif factual_float is not None:
            if abs(factual_float) < 10:
                min_value = 0.0 if math.isclose(factual_float, 0.0, abs_tol=1e-9) else 1.0
                low = max(min_value, factual_float - 10.0)
                high = max(min_value, factual_float + 10.0)
            else:
                low = factual_float * 0.8
                high = factual_float * 1.2
                if low > high:
                    low, high = high, low
        else:
            low = adjusted
            high = adjusted
        adjusted = min(max(adjusted, low), high)

        if factual_float is not None and math.isclose(adjusted, factual_float, abs_tol=1e-9) and high > low:
            epsilon = min(0.1, max(0.01, (high - low) / 10))
            if adjusted + epsilon <= high:
                adjusted += epsilon
            elif adjusted - epsilon >= low:
                adjusted -= epsilon
        return adjusted

    def _number_actual_bounds(
        self,
        number_id: str,
        required_attrs: Optional[Set[str]] = None,
    ) -> tuple[float, float]:
        number_kind = self._number_kind_for_generation(number_id, required_attrs)
        if number_kind in {"int", "fraction"}:
            low, high = self._number_base_range(number_id)
            return float(low), float(high)

        factual_number = self._factual_number_entity(number_id)
        factual_value = self._number_field_value(factual_number, number_kind)
        try:
            factual_float = float(factual_value) if factual_value is not None else None
        except (TypeError, ValueError):
            factual_float = None

        bounds_rule = getattr(self, "_implicit_rule_lookup", {}).get(f"{number_id}.{number_kind}")
        if bounds_rule is not None:
            return float(bounds_rule.lower_bound), float(bounds_rule.upper_bound)
        if factual_float is not None:
            if abs(factual_float) < 10:
                min_value = 0.0 if math.isclose(factual_float, 0.0, abs_tol=1e-9) else 1.0
                low = max(min_value, factual_float - 10.0)
                high = max(min_value, factual_float + 10.0)
            else:
                low = factual_float * 0.8
                high = factual_float * 1.2
            return (low, high) if low <= high else (high, low)
        actual = factual_float if factual_float is not None else float(self._factual_number_int(number_id) or 0)
        return actual, actual

    def _number_actual_value(self, number_entity: NumberEntity) -> Optional[float]:
        for field in ("float", "percent", "proportion", "int"):
            value = self._number_field_value(number_entity, field)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        fraction_value = self._number_field_value(number_entity, "fraction")
        parsed_fraction = self._parse_fraction_surface(fraction_value)
        if parsed_fraction is None:
            return None
        numerator, denominator = parsed_fraction
        try:
            return float(numerator) / float(denominator)
        except ZeroDivisionError:
            return None

    def _set_number_actual_value(
        self,
        number_id: str,
        number_entity: NumberEntity,
        actual_value: float,
        *,
        required_attrs: Optional[Set[str]] = None,
    ) -> None:
        number_kind = self._number_kind_for_generation(number_id, required_attrs)
        if number_kind == "percent":
            rounded_value = self._round_non_integer_surface_value(actual_value)
            number_entity.percent = float(rounded_value)
            number_entity.int = int(round(actual_value))
            return
        if number_kind == "proportion":
            rounded_value = self._round_non_integer_surface_value(actual_value)
            number_entity.proportion = float(rounded_value)
            number_entity.int = int(round(actual_value))
            return
        if number_kind == "float":
            rounded_value = self._round_non_integer_surface_value(actual_value)
            number_entity.float = float(rounded_value)
            number_entity.int = int(round(actual_value))
            return
        int_value = int(round(actual_value))
        rebuilt = self._build_number_entity(
            number_id,
            int_value,
            required_attrs=required_attrs,
            allow_non_integer_adjustment=False,
        )
        number_entity.int = rebuilt.int
        number_entity.str = rebuilt.str
        number_entity.float = rebuilt.float
        number_entity.fraction = rebuilt.fraction
        number_entity.percent = rebuilt.percent
        number_entity.proportion = rebuilt.proportion
        number_entity.int_surface_format = rebuilt.int_surface_format
        number_entity.str_surface_format = rebuilt.str_surface_format
