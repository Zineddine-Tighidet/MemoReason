"""Number-entity representation, conversion, and sampling-bound logic."""

import re
from collections.abc import Iterable
from typing import ClassVar

from memoreason.benchmark_definition.document_schema import NumberEntity
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

from .number_entity_values_and_construction import NumberActualValueMixin, NumberEntityBuilderMixin
from .candidate_values import NumberCandidateValueMixin

_MIN_DISTINCT_VALUES_PER_SERIES = 20


class NumberValueStrategyMixin(NumberCandidateValueMixin, NumberEntityBuilderMixin, NumberActualValueMixin):
    """Shared number-entity value logic used across generation stages."""

    _FRACTION_ORDINAL_WORDS: ClassVar[dict[int, tuple[str, str]]] = {
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
    _FRACTION_ORDINAL_ALIASES: ClassVar[dict[str, int]] = {
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
    def _number_entity_int_value(cls, number_entity: NumberEntity | dict | None) -> int | None:
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
    def _parse_fraction_surface(value: str | None) -> tuple[int, int] | None:
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


__all__ = ["NumberValueStrategyMixin"]
