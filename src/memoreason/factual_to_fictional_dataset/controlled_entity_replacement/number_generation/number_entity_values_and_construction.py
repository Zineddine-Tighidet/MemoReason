# ruff: noqa: RUF046
"""Convert, bound, and construct typed number entities."""

import math

from memoreason.benchmark_definition.document_schema import NumberEntity
from memoreason.benchmark_definition.entity_taxonomy import render_word_surface_number

from ..generation_limits import (
    _DEFAULT_NUMBER_MAX,
    _DEFAULT_NUMBER_MIN,
    _relative_int_window,
)

_MIN_DISTINCT_VALUES_PER_SERIES = 20


class NumberActualValueMixin:
    """Convert between typed number entities and their actual numerical values."""

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
        required_attrs: set[str] | None = None,
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

    def _number_actual_value(self, number_entity: NumberEntity) -> float | None:
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
        required_attrs: set[str] | None = None,
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


class NumberEntityBuilderMixin:
    """Choose number ranges and construct typed number entities."""

    def _factual_number_int(self, number_id: str) -> int | None:
        return self._number_entity_int_value(self._factual_number_entity(number_id))

    def _number_base_range(self, number_id: str) -> tuple[int, int]:
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

    def _factual_number_surface_formats(self, number_id: str) -> tuple[str | None, str | None]:
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

    def _number_kind_for_generation(self, number_id: str, required_attrs: set[str] | None = None) -> str:
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
        required_attrs: set[str] | None = None,
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


__all__ = ["NumberActualValueMixin", "NumberEntityBuilderMixin"]
