"""Number-entity representation, conversion, and sampling-bound logic."""

import random
from collections.abc import Iterable

from memoreason.benchmark_definition.entity_taxonomy import render_word_surface_number
from ..generation_limits import (
    _CSP_EXACT_DOMAIN_LIMIT,
    _CSP_MAX_SAMPLE_CANDIDATES,
    _DEFAULT_NUMBER_MIN,
)

_MIN_DISTINCT_VALUES_PER_SERIES = 20


class NumberCandidateValueMixin:
    """Build candidate integer domains while respecting forbidden values."""

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

    def _sample_int_in_range(self, low: int, high: int, avoid: int | Iterable[int] | None = None) -> int:
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

    def _candidate_values(self, low: int, high: int, avoid: int | Iterable[int] | None = None) -> list[int]:
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
        avoid: int | Iterable[int] | None = None,
        min_value: int = _DEFAULT_NUMBER_MIN,
    ) -> tuple[int, int]:
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
    ) -> tuple[int, int]:
        if low > high:
            return low, high
        new_low = int(low)
        new_high = int(high)
        while (new_high - new_low + 1) < target_size:
            if new_low > min_value:
                new_low -= 1
            new_high += 1
        return new_low, new_high


__all__ = ["NumberCandidateValueMixin"]
