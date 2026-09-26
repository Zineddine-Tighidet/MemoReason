"""Parse numeric and temporal surfaces found in MemoReason annotations."""

from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from .entity_taxonomy import ORDINAL_WORD_TO_NUMBER, WORD_TO_NUMBER, parse_integer_surface_number, parse_word_number

_SPECIAL_NUMBER_WORDS: dict[str, int] = {
    "once": 1,
    "twice": 2,
    "thrice": 3,
}
_NUMBER_SCALE_WORDS: dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "trillion": 1_000_000_000_000,
}
_NUMBER_WORD_CONNECTORS: frozenset[str] = frozenset({"and", "a", "an", "of"})
_FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)$")
_YEAR_ONLY_PATTERN = re.compile(r"^(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})$")
_APPROX_EQUAL_REL_TOL = 0.02
_APPROX_EQUAL_ABS_TOL = 0.05


def _parse_word_number_extended(text: str) -> int | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None

    canonical = raw.replace(" ", "-")
    parsed_basic = parse_word_number(canonical)
    if parsed_basic is not None:
        return parsed_basic
    if raw in _SPECIAL_NUMBER_WORDS:
        return _SPECIAL_NUMBER_WORDS[raw]

    normalized = re.sub(r"[,\u2013\u2014]", " ", raw).replace("-", " ")
    tokens = [token for token in normalized.split() if token]
    if not tokens:
        return None

    total = 0
    current = 0
    seen_number_token = False
    for token in tokens:
        if token in _NUMBER_WORD_CONNECTORS:
            continue
        if token in _SPECIAL_NUMBER_WORDS:
            current += _SPECIAL_NUMBER_WORDS[token]
            seen_number_token = True
            continue
        if token in WORD_TO_NUMBER:
            current += WORD_TO_NUMBER[token]
            seen_number_token = True
            continue
        if token in _NUMBER_SCALE_WORDS:
            seen_number_token = True
            scale = _NUMBER_SCALE_WORDS[token]
            if scale == 100:
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * scale
                current = 0
            continue
        singular = token[:-1] if token.endswith("s") else token
        if singular in _NUMBER_SCALE_WORDS:
            seen_number_token = True
            scale = _NUMBER_SCALE_WORDS[singular]
            if scale == 100:
                current = max(current, 1) * 100
            else:
                total += max(current, 1) * scale
                current = 0
            continue
        if token in ORDINAL_WORD_TO_NUMBER:
            current += ORDINAL_WORD_TO_NUMBER[token]
            seen_number_token = True
            continue
        return None

    if not seen_number_token:
        return None
    return total + current


def _coerce_numeric_surface(value: Any) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value

    raw = str(value).strip()
    if not raw:
        return None
    compact = raw.replace(",", "").strip()
    if not compact:
        return None

    parsed_int_surface = parse_integer_surface_number(compact)
    if parsed_int_surface is not None:
        return parsed_int_surface

    if compact.endswith("%"):
        compact = compact[:-1].strip()
        if not compact:
            return None

    fraction_match = re.fullmatch(r"([+-]?\d+)\s*/\s*(\d+)", compact)
    if fraction_match:
        denominator = int(fraction_match.group(2))
        if denominator != 0:
            return int(fraction_match.group(1)) / denominator

    if _FLOAT_PATTERN.fullmatch(compact):
        try:
            as_float = float(compact)
        except ValueError:
            as_float = None
        if as_float is not None:
            if as_float.is_integer():
                return int(as_float)
            return as_float

    digit_scale_match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
        r"(hundreds?|thousands?|millions?|billions?|trillions?)",
        compact,
        flags=re.IGNORECASE,
    )
    if digit_scale_match:
        try:
            amount = float(digit_scale_match.group(1))
            scale_label = digit_scale_match.group(2).lower()
            if scale_label.endswith("s"):
                scale_label = scale_label[:-1]
            scale_value = _NUMBER_SCALE_WORDS.get(scale_label)
            if scale_value is not None:
                scaled = amount * scale_value
                if scaled.is_integer():
                    return int(scaled)
                return scaled
        except ValueError:
            pass

    parsed_word = _parse_word_number_extended(compact)
    if parsed_word is not None:
        return parsed_word

    return None


@lru_cache(maxsize=8192)
def _parse_date_surface_cached(raw: str) -> date | None:
    normalized = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    candidates = [normalized, normalized.replace(",", ""), normalized.replace("  ", " ").strip()]

    formats = (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %Y",
        "%b %Y",
        "%Y",
    )
    for candidate in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue

    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", normalized)
    if year_match:
        try:
            return date(int(year_match.group(1)), 1, 1)
        except ValueError:
            return None
    return None


@lru_cache(maxsize=8192)
def _parse_date_surface_components_cached(raw: str) -> dict[str, Any]:
    normalized = re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    normalized = normalized.replace("\u2013", "-").replace("\u2014", "-")
    candidates = [normalized, normalized.replace(",", ""), normalized.replace("  ", " ").strip()]

    component_formats: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("%Y-%m-%d", ("year", "month", "day_of_month")),
        ("%Y/%m/%d", ("year", "month", "day_of_month")),
        ("%d/%m/%Y", ("year", "month", "day_of_month")),
        ("%m/%d/%Y", ("year", "month", "day_of_month")),
        ("%B %d, %Y", ("year", "month", "day_of_month")),
        ("%b %d, %Y", ("year", "month", "day_of_month")),
        ("%B %d %Y", ("year", "month", "day_of_month")),
        ("%b %d %Y", ("year", "month", "day_of_month")),
        ("%d %B %Y", ("year", "month", "day_of_month")),
        ("%d %b %Y", ("year", "month", "day_of_month")),
        ("%B %Y", ("year", "month")),
        ("%b %Y", ("year", "month")),
        ("%B %d", ("month", "day_of_month")),
        ("%b %d", ("month", "day_of_month")),
        ("%Y", ("year",)),
    )

    for candidate in candidates:
        for fmt, fields in component_formats:
            try:
                parsed = datetime.strptime(candidate, fmt)
            except ValueError:
                continue
            components: dict[str, Any] = {}
            if "year" in fields:
                components["year"] = int(parsed.year)
            if "month" in fields:
                components["month"] = parsed.strftime("%B")
            if "day_of_month" in fields:
                components["day_of_month"] = int(parsed.day)
            return components

    year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", normalized)
    if year_match:
        try:
            return {"year": int(year_match.group(1))}
        except ValueError:
            return {}
    return {}


def _parse_date_surface(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (int, float)):
        year = int(value)
        if 1000 <= year <= 9999:
            try:
                return date(year, 1, 1)
            except ValueError:
                return None
        return None

    raw = str(value).strip()
    if not raw:
        return None
    return _parse_date_surface_cached(raw)


def _parse_date_surface_components(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return {
            "year": int(value.year),
            "month": value.strftime("%B"),
            "day_of_month": int(value.day),
        }
    raw = str(value).strip()
    if not raw:
        return {}
    return dict(_parse_date_surface_components_cached(raw))


def _weekday_from_date_surface(value: Any) -> str | None:
    """Return the weekday only when ``value`` identifies a complete date."""
    components = _parse_date_surface_components(value)
    if not {"year", "month", "day_of_month"}.issubset(components):
        return None
    try:
        month_number = datetime.strptime(str(components["month"]), "%B").month
        parsed = date(
            int(components["year"]),
            month_number,
            int(components["day_of_month"]),
        )
    except (TypeError, ValueError):
        return None
    return parsed.strftime("%A")


def _parse_timestamp_surface(value: Any) -> int | None:
    """Parse a time-of-day surface into minutes since midnight.

    Supports common document formats such as ``5:34 p.m. CDT`` and ``6:12 p.m``.
    Timezone suffixes are ignored because generation rules only rely on local
    same-document differences/orderings, not absolute timezone conversion.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None

    normalized = raw.casefold().strip()
    normalized = normalized.replace("\u202f", " ").replace("\xa0", " ")
    normalized = re.sub(r"\b(a|p)\.(m)\.\b", r"\1m", normalized)
    normalized = re.sub(r"\b(a|p)\.m\b", r"\1m", normalized)
    normalized = re.sub(r"\b(am|pm)\b.*$", r"\1", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\s*(am|pm))?", normalized)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    ampm = match.group(3)

    if minute >= 60:
        return None
    if ampm is None:
        if hour >= 24:
            return None
        return hour * 60 + minute
    if hour < 1 or hour > 12:
        return None
    if hour == 12:
        hour = 0
    if ampm == "pm":
        hour += 12
    return hour * 60 + minute


def _add_years_safe(base_date: date, years: int) -> date:
    target_year = int(base_date.year) + int(years)
    if target_year < 1:
        target_year = 1
    if target_year > 9999:
        target_year = 9999
    try:
        return base_date.replace(year=target_year)
    except ValueError:
        # Leap-day fallback for non-leap target years.
        return base_date.replace(year=target_year, month=2, day=28)
