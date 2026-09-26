"""Schema-aware canonicalization for MemoReason short answers."""

from __future__ import annotations

import math
import re
from datetime import datetime

from memoreason.benchmark_definition.answer_matching import normalize_answer, try_parse_float

from .answer_schema_data_contracts import (
    _ACRONYM_STOPWORDS,
    _ACRONYM_TOKEN_RE,
    _CHAT_CONTROL_TOKEN_RE,
    _DAY_MONTH_YEAR_RE,
    _ENTITY_ID_RE,
    _ENTITY_TITLE_PREFIX_RE,
    _LEADING_BEFORE_AFTER_RE,
    _LEADING_YES_NO_RE,
    _MONTH_DAY_YEAR_RE,
    _QUANTITY_ABS_TOL,
    _TRAILING_ACRONYM_RE,
    _UNANSWERABLE_PREFIXES,
    _UNANSWERABLE_VALUES,
    _UNICODE_DASH_TRANSLATION,
    _YEAR_RE,
    UNANSWERABLE,
)

_QUANTITY_SCALE_VALUES: dict[str, float] = {
    "hundred": 1e2,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}
_SCALED_QUANTITY_RE = re.compile(
    r"^\s*(?:(?:approximately|approx\.?|about|around)\s+)?"
    r"(?:[A-Za-z]{0,3}\s*)?[\$\u00a3\u20ac\u00a5]?\s*"
    r"(?P<amount>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*"
    r"(?P<scale>hundreds?|thousands?|millions?|billions?|trillions?)"
    r"(?:\s+[A-Za-z][A-Za-z\s-]*)?\s*$",
    flags=re.IGNORECASE,
)
_FRACTION_QUANTITY_RE = re.compile(
    r"^\s*(?P<numerator>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*/\s*"
    r"(?P<denominator>[-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*$"
)


def _clean_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = _CHAT_CONTROL_TOKEN_RE.sub(" ", text)
    text = text.strip("`").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_ordinal_suffixes(text: str) -> str:
    return re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", text, flags=re.IGNORECASE)


def _normalize_dash_like_characters(text: str) -> str:
    return str(text or "").translate(_UNICODE_DASH_TRANSLATION)


def _is_unanswerable_text(text: str) -> bool:
    cleaned = normalize_answer(_clean_text(text))
    if not cleaned:
        return False
    if cleaned in _UNANSWERABLE_VALUES:
        return True
    return any(cleaned.startswith(f"{prefix} ") for prefix in _UNANSWERABLE_PREFIXES)


def _canonical_yes_no(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    cleaned = normalize_answer(_clean_text(text))
    if cleaned in {"yes", "true"}:
        return "YES"
    if cleaned in {"no", "false"}:
        return "NO"
    leading = _leading_yes_no_answer(_clean_text(text))
    if leading == "Yes":
        return "YES"
    if leading == "No":
        return "NO"
    return ""


def _canonical_before_after(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    cleaned = normalize_answer(_clean_text(text))
    if cleaned == "before":
        return "BEFORE"
    if cleaned == "after":
        return "AFTER"
    if cleaned == "earlier":
        return "BEFORE"
    if cleaned == "later":
        return "AFTER"
    leading = _leading_before_after_answer(_clean_text(text))
    if leading == "Before":
        return "BEFORE"
    if leading == "After":
        return "AFTER"
    return ""


def _canonical_quantity(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    cleaned = _clean_text(text).replace(",", "")
    normalized = normalize_answer(cleaned)
    value = None

    fraction_match = _FRACTION_QUANTITY_RE.fullmatch(cleaned)
    if fraction_match:
        numerator = float(fraction_match.group("numerator"))
        denominator = float(fraction_match.group("denominator"))
        if denominator != 0.0:
            value = numerator / denominator

    if value is None:
        scaled_match = _SCALED_QUANTITY_RE.fullmatch(cleaned)
        if scaled_match:
            amount = float(scaled_match.group("amount"))
            scale_label = scaled_match.group("scale").lower().removesuffix("s")
            value = amount * _QUANTITY_SCALE_VALUES[scale_label]

    if value is None:
        value = try_parse_float(normalized)
    if value is None or not math.isfinite(value):
        return ""
    rounded_value = round(value, 12)
    nearest_int = round(rounded_value)
    if math.isclose(rounded_value, nearest_int, rel_tol=0.0, abs_tol=_QUANTITY_ABS_TOL):
        return str(int(nearest_int))
    return format(rounded_value, ".12g")


def _canonical_year(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    matches = _YEAR_RE.findall(_clean_text(text))
    if len(matches) != 1:
        return ""
    return matches[0]


def _canonical_date(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    cleaned = _strip_ordinal_suffixes(_clean_text(text))
    for candidate in _DAY_MONTH_YEAR_RE.findall(cleaned):
        parsed = _parse_date(candidate)
        if parsed is not None:
            return parsed
    for candidate in _MONTH_DAY_YEAR_RE.findall(cleaned):
        parsed = _parse_date(candidate)
        if parsed is not None:
            return parsed
    return ""


def _parse_date(text: str) -> str | None:
    formats = (
        "%d %B %Y",
        "%d %b %Y",
        "%B %d, %Y",
        "%b %d, %Y",
    )
    for date_format in formats:
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        return f"{parsed.day} {parsed.strftime('%B %Y')}"
    return None


def _canonical_free_text(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    cleaned = _normalize_dash_like_characters(_strip_matching_parenthetical_acronym(_clean_text(text)))
    cleaned = cleaned.strip("\"'.,;:!?()[]{} ")
    return normalize_answer(cleaned)


def _canonical_entity_text(text: str) -> str:
    if _is_unanswerable_text(text):
        return UNANSWERABLE
    cleaned = _normalize_dash_like_characters(_strip_matching_parenthetical_acronym(_clean_text(text)))
    cleaned = cleaned.strip("\"'.,;:!?()[]{} ")
    if len(cleaned.split()) >= 2:
        cleaned = _ENTITY_TITLE_PREFIX_RE.sub("", cleaned).strip()
    return normalize_answer(cleaned)


def canonicalize_answer(answer_schema: str, text: str) -> str:
    """Canonicalize one answer according to its schema."""
    if answer_schema == "yes_no":
        return _canonical_yes_no(text)
    if answer_schema == "before_after":
        return _canonical_before_after(text)
    if answer_schema == "quantity":
        return _canonical_quantity(text)
    if answer_schema == "year":
        return _canonical_year(text)
    if answer_schema == "date":
        return _canonical_date(text)
    if answer_schema == "entity_span":
        return _canonical_entity_text(text)
    if answer_schema == "span":
        return _canonical_free_text(text)
    return _canonical_free_text(text)


def _should_add_parenthetical_acronym_alias(value: str) -> bool:
    cleaned = _clean_text(value)
    if not cleaned:
        return False
    token_list = _ACRONYM_TOKEN_RE.findall(cleaned)
    if len(token_list) < 3:
        return False
    if any(character in cleaned for character in ",&/-"):
        return True
    return any(token.lower() in _ACRONYM_STOPWORDS for token in token_list)


def _derive_acronym(value: str) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    acronym_match = _TRAILING_ACRONYM_RE.match(cleaned)
    if acronym_match:
        cleaned = acronym_match.group("long").strip()

    initials: list[str] = []
    for token in _ACRONYM_TOKEN_RE.findall(cleaned):
        lowered = token.lower()
        if lowered in _ACRONYM_STOPWORDS:
            continue
        if not any(character.isalpha() for character in token):
            continue
        initials.append(token[0].upper())

    if len(initials) < 2:
        return None
    acronym = "".join(initials)
    if 2 <= len(acronym) <= 8:
        return acronym
    return None


def _strip_matching_parenthetical_acronym(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    acronym_match = _TRAILING_ACRONYM_RE.match(cleaned)
    if not acronym_match:
        return cleaned
    long_form = acronym_match.group("long").strip()
    short_form = acronym_match.group("short").strip()
    derived_acronym = _derive_acronym(long_form)
    if derived_acronym and normalize_answer(short_form) == normalize_answer(derived_acronym):
        return long_form
    return cleaned


def _strip_parenthetical_segments(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    stripped = cleaned
    while True:
        updated = re.sub(r"\s*\([^()]*\)", "", stripped)
        updated = re.sub(r"\s+", " ", updated).strip(" \t\r\n,.;:!?")
        if updated == stripped:
            return updated
        stripped = updated


def _leading_yes_no_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    opening_match = _LEADING_YES_NO_RE.match(cleaned)
    if opening_match is None:
        return ""
    opening_fragment = re.split(r"(?:[\n\r]+|(?<=[.?!;])\s+)", cleaned, maxsplit=1)[0].strip()
    leading_answer = opening_match.group("answer").lower()
    opposite_answers = ("no", "false") if leading_answer in {"yes", "true"} else ("yes", "true")
    if any(
        re.search(rf"\b{opposite_answer}\b", opening_fragment, re.IGNORECASE) for opposite_answer in opposite_answers
    ):
        return ""
    return "Yes" if leading_answer in {"yes", "true"} else "No"


def _leading_before_after_answer(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    opening_match = _LEADING_BEFORE_AFTER_RE.match(cleaned)
    if opening_match is None:
        return ""
    opening_fragment = re.split(r"(?:[\n\r]+|(?<=[.?!;])\s+)", cleaned, maxsplit=1)[0].strip()
    leading_answer = opening_match.group("answer").lower()
    opposite_answers = ("after", "later") if leading_answer in {"before", "earlier"} else ("before", "earlier")
    if any(
        re.search(rf"\b{opposite_answer}\b", opening_fragment, re.IGNORECASE) for opposite_answer in opposite_answers
    ):
        return ""
    return "Before" if leading_answer in {"before", "earlier"} else "After"


def _entity_type_from_id(entity_id: str) -> str | None:
    match = _ENTITY_ID_RE.match(entity_id)
    if not match:
        return None
    return match.group("entity_type")


def _dedupe_preserve_order(values: tuple[str, ...] | list[str] | tuple[object, ...]) -> tuple[str, ...]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return tuple(deduped)
