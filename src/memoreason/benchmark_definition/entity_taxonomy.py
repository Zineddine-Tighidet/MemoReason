"""
Taxonomy and normalization constants shared across utility modules.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from num2words import num2words

from .organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    ORG_ENTITY_TYPES,
    canonicalize_organization_type,
    infer_organization_kind,
    is_organization_entity_type,
)

ANNOTATION_ENTITY_TYPE_ORDER: tuple[str, ...] = (
    "person",
    "place",
    "event",
    "military_org",
    "entreprise_org",
    "ngo",
    "government_org",
    "educational_org",
    "media_org",
    "temporal",
    "number",
    "award",
    "legal",
    "product",
)

# Valid entity types (used for annotation validation, not ablation).
VALID_ENTITY_TYPES: frozenset = frozenset(
    {
        "person",
        "place",
        "event",
        *ORG_ENTITY_TYPES,
        "number",
        "temporal",
        "award",
        "legal",
        "product",
    }
)

# ---------------------------------------------------------------------------
# Replacement-mode ablation constants
# ---------------------------------------------------------------------------
REPLACE_MODE_ALL = "all"
REPLACE_MODE_NUMERICAL = "numerical"
REPLACE_MODE_TEMPORAL = "temporal"
REPLACE_MODE_NUMERICAL_TEMPORAL = "numerical_temporal"
REPLACE_MODE_NON_NUMERICAL = "non_numerical"

VALID_REPLACE_MODES: frozenset = frozenset(
    {
        REPLACE_MODE_ALL,
        REPLACE_MODE_NUMERICAL,
        REPLACE_MODE_TEMPORAL,
        REPLACE_MODE_NUMERICAL_TEMPORAL,
        REPLACE_MODE_NON_NUMERICAL,
    }
)

# Entity types that are *fully* replaced (every attribute) in each mode.
FULL_REPLACE_ENTITY_TYPES: Dict[str, frozenset] = {
    REPLACE_MODE_ALL: frozenset(
        {
            "person",
            "place",
            "event",
            *CANONICAL_ORGANIZATION_TYPES,
            "number",
            "temporal",
            "award",
            "legal",
            "product",
        }
    ),
    REPLACE_MODE_NUMERICAL: frozenset({"number"}),
    REPLACE_MODE_TEMPORAL: frozenset({"temporal"}),
    REPLACE_MODE_NUMERICAL_TEMPORAL: frozenset({"number", "temporal"}),
    REPLACE_MODE_NON_NUMERICAL: frozenset(
        {
            "place",
            "event",
            *CANONICAL_ORGANIZATION_TYPES,
            "award",
            "legal",
            "product",
        }
    ),
}

# Entity types where only *some* attributes are replaced (the rest stay factual).
# Maps mode -> entity_type -> set of attributes to replace.
PARTIAL_REPLACE_ATTRIBUTES: Dict[str, Dict[str, frozenset]] = {
    REPLACE_MODE_NUMERICAL: {
        "person": frozenset({"age"}),
    },
    REPLACE_MODE_NUMERICAL_TEMPORAL: {
        "person": frozenset({"age"}),
    },
    REPLACE_MODE_NON_NUMERICAL: {
        "person": frozenset(
            {
                "full_name",
                "first_name",
                "last_name",
                "gender",
                "nationality",
                "ethnicity",
                "honorific",
            }
        ),
    },
    # temporal and all modes have no partial entities.
}


def replace_mode_label(mode: str) -> Optional[str]:
    """Return the output-directory subfolder name for a replacement mode."""
    if mode == REPLACE_MODE_ALL:
        return None
    return f"replace_{mode}"


# ---------------------------------------------------------------------------
# Annotation taxonomy (from TAXONOMY.md) -- used for validation
# ---------------------------------------------------------------------------
# Maps each entity type to the set of valid attribute names.
# "relationship.person_Y" is the directional form, and "relationship" is allowed
# as a generic person-level relationship label.
ENTITY_TAXONOMY: Dict[str, frozenset] = {
    "person": frozenset(
        {
            "full_name",
            "first_name",
            "last_name",
            "age",
            "gender",
            "nationality",
            "ethnicity",
            "subj_pronoun",
            "obj_pronoun",
            "poss_det_pronoun",
            "poss_pro_pronoun",
            "refl_pronoun",
            "honorific",
            "middle_name",
            "relationship",
        }
    ),
    "place": frozenset(
        {
            "city",
            "region",
            "state",
            "country",
            "street",
            "natural_site",
            "continent",
            "demonym",
        }
    ),
    "event": frozenset({"name", "type"}),
    "military_org": frozenset({"name"}),
    "entreprise_org": frozenset({"name"}),
    "ngo": frozenset({"name"}),
    "government_org": frozenset({"name"}),
    "educational_org": frozenset({"name"}),
    "media_org": frozenset({"name"}),
    "temporal": frozenset(
        {
            "day",
            "date",
            "year",
            "month",
            "timestamp",
            "day_of_month",
        }
    ),
    "number": frozenset({"int", "str", "float", "fraction"}),
    "award": frozenset({"name"}),
    "legal": frozenset({"name", "reference_code"}),
    "product": frozenset({"name"}),
}

ENTITY_TAXONOMY["number"] = frozenset({"int", "str", "float", "fraction", "percent", "proportion"})

# Legacy attributes kept for backward compatibility with older annotations.
# These are accepted by validation, but not part of the current taxonomy.
LEGACY_ENTITY_ATTRIBUTES: Dict[str, frozenset] = {
    "place": frozenset({"nationality"}),
}


_ENTITY_ID_PATTERN = re.compile(r"^([a-z_]+)_(\d+)$")


def parse_entity_id(entity_id: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse ``type_index`` entity IDs, supporting multi-word types."""
    if not entity_id:
        return None, None
    match = _ENTITY_ID_PATTERN.match(entity_id.strip())
    if not match:
        return None, None
    entity_type = match.group(1)
    try:
        entity_index = int(match.group(2))
    except ValueError:
        return None, None
    return entity_type, entity_index


# Shared word-to-number mapping (used by entity extraction, arithmetic evaluation, answer normalization).
WORD_TO_NUMBER: Dict[str, int] = {
    "zero": 0,
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
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

ORDINAL_WORD_TO_NUMBER: Dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
    "fortieth": 40,
    "fiftieth": 50,
    "sixtieth": 60,
    "seventieth": 70,
    "eightieth": 80,
    "ninetieth": 90,
    "hundredth": 100,
}

NUMBER_SURFACE_CARDINAL_DIGITS = "cardinal_digits"
NUMBER_SURFACE_ORDINAL_DIGITS = "ordinal_digits"
NUMBER_SURFACE_CARDINAL_WORDS = "cardinal_words"
NUMBER_SURFACE_ORDINAL_WORDS = "ordinal_words"

_ORDINAL_DIGIT_PATTERN = re.compile(r"^\s*([0-9][0-9,]*)\s*(st|nd|rd|th)\s*$", re.IGNORECASE)
_CARDINAL_DIGIT_PATTERN = re.compile(r"^\s*([0-9][0-9,]*)\s*$")


def parse_integer_surface_number(text: str) -> Optional[int]:
    """Parse an integer surface string, including ordinal digit forms like ``21st``."""
    s = text.strip()
    if not s:
        return None

    ordinal_match = _ORDINAL_DIGIT_PATTERN.match(s)
    if ordinal_match:
        try:
            return int(ordinal_match.group(1).replace(",", ""))
        except ValueError:
            return None

    cardinal_match = _CARDINAL_DIGIT_PATTERN.match(s)
    if cardinal_match:
        try:
            return int(cardinal_match.group(1).replace(",", ""))
        except ValueError:
            return None

    return None


def infer_int_surface_format(text: str) -> Optional[str]:
    """Infer whether an integer annotation uses cardinal or ordinal digits."""
    s = text.strip()
    if not s:
        return None
    if _ORDINAL_DIGIT_PATTERN.match(s):
        return NUMBER_SURFACE_ORDINAL_DIGITS
    if _CARDINAL_DIGIT_PATTERN.match(s):
        return NUMBER_SURFACE_CARDINAL_DIGITS
    return None


def infer_str_surface_format(text: str) -> Optional[str]:
    """Infer whether a word-form number annotation is cardinal or ordinal."""
    s = text.strip().lower().replace(" ", "-")
    if not s:
        return None
    if s in WORD_TO_NUMBER:
        return NUMBER_SURFACE_CARDINAL_WORDS
    if s in ORDINAL_WORD_TO_NUMBER:
        return NUMBER_SURFACE_ORDINAL_WORDS
    if "-" in s:
        first_part, second_part = s.split("-", 1)
        if first_part in WORD_TO_NUMBER and second_part in WORD_TO_NUMBER:
            return NUMBER_SURFACE_CARDINAL_WORDS
        if first_part in WORD_TO_NUMBER and second_part in ORDINAL_WORD_TO_NUMBER:
            return NUMBER_SURFACE_ORDINAL_WORDS
    return None


def parse_word_number(text: str) -> Optional[int]:
    """Parse a word-form number to int, supporting cardinals and ordinals."""
    s = text.strip().lower().replace(" ", "-")
    if not s:
        return None
    if s in WORD_TO_NUMBER:
        return WORD_TO_NUMBER[s]
    if s in ORDINAL_WORD_TO_NUMBER:
        return ORDINAL_WORD_TO_NUMBER[s]
    if "-" in s:
        parts = s.split("-")
        if len(parts) == 2 and all(part in WORD_TO_NUMBER for part in parts):
            return WORD_TO_NUMBER[parts[0]] + WORD_TO_NUMBER[parts[1]]
        if len(parts) == 2 and parts[0] in WORD_TO_NUMBER and parts[1] in ORDINAL_WORD_TO_NUMBER:
            return WORD_TO_NUMBER[parts[0]] + ORDINAL_WORD_TO_NUMBER[parts[1]]
    return None


def render_integer_surface_number(value: int, surface_format: str | None) -> str:
    """Render an integer using a cardinal or ordinal digit surface form."""
    if surface_format == NUMBER_SURFACE_ORDINAL_DIGITS:
        if 10 <= (value % 100) <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
        return f"{value}{suffix}"
    return str(value)


def render_word_surface_number(value: int, surface_format: str | None) -> str:
    """Render an integer using a cardinal or ordinal word surface form."""
    if surface_format == NUMBER_SURFACE_ORDINAL_WORDS:
        return str(num2words(value, to="ordinal"))
    return str(num2words(value))


WEEKDAYS: Tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_WEEKDAY_TO_INDEX: Dict[str, int] = {day.lower(): i for i, day in enumerate(WEEKDAYS)}
_WEEKDAY_ALIASES: Dict[str, str] = {
    "mon": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "wed": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "fri": "friday",
    "sat": "saturday",
    "sun": "sunday",
}


__all__ = [
    "ANNOTATION_ENTITY_TYPE_ORDER",
    "ENTITY_TAXONOMY",
    "FULL_REPLACE_ENTITY_TYPES",
    "LEGACY_ENTITY_ATTRIBUTES",
    "NUMBER_SURFACE_CARDINAL_DIGITS",
    "NUMBER_SURFACE_CARDINAL_WORDS",
    "NUMBER_SURFACE_ORDINAL_DIGITS",
    "NUMBER_SURFACE_ORDINAL_WORDS",
    "ORG_ENTITY_TYPES",
    "PARTIAL_REPLACE_ATTRIBUTES",
    "ORDINAL_WORD_TO_NUMBER",
    "REPLACE_MODE_ALL",
    "REPLACE_MODE_NON_NUMERICAL",
    "REPLACE_MODE_NUMERICAL",
    "REPLACE_MODE_NUMERICAL_TEMPORAL",
    "REPLACE_MODE_TEMPORAL",
    "VALID_ENTITY_TYPES",
    "VALID_REPLACE_MODES",
    "WEEKDAYS",
    "WORD_TO_NUMBER",
    "_WEEKDAY_ALIASES",
    "_WEEKDAY_TO_INDEX",
    "canonicalize_organization_type",
    "infer_int_surface_format",
    "infer_str_surface_format",
    "infer_organization_kind",
    "is_organization_entity_type",
    "parse_entity_id",
    "parse_integer_surface_number",
    "parse_word_number",
    "render_integer_surface_number",
    "render_word_surface_number",
    "replace_mode_label",
]
