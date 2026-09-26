# ruff: noqa: RUF001
"""Regular expressions and labels shared by dataset-record validators."""

import re

INLINE_ANNOTATION_PATTERN = re.compile(r"\[[^\]]+?;\s*[^\]]+?\]")
MONTH_NAME_PATTERN = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
SHORT_YEAR_DATE_PATTERN = re.compile(
    rf"\b(?:in|on|at|by|from)\s+\d{{1,2}}\s+{MONTH_NAME_PATTERN}\s+\d{{2}}\b",
    re.IGNORECASE,
)
DOUBLE_YEAR_DATE_PATTERN = re.compile(
    r"\b(?:in|on|at|by|from)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d{4}\b",
    re.IGNORECASE,
)
BIRTH_YEAR_PATTERN = re.compile(r"\bborn\b[^.]{0,80}?(\d{4})", re.IGNORECASE)
AGE_YEAR_PATTERN = re.compile(r"\bat age\s+(\d{1,3})\s+in\s+(?:[A-Za-z]+\s+)?(\d{4})\b", re.IGNORECASE)
ALIAS_TAUTOLOGY_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z0-9'’\" -]{2,}?)\s*,\s*also known as\s+(?P=name)\b")
PAREN_TAUTOLOGY_PATTERN = re.compile(r"\b(?P<name>[A-Z][A-Za-z0-9'’\" -]{2,}?)\s*\(\s*(?P=name)\s*\)")
THOUSANDS_COMMA_PATTERN = re.compile(r"(?<!\d)(\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?!\d)")
BLANKED_NUMTEMP_RENDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bshared the\s{2,}", re.IGNORECASE),
    re.compile(r"\bwon the\s{2,}", re.IGNORECASE),
    re.compile(r"\bin\s+,", re.IGNORECASE),
    re.compile(r"\baged\s+,", re.IGNORECASE),
    re.compile(r"\bdied in\s+,", re.IGNORECASE),
    re.compile(r"\bin\s{2,}different scientific fields", re.IGNORECASE),
    re.compile(r"\blegacy of\s{2,}", re.IGNORECASE),
    re.compile(r"\bfounded .* in\s+,", re.IGNORECASE),
    re.compile(r"\bdeclared\s+ the Year", re.IGNORECASE),
)
FACTUAL_LEAK_ATTRIBUTES = frozenset(
    {
        "full_name",
        "first_name",
        "last_name",
        "name",
        "nationality",
        "demonym",
        "country",
        "state",
        "region",
        "continent",
        "city",
    }
)
GENERIC_SINGLE_TOKEN_NAME_LITERALS = frozenset({"allied", "united"})
QUESTION_TYPE_ALIASES = {
    "arthmetic": "arithmetic",
    "arith": "arithmetic",
    "temporal_reasoning": "temporal",
    "temporal-reasoning": "temporal",
    "temporal reasoning": "temporal",
}
