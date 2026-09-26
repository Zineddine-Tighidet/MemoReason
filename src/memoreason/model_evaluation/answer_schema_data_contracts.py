"""Schema constants and immutable data contracts for short-answer evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass

UNANSWERABLE = "UNANSWERABLE"
ANSWER_SCHEMAS = frozenset({"yes_no", "before_after", "quantity", "year", "date", "span", "entity_span"})
ANSWER_PARSER_VERSION = "schema_answer_v3"

_ANSWER_TAG_RE = re.compile(r"(?:assistantfinalanswer|final answer|answer)\s*[:\-]", re.IGNORECASE)
_ENTITY_ID_RE = re.compile(r"^(?P<entity_type>[a-z_]+)_\d+$")
_MONTH_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
)
_DAY_MONTH_YEAR_RE = re.compile(rf"\b\d{{1,2}}\s+{_MONTH_PATTERN}\s+\d{{4}}\b", re.IGNORECASE)
_MONTH_DAY_YEAR_RE = re.compile(rf"\b{_MONTH_PATTERN}\s+\d{{1,2}},\s+\d{{4}}\b", re.IGNORECASE)
_YEAR_RE = re.compile(r"\b\d{4}\b")
_TRAILING_ACRONYM_RE = re.compile(r"^(?P<long>.+?)\s*\((?P<short>[A-Z][A-Z0-9&./-]{1,})\)$")
_CHAT_CONTROL_TOKEN_RE = re.compile(r"(?:<\|[^<>|]+\|>|<[^<>|>]+\|>)")
_ACRONYM_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_UNANSWERABLE_VALUES = {
    "cannot",
    "cannot be",
    "unanswerable",
    "cannot be determined",
    "cant be determined",
    "can't be determined",
    "cant",
    "can't",
    "cant be",
    "can't be",
    "cannot determine",
    "cannot answer",
    "not enough information",
    "insufficient information",
    "not stated",
    "not mentioned",
    "unknown",
}
_UNANSWERABLE_PREFIXES = (
    "cannot be determined",
    "cant be determined",
    "can't be determined",
    "cannot determine",
    "cannot answer",
    "not enough information",
    "insufficient information",
    "not stated",
    "not mentioned",
)
_ENTITY_TITLE_PREFIX_RE = re.compile(
    r"^(?:(?:mr|mrs|ms|dr|judge|president|prime minister|senator|minister|professor|sir)\.?\s+)+",
    re.IGNORECASE,
)
_ENTITY_EXPLANATION_SPLIT_RE = re.compile(
    r"^(?P<head>.+?)\s+(?:is|was|were|are|becomes?|became|remains?)\s+.+$", re.IGNORECASE
)
_PERSON_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "vi"}
_QUANTITY_REL_TOL = 1e-4
_QUANTITY_ABS_TOL = 1e-8
_HARMONY_ANALYSIS_PREFIX_RE = re.compile(r"^\s*<\|channel\|>analysis<\|message\|>", re.IGNORECASE)
_HARMONY_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>|]+\|>")
_REASONING_ANSWER_CUE_RE = re.compile(
    r"(?:^|[\n.]\s*)(?:so\s+)?(?:the\s+)?(?:final\s+answer|answer)(?:\s+is)?\s*[:\-]?\s*(?P<answer>.+?)(?=(?:[\n.?!]|$))",
    re.IGNORECASE | re.DOTALL,
)
_UNANSWERABLE_CUE_RE = re.compile(
    r"(?:cannot be determined|cannot determine|not enough information|insufficient information|"
    r"document does not (?:state|say|mention|provide)|no mention|not specified|"
    r"unclear|no (?:date|year|month|day|number|exact date|exact year|exact month|exact number) given|"
    r"not given|not provided|could be less|could be more)",
    re.IGNORECASE,
)
_TRAILING_NUMERIC_REASONING_RE = re.compile(
    r"(?:=|equals?|is)\s*(?P<answer>-?\d+(?:\.\d+)?)\s*(?:[A-Za-z%$€£¥/_-]+)?\s*$",
    re.IGNORECASE,
)
_LEADING_YES_NO_RE = re.compile(r'^[\s"\'`([{<]*\b(?P<answer>yes|no|true|false)\b', re.IGNORECASE)
_LEADING_BEFORE_AFTER_RE = re.compile(
    r'^[\s"\'`([{<]*\b(?P<answer>before|after|earlier|later)\b',
    re.IGNORECASE,
)
_LEADING_ARTICLES = {"the", "a", "an"}
_SHORT_NAME_QUESTION_KEYWORDS = (
    "team",
    "club",
    "franchise",
    "publication",
    "magazine",
    "newspaper",
    "organization",
    "organisation",
    "company",
    "corporation",
    "foundation",
    "trust",
    "initiative",
    "institute",
    "association",
    "school",
    "college",
    "university",
    "hospital",
)
_PROFESSION_QUESTION_KEYWORDS = (
    "profession",
    "occupation",
    "job",
    "work as",
    "worked as",
    "what was",
)
_DEGREE_QUESTION_PREFIXES = (
    "what degree",
    "which degree",
    "what university degree",
    "which university degree",
)
_DEGREE_QUESTION_EXCLUSION_KEYWORDS = (
    "subject",
    "major",
    "field",
    "discipline",
    "specialization",
    "specialisation",
)
_DOCUMENT_SURFACE_ORG_SUFFIXES = (
    "Group",
    "Company",
    "Corporation",
    "Corp.",
    "Corp",
    "Inc.",
    "Inc",
    "Ltd.",
    "Ltd",
    "LLC",
    "PLC",
    "AG",
    "SE",
    "NV",
    "Holdings",
    "Magazine",
    "Tribune",
    "Times",
    "Herald",
    "Post",
    "Journal",
    "Gazette",
    "Chronicle",
    "Observer",
    "Review",
    "Press",
    "Daily",
    "Weekly",
    "Standard",
    "Bulletin",
    "Record",
    "Mirror",
)
_DEGREE_LABEL_RE = re.compile(
    r"^(?:Juris Doctor|Bachelor(?:'s)?(?: of [A-Za-z][A-Za-z' -]+)?|"
    r"Master(?:'s)?(?: of [A-Za-z][A-Za-z' -]+)?|"
    r"Associate(?:'s)?(?: of [A-Za-z][A-Za-z' -]+)?|"
    r"Doctor(?:ate)?(?: of [A-Za-z][A-Za-z' -]+)?)$",
    re.IGNORECASE,
)
_PERSON_ORIGIN_QUALIFIER_RE = re.compile(
    r"^(?P<base>.+?)\s+(?:of|from)\s+(?P<place>[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,4})$"  # noqa: RUF001
)
_POSSESSIVE_DESCRIPTOR_SUFFIX_RE = re.compile(
    r"^(?P<head>.+?)(?:['’]s|s['’])\s+(?P<tail>.+)$"  # noqa: RUF001
)
_UNICODE_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2043": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)
_LOOSE_MATCH_IGNORED_TOKENS = {"a", "an", "the", "and"}
_GENERIC_TRAILING_DESCRIPTOR_TOKENS = {"format"}
_BOOL_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "has ",
    "have ",
    "had ",
    "can ",
    "could ",
    "should ",
    "would ",
    "will ",
)
_ACRONYM_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "de",
    "del",
    "des",
    "di",
    "do",
    "dos",
    "du",
    "for",
    "from",
    "in",
    "la",
    "le",
    "les",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "without",
    "y",
}


@dataclass(frozen=True)
class AnswerSpec:
    """One schema-aware answer contract derived from dataset metadata."""

    answer_schema: str
    ground_truth_canonical: str
    accepted_answers: tuple[str, ...]
    accepted_answers_canonical: tuple[str, ...]


@dataclass(frozen=True)
class ParseResult:
    """One parsed model answer plus audit metadata."""

    parsed_output: str
    canonical_output: str
    parse_status: str
    format_compliant: bool
