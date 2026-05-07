"""Schema-aware answer specifications, parsing, and scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import re

from src.core.annotation_runtime import find_entity_refs
from src.core.answer_matching import normalize_answer, try_parse_float
from src.core.document_schema import EntityCollection, PersonEntity

from .parsing import parse_short_answer


UNANSWERABLE = "UNANSWERABLE"
ANSWER_SCHEMAS = frozenset({"yes_no", "before_after", "quantity", "year", "date", "span", "entity_span"})

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
_ENTITY_EXPLANATION_SPLIT_RE = re.compile(r"^(?P<head>.+?)\s+(?:is|was|were|are|becomes?|became|remains?)\s+.+$", re.IGNORECASE)
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
_LEADING_BEFORE_AFTER_RE = re.compile(r'^[\s"\'`([{<]*\b(?P<answer>before|after|earlier|later)\b', re.IGNORECASE)
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
    r"^(?P<base>.+?)\s+(?:of|from)\s+(?P<place>[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){0,4})$"
)
_POSSESSIVE_DESCRIPTOR_SUFFIX_RE = re.compile(r"^(?P<head>.+?)(?:['’]s|s['’])\s+(?P<tail>.+)$")
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
    normalized = normalize_answer(_clean_text(text))
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


def _display_answer(answer_schema: str, text: str, canonical: str) -> str:
    if not canonical:
        return _clean_text(text)
    if canonical == UNANSWERABLE or answer_schema in {"yes_no", "before_after", "quantity", "year", "date"}:
        return canonical
    return _clean_text(text).strip("\"' ")


def _parse_schema_answer_impl(
    raw_text: str,
    answer_schema: str,
    *,
    accepted_answers: tuple[str, ...],
) -> ParseResult:
    """Implementation hook that optionally exploits accepted answers for recovery."""
    raw = str(raw_text or "")
    if not raw.strip():
        return ParseResult(parsed_output="", canonical_output="", parse_status="empty", format_compliant=False)

    format_compliant = bool(_ANSWER_TAG_RE.search(raw))
    candidate = parse_short_answer(raw)
    if answer_schema == "entity_span":
        candidate = _trim_entity_candidate(candidate)
    canonical = canonicalize_answer(answer_schema, candidate)
    parsed_output = _display_answer(answer_schema, candidate, canonical)
    parse_status = "answer_tag" if format_compliant else "fallback"
    recovered = _recover_from_reasoning_payload(
        raw,
        answer_schema=answer_schema,
        accepted_answers=accepted_answers,
    )
    should_use_reasoning_recovery = (
        bool(_HARMONY_ANALYSIS_PREFIX_RE.match(raw))
        or not format_compliant
        or not canonical
    )
    if recovered and should_use_reasoning_recovery:
        recovered_canonical = canonicalize_answer(answer_schema, recovered)
        if recovered_canonical:
            canonical = recovered_canonical
            parsed_output = _display_answer(answer_schema, recovered, recovered_canonical)
            if not format_compliant:
                parse_status = "reasoning_fallback"
    if not parsed_output and not canonical:
        parse_status = "empty"
    return ParseResult(
        parsed_output=parsed_output,
        canonical_output=canonical,
        parse_status=parse_status,
        format_compliant=format_compliant,
    )


def parse_schema_answer(
    raw_text: str,
    answer_schema: str,
    *,
    accepted_answers: tuple[str, ...] | list[str] | None = None,
) -> ParseResult:
    """Parse one raw model output using the configured answer schema."""
    coerced_answers = tuple(str(answer).strip() for answer in (accepted_answers or ()) if str(answer).strip())
    return _parse_schema_answer_impl(raw_text, answer_schema, accepted_answers=coerced_answers)


def score_canonical_prediction(predicted_canonical: str, accepted_answers_canonical: tuple[str, ...]) -> bool:
    """Return True when the prediction matches one accepted canonical answer."""
    if not predicted_canonical:
        return False
    return predicted_canonical in set(accepted_answers_canonical)


def quantity_match_is_close(predicted_canonical: str, accepted_answers_canonical: tuple[str, ...]) -> bool:
    """Return True when the quantity is numerically close to one accepted answer."""
    predicted_value = try_parse_float(predicted_canonical)
    if predicted_value is None or not math.isfinite(predicted_value):
        return False

    for accepted in accepted_answers_canonical:
        accepted_value = try_parse_float(accepted)
        if accepted_value is None or not math.isfinite(accepted_value):
            continue
        if math.isclose(
            predicted_value,
            accepted_value,
            rel_tol=_QUANTITY_REL_TOL,
            abs_tol=_QUANTITY_ABS_TOL,
        ):
            return True
    return False


def _loose_match_tokens(text: str) -> tuple[str, ...]:
    cleaned = _normalize_dash_like_characters(_clean_text(text))
    if not cleaned:
        return tuple()
    cleaned = cleaned.lower().replace("/", " ").replace("-", " ")
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return tuple()
    return tuple(token for token in cleaned.split() if token and token not in _LOOSE_MATCH_IGNORED_TOKENS)


def _loose_span_match(predicted_text: str, accepted_texts: tuple[str, ...]) -> bool:
    predicted_tokens = _loose_match_tokens(predicted_text)
    if not predicted_tokens:
        return False
    for accepted_text in accepted_texts:
        accepted_tokens = _loose_match_tokens(accepted_text)
        if not accepted_tokens:
            continue
        if predicted_tokens == accepted_tokens:
            return True
        if (
            len(predicted_tokens) == len(accepted_tokens) + 1
            and predicted_tokens[-1] in _GENERIC_TRAILING_DESCRIPTOR_TOKENS
            and predicted_tokens[:-1] == accepted_tokens
        ):
            return True
        if (
            len(accepted_tokens) == len(predicted_tokens) + 1
            and accepted_tokens[-1] in _GENERIC_TRAILING_DESCRIPTOR_TOKENS
            and accepted_tokens[:-1] == predicted_tokens
        ):
            return True
        predicted_years = tuple(token for token in predicted_tokens if _YEAR_RE.fullmatch(token))
        accepted_years = tuple(token for token in accepted_tokens if _YEAR_RE.fullmatch(token))
        if len(predicted_years) == len(accepted_years) == 1 and predicted_years == accepted_years:
            predicted_without_year = tuple(token for token in predicted_tokens if not _YEAR_RE.fullmatch(token))
            accepted_without_year = tuple(token for token in accepted_tokens if not _YEAR_RE.fullmatch(token))
            if predicted_without_year == accepted_without_year:
                return True
    return False


def score_prediction_with_schema(
    predicted_canonical: str,
    accepted_answers_canonical: tuple[str, ...],
    *,
    answer_schema: str,
    raw_prediction: str | None = None,
) -> bool:
    """Return True when the prediction matches under the schema-aware scorer."""
    if score_canonical_prediction(predicted_canonical, accepted_answers_canonical):
        return True
    if answer_schema == "quantity":
        return quantity_match_is_close(predicted_canonical, accepted_answers_canonical)
    if answer_schema in {"span", "entity_span"}:
        stripped_prediction = _strip_parenthetical_segments(raw_prediction or "")
        if stripped_prediction and _clean_text(stripped_prediction) != _clean_text(raw_prediction or ""):
            stripped_canonical = canonicalize_answer(answer_schema, stripped_prediction)
            if score_canonical_prediction(stripped_canonical, accepted_answers_canonical):
                return True
        stripped_person_qualifier = _strip_person_origin_qualifier(raw_prediction or "")
        if stripped_person_qualifier and _clean_text(stripped_person_qualifier) != _clean_text(raw_prediction or ""):
            stripped_canonical = canonicalize_answer(answer_schema, stripped_person_qualifier)
            if score_canonical_prediction(stripped_canonical, accepted_answers_canonical):
                return True
        if _loose_span_match(predicted_canonical, accepted_answers_canonical):
            return True
    if answer_schema == "entity_span":
        stripped_possessive_descriptor = _strip_possessive_descriptor_suffix(raw_prediction or "")
        if stripped_possessive_descriptor and _clean_text(stripped_possessive_descriptor) != _clean_text(raw_prediction or ""):
            stripped_canonical = canonicalize_answer(answer_schema, stripped_possessive_descriptor)
            if score_canonical_prediction(stripped_canonical, accepted_answers_canonical):
                return True
    return False


def infer_answer_schema(
    *,
    question_text: str,
    answer_expression: str,
    evaluated_answer: str,
) -> str:
    """Infer the answer schema from the exported answer metadata."""
    stripped_expr = str(answer_expression or "").strip()
    lowered_question = str(question_text or "").strip().lower()
    refs = find_entity_refs(stripped_expr)
    if len(refs) == 1 and stripped_expr == refs[0]:
        ref = refs[0]
        _, _, attr = ref.partition(".")
        if attr == "year":
            return "year"
        if attr in {"date", "timestamp"}:
            return "date"
        if attr in {"int", "float", "percent", "proportion", "str", "fraction"}:
            return "quantity"
        return "entity_span"

    yes_no_value = _canonical_yes_no(evaluated_answer)
    if yes_no_value and yes_no_value != UNANSWERABLE:
        return "yes_no"
    before_after_value = _canonical_before_after(evaluated_answer)
    if before_after_value and before_after_value != UNANSWERABLE:
        return "before_after"
    if " before or after " in lowered_question:
        return "before_after"
    if " earlier or later " in lowered_question:
        return "before_after"
    if lowered_question.startswith(_BOOL_PREFIXES):
        return "yes_no"
    if lowered_question.startswith(("how many ", "how much ", "how old ")):
        return "quantity"
    if lowered_question.startswith(("in what year", "what year", "which year")):
        return "year"
    if lowered_question.startswith("when "):
        return "date"
    if lowered_question.startswith("who ") or "name of the person" in lowered_question:
        return "entity_span"
    if lowered_question.startswith("where ") or "what is the name of" in lowered_question:
        return "entity_span"
    date_value = _canonical_date(evaluated_answer)
    if date_value and date_value != UNANSWERABLE:
        return "date"
    integer_value = _canonical_quantity(evaluated_answer)
    if integer_value and integer_value != UNANSWERABLE:
        return "quantity"
    year_value = _canonical_year(evaluated_answer)
    if year_value and year_value != UNANSWERABLE:
        return "year"
    return "span"


def build_answer_spec(
    *,
    question_text: str,
    answer_expression: str,
    evaluated_answer: str,
    document_text: str = "",
    entities_used: EntityCollection | None,
    accepted_answer_overrides: tuple[str, ...] | list[str] | None = None,
) -> AnswerSpec:
    """Build the schema-aware answer contract for one benchmark question."""
    answer_schema = infer_answer_schema(
        question_text=question_text,
        answer_expression=answer_expression,
        evaluated_answer=evaluated_answer,
    )
    ground_truth_canonical = canonicalize_answer(answer_schema, evaluated_answer)
    accepted_answers = _build_accepted_answers(
        answer_schema=answer_schema,
        question_text=question_text,
        answer_expression=answer_expression,
        evaluated_answer=evaluated_answer,
        document_text=document_text,
        entities_used=entities_used,
        accepted_answer_overrides=accepted_answer_overrides,
    )
    accepted_answers_canonical = tuple(
        canonical
        for canonical in (
            canonicalize_answer(answer_schema, accepted_answer) for accepted_answer in accepted_answers
        )
        if canonical
    )
    if not accepted_answers_canonical and ground_truth_canonical:
        accepted_answers_canonical = (ground_truth_canonical,)
    return AnswerSpec(
        answer_schema=answer_schema,
        ground_truth_canonical=ground_truth_canonical,
        accepted_answers=accepted_answers,
        accepted_answers_canonical=_dedupe_preserve_order(accepted_answers_canonical),
    )


def _build_accepted_answers(
    *,
    answer_schema: str,
    question_text: str,
    answer_expression: str,
    evaluated_answer: str,
    document_text: str,
    entities_used: EntityCollection | None,
    accepted_answer_overrides: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    override_answers = _coerce_accepted_answer_overrides(accepted_answer_overrides)
    if canonicalize_answer(answer_schema, evaluated_answer) == UNANSWERABLE:
        return (UNANSWERABLE,)

    if answer_schema == "yes_no":
        canonical = _canonical_yes_no(evaluated_answer)
        return (canonical,) if canonical else tuple()
    if answer_schema == "before_after":
        canonical = _canonical_before_after(evaluated_answer)
        return (canonical,) if canonical else tuple()
    if answer_schema == "quantity":
        canonical = _canonical_quantity(evaluated_answer)
        accepted = []
        cleaned_answer = _clean_text(evaluated_answer)
        if cleaned_answer:
            accepted.append(cleaned_answer)
        accepted.extend(override_answers)
        if canonical:
            accepted.append(canonical)
        return _dedupe_preserve_order(answer for answer in accepted if answer)
    if answer_schema == "year":
        canonical = _canonical_year(evaluated_answer)
        return (canonical,) if canonical else tuple()
    if answer_schema == "date":
        canonical = _canonical_date(evaluated_answer)
        return (canonical,) if canonical else tuple()

    accepted = [_clean_text(evaluated_answer), *override_answers]
    if answer_schema in {"span", "entity_span"}:
        accepted.extend(_name_like_aliases(_clean_text(evaluated_answer)))
        accepted.extend(_degree_core_aliases(_clean_text(evaluated_answer), question_text))
        accepted.extend(_surface_form_aliases(_clean_text(evaluated_answer), question_text))
        accepted.extend(
            _document_surface_org_aliases(
                _clean_text(evaluated_answer),
                question_text=question_text,
                document_text=document_text,
            )
        )
        if entities_used is not None:
            accepted.extend(_profession_modifier_aliases(_clean_text(evaluated_answer), question_text, entities_used))
    if entities_used is not None:
        refs = find_entity_refs(str(answer_expression or "").strip())
        if answer_schema == "entity_span" and len(refs) == 1 and str(answer_expression or "").strip() == refs[0]:
            accepted.extend(_entity_aliases(refs[0], question_text, entities_used))
        accepted.extend(_composite_expression_aliases(answer_expression, question_text, entities_used))
    return _dedupe_preserve_order(answer for answer in accepted if answer)


def _coerce_accepted_answer_overrides(
    accepted_answer_overrides: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if not accepted_answer_overrides:
        return tuple()
    return _dedupe_preserve_order(
        _clean_text(answer)
        for answer in accepted_answer_overrides
        if _clean_text(answer)
    )


def _surface_form_aliases(value: str, question_text: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []

    aliases = [cleaned]
    lowered_question = str(question_text or "").strip().lower()
    lowered_value = cleaned.lower()

    for prefix in ("the ", "its ", "a ", "an "):
        if lowered_value.startswith(prefix):
            aliases.append(cleaned[len(prefix) :].strip())

    if lowered_value.endswith(" itself"):
        aliases.append(cleaned[:-7].strip())

    if "degree" in lowered_question and re.search(r"\bdegree\b", cleaned, flags=re.IGNORECASE):
        aliases.append(re.sub(r"\bdegree\b\s*", "", cleaned, flags=re.IGNORECASE).strip())

    if "trophy" in lowered_question:
        if lowered_value.endswith(" trophy"):
            aliases.append(cleaned[: -len(" Trophy")].strip())
        else:
            aliases.append(f"{cleaned} Trophy")

    if " " not in cleaned:
        if lowered_value.endswith("s") and len(cleaned) > 3:
            aliases.append(cleaned[:-1])
        elif len(cleaned) > 2:
            aliases.append(f"{cleaned}s")
        if lowered_value.endswith("ially") and len(cleaned) > 5:
            aliases.append(cleaned[:-2])
        elif lowered_value.endswith("ial") and len(cleaned) > 4:
            aliases.append(f"{cleaned}ly")

    if lowered_value.endswith("-shaped"):
        aliases.append(cleaned[: -len("-shaped")].strip())

    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _degree_core_aliases(value: str, question_text: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned or not _question_allows_degree_core_alias(question_text):
        return []

    core = ""
    degree_in_match = re.match(r"^(?P<core>.+?)\s+degree\s+in\s+.+$", cleaned, flags=re.IGNORECASE)
    if degree_in_match:
        candidate = re.sub(r"\s+", " ", degree_in_match.group("core")).strip()
        if _DEGREE_LABEL_RE.match(candidate):
            core = candidate
    if not core:
        plain_in_match = re.match(r"^(?P<core>.+?)\s+in\s+.+$", cleaned, flags=re.IGNORECASE)
        if plain_in_match:
            candidate = re.sub(r"\s+", " ", plain_in_match.group("core")).strip()
            if _DEGREE_LABEL_RE.match(candidate):
                core = candidate
    if not core:
        return []

    if not core or normalize_answer(core) == normalize_answer(cleaned):
        return []
    return [core]


def _question_allows_degree_core_alias(question_text: str) -> bool:
    lowered = str(question_text or "").strip().lower()
    if "degree" not in lowered:
        return False
    if any(keyword in lowered for keyword in _DEGREE_QUESTION_EXCLUSION_KEYWORDS):
        return False
    return any(lowered.startswith(prefix) for prefix in _DEGREE_QUESTION_PREFIXES)


def _document_surface_org_aliases(
    value: str,
    *,
    question_text: str,
    document_text: str,
) -> list[str]:
    cleaned = _clean_text(value)
    source_text = str(document_text or "")
    if not cleaned or not source_text.strip():
        return []
    if not _question_allows_short_name_alias(question_text):
        return []
    if len(cleaned.split()) > 3:
        return []

    escaped_base = re.escape(cleaned)
    candidates: list[str] = []
    for suffix in _DOCUMENT_SURFACE_ORG_SUFFIXES:
        pattern = re.compile(
            rf"(?<!\w){escaped_base}\s+{re.escape(suffix)}(?!\w)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(source_text):
            candidate = re.sub(r"\s+", " ", match.group(0)).strip(" \t\r\n,.;:!?")
            if candidate:
                candidates.append(candidate)

    if not candidates:
        return []

    deduped_candidates: list[str] = []
    seen_canonical: set[str] = set()
    base_canonical = canonicalize_answer("entity_span", cleaned)
    for candidate in candidates:
        candidate_canonical = canonicalize_answer("entity_span", candidate)
        if not candidate_canonical or candidate_canonical == base_canonical:
            continue
        if candidate_canonical in seen_canonical:
            continue
        seen_canonical.add(candidate_canonical)
        deduped_candidates.append(candidate)

    if len(deduped_candidates) != 1:
        return []
    return deduped_candidates


def _entity_ref_value(entity_ref: str, entities_used: EntityCollection) -> str | None:
    entity_id, _, attr = entity_ref.partition(".")
    if not entity_id or not attr:
        return None
    entity_type = _entity_type_from_id(entity_id)
    if entity_type is None:
        return None
    try:
        collection = entities_used.get_collection(entity_type)
    except ValueError:
        return None
    entity = collection.get(entity_id)
    if entity is None:
        return None
    value = getattr(entity, attr, None)
    if value is None:
        return None
    return _clean_text(value)


def _composite_expression_aliases(
    answer_expression: str,
    question_text: str,
    entities_used: EntityCollection,
) -> list[str]:
    stripped_expr = str(answer_expression or "").strip()
    refs = find_entity_refs(stripped_expr)
    if len(refs) < 2:
        return []

    expression_skeleton = stripped_expr
    for ref in refs:
        expression_skeleton = expression_skeleton.replace(ref, "REF")
    collapsed_skeleton = re.sub(r"\s+", "", expression_skeleton)
    if any(operator in collapsed_skeleton for operator in ("+", "-", "*", "/")):
        return []
    if re.search(r"[A-Za-z0-9_]+\s*\(", expression_skeleton):
        return []

    values = [_entity_ref_value(ref, entities_used) for ref in refs]
    if any(not value for value in values):
        return []
    resolved_values = [value for value in values if value]

    aliases: list[str] = []
    if len(resolved_values) == 2:
        first, second = resolved_values
        aliases.extend(
            (
                f"{first} and {second}",
                f"{second} and {first}",
                f"{first}, {second}",
                f"{second}, {first}",
                f"('{first}', '{second}')",
                f"('{second}', '{first}')",
            )
        )
    else:
        aliases.append(", ".join(resolved_values))

    lowered_question = str(question_text or "").strip().lower()
    if "which two" in lowered_question or "what two" in lowered_question:
        aliases.extend(resolved_values)

    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _entity_aliases(entity_ref: str, question_text: str, entities_used: EntityCollection) -> list[str]:
    entity_id, _, attr = entity_ref.partition(".")
    if not entity_id or not attr:
        return []
    entity_type = _entity_type_from_id(entity_id)
    if entity_type is None:
        return []

    try:
        collection = entities_used.get_collection(entity_type)
    except ValueError:
        return []
    entity = collection.get(entity_id)
    if entity is None:
        return []

    if entity_type != "person":
        value = _clean_text(getattr(entity, attr, None))
        aliases = [value] if value else []
        aliases.extend(_name_like_aliases(value))
        if attr == "name" and _question_allows_short_name_alias(question_text):
            leading_alias = _leading_name_alias(value)
            if leading_alias and _is_unique_non_person_leading_alias(collection, leading_alias):
                aliases.append(leading_alias)
        return _dedupe_preserve_order(alias for alias in aliases if alias)

    person = entities_used.persons.get(entity_id)
    if person is None:
        return []
    requested_component = _requested_person_name_component(question_text)
    if requested_component is not None:
        component_value = _person_component_value(person, requested_component)
        return [component_value] if component_value else []

    aliases = []
    if person.full_name:
        aliases.append(person.full_name)
    full_name_without_suffix = _person_full_name_without_suffix(person)
    if full_name_without_suffix:
        aliases.append(full_name_without_suffix)
    first_last = _first_last_name(person)
    if first_last:
        aliases.append(first_last)
    first_token = _person_first_token(person)
    if first_token and _is_unique_person_first_token(entities_used, first_token):
        aliases.append(first_token)
    if person.last_name and _is_unique_person_attribute(entities_used, "last_name", person.last_name):
        aliases.append(person.last_name)
    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _requested_person_name_component(question_text: str) -> str | None:
    lowered = str(question_text or "").strip().lower()
    if "full name" in lowered:
        return "full_name"
    if "first name" in lowered or "given name" in lowered:
        return "first_name"
    if "last name" in lowered or "surname" in lowered or "family name" in lowered:
        return "last_name"
    if "middle name" in lowered:
        return "middle_name"
    return None


def _person_component_value(person: PersonEntity, component: str) -> str | None:
    if component == "full_name":
        if person.full_name:
            return person.full_name
        return _first_last_name(person)
    if component == "first_name":
        return _person_first_token(person)
    value = getattr(person, component, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_last_name(person: PersonEntity) -> str | None:
    first_name = _person_first_token(person)
    if first_name and person.last_name:
        return f"{first_name} {person.last_name}".strip()
    return None


def _person_first_token(person: PersonEntity) -> str | None:
    if person.first_name and person.first_name.strip():
        return person.first_name.strip()
    if not person.full_name:
        return None
    tokens = _person_name_tokens_without_suffix(person.full_name)
    if not tokens:
        return None
    return tokens[0]


def _person_full_name_without_suffix(person: PersonEntity) -> str | None:
    if not person.full_name:
        return None
    tokens = _person_name_tokens_without_suffix(person.full_name)
    if not tokens:
        return None
    candidate = " ".join(tokens).strip()
    if candidate == person.full_name.strip():
        return None
    return candidate


def _person_name_tokens_without_suffix(full_name: str) -> list[str]:
    tokens = [token for token in _clean_text(full_name).split() if token]
    while tokens:
        suffix = tokens[-1].rstrip(".").lower()
        if suffix not in _PERSON_SUFFIXES:
            break
        tokens.pop()
    return tokens


def _is_unique_person_attribute(entities_used: EntityCollection, attribute: str, value: str) -> bool:
    target = normalize_answer(value)
    matches = 0
    for person in entities_used.persons.values():
        candidate = getattr(person, attribute, None)
        if isinstance(candidate, str) and normalize_answer(candidate) == target:
            matches += 1
    return matches == 1


def _is_unique_person_first_token(entities_used: EntityCollection, value: str) -> bool:
    target = normalize_answer(value)
    matches = 0
    for person in entities_used.persons.values():
        candidate = _person_first_token(person)
        if candidate and normalize_answer(candidate) == target:
            matches += 1
    return matches == 1


def _name_like_aliases(value: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []
    aliases = [cleaned]
    acronym_match = _TRAILING_ACRONYM_RE.match(cleaned)
    if acronym_match:
        aliases.append(acronym_match.group("long").strip())
        aliases.append(acronym_match.group("short").strip())
    else:
        derived_acronym = _derive_acronym(cleaned) if _should_add_parenthetical_acronym_alias(cleaned) else None
        if derived_acronym:
            aliases.append(f"{cleaned} ({derived_acronym})")
    return _dedupe_preserve_order(alias for alias in aliases if alias)


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


def _question_allows_short_name_alias(question_text: str) -> bool:
    lowered = str(question_text or "").strip().lower()
    return any(keyword in lowered for keyword in _SHORT_NAME_QUESTION_KEYWORDS)


def _leading_name_alias(value: str) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    tokens = [token for token in cleaned.split() if token]
    while tokens and tokens[0].lower() in _LEADING_ARTICLES:
        tokens.pop(0)
    if len(tokens) < 2:
        return None
    return tokens[0]


def _is_unique_non_person_leading_alias(collection: dict[str, object], alias: str) -> bool:
    target = normalize_answer(alias)
    matches = 0
    for entity in collection.values():
        name = _clean_text(getattr(entity, "name", None))
        candidate = _leading_name_alias(name)
        if candidate and normalize_answer(candidate) == target:
            matches += 1
    return matches == 1


def _profession_modifier_aliases(value: str, question_text: str, entities_used: EntityCollection) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned or not _question_allows_profession_alias(question_text):
        return []
    if cleaned.lower() != cleaned:
        return []
    if len(cleaned.split()) > 3:
        return []

    modifiers: list[str] = []
    for person in entities_used.persons.values():
        if person.nationality and person.nationality.strip():
            modifiers.append(person.nationality.strip())
    for place in entities_used.places.values():
        for attr in ("demonym", "nationality"):
            modifier = getattr(place, attr, None)
            if isinstance(modifier, str) and modifier.strip():
                modifiers.append(modifier.strip())

    return _dedupe_preserve_order(f"{modifier} {cleaned}" for modifier in modifiers if modifier)


def _question_allows_profession_alias(question_text: str) -> bool:
    lowered = str(question_text or "").strip().lower()
    if "profession" in lowered or "occupation" in lowered or "job" in lowered:
        return True
    return "what was" in lowered and any(keyword in lowered for keyword in ("profession", "occupation", "job"))


def _strip_person_origin_qualifier(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    match = _PERSON_ORIGIN_QUALIFIER_RE.match(cleaned)
    if not match:
        return ""
    base = re.sub(r"\s+", " ", match.group("base")).strip()
    place = re.sub(r"\s+", " ", match.group("place")).strip()
    if len(base.split()) < 2:
        return ""
    if not place:
        return ""
    return base


def _strip_possessive_descriptor_suffix(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    match = _POSSESSIVE_DESCRIPTOR_SUFFIX_RE.match(cleaned)
    if not match:
        return ""
    head = re.sub(r"\s+", " ", match.group("head")).strip()
    tail = re.sub(r"\s+", " ", match.group("tail")).strip()
    if len(head.split()) < 2:
        return ""
    if len(tail.split()) < 2 and not any(character.isdigit() for character in tail):
        return ""
    return head


def _trim_entity_candidate(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (",", ";", " — ", " – "):
        if separator in cleaned:
            head = cleaned.split(separator, 1)[0].strip()
            if 1 <= len(head.split()) <= 8:
                cleaned = head
                break
    split_match = _ENTITY_EXPLANATION_SPLIT_RE.match(cleaned)
    if split_match:
        head = split_match.group("head").strip()
        if 1 <= len(head.split()) <= 8:
            return head
    return cleaned


def _recover_from_reasoning_payload(
    raw_text: str,
    *,
    answer_schema: str,
    accepted_answers: tuple[str, ...],
) -> str:
    raw = str(raw_text or "")
    if not raw.strip():
        return ""

    cleaned = _HARMONY_CONTROL_TOKEN_RE.sub(" ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""

    cue_match = _REASONING_ANSWER_CUE_RE.search(cleaned)
    if cue_match:
        return _clean_text(cue_match.group("answer"))

    if _contains_unanswerable_cue(cleaned):
        return "Cannot be determined"

    matched_accepted_answers = _collapse_nested_answer_matches(
        _accepted_answers_present_in_text(cleaned, accepted_answers)
    )
    if len(matched_accepted_answers) == 1:
        return matched_accepted_answers[0]

    if answer_schema == "yes_no":
        lowered = cleaned.lower()
        if re.search(r"\bso\b.{0,20}\byes\b", lowered):
            return "Yes"
        if re.search(r"\bso\b.{0,20}\bno\b", lowered):
            return "No"

    if answer_schema == "before_after":
        lowered = cleaned.lower()
        leading_before_after = _leading_before_after_answer(cleaned)
        if leading_before_after:
            return leading_before_after
        if re.search(r"\bso\b.{0,20}\bbefore\b", lowered):
            return "Before"
        if re.search(r"\bso\b.{0,20}\bafter\b", lowered):
            return "After"

    if answer_schema in {"quantity", "year"}:
        trailing_match = _TRAILING_NUMERIC_REASONING_RE.search(cleaned.rstrip(".?! "))
        if trailing_match:
            return trailing_match.group("answer").strip()

    return ""


def _contains_unanswerable_cue(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    if _is_unanswerable_text(cleaned):
        return True
    return bool(_UNANSWERABLE_CUE_RE.search(cleaned))


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
    if any(re.search(rf"\b{opposite_answer}\b", opening_fragment, re.IGNORECASE) for opposite_answer in opposite_answers):
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
    if any(re.search(rf"\b{opposite_answer}\b", opening_fragment, re.IGNORECASE) for opposite_answer in opposite_answers):
        return ""
    return "Before" if leading_answer in {"before", "earlier"} else "After"


def _accepted_answers_present_in_text(text: str, accepted_answers: tuple[str, ...]) -> tuple[str, ...]:
    matched: list[str] = []
    for answer in sorted((a for a in accepted_answers if a and a != UNANSWERABLE), key=lambda value: (-len(value), value.lower())):
        escaped = re.escape(answer)
        if re.fullmatch(r"[\w\s&./:-]+", answer):
            pattern = re.compile(rf"(?<!\w){escaped}(?:'s)?(?!\w)", flags=re.IGNORECASE)
        else:
            pattern = re.compile(escaped, flags=re.IGNORECASE)
        if pattern.search(text):
            matched.append(answer)
    return _dedupe_preserve_order(matched)


def _collapse_nested_answer_matches(matched_answers: tuple[str, ...]) -> tuple[str, ...]:
    if len(matched_answers) <= 1:
        return matched_answers

    filtered: list[str] = []
    for answer in matched_answers:
        normalized_answer = normalize_answer(answer)
        is_contained = False
        for other in matched_answers:
            if other == answer:
                continue
            normalized_other = normalize_answer(other)
            if len(normalized_other) <= len(normalized_answer):
                continue
            if re.search(rf"(?<!\w){re.escape(normalized_answer)}(?!\w)", normalized_other):
                is_contained = True
                break
        if not is_contained:
            filtered.append(answer)
    if not filtered:
        return matched_answers
    return _dedupe_preserve_order(filtered)


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


def answer_format_instructions(answer_schema: str) -> str:
    """Return the schema-specific output contract used in prompting."""
    if answer_schema == "yes_no":
        return (
            "Expected answer type: YES_NO\n"
            "Allowed values:\n"
            "- YES\n"
            "- NO\n"
            f"- {UNANSWERABLE}"
        )
    if answer_schema == "before_after":
        return (
            "Expected answer type: BEFORE_AFTER\n"
            "Allowed values:\n"
            "- BEFORE\n"
            "- AFTER\n"
            f"- {UNANSWERABLE}"
        )
    if answer_schema == "quantity":
        return (
            "Expected answer type: QUANTITY\n"
            "Rules:\n"
            "- Output a quantity only.\n"
            "- Digits or number words are both acceptable.\n"
            "- No units or explanation.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    if answer_schema == "year":
        return (
            "Expected answer type: YEAR\n"
            "Rules:\n"
            "- Output one 4-digit year only.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    if answer_schema == "date":
        return (
            "Expected answer type: DATE\n"
            "Rules:\n"
            "- Output one date only.\n"
            "- Prefer the exact date stated in the document.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    if answer_schema == "entity_span":
        return (
            "Expected answer type: ENTITY_SPAN\n"
            "Rules:\n"
            "- Copy the shortest entity name span from the document.\n"
            "- Do not paraphrase.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    return (
        "Expected answer type: TEXT_SPAN\n"
        "Rules:\n"
        "- Copy the shortest answer span from the document.\n"
        "- Do not paraphrase.\n"
        f"- If the document does not determine the answer, output {UNANSWERABLE}."
    )


def suggested_max_tokens_for_schema(answer_schema: str) -> int:
    """Return a conservative decoding budget for one answer schema."""
    if answer_schema in {"yes_no", "before_after"}:
        return 4
    if answer_schema in {"quantity", "year"}:
        return 8
    if answer_schema == "date":
        return 16
    return 24
