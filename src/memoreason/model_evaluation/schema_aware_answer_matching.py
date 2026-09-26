"""Parse generated answers and apply schema-aware exact-match scoring."""

from __future__ import annotations

import math
import re

from memoreason.benchmark_definition.answer_matching import normalize_answer, try_parse_float

from .accepted_answer_alias_generation import (
    _strip_person_origin_qualifier,
    _strip_possessive_descriptor_suffix,
    _trim_entity_candidate,
)
from .answer_normalization import (
    _clean_text,
    _dedupe_preserve_order,
    _is_unanswerable_text,
    _leading_before_after_answer,
    _normalize_dash_like_characters,
    _strip_parenthetical_segments,
    canonicalize_answer,
)
from .answer_schema_data_contracts import (
    _ANSWER_TAG_RE,
    _GENERIC_TRAILING_DESCRIPTOR_TOKENS,
    _HARMONY_ANALYSIS_PREFIX_RE,
    _HARMONY_CONTROL_TOKEN_RE,
    _LOOSE_MATCH_IGNORED_TOKENS,
    _QUANTITY_ABS_TOL,
    _QUANTITY_REL_TOL,
    _REASONING_ANSWER_CUE_RE,
    _TRAILING_NUMERIC_REASONING_RE,
    _UNANSWERABLE_CUE_RE,
    _YEAR_RE,
    UNANSWERABLE,
    ParseResult,
)
from .short_answer_extraction import parse_short_answer


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
        untrimmed_canonical = canonicalize_answer(answer_schema, candidate)
        accepted_canonicals = {
            canonicalize_answer(answer_schema, answer)
            for answer in accepted_answers
            if str(answer).strip()
        }
        if not untrimmed_canonical or untrimmed_canonical not in accepted_canonicals:
            candidate = _trim_entity_candidate(candidate)
    canonical = canonicalize_answer(answer_schema, candidate)
    if answer_schema == "quantity" and accepted_answers:
        numeric_surface_match = re.search(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", candidate.replace(",", ""))
        if numeric_surface_match:
            unscaled_candidate = numeric_surface_match.group(0)
            unscaled_canonical = canonicalize_answer(answer_schema, unscaled_candidate)
            accepted_canonicals = tuple(
                canonicalize_answer(answer_schema, answer)
                for answer in accepted_answers
                if str(answer).strip()
            )
            if score_prediction_with_schema(
                unscaled_canonical,
                accepted_canonicals,
                answer_schema=answer_schema,
                raw_prediction=unscaled_candidate,
            ):
                candidate = unscaled_candidate
                canonical = unscaled_canonical
    parsed_output = _display_answer(answer_schema, candidate, canonical)
    parse_status = "answer_tag" if format_compliant else "fallback"
    recovered = _recover_from_reasoning_payload(
        raw,
        answer_schema=answer_schema,
        accepted_answers=accepted_answers,
    )
    should_use_reasoning_recovery = (
        bool(_HARMONY_ANALYSIS_PREFIX_RE.match(raw)) or not format_compliant or not canonical
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
        decimal_match = re.fullmatch(r"[-+]?\d+\.(?P<fraction>\d+)", accepted.strip())
        display_precision_tolerance = 0.0
        if decimal_match:
            display_precision_tolerance = 0.5 * (10 ** -len(decimal_match.group("fraction")))
        if math.isclose(
            predicted_value,
            accepted_value,
            rel_tol=_QUANTITY_REL_TOL,
            abs_tol=max(_QUANTITY_ABS_TOL, display_precision_tolerance),
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
        if stripped_possessive_descriptor and _clean_text(stripped_possessive_descriptor) != _clean_text(
            raw_prediction or ""
        ):
            stripped_canonical = canonicalize_answer(answer_schema, stripped_possessive_descriptor)
            if score_canonical_prediction(stripped_canonical, accepted_answers_canonical):
                return True
    return False


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


def _accepted_answers_present_in_text(text: str, accepted_answers: tuple[str, ...]) -> tuple[str, ...]:
    matched: list[str] = []
    for answer in sorted(
        (a for a in accepted_answers if a and a != UNANSWERABLE), key=lambda value: (-len(value), value.lower())
    ):
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
