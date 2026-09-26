# ruff: noqa: RUF001
"""Normalize text after replacing factual mentions with fictional values."""

import re
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import AnnotationParser
from memoreason.benchmark_definition.document_schema import AnnotatedDocument, EntityCollection

from .fictional_document_rendering_constants import (
    _AGE_IN_YEAR_PATTERN,
    _ARTICLELESS_NUMBER_WORDS,
    _BIRTH_YEAR_PATTERN,
    _DEFINITE_ARTICLE_KEEP_TOKENS,
    _DEFINITE_ARTICLE_SINGLETONS,
    _DUPLICATE_PAREN_PATTERN,
    _DUPLICATE_SUFFIXES,
    _GLOBAL_LITERAL_REWRITE_ATTRS,
)
from .fictional_document_surface_rendering import (
    _capitalize_first_alpha,
    _capitalize_sentence_starts,
    _get_fictional_value,
    _is_sentence_start_annotation,
)


def _rewrite_leftover_factual_literals(
    annotated_doc: AnnotatedDocument,
    fictional_entities: EntityCollection,
    generated_text: str,
    generated_questions: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    # Only explicit annotations authorize entity replacement.  Bare source
    # literals must remain visible so the export verifier can flag a
    # missing annotation instead of silently rewriting unrelated prose.
    del annotated_doc, fictional_entities
    normalized_text = _normalize_replaced_text(generated_text)
    normalized_questions: list[dict[str, Any]] = []
    for question_entry in generated_questions:
        updated_entry = dict(question_entry)
        updated_entry["question"] = _normalize_replaced_text(str(question_entry.get("question") or ""))
        normalized_questions.append(updated_entry)
    return normalized_text, normalized_questions


def _normalize_replaced_text(text: str) -> str:
    normalized = _collapse_duplicate_surface_suffixes(text)
    normalized = _fix_indefinite_articles(normalized)
    normalized = _fix_definite_articles(normalized)
    normalized = _collapse_duplicate_definite_articles(normalized)
    normalized = _collapse_duplicate_parentheticals(normalized)
    normalized = _repair_birth_age_chronology(normalized)
    normalized = _capitalize_sentence_starts(normalized)
    return normalized


def _literal_rewrite_map(
    annotated_doc: AnnotatedDocument,
    fictional_entities: EntityCollection,
) -> dict[str, str]:
    replacements: dict[str, str] = {}
    for ann in AnnotationParser.parse_annotations(annotated_doc.document_to_annotate):
        if not ann.attribute or ann.attribute not in _GLOBAL_LITERAL_REWRITE_ATTRS:
            continue
        original_text = " ".join(str(ann.original_text or "").split())
        if len(original_text) < 4 or not re.search(r"[A-Za-z]", original_text):
            continue
        replacement = " ".join(
            _get_fictional_value(
                fictional_entities,
                ann.entity_id,
                ann.attribute,
                ann.original_text,
            ).split()
        )
        if not replacement or replacement == original_text:
            continue
        replacements.setdefault(original_text, replacement)
        if original_text.lower() != original_text:
            replacements.setdefault(original_text.lower(), replacement.lower())
        if " " in original_text and not original_text.endswith("s") and not replacement.endswith("s"):
            replacements.setdefault(f"{original_text}s", f"{replacement}s")
            replacements.setdefault(f"{original_text.lower()}s", f"{replacement.lower()}s")
        if ann.attribute == "full_name":
            original_parts = original_text.split()
            replacement_parts = replacement.split()
            if len(original_parts) >= 2 and len(replacement_parts) >= 2:
                replacements.setdefault(original_parts[-1], replacement_parts[-1])
                replacements.setdefault(original_parts[0], replacement_parts[0])
    return replacements


def _apply_literal_rewrites(text: str, replacements: dict[str, str]) -> str:
    if not replacements:
        return text
    ordered_literals = sorted(replacements, key=lambda value: (-len(value), value))
    pattern = re.compile("|".join(rf"(?<!\w){re.escape(literal)}(?!\w)" for literal in ordered_literals))

    def _replace(match: re.Match[str]) -> str:
        original = match.group(0)
        replacement = replacements.get(original, original)
        if replacement == original:
            return replacement
        if _is_sentence_start_annotation(text, match.start()):
            replacement = _capitalize_first_alpha(replacement)
        return replacement

    return pattern.sub(_replace, text)


def _fix_indefinite_articles(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        article = match.group(1)
        next_word = match.group(2)
        if _should_drop_indefinite_article(next_word):
            return next_word
        needs_an = _needs_an_for_token(next_word)
        desired = "an" if needs_an else "a"
        if article[0].isupper():
            desired = desired.capitalize()
        return f"{desired} {next_word}"

    return re.sub(r"\b([Aa]n?)\s+([A-Za-z0-9][A-Za-z0-9'’.\-]*)\b", _replace, text)


def _should_drop_indefinite_article(token: str) -> bool:
    lowered = str(token or "").strip().lower()
    return lowered in _ARTICLELESS_NUMBER_WORDS


def _needs_an_for_token(token: str) -> bool:
    stripped = str(token or "").strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered[:1] in {"a", "e", "i", "o", "u"}:
        return True
    if stripped[:1].isdigit():
        digits = re.sub(r"\D", "", stripped)
        return digits.startswith(("8", "11", "18"))
    return False


def _collapse_duplicate_surface_suffixes(text: str) -> str:
    normalized = text
    for suffix in _DUPLICATE_SUFFIXES:
        pattern = re.compile(
            rf"\b(?P<phrase>[A-Z][A-Za-z0-9'’\-]*(?:\s+[A-Z][A-Za-z0-9'’\-]*)*\s+{suffix})\s+{suffix}\b",
            flags=re.IGNORECASE,
        )
        normalized = pattern.sub(lambda match: match.group("phrase"), normalized)
    return normalized


def _should_keep_definite_article(phrase: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z'’-]*", phrase)
    if not tokens:
        return True
    lowered = [token.lower() for token in tokens]
    if len(lowered) == 1 and lowered[0] in _DEFINITE_ARTICLE_SINGLETONS:
        return True
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) >= 2 and token.isupper():
            return True
        return False
    if any(token in _DEFINITE_ARTICLE_KEEP_TOKENS for token in lowered):
        return True
    return True


def _fix_definite_articles(text: str) -> str:
    pattern = re.compile(r"\b([Tt]he)\s+([A-Z][A-Za-z'’.\-]*(?:\s+(?:of|the|and|[A-Z][A-Za-z'’.\-]*)){0,8})\b")

    def _replace(match: re.Match[str]) -> str:
        article = match.group(1)
        phrase = match.group(2)
        trailing_context = text[match.end() : match.end() + 40]
        next_token_match = re.match(r"\s+([A-Za-z][A-Za-z'’-]*)", trailing_context)
        if next_token_match and next_token_match.group(1)[:1].islower():
            return match.group(0)
        if next_token_match and next_token_match.group(1).lower() in _DEFINITE_ARTICLE_KEEP_TOKENS:
            return match.group(0)
        if _should_keep_definite_article(phrase):
            return match.group(0)
        replacement = phrase
        if article[:1].isupper() or _is_sentence_start_annotation(text, match.start()):
            replacement = _capitalize_first_alpha(replacement)
        return replacement

    return pattern.sub(_replace, text)


def _collapse_duplicate_definite_articles(text: str) -> str:
    return re.sub(r"\b([Tt]he)\s+[Tt]he\s+", lambda match: f"{match.group(1)} ", text)


def _collapse_duplicate_parentheticals(text: str) -> str:
    return _DUPLICATE_PAREN_PATTERN.sub(lambda match: match.group("name"), text)


def _repair_birth_age_chronology(text: str) -> str:
    birth_match = _BIRTH_YEAR_PATTERN.search(text)
    if not birth_match:
        return text
    birth_year = int(birth_match.group(1))

    def _replace(match: re.Match[str]) -> str:
        rendered_age = int(match.group(1))
        middle = match.group(2)
        event_year = int(match.group(3))
        implied_age = max(0, event_year - birth_year)
        if abs(implied_age - rendered_age) <= 1:
            return match.group(0)
        return f"at age {implied_age} in {middle}{event_year}"

    return _AGE_IN_YEAR_PATTERN.sub(_replace, text)
