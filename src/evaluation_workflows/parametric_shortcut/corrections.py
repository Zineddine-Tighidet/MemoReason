"""Helpers for applying manual answer corrections from review YAML files."""

from __future__ import annotations

from collections.abc import Sequence

from src.core.answer_evaluation import AnswerEvaluator
from src.core.document_schema import EntityCollection


def normalize_corrected_answer_expression(value: object) -> tuple[str, ...]:
    """Normalize one correction payload into a non-empty tuple of answer entries."""
    if value is None:
        return tuple()
    if isinstance(value, str):
        cleaned = value.strip()
        return (cleaned,) if cleaned else tuple()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        cleaned_entries = tuple(str(item).strip() for item in value if str(item).strip())
        return cleaned_entries
    cleaned = str(value).strip()
    return (cleaned,) if cleaned else tuple()


def split_corrected_answer_entries(
    value: object,
    *,
    entities: EntityCollection,
) -> tuple[str, tuple[str, ...]]:
    """Return ``(canonical_expression, accepted_answer_overrides)`` for one correction payload.

    The first list item is treated as the canonical template answer expression. Any remaining
    items are resolved against the factual entities and stored as literal accepted-answer
    overrides so the template remains compatible with the existing export/evaluation pipeline.
    """
    entries = normalize_corrected_answer_expression(value)
    if not entries:
        return "", tuple()

    canonical_expression = entries[0]
    canonical_surface = str(AnswerEvaluator.evaluate_answer(canonical_expression, entities) or "").strip()

    overrides: list[str] = []
    seen = {canonical_surface} if canonical_surface else set()
    for entry in entries[1:]:
        resolved = str(AnswerEvaluator.evaluate_answer(entry, entities) or "").strip()
        if not resolved or resolved in seen:
            continue
        seen.add(resolved)
        overrides.append(resolved)
    return canonical_expression, tuple(overrides)
