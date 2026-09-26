"""Construct schema-aware ground-truth contracts from reviewed questions."""

from __future__ import annotations

from memoreason.benchmark_definition.annotation_runtime import find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection

from .accepted_answer_alias_generation import (
    _composite_expression_aliases,
    _degree_core_aliases,
    _document_surface_org_aliases,
    _entity_aliases,
    _name_like_aliases,
    _profession_modifier_aliases,
    _surface_form_aliases,
)
from .answer_schema_data_contracts import AnswerSpec, UNANSWERABLE, _BOOL_PREFIXES
from .answer_normalization import (
    _canonical_before_after,
    _canonical_date,
    _canonical_quantity,
    _canonical_year,
    _canonical_yes_no,
    _clean_text,
    _dedupe_preserve_order,
    canonicalize_answer,
)


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
    if " positively or negatively " in lowered_question:
        return "span"
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
        for canonical in (canonicalize_answer(answer_schema, accepted_answer) for accepted_answer in accepted_answers)
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
    return _dedupe_preserve_order(_clean_text(answer) for answer in accepted_answer_overrides if _clean_text(answer))
