"""Load the benchmark documents used for model evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import yaml

from memoreason import PROJECT_ROOT_DIRECTORY
from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.answer_expression_evaluation import AnswerEvaluator
from memoreason.factual_to_fictional_dataset.dataset_settings import parse_dataset_setting
from memoreason.factual_to_fictional_dataset.dataset_paths import (
    format_document_variant_id,
    iter_document_variant_paths,
    split_document_variant_stem,
    template_path_by_identity,
    unique_question_key,
)
from .ground_truth_answer_specification import build_answer_spec
from .benchmark_scoring_references import load_scoring_references


_QUESTION_TYPE_ALIASES = {
    "arthmetic": "arithmetic",
    "arith": "arithmetic",
    "temporal_reasoning": "temporal",
    "temporal-reasoning": "temporal",
    "temporal reasoning": "temporal",
}

# `bankreg_11` is an incomplete benchmark outlier: it has fictional variants on disk
# but no matching factual document, so we exclude it from evaluation runs.
EXCLUDED_EVALUATION_DOCUMENT_IDS = frozenset({"bankreg_11"})
_ANNOTATED_SPAN = re.compile(r"\[([^;\]]+);\s*[^\]]+\]")


def normalize_question_type(question_type: str | None) -> str:
    """Normalize question-type labels from legacy templates."""
    if not question_type:
        return "unknown"
    cleaned = str(question_type).strip().lower()
    return _QUESTION_TYPE_ALIASES.get(cleaned, cleaned)


def answer_behavior_label(
    answer_type: str | bool | None = None,
    is_answer_invariant: bool | None = None,
) -> str:
    """Return normalized answer behavior label (variant, invariant, refusal)."""
    if isinstance(answer_type, bool) and is_answer_invariant is None:
        is_answer_invariant = answer_type
        answer_type = None

    cleaned = str(answer_type or "").strip().lower()
    if cleaned in {"variant", "invariant", "refusal"}:
        return cleaned
    return "invariant" if bool(is_answer_invariant) else "variant"


@dataclass(frozen=True)
class BenchmarkQuestionForEvaluation:
    """One benchmark question ready for model evaluation."""

    question_id: str
    question_type: str
    answer_behavior: str
    question_text: str
    ground_truth: str
    ground_truth_canonical: str
    answer_schema: str
    accepted_answers: tuple[str, ...]
    accepted_answers_canonical: tuple[str, ...]
    accepted_answer_overrides: tuple[str, ...]
    answer_expression: str
    pair_key: str


@dataclass(frozen=True)
class BenchmarkDocumentForEvaluation:
    """One benchmark document ready for evaluation."""

    document_id: str
    document_theme: str
    document_setting: str
    document_setting_family: str
    document_variant_id: str
    document_variant_index: int
    replacement_proportion: float
    document_text: str
    source_path: Path
    questions: list[BenchmarkQuestionForEvaluation]


def _coerce_ground_truth(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_accepted_answer_overrides(value: object) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        cleaned = str(value).strip()
        return (cleaned,) if cleaned else tuple()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    cleaned = str(value).strip()
    return (cleaned,) if cleaned else tuple()


def _looks_unresolved_ground_truth(value: str, *, answer_expression: str) -> bool:
    cleaned_value = _coerce_ground_truth(value)
    if not cleaned_value:
        return True
    if find_entity_refs(cleaned_value):
        return True
    cleaned_expression = _coerce_ground_truth(answer_expression)
    if cleaned_expression and cleaned_value == cleaned_expression:
        return True
    return False


def _render_composite_ref_expression(answer_expression: str, entities_used: EntityCollection) -> str:
    """Render textual answer expressions containing multiple entity references.

    ``AnswerEvaluator`` intentionally focuses on arithmetic/rule expressions. Some
    curated answers are textual composites such as
    ``place_3.country, place_9.country, and place_10.country``; if left unresolved,
    the judge ends up comparing model outputs against symbolic references.
    """
    cleaned_expression = _coerce_ground_truth(answer_expression)
    refs = find_entity_refs(cleaned_expression)
    if not refs:
        return ""

    rendered = cleaned_expression
    for ref in sorted(set(refs), key=len, reverse=True):
        value = RuleEngine._get_entity_value(entities_used, ref)
        if value is None:
            return ""
        rendered = rendered.replace(ref, str(value).strip())
    if _looks_unresolved_ground_truth(rendered, answer_expression=cleaned_expression):
        return ""
    return rendered.strip()


def _resolved_ground_truth(answer_expression: str, stored_ground_truth: str, entities_used: EntityCollection) -> str:
    cleaned_expression = _coerce_ground_truth(answer_expression)
    cleaned_ground_truth = _coerce_ground_truth(stored_ground_truth)
    computed_ground_truth = _coerce_ground_truth(AnswerEvaluator.evaluate_answer(cleaned_expression, entities_used))
    composite_ground_truth = _render_composite_ref_expression(cleaned_expression, entities_used)
    if computed_ground_truth and not _looks_unresolved_ground_truth(
        computed_ground_truth,
        answer_expression=cleaned_expression,
    ):
        if _looks_unresolved_ground_truth(cleaned_ground_truth, answer_expression=cleaned_expression):
            return computed_ground_truth
    if composite_ground_truth and (
        _looks_unresolved_ground_truth(cleaned_ground_truth, answer_expression=cleaned_expression)
        or _looks_unresolved_ground_truth(computed_ground_truth, answer_expression=cleaned_expression)
    ):
        return composite_ground_truth
    return cleaned_ground_truth or computed_ground_truth


def is_excluded_evaluation_document(document_id: str) -> bool:
    """Return whether one document id should be skipped during evaluation."""
    return str(document_id).strip() in EXCLUDED_EVALUATION_DOCUMENT_IDS


def _normalized_text(value: object) -> str:
    collapsed = " ".join(str(value or "").split())
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", collapsed)


def _render_factual_question(value: object) -> str:
    return _normalized_text(_ANNOTATED_SPAN.sub(r"\1", str(value or "")))


@lru_cache(maxsize=None)
def _load_human_question_contracts(template_path: Path) -> dict[str, dict[str, str]]:
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    document = payload.get("document")
    if not isinstance(document, dict):
        raise ValueError(f"{template_path}: human-reviewed template has no document mapping.")
    contracts: dict[str, dict[str, str]] = {}
    for index, question in enumerate(document.get("questions") or []):
        if not isinstance(question, dict):
            raise ValueError(f"{template_path}: human-reviewed question {index} is not a mapping.")
        question_id = str(question.get("question_id") or "").strip()
        if not question_id or question_id in contracts:
            raise ValueError(f"{template_path}: missing or duplicate human question_id {question_id!r}.")
        answer_behavior = str(question.get("answer_type") or "").strip().lower()
        if answer_behavior not in {"variant", "invariant", "refusal"}:
            raise ValueError(f"{template_path}: invalid human answer_type {answer_behavior!r} for {question_id}.")
        contracts[question_id] = {
            "question_type": normalize_question_type(question.get("question_type")),
            "answer_behavior": answer_behavior,
            "question_text_factual": _render_factual_question(question.get("question")),
        }
    return contracts


def _resolve_human_template_path(
    payload: dict,
    *,
    document_theme: str,
    document_id: str,
) -> Path:
    raw_path = str(payload.get("source_template_path") or "").strip()
    if raw_path:
        candidate = Path(raw_path).expanduser()
        path = candidate if candidate.is_absolute() else PROJECT_ROOT_DIRECTORY / candidate
    else:
        path = template_path_by_identity(document_theme, document_id)
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Human-reviewed template is required for {document_theme}/{document_id}: {path}")
    return path


def load_evaluation_document(document_path: Path) -> BenchmarkDocumentForEvaluation:
    """Load one benchmark document YAML file."""
    payload = yaml.safe_load(document_path.read_text(encoding="utf-8")) or {}
    entities_used = EntityCollection.model_validate(payload.get("entities_used") or {})
    stem_document_id, stem_variant_index = split_document_variant_stem(document_path.stem)
    document_id = str(payload.get("document_id") or stem_document_id)
    document_theme = str(payload.get("document_theme") or document_path.parent.name)
    default_setting = (
        document_path.parent.parent.name if document_path.parent.parent.name != "FICTIONAL_DOCUMENTS" else "fictional"
    )
    document_setting = str(payload.get("document_setting") or default_setting).lower()
    setting_spec = parse_dataset_setting(document_setting)
    document_variant_index = int(payload.get("document_variant_index") or stem_variant_index or 1)
    document_variant_id = str(payload.get("document_variant_id") or format_document_variant_id(document_variant_index))
    document_text = str(payload.get("document_text") or payload.get("generated_document") or "")
    scoring_references = load_scoring_references()

    raw_questions = payload.get("questions", []) or []
    contracts: dict[str, dict[str, str]] = {}
    if raw_questions:
        template_path = _resolve_human_template_path(
            payload,
            document_theme=document_theme,
            document_id=document_id,
        )
        contracts = _load_human_question_contracts(template_path)

    questions: list[BenchmarkQuestionForEvaluation] = []
    for raw_question in raw_questions:
        question_id = str(raw_question.get("question_id") or "")
        contract = contracts.get(question_id)
        if contract is None:
            raise ValueError(f"{document_path}: no human-reviewed question contract for {document_id}/{question_id}.")
        question_type = contract["question_type"]
        answer_behavior = contract["answer_behavior"]
        original_question_text = str(raw_question.get("question_text") or raw_question.get("question") or "")
        question_text = original_question_text.strip()
        answer_expression = str(raw_question.get("answer_expression") or raw_question.get("answer") or "").strip()
        stored_question_type = normalize_question_type(raw_question.get("question_type"))
        if stored_question_type != question_type:
            raise ValueError(
                f"{document_path}: generated question_type {stored_question_type!r} contradicts "
                f"human annotation {question_type!r} for {question_id}."
            )
        if setting_spec.is_factual and _normalized_text(question_text) != contract["question_text_factual"]:
            raise ValueError(
                f"{document_path}: factual question text drift for {question_id}; generated file "
                "does not match the human-reviewed template version."
            )
        reference = scoring_references.match(
            setting=setting_spec.setting_id,
            document_id=document_id,
            variant_id=document_variant_id,
            question_id=question_id,
            document_text=document_text,
            question_text=original_question_text,
            frozen_ground_truth=str(raw_question.get("evaluated_answer")),
        )
        if reference is not None:
            ground_truth = reference.ground_truth
            answer_expression = reference.answer_expression
            answer_spec = reference
            accepted_answer_overrides = reference.accepted_answer_overrides
        else:
            ground_truth = (
                "Cannot be determined"
                if answer_behavior == "refusal"
                else _resolved_ground_truth(
                    answer_expression,
                    _coerce_ground_truth(raw_question.get("evaluated_answer")),
                    entities_used,
                )
            )
            accepted_answer_overrides = _coerce_accepted_answer_overrides(raw_question.get("accepted_answer_overrides"))
            answer_spec = build_answer_spec(
                question_text=question_text,
                answer_expression=answer_expression,
                evaluated_answer=ground_truth,
                document_text=document_text,
                entities_used=entities_used,
                accepted_answer_overrides=accepted_answer_overrides,
            )
        questions.append(
            BenchmarkQuestionForEvaluation(
                question_id=question_id,
                question_type=question_type,
                answer_behavior=answer_behavior,
                question_text=question_text,
                ground_truth=ground_truth,
                ground_truth_canonical=answer_spec.ground_truth_canonical,
                answer_schema=answer_spec.answer_schema,
                accepted_answers=answer_spec.accepted_answers,
                accepted_answers_canonical=answer_spec.accepted_answers_canonical,
                accepted_answer_overrides=accepted_answer_overrides,
                answer_expression=answer_expression,
                pair_key=unique_question_key(document_theme, document_id, question_id),
            )
        )

    return BenchmarkDocumentForEvaluation(
        document_id=document_id,
        document_theme=document_theme,
        document_setting=setting_spec.setting_id,
        document_setting_family=str(payload.get("document_setting_family") or setting_spec.setting_family),
        document_variant_id=document_variant_id,
        document_variant_index=document_variant_index,
        replacement_proportion=float(payload.get("replacement_proportion") or setting_spec.replacement_proportion),
        document_text=document_text,
        source_path=document_path,
        questions=questions,
    )


def iter_evaluation_documents(
    *,
    settings: Sequence[str],
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
) -> Iterator[BenchmarkDocumentForEvaluation]:
    """Yield evaluation documents from the benchmark document directories."""
    for setting in settings:
        for document_path in iter_document_variant_paths(
            setting=setting,
            themes=themes,
            document_ids=document_ids,
        ):
            base_document_id, _ = split_document_variant_stem(document_path.stem)
            if is_excluded_evaluation_document(base_document_id):
                continue
            document = load_evaluation_document(document_path)
            if is_excluded_evaluation_document(document.document_id):
                continue
            yield document


def load_document_pairs(
    *,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
) -> dict[tuple[str, str], dict[str, BenchmarkDocumentForEvaluation]]:
    """Return grouped document settings keyed by ``(theme, document_id)``."""
    pairs: dict[tuple[str, str], dict[str, BenchmarkDocumentForEvaluation]] = {}
    for document in iter_evaluation_documents(
        settings=("factual", "fictional"), themes=themes, document_ids=document_ids
    ):
        key = (document.document_theme, document.document_id)
        pairs.setdefault(key, {})
        pair_setting_key = document.document_setting
        if not (document.document_setting == "factual" and document.document_variant_index == 1):
            pair_setting_key = f"{pair_setting_key}:{document.document_variant_id}"
        pairs[key][pair_setting_key] = document
    return pairs
