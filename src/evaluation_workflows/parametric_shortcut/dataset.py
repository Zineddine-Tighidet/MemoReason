"""Load the benchmark documents used for model evaluation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from src.core.annotation_runtime import RuleEngine, find_entity_refs
from src.core.document_schema import EntityCollection
from src.core.answer_evaluation import AnswerEvaluator
from src.dataset_export.dataset_settings import parse_dataset_setting
from src.dataset_export.dataset_paths import (
    format_document_variant_id,
    iter_template_paths,
    iter_document_variant_paths,
    split_document_variant_stem,
    unique_question_key,
)
from .answer_handling import build_answer_spec


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

DEFAULT_HF_DATASET_ID = "memoreason-anonymous/MemoReason"
HF_DOCUMENT_THEME_FALLBACK = "huggingface"
HF_VIRTUAL_SOURCE_PREFIX = "hf://datasets/"


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
class EvaluationQuestion:
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
class EvaluationDocument:
    """One benchmark document ready for evaluation."""

    document_id: str
    document_theme: str
    document_setting: str
    document_setting_family: str
    document_variant_id: str
    document_variant_index: int
    replacement_proportion: float
    document_text: str
    source_path: Path | str
    questions: list[EvaluationQuestion]


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


def load_evaluation_document(document_path: Path) -> EvaluationDocument:
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

    questions: list[EvaluationQuestion] = []
    for raw_question in payload.get("questions", []) or []:
        question_id = str(raw_question.get("question_id") or "")
        question_type = normalize_question_type(raw_question.get("question_type"))
        answer_behavior = str(
            raw_question.get("answer_behavior")
            or raw_question.get("answer_type")
            or answer_behavior_label(raw_question.get("is_answer_invariant"))
        ).lower()
        question_text = str(raw_question.get("question_text") or raw_question.get("question") or "").strip()
        answer_expression = str(raw_question.get("answer_expression") or raw_question.get("answer") or "").strip()
        ground_truth = _resolved_ground_truth(
            answer_expression,
            _coerce_ground_truth(raw_question.get("evaluated_answer")),
            entities_used,
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
            EvaluationQuestion(
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

    return EvaluationDocument(
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


@cache
def _template_theme_by_document_id() -> dict[str, str]:
    """Return a best-effort map from public template ids to their theme folder."""
    return {template_path.stem: template_path.parent.name for template_path in iter_template_paths()}


def _document_theme_for_hf_document(document_id: str) -> str:
    return _template_theme_by_document_id().get(document_id, HF_DOCUMENT_THEME_FALLBACK)


def _parse_hf_row_id(row_id: object, setting: str) -> tuple[str, str, str, int]:
    """Return ``(document_instance, document_id, question_id, variant_index)`` for one HF row."""
    row_id_text = str(row_id or "").strip()
    parts = row_id_text.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unexpected Hugging Face row id format: {row_id_text!r}")

    split_name, document_instance, question_id = parts
    if split_name != setting:
        raise ValueError(f"HF row id split {split_name!r} does not match requested split {setting!r}.")

    if setting == "factual":
        suffix = "_factual"
        document_id = (
            document_instance[: -len(suffix)] if document_instance.endswith(suffix) else document_instance
        )
        return document_instance, document_id, question_id, 1

    marker = f"_{setting}_v"
    if marker not in document_instance:
        raise ValueError(
            f"Unexpected Hugging Face document instance {document_instance!r} for split {setting!r}."
        )
    document_id, variant_text = document_instance.rsplit(marker, 1)
    return document_instance, document_id, question_id, int(variant_text)


def _hf_source_path(*, hf_dataset: str, hf_config: str | None, setting: str, document_instance: str) -> str:
    config_token = f"/{hf_config}" if hf_config else ""
    return f"{HF_VIRTUAL_SOURCE_PREFIX}{hf_dataset}{config_token}/{setting}/{document_instance}"


def _load_hf_split(*, hf_dataset: str, hf_config: str | None, setting: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject
        raise RuntimeError(
            "Loading MemoReason from Hugging Face requires the `datasets` package. "
            "Install the project with `uv sync` first."
        ) from exc

    if hf_config:
        return load_dataset(hf_dataset, hf_config, split=setting)
    return load_dataset(hf_dataset, split=setting)


def iter_hf_evaluation_documents(
    *,
    settings: Sequence[str],
    hf_dataset: str = DEFAULT_HF_DATASET_ID,
    hf_config: str | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
) -> Iterator[EvaluationDocument]:
    """Yield evaluation documents by grouping rows from the published HF dataset."""
    theme_filter = set(themes or [])
    document_filter = set(document_ids or [])

    for raw_setting in settings:
        setting_spec = parse_dataset_setting(raw_setting)
        setting = setting_spec.setting_id
        split = _load_hf_split(hf_dataset=hf_dataset, hf_config=hf_config, setting=setting)
        grouped_rows: dict[str, dict[str, object]] = {}

        for row in split:
            document_instance, document_id, question_id, variant_index = _parse_hf_row_id(
                row.get("id"),
                setting,
            )
            if is_excluded_evaluation_document(document_id):
                continue
            if document_filter and document_id not in document_filter:
                continue

            document_theme = _document_theme_for_hf_document(document_id)
            if theme_filter and document_theme not in theme_filter:
                continue

            document_text = str(row.get("document") or "")
            group = grouped_rows.setdefault(
                document_instance,
                {
                    "document_id": document_id,
                    "document_theme": document_theme,
                    "document_variant_index": variant_index,
                    "document_text": document_text,
                    "questions": [],
                },
            )
            if str(group["document_text"]) != document_text:
                raise ValueError(f"HF document text changed within instance {document_instance!r}.")

            question_text = str(row.get("question") or "").strip()
            ground_truth = _coerce_ground_truth(row.get("answer"))
            answer_spec = build_answer_spec(
                question_text=question_text,
                answer_expression=ground_truth,
                evaluated_answer=ground_truth,
                document_text=document_text,
                entities_used=None,
                accepted_answer_overrides=(),
            )
            questions = group["questions"]
            assert isinstance(questions, list)
            questions.append(
                EvaluationQuestion(
                    question_id=question_id,
                    question_type=normalize_question_type(row.get("question_type")),
                    answer_behavior=answer_behavior_label(row.get("answer_type")),
                    question_text=question_text,
                    ground_truth=ground_truth,
                    ground_truth_canonical=answer_spec.ground_truth_canonical,
                    answer_schema=answer_spec.answer_schema,
                    accepted_answers=answer_spec.accepted_answers,
                    accepted_answers_canonical=answer_spec.accepted_answers_canonical,
                    accepted_answer_overrides=tuple(),
                    answer_expression=ground_truth,
                    pair_key=unique_question_key(document_theme, document_id, question_id),
                )
            )

        for document_instance, group in grouped_rows.items():
            variant_index = int(group["document_variant_index"])
            yield EvaluationDocument(
                document_id=str(group["document_id"]),
                document_theme=str(group["document_theme"]),
                document_setting=setting,
                document_setting_family=setting_spec.setting_family,
                document_variant_id=format_document_variant_id(variant_index),
                document_variant_index=variant_index,
                replacement_proportion=setting_spec.replacement_proportion,
                document_text=str(group["document_text"]),
                source_path=_hf_source_path(
                    hf_dataset=hf_dataset,
                    hf_config=hf_config,
                    setting=setting,
                    document_instance=document_instance,
                ),
                questions=list(group["questions"]),
            )


def iter_evaluation_documents(
    *,
    settings: Sequence[str],
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    dataset_source: str = "local",
    hf_dataset: str = DEFAULT_HF_DATASET_ID,
    hf_config: str | None = None,
) -> Iterator[EvaluationDocument]:
    """Yield evaluation documents from local YAML files or the published HF dataset."""
    normalized_source = str(dataset_source or "local").strip().lower()
    if normalized_source in {"hf", "huggingface", "hugging_face"}:
        yield from iter_hf_evaluation_documents(
            settings=settings,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            themes=themes,
            document_ids=document_ids,
        )
        return
    if normalized_source != "local":
        raise ValueError("dataset_source must be either 'local' or 'huggingface'.")

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
) -> dict[tuple[str, str], dict[str, EvaluationDocument]]:
    """Return grouped document settings keyed by ``(theme, document_id)``."""
    pairs: dict[tuple[str, str], dict[str, EvaluationDocument]] = {}
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
