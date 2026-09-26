"""Core serialization and factual-record construction helpers."""

from __future__ import annotations

import re
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from memoreason import PROJECT_ROOT_DIRECTORY
from memoreason.benchmark_definition.annotation_runtime import AnnotationParser, load_annotated_document
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number
from memoreason.benchmark_definition.answer_expression_evaluation import AnswerEvaluator

from .dataset_paths import (
    FACTUAL_DOCUMENTS_DIR,
    FICTIONAL_DOCUMENTS_DIR,
    GENERATED_FICTIONAL_ENTITIES_DIR,
    HUMAN_ANNOTATED_TEMPLATES_DIR,
    document_variant_path,
    format_document_variant_id,
    resolve_template_identity,
)
from .dataset_record_constants import QUESTION_TYPE_ALIASES, THOUSANDS_COMMA_PATTERN
from .dataset_settings import FactualToFictionalDatasetSetting, factual_setting


def _semantically_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if isinstance(lhs, bool) or isinstance(rhs, bool):
        return lhs is rhs
    normalized_candidates = []
    for value in (lhs, rhs):
        normalized = None
        for parser in (parse_word_number, parse_integer_surface_number):
            try:
                normalized = parser(str(value).strip())
            except Exception:
                normalized = None
            if normalized is not None:
                break
        normalized_candidates.append(normalized)
    if normalized_candidates[0] is not None and normalized_candidates[1] is not None:
        return normalized_candidates[0] == normalized_candidates[1]
    for parser in (parse_word_number, parse_integer_surface_number):
        try:
            lhs_parsed = parser(str(lhs).strip())
            rhs_parsed = parser(str(rhs).strip())
        except Exception:
            lhs_parsed = None
            rhs_parsed = None
        if lhs_parsed is not None and rhs_parsed is not None:
            return lhs_parsed == rhs_parsed
    try:
        return abs(float(lhs) - float(rhs)) <= 1e-9
    except (TypeError, ValueError):
        return str(lhs).strip().casefold() == str(rhs).strip().casefold()


def normalize_question_type(question_type: str | None) -> str:
    """Normalize question-type labels from legacy templates."""
    if not question_type:
        return "unknown"
    cleaned = str(question_type).strip().lower()
    return QUESTION_TYPE_ALIASES.get(cleaned, cleaned)


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


def strip_inline_annotations(text: str) -> str:
    """Remove inline annotation metadata while keeping the visible surface text."""
    stripped = re.sub(r"\[([^\]]+?);\s*[^\]]+?\]", r"\1", text or "")
    return _normalize_numeric_surface_commas(stripped)


def _normalize_numeric_surface_commas(text: str) -> str:
    """Normalize thousands-separated numerals to plain digits for factual export."""
    return THOUSANDS_COMMA_PATTERN.sub(lambda match: match.group(1).replace(",", ""), text or "")


def _relative_to_project(path: Path) -> str:
    candidate = Path(path)
    candidate = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT_DIRECTORY / candidate).resolve()
    portable_roots = (
        (HUMAN_ANNOTATED_TEMPLATES_DIR.resolve(), Path("data/HUMAN_ANNOTATED_TEMPLATES")),
        (GENERATED_FICTIONAL_ENTITIES_DIR.resolve(), Path("data/GENERATED_FICTIONAL_ENTITIES")),
        (FACTUAL_DOCUMENTS_DIR.resolve(), Path("data/FACTUAL_DOCUMENTS")),
        (FICTIONAL_DOCUMENTS_DIR.resolve(), Path("data/FICTIONAL_DOCUMENTS")),
    )
    for configured_root, portable_root in portable_roots:
        try:
            relative_path = candidate.relative_to(configured_root)
        except ValueError:
            continue
        return str(portable_root / relative_path)
    try:
        return str(candidate.relative_to(PROJECT_ROOT_DIRECTORY))
    except ValueError:
        return str(candidate)


def _serialize_entities(entities: EntityCollection) -> dict[str, Any]:
    return {
        "persons": {key: value.model_dump(warnings=False) for key, value in entities.persons.items()},
        "places": {key: value.model_dump(warnings=False) for key, value in entities.places.items()},
        "events": {key: value.model_dump(warnings=False) for key, value in entities.events.items()},
        "organizations": {key: value.model_dump(warnings=False) for key, value in entities.organizations.items()},
        "awards": {key: value.model_dump(warnings=False) for key, value in entities.awards.items()},
        "legals": {key: value.model_dump(warnings=False) for key, value in entities.legals.items()},
        "products": {key: value.model_dump(warnings=False) for key, value in entities.products.items()},
        "temporals": {key: value.model_dump(warnings=False) for key, value in entities.temporals.items()},
        "numbers": {key: value.model_dump(warnings=False) for key, value in entities.numbers.items()},
    }


def _deserialize_entities(payload: dict[str, Any] | None) -> EntityCollection:
    return EntityCollection.model_validate(payload or {})


def _question_entries_from_template(document) -> list[dict[str, Any]]:
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    questions_payload = [
        {
            "question_id": question.question_id,
            "question": strip_inline_annotations(question.question),
            "answer": question.answer,
            "question_type": normalize_question_type(question.question_type),
            "answer_type": answer_behavior_label(question.answer_type),
            "reasoning_chain": list(question.reasoning_chain or []),
        }
        for question in document.questions
    ]
    evaluated_answers = AnswerEvaluator.evaluate_all_answers(questions_payload, factual_entities)
    question_entries = AnswerEvaluator.build_question_entries_with_answers(
        questions_payload,
        evaluated_answers,
        factual_entities,
    )
    question_entries_for_export: list[dict[str, Any]] = []
    for question_entry, source_question in zip(question_entries, document.questions, strict=True):
        question_entries_for_export.append(
            {
                "question_id": question_entry["question_id"],
                "question_type": normalize_question_type(question_entry.get("question_type")),
                "answer_behavior": answer_behavior_label(
                    getattr(source_question, "answer_type", None),
                ),
                "answer_type": answer_behavior_label(
                    getattr(source_question, "answer_type", None),
                ),
                "question_text": question_entry["question"],
                "reasoning_chain": [
                    _normalize_numeric_surface_commas(step) for step in (source_question.reasoning_chain or [])
                ],
                "answer_expression": question_entry.get("answer_expression", ""),
                "answer_entities": question_entry.get("answer_entities"),
                "evaluated_answer": str(question_entry.get("evaluated_answer", "")).strip(),
                "accepted_answer_overrides": [
                    _normalize_numeric_surface_commas(str(value))
                    for value in (getattr(source_question, "accepted_answer_overrides", []) or [])
                ],
            }
        )
    return question_entries_for_export


def build_factual_dataset_record(template_path: Path, *, seed: int) -> dict[str, Any]:
    """Build the factual dataset record for one template."""
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    theme, document_id = resolve_template_identity(template_path)
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    setting_spec = factual_setting()
    return {
        "document_id": document_id,
        "document_theme": theme,
        "document_setting": setting_spec.setting_id,
        "document_setting_family": setting_spec.setting_family,
        "document_variant_id": format_document_variant_id(1),
        "document_variant_index": 1,
        "replacement_proportion": setting_spec.replacement_proportion,
        "generation_seed": seed,
        "source_template_path": _relative_to_project(template_path),
        "document_text": strip_inline_annotations(document.document_to_annotate),
        "num_entities_replaced": 0,
        "replaced_factual_entities": {},
        "questions": _question_entries_from_template(document),
        "entities_used": _serialize_entities(factual_entities),
    }


def _write_yaml(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10000)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def remove_stale_fictional_dataset_exports(
    *,
    template_path: Path,
    setting_spec: FactualToFictionalDatasetSetting,
) -> None:
    theme, document_id = resolve_template_identity(template_path)
    output_dir = document_variant_path(theme, document_id, setting_spec.setting_id).parent
    if not output_dir.exists():
        return
    for stale_path in sorted(output_dir.glob(f"{document_id}.yaml")):
        stale_path.unlink()
    for stale_path in sorted(output_dir.glob(f"{document_id}_v*.yaml")):
        stale_path.unlink()
