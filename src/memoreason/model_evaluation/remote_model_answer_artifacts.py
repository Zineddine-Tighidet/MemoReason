"""Read, validate, reuse, and publish raw answers from remote paper models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from memoreason.factual_to_fictional_dataset.dataset_paths import MODEL_EVAL_RAW_OUTPUTS_DIR
from memoreason.model_providers.groq_client import (
    GPT_OSS_REASONING_BUDGET_EXHAUSTED,
    gpt_oss_reasoning_budget_exhaustion,
)

from .model_evaluation_artifact_reuse_contract import (
    _effective_generation_max_tokens,
    _model_generation_context,
    _raw_payload_has_current_effective_budgets,
    _raw_payload_matches_execution_context,
)
from .model_evaluation_artifact_io import (
    _saved_payload_is_current,
    _serialized_source_path,
    _write_text_exclusive,
)
from .document_question_answering_prompt import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    PROMPT_FORMAT_VERSION,
    build_document_question_prompt,
)


try:
    YAML_LOADER = yaml.CSafeLoader
    YAML_DUMPER = yaml.CSafeDumper
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader
    YAML_DUMPER = yaml.SafeDumper


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load one YAML artifact and require a mapping at its root."""
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def write_yaml_exclusive(payload: dict[str, Any], output_path: Path) -> None:
    """Publish one complete YAML atomically and refuse an existing target."""
    _write_text_exclusive(
        yaml.dump(
            payload,
            Dumper=YAML_DUMPER,
            sort_keys=False,
            allow_unicode=True,
            width=10000,
        ),
        output_path,
    )


def write_json_exclusive(payload: dict[str, Any], output_path: Path) -> None:
    """Write one run summary and refuse an existing target."""
    _write_text_exclusive(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        output_path,
    )


def _document_metadata_matches(
    payload: dict[str, Any],
    *,
    model_configuration: object,
    document: object,
) -> bool:
    expected = {
        "model_name": model_configuration.model_id,
        "model_provider": model_configuration.provider,
        "provider_model_name": model_configuration.model_name,
        "generation_config": _model_generation_context(model_configuration),
        "document_theme": document.document_theme,
        "document_id": document.document_id,
        "document_setting": document.document_setting,
        "document_setting_family": document.document_setting_family,
        "document_variant_id": document.document_variant_id,
        "document_variant_index": document.document_variant_index,
        "replacement_proportion": document.replacement_proportion,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "system_prompt": DOCUMENT_QA_SYSTEM_PROMPT,
    }
    return all(payload.get(key) == value for key, value in expected.items())


def _current_result_from_seed(
    seed_result: dict[str, Any],
    *,
    model_configuration: object,
    document: object,
    question: object,
) -> dict[str, Any] | None:
    expected_prompt = build_document_question_prompt(
        document.document_text,
        question.question_text,
        answer_schema=question.answer_schema,
    )
    if str(seed_result.get("user_prompt") or "") != expected_prompt:
        return None
    raw_output = str(seed_result.get("raw_output") or "")
    special_outcome: dict[str, object] = {}
    if not raw_output.strip():
        if seed_result.get("generation_outcome") != GPT_OSS_REASONING_BUDGET_EXHAUSTED:
            return None
        try:
            raw_provider_response = json.loads(str(seed_result.get("raw_provider_response") or ""))
        except json.JSONDecodeError:
            return None
        termination = gpt_oss_reasoning_budget_exhaustion(
            raw_provider_response,
            max_completion_tokens=_effective_generation_max_tokens(
                model_configuration,
                question,
            ),
        )
        if termination is None or seed_result.get("generation_termination") != termination:
            return None
        special_outcome = {
            "generation_outcome": GPT_OSS_REASONING_BUDGET_EXHAUSTED,
            "generation_termination": termination,
        }
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "answer_behavior": question.answer_behavior,
        "question_text": question.question_text,
        "ground_truth": question.ground_truth,
        "ground_truth_canonical": question.ground_truth_canonical,
        "answer_schema": question.answer_schema,
        "answer_expression": question.answer_expression,
        "accepted_answer_overrides": list(question.accepted_answer_overrides),
        "accepted_answers": list(question.accepted_answers),
        "accepted_answers_canonical": list(question.accepted_answers_canonical),
        "pair_key": question.pair_key,
        "user_prompt": expected_prompt,
        "effective_max_tokens": _effective_generation_max_tokens(model_configuration, question),
        "raw_output": raw_output,
        "raw_reasoning": str(seed_result.get("raw_reasoning") or ""),
        "raw_provider_response": str(seed_result.get("raw_provider_response") or ""),
        **special_outcome,
    }


def seed_results_for_document(
    seed_path: Path | None,
    *,
    model_configuration: object,
    document: object,
) -> dict[str, dict[str, Any]]:
    """Reuse only prompt-identical answers from a compatible frozen seed."""
    if seed_path is None or not seed_path.is_file():
        return {}
    payload = read_yaml_mapping(seed_path)
    if not _document_metadata_matches(payload, model_configuration=model_configuration, document=document):
        return {}
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return {}
    by_id = {str(result.get("question_id") or ""): result for result in raw_results if isinstance(result, dict)}
    if len(by_id) != len(raw_results):
        return {}
    reusable: dict[str, dict[str, Any]] = {}
    for question in document.questions:
        seed_result = by_id.get(question.question_id)
        if seed_result is None:
            continue
        current = _current_result_from_seed(
            seed_result,
            model_configuration=model_configuration,
            document=document,
            question=question,
        )
        if current is not None:
            reusable[question.question_id] = current
    return reusable


def build_raw_document_payload(
    *,
    model_configuration: object,
    document: object,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the frozen per-document raw-answer schema."""
    return {
        "model_name": model_configuration.model_id,
        "model_provider": model_configuration.provider,
        "provider_model_name": model_configuration.model_name,
        "generation_config": _model_generation_context(model_configuration),
        "document_theme": document.document_theme,
        "document_id": document.document_id,
        "document_setting": document.document_setting,
        "document_setting_family": document.document_setting_family,
        "document_variant_id": document.document_variant_id,
        "document_variant_index": document.document_variant_index,
        "replacement_proportion": document.replacement_proportion,
        "source_document_path": _serialized_source_path(document.source_path),
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "system_prompt": DOCUMENT_QA_SYSTEM_PROMPT,
        "results": results,
    }


def existing_target_is_current(
    output_path: Path,
    *,
    model_configuration: object,
    document: object,
) -> bool:
    """Return whether an already-published target matches the frozen context."""
    if not output_path.is_file():
        return False
    payload = read_yaml_mapping(output_path)
    return (
        _raw_payload_matches_execution_context(payload, model_configuration, document)
        and _raw_payload_has_current_effective_budgets(payload, model_configuration, document)
        and _saved_payload_is_current(output_path, check_score_metadata=False)
    )


def matching_seed_path(
    source_raw_root: Path | None,
    output_path: Path,
) -> Path | None:
    """Map a target artifact to the same relative path in a seed tree."""
    if source_raw_root is None:
        return None
    relative = output_path.relative_to(MODEL_EVAL_RAW_OUTPUTS_DIR)
    return source_raw_root / relative


__all__ = [
    "build_raw_document_payload",
    "existing_target_is_current",
    "matching_seed_path",
    "read_yaml_mapping",
    "seed_results_for_document",
    "write_json_exclusive",
    "write_yaml_exclusive",
]
