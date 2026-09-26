"""Prompt, generation, and scoring contexts used to decide safe reuse."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


from memoreason.model_providers.text_generation import TextGenerationRequest, generate_text, provider_generation_context
from .answer_schema_data_contracts import ANSWER_PARSER_VERSION
from .benchmark_document_loading import normalize_question_type
from .document_question_answering_prompt import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    GENERATION_TOKEN_BUDGET_POLICY_VERSION,
    PROMPT_FORMAT_VERSION,
    build_document_question_prompt,
    suggested_generation_max_tokens,
)
from .exact_and_judge_match_scoring import (
    SCORING_PROTOCOL_VERSION,
    JudgeMatchConfiguration,
)

from .model_evaluation_artifact_io import _serialized_source_path, _sha256_file, _should_preserve_raw_metadata


def _normalized_question_ids(question_ids: Sequence[str] | None) -> set[str]:
    return {str(question_id).strip() for question_id in (question_ids or []) if str(question_id).strip()}


def _normalized_question_types(question_types: Sequence[str] | None) -> set[str]:
    return {
        normalize_question_type(str(question_type).strip())
        for question_type in (question_types or [])
        if str(question_type).strip()
    }


def _question_matches_filter(
    question: object,
    *,
    question_ids: set[str],
    question_types: set[str],
) -> bool:
    if question_ids and str(getattr(question, "question_id", "") or "") not in question_ids:
        return False
    if question_types and normalize_question_type(getattr(question, "question_type", None)) not in question_types:
        return False
    return True


def _effective_generation_max_tokens(model_configuration: object, question: object) -> int:
    return min(
        int(model_configuration.max_tokens),
        suggested_generation_max_tokens(
            answer_schema=str(getattr(question, "answer_schema", "") or ""),
            model_name=str(model_configuration.model_id),
        ),
    )


def _raw_result_is_current_for_question(
    result: dict | None,
    document: object,
    question: object,
    model_configuration: object,
) -> bool:
    if not isinstance(result, dict):
        return False
    expected_prompt = build_document_question_prompt(
        document.document_text,
        question.question_text,
        answer_schema=question.answer_schema,
    )
    if str(result.get("user_prompt") or "") != expected_prompt:
        return False
    if result.get("effective_max_tokens") != _effective_generation_max_tokens(model_configuration, question):
        return False
    return not _should_preserve_raw_metadata(result, question)


def _model_generation_context(model_configuration: object) -> dict[str, object]:
    """Return the model parameters that must match before raw rows are reused."""
    context = {
        "temperature": float(model_configuration.temperature),
        "configured_max_tokens_cap": int(model_configuration.max_tokens),
        "token_budget_policy_version": GENERATION_TOKEN_BUDGET_POLICY_VERSION,
        "seed": model_configuration.seed,
    }
    context.update(
        provider_generation_context(
            provider=str(model_configuration.provider),
            model=str(model_configuration.model_name),
        )
    )
    return context


def _raw_payload_has_current_effective_budgets(
    payload: dict,
    model_configuration: object,
    document: object,
) -> bool:
    results = payload.get("results")
    if not isinstance(results, list):
        return False
    results_by_question_id: dict[str, dict] = {}
    for result in results:
        if not isinstance(result, dict):
            return False
        question_id = str(result.get("question_id") or "").strip()
        if not question_id or question_id in results_by_question_id:
            return False
        results_by_question_id[question_id] = result
    questions = list(getattr(document, "questions", ()) or ())
    if set(results_by_question_id) != {
        str(getattr(question, "question_id", "") or "").strip() for question in questions
    }:
        return False
    return all(
        results_by_question_id[str(question.question_id)].get("effective_max_tokens")
        == _effective_generation_max_tokens(model_configuration, question)
        for question in questions
    )


def _judge_execution_context(judge_config: JudgeMatchConfiguration | None) -> dict[str, object] | None:
    if judge_config is None:
        return None
    return {
        "provider": str(judge_config.provider).strip().lower(),
        "model_name": judge_config.model_name,
        "temperature": float(judge_config.temperature),
        "max_tokens": int(judge_config.max_tokens),
        "seed": judge_config.seed,
    }


def _parsed_payload_matches_raw_context(payload: dict, raw_output_path: Path) -> bool:
    return payload.get("answer_parser_version") == ANSWER_PARSER_VERSION and payload.get(
        "raw_source_sha256"
    ) == _sha256_file(raw_output_path)


def _evaluated_payload_matches_scoring_context(
    payload: dict,
    *,
    parsed_output_path: Path,
    judge_config: JudgeMatchConfiguration | None,
) -> bool:
    is_current, _detail = _evaluated_payload_scoring_currentness(
        payload,
        parsed_output_path=parsed_output_path,
        judge_config=judge_config,
    )
    return is_current


def _evaluated_payload_scoring_currentness(
    payload: dict,
    *,
    parsed_output_path: Path,
    judge_config: JudgeMatchConfiguration | None,
) -> tuple[bool, str]:
    if payload.get("answer_parser_version") != ANSWER_PARSER_VERSION:
        return False, "answer_parser_version mismatch"
    if payload.get("scoring_protocol_version") != SCORING_PROTOCOL_VERSION:
        return False, "scoring_protocol_version mismatch"
    if not parsed_output_path.is_file():
        return False, f"missing parsed source artifact: {parsed_output_path}"
    if payload.get("parsed_source_sha256") != _sha256_file(parsed_output_path):
        return False, "parsed_source_sha256 mismatch"
    expected_judge_config = _judge_execution_context(judge_config)
    if payload.get("judge_config") != expected_judge_config:
        return False, (
            f"judge_config mismatch: expected={expected_judge_config!r} actual={payload.get('judge_config')!r}"
        )
    return True, ""


def _source_document_identity(value: object) -> str:
    """Return the dataset-root-independent identity of one benchmark document."""
    normalized = str(value or "").replace("\\", "/")
    for marker in ("/FACTUAL_DOCUMENTS/", "/FICTIONAL_DOCUMENTS/"):
        if marker in normalized:
            return marker.lstrip("/") + normalized.split(marker, 1)[1]
    return normalized


def _raw_payload_matches_execution_context(
    payload: dict,
    model_configuration: object,
    document: object,
    *,
    allow_source_rebase: bool = False,
) -> bool:
    """Return whether rows from a raw payload can be reused for this execution.

    Per-question prompt comparison is sufficient for document/question edits, but
    it must not preserve rows produced with a different model, system prompt, or
    dataset variant when a filtered rerun is requested.
    """
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
    if not all(payload.get(key) == value for key, value in expected.items()):
        return False
    expected_source = _serialized_source_path(document.source_path)
    actual_source = payload.get("source_document_path")
    if actual_source == expected_source:
        return True
    return allow_source_rebase and (
        _source_document_identity(actual_source) == _source_document_identity(expected_source)
    )


def _raw_result_with_current_metadata(result: dict, question: object) -> dict:
    """Refresh score metadata while preserving an exactly reusable model response."""
    refreshed = dict(result)
    refreshed.update(
        {
            "question_id": question.question_id,
            "question_type": question.question_type,
            "answer_behavior": question.answer_behavior,
            "question_text": question.question_text,
            "ground_truth": question.ground_truth,
            "ground_truth_canonical": question.ground_truth_canonical,
            "answer_schema": question.answer_schema,
            "answer_expression": question.answer_expression,
            "accepted_answer_overrides": list(getattr(question, "accepted_answer_overrides", ()) or ()),
            "accepted_answers": list(question.accepted_answers),
            "accepted_answers_canonical": list(question.accepted_answers_canonical),
            "pair_key": question.pair_key,
        }
    )
    return refreshed


def _raw_result_payload(*, model_configuration: object, document: object, question: object) -> dict:
    user_prompt = build_document_question_prompt(
        document.document_text,
        question.question_text,
        answer_schema=question.answer_schema,
    )
    effective_max_tokens = _effective_generation_max_tokens(model_configuration, question)
    response = generate_text(
        TextGenerationRequest(
            provider=model_configuration.provider,
            model=model_configuration.model_name,
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=model_configuration.temperature,
            max_tokens=effective_max_tokens,
            seed=model_configuration.seed,
        )
    )
    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "answer_behavior": question.answer_behavior,
        "question_text": question.question_text,
        "ground_truth": question.ground_truth,
        "ground_truth_canonical": question.ground_truth_canonical,
        "answer_schema": question.answer_schema,
        "answer_expression": question.answer_expression,
        "accepted_answer_overrides": list(getattr(question, "accepted_answer_overrides", ()) or ()),
        "accepted_answers": list(question.accepted_answers),
        "accepted_answers_canonical": list(question.accepted_answers_canonical),
        "pair_key": question.pair_key,
        "user_prompt": user_prompt,
        "effective_max_tokens": effective_max_tokens,
        "raw_output": response.text,
        "raw_reasoning": response.reasoning_text,
        "raw_provider_response": response.raw_response,
    }
