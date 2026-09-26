"""Parse and Judge Match evaluation stages for MemoReason."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .answer_schema_data_contracts import ANSWER_PARSER_VERSION
from .exact_and_judge_match_scoring import (
    SCORING_PROTOCOL_VERSION,
    JudgeMatchConfiguration,
    accepted_answer_match_is_correct,
    judge_match_is_allowed,
    judge_prediction,
)
from .ground_truth_answer_specification import build_answer_spec
from .model_evaluation_artifact_io import (
    _attach_reproducibility_manifest_reference,
    _drop_downstream_heavy_fields,
    _evaluated_output_path,
    _iter_stage_paths,
    _read_yaml_payload,
    _remove_evaluated_stage_artifact,
    _resolve_source_question,
    _saved_payload_is_current,
    _sha256_file,
    _should_preserve_raw_metadata,
    _should_skip_deleted_question,
    _write_yaml,
)
from .model_evaluation_artifact_reuse_contract import (
    _evaluated_payload_matches_scoring_context,
    _judge_execution_context,
    _parsed_payload_matches_raw_context,
)
from .model_evaluation_run_manifest import ModelEvaluationRunManifest
from .schema_aware_answer_matching import parse_schema_answer
from .short_answer_extraction import parse_short_answer


def _judge_prediction_text(
    result: dict[str, object],
    *,
    parsed_output: str,
    answer_schema: str,
) -> str:
    """Preserve semantic units/composites that canonical parsing may discard."""
    raw_output = str(result.get("raw_output") or "").strip()
    if not parsed_output:
        return raw_output
    if answer_schema in {"quantity", "entity_span"}:
        raw_short_answer = parse_short_answer(raw_output)
        if raw_short_answer:
            return raw_short_answer
    return parsed_output


def parse_raw_model_answers(
    *,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    overwrite: bool = False,
    reproducibility_manifest: ModelEvaluationRunManifest | None = None,
) -> list[Path]:
    """Parse the raw outputs into short answers."""
    written_paths: list[Path] = []
    for raw_output_path in _iter_stage_paths(
        stage_suffix="raw_outputs",
        model_names=model_names,
        themes=themes,
        document_ids=document_ids,
        settings=settings,
    ):
        parsed_output_path = raw_output_path.with_name(
            raw_output_path.name.replace("_raw_outputs.yaml", "_parsed_outputs.yaml")
        )
        if parsed_output_path.exists() and not overwrite:
            existing_parsed_payload = _read_yaml_payload(parsed_output_path)
            if _parsed_payload_matches_raw_context(
                existing_parsed_payload,
                raw_output_path,
            ) and _saved_payload_is_current(parsed_output_path, check_score_metadata=True):
                written_paths.append(parsed_output_path)
                continue

        payload = _read_yaml_payload(raw_output_path)
        parsed_results = []
        for result in payload.get("results", []) or []:
            source_question = _resolve_source_question(payload, result)
            if _should_skip_deleted_question(source_question):
                continue
            answer_schema = str(result.get("answer_schema") or "").strip()
            accepted_answers = tuple(
                str(answer) for answer in (result.get("accepted_answers") or []) if str(answer).strip()
            )
            accepted_answers_canonical = tuple(
                str(answer) for answer in (result.get("accepted_answers_canonical") or []) if str(answer).strip()
            )
            ground_truth_canonical = str(result.get("ground_truth_canonical") or "").strip()
            if source_question is not None and not _should_preserve_raw_metadata(result, source_question):
                result = {
                    **result,
                    "question_type": source_question.question_type,
                    "answer_behavior": source_question.answer_behavior,
                    "question_text": source_question.question_text,
                    "ground_truth": source_question.ground_truth,
                    "answer_expression": source_question.answer_expression,
                    "accepted_answer_overrides": list(getattr(source_question, "accepted_answer_overrides", ()) or ()),
                }
                answer_schema = source_question.answer_schema
                accepted_answers = source_question.accepted_answers
                accepted_answers_canonical = source_question.accepted_answers_canonical
                ground_truth_canonical = source_question.ground_truth_canonical
            elif not answer_schema:
                fallback_spec = build_answer_spec(
                    question_text=str(result.get("question_text") or "").strip(),
                    answer_expression=str(result.get("answer_expression") or "").strip(),
                    evaluated_answer=str(result.get("ground_truth") or "").strip(),
                    document_text=str(payload.get("document_text") or payload.get("generated_document") or ""),
                    entities_used=None,
                    accepted_answer_overrides=result.get("accepted_answer_overrides"),
                )
                answer_schema = fallback_spec.answer_schema
                accepted_answers = fallback_spec.accepted_answers
                accepted_answers_canonical = fallback_spec.accepted_answers_canonical
                ground_truth_canonical = fallback_spec.ground_truth_canonical
            raw_output_text = str(result.get("raw_output", ""))
            raw_reasoning_text = str(result.get("raw_reasoning", ""))
            parse_source_text = raw_output_text
            if raw_reasoning_text.strip():
                stripped_raw_output = raw_output_text.lstrip()
                if (
                    not raw_output_text.strip()
                    or stripped_raw_output.startswith("<|channel|>analysis<|message|>")
                    or "answer:" not in raw_output_text.lower()
                ):
                    parse_source_text = raw_reasoning_text
            parse_result = parse_schema_answer(
                parse_source_text,
                answer_schema,
                accepted_answers=accepted_answers,
            )
            parsed_results.append(
                {
                    **_drop_downstream_heavy_fields(result),
                    "ground_truth_canonical": ground_truth_canonical,
                    "answer_schema": answer_schema,
                    "accepted_answers": list(accepted_answers),
                    "accepted_answers_canonical": list(accepted_answers_canonical),
                    "parsed_output": parse_result.parsed_output,
                    "parsed_output_canonical": parse_result.canonical_output,
                    "parse_status": parse_result.parse_status,
                    "format_compliant": parse_result.format_compliant,
                }
            )
        parsed_payload = _attach_reproducibility_manifest_reference(
            {
                **payload,
                "answer_parser_version": ANSWER_PARSER_VERSION,
                "raw_source_sha256": _sha256_file(raw_output_path),
                "results": parsed_results,
            },
            reproducibility_manifest=reproducibility_manifest,
            stage_name="parse",
        )
        _remove_evaluated_stage_artifact(parsed_output_path)
        written_paths.append(_write_yaml(parsed_payload, parsed_output_path))
    return written_paths


def score_parsed_model_answers(
    *,
    judge_config: JudgeMatchConfiguration | None,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    overwrite: bool = False,
    reproducibility_manifest: ModelEvaluationRunManifest | None = None,
) -> list[Path]:
    """Score parsed outputs with exact match and, when needed, an LLM judge."""
    written_paths: list[Path] = []
    for parsed_output_path in _iter_stage_paths(
        stage_suffix="parsed_outputs",
        model_names=model_names,
        themes=themes,
        document_ids=document_ids,
        settings=settings,
    ):
        evaluated_output_path = _evaluated_output_path(parsed_output_path)
        if evaluated_output_path.exists() and not overwrite:
            existing_evaluated_payload = _read_yaml_payload(evaluated_output_path)
            if _evaluated_payload_matches_scoring_context(
                existing_evaluated_payload,
                parsed_output_path=parsed_output_path,
                judge_config=judge_config,
            ) and _saved_payload_is_current(evaluated_output_path, check_score_metadata=True):
                written_paths.append(evaluated_output_path)
                continue

        payload = _read_yaml_payload(parsed_output_path)
        evaluated_results = []
        for result in payload.get("results", []) or []:
            source_question = _resolve_source_question(payload, result)
            if _should_skip_deleted_question(source_question):
                continue
            parsed_output = str(result.get("parsed_output", "")).strip()
            parsed_output_canonical = str(result.get("parsed_output_canonical", "")).strip()
            ground_truth = str(result.get("ground_truth", "")).strip()
            ground_truth_canonical = str(result.get("ground_truth_canonical") or "").strip()
            answer_schema = str(result.get("answer_schema") or "").strip()
            accepted_answers = tuple(
                str(answer).strip() for answer in (result.get("accepted_answers") or []) if str(answer).strip()
            )
            accepted_answers_canonical = tuple(
                str(answer).strip()
                for answer in (result.get("accepted_answers_canonical") or [])
                if str(answer).strip()
            )
            if source_question is not None and not _should_preserve_raw_metadata(result, source_question):
                result = {
                    **result,
                    "question_type": source_question.question_type,
                    "answer_behavior": source_question.answer_behavior,
                    "question_text": source_question.question_text,
                    "ground_truth": source_question.ground_truth,
                    "answer_expression": source_question.answer_expression,
                    "accepted_answer_overrides": list(getattr(source_question, "accepted_answer_overrides", ()) or ()),
                }
                ground_truth = source_question.ground_truth
                ground_truth_canonical = source_question.ground_truth_canonical
                answer_schema = source_question.answer_schema
                accepted_answers = source_question.accepted_answers
                accepted_answers_canonical = source_question.accepted_answers_canonical
            elif not accepted_answers_canonical:
                fallback_spec = build_answer_spec(
                    question_text=str(result.get("question_text") or "").strip(),
                    answer_expression=str(result.get("answer_expression") or "").strip(),
                    evaluated_answer=ground_truth,
                    document_text=str(payload.get("document_text") or payload.get("generated_document") or ""),
                    entities_used=None,
                    accepted_answer_overrides=result.get("accepted_answer_overrides"),
                )
                answer_schema = answer_schema or fallback_spec.answer_schema
                accepted_answers = fallback_spec.accepted_answers
                accepted_answers_canonical = fallback_spec.accepted_answers_canonical
                ground_truth_canonical = fallback_spec.ground_truth_canonical
            exact_match = accepted_answer_match_is_correct(
                parsed_output_canonical,
                accepted_answers_canonical,
                answer_schema=answer_schema,
                raw_prediction=parsed_output,
            )
            judge_match = None
            judge_raw_output = None
            judge_skip_reason = None
            if (
                not exact_match
                and judge_config is not None
                and judge_match_is_allowed(
                    answer_schema=answer_schema,
                    parsed_output_canonical=parsed_output_canonical,
                    raw_prediction=str(result.get("raw_output") or ""),
                )
            ):
                judge_match, judge_raw_output = judge_prediction(
                    question_text=str(result.get("question_text", "")).strip(),
                    ground_truth=ground_truth,
                    predicted_answer=_judge_prediction_text(
                        result,
                        parsed_output=parsed_output,
                        answer_schema=answer_schema,
                    ),
                    judge_config=judge_config,
                )
            elif not exact_match and judge_config is not None:
                judge_skip_reason = "schema_incompatible_prediction"
            final_is_correct = exact_match or bool(judge_match)
            evaluated_results.append(
                {
                    **_drop_downstream_heavy_fields(result),
                    "ground_truth_canonical": ground_truth_canonical,
                    "answer_schema": answer_schema,
                    "accepted_answers": list(accepted_answers),
                    "accepted_answers_canonical": list(accepted_answers_canonical),
                    "exact_match": exact_match,
                    "judge_match": judge_match,
                    "judge_raw_output": judge_raw_output,
                    "judge_skip_reason": judge_skip_reason,
                    "final_is_correct": final_is_correct,
                }
            )

        evaluated_payload = _attach_reproducibility_manifest_reference(
            {
                **payload,
                "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
                "parsed_source_sha256": _sha256_file(parsed_output_path),
                "judge_provider": judge_config.provider if judge_config is not None else None,
                "judge_model_name": judge_config.model_name if judge_config is not None else None,
                "judge_config": _judge_execution_context(judge_config),
                "results": evaluated_results,
            },
            reproducibility_manifest=reproducibility_manifest,
            stage_name="evaluate",
        )
        written_paths.append(_write_yaml(evaluated_payload, evaluated_output_path))
    return written_paths
