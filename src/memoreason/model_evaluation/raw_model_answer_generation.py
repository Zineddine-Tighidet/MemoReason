"""Raw model generation stage for MemoReason."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


from memoreason.factual_to_fictional_dataset.dataset_paths import (
    model_eval_artifact_path,
)
from .benchmark_document_loading import iter_evaluation_documents
from .document_question_answering_prompt import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    PROMPT_FORMAT_VERSION,
)
from .model_evaluation_run_manifest import ModelEvaluationRunManifest
from .paper_model_registry import resolve_paper_model_configurations

from .model_evaluation_artifact_reuse_contract import (
    _model_generation_context,
    _normalized_question_ids,
    _normalized_question_types,
    _question_matches_filter,
    _raw_payload_has_current_effective_budgets,
    _raw_payload_matches_execution_context,
    _raw_result_is_current_for_question,
    _raw_result_payload,
    _raw_result_with_current_metadata,
)
from .model_evaluation_artifact_io import (
    _attach_reproducibility_manifest_reference,
    _read_yaml_payload,
    _remove_downstream_stage_artifacts,
    _saved_payload_is_current,
    _serialized_source_path,
    _write_yaml,
)


def generate_raw_model_answers(
    *,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] = ("factual", "fictional"),
    question_ids: Sequence[str] | None = None,
    question_types: Sequence[str] | None = None,
    overwrite: bool = False,
    refresh_stale_only: bool = False,
    generation_temperature: float | None = None,
    generation_seed: int | None = None,
    reproducibility_manifest: ModelEvaluationRunManifest | None = None,
) -> list[Path]:
    """Call the configured models on the requested benchmark document settings."""
    question_id_filter = _normalized_question_ids(question_ids)
    question_type_filter = _normalized_question_types(question_types)
    has_question_filter = bool(question_id_filter or question_type_filter)
    if refresh_stale_only and has_question_filter:
        raise ValueError("refresh_stale_only cannot be combined with question filters.")
    if refresh_stale_only and overwrite:
        raise ValueError("refresh_stale_only cannot be combined with overwrite.")
    if generation_temperature is None and generation_seed is None:
        model_configurations = resolve_paper_model_configurations(model_names)
    else:
        model_configurations = resolve_paper_model_configurations(
            model_names,
            temperature=generation_temperature,
            seed=generation_seed,
        )
    evaluation_documents = list(
        iter_evaluation_documents(
            settings=settings,
            themes=themes,
            document_ids=document_ids,
        )
    )
    written_paths: list[Path] = []

    for model_configuration in model_configurations:
        for document in evaluation_documents:
            output_path = model_eval_artifact_path(
                theme=document.document_theme,
                model_name=model_configuration.model_id,
                document_id=document.document_id,
                setting=document.document_setting,
                stage_suffix="raw_outputs",
                variant_id=None
                if document.document_variant_index == 1 and document.source_path.stem == document.document_id
                else document.document_variant_id,
            )
            selected_questions = [
                question
                for question in document.questions
                if _question_matches_filter(
                    question,
                    question_ids=question_id_filter,
                    question_types=question_type_filter,
                )
            ]
            if has_question_filter and not selected_questions:
                continue

            existing_payload = _read_yaml_payload(output_path) if output_path.exists() else {}
            execution_context_matches = _raw_payload_matches_execution_context(
                existing_payload,
                model_configuration,
                document,
                allow_source_rebase=refresh_stale_only,
            )
            if output_path.exists() and refresh_stale_only and not execution_context_matches:
                raise ValueError(f"Refusing stale-only reuse from incompatible raw execution context: {output_path}")
            if output_path.exists() and not overwrite and not has_question_filter and not refresh_stale_only:
                if (
                    execution_context_matches
                    and _raw_payload_has_current_effective_budgets(existing_payload, model_configuration, document)
                    and _saved_payload_is_current(
                        output_path,
                        check_score_metadata=False,
                    )
                ):
                    written_paths.append(output_path)
                    continue
                _remove_downstream_stage_artifacts(output_path)

            existing_results_by_qid: dict[str, dict] = {}
            if (has_question_filter or refresh_stale_only) and execution_context_matches:
                existing_results_by_qid = {
                    str(result.get("question_id") or ""): result
                    for result in (existing_payload.get("results") or [])
                    if isinstance(result, dict)
                }

            results = []
            regenerated_any = False
            for question in document.questions:
                existing_result = existing_results_by_qid.get(question.question_id)
                if refresh_stale_only:
                    should_generate = not _raw_result_is_current_for_question(
                        existing_result,
                        document,
                        question,
                        model_configuration,
                    )
                else:
                    should_generate = not has_question_filter or _question_matches_filter(
                        question,
                        question_ids=question_id_filter,
                        question_types=question_type_filter,
                    )
                    if has_question_filter and not should_generate:
                        should_generate = not _raw_result_is_current_for_question(
                            existing_result,
                            document,
                            question,
                            model_configuration,
                        )
                if should_generate:
                    regenerated_any = True
                    results.append(
                        _raw_result_payload(
                            model_configuration=model_configuration,
                            document=document,
                            question=question,
                        )
                    )
                else:
                    results.append(_raw_result_with_current_metadata(existing_result, question))

            payload = {
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
            if regenerated_any:
                _remove_downstream_stage_artifacts(output_path)
            written_paths.append(
                _write_yaml(
                    _attach_reproducibility_manifest_reference(
                        payload,
                        reproducibility_manifest=reproducibility_manifest,
                        stage_name="raw",
                    ),
                    output_path,
                )
            )
    return written_paths
