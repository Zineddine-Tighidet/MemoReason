"""Run Parametric Shortcut evaluation from model calls to plots."""

from __future__ import annotations

from collections.abc import Sequence
from collections import defaultdict
from functools import cache
import json
import logging
from pathlib import Path

import yaml

from src.dataset_export.dataset_paths import (
    MODEL_EVAL_METRICS_DIR,
    MODEL_EVAL_RAW_OUTPUTS_DIR,
    PROJECT_ROOT,
    ensure_dataset_artifact_directories,
    metrics_output_path,
    model_eval_artifact_path,
    sanitize_model_name,
)
from src.llm.text_generation import TextGenerationRequest, generate_text
from .answer_handling import (
    build_answer_spec,
    parse_schema_answer,
)
from .dataset import iter_evaluation_documents, load_evaluation_document, normalize_question_type
from .prompting import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    PROMPT_FORMAT_VERSION,
    build_document_question_prompt,
    suggested_generation_max_tokens,
)
from .reproducibility_manifest import EvaluationReproducibilityManifest
from .registry import resolve_model_specs
from .scoring import JudgeConfig, accepted_answer_match_is_correct, judge_match_is_allowed, judge_prediction
from .version_audit import audit_saved_output_payload, blocking_issues, format_issue_summary


logger = logging.getLogger(__name__)

try:
    YAML_LOADER = yaml.CSafeLoader
    YAML_DUMPER = yaml.CSafeDumper
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader
    YAML_DUMPER = yaml.SafeDumper


def _drop_downstream_heavy_fields(result: dict) -> dict:
    """Keep raw-provider payloads in raw outputs only; downstream stages need the answer fields."""
    return {key: value for key, value in result.items() if key != "raw_provider_response"}


@cache
def _source_questions_by_id(source_document_path: str) -> dict[str, object]:
    if not source_document_path:
        return {}
    source_path = PROJECT_ROOT / source_document_path
    if not source_path.exists():
        return {}
    document = load_evaluation_document(source_path)
    return {question.question_id: question for question in document.questions}


def _resolve_source_question(payload: dict, result: dict) -> object | None:
    source_document_path = str(payload.get("source_document_path") or "").strip()
    question_id = str(result.get("question_id") or "").strip()
    if not source_document_path or not question_id:
        return None
    return _source_questions_by_id(source_document_path).get(question_id)


def _should_skip_deleted_question(source_question: object | None) -> bool:
    # If a question no longer exists in the source template, drop stale raw artifacts
    # instead of silently rescoring them from outdated stored metadata.
    return source_question is None


def _should_preserve_raw_metadata(result: dict, source_question: object | None) -> bool:
    if source_question is None:
        return False
    question_id = str(result.get("question_id") or "").strip()
    raw_question_text = str(result.get("question_text") or "").strip().lower()
    source_question_text = str(getattr(source_question, "question_text", "") or "").strip().lower()
    if (
        question_id == "awards_11_q05"
        and "including" in raw_question_text
        and "excluding" in source_question_text
    ):
        # This question changed semantics after the original model calls. Keep the
        # historical raw metadata until the question is rerun.
        return True
    return False


def _write_yaml(payload: dict, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.dump(payload, Dumper=YAML_DUMPER, sort_keys=False, allow_unicode=True, width=10000),
        encoding="utf-8",
    )
    return output_path


def _read_yaml_payload(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}


def _saved_payload_is_current(
    path: Path,
    *,
    check_score_metadata: bool,
) -> bool:
    payload = _read_yaml_payload(path)
    issues = blocking_issues(
        audit_saved_output_payload(
            payload,
            path=path,
            check_score_metadata=check_score_metadata,
        )
    )
    if issues:
        logger.warning("Will refresh stale %s: %s", path, format_issue_summary(issues))
        return False
    return True


def _remove_downstream_stage_artifacts(raw_output_path: Path) -> None:
    """Remove parse/evaluate siblings after a raw prompt-version refresh."""
    parsed_output_path = raw_output_path.with_name(
        raw_output_path.name.replace("_raw_outputs.yaml", "_parsed_outputs.yaml")
    )
    evaluated_output_path = raw_output_path.with_name(
        raw_output_path.name.replace("_raw_outputs.yaml", "_evaluated_outputs.yaml")
    )
    for path in (parsed_output_path, evaluated_output_path):
        if path.exists():
            path.unlink()


def _attach_reproducibility_manifest_reference(
    payload: dict,
    *,
    reproducibility_manifest: EvaluationReproducibilityManifest | None,
    stage_name: str,
) -> dict:
    if reproducibility_manifest is None:
        return payload
    return {
        **payload,
        "reproducibility_manifest": reproducibility_manifest.stage_reference(stage_name),
    }


def _iter_stage_paths(
    *,
    stage_suffix: str,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
) -> list[Path]:
    model_filter = {sanitize_model_name(name) for name in (model_names or [])}
    theme_filter = set(themes or [])
    document_filter = set(document_ids or [])
    setting_filter = {str(setting).strip().lower() for setting in (settings or []) if str(setting).strip()}

    matched_paths: list[Path] = []
    for artifact_path in sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.rglob(f"*_{stage_suffix}.yaml")):
        theme_name = artifact_path.parents[1].name
        model_folder = artifact_path.parent.name
        if theme_filter and theme_name not in theme_filter:
            continue
        if model_filter and model_folder not in model_filter:
            continue
        if document_filter:
            if not any(artifact_path.name.startswith(f"{document_id}_") for document_id in document_filter):
                continue
        if setting_filter:
            payload = _read_yaml_payload(artifact_path)
            document_setting = str(payload.get("document_setting") or "").strip().lower()
            if document_setting not in setting_filter:
                continue
        matched_paths.append(artifact_path)
    return matched_paths


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


def _raw_result_is_current_for_question(result: dict | None, document: object, question: object) -> bool:
    if not isinstance(result, dict):
        return False
    expected_prompt = build_document_question_prompt(
        document.document_text,
        question.question_text,
        answer_schema=question.answer_schema,
    )
    if str(result.get("user_prompt") or "") != expected_prompt:
        return False
    if str(result.get("raw_output") or "").strip() == "":
        return False
    if not str(result.get("raw_output") or "").lstrip().startswith("ANSWER:"):
        return False
    raw_provider_response = result.get("raw_provider_response")
    if raw_provider_response:
        try:
            raw_payload = json.loads(str(raw_provider_response))
        except json.JSONDecodeError:
            raw_payload = {}
        if raw_payload.get("stop_reason") == "max_tokens":
            return False
    return not _should_preserve_raw_metadata(result, question)


def _raw_result_payload(*, model_spec: object, document: object, question: object) -> dict:
    user_prompt = build_document_question_prompt(
        document.document_text,
        question.question_text,
        answer_schema=question.answer_schema,
    )
    response = generate_text(
        TextGenerationRequest(
            provider=model_spec.provider,
            model=model_spec.model_name,
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=model_spec.temperature,
            max_tokens=min(
                model_spec.max_tokens,
                suggested_generation_max_tokens(
                    answer_schema=question.answer_schema,
                    model_name=model_spec.registry_name,
                ),
            ),
            seed=model_spec.seed,
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
        "raw_output": response.text,
        "raw_reasoning": response.reasoning_text,
        "raw_provider_response": response.raw_response,
    }


def run_raw_model_calls(
    *,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] = ("factual", "fictional"),
    question_ids: Sequence[str] | None = None,
    question_types: Sequence[str] | None = None,
    overwrite: bool = False,
    reproducibility_manifest: EvaluationReproducibilityManifest | None = None,
) -> list[Path]:
    """Call the configured models on the requested benchmark document settings."""
    ensure_dataset_artifact_directories()
    question_id_filter = _normalized_question_ids(question_ids)
    question_type_filter = _normalized_question_types(question_types)
    has_question_filter = bool(question_id_filter or question_type_filter)
    model_specs = resolve_model_specs(model_names)
    evaluation_documents = list(
        iter_evaluation_documents(
            settings=settings,
            themes=themes,
            document_ids=document_ids,
        )
    )
    written_paths: list[Path] = []

    for model_spec in model_specs:
        for document in evaluation_documents:
            output_path = model_eval_artifact_path(
                theme=document.document_theme,
                model_name=model_spec.registry_name,
                document_id=document.document_id,
                setting=document.document_setting,
                stage_suffix="raw_outputs",
                variant_id=None
                if document.document_variant_index == 1 and document.source_path.stem == document.document_id
                else document.document_variant_id,
            )
            output_payload_is_current = False
            if output_path.exists() and not overwrite:
                output_payload_is_current = _saved_payload_is_current(output_path, check_score_metadata=False)
                if not output_payload_is_current:
                    _remove_downstream_stage_artifacts(output_path)

            existing_results_by_qid: dict[str, dict] = {}
            if output_path.exists() and not overwrite and output_payload_is_current:
                existing_payload = _read_yaml_payload(output_path)
                existing_results_by_qid = {
                    str(result.get("question_id") or ""): result
                    for result in (existing_payload.get("results") or [])
                    if isinstance(result, dict)
                }
                if not any(
                    not _raw_result_is_current_for_question(
                        existing_results_by_qid.get(question.question_id),
                        document,
                        question,
                    )
                    for question in document.questions
                ):
                    written_paths.append(output_path)
                    continue
                _remove_downstream_stage_artifacts(output_path)
            elif has_question_filter and output_path.exists():
                existing_payload = _read_yaml_payload(output_path)
                existing_results_by_qid = {
                    str(result.get("question_id") or ""): result
                    for result in (existing_payload.get("results") or [])
                    if isinstance(result, dict)
                }

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

            results = []
            regenerated_any = False
            for question in document.questions:
                existing_result = existing_results_by_qid.get(question.question_id)
                existing_result_is_current = _raw_result_is_current_for_question(
                    existing_result,
                    document,
                    question,
                )
                should_generate = not existing_result_is_current
                if has_question_filter and not should_generate:
                    should_generate = _question_matches_filter(
                        question,
                        question_ids=question_id_filter,
                        question_types=question_type_filter,
                    )
                if should_generate:
                    regenerated_any = True
                    results.append(
                        _raw_result_payload(
                            model_spec=model_spec,
                            document=document,
                            question=question,
                        )
                    )
                else:
                    results.append(existing_result)

            payload = {
                "model_name": model_spec.registry_name,
                "model_provider": model_spec.provider,
                "provider_model_name": model_spec.model_name,
                "document_theme": document.document_theme,
                "document_id": document.document_id,
                "document_setting": document.document_setting,
                "document_setting_family": document.document_setting_family,
                "document_variant_id": document.document_variant_id,
                "document_variant_index": document.document_variant_index,
                "replacement_proportion": document.replacement_proportion,
                "source_document_path": str(document.source_path.relative_to(PROJECT_ROOT)),
                "prompt_format_version": PROMPT_FORMAT_VERSION,
                "system_prompt": DOCUMENT_QA_SYSTEM_PROMPT,
                "results": results,
            }
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
            if regenerated_any:
                _remove_downstream_stage_artifacts(output_path)
    return written_paths


def parse_saved_outputs(
    *,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    overwrite: bool = False,
    reproducibility_manifest: EvaluationReproducibilityManifest | None = None,
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
            if _saved_payload_is_current(parsed_output_path, check_score_metadata=True):
                written_paths.append(parsed_output_path)
                continue

        payload = _read_yaml_payload(raw_output_path)
        parsed_results = []
        for result in payload.get("results", []) or []:
            source_question = _resolve_source_question(payload, result)
            if _should_skip_deleted_question(source_question):
                continue
            answer_schema = str(result.get("answer_schema") or "").strip()
            accepted_answers = tuple(str(answer) for answer in (result.get("accepted_answers") or []) if str(answer).strip())
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
                    "accepted_answer_overrides": list(
                        getattr(source_question, "accepted_answer_overrides", ()) or ()
                    ),
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
            {**payload, "results": parsed_results},
            reproducibility_manifest=reproducibility_manifest,
            stage_name="parse",
        )
        written_paths.append(_write_yaml(parsed_payload, parsed_output_path))
    return written_paths


def evaluate_saved_outputs(
    *,
    judge_config: JudgeConfig | None,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    overwrite: bool = False,
    reproducibility_manifest: EvaluationReproducibilityManifest | None = None,
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
        evaluated_output_path = parsed_output_path.with_name(
            parsed_output_path.name.replace("_parsed_outputs.yaml", "_evaluated_outputs.yaml")
        )
        if evaluated_output_path.exists() and not overwrite:
            if _saved_payload_is_current(evaluated_output_path, check_score_metadata=True):
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
            accepted_answers = tuple(str(answer).strip() for answer in (result.get("accepted_answers") or []) if str(answer).strip())
            accepted_answers_canonical = tuple(
                str(answer).strip() for answer in (result.get("accepted_answers_canonical") or []) if str(answer).strip()
            )
            if source_question is not None and not _should_preserve_raw_metadata(result, source_question):
                result = {
                    **result,
                    "question_type": source_question.question_type,
                    "answer_behavior": source_question.answer_behavior,
                    "question_text": source_question.question_text,
                    "ground_truth": source_question.ground_truth,
                    "answer_expression": source_question.answer_expression,
                    "accepted_answer_overrides": list(
                        getattr(source_question, "accepted_answer_overrides", ()) or ()
                    ),
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
            if not exact_match and judge_config is not None and judge_match_is_allowed(
                answer_schema=answer_schema,
                parsed_output_canonical=parsed_output_canonical,
            ):
                judge_match, judge_raw_output = judge_prediction(
                    question_text=str(result.get("question_text", "")).strip(),
                    ground_truth=ground_truth,
                    predicted_answer=parsed_output or str(result.get("raw_output") or "").strip(),
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
                "judge_provider": judge_config.provider if judge_config is not None else None,
                "judge_model_name": judge_config.model_name if judge_config is not None else None,
                "results": evaluated_results,
            },
            reproducibility_manifest=reproducibility_manifest,
            stage_name="evaluate",
        )
        written_paths.append(_write_yaml(evaluated_payload, evaluated_output_path))
    return written_paths


def compute_and_save_metrics(
    *,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
    reproducibility_manifest: EvaluationReproducibilityManifest | None = None,
) -> list[Path]:
    """Aggregate evaluated outputs into one metrics YAML file per model."""
    from .metrics import compute_metrics_for_model

    payloads_by_model: dict[str, list[dict]] = defaultdict(list)
    for evaluated_output_path in _iter_stage_paths(
        stage_suffix="evaluated_outputs",
        model_names=model_names,
        themes=themes,
        document_ids=document_ids,
        settings=settings,
    ):
        payload = _read_yaml_payload(evaluated_output_path)
        issues = blocking_issues(
            audit_saved_output_payload(
                payload,
                path=evaluated_output_path,
                check_score_metadata=True,
            )
        )
        if issues:
            raise RuntimeError(
                f"Refusing to aggregate stale evaluated output into metrics: {format_issue_summary(issues)}"
            )
        model_name = str(payload.get("model_name") or evaluated_output_path.parent.name)
        payloads_by_model[model_name].append(payload)

    written_paths: list[Path] = []
    MODEL_EVAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for model_name, payloads in sorted(payloads_by_model.items()):
        metrics_payload = compute_metrics_for_model(model_name, payloads)
        written_paths.append(
            _write_yaml(
                _attach_reproducibility_manifest_reference(
                    metrics_payload,
                    reproducibility_manifest=reproducibility_manifest,
                    stage_name="metrics",
                ),
                metrics_output_path(model_name),
            )
        )
    return written_paths


def build_metric_plots(*, model_names: Sequence[str] | None = None) -> list[Path]:
    """Render the benchmark plots from the saved metrics files."""
    from .metrics import load_metrics_payload
    from .plots import (
        export_paired_performance_drop_tables,
        plot_accuracy_breakdown,
        plot_accuracy_by_question_type_and_answer_behavior,
        plot_factual_vs_fictional_accuracy_bars,
        plot_paired_reasoning_fragility_effects,
        plot_question_type_accuracy_heatmap,
        plot_question_type_answer_behavior_variant_distributions,
        plot_question_type_answer_behavior_variant_distributions_by_model,
        plot_question_type_variant_distributions,
        plot_question_type_variant_distributions_by_model,
        plot_theme_accuracy_curves,
        plot_conversion_ratios,
        plot_factual_vs_fictional_variant_distributions,
        plot_setting_accuracy,
    )

    if model_names:
        metric_paths = [
            metrics_output_path(model_name) for model_name in model_names if metrics_output_path(model_name).exists()
        ]
    else:
        metric_paths = sorted(MODEL_EVAL_METRICS_DIR.glob("*_metrics.yaml"))
    metrics_payloads = [load_metrics_payload(metric_path) for metric_path in metric_paths]
    if not metrics_payloads:
        return []
    written_paths = [
        plot_setting_accuracy(metrics_payloads),
        plot_factual_vs_fictional_accuracy_bars(metrics_payloads),
        plot_accuracy_breakdown(metrics_payloads, breakdown_key="question_type"),
        plot_question_type_accuracy_heatmap(metrics_payloads),
        plot_accuracy_breakdown(metrics_payloads, breakdown_key="answer_behavior"),
        plot_accuracy_by_question_type_and_answer_behavior(metrics_payloads),
        plot_conversion_ratios(metrics_payloads),
        plot_factual_vs_fictional_variant_distributions(metrics_payloads),
        plot_question_type_variant_distributions(metrics_payloads),
        plot_question_type_answer_behavior_variant_distributions(metrics_payloads),
    ]
    written_paths.extend(plot_paired_reasoning_fragility_effects(metrics_payloads))
    written_paths.extend(export_paired_performance_drop_tables(metrics_payloads))
    written_paths.extend(plot_question_type_answer_behavior_variant_distributions_by_model(metrics_payloads))
    written_paths.extend(plot_question_type_variant_distributions_by_model(metrics_payloads))
    written_paths.extend(plot_theme_accuracy_curves(metrics_payloads))
    return written_paths


def run_parametric_shortcut_evaluation(
    *,
    steps: Sequence[str],
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] = ("factual", "fictional"),
    question_ids: Sequence[str] | None = None,
    question_types: Sequence[str] | None = None,
    overwrite: bool = False,
    judge_config: JudgeConfig | None = None,
    run_label: str | None = None,
    run_notes: str | None = None,
    entrypoint: str | None = None,
    invocation_command: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    """Run any subset of the benchmark model-evaluation stages."""
    executed: dict[str, list[Path]] = {}
    normalized_steps = list(steps)
    if "all" in normalized_steps:
        normalized_steps = ["raw", "parse", "evaluate", "metrics", "plot"]

    reproducibility_manifest = EvaluationReproducibilityManifest(
        steps=normalized_steps,
        model_names=model_names,
        themes=themes,
        document_ids=document_ids,
        settings=settings,
        overwrite=overwrite,
        judge_config=judge_config,
        run_label=run_label,
        run_notes=run_notes,
        entrypoint=entrypoint,
        invocation_command=invocation_command,
    )

    try:
        if "raw" in normalized_steps:
            executed["raw"] = run_raw_model_calls(
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                question_ids=question_ids,
                question_types=question_types,
                overwrite=overwrite,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("raw", executed["raw"])
        if "parse" in normalized_steps:
            executed["parse"] = parse_saved_outputs(
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                overwrite=overwrite,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("parse", executed["parse"])
        if "evaluate" in normalized_steps:
            executed["evaluate"] = evaluate_saved_outputs(
                judge_config=judge_config,
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                overwrite=overwrite,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("evaluate", executed["evaluate"])
        if "metrics" in normalized_steps:
            executed["metrics"] = compute_and_save_metrics(
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("metrics", executed["metrics"])
        if "plot" in normalized_steps:
            executed["plot"] = build_metric_plots(model_names=model_names)
            reproducibility_manifest.record_stage("plot", executed["plot"])
    except Exception as exc:
        executed["reproducibility_manifest"] = [reproducibility_manifest.mark_failed(exc)]
        raise

    executed["reproducibility_manifest"] = [reproducibility_manifest.mark_completed()]
    return executed
