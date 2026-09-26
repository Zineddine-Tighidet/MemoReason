"""Orchestrate raw generation, answer parsing, and answer scoring."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .model_answer_parsing_and_scoring import score_parsed_model_answers, parse_raw_model_answers
from .model_evaluation_run_manifest import ModelEvaluationRunManifest
from .raw_model_answer_generation import generate_raw_model_answers
from .exact_and_judge_match_scoring import JudgeMatchConfiguration

__all__ = ["generate_parse_and_score_model_answers"]


def generate_parse_and_score_model_answers(
    *,
    steps: Sequence[str],
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
    judge_config: JudgeMatchConfiguration | None = None,
    run_label: str | None = None,
    run_notes: str | None = None,
    dataset_revision_manifest: Path | None = None,
    entrypoint: str | None = None,
    invocation_command: Sequence[str] | None = None,
) -> dict[str, list[Path]]:
    """Run any subset of the immutable raw, parse, and evaluate stages."""
    executed: dict[str, list[Path]] = {}
    normalized_steps = list(steps)
    if "all" in normalized_steps:
        normalized_steps = ["raw", "parse", "evaluate"]
    unknown_steps = sorted(set(normalized_steps) - {"raw", "parse", "evaluate"})
    if unknown_steps:
        raise ValueError(f"Unknown model-evaluation stages: {unknown_steps}")

    reproducibility_manifest = ModelEvaluationRunManifest(
        steps=normalized_steps,
        model_names=model_names,
        themes=themes,
        document_ids=document_ids,
        settings=settings,
        question_ids=question_ids,
        question_types=question_types,
        overwrite=overwrite,
        refresh_stale_only=refresh_stale_only,
        generation_temperature=generation_temperature,
        generation_seed=generation_seed,
        judge_config=judge_config,
        run_label=run_label,
        run_notes=run_notes,
        dataset_revision_manifest=dataset_revision_manifest,
        entrypoint=entrypoint,
        invocation_command=invocation_command,
    )

    try:
        if "raw" in normalized_steps:
            executed["raw"] = generate_raw_model_answers(
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                question_ids=question_ids,
                question_types=question_types,
                overwrite=overwrite,
                refresh_stale_only=refresh_stale_only,
                generation_temperature=generation_temperature,
                generation_seed=generation_seed,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("raw", executed["raw"])
        if "parse" in normalized_steps:
            executed["parse"] = parse_raw_model_answers(
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                overwrite=overwrite,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("parse", executed["parse"])
        if "evaluate" in normalized_steps:
            executed["evaluate"] = score_parsed_model_answers(
                judge_config=judge_config,
                model_names=model_names,
                themes=themes,
                document_ids=document_ids,
                settings=settings,
                overwrite=overwrite,
                reproducibility_manifest=reproducibility_manifest,
            )
            reproducibility_manifest.record_stage("evaluate", executed["evaluate"])
    except Exception as exc:
        executed["reproducibility_manifest"] = [reproducibility_manifest.mark_failed(exc)]
        raise

    executed["reproducibility_manifest"] = [reproducibility_manifest.mark_completed()]
    return executed
