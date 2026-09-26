#!/usr/bin/env python3
"""Generate, parse, and score MemoReason model answers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
os.chdir(PROJECT_ROOT)


def _resolved_cli_path(path: Path) -> Path:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = PROJECT_ROOT / expanded
    return expanded.resolve()


def _display_artifact_path(path: Path) -> Path:
    """Render repository artifacts relatively and external artifacts absolutely."""
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT)
    except ValueError:
        return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the benchmark model-evaluation workflow")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", "raw", "parse", "evaluate"],
        required=True,
        help="Workflow stages to execute",
    )
    parser.add_argument("--models", nargs="+", required=True, help="Explicit model registry names to evaluate")
    parser.add_argument("--themes", nargs="+", default=None, help="Theme folders to evaluate")
    parser.add_argument("--docs", nargs="+", default=None, help="Document ids to evaluate")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="Question ids to regenerate during the raw stage; unchanged rows are reused.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        help="Question types to regenerate during the raw stage; unchanged rows are reused.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        required=True,
        help="Explicit benchmark setting ids, for example: factual fictional fictional_20pct",
    )
    parser.add_argument(
        "--fictional-proportions",
        nargs="+",
        type=float,
        default=None,
        help="Replacement proportions for fictional settings, for example: 0.2 0.5 0.8 1.0",
    )
    parser.add_argument("--skip-factual", action="store_true", help="Do not evaluate the factual document setting")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Regenerate complete artifacts when no question filter is supplied; "
            "with question filters, regenerate matching raw rows and reuse current non-matching rows."
        ),
    )
    parser.add_argument(
        "--refresh-stale-only",
        action="store_true",
        help=(
            "During the raw stage, reuse rows whose rebuilt prompt and execution context are current, "
            "and regenerate only stale rows. Compatible dataset-root rebases are verified by relative "
            "source-document identity."
        ),
    )
    parser.add_argument("--run-label", default=None, help="Short label stored in the reproducibility manifest")
    parser.add_argument("--run-notes", default=None, help="Free-form notes stored in the reproducibility manifest")
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=None,
        help="Override the selected model registry temperature for this isolated run.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Override the selected model registry seed for this isolated run.",
    )
    parser.add_argument(
        "--factual-documents-dir",
        type=Path,
        default=None,
        help="Read factual YAML files from this directory instead of data/FACTUAL_DOCUMENTS.",
    )
    parser.add_argument(
        "--fictional-documents-dir",
        type=Path,
        default=None,
        help=(
            "Read fictional YAML files from this directory instead of data/FICTIONAL_DOCUMENTS. "
            "Paper-era snapshots may contain partial_fictional_replacements/ beneath it."
        ),
    )
    parser.add_argument(
        "--model-eval-dir",
        type=Path,
        required=True,
        help="Write raw, parsed, evaluated, and provenance artifacts below this directory.",
    )
    parser.add_argument(
        "--dataset-revision-manifest",
        type=Path,
        default=None,
        help="Optional immutable dataset-overlay manifest to hash and record in the run manifest.",
    )
    parser.add_argument("--skip-judge", action="store_true", help="Skip the LLM-as-a-judge pass")
    parser.add_argument(
        "--judge-provider",
        default=None,
        choices=["anthropic", "groq", "local"],
        help="Explicit provider for a fresh Judge Match run; there is no implicit paper-protocol default",
    )
    parser.add_argument("--judge-model", default=None, help="Explicit model used for fresh judge scoring")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--judge-max-tokens", type=int, default=8, help="Judge max tokens")
    parser.add_argument("--judge-seed", type=int, default=23, help="Judge seed when supported")
    parser.add_argument(
        "--allow-model-execution", action="store_true",
        help="Explicitly allow GPU inference or paid provider calls; unnecessary for parse or evaluate --skip-judge",
    )
    args = parser.parse_args()

    generation = bool({"all", "raw"}.intersection(args.steps))
    judge = bool({"all", "evaluate"}.intersection(args.steps)) and not args.skip_judge
    if (generation or judge) and not args.allow_model_execution:
        parser.error("Inference/Judge Match requires explicit --allow-model-execution; inspect the selected scope first")
    if judge and (args.judge_provider is None or args.judge_model is None):
        parser.error("Fresh Judge Match requires explicit --judge-provider and --judge-model")
    for attribute_name, label in (
        ("factual_documents_dir", "Factual documents directory"),
        ("fictional_documents_dir", "Fictional documents directory"),
    ):
        configured_path = getattr(args, attribute_name)
        if configured_path is None:
            continue
        resolved_path = _resolved_cli_path(configured_path)
        if not resolved_path.is_dir():
            parser.error(f"{label} must already exist and be a directory: {resolved_path}")
        setattr(args, attribute_name, resolved_path)
    if args.model_eval_dir is not None:
        args.model_eval_dir = _resolved_cli_path(args.model_eval_dir)
        if args.model_eval_dir.exists() and not args.model_eval_dir.is_dir():
            parser.error(f"Model-evaluation output path is not a directory: {args.model_eval_dir}")

    path_overrides = {
        "MEMOREASON_FACTUAL_DOCUMENTS_DIR": args.factual_documents_dir,
        "MEMOREASON_FICTIONAL_DOCUMENTS_DIR": args.fictional_documents_dir,
        "MEMOREASON_MODEL_EVAL_DIR": args.model_eval_dir,
    }
    for environment_variable, configured_path in path_overrides.items():
        if configured_path is not None:
            os.environ[environment_variable] = str(configured_path.expanduser().resolve())

    # Import after applying path overrides: dataset_paths intentionally resolves
    # these process-level inputs once, making every downstream stage agree on
    # the exact same input and output roots.
    from memoreason.factual_to_fictional_dataset.dataset_settings import resolve_dataset_settings
    from memoreason.model_evaluation.model_answer_evaluation_pipeline import (
        generate_parse_and_score_model_answers,
    )
    from memoreason.model_evaluation.exact_and_judge_match_scoring import JudgeMatchConfiguration

    judge_config = None
    if judge:
        judge_config = JudgeMatchConfiguration(
            provider=args.judge_provider,
            model_name=args.judge_model,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            seed=args.judge_seed,
        )

    setting_specs = resolve_dataset_settings(
        explicit_settings=args.settings,
        include_factual=not args.skip_factual,
        fictional_proportions=args.fictional_proportions,
    )
    executed = generate_parse_and_score_model_answers(
        steps=args.steps,
        model_names=args.models,
        themes=args.themes,
        document_ids=args.docs,
        settings=[spec.setting_id for spec in setting_specs],
        question_ids=args.question_ids,
        question_types=args.question_types,
        overwrite=args.overwrite,
        refresh_stale_only=args.refresh_stale_only,
        generation_temperature=args.generation_temperature,
        generation_seed=args.generation_seed,
        judge_config=judge_config,
        run_label=args.run_label,
        run_notes=args.run_notes,
        dataset_revision_manifest=args.dataset_revision_manifest,
        entrypoint=str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        invocation_command=sys.argv,
    )
    for stage_name, output_paths in executed.items():
        print(f"{stage_name}: {len(output_paths)} artifact(s)")
        for output_path in output_paths:
            print(_display_artifact_path(output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
