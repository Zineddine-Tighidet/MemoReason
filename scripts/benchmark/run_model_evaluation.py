#!/usr/bin/env python3
"""Run the raw/parse/evaluate/metrics/plot stages of the benchmark model-evaluation workflow."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)


def main() -> int:
    from src.dataset_export.dataset_settings import resolve_dataset_settings
    from src.evaluation_workflows.parametric_shortcut.dataset import DEFAULT_HF_DATASET_ID
    from src.evaluation_workflows.parametric_shortcut.parametric_shortcut_evaluation import (
        run_parametric_shortcut_evaluation,
    )
    from src.evaluation_workflows.parametric_shortcut.scoring import JudgeConfig

    parser = argparse.ArgumentParser(description="Run the benchmark model-evaluation workflow")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["all", "raw", "parse", "evaluate", "metrics", "plot"],
        default=["all"],
        help="Workflow stages to execute",
    )
    parser.add_argument("--models", nargs="+", default=None, help="Model registry names to evaluate")
    parser.add_argument("--themes", nargs="+", default=None, help="Theme folders to evaluate")
    parser.add_argument("--docs", nargs="+", default=None, help="Document ids to evaluate")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="Optional question ids to regenerate during the raw stage only.",
    )
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=None,
        help="Optional question types to regenerate during the raw stage only.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="Explicit benchmark setting ids, for example: factual fictional fictional_20pct",
    )
    parser.add_argument(
        "--dataset-source",
        choices=["local", "huggingface"],
        default="local",
        help="Read evaluation inputs from generated local YAML files or from the Hugging Face dataset.",
    )
    parser.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET_ID,
        help="Hugging Face dataset id used when --dataset-source huggingface is set.",
    )
    parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--fictional-proportions",
        nargs="+",
        type=float,
        default=None,
        help="Replacement proportions for fictional settings, for example: 0.2 0.5 0.8 1.0",
    )
    parser.add_argument("--skip-factual", action="store_true", help="Do not evaluate the factual document setting")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate artifacts even when they exist already")
    parser.add_argument("--run-label", default=None, help="Short label stored in the reproducibility manifest")
    parser.add_argument("--run-notes", default=None, help="Free-form notes stored in the reproducibility manifest")
    parser.add_argument("--skip-judge", action="store_true", help="Skip the LLM-as-a-judge pass")
    parser.add_argument(
        "--judge-provider",
        default="anthropic",
        choices=["anthropic", "gemini", "groq", "local"],
        help="Provider used for judge scoring",
    )
    parser.add_argument("--judge-model", default="claude-opus-4-6", help="Model used for judge scoring")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--judge-max-tokens", type=int, default=8, help="Judge max tokens")
    parser.add_argument("--judge-seed", type=int, default=23, help="Judge seed when supported")
    args = parser.parse_args()

    judge_config = None
    if not args.skip_judge:
        judge_config = JudgeConfig(
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
    executed = run_parametric_shortcut_evaluation(
        steps=args.steps,
        model_names=args.models,
        themes=args.themes,
        document_ids=args.docs,
        settings=[spec.setting_id for spec in setting_specs],
        question_ids=args.question_ids,
        question_types=args.question_types,
        dataset_source=args.dataset_source,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        overwrite=args.overwrite,
        judge_config=judge_config,
        run_label=args.run_label,
        run_notes=args.run_notes,
        entrypoint=str(Path(__file__).resolve().relative_to(_ROOT)),
        invocation_command=sys.argv,
    )
    for stage_name, output_paths in executed.items():
        print(f"{stage_name}: {len(output_paths)} artifact(s)")
        for output_path in output_paths:
            print(output_path.relative_to(_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
