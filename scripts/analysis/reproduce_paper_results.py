#!/usr/bin/env python3
"""Recompute the tables and figures used by the MemoReason paper."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_OUTPUTS_DIR = PROJECT_ROOT / "data" / "MODEL_EVAL" / "RAW_OUTPUTS"
FINAL_MODELS = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "claude-sonnet-4-6",
)
PARTIAL_MODELS = (
    "olmo-3-7b-instruct",
    "gpt-oss-120b-groq",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
)
FINAL_SETTINGS = (
    "factual",
    "fictional",
    "fictional_10pct",
    "fictional_20pct",
    "fictional_30pct",
    "fictional_50pct",
    "fictional_80pct",
    "fictional_90pct",
)


def _has_evaluated_outputs(raw_outputs_dir: Path) -> bool:
    return raw_outputs_dir.exists() and any(raw_outputs_dir.rglob("*_evaluated_outputs.yaml"))


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _python() -> str:
    local_python = PROJECT_ROOT / ".venv" / "bin" / "python3.11"
    return str(local_python if local_python.exists() else sys.executable)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(FINAL_MODELS))
    parser.add_argument("--partial-models", nargs="+", default=list(PARTIAL_MODELS))
    parser.add_argument("--allow-missing-inputs", action="store_true")
    parser.add_argument("--allow-stale-outputs", action="store_true")
    parser.add_argument("--allow-live-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not _has_evaluated_outputs(RAW_OUTPUTS_DIR) and not args.allow_missing_inputs:
        print(
            "No evaluated model outputs were found under data/MODEL_EVAL/RAW_OUTPUTS.\n"
            "This public repository intentionally does not vendor model outputs. "
            "Either rerun evaluations with scripts/benchmark/run_model_evaluation.py, "
            "or place compatible *_evaluated_outputs.yaml files in data/MODEL_EVAL/RAW_OUTPUTS "
            "and rerun this command. Use --allow-missing-inputs only to print the command plan.",
            file=sys.stderr,
        )
        return 2

    python = _python()
    models = [str(model) for model in args.models]
    partial_models = [str(model) for model in args.partial_models]
    stale_flag = ["--allow-stale-outputs"] if args.allow_stale_outputs else []
    judge_flag = ["--allow-live-judge"] if args.allow_live_judge else []

    commands = [
        [
            python,
            "scripts/analysis/audit_model_eval_version_consistency.py",
            "--models",
            *models,
            "--settings",
            *FINAL_SETTINGS,
        ],
        [
            python,
            "scripts/analysis/build_judge_match_drop_by_question_type_figure.py",
            "--models",
            *models,
            "--comparison-setting",
            "fictional",
            "--question-groups",
            "all",
            "reasoning",
            "extractive",
            "--output-stem",
            "paper_figure3_judge_match_drop_all_reasoning_extractive",
            "--variant-level-correction",
            "bonferroni",
            *stale_flag,
            *judge_flag,
        ],
        [
            python,
            "scripts/analysis/build_question_answer_type_drop_heatmap.py",
            "--models",
            *models,
            "--comparison-setting",
            "fictional",
            "--output-stem",
            "paper_figure3_question_answer_type_drop_heatmap",
            "--pvalue-correction",
            "bonferroni",
            *stale_flag,
            *judge_flag,
        ],
        [
            python,
            "scripts/analysis/build_question_answer_type_drop_latex_table.py",
        ],
        [
            python,
            "scripts/analysis/build_partial_replacement_drop_report.py",
            "--models",
            *partial_models,
            "--proportions",
            "0.1",
            "0.2",
            "0.3",
            "0.5",
            "0.8",
            "0.9",
        ],
        [
            python,
            "scripts/analysis/plot_partial_replacement_performance_curves.py",
        ],
        [
            python,
            "scripts/analysis/build_variant_factual_answer_shortcut_report.py",
            "--models",
            *models,
            *judge_flag,
        ],
        [
            python,
            "scripts/analysis/build_performance_vs_inference_flops_scatter.py",
        ],
    ]

    for command in commands:
        _run(command, dry_run=bool(args.dry_run or args.allow_missing_inputs and not _has_evaluated_outputs(RAW_OUTPUTS_DIR)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
