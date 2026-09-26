#!/usr/bin/env python3
"""Run paper experiments or build tables/figures from your evaluated outputs."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from analysis import ablations, effort, endpoint, plots, temperature  # noqa: E402
from analysis import paper_reports as reports  # noqa: E402
from analysis.entity_counts import collect_counts  # noqa: E402
from analysis.paper_inputs import JUDGE_MODEL  # noqa: E402

EXPERIMENTS = ("main", "partial", "effort", "ablations", "temperature")


def experiment_jobs(experiment):
    """(relative output, model, settings, temperature, seed, reasoning effort)."""
    if experiment == "main":
        return [(Path("main") / m, m, reports.ENDPOINTS, 0.0, 23, "low") for m in endpoint.MODEL_ORDER]
    if experiment == "partial":
        return [(Path("partial") / m, m, reports.PARTIAL, 0.0, 23, "low") for m, _ in plots.PARTIAL_MODELS]
    if experiment == "effort":
        return [
            (Path("effort") / level, effort.MODEL, reports.ENDPOINTS, 0.0, 23, level) for level in ("medium", "high")
        ]
    if experiment == "ablations":
        return [
            (
                Path("ablations") / m,
                m,
                ("fictional_named", "fictional_numtemp") if m in ablations.MODELS else ("fictional_numtemp",),
                0.0,
                23,
                "low",
            )
            for m in (*ablations.MODELS, "claude-sonnet-4-6")
        ]
    if experiment == "temperature":
        return [
            (Path("temperature") / condition / m, m, reports.ENDPOINTS, t, seed, "low")
            for condition, t, seed in temperature.CONDITIONS
            if t != 0
            for m in temperature.MODELS
        ]
    raise ValueError(f"Unknown experiment: {experiment}")


def commands_for(job, dataset, runs):
    relative, model, settings, value, seed, level = job
    base = [
        sys.executable,
        str(ROOT / "scripts/model_evaluation/generate_parse_and_score_model_answers.py"),
        "--models",
        model,
        "--settings",
        *settings,
        "--factual-documents-dir",
        str(dataset / "FACTUAL_DOCUMENTS"),
        "--fictional-documents-dir",
        str(dataset / "FICTIONAL_DOCUMENTS"),
        "--model-eval-dir",
        str(runs / relative),
        "--generation-temperature",
        str(value),
        "--allow-model-execution",
    ]
    if model != "claude-sonnet-4-6":
        base += ["--generation-seed", str(seed)]
    # Separate passes: varying answer-model effort must not also vary the judge.
    return [
        (level, [*base, "--steps", "raw", "parse", "--skip-judge"]),
        (
            "low",
            [
                *base,
                "--steps",
                "evaluate",
                "--judge-provider",
                "groq",
                "--judge-model",
                JUDGE_MODEL,
                "--judge-max-tokens",
                "256",
                "--judge-temperature",
                "0",
                "--judge-seed",
                "23",
            ],
        ),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Print the run plan; add --execute for GPU/API execution")
    run.add_argument("experiment", choices=EXPERIMENTS)
    run.add_argument("--dataset", type=Path, default=Path("data/generated"))
    run.add_argument("--runs", type=Path, default=Path("output/runs"))
    run.add_argument("--models", nargs="+", help="Optional model subset for incremental execution")
    run.add_argument("--execute", action="store_true")
    analyze = sub.add_parser("analyze", help="CPU-only scoring summaries and plots; no model calls")
    analyze.add_argument("experiment", choices=("dataset", *EXPERIMENTS, "frequency"))
    analyze.add_argument("--runs", type=Path, default=Path("output/runs"))
    analyze.add_argument("--dataset", type=Path, default=Path("data/generated"))
    analyze.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite")
    analyze.add_argument("--bootstrap-samples", type=int, default=100_000)
    analyze.add_argument("--counts", type=Path, help="Document-frequency CSV from the counts command")
    counts = sub.add_parser("counts", help="Collect corpus-frequency inputs for Figure 4")
    counts.add_argument("--dataset", type=Path, default=Path("data/generated"))
    counts.add_argument("--output", type=Path, required=True)
    counts.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.dataset = args.dataset.expanduser().resolve()
    if args.command == "run":
        args.runs = args.runs.expanduser().resolve()
        jobs = experiment_jobs(args.experiment)
        if args.models:
            unknown = set(args.models) - {job[1] for job in jobs}
            if unknown:
                parser.error(f"Models outside this experiment: {sorted(unknown)}")
            jobs = [job for job in jobs if job[1] in args.models]
        for job in jobs:
            for level, command in commands_for(job, args.dataset, args.runs):
                print(
                    f"GROQ_GPT_OSS_REASONING_EFFORT={level} GROQ_INCLUDE_REASONING=false {shlex.join(command)}",
                    flush=True,
                )
                if args.execute:
                    env = dict(os.environ, GROQ_GPT_OSS_REASONING_EFFORT=level, GROQ_INCLUDE_REASONING="false")
                    subprocess.run(command, env=env, cwd=ROOT, check=True)
        return 0
    if args.command == "counts":
        collect_counts(args.dataset, args.output.expanduser().resolve(), execute=args.execute)
        return 0
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.experiment == "frequency" and args.counts is None:
        parser.error("frequency requires --counts (see the counts command)")
    out = args.output.expanduser().resolve()
    if out.exists():
        parser.error("Use a new --output directory")
    out.mkdir(parents=True, exist_ok=False)
    if args.experiment == "dataset":
        reports.dataset_report(args.dataset, out)
    elif args.experiment == "frequency":
        reports.frequency_report(args.runs.resolve(), out, args.bootstrap_samples, args.counts.resolve())
    else:
        functions = {
            "main": reports.main_report,
            "partial": reports.partial_report,
            "effort": reports.effort_report,
            "ablations": reports.ablation_report,
            "temperature": reports.temperature_report,
        }
        functions[args.experiment](args.runs.resolve(), out, args.bootstrap_samples)
    print(f"Wrote {args.experiment} tables/figures to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
