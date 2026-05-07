#!/usr/bin/env python3
"""Plot factual/fictional performance against estimated inference FLOPs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)


DEFAULT_PERFORMANCE_CSV = (
    _ROOT
    / "data"
    / "MODEL_EVAL"
    / "PLOTS"
    / "performance_drop_ttest"
    / "paper_figure3_judge_match_drop_all_reasoning_extractive.csv"
)
DEFAULT_OUTPUT_DIR = _ROOT / "data" / "MODEL_EVAL" / "PLOTS" / "inference_cost_performance"
DEFAULT_PROMPT_TOKENS = 612
DEFAULT_GENERATED_TOKENS = 3


@dataclass(frozen=True)
class ModelSpec:
    registry_name: str
    label: str
    provider_model_id: str
    active_params_b: float | None
    total_params_b: float | None
    cached_flops_t: float | None
    compute_status: str = "open"


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        registry_name="olmo-3-7b-think",
        label="OLMo-3 7B Think",
        provider_model_id="allenai/Olmo-3-7B-Think",
        active_params_b=7.0,
        total_params_b=7.0,
        cached_flops_t=8.017284317184,
    ),
    ModelSpec(
        registry_name="olmo-3-7b-instruct",
        label="OLMo-3 7B Instruct",
        provider_model_id="allenai/Olmo-3-7B-Instruct",
        active_params_b=7.0,
        total_params_b=7.0,
        cached_flops_t=8.017284317184,
    ),
    ModelSpec(
        registry_name="gpt-oss-20b-groq",
        label="GPT-OSS 20B",
        provider_model_id="openai/gpt-oss-20b",
        active_params_b=3.6,
        total_params_b=21.0,
        cached_flops_t=3.53204527104,
    ),
    ModelSpec(
        registry_name="gemma-4-26b-a4b-it",
        label="Gemma-4 26B-A4B-IT",
        provider_model_id="google/gemma-4-26B-A4B-it",
        active_params_b=4.0,
        total_params_b=26.0,
        cached_flops_t=2.805438096384,
    ),
    ModelSpec(
        registry_name="gpt-oss-120b-groq",
        label="GPT-OSS 120B",
        provider_model_id="openai/gpt-oss-120b",
        active_params_b=5.1,
        total_params_b=117.0,
        cached_flops_t=5.30857304064,
    ),
    ModelSpec(
        registry_name="qwen3.5-27b",
        label="Qwen3.5 27B",
        provider_model_id="Qwen/Qwen3.5-27B",
        active_params_b=27.0,
        total_params_b=27.0,
        cached_flops_t=26.13229092864,
    ),
    ModelSpec(
        registry_name="qwen3.5-35b-a3b",
        label="Qwen3.5 35B-A3B",
        provider_model_id="Qwen/Qwen3.5-35B-A3B",
        active_params_b=3.0,
        total_params_b=35.0,
        cached_flops_t=1.96850737152,
    ),
    ModelSpec(
        registry_name="claude-sonnet-4-6",
        label="Claude Sonnet 4.6",
        provider_model_id="claude-sonnet-4-6",
        active_params_b=None,
        total_params_b=None,
        cached_flops_t=None,
        compute_status="closed_unavailable",
    ),
)

PROVIDER_COLORS = {
    "OLMo": "#3B6EA8",
    "GPT": "#2E9D73",
    "Gemma": "#C44E52",
    "Qwen": "#8A5FBF",
}

LABEL_OFFSETS = {
    "qwen3.5-35b-a3b": (12, -20),
    "gemma-4-26b-a4b-it": (14, 13),
    "gpt-oss-20b-groq": (14, -6),
    "gpt-oss-120b-groq": (16, 15),
    "olmo-3-7b-think": (12, 11),
    "olmo-3-7b-instruct": (12, -25),
    "qwen3.5-27b": (-14, 13),
}

LABEL_OFFSETS_BY_METRIC = {
    "factual_accuracy_pp": {
        "qwen3.5-35b-a3b": (12, -25),
        "gemma-4-26b-a4b-it": (-16, 26),
        "gpt-oss-20b-groq": (18, -30),
        "gpt-oss-120b-groq": (16, 18),
    }
}


def _provider_group(registry_name: str) -> str:
    if registry_name.startswith("olmo"):
        return "OLMo"
    if registry_name.startswith("gpt"):
        return "GPT"
    if registry_name.startswith("gemma"):
        return "Gemma"
    if registry_name.startswith("qwen"):
        return "Qwen"
    return "Other"


def _read_performance_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("question_type") == "all":
            selected[str(row["model_name"])] = row
    return selected


def _refresh_flops(model_id: str, prompt_tokens: int, generated_tokens: int) -> float:
    raise RuntimeError(
        "The anonymous public release ships cached FLOPs estimates only. "
        f"Cannot refresh {model_id!r} from local architecture helpers."
    )


def _joined_rows(
    *,
    performance_rows: dict[str, dict[str, str]],
    refresh_flops: bool,
    prompt_tokens: int,
    generated_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in MODEL_SPECS:
        performance = performance_rows.get(model.registry_name)
        if performance is None:
            raise ValueError(f"Missing all-question performance row for {model.registry_name!r}.")

        flops_t = model.cached_flops_t
        flops_source = "cached_kv_cache_formula"
        omitted_reason = ""
        if model.compute_status != "open":
            flops_t = None
            flops_source = "unavailable"
            omitted_reason = "closed model; public architecture/active-parameter details unavailable"
        elif refresh_flops:
            flops_t = _refresh_flops(model.provider_model_id, prompt_tokens, generated_tokens)
            flops_source = "refreshed_hf_config_kv_cache_formula"

        rows.append(
            {
                "model_name": model.registry_name,
                "model_label": model.label,
                "provider_model_id": model.provider_model_id,
                "provider_group": _provider_group(model.registry_name),
                "active_params_b": model.active_params_b,
                "total_params_b": model.total_params_b,
                "flops_t": flops_t,
                "flops": None if flops_t is None else flops_t * 1e12,
                "flops_source": flops_source,
                "prompt_tokens": prompt_tokens if flops_t is not None else None,
                "generated_tokens": generated_tokens if flops_t is not None else None,
                "plot_included": flops_t is not None,
                "omitted_reason": omitted_reason,
                "count_pairs": int(performance["count_pairs"]),
                "factual_score": float(performance["factual_score"]),
                "fictional_score": float(performance["fictional_score"]),
                "factual_accuracy_pp": 100.0 * float(performance["factual_score"]),
                "fictional_accuracy_pp": 100.0 * float(performance["fictional_score"]),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, rows: list[dict[str, Any]], *, prompt_tokens: int, generated_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "score_definition": "final_is_correct = exact_match OR judge_match",
        "flops_definition": "average inference FLOPs per QA example, KV-cache-aware estimate",
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plot_one(rows: list[dict[str, Any]], *, metric: str, title: str, output_stem: Path) -> None:
    plot_rows = [row for row in rows if row["plot_included"]]
    if not plot_rows:
        raise ValueError("No models have valid FLOPs for plotting.")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.4,
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    _draw_scatter(ax, plot_rows, metric=metric, title=title)
    fig.tight_layout()
    _save_all(fig, output_stem)
    plt.close(fig)


def _plot_combined(rows: list[dict[str, Any]], output_stem: Path) -> None:
    plot_rows = [row for row in rows if row["plot_included"]]
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.6), sharex=True)
    _draw_scatter(axes[0], plot_rows, metric="fictional_accuracy_pp", title="Fictional", annotate=False)
    _draw_scatter(axes[1], plot_rows, metric="factual_accuracy_pp", title="Factual", annotate=False)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save_all(fig, output_stem)
    plt.close(fig)


def _draw_scatter(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    annotate: bool = True,
) -> None:
    rows = sorted(rows, key=lambda row: float(row["flops_t"]))
    x_values = [float(row["flops_t"]) for row in rows]
    y_values = [float(row[metric]) for row in rows]
    ax.set_xscale("log")
    ax.set_xlim(min(x_values) / 1.25, max(x_values) * 1.35)
    y_min = math.floor((min(y_values) - 3.0) / 5.0) * 5.0
    y_max = math.ceil((max(y_values) + 3.0) / 5.0) * 5.0
    ax.set_ylim(max(0.0, y_min), min(100.0, y_max))

    for provider in ("Qwen", "Gemma", "GPT", "OLMo"):
        group = [row for row in rows if row["provider_group"] == provider]
        if not group:
            continue
        ax.scatter(
            [float(row["flops_t"]) for row in group],
            [float(row[metric]) for row in group],
            s=120,
            color=PROVIDER_COLORS[provider],
            edgecolor="white",
            linewidth=1.2,
            alpha=0.95,
            label=provider,
            zorder=4,
        )

    if annotate:
        metric_offsets = LABEL_OFFSETS_BY_METRIC.get(metric, {})
        for row in rows:
            model_name = str(row["model_name"])
            dx, dy = metric_offsets.get(model_name, LABEL_OFFSETS.get(model_name, (10, 10)))
            ax.annotate(
                str(row["model_label"]),
                xy=(float(row["flops_t"]), float(row[metric])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=10.2,
                fontweight="bold",
                color="#141414",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "#D5DCE5",
                    "linewidth": 0.8,
                    "alpha": 0.9,
                },
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#8A96A8",
                    "lw": 0.8,
                    "alpha": 0.8,
                    "shrinkA": 2,
                    "shrinkB": 2,
                },
                zorder=5,
            )

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Average inference cost per QA example (TFLOPs, log scale)", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.32, linewidth=0.9, color="#AAB3C0")
    ax.grid(axis="x", linestyle="--", alpha=0.18, linewidth=0.8, color="#CBD2DC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.legend(loc="lower right", frameon=True, edgecolor="#222222", framealpha=0.95)


def _save_all(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 300
        fig.savefig(output_stem.with_suffix(suffix), **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance-csv", type=Path, default=DEFAULT_PERFORMANCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--refresh-flops",
        action="store_true",
        help="Recompute FLOPs from Hugging Face configs instead of using cached estimates.",
    )
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--generated-tokens", type=int, default=DEFAULT_GENERATED_TOKENS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    performance_rows = _read_performance_rows(args.performance_csv)
    joined = _joined_rows(
        performance_rows=performance_rows,
        refresh_flops=bool(args.refresh_flops),
        prompt_tokens=int(args.prompt_tokens),
        generated_tokens=int(args.generated_tokens),
    )

    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "performance_vs_inference_flops.csv", joined)
    _write_json(
        output_dir / "performance_vs_inference_flops.json",
        joined,
        prompt_tokens=int(args.prompt_tokens),
        generated_tokens=int(args.generated_tokens),
    )
    _plot_one(
        joined,
        metric="fictional_accuracy_pp",
        title="Fictional Performance vs. Inference Cost",
        output_stem=output_dir / "fictional_performance_vs_inference_flops",
    )
    _plot_one(
        joined,
        metric="factual_accuracy_pp",
        title="Factual Performance vs. Inference Cost",
        output_stem=output_dir / "factual_performance_vs_inference_flops",
    )
    _plot_combined(joined, output_dir / "factual_fictional_performance_vs_inference_flops")

    print(f"output_dir: {output_dir}")
    print(f"rows: {len(joined)}")
    print(f"plotted_rows: {sum(1 for row in joined if row['plot_included'])}")
    print("omitted_rows:")
    for row in joined:
        if not row["plot_included"]:
            print(f"  {row['model_name']}: {row['omitted_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
