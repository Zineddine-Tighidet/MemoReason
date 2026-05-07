#!/usr/bin/env python3
"""Plot paired factual-to-fictional drops by question type and answer type."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.build_judge_match_drop_by_question_type_figure import (
    BASELINE_SETTING,
    DEFAULT_COMPARISON_SETTING,
    EXPECTED_VARIANT_IDS,
    OUTPUT_DIR,
    _display_model_name,
    _load_cache,
    _load_pair_outcomes,
    _paired_t_test_and_ci,
)


DEFAULT_MODELS = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "claude-sonnet-4-6",
)
BASE_QUESTION_TYPES = ("arithmetic", "temporal", "inference", "extractive")
QUESTION_TYPES = ("reasoning", "arithmetic", "temporal", "inference", "extractive")
QUESTION_TYPE_MEMBERS = {
    "reasoning": ("arithmetic", "temporal", "inference"),
    "arithmetic": ("arithmetic",),
    "temporal": ("temporal",),
    "inference": ("inference",),
    "extractive": ("extractive",),
}
BASE_ANSWER_BEHAVIORS = ("variant", "invariant", "refusal")
ANSWER_BEHAVIORS = ("non_refusal", "variant", "invariant", "refusal")
ANSWER_BEHAVIOR_MEMBERS = {
    "non_refusal": ("variant", "invariant"),
    "variant": ("variant",),
    "invariant": ("invariant",),
    "refusal": ("refusal",),
}
QUESTION_TYPE_LABELS = {
    "reasoning": "Reasoning",
    "arithmetic": "Arithmetic",
    "temporal": "Temporal",
    "inference": "Inference",
    "extractive": "Extractive",
}
ANSWER_BEHAVIOR_LABELS = {
    "non_refusal": "Non-refusal",
    "variant": "Variant",
    "invariant": "Invariant",
    "refusal": "Refusal",
}


@dataclass(frozen=True)
class CellDrop:
    model_name: str
    question_type: str
    answer_behavior: str
    count_pairs: int
    factual_score: float
    fictional_score: float
    mean_difference: float
    mean_difference_pp: float
    standard_deviation_difference: float
    standard_error_difference: float
    degrees_of_freedom: int
    ci_low: float
    ci_high: float
    ci_low_pp: float
    ci_high_pp: float
    t_statistic: float
    p_value: float
    p_value_adjusted: float | None
    pvalue_correction: str | None
    correction_family_size: int | None
    significance_alpha: float
    significant: bool
    stars: str
    significance_source: str
    variant_p_values: dict[str, float]
    variant_p_values_adjusted: dict[str, float]
    significant_variant_ids: tuple[str, ...]


def _build_cells_for_model(
    model_name: str,
    *,
    comparison_setting: str,
    allow_live_judge: bool,
    require_current_outputs: bool,
) -> list[CellDrop]:
    outcomes, _judged_rows = _load_pair_outcomes(
        model_name,
        _load_cache(),
        allow_live_judge=allow_live_judge,
        comparison_setting=comparison_setting,
        require_current_outputs=require_current_outputs,
    )
    expected_variant_ids = set(EXPECTED_VARIANT_IDS)
    grouped_pairs: dict[tuple[str, str], list[tuple[float, float]]] = {
        (question_type, answer_behavior): []
        for answer_behavior in ANSWER_BEHAVIORS
        for question_type in QUESTION_TYPES
    }
    grouped_variant_pairs: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {
        (question_type, answer_behavior): {variant_id: [] for variant_id in EXPECTED_VARIANT_IDS}
        for answer_behavior in ANSWER_BEHAVIORS
        for question_type in QUESTION_TYPES
    }

    for entry in outcomes.values():
        factual = entry.get(BASELINE_SETTING)
        fictional = entry.get(comparison_setting) or {}
        question_type = str(entry.get("question_type") or "").strip().lower()
        answer_behavior = str(entry.get("answer_behavior") or "").strip().lower()
        if factual is None:
            continue
        if question_type not in BASE_QUESTION_TYPES or answer_behavior not in BASE_ANSWER_BEHAVIORS:
            continue
        if set(str(variant_id) for variant_id in fictional.keys()) != expected_variant_ids:
            continue

        fictional_mean = float(
            np.mean([float(fictional[variant_id]) for variant_id in EXPECTED_VARIANT_IDS])
        )
        for group_name in QUESTION_TYPES:
            if question_type not in QUESTION_TYPE_MEMBERS[group_name]:
                continue
            for answer_group in ANSWER_BEHAVIORS:
                if answer_behavior not in ANSWER_BEHAVIOR_MEMBERS[answer_group]:
                    continue
                grouped_pairs[(group_name, answer_group)].append((float(factual), fictional_mean))
                for variant_id in EXPECTED_VARIANT_IDS:
                    grouped_variant_pairs[(group_name, answer_group)][variant_id].append(
                        (float(factual), float(fictional[variant_id]))
                    )

    cells: list[CellDrop] = []
    for answer_behavior in ANSWER_BEHAVIORS:
        for question_type in QUESTION_TYPES:
            pairs = grouped_pairs[(question_type, answer_behavior)]
            if not pairs:
                continue
            factual_arr = np.asarray([pair[0] for pair in pairs], dtype=float)
            fictional_arr = np.asarray([pair[1] for pair in pairs], dtype=float)
            diff_arr = fictional_arr - factual_arr
            (
                mean_difference,
                standard_deviation_difference,
                degrees_of_freedom,
                standard_error_difference,
                ci_low,
                ci_high,
                t_statistic,
                p_value,
            ) = _paired_t_test_and_ci(diff_arr)
            variant_p_values: dict[str, float] = {}
            for variant_id in EXPECTED_VARIANT_IDS:
                variant_pairs = grouped_variant_pairs[(question_type, answer_behavior)][variant_id]
                if not variant_pairs:
                    continue
                variant_diff_arr = np.asarray([pair[1] - pair[0] for pair in variant_pairs], dtype=float)
                (
                    _variant_mean_difference,
                    _variant_standard_deviation_difference,
                    _variant_degrees_of_freedom,
                    _variant_standard_error_difference,
                    _variant_ci_low,
                    _variant_ci_high,
                    _variant_t_statistic,
                    variant_p_value,
                ) = _paired_t_test_and_ci(variant_diff_arr)
                variant_p_values[variant_id] = float(variant_p_value)
            significant = bool(p_value < 0.05)
            cells.append(
                CellDrop(
                    model_name=model_name,
                    question_type=question_type,
                    answer_behavior=answer_behavior,
                    count_pairs=int(diff_arr.size),
                    factual_score=float(np.mean(factual_arr)),
                    fictional_score=float(np.mean(fictional_arr)),
                    mean_difference=float(mean_difference),
                    mean_difference_pp=float(100.0 * mean_difference),
                    standard_deviation_difference=float(standard_deviation_difference),
                    standard_error_difference=float(standard_error_difference),
                    degrees_of_freedom=int(degrees_of_freedom),
                    ci_low=float(ci_low),
                    ci_high=float(ci_high),
                    ci_low_pp=float(100.0 * ci_low),
                    ci_high_pp=float(100.0 * ci_high),
                    t_statistic=float(t_statistic),
                    p_value=float(p_value),
                    p_value_adjusted=None,
                    pvalue_correction=None,
                    correction_family_size=None,
                    significance_alpha=0.05,
                    significant=significant,
                    stars="*" if significant else "",
                    significance_source="aggregate_paired_mean",
                    variant_p_values=variant_p_values,
                    variant_p_values_adjusted={},
                    significant_variant_ids=(),
                )
            )
    return cells


def build_cells(
    models: tuple[str, ...],
    *,
    comparison_setting: str,
    allow_live_judge: bool,
    require_current_outputs: bool,
) -> list[CellDrop]:
    cells: list[CellDrop] = []
    for model_name in models:
        model_cells = _build_cells_for_model(
            model_name,
            comparison_setting=comparison_setting,
            allow_live_judge=allow_live_judge,
            require_current_outputs=require_current_outputs,
        )
        if not model_cells:
            raise RuntimeError(f"No complete paired cells found for {model_name}.")
        cells.extend(model_cells)
        print(f"processed {model_name}: {len(model_cells)} cells", flush=True)
    return cells


def apply_pvalue_correction(
    cells: list[CellDrop],
    *,
    method: str,
    alpha: float,
) -> list[CellDrop]:
    method = str(method or "none").strip().lower()
    if method in {"", "none"}:
        return [
            replace(
                cell,
                p_value_adjusted=None,
                pvalue_correction=None,
                correction_family_size=None,
                significance_alpha=float(alpha),
                significant=bool(cell.p_value < float(alpha)),
                stars="*" if cell.p_value < float(alpha) else "",
                significance_source="aggregate_paired_mean",
                variant_p_values_adjusted={},
                significant_variant_ids=(),
            )
            for cell in cells
        ]
    if method != "bonferroni":
        raise ValueError(f"Unsupported p-value correction: {method}")

    family_size = len(EXPECTED_VARIANT_IDS)
    if family_size <= 0:
        return cells
    corrected_cells: list[CellDrop] = []
    for cell in cells:
        variant_p_values_adjusted = {
            variant_id: min(1.0, float(p_value) * float(family_size))
            for variant_id, p_value in cell.variant_p_values.items()
            if np.isfinite(float(p_value))
        }
        significant_variant_ids = tuple(
            variant_id
            for variant_id in EXPECTED_VARIANT_IDS
            if variant_p_values_adjusted.get(variant_id, 1.0) < float(alpha)
        )
        p_value_adjusted = min(variant_p_values_adjusted.values()) if variant_p_values_adjusted else None
        significant = bool(significant_variant_ids)
        corrected_cells.append(
            replace(
                cell,
                p_value_adjusted=p_value_adjusted,
                pvalue_correction="bonferroni",
                correction_family_size=family_size,
                significance_alpha=float(alpha),
                significant=significant,
                stars="*" if significant else "",
                significance_source="variant_level_factual_vs_each_fictional",
                variant_p_values_adjusted=variant_p_values_adjusted,
                significant_variant_ids=significant_variant_ids,
            )
        )
    return corrected_cells


def write_cells(
    cells: list[CellDrop],
    *,
    output_stem: str,
    models: tuple[str, ...],
    comparison_setting: str,
    pvalue_correction: str,
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    json_path = OUTPUT_DIR / f"{output_stem}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(asdict(cells[0]).keys()) if cells else [field.name for field in CellDrop.__dataclass_fields__.values()]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            record = asdict(cell)
            record["variant_p_values"] = json.dumps(record["variant_p_values"], sort_keys=True)
            record["variant_p_values_adjusted"] = json.dumps(
                record["variant_p_values_adjusted"],
                sort_keys=True,
            )
            record["significant_variant_ids"] = json.dumps(list(record["significant_variant_ids"]))
            writer.writerow(record)

    payload = {
        "metric": "mean paired difference, fictional - factual",
        "unit": "percentage points for *_pp fields, fraction for score fields",
        "comparison_setting": comparison_setting,
        "pvalue_correction": pvalue_correction,
        "correction_family_size": cells[0].correction_family_size if cells else None,
        "significance_alpha": cells[0].significance_alpha if cells else 0.05,
        "significance_source": cells[0].significance_source if cells else None,
        "models": list(models),
        "question_types": list(QUESTION_TYPES),
        "answer_behaviors": list(ANSWER_BEHAVIORS),
        "cells": [asdict(cell) for cell in cells],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return [csv_path, json_path]


def _cell_lookup(cells: list[CellDrop]) -> dict[tuple[str, str, str], CellDrop]:
    return {(cell.model_name, cell.question_type, cell.answer_behavior): cell for cell in cells}


def plot_heatmap(
    cells: list[CellDrop],
    *,
    models: tuple[str, ...],
    output_stem: str,
    figure_width: float,
    figure_height: float,
    color_limit: float | None,
) -> list[Path]:
    lookup = _cell_lookup(cells)
    values = np.asarray([cell.mean_difference_pp for cell in cells], dtype=float)
    finite_values = values[np.isfinite(values)]
    if color_limit is None:
        max_abs = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
        color_limit = max(5.0, float(np.ceil(max_abs / 2.5) * 2.5))

    ncols = 4
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for model_index, model_name in enumerate(models):
        row_index = model_index // ncols
        col_index = model_index % ncols
        axis = axes[row_index][col_index]
        matrix = np.full((len(ANSWER_BEHAVIORS), len(QUESTION_TYPES)), np.nan, dtype=float)
        for y_index, answer_behavior in enumerate(ANSWER_BEHAVIORS):
            for x_index, question_type in enumerate(QUESTION_TYPES):
                cell = lookup.get((model_name, question_type, answer_behavior))
                if cell is not None:
                    matrix[y_index, x_index] = cell.mean_difference_pp
        image = axis.imshow(matrix, cmap="RdBu", vmin=-color_limit, vmax=color_limit, aspect="auto")
        axis.set_title(_display_model_name(model_name), fontsize=10, pad=8)
        axis.set_xticks(np.arange(len(QUESTION_TYPES)))
        axis.set_xticklabels([QUESTION_TYPE_LABELS[item] for item in QUESTION_TYPES], rotation=30, ha="right", fontsize=8)
        axis.set_yticks(np.arange(len(ANSWER_BEHAVIORS)))
        if col_index == 0:
            axis.set_yticklabels([ANSWER_BEHAVIOR_LABELS[item] for item in ANSWER_BEHAVIORS], fontsize=8)
        else:
            axis.set_yticklabels([])
        axis.tick_params(length=0)

        axis.set_xticks(np.arange(-0.5, len(QUESTION_TYPES), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(ANSWER_BEHAVIORS), 1), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        axis.tick_params(which="minor", bottom=False, left=False)

        for y_index, answer_behavior in enumerate(ANSWER_BEHAVIORS):
            for x_index, question_type in enumerate(QUESTION_TYPES):
                cell = lookup.get((model_name, question_type, answer_behavior))
                if cell is None:
                    label = "n/a"
                    text_color = "#334155"
                else:
                    label = f"{cell.mean_difference_pp:+.1f}{cell.stars}"
                    text_color = "white" if abs(cell.mean_difference_pp) > 0.55 * float(color_limit) else "#111827"
                axis.text(x_index, y_index, label, ha="center", va="center", fontsize=8, color=text_color)

    for empty_index in range(len(models), nrows * ncols):
        axes[empty_index // ncols][empty_index % ncols].axis("off")

    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.012)
        colorbar.set_label(r"Mean paired $\Delta_i$ (fictional - factual, pp)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    output_paths: list[Path] = []
    for suffix, dpi in (("png", 300), ("pdf", None), ("svg", None)):
        path = OUTPUT_DIR / f"{output_stem}.{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        output_paths.append(path)
    plt.close(fig)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--comparison-setting", default=DEFAULT_COMPARISON_SETTING)
    parser.add_argument("--output-stem", default="paper_figure3_question_answer_type_drop_heatmap")
    parser.add_argument(
        "--pvalue-correction",
        default="none",
        choices=("none", "bonferroni"),
        help=(
            "Correction used for stars. bonferroni applies within each cell "
            "across the 10 factual-vs-individual-fictional paired t-tests."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--allow-live-judge", action="store_true")
    parser.add_argument(
        "--allow-stale-outputs",
        action="store_true",
        help="Skip evaluated-output version audits while drafting a visual.",
    )
    parser.add_argument("--figure-width", type=float, default=13.2)
    parser.add_argument("--figure-height", type=float, default=6.7)
    parser.add_argument(
        "--color-limit",
        type=float,
        help="Symmetric color scale limit in percentage points. Defaults to the next 2.5pp tick above the max absolute cell.",
    )
    args = parser.parse_args()

    models = tuple(str(model).strip() for model in args.models if str(model).strip())
    if not models:
        raise SystemExit("No models were provided.")

    cells = build_cells(
        models,
        comparison_setting=str(args.comparison_setting),
        allow_live_judge=bool(args.allow_live_judge),
        require_current_outputs=not bool(args.allow_stale_outputs),
    )
    cells = apply_pvalue_correction(
        cells,
        method=str(args.pvalue_correction),
        alpha=float(args.alpha),
    )
    output_paths = []
    output_paths.extend(
        write_cells(
            cells,
            output_stem=str(args.output_stem),
            models=models,
            comparison_setting=str(args.comparison_setting),
            pvalue_correction=str(args.pvalue_correction),
        )
    )
    output_paths.extend(
        plot_heatmap(
            cells,
            models=models,
            output_stem=str(args.output_stem),
            figure_width=float(args.figure_width),
            figure_height=float(args.figure_height),
            color_limit=args.color_limit,
        )
    )
    for output_path in output_paths:
        print(output_path.relative_to(Path.cwd()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
