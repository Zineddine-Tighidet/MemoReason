#!/usr/bin/env python3
"""Plot model accuracy across partial fictional replacement proportions."""

from __future__ import annotations

import argparse
import csv
from math import sqrt
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.dataset_export.dataset_paths import MODEL_EVAL_PLOTS_DIR

DEFAULT_INPUT_PATH = (
    MODEL_EVAL_PLOTS_DIR
    / "partial_fictional_replacements"
    / "partial_replacement_drop_by_model.csv"
)
DEFAULT_QUESTION_TYPE_INPUT_PATH = (
    MODEL_EVAL_PLOTS_DIR
    / "partial_fictional_replacements"
    / "partial_replacement_drop_by_model_question_type.csv"
)
DEFAULT_ANSWER_TYPE_INPUT_PATH = (
    MODEL_EVAL_PLOTS_DIR
    / "partial_fictional_replacements"
    / "partial_replacement_drop_by_model_answer_type.csv"
)
DEFAULT_OUTPUT_DIR = MODEL_EVAL_PLOTS_DIR / "partial_fictional_replacements"
DEFAULT_OUTPUT_STEM = "partial_replacement_accuracy_curves_by_model_zoomed"
DEFAULT_QUESTION_TYPE_OUTPUT_STEM = "partial_replacement_accuracy_curves_by_question_type_and_model_zoomed"
DEFAULT_QUESTION_TYPE_OVERLAY_OUTPUT_STEM = (
    "partial_replacement_accuracy_curves_by_model_question_type_overlay_zoomed"
)
DEFAULT_ANSWER_TYPE_OVERLAY_OUTPUT_STEM = (
    "partial_replacement_accuracy_curves_by_model_answer_type_overlay_zoomed"
)

MODEL_LABELS = {
    "olmo-3-7b-instruct": "OLMo 3 7B Instruct",
    "gpt-oss-120b-groq": "GPT-OSS 120B",
    "gpt-oss-20b-groq": "GPT-OSS 20B",
    "gemma-4-26b-a4b-it": "Gemma 4 26B A4B IT",
    "qwen3.5-27b": "Qwen3.5 27B",
    "qwen3.5-35b-a3b": "Qwen3.5 35B A3B",
}

MODEL_ORDER = tuple(MODEL_LABELS)
MODEL_TITLE_LABELS = {
    "olmo-3-7b-instruct": "OLMo 3 7B\nInstruct",
    "gpt-oss-120b-groq": "GPT-OSS\n120B",
    "gpt-oss-20b-groq": "GPT-OSS\n20B",
    "gemma-4-26b-a4b-it": "Gemma 4 26B\nA4B IT",
    "qwen3.5-27b": "Qwen3.5\n27B",
    "qwen3.5-35b-a3b": "Qwen3.5 35B\nA3B",
}
QUESTION_TYPE_LABELS = {
    "all": "All",
    "extractive": "Extractive",
    "arithmetic": "Arithmetic",
    "temporal": "Temporal",
    "inference": "Inference",
}
QUESTION_TYPE_ORDER = tuple(QUESTION_TYPE_LABELS)
QUESTION_TYPE_COLORS = {
    "all": "#e49a35",
    "extractive": "#1f77b4",
    "arithmetic": "#2ca02c",
    "temporal": "#9467bd",
    "inference": "#d62728",
}
ANSWER_TYPE_LABELS = {
    "all": "All",
    "variant": "Variant",
    "invariant": "Invariant",
    "refusal": "Refusal",
}
ANSWER_TYPE_ORDER = tuple(ANSWER_TYPE_LABELS)
ANSWER_TYPE_COLORS = {
    "all": "#e49a35",
    "variant": "#1f77b4",
    "invariant": "#d97706",
    "refusal": "#d9467a",
}
LINE_COLOR = "#e49a35"
ERROR_BAND_COLOR = "#f5d4aa"
FACTUAL_COLOR = "#c62828"
WALD_Z_95 = 1.96


@dataclass(frozen=True)
class CurvePoint:
    proportion: float
    accuracy_pp: float
    accuracy_error_low_pp: float
    accuracy_error_high_pp: float


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series_by_model(
    rows: list[dict[str, str]],
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, list[CurvePoint]]:
    partial_points: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    factual_accuracy: dict[str, float] = {}
    factual_count_total: dict[str, int] = {}
    factual_count_correct: dict[str, int] = {}

    for row in rows:
        model = str(row["model_name"])
        proportion = float(row["replacement_proportion"])
        accuracy_pp = float(row["accuracy_pp"])
        count_total = int(row["count_total"])
        count_correct = int(row["count_correct"])
        partial_points[model].append((proportion, accuracy_pp, count_total, count_correct))
        factual_accuracy.setdefault(model, float(row["factual_accuracy_pp"]))
        factual_count_total.setdefault(model, int(row["factual_count_total"]))
        factual_count_correct.setdefault(
            model,
            round(float(row["factual_accuracy"]) * int(row["factual_count_total"])),
        )

    series: dict[str, list[CurvePoint]] = {}
    for model_index, (model, points) in enumerate(sorted(partial_points.items())):
        if model not in factual_accuracy:
            raise ValueError(f"Missing factual accuracy for model {model!r}.")
        if model not in factual_count_total:
            raise ValueError(f"Missing factual count for model {model!r}.")
        if model not in factual_count_correct:
            raise ValueError(f"Missing factual correct count for model {model!r}.")
        deduped = {
            0.0: (
                factual_accuracy[model],
                factual_count_total[model],
                factual_count_correct[model],
            )
        }
        for proportion, accuracy_pp, count_total, count_correct in points:
            deduped[proportion] = (accuracy_pp, count_total, count_correct)
        model_series: list[CurvePoint] = []
        for point_index, (proportion, (accuracy_pp, count_total, count_correct)) in enumerate(
            sorted(deduped.items(), key=lambda item: item[0])
        ):
            error_low_pp, error_high_pp = _accuracy_error_interval_pp(
                accuracy_pp,
                count_total,
                count_correct,
                method=error_method,
                bootstrap_resamples=bootstrap_resamples,
                bootstrap_seed=bootstrap_seed + model_index * 1000 + point_index,
            )
            model_series.append(
                CurvePoint(
                    proportion=proportion,
                    accuracy_pp=accuracy_pp,
                    accuracy_error_low_pp=error_low_pp,
                    accuracy_error_high_pp=error_high_pp,
                )
            )
        series[model] = model_series
    return series


def _accuracy_error_interval_pp(
    accuracy_pp: float,
    count_total: int,
    count_correct: int,
    *,
    method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[float, float]:
    if method == "wald":
        margin = _wald_95_margin_pp(accuracy_pp, count_total)
        return margin, margin
    if method == "wilson":
        return _wilson_95_margins_pp(
            count_correct=count_correct,
            count_total=count_total,
        )
    if method == "bootstrap":
        return _bootstrap_95_margins_pp(
            count_correct=count_correct,
            count_total=count_total,
            n_resamples=bootstrap_resamples,
            seed=bootstrap_seed,
        )
    raise ValueError(f"Unsupported error method: {method!r}")


def _wald_95_margin_pp(accuracy_pp: float, count_total: int) -> float:
    if count_total <= 0:
        return 0.0
    accuracy = min(max(accuracy_pp / 100.0, 0.0), 1.0)
    standard_error = sqrt(max(accuracy * (1.0 - accuracy), 0.0) / count_total)
    return 100.0 * WALD_Z_95 * standard_error


def _wilson_95_margins_pp(*, count_correct: int, count_total: int) -> tuple[float, float]:
    if count_total <= 0:
        return 0.0, 0.0
    accuracy = min(max(count_correct / count_total, 0.0), 1.0)
    z_squared = WALD_Z_95**2
    denominator = 1.0 + z_squared / count_total
    center = (accuracy + z_squared / (2.0 * count_total)) / denominator
    half_width = (
        WALD_Z_95
        * sqrt((accuracy * (1.0 - accuracy) + z_squared / (4.0 * count_total)) / count_total)
        / denominator
    )
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return max(0.0, 100.0 * (accuracy - low)), max(0.0, 100.0 * (high - accuracy))


def _bootstrap_95_margins_pp(
    *,
    count_correct: int,
    count_total: int,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    if count_total <= 0 or n_resamples <= 0:
        return 0.0, 0.0
    import numpy as np

    accuracy = min(max(count_correct / count_total, 0.0), 1.0)
    rng = np.random.default_rng(seed)
    bootstrap_accuracy_pp = 100.0 * rng.binomial(
        n=count_total,
        p=accuracy,
        size=n_resamples,
    ) / count_total
    low, high = np.percentile(bootstrap_accuracy_pp, [2.5, 97.5])
    center = 100.0 * accuracy
    return max(0.0, center - float(low)), max(0.0, float(high) - center)


def _format_x_tick_label(proportion: float) -> str:
    percentage = int(round(proportion * 100))
    return str(percentage)


def _draw_x_tick_labels(
    ax: Any,
    x_ticks: list[float],
    x_tick_labels: list[str],
) -> None:
    ax.set_xticklabels([])
    if not ax.get_subplotspec().is_last_row():
        return
    for tick, label in zip(
        x_ticks,
        x_tick_labels,
        strict=True,
    ):
        ax.text(
            tick,
            -0.08,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=96,
            color="#222222",
            clip_on=False,
        )


def _draw_curve_panel(
    ax: Any,
    points: list[CurvePoint],
    *,
    x_position_by_proportion: dict[float, int],
    x_tick_positions: list[int],
    x_tick_labels: list[str],
    show_x_tick_labels: bool,
    title: str | None = None,
    title_fontsize: float = 28.0,
    tick_fontsize: float = 24.0,
    line_width: float = 5.0,
    factual_line_width: float = 3.5,
    marker_size: float = 5.0,
) -> None:
    from matplotlib.ticker import MaxNLocator

    xs = [x_position_by_proportion[round(point.proportion, 1)] for point in points]
    ys = [point.accuracy_pp for point in points]
    y_error_lows = [point.accuracy_error_low_pp for point in points]
    y_error_highs = [point.accuracy_error_high_pp for point in points]
    y_lows = [
        max(0.0, accuracy - error)
        for accuracy, error in zip(ys, y_error_lows, strict=True)
    ]
    y_highs = [
        min(100.0, accuracy + error)
        for accuracy, error in zip(ys, y_error_highs, strict=True)
    ]
    factual_y = {point.proportion: point.accuracy_pp for point in points}[0.0]
    y_span = max(y_highs) - min(y_lows)
    bottom_padding = max(0.15, y_span * 0.08)
    top_padding = max(0.35, y_span * 0.16)
    y_min = max(0.0, min(y_lows) - bottom_padding)
    y_max = min(100.0, max(y_highs) + top_padding)

    ax.fill_between(
        xs,
        y_lows,
        y_highs,
        color=ERROR_BAND_COLOR,
        alpha=0.36,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        xs,
        ys,
        color=LINE_COLOR,
        linewidth=line_width,
        marker="o",
        markersize=marker_size,
        markerfacecolor="white",
        markeredgewidth=max(1.2, line_width * 0.23),
        zorder=3,
    )
    ax.axhline(
        factual_y,
        color=FACTUAL_COLOR,
        linewidth=factual_line_width,
        linestyle=(0, (4, 3)),
        alpha=0.9,
        zorder=2,
    )
    ax.set_xlim(-0.25, len(x_tick_positions) - 0.75)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels if show_x_tick_labels else [])
    ax.tick_params(axis="x", labelsize=tick_fontsize, colors="#000000", width=1.8, length=6)
    ax.tick_params(axis="y", labelsize=tick_fontsize, colors="#000000", width=1.8, length=6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
    ax.grid(axis="x", color="#ededed", linewidth=0.6)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=12)


def build_figure(
    input_path: Path,
    output_dir: Path,
    output_stem: str,
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    rows = _read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}.")

    series = _series_by_model(
        rows,
        error_method=error_method,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    models = [model for model in MODEL_ORDER if model in series]
    models.extend(sorted(model for model in series if model not in MODEL_ORDER))
    if not models:
        raise ValueError("No model series to plot.")

    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    x_tick_positions = list(range(len(x_ticks)))
    x_position_by_proportion = {
        round(proportion, 1): position
        for position, proportion in zip(x_tick_positions, x_ticks, strict=True)
    }
    x_tick_labels = [_format_x_tick_label(tick) for tick in x_ticks]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titleweight": "bold",
            "axes.titlesize": 100,
            "font.size": 12,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(84, 42),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, model in zip(flat_axes, models, strict=False):
        points = series[model]
        xs = [x_position_by_proportion[round(point.proportion, 1)] for point in points]
        ys = [point.accuracy_pp for point in points]
        y_error_lows = [point.accuracy_error_low_pp for point in points]
        y_error_highs = [point.accuracy_error_high_pp for point in points]
        y_lows = [
            max(0.0, accuracy - error)
            for accuracy, error in zip(ys, y_error_lows, strict=True)
        ]
        y_highs = [
            min(100.0, accuracy + error)
            for accuracy, error in zip(ys, y_error_highs, strict=True)
        ]
        factual_y = {point.proportion: point.accuracy_pp for point in points}[0.0]
        y_span = max(y_highs) - min(y_lows)
        bottom_padding = max(0.2, y_span * 0.08)
        top_padding = max(0.45, y_span * 0.18)
        y_min = max(0.0, min(y_lows) - bottom_padding)
        y_max = min(100.0, max(y_highs) + top_padding)

        ax.fill_between(
            xs,
            y_lows,
            y_highs,
            color=ERROR_BAND_COLOR,
            alpha=0.36,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            xs,
            ys,
            color=LINE_COLOR,
            linewidth=22.0,
            marker="o",
            markersize=18.0,
            markerfacecolor="white",
            markeredgewidth=7.0,
            zorder=3,
        )
        ax.axhline(
            factual_y,
            color=FACTUAL_COLOR,
            linewidth=12.0,
            linestyle=(0, (4, 3)),
            alpha=0.9,
            zorder=2,
        )
        ax.scatter([0.0], [factual_y], color=FACTUAL_COLOR, s=110, zorder=4, label="Factual")
        ax.text(
            0.5,
            0.98,
            MODEL_LABELS.get(model, model),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=100,
            fontweight="bold",
            color="#000000",
            zorder=5,
        )
        ax.set_xlim(-0.25, len(x_ticks) - 0.75)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(x_tick_positions)
        _draw_x_tick_labels(ax, x_tick_positions, x_tick_labels)
        ax.tick_params(axis="x", pad=40, labelsize=96, colors="#000000", width=3.5, length=10)
        ax.tick_params(axis="y", labelsize=96, colors="#000000", width=3.5, length=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
        ax.grid(axis="x", color="#ededed", linewidth=0.6)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.spines["left"].set_linewidth(3.5)
        ax.spines["bottom"].set_linewidth(3.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(models) :]:
        ax.axis("off")

    axis_label_size = 120
    fig.supxlabel(
        "Replaced Entities (%)",
        fontsize=axis_label_size,
    )
    fig.supylabel("Accuracy (%)", fontsize=axis_label_size)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=FACTUAL_COLOR,
                linewidth=12.0,
                linestyle=(0, (4, 3)),
                label="Factual baseline",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=88,
        handlelength=4.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
        output_dir / f"{output_stem}.svg",
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(fig)
    return output_paths


def build_question_type_figure(
    input_path: Path,
    output_dir: Path,
    output_stem: str,
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = _read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}.")

    series_by_question_type: dict[str, dict[str, list[CurvePoint]]] = {}
    for question_type in QUESTION_TYPE_ORDER:
        question_rows = [row for row in rows if str(row.get("question_type") or "") == question_type]
        if not question_rows:
            continue
        series_by_question_type[question_type] = _series_by_model(
            question_rows,
            error_method=error_method,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + 100_000 * len(series_by_question_type),
        )

    question_types = [question_type for question_type in QUESTION_TYPE_ORDER if question_type in series_by_question_type]
    models = [
        model
        for model in MODEL_ORDER
        if any(model in model_series for model_series in series_by_question_type.values())
    ]
    models.extend(
        sorted(
            {
                model
                for model_series in series_by_question_type.values()
                for model in model_series
            }
            - set(models)
        )
    )
    if not question_types or not models:
        raise ValueError("No question-type/model series to plot.")

    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    x_tick_positions = list(range(len(x_ticks)))
    x_position_by_proportion = {
        round(proportion, 1): position
        for position, proportion in zip(x_tick_positions, x_ticks, strict=True)
    }
    x_tick_labels = [_format_x_tick_label(tick) for tick in x_ticks]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "font.size": 12,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )

    fig = plt.figure(figsize=(38, 27))
    grid = fig.add_gridspec(
        len(question_types),
        len(models) + 1,
        left=0.015,
        right=0.995,
        bottom=0.08,
        top=0.90,
        width_ratios=(0.42, *([1.0] * len(models))),
        wspace=0.18,
        hspace=0.38,
    )

    for row_index, question_type in enumerate(question_types):
        label_axis = fig.add_subplot(grid[row_index, 0])
        label_axis.axis("off")
        label_axis.text(
            0.98,
            0.5,
            QUESTION_TYPE_LABELS.get(question_type, question_type.title()),
            transform=label_axis.transAxes,
            ha="right",
            va="center",
            fontsize=31,
            fontweight="bold",
            rotation=90,
        )
        for col_index, model in enumerate(models):
            ax = fig.add_subplot(grid[row_index, col_index + 1])
            points = series_by_question_type.get(question_type, {}).get(model)
            if not points:
                ax.axis("off")
                continue
            _draw_curve_panel(
                ax,
                points,
                x_position_by_proportion=x_position_by_proportion,
                x_tick_positions=x_tick_positions,
                x_tick_labels=x_tick_labels,
                show_x_tick_labels=row_index == len(question_types) - 1,
                title=MODEL_TITLE_LABELS.get(model, MODEL_LABELS.get(model, model)) if row_index == 0 else None,
                title_fontsize=29,
                tick_fontsize=22,
                line_width=5.8,
                factual_line_width=3.8,
                marker_size=5.2,
            )

    fig.supxlabel("Replaced Entities (%)", fontsize=36)
    fig.supylabel("Accuracy (%)", fontsize=36)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=FACTUAL_COLOR,
                linewidth=3.8,
                linestyle=(0, (4, 3)),
                label="Factual baseline",
            )
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.935),
        frameon=False,
        fontsize=29,
        handlelength=3.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
        output_dir / f"{output_stem}.svg",
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(fig)
    return output_paths


def _question_type_series(
    rows: list[dict[str, str]],
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, list[CurvePoint]]]:
    series_by_question_type: dict[str, dict[str, list[CurvePoint]]] = {}
    for question_type in QUESTION_TYPE_ORDER:
        question_rows = [row for row in rows if str(row.get("question_type") or "") == question_type]
        if not question_rows:
            continue
        series_by_question_type[question_type] = _series_by_model(
            question_rows,
            error_method=error_method,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + 100_000 * len(series_by_question_type),
        )
    return series_by_question_type


def build_question_type_overlay_figure(
    input_path: Path,
    output_dir: Path,
    output_stem: str,
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    rows = _read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}.")

    series_by_question_type = _question_type_series(
        rows,
        error_method=error_method,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    models = [
        model
        for model in MODEL_ORDER
        if any(model in model_series for model_series in series_by_question_type.values())
    ]
    models.extend(
        sorted(
            {
                model
                for model_series in series_by_question_type.values()
                for model in model_series
            }
            - set(models)
        )
    )
    if not models:
        raise ValueError("No model series to plot.")

    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    x_tick_positions = list(range(len(x_ticks)))
    x_position_by_proportion = {
        round(proportion, 1): position
        for position, proportion in zip(x_tick_positions, x_ticks, strict=True)
    }
    x_tick_labels = [_format_x_tick_label(tick) for tick in x_ticks]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titleweight": "bold",
            "axes.titlesize": 100,
            "font.size": 12,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(84, 42),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, model in zip(flat_axes, models, strict=False):
        panel_y_lows: list[float] = []
        panel_y_highs: list[float] = []
        for question_type in QUESTION_TYPE_ORDER:
            points = series_by_question_type.get(question_type, {}).get(model, [])
            if not points:
                continue
            xs = [x_position_by_proportion[round(point.proportion, 1)] for point in points]
            ys = [point.accuracy_pp for point in points]
            y_lows = [
                max(0.0, point.accuracy_pp - point.accuracy_error_low_pp)
                for point in points
            ]
            y_highs = [
                min(100.0, point.accuracy_pp + point.accuracy_error_high_pp)
                for point in points
            ]
            panel_y_lows.extend(y_lows)
            panel_y_highs.extend(y_highs)
            color = QUESTION_TYPE_COLORS.get(question_type, "#64748b")
            ax.fill_between(
                xs,
                y_lows,
                y_highs,
                color=color,
                alpha=0.075 if question_type != "all" else 0.13,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=16.0 if question_type == "all" else 12.0,
                marker="o",
                markersize=12.0 if question_type == "all" else 10.0,
                markerfacecolor="white",
                markeredgewidth=4.5,
                zorder=4 if question_type == "all" else 3,
            )

        if panel_y_lows and panel_y_highs:
            y_span = max(panel_y_highs) - min(panel_y_lows)
            bottom_padding = max(0.35, y_span * 0.08)
            top_padding = max(0.65, y_span * 0.14)
            ax.set_ylim(
                max(0.0, min(panel_y_lows) - bottom_padding),
                min(100.0, max(panel_y_highs) + top_padding),
            )

        ax.text(
            0.5,
            0.98,
            MODEL_LABELS.get(model, model),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=100,
            fontweight="bold",
            color="#000000",
            zorder=5,
        )
        ax.set_xlim(-0.25, len(x_ticks) - 0.75)
        ax.set_xticks(x_tick_positions)
        _draw_x_tick_labels(ax, x_tick_positions, x_tick_labels)
        ax.tick_params(axis="x", pad=40, labelsize=96, colors="#000000", width=3.5, length=10)
        ax.tick_params(axis="y", labelsize=96, colors="#000000", width=3.5, length=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
        ax.grid(axis="x", color="#ededed", linewidth=0.6)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.spines["left"].set_linewidth(3.5)
        ax.spines["bottom"].set_linewidth(3.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(models) :]:
        ax.axis("off")

    axis_label_size = 120
    fig.supxlabel("Replaced Entities (%)", fontsize=axis_label_size)
    fig.supylabel("Accuracy (%)", fontsize=axis_label_size)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=QUESTION_TYPE_COLORS[question_type],
            linewidth=12.0 if question_type == "all" else 9.0,
            marker="o",
            markersize=15,
            markerfacecolor="white",
            markeredgewidth=4.0,
            label=QUESTION_TYPE_LABELS[question_type],
        )
        for question_type in QUESTION_TYPE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=74,
        handlelength=2.8,
        ncol=6,
        columnspacing=1.0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
        output_dir / f"{output_stem}.svg",
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(fig)
    return output_paths


def _answer_type_series(
    rows: list[dict[str, str]],
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, list[CurvePoint]]]:
    series_by_answer_type: dict[str, dict[str, list[CurvePoint]]] = {}
    for answer_type in ANSWER_TYPE_ORDER:
        answer_rows = [row for row in rows if str(row.get("answer_type") or "") == answer_type]
        if not answer_rows:
            continue
        series_by_answer_type[answer_type] = _series_by_model(
            answer_rows,
            error_method=error_method,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + 100_000 * len(series_by_answer_type),
        )
    return series_by_answer_type


def build_answer_type_overlay_figure(
    input_path: Path,
    output_dir: Path,
    output_stem: str,
    *,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    rows = _read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}.")

    series_by_answer_type = _answer_type_series(
        rows,
        error_method=error_method,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    models = [
        model
        for model in MODEL_ORDER
        if any(model in model_series for model_series in series_by_answer_type.values())
    ]
    models.extend(
        sorted(
            {
                model
                for model_series in series_by_answer_type.values()
                for model in model_series
            }
            - set(models)
        )
    )
    if not models:
        raise ValueError("No model series to plot.")

    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    x_tick_positions = list(range(len(x_ticks)))
    x_position_by_proportion = {
        round(proportion, 1): position
        for position, proportion in zip(x_tick_positions, x_ticks, strict=True)
    }
    x_tick_labels = [_format_x_tick_label(tick) for tick in x_ticks]

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titleweight": "bold",
            "axes.titlesize": 100,
            "font.size": 12,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(84, 42),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, model in zip(flat_axes, models, strict=False):
        panel_y_lows: list[float] = []
        panel_y_highs: list[float] = []

        for answer_type in ANSWER_TYPE_ORDER:
            points = series_by_answer_type.get(answer_type, {}).get(model, [])
            if not points:
                continue
            xs = [x_position_by_proportion[round(point.proportion, 1)] for point in points]
            ys = [point.accuracy_pp for point in points]
            y_lows = [
                max(0.0, point.accuracy_pp - point.accuracy_error_low_pp)
                for point in points
            ]
            y_highs = [
                min(100.0, point.accuracy_pp + point.accuracy_error_high_pp)
                for point in points
            ]
            panel_y_lows.extend(y_lows)
            panel_y_highs.extend(y_highs)
            color = ANSWER_TYPE_COLORS.get(answer_type, "#64748b")
            ax.fill_between(
                xs,
                y_lows,
                y_highs,
                color=color,
                alpha=0.075 if answer_type != "all" else 0.13,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=16.0 if answer_type == "all" else 12.0,
                marker="o",
                markersize=12.0 if answer_type == "all" else 10.0,
                markerfacecolor="white",
                markeredgewidth=4.5,
                zorder=4 if answer_type == "all" else 3,
            )

        if panel_y_lows and panel_y_highs:
            y_span = max(panel_y_highs) - min(panel_y_lows)
            bottom_padding = max(0.35, y_span * 0.08)
            top_padding = max(0.65, y_span * 0.14)
            ax.set_ylim(
                max(0.0, min(panel_y_lows) - bottom_padding),
                min(100.0, max(panel_y_highs) + top_padding),
            )

        ax.text(
            0.5,
            0.98,
            MODEL_LABELS.get(model, model),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=100,
            fontweight="bold",
            color="#000000",
            zorder=5,
        )
        ax.set_xlim(-0.25, len(x_ticks) - 0.75)
        ax.set_xticks(x_tick_positions)
        _draw_x_tick_labels(ax, x_tick_positions, x_tick_labels)
        ax.tick_params(axis="x", pad=40, labelsize=96, colors="#000000", width=3.5, length=10)
        ax.tick_params(axis="y", labelsize=96, colors="#000000", width=3.5, length=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
        ax.grid(axis="x", color="#ededed", linewidth=0.6)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.spines["left"].set_linewidth(3.5)
        ax.spines["bottom"].set_linewidth(3.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(models) :]:
        ax.axis("off")

    axis_label_size = 120
    fig.supxlabel("Replaced Entities (%)", fontsize=axis_label_size)
    fig.supylabel("Accuracy (%)", fontsize=axis_label_size)
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=ANSWER_TYPE_COLORS[answer_type],
            linewidth=12.0 if answer_type == "all" else 9.0,
            marker="o",
            markersize=15,
            markerfacecolor="white",
            markeredgewidth=4.0,
            label=ANSWER_TYPE_LABELS[answer_type],
        )
        for answer_type in ANSWER_TYPE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=80,
        handlelength=2.8,
        ncol=4,
        columnspacing=1.15,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
        output_dir / f"{output_stem}.svg",
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(fig)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--question-type-input-path", type=Path, default=DEFAULT_QUESTION_TYPE_INPUT_PATH)
    parser.add_argument("--answer-type-input-path", type=Path, default=DEFAULT_ANSWER_TYPE_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--question-type-output-stem", default=DEFAULT_QUESTION_TYPE_OUTPUT_STEM)
    parser.add_argument(
        "--question-type-overlay-output-stem",
        default=DEFAULT_QUESTION_TYPE_OVERLAY_OUTPUT_STEM,
    )
    parser.add_argument(
        "--answer-type-overlay-output-stem",
        default=DEFAULT_ANSWER_TYPE_OVERLAY_OUTPUT_STEM,
    )
    parser.add_argument(
        "--skip-question-type-figure",
        action="store_true",
        help="Only render the global model-grid figure.",
    )
    parser.add_argument(
        "--skip-question-type-overlay-figure",
        action="store_true",
        help="Do not render the 2x3 model-grid question-type overlay figure.",
    )
    parser.add_argument(
        "--skip-answer-type-overlay-figure",
        action="store_true",
        help="Do not render the 2x3 model-grid answer-type overlay figure.",
    )
    parser.add_argument(
        "--error-method",
        choices=("wald", "wilson", "bootstrap"),
        default="wald",
        help="Uncertainty band method for accuracy.",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=10000,
        help="Number of resamples when --error-method=bootstrap.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=23,
        help="Random seed used when --error-method=bootstrap.",
    )
    args = parser.parse_args()

    output_paths = build_figure(
        args.input_path,
        args.output_dir,
        args.output_stem,
        error_method=args.error_method,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    if not args.skip_question_type_figure:
        output_paths.extend(
            build_question_type_figure(
                args.question_type_input_path,
                args.output_dir,
                args.question_type_output_stem,
                error_method=args.error_method,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    if not args.skip_question_type_overlay_figure:
        output_paths.extend(
            build_question_type_overlay_figure(
                args.question_type_input_path,
                args.output_dir,
                args.question_type_overlay_output_stem,
                error_method=args.error_method,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    if not args.skip_answer_type_overlay_figure:
        output_paths.extend(
            build_answer_type_overlay_figure(
                args.answer_type_input_path,
                args.output_dir,
                args.answer_type_overlay_output_stem,
                error_method=args.error_method,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
            )
        )
    for output_path in output_paths:
        print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
