"""Deterministic renderers for paper Figures 3, 5 and 6."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from memoreason.paper_results.paper_results_data_model import ANSWER_BEHAVIORS, MODEL_LABELS


QUESTION_LABELS = {
    "all": "All",
    "extractive": "Extractive",
    "arithmetic": "Arithmetic",
    "temporal": "Temporal",
    "inference": "Inference",
}
QUESTION_COLORS = {
    "all": "#e89b2d",
    "extractive": "#1f77b4",
    "arithmetic": "#2ca02c",
    "temporal": "#9467bd",
    "inference": "#d62728",
}
ANSWER_LABELS = {
    "all": "All",
    "variant": "Variant",
    "invariant": "Invariant",
    "refusal": "Refusal",
}
ANSWER_COLORS = {
    "all": "#e89b2d",
    "variant": "#1f77b4",
    "invariant": "#d97706",
    "refusal": "#e83e78",
}

# Figure 3 uses every cell of its 2x3 grid.  Keep its single shared legend in
# a dedicated band above the model titles instead of placing it on top of the
# middle title.
FIGURE_3_LEGEND_ANCHOR = (0.5, 1.07)
FIGURE_3_SIZE_INCHES = (10.5, 5.2)
FIGURE_3_REPLACEMENT_PERCENT_LABELS = (0, 10, 20, 30, 50, 80, 90, 100)
FIGURE_3_EQUALLY_SPACED_X_POSITIONS = tuple(range(len(FIGURE_3_REPLACEMENT_PERCENT_LABELS)))
FIGURE_3_TICK_LABEL_FONT_SIZE = 11
FIGURE_3_LEGEND_FONT_SIZE = 11
FIGURE_3_TITLE_Y_POSITION = 0.94
FIGURE_3_Y_AXIS_PADDING = 0.2
FIGURE_3_MODEL_LABELS = {
    "olmo-3-7b-instruct": "OLMo 3 7B Instruct",
    "gpt-oss-120b-groq": "GPT-OSS 120B",
    "gpt-oss-20b-groq": "GPT-OSS 20B",
    "gemma-4-26b-a4b-it": "Gemma 4 26B A4B IT",
    "qwen3.5-27b": "Qwen3.5 27B",
    "qwen3.5-35b-a3b": "Qwen3.5 35B A3B",
}
FIGURE_3_Y_TICKS_BY_MODEL = {
    "olmo-3-7b-instruct": (64.5, 66.0, 67.5, 69.0, 70.5, 72.0, 73.5),
    "gpt-oss-120b-groq": (85.5, 87.0, 88.5, 90.0, 91.5, 93.0),
    "gpt-oss-20b-groq": (82.5, 84.0, 85.5, 87.0, 88.5, 90.0, 91.5),
    "gemma-4-26b-a4b-it": (81.0, 82.5, 84.0, 85.5, 87.0, 88.5, 90.0),
    "qwen3.5-27b": (84.0, 85.5, 87.0, 88.5, 90.0, 91.5, 93.0),
    "qwen3.5-35b-a3b": (81.0, 82.5, 84.0, 85.5, 87.0, 88.5, 90.0, 91.5),
}


def _configure_axes(
    ax: Any,
    *,
    title: str,
    y_values: Sequence[float],
    title_y_position: float | None = None,
) -> None:
    title_options = {"fontweight": "bold", "fontsize": 13, "pad": 8}
    if title_y_position is not None:
        title_options.update({"y": title_y_position, "pad": 0})
    ax.set_title(title, **title_options)
    ax.set_xlim(-2, 102)
    if y_values:
        low = min(y_values)
        high = max(y_values)
        span = max(2.0, high - low)
        ax.set_ylim(max(0.0, low - 0.18 * span), min(100.0, high + 0.25 * span))
    ax.set_xticks((0, 10, 20, 30, 50, 80, 90, 100))
    ax.grid(True, color="#dedede", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: Any, output_stem: Path) -> tuple[Path, Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    symbolic_links = [path for path in (png_path, pdf_path) if path.is_symlink()]
    if symbolic_links:
        raise ValueError(f"Refusing to write figure artifacts through symbolic links: {symbolic_links}")
    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": "MemoReason paper-artifact renderer"},
    )
    # Matplotlib otherwise inserts the current wall-clock time into PDF
    # metadata, changing the artifact hash across identical runs.
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Creator": "MemoReason paper-artifact renderer",
            "Producer": "Matplotlib",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return png_path, pdf_path


def _rows_for(
    rows: Sequence[Mapping[str, Any]],
    *,
    model: str,
    group_key: str | None = None,
    group: str = "all",
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if str(row["model_name"]) == model]
    if group_key:
        selected = [row for row in selected if str(row[group_key]) == group]
    return sorted(selected, key=lambda row: float(row["replacement_proportion"]))


def build_figure3(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    output_stem: Path,
) -> tuple[Path, Path]:
    if not models or len(models) > 6:
        raise ValueError("Figure 3 requires between one and six models for its 2x3 panel grid.")
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_3_SIZE_INCHES, constrained_layout=True)
    flat_axes = list(axes.flat)
    for panel_index, (ax, model) in enumerate(zip(flat_axes, models, strict=False)):
        points = _rows_for(curve_rows, model=model)
        replacement_percentages = tuple(round(100.0 * float(row["replacement_proportion"])) for row in points)
        if replacement_percentages != FIGURE_3_REPLACEMENT_PERCENT_LABELS:
            raise ValueError(
                f"Figure 3 requires replacement percentages {FIGURE_3_REPLACEMENT_PERCENT_LABELS}, "
                f"got {replacement_percentages} for {model}."
            )
        xs = FIGURE_3_EQUALLY_SPACED_X_POSITIONS
        ys = [float(row["accuracy_pp"]) for row in points]
        lows = [float(row["ci95_wald_low_pp"]) for row in points]
        highs = [float(row["ci95_wald_high_pp"]) for row in points]
        factual = next(float(row["accuracy_pp"]) for row in points if float(row["replacement_proportion"]) == 0.0)
        ax.fill_between(xs, lows, highs, color="#f6d9b2", alpha=0.55, linewidth=0)
        ax.plot(
            xs,
            ys,
            color="#e89b2d",
            linewidth=2.5,
            marker="o",
            markersize=3.2,
            markerfacecolor="white",
        )
        ax.axhline(factual, color="#d62728", linestyle=(0, (4, 3)), linewidth=1.5)
        _configure_axes(
            ax,
            title=FIGURE_3_MODEL_LABELS.get(model, MODEL_LABELS.get(model, model)),
            y_values=(*lows, *highs),
            title_y_position=FIGURE_3_TITLE_Y_POSITION,
        )
        ax.set_xlim(-0.25, len(FIGURE_3_EQUALLY_SPACED_X_POSITIONS) - 0.75)
        ax.set_xticks(FIGURE_3_EQUALLY_SPACED_X_POSITIONS, labels=FIGURE_3_REPLACEMENT_PERCENT_LABELS)
        if model in FIGURE_3_Y_TICKS_BY_MODEL:
            y_ticks = FIGURE_3_Y_TICKS_BY_MODEL[model]
            ax.set_yticks(y_ticks)
            ax.set_ylim(
                y_ticks[0] - FIGURE_3_Y_AXIS_PADDING,
                y_ticks[-1] + FIGURE_3_Y_AXIS_PADDING,
            )
        ax.tick_params(axis="both", labelsize=FIGURE_3_TICK_LABEL_FONT_SIZE)
        if panel_index < 3:
            ax.tick_params(axis="x", labelbottom=False)
    for ax in flat_axes[len(models) :]:
        ax.axis("off")
    fig.supxlabel("Replaced Entities (%)", fontsize=15)
    fig.supylabel("Accuracy (%)", fontsize=15)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#d62728",
                linestyle=(0, (4, 3)),
                label="Factual baseline",
            )
        ],
        loc="upper center",
        bbox_to_anchor=FIGURE_3_LEGEND_ANCHOR,
        borderaxespad=0.0,
        frameon=False,
        fontsize=FIGURE_3_LEGEND_FONT_SIZE,
        handlelength=3.2,
    )
    return _save_figure(fig, output_stem)


def build_overlay_figure(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    group_key: str,
    group_order: Sequence[str],
    labels: Mapping[str, str],
    colors: Mapping[str, str],
    output_stem: Path,
) -> tuple[Path, Path]:
    if not models or len(models) > 6:
        raise ValueError("Overlay figures require between one and six models for their 2x3 panel grid.")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    flat_axes = list(axes.flat)
    for ax, model in zip(flat_axes, models, strict=False):
        panel_bounds: list[float] = []
        for group in group_order:
            points = _rows_for(curve_rows, model=model, group_key=group_key, group=group)
            xs = [100.0 * float(row["replacement_proportion"]) for row in points]
            ys = [float(row["accuracy_pp"]) for row in points]
            lows = [float(row["ci95_wald_low_pp"]) for row in points]
            highs = [float(row["ci95_wald_high_pp"]) for row in points]
            panel_bounds.extend((*lows, *highs))
            color = colors[group]
            ax.fill_between(xs, lows, highs, color=color, alpha=0.07, linewidth=0)
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.0 if group == "all" else 1.6,
                marker="o",
                markersize=2.6,
                markerfacecolor="white",
                label=labels[group],
            )
        _configure_axes(ax, title=MODEL_LABELS.get(model, model), y_values=panel_bounds)
    for ax in flat_axes[len(models) :]:
        ax.axis("off")
    fig.supxlabel("Replaced Entities (%)", fontsize=15)
    fig.supylabel("Accuracy (%)", fontsize=15)
    fig.legend(
        handles=[
            Line2D([0], [0], color=colors[group], marker="o", markersize=4, label=labels[group])
            for group in group_order
        ],
        loc="upper center",
        ncol=len(group_order),
        frameon=False,
    )
    return _save_figure(fig, output_stem)


def build_figure5(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    output_stem: Path,
) -> tuple[Path, Path]:
    return build_overlay_figure(
        curve_rows,
        models=models,
        group_key="question_type",
        group_order=("all", "extractive", "arithmetic", "temporal", "inference"),
        labels=QUESTION_LABELS,
        colors=QUESTION_COLORS,
        output_stem=output_stem,
    )


def build_figure6(
    curve_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    output_stem: Path,
) -> tuple[Path, Path]:
    return build_overlay_figure(
        curve_rows,
        models=models,
        group_key="answer_behavior",
        group_order=("all", *ANSWER_BEHAVIORS),
        labels=ANSWER_LABELS,
        colors=ANSWER_COLORS,
        output_stem=output_stem,
    )
