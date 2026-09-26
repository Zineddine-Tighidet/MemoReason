"""Render partial-replacement figures from caller-supplied computed CSV tables.

The input directory must contain partial/<model>.csv for the seven models below,
with replaced_percent, accuracy_percent, and count_total columns.
"""
from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PARTIAL_MODELS = (
    ("olmo-3-7b-instruct", "OLMo 3 7B Instruct"),
    ("gpt-oss-120b-groq", "GPT-OSS 120B"),
    ("gpt-oss-20b-groq", "GPT-OSS 20B"),
    ("gemma-4-26b-a4b-it", "Gemma 4 26B A4B IT"),
    ("qwen3.5-27b", "Qwen3.5 27B"),
    ("qwen3.5-35b-a3b", "Qwen3.5 35B A3B"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def partial_curves(output: Path) -> dict:
    proportions = (0, 10, 20, 30, 50, 80, 90, 100)
    figure, axes = plt.subplots(2, 4, figsize=(12.2, 6.3))
    sources = {}
    for axis, (model, label) in zip(axes.flat, PARTIAL_MODELS, strict=False):
        table = output / "partial" / f"{model}.csv"
        rows = read_csv(table)
        if tuple(int(row["replaced_percent"]) for row in rows) != proportions:
            raise ValueError(f"Incomplete plotting scope: {model}")
        accuracy = np.array([float(row["accuracy_percent"]) for row in rows])
        counts = np.array([int(row["count_total"]) for row in rows])
        probability = accuracy / 100
        ci = 100 * 1.96 * np.sqrt(probability * (1 - probability) / counts)
        x = np.arange(len(proportions))
        axis.axhline(accuracy[0], color="#e53b3b", linestyle=(0, (4, 3)), linewidth=1.5,
                     label="Factual baseline")
        axis.fill_between(x, accuracy - ci, accuracy + ci, color="#ee9820", alpha=0.18, linewidth=0)
        axis.plot(x, accuracy, color="#e8951e", marker="o", markersize=4, markerfacecolor="white", linewidth=2)
        axis.set_xticks(x, proportions, fontsize=8)
        axis.set_title(label, fontsize=11, fontweight="bold")
        axis.grid(color="#e5e7eb", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        sources[model] = f"partial/{model}.csv"
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", frameon=False, fontsize=11)
    figure.supxlabel("Replaced Entities (%)", fontsize=13)
    figure.supylabel("Accuracy (%)", fontsize=13)
    figure.tight_layout(rect=(0.01, 0.02, 1, 0.94), h_pad=1.5)
    save(figure, output, "partial_accuracy_curves")
    return {"sources": sources, "intervals": "95% binomial normal-approximation intervals: 1.96 sqrt(p(1-p)/n)",
            "x_axis": "equally spaced intervention levels, labels give the actual percentages"}


def representative_curve(output: Path) -> dict:
    """Render the representative curve from the supplied eight-point table."""
    model = "qwen3.5-27b"
    rows = read_csv(output / "partial" / f"{model}.csv")
    percentages = (0, 10, 20, 30, 50, 80, 90, 100)
    if tuple(int(row["replaced_percent"]) for row in rows) != percentages:
        raise ValueError("Representative curve must contain all eight evaluated levels")
    values = np.array([float(row["accuracy_percent"]) for row in rows])
    counts = np.array([int(row["count_total"]) for row in rows])
    half = 100 * 1.96 * np.sqrt((values / 100) * (1 - values / 100) / counts)
    figure, axis = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    from matplotlib.colors import to_rgb
    green, red, blue = map(np.array, map(to_rgb, ("#73b77c", "#e45756", "#75b8e7")))
    edges = np.linspace(-0.25, 7.25, 401)
    for left, right in pairwise(edges):
        center = (left + right) / 2
        if center < 0.05:
            color = green
        elif center < 0.8:
            weight = (center - 0.05) / 0.75
            color = green * (1 - weight) + red * weight
        elif center <= 6.2:
            color = red
        elif center < 6.95:
            weight = (center - 6.2) / 0.75
            color = red * (1 - weight) + blue * weight
        else:
            color = blue
        axis.axvspan(left, right, color=color, alpha=0.16, linewidth=0, zorder=0)
    x = np.arange(8)
    axis.fill_between(x, values - half, values + half, color="#ee9820", alpha=0.20, linewidth=0)
    axis.plot(x, values, color="#e8951e", linewidth=2.4, marker="o", markersize=4,
              markerfacecolor="white", zorder=3)
    axis.axhline(values[0], color="#e53b3b", linestyle=(0, (4, 3)), linewidth=1.5,
                 label="Factual baseline")
    for location, label, color, alignment in ((-0.15, "FACTUAL", "#2f7f45", "left"),
                                    (3.55, "COUNTERFACTUAL", "#9e3434", "center"),
                                    (7.15, "FICTITIOUS", "#2d6f9f", "right")):
        axis.text(location, values[0] + 0.18, label, ha=alignment, va="bottom",
                  fontsize=7.2, fontweight="bold", color=color,
                  bbox={"boxstyle": "round,pad=0.18", "facecolor": "white",
                        "edgecolor": "none", "alpha": 0.8})
    axis.set_xlim(-0.25, 7.25)
    axis.set_xticks(x, percentages)
    axis.set_xlabel("Replaced Entities (%)", fontsize=11)
    axis.set_ylabel("Accuracy (%)", fontsize=11)
    axis.grid(color="#dedede", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    legend = axis.legend(loc="upper right", frameon=True, facecolor="white",
                         edgecolor="black", framealpha=0.92, fontsize=8)
    legend.get_frame().set_linewidth(0.5)
    save(figure, output, "partial_accuracy_qwen27")
    return {"model": model, "source": "partial/qwen3.5-27b.csv",
            "regimes": "narrow factual/fictitious endpoint zones; counterfactual interior"}




def render_all(output: Path) -> dict:
    """Render figures beside supplied tables and return their input descriptions."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    return {"partial_curves": partial_curves(output),
            "representative_curve": representative_curve(output)}
