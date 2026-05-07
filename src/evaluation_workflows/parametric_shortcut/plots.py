"""Plots for model-evaluation metrics with proportion-aware benchmark settings."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from collections import defaultdict
from functools import lru_cache
from math import ceil, sqrt
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch
from scipy.stats import ttest_rel

from src.dataset_export.dataset_settings import DatasetSettingSpec, order_dataset_settings
from src.dataset_export.dataset_paths import MODEL_EVAL_PLOTS_DIR, MODEL_EVAL_RAW_OUTPUTS_DIR, PROJECT_ROOT, sanitize_model_name
from .dataset import is_excluded_evaluation_document, iter_evaluation_documents


_CONVERSION_COLORS = {
    "correct_to_correct": "#ffd54f",
    "correct_to_incorrect": "#ff8c00",
    "incorrect_to_correct": "#f5c26b",
    "incorrect_to_incorrect": "#d95f02",
}
_CONVERSION_CATEGORIES = tuple(_CONVERSION_COLORS.keys())
_PREFERRED_QUESTION_TYPES = ("extractive", "arithmetic", "temporal", "inference")
_QUESTION_TYPE_COLORS = {
    "extractive": "#0b72d0",
    "arithmetic": "#d97706",
    "temporal": "#0f9d8f",
    "inference": "#d9467a",
}
_ANSWER_BEHAVIOR_COLORS = {
    "variant": "#0b72d0",
    "invariant": "#d97706",
    "refusal": "#d9467a",
}
_ANSWER_BEHAVIOR_LEGEND_LABELS = {
    "variant": "Variant",
    "invariant": "Invariant",
    "refusal": "Refusal",
}
_EXPECTED_VARIANT_IDS = tuple(f"v{index:02d}" for index in range(1, 11))
_ANSWER_BEHAVIOR_DISTRIBUTION_BIN_COUNT = 20


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.6,
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        }
    )


def _save_figure(filename: str) -> Path:
    MODEL_EVAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_EVAL_PLOTS_DIR / filename


@lru_cache(maxsize=1)
def _expected_variant_question_counts() -> tuple[int, dict[str, int]]:
    factual_question_total = sum(len(document.questions) for document in iter_evaluation_documents(settings=("factual",)))
    fictional_counts: dict[str, int] = defaultdict(int)
    for document in iter_evaluation_documents(settings=("fictional",)):
        fictional_counts[str(document.document_variant_id)] += len(document.questions)
    return factual_question_total, dict(sorted(fictional_counts.items()))


def _kernel_density_count_curve(samples: np.ndarray, x_grid: np.ndarray, *, bin_width: float) -> np.ndarray | None:
    if samples.size < 2:
        return None
    std = float(np.std(samples, ddof=1))
    if not np.isfinite(std) or std == 0.0:
        return None
    q75, q25 = np.percentile(samples, [75, 25])
    iqr_scale = float((q75 - q25) / 1.34) if q75 > q25 else std
    sigma = min(std, iqr_scale) if iqr_scale > 0 else std
    bandwidth = 0.9 * sigma * (samples.size ** (-1.0 / 5.0))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        return None
    distances = (x_grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * distances**2).sum(axis=1) / (samples.size * bandwidth * np.sqrt(2.0 * np.pi))
    return density * samples.size * bin_width


def _empirical_q_value(samples: np.ndarray, value: float) -> float | None:
    if samples.size == 0:
        return None
    less = float(np.sum(samples < value))
    equal = float(np.sum(samples == value))
    return (less + 0.5 * equal) / float(samples.size)


def _format_percentile_label(q_value: float) -> str:
    bounded = min(1.0, max(0.0, float(q_value)))
    percentile = int(bounded * 100.0 + 0.5)
    if 10 <= percentile % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(percentile % 10, "th")
    return f"Percentile={percentile}{suffix}"


def _display_model_name(model_name: str) -> str:
    display = str(model_name).strip()
    return display.replace("-it", "-IT").replace("-oss", "-OSS")


def _answer_behavior_legend_handles(answer_behaviors: tuple[str, ...]) -> list[Patch]:
    handles: list[Patch] = []
    for answer_behavior in answer_behaviors:
        color = _ANSWER_BEHAVIOR_COLORS.get(answer_behavior, "#64748b")
        handles.append(
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.18,
                linewidth=1.2,
                label=_ANSWER_BEHAVIOR_LEGEND_LABELS.get(answer_behavior, answer_behavior.title()),
            )
        )
    return handles


@lru_cache(maxsize=1)
def _expected_variant_question_type_counts() -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    factual_counts: dict[str, int] = defaultdict(int)
    fictional_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for document in iter_evaluation_documents(settings=("factual",)):
        for question in document.questions:
            factual_counts[str(question.question_type)] += 1

    for document in iter_evaluation_documents(settings=("fictional",)):
        variant_id = str(document.document_variant_id)
        for question in document.questions:
            fictional_counts[variant_id][str(question.question_type)] += 1

    return (
        dict(sorted(factual_counts.items())),
        {variant_id: dict(sorted(counts.items())) for variant_id, counts in sorted(fictional_counts.items())},
    )


def _load_all_variant_accuracy_distributions(model_names: list[str]) -> dict[str, dict]:
    model_names = [str(model_name).strip() for model_name in model_names if str(model_name).strip()]
    _, fictional_expected = _expected_variant_question_counts()
    variant_ids = [variant_id for variant_id in fictional_expected]
    tallies = _load_model_distribution_tallies(tuple(model_names))

    distributions: dict[str, dict] = {}
    for model_name, tally in tallies.items():
        overall = tally["overall"]
        factual_total = int(overall["factual_total"])
        if factual_total <= 0:
            continue

        variant_accuracies: list[float] = []
        fictional_total = overall["fictional_total"]
        fictional_correct = overall["fictional_correct"]
        incomplete = False
        for variant_id in variant_ids:
            actual_total = int(fictional_total.get(variant_id, 0))
            if actual_total != factual_total:
                incomplete = True
                break
            variant_accuracies.append(100.0 * int(fictional_correct.get(variant_id, 0)) / actual_total)
        if incomplete:
            continue

        distributions[model_name] = {
            "model_name": model_name,
            "factual_accuracy": 100.0 * int(overall["factual_correct"]) / factual_total,
            "variant_accuracies": np.array(variant_accuracies, dtype=float),
        }
    return distributions


def _load_variant_accuracy_distributions_by_question_type(
    model_names: list[str],
    question_types: tuple[str, ...],
) -> dict[str, dict]:
    model_names = [str(model_name).strip() for model_name in model_names if str(model_name).strip()]
    _, expected_fictional = _expected_variant_question_type_counts()
    variant_ids = [variant_id for variant_id in expected_fictional]
    tallies = _load_model_distribution_tallies(tuple(model_names))

    distributions: dict[str, dict] = {}
    for model_name, tally in tallies.items():
        question_type_tallies = tally["question_type"]
        fictional_accuracies_by_question_type: dict[str, np.ndarray] = {}
        factual_accuracy_by_question_type: dict[str, float] = {}
        incomplete = False
        for question_type in question_types:
            expected_total = int(question_type_tallies["factual_total"].get(question_type, 0))
            if expected_total <= 0:
                incomplete = True
                break
            factual_accuracy_by_question_type[question_type] = (
                100.0 * float(question_type_tallies["factual_correct"].get(question_type, 0)) / expected_total
            )
            variant_values: list[float] = []
            for variant_id in variant_ids:
                actual_variant_total = int(
                    (question_type_tallies["fictional_total"].get(variant_id) or {}).get(question_type, 0)
                )
                if actual_variant_total != expected_total:
                    incomplete = True
                    break
                variant_values.append(
                    100.0
                    * float((question_type_tallies["fictional_correct"].get(variant_id) or {}).get(question_type, 0))
                    / actual_variant_total
                )
            if incomplete:
                break
            fictional_accuracies_by_question_type[question_type] = np.array(variant_values, dtype=float)
        if incomplete:
            continue

        distributions[model_name] = {
            "model_name": model_name,
            "factual_accuracy_by_question_type": factual_accuracy_by_question_type,
            "fictional_accuracies_by_question_type": fictional_accuracies_by_question_type,
        }
    return distributions


@lru_cache(maxsize=1)
def _expected_variant_question_type_answer_behavior_counts() -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    factual_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fictional_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for document in iter_evaluation_documents(settings=("factual",)):
        for question in document.questions:
            factual_counts[str(question.question_type)][str(question.answer_behavior)] += 1

    for document in iter_evaluation_documents(settings=("fictional",)):
        variant_id = str(document.document_variant_id)
        for question in document.questions:
            fictional_counts[variant_id][str(question.question_type)][str(question.answer_behavior)] += 1

    return (
        {
            question_type: dict(sorted(answer_counts.items()))
            for question_type, answer_counts in sorted(factual_counts.items())
        },
        {
            variant_id: {
                question_type: dict(sorted(answer_counts.items()))
                for question_type, answer_counts in sorted(question_counts.items())
            }
            for variant_id, question_counts in sorted(fictional_counts.items())
        },
    )


def _load_variant_accuracy_distributions_by_question_type_and_answer_behavior(
    model_names: list[str],
    question_types: tuple[str, ...],
    answer_behaviors: tuple[str, ...],
) -> dict[str, dict]:
    model_names = [str(model_name).strip() for model_name in model_names if str(model_name).strip()]
    _, expected_fictional = _expected_variant_question_type_answer_behavior_counts()
    variant_ids = [variant_id for variant_id in expected_fictional]
    tallies = _load_model_distribution_tallies(tuple(model_names))

    distributions: dict[str, dict] = {}
    for model_name, tally in tallies.items():
        nested_tallies = tally["question_type_answer_behavior"]
        incomplete = False
        factual_accuracy: dict[str, dict[str, float | None]] = {}
        factual_count: dict[str, dict[str, int]] = {}
        fictional_accuracies: dict[str, dict[str, np.ndarray]] = {}

        for question_type in question_types:
            factual_accuracy[question_type] = {}
            factual_count[question_type] = {}
            fictional_accuracies[question_type] = {}
            for answer_behavior in answer_behaviors:
                expected_total = int((nested_tallies["factual_total"].get(question_type) or {}).get(answer_behavior, 0))
                if expected_total < 0:
                    incomplete = True
                    break
                factual_count[question_type][answer_behavior] = expected_total
                factual_accuracy[question_type][answer_behavior] = (
                    100.0
                    * float((nested_tallies["factual_correct"].get(question_type) or {}).get(answer_behavior, 0))
                    / expected_total
                    if expected_total
                    else None
                )
                variant_values: list[float] = []
                for variant_id in variant_ids:
                    actual_variant_total = int(
                        (((nested_tallies["fictional_total"].get(variant_id) or {}).get(question_type) or {}).get(answer_behavior, 0))
                    )
                    if actual_variant_total != expected_total:
                        incomplete = True
                        break
                    if actual_variant_total:
                        correct_count = int(
                            (((nested_tallies["fictional_correct"].get(variant_id) or {}).get(question_type) or {}).get(answer_behavior, 0))
                        )
                        variant_values.append(100.0 * float(correct_count) / actual_variant_total)
                if incomplete:
                    break
                fictional_accuracies[question_type][answer_behavior] = np.array(variant_values, dtype=float)
            if incomplete:
                break
        if incomplete:
            continue

        distributions[model_name] = {
            "model_name": model_name,
            "factual_accuracy_by_question_type_and_answer_behavior": factual_accuracy,
            "factual_count_by_question_type_and_answer_behavior": factual_count,
            "fictional_accuracies_by_question_type_and_answer_behavior": fictional_accuracies,
        }
    return distributions


def _model_evaluated_paths(model_folder: str) -> list[Path]:
    paths: list[Path] = []
    for theme_dir in sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.iterdir()):
        if not theme_dir.is_dir():
            continue
        candidate_dir = theme_dir / model_folder
        if candidate_dir.is_dir():
            paths.extend(sorted(candidate_dir.glob("*_evaluated_outputs.yaml")))
    return paths


def _increment_nested(mapping: dict, *keys: str, amount: int) -> None:
    current = mapping
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = int(current.get(keys[-1], 0)) + int(amount)


def _aggregate_model_distribution_tally(model_folder: str) -> tuple[str, dict]:
    overall = {
        "factual_correct": 0,
        "factual_total": 0,
        "fictional_correct": {},
        "fictional_total": {},
    }
    question_type = {
        "factual_correct": {},
        "factual_total": {},
        "fictional_correct": {},
        "fictional_total": {},
    }
    question_type_answer_behavior = {
        "factual_correct": {},
        "factual_total": {},
        "fictional_correct": {},
        "fictional_total": {},
    }

    for evaluated_path in _model_evaluated_paths(model_folder):
        payload = yaml.safe_load(evaluated_path.read_text(encoding="utf-8")) or {}
        if is_excluded_evaluation_document(str(payload.get("document_id") or "")):
            continue
        setting_id = str(payload.get("document_setting") or "").strip().lower()
        variant_id = str(payload.get("document_variant_id") or "v01")

        for row in payload.get("results") or []:
            question_name = str(row.get("question_type") or "unknown")
            answer_behavior = str(row.get("answer_behavior") or "unknown")
            correct = 1 if row.get("final_is_correct") is True else 0

            if setting_id == "factual":
                overall["factual_correct"] += correct
                overall["factual_total"] += 1
                _increment_nested(question_type["factual_correct"], question_name, amount=correct)
                _increment_nested(question_type["factual_total"], question_name, amount=1)
                _increment_nested(question_type_answer_behavior["factual_correct"], question_name, answer_behavior, amount=correct)
                _increment_nested(question_type_answer_behavior["factual_total"], question_name, answer_behavior, amount=1)
            elif setting_id == "fictional":
                _increment_nested(overall["fictional_correct"], variant_id, amount=correct)
                _increment_nested(overall["fictional_total"], variant_id, amount=1)
                _increment_nested(question_type["fictional_correct"], variant_id, question_name, amount=correct)
                _increment_nested(question_type["fictional_total"], variant_id, question_name, amount=1)
                _increment_nested(
                    question_type_answer_behavior["fictional_correct"],
                    variant_id,
                    question_name,
                    answer_behavior,
                    amount=correct,
                )
                _increment_nested(
                    question_type_answer_behavior["fictional_total"],
                    variant_id,
                    question_name,
                    answer_behavior,
                    amount=1,
                )

    return model_folder, {
        "overall": overall,
        "question_type": question_type,
        "question_type_answer_behavior": question_type_answer_behavior,
    }


@lru_cache(maxsize=8)
def _load_model_distribution_tallies(model_names_key: tuple[str, ...]) -> dict[str, dict]:
    model_names = [str(model_name).strip() for model_name in model_names_key if str(model_name).strip()]
    model_folders = {sanitize_model_name(model_name): model_name for model_name in model_names}
    folders = sorted(model_folders)
    if not folders:
        return {}

    max_workers = min(len(folders), max(1, min(os.cpu_count() or 1, 8)))
    if len(folders) == 1:
        aggregated = [_aggregate_model_distribution_tally(folders[0])]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            aggregated = list(executor.map(_aggregate_model_distribution_tally, folders))

    return {model_folders[folder]: tally for folder, tally in aggregated}


@lru_cache(maxsize=16)
def _load_model_pair_outcomes(model_name: str) -> dict[str, dict]:
    """Load per-question factual and fictional correctness outcomes for one model."""
    model_folder = sanitize_model_name(model_name)
    pair_outcomes: dict[str, dict] = {}
    allowed_settings = {"factual", "fictional"}

    for evaluated_path in _model_evaluated_paths(model_folder):
        payload = yaml.safe_load(evaluated_path.read_text(encoding="utf-8")) or {}
        if is_excluded_evaluation_document(str(payload.get("document_id") or "")):
            continue
        setting_id = str(payload.get("document_setting") or "").strip().lower()
        if setting_id not in allowed_settings:
            continue
        variant_id = str(payload.get("document_variant_id") or "v01")
        document_id = str(payload.get("document_id") or "")

        for row in payload.get("results") or []:
            pair_key = str(row.get("pair_key") or f"{document_id}::{row.get('question_id') or ''}")
            question_type = str(row.get("question_type") or "unknown").strip().lower()
            entry = pair_outcomes.setdefault(
                pair_key,
                {
                    "document_id": document_id,
                    "question_id": str(row.get("question_id") or ""),
                    "question_type": question_type,
                    "answer_behavior": str(row.get("answer_behavior") or "unknown").strip().lower(),
                    "factual": None,
                    "fictional": {},
                },
            )
            if setting_id == "factual":
                entry["factual"] = 1.0 if row.get("final_is_correct") is True else 0.0
            else:
                entry[setting_id][variant_id] = 1.0 if row.get("final_is_correct") is True else 0.0

    return pair_outcomes


def _bootstrap_mean_ci(
    samples: np.ndarray,
    *,
    seed: int,
    n_resamples: int = 10000,
    confidence: float = 0.95,
    batch_size: int = 2048,
) -> tuple[float, float]:
    if samples.size == 0:
        return 0.0, 0.0
    if samples.size == 1:
        value = float(samples[0])
        return value, value

    rng = np.random.default_rng(seed)
    lower_q = 100.0 * (1.0 - confidence) / 2.0
    upper_q = 100.0 - lower_q
    means: list[np.ndarray] = []
    remaining = int(n_resamples)
    while remaining > 0:
        current = min(batch_size, remaining)
        indices = rng.integers(0, samples.size, size=(current, samples.size))
        means.append(np.mean(samples[indices], axis=1))
        remaining -= current
    bootstrap_distribution = np.concatenate(means, axis=0)
    return (
        float(np.percentile(bootstrap_distribution, lower_q)),
        float(np.percentile(bootstrap_distribution, upper_q)),
    )


def _paired_permutation_p_value(
    samples: np.ndarray,
    *,
    seed: int,
    n_permutations: int = 20000,
    batch_size: int = 2048,
) -> float:
    if samples.size == 0:
        return 1.0
    observed = float(abs(np.mean(samples)))
    if not np.isfinite(observed) or observed == 0.0:
        return 1.0

    rng = np.random.default_rng(seed)
    exceed_count = 0
    generated = 0
    while generated < n_permutations:
        current = min(batch_size, n_permutations - generated)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(current, samples.size))
        permuted = np.abs(np.mean(signs * samples[None, :], axis=1))
        exceed_count += int(np.count_nonzero(permuted >= observed - 1e-12))
        generated += current
    return float((exceed_count + 1) / (n_permutations + 1))


def _paired_t_test_p_value(
    factual_values: np.ndarray,
    perturbed_values: np.ndarray,
) -> tuple[float, float]:
    if factual_values.size == 0 or perturbed_values.size == 0:
        return 0.0, 1.0
    if factual_values.size != perturbed_values.size:
        raise ValueError("Paired t-test requires arrays of the same length.")
    if factual_values.size < 2:
        return 0.0, 1.0
    if np.allclose(factual_values, perturbed_values):
        return 0.0, 1.0
    result = ttest_rel(factual_values, perturbed_values, nan_policy="omit", alternative="two-sided")
    t_stat = float(result.statistic) if np.isfinite(result.statistic) else 0.0
    p_value = float(result.pvalue) if np.isfinite(result.pvalue) else 1.0
    return t_stat, p_value


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    adjusted = [1.0 for _ in p_values]
    order = np.argsort(np.asarray(p_values, dtype=float))
    running_min = 1.0
    total = float(len(p_values))
    for rank_index in range(len(order) - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = float(rank_index + 1)
        candidate = min(1.0, float(p_values[original_index]) * total / rank)
        running_min = min(running_min, candidate)
        adjusted[original_index] = running_min
    return adjusted


def _significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    return "*" if p_value < 0.05 else ""


def _paired_effect_panel_rows(
    model_names: list[str],
    *,
    target_setting: str,
    allowed_question_types: set[str],
    seed_offset: int,
) -> tuple[list[dict], tuple[str, ...]]:
    common_pair_keys: set[str] | None = None
    model_to_outcomes = {model_name: _load_model_pair_outcomes(model_name) for model_name in model_names}

    for model_name in model_names:
        eligible: set[str] = set()
        for pair_key, entry in model_to_outcomes[model_name].items():
            if str(entry.get("question_type") or "") not in allowed_question_types:
                continue
            if entry.get("factual") is None:
                continue
            variant_map = entry.get(target_setting) or {}
            if set(str(variant_id) for variant_id in variant_map.keys()) != set(_EXPECTED_VARIANT_IDS):
                continue
            eligible.add(pair_key)
        common_pair_keys = eligible if common_pair_keys is None else (common_pair_keys & eligible)

    ordered_keys = tuple(sorted(common_pair_keys or set()))
    rows: list[dict] = []
    raw_p_values: list[float] = []

    for model_index, model_name in enumerate(model_names):
        outcomes = model_to_outcomes[model_name]
        factual_values = np.asarray([float(outcomes[pair_key]["factual"]) for pair_key in ordered_keys], dtype=float)
        perturbed_values = np.asarray(
            [
                float(
                    np.mean([float((outcomes[pair_key][target_setting] or {}).get(variant_id, 0.0)) for variant_id in _EXPECTED_VARIANT_IDS])
                )
                for pair_key in ordered_keys
            ],
            dtype=float,
        )
        paired_drop = factual_values - perturbed_values
        mean_drop = float(np.mean(paired_drop)) if paired_drop.size else 0.0
        ci_low, ci_high = _bootstrap_mean_ci(paired_drop, seed=1729 + seed_offset + model_index)
        t_stat, p_value = _paired_t_test_p_value(factual_values, perturbed_values)
        raw_p_values.append(p_value)
        rows.append(
            {
                "model_name": model_name,
                "count_pairs": int(paired_drop.size),
                "factual_accuracy": float(np.mean(factual_values)) if factual_values.size else 0.0,
                "perturbed_accuracy": float(np.mean(perturbed_values)) if perturbed_values.size else 0.0,
                "mean_drop": mean_drop,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "t_statistic": t_stat,
                "p_value": p_value,
            }
        )

    adjusted_p_values = _benjamini_hochberg(raw_p_values)
    for row, adjusted_p_value in zip(rows, adjusted_p_values, strict=True):
        row["p_value_bh"] = adjusted_p_value
        row["stars"] = _significance_stars(float(row["p_value"]))

    return rows, ordered_keys


def _performance_drop_output_dir() -> Path:
    output_dir = MODEL_EVAL_PLOTS_DIR / "performance_drop_ttest"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _paired_effect_rows_for_setting_and_question_types(
    model_names: list[str],
    *,
    target_setting: str,
    allowed_question_types: set[str] | None,
    seed_offset: int,
) -> list[dict]:
    rows, _ = _paired_effect_panel_rows(
        model_names,
        target_setting=target_setting,
        allowed_question_types=allowed_question_types or set(_PREFERRED_QUESTION_TYPES),
        seed_offset=seed_offset,
    )
    return rows


def export_paired_performance_drop_tables(metrics_payloads: list[dict]) -> list[Path]:
    model_names = [str(payload.get("model_name") or "").strip() for payload in metrics_payloads if str(payload.get("model_name") or "").strip()]
    if not model_names:
        return []

    output_dir = _performance_drop_output_dir()
    setting_specs = (
        ("fictional", "full_replacement"),
    )
    available_question_types = set()
    for model_name in model_names:
        for entry in _load_model_pair_outcomes(model_name).values():
            question_type = str(entry.get("question_type") or "").strip().lower()
            if question_type:
                available_question_types.add(question_type)
    ordered_question_types = [qt for qt in _PREFERRED_QUESTION_TYPES if qt in available_question_types]
    ordered_question_types.extend(sorted(available_question_types - set(ordered_question_types)))

    overall_rows: list[dict] = []
    by_question_type_rows: list[dict] = []
    for setting_index, (setting_id, setting_label) in enumerate(setting_specs):
        setting_rows, _ = _paired_effect_panel_rows(
            model_names,
            target_setting=setting_id,
            allowed_question_types=available_question_types or set(_PREFERRED_QUESTION_TYPES),
            seed_offset=10000 * (setting_index + 1),
        )
        for row in setting_rows:
            overall_rows.append(
                {
                    "setting_id": setting_id,
                    "setting_label": setting_label,
                    **row,
                }
            )
        for question_index, question_type in enumerate(ordered_question_types):
            question_rows, _ = _paired_effect_panel_rows(
                model_names,
                target_setting=setting_id,
                allowed_question_types={question_type},
                seed_offset=10000 * (setting_index + 1) + 100 * (question_index + 1),
            )
            for row in question_rows:
                by_question_type_rows.append(
                    {
                        "setting_id": setting_id,
                        "setting_label": setting_label,
                        "question_type": question_type,
                        **row,
                    }
                )

    json_payload = {
        "test": "paired_t_test",
        "overall": overall_rows,
        "by_question_type": by_question_type_rows,
    }
    json_path = output_dir / "paired_performance_drop_summary.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    overall_csv_path = output_dir / "paired_performance_drop_overall.csv"
    with overall_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting_id",
                "setting_label",
                "model_name",
                "count_pairs",
                "factual_accuracy",
                "perturbed_accuracy",
                "mean_drop",
                "ci_low",
                "ci_high",
                "t_statistic",
                "p_value",
                "p_value_bh",
                "stars",
            ],
        )
        writer.writeheader()
        writer.writerows(overall_rows)

    by_question_type_csv_path = output_dir / "paired_performance_drop_by_question_type.csv"
    with by_question_type_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "setting_id",
                "setting_label",
                "question_type",
                "model_name",
                "count_pairs",
                "factual_accuracy",
                "perturbed_accuracy",
                "mean_drop",
                "ci_low",
                "ci_high",
                "t_statistic",
                "p_value",
                "p_value_bh",
                "stars",
            ],
        )
        writer.writeheader()
        writer.writerows(by_question_type_rows)

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Paired Performance Drop (Paired t-test)",
                "",
                "- `paired_performance_drop_overall.csv`: mean paired drop over the whole dataset for each perturbation setting.",
                "- `paired_performance_drop_by_question_type.csv`: same statistic broken down by question type.",
                "- `paired_performance_drop_summary.json`: combined machine-readable export.",
                "",
                "Definition:",
                "- `mean_drop = mean(Y_factual - Y_perturbed)` at the paired question level.",
                "- Significance uses a two-sided paired t-test.",
            ]
        ),
        encoding="utf-8",
    )
    return [json_path, overall_csv_path, by_question_type_csv_path, readme_path]


def plot_paired_reasoning_fragility_effects(metrics_payloads: list[dict]) -> list[Path]:
    """Plot paired performance drops for the final full-replacement analysis."""
    model_names = [str(payload.get("model_name") or "").strip() for payload in metrics_payloads if str(payload.get("model_name") or "").strip()]
    if not model_names:
        return [
            _plot_empty_state(
                "paired_reasoning_fragility_full_replacement.png",
                "Paired Reasoning Fragility Effects",
                "No model metrics were available.",
            )
        ]

    panel_specs = (
        {
            "filename": "paired_reasoning_fragility_full_replacement.png",
            "setting": "fictional",
            "question_types": {"arithmetic", "temporal", "inference"},
            "seed_offset": 0,
        },
    )

    panel_results: list[dict] = []
    for spec in panel_specs:
        rows, ordered_keys = _paired_effect_panel_rows(
            model_names,
            target_setting=str(spec["setting"]),
            allowed_question_types=set(spec["question_types"]),
            seed_offset=int(spec["seed_offset"]),
        )
        panel_results.append({"spec": spec, "rows": rows, "pair_keys": ordered_keys})

    main_panel_rows = panel_results[0]["rows"]
    if not main_panel_rows:
        return [
            _plot_empty_state(
                "paired_reasoning_fragility_full_replacement.png",
                "Paired Reasoning Fragility Effects",
                "No complete paired evaluation set was available for the requested models.",
            )
        ]

    model_order = [
        row["model_name"]
        for row in sorted(main_panel_rows, key=lambda row: float(row.get("mean_drop", 0.0)), reverse=True)
    ]
    model_colors = {model_name: _line_color(index, len(model_order)) for index, model_name in enumerate(model_order)}

    y_min = 0.0
    y_max = 0.0
    for panel in panel_results:
        for row in panel["rows"]:
            y_min = min(y_min, float(row["ci_low"]) * 100.0)
            y_max = max(y_max, float(row["ci_high"]) * 100.0)
    y_span = max(y_max - y_min, 1.0)
    pad = 0.16 * y_span
    y_limits = (y_min - pad, y_max + pad)
    written_paths: list[Path] = []

    for panel in panel_results:
        _apply_plot_style()
        spec = panel["spec"]
        rows_by_model = {str(row["model_name"]): row for row in panel["rows"]}
        panel_model_order = [model_name for model_name in model_order if model_name in rows_by_model]
        fig, axis = plt.subplots(figsize=(max(8.5, len(panel_model_order) * 2.15), 5.2))
        x_positions = np.arange(len(panel_model_order), dtype=float)
        x_labels = [_display_model_name(model_name) for model_name in panel_model_order]
        means = np.array([100.0 * float(rows_by_model[model_name]["mean_drop"]) for model_name in panel_model_order], dtype=float)
        ci_low = np.array([100.0 * float(rows_by_model[model_name]["ci_low"]) for model_name in panel_model_order], dtype=float)
        ci_high = np.array([100.0 * float(rows_by_model[model_name]["ci_high"]) for model_name in panel_model_order], dtype=float)
        lower_err = means - ci_low
        upper_err = ci_high - means
        colors = [model_colors[model_name] for model_name in panel_model_order]

        axis.bar(x_positions, means, color=colors, width=0.74, edgecolor="black", linewidth=0.8, alpha=0.9)
        axis.errorbar(
            x_positions,
            means,
            yerr=np.vstack([lower_err, upper_err]),
            fmt="none",
            ecolor="black",
            elinewidth=1.25,
            capsize=4,
            capthick=1.25,
            zorder=3,
        )
        axis.axhline(0.0, color="#475569", linewidth=1.2, linestyle="--", alpha=0.75)
        axis.set_ylim(*y_limits)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(x_labels, rotation=25, ha="right")
        axis.grid(axis="y", linestyle="--", alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylabel(
            r"Mean Paired Accuracy Drop ($\mathbb{E}[Y_i^{\mathrm{factual}} - Y_i^{\mathrm{fictional}}]$)",
            labelpad=6,
            fontsize=11,
        )
        for idx, model_name in enumerate(panel_model_order):
            star = str(rows_by_model[model_name].get("stars") or "")
            if not star:
                continue
            mean_value = means[idx]
            low_value = ci_low[idx]
            high_value = ci_high[idx]
            offset = 0.04 * y_span
            if mean_value >= 0:
                star_y = high_value + offset
                va = "bottom"
            else:
                star_y = low_value - offset
                va = "top"
            axis.text(
                x_positions[idx],
                star_y,
                star,
                ha="center",
                va=va,
                fontsize=13,
                fontweight="bold",
                color="#111827",
            )

        fig.subplots_adjust(left=0.24, bottom=0.22, right=0.98, top=0.98)
        output_path = _save_figure(str(spec["filename"]))
        fig.savefig(output_path, dpi=250)
        plt.close(fig)
        written_paths.append(output_path)
    return written_paths


def _ordered_setting_specs(metrics_payloads: list[dict]) -> list[DatasetSettingSpec]:
    setting_ids: list[str] = []
    for payload in metrics_payloads:
        setting_ids.extend((payload.get("document_settings") or {}).keys())
        setting_ids.extend((payload.get("overall_accuracy") or {}).keys())
    return order_dataset_settings(setting_ids)


def _ordered_question_types(metrics_payloads: list[dict]) -> tuple[str, ...]:
    discovered: set[str] = set()
    for payload in metrics_payloads:
        discovered.update((payload.get("accuracy_by_question_type") or {}).keys())
        discovered.update((payload.get("accuracy_by_question_type_and_answer_behavior") or {}).keys())

    ordered = [question_type for question_type in _PREFERRED_QUESTION_TYPES if question_type in discovered]
    ordered.extend(sorted(discovered - set(_PREFERRED_QUESTION_TYPES)))
    return tuple(ordered or _PREFERRED_QUESTION_TYPES)


def _error_bars(summary: dict, key: str) -> tuple[float, np.ndarray]:
    value = float(summary.get(key, 0.0))
    lower = max(0.0, value - float(summary.get("ci_low", 0.0)))
    upper = max(0.0, float(summary.get("ci_high", 0.0)) - value)
    return value, np.array([[lower], [upper]])


def _wald_error_bars(summary: dict, key: str) -> tuple[float, np.ndarray]:
    value = float(summary.get(key, 0.0))
    total = int(summary.get("count_total", 0))
    if total <= 0:
        return value, np.array([[0.0], [0.0]])
    z_value = 1.96
    standard_error = sqrt(max(value * (1.0 - value), 0.0) / total)
    margin = z_value * standard_error
    lower = min(value, margin)
    upper = min(1.0 - value, margin)
    return value, np.array([[lower], [upper]])


def _line_color(index: int, count: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab10", max(count, 1))
    return cmap(index)


def _series_color(index: int, count: int) -> tuple[float, float, float, float]:
    cmap_name = "tab20" if count > 10 else "tab10"
    cmap = plt.get_cmap(cmap_name, max(count, 1))
    return cmap(index)


def _setting_axis(
    specs: list[DatasetSettingSpec], *, include_factual: bool = True
) -> tuple[np.ndarray, list[str], list[DatasetSettingSpec]]:
    ordered_specs = [spec for spec in specs if include_factual or not spec.is_factual]
    x_values = np.array([spec.replacement_proportion * 100.0 for spec in ordered_specs], dtype=float)
    tick_labels = [spec.short_label for spec in ordered_specs]
    return x_values, tick_labels, ordered_specs


def _plot_empty_state(filename: str, title: str, message: str) -> Path:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    output_path = _save_figure(filename)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_metric_lines(
    metrics_payloads: list[dict],
    *,
    filename: str,
    title: str,
    ylabel: str,
    metric_key: str,
    summary_lookup,
    include_factual: bool = True,
) -> Path:
    setting_specs = _ordered_setting_specs(metrics_payloads)
    if not setting_specs:
        return _plot_empty_state(filename, title, "No metrics were available.")

    x_values, tick_labels, ordered_specs = _setting_axis(setting_specs, include_factual=include_factual)
    if len(ordered_specs) == 0:
        return _plot_empty_state(filename, title, "No fictional settings were available.")

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(8, len(ordered_specs) * 1.5), 5.5))
    error_style = dict(elinewidth=1.2, capsize=4, capthick=1.2)

    for index, payload in enumerate(metrics_payloads):
        y_values: list[float] = []
        y_errors: list[np.ndarray] = []
        for spec in ordered_specs:
            summary = summary_lookup(payload, spec.setting_id)
            value, error = _error_bars(summary, metric_key)
            y_values.append(value * 100.0)
            y_errors.append(error[:, 0] * 100.0)

        ax.errorbar(
            x_values,
            y_values,
            yerr=np.array(y_errors).T,
            fmt="-o",
            linewidth=2.0,
            markersize=6,
            color=_line_color(index, len(metrics_payloads)),
            label=payload["model_name"],
            **error_style,
        )

    ax.set_xlabel("Replacement Proportion")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, edgecolor="black", framealpha=1.0)
    plt.tight_layout()
    output_path = _save_figure(filename)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_setting_accuracy(metrics_payloads: list[dict]) -> Path:
    """Plot model accuracy across factual and fictional replacement proportions."""
    return _plot_metric_lines(
        metrics_payloads,
        filename="accuracy_by_replacement_proportion.png",
        title="Model Accuracy by Replacement Proportion",
        ylabel="Accuracy (%)",
        metric_key="accuracy",
        summary_lookup=lambda payload, setting_id: (payload.get("overall_accuracy") or {}).get(setting_id, {}),
    )


def plot_factual_vs_fictional_accuracy_bars(metrics_payloads: list[dict]) -> Path:
    """Render a grouped bar chart comparing factual and fictional EM with Wald confidence intervals."""
    comparable_rows: list[dict] = []
    max_factual_total = max(
        int(((payload.get("overall_accuracy") or {}).get("factual") or {}).get("count_total", 0))
        for payload in metrics_payloads
    )
    max_fictional_total = max(
        int(((payload.get("overall_accuracy") or {}).get("fictional") or {}).get("count_total", 0))
        for payload in metrics_payloads
    )

    for payload in metrics_payloads:
        overall = payload.get("overall_accuracy") or {}
        factual = overall.get("factual") or {}
        fictional = overall.get("fictional") or {}
        factual_total = int(factual.get("count_total", 0))
        fictional_total = int(fictional.get("count_total", 0))
        if factual_total != max_factual_total or fictional_total != max_fictional_total:
            continue
        comparable_rows.append(
            {
                "model_name": str(payload.get("model_name") or "unknown"),
                "factual": factual,
                "fictional": fictional,
            }
        )

    if not comparable_rows:
        return _plot_empty_state(
            "factual_vs_fictional_accuracy_bars.png",
            "Factual vs Fictional Accuracy",
            "No model had complete factual and fictional outputs.",
        )

    comparable_rows.sort(key=lambda row: float((row["factual"] or {}).get("accuracy", 0.0)), reverse=True)

    labels = [_display_model_name(row["model_name"]) for row in comparable_rows]
    factual_values: list[float] = []
    factual_errors: list[np.ndarray] = []
    fictional_values: list[float] = []
    fictional_errors: list[np.ndarray] = []

    for row in comparable_rows:
        factual_value, factual_error = _wald_error_bars(row["factual"], "accuracy")
        fictional_value, fictional_error = _wald_error_bars(row["fictional"], "accuracy")
        factual_values.append(factual_value * 100.0)
        factual_errors.append(factual_error[:, 0] * 100.0)
        fictional_values.append(fictional_value * 100.0)
        fictional_errors.append(fictional_error[:, 0] * 100.0)

    x_positions = np.arange(len(comparable_rows), dtype=float)
    bar_width = 0.36

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(max(9.5, len(comparable_rows) * 1.35), 5.8))
    error_style = dict(elinewidth=1.2, capsize=4, capthick=1.2, ecolor="#333333")

    ax.bar(
        x_positions - bar_width / 2.0,
        factual_values,
        width=bar_width,
        color="#f59e0b",
        edgecolor="black",
        linewidth=1.0,
        yerr=np.array(factual_errors).T,
        label="Factual",
        error_kw=error_style,
    )
    ax.bar(
        x_positions + bar_width / 2.0,
        fictional_values,
        width=bar_width,
        color="#fcd34d",
        edgecolor="black",
        linewidth=1.0,
        yerr=np.array(fictional_errors).T,
        label="Fictional",
        error_kw=error_style,
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=True, edgecolor="black", framealpha=1.0)
    plt.tight_layout()
    output_path = _save_figure("factual_vs_fictional_accuracy_bars.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_accuracy_breakdown(metrics_payloads: list[dict], *, breakdown_key: str) -> Path:
    """Plot accuracy across proportions by question type or answer behavior."""
    if breakdown_key == "question_type":
        categories = _ordered_question_types(metrics_payloads)
        title = "Accuracy by Question Type"
        filename = "accuracy_by_question_type_over_proportion.png"
        payload_key = "accuracy_by_question_type"
    elif breakdown_key == "answer_behavior":
        categories = ("variant", "invariant")
        title = "Accuracy by Answer Behavior"
        filename = "accuracy_by_answer_behavior_over_proportion.png"
        payload_key = "accuracy_by_answer_behavior"
    else:
        raise ValueError(f"Unknown breakdown key: {breakdown_key!r}")

    setting_specs = _ordered_setting_specs(metrics_payloads)
    if not setting_specs:
        return _plot_empty_state(filename, title, "No metrics were available.")

    x_values, tick_labels, ordered_specs = _setting_axis(setting_specs, include_factual=True)
    _apply_plot_style()
    fig, axes = plt.subplots(1, len(categories), figsize=(max(10, len(categories) * 4.5), 4.8), sharey=True)
    if len(categories) == 1:
        axes = [axes]

    error_style = dict(elinewidth=1.1, capsize=4, capthick=1.1)
    for axis, category in zip(axes, categories, strict=True):
        for index, payload in enumerate(metrics_payloads):
            y_values: list[float] = []
            y_errors: list[np.ndarray] = []
            for spec in ordered_specs:
                summary = ((payload.get(payload_key) or {}).get(category) or {}).get(spec.setting_id, {})
                value, error = _error_bars(summary, "accuracy")
                y_values.append(value * 100.0)
                y_errors.append(error[:, 0] * 100.0)

            axis.errorbar(
                x_values,
                y_values,
                yerr=np.array(y_errors).T,
                fmt="-o",
                linewidth=2.0,
                markersize=6,
                color=_line_color(index, len(metrics_payloads)),
                label=payload["model_name"],
                **error_style,
            )

        axis.set_xticks(x_values)
        axis.set_xticklabels(tick_labels)
        axis.set_ylim(0, 110)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Accuracy (%)")
    axes[0].legend(frameon=True, edgecolor="black", framealpha=1.0)
    plt.tight_layout()
    output_path = _save_figure(filename)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_question_type_accuracy_heatmap(metrics_payloads: list[dict]) -> Path:
    """Render one compact heatmap view for factual EM and fictional-vs-factual deltas by question type."""
    question_types = _ordered_question_types(metrics_payloads)
    if not question_types:
        return _plot_empty_state(
            "question_type_accuracy_heatmap.png",
            "Question-Type Accuracy Summary",
            "No question-type metrics were available.",
        )

    expected_factual = {
        question_type: max(
            int((((payload.get("accuracy_by_question_type") or {}).get(question_type) or {}).get("factual") or {}).get("count_total", 0))
            for payload in metrics_payloads
        )
        for question_type in question_types
    }
    expected_fictional = {
        question_type: max(
            int(
                (((payload.get("accuracy_by_question_type") or {}).get(question_type) or {}).get("fictional") or {}).get(
                    "count_total", 0
                )
            )
            for payload in metrics_payloads
        )
        for question_type in question_types
    }

    model_labels: list[str] = []
    factual_matrix: list[list[float]] = []
    delta_matrix: list[list[float]] = []
    has_partial = False

    for payload in metrics_payloads:
        by_question_type = payload.get("accuracy_by_question_type") or {}
        partial = False
        factual_row: list[float] = []
        delta_row: list[float] = []
        for question_type in question_types:
            summaries = by_question_type.get(question_type) or {}
            factual_summary = summaries.get("factual") or {}
            fictional_summary = summaries.get("fictional") or {}

            factual_accuracy = float(factual_summary.get("accuracy", 0.0)) * 100.0
            fictional_accuracy = float(fictional_summary.get("accuracy", 0.0)) * 100.0
            factual_row.append(factual_accuracy)
            delta_row.append(fictional_accuracy - factual_accuracy)

            if int(factual_summary.get("count_total", 0)) != expected_factual[question_type]:
                partial = True
            if int(fictional_summary.get("count_total", 0)) != expected_fictional[question_type]:
                partial = True

        label = _display_model_name(str(payload.get("model_name") or "unknown"))
        if partial:
            label = f"{label}*"
            has_partial = True
        model_labels.append(label)
        factual_matrix.append(factual_row)
        delta_matrix.append(delta_row)

    factual_values = np.array(factual_matrix, dtype=float)
    delta_values = np.array(delta_matrix, dtype=float)
    max_abs_delta = max(1.0, float(np.max(np.abs(delta_values))))

    _apply_plot_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(12, len(question_types) * 2.8), max(4.8, len(model_labels) * 0.75 + 2.4)),
        gridspec_kw={"width_ratios": [1.0, 1.05]},
    )

    heatmaps = [
        (
            axes[0],
            factual_values,
            "Blues",
            None,
            "Factual EM (%)",
            lambda value: f"{value:.1f}",
        ),
        (
            axes[1],
            delta_values,
            "RdBu_r",
            TwoSlopeNorm(vmin=-max_abs_delta, vcenter=0.0, vmax=max_abs_delta),
            "Fictional - Factual (pp)",
            lambda value: f"{value:+.1f}",
        ),
    ]

    for axis, values, cmap, norm, title, formatter in heatmaps:
        image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        axis.set_xticks(np.arange(len(question_types)))
        axis.set_xticklabels([question_type.title() for question_type in question_types], rotation=20, ha="right")
        axis.set_yticks(np.arange(len(model_labels)))
        axis.set_yticklabels(model_labels)
        axis.set_xticks(np.arange(-0.5, len(question_types), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(model_labels), 1), minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

        cmap_obj = plt.get_cmap(cmap)
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                rgba = cmap_obj(norm(values[row_index, column_index]) if norm is not None else values[row_index, column_index] / 100.0)
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                text_color = "black" if luminance > 0.6 else "white"
                axis.text(
                    column_index,
                    row_index,
                    formatter(float(values[row_index, column_index])),
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=11,
                    fontweight="semibold",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)

    if has_partial:
        fig.text(
            0.01,
            -0.02,
            "* Partial local snapshot: at least one question type is missing some factual or fictional outputs.",
            ha="left",
            va="top",
            fontsize=10,
        )
    plt.tight_layout()
    output_path = _save_figure("question_type_accuracy_heatmap.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_accuracy_by_question_type_and_answer_behavior(metrics_payloads: list[dict]) -> Path:
    """Plot accuracy across proportions for each question-type / answer-behavior pair."""
    setting_specs = _ordered_setting_specs(metrics_payloads)
    if not setting_specs:
        return _plot_empty_state(
            "accuracy_by_question_type_and_answer_behavior_over_proportion.png",
            "Accuracy by Question Type and Answer Behavior",
            "No metrics were available.",
        )

    x_values, tick_labels, ordered_specs = _setting_axis(setting_specs, include_factual=True)
    question_types = _ordered_question_types(metrics_payloads)
    answer_behaviors = ("variant", "invariant")

    _apply_plot_style()
    fig, axes = plt.subplots(
        len(answer_behaviors),
        len(question_types),
        figsize=(max(12, len(question_types) * 4.0), 8.0),
        sharex=True,
        sharey=True,
    )
    axes_array = np.array(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array[:, np.newaxis]
    error_style = dict(elinewidth=1.1, capsize=4, capthick=1.1)

    for row_index, answer_behavior in enumerate(answer_behaviors):
        for column_index, question_type in enumerate(question_types):
            axis = axes_array[row_index, column_index]
            for model_index, payload in enumerate(metrics_payloads):
                y_values: list[float] = []
                y_errors: list[np.ndarray] = []
                nested_summary = (
                    (payload.get("accuracy_by_question_type_and_answer_behavior") or {}).get(question_type) or {}
                ).get(answer_behavior) or {}
                for spec in ordered_specs:
                    summary = nested_summary.get(spec.setting_id, {})
                    value, error = _error_bars(summary, "accuracy")
                    y_values.append(value * 100.0)
                    y_errors.append(error[:, 0] * 100.0)

                axis.errorbar(
                    x_values,
                    y_values,
                    yerr=np.array(y_errors).T,
                    fmt="-o",
                    linewidth=2.0,
                    markersize=6,
                    color=_line_color(model_index, len(metrics_payloads)),
                    label=payload["model_name"],
                    **error_style,
                )

            axis.set_xticks(x_values)
            axis.set_xticklabels(tick_labels)
            axis.set_ylim(0, 110)
            axis.grid(axis="y", linestyle="--", alpha=0.25)
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            if row_index == 0:
                axis.set_title(question_type.title(), pad=14)

    axes_array[0, 0].legend(frameon=True, edgecolor="black", framealpha=1.0)
    axes_array[0, 0].set_ylabel("Accuracy (%)")
    axes_array[1, 0].set_ylabel("Accuracy (%)")
    plt.tight_layout()
    output_path = _save_figure("accuracy_by_question_type_and_answer_behavior_over_proportion.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_conversion_ratios(metrics_payloads: list[dict]) -> Path:
    """Plot factual-to-fictional conversion ratios across replacement proportions."""
    setting_specs = [spec for spec in _ordered_setting_specs(metrics_payloads) if not spec.is_factual]
    if not setting_specs:
        return _plot_empty_state(
            "conversion_ratios_over_proportion.png", "Conversion Ratios", "No fictional settings were available."
        )

    x_values, tick_labels, ordered_specs = _setting_axis(setting_specs, include_factual=False)
    _apply_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(max(10, len(ordered_specs) * 2.0), 7.5), sharey=True)
    axes = axes.flatten()
    error_style = dict(elinewidth=1.1, capsize=4, capthick=1.1)

    for axis, category in zip(axes, _CONVERSION_CATEGORIES, strict=True):
        for index, payload in enumerate(metrics_payloads):
            y_values: list[float] = []
            y_errors: list[np.ndarray] = []
            for spec in ordered_specs:
                summary = (
                    ((payload.get("conversion_from_factual") or {}).get(spec.setting_id) or {}).get("overall") or {}
                ).get(category, {})
                value, error = _error_bars(summary, "ratio")
                y_values.append(value * 100.0)
                y_errors.append(error[:, 0] * 100.0)

            axis.errorbar(
                x_values,
                y_values,
                yerr=np.array(y_errors).T,
                fmt="-o",
                linewidth=2.0,
                markersize=6,
                color=_line_color(index, len(metrics_payloads)),
                label=payload["model_name"],
                **error_style,
            )

        axis.set_xticks(x_values)
        axis.set_xticklabels(tick_labels)
        axis.set_ylim(0, 110)
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Ratio (%)")
    axes[2].set_ylabel("Ratio (%)")
    axes[0].legend(frameon=True, edgecolor="black", framealpha=1.0)
    plt.tight_layout()
    output_path = _save_figure("conversion_ratios_over_proportion.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_theme_accuracy_curves(metrics_payloads: list[dict]) -> list[Path]:
    """Render one theme-level accuracy figure per model."""
    setting_specs = _ordered_setting_specs(metrics_payloads)
    if not setting_specs:
        return []

    x_values, tick_labels, ordered_specs = _setting_axis(setting_specs, include_factual=True)
    written_paths: list[Path] = []
    error_style = dict(elinewidth=1.1, capsize=4, capthick=1.1)

    for payload in metrics_payloads:
        theme_accuracy = payload.get("accuracy_by_theme") or {}
        if not theme_accuracy:
            continue

        _apply_plot_style()
        fig, ax = plt.subplots(figsize=(max(9, len(ordered_specs) * 1.6), 5.8))
        theme_names = sorted(theme_accuracy)

        for theme_index, theme_name in enumerate(theme_names):
            y_values: list[float] = []
            y_errors: list[np.ndarray] = []
            theme_summary = theme_accuracy.get(theme_name, {})
            for spec in ordered_specs:
                summary = theme_summary.get(spec.setting_id, {})
                value, error = _error_bars(summary, "accuracy")
                y_values.append(value * 100.0)
                y_errors.append(error[:, 0] * 100.0)

            ax.errorbar(
                x_values,
                y_values,
                yerr=np.array(y_errors).T,
                fmt="-o",
                linewidth=1.9,
                markersize=5.5,
                color=_series_color(theme_index, len(theme_names)),
                label=theme_name,
                **error_style,
            )

        ax.set_xlabel("Replacement Proportion")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x_values)
        ax.set_xticklabels(tick_labels)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=True, edgecolor="black", framealpha=1.0, ncol=2 if len(theme_names) > 6 else 1)
        plt.tight_layout()
        output_path = _save_figure(
            f"{sanitize_model_name(payload['model_name'])}_accuracy_by_theme_over_proportion.png"
        )
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        written_paths.append(output_path)

    return written_paths


def plot_factual_vs_fictional_variant_distributions(metrics_payloads: list[dict]) -> Path:
    """Plot factual EM against the distribution across the ten fictional benchmark variants."""
    distributions = _load_all_variant_accuracy_distributions(
        [str(payload.get("model_name") or "") for payload in metrics_payloads]
    )
    plot_rows = [
        distributions[model_name]
        for model_name in [str(payload.get("model_name") or "") for payload in metrics_payloads]
        if model_name in distributions
    ]

    if not plot_rows:
        return _plot_empty_state(
            "factual_vs_fictional_variant_accuracy_distribution.png",
            "Factual vs Fictional Variant Accuracy",
            "No model had a complete factual + 10-variant fictional evaluation set.",
        )

    _apply_plot_style()
    column_count = min(3, len(plot_rows))
    row_count = ceil(len(plot_rows) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(max(10, column_count * 4.3), max(4.6, row_count * 3.7)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    hist_face = "#dfeaf4"
    hist_edge = "#79bfff"
    kde_color = "#0b72d0"
    line_color = "#6e6e6e"

    for axis, row in zip(axes_flat, plot_rows, strict=False):
        fictional_values = np.asarray(row["variant_accuracies"], dtype=float)
        factual_accuracy = float(row["factual_accuracy"])
        lower = min(float(np.min(fictional_values)), factual_accuracy)
        upper = max(float(np.max(fictional_values)), factual_accuracy)
        spread = max(upper - lower, 1.0)
        margin = max(0.75, 0.12 * spread)
        bins = np.linspace(lower - margin, upper + margin, 7)
        counts, _, _ = axis.hist(
            fictional_values,
            bins=bins,
            color=hist_face,
            edgecolor=hist_edge,
            linewidth=1.1,
        )
        bin_width = float(bins[1] - bins[0])
        x_grid = np.linspace(bins[0], bins[-1], 300)
        kde_curve = _kernel_density_count_curve(fictional_values, x_grid, bin_width=bin_width)
        if kde_curve is not None:
            axis.plot(x_grid, kde_curve, color=kde_color, linewidth=2.0)

        axis.axvline(factual_accuracy, color=line_color, linestyle="--", linewidth=2.0)
        axis.set_xlabel("Fictional Accuracy (%) - EM")
        axis.grid(axis="y", linestyle="--", alpha=0.18)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(bins[0], bins[-1])
        axis.set_ylim(0, max(2.5, float(np.max(counts)) * 1.25 if counts.size else 2.5))
        axis.text(
            0.03,
            0.95,
            f"Factual {factual_accuracy:.1f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="black",
            fontsize=12,
        )
        axis.text(
            0.03,
            0.84,
            f"Fictional {np.mean(fictional_values):.1f} (\u00b1{np.std(fictional_values, ddof=0):.1f})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=kde_color,
            fontsize=12,
        )

    for axis_index, axis in enumerate(axes_flat):
        if axis_index >= len(plot_rows):
            axis.axis("off")
        elif axis_index % column_count == 0:
            axis.set_ylabel("Frequency")

    plt.tight_layout()
    output_path = _save_figure("factual_vs_fictional_variant_accuracy_distribution.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_question_type_variant_distributions(metrics_payloads: list[dict]) -> Path:
    """Plot factual-vs-fictional distributions split by question type for each model."""
    question_types = _ordered_question_types(metrics_payloads)
    distributions = _load_variant_accuracy_distributions_by_question_type(
        [str(payload.get("model_name") or "") for payload in metrics_payloads],
        question_types,
    )
    plot_rows = [
        distributions[model_name]
        for model_name in [str(payload.get("model_name") or "") for payload in metrics_payloads]
        if model_name in distributions
    ]

    if not plot_rows:
        return _plot_empty_state(
            "factual_vs_fictional_variant_accuracy_by_question_type_distribution.png",
            "Factual vs Fictional Variant Accuracy by Question Type",
            "No model had a complete factual + 10-variant fictional evaluation set.",
        )

    _apply_plot_style()
    column_count = min(3, len(plot_rows))
    row_count = ceil(len(plot_rows) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(max(12, column_count * 5.2), max(5.0, row_count * 4.3)),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for axis, row in zip(axes_flat, plot_rows, strict=False):
        _draw_question_type_variant_distribution_axis(
            axis,
            row=row,
            question_types=question_types,
            title=_display_model_name(str(row["model_name"])),
        )

    for axis_index, axis in enumerate(axes_flat):
        if axis_index >= len(plot_rows):
            axis.axis("off")
        elif axis_index % column_count == 0:
            axis.set_ylabel("Frequency")

    plt.tight_layout()
    output_path = _save_figure("factual_vs_fictional_variant_accuracy_by_question_type_distribution.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_question_type_variant_distribution_axis(
    axis,
    *,
    row: dict,
    question_types: tuple[str, ...],
    title: str,
) -> None:
    factual_by_type = row["factual_accuracy_by_question_type"]
    fictional_by_type = row["fictional_accuracies_by_question_type"]
    all_values: list[float] = []
    for question_type in question_types:
        all_values.extend(np.asarray(fictional_by_type[question_type], dtype=float).tolist())
        all_values.append(float(factual_by_type[question_type]))

    lower = min(all_values)
    upper = max(all_values)
    spread = max(upper - lower, 1.0)
    margin = max(0.8, 0.12 * spread)
    bins = np.linspace(lower - margin, upper + margin, 13)
    x_grid = np.linspace(bins[0], bins[-1], 400)
    bin_width = float(bins[1] - bins[0])
    max_count = 0.0

    for index, question_type in enumerate(question_types):
        color = _QUESTION_TYPE_COLORS.get(question_type, _series_color(index, len(question_types)))
        fictional_values = np.asarray(fictional_by_type[question_type], dtype=float)
        factual_accuracy = float(factual_by_type[question_type])
        counts, _, _ = axis.hist(
            fictional_values,
            bins=bins,
            color=color,
            edgecolor=color,
            linewidth=1.0,
            alpha=0.16,
        )
        max_count = max(max_count, float(np.max(counts)) if counts.size else 0.0)
        kde_curve = _kernel_density_count_curve(fictional_values, x_grid, bin_width=bin_width)
        if kde_curve is not None:
            axis.plot(x_grid, kde_curve, color=color, linewidth=2.0)
            max_count = max(max_count, float(np.max(kde_curve)))
        axis.axvline(factual_accuracy, color=color, linestyle="--", linewidth=2.0, alpha=0.95)
        axis.text(
            0.03,
            0.96 - index * 0.085,
            f"{question_type.title()} F {factual_accuracy:.1f} | Fi {np.mean(fictional_values):.1f} (\u00b1{np.std(fictional_values, ddof=0):.1f})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color=color,
            fontsize=10.5,
        )

    axis.set_xlabel("Accuracy (%) - EM")
    axis.grid(axis="y", linestyle="--", alpha=0.18)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlim(bins[0], bins[-1])
    axis.set_ylim(0, max(3.0, max_count * 1.22))


def plot_question_type_variant_distributions_by_model(metrics_payloads: list[dict]) -> list[Path]:
    """Render one question-type fictional-vs-factual distribution figure per model."""
    question_types = _ordered_question_types(metrics_payloads)
    distributions = _load_variant_accuracy_distributions_by_question_type(
        [str(payload.get("model_name") or "") for payload in metrics_payloads],
        question_types,
    )
    plot_rows = [
        distributions[model_name]
        for model_name in [str(payload.get("model_name") or "") for payload in metrics_payloads]
        if model_name in distributions
    ]

    written_paths: list[Path] = []
    for row in plot_rows:
        _apply_plot_style()
        fig, axis = plt.subplots(figsize=(8.2, 5.4))
        model_name = str(row["model_name"])
        _draw_question_type_variant_distribution_axis(
            axis,
            row=row,
            question_types=question_types,
            title=_display_model_name(model_name),
        )
        axis.set_ylabel("Frequency")
        plt.tight_layout()
        output_path = _save_figure(
            f"{sanitize_model_name(model_name)}_factual_vs_fictional_variant_accuracy_by_question_type_distribution.png"
        )
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        written_paths.append(output_path)

    return written_paths


def plot_question_type_answer_behavior_variant_distributions(metrics_payloads: list[dict]) -> Path:
    """Plot per-model, per-question-type fictional distributions split by answer behavior."""
    question_types = _ordered_question_types(metrics_payloads)
    answer_behaviors = ("variant", "invariant", "refusal")
    distributions = _load_variant_accuracy_distributions_by_question_type_and_answer_behavior(
        [str(payload.get("model_name") or "") for payload in metrics_payloads],
        question_types,
        answer_behaviors,
    )
    plot_rows = [
        distributions[model_name]
        for model_name in [str(payload.get("model_name") or "") for payload in metrics_payloads]
        if model_name in distributions
    ]

    if not plot_rows:
        return _plot_empty_state(
            "factual_vs_fictional_variant_accuracy_by_question_type_and_answer_behavior_distribution.png",
            "Factual vs Fictional Variant Accuracy by Question Type and Answer Behavior",
            "No model had a complete factual + 10-variant fictional evaluation set.",
        )

    _apply_plot_style()
    fig, axes = plt.subplots(
        len(plot_rows),
        len(question_types),
        figsize=(max(16, len(question_types) * 4.3), max(5.2, len(plot_rows) * 3.5)),
        squeeze=False,
    )
    legend_handles = _answer_behavior_legend_handles(answer_behaviors)

    for row_index, row in enumerate(plot_rows):
        model_name = _display_model_name(str(row["model_name"]))
        for col_index, question_type in enumerate(question_types):
            axis = axes[row_index][col_index]
            _draw_question_type_answer_behavior_variant_distribution_axis(
                axis,
                row=row,
                question_type=question_type,
                answer_behaviors=answer_behaviors,
            )

            if row_index == 0:
                axis.set_title(question_type.title())
            if row_index == len(plot_rows) - 1:
                axis.set_xlabel("Accuracy (%)")
            if col_index == 0:
                axis.set_ylabel(f"{model_name}\nCount")

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(legend_handles),
        frameon=False,
        fontsize=13,
        handlelength=2.1,
        handleheight=1.0,
        columnspacing=1.6,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    output_path = _save_figure(
        "factual_vs_fictional_variant_accuracy_by_question_type_and_answer_behavior_distribution.png"
    )
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _draw_question_type_answer_behavior_variant_distribution_axis(
    axis,
    *,
    row: dict,
    question_type: str,
    answer_behaviors: tuple[str, ...],
) -> None:
    factual_by_behavior = row["factual_accuracy_by_question_type_and_answer_behavior"][question_type]
    factual_count_by_behavior = row["factual_count_by_question_type_and_answer_behavior"][question_type]
    fictional_by_behavior = row["fictional_accuracies_by_question_type_and_answer_behavior"][question_type]

    all_values: list[float] = []
    for answer_behavior in answer_behaviors:
        values = np.asarray(fictional_by_behavior.get(answer_behavior, np.array([], dtype=float)), dtype=float)
        if values.size:
            all_values.extend(values.tolist())
        factual_value = factual_by_behavior.get(answer_behavior)
        if factual_value is not None:
            all_values.append(float(factual_value))

    if not all_values:
        axis.axis("off")
        return

    lower = min(all_values)
    upper = max(all_values)
    spread = max(upper - lower, 1.0)
    margin = max(0.8, 0.12 * spread)
    bins = np.linspace(lower - margin, upper + margin, _ANSWER_BEHAVIOR_DISTRIBUTION_BIN_COUNT + 1)
    x_grid = np.linspace(bins[0], bins[-1], 350)
    bin_width = float(bins[1] - bins[0])
    max_count = 0.0
    q_annotations: list[tuple[int, float, float, str]] = []

    for behavior_index, answer_behavior in enumerate(answer_behaviors):
        color = _ANSWER_BEHAVIOR_COLORS.get(answer_behavior, _series_color(behavior_index, len(answer_behaviors)))
        values = np.asarray(fictional_by_behavior.get(answer_behavior, np.array([], dtype=float)), dtype=float)
        factual_value = factual_by_behavior.get(answer_behavior)
        count_total = int(factual_count_by_behavior.get(answer_behavior, 0))
        if values.size:
            counts, _, _ = axis.hist(
                values,
                bins=bins,
                color=color,
                edgecolor=color,
                linewidth=0.9,
                alpha=0.14,
            )
            max_count = max(max_count, float(np.max(counts)) if counts.size else 0.0)
            kde_curve = _kernel_density_count_curve(values, x_grid, bin_width=bin_width)
            if kde_curve is not None:
                axis.plot(x_grid, kde_curve, color=color, linewidth=1.9)
                max_count = max(max_count, float(np.max(kde_curve)))
        if factual_value is not None and count_total > 0:
            factual_float = float(factual_value)
            axis.axvline(factual_float, color=color, linestyle="--", linewidth=1.8, alpha=0.95)
            q_value = _empirical_q_value(values, factual_float) if values.size else None
            if q_value is not None:
                q_annotations.append((behavior_index, factual_float, q_value, color))

    y_upper = max(2.5, max_count * 1.22)
    axis.grid(axis="y", linestyle="--", alpha=0.18)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_xlim(bins[0], bins[-1])
    axis.set_ylim(0, y_upper)

    for behavior_index, factual_float, q_value, color in q_annotations:
        vertical_anchor = 0.95 - behavior_index * 0.09
        axis.annotate(
            _format_percentile_label(q_value),
            xy=(factual_float, y_upper * vertical_anchor),
            xytext=(4 if behavior_index % 2 == 0 else -4, 0),
            textcoords="offset points",
            rotation=90,
            ha="left" if behavior_index % 2 == 0 else "right",
            va="top",
            color=color,
            fontsize=8.8,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.2,
            },
            clip_on=False,
        )


def plot_question_type_answer_behavior_variant_distributions_by_model(metrics_payloads: list[dict]) -> list[Path]:
    """Render one question-type/answer-behavior fictional-vs-factual distribution figure per model."""
    question_types = _ordered_question_types(metrics_payloads)
    answer_behaviors = ("variant", "invariant", "refusal")
    distributions = _load_variant_accuracy_distributions_by_question_type_and_answer_behavior(
        [str(payload.get("model_name") or "") for payload in metrics_payloads],
        question_types,
        answer_behaviors,
    )
    written_paths: list[Path] = []
    legend_handles = _answer_behavior_legend_handles(answer_behaviors)
    ordered_model_names = [str(payload.get("model_name") or "") for payload in metrics_payloads]
    for model_name in ordered_model_names:
        row = distributions.get(model_name)
        output_filename = (
            f"{sanitize_model_name(model_name)}_factual_vs_fictional_variant_accuracy_by_question_type_and_answer_behavior_distribution.png"
        )
        if row is None:
            written_paths.append(
                _plot_empty_state(
                    output_filename,
                    f"{_display_model_name(model_name)}: Factual vs Fictional-Variant EM",
                    "No complete factual + 10-variant fictional evaluation set was available for this model.",
                )
            )
            continue

        _apply_plot_style()
        fig, axes = plt.subplots(
            1,
            len(question_types),
            figsize=(max(17, len(question_types) * 4.6), 4.8),
            squeeze=False,
        )

        for col_index, question_type in enumerate(question_types):
            axis = axes[0][col_index]
            _draw_question_type_answer_behavior_variant_distribution_axis(
                axis,
                row=row,
                question_type=question_type,
                answer_behaviors=answer_behaviors,
            )
            axis.set_title(question_type.title(), pad=14)
            axis.set_xlabel("Accuracy (%)")
            if col_index == 0:
                axis.set_ylabel("Count")

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=len(legend_handles),
            frameon=False,
            fontsize=13,
            handlelength=2.1,
            handleheight=1.0,
            columnspacing=1.6,
        )
        plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
        output_path = _save_figure(output_filename)
        fig.savefig(output_path, dpi=250, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close(fig)
        written_paths.append(output_path)

    return written_paths
