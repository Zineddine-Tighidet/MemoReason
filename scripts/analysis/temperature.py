"""Frozen paired source-document temperature bootstrap from compact file scores."""
from __future__ import annotations
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
import numpy as np

CONDITIONS: tuple[tuple[str, float, int], ...] = (
    ("t00_s23", 0.0, 23),
    ("t05_s23", 0.5, 23),
    ("t05_s24", 0.5, 24),
    ("t05_s25", 0.5, 25),
    ("t10_s23", 1.0, 23),
    ("t10_s24", 1.0, 24),
    ("t10_s25", 1.0, 25),
)

MODELS = ("olmo-3-7b-instruct", "qwen3.5-27b")

EXPECTED_QUESTIONS_PER_FILE = 12

EXPECTED_VARIANTS = {
    "factual": frozenset({"v01"}),
    "fictional": frozenset(f"v{index:02d}" for index in range(1, 11)),
}

EXPECTED_SEEDS = {
    0.0: frozenset({23}),
    0.5: frozenset({23, 24, 25}),
    1.0: frozenset({23, 24, 25}),
}

def percentile(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)

def compute(records: list[dict[str, Any]], *, samples: int, seed: int) -> list[dict[str, Any]]:
    if len(records) != 15400:
        raise ValueError("Expected 15,400 compact temperature records")
    seen = set()
    documents_by_group = []
    for group_id in range(14):
        selected = [row for row in records if row["group_id"] == group_id]
        condition, value, source_seed = CONDITIONS[group_id // 2]
        model = MODELS[group_id % 2]
        docs = set()
        for row in selected:
            key = (group_id, row["document"], row["setting"], row["variant_id"])
            if key in seen or row["total"] != 12 or not 0 <= row["final_correct"] <= 12:
                raise ValueError("Duplicate or invalid temperature record")
            seen.add(key)
            if (row["condition"], row["temperature"], row["seed"], row["model"]) != (condition, value, source_seed, model):
                raise ValueError("Temperature group contract differs")
            docs.add(row["document"])
        if len(selected) != 1100 or len(docs) != 100:
            raise ValueError("Incomplete temperature group")
        documents_by_group.append(docs)
    if any(docs != documents_by_group[0] for docs in documents_by_group):
        raise ValueError("Mismatched document inventories")
    args = SimpleNamespace(resamples=samples, seed=seed, chunk_size=5000)
    temperature = SimpleNamespace(CONDITIONS=CONDITIONS, MODELS=MODELS,
        EXPECTED_QUESTIONS_PER_FILE=EXPECTED_QUESTIONS_PER_FILE,
        EXPECTED_VARIANTS=EXPECTED_VARIANTS, EXPECTED_SEEDS=EXPECTED_SEEDS)
    documents = sorted(documents_by_group[0])
    aggregates: dict[tuple[str, float, str, str], list[int]] = defaultdict(lambda: [0, 0])
    seed_aggregates: dict[tuple[str, float, int, str], list[int]] = defaultdict(lambda: [0, 0])
    variants: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    seeds: dict[tuple[str, float], set[int]] = defaultdict(set)
    for row in records:
        model = str(row["model"])
        value = float(row["temperature"])
        seed = int(row["seed"])
        document = str(row["document"])
        setting = str(row["setting"])
        condition = str(row["condition"])
        seeds[(model, value)].add(seed)
        variants[(condition, model, document, setting)].add(str(row["variant_id"]))
        bucket = aggregates[(model, value, document, setting)]
        bucket[0] += int(row["final_correct"])
        bucket[1] += int(row["total"])
        seed_bucket = seed_aggregates[(model, value, seed, setting)]
        seed_bucket[0] += int(row["final_correct"])
        seed_bucket[1] += int(row["total"])

    for model in temperature.MODELS:
        for value in (0.0, 0.5, 1.0):
            if frozenset(seeds[(model, value)]) != temperature.EXPECTED_SEEDS[value]:
                raise ValueError(f"seed lattice differs: {model}/T={value}")
            for document in documents:
                for setting, expected_variants in temperature.EXPECTED_VARIANTS.items():
                    expected_total = (
                        len(temperature.EXPECTED_SEEDS[value])
                        * len(expected_variants)
                        * temperature.EXPECTED_QUESTIONS_PER_FILE
                    )
                    if aggregates[(model, value, document, setting)][1] != expected_total:
                        raise ValueError(f"within-document denominator differs: {model}/T={value}/{document}")
    for condition, _value, _seed in temperature.CONDITIONS:
        for model in temperature.MODELS:
            for document in documents:
                for setting, expected in temperature.EXPECTED_VARIANTS.items():
                    if frozenset(variants[(condition, model, document, setting)]) != expected:
                        raise ValueError(f"variant lattice differs: {condition}/{model}/{document}/{setting}")

    arrays: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for model in temperature.MODELS:
        for value in (0.0, 0.5, 1.0):
            factual = np.asarray(
                [aggregates[(model, value, doc, "factual")][0] / aggregates[(model, value, doc, "factual")][1] for doc in documents]
            )
            fictional = np.asarray(
                [aggregates[(model, value, doc, "fictional")][0] / aggregates[(model, value, doc, "fictional")][1] for doc in documents]
            )
            arrays[(model, value)] = factual, fictional, factual - fictional

    rng = np.random.default_rng(args.seed)
    draws = {key: np.empty((3, args.resamples), dtype=np.float64) for key in arrays}
    for start in range(0, args.resamples, args.chunk_size):
        stop = min(start + args.chunk_size, args.resamples)
        indices = rng.integers(0, len(documents), size=(stop - start, len(documents)))
        for key, values in arrays.items():
            for metric_index, metric_values in enumerate(values):
                draws[key][metric_index, start:stop] = metric_values[indices].mean(axis=1)

    results: list[dict[str, Any]] = []
    for model in temperature.MODELS:
        for value in (0.0, 0.5, 1.0):
            values = arrays[(model, value)]
            cell_draws = draws[(model, value)]
            row: dict[str, Any] = {
                "model": model,
                "temperature": value,
                "n_documents": len(documents),
                "n_seeds": len(temperature.EXPECTED_SEEDS[value]),
                "bootstrap_resamples": args.resamples,
            }
            for index, metric in enumerate(("factual_accuracy", "fictional_accuracy", "factual_minus_fictional_drop")):
                low, high = percentile(cell_draws[index])
                row[f"{metric}_pp"] = 100.0 * float(values[index].mean())
                row[f"{metric}_ci95_low_pp"] = 100.0 * low
                row[f"{metric}_ci95_high_pp"] = 100.0 * high
            seed_values = []
            for seed in sorted(temperature.EXPECTED_SEEDS[value]):
                fc, ft = seed_aggregates[(model, value, seed, "factual")]
                xc, xt = seed_aggregates[(model, value, seed, "fictional")]
                seed_values.append((fc / ft, xc / xt, fc / ft - xc / xt))
            seed_array = np.asarray(seed_values)
            for index, metric in enumerate(("factual_accuracy", "fictional_accuracy", "factual_minus_fictional_drop")):
                row[f"{metric}_across_seed_sd_pp"] = (
                    100.0 * float(seed_array[:, index].std(ddof=1)) if len(seed_values) > 1 else None
                )
            contrast = values[2] - arrays[(model, 0.0)][2]
            contrast_draws = cell_draws[2] - draws[(model, 0.0)][2]
            low, high = percentile(contrast_draws)
            row["drop_change_vs_temperature_0_pp"] = 100.0 * float(contrast.mean())
            row["drop_change_vs_temperature_0_ci95_low_pp"] = 100.0 * low
            row["drop_change_vs_temperature_0_ci95_high_pp"] = 100.0 * high
            results.append(row)

    return results
