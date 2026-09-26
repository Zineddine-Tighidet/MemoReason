"""Frozen document-frequency statistics; scientific routines retain release calculations."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
from typing import Any
import numpy as np

def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()

def _document_rows(
    question_rows: list[dict[str, Any]],
    frequencies: dict[tuple[str, str], dict[str, float]],
    *,
    require_complete_frequency_coverage: bool = True,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in question_rows:
        grouped[(row["document_theme"], row["document_id"])].append(row)
    if require_complete_frequency_coverage and set(grouped) != set(frequencies):
        missing_frequency = sorted(set(grouped) - set(frequencies))
        missing_scores = sorted(set(frequencies) - set(grouped))
        raise ValueError(
            f"document-key mismatch: missing_frequency={missing_frequency}, missing_scores={missing_scores}"
        )
    if not set(frequencies).issubset(grouped):
        missing_scores = sorted(set(frequencies) - set(grouped))
        raise ValueError(f"frequency rows without matching scores: {missing_scores}")

    output: list[dict[str, Any]] = []
    for key in sorted(frequencies):
        rows = grouped[key]
        factual_successes = sum(int(row["factual_correct"]) for row in rows)
        fictional_successes = sum(int(row["fictional_successes"]) for row in rows)
        fictional_total = sum(int(row["fictional_total"]) for row in rows)
        factual_total = len(rows)
        if factual_total != 12 or fictional_total != 120:
            raise ValueError(f"unexpected within-document denominators for {key}: {factual_total}/{fictional_total}")
        factual_accuracy = factual_successes / factual_total
        fictional_accuracy = fictional_successes / fictional_total
        frequency = frequencies[key]
        mean_count = frequency["mean_named_entity_count"]
        output.append(
            {
                "document_theme": key[0],
                "document_id": key[1],
                "factual_successes": factual_successes,
                "factual_total": factual_total,
                "fictional_successes": fictional_successes,
                "fictional_total": fictional_total,
                "factual_accuracy_percent": 100.0 * factual_accuracy,
                "fictional_accuracy_percent": 100.0 * fictional_accuracy,
                "performance_drop_pp": 100.0 * (factual_accuracy - fictional_accuracy),
                **frequency,
                "log10_1p_mean_named_entity_count": math.log10(1.0 + mean_count),
            }
        )
    return output

def _rowwise_pearson(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=1, keepdims=True)
    y_centered = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)

def _bootstrap_distributions(
    documents: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 5000,
) -> tuple[dict[str, np.ndarray], str]:
    from scipy.stats import rankdata

    n_documents = len(documents)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_documents, size=(replicates, n_documents), dtype=np.int16)
    draws_sha256 = _sha256_bytes(draws.tobytes())

    factual_successes = np.asarray([row["factual_successes"] for row in documents], dtype=float)
    factual_total = np.asarray([row["factual_total"] for row in documents], dtype=float)
    fictional_successes = np.asarray([row["fictional_successes"] for row in documents], dtype=float)
    fictional_total = np.asarray([row["fictional_total"] for row in documents], dtype=float)
    x = np.asarray([row["log10_1p_mean_named_entity_count"] for row in documents], dtype=float)
    y = np.asarray([row["performance_drop_pp"] for row in documents], dtype=float)

    distributions = {
        name: np.empty(replicates, dtype=np.float64)
        for name in (
            "factual_accuracy_percent",
            "fictional_accuracy_percent",
            "performance_drop_pp",
            "pearson_r",
            "spearman_r",
            "ols_slope_pp_per_log10_decade",
        )
    }

    for start in range(0, replicates, chunk_size):
        stop = min(replicates, start + chunk_size)
        sample = draws[start:stop]
        factual = factual_successes[sample].sum(axis=1) / factual_total[sample].sum(axis=1)
        fictional = fictional_successes[sample].sum(axis=1) / fictional_total[sample].sum(axis=1)
        x_sample = x[sample]
        y_sample = y[sample]
        x_centered = x_sample - np.mean(x_sample, axis=1, keepdims=True)
        y_centered = y_sample - np.mean(y_sample, axis=1, keepdims=True)
        ss_x = np.sum(x_centered**2, axis=1)
        covariance = np.sum(x_centered * y_centered, axis=1)

        distributions["factual_accuracy_percent"][start:stop] = 100.0 * factual
        distributions["fictional_accuracy_percent"][start:stop] = 100.0 * fictional
        distributions["performance_drop_pp"][start:stop] = 100.0 * (factual - fictional)
        distributions["pearson_r"][start:stop] = _rowwise_pearson(x_sample, y_sample)
        distributions["ols_slope_pp_per_log10_decade"][start:stop] = np.divide(
            covariance,
            ss_x,
            out=np.full_like(covariance, np.nan),
            where=ss_x > 0,
        )
        distributions["spearman_r"][start:stop] = _rowwise_pearson(
            rankdata(x_sample, axis=1),
            rankdata(y_sample, axis=1),
        )

    return distributions, draws_sha256

def _interval(distribution: np.ndarray) -> tuple[float, float]:
    finite = distribution[np.isfinite(distribution)]
    if finite.size != distribution.size:
        raise ValueError(f"non-finite bootstrap replicates: {distribution.size - finite.size}")
    low, high = np.percentile(finite, [2.5, 97.5], method="linear")
    return float(low), float(high)

def _point_metrics(documents: list[dict[str, Any]]) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    factual_successes = sum(int(row["factual_successes"]) for row in documents)
    factual_total = sum(int(row["factual_total"]) for row in documents)
    fictional_successes = sum(int(row["fictional_successes"]) for row in documents)
    fictional_total = sum(int(row["fictional_total"]) for row in documents)
    x = np.asarray([row["log10_1p_mean_named_entity_count"] for row in documents], dtype=float)
    y = np.asarray([row["performance_drop_pp"] for row in documents], dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return {
        "factual_accuracy_percent": 100.0 * factual_successes / factual_total,
        "fictional_accuracy_percent": 100.0 * fictional_successes / fictional_total,
        "performance_drop_pp": 100.0 * (
            factual_successes / factual_total - fictional_successes / fictional_total
        ),
        "pearson_r": float(pearsonr(x, y).statistic),
        "spearman_r": float(spearmanr(x, y).statistic),
        "ols_slope_pp_per_log10_decade": slope,
    }

def _metric_rows(
    points: dict[str, float], distributions: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    labels = {
        "factual_accuracy_percent": "Factual accuracy",
        "fictional_accuracy_percent": "Fully fictional accuracy",
        "performance_drop_pp": "Factual-to-fictional performance drop",
        "pearson_r": "Pearson r: log10(1+mean named-entity frequency) vs document drop",
        "spearman_r": "Spearman rho: mean named-entity frequency vs document drop",
        "ols_slope_pp_per_log10_decade": "OLS slope: drop pp per 10x mean named-entity frequency",
    }
    units = {
        "factual_accuracy_percent": "percent",
        "fictional_accuracy_percent": "percent",
        "performance_drop_pp": "percentage_points",
        "pearson_r": "correlation",
        "spearman_r": "correlation",
        "ols_slope_pp_per_log10_decade": "percentage_points_per_log10_decade",
    }
    rows = []
    for name in labels:
        low, high = _interval(distributions[name])
        rows.append(
            {
                "metric": name,
                "label": labels[name],
                "unit": units[name],
                "point": points[name],
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return rows

def _quartile_rows(documents: list[dict[str, Any]], draws_seed: int, replicates: int) -> list[dict[str, Any]]:
    x = np.asarray([row["mean_named_entity_count"] for row in documents], dtype=float)
    y = np.asarray([row["performance_drop_pp"] for row in documents], dtype=float)
    order = np.argsort(x, kind="stable")
    groups = np.empty(len(documents), dtype=np.int8)
    for group_index, indices in enumerate(np.array_split(order, 4), start=1):
        groups[indices] = group_index

    rng = np.random.default_rng(draws_seed)
    draws = rng.integers(0, len(documents), size=(replicates, len(documents)), dtype=np.int16)
    rows: list[dict[str, Any]] = []
    for group_index in range(1, 5):
        mask = groups == group_index
        sampled_in_group = mask[draws]
        sampled_y = y[draws]
        numerator = np.sum(sampled_y * sampled_in_group, axis=1)
        denominator = np.sum(sampled_in_group, axis=1)
        if np.any(denominator == 0):
            raise ValueError(f"bootstrap replicate without frequency-quartile {group_index} document")
        distribution = numerator / denominator
        low, high = _interval(distribution)
        rows.append(
            {
                "frequency_quartile": group_index,
                "documents": int(mask.sum()),
                "mean_frequency_min": float(np.min(x[mask])),
                "mean_frequency_max": float(np.max(x[mask])),
                "mean_frequency": float(np.mean(x[mask])),
                "mean_performance_drop_pp": float(np.mean(y[mask])),
                "ci95_low_pp": low,
                "ci95_high_pp": high,
            }
        )
    return rows
