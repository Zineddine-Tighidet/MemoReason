"""Frozen scientific routines used by the submitted result release."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np

MODEL_ORDER = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "claude-sonnet-4-6",
    "llama-3.1-8b-instruct",
)

DISPLAY = {
    "olmo-3-7b-think": "OLMO-3 7B-THINK",
    "olmo-3-7b-instruct": "OLMO-3 7B-INSTRUCT",
    "gpt-oss-20b-groq": "GPT-OSS 20B",
    "gemma-4-26b-a4b-it": "GEMMA-4 26B-A4B-IT",
    "gpt-oss-120b-groq": "GPT-OSS 120B",
    "qwen3.5-27b": "QWEN3.5 27B",
    "qwen3.5-35b-a3b": "QWEN3.5 35B-A3B",
    "claude-sonnet-4-6": "CLAUDE SONNET 4.6",
    "llama-3.1-8b-instruct": "LLAMA-3.1 8B-INSTRUCT",
}

def compute_model(
    model: str,
    rows: list[dict[str, Any]],
    *,
    draws: np.ndarray,
    clusters: tuple[tuple[str, str], ...],
    question_scope: str = "all_12_questions",
    expected_factual_questions: int = 1_200,
) -> dict[str, Any]:
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    factual: dict[str, tuple[bool, tuple[str, str]]] = {}
    fictional: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        cluster = (str(row["document_theme"]), str(row["document_id"]))
        if cluster not in cluster_index:
            raise ValueError(f"{model}: unexpected cluster {cluster}")
        pair_key = str(row["pair_key"])
        correct = bool(row["new_final_is_correct"])
        if row["document_setting"] == "factual":
            if pair_key in factual:
                raise ValueError(f"{model}: duplicate factual pair {pair_key}")
            factual[pair_key] = (correct, cluster)
        else:
            variant = str(row["document_variant_id"])
            if variant in fictional[pair_key]:
                raise ValueError(f"{model}: duplicate fictional pair {pair_key}/{variant}")
            fictional[pair_key][variant] = correct
    if len(factual) != expected_factual_questions:
        raise ValueError(
            f"{model}: expected {expected_factual_questions:,} factual questions, found {len(factual)}"
        )

    n11 = np.zeros(len(clusters), dtype=np.int64)
    n10 = np.zeros(len(clusters), dtype=np.int64)
    n01 = np.zeros(len(clusters), dtype=np.int64)
    n00 = np.zeros(len(clusters), dtype=np.int64)
    expected_variants = {f"v{index:02d}" for index in range(1, 11)}
    for pair_key, (factual_correct, cluster) in factual.items():
        variants = fictional.get(pair_key, {})
        if set(variants) != expected_variants:
            raise ValueError(f"{model}: incomplete fictional variants for {pair_key}")
        index = cluster_index[cluster]
        for fictional_correct in variants.values():
            if factual_correct and fictional_correct:
                n11[index] += 1
            elif factual_correct:
                n10[index] += 1
            elif fictional_correct:
                n01[index] += 1
            else:
                n00[index] += 1

    forward_den = n10 + n11
    mirror_den = n01 + n11
    forward_point = float(n10.sum() / forward_den.sum())
    mirror_point = float(n01.sum() / mirror_den.sum())
    forward_dist = n10[draws].sum(axis=1) / forward_den[draws].sum(axis=1)
    mirror_dist = n01[draws].sum(axis=1) / mirror_den[draws].sum(axis=1)
    difference_dist = forward_dist - mirror_dist

    def metric(point: float, distribution: np.ndarray) -> dict[str, float]:
        low, high = np.percentile(distribution, [2.5, 97.5])
        return {
            "point_percent": 100.0 * point,
            "ci_low_percent": 100.0 * float(low),
            "ci_high_percent": 100.0 * float(high),
        }

    return {
        "model": model,
        "display_name": DISPLAY[model],
        "question_scope": question_scope,
        "forward": metric(forward_point, forward_dist),
        "mirror": metric(mirror_point, mirror_dist),
        "difference": metric(forward_point - mirror_point, difference_dist),
        "transitions": {"n11": int(n11.sum()), "n10": int(n10.sum()), "n01": int(n01.sum()), "n00": int(n00.sum())},
    }
