"""Frozen scientific routines used by the submitted result release."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np

ATOMIC_QUESTION_TYPES = ("arithmetic", "temporal", "inference", "extractive")

ANSWER_BEHAVIORS = ("variant", "invariant", "refusal")

def validate_and_index(
    factual_rows: list[dict[str, Any]],
    fictional_rows: list[dict[str, Any]],
    *,
    expected_documents: int,
    expected_variants: int,
) -> tuple[
    tuple[tuple[str, str], ...],
    dict[tuple[str, str, str, str], float],
    dict[tuple[str, str, str, str], float],
]:
    factual: dict[tuple[str, str, str, str], list[bool]] = defaultdict(list)
    fictional: dict[tuple[str, str, str, str], list[bool]] = defaultdict(list)
    for row in factual_rows:
        document = (str(row["document_theme"]), str(row["document_id"]))
        question_type = str(row["question_type"])
        answer_behavior = str(row["answer_behavior"])
        if question_type not in ATOMIC_QUESTION_TYPES or answer_behavior not in ANSWER_BEHAVIORS:
            raise ValueError(f"Unexpected factual cell: {question_type}/{answer_behavior}")
        factual[(*document, question_type, answer_behavior)].append(bool(row["new_final_is_correct"]))
    for row in fictional_rows:
        document = (str(row["document_theme"]), str(row["document_id"]))
        question_type = str(row["question_type"])
        answer_behavior = str(row["answer_behavior"])
        if question_type not in ATOMIC_QUESTION_TYPES or answer_behavior not in ANSWER_BEHAVIORS:
            raise ValueError(f"Unexpected fictional cell: {question_type}/{answer_behavior}")
        fictional[(*document, question_type, answer_behavior)].append(bool(row["new_final_is_correct"]))

    documents = tuple(sorted({(key[0], key[1]) for key in factual}))
    if len(documents) != expected_documents:
        raise ValueError(f"Expected {expected_documents} documents, found {len(documents)}")
    factual_means: dict[tuple[str, str, str, str], float] = {}
    fictional_means: dict[tuple[str, str, str, str], float] = {}
    for document in documents:
        for question_type in ATOMIC_QUESTION_TYPES:
            for answer_behavior in ANSWER_BEHAVIORS:
                key = (*document, question_type, answer_behavior)
                factual_values = factual.get(key, [])
                fictional_values = fictional.get(key, [])
                if len(factual_values) != 1:
                    raise ValueError(f"{key}: expected one factual row, found {len(factual_values)}")
                if len(fictional_values) != expected_variants:
                    raise ValueError(
                        f"{key}: expected {expected_variants} fictional rows, found {len(fictional_values)}"
                    )
                factual_means[key] = float(np.mean(factual_values))
                fictional_means[key] = float(np.mean(fictional_values))
    return documents, factual_means, fictional_means

def cell_values(
    documents: tuple[tuple[str, str], ...],
    factual: dict[tuple[str, str, str, str], float],
    fictional: dict[tuple[str, str, str, str], float],
    *,
    question_type: str,
    answer_behavior: str,
) -> tuple[np.ndarray, np.ndarray]:
    atomic_types = ATOMIC_QUESTION_TYPES[:3] if question_type == "reasoning" else (question_type,)
    factual_values = np.array(
        [
            np.mean([factual[(*document, atomic_type, answer_behavior)] for atomic_type in atomic_types])
            for document in documents
        ],
        dtype=np.float64,
    )
    fictional_values = np.array(
        [
            np.mean([fictional[(*document, atomic_type, answer_behavior)] for atomic_type in atomic_types])
            for document in documents
        ],
        dtype=np.float64,
    )
    return factual_values, fictional_values

def bootstrap_interval(
    document_contrasts: np.ndarray,
    *,
    resamples: int,
    seed: int,
    chunk_size: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    remaining = resamples
    while remaining:
        current = min(chunk_size, remaining)
        indices = rng.integers(0, len(document_contrasts), size=(current, len(document_contrasts)))
        draws.append(document_contrasts[indices].mean(axis=1) * 100.0)
        remaining -= current
    values = np.concatenate(draws)
    lower, upper = np.quantile(values, (0.025, 0.975))
    return float(lower), float(upper)
