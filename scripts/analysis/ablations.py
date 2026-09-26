"""Frozen scientific routines used by the submitted result release."""
from __future__ import annotations
from collections import defaultdict
from typing import Any
import hashlib
import numpy as np

CONTRACT = "dataset100_reconciled_rebuttal_ablation_release_v1"

MODELS = ("olmo-3-7b-instruct", "olmo-3-7b-think", "qwen3.5-35b-a3b")

ATOMIC_TYPES = ("arithmetic", "temporal", "inference", "extractive")

REASONING_TYPES = ("arithmetic", "temporal", "inference")

SETTINGS = ("factual", "fictional_numtemp", "fictional_named", "fictional")

def factual_answers(endpoint: dict[str, list[dict[str, Any]]]) -> dict[str, set[str]]:
    observed: dict[str, set[frozenset[str]]] = defaultdict(set)
    for rows in endpoint.values():
        for row in rows:
            if row["document_setting"] != "factual" or row.get("answer_behavior") != "variant":
                continue
            accepted = frozenset(
                str(value).strip()
                for value in row.get("new_accepted_answers_canonical") or []
                if str(value).strip()
            )
            fallback = str(row.get("new_ground_truth_canonical") or "").strip()
            if not accepted and fallback:
                accepted = frozenset({fallback})
            if accepted:
                observed[str(row["pair_key"])].add(accepted)
    if len(observed) != 400 or any(len(values) != 1 for values in observed.values()):
        raise ValueError("Variant factual reference sets are incomplete or inconsistent")
    return {key: set(next(iter(values))) for key, values in observed.items()}

def compute_numtemp_shortcut(
    numtemp: dict[str, list[dict[str, Any]]],
    references: dict[str, set[str]],
    *,
    models: tuple[str, ...] = MODELS,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in models:
        by_question: dict[str, dict[str, Any]] = {}
        variants: dict[str, set[str]] = defaultdict(set)
        for row in numtemp[model]:
            if row["document_setting"] != "fictional_numtemp" or row.get("answer_behavior") != "variant":
                continue
            pair = str(row["pair_key"])
            question_type = str(row["question_type"])
            cell = by_question.setdefault(pair, {"question_type": question_type, "failures": 0, "copies": 0})
            if cell["question_type"] != question_type:
                raise ValueError(f"{model}/{pair}: question-type drift")
            variants[pair].add(str(row["document_variant_id"]))
            if row["new_final_is_correct"]:
                continue
            cell["failures"] += 1
            prediction = str(row.get("new_parsed_output_canonical") or "").strip()
            if prediction and prediction in references[pair]:
                cell["copies"] += 1
        if len(by_question) != 400 or any(len(values) != 10 for values in variants.values()):
            raise ValueError(f"{model}: incomplete numtemp variant-question inventory")
        groups = [*((name, {name}) for name in ATOMIC_TYPES), ("reasoning", set(REASONING_TYPES))]
        for label, types in groups:
            selected = [cell for cell in by_question.values() if cell["question_type"] in types]
            if len(selected) != 100 * len(types):
                raise ValueError(f"{model}/{label}: incomplete question scope")
            failures = sum(cell["failures"] for cell in selected)
            copies = sum(cell["copies"] for cell in selected)
            shortcut = 100.0 * sum(
                cell["copies"] / cell["failures"] if cell["failures"] else 0.0
                for cell in selected
            ) / len(selected)
            output.append(
                {
                    "model_name": model,
                    "question_type": label,
                    "base_question_count": len(selected),
                    "questions_with_failures": sum(cell["failures"] > 0 for cell in selected),
                    "failed_variant_count": failures,
                    "factual_answer_copy_count": copies,
                    "shortcut_rate_percent": shortcut,
                    "pooled_copy_rate_percent_diagnostic": 100.0 * copies / failures if failures else 0.0,
                }
            )
    return output

def invariant_document_matrix(
    endpoint: dict[str, list[dict[str, Any]]],
    numtemp: dict[str, list[dict[str, Any]]],
    named: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], np.ndarray]:
    source_for = {
        "factual": endpoint,
        "fictional": endpoint,
        "fictional_numtemp": numtemp,
        "fictional_named": named,
    }
    scores: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        model: {setting: defaultdict(lambda: defaultdict(list)) for setting in SETTINGS}
        for model in MODELS
    }
    taxonomy: dict[str, tuple[str, str]] = {}
    for setting in SETTINGS:
        for model in MODELS:
            for row in source_for[setting][model]:
                if row["document_setting"] != setting or row.get("answer_behavior") != "invariant":
                    continue
                pair = str(row["pair_key"])
                question_type = str(row["question_type"])
                if question_type not in ATOMIC_TYPES:
                    raise ValueError(f"{model}/{pair}: invalid invariant question type")
                prior = taxonomy.setdefault(pair, (question_type, "invariant"))
                if prior != (question_type, "invariant"):
                    raise ValueError(f"{pair}: taxonomy drift")
                document = f"{row['document_theme']}::{row['document_id']}"
                scores[model][setting][document][pair].append(float(bool(row["new_final_is_correct"])))
    documents = sorted(scores[MODELS[0]]["factual"])
    if len(documents) != 100:
        raise ValueError(f"Expected 100 documents, found {len(documents)}")
    matrix = np.empty((len(MODELS), len(documents), len(SETTINGS)), dtype=float)
    for model_index, model in enumerate(MODELS):
        for setting_index, setting in enumerate(SETTINGS):
            if sorted(scores[model][setting]) != documents:
                raise ValueError(f"{model}/{setting}: document scope differs")
            for document_index, document in enumerate(documents):
                pairs = scores[model][setting][document]
                if len(pairs) != 4 or {taxonomy[pair][0] for pair in pairs} != set(ATOMIC_TYPES):
                    raise ValueError(f"{model}/{setting}/{document}: incomplete invariant 4-type grid")
                expected = 1 if setting == "factual" else 10
                if any(len(values) != expected for values in pairs.values()):
                    raise ValueError(f"{model}/{setting}/{document}: invalid variant count")
                matrix[model_index, document_index, setting_index] = float(
                    np.mean([np.mean(values) for values in pairs.values()])
                )
    return documents, matrix

def compute_four_setting(
    documents: list[str],
    matrix: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    points = matrix.mean(axis=1)
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(MODELS), len(SETTINGS)), dtype=float)
    digest = hashlib.sha256()
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(0, len(documents), size=(stop - start, len(documents)), endpoint=False)
        digest.update(np.asarray(indices, dtype="<i8").tobytes(order="C"))
        draws[start:stop] = matrix[:, indices, :].mean(axis=2).transpose(1, 0, 2)

    by_model: dict[str, Any] = {}
    factual_index = SETTINGS.index("factual")
    for model_index, model in enumerate(MODELS):
        settings: dict[str, Any] = {}
        contrasts: dict[str, Any] = {}
        for setting_index, setting in enumerate(SETTINGS):
            lower, upper = np.quantile(draws[:, model_index, setting_index], (0.025, 0.975), method="linear")
            settings[setting] = {
                "percent": 100.0 * float(points[model_index, setting_index]),
                "cluster_bootstrap_95ci_percent": [100.0 * float(lower), 100.0 * float(upper)],
                "total_rows": 400 if setting == "factual" else 4_000,
            }
            if setting == "factual":
                continue
            distribution = draws[:, model_index, setting_index] - draws[:, model_index, factual_index]
            low, high = np.quantile(distribution, (0.025, 0.975), method="linear")
            contrasts[f"{setting}_minus_factual"] = {
                "percentage_points": 100.0 * float(
                    points[model_index, setting_index] - points[model_index, factual_index]
                ),
                "cluster_bootstrap_95ci_percentage_points": [100.0 * float(low), 100.0 * float(high)],
                "ci_excludes_zero": bool(low > 0 or high < 0),
            }
        by_model[model] = {"settings": settings, "contrasts": contrasts}
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "complete",
        "analysis": "four-setting invariant-answer accuracy",
        "models": list(MODELS),
        "settings": list(SETTINGS),
        "scope": {
            "answer_behaviors": ["invariant"],
            "question_types": list(ATOMIC_TYPES),
            "base_questions_per_model": 400,
            "fictional_variant_evaluations_per_model_and_setting": 4_000,
        },
        "bootstrap": {
            "method": "paired nonparametric document-cluster percentile bootstrap",
            "replicates": samples,
            "seed": seed,
            "document_clusters": len(documents),
            "document_index_draws_sha256": digest.hexdigest(),
        },
        "by_model": by_model,
    }
