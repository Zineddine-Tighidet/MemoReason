"""Frozen scientific routines used by the submitted result release."""
from __future__ import annotations
from typing import Any
import numpy as np
from .shared_bootstrap import bootstrap_shared_documents

CONTRACT = "dataset100_reconciled_reasoning_effort_release_v1"

MODEL = "gpt-oss-20b-groq"

EFFORTS = ("low", "medium", "high")

METRICS = ("factual_accuracy", "fully_fictional_accuracy", "gap")

CONTRASTS = (
    ("medium_minus_low", "medium", "low"),
    ("high_minus_low", "high", "low"),
    ("high_minus_medium", "high", "medium"),
)

QUESTION_TYPES = ("arithmetic", "temporal", "inference", "extractive")

ANSWER_BEHAVIORS = ("variant", "invariant", "refusal")

def build_document_metrics(
    loaded: dict[str, list[dict[str, Any]]]
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    scores: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for effort, rows in loaded.items():
        by_document: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            document = f"{row['document_theme']}::{row['document_id']}"
            pair = str(row["pair_key"])
            question_type = str(row.get("question_type") or "")
            answer_behavior = str(row.get("answer_behavior") or "")
            if question_type not in QUESTION_TYPES or answer_behavior not in ANSWER_BEHAVIORS:
                raise ValueError(f"{effort}/{pair}: unexpected question taxonomy")
            bucket = by_document.setdefault(document, {}).setdefault(
                pair,
                {
                    "question_type": question_type,
                    "answer_behavior": answer_behavior,
                    "factual": [],
                    "fictional": {},
                },
            )
            if (bucket["question_type"], bucket["answer_behavior"]) != (
                question_type,
                answer_behavior,
            ):
                raise ValueError(f"{effort}/{pair}: taxonomy drift")
            score = float(bool(row["new_final_is_correct"]))
            if row["document_setting"] == "factual":
                bucket["factual"].append(score)
            else:
                variant = str(row["document_variant_id"])
                if variant in bucket["fictional"]:
                    raise ValueError(f"{effort}/{pair}: duplicate fictional variant {variant}")
                bucket["fictional"][variant] = score
        scores[effort] = by_document

    documents = sorted(scores["low"])
    if len(documents) != 100:
        raise ValueError(f"Expected 100 document clusters, found {len(documents)}")
    matrix = np.empty((len(EFFORTS), len(documents), len(METRICS)), dtype=float)
    audit: dict[str, Any] = {}
    reference_pairs = {document: set(scores["low"][document]) for document in documents}
    reference_taxonomy = {
        document: {
            pair: (
                scores["low"][document][pair]["question_type"],
                scores["low"][document][pair]["answer_behavior"],
            )
            for pair in scores["low"][document]
        }
        for document in documents
    }
    expected_taxonomy = set((q, a) for q in QUESTION_TYPES for a in ANSWER_BEHAVIORS)
    for effort_index, effort in enumerate(EFFORTS):
        if sorted(scores[effort]) != documents:
            raise ValueError(f"{effort}: document scope differs from low")
        effort_audit: dict[str, Any] = {}
        for document_index, document in enumerate(documents):
            pairs = scores[effort][document]
            if set(pairs) != reference_pairs[document]:
                raise ValueError(f"{effort}/{document}: pair scope differs from low")
            taxonomy = {
                pair: (bucket["question_type"], bucket["answer_behavior"])
                for pair, bucket in pairs.items()
            }
            if taxonomy != reference_taxonomy[document]:
                raise ValueError(f"{effort}/{document}: taxonomy differs from low")
            if len(pairs) != 12 or set(taxonomy.values()) != expected_taxonomy:
                raise ValueError(f"{effort}/{document}: incomplete 4x3 question grid")
            factual_values: list[float] = []
            fictional_question_values: list[float] = []
            for pair, bucket in pairs.items():
                if len(bucket["factual"]) != 1 or len(bucket["fictional"]) != 10:
                    raise ValueError(f"{effort}/{pair}: expected one factual and ten fictional scores")
                factual_values.extend(bucket["factual"])
                fictional_question_values.append(float(np.mean(list(bucket["fictional"].values()))))
            factual = float(np.mean(factual_values))
            fictional = float(np.mean(fictional_question_values))
            matrix[effort_index, document_index] = factual, fictional, factual - fictional
            effort_audit[document] = {
                "base_questions": len(pairs),
                "factual_rows": len(factual_values),
                "fully_fictional_rows": sum(len(bucket["fictional"]) for bucket in pairs.values()),
            }
        audit[effort] = effort_audit
    if not np.allclose(matrix[:, :, 2], matrix[:, :, 0] - matrix[:, :, 1], atol=1e-15, rtol=0):
        raise ValueError("Paired document gap identity failed")
    return documents, matrix, audit

def cell(point: float, distribution: np.ndarray) -> dict[str, Any]:
    lower, upper = np.quantile(distribution, (0.025, 0.975), method="linear")
    return {
        "value_percent": 100.0 * float(point),
        "ci95_percent": [100.0 * float(lower), 100.0 * float(upper)],
        "bootstrap_standard_error_percent": 100.0 * float(np.std(distribution, ddof=1)),
        "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
    }

def build_report(
    documents: list[str],
    matrix: np.ndarray,
    audit: dict[str, Any],
    *,
    samples: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    points, draws, draw_sha = bootstrap_shared_documents(
        document_metrics=matrix,
        samples=samples,
        seed=seed,
        chunk_size=chunk_size,
    )
    metrics: dict[str, Any] = {}
    for effort_index, effort in enumerate(EFFORTS):
        metrics[effort] = {
            metric: cell(points[effort_index, metric_index], draws[:, effort_index, metric_index])
            for metric_index, metric in enumerate(METRICS)
        }
    contrasts: dict[str, Any] = {}
    for name, minuend, subtrahend in CONTRASTS:
        i = EFFORTS.index(minuend)
        j = EFFORTS.index(subtrahend)
        contrasts[name] = {
            metric: cell(
                points[i, metric_index] - points[j, metric_index],
                draws[:, i, metric_index] - draws[:, j, metric_index],
            )
            for metric_index, metric in enumerate(METRICS)
        }
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "complete",
        "model": MODEL,
        "question_scope": "all_12_questions",
        "counts": {
            "document_clusters": 100,
            "base_questions_per_document": 12,
            "factual_rows_per_effort": 1_200,
            "fully_fictional_rows_per_effort": 12_000,
            "fully_fictional_variants_per_question": 10,
        },
        "metric_definition": {
            "score": "reconciled final correctness",
            "factual_accuracy": "mean of all 12 factual questions within each document",
            "fully_fictional_accuracy": "mean K=10 variants within question, then 12 questions within document",
            "gap": "paired factual minus fully fictional document accuracy",
        },
        "bootstrap": {
            "method": "paired nonparametric percentile bootstrap over whole source documents",
            "samples": samples,
            "seed": seed,
            "confidence_level": 0.95,
            "pairing_rule": "one document-index draw shared by all efforts and metrics",
            "numpy_quantile_method": "linear",
            "document_index_draws_sha256": draw_sha,
        },
        "metrics": metrics,
        "contrasts": contrasts,
        "documents": documents,
        "audits": {
            "same_document_and_pair_scope": "PASS",
            "complete_4x3_question_taxonomy": "PASS",
            "paired_gap_identity": "PASS",
            "per_document_counts": audit,
        },
    }
