"""Document-clustered bootstrap analysis for the rebuttal Table 2 claims.

The benchmark contains repeated questions within each source document.  This
module therefore treats the document, not the base question or fictional
variant, as the independent resampling unit.  Each base-question contrast is
still paired and its fictional score is the mean over the manifest-declared
variants.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .evaluated_model_answer_loading import load_evaluated_answers
from .frozen_paper_results_input_loading import load_paper_results_manifest, sha256_file
from .paper_artifact_serialization import (
    latex_escape,
    write_artifact_manifest,
    write_csv,
    write_json,
    write_text,
)
from .paper_results_data_model import (
    ANSWER_BEHAVIORS,
    ANSWER_GROUPS,
    MODEL_LABELS,
    QUESTION_GROUPS,
    QUESTION_TYPES,
    REASONING_TYPES,
    EvaluatedAnswer,
)
from .tables_1_to_4_and_figures_3_5_6 import (
    TABLE_3_CHAIN_OF_THOUGHT_MODELS,
    TABLES_2_AND_4_MODELS,
)
from .tables_2_and_3_factual_to_fictional_performance import (
    _pair_factual_accuracy_with_mean_fictional_accuracy,
    compute_table_3_chain_of_thought_comparison,
    compute_table_2_performance_drop,
)


DEFAULT_RESAMPLES = 100_000
DEFAULT_SEED = 20_260_724
PRIMARY_QUESTION_TYPES = QUESTION_TYPES
REASON_SUMMARY = "reasoning"
CLUSTER_TEST = (
    "two_sided_studentized_rademacher_wild_cluster_"
    "null_restricted_centered_document_effects_plus_one"
)
CLUSTER_TEST_STATISTIC = "absolute_one_sample_t_on_equal_document_means"


@dataclass(frozen=True)
class DocumentClusterBootstrapSummary:
    """Bootstrap uncertainty for the equal-document mean paired effect."""

    document_clusters: int
    nested_observations: int
    observations_per_document: int
    mean: float
    bootstrap_standard_error: float
    ci_low: float
    ci_high: float
    p_value: float
    cohens_dz: float | None
    hedges_gz: float | None

    @property
    def ci_half_width(self) -> float:
        return max(self.mean - self.ci_low, self.ci_high - self.mean)


def _stable_cell_seed(base_seed: int, *labels: str) -> int:
    encoded = ":".join((str(base_seed), *labels)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], byteorder="big", signed=False)


def summarize_document_clusters(
    values_by_document: Mapping[str, Sequence[float]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence_level: float = 0.95,
    batch_size: int = 5_000,
) -> DocumentClusterBootstrapSummary:
    """Resample whole documents while preserving every nested observation.

    All documents must contribute the same number of observations.  MemoReason
    is balanced by construction (one question in every elementary
    answer-behavior/question-type stratum and three questions in each Reason
    summary).  Rejecting imbalance prevents an accidental change from the
    equal-document estimand to a question-count-weighted estimand.

    The confidence interval is the non-parametric percentile interval.  The
    two-sided p-value uses a studentized Rademacher wild-cluster bootstrap at
    the document level and includes the usual plus-one finite-Monte-Carlo
    correction.
    """

    if resamples < 999:
        raise ValueError("resamples must be at least 999 for a stable 95% interval.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between zero and one.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not values_by_document:
        raise ValueError("Cannot bootstrap zero document clusters.")

    ordered = []
    cluster_sizes: set[int] = set()
    for document_key in sorted(values_by_document):
        values = np.asarray(tuple(values_by_document[document_key]), dtype=float)
        if values.size == 0:
            raise ValueError(f"Document cluster {document_key!r} is empty.")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Document cluster {document_key!r} contains non-finite values.")
        cluster_sizes.add(int(values.size))
        ordered.append(values)
    if len(cluster_sizes) != 1:
        raise ValueError(
            "Document clusters must be balanced for the equal-document estimand; "
            f"observed sizes={sorted(cluster_sizes)}."
        )

    observations_per_document = next(iter(cluster_sizes))
    document_means = np.asarray([float(np.mean(values)) for values in ordered], dtype=float)
    document_clusters = int(document_means.size)
    if document_clusters < 2:
        raise ValueError("Document-cluster inference requires at least two document clusters.")
    observed_mean = float(np.mean(document_means))

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(
            0,
            document_clusters,
            size=(stop - start, document_clusters),
            endpoint=False,
        )
        bootstrap_means[start:stop] = np.mean(document_means[indices], axis=1)

    alpha = 1.0 - confidence_level
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        (alpha / 2.0, 1.0 - alpha / 2.0),
        method="linear",
    )
    standard_deviation = (
        float(np.std(document_means, ddof=1)) if document_clusters > 1 else 0.0
    )
    if standard_deviation > 0.0 and document_clusters > 1:
        observed_t = observed_mean / (standard_deviation / math.sqrt(document_clusters))
        # Impose the null by centering the document effects, then apply
        # Rademacher weights to those null-restricted residuals.  Both the
        # observed and bootstrap statistics are absolute one-sample t values.
        centered_document_means = document_means - observed_mean
        wild_rng = np.random.default_rng(seed ^ 0x9E3779B97F4A7C15)
        extreme_count = 0
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            weights = wild_rng.integers(
                0,
                2,
                size=(stop - start, document_clusters),
                endpoint=False,
                dtype=np.int8,
            )
            weights = 2.0 * weights - 1.0
            wild_samples = weights * centered_document_means
            wild_means = np.mean(wild_samples, axis=1)
            wild_standard_deviations = np.std(wild_samples, axis=1, ddof=1)
            wild_standard_errors = wild_standard_deviations / math.sqrt(document_clusters)
            wild_t = np.divide(
                wild_means,
                wild_standard_errors,
                out=np.zeros_like(wild_means),
                where=wild_standard_errors > 0.0,
            )
            extreme_count += int(np.count_nonzero(np.abs(wild_t) >= abs(observed_t)))
        p_value = (extreme_count + 1.0) / (resamples + 1.0)
    elif np.isclose(observed_mean, 0.0):
        p_value = 1.0
    else:
        p_value = 1.0 / (resamples + 1.0)

    if standard_deviation > 0.0 and math.isfinite(standard_deviation):
        cohens_dz: float | None = observed_mean / standard_deviation
        correction = 1.0 - (3.0 / (4.0 * document_clusters - 5.0))
        hedges_gz: float | None = cohens_dz * correction
    elif np.isclose(observed_mean, 0.0):
        cohens_dz = 0.0
        hedges_gz = 0.0
    else:
        # Infinite standardized effects are not valid JSON and are not useful
        # for a degenerate all-identical sample.  The raw effect remains exact.
        cohens_dz = None
        hedges_gz = None

    return DocumentClusterBootstrapSummary(
        document_clusters=document_clusters,
        nested_observations=document_clusters * observations_per_document,
        observations_per_document=observations_per_document,
        mean=observed_mean,
        bootstrap_standard_error=float(np.std(bootstrap_means, ddof=1)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
        cohens_dz=cohens_dz,
        hedges_gz=hedges_gz,
    )


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm--Bonferroni family-wise adjusted p-values."""

    if not p_values:
        return []
    values = np.asarray(tuple(p_values), dtype=float)
    if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Holm adjustment requires finite p-values in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(values.size, dtype=float)
    running_max = 0.0
    family_size = int(values.size)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (family_size - rank) * float(values[original_index]))
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return [float(value) for value in adjusted]


def _document_key(row: EvaluatedAnswer) -> str:
    return f"{row.document_theme}::{row.document_id}"


def _balanced_document_values(
    selected: Sequence[tuple[EvaluatedAnswer, float]],
    *,
    expected_documents: set[str],
    expected_question_types: Sequence[str],
    model: str,
    answer_group: str,
    question_group: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    differences_by_document: dict[str, list[float]] = defaultdict(list)
    factual_by_document: dict[str, list[float]] = defaultdict(list)
    fictional_by_document: dict[str, list[float]] = defaultdict(list)
    observed_types: dict[str, list[str]] = defaultdict(list)
    for factual, fictional_mean in selected:
        key = _document_key(factual)
        factual_score = float(factual.final_is_correct)
        differences_by_document[key].append(fictional_mean - factual_score)
        factual_by_document[key].append(factual_score)
        fictional_by_document[key].append(fictional_mean)
        observed_types[key].append(factual.question_type)

    observed_documents = set(differences_by_document)
    if observed_documents != expected_documents:
        missing = sorted(expected_documents - observed_documents)
        extra = sorted(observed_documents - expected_documents)
        raise ValueError(
            f"{model}/{answer_group}/{question_group}: document scope is not balanced; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    expected_types = sorted(expected_question_types)
    for key in sorted(expected_documents):
        if sorted(observed_types[key]) != expected_types:
            raise ValueError(
                f"{model}/{answer_group}/{question_group}/{key}: expected question types "
                f"{expected_types}, got {sorted(observed_types[key])}."
            )
    return (
        dict(differences_by_document),
        dict(factual_by_document),
        dict(fictional_by_document),
    )


def compute_document_clustered_table_2(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    models: Sequence[str],
    comparison_setting: str,
    variant_ids: Sequence[str],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    expected_document_clusters: int | None = None,
) -> list[dict[str, Any]]:
    """Compute Table 2 with documents as the independent bootstrap units."""

    result_rows: list[dict[str, Any]] = []
    for model in models:
        paired = _pair_factual_accuracy_with_mean_fictional_accuracy(
            evaluated_answers,
            model=model,
            comparison_setting=comparison_setting,
            variant_ids=variant_ids,
        )
        expected_documents = {_document_key(factual) for factual, _ in paired}
        if (
            expected_document_clusters is not None
            and len(expected_documents) != expected_document_clusters
        ):
            raise ValueError(
                f"{model}: expected {expected_document_clusters} document clusters, "
                f"got {len(expected_documents)}."
            )
        expected_pairs = len(expected_documents) * len(ANSWER_BEHAVIORS) * len(QUESTION_TYPES)
        if len(paired) != expected_pairs:
            raise ValueError(
                f"{model}: expected one question per document in each of the "
                f"{len(ANSWER_BEHAVIORS) * len(QUESTION_TYPES)} elementary strata; "
                f"got {len(paired)} pairs across {len(expected_documents)} documents."
            )

        for answer_group, answer_members in ANSWER_GROUPS.items():
            for question_group, question_members in QUESTION_GROUPS.items():
                selected = [
                    (factual, fictional_mean)
                    for factual, fictional_mean in paired
                    if factual.answer_behavior in answer_members
                    and factual.question_type in question_members
                ]
                if not selected:
                    raise ValueError(
                        f"{model}: empty clustered Table 2 cell {answer_group}/{question_group}."
                    )
                differences, factual_scores, fictional_scores = _balanced_document_values(
                    selected,
                    expected_documents=expected_documents,
                    expected_question_types=question_members,
                    model=model,
                    answer_group=answer_group,
                    question_group=question_group,
                )
                cell_seed = _stable_cell_seed(seed, model, answer_group, question_group)
                summary = summarize_document_clusters(
                    differences,
                    resamples=resamples,
                    seed=cell_seed,
                )
                factual_mean = float(
                    np.mean([np.mean(values) for values in factual_scores.values()])
                )
                fictional_mean = float(
                    np.mean([np.mean(values) for values in fictional_scores.values()])
                )
                primary = question_group in PRIMARY_QUESTION_TYPES
                result_rows.append(
                    {
                        "model_name": model,
                        "answer_behavior": answer_group,
                        "question_type": question_group,
                        "result_role": "primary_question_class" if primary else "reason_summary",
                        "count_document_clusters": summary.document_clusters,
                        "count_base_questions": summary.nested_observations,
                        "questions_per_document_in_cell": summary.observations_per_document,
                        "variants_per_base_question": len(tuple(variant_ids)),
                        "count_fictional_variant_evaluations": (
                            summary.nested_observations * len(tuple(variant_ids))
                        ),
                        "factual_accuracy_pp": 100.0 * factual_mean,
                        "fictional_accuracy_pp": 100.0 * fictional_mean,
                        "contrast_definition": "fictional_minus_factual",
                        "fictional_minus_factual_pp": 100.0 * summary.mean,
                        "factual_minus_fictional_drop_pp": -100.0 * summary.mean,
                        "document_cluster_bootstrap_standard_error_pp": (
                            100.0 * summary.bootstrap_standard_error
                        ),
                        "fictional_minus_factual_ci95_half_width_pp": (
                            100.0 * summary.ci_half_width
                        ),
                        "fictional_minus_factual_ci95_low_pp": 100.0 * summary.ci_low,
                        "fictional_minus_factual_ci95_high_pp": 100.0 * summary.ci_high,
                        "factual_minus_fictional_drop_ci95_low_pp": -100.0 * summary.ci_high,
                        "factual_minus_fictional_drop_ci95_high_pp": -100.0 * summary.ci_low,
                        "cluster_wild_bootstrap_p_value": summary.p_value,
                        "document_cohens_dz_fictional_minus_factual": summary.cohens_dz,
                        "document_hedges_gz_fictional_minus_factual": summary.hedges_gz,
                        "document_hedges_gz_factual_minus_fictional_drop": (
                            None if summary.hedges_gz is None else -summary.hedges_gz
                        ),
                        "significant_cluster_bootstrap_ci_0_05": (
                            summary.ci_low > 0.0 or summary.ci_high < 0.0
                        ),
                        "bootstrap_resamples": resamples,
                        "bootstrap_cell_seed": cell_seed,
                        "holm_adjusted_p_value_within_model_12": None,
                        "holm_adjusted_p_value_global_primary": None,
                        "significant_holm_within_model_0_05": False,
                        "significant_holm_global_0_05": False,
                    }
                )

    for model in models:
        primary_rows = [
            row
            for row in result_rows
            if row["model_name"] == model and row["result_role"] == "primary_question_class"
        ]
        if len(primary_rows) != len(ANSWER_BEHAVIORS) * len(PRIMARY_QUESTION_TYPES):
            raise AssertionError(f"{model}: expected 12 primary Table 2 tests.")
        adjusted = holm_adjust([float(row["cluster_wild_bootstrap_p_value"]) for row in primary_rows])
        for row, adjusted_p in zip(primary_rows, adjusted, strict=True):
            row["holm_adjusted_p_value_within_model_12"] = adjusted_p
            row["significant_holm_within_model_0_05"] = bool(adjusted_p < 0.05)

    all_primary_rows = [
        row for row in result_rows if row["result_role"] == "primary_question_class"
    ]
    global_adjusted = holm_adjust(
        [float(row["cluster_wild_bootstrap_p_value"]) for row in all_primary_rows]
    )
    for row, adjusted_p in zip(all_primary_rows, global_adjusted, strict=True):
        row["holm_adjusted_p_value_global_primary"] = adjusted_p
        row["significant_holm_global_0_05"] = bool(adjusted_p < 0.05)
    return result_rows


def compute_document_clustered_table_3(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    models: Sequence[str],
    comparison_setting: str,
    variant_ids: Sequence[str],
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    expected_document_clusters: int | None = None,
) -> list[dict[str, Any]]:
    """Recompute the secondary all-reasoning Table 3 by document cluster."""

    result_rows: list[dict[str, Any]] = []
    expected_strata = {
        (answer_behavior, question_type)
        for answer_behavior in ANSWER_BEHAVIORS
        for question_type in REASONING_TYPES
    }
    for model in models:
        paired = [
            item
            for item in _pair_factual_accuracy_with_mean_fictional_accuracy(
                evaluated_answers,
                model=model,
                comparison_setting=comparison_setting,
                variant_ids=variant_ids,
            )
            if item[0].question_type in REASONING_TYPES
        ]
        by_document: dict[str, list[tuple[EvaluatedAnswer, float]]] = defaultdict(list)
        for item in paired:
            by_document[_document_key(item[0])].append(item)
        if (
            expected_document_clusters is not None
            and len(by_document) != expected_document_clusters
        ):
            raise ValueError(
                f"{model}: expected {expected_document_clusters} Table 3 document clusters, "
                f"got {len(by_document)}."
            )
        factual_by_document: dict[str, list[float]] = {}
        fictional_by_document: dict[str, list[float]] = {}
        differences_by_document: dict[str, list[float]] = {}
        for document_key, items in sorted(by_document.items()):
            observed_strata = {
                (factual.answer_behavior, factual.question_type) for factual, _ in items
            }
            if observed_strata != expected_strata or len(items) != len(expected_strata):
                raise ValueError(
                    f"{model}/{document_key}: incomplete Table 3 reasoning design; "
                    f"expected {sorted(expected_strata)}, got {sorted(observed_strata)}."
                )
            factual_values = [float(factual.final_is_correct) for factual, _ in items]
            fictional_values = [fictional_mean for _, fictional_mean in items]
            factual_by_document[document_key] = factual_values
            fictional_by_document[document_key] = fictional_values
            differences_by_document[document_key] = [
                fictional - factual
                for factual, fictional in zip(factual_values, fictional_values, strict=True)
            ]

        cell_seed = _stable_cell_seed(seed, model, "table3", "all_reasoning")
        # The same seed and sorted document keys deliberately reuse identical
        # document resamples for factual, fictional, and paired-difference CIs.
        factual_summary = summarize_document_clusters(
            factual_by_document,
            resamples=resamples,
            seed=cell_seed,
        )
        fictional_summary = summarize_document_clusters(
            fictional_by_document,
            resamples=resamples,
            seed=cell_seed,
        )
        difference_summary = summarize_document_clusters(
            differences_by_document,
            resamples=resamples,
            seed=cell_seed,
        )
        result_rows.append(
            {
                "model_name": model,
                "result_role": "secondary_all_reasoning_summary",
                "question_types": "+".join(REASONING_TYPES),
                "answer_behaviors": "+".join(ANSWER_BEHAVIORS),
                "count_document_clusters": difference_summary.document_clusters,
                "count_base_questions": difference_summary.nested_observations,
                "questions_per_document_in_cell": difference_summary.observations_per_document,
                "variants_per_base_question": len(tuple(variant_ids)),
                "count_fictional_variant_evaluations": (
                    difference_summary.nested_observations * len(tuple(variant_ids))
                ),
                "factual_accuracy_pp": 100.0 * factual_summary.mean,
                "factual_document_cluster_ci95_low_pp": 100.0 * factual_summary.ci_low,
                "factual_document_cluster_ci95_high_pp": 100.0 * factual_summary.ci_high,
                "fictional_accuracy_pp": 100.0 * fictional_summary.mean,
                "fictional_document_cluster_ci95_low_pp": 100.0 * fictional_summary.ci_low,
                "fictional_document_cluster_ci95_high_pp": 100.0 * fictional_summary.ci_high,
                "contrast_definition": "fictional_minus_factual",
                "fictional_minus_factual_pp": 100.0 * difference_summary.mean,
                "factual_minus_fictional_drop_pp": -100.0 * difference_summary.mean,
                "fictional_minus_factual_ci95_low_pp": 100.0 * difference_summary.ci_low,
                "fictional_minus_factual_ci95_high_pp": 100.0 * difference_summary.ci_high,
                "factual_minus_fictional_drop_ci95_low_pp": -100.0 * difference_summary.ci_high,
                "factual_minus_fictional_drop_ci95_high_pp": -100.0 * difference_summary.ci_low,
                "cluster_wild_bootstrap_p_value": difference_summary.p_value,
                "document_cohens_dz_fictional_minus_factual": difference_summary.cohens_dz,
                "document_hedges_gz_factual_minus_fictional_drop": (
                    None if difference_summary.hedges_gz is None else -difference_summary.hedges_gz
                ),
                "significant_document_cluster_ci_0_05": (
                    difference_summary.ci_low > 0.0 or difference_summary.ci_high < 0.0
                ),
                "bootstrap_resamples": resamples,
                "bootstrap_cell_seed": cell_seed,
                "holm_adjusted_p_value_across_table3_models": None,
                "significant_holm_across_table3_models_0_05": False,
            }
        )

    adjusted = holm_adjust([float(row["cluster_wild_bootstrap_p_value"]) for row in result_rows])
    for row, adjusted_p in zip(result_rows, adjusted, strict=True):
        row["holm_adjusted_p_value_across_table3_models"] = adjusted_p
        row["significant_holm_across_table3_models_0_05"] = bool(adjusted_p < 0.05)
    return result_rows


def compare_legacy_and_clustered_rows(
    legacy_rows: Sequence[Mapping[str, Any]],
    clustered_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Join the old question-level t results to the hardened analysis."""

    legacy = {
        (str(row["model_name"]), str(row["answer_behavior"]), str(row["question_type"])): row
        for row in legacy_rows
    }
    comparison: list[dict[str, Any]] = []
    for row in clustered_rows:
        key = (
            str(row["model_name"]),
            str(row["answer_behavior"]),
            str(row["question_type"]),
        )
        old = legacy[key]
        old_mean = float(old["mean_difference_pp"])
        new_mean = float(row["fictional_minus_factual_pp"])
        if not np.isclose(old_mean, new_mean, rtol=0.0, atol=1e-10):
            raise ValueError(f"Point estimand drift for {key}: legacy={old_mean}, clustered={new_mean}.")
        legacy_significant = bool(old["significant_paired_t_0_05"])
        cluster_significant = bool(row["significant_cluster_bootstrap_ci_0_05"])
        if legacy_significant and cluster_significant:
            status = "survives_document_clustering"
        elif legacy_significant:
            status = "no_longer_significant_after_document_clustering"
        elif cluster_significant:
            status = "significant_only_after_document_clustering"
        else:
            status = "not_significant_in_either"
        comparison.append(
            {
                "model_name": key[0],
                "answer_behavior": key[1],
                "question_type": key[2],
                "result_role": row["result_role"],
                "contrast_definition": "fictional_minus_factual",
                "fictional_minus_factual_pp": new_mean,
                "factual_minus_fictional_drop_pp": -new_mean,
                "legacy_t_ci95_low_pp": float(old["ci95_low_pp"]),
                "legacy_t_ci95_high_pp": float(old["ci95_high_pp"]),
                "legacy_t_p_value": float(old["p_value"]),
                "legacy_significant_paired_t_0_05": legacy_significant,
                "fictional_minus_factual_ci95_low_pp": float(
                    row["fictional_minus_factual_ci95_low_pp"]
                ),
                "fictional_minus_factual_ci95_high_pp": float(
                    row["fictional_minus_factual_ci95_high_pp"]
                ),
                "factual_minus_fictional_drop_ci95_low_pp": float(
                    row["factual_minus_fictional_drop_ci95_low_pp"]
                ),
                "factual_minus_fictional_drop_ci95_high_pp": float(
                    row["factual_minus_fictional_drop_ci95_high_pp"]
                ),
                "cluster_wild_bootstrap_p_value": float(row["cluster_wild_bootstrap_p_value"]),
                "cluster_significant_ci_0_05": cluster_significant,
                "holm_adjusted_p_value_within_model_12": row[
                    "holm_adjusted_p_value_within_model_12"
                ],
                "significant_holm_within_model_0_05": row[
                    "significant_holm_within_model_0_05"
                ],
                "significance_status": status,
            }
        )
    return comparison


def _format_interval_cell(row: Mapping[str, Any], *, include_holm_marker: bool) -> str:
    marker = "*" if include_holm_marker and bool(row["significant_holm_within_model_0_05"]) else ""
    return (
        f"{float(row['factual_minus_fictional_drop_pp']):+.1f}{marker} "
        f"[{float(row['factual_minus_fictional_drop_ci95_low_pp']):+.1f}, "
        f"{float(row['factual_minus_fictional_drop_ci95_high_pp']):+.1f}]"
    )


def _single_design_count(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    values = {int(row[field]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"Expected one consistent {field!r} value, got {sorted(values)}.")
    return next(iter(values))


def render_clustered_table_2_latex(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[str],
) -> str:
    """Render primary class results separately from the Reason summary."""

    document_clusters = _single_design_count(rows, "count_document_clusters")
    variants_per_question = _single_design_count(rows, "variants_per_base_question")
    lookup = {
        (str(row["model_name"]), str(row["answer_behavior"]), str(row["question_type"])): row
        for row in rows
    }
    lines = [
        r"% Primary per-question-class results. * denotes Holm-adjusted p < 0.05 within model.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model & Answer behavior & Arithmetic & Temporal & Inference & Extractive \\",
        r"\midrule",
    ]
    for model in models:
        for answer_behavior in ANSWER_BEHAVIORS:
            cells = [
                _format_interval_cell(
                    lookup[(model, answer_behavior, question_type)],
                    include_holm_marker=True,
                )
                for question_type in PRIMARY_QUESTION_TYPES
            ]
            lines.append(
                f"{latex_escape(MODEL_LABELS.get(model, model))} & "
                f"{latex_escape(answer_behavior.title())} & "
                + " & ".join(cells)
                + r" \\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Primary factual-to-fictional accuracy drops (factual minus fictional; positive values denote degradation) in percentage points with 95\% document-cluster bootstrap intervals. The bootstrap resamples {document_clusters} source documents and retains all paired questions and all {variants_per_question} fictional variants within each sampled document. Asterisks denote Holm-adjusted $p<0.05$ over the 12 elementary answer-behavior $\times$ question-class tests within each model.}}",
            r"\label{tab:question-answer-type-drop-clustered-primary}",
            r"\end{table*}",
            "",
            r"% The aggregated Reason result is intentionally a secondary summary.",
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Model & Variant Reason & Invariant Reason & Refusal Reason \\",
            r"\midrule",
        ]
    )
    for model in models:
        cells = [
            _format_interval_cell(
                lookup[(model, answer_behavior, REASON_SUMMARY)],
                include_holm_marker=False,
            )
            for answer_behavior in ANSWER_BEHAVIORS
        ]
        lines.append(
            f"{latex_escape(MODEL_LABELS.get(model, model))} & " + " & ".join(cells) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Secondary aggregated Reason summary. Each document contributes the mean of its arithmetic, temporal, and inference contrasts; heterogeneous class-specific effects in Table~\ref{tab:question-answer-type-drop-clustered-primary} remain the primary evidence. No multiplicity-adjusted significance claim is attached to this summary.}",
            r"\label{tab:question-answer-type-drop-clustered-reason-summary}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_clustered_table_3_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the all-reasoning aggregate explicitly as a secondary summary."""

    document_clusters = _single_design_count(rows, "count_document_clusters")
    questions_per_document = _single_design_count(rows, "questions_per_document_in_cell")
    variants_per_question = _single_design_count(rows, "variants_per_base_question")
    lines = [
        r"% Secondary aggregate over nine reasoning questions per document.",
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Factual & Fictional & Accuracy drop \\",
        r"\midrule",
    ]
    for row in rows:
        factual = (
            f"{float(row['factual_accuracy_pp']):.1f} "
            f"[{float(row['factual_document_cluster_ci95_low_pp']):.1f}, "
            f"{float(row['factual_document_cluster_ci95_high_pp']):.1f}]"
        )
        fictional = (
            f"{float(row['fictional_accuracy_pp']):.1f} "
            f"[{float(row['fictional_document_cluster_ci95_low_pp']):.1f}, "
            f"{float(row['fictional_document_cluster_ci95_high_pp']):.1f}]"
        )
        drop = (
            f"{float(row['factual_minus_fictional_drop_pp']):+.1f} "
            f"[{float(row['factual_minus_fictional_drop_ci95_low_pp']):+.1f}, "
            f"{float(row['factual_minus_fictional_drop_ci95_high_pp']):+.1f}]"
        )
        lines.append(
            f"{latex_escape(MODEL_LABELS.get(str(row['model_name']), str(row['model_name'])))} "
            f"& {factual} & {fictional} & {drop} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{Secondary all-reasoning summary over {questions_per_document} questions per document. Values are accuracies or factual-minus-fictional accuracy drops in percentage points with 95\% bootstrap intervals over {document_clusters} document clusters; each base question averages {variants_per_question} fictional variants. Class-specific Table~\ref{{tab:question-answer-type-drop-clustered-primary}} results are primary.}}",
            r"\label{tab:cot-effect-document-cluster-summary}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_w2_markdown_report(
    rows: Sequence[Mapping[str, Any]],
    table_3_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[str],
    resamples: int,
    seed: int,
) -> str:
    primary = [row for row in rows if row["result_role"] == "primary_question_class"]
    reason = [row for row in rows if row["result_role"] == "reason_summary"]
    if not models:
        raise ValueError("At least one Table 2 model is required to render the W2 report.")
    document_clusters = _single_design_count(rows, "count_document_clusters")
    variants_per_question = _single_design_count(rows, "variants_per_base_question")
    first_model_primary = [row for row in primary if row["model_name"] == models[0]]
    base_questions_per_model = sum(int(row["count_base_questions"]) for row in first_model_primary)
    questions_per_document = base_questions_per_model // document_clusters
    if questions_per_document * document_clusters != base_questions_per_model:
        raise ValueError("Base-question count is not divisible by the document-cluster count.")
    fictional_evaluations_per_model = base_questions_per_model * variants_per_question
    survived = [
        row
        for row in comparison_rows
        if row["result_role"] == "primary_question_class"
        and row["significance_status"] == "survives_document_clustering"
    ]
    lost = [
        row
        for row in comparison_rows
        if row["result_role"] == "primary_question_class"
        and row["significance_status"] == "no_longer_significant_after_document_clustering"
    ]
    holm_model = sum(bool(row["significant_holm_within_model_0_05"]) for row in primary)
    holm_global = sum(bool(row["significant_holm_global_0_05"]) for row in primary)

    lines = [
        "# W2 statistics hardening",
        "",
        "## Design and inferential unit",
        "",
        f"MemoReason contains {document_clusters} source documents, {questions_per_document} "
        f"pre-specified base questions per document ({base_questions_per_model:,} paired "
        f"base-question contrasts), and {variants_per_question} fully fictional variants per base question. "
        "The variants are document-level realizations shared by all 12 questions. They are averaged "
        "within each base question and are not treated as independent. "
        f"All W2 Table 2/3 uncertainty estimates resample the {document_clusters} documents and retain their nested "
        "questions and variants.",
        "",
        "The four elementary question classes are primary. The aggregated `Reason` value is a secondary "
        "equal-document summary of arithmetic, temporal, and inference and is not used as an additional "
        "multiplicity-adjusted hypothesis test.",
        "",
        "## Method",
        "",
        f"- Non-parametric paired document-cluster bootstrap: {resamples:,} resamples.",
        f"- Reproducibility seed: {seed}; a stable per-cell seed is derived from the model and stratum.",
        "- Two-sided studentized Rademacher wild-cluster p-values with a plus-one Monte Carlo correction.",
        "- Primary presentation: factual minus fictional accuracy drop in percentage points (positive means degradation).",
        "- The CSV/JSON also retain the exact opposite-signed `fictional_minus_factual` contrast for compatibility with Table 2.",
        "- Standardized effect: document-level paired Cohen's dz and small-sample-corrected Hedges' gz, in both sign conventions.",
        "- Multiplicity: Holm correction across 12 primary answer-behavior x question-class tests per model; "
        "a global correction across every primary model/stratum test is also reported as a sensitivity analysis.",
        "",
        "## Robustness summary",
        "",
        f"Across {len(primary)} primary cells ({len(models)} models x 12 strata), "
        f"{len(survived)} legacy significant cells retain a 95% document-cluster interval excluding zero, "
        f"and {len(lost)} no longer do. {holm_model} remain significant after within-model Holm correction; "
        f"{holm_global} remain significant under the global Holm sensitivity analysis.",
        "",
        "### Variant-answer temporal and inference effects",
        "",
        "| Model | Class | Delta pp | 95% document-cluster CI | Hedges gz | Holm p (12/model) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        for question_type in ("temporal", "inference"):
            row = next(
                candidate
                for candidate in rows
                if candidate["model_name"] == model
                and candidate["answer_behavior"] == "variant"
                and candidate["question_type"] == question_type
            )
            hedges = row["document_hedges_gz_factual_minus_fictional_drop"]
            hedges_text = "NA" if hedges is None else f"{float(hedges):+.2f}"
            lines.append(
                f"| {MODEL_LABELS.get(model, model)} | {question_type.title()} | "
                f"{float(row['factual_minus_fictional_drop_pp']):+.1f} | "
                f"[{float(row['factual_minus_fictional_drop_ci95_low_pp']):+.1f}, "
                f"{float(row['factual_minus_fictional_drop_ci95_high_pp']):+.1f}] | "
                f"{hedges_text} | "
                f"{float(row['holm_adjusted_p_value_within_model_12']):.4g} |"
            )
    lines.extend(
        [
            "",
            "### Secondary all-reasoning/CoT summary",
            "",
            "| Model | Factual accuracy | Fictional accuracy | Drop pp | 95% document-cluster CI |",
            "|---|---:|---:|---:|---:|",
            *[
                (
                    f"| {MODEL_LABELS.get(str(row['model_name']), str(row['model_name']))} | "
                    f"{float(row['factual_accuracy_pp']):.1f} | "
                    f"{float(row['fictional_accuracy_pp']):.1f} | "
                    f"{float(row['factual_minus_fictional_drop_pp']):+.1f} | "
                    f"[{float(row['factual_minus_fictional_drop_ci95_low_pp']):+.1f}, "
                    f"{float(row['factual_minus_fictional_drop_ci95_high_pp']):+.1f}] |"
                )
                for row in table_3_rows
            ],
            "",
            "This nine-question-per-document aggregate is secondary for the same reason as the `Reason` "
            "column: it can conceal arithmetic, temporal, and inference heterogeneity.",
            "",
            "## Rebuttal-ready wording",
            "",
            "We agree that questions from the same source document are not independent and that pooling "
            "arithmetic, temporal, and inference items can hide heterogeneous behavior. We therefore reran "
            f"the factual-to-fictional paired analysis with a document-cluster bootstrap: each of {resamples:,} "
            f"replicates samples the {document_clusters} source documents with replacement and retains all "
            f"questions and all {variants_per_question} fictional variants nested in each selected document. "
            "We now report arithmetic, temporal, "
            "inference, and extractive results as the primary analysis, with raw percentage-point effects, "
            "95% cluster-bootstrap intervals, document-level standardized effects, and Holm-adjusted "
            "significance. The pooled Reason value is retained only as a secondary summary with an explicit "
            "heterogeneity caveat.",
            "",
            f"The sample-size statement has also been corrected: the design has {document_clusters} "
            f"independent document clusters, containing {base_questions_per_model:,} paired question "
            f"contrasts whose fictional score is estimated from {variants_per_question} variants each. "
            f"The nested replication improves precision, but we do not count the "
            f"{fictional_evaluations_per_model:,} fictional evaluations as independent samples.",
            "",
            "## Reason summary caveat",
            "",
            f"The machine-readable output contains {len(reason)} Reason summary cells. They preserve equal "
            "weight for arithmetic, temporal, and inference within each document, but all claims should be "
            "grounded first in the corresponding class-specific rows.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksum_manifest(output_dir: Path, paths: Sequence[Path]) -> Path:
    checksum_path = output_dir / "ARTIFACTS.sha256"
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return checksum_path


def run_analysis(
    *,
    manifest_path: Path,
    output_dir: Path,
    models: Sequence[str],
    table_3_models: Sequence[str],
    comparison_setting: str,
    resamples: int,
    seed: int,
    expected_document_clusters: int,
    code_revision: str | None = None,
    code_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing W2 output directory: {output_dir}")
    paper_manifest = load_paper_results_manifest(manifest_path)
    code_manifest_sha256 = (
        sha256_file(code_manifest_path) if code_manifest_path is not None else None
    )
    evaluated_answers = load_evaluated_answers(paper_manifest)
    clustered_rows = compute_document_clustered_table_2(
        evaluated_answers,
        models=models,
        comparison_setting=comparison_setting,
        variant_ids=paper_manifest.variant_ids,
        resamples=resamples,
        seed=seed,
        expected_document_clusters=expected_document_clusters,
    )
    table_3_rows = compute_document_clustered_table_3(
        evaluated_answers,
        models=table_3_models,
        comparison_setting=comparison_setting,
        variant_ids=paper_manifest.variant_ids,
        resamples=resamples,
        seed=seed,
        expected_document_clusters=expected_document_clusters,
    )
    legacy_rows = compute_table_2_performance_drop(
        evaluated_answers,
        models=models,
        comparison_setting=comparison_setting,
        variant_ids=paper_manifest.variant_ids,
    )
    comparison_rows = compare_legacy_and_clustered_rows(legacy_rows, clustered_rows)
    legacy_table_3_rows = compute_table_3_chain_of_thought_comparison(
        evaluated_answers,
        models=table_3_models,
        comparison_setting=comparison_setting,
        variant_ids=paper_manifest.variant_ids,
    )
    for clustered_row, legacy_row in zip(table_3_rows, legacy_table_3_rows, strict=True):
        checks = {
            "factual": (
                float(clustered_row["factual_accuracy_pp"]),
                float(legacy_row["factual_accuracy_pp"]),
            ),
            "fictional": (
                float(clustered_row["fictional_accuracy_pp"]),
                float(legacy_row["fictional_accuracy_pp"]),
            ),
            "contrast": (
                float(clustered_row["fictional_minus_factual_pp"]),
                float(legacy_row["delta_fictional_minus_factual_pp"]),
            ),
        }
        if any(
            not np.isclose(left, right, rtol=0.0, atol=1e-10)
            for left, right in checks.values()
        ):
            raise ValueError(
                f"Table 3 point estimand drift for {clustered_row['model_name']}: {checks}."
            )

    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = write_csv(output_dir / "table2_document_cluster_bootstrap.csv", clustered_rows)
    comparison_path = write_csv(
        output_dir / "legacy_t_vs_document_cluster_bootstrap.csv", comparison_rows
    )
    table_3_csv_path = write_csv(
        output_dir / "table3_document_cluster_bootstrap.csv",
        table_3_rows,
    )
    json_path = write_json(
        output_dir / "w2_document_cluster_bootstrap_results.json",
        {
            "schema_version": 1,
            "analysis": "paired_document_cluster_bootstrap",
            "design": {
                "independent_resampling_unit": "source_document",
                "document_clusters": clustered_rows[0]["count_document_clusters"],
                "expected_document_clusters": expected_document_clusters,
                "base_questions_per_model": sum(
                    int(row["count_base_questions"])
                    for row in clustered_rows
                    if row["model_name"] == models[0]
                    and row["result_role"] == "primary_question_class"
                ),
                "fictional_variants_per_base_question": len(paper_manifest.variant_ids),
                "primary_question_types": list(PRIMARY_QUESTION_TYPES),
                "secondary_summary": REASON_SUMMARY,
            },
            "bootstrap": {
                "resamples": resamples,
                "base_seed": seed,
                "confidence_level": 0.95,
                "interval": "percentile",
                "test": CLUSTER_TEST,
                "test_statistic": CLUSTER_TEST_STATISTIC,
                "null_restriction": "subtract_observed_equal_document_mean_before_weighting",
            },
            "multiplicity": {
                "primary_family_within_model": 12,
                "global_primary_family": len(models) * 12,
                "method": "Holm-Bonferroni",
            },
            "code_provenance": {
                "revision": code_revision,
                "code_manifest_path": (
                    None if code_manifest_path is None else str(code_manifest_path)
                ),
                "code_manifest_sha256": code_manifest_sha256,
            },
            "table_2_rows": clustered_rows,
            "table_3_rows": table_3_rows,
        },
    )
    tex_path = write_text(
        output_dir / "table2_document_cluster_bootstrap.tex",
        render_clustered_table_2_latex(clustered_rows, models),
    )
    table_3_tex_path = write_text(
        output_dir / "table3_document_cluster_bootstrap.tex",
        render_clustered_table_3_latex(table_3_rows),
    )
    report_path = write_text(
        output_dir / "W2_STATISTICS_HARDENING_REPORT.md",
        render_w2_markdown_report(
            clustered_rows,
            table_3_rows,
            comparison_rows,
            models=models,
            resamples=resamples,
            seed=seed,
        ),
    )
    outputs = (
        csv_path,
        comparison_path,
        table_3_csv_path,
        json_path,
        tex_path,
        table_3_tex_path,
        report_path,
    )
    analysis_path = Path(__file__).resolve()
    artifact_manifest_path = write_artifact_manifest(
        paper_results_manifest=paper_manifest,
        artifact_name="rebuttal_tables_2_3_document_cluster_bootstrap",
        outputs=outputs,
        metadata={
            "models": list(models),
            "table_3_models": list(table_3_models),
            "comparison_setting": comparison_setting,
            "independent_resampling_unit": "source_document",
            "expected_document_clusters": expected_document_clusters,
            "nested_unit": "paired_base_question_mean_over_expected_variants",
            "primary_question_types": list(PRIMARY_QUESTION_TYPES),
            "reason_column_role": "secondary_summary_with_explicit_heterogeneity_caveat",
            "bootstrap_resamples": resamples,
            "bootstrap_base_seed": seed,
            "confidence_interval": "document_cluster_percentile_95",
            "test": CLUSTER_TEST,
            "test_statistic": CLUSTER_TEST_STATISTIC,
            "wild_cluster_null_restriction": (
                "subtract_observed_equal_document_mean_before_rademacher_weighting"
            ),
            "multiplicity": "Holm_within_model_12_and_global_primary_sensitivity",
            "code_revision": code_revision,
            "code_manifest": {
                "path": None if code_manifest_path is None else str(code_manifest_path),
                "sha256": code_manifest_sha256,
            },
            "analysis_code": {
                "path": str(analysis_path),
                "sha256": sha256_file(analysis_path),
            },
        },
        output_path=output_dir / "w2_document_cluster_bootstrap.manifest.json",
    )
    checksum_path = _write_checksum_manifest(
        output_dir,
        (*outputs, artifact_manifest_path),
    )
    complete_path = output_dir / "CLUSTER_BOOTSTRAP_COMPLETE"
    complete_path.write_text(
        f"complete resamples={resamples} seed={seed}\n",
        encoding="utf-8",
    )
    primary_rows = [
        row for row in clustered_rows if row["result_role"] == "primary_question_class"
    ]
    return {
        "status": "complete",
        "output_dir": str(output_dir),
        "manifest": str(artifact_manifest_path),
        "checksums": str(checksum_path),
        "completion_marker": str(complete_path),
        "document_clusters": clustered_rows[0]["count_document_clusters"],
        "primary_tests": len(primary_rows),
        "secondary_table_3_models": len(table_3_rows),
        "significant_holm_within_model": sum(
            bool(row["significant_holm_within_model_0_05"]) for row in primary_rows
        ),
        "significant_holm_global": sum(
            bool(row["significant_holm_global_0_05"]) for row in primary_rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--table3-models", nargs="+", default=None)
    parser.add_argument("--comparison-setting", default="fictional")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--expected-document-clusters", type=int, default=87)
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--code-manifest", type=Path, default=None)
    args = parser.parse_args()
    models = tuple(args.models or TABLES_2_AND_4_MODELS)
    table_3_models = tuple(args.table3_models or TABLE_3_CHAIN_OF_THOUGHT_MODELS)
    summary = run_analysis(
        manifest_path=args.manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        models=models,
        table_3_models=table_3_models,
        comparison_setting=args.comparison_setting,
        resamples=args.resamples,
        seed=args.seed,
        expected_document_clusters=args.expected_document_clusters,
        code_revision=args.code_revision,
        code_manifest_path=(
            None if args.code_manifest is None else args.code_manifest.expanduser().resolve()
        ),
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
