"""Compute the factual-to-fictional comparisons in paper Tables 2 and 3."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np
from scipy.stats import t

from .paper_results_data_model import (
    ANSWER_GROUPS,
    QUESTION_GROUPS,
    REASONING_TYPES,
    EvaluatedAnswer,
    PairedAccuracyStatistics,
)


def summarize_paired_observations_with_t_interval(values: Sequence[float]) -> PairedAccuracyStatistics:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot compute a t summary over zero observations.")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return PairedAccuracyStatistics(1, mean, 0.0, 0.0, mean, mean, 0.0, 1.0)
    standard_deviation = float(np.std(arr, ddof=1))
    standard_error = standard_deviation / sqrt(float(arr.size))
    degrees_of_freedom = int(arr.size - 1)
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        p_value = 1.0 if np.isclose(mean, 0.0) else 0.0
        return PairedAccuracyStatistics(
            int(arr.size), mean, standard_deviation, standard_error, mean, mean, 0.0, p_value
        )
    statistic = mean / standard_error
    margin = float(t.ppf(0.975, df=degrees_of_freedom) * standard_error)
    p_value = 2.0 * float(t.sf(abs(statistic), df=degrees_of_freedom))
    return PairedAccuracyStatistics(
        int(arr.size),
        mean,
        standard_deviation,
        standard_error,
        mean - margin,
        mean + margin,
        float(statistic),
        p_value,
    )


def _pair_factual_accuracy_with_mean_fictional_accuracy(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    model: str,
    comparison_setting: str,
    variant_ids: Sequence[str],
) -> list[tuple[EvaluatedAnswer, float]]:
    factual: dict[str, EvaluatedAnswer] = {}
    fictional: dict[str, dict[str, EvaluatedAnswer]] = defaultdict(dict)
    expected = set(variant_ids)
    for row in evaluated_answers:
        if row.model_name != model:
            continue
        if row.setting == "factual":
            factual[row.pair_key] = row
        elif row.setting == comparison_setting:
            fictional[row.pair_key][row.variant_id] = row

    missing_factual = sorted(set(fictional) - set(factual))
    if missing_factual:
        raise ValueError(f"{model}: fictional evaluated_answers have no factual baseline for {missing_factual[:5]}.")
    incomplete = {
        pair_key: sorted(expected - set(by_variant))
        for pair_key, by_variant in fictional.items()
        if set(by_variant) != expected
    }
    if incomplete:
        example_key = next(iter(incomplete))
        raise ValueError(
            f"{model}: incomplete {comparison_setting} variants for {example_key}: {incomplete[example_key]}."
        )
    factual_and_fictional_pairs: list[tuple[EvaluatedAnswer, float]] = []
    for pair_key in sorted(factual):
        by_variant = fictional.get(pair_key)
        if by_variant is None:
            raise ValueError(f"{model}: factual pair {pair_key} is missing {comparison_setting} variants.")
        factual_row = factual[pair_key]
        metadata_fields = [
            "document_theme",
            "document_id",
            "question_type",
            "answer_schema",
        ]
        for variant_id, variant_row in by_variant.items():
            metadata_mismatches = {
                field: (getattr(factual_row, field), getattr(variant_row, field))
                for field in metadata_fields
                if getattr(factual_row, field) != getattr(variant_row, field)
            }
            if metadata_mismatches:
                raise ValueError(
                    f"{model}/{pair_key}/{variant_id}: factual/{comparison_setting} "
                    f"metadata mismatch: {metadata_mismatches}."
                )
        comparison_behaviors = {row.answer_behavior for row in by_variant.values()}
        if comparison_behaviors != {factual_row.answer_behavior}:
            raise ValueError(
                f"{model}/{pair_key}: answer_behavior drift after human-contract validation; "
                f"factual={factual_row.answer_behavior!r}, "
                f"{comparison_setting}={sorted(comparison_behaviors)}."
            )
        values = [float(by_variant[variant_id].final_is_correct) for variant_id in variant_ids]
        factual_and_fictional_pairs.append((factual_row, float(np.mean(values))))
    if not factual_and_fictional_pairs:
        raise ValueError(f"{model}: no complete factual/{comparison_setting} pairs.")
    return factual_and_fictional_pairs


def compute_table_2_performance_drop(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    models: Sequence[str],
    comparison_setting: str,
    variant_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute paper Table 2 using one factual_and_fictional_pairs observation per base QA."""
    result_rows: list[dict[str, Any]] = []
    for model in models:
        factual_and_fictional_pairs = _pair_factual_accuracy_with_mean_fictional_accuracy(
            evaluated_answers, model=model, comparison_setting=comparison_setting, variant_ids=variant_ids
        )
        for answer_group, answer_members in ANSWER_GROUPS.items():
            for question_group, question_members in QUESTION_GROUPS.items():
                selected = [
                    (factual, fictional_mean)
                    for factual, fictional_mean in factual_and_fictional_pairs
                    if factual.answer_behavior in answer_members and factual.question_type in question_members
                ]
                if not selected:
                    raise ValueError(f"{model}: empty Table 2 cell {answer_group}/{question_group}.")
                factual_values = [float(factual.final_is_correct) for factual, _ in selected]
                fictional_values = [fictional_mean for _, fictional_mean in selected]
                differences = [
                    fictional - factual for factual, fictional in zip(factual_values, fictional_values, strict=True)
                ]
                summary = summarize_paired_observations_with_t_interval(differences)
                result_rows.append(
                    {
                        "model_name": model,
                        "answer_behavior": answer_group,
                        "question_type": question_group,
                        "count_pairs": summary.count,
                        "factual_accuracy_pp": 100.0 * float(np.mean(factual_values)),
                        "fictional_accuracy_pp": 100.0 * float(np.mean(fictional_values)),
                        "mean_difference_pp": 100.0 * summary.mean,
                        "ci95_half_width_pp": 100.0 * summary.ci_half_width,
                        "ci95_low_pp": 100.0 * summary.ci_low,
                        "ci95_high_pp": 100.0 * summary.ci_high,
                        "t_statistic": summary.t_statistic,
                        "p_value": summary.p_value,
                        "significant_paired_t_0_05": summary.p_value < 0.05,
                    }
                )
    return result_rows


def compute_table_3_chain_of_thought_comparison(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    models: Sequence[str],
    comparison_setting: str,
    variant_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Compute Table 3 over base reasoning QAs (not 10x pseudo-replicates)."""
    result_rows: list[dict[str, Any]] = []
    for model in models:
        factual_and_fictional_pairs = [
            item
            for item in _pair_factual_accuracy_with_mean_fictional_accuracy(
                evaluated_answers,
                model=model,
                comparison_setting=comparison_setting,
                variant_ids=variant_ids,
            )
            if item[0].question_type in REASONING_TYPES
        ]
        if not factual_and_fictional_pairs:
            raise ValueError(f"{model}: no reasoning pairs for Table 3.")
        factual_values = [float(factual.final_is_correct) for factual, _ in factual_and_fictional_pairs]
        fictional_values = [fictional for _, fictional in factual_and_fictional_pairs]
        factual_summary = summarize_paired_observations_with_t_interval(factual_values)
        fictional_summary = summarize_paired_observations_with_t_interval(fictional_values)
        delta_summary = summarize_paired_observations_with_t_interval(
            [fictional - factual for factual, fictional in zip(factual_values, fictional_values, strict=True)]
        )
        result_rows.append(
            {
                "model_name": model,
                "count_base_questions": len(factual_and_fictional_pairs),
                "factual_accuracy_pp": 100.0 * factual_summary.mean,
                "factual_ci95_half_width_pp": 100.0 * factual_summary.ci_half_width,
                "fictional_accuracy_pp": 100.0 * fictional_summary.mean,
                "fictional_ci95_half_width_pp": 100.0 * fictional_summary.ci_half_width,
                "delta_fictional_minus_factual_pp": 100.0 * delta_summary.mean,
                "delta_ci95_half_width_pp": 100.0 * delta_summary.ci_half_width,
            }
        )
    return result_rows
