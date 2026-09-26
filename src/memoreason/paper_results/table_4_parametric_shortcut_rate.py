"""Compute paper Table 4's strict canonical Parametric Shortcut Rate."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .paper_results_data_model import QUESTION_GROUPS, PaperResultsManifest, EvaluatedAnswer


def compute_table_4_parametric_shortcut_rate(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    paper_results_manifest: PaperResultsManifest,
    models: Sequence[str],
    comparison_setting: str,
    missing_cache_policy: str = "error",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute strict copying from parsed canonical final answers only.

    A copy is counted only when the parsed canonical prediction is exactly one
    of the paired factual question's canonical accepted answers.  Raw reasoning,
    semantic Judge Match, and legacy ``factual_answer_match`` fields are outside
    this diagnostic's estimand.
    """
    if missing_cache_policy not in {"error", "negative"}:
        raise ValueError("missing_cache_policy must be error or negative.")
    expected_variants = set(paper_results_manifest.variant_ids)
    by_factual: dict[tuple[str, str], EvaluatedAnswer] = {
        (row.model_name, row.pair_key): row for row in evaluated_answers if row.setting == "factual"
    }
    comparison_groups: dict[tuple[str, str], list[EvaluatedAnswer]] = defaultdict(list)
    model_set = set(models)
    for row in evaluated_answers:
        if row.setting == comparison_setting and row.model_name in model_set:
            comparison_groups[(row.model_name, row.pair_key)].append(row)

    predictions: dict[tuple[str, str], list[EvaluatedAnswer]] = {}
    for key, group in comparison_groups.items():
        behaviors = {row.answer_behavior for row in group}
        if len(behaviors) != 1:
            raise ValueError(
                f"{key[0]}/{key[1]}: inconsistent {comparison_setting} answer_behavior "
                f"across variants: {sorted(behaviors)}."
            )
        if behaviors == {"variant"}:
            predictions[key] = group

    # Human-reviewed contracts are authoritative for answer behavior.  The
    # factual-only fallback solely detects a deleted comparison group.
    factual_variant_pair_keys = {
        (row.model_name, row.pair_key)
        for row in evaluated_answers
        if row.setting == "factual" and row.model_name in model_set and row.answer_behavior == "variant"
    }
    expected_pair_keys = set(predictions) | (factual_variant_pair_keys - set(comparison_groups))
    missing_prediction_groups = sorted(expected_pair_keys - set(predictions))
    unexpected_prediction_groups = sorted(set(predictions) - expected_pair_keys)
    if missing_prediction_groups or unexpected_prediction_groups:
        raise ValueError(
            f"Incomplete Table 4 base-question scope: missing={missing_prediction_groups[:5]}, "
            f"unexpected={unexpected_prediction_groups[:5]}."
        )

    question_rows: list[dict[str, Any]] = []
    strict_canonical_matches = 0
    for (model, pair_key), group in sorted(predictions.items()):
        variant_set = {row.variant_id for row in group}
        if variant_set != expected_variants:
            raise ValueError(
                f"{model}/{pair_key}: expected variants {sorted(expected_variants)}, got {sorted(variant_set)}."
            )
        factual = by_factual.get((model, pair_key))
        if factual is None:
            raise ValueError(f"{model}/{pair_key}: missing factual row for Table 4.")
        for prediction in group:
            metadata_mismatches = {
                field: (getattr(factual, field), getattr(prediction, field))
                for field in (
                    "document_theme",
                    "document_id",
                    "question_type",
                    "answer_schema",
                )
                if getattr(factual, field) != getattr(prediction, field)
            }
            if metadata_mismatches:
                raise ValueError(
                    f"{model}/{pair_key}/{prediction.variant_id}: factual/{comparison_setting} "
                    f"metadata mismatch: {metadata_mismatches}."
                )
        failures = 0
        matches = 0
        for prediction in group:
            if prediction.final_is_correct:
                continue
            failures += 1
            predicted_canonical = prediction.parsed_output_canonical.strip()
            copied = bool(predicted_canonical) and predicted_canonical in set(
                factual.accepted_answers_canonical
            )
            if copied:
                strict_canonical_matches += 1
                matches += 1
        # The submitted table macro-averages over every variant base QA.  A QA
        # with no failures contributes 0.0 (the convention used by the paper
        # script), while the parenthesized count remains the number of failed
        # fictional examples only.
        question_rows.append(
            {
                "model_name": model,
                "pair_key": pair_key,
                "question_type": factual.question_type,
                "failcase_count": failures,
                "factual_answer_match_count": matches,
                "shortcut_rate_among_failures": matches / failures if failures else 0.0,
            }
        )

    summary: list[dict[str, Any]] = []
    for model in models:
        model_rows = [row for row in question_rows if row["model_name"] == model]
        for question_group, members in QUESTION_GROUPS.items():
            selected = [row for row in model_rows if row["question_type"] in members]
            if not selected:
                raise ValueError(f"{model}: empty Table 4 cell {question_group}.")
            rates = [float(row["shortcut_rate_among_failures"]) for row in selected]
            summary.append(
                {
                    "model_name": model,
                    "question_type": question_group,
                    "count_base_questions": len(selected),
                    "count_questions_with_failures": sum(int(row["failcase_count"] > 0) for row in selected),
                    "total_failcases": sum(int(row["failcase_count"]) for row in selected),
                    "mean_shortcut_rate_pp": 100.0 * float(np.mean(rates)),
                }
            )
    diagnostics = {
        "missing_cache_policy": missing_cache_policy,
        "matching_rule": "parsed_output_canonical_exactly_matches_factual_accepted_answer_canonical",
        "semantic_judge_used": False,
        "raw_output_used": False,
        "legacy_factual_answer_match_used": False,
        "strict_canonical_matches": strict_canonical_matches,
        "deterministic_exact_hits": strict_canonical_matches,
        "frozen_cache_hits": 0,
        "missing_cache_entries": 0,
        "question_rows": question_rows,
    }
    return summary, diagnostics
