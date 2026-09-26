"""Compute partial-replacement accuracy curves for paper Figures 3, 5, and 6."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import sqrt
from typing import Any

from .paper_results_data_model import ANSWER_BEHAVIORS, QUESTION_TYPES, EvaluatedAnswer
from .evaluated_model_answer_loading import replacement_proportion


def compute_partial_replacement_accuracy_curves(
    evaluated_answers: Sequence[EvaluatedAnswer],
    *,
    models: Sequence[str],
    settings: Sequence[str],
    group_by: str | None = None,
    variant_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate Figure 3/5/6 accuracies with 95% Wald intervals."""
    if group_by not in {None, "question_type", "answer_behavior"}:
        raise ValueError(f"Unsupported curve grouping: {group_by!r}")
    expected_variants = set(variant_ids or ())
    if variant_ids is not None and (not expected_variants or len(expected_variants) != len(tuple(variant_ids))):
        raise ValueError("variant_ids must contain distinct ids when curve scope is validated.")
    curve_rows: list[dict[str, Any]] = []
    for model in models:
        model_evaluated_answers = [row for row in evaluated_answers if row.model_name == model]
        if not model_evaluated_answers:
            raise ValueError(f"No curve evaluated_answers found for model {model}.")
        if variant_ids is not None:
            factual_evaluated_answers = [row for row in model_evaluated_answers if row.setting == "factual"]
            factual_by_pair = {row.pair_key: row for row in factual_evaluated_answers}
            factual_pair_keys = set(factual_by_pair)
            if not factual_pair_keys:
                raise ValueError(f"{model}: no factual curve baseline evaluated_answers.")
            if len(factual_evaluated_answers) != len(factual_by_pair):
                raise ValueError(f"{model}: duplicate factual curve evaluated_answers detected.")
            for setting in settings:
                setting_evaluated_answers = [row for row in model_evaluated_answers if row.setting == setting]
                setting_pair_keys = {row.pair_key for row in setting_evaluated_answers}
                missing_pairs = sorted(factual_pair_keys - setting_pair_keys)
                unexpected_pairs = sorted(setting_pair_keys - factual_pair_keys)
                if missing_pairs or unexpected_pairs:
                    raise ValueError(
                        f"{model}/{setting}: incomplete curve base-question scope; "
                        f"missing={missing_pairs[:5]}, unexpected={unexpected_pairs[:5]}."
                    )
                if setting == "factual":
                    if len(setting_evaluated_answers) != len(factual_pair_keys):
                        raise ValueError(f"{model}: duplicate factual curve evaluated_answers detected.")
                    continue
                metadata_fields = (
                    "document_theme",
                    "document_id",
                    "question_type",
                    "answer_schema",
                )
                for row in setting_evaluated_answers:
                    factual_row = factual_by_pair[row.pair_key]
                    metadata_mismatches = {
                        field: (getattr(factual_row, field), getattr(row, field))
                        for field in metadata_fields
                        if getattr(factual_row, field) != getattr(row, field)
                    }
                    if metadata_mismatches:
                        raise ValueError(
                            f"{model}/{setting}/{row.pair_key}/{row.variant_id}: "
                            f"factual/curve metadata mismatch: {metadata_mismatches}."
                        )
                variants_by_pair: dict[str, list[str]] = defaultdict(list)
                behaviors_by_pair: dict[str, set[str]] = defaultdict(set)
                for row in setting_evaluated_answers:
                    variants_by_pair[row.pair_key].append(row.variant_id)
                    behaviors_by_pair[row.pair_key].add(row.answer_behavior)
                incomplete_variants = {
                    pair_key: sorted(expected_variants - set(observed))
                    for pair_key, observed in variants_by_pair.items()
                    if set(observed) != expected_variants or len(observed) != len(expected_variants)
                }
                if incomplete_variants:
                    example_pair = next(iter(incomplete_variants))
                    raise ValueError(
                        f"{model}/{setting}: incomplete curve variants for {example_pair}: "
                        f"{incomplete_variants[example_pair]}."
                    )
                inconsistent_behaviors = {
                    pair_key: sorted(observed) for pair_key, observed in behaviors_by_pair.items() if len(observed) != 1
                }
                if inconsistent_behaviors:
                    example_pair = next(iter(inconsistent_behaviors))
                    raise ValueError(
                        f"{model}/{setting}: inconsistent answer_behavior across variants for "
                        f"{example_pair}: {inconsistent_behaviors[example_pair]}."
                    )
        if group_by == "question_type":
            groups = ("all", *QUESTION_TYPES)
        elif group_by == "answer_behavior":
            groups = ("all", *ANSWER_BEHAVIORS)
        else:
            groups = ("all",)
        for group in groups:
            for setting in settings:
                selected = [row for row in model_evaluated_answers if row.setting == setting]
                if group_by == "question_type" and group != "all":
                    selected = [row for row in selected if row.question_type == group]
                if group_by == "answer_behavior" and group != "all":
                    selected = [row for row in selected if row.answer_behavior == group]
                if not selected:
                    raise ValueError(f"Missing curve cell: model={model}, group={group}, setting={setting}.")
                total = len(selected)
                correct = sum(row.final_is_correct for row in selected)
                accuracy = correct / total
                margin = 1.96 * sqrt(accuracy * (1.0 - accuracy) / total)
                record: dict[str, Any] = {
                    "model_name": model,
                    "document_setting": setting,
                    "replacement_proportion": replacement_proportion(setting),
                    "count_total": total,
                    "count_correct": correct,
                    "accuracy_pp": 100.0 * accuracy,
                    "ci95_wald_low_pp": 100.0 * max(0.0, accuracy - margin),
                    "ci95_wald_high_pp": 100.0 * min(1.0, accuracy + margin),
                }
                if group_by:
                    record[group_by] = group
                curve_rows.append(record)
    return curve_rows
