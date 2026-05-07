"""Aggregate evaluated outputs into setting-aware accuracy and conversion metrics."""

from __future__ import annotations

from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path

import yaml
from statsmodels.stats.proportion import proportion_confint

from src.dataset_export.dataset_settings import order_dataset_settings


_CONVERSION_CATEGORIES = (
    "correct_to_correct",
    "correct_to_incorrect",
    "incorrect_to_correct",
    "incorrect_to_incorrect",
)


def _accuracy_summary(rows: Iterable[dict]) -> dict[str, float | int]:
    rows = list(rows)
    total = len(rows)
    correct = sum(1 for row in rows if row.get("final_is_correct") is True)
    if total == 0:
        return {
            "count_correct": 0,
            "count_total": 0,
            "accuracy": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    lower, upper = proportion_confint(count=correct, nobs=total, alpha=0.05, method="wilson")
    return {
        "count_correct": correct,
        "count_total": total,
        "accuracy": correct / total,
        "ci_low": float(lower),
        "ci_high": float(upper),
    }


def _conversion_summary(pairs: Iterable[dict], category: str) -> dict[str, float | int]:
    pairs = list(pairs)
    total = len(pairs)
    count = sum(1 for pair in pairs if pair.get("conversion_category") == category)
    if total == 0:
        return {
            "count": 0,
            "total_pairs": 0,
            "ratio": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
        }
    lower, upper = proportion_confint(count=count, nobs=total, alpha=0.05, method="wilson")
    return {
        "count": count,
        "total_pairs": total,
        "ratio": count / total,
        "ci_low": float(lower),
        "ci_high": float(upper),
    }


def _setting_ids(rows: list[dict]) -> list[str]:
    return [
        spec.setting_id for spec in order_dataset_settings([str(row.get("document_setting") or "") for row in rows])
    ]


def _group_accuracy_by_setting(rows: list[dict], setting_ids: list[str]) -> dict[str, dict[str, float | int]]:
    return {
        setting_id: _accuracy_summary(row for row in rows if row.get("document_setting") == setting_id)
        for setting_id in setting_ids
    }


def _group_accuracy(
    rows: list[dict],
    group_key: str,
    setting_ids: list[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row.get(group_key) or "unknown")].append(row)
    return {
        group_value: _group_accuracy_by_setting(group_rows, setting_ids)
        for group_value, group_rows in sorted(grouped_rows.items())
    }


def _group_accuracy_nested(
    rows: list[dict],
    setting_ids: list[str],
) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    nested_rows: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        question_type = str(row.get("question_type") or "unknown")
        answer_behavior = str(row.get("answer_behavior") or "unknown")
        nested_rows[question_type][answer_behavior].append(row)

    output: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for question_type, answer_groups in sorted(nested_rows.items()):
        output[question_type] = {}
        for answer_behavior, grouped_rows in sorted(answer_groups.items()):
            output[question_type][answer_behavior] = _group_accuracy_by_setting(grouped_rows, setting_ids)
    return output


def _build_conversion_pairs_by_setting(rows: list[dict]) -> dict[str, list[dict]]:
    rows_by_pair_key_and_setting: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        pair_key = str(row.get("pair_key") or "")
        setting_id = str(row.get("document_setting") or "")
        rows_by_pair_key_and_setting[pair_key][setting_id].append(row)

    pairs_by_setting: dict[str, list[dict]] = defaultdict(list)
    for pair_key, setting_rows in rows_by_pair_key_and_setting.items():
        factual_rows = setting_rows.get("factual") or []
        if not factual_rows:
            continue
        factual_row = factual_rows[0]

        for setting_id, compared_rows in setting_rows.items():
            if setting_id == "factual":
                continue

            for compared_row in compared_rows:
                factual_correct = bool(factual_row.get("final_is_correct"))
                compared_correct = bool(compared_row.get("final_is_correct"))
                if factual_correct and compared_correct:
                    category = "correct_to_correct"
                elif factual_correct and not compared_correct:
                    category = "correct_to_incorrect"
                elif not factual_correct and compared_correct:
                    category = "incorrect_to_correct"
                else:
                    category = "incorrect_to_incorrect"

                pairs_by_setting[setting_id].append(
                    {
                        "pair_key": pair_key,
                        "document_theme": factual_row.get("document_theme", "unknown"),
                        "document_id": factual_row.get("document_id", "unknown"),
                        "document_variant_id": compared_row.get("document_variant_id", "v01"),
                        "question_type": factual_row.get("question_type", "unknown"),
                        "answer_behavior": factual_row.get("answer_behavior", "unknown"),
                        "conversion_category": category,
                    }
                )
    return pairs_by_setting


def _group_conversion(
    pairs: list[dict],
    group_key: str,
) -> dict[str, dict[str, dict[str, float | int]]]:
    grouped_pairs: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped_pairs[str(pair.get(group_key) or "unknown")].append(pair)
    return {
        group_value: {category: _conversion_summary(group_rows, category) for category in _CONVERSION_CATEGORIES}
        for group_value, group_rows in sorted(grouped_pairs.items())
    }


def _group_conversion_nested(pairs: list[dict]) -> dict[str, dict[str, dict[str, dict[str, float | int]]]]:
    nested_pairs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for pair in pairs:
        question_type = str(pair.get("question_type") or "unknown")
        answer_behavior = str(pair.get("answer_behavior") or "unknown")
        nested_pairs[question_type][answer_behavior].append(pair)

    output: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for question_type, answer_groups in sorted(nested_pairs.items()):
        output[question_type] = {}
        for answer_behavior, grouped_pairs in sorted(answer_groups.items()):
            output[question_type][answer_behavior] = {
                category: _conversion_summary(grouped_pairs, category) for category in _CONVERSION_CATEGORIES
            }
    return output


def _conversion_metrics(pairs: list[dict]) -> dict:
    theme_grouped_pairs: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        theme_grouped_pairs[str(pair.get("document_theme") or "unknown")].append(pair)

    return {
        "overall": {category: _conversion_summary(pairs, category) for category in _CONVERSION_CATEGORIES},
        "by_theme": {
            theme: {category: _conversion_summary(group_pairs, category) for category in _CONVERSION_CATEGORIES}
            for theme, group_pairs in sorted(theme_grouped_pairs.items())
        },
        "by_question_type": _group_conversion(pairs, "question_type"),
        "by_answer_behavior": _group_conversion(pairs, "answer_behavior"),
        "by_question_type_and_answer_behavior": _group_conversion_nested(pairs),
    }


def flatten_evaluated_payload(evaluated_payload: dict) -> list[dict]:
    """Flatten an evaluated-output payload into row records."""
    rows: list[dict] = []
    for result in evaluated_payload.get("results", []) or []:
        rows.append(
            {
                "model_name": evaluated_payload.get("model_name"),
                "document_theme": evaluated_payload.get("document_theme"),
                "document_id": evaluated_payload.get("document_id"),
                "document_setting": evaluated_payload.get("document_setting"),
                "document_setting_family": evaluated_payload.get("document_setting_family"),
                "document_variant_id": evaluated_payload.get("document_variant_id"),
                "document_variant_index": evaluated_payload.get("document_variant_index"),
                "replacement_proportion": evaluated_payload.get("replacement_proportion"),
                **result,
            }
        )
    return rows


def compute_metrics_for_model(model_name: str, evaluated_payloads: list[dict]) -> dict:
    """Compute all metrics for one model."""
    all_rows: list[dict] = []
    for payload in evaluated_payloads:
        all_rows.extend(flatten_evaluated_payload(payload))

    setting_specs = order_dataset_settings([str(row.get("document_setting") or "") for row in all_rows])
    setting_ids = [spec.setting_id for spec in setting_specs]
    conversion_pairs_by_setting = _build_conversion_pairs_by_setting(all_rows)
    source_run_ids = sorted(
        {
            str((payload.get("reproducibility_manifest") or {}).get("run_id") or "").strip()
            for payload in evaluated_payloads
            if str((payload.get("reproducibility_manifest") or {}).get("run_id") or "").strip()
        }
    )
    source_providers = sorted(
        {
            str(payload.get("model_provider") or "").strip()
            for payload in evaluated_payloads
            if str(payload.get("model_provider") or "").strip()
        }
    )
    source_provider_models = sorted(
        {
            str(payload.get("provider_model_name") or "").strip()
            for payload in evaluated_payloads
            if str(payload.get("provider_model_name") or "").strip()
        }
    )

    theme_grouped_rows: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        theme_grouped_rows[str(row.get("document_theme") or "unknown")].append(row)

    return {
        "model_name": model_name,
        "source_runs": source_run_ids,
        "model_providers": source_providers,
        "provider_model_names": source_provider_models,
        "document_settings": {spec.setting_id: spec.to_payload() for spec in setting_specs},
        "overall_accuracy": _group_accuracy_by_setting(all_rows, setting_ids),
        "accuracy_by_theme": {
            theme: _group_accuracy_by_setting(group_rows, setting_ids)
            for theme, group_rows in sorted(theme_grouped_rows.items())
        },
        "accuracy_by_question_type": _group_accuracy(all_rows, "question_type", setting_ids),
        "accuracy_by_answer_behavior": _group_accuracy(all_rows, "answer_behavior", setting_ids),
        "accuracy_by_question_type_and_answer_behavior": _group_accuracy_nested(all_rows, setting_ids),
        "conversion_from_factual": {
            spec.setting_id: _conversion_metrics(conversion_pairs_by_setting.get(spec.setting_id, []))
            for spec in setting_specs
            if not spec.is_factual
        },
    }


def load_metrics_payload(metrics_path: Path) -> dict:
    """Load a metrics YAML file."""
    return yaml.safe_load(metrics_path.read_text(encoding="utf-8")) or {}
