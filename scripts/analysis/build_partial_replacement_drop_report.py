#!/usr/bin/env python3
"""Report Parametric Shortcut performance drops across partial replacement settings."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from src.dataset_export.dataset_paths import MODEL_EVAL_PLOTS_DIR, MODEL_EVAL_RAW_OUTPUTS_DIR
from src.dataset_export.dataset_settings import fictional_setting, order_dataset_settings, parse_dataset_setting

DEFAULT_MODELS = (
    "olmo-3-7b-instruct",
    "gpt-oss-120b-groq",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
)
DEFAULT_PROPORTIONS = (0.1, 0.2, 0.3, 0.5, 0.8, 0.9)
DEFAULT_OUTPUT_DIR = MODEL_EVAL_PLOTS_DIR / "partial_fictional_replacements"
DEFAULT_EXPECTED_FACTUAL_SCORE_COUNT = 1044
DEFAULT_EXPECTED_FICTIONAL_SCORE_COUNT = 10440
QUESTION_TYPE_ORDER = ("all", "extractive", "arithmetic", "temporal", "inference")
ANSWER_TYPE_ORDER = ("all", "variant", "invariant", "refusal")

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


@dataclass(frozen=True)
class ScoreRow:
    model_name: str
    document_setting: str
    replacement_proportion: float
    question_type: str
    answer_type: str
    pair_key: str
    exact_match: bool
    judge_match: bool | None
    final_is_correct: bool


def _read_yaml_payload(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}


def _settings_for_report(proportions: tuple[float, ...], *, include_full_fictional: bool) -> list[str]:
    settings = [fictional_setting(proportion).setting_id for proportion in proportions]
    if include_full_fictional:
        settings.append("fictional")
    return [spec.setting_id for spec in order_dataset_settings(["factual", *settings])]


def _iter_score_rows(*, models: set[str], settings: set[str]) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    for path in sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.rglob("*_evaluated_outputs.yaml")):
        payload = _read_yaml_payload(path)
        model_name = str(payload.get("model_name") or path.parent.name)
        document_setting = str(payload.get("document_setting") or "").strip().lower()
        if model_name not in models or document_setting not in settings:
            continue
        replacement_proportion = float(
            payload.get("replacement_proportion")
            or parse_dataset_setting(document_setting).replacement_proportion
        )
        for result in payload.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            pair_key = str(result.get("pair_key") or "").strip()
            if not pair_key:
                continue
            rows.append(
                ScoreRow(
                    model_name=model_name,
                    document_setting=document_setting,
                    replacement_proportion=replacement_proportion,
                    question_type=str(result.get("question_type") or "").strip().lower(),
                    answer_type=str(
                        result.get("answer_type") or result.get("answer_behavior") or ""
                    ).strip().lower(),
                    pair_key=pair_key,
                    exact_match=bool(result.get("exact_match")),
                    judge_match=(
                        None if result.get("judge_match") is None else bool(result.get("judge_match"))
                    ),
                    final_is_correct=bool(result.get("final_is_correct")),
                )
            )
    return rows


def _accuracy(rows: list[ScoreRow], attr: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if bool(getattr(row, attr))) / len(rows)


def _conversion_counts(setting_rows: list[ScoreRow], factual_by_pair: dict[str, bool]) -> dict[str, int]:
    counts = {
        "correct_to_correct": 0,
        "correct_to_incorrect": 0,
        "incorrect_to_correct": 0,
        "incorrect_to_incorrect": 0,
        "missing_factual_pair": 0,
    }
    for row in setting_rows:
        factual_correct = factual_by_pair.get(row.pair_key)
        if factual_correct is None:
            counts["missing_factual_pair"] += 1
            continue
        if factual_correct and row.final_is_correct:
            counts["correct_to_correct"] += 1
        elif factual_correct and not row.final_is_correct:
            counts["correct_to_incorrect"] += 1
        elif not factual_correct and row.final_is_correct:
            counts["incorrect_to_correct"] += 1
        else:
            counts["incorrect_to_incorrect"] += 1
    return counts


def _build_model_rows_for_question_type(
    score_rows: list[ScoreRow],
    *,
    models: tuple[str, ...],
    settings: list[str],
    question_type: str | None = None,
    include_question_type: bool = False,
) -> list[dict[str, Any]]:
    if question_type is not None:
        score_rows = [row for row in score_rows if row.question_type == question_type]

    rows_by_model_setting: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in score_rows:
        rows_by_model_setting[(row.model_name, row.document_setting)].append(row)

    report_rows: list[dict[str, Any]] = []
    for model_name in models:
        factual_rows = rows_by_model_setting.get((model_name, "factual"), [])
        factual_accuracy = _accuracy(factual_rows, "final_is_correct")
        factual_by_pair = {row.pair_key: row.final_is_correct for row in factual_rows}
        for setting in settings:
            if setting == "factual":
                continue
            setting_rows = rows_by_model_setting.get((model_name, setting), [])
            accuracy = _accuracy(setting_rows, "final_is_correct")
            exact_accuracy = _accuracy(setting_rows, "exact_match")
            judge_used = sum(1 for row in setting_rows if row.judge_match is not None)
            judge_correct = sum(1 for row in setting_rows if row.judge_match is True)
            missing_judge_for_non_exact = sum(
                1 for row in setting_rows if not row.exact_match and row.judge_match is None
            )
            judge_present_for_exact = sum(1 for row in setting_rows if row.exact_match and row.judge_match is not None)
            inconsistent_final_score = sum(
                1
                for row in setting_rows
                if row.final_is_correct != (row.exact_match or row.judge_match is True)
            )
            conversion_counts = _conversion_counts(setting_rows, factual_by_pair)
            paired_total = len(setting_rows) - conversion_counts["missing_factual_pair"]
            drop = factual_accuracy - accuracy
            record = {
                "model_name": model_name,
                "document_setting": setting,
                "replacement_proportion": parse_dataset_setting(setting).replacement_proportion,
                "count_total": len(setting_rows),
                "count_correct": sum(1 for row in setting_rows if row.final_is_correct),
                "accuracy": accuracy,
                "accuracy_pp": accuracy * 100.0,
                "exact_match_accuracy": exact_accuracy,
                "exact_match_accuracy_pp": exact_accuracy * 100.0,
                "judge_used_count": judge_used,
                "judge_correct_count": judge_correct,
                "missing_judge_for_non_exact_count": missing_judge_for_non_exact,
                "judge_present_for_exact_count": judge_present_for_exact,
                "inconsistent_final_score_count": inconsistent_final_score,
                "factual_count_total": len(factual_rows),
                "factual_accuracy": factual_accuracy,
                "factual_accuracy_pp": factual_accuracy * 100.0,
                "drop_factual_minus_setting": drop,
                "drop_factual_minus_setting_pp": drop * 100.0,
                "paired_total": paired_total,
                "correct_to_incorrect_count": conversion_counts["correct_to_incorrect"],
                "correct_to_incorrect_ratio": (
                    conversion_counts["correct_to_incorrect"] / paired_total if paired_total else 0.0
                ),
                "incorrect_to_correct_count": conversion_counts["incorrect_to_correct"],
                "incorrect_to_correct_ratio": (
                    conversion_counts["incorrect_to_correct"] / paired_total if paired_total else 0.0
                ),
                "missing_factual_pair_count": conversion_counts["missing_factual_pair"],
            }
            if include_question_type:
                record = {
                    "question_type": question_type or "all",
                    **record,
                }
            report_rows.append(record)
    return report_rows


def _build_model_rows(score_rows: list[ScoreRow], *, models: tuple[str, ...], settings: list[str]) -> list[dict[str, Any]]:
    return _build_model_rows_for_question_type(score_rows, models=models, settings=settings)


def _build_question_type_model_rows(
    score_rows: list[ScoreRow],
    *,
    models: tuple[str, ...],
    settings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_type in QUESTION_TYPE_ORDER:
        rows.extend(
            _build_model_rows_for_question_type(
                score_rows,
                models=models,
                settings=settings,
                question_type=None if question_type == "all" else question_type,
                include_question_type=True,
            )
        )
    return rows


def _build_model_rows_for_answer_type(
    score_rows: list[ScoreRow],
    *,
    models: tuple[str, ...],
    settings: list[str],
    answer_type: str | None = None,
    include_answer_type: bool = False,
) -> list[dict[str, Any]]:
    if answer_type is not None:
        score_rows = [row for row in score_rows if row.answer_type == answer_type]

    rows_by_model_setting: dict[tuple[str, str], list[ScoreRow]] = defaultdict(list)
    for row in score_rows:
        rows_by_model_setting[(row.model_name, row.document_setting)].append(row)

    report_rows: list[dict[str, Any]] = []
    for model_name in models:
        factual_rows = rows_by_model_setting.get((model_name, "factual"), [])
        factual_accuracy = _accuracy(factual_rows, "final_is_correct")
        factual_by_pair = {row.pair_key: row.final_is_correct for row in factual_rows}
        for setting in settings:
            if setting == "factual":
                continue
            setting_rows = rows_by_model_setting.get((model_name, setting), [])
            accuracy = _accuracy(setting_rows, "final_is_correct")
            exact_accuracy = _accuracy(setting_rows, "exact_match")
            judge_used = sum(1 for row in setting_rows if row.judge_match is not None)
            judge_correct = sum(1 for row in setting_rows if row.judge_match is True)
            missing_judge_for_non_exact = sum(
                1 for row in setting_rows if not row.exact_match and row.judge_match is None
            )
            judge_present_for_exact = sum(1 for row in setting_rows if row.exact_match and row.judge_match is not None)
            inconsistent_final_score = sum(
                1
                for row in setting_rows
                if row.final_is_correct != (row.exact_match or row.judge_match is True)
            )
            conversion_counts = _conversion_counts(setting_rows, factual_by_pair)
            paired_total = len(setting_rows) - conversion_counts["missing_factual_pair"]
            drop = factual_accuracy - accuracy
            record = {
                "model_name": model_name,
                "document_setting": setting,
                "replacement_proportion": parse_dataset_setting(setting).replacement_proportion,
                "count_total": len(setting_rows),
                "count_correct": sum(1 for row in setting_rows if row.final_is_correct),
                "accuracy": accuracy,
                "accuracy_pp": accuracy * 100.0,
                "exact_match_accuracy": exact_accuracy,
                "exact_match_accuracy_pp": exact_accuracy * 100.0,
                "judge_used_count": judge_used,
                "judge_correct_count": judge_correct,
                "missing_judge_for_non_exact_count": missing_judge_for_non_exact,
                "judge_present_for_exact_count": judge_present_for_exact,
                "inconsistent_final_score_count": inconsistent_final_score,
                "factual_count_total": len(factual_rows),
                "factual_accuracy": factual_accuracy,
                "factual_accuracy_pp": factual_accuracy * 100.0,
                "drop_factual_minus_setting": drop,
                "drop_factual_minus_setting_pp": drop * 100.0,
                "paired_total": paired_total,
                "correct_to_incorrect_count": conversion_counts["correct_to_incorrect"],
                "correct_to_incorrect_ratio": (
                    conversion_counts["correct_to_incorrect"] / paired_total if paired_total else 0.0
                ),
                "incorrect_to_correct_count": conversion_counts["incorrect_to_correct"],
                "incorrect_to_correct_ratio": (
                    conversion_counts["incorrect_to_correct"] / paired_total if paired_total else 0.0
                ),
                "missing_factual_pair_count": conversion_counts["missing_factual_pair"],
            }
            if include_answer_type:
                record = {
                    "answer_type": answer_type or "all",
                    **record,
                }
            report_rows.append(record)
    return report_rows


def _build_answer_type_model_rows(
    score_rows: list[ScoreRow],
    *,
    models: tuple[str, ...],
    settings: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for answer_type in ANSWER_TYPE_ORDER:
        rows.extend(
            _build_model_rows_for_answer_type(
                score_rows,
                models=models,
                settings=settings,
                answer_type=None if answer_type == "all" else answer_type,
                include_answer_type=True,
            )
        )
    return rows


def _build_summary_rows(model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_setting: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        if int(row["count_total"]) > 0:
            rows_by_setting[str(row["document_setting"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for setting in [spec.setting_id for spec in order_dataset_settings(rows_by_setting.keys())]:
        rows = rows_by_setting[setting]
        summary_rows.append(
            {
                "document_setting": setting,
                "replacement_proportion": parse_dataset_setting(setting).replacement_proportion,
                "model_count": len(rows),
                "mean_accuracy_pp": mean(float(row["accuracy_pp"]) for row in rows),
                "mean_factual_accuracy_pp": mean(float(row["factual_accuracy_pp"]) for row in rows),
                "mean_drop_factual_minus_setting_pp": mean(
                    float(row["drop_factual_minus_setting_pp"]) for row in rows
                ),
                "mean_correct_to_incorrect_ratio": mean(
                    float(row["correct_to_incorrect_ratio"]) for row in rows
                ),
            }
        )
    return summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_markdown(summary_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], models: tuple[str, ...]) -> str:
    model_drop_by_setting = {
        (str(row["document_setting"]), str(row["model_name"])): float(row["drop_factual_minus_setting_pp"])
        for row in model_rows
    }
    lines = [
        "# Partial Fictional Replacement Drop Report",
        "",
        "Positive drop means factual accuracy minus partial-fictional accuracy, in percentage points.",
        "",
        "| Proportion | Mean drop pp | Mean accuracy pp | Models with data | "
        + " | ".join(models)
        + " |",
        "|---:|---:|---:|---:|" + "---:|" * len(models),
    ]
    for row in summary_rows:
        setting = str(row["document_setting"])
        model_cells = [
            f"{model_drop_by_setting[(setting, model)]:.2f}"
            if (setting, model) in model_drop_by_setting
            else "NA"
            for model in models
        ]
        lines.append(
            f"| {float(row['replacement_proportion']):.1f} | "
            f"{float(row['mean_drop_factual_minus_setting_pp']):.2f} | "
            f"{float(row['mean_accuracy_pp']):.2f} | "
            f"{int(row['model_count'])} | "
            + " | ".join(model_cells)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build partial replacement performance-drop tables.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help="Model registry names to include.")
    parser.add_argument(
        "--proportions",
        nargs="+",
        type=float,
        default=list(DEFAULT_PROPORTIONS),
        help="Partial replacement proportions to report.",
    )
    parser.add_argument(
        "--include-full-fictional",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the 100% fictional setting as a reference row.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allow-missing", action="store_true", help="Write partial reports even when rows are missing.")
    parser.add_argument(
        "--expected-score-count",
        type=int,
        default=DEFAULT_EXPECTED_FICTIONAL_SCORE_COUNT,
        help=(
            "Expected number of scored QA rows for each in-scope fictional setting/model pair. "
            "The default matches the current 87-document benchmark after excluding bankreg_11."
        ),
    )
    parser.add_argument(
        "--expected-factual-score-count",
        type=int,
        default=DEFAULT_EXPECTED_FACTUAL_SCORE_COUNT,
        help=(
            "Expected number of scored QA rows for each in-scope factual/model pair. "
            "The default matches the current 87-document factual benchmark after excluding bankreg_11."
        ),
    )
    args = parser.parse_args()

    models = tuple(args.models)
    proportions = tuple(float(proportion) for proportion in args.proportions)
    settings = _settings_for_report(proportions, include_full_fictional=args.include_full_fictional)
    score_rows = _iter_score_rows(models=set(models), settings=set(settings))
    model_rows = _build_model_rows(score_rows, models=models, settings=settings)
    question_type_model_rows = _build_question_type_model_rows(score_rows, models=models, settings=settings)
    answer_type_model_rows = _build_answer_type_model_rows(score_rows, models=models, settings=settings)

    missing = [
        {"model_name": row["model_name"], "document_setting": row["document_setting"]}
        for row in model_rows
        if int(row["count_total"]) == 0
    ]
    if missing and not args.allow_missing:
        rendered = ", ".join(f"{item['model_name']}:{item['document_setting']}" for item in missing[:20])
        raise SystemExit(
            f"Missing evaluated outputs for {len(missing)} model/setting pair(s): {rendered}. "
            "Use --allow-missing to write an incomplete diagnostic report."
        )
    incomplete = [
        {
            "model_name": row["model_name"],
            "document_setting": row["document_setting"],
            "count_total": int(row["count_total"]),
            "factual_count_total": int(row["factual_count_total"]),
            "expected_score_count": int(args.expected_score_count),
            "expected_factual_score_count": int(args.expected_factual_score_count),
        }
        for row in model_rows
        if int(row["count_total"]) not in {0, int(args.expected_score_count)}
        or int(row["factual_count_total"]) not in {0, int(args.expected_factual_score_count)}
    ]
    if incomplete and not args.allow_missing:
        rendered = ", ".join(
            f"{item['model_name']}:{item['document_setting']}="
            f"{item['count_total']}/{item['expected_score_count']} setting rows, "
            f"{item['factual_count_total']}/{item['expected_factual_score_count']} factual rows"
            for item in incomplete[:20]
        )
        raise SystemExit(
            f"Incomplete evaluated outputs for {len(incomplete)} model/setting pair(s): {rendered}. "
            "Use --allow-missing only for diagnostics."
        )
    protocol_violations = [
        {
            "model_name": row["model_name"],
            "document_setting": row["document_setting"],
            "missing_judge_for_non_exact_count": int(row["missing_judge_for_non_exact_count"]),
            "judge_present_for_exact_count": int(row["judge_present_for_exact_count"]),
            "inconsistent_final_score_count": int(row["inconsistent_final_score_count"]),
        }
        for row in model_rows
        if int(row["missing_judge_for_non_exact_count"])
        or int(row["judge_present_for_exact_count"])
        or int(row["inconsistent_final_score_count"])
    ]
    if protocol_violations and not args.allow_missing:
        rendered = ", ".join(
            f"{item['model_name']}:{item['document_setting']}="
            f"missing_judge_for_non_exact:{item['missing_judge_for_non_exact_count']}, "
            f"judge_present_for_exact:{item['judge_present_for_exact_count']}, "
            f"inconsistent_final_score:{item['inconsistent_final_score_count']}"
            for item in protocol_violations[:20]
        )
        raise SystemExit(
            f"Protocol violations in {len(protocol_violations)} model/setting pair(s): {rendered}. "
            "Expected raw -> parsed -> exact match -> Judge Match only when exact match is false."
        )

    summary_rows = _build_summary_rows(model_rows)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_csv = output_dir / "partial_replacement_drop_by_model.csv"
    question_type_model_csv = output_dir / "partial_replacement_drop_by_model_question_type.csv"
    answer_type_model_csv = output_dir / "partial_replacement_drop_by_model_answer_type.csv"
    summary_csv = output_dir / "partial_replacement_drop_summary_by_proportion.csv"
    json_path = output_dir / "partial_replacement_drop_report.json"
    markdown_path = output_dir / "partial_replacement_drop_report.md"

    _write_csv(model_csv, model_rows)
    _write_csv(question_type_model_csv, question_type_model_rows)
    _write_csv(answer_type_model_csv, answer_type_model_rows)
    _write_csv(summary_csv, summary_rows)
    markdown_path.write_text(_format_markdown(summary_rows, model_rows, models), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "models": list(models),
                "settings": settings,
                "missing_model_setting_pairs": missing,
                "incomplete_model_setting_pairs": incomplete,
                "protocol_violations": protocol_violations,
                "by_model": model_rows,
                "by_model_question_type": question_type_model_rows,
                "by_model_answer_type": answer_type_model_rows,
                "summary_by_proportion": summary_rows,
                "outputs": {
                    "by_model_csv": str(model_csv.relative_to(_ROOT)),
                    "by_model_question_type_csv": str(question_type_model_csv.relative_to(_ROOT)),
                    "by_model_answer_type_csv": str(answer_type_model_csv.relative_to(_ROOT)),
                    "summary_csv": str(summary_csv.relative_to(_ROOT)),
                    "markdown": str(markdown_path.relative_to(_ROOT)),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(model_csv.relative_to(_ROOT))
    print(question_type_model_csv.relative_to(_ROOT))
    print(answer_type_model_csv.relative_to(_ROOT))
    print(summary_csv.relative_to(_ROOT))
    print(markdown_path.relative_to(_ROOT))
    print(json_path.relative_to(_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
