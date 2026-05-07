#!/usr/bin/env python3
"""Report how often variant fictional failures reuse the factual answer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import t

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.build_judge_match_drop_by_question_type_figure import (  # noqa: E402
    CACHE_PATH,
    EXPECTED_VARIANT_IDS,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    JUDGE_SEED,
    _display_model_name,
    _judge_cache_key,
)
from src.core.answer_matching import normalize_answer  # noqa: E402
from src.dataset_export.dataset_paths import MODEL_EVAL_PLOTS_DIR  # noqa: E402
from src.evaluation_workflows.parametric_shortcut.dataset import (  # noqa: E402
    is_excluded_evaluation_document,
)
from src.evaluation_workflows.parametric_shortcut.prompting import (  # noqa: E402
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)
from src.evaluation_workflows.parametric_shortcut.scoring import (  # noqa: E402
    accepted_answer_match_is_correct,
)
from src.llm.text_generation import TextGenerationRequest, generate_text  # noqa: E402


RAW_OUTPUTS_DIR = PROJECT_ROOT / "data" / "MODEL_EVAL" / "RAW_OUTPUTS"
OUTPUT_DIR = MODEL_EVAL_PLOTS_DIR / "factual_answer_shortcuts"
DEFAULT_MODELS = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "claude-sonnet-4-6",
)
BASE_QUESTION_TYPES = ("arithmetic", "temporal", "inference", "extractive")
QUESTION_COLUMNS = ("arithmetic", "temporal", "inference", "reasoning", "extractive")
QUESTION_MEMBERS = {
    "arithmetic": ("arithmetic",),
    "temporal": ("temporal",),
    "inference": ("inference",),
    "reasoning": ("arithmetic", "temporal", "inference"),
    "extractive": ("extractive",),
}
QUESTION_LABELS = {
    "arithmetic": "Arith.",
    "temporal": "Temp.",
    "inference": "Infer.",
    "reasoning": "Reason.",
    "extractive": "Extr.",
}
try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


@dataclass(frozen=True)
class FactualAnswerSpec:
    pair_key: str
    question_text: str
    question_type: str
    ground_truth: str
    accepted_answers_canonical: tuple[str, ...]
    answer_schema: str


@dataclass(frozen=True)
class VariantPrediction:
    model_name: str
    pair_key: str
    question_type: str
    variant_id: str
    final_is_correct: bool
    parsed_output: str
    parsed_output_canonical: str
    raw_output: str


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}


def _load_judge_cache() -> dict[str, dict[str, Any]]:
    if not CACHE_PATH.exists():
        return {}
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _save_judge_cache(cache: dict[str, dict[str, Any]]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _iter_evaluated_paths(model_name: str, setting: str) -> list[Path]:
    return sorted(RAW_OUTPUTS_DIR.rglob(f"*_{setting}_{model_name}_evaluated_outputs.yaml"))


def _load_factual_specs(models: tuple[str, ...]) -> dict[str, FactualAnswerSpec]:
    specs: dict[str, FactualAnswerSpec] = {}
    for model_name in models:
        for path in _iter_evaluated_paths(model_name, "factual"):
            payload = _read_yaml(path)
            if str(payload.get("document_setting") or "").strip().lower() != "factual":
                continue
            if is_excluded_evaluation_document(str(payload.get("document_id") or "")):
                continue
            for row in payload.get("results") or []:
                pair_key = str(row.get("pair_key") or "").strip()
                if not pair_key or pair_key in specs:
                    continue
                specs[pair_key] = FactualAnswerSpec(
                    pair_key=pair_key,
                    question_text=str(row.get("question_text") or "").strip(),
                    question_type=str(row.get("question_type") or "").strip().lower(),
                    ground_truth=str(row.get("ground_truth") or "").strip(),
                    accepted_answers_canonical=tuple(
                        str(answer).strip()
                        for answer in (row.get("accepted_answers_canonical") or [])
                        if str(answer).strip()
                    ),
                    answer_schema=str(row.get("answer_schema") or "").strip(),
                )
    return specs


def _load_variant_predictions(models: tuple[str, ...]) -> list[VariantPrediction]:
    predictions: list[VariantPrediction] = []
    expected = set(EXPECTED_VARIANT_IDS)
    for model_name in models:
        for path in _iter_evaluated_paths(model_name, "fictional"):
            payload = _read_yaml(path)
            if str(payload.get("document_setting") or "").strip().lower() != "fictional":
                continue
            if is_excluded_evaluation_document(str(payload.get("document_id") or "")):
                continue
            variant_id = str(payload.get("document_variant_id") or "").strip()
            if variant_id not in expected:
                continue
            for row in payload.get("results") or []:
                answer_type = str(row.get("answer_behavior") or row.get("answer_type") or "").strip().lower()
                question_type = str(row.get("question_type") or "").strip().lower()
                if answer_type != "variant" or question_type not in BASE_QUESTION_TYPES:
                    continue
                pair_key = str(row.get("pair_key") or "").strip()
                if not pair_key:
                    continue
                predictions.append(
                    VariantPrediction(
                        model_name=model_name,
                        pair_key=pair_key,
                        question_type=question_type,
                        variant_id=variant_id,
                        final_is_correct=bool(row.get("final_is_correct")),
                        parsed_output=str(row.get("parsed_output") or "").strip(),
                        parsed_output_canonical=str(row.get("parsed_output_canonical") or "").strip(),
                        raw_output=str(row.get("raw_output") or "").strip(),
                    )
                )
    return predictions


def _parse_judge_verdict(text: str) -> bool | None:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().upper())
    trimmed = normalized.lstrip(" `\"'([{:-")
    if trimmed.startswith("INCORRECT"):
        return False
    if trimmed.startswith("CORRECT"):
        return True
    match = re.search(r"\bVERDICT\s*[:=-]\s*(INCORRECT|CORRECT)\b", normalized)
    if match:
        return match.group(1) == "CORRECT"
    verdict_tokens = re.findall(r"\b(INCORRECT|CORRECT)\b", normalized)
    if verdict_tokens:
        return verdict_tokens[-1] == "CORRECT"
    return None


def _likely_needs_judge(predicted_answer: str, factual_spec: FactualAnswerSpec) -> bool:
    pred_norm = normalize_answer(predicted_answer)
    truth_norm = normalize_answer(factual_spec.ground_truth)
    if not pred_norm or not truth_norm:
        return False
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return True
    pred_tokens = set(pred_norm.split())
    truth_tokens = set(truth_norm.split())
    if pred_tokens and truth_tokens and len(pred_tokens & truth_tokens) / min(len(pred_tokens), len(truth_tokens)) >= 0.5:
        return True
    for accepted in factual_spec.accepted_answers_canonical:
        accepted_tokens = set(str(accepted).split())
        if pred_tokens and accepted_tokens and len(pred_tokens & accepted_tokens) / min(len(pred_tokens), len(accepted_tokens)) >= 0.5:
            return True
    return False


def _judge_factual_match(
    *,
    question_text: str,
    factual_answer: str,
    predicted_answer: str,
    cache: dict[str, dict[str, Any]],
    allow_live_judge: bool,
) -> tuple[bool, str]:
    cache_key = _judge_cache_key(
        question_text=question_text,
        ground_truth=factual_answer,
        predicted_answer=predicted_answer,
    )
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("judge_match"), bool):
        return bool(cached["judge_match"]), "cached_judge_match"
    if not allow_live_judge:
        return False, "not_judged"

    response = generate_text(
        TextGenerationRequest(
            provider=JUDGE_PROVIDER,
            model=JUDGE_MODEL,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=build_judge_prompt(question_text, factual_answer, predicted_answer),
            temperature=0.0,
            max_tokens=64,
            seed=JUDGE_SEED,
        )
    )
    raw_text = response.text or response.reasoning_text or response.raw_response
    verdict = _parse_judge_verdict(raw_text)
    if verdict is None:
        raise RuntimeError(f"Unparseable judge response for factual-answer shortcut check: {raw_text!r}")
    cache[cache_key] = {
        "judge_match": bool(verdict),
        "judge_raw_output": raw_text,
        "judge_task": "variant_fictional_prediction_matches_factual_answer",
    }
    return bool(verdict), "live_judge_match"


def _ci_half_width(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    se = float(np.std(arr, ddof=1) / sqrt(float(arr.size)))
    half_width = float(t.ppf(0.975, df=int(arr.size - 1)) * se) if se > 0 else 0.0
    return mean, half_width


def build_report(
    *,
    models: tuple[str, ...],
    allow_live_judge: bool,
    live_judge_likely_only: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    factual_specs = _load_factual_specs(models)
    predictions = _load_variant_predictions(models)
    cache = _load_judge_cache()
    cache_changed = False

    by_question: dict[tuple[str, str], list[VariantPrediction]] = {}
    for pred in predictions:
        by_question.setdefault((pred.model_name, pred.pair_key), []).append(pred)

    question_rows: list[dict[str, Any]] = []
    judge_counts = {
        "exact_factual_matches": 0,
        "cached_judge_requests": 0,
        "cached_judge_positive": 0,
        "live_judge_requests": 0,
        "live_judge_positive": 0,
        "not_judged_low_overlap": 0,
        "not_judged_live_disabled": 0,
        "missing_factual_specs": 0,
    }
    expected_count = len(EXPECTED_VARIANT_IDS)
    for (model_name, pair_key), group in sorted(by_question.items()):
        factual_spec = factual_specs.get(pair_key)
        if factual_spec is None:
            judge_counts["missing_factual_specs"] += 1
            continue
        qtype = factual_spec.question_type or group[0].question_type
        if qtype not in BASE_QUESTION_TYPES:
            continue

        total_variants = len({pred.variant_id for pred in group})
        failcase_count = 0
        factual_answer_match_count = 0
        exact_match_count = 0
        cached_judge_match_count = 0
        live_judge_match_count = 0
        not_judged_count = 0
        for pred in group:
            if pred.final_is_correct:
                continue
            failcase_count += 1
            predicted_answer = pred.parsed_output or pred.raw_output
            exact_match = accepted_answer_match_is_correct(
                pred.parsed_output_canonical,
                factual_spec.accepted_answers_canonical,
                answer_schema=factual_spec.answer_schema,
                raw_prediction=predicted_answer,
            )
            if exact_match:
                factual_answer_match_count += 1
                exact_match_count += 1
                judge_counts["exact_factual_matches"] += 1
                continue

            try:
                judge_match, source = _judge_factual_match(
                    question_text=factual_spec.question_text,
                    factual_answer=factual_spec.ground_truth,
                    predicted_answer=pred.raw_output or pred.parsed_output,
                    cache=cache,
                    allow_live_judge=False,
                )
            except Exception as exc:  # keep the report moving for meeting-time usage
                judge_match = False
                source = f"judge_error:{type(exc).__name__}"
                not_judged_count += 1
            if source == "not_judged":
                judge_allowed = (not live_judge_likely_only) or _likely_needs_judge(predicted_answer, factual_spec)
                if not judge_allowed:
                    not_judged_count += 1
                    judge_counts["not_judged_low_overlap"] += 1
                    continue
                try:
                    judge_match, source = _judge_factual_match(
                        question_text=factual_spec.question_text,
                        factual_answer=factual_spec.ground_truth,
                        predicted_answer=pred.raw_output or pred.parsed_output,
                        cache=cache,
                        allow_live_judge=allow_live_judge,
                    )
                except Exception as exc:  # keep the report moving for meeting-time usage
                    judge_match = False
                    source = f"judge_error:{type(exc).__name__}"
                    not_judged_count += 1
            if source == "cached_judge_match":
                judge_counts["cached_judge_requests"] += 1
                if judge_match:
                    cached_judge_match_count += 1
                    judge_counts["cached_judge_positive"] += 1
            elif source == "live_judge_match":
                cache_changed = True
                judge_counts["live_judge_requests"] += 1
                if judge_match:
                    live_judge_match_count += 1
                    judge_counts["live_judge_positive"] += 1
            elif source == "not_judged":
                judge_counts["not_judged_live_disabled"] += 1
                not_judged_count += 1
            if judge_match:
                factual_answer_match_count += 1

        all_variant_rate = factual_answer_match_count / expected_count
        observed_variant_rate = factual_answer_match_count / total_variants if total_variants else 0.0
        failcase_rate = factual_answer_match_count / failcase_count if failcase_count else 0.0
        question_rows.append(
            {
                "model_name": model_name,
                "pair_key": pair_key,
                "question_type": qtype,
                "total_variants_observed": total_variants,
                "expected_variants": expected_count,
                "failcase_count": failcase_count,
                "factual_answer_match_count": factual_answer_match_count,
                "factual_answer_exact_match_count": exact_match_count,
                "factual_answer_cached_judge_match_count": cached_judge_match_count,
                "factual_answer_live_judge_match_count": live_judge_match_count,
                "not_judged_count": not_judged_count,
                "factual_answer_rate_all_expected_variants": all_variant_rate,
                "factual_answer_rate_observed_variants": observed_variant_rate,
                "factual_answer_rate_among_failcases": failcase_rate,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for model_name in models:
        model_rows = [row for row in question_rows if row["model_name"] == model_name]
        for question_group in QUESTION_COLUMNS:
            members = QUESTION_MEMBERS[question_group]
            rows = [row for row in model_rows if row["question_type"] in members]
            if not rows:
                continue
            all_rates = [100.0 * float(row["factual_answer_rate_all_expected_variants"]) for row in rows]
            failcase_rates = [100.0 * float(row["factual_answer_rate_among_failcases"]) for row in rows]
            all_mean, all_ci = _ci_half_width(all_rates)
            fail_mean, fail_ci = _ci_half_width(failcase_rates)
            summary_rows.append(
                {
                    "model_name": model_name,
                    "question_type": question_group,
                    "question_type_label": QUESTION_LABELS[question_group],
                    "n_question_percentages": len(rows),
                    "total_failcases": sum(int(row["failcase_count"]) for row in rows),
                    "total_factual_answer_matches": sum(int(row["factual_answer_match_count"]) for row in rows),
                    "mean_factual_answer_rate_all_variants_pp": all_mean,
                    "ci95_half_width_all_variants_pp": all_ci,
                    "mean_factual_answer_rate_among_failcases_pp": fail_mean,
                    "ci95_half_width_among_failcases_pp": fail_ci,
                }
            )

    if cache_changed:
        _save_judge_cache(cache)

    diagnostics = {
        "models": list(models),
        "expected_variant_ids": list(EXPECTED_VARIANT_IDS),
        "n_factual_specs": len(factual_specs),
        "n_variant_predictions": len(predictions),
        "n_question_rows": len(question_rows),
        "judge_counts": judge_counts,
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL,
        "live_judge_likely_only": live_judge_likely_only,
        "allow_live_judge": allow_live_judge,
    }
    return summary_rows, question_rows, diagnostics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_cell(mean: float, ci: float) -> str:
    return f"{mean:.1f} ± {ci:.1f}"


def _print_markdown_table(summary_rows: list[dict[str, Any]], *, metric: str) -> None:
    by_key = {(row["model_name"], row["question_type"]): row for row in summary_rows}
    print(f"\n{metric}")
    print("| Model | Arith. | Temp. | Infer. | Reason. | Extr. |")
    print("|---|---:|---:|---:|---:|---:|")
    for model_name in DEFAULT_MODELS:
        cells = []
        for question_type in QUESTION_COLUMNS:
            row = by_key.get((model_name, question_type))
            if row is None:
                cells.append("")
                continue
            if metric == "all_variants":
                cells.append(
                    _format_cell(
                        float(row["mean_factual_answer_rate_all_variants_pp"]),
                        float(row["ci95_half_width_all_variants_pp"]),
                    )
                )
            else:
                cells.append(
                    _format_cell(
                        float(row["mean_factual_answer_rate_among_failcases_pp"]),
                        float(row["ci95_half_width_among_failcases_pp"]),
                    )
                )
        print(f"| {_display_model_name(model_name).replace(chr(10), ' ')} | " + " | ".join(cells) + " |")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--allow-live-judge", action="store_true")
    parser.add_argument(
        "--judge-all-non-exact",
        action="store_true",
        help="Judge every non-exact failcase. By default, live judging is limited to lexical-overlap candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = tuple(str(model) for model in args.models)
    summary_rows, question_rows, diagnostics = build_report(
        models=models,
        allow_live_judge=bool(args.allow_live_judge),
        live_judge_likely_only=not bool(args.judge_all_non_exact),
    )
    output_dir = args.output_dir
    _write_csv(output_dir / "variant_factual_answer_shortcut_summary.csv", summary_rows)
    _write_csv(output_dir / "variant_factual_answer_shortcut_by_question.csv", question_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "variant_factual_answer_shortcut_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2, ensure_ascii=True))
    _print_markdown_table(summary_rows, metric="all_variants")
    _print_markdown_table(summary_rows, metric="among_failcases")


if __name__ == "__main__":
    main()
