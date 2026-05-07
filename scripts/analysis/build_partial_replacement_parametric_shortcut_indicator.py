#!/usr/bin/env python3
"""Overlay partial-replacement accuracy with a parametric shortcut indicator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from scripts.analysis.plot_partial_replacement_performance_curves import (  # noqa: E402
    DEFAULT_INPUT_PATH,
    ERROR_BAND_COLOR,
    FACTUAL_COLOR,
    LINE_COLOR,
    MODEL_LABELS,
    MODEL_ORDER,
    _accuracy_error_interval_pp,
    _draw_x_tick_labels,
    _format_x_tick_label,
    _read_rows,
    _series_by_model,
)
from src.core.answer_matching import normalize_answer  # noqa: E402
from src.dataset_export.dataset_paths import MODEL_EVAL_PLOTS_DIR, MODEL_EVAL_RAW_OUTPUTS_DIR  # noqa: E402
from src.dataset_export.dataset_settings import parse_dataset_setting  # noqa: E402
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


DEFAULT_MODELS = MODEL_ORDER
DEFAULT_OUTPUT_DIR = MODEL_EVAL_PLOTS_DIR / "partial_fictional_replacements"
DEFAULT_OUTPUT_STEM = "partial_replacement_accuracy_with_parametric_shortcut_indicator_by_model_zoomed"
DEFAULT_PSI_CSV = DEFAULT_OUTPUT_DIR / "partial_replacement_parametric_shortcut_indicator_by_model.csv"
DEFAULT_DIAGNOSTICS_PATH = (
    DEFAULT_OUTPUT_DIR / "partial_replacement_parametric_shortcut_indicator_diagnostics.json"
)
DEFAULT_CACHE_PATH = (
    _ROOT / "data" / "JUDGE_EVAL" / "partial_replacement_parametric_shortcut_indicator_judge_cache.json"
)
DEFAULT_SETTINGS = (
    "fictional_10pct",
    "fictional_20pct",
    "fictional_30pct",
    "fictional_50pct",
    "fictional_80pct",
    "fictional_90pct",
    "fictional",
)
JUDGE_PROVIDER = "groq"
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_SEED = 23
PSI_COLOR = "#2563eb"
PSI_ERROR_COLOR = "#93c5fd"

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


@dataclass(frozen=True)
class FactualAnswerSpec:
    pair_key: str
    question_text: str
    ground_truth: str
    accepted_answers_canonical: tuple[str, ...]
    answer_schema: str


@dataclass(frozen=True)
class VariantFailure:
    model_name: str
    document_setting: str
    replacement_proportion: float
    pair_key: str
    parsed_output: str
    parsed_output_canonical: str
    raw_output: str


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}


def _load_json_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _save_json_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _judge_cache_key(*, question_text: str, factual_answer: str, predicted_answer: str) -> str:
    payload = {
        "task": "partial_replacement_parametric_shortcut_indicator",
        "judge_model": JUDGE_MODEL,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "question_text": question_text,
        "factual_answer": factual_answer,
        "predicted_answer": predicted_answer,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _parse_judge_verdict(text: str) -> bool | None:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().upper())
    trimmed = normalized.lstrip(" `\"'([{:-")
    if trimmed.startswith("INCORRECT") or trimmed == "INCOR":
        return False
    if trimmed.startswith("CORRECT") or trimmed == "CORR":
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
        accepted_tokens = set(normalize_answer(str(accepted)).split())
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
        factual_answer=factual_answer,
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
        raise RuntimeError(f"Unparseable judge response: {raw_text!r}")
    cache[cache_key] = {
        "judge_match": bool(verdict),
        "judge_raw_output": raw_text,
        "judge_task": "partial_replacement_parametric_shortcut_indicator",
    }
    return bool(verdict), "live_judge_match"


def _run_live_judge_task(task: dict[str, str]) -> dict[str, Any]:
    response = generate_text(
        TextGenerationRequest(
            provider=JUDGE_PROVIDER,
            model=JUDGE_MODEL,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=build_judge_prompt(
                task["question_text"],
                task["factual_answer"],
                task["predicted_answer"],
            ),
            temperature=0.0,
            max_tokens=64,
            seed=JUDGE_SEED,
        )
    )
    raw_text = response.text or response.reasoning_text or response.raw_response
    verdict = _parse_judge_verdict(raw_text)
    if verdict is None:
        raise RuntimeError(f"Unparseable judge response: {raw_text!r}")
    return {
        "cache_key": task["cache_key"],
        "judge_match": bool(verdict),
        "judge_raw_output": raw_text,
    }


def _load_factual_specs(models: tuple[str, ...]) -> dict[str, FactualAnswerSpec]:
    specs: dict[str, FactualAnswerSpec] = {}
    model_set = set(models)
    for path in sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.rglob("*_factual_*_evaluated_outputs.yaml")):
        payload = _read_yaml(path)
        model_name = str(payload.get("model_name") or path.parent.name)
        if model_name not in model_set:
            continue
        if str(payload.get("document_setting") or "").strip().lower() != "factual":
            continue
        if is_excluded_evaluation_document(str(payload.get("document_id") or "")):
            continue
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            pair_key = str(result.get("pair_key") or "").strip()
            if not pair_key or pair_key in specs:
                continue
            specs[pair_key] = FactualAnswerSpec(
                pair_key=pair_key,
                question_text=str(result.get("question_text") or "").strip(),
                ground_truth=str(result.get("ground_truth") or "").strip(),
                accepted_answers_canonical=tuple(
                    str(answer).strip()
                    for answer in (result.get("accepted_answers_canonical") or [])
                    if str(answer).strip()
                ),
                answer_schema=str(result.get("answer_schema") or "").strip(),
            )
    return specs


def _iter_variant_failures(
    *,
    models: tuple[str, ...],
    settings: tuple[str, ...],
) -> list[VariantFailure]:
    model_set = set(models)
    setting_set = set(settings)
    failures: list[VariantFailure] = []
    for path in sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.rglob("*_evaluated_outputs.yaml")):
        payload = _read_yaml(path)
        model_name = str(payload.get("model_name") or path.parent.name)
        document_setting = str(payload.get("document_setting") or "").strip().lower()
        if model_name not in model_set or document_setting not in setting_set:
            continue
        if is_excluded_evaluation_document(str(payload.get("document_id") or "")):
            continue
        replacement_proportion = float(
            payload.get("replacement_proportion")
            or parse_dataset_setting(document_setting).replacement_proportion
        )
        for result in payload.get("results") or []:
            if not isinstance(result, dict):
                continue
            answer_type = str(result.get("answer_type") or result.get("answer_behavior") or "").strip().lower()
            if answer_type != "variant" or bool(result.get("final_is_correct")):
                continue
            pair_key = str(result.get("pair_key") or "").strip()
            if not pair_key:
                continue
            failures.append(
                VariantFailure(
                    model_name=model_name,
                    document_setting=document_setting,
                    replacement_proportion=replacement_proportion,
                    pair_key=pair_key,
                    parsed_output=str(result.get("parsed_output") or "").strip(),
                    parsed_output_canonical=str(result.get("parsed_output_canonical") or "").strip(),
                    raw_output=str(result.get("raw_output") or "").strip(),
                )
            )
    return failures


def _wald_95_margin_pp(success_count: int, total_count: int) -> float:
    if total_count <= 0:
        return 0.0
    p = min(max(success_count / total_count, 0.0), 1.0)
    return 100.0 * 1.96 * sqrt(max(p * (1.0 - p), 0.0) / total_count)


def prefill_judge_cache(
    *,
    models: tuple[str, ...],
    settings: tuple[str, ...],
    cache_path: Path,
    live_judge_likely_only: bool,
    max_workers: int,
) -> dict[str, Any]:
    factual_specs = _load_factual_specs(models)
    failures = _iter_variant_failures(models=models, settings=settings)
    cache = _load_json_cache(cache_path)
    tasks_by_key: dict[str, dict[str, str]] = {}
    counts: Counter[str] = Counter()

    for failure in failures:
        factual_spec = factual_specs.get(failure.pair_key)
        if factual_spec is None:
            counts["missing_factual_spec_count"] += 1
            continue
        predicted_answer = failure.parsed_output or failure.raw_output
        exact_match = accepted_answer_match_is_correct(
            failure.parsed_output_canonical,
            factual_spec.accepted_answers_canonical,
            answer_schema=factual_spec.answer_schema,
            raw_prediction=predicted_answer,
        )
        if exact_match:
            counts["exact_factual_match_count"] += 1
            continue
        if live_judge_likely_only and not _likely_needs_judge(predicted_answer, factual_spec):
            counts["not_judged_low_overlap_count"] += 1
            continue
        cache_key = _judge_cache_key(
            question_text=factual_spec.question_text,
            factual_answer=factual_spec.ground_truth,
            predicted_answer=predicted_answer,
        )
        if cache_key in cache:
            counts["already_cached_candidate_count"] += 1
            continue
        tasks_by_key.setdefault(
            cache_key,
            {
                "cache_key": cache_key,
                "question_text": factual_spec.question_text,
                "factual_answer": factual_spec.ground_truth,
                "predicted_answer": predicted_answer,
            },
        )

    tasks = list(tasks_by_key.values())
    if not tasks:
        return {
            "candidate_unique_tasks": 0,
            "completed_live_judge_tasks": 0,
            "failed_live_judge_tasks": 0,
            "prefill_counts": dict(counts),
        }

    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = [executor.submit(_run_live_judge_task, task) for task in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception:
                failed += 1
                continue
            cache[result["cache_key"]] = {
                "judge_match": bool(result["judge_match"]),
                "judge_raw_output": result["judge_raw_output"],
                "judge_task": "partial_replacement_parametric_shortcut_indicator",
            }
            completed += 1
            if completed % 50 == 0:
                _save_json_cache(cache_path, cache)
                print(f"prefill_judge_cache completed={completed}/{len(tasks)} failed={failed}", flush=True)
    _save_json_cache(cache_path, cache)
    return {
        "candidate_unique_tasks": len(tasks),
        "completed_live_judge_tasks": completed,
        "failed_live_judge_tasks": failed,
        "prefill_counts": dict(counts),
    }


def build_indicator_rows(
    *,
    models: tuple[str, ...],
    settings: tuple[str, ...],
    cache_path: Path,
    allow_live_judge: bool,
    live_judge_likely_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factual_specs = _load_factual_specs(models)
    failures = _iter_variant_failures(models=models, settings=settings)
    cache = _load_json_cache(cache_path)
    cache_changed = False
    by_model_proportion: dict[tuple[str, float], Counter[str]] = defaultdict(Counter)
    judge_counts: Counter[str] = Counter()

    for failure in failures:
        key = (failure.model_name, round(failure.replacement_proportion, 1))
        by_model_proportion[key]["failcase_count"] += 1
        factual_spec = factual_specs.get(failure.pair_key)
        if factual_spec is None:
            by_model_proportion[key]["missing_factual_spec_count"] += 1
            judge_counts["missing_factual_spec_count"] += 1
            continue

        predicted_answer = failure.parsed_output or failure.raw_output
        exact_match = accepted_answer_match_is_correct(
            failure.parsed_output_canonical,
            factual_spec.accepted_answers_canonical,
            answer_schema=factual_spec.answer_schema,
            raw_prediction=predicted_answer,
        )
        if exact_match:
            by_model_proportion[key]["factual_answer_match_count"] += 1
            by_model_proportion[key]["exact_factual_match_count"] += 1
            judge_counts["exact_factual_match_count"] += 1
            continue

        judge_match, source = _judge_factual_match(
            question_text=factual_spec.question_text,
            factual_answer=factual_spec.ground_truth,
            predicted_answer=predicted_answer,
            cache=cache,
            allow_live_judge=False,
        )
        if source == "not_judged":
            judge_allowed = (not live_judge_likely_only) or _likely_needs_judge(predicted_answer, factual_spec)
            if not judge_allowed:
                by_model_proportion[key]["not_judged_low_overlap_count"] += 1
                judge_counts["not_judged_low_overlap_count"] += 1
                continue
            try:
                judge_match, source = _judge_factual_match(
                    question_text=factual_spec.question_text,
                    factual_answer=factual_spec.ground_truth,
                    predicted_answer=predicted_answer,
                    cache=cache,
                    allow_live_judge=allow_live_judge,
                )
            except Exception as exc:
                by_model_proportion[key][f"judge_error_{type(exc).__name__}_count"] += 1
                judge_counts[f"judge_error_{type(exc).__name__}_count"] += 1
                by_model_proportion[key]["not_judged_judge_error_count"] += 1
                judge_counts["not_judged_judge_error_count"] += 1
                continue

        if source == "cached_judge_match":
            by_model_proportion[key]["cached_judge_request_count"] += 1
            judge_counts["cached_judge_request_count"] += 1
            if judge_match:
                by_model_proportion[key]["cached_judge_positive_count"] += 1
                judge_counts["cached_judge_positive_count"] += 1
        elif source == "live_judge_match":
            cache_changed = True
            by_model_proportion[key]["live_judge_request_count"] += 1
            judge_counts["live_judge_request_count"] += 1
            if judge_counts["live_judge_request_count"] % 50 == 0:
                _save_json_cache(cache_path, cache)
            if judge_match:
                by_model_proportion[key]["live_judge_positive_count"] += 1
                judge_counts["live_judge_positive_count"] += 1
        elif source == "not_judged":
            by_model_proportion[key]["not_judged_live_disabled_count"] += 1
            judge_counts["not_judged_live_disabled_count"] += 1

        if judge_match:
            by_model_proportion[key]["factual_answer_match_count"] += 1

    if cache_changed:
        _save_json_cache(cache_path, cache)

    rows: list[dict[str, Any]] = []
    setting_by_proportion = {
        round(parse_dataset_setting(setting).replacement_proportion, 1): setting
        for setting in settings
    }
    for model_name in models:
        for proportion in sorted(setting_by_proportion):
            counts = by_model_proportion.get((model_name, proportion), Counter())
            failcase_count = int(counts["failcase_count"])
            match_count = int(counts["factual_answer_match_count"])
            psi = match_count / failcase_count if failcase_count else 0.0
            rows.append(
                {
                    "model_name": model_name,
                    "document_setting": setting_by_proportion[proportion],
                    "replacement_proportion": proportion,
                    "failcase_count": failcase_count,
                    "factual_answer_match_count": match_count,
                    "parametric_shortcut_indicator": psi,
                    "parametric_shortcut_indicator_pp": 100.0 * psi,
                    "parametric_shortcut_indicator_ci95_half_width_pp": _wald_95_margin_pp(
                        match_count,
                        failcase_count,
                    ),
                    "exact_factual_match_count": int(counts["exact_factual_match_count"]),
                    "cached_judge_request_count": int(counts["cached_judge_request_count"]),
                    "cached_judge_positive_count": int(counts["cached_judge_positive_count"]),
                    "live_judge_request_count": int(counts["live_judge_request_count"]),
                    "live_judge_positive_count": int(counts["live_judge_positive_count"]),
                    "not_judged_low_overlap_count": int(counts["not_judged_low_overlap_count"]),
                    "not_judged_live_disabled_count": int(counts["not_judged_live_disabled_count"]),
                    "missing_factual_spec_count": int(counts["missing_factual_spec_count"]),
                }
            )

    diagnostics = {
        "models": list(models),
        "settings": list(settings),
        "cache_path": str(cache_path.relative_to(_ROOT)),
        "n_factual_specs": len(factual_specs),
        "n_variant_failures": len(failures),
        "judge_counts": dict(judge_counts),
        "allow_live_judge": allow_live_judge,
        "live_judge_likely_only": live_judge_likely_only,
        "judge_provider": JUDGE_PROVIDER,
        "judge_model": JUDGE_MODEL,
    }
    return rows, diagnostics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _indicator_series_by_model(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float]]]:
    by_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        by_model[str(row["model_name"])].append(
            {
                "proportion": float(row["replacement_proportion"]),
                "indicator_pp": float(row["parametric_shortcut_indicator_pp"]),
                "indicator_ci_pp": float(row["parametric_shortcut_indicator_ci95_half_width_pp"]),
            }
        )
    return {
        model_name: sorted(points, key=lambda point: point["proportion"])
        for model_name, points in by_model.items()
    }


def build_dual_axis_figure(
    *,
    accuracy_input_path: Path,
    indicator_rows: list[dict[str, Any]],
    output_dir: Path,
    output_stem: str,
    error_method: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    accuracy_rows = _read_rows(accuracy_input_path)
    accuracy_series = _series_by_model(
        accuracy_rows,
        error_method=error_method,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    indicator_series = _indicator_series_by_model(indicator_rows)
    models = [model for model in MODEL_ORDER if model in accuracy_series]
    models.extend(sorted(model for model in accuracy_series if model not in MODEL_ORDER))
    if not models:
        raise ValueError("No model series to plot.")

    x_ticks = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0]
    x_tick_positions = list(range(len(x_ticks)))
    x_position_by_proportion = {
        round(proportion, 1): position
        for position, proportion in zip(x_tick_positions, x_ticks, strict=True)
    }
    x_tick_labels = [_format_x_tick_label(tick) for tick in x_ticks]
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#000000",
            "axes.labelcolor": "#000000",
            "axes.titleweight": "bold",
            "axes.titlesize": 100,
            "font.size": 12,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "#000000",
            "ytick.color": "#000000",
        }
    )

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(84, 42),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]
    right_axes = []

    for ax, model in zip(flat_axes, models, strict=False):
        points = accuracy_series[model]
        xs = [x_position_by_proportion[round(point.proportion, 1)] for point in points]
        ys = [point.accuracy_pp for point in points]
        y_lows = [
            max(0.0, point.accuracy_pp - point.accuracy_error_low_pp)
            for point in points
        ]
        y_highs = [
            min(100.0, point.accuracy_pp + point.accuracy_error_high_pp)
            for point in points
        ]
        factual_y = {point.proportion: point.accuracy_pp for point in points}[0.0]
        y_span = max(y_highs) - min(y_lows)
        ax.set_ylim(
            max(0.0, min(y_lows) - max(0.2, y_span * 0.08)),
            min(100.0, max(y_highs) + max(0.45, y_span * 0.18)),
        )
        ax.fill_between(
            xs,
            y_lows,
            y_highs,
            color=ERROR_BAND_COLOR,
            alpha=0.32,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            xs,
            ys,
            color=LINE_COLOR,
            linewidth=18.0,
            marker="o",
            markersize=15.0,
            markerfacecolor="white",
            markeredgewidth=5.5,
            zorder=3,
        )
        ax.axhline(
            factual_y,
            color=FACTUAL_COLOR,
            linewidth=9.5,
            linestyle=(0, (4, 3)),
            alpha=0.9,
            zorder=2,
        )

        ax2 = ax.twinx()
        right_axes.append(ax2)
        psi_points = indicator_series.get(model, [])
        psi_xs = [x_position_by_proportion[round(point["proportion"], 1)] for point in psi_points]
        psi_ys = [point["indicator_pp"] for point in psi_points]
        psi_lows = [max(0.0, point["indicator_pp"] - point["indicator_ci_pp"]) for point in psi_points]
        psi_highs = [min(100.0, point["indicator_pp"] + point["indicator_ci_pp"]) for point in psi_points]
        if psi_points:
            psi_span = max(psi_highs) - min(psi_lows)
            psi_padding = max(0.35, psi_span * 0.18)
            ax2.set_ylim(
                max(0.0, min(psi_lows) - psi_padding),
                min(100.0, max(psi_highs) + psi_padding),
            )
            ax2.fill_between(
                psi_xs,
                psi_lows,
                psi_highs,
                color=PSI_ERROR_COLOR,
                alpha=0.18,
                linewidth=0,
                zorder=1,
            )
            ax2.plot(
                psi_xs,
                psi_ys,
                color=PSI_COLOR,
                linewidth=11.0,
                marker="s",
                markersize=12.0,
                markerfacecolor="white",
                markeredgewidth=4.0,
                zorder=4,
            )
        else:
            ax2.set_ylim(0.0, 5.0)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax2.tick_params(
            axis="y",
            labelsize=58,
            colors=PSI_COLOR,
            width=3.0,
            length=7,
            labelright=True,
            right=True,
            pad=7,
        )
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(PSI_COLOR)
        ax2.spines["right"].set_linewidth(3.0)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.grid(False)

        ax.text(
            0.5,
            0.98,
            MODEL_LABELS.get(model, model),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=92,
            fontweight="bold",
            color="#000000",
            zorder=5,
        )
        ax.set_xlim(-0.25, len(x_ticks) - 0.75)
        ax.set_xticks(x_tick_positions)
        _draw_x_tick_labels(ax, x_tick_positions, x_tick_labels)
        ax.tick_params(axis="x", pad=40, labelsize=88, colors="#000000", width=3.5, length=10)
        ax.tick_params(axis="y", labelsize=88, colors="#000000", width=3.5, length=10)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=5))
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.8)
        ax.grid(axis="x", color="#ededed", linewidth=0.6)
        ax.spines["left"].set_color("#000000")
        ax.spines["bottom"].set_color("#000000")
        ax.spines["left"].set_linewidth(3.5)
        ax.spines["bottom"].set_linewidth(3.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(models) :]:
        ax.axis("off")

    axis_label_size = 108
    fig.supxlabel("Replaced Entities (%)", fontsize=axis_label_size)
    fig.supylabel("Accuracy (%)", fontsize=axis_label_size)
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=FACTUAL_COLOR,
                linewidth=9.5,
                linestyle=(0, (4, 3)),
                label="Factual baseline",
            ),
            Line2D(
                [0],
                [0],
                color=LINE_COLOR,
                linewidth=12.0,
                marker="o",
                markersize=15,
                markerfacecolor="white",
                markeredgewidth=4.0,
                label="Accuracy",
            ),
            Line2D(
                [0],
                [0],
                color=PSI_COLOR,
                linewidth=9.0,
                marker="s",
                markersize=14,
                markerfacecolor="white",
                markeredgewidth=3.5,
                label="Parametric Shortcut Indicator",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        frameon=False,
        fontsize=72,
        handlelength=3.2,
        ncol=3,
        columnspacing=1.1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / f"{output_stem}.png",
        output_dir / f"{output_stem}.pdf",
        output_dir / f"{output_stem}.svg",
    ]
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300 if output_path.suffix == ".png" else None)
    plt.close(fig)
    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--settings", nargs="+", default=list(DEFAULT_SETTINGS))
    parser.add_argument("--accuracy-input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--psi-csv", type=Path, default=DEFAULT_PSI_CSV)
    parser.add_argument("--diagnostics-path", type=Path, default=DEFAULT_DIAGNOSTICS_PATH)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--allow-live-judge", action="store_true")
    parser.add_argument(
        "--judge-workers",
        type=int,
        default=1,
        help="When >1 with --allow-live-judge, prefill missing Judge Match cache entries concurrently.",
    )
    parser.add_argument(
        "--judge-all-non-exact",
        action="store_true",
        help="Send every non-exact variant failure to Judge Match. Default only judges plausible lexical overlaps.",
    )
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument(
        "--error-method",
        choices=("wald", "wilson", "bootstrap"),
        default="wald",
        help="Uncertainty band method for accuracy.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=23)
    args = parser.parse_args()

    prefill_diagnostics: dict[str, Any] = {}
    live_judge_in_report = args.allow_live_judge
    if args.allow_live_judge and int(args.judge_workers) > 1:
        prefill_diagnostics = prefill_judge_cache(
            models=tuple(args.models),
            settings=tuple(args.settings),
            cache_path=args.cache_path,
            live_judge_likely_only=not args.judge_all_non_exact,
            max_workers=int(args.judge_workers),
        )
        live_judge_in_report = False

    indicator_rows, diagnostics = build_indicator_rows(
        models=tuple(args.models),
        settings=tuple(args.settings),
        cache_path=args.cache_path,
        allow_live_judge=live_judge_in_report,
        live_judge_likely_only=not args.judge_all_non_exact,
    )
    if prefill_diagnostics:
        diagnostics["prefill_judge_cache"] = prefill_diagnostics
    _write_csv(args.psi_csv, indicator_rows)
    args.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(args.psi_csv)
    print(args.diagnostics_path)

    if not args.skip_plot:
        output_paths = build_dual_axis_figure(
            accuracy_input_path=args.accuracy_input_path,
            indicator_rows=indicator_rows,
            output_dir=args.output_dir,
            output_stem=args.output_stem,
            error_method=args.error_method,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
        for output_path in output_paths:
            print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
