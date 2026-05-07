from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
import argparse
import csv
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.stats import multivariate_t, t
from statsmodels.stats.multitest import multipletests

from src.dataset_export.dataset_paths import MODEL_EVAL_PLOTS_DIR, PROJECT_ROOT
from src.core.annotation_runtime import find_entity_refs
from src.evaluation_workflows.parametric_shortcut.dataset import is_excluded_evaluation_document
from src.evaluation_workflows.parametric_shortcut.prompting import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from src.evaluation_workflows.parametric_shortcut.scoring import judge_match_is_allowed
from src.evaluation_workflows.parametric_shortcut.version_audit import (
    audit_saved_output_payload,
    blocking_issues,
    format_issue_summary,
)
from src.llm.text_generation import TextGenerationRequest, generate_text


RAW_OUTPUTS_DIR = PROJECT_ROOT / "data" / "MODEL_EVAL" / "RAW_OUTPUTS"
OUTPUT_DIR = MODEL_EVAL_PLOTS_DIR / "performance_drop_ttest"
CACHE_DIR = PROJECT_ROOT / "data" / "JUDGE_EVAL"
CACHE_PATH = CACHE_DIR / "full_groq_judge_eval_cache.json"
EXPECTED_VARIANT_IDS = tuple(f"v{index:02d}" for index in range(1, 11))
BASELINE_SETTING = "factual"
DEFAULT_COMPARISON_SETTING = "fictional"
SUPPORTED_COMPARISON_SETTINGS = {"fictional"}
SUPPORTED_ANSWER_BEHAVIORS = {"variant", "invariant", "refusal"}
SUPPORTED_VARIANT_LEVEL_CORRECTIONS = (
    "bonferroni",
    "holm",
    "sidak",
    "hochberg",
    "hommel",
    "dunnett",
    "max_t",
)
TARGET_MODELS = (
    "gpt-oss-120b-groq",
    "gpt-oss-20b-groq",
    "qwen3.5-0.8b",
    "qwen3.5-4b",
    "qwen3.5-9b",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "gemma-4-26b-a4b-it",
    "magistral-small-2509",
    "olmo-3-7b-instruct",
    "llama-3.2-3b-instruct",
)
QUESTION_TYPE_ORDER = ("all", "arithmetic", "temporal", "inference", "extractive")
QUESTION_GROUP_MEMBERS = {
    "all": ("arithmetic", "temporal", "inference", "extractive"),
    "reasoning": ("arithmetic", "temporal", "inference"),
    "arithmetic_temporal": ("arithmetic", "temporal"),
    "arithmetic": ("arithmetic",),
    "temporal": ("temporal",),
    "inference": ("inference",),
    "extractive": ("extractive",),
}
QUESTION_TYPE_LABELS = {
    "all": "All",
    "reasoning": "Reasoning",
    "arithmetic_temporal": "All",
    "arithmetic": "Arithmetic",
    "temporal": "Temporal",
    "inference": "Inference",
    "extractive": "Extractive",
}
COMPACT_QUESTION_TYPE_LABELS = {
    "all": "All",
    "reasoning": "Reason.",
    "arithmetic_temporal": "All",
    "arithmetic": "Arith.",
    "temporal": "Temp.",
    "inference": "Infer.",
    "extractive": "Extr.",
}
QUESTION_TYPE_COLORS = {
    "all": "#c7d7f7",
    "reasoning": "#e8b27c",
    "arithmetic_temporal": "#e8b27c",
    "arithmetic": "#f3c98b",
    "temporal": "#80b1d3",
    "inference": "#d8c3e6",
    "extractive": "#d9d9d9",
}
QUESTION_TYPE_HATCHES = {
    "all": "///",
    "reasoning": "\\\\\\",
    "arithmetic_temporal": "///",
    "arithmetic": "///",
    "temporal": "\\\\\\",
    "inference": "xx",
    "extractive": "..",
}
JUDGE_PROVIDER = "groq"
JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_MAX_TOKENS = 256
JUDGE_RETRY_MAX_TOKENS = 1024
JUDGE_SEED = 23
SIGNIFICANCE_TEST = "paired_t_test"
CONFIDENCE_INTERVAL = "paired_t_95"
BOOTSTRAP_CONFIDENCE_INTERVAL = "paired_bootstrap_percentile_95"
BOOTSTRAP_SIGNIFICANCE_TEST = "paired_bootstrap_ci_excludes_zero"
DEFAULT_BOOTSTRAP_ITERATIONS = 10000
DEFAULT_BOOTSTRAP_SEED = 23
DEFAULT_MAX_T_PERMUTATIONS = 10000
DEFAULT_MAX_T_SEED = 23
DEFAULT_DUNNETT_MAXPTS = 25000
try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


@dataclass
class DropRow:
    model_name: str
    question_type: str
    count_pairs: int
    factual_score: float
    fictional_score: float
    mean_difference: float
    standard_deviation_difference: float
    standard_error_difference: float
    degrees_of_freedom: int
    ci_low: float
    ci_high: float
    t_statistic: float
    p_value: float
    p_value_adjusted: float | None = None
    pvalue_correction: str | None = None
    correction_family_size: int | None = None
    significance_alpha: float = 0.05
    significant: bool | None = None
    significance_source: str = "aggregate_paired_mean"
    variant_p_values: dict[str, float] = field(default_factory=dict)
    variant_p_values_adjusted: dict[str, float] = field(default_factory=dict)
    variant_t_statistics: dict[str, float] = field(default_factory=dict)
    variant_differences: dict[str, tuple[float, ...]] = field(default_factory=dict, repr=False)
    significant_variant_ids: tuple[str, ...] = ()

    @property
    def is_significant_95_ci(self) -> bool:
        return bool(np.isfinite(self.ci_low) and np.isfinite(self.ci_high) and (self.ci_high < 0.0 or self.ci_low > 0.0))

    @property
    def is_significant(self) -> bool:
        if self.significant is not None:
            return bool(self.significant)
        return self.is_significant_95_ci

    @property
    def stars(self) -> str:
        return "*" if self.is_significant else ""


def _display_model_name(model_name: str) -> str:
    custom_labels = {
        "gpt-oss-120b-groq": "GPT-OSS\n120B",
        "gpt-oss-20b-groq": "GPT-OSS\n20B",
        "llama-3.2-3b-instruct": "LLAMA-3.2\n3B-INSTRUCT",
        "qwen3.5-27b": "QWEN3.5\n27B",
        "qwen3.5-35b-a3b": "QWEN3.5\n35B-A3B",
        "gemma-4-26b-a4b-it": "GEMMA-4\n26B-A4B-IT",
        "olmo-3-7b-instruct": "OLMO-3\n7B-INSTRUCT",
        "olmo-3-7b-think": "OLMO-3\n7B-THINK",
        "claude-sonnet-4-6": "CLAUDE\nSONNET 4.6",
    }
    if model_name in custom_labels:
        return custom_labels[model_name]
    display = str(model_name).replace("-it", "-IT").replace("-oss", "-OSS")
    return display.replace("-groq", "").upper()


def _paired_t_test_and_ci(samples: np.ndarray) -> tuple[float, float, int, float, float, float, float, float]:
    """Paired two-sided t-test and 95% CI over paired differences."""
    if samples.size == 0:
        return 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 1.0
    mean_difference = float(np.mean(samples))
    degrees_of_freedom = int(samples.size - 1)
    if samples.size == 1:
        return mean_difference, 0.0, degrees_of_freedom, 0.0, mean_difference, mean_difference, 0.0, 1.0
    standard_deviation = float(np.std(samples, ddof=1))
    standard_error = standard_deviation / sqrt(float(samples.size))
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        p_value = 1.0 if np.isclose(mean_difference, 0.0) else 0.0
        return mean_difference, standard_deviation, degrees_of_freedom, standard_error, mean_difference, mean_difference, 0.0, p_value
    t_statistic = mean_difference / standard_error
    p_value = 2.0 * float(t.sf(abs(t_statistic), df=degrees_of_freedom))
    margin = float(t.ppf(0.975, df=degrees_of_freedom) * standard_error)
    return (
        mean_difference,
        standard_deviation,
        degrees_of_freedom,
        standard_error,
        mean_difference - margin,
        mean_difference + margin,
        float(t_statistic),
        p_value,
    )


def _paired_bootstrap_ci(
    samples: np.ndarray,
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float, int, float, float, float, float, float]:
    """Mean paired difference with percentile bootstrap 95% CI."""
    (
        mean_difference,
        standard_deviation,
        degrees_of_freedom,
        standard_error,
        _t_ci_low,
        _t_ci_high,
        t_statistic,
        p_value,
    ) = _paired_t_test_and_ci(samples)
    if samples.size <= 1:
        return (
            mean_difference,
            standard_deviation,
            degrees_of_freedom,
            standard_error,
            mean_difference,
            mean_difference,
            t_statistic,
            p_value,
        )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, samples.size, size=(int(iterations), samples.size))
    bootstrap_means = np.mean(samples[sample_indices], axis=1)
    ci_low, ci_high = np.percentile(bootstrap_means, [2.5, 97.5])
    return (
        mean_difference,
        standard_deviation,
        degrees_of_freedom,
        standard_error,
        float(ci_low),
        float(ci_high),
        t_statistic,
        p_value,
    )


def _iter_model_evaluated_paths(model_name: str, *, comparison_setting: str) -> list[Path]:
    """Return only evaluated outputs for the exact full-replacement comparison.

    RAW_OUTPUTS may also contain partial-replacement settings such as
    `fictional_10pct`. Keep this figure input explicit so stale auxiliary
    runs cannot silently participate in aggregation.
    """
    settings = (BASELINE_SETTING, comparison_setting)
    paths: list[Path] = []
    for setting in settings:
        paths.extend(RAW_OUTPUTS_DIR.rglob(f"*_{setting}_{model_name}_evaluated_outputs.yaml"))
    return sorted(paths)


def _load_cache() -> dict[str, dict]:
    if not CACHE_PATH.exists():
        return {}
    payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _judge_cache_key(*, question_text: str, ground_truth: str, predicted_answer: str) -> str:
    payload = {
        "judge_model": JUDGE_MODEL,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "question_text": question_text,
        "ground_truth": ground_truth,
        "predicted_answer": predicted_answer,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def _judge_prediction(
    *,
    question_text: str,
    ground_truth: str,
    predicted_answer: str,
    cache: dict[str, dict],
    allow_live_judge: bool = False,
) -> bool:
    cache_key = _judge_cache_key(
        question_text=question_text,
        ground_truth=ground_truth,
        predicted_answer=predicted_answer,
    )
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("judge_match"), bool):
        return bool(cached["judge_match"])
    if not allow_live_judge:
        raise RuntimeError(
            "Missing Judge Match cache entry while live judging is disabled. "
            "Run scripts/analysis/run_full_groq_judge_eval.py for this model first, "
            "or pass --allow-live-judge."
        )

    user_prompt = build_judge_prompt(question_text, ground_truth, predicted_answer)
    response = generate_text(
        TextGenerationRequest(
            provider=JUDGE_PROVIDER,
            model=JUDGE_MODEL,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=JUDGE_MAX_TOKENS,
            seed=JUDGE_SEED,
        )
    )
    normalized = str(response.text or "").strip().upper()
    if not normalized:
        response = generate_text(
            TextGenerationRequest(
                provider=JUDGE_PROVIDER,
                model=JUDGE_MODEL,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
                max_tokens=JUDGE_RETRY_MAX_TOKENS,
                seed=JUDGE_SEED,
            )
        )
        normalized = str(response.text or "").strip().upper()
    if normalized.startswith("CORRECT"):
        verdict = True
    elif normalized.startswith("INCORRECT"):
        verdict = False
    else:
        raise RuntimeError(f"Unparseable judge response: {response.text!r}")

    cache[cache_key] = {
        "judge_match": verdict,
        "judge_raw_output": response.text,
    }
    return verdict


def _normalize_question_type(qtype: str) -> str:
    value = str(qtype or "").strip().lower()
    if value in QUESTION_TYPE_ORDER:
        return value
    return value


def _normalize_answer_behavior(value: str) -> str:
    return str(value or "").strip().lower()


def _load_pair_outcomes(
    model_name: str,
    cache: dict[str, dict],
    *,
    allow_live_judge: bool = False,
    comparison_setting: str = DEFAULT_COMPARISON_SETTING,
    answer_behaviors: set[str] | None = None,
    require_current_outputs: bool = True,
) -> tuple[dict[str, dict], int]:
    pair_outcomes: dict[str, dict] = {}
    judged_rows = 0
    for evaluated_path in _iter_model_evaluated_paths(model_name, comparison_setting=comparison_setting):
        payload = yaml.load(evaluated_path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}
        setting_id = str(payload.get("document_setting") or "").strip().lower()
        if setting_id not in {BASELINE_SETTING, comparison_setting}:
            continue
        has_saved_judge_provenance = bool(
            str(payload.get("judge_provider") or "").strip()
            and str(payload.get("judge_model_name") or "").strip()
        )
        if require_current_outputs:
            issues = blocking_issues(
                audit_saved_output_payload(
                    payload,
                    path=evaluated_path,
                    check_score_metadata=True,
                )
            )
            if issues:
                raise RuntimeError(
                    f"Stale or inconsistent evaluated output for {model_name}: "
                    f"{format_issue_summary(issues)}"
                )
        document_id = str(payload.get("document_id") or "").strip()
        if is_excluded_evaluation_document(document_id):
            continue
        variant_id = str(payload.get("document_variant_id") or "v01").strip()

        for row in payload.get("results") or []:
            answer_behavior = _normalize_answer_behavior(
                str(row.get("answer_behavior") or row.get("answer_type") or "")
            )
            if answer_behaviors is not None and answer_behavior not in answer_behaviors:
                continue
            pair_key = str(row.get("pair_key") or f"{document_id}::{row.get('question_id') or ''}")
            entry = pair_outcomes.setdefault(
                pair_key,
                {
                    "question_type": _normalize_question_type(str(row.get("question_type") or "")),
                    "answer_behavior": answer_behavior,
                    BASELINE_SETTING: None,
                    comparison_setting: {},
                },
            )
            if not entry.get("question_type"):
                entry["question_type"] = _normalize_question_type(str(row.get("question_type") or ""))
            if not entry.get("answer_behavior"):
                entry["answer_behavior"] = answer_behavior

            exact_match = row.get("exact_match") is True
            if exact_match:
                score = 1.0
            else:
                ground_truth = str(row.get("ground_truth") or "")
                if find_entity_refs(ground_truth):
                    raise ValueError(
                        f"Unresolved ground truth in {evaluated_path}: "
                        f"{row.get('question_id')}: {ground_truth!r}"
                    )
                if not judge_match_is_allowed(
                    answer_schema=str(row.get("answer_schema") or ""),
                    parsed_output_canonical=str(row.get("parsed_output_canonical") or ""),
                ):
                    score = 0.0
                else:
                    # When a run records explicit judge provenance, keep its
                    # saved verdicts aligned with the published metrics. Older
                    # YAMLs without provenance still use the versioned cache.
                    question_text = str(row.get("question_text") or "")
                    predicted_answer = str(row.get("raw_output") or row.get("parsed_output") or "")
                    cache_key = _judge_cache_key(
                        question_text=question_text,
                        ground_truth=ground_truth,
                        predicted_answer=predicted_answer,
                    )
                    cached = cache.get(cache_key)
                    if has_saved_judge_provenance and isinstance(row.get("judge_match"), bool):
                        judge_match = bool(row["judge_match"])
                    elif isinstance(cached, dict) and isinstance(cached.get("judge_match"), bool):
                        judge_match = bool(cached["judge_match"])
                    elif isinstance(row.get("judge_match"), bool):
                        judge_match = bool(row["judge_match"])
                    else:
                        judge_match = _judge_prediction(
                            question_text=question_text,
                            ground_truth=ground_truth,
                            predicted_answer=predicted_answer,
                            cache=cache,
                            allow_live_judge=allow_live_judge,
                        )
                    judged_rows += 1
                    score = 1.0 if judge_match else 0.0

            if setting_id == BASELINE_SETTING:
                entry[BASELINE_SETTING] = score
            else:
                entry[comparison_setting][variant_id] = score
    return pair_outcomes, judged_rows


def _build_rows_for_model(
    model_name: str,
    cache: dict[str, dict],
    *,
    question_groups: tuple[str, ...] = QUESTION_TYPE_ORDER,
    allow_live_judge: bool = False,
    comparison_setting: str = DEFAULT_COMPARISON_SETTING,
    answer_behaviors: set[str] | None = None,
    ci_method: str = "paired_t",
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    require_current_outputs: bool = True,
) -> tuple[list[DropRow], int]:
    outcomes, judged_rows = _load_pair_outcomes(
        model_name,
        cache,
        allow_live_judge=allow_live_judge,
        comparison_setting=comparison_setting,
        answer_behaviors=answer_behaviors,
        require_current_outputs=require_current_outputs,
    )
    grouped_pairs: dict[str, list[tuple[float, float]]] = {qtype: [] for qtype in question_groups}
    grouped_variant_pairs: dict[str, dict[str, list[tuple[float, float]]]] = {
        qtype: {variant_id: [] for variant_id in EXPECTED_VARIANT_IDS}
        for qtype in question_groups
    }

    for entry in outcomes.values():
        factual = entry.get(BASELINE_SETTING)
        fictional = entry.get(comparison_setting) or {}
        qtype = _normalize_question_type(entry.get("question_type") or "")
        if factual is None:
            continue
        if set(str(v) for v in fictional.keys()) != set(EXPECTED_VARIANT_IDS):
            continue
        if qtype not in QUESTION_TYPE_ORDER:
            continue
        variant_scores = {variant_id: float(fictional[variant_id]) for variant_id in EXPECTED_VARIANT_IDS}
        factual_score = float(factual)
        fictional_mean = float(np.mean([variant_scores[variant_id] for variant_id in EXPECTED_VARIANT_IDS]))
        for group_name in question_groups:
            if qtype in QUESTION_GROUP_MEMBERS[group_name]:
                grouped_pairs[group_name].append((factual_score, fictional_mean))
                for variant_id, variant_score in variant_scores.items():
                    grouped_variant_pairs[group_name][variant_id].append((factual_score, variant_score))

    rows: list[DropRow] = []
    for qtype in question_groups:
        pairs = grouped_pairs[qtype]
        if not pairs:
            continue
        factual_arr = np.asarray([p[0] for p in pairs], dtype=float)
        fictional_arr = np.asarray([p[1] for p in pairs], dtype=float)
        diff_arr = fictional_arr - factual_arr
        if ci_method == "bootstrap":
            estimate = _paired_bootstrap_ci(
                diff_arr,
                iterations=bootstrap_iterations,
                seed=bootstrap_seed,
            )
        else:
            estimate = _paired_t_test_and_ci(diff_arr)
        (
            mean_difference,
            standard_deviation_difference,
            degrees_of_freedom,
            standard_error_difference,
            ci_low,
            ci_high,
            t_statistic,
            p_value,
        ) = estimate
        variant_p_values: dict[str, float] = {}
        variant_t_statistics: dict[str, float] = {}
        variant_differences: dict[str, tuple[float, ...]] = {}
        for variant_id in EXPECTED_VARIANT_IDS:
            variant_pairs = grouped_variant_pairs[qtype][variant_id]
            if not variant_pairs:
                continue
            variant_diff_arr = np.asarray([p[1] - p[0] for p in variant_pairs], dtype=float)
            (
                _variant_mean_difference,
                _variant_standard_deviation_difference,
                _variant_degrees_of_freedom,
                _variant_standard_error_difference,
                _variant_ci_low,
                _variant_ci_high,
                variant_t_statistic,
                variant_p_value,
            ) = _paired_t_test_and_ci(variant_diff_arr)
            variant_p_values[variant_id] = float(variant_p_value)
            variant_t_statistics[variant_id] = float(variant_t_statistic)
            variant_differences[variant_id] = tuple(float(value) for value in variant_diff_arr)
        rows.append(
            DropRow(
                model_name=model_name,
                question_type=qtype,
                count_pairs=int(diff_arr.size),
                factual_score=float(np.mean(factual_arr)),
                fictional_score=float(np.mean(fictional_arr)),
                mean_difference=mean_difference,
                standard_deviation_difference=standard_deviation_difference,
                standard_error_difference=standard_error_difference,
                degrees_of_freedom=degrees_of_freedom,
                ci_low=ci_low,
                ci_high=ci_high,
                t_statistic=t_statistic,
                p_value=p_value,
                variant_p_values=variant_p_values,
                variant_t_statistics=variant_t_statistics,
                variant_differences=variant_differences,
            )
        )
    return rows, judged_rows


DEFAULT_OUTPUT_STEM = "judge_match_drop_by_question_type_groq_current"


def _write_csv(
    rows: list[DropRow],
    *,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    significance_test: str = SIGNIFICANCE_TEST,
    confidence_interval: str = CONFIDENCE_INTERVAL,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{output_stem}.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "question_type",
                "count_pairs",
                "factual_score",
                "fictional_score",
                "mean_difference",
                "standard_deviation_difference",
                "standard_error_difference",
                "degrees_of_freedom",
                "ci_low",
                "ci_high",
                "t_statistic",
                "p_value",
                "p_value_adjusted",
                "pvalue_correction",
                "correction_family_size",
                "significance_alpha",
                "significant",
                "significance_source",
                "variant_p_values",
                "variant_p_values_adjusted",
                "variant_t_statistics",
                "significant_variant_ids",
                "stars",
                "significance_test",
                "confidence_interval",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row.model_name,
                    "question_type": row.question_type,
                    "count_pairs": row.count_pairs,
                    "factual_score": row.factual_score,
                    "fictional_score": row.fictional_score,
                    "mean_difference": row.mean_difference,
                    "standard_deviation_difference": row.standard_deviation_difference,
                    "standard_error_difference": row.standard_error_difference,
                    "degrees_of_freedom": row.degrees_of_freedom,
                    "ci_low": row.ci_low,
                    "ci_high": row.ci_high,
                    "t_statistic": row.t_statistic,
                    "p_value": row.p_value,
                    "p_value_adjusted": row.p_value_adjusted,
                    "pvalue_correction": row.pvalue_correction,
                    "correction_family_size": row.correction_family_size,
                    "significance_alpha": row.significance_alpha,
                    "significant": row.is_significant,
                    "significance_source": row.significance_source,
                    "variant_p_values": json.dumps(row.variant_p_values, sort_keys=True),
                    "variant_p_values_adjusted": json.dumps(row.variant_p_values_adjusted, sort_keys=True),
                    "variant_t_statistics": json.dumps(row.variant_t_statistics, sort_keys=True),
                    "significant_variant_ids": json.dumps(list(row.significant_variant_ids)),
                    "stars": row.stars,
                    "significance_test": significance_test,
                    "confidence_interval": confidence_interval,
                }
            )
    return output_path


def _read_rows_from_csv(*, output_stem: str = DEFAULT_OUTPUT_STEM) -> list[DropRow]:
    input_path = OUTPUT_DIR / f"{output_stem}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Cached score CSV not found: {input_path}")
    rows: list[DropRow] = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            rows.append(
                DropRow(
                    model_name=str(record["model_name"]),
                    question_type=str(record["question_type"]),
                    count_pairs=int(record["count_pairs"]),
                    factual_score=float(record["factual_score"]),
                    fictional_score=float(record["fictional_score"]),
                    mean_difference=float(record["mean_difference"]),
                    standard_deviation_difference=float(record["standard_deviation_difference"]),
                    standard_error_difference=float(record["standard_error_difference"]),
                    degrees_of_freedom=int(record["degrees_of_freedom"]),
                    ci_low=float(record["ci_low"]),
                    ci_high=float(record["ci_high"]),
                    t_statistic=float(record["t_statistic"]),
                    p_value=float(record["p_value"]),
                    p_value_adjusted=(
                        float(record["p_value_adjusted"])
                        if record.get("p_value_adjusted")
                        else None
                    ),
                    pvalue_correction=record.get("pvalue_correction") or None,
                    correction_family_size=(
                        int(record["correction_family_size"])
                        if record.get("correction_family_size")
                        else None
                    ),
                    significance_alpha=(
                        float(record["significance_alpha"])
                        if record.get("significance_alpha")
                        else 0.05
                    ),
                    significant=(
                        str(record["significant"]).strip().lower() == "true"
                        if record.get("significant")
                        else None
                    ),
                    significance_source=record.get("significance_source") or "aggregate_paired_mean",
                    variant_p_values=(
                        {
                            str(key): float(value)
                            for key, value in json.loads(record["variant_p_values"]).items()
                        }
                        if record.get("variant_p_values")
                        else {}
                    ),
                    variant_p_values_adjusted=(
                        {
                            str(key): float(value)
                            for key, value in json.loads(record["variant_p_values_adjusted"]).items()
                        }
                        if record.get("variant_p_values_adjusted")
                        else {}
                    ),
                    variant_t_statistics=(
                        {
                            str(key): float(value)
                            for key, value in json.loads(record["variant_t_statistics"]).items()
                        }
                        if record.get("variant_t_statistics")
                        else {}
                    ),
                    significant_variant_ids=(
                        tuple(str(value) for value in json.loads(record["significant_variant_ids"]))
                        if record.get("significant_variant_ids")
                        else ()
                    ),
                )
            )
    return rows


def _read_judged_rows_from_json(*, output_stem: str = DEFAULT_OUTPUT_STEM) -> int:
    input_path = OUTPUT_DIR / f"{output_stem}.json"
    if not input_path.exists():
        return 0
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return int(payload.get("judged_non_exact_rows") or 0)


def _apply_pvalue_correction(
    rows: list[DropRow],
    *,
    method: str = "none",
    alpha: float = 0.05,
    family_size: int | None = None,
) -> None:
    method_norm = str(method or "none").strip().lower()
    if method_norm not in {"none", "bonferroni"}:
        raise ValueError(f"Unsupported p-value correction method: {method}")
    if method_norm == "none":
        return

    finite_p_rows = [row for row in rows if np.isfinite(row.p_value)]
    resolved_family_size = int(family_size) if family_size is not None else len(finite_p_rows)
    if resolved_family_size <= 0:
        raise ValueError("Bonferroni correction requires a positive family size.")

    for row in rows:
        row.pvalue_correction = method_norm
        row.correction_family_size = resolved_family_size
        row.significance_alpha = float(alpha)
        if not np.isfinite(row.p_value):
            row.p_value_adjusted = None
            row.significant = False
            continue
        row.p_value_adjusted = min(1.0, float(row.p_value) * float(resolved_family_size))
        row.significant = bool(row.p_value_adjusted < float(alpha))


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _variant_ids_and_p_values(row: DropRow) -> tuple[list[str], np.ndarray]:
    variant_ids = [variant_id for variant_id in EXPECTED_VARIANT_IDS if variant_id in row.variant_p_values]
    p_values = np.asarray([float(row.variant_p_values[variant_id]) for variant_id in variant_ids], dtype=float)
    return variant_ids, p_values


def _variant_difference_matrix(row: DropRow) -> tuple[list[str], np.ndarray]:
    variant_ids = [variant_id for variant_id in EXPECTED_VARIANT_IDS if variant_id in row.variant_differences]
    if not variant_ids:
        raise ValueError(
            "This correction requires per-variant paired differences. "
            "Regenerate from evaluated YAMLs instead of --plot-from-cache."
        )
    lengths = {len(row.variant_differences[variant_id]) for variant_id in variant_ids}
    if len(lengths) != 1:
        raise ValueError(f"Mismatched variant paired-difference lengths for {row.model_name}/{row.question_type}.")
    matrix = np.asarray(
        [[row.variant_differences[variant_id][idx] for variant_id in variant_ids] for idx in range(next(iter(lengths)))],
        dtype=float,
    )
    return variant_ids, matrix


def _correlation_matrix_from_columns(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Expected a non-empty 2D matrix for correlation estimation.")
    column_stds = np.std(matrix, axis=0, ddof=1)
    corr = np.eye(matrix.shape[1], dtype=float)
    variable = column_stds > 1e-12
    if int(np.sum(variable)) >= 2:
        variable_corr = np.corrcoef(matrix[:, variable], rowvar=False)
        corr[np.ix_(variable, variable)] = variable_corr
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    if float(np.min(eigvals)) < 1e-8:
        eigvals = np.clip(eigvals, 1e-8, None)
        corr = (eigvecs * eigvals) @ eigvecs.T
        scale = np.sqrt(np.diag(corr))
        corr = corr / np.outer(scale, scale)
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
    return corr


def _apply_variant_pvalue_method(
    row: DropRow,
    *,
    method: str,
    alpha: float,
) -> None:
    variant_ids, p_values = _variant_ids_and_p_values(row)
    if not variant_ids:
        raise ValueError(
            "Variant-level correction requires per-variant p-values. "
            "Regenerate from evaluated YAMLs first."
        )
    method_map = {
        "bonferroni": "bonferroni",
        "holm": "holm",
        "sidak": "sidak",
        "hochberg": "simes-hochberg",
        "hommel": "hommel",
    }
    reject, adjusted_values, _, _ = multipletests(
        p_values,
        alpha=float(alpha),
        method=method_map[method],
    )
    row.variant_p_values_adjusted = {
        variant_id: float(adjusted_value)
        for variant_id, adjusted_value in zip(variant_ids, adjusted_values, strict=True)
    }
    row.significant_variant_ids = tuple(
        variant_id for variant_id, is_rejected in zip(variant_ids, reject, strict=True) if bool(is_rejected)
    )


def _apply_variant_dunnett_method(
    row: DropRow,
    *,
    alpha: float,
    maxpts: int,
    seed: int,
) -> None:
    variant_ids, matrix = _variant_difference_matrix(row)
    corr = _correlation_matrix_from_columns(matrix)
    df = max(int(matrix.shape[0] - 1), 1)
    adjusted: dict[str, float] = {}
    for variant_id in variant_ids:
        observed_t = abs(float(row.variant_t_statistics.get(variant_id, 0.0)))
        if not np.isfinite(observed_t):
            adjusted[variant_id] = 0.0
            continue
        if observed_t <= 0.0:
            adjusted[variant_id] = 1.0
            continue
        upper = np.full(len(variant_ids), observed_t, dtype=float)
        lower = -upper
        random_state = _stable_seed(seed, row.model_name, row.question_type, variant_id, "dunnett")
        inside_probability = multivariate_t.cdf(
            upper,
            loc=np.zeros(len(variant_ids), dtype=float),
            shape=corr,
            df=df,
            allow_singular=True,
            maxpts=int(maxpts),
            lower_limit=lower,
            random_state=random_state,
        )
        adjusted[variant_id] = float(min(1.0, max(0.0, 1.0 - float(inside_probability))))
    row.variant_p_values_adjusted = adjusted
    row.significant_variant_ids = tuple(
        variant_id
        for variant_id in variant_ids
        if adjusted.get(variant_id, 1.0) < float(alpha)
    )


def _apply_variant_max_t_method(
    row: DropRow,
    *,
    alpha: float,
    permutations: int,
    seed: int,
) -> None:
    variant_ids, matrix = _variant_difference_matrix(row)
    n_pairs, n_variants = matrix.shape
    if n_pairs < 2:
        adjusted = {variant_id: 1.0 for variant_id in variant_ids}
        row.variant_p_values_adjusted = adjusted
        row.significant_variant_ids = ()
        return
    observed = np.asarray(
        [abs(float(row.variant_t_statistics.get(variant_id, 0.0))) for variant_id in variant_ids],
        dtype=float,
    )
    rng = np.random.default_rng(_stable_seed(seed, row.model_name, row.question_type, "max_t"))
    sum_squares = np.sum(matrix * matrix, axis=0)
    counts = np.zeros(n_variants, dtype=np.int64)
    generated = 0
    batch_size = 1000
    while generated < int(permutations):
        current = min(batch_size, int(permutations) - generated)
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=(current, n_pairs))
        permuted_sums = signs @ matrix
        means = permuted_sums / float(n_pairs)
        variances = (sum_squares[None, :] - float(n_pairs) * means * means) / float(n_pairs - 1)
        variances = np.maximum(variances, 0.0)
        standard_errors = np.sqrt(variances / float(n_pairs))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.where(standard_errors > 0.0, means / standard_errors, 0.0)
        max_abs_t = np.max(np.abs(t_values), axis=1)
        counts += np.sum(max_abs_t[:, None] >= observed[None, :], axis=0)
        generated += current

    adjusted_values = (counts.astype(float) + 1.0) / (float(permutations) + 1.0)
    row.variant_p_values_adjusted = {
        variant_id: float(adjusted_value)
        for variant_id, adjusted_value in zip(variant_ids, adjusted_values, strict=True)
    }
    row.significant_variant_ids = tuple(
        variant_id
        for variant_id, adjusted_value in zip(variant_ids, adjusted_values, strict=True)
        if float(adjusted_value) < float(alpha)
    )


def _apply_variant_level_correction_stars(
    rows: list[DropRow],
    *,
    method: str,
    alpha: float = 0.05,
    family_size: int | None = None,
    max_t_permutations: int = DEFAULT_MAX_T_PERMUTATIONS,
    max_t_seed: int = DEFAULT_MAX_T_SEED,
    dunnett_maxpts: int = DEFAULT_DUNNETT_MAXPTS,
) -> None:
    method_norm = str(method or "bonferroni").strip().lower()
    if method_norm not in SUPPORTED_VARIANT_LEVEL_CORRECTIONS:
        raise ValueError(f"Unsupported variant-level correction method: {method}")
    resolved_family_size = int(family_size) if family_size is not None else len(EXPECTED_VARIANT_IDS)
    if resolved_family_size <= 0:
        raise ValueError("Variant-level correction requires a positive family size.")

    for row in rows:
        if method_norm in {"bonferroni", "holm", "sidak", "hochberg", "hommel"}:
            _apply_variant_pvalue_method(row, method=method_norm, alpha=float(alpha))
            if family_size is not None and method_norm == "bonferroni":
                row.variant_p_values_adjusted = {
                    variant_id: min(1.0, float(p_value) * float(resolved_family_size))
                    for variant_id, p_value in row.variant_p_values.items()
                    if np.isfinite(float(p_value))
                }
                row.significant_variant_ids = tuple(
                    variant_id
                    for variant_id in EXPECTED_VARIANT_IDS
                    if row.variant_p_values_adjusted.get(variant_id, 1.0) < float(alpha)
                )
        elif method_norm == "dunnett":
            _apply_variant_dunnett_method(
                row,
                alpha=float(alpha),
                maxpts=int(dunnett_maxpts),
                seed=int(max_t_seed),
            )
        elif method_norm == "max_t":
            _apply_variant_max_t_method(
                row,
                alpha=float(alpha),
                permutations=int(max_t_permutations),
                seed=int(max_t_seed),
            )
        row.correction_family_size = resolved_family_size
        row.significance_alpha = float(alpha)
        row.significance_source = "variant_level_factual_vs_each_fictional"
        row.pvalue_correction = method_norm
        row.p_value_adjusted = (
            min(row.variant_p_values_adjusted.values())
            if row.variant_p_values_adjusted
            else None
        )
        row.significant = bool(row.significant_variant_ids)


def _write_json(
    rows: list[DropRow],
    *,
    judged_rows: int,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    significance_test: str = SIGNIFICANCE_TEST,
    confidence_interval: str = CONFIDENCE_INTERVAL,
    bootstrap_iterations: int | None = None,
    bootstrap_seed: int | None = None,
    pvalue_correction: str | None = None,
    correction_family_size: int | None = None,
    significance_alpha: float | None = None,
    max_t_permutations: int | None = None,
    max_t_seed: int | None = None,
    dunnett_maxpts: int | None = None,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{output_stem}.json"
    payload = {
        "judge_model": JUDGE_MODEL,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "significance_test": significance_test,
        "confidence_interval": confidence_interval,
        "judged_non_exact_rows": judged_rows,
        "rows": [
            {
                "model_name": row.model_name,
                "question_type": row.question_type,
                "count_pairs": row.count_pairs,
                "factual_score": row.factual_score,
                "fictional_score": row.fictional_score,
                "mean_difference": row.mean_difference,
                "standard_deviation_difference": row.standard_deviation_difference,
                "standard_error_difference": row.standard_error_difference,
                "degrees_of_freedom": row.degrees_of_freedom,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "t_statistic": row.t_statistic,
                "p_value": row.p_value,
                "p_value_adjusted": row.p_value_adjusted,
                "pvalue_correction": row.pvalue_correction,
                "correction_family_size": row.correction_family_size,
                "significance_alpha": row.significance_alpha,
                "significant": row.is_significant,
                "significance_source": row.significance_source,
                "variant_p_values": row.variant_p_values,
                "variant_p_values_adjusted": row.variant_p_values_adjusted,
                "variant_t_statistics": row.variant_t_statistics,
                "significant_variant_ids": list(row.significant_variant_ids),
                "stars": row.stars,
            }
            for row in rows
        ],
    }
    if bootstrap_iterations is not None:
        payload["bootstrap_iterations"] = int(bootstrap_iterations)
    if bootstrap_seed is not None:
        payload["bootstrap_seed"] = int(bootstrap_seed)
    if pvalue_correction:
        payload["pvalue_correction"] = str(pvalue_correction)
    if correction_family_size is not None:
        payload["correction_family_size"] = int(correction_family_size)
    if significance_alpha is not None:
        payload["significance_alpha"] = float(significance_alpha)
    if max_t_permutations is not None:
        payload["max_t_permutations"] = int(max_t_permutations)
    if max_t_seed is not None:
        payload["max_t_seed"] = int(max_t_seed)
    if dunnett_maxpts is not None:
        payload["dunnett_maxpts"] = int(dunnett_maxpts)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _plot(
    rows: list[DropRow],
    *,
    target_models: tuple[str, ...],
    question_groups: tuple[str, ...] = QUESTION_TYPE_ORDER,
    output_stem: str = DEFAULT_OUTPUT_STEM,
    group_gap: float = 1.7,
    bar_width: float = 0.82,
    extra_gap_after: dict[str, float] | None = None,
    clamp_positive_axis: bool = False,
    use_hatches: bool = True,
    legend_placement: str = "inside-right",
    figure_width: float | None = None,
    figure_height: float | None = None,
    left_margin: float | None = None,
    ylabel: str | None = None,
    ylabel_fontsize: float | None = None,
    ylabel_y: float | None = None,
    model_label_fontsize: float = 12.0,
    legend_fontsize: float | None = None,
    show_model_labels: bool = True,
    show_x_tick_labels: bool = False,
) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.3,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    extra_gap_after = extra_gap_after or {}
    fig_width = float(figure_width) if figure_width else max(10.8, 1.62 * len(target_models))
    fig_height = float(figure_height) if figure_height else 6.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    group_size = len(question_groups)

    x_positions: list[float] = []
    heights: list[float] = []
    lower_errs: list[float] = []
    upper_errs: list[float] = []
    colors: list[str] = []
    hatches: list[str] = []
    stars_meta: list[tuple[float, float, str, str]] = []
    model_centers: list[tuple[float, str]] = []

    row_lookup = {(row.model_name, row.question_type): row for row in rows}
    global_min = 0.0
    global_max = 0.0

    base = 0.0
    for model_name in target_models:
        first_x = base
        last_x = base + group_size - 1
        model_centers.append(((first_x + last_x) / 2.0, _display_model_name(model_name)))
        for offset, qtype in enumerate(question_groups):
            row = row_lookup.get((model_name, qtype))
            if row is None:
                continue
            x = base + offset
            mean = 100.0 * row.mean_difference
            ci_low = 100.0 * row.ci_low
            ci_high = 100.0 * row.ci_high
            x_positions.append(x)
            heights.append(mean)
            lower_errs.append(mean - ci_low)
            upper_errs.append(ci_high - mean)
            colors.append(QUESTION_TYPE_COLORS[qtype])
            hatches.append(QUESTION_TYPE_HATCHES[qtype] if use_hatches else "")
            global_min = min(global_min, ci_low)
            global_max = max(global_max, ci_high)
            if row.stars:
                stars_meta.append((x, ci_low - 0.38, row.stars, "top"))
        base += group_size + group_gap + float(extra_gap_after.get(model_name, 0.0))

    bars = ax.bar(
        x_positions,
        heights,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=1.7,
        zorder=2,
    )
    if use_hatches:
        for bar, hatch in zip(bars, hatches, strict=True):
            bar.set_hatch(hatch)

    ax.errorbar(
        x_positions,
        heights,
        yerr=np.vstack([lower_errs, upper_errs]),
        fmt="none",
        ecolor="black",
        elinewidth=1.7,
        capsize=5,
        capthick=1.7,
        zorder=3,
    )

    ax.axhline(0.0, color="black", linewidth=1.3, zorder=1)
    ax.set_ylabel(ylabel or r"Mean Performance Drop (Factual $\rightarrow$ Fictional - %)")
    if ylabel_fontsize is not None:
        ax.yaxis.label.set_size(float(ylabel_fontsize))
    if ylabel_y is not None:
        ax.yaxis.set_label_coords(-0.06, float(ylabel_y))
    if show_x_tick_labels:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [QUESTION_TYPE_LABELS[qtype] for _model_name in target_models for qtype in question_groups],
            rotation=25,
            ha="right",
        )
    else:
        ax.set_xticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.18, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ymin = min(global_min - 1.4, -1.0)
    if legend_placement == "inside-bottom" and len(question_groups) > 1:
        ymin -= 1.7
    if clamp_positive_axis:
        ymax = 0.0
    else:
        y_span = max(global_max - ymin, 1.0)
        # Keep just enough headroom to avoid clipping caps/stars without wasting vertical space.
        ymax = max(global_max + min(max(0.12, 0.025 * y_span), 0.45), 0.12)
    ax.set_ylim(ymin, ymax)

    if show_model_labels:
        for x, label in model_centers:
            ax.text(
                x,
                1.035,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=float(model_label_fontsize),
                clip_on=False,
            )

    for x, y, stars, va in stars_meta:
        ax.text(x, y, stars, ha="center", va=va, fontsize=13, fontweight="bold")

    if len(question_groups) > 1 and legend_placement != "none":
        legend_handles = []
        legend_labels = []
        resolved_legend_fontsize = float(legend_fontsize) if legend_fontsize is not None else None
        from matplotlib.patches import Patch
        for qtype in question_groups:
            legend_handles.append(
                Patch(
                    facecolor=QUESTION_TYPE_COLORS[qtype],
                    edgecolor="black",
                    hatch=QUESTION_TYPE_HATCHES[qtype] if use_hatches else "",
                    linewidth=1.5,
                )
            )
            if legend_placement.startswith("compact-"):
                legend_labels.append(COMPACT_QUESTION_TYPE_LABELS[qtype])
            else:
                legend_labels.append(QUESTION_TYPE_LABELS[qtype])

        if legend_placement == "below":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=len(question_groups),
                frameon=False,
                fontsize=resolved_legend_fontsize or 15.0,
                handlelength=2.45,
                handleheight=1.25,
                columnspacing=1.35,
                labelspacing=0.8,
            )
        elif legend_placement == "lower-left":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                bbox_to_anchor=(0.02, -0.025),
                ncol=len(question_groups),
                frameon=False,
                fontsize=resolved_legend_fontsize or 15.0,
                handlelength=2.45,
                handleheight=1.25,
                columnspacing=1.35,
                labelspacing=0.8,
            )
        elif legend_placement == "below-tight":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.025),
                ncol=len(question_groups),
                frameon=False,
                fontsize=resolved_legend_fontsize or 13.5,
                handlelength=2.0,
                handleheight=1.0,
                columnspacing=1.05,
                labelspacing=0.45,
            )
        elif legend_placement == "inside-bottom":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.035),
                ncol=len(question_groups),
                frameon=False,
                fontsize=resolved_legend_fontsize or 15.0,
                handlelength=2.0,
                handleheight=1.0,
                borderaxespad=0.12,
                columnspacing=1.05,
                labelspacing=0.45,
            )
        elif legend_placement == "compact-upper-right":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                frameon=False,
                fontsize=resolved_legend_fontsize or 10.5,
                handlelength=1.25,
                handleheight=0.75,
                borderaxespad=0.15,
                labelspacing=0.28,
            )
        elif legend_placement == "upper-right":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                frameon=False,
                fontsize=resolved_legend_fontsize or 9.5,
                handlelength=1.25,
                handleheight=0.72,
                borderaxespad=0.15,
                labelspacing=0.25,
            )
        elif legend_placement == "upper-right-2col":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.985),
                ncol=2,
                frameon=False,
                fontsize=resolved_legend_fontsize or 12.5,
                handlelength=1.45,
                handleheight=0.86,
                borderaxespad=0.12,
                columnspacing=0.95,
                labelspacing=0.45,
            )
        elif legend_placement == "lower-right-3col":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="lower right",
                bbox_to_anchor=(0.985, 0.02),
                ncol=3,
                frameon=False,
                fontsize=resolved_legend_fontsize or 12.5,
                handlelength=1.45,
                handleheight=0.86,
                borderaxespad=0.1,
                columnspacing=0.95,
                labelspacing=0.45,
            )
        elif legend_placement == "compact-upper-left":
            ax.legend(
                legend_handles,
                legend_labels,
                loc="upper left",
                bbox_to_anchor=(0.02, 0.985),
                frameon=False,
                fontsize=resolved_legend_fontsize or 10.5,
                handlelength=1.25,
                handleheight=0.75,
                borderaxespad=0.15,
                labelspacing=0.28,
            )
        else:
            ax.legend(
                legend_handles,
                legend_labels,
                loc="center right",
                bbox_to_anchor=(0.985, 0.24),
                frameon=False,
                fontsize=resolved_legend_fontsize or 15.0,
                handlelength=2.45,
                handleheight=1.25,
                labelspacing=0.8,
            )

    if legend_placement == "below-tight":
        bottom_margin = 0.105
    else:
        bottom_margin = 0.16 if legend_placement in {"below", "lower-left"} else 0.08
    if show_x_tick_labels:
        bottom_margin = max(bottom_margin, 0.22)
    top_margin = 0.88 if show_model_labels else 0.97
    fig.subplots_adjust(
        left=0.105 if left_margin is None else float(left_margin),
        right=0.985,
        top=top_margin,
        bottom=bottom_margin,
    )
    output_path = OUTPUT_DIR / f"{output_stem}.png"
    fig.savefig(output_path, dpi=260, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Judge Match drop figure by question type.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(TARGET_MODELS),
        help="Model registry names to include in the figure.",
    )
    parser.add_argument(
        "--allow-live-judge",
        action="store_true",
        help="Allow missing non-exact examples to trigger live Groq Judge Match calls.",
    )
    parser.add_argument(
        "--plot-from-cache",
        action="store_true",
        help="Only redraw from the existing output CSV; do not reload evaluated YAMLs or recompute scores.",
    )
    parser.add_argument(
        "--allow-stale-outputs",
        action="store_true",
        help="Disable current-dataset prompt/ground-truth consistency checks before scoring.",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help="Output filename stem for the PNG/CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--input-stem",
        help="Cached CSV/JSON stem to read when --plot-from-cache is used. Defaults to --output-stem.",
    )
    parser.add_argument(
        "--pvalue-correction",
        default="none",
        choices=("none", "bonferroni"),
        help="Optional correction applied to plotted significance stars.",
    )
    parser.add_argument(
        "--significance-alpha",
        type=float,
        default=0.05,
        help="Alpha threshold used after p-value correction.",
    )
    parser.add_argument(
        "--pvalue-family-size",
        type=int,
        help="Optional Bonferroni family size override. Defaults to the number of plotted finite p-values.",
    )
    parser.add_argument(
        "--variant-level-bonferroni-stars",
        action="store_true",
        help=(
            "Use stars from 10 factual-vs-individual-fictional paired t-tests per bar, "
            "Bonferroni-corrected within that variant family."
        ),
    )
    parser.add_argument(
        "--variant-level-correction",
        choices=SUPPORTED_VARIANT_LEVEL_CORRECTIONS,
        help="Use stars from 10 factual-vs-individual-fictional paired t-tests per bar with this correction.",
    )
    parser.add_argument(
        "--max-t-permutations",
        type=int,
        default=DEFAULT_MAX_T_PERMUTATIONS,
        help="Number of paired sign-flip permutations for --variant-level-correction max_t.",
    )
    parser.add_argument(
        "--max-t-seed",
        type=int,
        default=DEFAULT_MAX_T_SEED,
        help="Random seed for max-t permutation and Dunnett numerical integration.",
    )
    parser.add_argument(
        "--dunnett-maxpts",
        type=int,
        default=DEFAULT_DUNNETT_MAXPTS,
        help="Maximum integration points for paired Dunnett-style multivariate-t correction.",
    )
    parser.add_argument(
        "--comparison-setting",
        default=DEFAULT_COMPARISON_SETTING,
        choices=sorted(SUPPORTED_COMPARISON_SETTINGS),
        help="Dataset setting to compare against factual.",
    )
    parser.add_argument(
        "--question-groups",
        nargs="+",
        default=list(QUESTION_TYPE_ORDER),
        choices=list(QUESTION_GROUP_MEMBERS),
        help="Question groups to show as bars. Use reasoning to combine arithmetic, temporal, and inference.",
    )
    parser.add_argument(
        "--answer-behaviors",
        nargs="+",
        choices=sorted(SUPPORTED_ANSWER_BEHAVIORS),
        help="Optional answer behavior filter, e.g. refusal for refusal-only questions.",
    )
    parser.add_argument(
        "--ci-method",
        default="paired_t",
        choices=("paired_t", "bootstrap"),
        help="Method used to compute plotted confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
        help="Number of paired bootstrap resamples when --ci-method bootstrap is used.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Random seed for paired bootstrap confidence intervals.",
    )
    parser.add_argument("--group-gap", type=float, default=1.7, help="Horizontal gap between model bar groups.")
    parser.add_argument("--bar-width", type=float, default=0.82, help="Width of each bar.")
    parser.add_argument("--no-hatches", action="store_true", help="Disable hatch patterns on bars and legend.")
    parser.add_argument(
        "--extra-gap-after",
        nargs="*",
        default=[],
        metavar="MODEL:GAP",
        help="Optional extra horizontal gap after selected model groups, e.g. gemma-4-26b-a4b-it:0.45.",
    )
    parser.add_argument(
        "--clamp-positive-axis",
        action="store_true",
        help="Clamp the upper y-axis limit to zero for drop-only figures.",
    )
    parser.add_argument(
        "--legend-placement",
        default="inside-right",
        choices=(
            "inside-right",
            "lower-left",
            "below",
            "compact-upper-right",
            "upper-right",
            "upper-right-2col",
            "lower-right-3col",
            "compact-upper-left",
            "below-tight",
            "inside-bottom",
            "none",
        ),
        help="Legend placement. Use below for compact figures where an inside legend can hide significance stars.",
    )
    parser.add_argument("--figure-width", type=float, help="Optional figure width in inches.")
    parser.add_argument("--figure-height", type=float, help="Optional figure height in inches.")
    parser.add_argument("--left-margin", type=float, help="Optional Matplotlib left subplot margin.")
    parser.add_argument("--ylabel", help="Optional y-axis label override.")
    parser.add_argument("--ylabel-fontsize", type=float, help="Optional y-axis label font size.")
    parser.add_argument("--ylabel-y", type=float, help="Optional y-axis label vertical coordinate in axes units.")
    parser.add_argument("--model-label-fontsize", type=float, default=12.0, help="Font size for model labels above bar groups.")
    parser.add_argument("--legend-fontsize", type=float, help="Optional legend font size override.")
    parser.add_argument("--hide-model-labels", action="store_true", help="Do not draw model names above bar groups.")
    parser.add_argument("--show-x-tick-labels", action="store_true", help="Label bars directly on the x-axis.")
    args = parser.parse_args()
    target_models = tuple(str(model).strip() for model in args.models if str(model).strip())
    question_groups = tuple(str(group).strip() for group in args.question_groups if str(group).strip())
    answer_behaviors = (
        {str(answer_behavior).strip().lower() for answer_behavior in args.answer_behaviors if str(answer_behavior).strip()}
        if args.answer_behaviors
        else None
    )
    extra_gap_after: dict[str, float] = {}
    for item in args.extra_gap_after:
        model_name, sep, gap_value = str(item).partition(":")
        if not sep:
            raise SystemExit(f"Invalid --extra-gap-after value {item!r}; expected MODEL:GAP.")
        extra_gap_after[model_name] = float(gap_value)

    input_stem = str(args.input_stem or args.output_stem)
    confidence_interval = BOOTSTRAP_CONFIDENCE_INTERVAL if str(args.ci_method) == "bootstrap" else CONFIDENCE_INTERVAL
    significance_test = BOOTSTRAP_SIGNIFICANCE_TEST if str(args.ci_method) == "bootstrap" else SIGNIFICANCE_TEST

    if bool(args.plot_from_cache):
        rows = _read_rows_from_csv(output_stem=input_stem)
        judged_rows = _read_judged_rows_from_json(output_stem=input_stem)
        input_csv_path = OUTPUT_DIR / f"{input_stem}.csv"
        print(f"loaded cached rows from {input_csv_path}", flush=True)
    else:
        cache = _load_cache()
        rows: list[DropRow] = []
        judged_rows = 0
        for model_name in target_models:
            model_rows, model_judged_rows = _build_rows_for_model(
                model_name,
                cache,
                question_groups=question_groups,
                allow_live_judge=bool(args.allow_live_judge),
                comparison_setting=str(args.comparison_setting),
                answer_behaviors=answer_behaviors,
                ci_method=str(args.ci_method),
                bootstrap_iterations=int(args.bootstrap_iterations),
                bootstrap_seed=int(args.bootstrap_seed),
                require_current_outputs=not bool(args.allow_stale_outputs),
            )
            rows.extend(model_rows)
            judged_rows += model_judged_rows
            _save_cache(cache)
            print(f"processed {model_name} judged_non_exact_rows={model_judged_rows}", flush=True)

        if not rows:
            raise SystemExit(f"No complete factual/{args.comparison_setting} Judge Match pairs found.")

    if not rows:
        raise SystemExit(f"No complete factual/{args.comparison_setting} Judge Match pairs found.")

    variant_level_correction = (
        str(args.variant_level_correction)
        if args.variant_level_correction
        else ("bonferroni" if bool(args.variant_level_bonferroni_stars) else None)
    )
    if variant_level_correction:
        _apply_variant_level_correction_stars(
            rows,
            method=variant_level_correction,
            alpha=float(args.significance_alpha),
            family_size=args.pvalue_family_size,
            max_t_permutations=int(args.max_t_permutations),
            max_t_seed=int(args.max_t_seed),
            dunnett_maxpts=int(args.dunnett_maxpts),
        )
    else:
        _apply_pvalue_correction(
            rows,
            method=str(args.pvalue_correction),
            alpha=float(args.significance_alpha),
            family_size=args.pvalue_family_size,
        )
    corrected_family_size = next((row.correction_family_size for row in rows if row.correction_family_size), None)
    csv_path = _write_csv(
        rows,
        output_stem=str(args.output_stem),
        significance_test=significance_test,
        confidence_interval=confidence_interval,
    )
    output_pvalue_correction = (
        f"variant_level_{variant_level_correction}"
        if variant_level_correction
        else str(args.pvalue_correction)
    )
    json_path = _write_json(
        rows,
        judged_rows=judged_rows,
        output_stem=str(args.output_stem),
        significance_test=significance_test,
        confidence_interval=confidence_interval,
        bootstrap_iterations=int(args.bootstrap_iterations) if str(args.ci_method) == "bootstrap" else None,
        bootstrap_seed=int(args.bootstrap_seed) if str(args.ci_method) == "bootstrap" else None,
        pvalue_correction=output_pvalue_correction,
        correction_family_size=corrected_family_size,
        significance_alpha=float(args.significance_alpha),
        max_t_permutations=(
            int(args.max_t_permutations)
            if variant_level_correction == "max_t"
            else None
        ),
        max_t_seed=(
            int(args.max_t_seed)
            if variant_level_correction in {"max_t", "dunnett"}
            else None
        ),
        dunnett_maxpts=(
            int(args.dunnett_maxpts)
            if variant_level_correction == "dunnett"
            else None
        ),
    )
    fig_path = _plot(
        rows,
        target_models=target_models,
        question_groups=question_groups,
        output_stem=str(args.output_stem),
        group_gap=float(args.group_gap),
        bar_width=float(args.bar_width),
        extra_gap_after=extra_gap_after,
        clamp_positive_axis=bool(args.clamp_positive_axis),
        use_hatches=not bool(args.no_hatches),
        legend_placement=str(args.legend_placement),
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        left_margin=args.left_margin,
        ylabel=args.ylabel,
        ylabel_fontsize=args.ylabel_fontsize,
        ylabel_y=args.ylabel_y,
        model_label_fontsize=float(args.model_label_fontsize),
        legend_fontsize=args.legend_fontsize,
        show_model_labels=not bool(args.hide_model_labels),
        show_x_tick_labels=bool(args.show_x_tick_labels),
    )
    print(csv_path)
    print(json_path)
    print(fig_path)
    for row in rows:
        print(
            f"{row.model_name} {row.question_type}: n={row.count_pairs}, factual_score={row.factual_score:.4f}, "
            f"fictional_score={row.fictional_score:.4f}, mean_diff={row.mean_difference:.4f}, "
            f"ci=[{row.ci_low:.4f}, {row.ci_high:.4f}], t={row.t_statistic:.4f}, "
            f"df={row.degrees_of_freedom}, p={row.p_value:.4g}, stars={row.stars}"
        )


if __name__ == "__main__":
    main()
