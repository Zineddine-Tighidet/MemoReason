"""Frozen scientific routines used by the submitted result release."""
from __future__ import annotations
from collections import defaultdict
from typing import Any

MODEL_ORDER = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "llama-3.1-8b-instruct",
    "claude-sonnet-4-6",
)

DISPLAY = {
    "olmo-3-7b-think": "OLMO-3 7B-THINK",
    "olmo-3-7b-instruct": "OLMO-3 7B-INSTRUCT",
    "gpt-oss-20b-groq": "GPT-OSS 20B",
    "gemma-4-26b-a4b-it": "GEMMA-4 26B-A4B-IT",
    "gpt-oss-120b-groq": "GPT-OSS 120B",
    "qwen3.5-27b": "QWEN3.5 27B",
    "qwen3.5-35b-a3b": "QWEN3.5 35B-A3B",
    "llama-3.1-8b-instruct": "LLAMA-3.1 8B-INSTRUCT",
    "claude-sonnet-4-6": "CLAUDE SONNET 4.6",
}

ATOMIC_TYPES = ("arithmetic", "temporal", "inference", "extractive")

REASONING_TYPES = ("arithmetic", "temporal", "inference")

def reference_answers_from_reconciled_rows(
    loaded: dict[str, list[dict[str, Any]]]
) -> dict[str, set[str]]:
    """Recover model-independent canonical answers from reconciled rows.

    The scoring exports repeat the same dataset reference fields for every
    model. A small number of legacy exports omit the canonical-answer list for
    one model while another model's row for the same Dataset100 question still
    carries it. Resolve that export omission without introducing any literal
    answer: require one and only one non-empty canonical set across all models.
    """

    observed: dict[str, set[frozenset[str]]] = defaultdict(set)
    for rows in loaded.values():
        for row in rows:
            if row["document_setting"] != "factual" or row.get("answer_behavior") != "variant":
                continue
            accepted = frozenset(
                str(value).strip()
                for value in row.get("new_accepted_answers_canonical") or []
                if str(value).strip()
            )
            fallback = str(row.get("new_ground_truth_canonical") or "").strip()
            if not accepted and fallback:
                accepted = frozenset({fallback})
            if accepted:
                observed[str(row["pair_key"])].add(accepted)
    if len(observed) != 400:
        raise ValueError(f"Expected canonical references for 400 variant questions, found {len(observed)}")
    inconsistent = {key: values for key, values in observed.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Inconsistent canonical reference sets: {sorted(inconsistent)[:5]}")
    return {key: set(next(iter(values))) for key, values in observed.items()}

def endpoint_and_shortcut_rows(
    loaded: dict[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    endpoint_rows: list[dict[str, Any]] = []
    shortcut_rows: list[dict[str, Any]] = []
    factual_answers = reference_answers_from_reconciled_rows(loaded)
    for model in MODEL_ORDER:
        rows = loaded[model]
        factual_rows = [row for row in rows if row["document_setting"] == "factual"]
        fictional_rows = [row for row in rows if row["document_setting"] == "fictional"]
        factual_correct = sum(bool(row["new_final_is_correct"]) for row in factual_rows)
        fictional_correct = sum(bool(row["new_final_is_correct"]) for row in fictional_rows)
        factual_accuracy = 100.0 * factual_correct / len(factual_rows)
        fictional_accuracy = 100.0 * fictional_correct / len(fictional_rows)
        endpoint_rows.append(
            {
                "model_name": model,
                "factual_correct": factual_correct,
                "factual_total": len(factual_rows),
                "factual_accuracy_percent": factual_accuracy,
                "fictional_correct": fictional_correct,
                "fictional_total": len(fictional_rows),
                "fictional_accuracy_percent": fictional_accuracy,
                "gap_pp": factual_accuracy - fictional_accuracy,
                "error_ratio": (100.0 - fictional_accuracy) / (100.0 - factual_accuracy),
            }
        )

        by_question: dict[str, dict[str, Any]] = {}
        variants: dict[str, set[str]] = defaultdict(set)
        for row in fictional_rows:
            if row.get("answer_behavior") != "variant":
                continue
            pair_key = str(row["pair_key"])
            question_type = str(row["question_type"])
            if question_type not in ATOMIC_TYPES:
                raise ValueError(f"{model}: unexpected question type {question_type}")
            cell = by_question.setdefault(
                pair_key,
                {"question_type": question_type, "failures": 0, "matches": 0},
            )
            if cell["question_type"] != question_type:
                raise ValueError(f"{model}: question-type drift for {pair_key}")
            variants[pair_key].add(str(row["document_variant_id"]))
            if row["new_final_is_correct"]:
                continue
            cell["failures"] += 1
            prediction = str(row.get("new_parsed_output_canonical") or "").strip()
            if prediction and prediction in factual_answers[pair_key]:
                cell["matches"] += 1
        if len(by_question) != 400 or any(len(values) != 10 for values in variants.values()):
            raise ValueError(f"{model}: invalid variant-question inventory")
        for question_type, members in [
            *((name, {name}) for name in ATOMIC_TYPES),
            ("reasoning", set(REASONING_TYPES)),
        ]:
            selected = [cell for cell in by_question.values() if cell["question_type"] in members]
            expected = 100 * len(members)
            if len(selected) != expected:
                raise ValueError(f"{model}/{question_type}: expected {expected} base questions")
            failures = sum(int(cell["failures"]) for cell in selected)
            matches = sum(int(cell["matches"]) for cell in selected)
            macro = 100.0 * sum(
                cell["matches"] / cell["failures"] if cell["failures"] else 0.0
                for cell in selected
            ) / len(selected)
            shortcut_rows.append(
                {
                    "model_name": model,
                    "question_type": question_type,
                    "base_question_count": len(selected),
                    "questions_with_failures": sum(cell["failures"] > 0 for cell in selected),
                    "failed_variant_count": failures,
                    "factual_answer_match_count": matches,
                    "shortcut_rate_pp": macro,
                    "pooled_match_rate_pp_diagnostic": 100.0 * matches / failures if failures else 0.0,
                }
            )
    return endpoint_rows, shortcut_rows
