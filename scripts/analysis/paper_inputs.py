"""Adapt newly evaluated YAML files to the paper's statistical routines."""

from collections import defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path

import yaml

from memoreason.model_evaluation.benchmark_scoring_references import load_scoring_references
from memoreason.factual_to_fictional_dataset.dataset_paths import unique_question_key
from memoreason.model_evaluation.benchmark_document_loading import normalize_question_type, answer_behavior_label

ROOT = Path(__file__).resolve().parents[2]
JUDGE_MODEL = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def question_metadata():
    metadata = {}
    for path in (ROOT / "data/HUMAN_ANNOTATED_TEMPLATES").glob("*/*.yaml"):
        document = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)["document"]
        for question in document["questions"]:
            metadata[path.stem, question["question_id"]] = (
                path.parent.name,
                normalize_question_type(question["question_type"]),
                answer_behavior_label(question.get("answer_type"), question.get("is_answer_invariant")),
            )
    return metadata


def load_run(root, models, settings, *, temperature=0.0, seed=23, reasoning_effort="low"):
    """Require the complete requested benchmark grid, scored against active golds.

    No model or judge calls occur here. Raw/parsed files and other models/settings
    are ignored; duplicates, missing decisions and stale reference answers fail.
    """
    references = load_scoring_references()
    themes = {p.stem: p.parent.name for p in (ROOT / "data/HUMAN_ANNOTATED_TEMPLATES").glob("*/*.yaml")}
    expected = {key for key in references.records if key[0] in settings}
    if not expected or set(settings) - {key[0] for key in expected}:
        raise ValueError("Unknown or empty paper setting scope")
    loaded = {model: [] for model in models}
    seen = defaultdict(set)
    for path in sorted(Path(root).rglob("*_evaluated_outputs.yaml")):
        payload = yaml.load(path.read_text(), Loader=yaml.CSafeLoader)
        model, setting = payload.get("model_name"), payload.get("document_setting")
        if model not in loaded or setting not in settings:
            continue
        if payload.get("judge_model_name") != JUDGE_MODEL:
            raise ValueError(f"{path.name}: paper analysis requires EM + {JUDGE_MODEL} judge scoring")
        if payload.get("judge_config") != {
            "provider": "groq",
            "model_name": JUDGE_MODEL,
            "temperature": 0.0,
            "max_tokens": 256,
            "seed": 23,
        }:
            raise ValueError(f"{path.name}: judge configuration differs from the paper protocol")
        config = payload.get("generation_config") or {}
        expected_seed = None if model == "claude-sonnet-4-6" else seed
        if config.get("temperature") != temperature or config.get("seed") != expected_seed:
            raise ValueError(f"{path.name}: generation temperature/seed mismatch")
        if model.startswith("gpt-oss") and config.get("reasoning_effort") != reasoning_effort:
            raise ValueError(f"{path.name}: reasoning effort mismatch")
        document, variant = str(payload.get("document_id")), str(payload.get("document_variant_id"))
        theme = str(payload.get("document_theme"))
        if themes.get(document) != theme:
            raise ValueError(f"{path.name}: unknown document/theme")
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 12:
            raise ValueError(f"{path.name}: expected 12 evaluated questions")
        for result in results:
            question = str(result.get("question_id"))
            key = (setting, document, variant, question)
            if key not in expected or key in seen[model]:
                raise ValueError(f"{model}: duplicate or unexpected question {key}")
            reference = references.records[key]
            if (
                hashlib.sha256(str(result.get("question_text") or "").encode()).hexdigest()
                != reference.question_text_sha256
            ):
                raise ValueError(f"{model}/{key}: changed question text")
            if result.get("ground_truth_canonical") != reference.ground_truth_canonical or set(
                result.get("accepted_answers_canonical") or []
            ) != set(reference.accepted_answers_canonical):
                raise ValueError(f"{model}/{key}: stale reference answers; re-evaluate first")
            exact, judge, final = (result.get(field) for field in ("exact_match", "judge_match", "final_is_correct"))
            if type(exact) is not bool or type(final) is not bool or (judge is not None and type(judge) is not bool):
                raise ValueError(f"{model}/{key}: missing or non-boolean scoring decision")
            if not exact and judge is None and result.get("judge_skip_reason") != "schema_incompatible_prediction":
                raise ValueError(f"{model}/{key}: missing judge decision")
            if final != (exact or bool(judge)):
                raise ValueError(f"{model}/{key}: inconsistent final correctness")
            pair = unique_question_key(theme, document, question)
            if result.get("pair_key") != pair:
                raise ValueError(f"{model}/{key}: pairing identity mismatch")
            if (theme, result.get("question_type"), result.get("answer_behavior")) != question_metadata().get(
                (document, question)
            ):
                raise ValueError(f"{model}/{key}: question taxonomy differs from source template")
            seen[model].add(key)
            loaded[model].append(
                {
                    "model_name": model,
                    "document_theme": theme,
                    "document_id": document,
                    "document_setting": setting,
                    "document_variant_id": variant,
                    "question_id": question,
                    "pair_key": pair,
                    "question_type": result["question_type"],
                    "answer_behavior": result["answer_behavior"],
                    "new_final_is_correct": final,
                    "new_ground_truth_canonical": reference.ground_truth_canonical,
                    "new_accepted_answers_canonical": list(reference.accepted_answers_canonical),
                    "new_parsed_output_canonical": str(result.get("parsed_output_canonical") or ""),
                }
            )
    for model in models:
        if seen[model] != expected:
            raise ValueError(f"{model}: incomplete input scope ({len(seen[model])}/{len(expected)} questions)")
    return loaded


def temperature_records(runs):
    """Aggregate question decisions into the existing temperature kernel format."""
    from .temperature import CONDITIONS, MODELS

    records = []
    for condition_index, (condition, value, seed) in enumerate(CONDITIONS):
        for model_index, model in enumerate(MODELS):
            grouped = defaultdict(list)
            for row in runs[condition][model]:
                grouped[
                    (row["document_theme"], row["document_id"], row["document_setting"], row["document_variant_id"])
                ].append(row)
            for (theme, document, setting, variant), rows in sorted(grouped.items()):
                records.append(
                    {
                        "group_id": condition_index * 2 + model_index,
                        "condition": condition,
                        "temperature": value,
                        "seed": seed,
                        "model": model,
                        "document": f"{theme}::{document}",
                        "setting": setting,
                        "variant_id": variant,
                        "total": len(rows),
                        "final_correct": sum(r["new_final_is_correct"] for r in rows),
                    }
                )
    return records
