"""Load evaluated model answers into the paper's normalized score evaluated_answers."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from .frozen_paper_results_input_loading import _normalized_text, _read_yaml
from .paper_results_data_model import PaperResultsManifest, EvaluatedAnswer


def _strict_bool(value: Any, *, field: str, path: Path) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{path}: {field} must be boolean, got {value!r}.")


def _optional_bool(value: Any, *, field: str, path: Path) -> bool | None:
    if value is None:
        return None
    return _strict_bool(value, field=field, path=path)


def replacement_proportion(setting: str, payload_value: Any = None) -> float:
    inferred: float | None = None
    if setting == "factual":
        inferred = 0.0
    elif setting == "fictional":
        inferred = 1.0
    else:
        prefix = "fictional_"
        suffix = "pct"
        if setting.startswith(prefix) and setting.endswith(suffix):
            try:
                inferred = float(setting[len(prefix) : -len(suffix)]) / 100.0
            except ValueError as exc:
                raise ValueError(f"Cannot infer replacement proportion from setting {setting!r}.") from exc

    if payload_value is None:
        if inferred is None:
            raise ValueError(f"Cannot infer replacement proportion from setting {setting!r}.")
        return inferred
    try:
        value = float(payload_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid replacement_proportion {payload_value!r} for setting {setting!r}.") from exc
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"replacement_proportion must be finite and between 0 and 1, got {payload_value!r}.")
    if inferred is not None and not np.isclose(value, inferred, rtol=0.0, atol=1e-12):
        raise ValueError(f"replacement_proportion mismatch for setting {setting!r}: expected {inferred}, got {value}.")
    return value


def load_evaluated_answers(paper_results_manifest: PaperResultsManifest) -> list[EvaluatedAnswer]:
    """Flatten the manifest-selected evaluated YAML files with strict checks."""
    evaluated_answers: list[EvaluatedAnswer] = []
    unique_keys: set[tuple[str, str, str, str]] = set()
    allowed_settings = set(paper_results_manifest.settings)

    for path in paper_results_manifest.evaluated_output_paths:
        payload = _read_yaml(path)
        model_name = str(payload.get("model_name") or "").strip()
        document_theme = str(payload.get("document_theme") or "").strip()
        document_id = str(payload.get("document_id") or "").strip()
        setting = str(payload.get("document_setting") or "").strip().lower()
        if not model_name or not document_theme or not document_id or not setting:
            raise ValueError(f"{path}: model_name, document_theme, document_id and document_setting are required.")
        if setting not in allowed_settings:
            continue
        variant_id = str(payload.get("document_variant_id") or ("factual" if setting == "factual" else "")).strip()
        if setting != "factual" and variant_id not in set(paper_results_manifest.variant_ids):
            raise ValueError(f"{path}: unexpected variant id {variant_id!r}.")
        proportion = replacement_proportion(setting, payload.get("replacement_proportion"))

        raw_results = payload.get("results") or []
        if not isinstance(raw_results, list) or not raw_results:
            raise ValueError(f"{path}: results must be a non-empty list.")
        for index, result in enumerate(raw_results):
            if not isinstance(result, Mapping):
                raise ValueError(f"{path}: result {index} is not a mapping.")
            pair_key = str(result.get("pair_key") or "").strip()
            if not pair_key:
                raise ValueError(f"{path}: result {index} has no pair_key.")
            # Factual files historically carry ``document_variant_id: v01``.
            # Normalize their identity so selecting two factual runs cannot
            # evade duplicate detection through different payload variant ids.
            identity_variant_id = "factual" if setting == "factual" else variant_id
            unique_key = (model_name, setting, identity_variant_id, pair_key)
            if unique_key in unique_keys:
                raise ValueError(f"Duplicate evaluated row selected by manifest: {unique_key}")
            unique_keys.add(unique_key)

            exact = _strict_bool(result.get("exact_match"), field="exact_match", path=path)
            judge = _optional_bool(result.get("judge_match"), field="judge_match", path=path)
            final = _strict_bool(result.get("final_is_correct"), field="final_is_correct", path=path)
            if final != (exact or judge is True):
                raise ValueError(
                    f"{path}: inconsistent final score for {pair_key}; expected exact_match OR judge_match."
                )

            accepted = tuple(
                str(value).strip() for value in (result.get("accepted_answers_canonical") or []) if str(value).strip()
            )
            question_text = str(result.get("question_text") or "").strip()
            stored_question_type = str(result.get("question_type") or "").strip().lower()
            stored_answer_behavior = (
                str(result.get("answer_behavior") or result.get("answer_type") or "").strip().lower()
            )
            question_id = str(result.get("question_id") or pair_key.rsplit("::", 1)[-1]).strip()
            reviewed_question_key = f"{document_theme}::{document_id}::{question_id}"
            reviewed_question = paper_results_manifest.reviewed_questions_by_pair_key.get(reviewed_question_key)
            if reviewed_question is None:
                raise ValueError(
                    f"{path}: no human-reviewed metadata for {reviewed_question_key}; "
                    "generated outputs cannot supply answer behavior."
                )
            if pair_key != reviewed_question_key:
                raise ValueError(
                    f"{path}: pair_key {pair_key!r} does not match reviewed question {reviewed_question_key!r}."
                )
            if stored_question_type != reviewed_question.question_type:
                raise ValueError(
                    f"{path}: generated question_type {stored_question_type!r} contradicts human "
                    f"annotation {reviewed_question.question_type!r} for {pair_key}."
                )
            if stored_answer_behavior != reviewed_question.answer_behavior:
                raise ValueError(
                    f"{path}: generated answer_behavior {stored_answer_behavior!r} contradicts human "
                    f"annotation {reviewed_question.answer_behavior!r} for {pair_key}."
                )
            if setting == "factual" and _normalized_text(question_text) != reviewed_question.question_text_factual:
                raise ValueError(
                    f"{path}: factual question text drift for {pair_key}; the evaluated question "
                    "is not the human-reviewed template version."
                )
            question_type = reviewed_question.question_type
            answer_behavior = reviewed_question.answer_behavior
            ground_truth = str(result.get("ground_truth") or "").strip()
            answer_schema = str(result.get("answer_schema") or "").strip()
            if not question_text or not ground_truth or not answer_schema or not accepted:
                raise ValueError(
                    f"{path}: result {index} must contain question_text, ground_truth, "
                    "answer_schema and accepted_answers_canonical."
                )
            evaluated_answers.append(
                EvaluatedAnswer(
                    source_path=path,
                    model_name=model_name,
                    document_theme=document_theme,
                    document_id=document_id,
                    setting=setting,
                    replacement_proportion=proportion,
                    variant_id=variant_id,
                    pair_key=pair_key,
                    question_text=question_text,
                    question_type=question_type,
                    answer_behavior=answer_behavior,
                    ground_truth=ground_truth,
                    answer_schema=answer_schema,
                    accepted_answers_canonical=accepted,
                    parsed_output=str(result.get("parsed_output") or "").strip(),
                    parsed_output_canonical=str(result.get("parsed_output_canonical") or "").strip(),
                    raw_output=str(result.get("raw_output") or "").strip(),
                    exact_match=exact,
                    judge_match=judge,
                    final_is_correct=final,
                    factual_answer_match=_optional_bool(
                        result.get("factual_answer_match"), field="factual_answer_match", path=path
                    ),
                )
            )
    if not evaluated_answers:
        raise ValueError("The reporting manifest selected no evaluated score evaluated_answers.")
    return evaluated_answers
