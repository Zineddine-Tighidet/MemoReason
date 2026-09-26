"""Lazy public API for MemoReason model evaluation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "PAPER_MODEL_CONFIGURATIONS",
    "PaperModelConfiguration",
    "ModelEvaluationRunManifest",
    "JudgeMatchConfiguration",
    "generate_parse_and_score_model_answers",
]


def __getattr__(name: str) -> Any:
    if name in {"generate_parse_and_score_model_answers"}:
        module = import_module(".model_answer_evaluation_pipeline", __name__)
        return getattr(module, name)
    if name in {"ModelEvaluationRunManifest"}:
        module = import_module(".model_evaluation_run_manifest", __name__)
        return getattr(module, name)
    if name in {"PAPER_MODEL_CONFIGURATIONS", "PaperModelConfiguration"}:
        module = import_module(".paper_model_registry", __name__)
        return getattr(module, name)
    if name in {"JudgeMatchConfiguration"}:
        module = import_module(".exact_and_judge_match_scoring", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
