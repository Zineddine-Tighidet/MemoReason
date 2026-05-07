"""Parametric Shortcut model-evaluation package.

Keep imports lazy so consumers can reuse lightweight helpers without pulling in
the full evaluation stack and its optional dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BinaryJudgeCalibration",
    "BinaryJudgeConfusionCounts",
    "CalibratedSuccessEstimate",
    "DEFAULT_MODEL_REGISTRY",
    "EvaluatedModelSpec",
    "EvaluationReproducibilityManifest",
    "JudgeConfig",
    "calibrate_binary_judge",
    "compute_binary_judge_confusion_counts",
    "estimate_true_success_rate",
    "estimate_true_success_rate_from_labels",
    "estimate_true_success_rate_with_confidence",
    "run_parametric_shortcut_evaluation",
    "z_score_for_confidence",
]


def __getattr__(name: str) -> Any:
    if name in {"run_parametric_shortcut_evaluation"}:
        module = import_module(".parametric_shortcut_evaluation", __name__)
        return getattr(module, name)
    if name in {"EvaluationReproducibilityManifest"}:
        module = import_module(".reproducibility_manifest", __name__)
        return getattr(module, name)
    if name in {"DEFAULT_MODEL_REGISTRY", "EvaluatedModelSpec"}:
        module = import_module(".registry", __name__)
        return getattr(module, name)
    if name in {"JudgeConfig"}:
        module = import_module(".scoring", __name__)
        return getattr(module, name)
    if name in {
        "BinaryJudgeCalibration",
        "BinaryJudgeConfusionCounts",
        "CalibratedSuccessEstimate",
        "calibrate_binary_judge",
        "compute_binary_judge_confusion_counts",
        "estimate_true_success_rate",
        "estimate_true_success_rate_from_labels",
        "estimate_true_success_rate_with_confidence",
        "z_score_for_confidence",
    }:
        module = import_module(".judge_calibration", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
