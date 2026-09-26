"""Template-driven export of the factual and fictional datasets."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FactualToFictionalDatasetSetting",
    "FictionalEntityReplacementPoolGenerationConfiguration",
    "ensure_dataset_artifact_directories",
    "export_paired_factual_and_fictional_documents",
    "factual_setting",
    "fictional_setting",
    "generate_fictional_entity_replacement_pool_for_template",
    "iter_template_paths",
    "parse_dataset_setting",
    "resolve_dataset_settings",
]


def __getattr__(name: str) -> Any:
    if name in {
        "FactualToFictionalDatasetSetting",
        "factual_setting",
        "fictional_setting",
        "parse_dataset_setting",
        "resolve_dataset_settings",
    }:
        module = import_module(".dataset_settings", __name__)
        return getattr(module, name)
    if name in {"ensure_dataset_artifact_directories", "iter_template_paths"}:
        module = import_module(".dataset_paths", __name__)
        return getattr(module, name)
    if name in {"export_paired_factual_and_fictional_documents"}:
        module = import_module(".paired_factual_and_fictional_dataset_export", __name__)
        return getattr(module, name)
    if name in {
        "FictionalEntityReplacementPoolGenerationConfiguration",
        "generate_fictional_entity_replacement_pool_for_template",
    }:
        module = import_module(
            ".fictional_entity_pool_generation.fictional_entity_replacement_pool_generation",
            __name__,
        )
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
