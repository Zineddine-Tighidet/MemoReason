"""Template-driven export of the factual and fictional datasets."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DatasetSettingSpec",
    "ensure_dataset_artifact_directories",
    "export_dataset_documents",
    "factual_setting",
    "fictional_setting",
    "iter_template_paths",
    "parse_dataset_setting",
    "resolve_dataset_settings",
]


def __getattr__(name: str) -> Any:
    if name in {
        "DatasetSettingSpec",
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
    if name in {"export_dataset_documents"}:
        module = import_module(".export_workflow", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
