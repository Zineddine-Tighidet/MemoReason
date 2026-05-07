"""Factual dataset export built directly from reviewed templates."""

from __future__ import annotations

from pathlib import Path

from .dataset_paths import document_variant_path, resolve_template_identity
from .dataset_record_builder import _write_yaml, build_factual_dataset_record

__all__ = ["build_factual_dataset_record", "export_factual_dataset_document"]


def export_factual_dataset_document(template_path: Path, *, seed: int, overwrite: bool = False) -> Path:
    """Write one factual dataset document for one template."""
    theme, document_id = resolve_template_identity(template_path)
    output_path = document_variant_path(theme, document_id, "factual", variant_index=1, variant_count=1)
    if output_path.exists() and not overwrite:
        return output_path
    return _write_yaml(build_factual_dataset_record(template_path, seed=seed), output_path)
