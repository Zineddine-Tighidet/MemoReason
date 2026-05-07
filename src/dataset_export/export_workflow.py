"""Orchestrate factual and fictional dataset export for a set of templates."""

from __future__ import annotations

import logging
from pathlib import Path

from .dataset_settings import order_dataset_settings
from .dataset_record_builder import remove_stale_fictional_dataset_exports
from .factual_dataset import export_factual_dataset_document
from .fictional_dataset import export_fictional_dataset_documents_batch

logger = logging.getLogger(__name__)

__all__ = ["export_dataset_documents"]


def export_dataset_documents(
    template_paths: list[Path],
    *,
    seed: int,
    settings: list[str],
    fictional_version_count: int = 1,
    overwrite: bool = False,
    skip_missing_pools: bool = False,
) -> list[Path]:
    """Export dataset-ready factual and fictional documents for the requested settings."""
    if int(fictional_version_count) < 1:
        raise ValueError(f"fictional_version_count must be >= 1, got {fictional_version_count!r}.")

    written_paths: list[Path] = []
    setting_specs = order_dataset_settings(settings)
    for template_path in template_paths:
        print(f"[export] template={template_path}", flush=True)
        for setting_spec in setting_specs:
            if setting_spec.is_factual:
                print(
                    f"[export] factual template={template_path.name} setting={setting_spec.setting_id}",
                    flush=True,
                )
                written_paths.append(export_factual_dataset_document(template_path, seed=seed, overwrite=overwrite))
                continue

            if overwrite:
                remove_stale_fictional_dataset_exports(
                    template_path=template_path,
                    setting_spec=setting_spec,
                )

            print(
                f"[export] fictional template={template_path.name} setting={setting_spec.setting_id} variants=1..{fictional_version_count}",
                flush=True,
            )
            try:
                written_paths.extend(
                    export_fictional_dataset_documents_batch(
                        template_path,
                        setting_spec=setting_spec,
                        seed=seed,
                        variant_count=fictional_version_count,
                        overwrite=overwrite,
                    )
                )
            except FileNotFoundError:
                if not skip_missing_pools:
                    raise
                logger.warning(
                    "Skipping %s export because the fictional entity pool is missing: %s",
                    setting_spec.setting_id,
                    template_path,
                )
    return written_paths
