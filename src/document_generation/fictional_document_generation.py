"""Batch helpers for generating fictional document variants from one template."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from src.core.annotation_runtime import REPLACE_MODE_ALL, load_annotated_document, load_entity_pool, replace_mode_label

from .fictional_generation_algorithm import (
    FictionalGenerationTemplateInput,
    FictionalVariantRequest,
    MAX_VARIANT_GENERATION_RETRIES,
    fictional_generation,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_VARIANT_GENERATION_RETRIES",
    "create_entity_pool_template",
    "generate_fictional_documents",
    "run_fictional_document_generation",
]


def _variant_output_path(
    *,
    output_dir: Path,
    document_id: str,
    replacement_proportion: float,
    version_index: int,
) -> Path:
    return output_dir / f"{document_id}_partial_p{replacement_proportion:.1f}_v{version_index + 1}.yaml"


def _prepare_output_dir(
    *,
    output_base_dir: Path,
    document_id: str,
    replacement_proportion: float,
) -> Path:
    output_dir = output_base_dir / f"p{replacement_proportion:.1f}" / document_id
    if output_dir.exists():
        existing_version_files = list(output_dir.glob(f"{document_id}_partial_p{replacement_proportion:.1f}_v*.yaml"))
        if existing_version_files:
            logger.info("Overwriting %d existing file(s) in %s", len(existing_version_files), output_dir)
            for existing_file in existing_version_files:
                existing_file.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _generate_fictional_variant_task(task: dict[str, Any]) -> str:
    document = load_annotated_document(task["doc_path"])
    document_id = task["document_id"] or document.document_id
    raw_entity_pool: dict[str, Any] = (
        load_entity_pool(task["pool_path"])
        if task["replacement_proportion"] > 0.0 and task["pool_path"]
        else {}
    )
    output_dir = Path(task["output_base_dir"]) / f"p{task['replacement_proportion']:.1f}" / document_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _variant_output_path(
        output_dir=output_dir,
        document_id=document_id,
        replacement_proportion=task["replacement_proportion"],
        version_index=task["version_index"],
    )
    algorithm_result = fictional_generation(
        [
            FictionalGenerationTemplateInput(
                template_document=document,
                named_entity_pool=raw_entity_pool,
                variant_requests=(
                    FictionalVariantRequest(
                        base_seed=task["seed"] + task["version_index"],
                        output_path=output_path,
                        reference_variant_index=task["version_index"],
                        reference_variant_count=task["num_versions"],
                    ),
                ),
                replacement_proportion=task["replacement_proportion"],
                document_id=document_id,
                replace_mode=task["replace_mode"],
                named_entities_seed=task["seed"] + task["version_index"],
            )
        ]
    )[0]
    result_path, _successful_seed = algorithm_result.generated_variants[0]
    return str(result_path)


def generate_fictional_documents(
    doc_path: str,
    pool_path: str | None,
    replacement_proportion: float,
    num_versions: int,
    seed: int,
    output_base_dir: Path,
    document_id: str | None = None,
    replace_mode: str = REPLACE_MODE_ALL,
    num_workers: int = 1,
) -> Path:
    """Generate fictional variants for one template under one replacement setting."""
    if not 0.0 <= replacement_proportion <= 1.0:
        raise ValueError("replacement_proportion must be in [0.0, 1.0]")
    if replacement_proportion > 0.0 and (not pool_path or not pool_path.strip()):
        raise ValueError("pool_path is required when replacement_proportion > 0")
    if replacement_proportion == 0.0:
        num_versions = 1
    if num_workers < 1:
        num_workers = 1

    document = load_annotated_document(doc_path)
    document_id = document_id or document.document_id
    output_dir = _prepare_output_dir(
        output_base_dir=output_base_dir,
        document_id=document_id,
        replacement_proportion=replacement_proportion,
    )

    if num_workers <= 1:
        raw_entity_pool: dict[str, Any] = load_entity_pool(pool_path) if replacement_proportion > 0.0 and pool_path else {}
        eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] = {}
        variant_requests = [
            FictionalVariantRequest(
                base_seed=seed + version_index,
                output_path=_variant_output_path(
                    output_dir=output_dir,
                    document_id=document_id,
                    replacement_proportion=replacement_proportion,
                    version_index=version_index,
                ),
                reference_variant_index=version_index,
                reference_variant_count=num_versions,
            )
            for version_index in range(num_versions)
        ]
        fictional_generation(
            [
                FictionalGenerationTemplateInput(
                    template_document=document,
                    named_entity_pool=raw_entity_pool,
                    variant_requests=tuple(variant_requests),
                    replacement_proportion=replacement_proportion,
                    document_id=document_id,
                    replace_mode=replace_mode,
                    named_entities_seed=seed,
                    eligible_cache=eligible_cache,
                    max_generation_retries=MAX_VARIANT_GENERATION_RETRIES,
                )
            ]
        )
    else:
        tasks = [
            {
                "doc_path": doc_path,
                "pool_path": pool_path,
                "replacement_proportion": replacement_proportion,
                "version_index": version_index,
                "seed": seed,
                "num_versions": num_versions,
                "output_base_dir": str(output_base_dir),
                "document_id": document_id,
                "replace_mode": replace_mode,
            }
            for version_index in range(num_versions)
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_generate_fictional_variant_task, task): task["version_index"] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"p={replacement_proportion:.0%}"):
                exception = future.exception()
                if exception is not None:
                    raise exception

    return output_dir


def create_entity_pool_template(doc_path: str, output_path: str) -> None:
    """Create an empty entity-pool template for one annotated document."""
    load_annotated_document(doc_path)
    empty_pool_template = {
        "persons": [],
        "places": [],
        "events": [],
        "organizations": [],
        "awards": [],
        "legals": [],
        "products": [],
        "numbers": [],
        "temporals": [],
    }
    Path(output_path).write_text(
        yaml.safe_dump(empty_pool_template, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def run_fictional_document_generation(
    doc_path: str,
    pool_path: str | None = None,
    replacement_proportion: float = 1.0,
    num_versions: int = 1,
    seed: int = 42,
    subfolder: str | None = None,
    output_base_dir: Path | None = None,
    document_id: str | None = None,
    replace_mode: str = REPLACE_MODE_ALL,
    num_workers: int = 1,
) -> Path:
    """User-facing convenience wrapper around :func:`generate_fictional_documents`."""
    if not 0.0 <= replacement_proportion <= 1.0:
        raise ValueError("replacement_proportion must be in [0.0, 1.0]")
    if replacement_proportion > 0.0 and (not pool_path or not pool_path.strip()):
        raise ValueError("pool_path is required when replacement_proportion > 0")

    base_output_dir = (
        Path(output_base_dir)
        if output_base_dir
        else (Path(f"output/partial_fictional/{subfolder}") if subfolder else Path("output/partial_fictional"))
    )
    mode_label = replace_mode_label(replace_mode)
    if mode_label:
        base_output_dir = base_output_dir / mode_label
    base_output_dir.mkdir(parents=True, exist_ok=True)

    return generate_fictional_documents(
        doc_path=doc_path,
        pool_path=pool_path,
        replacement_proportion=replacement_proportion,
        num_versions=num_versions,
        seed=seed,
        output_base_dir=base_output_dir,
        document_id=document_id,
        replace_mode=replace_mode,
        num_workers=num_workers,
    )
