"""Entity pool loading and generation service for web previews."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memoreason.factual_to_fictional_dataset.fictional_entity_pool_generation.fictional_entity_replacement_pool_generation import (
    FictionalEntityReplacementPoolGenerationConfiguration,
    generate_fictional_entity_replacement_pool,
)
from memoreason.factual_to_fictional_dataset.dataset_paths import (
    GENERATED_FICTIONAL_ENTITIES_DIR,
    generated_entity_pool_path,
)
from memoreason.benchmark_definition.document_schema import AnnotatedDocument
from memoreason.benchmark_definition.annotation_runtime import load_entity_pool
from web.services import yaml_service
from web.services.persistence import (
    delete_work_prefix_from_gcs,
    restore_work_file_from_gcs,
    restore_worktree_from_gcs,
    sync_work_file_to_gcs,
)

logger = logging.getLogger(__name__)
RUNTIME_POOL_ROOT = yaml_service.WORK_DIR / "_generated_entity_pools"


def _env_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes"}


def _pool_generation_config(seed: int) -> FictionalEntityReplacementPoolGenerationConfiguration:
    return FictionalEntityReplacementPoolGenerationConfiguration(
        provider=os.environ.get("POOL_LLM_PROVIDER", "anthropic").strip().lower(),
        model=os.environ.get("POOL_LLM_MODEL", "claude-opus-4-6").strip(),
        temperature=float(os.environ.get("POOL_LLM_TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("POOL_LLM_MAX_TOKENS", "32768")),
        seed=seed,
        max_attempts=int(os.environ.get("POOL_LLM_MAX_ATTEMPTS", "12")),
        validate_against_wikipedia=_env_flag("POOL_WIKI_VALIDATE", True),
    )


def runtime_entity_pool_path(theme: str, document_id: str) -> Path:
    """Return the synced workspace pool path for one ``theme/doc`` pair."""
    normalized_theme = yaml_service.canonical_theme_id(theme)
    return RUNTIME_POOL_ROOT / normalized_theme / f"{document_id}_entity_pool.yaml"


def _theme_candidates(theme_id: Optional[str], doc: AnnotatedDocument) -> list[str]:
    candidates: list[str] = []
    for raw_theme in (theme_id, doc.document_theme):
        if not raw_theme:
            continue
        stripped = str(raw_theme).strip()
        if not stripped:
            continue
        normalized = yaml_service.canonical_theme_id(stripped)
        for value in (stripped, normalized):
            if value and value not in candidates:
                candidates.append(value)
    return candidates


def _pool_generation_theme(theme_id: Optional[str], doc: AnnotatedDocument) -> str:
    for candidate in _theme_candidates(theme_id, doc):
        if candidate:
            return candidate
    return str(doc.document_theme or "").strip()


def _single_pool_match_by_doc_id(root: Path, document_id: str) -> Optional[Path]:
    if not root.exists():
        return None
    matches = sorted(root.rglob(f"{document_id}_entity_pool.yaml"))
    if len(matches) == 1:
        return matches[0]
    return None


def _try_load_pool(path: Path, *, refresh_from_gcs: bool = False) -> Optional[Dict[str, Any]]:
    if refresh_from_gcs:
        restore_work_file_from_gcs(path)
    if not path.exists():
        return None
    try:
        return load_entity_pool(str(path))
    except Exception as exc:
        logger.warning("Failed to load pool %s: %s", path, exc)
        return None


def _load_pool_for_document(
    doc: AnnotatedDocument,
    *,
    theme_id: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Load a document-specific pool from synced workspace or benchmark data."""
    doc_id = (doc.document_id or "").strip()
    if not doc_id:
        return None, None

    for candidate_theme in _theme_candidates(theme_id, doc):
        runtime_path = runtime_entity_pool_path(candidate_theme, doc_id)
        pool = _try_load_pool(runtime_path, refresh_from_gcs=True)
        if pool is not None:
            return pool, runtime_path

    unique_runtime = _single_pool_match_by_doc_id(RUNTIME_POOL_ROOT, doc_id)
    if unique_runtime is not None:
        pool = _try_load_pool(unique_runtime, refresh_from_gcs=True)
        if pool is not None:
            return pool, unique_runtime

    for candidate_theme in _theme_candidates(theme_id, doc):
        benchmark_path = generated_entity_pool_path(candidate_theme, doc_id)
        pool = _try_load_pool(benchmark_path, refresh_from_gcs=False)
        if pool is not None:
            return pool, benchmark_path

    unique_benchmark = _single_pool_match_by_doc_id(GENERATED_FICTIONAL_ENTITIES_DIR, doc_id)
    if unique_benchmark is not None:
        pool = _try_load_pool(unique_benchmark, refresh_from_gcs=False)
        if pool is not None:
            return pool, unique_benchmark

    return None, None


def publish_runtime_entity_pool(
    *,
    source_pool_path: Path,
    theme: str,
    document_id: str,
    overwrite: bool = False,
) -> tuple[Path, bool]:
    """Copy one generated pool into the synced runtime workspace."""
    destination_path = runtime_entity_pool_path(theme, document_id)
    if not destination_path.exists():
        restore_work_file_from_gcs(destination_path)
    if destination_path.exists() and not overwrite:
        return destination_path, False
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_pool_path, destination_path)
    sync_work_file_to_gcs(destination_path)
    return destination_path, True


def clear_runtime_entity_pools() -> None:
    """Delete all synced runtime pools from local workspace and GCS."""
    restore_worktree_from_gcs()
    if RUNTIME_POOL_ROOT.exists():
        shutil.rmtree(RUNTIME_POOL_ROOT)
    delete_work_prefix_from_gcs(Path("_generated_entity_pools"))


def get_or_generate_pool(
    doc: AnnotatedDocument,
    required_entities: Dict[str, List[Tuple[str, List[str]]]],
    seed: int,
    *,
    theme_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, Optional[Path]]:
    """Return a loaded or generated pool for the web preview path."""
    pool, pool_path = _load_pool_for_document(doc, theme_id=theme_id)
    if pool is not None:
        return pool, "document_pool", pool_path

    if not _env_flag("ANNOTATION_ALLOW_POOL_GENERATION", False):
        raise ValueError(
            "No local entity pool is available for this document. Restore the release "
            "entity pools, or explicitly enable ANNOTATION_ALLOW_POOL_GENERATION=true "
            "and configure your own provider credentials."
        )
    config = _pool_generation_config(seed)
    generated_pool = generate_fictional_entity_replacement_pool(
        doc,
        required_entities,
        theme=_pool_generation_theme(theme_id, doc),
        config=config,
    )
    return generated_pool, "auto_generated", None
