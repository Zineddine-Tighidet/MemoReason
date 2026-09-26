"""Persistence helpers for validated fictional entity pools."""

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import yaml

from memoreason.benchmark_definition.annotation_runtime import load_entity_pool

from ..dataset_paths import existing_entity_pool_path
from .pool_normalization import POOL_BUCKETS


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable_pool_payload(pool: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(pool.get("_metadata", {})) if isinstance(pool.get("_metadata"), dict) else {}
    coverage = dict(pool.get("_coverage", {})) if isinstance(pool.get("_coverage"), dict) else {}
    reference_pools = dict(pool.get("_reference_pools", {})) if isinstance(pool.get("_reference_pools"), dict) else {}
    missing_references = {
        bucket: {
            entity_id: max(0, int(target) - int(actual))
            for entity_id, actual in bucket_counts.items()
            for target in [metadata.get("candidates_per_reference", 15)]
            if int(actual) < int(target)
        }
        for bucket, bucket_counts in coverage.items()
        if isinstance(bucket_counts, dict)
    }
    missing_references = {bucket: refs for bucket, refs in missing_references.items() if refs}
    payload: dict[str, Any] = {
        "metadata": {
            **metadata,
            "complete": bool(metadata.get("complete", True)) and not bool(missing_references),
        }
    }
    if coverage:
        payload["coverage"] = coverage
    if missing_references:
        payload["missing_references"] = missing_references
    for bucket in POOL_BUCKETS:
        bucket_refs = reference_pools.get(bucket, {})
        if not isinstance(bucket_refs, dict) or not bucket_refs:
            continue
        payload[bucket] = {}
        for entity_id, ref_payload in bucket_refs.items():
            if not isinstance(ref_payload, dict):
                continue
            payload[bucket][entity_id] = {
                "required_attributes": list(ref_payload.get("required_attributes", [])),
                "count": int(ref_payload.get("count", 0)),
                "variants": list(ref_payload.get("variants", [])),
            }
    return payload


def save_entity_pool(
    pool: dict[str, Any],
    output_path: Path,
    *,
    overwrite: bool = False,
    expected_existing_sha256: str | None = None,
) -> Path:
    """Atomically save a pool, refusing accidental or stale overwrites."""
    if expected_existing_sha256 is not None and not overwrite:
        raise ValueError("expected_existing_sha256 requires overwrite=True")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = yaml.safe_dump(
        _serializable_pool_payload(pool),
        sort_keys=False,
        allow_unicode=True,
        width=10000,
    ).encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())

        if overwrite:
            if not output_path.is_file():
                raise FileNotFoundError(f"Entity pool selected for overwrite no longer exists: {output_path}")
            if expected_existing_sha256 is not None:
                actual_sha256 = _file_sha256(output_path)
                if actual_sha256 != expected_existing_sha256:
                    raise RuntimeError(
                        f"Entity pool changed before atomic overwrite: {output_path}; "
                        f"expected {expected_existing_sha256}, found {actual_sha256}"
                    )
            os.chmod(temporary_path, output_path.stat().st_mode & 0o777)
            os.replace(temporary_path, output_path)
        else:
            os.chmod(temporary_path, 0o644)
            try:
                os.link(temporary_path, output_path)
            except FileExistsError as exc:
                raise FileExistsError(f"Refusing to overwrite existing entity pool: {output_path}") from exc
            temporary_path.unlink()
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def load_saved_entity_pool(theme: str, document_id: str) -> tuple[dict[str, Any] | None, Path | None]:
    """Load an existing pool from the dataset data layout."""
    pool_path = existing_entity_pool_path(theme, document_id)
    if pool_path is None:
        return None, None
    return load_entity_pool(str(pool_path)), pool_path
