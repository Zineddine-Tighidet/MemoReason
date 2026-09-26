"""Validation helpers for finalized fictional entity pools."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

from memoreason.benchmark_definition.organization_types import (
    ORGANIZATION_POOL_BUCKETS,
    normalize_organization_pool_entry,
)
from memoreason.factual_to_fictional_dataset.fictional_entity_pool_generation.wikipedia_validation import (
    check_entities,
    load_cache,
    normalize_query,
)
from .pool_normalization import _normalize_pool_dict
from .fictional_entity_replacement_pool_prompt_planning import (
    _POOL_WIKI_CACHE_PATH,
    _POOL_WIKI_MAX_RETRIES,
    _POOL_WIKI_THROTTLE,
    _POOL_WIKI_TIMEOUT,
)

_WIKIPEDIA_CACHE_PATH_ENV = "MEMOREASON_ENTITY_POOL_WIKIPEDIA_CACHE_PATH"
_WIKIPEDIA_CACHE_MODE_ENV = "MEMOREASON_ENTITY_POOL_WIKIPEDIA_CACHE_MODE"
_WIKIPEDIA_CACHE_MODES = frozenset({"read-write", "replay-only"})


def _wikipedia_cache_configuration() -> tuple[Path, str]:
    """Resolve the pool Wikipedia cache at call time for isolated/replay runs."""
    configured_path = str(os.environ.get(_WIKIPEDIA_CACHE_PATH_ENV, "")).strip()
    cache_path = Path(configured_path).expanduser().resolve() if configured_path else _POOL_WIKI_CACHE_PATH
    mode = str(os.environ.get(_WIKIPEDIA_CACHE_MODE_ENV, "read-write")).strip().lower()
    if mode not in _WIKIPEDIA_CACHE_MODES:
        allowed = ", ".join(sorted(_WIKIPEDIA_CACHE_MODES))
        raise ValueError(f"{_WIKIPEDIA_CACHE_MODE_ENV} must be one of: {allowed}")
    return cache_path, mode


def _iter_pool_strings(pool: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for person in pool.get("persons", []) or []:
        if not isinstance(person, dict):
            continue
        for key in ("first_name", "last_name", "full_name", "nationality", "ethnicity"):
            raw_value = person.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                values.append(raw_value.strip())

    for place in pool.get("places", []) or []:
        if not isinstance(place, dict):
            continue
        for raw_value in place.values():
            if isinstance(raw_value, str) and raw_value.strip():
                values.append(raw_value.strip())

    for event in pool.get("events", []) or []:
        if not isinstance(event, dict):
            continue
        for key, raw_value in event.items():
            if key == "type":
                continue
            if isinstance(raw_value, str) and raw_value.strip():
                values.append(raw_value.strip())

    for bucket_name in ORGANIZATION_POOL_BUCKETS:
        expected_type = None
        if bucket_name != "organizations":
            expected_type = bucket_name[:-1]
        if bucket_name == "ngos":
            expected_type = "ngo"
        for organization in pool.get(bucket_name, []) or []:
            if not isinstance(organization, dict):
                continue
            try:
                normalized = normalize_organization_pool_entry(
                    organization,
                    expected_entity_type=expected_type,
                )
            except ValueError:
                continue
            values.append(normalized["name"])

    for bucket in ("awards", "products"):
        for entry in pool.get(bucket, []) or []:
            if not isinstance(entry, dict):
                continue
            raw_name = entry.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                values.append(raw_name.strip())

    for legal in pool.get("legals", []) or []:
        if not isinstance(legal, dict):
            continue
        for key in ("name", "reference_code"):
            raw_value = legal.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                values.append(raw_value.strip())
    return values


def _pool_wikipedia_hits(pool: dict[str, Any]) -> list[str]:
    normalized_values = sorted({normalize_query(value) for value in _iter_pool_strings(pool) if normalize_query(value)})
    if not normalized_values:
        return []
    cache_path, cache_mode = _wikipedia_cache_configuration()
    cache = load_cache(cache_path) if cache_path.exists() else {}
    cache = check_entities(
        normalized_values,
        cache_path,
        timeout=_POOL_WIKI_TIMEOUT,
        cache=cache,
        throttle=_POOL_WIKI_THROTTLE,
        max_retries=_POOL_WIKI_MAX_RETRIES,
        save=cache_mode == "read-write",
        replay_only=cache_mode == "replay-only",
        progress={
            "count": 0,
            "start": time.time(),
            "total": len(normalized_values),
            "log_every": 25,
        },
    )
    hits: list[str] = []
    for value in normalized_values:
        result = cache.get(value)
        if result and result.exists:
            hits.append(value)
    return hits


def _finalize_pool_candidate(
    pool: dict[str, Any],
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    validate_against_wikipedia: bool,
) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    """Normalize one pool candidate and collect Wikipedia hits before any write."""
    normalized = _normalize_pool_dict(pool, required_entities)
    if not validate_against_wikipedia:
        return normalized, []
    return normalized, _pool_wikipedia_hits(normalized)
