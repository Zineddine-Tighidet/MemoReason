"""Pool payload parsing/normalization and support-shortage checks."""

from __future__ import annotations

from typing import Any

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.benchmark_definition.organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    organization_pool_bucket,
)
from .fictional_entity_replacement_pool_target_planning import (
    FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
    build_fictional_entity_replacement_pool_target_plan,
    compute_required_fictional_entity_candidates_per_pool_bucket,
)
from ..dataset_paths import DEFAULT_RANDOM_SEED


def _pool_bucket_shortages(
    pool: dict[str, list[dict[str, str]]],
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    target_plan_builder=None,
) -> dict[str, tuple[int, int]]:
    builder = target_plan_builder or build_fictional_entity_replacement_pool_target_plan
    target_plan = builder(required_entities)
    shortages: dict[str, tuple[int, int]] = {}
    for bucket, minimum_size in target_plan.bucket_sizes.items():
        actual_size = len(pool.get(bucket, []))
        if actual_size < minimum_size:
            shortages[bucket] = (actual_size, minimum_size)
    return shortages


def _reference_pool_shortages(
    pool: dict[str, Any],
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    candidates_per_required_entity: int = FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
) -> dict[str, dict[str, tuple[int, int]]]:
    shortages: dict[str, dict[str, tuple[int, int]]] = {}
    target_plan = build_fictional_entity_replacement_pool_target_plan(
        required_entities,
        candidates_per_required_entity=candidates_per_required_entity,
    )
    reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
    for bucket, bucket_targets in target_plan.reference_targets.items():
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        bucket_shortages: dict[str, tuple[int, int]] = {}
        for entity_id, target_count in bucket_targets.items():
            ref_payload = bucket_refs.get(entity_id, {}) if isinstance(bucket_refs, dict) else {}
            actual_count = int(ref_payload.get("count", 0)) if isinstance(ref_payload, dict) else 0
            if actual_count < target_count:
                bucket_shortages[entity_id] = (actual_count, target_count)
        if bucket_shortages:
            shortages[bucket] = bucket_shortages
    return shortages


def _render_bucket_shortages(shortages: dict[str, tuple[int, int]]) -> str:
    return ", ".join(f"{bucket}: {actual}/{minimum}" for bucket, (actual, minimum) in sorted(shortages.items()))


def _distinct_entry_count(entries: list[dict[str, str]]) -> int:
    seen: set[tuple[tuple[str, str], ...]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        seen.add(tuple(sorted((str(key), str(value)) for key, value in entry.items() if str(value).strip())))
    return len(seen)


def _pool_support_shortages(
    pool: dict[str, list[dict[str, str]]],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> list[str]:
    sampler = FictionalEntitySampler(pool, seed=DEFAULT_RANDOM_SEED)
    shortages = sampler.find_manual_pool_support_shortages(
        required_entities,
        candidates_per_required_entity=FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
    )
    globally_distinct_types = {
        "person",
        "event",
        *CANONICAL_ORGANIZATION_TYPES,
        "award",
        "legal",
        "product",
    }
    for entity_type, specs in required_entities.items():
        if entity_type not in globally_distinct_types or not specs:
            continue
        representative_attrs = sorted({attr for _entity_id, attrs in specs for attr in attrs})
        entity_ids = [entity_id for entity_id, _attrs in specs]
        if entity_type in CANONICAL_ORGANIZATION_TYPES:
            pool_bucket = organization_pool_bucket(entity_type)
        else:
            pool_bucket = {
                "person": "persons",
                "event": "events",
                "award": "awards",
                "legal": "legals",
                "product": "products",
            }.get(entity_type)
        reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
        bucket_refs = reference_pools.get(pool_bucket, {}) if pool_bucket and isinstance(reference_pools, dict) else {}
        has_reference_coverage = bool(bucket_refs) and all(entity_id in bucket_refs for entity_id in entity_ids)
        required_candidates = compute_required_fictional_entity_candidates_per_pool_bucket(
            len(entity_ids),
            candidates_per_required_entity=FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
        )
        if has_reference_coverage:
            distinct_variants = _distinct_entry_count(
                [
                    variant
                    for entity_id in entity_ids
                    for variant in bucket_refs.get(entity_id, {}).get("variants", [])
                    if isinstance(variant, dict)
                ]
            )
            if distinct_variants < required_candidates:
                shortages.append(
                    f"{entity_type} {sorted(entity_ids)}: need {required_candidates} total distinct candidates "
                    f"across all reference pools, found {distinct_variants}"
                )
            continue
        try:
            valid_entities = sampler._valid_pool_entities(
                entity_type,
                representative_attrs,
                [],
                entity_id=None if has_reference_coverage else (entity_ids[0] if entity_ids else None),
            )
        except ValueError as exc:
            shortages.append(f"{entity_type} {sorted(entity_ids)}: {exc}")
            continue
        if len(valid_entities) < required_candidates:
            shortages.append(
                f"{entity_type} {sorted(entity_ids)}: need {required_candidates} total distinct candidates "
                f"across all required attrs {representative_attrs}, found {len(valid_entities)}"
            )
    return shortages


__all__ = [
    "_pool_bucket_shortages",
    "_pool_support_shortages",
    "_reference_pool_shortages",
    "_render_bucket_shortages",
]
