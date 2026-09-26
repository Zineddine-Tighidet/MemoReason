"""Target-size planning for document-specific fictional entity pools."""

from __future__ import annotations

from dataclasses import dataclass, field

from memoreason.benchmark_definition.organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    organization_pool_bucket,
)

FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY = 15
_AUTO_GENERATED_PERSON_ATTRIBUTES = frozenset(
    {
        "age",
        "gender",
        "subj_pronoun",
        "obj_pronoun",
        "poss_det_pronoun",
        "poss_pro_pronoun",
        "refl_pronoun",
        "honorific",
        "relationship",
    }
)

POOL_BUCKET_TO_ENTITY_TYPES: dict[str, tuple[str, ...]] = {
    "persons": ("person",),
    "places": ("place",),
    "events": ("event",),
    **{
        organization_pool_bucket(organization_type) or "organizations": (organization_type,)
        for organization_type in CANONICAL_ORGANIZATION_TYPES
    },
    "awards": ("award",),
    "legals": ("legal",),
    "products": ("product",),
}


@dataclass(frozen=True)
class FictionalEntityReplacementPoolTargetPlan:
    """Required fictional candidate counts for every entity in one template."""

    bucket_sizes: dict[str, int]
    candidates_per_required_entity: int = FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY
    reference_targets: dict[str, dict[str, int]] = field(default_factory=dict)


def compute_required_fictional_entity_candidates_per_pool_bucket(
    required_entity_count: int,
    *,
    candidates_per_required_entity: int = FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
) -> int:
    """Return the candidate count needed for distinct variant assignments."""
    if required_entity_count <= 0:
        return 0
    return int(required_entity_count) * int(candidates_per_required_entity)


def build_reference_target_counts(
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    candidates_per_required_entity: int = FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
) -> dict[str, dict[str, int]]:
    """Return the requested candidate count for each required entity reference."""
    reference_targets: dict[str, dict[str, int]] = {}
    for bucket, entity_types in POOL_BUCKET_TO_ENTITY_TYPES.items():
        bucket_targets: dict[str, int] = {}
        for entity_type in entity_types:
            for entity_id, attrs in required_entities.get(entity_type, []):
                if entity_type == "person":
                    filtered_attrs = [attr for attr in (attrs or []) if attr not in _AUTO_GENERATED_PERSON_ATTRIBUTES]
                    if not filtered_attrs:
                        continue
                bucket_targets[entity_id] = int(candidates_per_required_entity)
        if bucket_targets:
            reference_targets[bucket] = bucket_targets
    return reference_targets


def build_fictional_entity_replacement_pool_target_plan(
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    candidates_per_required_entity: int = FICTIONAL_ENTITY_CANDIDATES_PER_REQUIRED_ENTITY,
) -> FictionalEntityReplacementPoolTargetPlan:
    """Compute pool targets from the required entities of one template."""
    bucket_sizes: dict[str, int] = {}
    for bucket, entity_types in POOL_BUCKET_TO_ENTITY_TYPES.items():
        required_count = sum(len(required_entities.get(entity_type, [])) for entity_type in entity_types)
        bucket_sizes[bucket] = compute_required_fictional_entity_candidates_per_pool_bucket(
            required_count,
            candidates_per_required_entity=candidates_per_required_entity,
        )

    return FictionalEntityReplacementPoolTargetPlan(
        bucket_sizes=bucket_sizes,
        candidates_per_required_entity=candidates_per_required_entity,
        reference_targets=build_reference_target_counts(
            required_entities,
            candidates_per_required_entity=candidates_per_required_entity,
        ),
    )
