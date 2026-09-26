"""Claude-backed generation of document-specific fictional entity pools."""

from __future__ import annotations

import logging
import re
from typing import Any


from memoreason.benchmark_definition.document_schema import AnnotatedDocument, EntityCollection, ENTITY_TYPE_TO_CLASS
from memoreason.benchmark_definition.organization_types import CANONICAL_ORGANIZATION_TYPES
from memoreason.benchmark_definition.organization_types import normalize_organization_pool_entry
from memoreason.benchmark_definition.annotation_runtime import (
    RuleEngine,
    find_entity_refs,
    parse_entity_id,
)
from .pool_normalization import (
    _pool_bucket_shortages as _pool_bucket_shortages_impl,
)
from .fictional_entity_replacement_pool_prompt_planning import (
    build_fictional_entity_replacement_pool_generation_prompt as _build_pool_generation_prompt_impl,
)
from .fictional_entity_replacement_pool_target_planning import (
    build_fictional_entity_replacement_pool_target_plan,
)
from .pool_generation_planning import _bucket_for_entity_type

logger = logging.getLogger(__name__)
_BANNED_POOL_SUFFIX_TOKENS = frozenset({"Alt", "Astra", "Nova", "Prime", "Sigma"})
_BANNED_POOL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in sorted(_BANNED_POOL_SUFFIX_TOKENS)) + r")\b"
)
_MANUAL_POOL_ENTITY_TYPES = {
    "person",
    "place",
    "event",
    *CANONICAL_ORGANIZATION_TYPES,
    "award",
    "legal",
    "product",
}
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


def build_fictional_entity_replacement_pool_generation_prompt(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    theme: str,
    feedback: str | None = None,
    forbidden_values: set[str] | None = None,
    accepted_pool: dict[str, Any] | None = None,
    shortage_targets: dict[str, dict[str, tuple[int, int]]] | None = None,
) -> tuple[str, str]:
    """Build prompts for pool generation (compatible with module-level monkeypatching)."""
    return _build_pool_generation_prompt_impl(
        document,
        required_entities,
        theme=theme,
        target_plan_builder=build_fictional_entity_replacement_pool_target_plan,
        feedback=feedback,
        forbidden_values=forbidden_values,
        accepted_pool=accepted_pool,
        shortage_targets=shortage_targets,
    )


def _pool_bucket_shortages(
    pool: dict[str, list[dict[str, str]]],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, tuple[int, int]]:
    """Return bucket shortages using the currently bound target-plan function."""
    return _pool_bucket_shortages_impl(
        pool,
        required_entities,
        target_plan_builder=build_fictional_entity_replacement_pool_target_plan,
    )


def _reference_pool_shortages(
    pool: dict[str, Any],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, dict[str, tuple[int, int]]]:
    target_plan = build_fictional_entity_replacement_pool_target_plan(required_entities)
    reference_targets = getattr(target_plan, "reference_targets", {}) or {}
    if not reference_targets:
        return {}
    reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
    shortages: dict[str, dict[str, tuple[int, int]]] = {}
    for bucket, bucket_targets in reference_targets.items():
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        bucket_shortages: dict[str, tuple[int, int]] = {}
        for entity_id, target_count in bucket_targets.items():
            ref_payload = bucket_refs.get(entity_id, {}) if isinstance(bucket_refs, dict) else {}
            actual_count = int(ref_payload.get("count", 0)) if isinstance(ref_payload, dict) else 0
            if actual_count < target_count:
                bucket_shortages[entity_id] = (actual_count, int(target_count))
        if bucket_shortages:
            shortages[bucket] = bucket_shortages
    return shortages


def _pool_rule_ref_supported(ref: str) -> bool:
    entity_id, attribute = ref.split(".", 1) if "." in ref else (ref, "")
    entity_type, _entity_index = parse_entity_id(entity_id)
    if entity_type in {None, "number", "temporal"}:
        return False
    if entity_type == "person" and attribute.split(".", 1)[0] in _AUTO_GENERATED_PERSON_ATTRIBUTES:
        return False
    return entity_type in _MANUAL_POOL_ENTITY_TYPES


def _generation_unit_rule_subset(
    document: AnnotatedDocument,
    generation_unit: dict[str, list[tuple[str, list[str]]]],
) -> list[str]:
    unit_entity_ids = {entity_id for specs in generation_unit.values() for entity_id, _required_attributes in specs}
    relevant_rules: list[str] = []
    for raw_rule in document.rules or []:
        rule = str(raw_rule)
        refs = find_entity_refs(rule)
        if not refs or not all(_pool_rule_ref_supported(ref) for ref in refs):
            continue
        referenced_entity_ids = {ref.split(".", 1)[0] for ref in refs}
        if referenced_entity_ids.issubset(unit_entity_ids):
            relevant_rules.append(rule)
    return relevant_rules


def _entity_collection_for_variant(
    pool: dict[str, Any],
    generation_unit: dict[str, list[tuple[str, list[str]]]],
    *,
    variant_index: int,
) -> EntityCollection:
    reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
    entities = EntityCollection()
    for entity_type, specs in generation_unit.items():
        bucket = _bucket_for_entity_type(entity_type)
        if bucket is None:
            continue
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        for entity_id, _required_attrs in specs:
            ref_payload = bucket_refs.get(entity_id, {}) if isinstance(bucket_refs, dict) else {}
            variants = ref_payload.get("variants", []) if isinstance(ref_payload, dict) else []
            if variant_index >= len(variants):
                raise ValueError(f"{entity_id} is missing variant {variant_index + 1}")
            raw_variant = variants[variant_index]
            if not isinstance(raw_variant, dict):
                raise ValueError(f"{entity_id} variant {variant_index + 1} is not a mapping")
            entity_payload = {key: str(value).strip() for key, value in raw_variant.items() if str(value).strip()}
            entity_cls = ENTITY_TYPE_TO_CLASS[entity_type]
            if entity_type in CANONICAL_ORGANIZATION_TYPES:
                entity_payload = normalize_organization_pool_entry(
                    entity_payload,
                    expected_entity_type=entity_type,
                )
                entity_payload["organization_kind"] = entity_type
            entity = entity_cls(**entity_payload)
            entities.add_entity(entity_type, entity_id, entity)
    return entities


def _generation_unit_alignment_failures(
    document: AnnotatedDocument,
    generation_unit: dict[str, list[tuple[str, list[str]]]],
    pool: dict[str, Any],
) -> list[str]:
    reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
    if not isinstance(reference_pools, dict) or not any(reference_pools.values()):
        return []

    variant_counts: list[int] = []
    for entity_type, specs in generation_unit.items():
        bucket = _bucket_for_entity_type(entity_type)
        if bucket is None:
            continue
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        for entity_id, _required_attrs in specs:
            ref_payload = bucket_refs.get(entity_id, {}) if isinstance(bucket_refs, dict) else {}
            variants = ref_payload.get("variants", []) if isinstance(ref_payload, dict) else []
            if variants:
                variant_counts.append(len(variants))
    if not variant_counts:
        return []

    relevant_rules = _generation_unit_rule_subset(document, generation_unit)
    failures: list[str] = []
    shared_variant_count = min(variant_counts)
    for variant_index in range(shared_variant_count):
        try:
            entities = _entity_collection_for_variant(pool, generation_unit, variant_index=variant_index)
        except Exception as exc:
            failures.append(str(exc))
            break

        for entity_id, place in entities.places.items():
            demonym = str(place.demonym or "").strip()
            nationality = str(place.nationality or "").strip()
            if demonym and nationality and demonym.casefold() != nationality.casefold():
                failures.append(
                    f"variant {variant_index + 1}: place {entity_id} uses demonym {demonym!r} but nationality {nationality!r}"
                )
                if len(failures) >= 6:
                    return failures

        if not relevant_rules:
            continue

        rule_results = RuleEngine.validate_all_rules(relevant_rules, entities)
        invalid_rules = [rule for rule, is_valid in rule_results if not is_valid]
        if invalid_rules:
            failures.append(f"variant {variant_index + 1}: rule mismatch for " + "; ".join(invalid_rules[:3]))
            if len(failures) >= 6:
                return failures
    return failures


def _is_non_retryable_provider_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    if not message:
        return False
    return "credit balance is too low" in message or "invalid_request_error" in message
