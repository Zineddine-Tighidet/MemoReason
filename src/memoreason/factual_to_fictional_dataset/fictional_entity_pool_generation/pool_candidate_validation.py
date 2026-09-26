"""Claude-backed generation of document-specific fictional entity pools."""

from __future__ import annotations

import logging
import re
from typing import Any


from memoreason.benchmark_definition.document_schema import AnnotatedDocument
from memoreason.benchmark_definition.organization_types import CANONICAL_ORGANIZATION_TYPES
from memoreason.benchmark_definition.annotation_runtime import (
    AnnotationParser,
    RuleEngine,
)
from memoreason.factual_to_fictional_dataset.fictional_entity_pool_generation.wikipedia_validation import (
    normalize_query,
)
from .pool_normalization import (
    POOL_BUCKETS,
    _empty_normalized_pool,
    _flatten_reference_pools,
    _normalize_pool_dict,
    _pool_support_shortages,
)
from .pool_validation import (
    _iter_pool_strings,
)
from .pool_generation_planning import (
    _augment_required_entities_with_rule_attrs,
    _bucket_for_entity_type,
    _manual_named_required_entities,
)
from .fictional_entity_replacement_pool_prompt_alignment import _reference_pool_shortages

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


def _reference_shortage_messages(
    shortages: dict[str, dict[str, tuple[int, int]]],
) -> list[str]:
    messages: list[str] = []
    for bucket, bucket_shortages in shortages.items():
        singular_type = bucket[:-1] if bucket.endswith("s") else bucket
        if bucket == "ngos":
            singular_type = "ngo"
        for entity_id, (actual, target) in sorted(bucket_shortages.items()):
            messages.append(f"{singular_type} {entity_id}: need {target}, found {actual}")
    return messages


def _variant_contains_banned_value(variant: dict[str, Any], banned_values: set[str]) -> bool:
    for raw_value in variant.values():
        value = str(raw_value or "").strip()
        if not value:
            continue
        if any(_value_conflicts_with_banned_literal(value, banned_value) for banned_value in banned_values):
            return True
    return False


def _prune_generation_unit_variants(
    pool: dict[str, Any],
    generation_unit: dict[str, list[tuple[str, list[str]]]],
    banned_values: set[str],
) -> dict[str, Any]:
    if not banned_values:
        return pool
    reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
    if not isinstance(reference_pools, dict) or not any(reference_pools.values()):
        return pool

    invalid_indexes: set[int] = set()
    for entity_type, specs in generation_unit.items():
        bucket = _bucket_for_entity_type(entity_type)
        if bucket is None:
            continue
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        for entity_id, _required_attrs in specs:
            ref_payload = bucket_refs.get(entity_id, {}) if isinstance(bucket_refs, dict) else {}
            variants = ref_payload.get("variants", []) if isinstance(ref_payload, dict) else []
            for variant_index, variant in enumerate(variants):
                if isinstance(variant, dict) and _variant_contains_banned_value(variant, banned_values):
                    invalid_indexes.add(variant_index)

    if not invalid_indexes:
        return pool

    pruned = _empty_normalized_pool()
    source_metadata = pool.get("_metadata", {}) if isinstance(pool, dict) else {}
    if isinstance(source_metadata, dict):
        pruned["_metadata"] = dict(source_metadata)

    for bucket in POOL_BUCKETS:
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        if not isinstance(bucket_refs, dict):
            continue
        pruned_bucket_refs: dict[str, Any] = {}
        for entity_id, ref_payload in bucket_refs.items():
            if not isinstance(ref_payload, dict):
                continue
            variants = [
                dict(variant)
                for variant_index, variant in enumerate(ref_payload.get("variants", []) or [])
                if variant_index not in invalid_indexes and isinstance(variant, dict)
            ]
            pruned_bucket_refs[entity_id] = {
                "required_attributes": list(ref_payload.get("required_attributes", [])),
                "count": len(variants),
                "variants": variants,
            }
        pruned["_reference_pools"][bucket] = pruned_bucket_refs
        pruned["_coverage"][bucket] = {
            entity_id: int(ref_payload.get("count", 0))
            for entity_id, ref_payload in pruned_bucket_refs.items()
            if isinstance(ref_payload, dict)
        }
        pruned[bucket] = _flatten_reference_pools({bucket: pruned_bucket_refs})[bucket]
    return pruned


def _value_conflicts_with_banned_literal(value: str, banned_value: str) -> bool:
    normalized_value = normalize_query(value)
    normalized_banned = normalize_query(banned_value)
    if not normalized_value or not normalized_banned:
        return False
    if normalized_value == normalized_banned:
        return True
    if len(normalized_value) >= 5 and len(normalized_banned) >= 5:
        if normalized_banned in normalized_value or normalized_value in normalized_banned:
            return True
    return False


def _pool_factual_literals(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> list[str]:
    """Return factual manual-entity literals that cannot appear in a fictional pool."""
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    factual_literals: set[str] = set()
    for entity_type, specs in required_entities.items():
        if entity_type not in _MANUAL_POOL_ENTITY_TYPES:
            continue
        for entity_id, attrs in specs:
            for attr in attrs:
                factual_value = RuleEngine._get_entity_value(factual_entities, f"{entity_id}.{attr}")
                text = str(factual_value or "").strip()
                if text:
                    factual_literals.add(text)
    return sorted(factual_literals)


def _pool_factual_literal_hits(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
    pool: dict[str, list[dict[str, str]]],
) -> list[str]:
    """Return factual manual-entity literals that still appear in a candidate pool."""
    factual_literals = set(_pool_factual_literals(document, required_entities))
    if not factual_literals:
        return []

    pool_values = [str(value).strip() for value in _iter_pool_strings(pool) if str(value).strip()]
    hits = {
        factual_literal
        for factual_literal in factual_literals
        if any(_value_conflicts_with_banned_literal(value, factual_literal) for value in pool_values)
    }
    return sorted(hits)


def _pool_banned_suffix_hits(pool: dict[str, list[dict[str, str]]]) -> list[str]:
    """Return pool values that use legacy suffix-fallback marker tokens."""
    hits = {
        value
        for value in (str(item).strip() for item in _iter_pool_strings(pool))
        if value and _BANNED_POOL_SUFFIX_PATTERN.search(value)
    }
    return sorted(hits)


def _validate_existing_pool(
    pool: dict[str, Any],
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, Any]:
    """Normalize and validate an existing pool without local synthetic repair."""
    manual_required_entities = _augment_required_entities_with_rule_attrs(
        document,
        _manual_named_required_entities(required_entities),
    )
    if isinstance(pool, dict) and isinstance(pool.get("_reference_pools"), dict):
        normalized_pool = pool
    else:
        normalized_pool = _normalize_pool_dict(pool, manual_required_entities)
    factual_hits = _pool_factual_literal_hits(document, manual_required_entities, normalized_pool)
    suffix_hits = _pool_banned_suffix_hits(normalized_pool)
    if factual_hits or suffix_hits:
        banned_hits = sorted({*factual_hits, *suffix_hits})
        raise ValueError(
            f"Existing entity pool for {document.document_id} reuses factual literals or legacy suffix tokens: "
            + ", ".join(banned_hits[:12])
        )
    reference_shortages = _reference_pool_shortages(normalized_pool, manual_required_entities)
    if reference_shortages:
        raise ValueError(
            f"Existing entity pool for {document.document_id} lacks enough per-reference Claude variants:\n"
            + "\n".join(_reference_shortage_messages(reference_shortages)[:24])
        )
    support_shortages = _pool_support_shortages(normalized_pool, manual_required_entities)
    if support_shortages:
        raise ValueError(
            f"Existing entity pool for {document.document_id} lacks enough Claude-generated support:\n"
            + "\n".join(support_shortages[:12])
        )
    return normalized_pool
