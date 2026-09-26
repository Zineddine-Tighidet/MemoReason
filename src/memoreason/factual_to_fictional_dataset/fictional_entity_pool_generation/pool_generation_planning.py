"""Claude-backed generation of document-specific fictional entity pools."""

from __future__ import annotations

import logging
import re
from typing import Any


from memoreason.benchmark_definition.document_schema import AnnotatedDocument
from memoreason.benchmark_definition.organization_types import CANONICAL_ORGANIZATION_TYPES, organization_pool_bucket
from memoreason.benchmark_definition.annotation_runtime import (
    find_entity_refs,
    parse_entity_id,
)
from .pool_normalization import (
    POOL_BUCKETS,
    _empty_normalized_pool,
)

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


def _merge_normalized_pools(
    base: dict[str, Any] | None,
    addition: dict[str, Any],
) -> dict[str, Any]:
    """Merge Claude-returned pool entries without inventing local replacements."""
    merged = _empty_normalized_pool()
    merged["_metadata"] = {}
    for source in (base or {}, addition):
        source_metadata = source.get("_metadata", {}) if isinstance(source, dict) else {}
        if isinstance(source_metadata, dict):
            merged["_metadata"].update(source_metadata)
        for bucket in POOL_BUCKETS:
            target_entries = merged.setdefault(bucket, [])
            source_entries = source.get(bucket, []) if isinstance(source, dict) else []
            seen = {tuple(sorted(entry.items())) for entry in target_entries if isinstance(entry, dict)}
            for entry in source_entries or []:
                if not isinstance(entry, dict):
                    continue
                entry_key = tuple(sorted(entry.items()))
                if entry_key in seen:
                    continue
                seen.add(entry_key)
                target_entries.append(dict(entry))
        source_reference_pools = source.get("_reference_pools", {}) if isinstance(source, dict) else {}
        target_reference_pools = merged.setdefault("_reference_pools", {bucket: {} for bucket in POOL_BUCKETS})
        target_coverage = merged.setdefault("_coverage", {bucket: {} for bucket in POOL_BUCKETS})
        global_seen = {
            bucket: {
                tuple(sorted(entry.items()))
                for ref_payload in target_reference_pools.get(bucket, {}).values()
                if isinstance(ref_payload, dict)
                for entry in ref_payload.get("variants", []) or []
                if isinstance(entry, dict)
            }
            for bucket in POOL_BUCKETS
        }
        for bucket in POOL_BUCKETS:
            bucket_refs = source_reference_pools.get(bucket, {}) if isinstance(source_reference_pools, dict) else {}
            if not isinstance(bucket_refs, dict):
                continue
            target_bucket_refs = target_reference_pools.setdefault(bucket, {})
            for entity_id, ref_payload in bucket_refs.items():
                if not isinstance(ref_payload, dict):
                    continue
                existing_ref = target_bucket_refs.get(entity_id, {})
                existing_variants = (
                    list(existing_ref.get("variants", []) or []) if isinstance(existing_ref, dict) else []
                )
                ref_seen = {tuple(sorted(entry.items())) for entry in existing_variants if isinstance(entry, dict)}
                variants: list[dict[str, str]] = [dict(entry) for entry in existing_variants if isinstance(entry, dict)]
                for entry in ref_payload.get("variants", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    entry_key = tuple(
                        sorted((key, str(value).strip()) for key, value in entry.items() if str(value).strip())
                    )
                    if not entry_key or entry_key in ref_seen or entry_key in global_seen[bucket]:
                        continue
                    cleaned_entry = {key: str(value).strip() for key, value in entry.items() if str(value).strip()}
                    ref_seen.add(entry_key)
                    global_seen[bucket].add(entry_key)
                    variants.append(cleaned_entry)
                target_bucket_refs[entity_id] = {
                    "required_attributes": list(
                        existing_ref.get("required_attributes", ref_payload.get("required_attributes", []))
                    ),
                    "count": len(variants),
                    "variants": variants,
                }
                target_coverage.setdefault(bucket, {})[entity_id] = len(variants)
    for bucket in POOL_BUCKETS:
        seen = set()
        flattened: list[dict[str, str]] = []
        for ref_payload in merged.get("_reference_pools", {}).get(bucket, {}).values():
            if not isinstance(ref_payload, dict):
                continue
            for entry in ref_payload.get("variants", []) or []:
                if not isinstance(entry, dict):
                    continue
                entry_key = tuple(sorted(entry.items()))
                if entry_key in seen:
                    continue
                seen.add(entry_key)
                flattened.append(dict(entry))
        if flattened:
            merged[bucket] = flattened
    return merged


def _bucket_for_entity_type(entity_type: str) -> str | None:
    if entity_type in CANONICAL_ORGANIZATION_TYPES:
        return organization_pool_bucket(entity_type)
    return {
        "person": "persons",
        "place": "places",
        "event": "events",
        "award": "awards",
        "legal": "legals",
        "product": "products",
    }.get(entity_type)


def _manual_named_required_entities(
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, list[tuple[str, list[str]]]]:
    filtered: dict[str, list[tuple[str, list[str]]]] = {}
    for entity_type, specs in required_entities.items():
        if entity_type not in _MANUAL_POOL_ENTITY_TYPES or not specs:
            continue
        kept_specs: list[tuple[str, list[str]]] = []
        for entity_id, attrs in specs:
            filtered_attrs = list(attrs or [])
            if entity_type == "person":
                filtered_attrs = [attr for attr in filtered_attrs if attr not in _AUTO_GENERATED_PERSON_ATTRIBUTES]
            if entity_type == "person" and not filtered_attrs:
                continue
            kept_specs.append((entity_id, filtered_attrs))
        if kept_specs:
            filtered[entity_type] = kept_specs
    return filtered


def _augment_required_entities_with_rule_attrs(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, list[tuple[str, list[str]]]]:
    augmented: dict[str, list[tuple[str, list[str]]]] = {
        entity_type: [(entity_id, list(attrs or [])) for entity_id, attrs in specs]
        for entity_type, specs in required_entities.items()
    }
    attrs_by_entity_id: dict[str, set[str]] = {
        entity_id: set(attrs or []) for specs in augmented.values() for entity_id, attrs in specs
    }

    for raw_rule in document.rules or []:
        for raw_ref in find_entity_refs(str(raw_rule)):
            entity_id, attribute = raw_ref.split(".", 1) if "." in raw_ref else (raw_ref, "")
            entity_type, _entity_index = parse_entity_id(entity_id)
            if entity_type not in _MANUAL_POOL_ENTITY_TYPES or entity_id not in attrs_by_entity_id:
                continue
            root_attribute = attribute.split(".", 1)[0].strip()
            if not root_attribute:
                continue
            if entity_type == "person" and root_attribute in _AUTO_GENERATED_PERSON_ATTRIBUTES:
                continue
            attrs_by_entity_id[entity_id].add(root_attribute)

    for entity_type, specs in augmented.items():
        augmented[entity_type] = [
            (entity_id, sorted(attrs_by_entity_id.get(entity_id, set(attrs or [])))) for entity_id, attrs in specs
        ]
    return augmented


def _rule_connected_entity_ids(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> list[set[str]]:
    required_ids = {entity_id for specs in required_entities.values() for entity_id, _attrs in specs}
    parent: dict[str, str] = {}

    def find(entity_id: str) -> str:
        parent.setdefault(entity_id, entity_id)
        while parent[entity_id] != entity_id:
            parent[entity_id] = parent[parent[entity_id]]
            entity_id = parent[entity_id]
        return entity_id

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for raw_rule in document.rules or []:
        refs: list[str] = []
        for raw_ref in find_entity_refs(str(raw_rule)):
            entity_id = raw_ref.split(".", 1)[0]
            entity_type, _entity_index = parse_entity_id(entity_id)
            if entity_type in {None, "number", "temporal"}:
                continue
            if entity_id not in required_ids:
                continue
            refs.append(entity_id)
        unique_refs = sorted(set(refs))
        for index in range(1, len(unique_refs)):
            union(unique_refs[0], unique_refs[index])

    components: dict[str, set[str]] = {}
    for entity_id in required_ids:
        root = find(entity_id) if entity_id in parent else entity_id
        components.setdefault(root, set()).add(entity_id)
    return [component for component in components.values() if len(component) > 1]


def _chunk_size_for_entity_type(entity_type: str) -> int:
    # Isolated references are more reliable when Claude handles one opaque id at a time.
    # Rule-connected components are still kept together by `_generation_units`.
    return 1


def _generation_units(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> list[dict[str, list[tuple[str, list[str]]]]]:
    manual_required = _manual_named_required_entities(required_entities)
    if not manual_required:
        return []

    entity_specs_by_id = {
        entity_id: (entity_type, attrs) for entity_type, specs in manual_required.items() for entity_id, attrs in specs
    }
    connected_components = _rule_connected_entity_ids(document, manual_required)
    covered_ids = {entity_id for component in connected_components for entity_id in component}

    units: list[dict[str, list[tuple[str, list[str]]]]] = []
    for component in connected_components:
        unit: dict[str, list[tuple[str, list[str]]]] = {}
        for entity_id in sorted(component):
            entity_type, attrs = entity_specs_by_id[entity_id]
            unit.setdefault(entity_type, []).append((entity_id, attrs))
        units.append(unit)

    for entity_type, specs in manual_required.items():
        isolated_specs = [(entity_id, attrs) for entity_id, attrs in specs if entity_id not in covered_ids]
        if not isolated_specs:
            continue
        chunk_size = _chunk_size_for_entity_type(entity_type)
        for start in range(0, len(isolated_specs), chunk_size):
            units.append({entity_type: isolated_specs[start : start + chunk_size]})
    return units
