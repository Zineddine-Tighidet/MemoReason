"""Named-entity sampling stage for fictional generation."""

from __future__ import annotations


class NamedEntityPoolValidationMixin:
    """Validate manual entity-pool coverage before named-entity sampling."""

    def find_manual_pool_shortages(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
    ) -> list[str]:
        shortages: list[str] = []
        for entity_type, specs in required_entities.items():
            if entity_type not in self._MANUAL_ENTITY_TYPES:
                continue
            grouped_specs: dict[tuple[tuple[str, ...], str | None], list[str]] = {}
            for entity_id, required_attrs in specs:
                event_family = self._factual_event_family(entity_id) if entity_type == "event" else None
                grouped_specs.setdefault((tuple(sorted(required_attrs)), event_family), []).append(entity_id)
            for (attr_key, _event_family), entity_ids in grouped_specs.items():
                try:
                    pool_bucket = self._reference_pool_bucket(entity_type)
                    reference_pools = (
                        self.entity_pool.get("_reference_pools", {}) if isinstance(self.entity_pool, dict) else {}
                    )
                    bucket_refs = (
                        reference_pools.get(pool_bucket, {})
                        if pool_bucket and isinstance(reference_pools, dict)
                        else {}
                    )
                    has_reference_coverage = bool(bucket_refs) and all(
                        entity_id in bucket_refs for entity_id in entity_ids
                    )
                    valid_entities = self._valid_pool_entities(
                        entity_type,
                        list(attr_key),
                        [],
                        entity_id=None if has_reference_coverage else (entity_ids[0] if entity_ids else None),
                    )
                except ValueError as exc:
                    shortages.append(f"{entity_type} {sorted(entity_ids)}: {exc}")
                    continue
                if not valid_entities:
                    shortages.append(
                        f"{entity_type} {sorted(entity_ids)}: no valid pool entities for attrs {list(attr_key)}"
                    )
                    continue
                if len(valid_entities) < len(entity_ids):
                    shortages.append(
                        f"{entity_type} {sorted(entity_ids)}: need {len(entity_ids)} distinct candidates "
                        f"with attrs {list(attr_key)}, found {len(valid_entities)}"
                    )
        return shortages

    def find_manual_pool_support_shortages(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        *,
        candidates_per_required_entity: int | None = None,
        min_supported_documents: int | None = None,
    ) -> list[str]:
        if candidates_per_required_entity is None:
            candidates_per_required_entity = int(min_supported_documents) if min_supported_documents is not None else 15
        shortages: list[str] = []
        for entity_type, specs in required_entities.items():
            if entity_type not in self._MANUAL_ENTITY_TYPES:
                continue
            grouped_specs: dict[tuple[tuple[str, ...], str | None], list[str]] = {}
            for entity_id, required_attrs in specs:
                event_family = self._factual_event_family(entity_id) if entity_type == "event" else None
                grouped_specs.setdefault((tuple(sorted(required_attrs)), event_family), []).append(entity_id)
            for (attr_key, _event_family), entity_ids in grouped_specs.items():
                try:
                    pool_bucket = self._reference_pool_bucket(entity_type)
                    reference_pools = (
                        self.entity_pool.get("_reference_pools", {}) if isinstance(self.entity_pool, dict) else {}
                    )
                    bucket_refs = (
                        reference_pools.get(pool_bucket, {})
                        if pool_bucket and isinstance(reference_pools, dict)
                        else {}
                    )
                    has_reference_coverage = bool(bucket_refs) and all(
                        entity_id in bucket_refs for entity_id in entity_ids
                    )
                    valid_entities = self._valid_pool_entities(
                        entity_type,
                        list(attr_key),
                        [],
                        entity_id=None if has_reference_coverage else (entity_ids[0] if entity_ids else None),
                    )
                except ValueError as exc:
                    shortages.append(f"{entity_type} {sorted(entity_ids)}: {exc}")
                    continue
                required_candidates = len(entity_ids) * int(candidates_per_required_entity)
                if len(valid_entities) < required_candidates:
                    shortages.append(
                        f"{entity_type} {sorted(entity_ids)}: need {required_candidates} candidates "
                        f"({candidates_per_required_entity} per entity) with attrs {list(attr_key)}, "
                        f"found {len(valid_entities)}"
                    )
        return shortages


__all__ = ["NamedEntityPoolValidationMixin"]
