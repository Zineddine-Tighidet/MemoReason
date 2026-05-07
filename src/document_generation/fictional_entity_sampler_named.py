"""Named-entity sampling stage for fictional generation."""

from __future__ import annotations

import copy
import random
import re
import time
from typing import Any

from src.core.annotation_runtime import RuleEngine, map_relationship_for_gender
from src.core.document_schema import ENTITY_TYPE_TO_CLASS, EntityCollection
from src.core.organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    normalize_organization_pool_entry,
    organization_attribute_value,
    organization_pool_bucket,
)

from .fictional_entity_sampler_common import logger
from .generation_limits import MAX_MANUAL_ATTEMPTS, SLOW_STAGE_LOG_SECONDS


class FictionalEntitySamplerNamedMixin:
    """Pool-based named-entity sampling helpers."""

    _AUTO_GENERATED_PERSON_ONLY_ATTRS = frozenset({"age"})
    _PERSON_GENDER_LITERAL_RULE_RE = re.compile(
        r'^\s*(person_\d+)\.gender\s*(==|=)\s*["\']?(male|female|neutral)["\']?\s*$',
        re.IGNORECASE,
    )

    @staticmethod
    def _has_primary_org_place_collision(entities: EntityCollection) -> bool:
        primary_org = entities.organizations.get("entreprise_org_1")
        primary_org_name = str(getattr(primary_org, "name", "") or "").strip()
        if not primary_org_name:
            return False
        normalized_org_name = re.sub(r"[^a-z0-9]+", " ", primary_org_name.lower()).strip()
        org_tokens = {token for token in normalized_org_name.split() if token}
        for place in entities.places.values():
            for attr in ("country", "city", "region", "state", "natural_site"):
                value = str(getattr(place, attr, "") or "").strip()
                normalized_place = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
                if value and normalized_place and (
                    normalized_org_name == normalized_place or normalized_place in org_tokens
                ):
                    return True
        return False

    @staticmethod
    def _reference_pool_bucket(entity_type: str) -> str | None:
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

    @staticmethod
    def _pool_entry_key(entry: Any) -> tuple[tuple[str, str], ...] | None:
        if isinstance(entry, dict):
            normalized = tuple(
                sorted((str(key), str(value)) for key, value in entry.items() if str(key).strip() and str(value).strip())
            )
            return normalized or None
        if hasattr(entry, "model_dump"):
            payload = entry.model_dump()
            if isinstance(payload, dict):
                normalized = tuple(
                    sorted(
                        (str(key), str(value))
                        for key, value in payload.items()
                        if str(key).strip() and str(value).strip()
                    )
                )
                return normalized or None
        return None

    def _explicit_person_gender_constraints(self, rules: list[str]) -> dict[str, str] | None:
        constraints: dict[str, str] = {}
        for raw_rule in rules or []:
            cleaned = self._strip_rule_comment(str(raw_rule))
            match = self._PERSON_GENDER_LITERAL_RULE_RE.fullmatch(cleaned)
            if not match:
                continue
            person_id = str(match.group(1) or "").strip()
            gender = str(match.group(3) or "").strip().lower()
            if not person_id or not gender:
                continue
            existing = constraints.get(person_id)
            if existing is not None and existing != gender:
                return None
            constraints[person_id] = gender
        return constraints

    def _sample_person_genders(
        self,
        person_ids: list[str],
        rules: list[str],
    ) -> dict[str, str] | None:
        fixed_genders = self._explicit_person_gender_constraints(rules)
        if fixed_genders is None:
            return None

        resolved: dict[str, str] = {
            person_id: gender
            for person_id, gender in fixed_genders.items()
            if person_id in person_ids
        }
        remaining_ids = [person_id for person_id in person_ids if person_id not in resolved]

        fixed_male = sum(1 for gender in resolved.values() if gender == "male")
        fixed_female = sum(1 for gender in resolved.values() if gender == "female")
        candidate_male_counts = list(range(len(remaining_ids) + 1))
        if not candidate_male_counts:
            return resolved

        def imbalance(extra_males: int) -> int:
            extra_females = len(remaining_ids) - extra_males
            return abs((fixed_male + extra_males) - (fixed_female + extra_females))

        best_imbalance = min(imbalance(extra_males) for extra_males in candidate_male_counts)
        best_male_counts = [
            extra_males for extra_males in candidate_male_counts if imbalance(extra_males) == best_imbalance
        ]
        sampled_male_count = random.choice(best_male_counts)
        sampled_genders = ["male"] * sampled_male_count + ["female"] * (len(remaining_ids) - sampled_male_count)
        random.shuffle(sampled_genders)
        resolved.update(dict(zip(remaining_ids, sampled_genders, strict=True)))
        return resolved

    def sample_named_entities(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        rules: list[str],
    ) -> EntityCollection | None:
        """Sample named entities from the fictional pool for one variant."""
        self._current_rules = list(rules or [])
        manual_required = {k: v for k, v in required_entities.items() if k in self._MANUAL_ENTITY_TYPES}
        manual_rules = [rule for rule in rules if self._is_manual_rule(str(rule))]

        base = None
        original_reference_variant_index = self.reference_variant_index
        manual_attempts = MAX_MANUAL_ATTEMPTS if manual_rules else 1
        if not manual_rules and original_reference_variant_index is not None:
            manual_attempts = min(MAX_MANUAL_ATTEMPTS, 4)
        manual_start = time.monotonic()
        try:
            for _manual_attempt in range(manual_attempts):
                if original_reference_variant_index is not None:
                    self.reference_variant_index = original_reference_variant_index + (
                        _manual_attempt * int(self.reference_variant_count or 1)
                    )
                if manual_rules:
                    base = self._sample_manual_entities_with_constraints(manual_required, manual_rules)
                    if base is None:
                        continue
                else:
                    base = EntityCollection()
                    used_entities: dict[str, list[Any]] = {}
                    for entity_type, specs in manual_required.items():
                        if entity_type not in used_entities:
                            used_entities[entity_type] = []
                        specs_in_sampling_order = sorted(
                            specs,
                            key=lambda spec: (
                                len(
                                    self._valid_pool_entities(
                                        entity_type,
                                        spec[1],
                                        used_entities[entity_type],
                                        entity_id=spec[0],
                                    )
                                ),
                                -len(spec[1]),
                                spec[0],
                            ),
                        )
                        for entity_id, required_attrs in specs_in_sampling_order:
                            try:
                                sampled = self._sample_entity_with_attributes(
                                    entity_type,
                                    required_attrs,
                                    used_entities[entity_type],
                                    entity_id=entity_id,
                                )
                            except ValueError as exc:
                                logger.debug("Cannot sample %s %s: %s", entity_type, entity_id, exc)
                                return None
                            used_entities[entity_type].append(sampled)
                            base.add_entity(entity_type, entity_id, sampled)

                person_ids = list(base.persons.keys())
                random.shuffle(person_ids)
                if person_ids:
                    gender_map = self._sample_person_genders(person_ids, rules)
                    if gender_map is None:
                        return None

                    for person_id in person_ids:
                        self._apply_gender_profile(base.persons[person_id], gender_map[person_id])

                if self.factual_entities and self.factual_entities.persons:
                    for person_id, person_entity in base.persons.items():
                        if person_id not in self.factual_entities.persons:
                            continue
                        orig = self.factual_entities.persons[person_id]
                        if orig.relationships:
                            person_entity.relationships = {
                                other_id: map_relationship_for_gender(rel, person_entity.gender)
                                for other_id, rel in orig.relationships.items()
                            }
                        elif getattr(orig, "relationship", None):
                            person_entity.relationship = map_relationship_for_gender(
                                orig.relationship,
                                person_entity.gender,
                            )

                if self._has_primary_org_place_collision(base):
                    base = None
                    continue

                if not manual_rules:
                    break
                manual_results = RuleEngine.validate_all_rules(manual_rules, base)
                if all(is_valid for _, is_valid in manual_results):
                    break
                base = None
        finally:
            self.reference_variant_index = original_reference_variant_index

        if base is None:
            return None
        manual_elapsed = time.monotonic() - manual_start
        if manual_elapsed > SLOW_STAGE_LOG_SECONDS:
            print(f"[slow] Manual sampling took {manual_elapsed:.1f}s (attempts={manual_attempts}).", flush=True)
        return base

    def _sample_entity_with_attributes(
        self,
        entity_type: str,
        required_attrs: list[str],
        used_entities: list[Any] | None = None,
        entity_id: str | None = None,
    ) -> Any:
        if used_entities is None:
            used_entities = []
        sampled_entity = None
        if entity_type == "person" and set(required_attrs or []).issubset(self._AUTO_GENERATED_PERSON_ONLY_ATTRS):
            sampled_entity = ENTITY_TYPE_TO_CLASS["person"]()
        else:
            valid_entities = self._valid_pool_entities(entity_type, required_attrs, used_entities, entity_id=entity_id)
            if not valid_entities:
                raise ValueError(f"No entities in pool for type '{entity_type}' with attributes {required_attrs}")
            if self.reference_variant_index is not None and valid_entities:
                pool_bucket = self._reference_pool_bucket(entity_type)
                reference_pools = self.entity_pool.get("_reference_pools", {}) if isinstance(self.entity_pool, dict) else {}
                ref_pool = (
                    reference_pools.get(pool_bucket, {}).get(entity_id, {}).get("variants", [])
                    if entity_id and pool_bucket and isinstance(reference_pools, dict)
                    else []
                )
                if ref_pool:
                    selected = valid_entities[0]
                else:
                    candidate_indices = self._reference_variant_candidates(len(valid_entities))
                    selected = valid_entities[candidate_indices[0] if candidate_indices else 0]
            else:
                selected = random.choice(valid_entities)
            if isinstance(selected, dict):
                entity_dict = copy.deepcopy(selected)
                if entity_type in CANONICAL_ORGANIZATION_TYPES:
                    entity_dict = normalize_organization_pool_entry(entity_dict, expected_entity_type=entity_type)
                entity_class = ENTITY_TYPE_TO_CLASS[entity_type]
                sampled_entity = entity_class(**entity_dict)
            else:
                sampled_entity = selected.model_copy(deep=True)
        if entity_type == "person":
            for attr in (
                "gender",
                "subj_pronoun",
                "obj_pronoun",
                "poss_det_pronoun",
                "poss_pro_pronoun",
                "refl_pronoun",
                "relationship",
                "relationships",
            ):
                setattr(sampled_entity, attr, None if attr != "relationships" else {})
        if entity_type == "person" and "age" in required_attrs:
            min_age, max_age = self._get_age_bounds_from_rules()
            low, high = self._age_window(entity_id or "person_0", min_age, max_age)
            current_age = getattr(sampled_entity, "age", None)
            try:
                current_age_int = int(current_age) if current_age is not None else None
            except (TypeError, ValueError):
                current_age_int = None
            factual_age = self._factual_person_age(entity_id) if entity_id else None
            if (
                current_age_int is None
                or current_age_int < low
                or current_age_int > high
                or (factual_age is not None and current_age_int == factual_age)
            ):
                sampled_entity.age = (
                    self._sample_person_age(entity_id, min_age, max_age) if entity_id else random.randint(low, high)
                )
        return sampled_entity

    def _valid_pool_entities(
        self,
        entity_type: str,
        required_attrs: list[str],
        used_entities: list[Any] | None = None,
        *,
        entity_id: str | None = None,
    ) -> list[Any]:
        if used_entities is None:
            used_entities = []
        excluded_from_pool = [
            "gender",
            "subj_pronoun",
            "obj_pronoun",
            "poss_det_pronoun",
            "poss_pro_pronoun",
            "refl_pronoun",
            "relationship",
            "relationships",
            "honorific",
        ]
        pool_map = {
            "person": self.entity_pool.get("persons", []),
            "place": self.entity_pool.get("places", []),
            "event": self.entity_pool.get("events", []),
            "award": self.entity_pool.get("awards", []),
            "legal": self.entity_pool.get("legals", []),
            "product": self.entity_pool.get("products", []),
        }
        reference_pools = self.entity_pool.get("_reference_pools", {}) if isinstance(self.entity_pool, dict) else {}
        if entity_type in CANONICAL_ORGANIZATION_TYPES:
            pool_bucket = organization_pool_bucket(entity_type) or "organizations"
            ref_pool = (
                reference_pools.get(pool_bucket, {}).get(entity_id, {}).get("variants", [])
                if entity_id and isinstance(reference_pools, dict)
                else []
            )
            bucket_pool = self.entity_pool.get(pool_bucket, [])
        else:
            pool_bucket = {
                "person": "persons",
                "place": "places",
                "event": "events",
                "award": "awards",
                "legal": "legals",
                "product": "products",
            }.get(entity_type)
            ref_pool = (
                reference_pools.get(pool_bucket, {}).get(entity_id, {}).get("variants", [])
                if entity_id and pool_bucket and isinstance(reference_pools, dict)
                else []
            )
            bucket_pool = pool_map.get(entity_type, [])

        if ref_pool and self.reference_variant_index is not None:
            candidate_indices = self._reference_variant_candidates(len(ref_pool))
            preferred_pool = [ref_pool[index] for index in candidate_indices]
        else:
            preferred_pool = list(ref_pool)

        pool: list[Any] = []
        seen_entry_keys: set[tuple[tuple[str, str], ...]] = set()
        for source_pool in (preferred_pool, list(bucket_pool or [])):
            for entry in source_pool:
                entry_key = self._pool_entry_key(entry)
                if entry_key is not None and entry_key in seen_entry_keys:
                    continue
                if entry_key is not None:
                    seen_entry_keys.add(entry_key)
                pool.append(entry)

        if not pool:
            raise ValueError(f"No entities in pool for type '{entity_type}'")
        if not pool:
            raise ValueError(f"No entities in pool for type '{entity_type}' at variant index {self.reference_variant_index}")

        factual_to_exclude = []
        if self.factual_entities:
            factual_map = {
                "person": self.factual_entities.persons,
                "place": self.factual_entities.places,
                "event": self.factual_entities.events,
                "award": self.factual_entities.awards,
                "legal": self.factual_entities.legals,
                "product": self.factual_entities.products,
            }
            factual_source = (
                self.factual_entities.organizations
                if entity_type in CANONICAL_ORGANIZATION_TYPES
                else factual_map.get(entity_type) or {}
            )
            factual_to_exclude = list(factual_source.values())

        pool_required = [attr for attr in required_attrs if attr not in {"age", *excluded_from_pool}]
        event_family = self._factual_event_family(entity_id) if entity_type == "event" else None
        cache_key = (
            entity_type,
            tuple(sorted(pool_required)),
            event_family,
            entity_id if ref_pool else None,
            self.reference_variant_index if ref_pool else None,
        )
        if cache_key in self._eligible_cache:
            eligible = self._eligible_cache[cache_key]
        else:
            eligible = []
            for entity in pool:

                def get_attr(ent: Any, attr: str) -> Any:
                    if entity_type in CANONICAL_ORGANIZATION_TYPES:
                        return organization_attribute_value(ent, attr)
                    return ent.get(attr) if isinstance(ent, dict) else getattr(ent, attr, None)

                has_all = all(get_attr(entity, attr) is not None for attr in pool_required)
                if not has_all:
                    continue
                if entity_type == "event" and event_family is not None:
                    candidate_family = self._candidate_event_family(entity)
                    if candidate_family is not None and candidate_family != event_family:
                        continue
                eligible.append(entity)
            self._eligible_cache[cache_key] = eligible

        valid_entities = []
        for entity in eligible:
            is_used = any(self._entities_equal(entity, used_entity, pool_required) for used_entity in used_entities)
            is_factual = any(
                self._entities_equal(entity, factual_entity, self._get_key_attributes(entity_type, required_attrs))
                for factual_entity in factual_to_exclude
            )
            if not is_used and not is_factual:
                valid_entities.append(entity)
        return valid_entities

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
                    reference_pools = self.entity_pool.get("_reference_pools", {}) if isinstance(self.entity_pool, dict) else {}
                    bucket_refs = reference_pools.get(pool_bucket, {}) if pool_bucket and isinstance(reference_pools, dict) else {}
                    has_reference_coverage = bool(bucket_refs) and all(entity_id in bucket_refs for entity_id in entity_ids)
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
                    reference_pools = self.entity_pool.get("_reference_pools", {}) if isinstance(self.entity_pool, dict) else {}
                    bucket_refs = reference_pools.get(pool_bucket, {}) if pool_bucket and isinstance(reference_pools, dict) else {}
                    has_reference_coverage = bool(bucket_refs) and all(entity_id in bucket_refs for entity_id in entity_ids)
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


__all__ = ["FictionalEntitySamplerNamedMixin"]
