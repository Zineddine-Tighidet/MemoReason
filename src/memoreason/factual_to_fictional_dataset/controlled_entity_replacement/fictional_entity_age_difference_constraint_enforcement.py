"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

import math
import random
import re
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine
from memoreason.benchmark_definition.document_schema import EntityCollection

from .fictional_entity_sampler_common import logger
from .generation_limits import (
    _DEFAULT_UNCONSTRAINED_MAX_AGE,
    _DEFAULT_UNCONSTRAINED_MIN_AGE,
    _MAX_AGE,
    _MIN_AGE,
    _relative_int_window,
)
from .sampling_checks import (
    _attr_requires_difference,
    _values_equal,
    merge_factual_entities,
    preserve_age_ordering,
    resample_equal_person_ages,
    validate_rules_with_details,
    verify_ordering_preserved,
    verify_required_differences,
)


class AgeDifferenceConstraintEnforcementMixin:
    """Enforce age bounds and required factual-to-fictional differences."""

    def _get_age_bounds_from_rules(self) -> tuple[int, int]:
        min_age, max_age = _MIN_AGE, _MAX_AGE
        if not getattr(self, "_current_rules", None):
            return _DEFAULT_UNCONSTRAINED_MIN_AGE, _DEFAULT_UNCONSTRAINED_MAX_AGE
        found_bound = False
        for rule in self._current_rules:
            match = re.search(r"person_\d+(?:\.age)?\s*>\s*(\d+)", rule.strip())
            if match:
                min_age = max(min_age, int(match.group(1)) + 1)
                found_bound = True
            match = re.search(r"(\d+)\s*<\s*person_\d+(?:\.age)?", rule.strip())
            if match:
                min_age = max(min_age, int(match.group(1)) + 1)
                found_bound = True
            match = re.search(r"person_\d+(?:\.age)?\s*<\s*(\d+)", rule.strip())
            if match:
                max_age = min(max_age, int(match.group(1)) - 1)
                found_bound = True
            match = re.search(r"(\d+)\s*>\s*person_\d+(?:\.age)?", rule.strip())
            if match:
                max_age = min(max_age, int(match.group(1)) - 1)
                found_bound = True
        if not found_bound:
            min_age = max(min_age, _DEFAULT_UNCONSTRAINED_MIN_AGE)
            max_age = min(max_age, _DEFAULT_UNCONSTRAINED_MAX_AGE)
        if min_age > max_age:
            raise ValueError(f"Age constraints contradictory: min_age={min_age}, max_age={max_age}")
        return min_age, max_age

    def _factual_person_age(self, person_id: str) -> int | None:
        if not self.factual_entities or not self.factual_entities.persons:
            return None
        factual_person = self.factual_entities.persons.get(person_id)
        if factual_person is None:
            return None
        raw_age = (
            getattr(factual_person, "age", None) if not isinstance(factual_person, dict) else factual_person.get("age")
        )
        if raw_age is None:
            return None
        try:
            return int(raw_age)
        except (TypeError, ValueError):
            return None

    def _age_window(self, person_id: str, min_age: int, max_age: int) -> tuple[int, int]:
        implicit_age_range = getattr(self, "_implicit_age_range", None)
        if callable(implicit_age_range):
            overridden = implicit_age_range(person_id, min_age=min_age, max_age=max_age)
            if overridden is not None:
                return overridden
        factual_age = self._factual_person_age(person_id)
        if factual_age is None:
            return min_age, max_age
        min_age = min(min_age, factual_age)
        max_age = max(max_age, factual_age)
        low, high = _relative_int_window(factual_age, min_value=min_age, max_value=max_age)
        return low, high

    def _implicit_age_range(
        self,
        person_id: str,
        *,
        min_age: int,
        max_age: int,
    ) -> tuple[int, int] | None:
        rule = self._implicit_age_rules.get(person_id)
        if rule is None:
            return None
        low = max(min_age, math.ceil(float(rule.lower_bound)))
        high = min(max_age, math.floor(float(rule.upper_bound)))
        if low > high:
            factual = round(float(rule.factual_value))
            factual = max(min_age, min(factual, max_age))
            return factual, factual
        return low, high

    def _sample_person_age(self, person_id: str, min_age: int, max_age: int) -> int:
        low, high = self._age_window(person_id, min_age, max_age)
        factual_age = self._factual_person_age(person_id)
        if factual_age is None:
            return random.randint(low, high)
        candidates = [age for age in range(low, high + 1) if age != factual_age]
        if candidates:
            return random.choice(candidates)
        return random.randint(low, high)

    def _merge_factual_entities(self, collection: EntityCollection) -> None:
        merge_factual_entities(collection, self.factual_entities)

    def _validate_rules_with_details(self, rules: list[str], entities: EntityCollection) -> dict:
        return validate_rules_with_details(rules, entities, logger)

    def _verify_ordering_preserved(
        self,
        entities: EntityCollection,
        *,
        preserve_temporal_ordering: bool = True,
        preserve_number_ordering: bool = False,
        forced_equal_entity_ids: set[str] | None = None,
    ) -> bool:
        return verify_ordering_preserved(
            entities,
            self.factual_entities,
            preserve_temporal_ordering=preserve_temporal_ordering,
            preserve_number_ordering=preserve_number_ordering,
            forced_equal_entity_ids=forced_equal_entity_ids,
        )

    def _verify_required_differences(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        sampled_entities: EntityCollection,
        forced_equal_entity_ids: set[str] | None = None,
        forced_equal_entity_refs: set[str] | None = None,
    ) -> bool:
        return verify_required_differences(
            required_entities=required_entities,
            sampled_entities=sampled_entities,
            factual_entities=self.factual_entities,
            person_diff_exempt_attrs=self._PERSON_DIFF_EXEMPT_ATTRS,
            forced_equal_entity_ids=forced_equal_entity_ids,
            forced_equal_entity_refs=forced_equal_entity_refs,
        )

    def _resample_equal_person_ages(
        self,
        collection: EntityCollection,
        required_entities: dict[str, list[tuple[str, list[str]]]],
    ) -> None:
        resample_equal_person_ages(
            collection=collection,
            required_entities=required_entities,
            factual_entities=self.factual_entities,
            age_bounds_getter=self._get_age_bounds_from_rules,
            age_window_getter=self._age_window,
            sample_person_age=self._sample_person_age,
        )

    def _preserve_age_ordering(self, collection: EntityCollection) -> None:
        preserve_age_ordering(collection, self.factual_entities)

    def _collect_unchanged_required_refs(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        collection: EntityCollection,
        *,
        forced_equal_entity_ids: set[str] | None = None,
        forced_equal_entity_refs: set[str] | None = None,
    ) -> list[tuple[str, str, str, Any]]:
        unchanged: list[tuple[str, str, str, Any]] = []
        if not self.factual_entities:
            return unchanged
        forced_equal_entity_ids = set(forced_equal_entity_ids or set())
        forced_equal_entity_refs = set(forced_equal_entity_refs or set())
        for entity_type, specs in required_entities.items():
            for entity_id, attrs in specs:
                if entity_id in forced_equal_entity_ids:
                    continue
                for attr in attrs:
                    entity_ref = f"{entity_id}.{attr}"
                    if entity_ref in forced_equal_entity_refs:
                        continue
                    if not _attr_requires_difference(entity_type, attr, self._PERSON_DIFF_EXEMPT_ATTRS):
                        continue
                    factual_value = RuleEngine._get_entity_value(self.factual_entities, entity_ref)
                    fictional_value = RuleEngine._get_entity_value(collection, entity_ref)
                    if factual_value is None or fictional_value is None:
                        continue
                    if _values_equal(factual_value, fictional_value):
                        unchanged.append((entity_type, entity_id, attr, factual_value))
        return unchanged

    def _repair_required_difference_violations(
        self,
        *,
        generator,
        collection: EntityCollection,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        rules_with_ordering: list[str],
        ordering_exempt_ids: set[str],
        fixed_required_refs: set[str],
        allowed_number_reuse_ids: set[str] | None = None,
    ) -> None:
        allowed_number_reuse_ids = set(allowed_number_reuse_ids or set())
        number_required_attr_map = {
            entity_id: set(attrs or []) for entity_id, attrs in required_entities.get("number", [])
        }
        for _ in range(max(1, len(number_required_attr_map))):
            unchanged = self._collect_unchanged_required_refs(
                required_entities,
                collection,
                forced_equal_entity_ids=ordering_exempt_ids,
                forced_equal_entity_refs=fixed_required_refs,
            )
            if not unchanged:
                return
            progress = False
            for entity_type, entity_id, attr, factual_value in unchanged:
                if entity_type == "number":
                    progress = (
                        self._repair_single_required_number_difference(
                            generator=generator,
                            collection=collection,
                            number_id=entity_id,
                            attr=attr,
                            factual_value=factual_value,
                            required_attrs=number_required_attr_map.get(entity_id, set()),
                            rules_with_ordering=rules_with_ordering,
                            ordering_exempt_ids=ordering_exempt_ids,
                            allowed_number_reuse_ids=allowed_number_reuse_ids,
                        )
                        or progress
                    )
                elif entity_type == "temporal" and attr == "year":
                    progress = (
                        self._repair_single_required_temporal_year_difference(
                            generator=generator,
                            collection=collection,
                            temporal_id=entity_id,
                            factual_value=factual_value,
                            rules_with_ordering=rules_with_ordering,
                            ordering_exempt_ids=ordering_exempt_ids,
                        )
                        or progress
                    )
            if not progress:
                self._force_simple_required_differences(
                    generator=generator,
                    collection=collection,
                    unchanged=unchanged,
                    rules_with_ordering=rules_with_ordering,
                    number_required_attr_map=number_required_attr_map,
                    ordering_exempt_ids=ordering_exempt_ids,
                    allowed_number_reuse_ids=allowed_number_reuse_ids,
                )
                return
        remaining_unchanged = self._collect_unchanged_required_refs(
            required_entities,
            collection,
            forced_equal_entity_ids=ordering_exempt_ids,
            forced_equal_entity_refs=fixed_required_refs,
        )
        if remaining_unchanged:
            self._force_simple_required_differences(
                generator=generator,
                collection=collection,
                unchanged=remaining_unchanged,
                rules_with_ordering=rules_with_ordering,
                number_required_attr_map=number_required_attr_map,
                ordering_exempt_ids=ordering_exempt_ids,
                allowed_number_reuse_ids=allowed_number_reuse_ids,
            )

    def _force_simple_required_differences(
        self,
        *,
        generator,
        collection: EntityCollection,
        unchanged: list[tuple[str, str, str, Any]],
        rules_with_ordering: list[str],
        number_required_attr_map: dict[str, set[str]],
        ordering_exempt_ids: set[str],
        allowed_number_reuse_ids: set[str] | None = None,
    ) -> None:
        allowed_number_reuse_ids = set(allowed_number_reuse_ids or set())
        for entity_type, entity_id, attr, factual_value in unchanged:
            if entity_type == "number":
                number_entity = collection.numbers.get(entity_id)
                if number_entity is None:
                    continue
                relevant_rules = self._rules_referencing_entity_ids(rules_with_ordering, {entity_id})
                original_snapshot = number_entity.model_copy(deep=True)
                required_attrs = number_required_attr_map.get(entity_id, set())
                low, high = generator._number_actual_bounds(entity_id, required_attrs)
                candidates = [
                    candidate
                    for candidate in range(int(math.ceil(low)), int(math.floor(high)) + 1)  # noqa: RUF046
                    if not _values_equal(factual_value, candidate)
                ]
                current_value = getattr(number_entity, "int", None)
                used_values = {
                    int(value)
                    for value in self.used_number_values_by_id.get(entity_id, set())
                    if isinstance(value, (int, float))
                }
                allow_previous_value_reuse = entity_id in allowed_number_reuse_ids
                candidates = [
                    candidate for candidate in candidates if allow_previous_value_reuse or candidate not in used_values
                ]
                candidates.sort(
                    key=lambda candidate: (
                        candidate in used_values,
                        abs(candidate - int(current_value)) if current_value is not None else 0,
                        abs(candidate),
                    )
                )
                for candidate in candidates:
                    generator._set_number_actual_value(
                        entity_id,
                        number_entity,
                        float(candidate),
                        required_attrs=required_attrs,
                    )
                    fictional_value = RuleEngine._get_entity_value(collection, f"{entity_id}.{attr}")
                    if _values_equal(factual_value, fictional_value):
                        continue
                    validation_result = self._validate_rules_with_details(relevant_rules, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        break
                else:
                    collection.numbers[entity_id] = original_snapshot
            elif entity_type == "temporal" and attr == "year":
                temporal_entity = collection.temporals.get(entity_id)
                if temporal_entity is None or getattr(temporal_entity, "year", None) is None:
                    continue
                relevant_rules = self._rules_referencing_entity_ids(rules_with_ordering, {entity_id})
                original_snapshot = temporal_entity.model_copy(deep=True)
                excluded_years = set(generator.exclude_temporals.get("years", set()))
                candidates = [
                    candidate
                    for candidate in self._order_preserving_temporal_year_candidates(
                        generator=generator,
                        collection=collection,
                        temporal_id=entity_id,
                        excluded_years=excluded_years,
                        decade_year_temporal_ids=set(),
                    )
                    if not _values_equal(factual_value, candidate)
                ]
                current_value = getattr(temporal_entity, "year", None)
                candidates.sort(
                    key=lambda candidate: (
                        abs(candidate - int(current_value)) if current_value is not None else 0,
                        abs(candidate),
                    )
                )
                for candidate in candidates:
                    collection.temporals[entity_id] = generator._update_temporal_year(original_snapshot, int(candidate))
                    fictional_value = RuleEngine._get_entity_value(collection, f"{entity_id}.year")
                    if _values_equal(factual_value, fictional_value):
                        continue
                    validation_result = self._validate_rules_with_details(relevant_rules, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        break
                else:
                    collection.temporals[entity_id] = original_snapshot


__all__ = ["AgeDifferenceConstraintEnforcementMixin"]
