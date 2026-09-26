"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

import math
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.century_expressions import has_century_function
from memoreason.benchmark_definition.document_schema import EntityCollection

from .fictional_entity_mixed_temporal_rule_constraint_enforcement import (
    _affine_two_year_candidate_pairs,
)
from .sampling_checks import (
    _values_equal,
)


class SingleDifferenceConstraintEnforcementMixin:
    """Enforce one required numerical or temporal difference."""

    def _repair_single_required_number_difference(
        self,
        *,
        generator,
        collection: EntityCollection,
        number_id: str,
        attr: str,
        factual_value: Any,
        required_attrs: set[str],
        rules_with_ordering: list[str],
        ordering_exempt_ids: set[str],
        allowed_number_reuse_ids: set[str] | None = None,
    ) -> bool:
        number_entity = collection.numbers.get(number_id)
        if number_entity is None:
            return False
        relevant_number_rules = self._rules_referencing_entity_ids(rules_with_ordering, {number_id})
        original_snapshot = number_entity.model_copy(deep=True)
        original_temporals = {
            temporal_id: temporal.model_copy(deep=True) for temporal_id, temporal in collection.temporals.items()
        }
        forced_number_refs = generator._forced_equal_refs_for_number(number_id)
        forced_number_refs_on_entry = set(generator.last_number_forced_equal_refs) & forced_number_refs

        def clear_stale_forced_equal_marker() -> None:
            generator.last_number_forced_equal_refs.difference_update(forced_number_refs)

        def restore_entry_forced_equal_marker() -> None:
            clear_stale_forced_equal_marker()
            generator.last_number_forced_equal_refs.update(forced_number_refs_on_entry)

        low, high = generator._number_actual_bounds(number_id, required_attrs)
        used_values = {
            int(value)
            for value in self.used_number_values_by_id.get(number_id, set())
            if isinstance(value, (int, float))
        }
        allow_previous_value_reuse = number_id in set(allowed_number_reuse_ids or set())
        candidates = [
            candidate
            for candidate in range(int(math.ceil(low)), int(math.floor(high)) + 1)  # noqa: RUF046
            if not _values_equal(factual_value, candidate)
            and (allow_previous_value_reuse or candidate not in used_values)
        ]
        current_value = getattr(number_entity, "int", None)
        candidates.sort(
            key=lambda candidate: (
                candidate in used_values,
                abs(candidate - int(current_value)) if current_value is not None else 0,
                abs(candidate),
            )
        )
        for candidate in candidates:
            generator._set_number_actual_value(
                number_id,
                number_entity,
                float(candidate),
                required_attrs=required_attrs,
            )
            fictional_value = RuleEngine._get_entity_value(collection, f"{number_id}.{attr}")
            if _values_equal(factual_value, fictional_value):
                collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                number_entity = collection.numbers[number_id]
                continue
            validation_result = self._validate_rules_with_details(relevant_number_rules, collection)
            if validation_result["all_valid"] and self._verify_ordering_preserved(
                collection,
                forced_equal_entity_ids=ordering_exempt_ids,
            ):
                clear_stale_forced_equal_marker()
                return True
            collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
            number_entity = collection.numbers[number_id]
            collection.temporals = {
                temporal_id: temporal.model_copy(deep=True) for temporal_id, temporal in original_temporals.items()
            }
        mixed_rules = [
            str(rule)
            for rule in rules_with_ordering
            if number_id in str(rule) and "temporal_" in str(rule) and not has_century_function(str(rule))
        ]
        if not mixed_rules:
            restore_entry_forced_equal_marker()
            return False

        excluded_years = set(generator.exclude_temporals.get("years", set()))
        temporal_domains_cache: dict[str, list[int]] = {}

        def temporal_year_domain(temporal_id: str) -> list[int]:
            cached = temporal_domains_cache.get(temporal_id)
            if cached is not None:
                return cached
            domain = self._order_preserving_temporal_year_candidates(
                generator=generator,
                collection=collection,
                temporal_id=temporal_id,
                excluded_years=excluded_years,
                decade_year_temporal_ids=set(),
            )
            temporal_domains_cache[temporal_id] = list(domain)
            return temporal_domains_cache[temporal_id]

        for candidate in candidates:
            generator._set_number_actual_value(
                number_id,
                number_entity,
                float(candidate),
                required_attrs=required_attrs,
            )
            fictional_value = RuleEngine._get_entity_value(collection, f"{number_id}.{attr}")
            if _values_equal(factual_value, fictional_value):
                collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                number_entity = collection.numbers[number_id]
                continue

            repaired = True
            for mixed_rule in mixed_rules:
                cleaned = str(mixed_rule or "").split("#", 1)[0].strip()
                temporal_ids = list(
                    dict.fromkeys(
                        ref.split(".", 1)[0]
                        for ref in find_entity_refs(cleaned)
                        if ref.startswith("temporal_") and ref.endswith(".year")
                    )
                )
                if len(temporal_ids) != 2:
                    repaired = False
                    break
                left_id, right_id = temporal_ids
                left_original = original_temporals.get(left_id)
                right_original = original_temporals.get(right_id)
                if left_original is None or right_original is None:
                    repaired = False
                    break
                pair_ignore_ids = {left_id, right_id}
                left_domain = self._order_preserving_temporal_year_candidates(
                    generator=generator,
                    collection=collection,
                    temporal_id=left_id,
                    excluded_years=excluded_years,
                    decade_year_temporal_ids=set(),
                    ignore_temporal_ids=pair_ignore_ids,
                )
                right_domain = self._order_preserving_temporal_year_candidates(
                    generator=generator,
                    collection=collection,
                    temporal_id=right_id,
                    excluded_years=excluded_years,
                    decade_year_temporal_ids=set(),
                    ignore_temporal_ids=pair_ignore_ids,
                )
                left_current = getattr(left_original, "year", None)
                right_current = getattr(right_original, "year", None)
                candidate_pairs = _affine_two_year_candidate_pairs(
                    generator=generator,
                    rule=cleaned,
                    collection=collection,
                    left_id=left_id,
                    right_id=right_id,
                    left_domain=left_domain,
                    right_domain=right_domain,
                )
                if candidate_pairs is None:
                    # This fast repair is deliberately exact and bounded.  A
                    # non-affine rule is left to the outer retry/fallback
                    # rather than materialising an unbounded Cartesian product.
                    repaired = False
                    break
                candidate_pairs = sorted(
                    candidate_pairs,
                    key=lambda pair: (
                        abs(pair[0] - (left_current if left_current is not None else pair[0]))
                        + abs(pair[1] - (right_current if right_current is not None else pair[1])),
                        abs(pair[1] - pair[0]),
                        pair[0],
                        pair[1],
                    ),
                )
                pair_found = False
                pair_relevant_rules = self._rules_referencing_entity_ids(
                    rules_with_ordering,
                    {number_id, left_id, right_id},
                )
                for left_year, right_year in candidate_pairs:
                    collection.temporals[left_id] = generator._update_temporal_year(left_original, int(left_year))
                    collection.temporals[right_id] = generator._update_temporal_year(right_original, int(right_year))
                    if not RuleEngine.evaluate_expression(cleaned, collection):
                        continue
                    validation_result = self._validate_rules_with_details(pair_relevant_rules, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        pair_found = True
                        break
                if not pair_found:
                    repaired = False
                    break
            if repaired:
                clear_stale_forced_equal_marker()
                return True
            collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
            number_entity = collection.numbers[number_id]
            collection.temporals = {
                temporal_id: temporal.model_copy(deep=True) for temporal_id, temporal in original_temporals.items()
            }

        temporal_rules = [
            str(rule)
            for rule in rules_with_ordering
            if "temporal_" in str(rule) and not has_century_function(str(rule))
        ]
        temporal_specs = sorted(
            (
                temporal_id,
                ["year"],
            )
            for temporal_id, temporal in collection.temporals.items()
            if getattr(temporal, "year", None) is not None
        )
        if temporal_rules and temporal_specs:
            existing_without_temporals = EntityCollection(
                persons={key: value.model_copy(deep=True) for key, value in collection.persons.items()},
                places={key: value.model_copy(deep=True) for key, value in collection.places.items()},
                events={key: value.model_copy(deep=True) for key, value in collection.events.items()},
                organizations={key: value.model_copy(deep=True) for key, value in collection.organizations.items()},
                awards={key: value.model_copy(deep=True) for key, value in collection.awards.items()},
                legals={key: value.model_copy(deep=True) for key, value in collection.legals.items()},
                products={key: value.model_copy(deep=True) for key, value in collection.products.items()},
                numbers={key: value.model_copy(deep=True) for key, value in collection.numbers.items()},
                temporals={},
            )
            for candidate in candidates:
                generator._set_number_actual_value(
                    number_id,
                    number_entity,
                    float(candidate),
                    required_attrs=required_attrs,
                )
                existing_without_temporals.numbers[number_id] = collection.numbers[number_id].model_copy(deep=True)
                fictional_value = RuleEngine._get_entity_value(collection, f"{number_id}.{attr}")
                if _values_equal(factual_value, fictional_value):
                    collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                    number_entity = collection.numbers[number_id]
                    existing_without_temporals.numbers[number_id] = number_entity.model_copy(deep=True)
                    continue
                try:
                    regenerated_temporals = generator.generate_temporals_with_rules(
                        temporal_specs,
                        self._concretize_temporal_rules_with_numbers(
                            generator=generator,
                            temporal_rules=temporal_rules,
                            collection=existing_without_temporals,
                        ),
                        existing_without_temporals,
                    )
                except Exception:
                    regenerated_temporals = None
                if regenerated_temporals:
                    collection.temporals = {
                        temporal_id: temporal.model_copy(deep=True)
                        for temporal_id, temporal in regenerated_temporals.items()
                    }
                    validation_result = self._validate_rules_with_details(rules_with_ordering, collection)
                    if validation_result["all_valid"] and self._verify_ordering_preserved(
                        collection,
                        forced_equal_entity_ids=ordering_exempt_ids,
                    ):
                        clear_stale_forced_equal_marker()
                        return True
                collection.numbers[number_id] = original_snapshot.model_copy(deep=True)
                number_entity = collection.numbers[number_id]
                existing_without_temporals.numbers[number_id] = number_entity.model_copy(deep=True)
                collection.temporals = {
                    temporal_id: temporal.model_copy(deep=True) for temporal_id, temporal in original_temporals.items()
                }
        restore_entry_forced_equal_marker()
        return False

    def _repair_single_required_temporal_year_difference(
        self,
        *,
        generator,
        collection: EntityCollection,
        temporal_id: str,
        factual_value: Any,
        rules_with_ordering: list[str],
        ordering_exempt_ids: set[str],
    ) -> bool:
        temporal_entity = collection.temporals.get(temporal_id)
        if temporal_entity is None or getattr(temporal_entity, "year", None) is None:
            return False
        relevant_temporal_rules = self._rules_referencing_entity_ids(rules_with_ordering, {temporal_id})
        original_snapshot = temporal_entity.model_copy(deep=True)
        excluded_years = set(generator.exclude_temporals.get("years", set()))
        candidates = self._order_preserving_temporal_year_candidates(
            generator=generator,
            collection=collection,
            temporal_id=temporal_id,
            excluded_years=excluded_years,
            decade_year_temporal_ids=set(),
        )
        current_value = getattr(temporal_entity, "year", None)
        candidates = [candidate for candidate in candidates if not _values_equal(factual_value, candidate)]
        candidates.sort(
            key=lambda candidate: (
                abs(candidate - int(current_value)) if current_value is not None else 0,
                abs(candidate),
            )
        )
        for candidate in candidates:
            collection.temporals[temporal_id] = generator._update_temporal_year(original_snapshot, int(candidate))
            fictional_value = RuleEngine._get_entity_value(collection, f"{temporal_id}.year")
            if _values_equal(factual_value, fictional_value):
                collection.temporals[temporal_id] = original_snapshot.model_copy(deep=True)
                continue
            validation_result = self._validate_rules_with_details(relevant_temporal_rules, collection)
            if validation_result["all_valid"] and self._verify_ordering_preserved(
                collection,
                forced_equal_entity_ids=ordering_exempt_ids,
            ):
                return True
            collection.temporals[temporal_id] = original_snapshot.model_copy(deep=True)
        return False


__all__ = ["SingleDifferenceConstraintEnforcementMixin"]
