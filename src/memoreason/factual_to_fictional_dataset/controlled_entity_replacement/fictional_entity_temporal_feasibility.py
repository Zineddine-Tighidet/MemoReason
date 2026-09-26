"""Numerical and temporal generation stage for fictional sampling."""

from __future__ import annotations

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection

from .generation_limits import (
    _DEFAULT_NUMBER_MIN,
)


class TemporalNumberFeasibilityMixin:
    """Joint temporal and numerical feasibility checks."""

    def _ensure_temporal_number_joint_feasibility(
        self,
        *,
        generator,
        collection: EntityCollection,
        required_temporals: list[tuple[str, list[str]]],
        temporal_rules: list[str],
        supporting_numeric_rules: list[str],
        required_attr_map: dict[str, set[str]],
        avoid_numbers: dict[str, set[int | float]],
        decade_year_temporal_ids: set[str],
    ) -> None:
        mixed_number_ids = sorted(
            {
                ref.split(".", 1)[0]
                for raw_rule in temporal_rules
                for ref in find_entity_refs(str(raw_rule))
                if ref.startswith("number_")
            }
        )
        if not mixed_number_ids:
            return
        required_mixed_number_ids = [number_id for number_id in mixed_number_ids if number_id in required_attr_map]

        def candidate_domain(number_id: str, *, expansion_padding: int = 0) -> list[int]:
            base_low, base_high = generator._number_base_range(number_id)
            implicit_bounds = generator._implicit_number_range(number_id)
            if implicit_bounds is not None:
                base_low = max(base_low, implicit_bounds[0])
                base_high = min(base_high, implicit_bounds[1])
            if supporting_numeric_rules:
                support_low, support_high = generator._get_number_range_from_rules(
                    number_id,
                    supporting_numeric_rules,
                    pre_assigned={},
                    existing_entities=collection,
                )
                base_low = max(base_low, support_low)
                if expansion_padding <= 0:
                    base_high = min(base_high, support_high)
            if expansion_padding > 0:
                base_low = max(_DEFAULT_NUMBER_MIN, int(base_low) - expansion_padding)
                base_high = int(base_high) + expansion_padding
            if base_low > base_high:
                return []
            forbidden = {int(value) for value in avoid_numbers.get(number_id, set()) if isinstance(value, (int, float))}
            base_low, base_high = generator._expand_int_domain_to_escape_forbidden(
                int(base_low),
                int(base_high),
                avoid=forbidden,
            )
            return [value for value in range(int(base_low), int(base_high) + 1) if value not in forbidden]

        def current_int(number_id: str) -> int | None:
            entity = collection.numbers.get(number_id)
            if entity is None:
                return None
            raw = getattr(entity, "int", None)
            if raw is None:
                return None
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None

        def assign_number(number_id: str, value: int) -> None:
            collection.numbers[number_id] = generator._build_number_entity(
                number_id,
                value,
                required_attrs=required_attr_map.get(number_id),
            )

        def partial_product_rules_ok(assigned_ids: set[str]) -> bool:
            for raw_rule in temporal_rules:
                cleaned = str(raw_rule or "").split("#", 1)[0].strip()
                match = self._TEMPORAL_PRODUCT_RULE_PATTERN.fullmatch(cleaned)
                if match is None:
                    continue
                source_number_id = match.group("left_number") or match.group("right_number")
                target_number_id = match.group("target_number")
                rule_number_ids = {source_number_id, target_number_id}
                resolved_ids = set(assigned_ids)
                resolved_ids.update(
                    number_id
                    for number_id in rule_number_ids
                    if number_id not in required_attr_map and generator._factual_number_int(number_id) is not None
                )
                if not rule_number_ids.issubset(resolved_ids):
                    continue
                source_value = supporting_rule_int(source_number_id)
                target_value = supporting_rule_int(target_number_id)
                if source_value is None or target_value is None:
                    continue
                if source_value <= 0 or target_value % source_value != 0:
                    return False
            return True

        original_numbers = {
            number_id: (
                collection.numbers[number_id].model_copy(deep=True) if number_id in collection.numbers else None
            )
            for number_id in mixed_number_ids
        }

        excluded_years = set(generator.exclude_temporals.get("years", set()))
        numeric_constraint_vars = {
            ref.split(".", 1)[0]
            for raw_rule in supporting_numeric_rules
            for ref in find_entity_refs(str(raw_rule))
            if ref.startswith("number_")
        }
        compiled_numeric_constraints = (
            generator._collect_linear_constraints(
                supporting_numeric_rules,
                numeric_constraint_vars,
                collection,
            )
            if supporting_numeric_rules and numeric_constraint_vars
            else None
        )

        def supporting_rule_int(number_id: str) -> int | None:
            if number_id not in required_attr_map:
                factual_value = generator._factual_number_int(number_id)
                if factual_value is not None:
                    return factual_value
            else:
                # A required number must have been generated in this attempt.
                # Falling back here would hide an incomplete active assignment.
                return current_int(number_id)
            return current_int(number_id)

        def collection_with_retained_factual_numbers(number_ids: set[str]) -> EntityCollection:
            retained_ids = {number_id for number_id in number_ids if number_id not in required_attr_map}
            if not retained_ids:
                return collection
            factual_numbers = getattr(getattr(generator, "factual_entities", None), "numbers", {}) or {}
            merged = collection.model_copy(deep=True)
            for number_id in sorted(retained_ids):
                factual_number = factual_numbers.get(number_id)
                if factual_number is not None:
                    merged.numbers[number_id] = factual_number.model_copy(deep=True)
            return merged

        def supporting_rule_collection() -> EntityCollection:
            return collection_with_retained_factual_numbers(numeric_constraint_vars)

        def restore_originals() -> None:
            for number_id, original in original_numbers.items():
                if original is None:
                    collection.numbers.pop(number_id, None)
                else:
                    collection.numbers[number_id] = original.model_copy(deep=True)

        def build_candidate_values(expansion_padding: int) -> dict[str, list[int]] | None:
            candidate_values: dict[str, list[int]] = {}
            total_search_space = 1
            for number_id in required_mixed_number_ids:
                domain = candidate_domain(number_id, expansion_padding=expansion_padding)
                if not domain:
                    return None
                current_value = current_int(number_id)
                factual_value = generator._factual_number_int(number_id)
                ordered_domain = sorted(
                    domain,
                    key=lambda candidate: (
                        candidate == factual_value,
                        abs(candidate - (factual_value if factual_value is not None else candidate)),
                        abs(candidate - (current_value if current_value is not None else candidate)),
                        candidate,
                    ),
                )
                candidate_values[number_id] = ordered_domain
                total_search_space *= len(ordered_domain)

            if total_search_space > 4096 or len(required_temporals) > 8:
                # Keep the search bounded, but less aggressively after expansion:
                # mixed temporal-number systems sometimes need the nearest value
                # just outside the normal relative window to satisfy all rules.
                keep = 6 if expansion_padding <= 0 else 12
                candidate_values = {number_id: values[:keep] for number_id, values in candidate_values.items()}
            return candidate_values

        def supporting_numeric_rules_ok() -> bool:
            if compiled_numeric_constraints is not None:
                assignments: dict[str, int] = {}
                domains: dict[str, tuple[int, int]] = {}
                for number_id in numeric_constraint_vars:
                    value = supporting_rule_int(number_id)
                    if value is None:
                        return False
                    assignments[number_id] = int(value)
                    domains[number_id] = (int(value), int(value))
                return generator._constraints_feasible(compiled_numeric_constraints, assignments, domains)
            return all(
                is_valid
                for _, is_valid in RuleEngine.validate_all_rules(
                    supporting_numeric_rules,
                    supporting_rule_collection(),
                )
            )

        temporal_feasibility_cache: dict[tuple[tuple[str, ...], tuple[tuple[str, int | None], ...]], bool] = {}

        def temporal_rules_admit_solution() -> bool:
            temporal_rule_collection = collection_with_retained_factual_numbers(set(mixed_number_ids))
            concretized_temporal_rules = self._concretize_temporal_rules_with_numbers(
                generator=generator,
                temporal_rules=temporal_rules,
                collection=temporal_rule_collection,
            )
            unresolved_number_ids = sorted(
                {
                    ref.split(".", 1)[0]
                    for rule in concretized_temporal_rules
                    for ref in find_entity_refs(rule)
                    if ref.startswith("number_")
                }
            )
            cache_key = (
                tuple(concretized_temporal_rules),
                tuple((number_id, supporting_rule_int(number_id)) for number_id in unresolved_number_ids),
            )
            cached = temporal_feasibility_cache.get(cache_key)
            if cached is not None:
                return cached
            solved_years = generator._solve_temporal_years(
                required_temporals,
                concretized_temporal_rules,
                temporal_rule_collection,
                excluded_years,
                decade_year_temporal_ids,
            )
            temporal_feasibility_cache[cache_key] = solved_years is not None
            return temporal_feasibility_cache[cache_key]

        # Fast path: if the exact-solved numbers already satisfy the supporting numeric
        # constraints and admit a temporal-year solution, there is nothing to search.
        if supporting_numeric_rules_ok() and temporal_rules_admit_solution():
            return

        def search(index: int, assigned_ids: set[str], candidate_values: dict[str, list[int]]) -> bool:
            if index >= len(required_mixed_number_ids):
                if not supporting_numeric_rules_ok():
                    return False
                return temporal_rules_admit_solution()

            number_id = required_mixed_number_ids[index]
            original_number = collection.numbers.get(number_id)
            for candidate in candidate_values[number_id]:
                assign_number(number_id, candidate)
                next_assigned_ids = set(assigned_ids)
                next_assigned_ids.add(number_id)
                if not partial_product_rules_ok(next_assigned_ids):
                    continue
                if search(index + 1, next_assigned_ids, candidate_values):
                    return True
            if original_number is not None:
                collection.numbers[number_id] = original_number
            else:
                collection.numbers.pop(number_id, None)
            return False

        for expansion_padding in (0, 8, 16, 32):
            restore_originals()
            candidate_values = build_candidate_values(expansion_padding)
            if candidate_values is None:
                continue
            if search(0, set(), candidate_values):
                return
        restore_originals()


__all__ = ["TemporalNumberFeasibilityMixin"]
