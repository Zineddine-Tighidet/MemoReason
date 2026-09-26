"""Solver-first workflow for generating numeric entity values."""

import logging

from memoreason.benchmark_definition.annotation_runtime import find_entity_refs
from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity
from ..generation_limits import (
    _DEFAULT_NUMBER_MIN,
    _NUMBER_GENERATION_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class IndependentNumberGenerationMixin:
    """Generate independent numbers when no cross-number constraints apply."""

    def _number_rules_are_independent(self, active_rules: list[str], number_ids: set[str]) -> bool:
        for raw_rule in active_rules:
            refs = {
                ref.split(".", 1)[0]
                for ref in find_entity_refs(str(raw_rule))
                if ref.startswith("number_") and ref.split(".", 1)[0] in number_ids
            }
            if len(refs) > 1:
                return False
        return True

    def _generate_numbers_independently(
        self,
        number_ids: list[str],
        *,
        required_attr_map: dict[str, set[str]],
        active_rules: list[str],
        existing_entities: EntityCollection | None,
        avoid_values: dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]],
        protected_decimal_numbers: set[str],
    ) -> dict[str, NumberEntity] | None:
        for _attempt in range(_NUMBER_GENERATION_MAX_RETRIES):
            self.last_number_forced_equal_refs = set()
            numbers: dict[str, NumberEntity] = {}
            for number_id in number_ids:
                min_val, max_val = self._get_number_range_from_rules(
                    number_id,
                    active_rules,
                    {},
                    existing_entities,
                )
                min_val = max(_DEFAULT_NUMBER_MIN, min_val)
                min_val, max_val = self._expand_int_domain_to_escape_forbidden(
                    min_val,
                    max_val,
                    avoid=avoid_values.get(number_id),
                    min_value=_DEFAULT_NUMBER_MIN,
                )
                if max_val < min_val:
                    numbers = {}
                    break
                sampled_value = self._sample_int_in_range(min_val, max_val, avoid=avoid_values.get(number_id))
                numbers[number_id] = self._build_number_entity(
                    number_id,
                    sampled_value,
                    required_attrs=required_attr_map.get(number_id),
                    allow_non_integer_adjustment=number_id not in protected_decimal_numbers,
                )
            if not numbers:
                continue

            self._preserve_number_ordering(
                numbers,
                required_attr_map=required_attr_map,
                active_rules=active_rules,
            )
            self._repair_number_order_violations(
                numbers,
                required_attr_map=required_attr_map,
                active_rules=active_rules,
                existing_entities=existing_entities,
            )
            self._nudge_numbers_away_from_factual(
                numbers,
                required_attr_map=required_attr_map,
                active_rules=active_rules,
                existing_entities=existing_entities,
                avoid_values=avoid_values,
            )
            try:
                if not self._validate_numbers_against_rules(numbers, active_rules, existing_entities):
                    continue
            except Exception:
                continue
            equal_ids = {
                num_id
                for num_id in number_ids
                if (num_id in avoid_values and self._number_hits_forbidden_value(num_id, numbers[num_id], avoid_values))
            }
            if equal_ids:
                continue
            return numbers
        return None


__all__ = ["IndependentNumberGenerationMixin"]
