"""Post-processing helpers for ordering and factual-divergence nudging."""

import math
from typing import Dict, List, Optional, Set

from src.core.document_schema import EntityCollection, NumberEntity
from src.core.annotation_runtime import find_entity_refs


class NumberSamplingAdjustmentMixin:
    """Repairs and nudges generated numbers after initial assignment."""

    def _preserve_number_ordering(
        self,
        numbers: Dict[str, NumberEntity],
        *,
        required_attr_map: Dict[str, Set[str]],
        active_rules: List[str],
    ) -> None:
        if not self.factual_entities or not self.factual_entities.numbers or len(numbers) <= 1:
            return

        constrained_ids = {
            ref.split(".", 1)[0]
            for rule in active_rules
            for ref in find_entity_refs(str(rule))
            if ref.startswith("number_")
        }
        ordering_excluded_number_ids = set(getattr(self, "ordering_excluded_number_ids", set()) or set())
        rows: list[tuple[float, str]] = []
        for number_id in numbers:
            if number_id in ordering_excluded_number_ids:
                continue
            if constrained_ids and number_id not in constrained_ids:
                continue
            factual_entity = self._factual_number_entity(number_id)
            factual_value = self._number_actual_value(factual_entity) if factual_entity is not None else None
            current_value = self._number_actual_value(numbers[number_id])
            if factual_value is None or current_value is None:
                continue
            rows.append((factual_value, number_id))
        if len(rows) <= 1:
            return

        rows.sort(key=lambda item: (item[0], item[1]))
        previous_group_max = -math.inf
        idx = 0
        while idx < len(rows):
            factual_value = rows[idx][0]
            group_ids: list[str] = []
            while idx < len(rows) and math.isclose(rows[idx][0], factual_value, abs_tol=1e-9):
                group_ids.append(rows[idx][1])
                idx += 1
            group_ids.sort(key=lambda number_id: self._number_actual_value(numbers[number_id]) or 0.0)
            group_max = previous_group_max
            for number_id in group_ids:
                current_value = self._number_actual_value(numbers[number_id])
                if current_value is None:
                    continue
                number_kind = self._number_kind_for_generation(number_id, required_attr_map.get(number_id))
                if number_id in constrained_ids:
                    current_actual = self._number_actual_value(numbers[number_id])
                    if current_actual is None:
                        return
                    if previous_group_max != -math.inf and current_actual <= previous_group_max + 1e-9:
                        return
                    group_max = max(group_max, current_actual)
                    continue
                low, high = self._number_actual_bounds(number_id, required_attr_map.get(number_id))
                low, high = (low, high) if low <= high else (high, low)
                if previous_group_max == -math.inf:
                    required_low = low
                elif number_kind in {"int", "fraction"}:
                    required_low = max(low, math.floor(previous_group_max + 1e-9) + 1.0)
                else:
                    required_low = max(low, previous_group_max + 0.1)
                adjusted = required_low
                factual_current = self._number_actual_value(self._factual_number_entity(number_id))
                if (
                    factual_current is not None
                    and math.isclose(adjusted, factual_current, abs_tol=1e-9)
                    and high > required_low
                ):
                    if number_kind in {"int", "fraction"}:
                        adjusted = min(high, adjusted + 1.0)
                    else:
                        adjusted = min(high, adjusted + 0.1)

                if adjusted < required_low - 1e-9 or adjusted > high + 1e-9:
                    return
                self._set_number_actual_value(
                    number_id,
                    numbers[number_id],
                    adjusted,
                    required_attrs=required_attr_map.get(number_id),
                )
                current_actual = self._number_actual_value(numbers[number_id])
                if current_actual is None:
                    return
                group_max = max(group_max, current_actual)
            previous_group_max = group_max

    def _repair_number_order_violations(
        self,
        numbers: Dict[str, NumberEntity],
        *,
        required_attr_map: Dict[str, Set[str]],
        active_rules: List[str],
        existing_entities: Optional[EntityCollection],
    ) -> None:
        if not self.factual_entities or not self.factual_entities.numbers or len(numbers) <= 1:
            return

        def rows() -> list[tuple[float, str]]:
            result: list[tuple[float, str]] = []
            constrained_ids = {
                ref.split(".", 1)[0]
                for rule in active_rules
                for ref in find_entity_refs(str(rule))
                if ref.startswith("number_")
            }
            ordering_excluded_number_ids = set(getattr(self, "ordering_excluded_number_ids", set()) or set())
            for number_id in numbers:
                if number_id in ordering_excluded_number_ids:
                    continue
                if constrained_ids and number_id not in constrained_ids:
                    continue
                factual_entity = self._factual_number_entity(number_id)
                factual_value = self._number_actual_value(factual_entity) if factual_entity is not None else None
                if factual_value is None:
                    continue
                result.append((factual_value, number_id))
            result.sort(key=lambda item: (item[0], item[1]))
            return result

        def try_set(number_id: str, target_value: float) -> bool:
            low, high = self._number_actual_bounds(number_id, required_attr_map.get(number_id))
            low, high = (low, high) if low <= high else (high, low)
            if target_value < low - 1e-9 or target_value > high + 1e-9:
                return False
            original = numbers[number_id].model_copy(deep=True)
            self._set_number_actual_value(
                number_id,
                numbers[number_id],
                target_value,
                required_attrs=required_attr_map.get(number_id),
            )
            if self._validate_numbers_against_rules(numbers, active_rules, existing_entities):
                if self._number_equals_factual_value(number_id, numbers[number_id]):
                    self.last_number_forced_equal_refs.update(self._forced_equal_refs_for_number(number_id))
                return True
            numbers[number_id] = original
            return False

        max_passes = max(8, len(rows()) * 4)
        for _ in range(max_passes):
            progress = False
            ordered = rows()
            for idx in range(len(ordered) - 1):
                factual_low, low_id = ordered[idx]
                for jdx in range(idx + 1, len(ordered)):
                    factual_high, high_id = ordered[jdx]
                    if math.isclose(factual_low, factual_high, abs_tol=1e-9):
                        continue
                    current_low = self._number_actual_value(numbers[low_id])
                    current_high = self._number_actual_value(numbers[high_id])
                    if current_low is None or current_high is None or current_low < current_high:
                        continue

                    low_kind = self._number_kind_for_generation(low_id, required_attr_map.get(low_id))
                    high_kind = self._number_kind_for_generation(high_id, required_attr_map.get(high_id))
                    if high_kind in {"int", "fraction"}:
                        raised_target = math.floor(current_low + 1e-9) + 1.0
                    else:
                        raised_target = current_low + 0.1
                    if try_set(high_id, raised_target):
                        progress = True
                        break

                    if low_kind in {"int", "fraction"}:
                        lowered_target = math.floor(current_high - 1e-9)
                    else:
                        lowered_target = current_high - 0.1
                    if try_set(low_id, lowered_target):
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                return

    def _nudge_numbers_away_from_factual(
        self,
        numbers: Dict[str, NumberEntity],
        *,
        required_attr_map: Dict[str, Set[str]],
        active_rules: List[str],
        existing_entities: Optional[EntityCollection],
        avoid_values: Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]]
        | None = None,
    ) -> None:
        avoid_values = avoid_values or {}
        for number_id, number_entity in numbers.items():
            forced_refs = self._forced_equal_refs_for_number(number_id)
            if forced_refs and forced_refs.issubset(self.last_number_forced_equal_refs):
                continue
            if not self._number_equals_factual_value(
                number_id,
                number_entity,
            ) and not self._number_hits_forbidden_value(number_id, number_entity, avoid_values):
                continue
            number_kind = self._number_kind_for_generation(number_id, required_attr_map.get(number_id))
            current_value = self._number_actual_value(number_entity)
            if current_value is None:
                continue
            low, high = self._number_actual_bounds(number_id, required_attr_map.get(number_id))
            low, high = (low, high) if low <= high else (high, low)
            if number_id in avoid_values:
                low, high = self._expand_int_domain_to_escape_forbidden(
                    int(math.floor(low)),
                    int(math.ceil(high)),
                    avoid=avoid_values.get(number_id),
                    min_value=1,
                )
            step = 1.0 if number_kind in {"int", "fraction"} else 0.1
            original = number_entity.model_copy(deep=True)
            moved = False
            if number_kind in {"int", "fraction"}:
                integer_candidates = [
                    float(candidate)
                    for candidate in range(int(math.ceil(low)), int(math.floor(high)) + 1)
                    if not math.isclose(float(candidate), float(current_value), abs_tol=1e-9)
                ]
                candidates = sorted(
                    integer_candidates,
                    key=lambda candidate: (abs(candidate - current_value), abs(candidate)),
                )
            else:
                candidates = [current_value + step, current_value - step]
            for candidate_value in candidates:
                if candidate_value < low - 1e-9 or candidate_value > high + 1e-9:
                    continue
                self._set_number_actual_value(
                    number_id,
                    number_entity,
                    candidate_value,
                    required_attrs=required_attr_map.get(number_id),
                )
                if self._validate_numbers_against_rules(
                    numbers, active_rules, existing_entities
                ) and not self._number_hits_forbidden_value(number_id, number_entity, avoid_values):
                    moved = True
                    break
                numbers[number_id] = original.model_copy(deep=True)
                number_entity = numbers[number_id]
            if not moved and not avoid_values:
                self.last_number_forced_equal_refs.update(forced_refs)

    def _forced_equal_refs_for_number(self, number_id: str) -> set[str]:
        number_kind = self._factual_number_kind(number_id)
        if number_kind == "int":
            return {f"{number_id}.int", f"{number_id}.str"}
        return {f"{number_id}.{number_kind}"}

    def _number_factual_equal_is_forced(self, number_id: str) -> bool:
        forced_refs = self._forced_equal_refs_for_number(number_id)
        return bool(forced_refs) and forced_refs.issubset(self.last_number_forced_equal_refs)

    def _number_equals_factual_value(self, number_id: str, number_entity: NumberEntity) -> bool:
        factual_entity = self._factual_number_entity(number_id)
        if factual_entity is None:
            return False
        number_kind = self._factual_number_kind(number_id)
        if number_kind == "fraction":
            return self._number_field_value(number_entity, "fraction") == self._number_field_value(
                factual_entity, "fraction"
            )
        if number_kind == "int":
            return self._number_entity_int_value(number_entity) == self._number_entity_int_value(factual_entity)
        factual_value = self._number_field_value(factual_entity, number_kind)
        generated_value = self._number_field_value(number_entity, number_kind)
        try:
            return math.isclose(float(factual_value), float(generated_value), abs_tol=1e-9)
        except (TypeError, ValueError):
            return factual_value == generated_value
