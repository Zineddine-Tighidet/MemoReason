"""Solver-first workflow for generating numeric entity values."""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple

from src.core.annotation_runtime import RuleEngine, find_entity_refs
from src.core.document_schema import EntityCollection, NumberEntity
from src.document_generation.generation_exceptions import StrictInterVariantUniquenessInfeasible
from ..generation_limits import (
    SLOW_SAMPLING_LOG_SECONDS,
    SLOW_STAGE_LOG_SECONDS,
    _DEFAULT_NUMBER_MIN,
    _NUMBER_GENERATION_MAX_RETRIES,
)

logger = logging.getLogger(__name__)


class NumberGenerationWorkflowMixin:
    """Generate numeric entities with constraint solving before retry sampling."""

    _EXACT_AVOID_EXPANSION_STEPS = (0, 2, 4, 8, 16, 32)

    def _implicit_number_bound_rules(self, number_ids: List[str]) -> List[str]:
        rules: List[str] = []
        for number_id in number_ids:
            low, high = self._number_base_range(number_id)
            rules.append(f"{number_id}.int >= {int(low)}")
            rules.append(f"{number_id}.int <= {int(high)}")
        return rules

    def _number_hits_forbidden_value(
        self,
        number_id: str,
        number_entity: NumberEntity,
        avoid_values: Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]],
    ) -> bool:
        number_kind = self._factual_number_kind(number_id)
        factual_entity = self._factual_number_entity(number_id)
        if number_kind in {"int", "fraction"}:
            current_value = self._number_entity_int_value(number_entity)
            if current_value is None:
                return False
            forbidden = self._coerce_forbidden_number_values(avoid_values.get(number_id))
            if current_value not in forbidden:
                return False
            factual_value = self._factual_number_int(number_id)
            if current_value == factual_value and self._number_factual_equal_is_forced(number_id):
                return False
            return True

        current_value = self._number_field_value(number_entity, number_kind)
        if current_value is None:
            return False
        forbidden = self._coerce_forbidden_actual_values(avoid_values.get(number_id))
        if not any(math.isclose(float(current_value), forbidden_value, abs_tol=1e-9) for forbidden_value in forbidden):
            return False
        factual_value = self._number_field_value(factual_entity, number_kind) if factual_entity is not None else None
        if (
            factual_value is not None
            and math.isclose(float(current_value), float(factual_value), abs_tol=1e-9)
            and self._number_factual_equal_is_forced(number_id)
        ):
            return False
        return True

    def generate_numbers(
        self,
        required_numbers: List[tuple],
        rules: List[str],
        existing_entities: Optional[EntityCollection] = None,
        avoid_values: Optional[
            Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]]
        ] = None,
    ) -> Dict[str, NumberEntity]:
        self.last_number_forced_equal_refs = set()
        self.last_relaxed_avoid_number_ids = set()
        number_ids = [nid for nid, _ in required_numbers]
        required_attr_map = {nid: set(attrs or []) for nid, attrs in required_numbers}
        pre_assigned: Dict[str, int] = {}
        avoid_values = avoid_values or {}
        previous_avoid_values = getattr(self, "_current_number_avoid_values", None)
        previous_avoid_expansion = getattr(self, "_current_number_avoid_expansion", None)
        self._current_number_avoid_values = avoid_values
        try:
            if existing_entities and existing_entities.numbers:
                for num_id, num_entity in existing_entities.numbers.items():
                    if num_id in number_ids:
                        continue
                    numeric_value = self._number_entity_int_value(num_entity)
                    if numeric_value is not None:
                        pre_assigned[num_id] = numeric_value

            def _build_exact_rule_bundle(expansion_padding: int) -> tuple[list[str], set[str], list[tuple] | None]:
                self._current_number_avoid_expansion = (
                    {number_id: expansion_padding for number_id in number_ids if number_id in avoid_values}
                    if expansion_padding > 0 and avoid_values
                    else {}
                )
                bundled_rules = self._number_evaluable_rules(rules, set(number_ids), existing_entities)
                ordering_rules = self._ordering_number_rules(number_ids, existing_entities)
                if ordering_rules:
                    bundled_rules = [*bundled_rules, *ordering_rules]
                implicit_bound_rules = self._implicit_number_bound_rules(number_ids)
                if implicit_bound_rules:
                    bundled_rules = [*bundled_rules, *implicit_bound_rules]
                protected_numbers = {
                    ref.split(".", 1)[0]
                    for rule in bundled_rules
                    for ref in find_entity_refs(rule)
                    if ref.startswith("number_")
                }
                bundled_constraints = self._collect_linear_constraints(bundled_rules, set(number_ids), existing_entities)
                if bundled_constraints is not None:
                    self._tighten_number_domains(
                        bundled_constraints,
                        {vid: self._number_base_range(vid) for vid in number_ids},
                    )
                return bundled_rules, protected_numbers, bundled_constraints

            active_rules, protected_decimal_numbers, exact_constraints = _build_exact_rule_bundle(0)
            if self._number_rules_are_independent(active_rules, set(number_ids)):
                numbers = self._generate_numbers_independently(
                    number_ids,
                    required_attr_map=required_attr_map,
                    active_rules=active_rules,
                    existing_entities=existing_entities,
                    avoid_values=avoid_values,
                    protected_decimal_numbers=protected_decimal_numbers,
                )
                if numbers is not None:
                    return numbers

            solve_start = time.monotonic()
            solved = None
            exact_solve_error = False
            exact_active_rules = active_rules
            exact_protected_decimal_numbers = protected_decimal_numbers
            exact_constraints_last = exact_constraints
            exact_expansion_steps = self._EXACT_AVOID_EXPANSION_STEPS if avoid_values else (0,)
            for expansion_padding in exact_expansion_steps:
                active_rules, protected_decimal_numbers, exact_constraints = _build_exact_rule_bundle(expansion_padding)
                exact_active_rules = active_rules
                exact_protected_decimal_numbers = protected_decimal_numbers
                exact_constraints_last = exact_constraints
                solved = self._solve_numbers_via_constraints(number_ids, active_rules, existing_entities, avoid_values)
                if solved is not None:
                    break
                if exact_constraints is None or not avoid_values:
                    break
                exact_solve_error = True
            if solved is not None:
                self.last_number_forced_equal_refs = set()
                missing_solved_ids = [number_id for number_id in number_ids if number_id not in solved]
                if missing_solved_ids:
                    logger.debug(
                        "Rejecting incomplete exact number solution; missing assignments for %s",
                        missing_solved_ids,
                    )
                else:
                    numbers = {}
                    for number_id in number_ids:
                        entity_value = int(solved[number_id])
                        numbers[number_id] = self._build_number_entity(
                            number_id,
                            entity_value,
                            required_attrs=required_attr_map.get(number_id),
                            allow_non_integer_adjustment=number_id not in exact_protected_decimal_numbers,
                        )
                    if not avoid_values:
                        self._nudge_numbers_away_from_factual(
                            numbers,
                            required_attr_map=required_attr_map,
                            active_rules=exact_active_rules,
                            existing_entities=existing_entities,
                        )
                    try:
                        if self._validate_numbers_against_rules(numbers, exact_active_rules, existing_entities):
                            if avoid_values:
                                for _ in range(max(1, len(number_ids))):
                                    before_values = {
                                        number_id: self._number_actual_value(numbers[number_id])
                                        for number_id in number_ids
                                    }
                                    self._nudge_numbers_away_from_factual(
                                        numbers,
                                        required_attr_map=required_attr_map,
                                        active_rules=exact_active_rules,
                                        existing_entities=existing_entities,
                                        avoid_values=avoid_values,
                                    )
                                    remaining_forbidden = {
                                        num_id
                                        for num_id in number_ids
                                        if num_id in avoid_values
                                        and self._number_hits_forbidden_value(
                                            num_id,
                                            numbers[num_id],
                                            avoid_values,
                                        )
                                    }
                                    if not remaining_forbidden:
                                        break
                                    after_values = {
                                        number_id: self._number_actual_value(numbers[number_id])
                                        for number_id in number_ids
                                    }
                                    if after_values == before_values:
                                        break
                                if not self._validate_numbers_against_rules(
                                    numbers,
                                    exact_active_rules,
                                    existing_entities,
                                ):
                                    raise ValueError("Avoid-aware number nudge invalidated exact solution.")
                                equal_ids = {
                                    num_id
                                    for num_id in number_ids
                                    if num_id in avoid_values
                                    and self._number_hits_forbidden_value(num_id, numbers[num_id], avoid_values)
                                }
                                blocking_equal_ids = equal_ids - set(getattr(self, "last_relaxed_avoid_number_ids", set()))
                                if blocking_equal_ids:
                                    logger.debug(
                                        "Rejecting exact number solution because it matches factual values: %s",
                                        sorted(blocking_equal_ids),
                                    )
                                    if getattr(self, "last_number_solution_used_relaxed_avoid", False):
                                        if not getattr(self, "allow_relaxed_factual_avoid_solution", False):
                                            raise StrictInterVariantUniquenessInfeasible(
                                                "Exact number solve required non-structural forbidden values."
                                            )
                                        return numbers
                                elif equal_ids:
                                    logger.debug(
                                        "Accepting exact number solution with structurally forced reused values: %s",
                                        sorted(equal_ids),
                                    )
                                    for num_id in equal_ids:
                                        self.last_number_forced_equal_refs.update(
                                            self._forced_equal_refs_for_number(num_id)
                                        )
                                    return numbers
                                else:
                                    return numbers
                            else:
                                return numbers
                    except Exception as e:
                        if isinstance(e, StrictInterVariantUniquenessInfeasible):
                            raise
                        logger.debug("Constraint-solve validation failed: %s", e)
            elapsed = time.monotonic() - solve_start
            if elapsed > SLOW_STAGE_LOG_SECONDS:
                print(f"[slow] CSP number solve took {elapsed:.1f}s for vars={','.join(number_ids)}", flush=True)

            if exact_constraints_last is not None and exact_solve_error:
                raise ValueError("Unable to generate numbers satisfying rules with exact MILP under forbidden values.")

            active_rules = exact_active_rules
            protected_decimal_numbers = exact_protected_decimal_numbers
            exact_constraints = exact_constraints_last

            equality_constraints, reverse_equality_constraints = self._parse_equality_constraints(active_rules, number_ids)
            parsed_equality_constraints = {
                constrained_var: self._parse_linear_expr(expr, set(number_ids), existing_entities)
                for constrained_var, expr in equality_constraints.items()
            }
            parsed_reverse_constraints = {
                constrained_var: (
                    self._parse_linear_expr(lhs_expr, set(number_ids), existing_entities),
                    rhs_var,
                )
                for constrained_var, (lhs_expr, rhs_var) in reverse_equality_constraints.items()
            }

            factual_vals = {}
            if self.factual_entities and self.factual_entities.numbers:
                for num_id, num_entity in self.factual_entities.numbers.items():
                    val = self._number_entity_int_value(num_entity)
                    if val is not None:
                        factual_vals[num_id] = val

            start_time = time.monotonic()
            for attempt in range(_NUMBER_GENERATION_MAX_RETRIES):
                self.last_number_forced_equal_refs = set()
                assigned = pre_assigned.copy()
                free_vars = {vid for vid in number_ids if vid not in equality_constraints and vid not in assigned}
                if attempt and attempt % 10 == 0 and (time.monotonic() - start_time) > SLOW_SAMPLING_LOG_SECONDS:
                    print(
                        f"[slow] Number generation retrying ({attempt}/{_NUMBER_GENERATION_MAX_RETRIES}) after {time.monotonic() - start_time:.1f}s vars={','.join(number_ids)}",
                        flush=True,
                    )
                failed = False
                while free_vars:
                    ranges: Dict[str, Tuple[int, int]] = {}
                    for var_id in free_vars:
                        min_val, max_val = self._get_number_range_from_rules(
                            var_id,
                            active_rules,
                            assigned,
                            existing_entities,
                        )
                        min_val = max(_DEFAULT_NUMBER_MIN, min_val)
                        min_val, max_val = self._expand_int_domain_to_escape_forbidden(
                            min_val,
                            max_val,
                            avoid=avoid_values.get(var_id),
                            min_value=_DEFAULT_NUMBER_MIN,
                        )
                        if max_val < min_val:
                            failed = True
                            break
                        ranges[var_id] = (min_val, max_val)
                    if failed:
                        break

                    next_var = min(
                        free_vars,
                        key=lambda vid: (
                            ranges[vid][1] - ranges[vid][0],
                            factual_vals.get(vid, float("inf")),
                        ),
                    )
                    min_val, max_val = ranges[next_var]
                    avoid = avoid_values.get(next_var)
                    forbidden = self._coerce_forbidden_number_values(avoid)
                    if forbidden and min_val == max_val:
                        if min_val in forbidden:
                            failed = True
                            break
                        assigned[next_var] = min_val
                    else:
                        assigned[next_var] = self._sample_int_in_range(min_val, max_val, avoid=avoid)
                    free_vars.remove(next_var)

                if failed:
                    continue

                assignment_conflict = False
                for constrained_var, (_lhs_expr, rhs_var) in list(reverse_equality_constraints.items()):
                    if constrained_var in assigned:
                        continue
                    rhs_value = assigned.get(rhs_var)
                    if rhs_value is None and existing_entities:
                        rhs_value = RuleEngine._get_entity_value(existing_entities, rhs_var)
                    if rhs_value is None:
                        continue
                    try:
                        parsed_linear, _ = parsed_reverse_constraints.get(constrained_var, (None, rhs_var))
                        if parsed_linear is None:
                            raise ValueError("unparsed reverse-equality expression")
                        coeffs, const = parsed_linear
                        coeff = coeffs.get(constrained_var, 0.0)
                        if math.isclose(coeff, 0.0, abs_tol=1e-9):
                            raise ValueError("missing constrained coefficient")
                        rest_linear = ({key: value for key, value in coeffs.items() if key != constrained_var}, const)
                        rest_value = self._evaluate_linear_expr(rest_linear, assigned, existing_entities)
                        if rest_value is None:
                            raise ValueError("unresolved reverse-equality refs")
                        assigned[constrained_var] = int(round((float(rhs_value) - float(rest_value)) / coeff))
                        if assigned[constrained_var] in self._coerce_forbidden_number_values(
                            avoid_values.get(constrained_var)
                        ):
                            assignment_conflict = True
                            break
                    except Exception as e:
                        logger.debug("Reverse-equality solve failed for %s: %s", constrained_var, e)
                        assignment_conflict = True
                        break
                if assignment_conflict:
                    continue

                for constrained_var, expr in equality_constraints.items():
                    if constrained_var in assigned:
                        continue
                    try:
                        parsed_linear = parsed_equality_constraints.get(constrained_var)
                        if parsed_linear is None:
                            raise ValueError("unparsed equality expression")
                        resolved_value = self._evaluate_linear_expr(parsed_linear, assigned, existing_entities)
                        if resolved_value is None:
                            raise ValueError("unresolved equality refs")
                        assigned[constrained_var] = int(round(resolved_value))
                        if assigned[constrained_var] in self._coerce_forbidden_number_values(
                            avoid_values.get(constrained_var)
                        ):
                            assignment_conflict = True
                            break
                    except Exception as e:
                        logger.debug("Equality solve failed for %s (%r): %s", constrained_var, expr, e)
                        assignment_conflict = True
                        break
                if assignment_conflict:
                    continue

                missing_assigned_ids = [number_id for number_id in number_ids if number_id not in assigned]
                if missing_assigned_ids:
                    logger.debug(
                        "Rejecting number generation attempt with missing assignments for %s",
                        missing_assigned_ids,
                    )
                    continue

                numbers = {}
                for number_id in number_ids:
                    entity_value = assigned[number_id]
                    numbers[number_id] = self._build_number_entity(
                        number_id,
                        entity_value,
                        required_attrs=required_attr_map.get(number_id),
                        allow_non_integer_adjustment=number_id not in protected_decimal_numbers,
                    )
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
                    if self._validate_numbers_against_rules(numbers, active_rules, existing_entities):
                        if avoid_values:
                            equal_ids = {
                                num_id
                                for num_id in number_ids
                                if (
                                    num_id in avoid_values
                                    and self._number_hits_forbidden_value(num_id, numbers[num_id], avoid_values)
                                )
                            }
                            if equal_ids:
                                continue
                        return numbers
                except Exception as e:
                    logger.debug("Number generation attempt %d failed: %s", attempt, e)
                    continue

            raise ValueError("Unable to generate numbers satisfying rules after retries.")
        finally:
            self._current_number_avoid_values = previous_avoid_values or {}
            self._current_number_avoid_expansion = previous_avoid_expansion or {}

    def _number_rules_are_independent(self, active_rules: List[str], number_ids: set[str]) -> bool:
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
        number_ids: List[str],
        *,
        required_attr_map: Dict[str, set[str]],
        active_rules: List[str],
        existing_entities: Optional[EntityCollection],
        avoid_values: Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]],
        protected_decimal_numbers: set[str],
    ) -> Dict[str, NumberEntity] | None:
        for _attempt in range(_NUMBER_GENERATION_MAX_RETRIES):
            self.last_number_forced_equal_refs = set()
            numbers: Dict[str, NumberEntity] = {}
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
                if (
                    num_id in avoid_values
                    and self._number_hits_forbidden_value(num_id, numbers[num_id], avoid_values)
                )
            }
            if equal_ids:
                continue
            return numbers
        return None
