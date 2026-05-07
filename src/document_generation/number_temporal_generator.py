"""Constraint-aware auto-entity generation facade.

Numbers follow the benchmark's solver-first MILP workflow, while temporals are
sampled with randomized constrained generation inside bounded local windows.
"""

import math
import random
from typing import Any

from src.core.implicit_numeric_rules import implicit_rule_bounds_lookup
from src.core.document_schema import ImplicitRule
from .generation_limits import _DEFAULT_NUMBER_MIN, _relative_int_window
from .century_generation import CenturyGenerationMixin
from .number_generation import NumberGenerationMixin
from .temporal_generation import TemporalGenerationMixin


class NumberTemporalGenerator(NumberGenerationMixin, TemporalGenerationMixin, CenturyGenerationMixin):
    """Generates numbers and temporals satisfying constraints."""

    def __init__(
        self,
        seed: int | None = None,
        exclude_numbers: set[int] | None = None,
        exclude_temporals: dict[str, set] | None = None,
        factual_entities: Any = None,
        implicit_rules: list[ImplicitRule] | list[dict[str, Any]] | None = None,
        ordering_excluded_number_ids: set[str] | None = None,
    ):
        if seed is not None:
            random.seed(seed)
        self.exclude_numbers = exclude_numbers or set()
        self.exclude_temporals = exclude_temporals or {}
        self.factual_entities = factual_entities
        self.implicit_rules = list(implicit_rules or [])
        self.ordering_excluded_number_ids = set(ordering_excluded_number_ids or set())
        self._implicit_rule_lookup = implicit_rule_bounds_lookup(self.implicit_rules)
        self._implicit_number_rules: dict[str, ImplicitRule] = {}
        self._implicit_number_rules_by_id: dict[str, list[ImplicitRule]] = {}
        self._implicit_age_rules: dict[str, ImplicitRule] = {}
        self._implicit_temporal_rules: dict[str, ImplicitRule] = {}
        for entity_ref, rule in self._implicit_rule_lookup.items():
            entity_id = entity_ref.split(".", 1)[0]
            if entity_ref.startswith("number_"):
                self._implicit_number_rules_by_id.setdefault(entity_id, []).append(rule)
                self._implicit_number_rules.setdefault(entity_id, rule)
            elif entity_ref.endswith(".age") and entity_ref.startswith("person_"):
                self._implicit_age_rules[entity_id] = rule
            elif entity_ref.startswith("temporal_"):
                self._implicit_temporal_rules[entity_ref] = rule
        self.last_century_forced_equal_ids: set[str] = set()
        self.last_implicit_forced_equal_refs: set[str] = set()
        self.last_number_forced_equal_refs: set[str] = set()
        self.last_number_solution_used_relaxed_avoid = False

    @staticmethod
    def _coerce_implicit_int_bounds(
        rule: ImplicitRule | None,
        *,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> tuple[int, int] | None:
        if rule is None:
            return None
        low = math.ceil(float(rule.lower_bound))
        high = math.floor(float(rule.upper_bound))
        if min_value is not None:
            low = max(low, min_value)
            high = max(high, min_value)
        if max_value is not None:
            low = min(low, max_value)
            high = min(high, max_value)
        if low > high:
            factual = round(float(rule.factual_value))
            if min_value is not None:
                factual = max(factual, min_value)
            if max_value is not None:
                factual = min(factual, max_value)
            low = high = factual
        return low, high

    def _implicit_number_range(self, number_id: str) -> tuple[int, int] | None:
        rules = self._implicit_number_rules_by_id.get(number_id) or []
        rule = self._implicit_number_rules.get(number_id)
        number_kind = self._factual_number_kind(number_id)
        factual_value = self._factual_number_int(number_id)
        if number_kind == "fraction" and rule is not None:
            try:
                lower_bound = float(rule.lower_bound)
                upper_bound = float(rule.upper_bound)
                factual_value = float(rule.factual_value)
            except (TypeError, ValueError):
                return None
            if lower_bound > 0.0 and upper_bound > 0.0 and factual_value > 0.0:
                low = max(2, math.floor(1.0 / upper_bound))
                high = max(low, math.ceil(1.0 / lower_bound))
                return low, high
        bounds_candidates: list[tuple[ImplicitRule, tuple[int, int]]] = []
        for candidate_rule in rules:
            candidate_bounds = self._coerce_implicit_int_bounds(candidate_rule)
            if candidate_bounds is not None:
                bounds_candidates.append((candidate_rule, candidate_bounds))
        if not bounds_candidates:
            bounds = self._coerce_implicit_int_bounds(rule)
        elif len(bounds_candidates) == 1:
            rule, bounds = bounds_candidates[0]
        else:
            matching_candidates = []
            if factual_value is not None:
                matching_candidates = [
                    (candidate_rule, candidate_bounds)
                    for candidate_rule, candidate_bounds in bounds_candidates
                    if candidate_bounds[0] <= factual_value <= candidate_bounds[1]
                ]
            selected_pool = matching_candidates or bounds_candidates
            intersection_low = max(candidate_bounds[0] for _candidate_rule, candidate_bounds in selected_pool)
            intersection_high = min(candidate_bounds[1] for _candidate_rule, candidate_bounds in selected_pool)
            if intersection_low <= intersection_high:
                rule = selected_pool[0][0]
                bounds = (intersection_low, intersection_high)
            else:
                rule, bounds = min(
                    selected_pool,
                    key=lambda item: (
                        abs((item[1][0] + item[1][1]) / 2 - (factual_value if factual_value is not None else 0)),
                        item[1][1] - item[1][0],
                    ),
                )
        if factual_value is not None and bounds is not None and bounds[0] == bounds[1] == factual_value:
            widened_bounds = _relative_int_window(
                factual_value,
                small_value_threshold=10,
                small_value_delta=10,
                min_value=0 if factual_value == 0 else _DEFAULT_NUMBER_MIN,
            )
            if widened_bounds != bounds:
                bounds = widened_bounds

        if number_kind == "int":
            self._remember_forced_equal_ref(f"{number_id}.int", rule, bounds)
            self._remember_forced_equal_ref(f"{number_id}.str", rule, bounds)
        else:
            self._remember_forced_equal_ref(f"{number_id}.{number_kind}", rule, bounds)
        return bounds

    def _implicit_age_range(self, person_id: str, *, min_age: int, max_age: int) -> tuple[int, int] | None:
        rule = self._implicit_age_rules.get(person_id)
        bounds = self._coerce_implicit_int_bounds(
            rule,
            min_value=min_age,
            max_value=max_age,
        )
        self._remember_forced_equal_ref(f"{person_id}.age", rule, bounds)
        return bounds

    def _implicit_temporal_range(self, temporal_id: str, attribute: str) -> tuple[int, int] | None:
        entity_ref = f"{temporal_id}.{attribute}"
        rule = self._implicit_temporal_rules.get(entity_ref)
        bounds = self._coerce_implicit_int_bounds(rule)
        self._remember_forced_equal_ref(entity_ref, rule, bounds)
        return bounds

    def _remember_forced_equal_ref(
        self,
        entity_ref: str,
        rule: ImplicitRule | None,
        bounds: tuple[int, int] | None,
    ) -> None:
        if rule is None or bounds is None:
            return
        low, high = bounds
        factual = round(float(rule.factual_value))
        if low == high == factual:
            self.last_implicit_forced_equal_refs.add(entity_ref)
