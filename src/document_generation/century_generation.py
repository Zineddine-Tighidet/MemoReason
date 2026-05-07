"""Joint generation helpers for century-aware number and temporal rules."""

from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Set, Tuple

from src.core.century_expressions import (
    CenturyConstraint,
    CenturyExpression,
    coerce_numeric_ref_value,
    century_end,
    century_expression_refs,
    century_of,
    century_start,
    parse_century_constraint,
)
from src.core.document_schema import EntityCollection, TemporalEntity
from src.core.annotation_runtime import RuleEngine, find_entity_refs
from .generation_limits import _CSP_EXACT_DOMAIN_LIMIT, _CSP_MAX_SAMPLE_CANDIDATES


class CenturyGenerationMixin:
    """Mixin that jointly solves century-aware rules over numbers and temporals."""

    _PURE_ORDERING_RULE_RE = re.compile(
        r"""
        ^\s*
        (?:number_\d+\.(?:int|str|float|percent|proportion)|temporal_\d+\.year)
        \s*(?:<=|>=|<|>)\s*
        (?:number_\d+\.(?:int|str|float|percent|proportion)|temporal_\d+\.year)
        \s*$
        """,
        re.VERBOSE,
    )

    def apply_century_constraints(
        self,
        collection: EntityCollection,
        rules: List[str],
        required_numbers: List[tuple],
        required_temporals: List[tuple],
        avoid_number_values: Optional[Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]]] = None,
        decade_year_temporal_ids: Optional[Set[str]] = None,
    ) -> bool:
        self.last_century_forced_equal_ids = set()
        century_constraints = self._parse_century_constraints(rules)
        if not century_constraints:
            return True

        century_rule_texts = [constraint.raw_expression for constraint in century_constraints]
        required_number_ids = {number_id for number_id, _ in required_numbers}
        required_temporal_ids = {
            temporal_id for temporal_id, attrs in required_temporals if "year" in attrs or "date" in attrs
        }

        active_entity_ids = self._collect_century_active_entities(
            century_constraints,
            rules=rules,
            required_number_ids=required_number_ids,
            required_temporal_ids=required_temporal_ids,
            collection=collection,
        )

        if not active_entity_ids:
            results = RuleEngine.validate_all_rules(century_rule_texts, collection)
            return all(is_valid for _, is_valid in results)

        decade_year_temporal_ids = set(decade_year_temporal_ids or set())
        avoid_number_values = avoid_number_values or {}

        domains = self._build_century_domains(
            active_entity_ids,
            collection=collection,
            avoid_number_values=avoid_number_values,
            decade_year_temporal_ids=decade_year_temporal_ids,
        )
        if domains is None:
            return False

        equality_groups = self._collect_active_equality_groups(rules, active_entity_ids)
        leader_domains = self._collapse_domains_by_equality_groups(domains, equality_groups)
        if leader_domains is None:
            return False

        variable_ids = sorted(leader_domains, key=lambda entity_id: (len(leader_domains[entity_id]), entity_id))

        def backtrack(assignments: Dict[str, int]) -> EntityCollection | None:
            if len(assignments) == len(variable_ids):
                expanded_assignments = self._expand_equality_assignments(assignments, equality_groups)
                candidate_collection = self._apply_century_assignments(collection, expanded_assignments)
                results = RuleEngine.validate_all_rules(rules, candidate_collection)
                if all(is_valid for _, is_valid in results):
                    return candidate_collection
                return None

            next_entity_id = min(
                (entity_id for entity_id in variable_ids if entity_id not in assignments),
                key=lambda entity_id: len(leader_domains[entity_id]),
            )

            for value in leader_domains[next_entity_id]:
                assignments[next_entity_id] = value
                expanded_assignments = self._expand_equality_assignments(assignments, equality_groups)
                if self._century_constraints_feasible(
                    century_constraints,
                    collection,
                    expanded_assignments,
                    domains,
                ):
                    solved = backtrack(assignments)
                    if solved is not None:
                        return solved
                assignments.pop(next_entity_id, None)
            return None

        solved_collection = backtrack({})
        if solved_collection is None:
            return False

        self.last_century_forced_equal_ids = self._collect_century_forced_equal_ids(
            solved_collection,
            active_entity_ids=active_entity_ids,
            avoid_number_values=avoid_number_values,
        )
        collection.numbers = solved_collection.numbers
        collection.temporals = solved_collection.temporals
        return True

    @staticmethod
    def _collect_active_equality_groups(rules: List[str], active_entity_ids: Set[str]) -> Dict[str, Set[str]]:
        parent: Dict[str, str] = {entity_id: entity_id for entity_id in active_entity_ids}

        def find(entity_id: str) -> str:
            while parent[entity_id] != entity_id:
                parent[entity_id] = parent[parent[entity_id]]
                entity_id = parent[entity_id]
            return entity_id

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

        temporal_matcher = re.compile(r"^(temporal_\d+)\.year\s*(?:==|=)\s*(temporal_\d+)\.year$")
        number_matcher = re.compile(
            r"^(number_\d+)\.(?:int|str)\s*(?:==|=)\s*(number_\d+)\.(?:int|str)$"
        )

        for raw_rule in rules:
            cleaned = str(raw_rule or "").split("#", 1)[0].strip()
            if not cleaned:
                continue
            match = temporal_matcher.fullmatch(cleaned) or number_matcher.fullmatch(cleaned)
            if match is None:
                continue
            left_id, right_id = match.group(1), match.group(2)
            if left_id not in active_entity_ids or right_id not in active_entity_ids:
                continue
            if left_id.split("_", 1)[0] != right_id.split("_", 1)[0]:
                continue
            union(left_id, right_id)

        groups: Dict[str, Set[str]] = {}
        for entity_id in active_entity_ids:
            groups.setdefault(find(entity_id), set()).add(entity_id)
        return groups

    @staticmethod
    def _collapse_domains_by_equality_groups(
        domains: Dict[str, List[int]],
        equality_groups: Dict[str, Set[str]],
    ) -> Dict[str, List[int]] | None:
        collapsed: Dict[str, List[int]] = {}
        for leader, members in equality_groups.items():
            member_domains = [set(domains[member]) for member in members]
            intersection = set.intersection(*member_domains) if member_domains else set()
            if not intersection:
                return None
            collapsed[leader] = sorted(intersection)
        return collapsed

    @staticmethod
    def _expand_equality_assignments(
        assignments: Dict[str, int],
        equality_groups: Dict[str, Set[str]],
    ) -> Dict[str, int]:
        expanded: Dict[str, int] = {}
        for leader, value in assignments.items():
            members = equality_groups.get(leader, {leader})
            for member in members:
                expanded[member] = value
        return expanded

    @staticmethod
    def _parse_century_constraints(rules: List[str]) -> List[CenturyConstraint]:
        constraints: List[CenturyConstraint] = []
        for raw_rule in rules:
            expression = str(raw_rule or "").split("#", 1)[0].strip()
            if not expression:
                continue
            parsed = parse_century_constraint(expression)
            if parsed is not None:
                constraints.append(parsed)
        return constraints

    @staticmethod
    def _entity_id_from_ref(ref: str) -> str:
        return str(ref or "").split(".", 1)[0].strip()

    def _collect_century_active_entities(
        self,
        constraints: List[CenturyConstraint],
        *,
        rules: List[str],
        required_number_ids: Set[str],
        required_temporal_ids: Set[str],
        collection: EntityCollection,
    ) -> Set[str]:
        active_entity_ids: Set[str] = set()
        for constraint in constraints:
            refs = century_expression_refs(constraint.lhs) | century_expression_refs(constraint.rhs)
            for ref in refs:
                entity_id = self._entity_id_from_ref(ref)
                if entity_id in required_number_ids and entity_id in collection.numbers:
                    active_entity_ids.add(entity_id)
                if entity_id in required_temporal_ids and entity_id in collection.temporals:
                    active_entity_ids.add(entity_id)

        if not active_entity_ids:
            return active_entity_ids

        expandable_ids = required_number_ids | required_temporal_ids
        changed = True
        while changed:
            changed = False
            for raw_rule in rules:
                cleaned = str(raw_rule or "").split("#", 1)[0].strip()
                if not cleaned:
                    continue
                if self._PURE_ORDERING_RULE_RE.fullmatch(cleaned):
                    continue
                refs = {
                    self._entity_id_from_ref(ref)
                    for ref in find_entity_refs(cleaned)
                    if ref.startswith(("number_", "temporal_"))
                }
                if not refs or refs.isdisjoint(active_entity_ids):
                    continue
                for entity_id in refs:
                    if entity_id not in expandable_ids or entity_id in active_entity_ids:
                        continue
                    if entity_id.startswith("number_") and entity_id not in collection.numbers:
                        continue
                    if entity_id.startswith("temporal_") and entity_id not in collection.temporals:
                        continue
                    active_entity_ids.add(entity_id)
                    changed = True
        return active_entity_ids

    def _build_century_domains(
        self,
        active_entity_ids: Set[str],
        *,
        collection: EntityCollection,
        avoid_number_values: Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]],
        decade_year_temporal_ids: Set[str],
    ) -> Dict[str, List[int]] | None:
        domains: Dict[str, List[int]] = {}
        for entity_id in active_entity_ids:
            if entity_id.startswith("number_"):
                low, high = self._number_base_range(entity_id)
                values = self._candidate_values(low, high, avoid=avoid_number_values.get(entity_id))
            elif entity_id.startswith("temporal_"):
                values = self._temporal_candidates_for_century_rules(
                    temporal_id=entity_id,
                    collection=collection,
                    decade_year_temporal_ids=decade_year_temporal_ids,
                )
            else:
                return None

            if not values:
                return None
            domains[entity_id] = values
        return domains

    def _temporal_candidates_for_century_rules(
        self,
        *,
        temporal_id: str,
        collection: EntityCollection,
        decade_year_temporal_ids: Set[str],
    ) -> List[int]:
        low, high = self._temporal_year_base_range(temporal_id)
        excluded_years = set(self.exclude_temporals.get("years", set()))
        factual_year = self._factual_temporal_year(temporal_id)
        current_year = getattr(collection.temporals.get(temporal_id), "year", None)

        values = [
            year for year in range(low, high + 1) if self._temporal_year_is_available(temporal_id, year, excluded_years)
        ]
        if temporal_id in decade_year_temporal_ids:
            values = [year for year in values if year % 10 == 0]

        # Century-aware constraints often need to move a year outside the
        # default local +/- window. Add a sparse set of representative years
        # across the full temporal horizon so century changes remain feasible
        # without exploding the search space.
        overall_low, overall_high = self._temporal_year_sampling_bounds(temporal_id)
        candidate_years = set(values)
        first_century = century_of(overall_low)
        last_century = century_of(overall_high)
        if first_century is not None and last_century is not None:
            for century in range(first_century, last_century + 1):
                century_low = max(overall_low, century_start(century) or overall_low)
                century_high = min(overall_high, century_end(century) or overall_high)
                if century_low > century_high:
                    continue
                representatives = {
                    century_low,
                    century_high,
                    (century_low + century_high) // 2,
                }
                for year in representatives:
                    if not self._temporal_year_is_available(temporal_id, year, excluded_years):
                        continue
                    if temporal_id in decade_year_temporal_ids and year % 10 != 0:
                        continue
                    candidate_years.add(year)

        values = sorted(candidate_years)

        if len(values) > _CSP_EXACT_DOMAIN_LIMIT:
            sampled = {low, high, (low + high) // 2}
            if current_year is not None:
                sampled.add(int(current_year))
            if factual_year is not None:
                sampled.add(int(factual_year))
            target_size = min(_CSP_MAX_SAMPLE_CANDIDATES, len(values))
            while len(sampled) < target_size:
                sampled.add(random.randint(low, high))
            values = sorted(
                year for year in sampled if self._temporal_year_is_available(temporal_id, year, excluded_years)
            )
            if temporal_id in decade_year_temporal_ids:
                values = [year for year in values if year % 10 == 0]

        random.shuffle(values)
        if factual_year in values and len(values) > 1:
            values = [year for year in values if year != factual_year] + [factual_year]
        return values

    def _century_constraints_feasible(
        self,
        constraints: List[CenturyConstraint],
        collection: EntityCollection,
        assignments: Dict[str, int],
        domains: Dict[str, List[int]],
    ) -> bool:
        for constraint in constraints:
            left_bounds = self._century_expression_bounds(constraint.lhs, collection, assignments, domains)
            right_bounds = self._century_expression_bounds(constraint.rhs, collection, assignments, domains)
            if left_bounds is None or right_bounds is None:
                return False

            left_min, left_max = left_bounds
            right_min, right_max = right_bounds
            operator = constraint.operator
            if operator == "==":
                if left_max < right_min or right_max < left_min:
                    return False
            elif operator == "!=":
                if left_min == left_max == right_min == right_max:
                    return False
            elif operator == "<":
                if left_min >= right_max:
                    return False
            elif operator == "<=":
                if left_min > right_max:
                    return False
            elif operator == ">":
                if left_max <= right_min:
                    return False
            elif operator == ">=":
                if left_max < right_min:
                    return False
        return True

    def _century_expression_bounds(
        self,
        expression: CenturyExpression,
        collection: EntityCollection,
        assignments: Dict[str, int],
        domains: Dict[str, List[int]],
    ) -> Tuple[int, int] | None:
        if expression.kind == "const" and expression.value is not None:
            value = int(expression.value)
            return value, value

        if expression.kind == "ref" and isinstance(expression.value, str):
            ref = expression.value
            entity_id = self._entity_id_from_ref(ref)
            if entity_id in assignments:
                value = assignments[entity_id]
                return value, value
            if entity_id in domains:
                domain_values = domains[entity_id]
                return min(domain_values), max(domain_values)
            resolved = coerce_numeric_ref_value(ref, RuleEngine._get_entity_value(collection, ref))
            if resolved is None:
                return None
            return resolved, resolved

        if expression.kind != "call" or expression.argument is None or expression.function_name is None:
            return None

        inner_bounds = self._century_expression_bounds(expression.argument, collection, assignments, domains)
        if inner_bounds is None:
            return None
        inner_low, inner_high = inner_bounds

        if expression.function_name == "century_of":
            low = century_of(inner_low)
            high = century_of(inner_high)
        elif expression.function_name == "century_start":
            low = century_start(inner_low)
            high = century_start(inner_high)
        elif expression.function_name == "century_end":
            low = century_end(inner_low)
            high = century_end(inner_high)
        else:
            return None

        if low is None or high is None:
            return None
        return min(low, high), max(low, high)

    def _apply_century_assignments(
        self,
        collection: EntityCollection,
        assignments: Dict[str, int],
    ) -> EntityCollection:
        updated = collection.model_copy(deep=True)
        for entity_id, value in assignments.items():
            if entity_id.startswith("number_"):
                updated.numbers[entity_id] = self._build_number_entity(entity_id, int(value))
                continue
            if entity_id.startswith("temporal_") and entity_id in updated.temporals:
                updated.temporals[entity_id] = self._update_temporal_year(updated.temporals[entity_id], int(value))
        return updated

    def _update_temporal_year(self, temporal_entity: TemporalEntity, year: int) -> TemporalEntity:
        updated = temporal_entity.model_copy(deep=True)
        updated.year = int(year)
        if updated.date:
            if updated.month:
                day_num = int(updated.day_of_month) if updated.day_of_month is not None else 1
                updated.date = f"{day_num} {updated.month} {year}"
            elif re.search(r"\b\d{4}\b", updated.date):
                updated.date = re.sub(r"\b\d{4}\b", str(year), updated.date, count=1)
            else:
                updated.date = f"{updated.date} {year}".strip()
        return updated

    def _collect_century_forced_equal_ids(
        self,
        collection: EntityCollection,
        *,
        active_entity_ids: Set[str],
        avoid_number_values: Dict[str, int | float | list[int | float] | set[int | float] | tuple[int | float, ...]],
    ) -> Set[str]:
        forced_equal_ids: Set[str] = set()
        for entity_id in active_entity_ids:
            if entity_id.startswith("number_"):
                forbidden_values = self._coerce_forbidden_number_values(avoid_number_values.get(entity_id))
                current_value = getattr(collection.numbers.get(entity_id), "int", None)
                if current_value is not None and current_value in forbidden_values:
                    forced_equal_ids.add(entity_id)
                continue
            if entity_id.startswith("temporal_"):
                factual_year = self._factual_temporal_year(entity_id)
                current_year = getattr(collection.temporals.get(entity_id), "year", None)
                if factual_year is not None and current_year == factual_year:
                    forced_equal_ids.add(entity_id)
        return forced_equal_ids
