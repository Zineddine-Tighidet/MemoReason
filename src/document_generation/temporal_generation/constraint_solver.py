"""Temporal year-constraint parsing and solving helpers."""

from __future__ import annotations

import ast
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from src.core.century_expressions import has_century_function
from src.core.document_schema import EntityCollection
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number

from ..generation_limits import _MIN_YEAR, SLOW_SAMPLING_LOG_SECONDS


class TemporalConstraintSolverMixin:
    """Parse temporal year rules and solve valid year assignments."""

    def _parse_temporal_linear_expr(
        self,
        expr: str,
        required_ids: Set[str],
        existing_entities: Optional[EntityCollection],
    ) -> Optional[Tuple[Dict[str, int], int]]:
        def combine(a: Tuple[Dict[str, int], int], b: Tuple[Dict[str, int], int], sign: int = 1):
            coeffs = dict(a[0])
            for var, coeff in b[0].items():
                coeffs[var] = coeffs.get(var, 0) + sign * coeff
            return coeffs, a[1] + sign * b[1]

        def scale(linear: Tuple[Dict[str, int], int], factor: float) -> Tuple[Dict[str, float], float]:
            coeffs, const = linear
            return {key: value * factor for key, value in coeffs.items()}, const * factor

        def coerce_number_entity_value(number: Any, attr: str) -> Optional[int]:
            if number is None:
                return None
            getter = number.get if isinstance(number, dict) else getattr
            if attr == "str":
                numeric = getter("int", None) if isinstance(number, dict) else getter(number, "int", None)
                if numeric is not None:
                    try:
                        return int(numeric)
                    except (TypeError, ValueError):
                        return None
                raw_text = getter("str", None) if isinstance(number, dict) else getter(number, "str", None)
                if isinstance(raw_text, str):
                    parsed = parse_word_number(raw_text)
                    if parsed is not None:
                        return int(parsed)
                    parsed = parse_integer_surface_number(raw_text)
                    if parsed is not None:
                        return int(parsed)
                return None
            raw_value = getter(attr, None) if isinstance(number, dict) else getter(number, attr, None)
            if raw_value is None:
                return None
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                return None
            if not numeric.is_integer():
                return None
            return int(numeric)

        def resolve_ref(ref: str) -> Optional[int]:
            if not existing_entities or "." not in ref:
                return None
            entity_id, attr = ref.split(".", 1)
            if entity_id in required_ids:
                return None
            if entity_id.startswith("temporal_"):
                temporal = existing_entities.temporals.get(entity_id)
                return self._temporal_year_from_entity(temporal)
            if entity_id.startswith("number_"):
                number = existing_entities.numbers.get(entity_id)
                if number is None:
                    return None
                return coerce_number_entity_value(number, attr)
            if entity_id.startswith("person_") and attr == "age":
                person = existing_entities.persons.get(entity_id)
                if person is None:
                    return None
                raw_age = person.get("age", None) if isinstance(person, dict) else getattr(person, "age", None)
                if raw_age is None:
                    return None
                try:
                    return int(raw_age)
                except (TypeError, ValueError):
                    return None
            return None

        def to_linear(node) -> Optional[Tuple[Dict[str, int], int]]:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return {}, int(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                inner = to_linear(node.operand)
                if inner is None:
                    return None
                coeffs, const = inner
                if isinstance(node.op, ast.USub):
                    return {k: -v for k, v in coeffs.items()}, -const
                return coeffs, const
            if isinstance(node, ast.Name):
                if node.id in required_ids:
                    return {node.id: 1}, 0
                parsed_word = parse_word_number(node.id)
                if parsed_word is not None:
                    return {}, int(parsed_word)
                parsed_integer = parse_integer_surface_number(node.id)
                if parsed_integer is not None:
                    return {}, int(parsed_integer)
                return None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                entity_id = node.value.id
                attr = node.attr
                if entity_id.startswith("temporal_"):
                    if attr not in {"year", "date"}:
                        return None
                    if entity_id in required_ids:
                        return {entity_id: 1}, 0
                    resolved = resolve_ref(f"{entity_id}.{attr}")
                    if resolved is not None:
                        return {}, resolved
                    return None
                if entity_id.startswith("number_"):
                    if attr not in {"int", "float", "percent", "proportion", "str"}:
                        return None
                    resolved = resolve_ref(f"{entity_id}.{attr}")
                    if resolved is not None:
                        return {}, resolved
                    return None
                if entity_id.startswith("person_"):
                    if attr != "age":
                        return None
                    resolved = resolve_ref(f"{entity_id}.{attr}")
                    if resolved is not None:
                        return {}, resolved
                    return None
                return None
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                left = to_linear(node.left)
                right = to_linear(node.right)
                if left is None or right is None:
                    return None
                sign = 1 if isinstance(node.op, ast.Add) else -1
                return combine(left, right, sign=sign)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                left = to_linear(node.left)
                right = to_linear(node.right)
                if left is None or right is None:
                    return None
                if left[0] and right[0]:
                    return None
                if left[0]:
                    return scale(left, right[1])
                if right[0]:
                    return scale(right, left[1])
                return {}, left[1] * right[1]
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
                left = to_linear(node.left)
                right = to_linear(node.right)
                if left is None or right is None:
                    return None
                if right[0] or right[1] == 0:
                    return None
                return scale(left, 1.0 / right[1])
            return None

        try:
            tree = ast.parse(expr, mode="eval").body
        except SyntaxError:
            return None
        return to_linear(tree)

    def _solve_temporal_years(
        self,
        required_temporals: List[tuple],
        rules: List[str],
        existing_entities: Optional[EntityCollection],
        excluded_years: Set[int],
        decade_year_temporal_ids: Set[str],
    ) -> Optional[Dict[str, int]]:
        constraints = self._collect_temporal_year_constraints(required_temporals, rules, existing_entities)
        if constraints is None:
            return None
        constraints.extend(self._ordering_temporal_year_constraints(required_temporals, existing_entities))
        if not constraints:
            return None

        required_ids = {
            tid
            for tid, attrs in required_temporals
            if any(attr in attrs for attr in ("year", "date"))
        }
        if not required_ids:
            return {}
        domains: Dict[str, List[int]] = {}
        for tid in required_ids:
            base_lo, base_hi = self._temporal_year_base_range(tid)
            domain = self._temporal_year_domain(
                tid,
                base_lo,
                base_hi,
                excluded_years,
                decade_year_temporal_ids,
            )
            if not domain:
                _, expanded_max_year = self._temporal_year_sampling_bounds(tid)
                domain = self._expand_temporal_year_domain(
                    tid,
                    _MIN_YEAR,
                    expanded_max_year,
                    excluded_years,
                    decade_year_temporal_ids,
                )
            if not domain:
                return None
            domains[tid] = domain
        domain_bounds = {tid: (min(values), max(values)) for tid, values in domains.items()}

        milp_solution = self._solve_temporal_years_via_milp(
            constraints,
            domains,
            domain_bounds,
            decade_year_temporal_ids,
        )
        if milp_solution is not None:
            return milp_solution

        def feasible(assignments: Dict[str, int]) -> bool:
            return self._constraints_feasible(constraints, assignments, domain_bounds)

        def choose_var(assignments: Dict[str, int]) -> Optional[str]:
            remaining = [tid for tid in required_ids if tid not in assignments]
            if not remaining:
                return None
            random.shuffle(remaining)
            return min(remaining, key=lambda tid: len(domains[tid]))

        steps = 0
        start_time = time.monotonic()
        randomized_step_budget = 1000 if len(required_ids) > 8 else 5000
        randomized_time_budget = 0.15 if len(required_ids) > 8 else 0.75

        def backtrack(assignments: Dict[str, int]) -> Optional[Dict[str, int]]:
            nonlocal steps
            steps += 1
            if randomized_step_budget is not None and steps > randomized_step_budget:
                return None
            if randomized_time_budget is not None and (time.monotonic() - start_time) > randomized_time_budget:
                return None
            if steps % 2000 == 0 and (time.monotonic() - start_time) > SLOW_SAMPLING_LOG_SECONDS:
                print(
                    f"[slow] Temporal solver still searching ({steps} steps, {time.monotonic() - start_time:.1f}s, {len(required_ids)} vars).",
                    flush=True,
                )
            var = choose_var(assignments)
            if var is None:
                return dict(assignments)
            candidates = list(domains[var])
            random.shuffle(candidates)
            for val in candidates:
                assignments[var] = val
                if feasible(assignments):
                    solved = backtrack(assignments)
                    if solved is not None:
                        return solved
                assignments.pop(var, None)
            return None

        solved = backtrack({})
        if solved is not None:
            return solved
        return None

    def _solve_temporal_years_via_milp(
        self,
        constraints: List[Tuple[Tuple[Dict[str, int], int], str, Tuple[Dict[str, int], int]]],
        domains: Dict[str, List[int]],
        domain_bounds: Dict[str, Tuple[int, int]],
        decade_year_temporal_ids: Set[str],
    ) -> Optional[Dict[str, int]]:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
        except Exception:
            return None

        ordered_ids = sorted(domains)
        index_by_id = {tid: idx for idx, tid in enumerate(ordered_ids)}
        decade_ids = [tid for tid in ordered_ids if tid in decade_year_temporal_ids]
        decade_index = {tid: len(ordered_ids) + idx for idx, tid in enumerate(decade_ids)}
        forbidden_pairs: list[tuple[str, int]] = []
        for tid in ordered_ids:
            low, high = domain_bounds[tid]
            allowed_years = set(int(year) for year in domains[tid])
            if tid in decade_year_temporal_ids:
                candidate_years = {year for year in range(low, high + 1) if year % 10 == 0}
            else:
                candidate_years = set(range(low, high + 1))
            forbidden_pairs.extend((tid, year) for year in sorted(candidate_years - allowed_years))
        forbidden_index = {pair: len(ordered_ids) + len(decade_ids) + idx for idx, pair in enumerate(forbidden_pairs)}
        not_equal_specs: list[tuple[Dict[str, float], float, float]] = []
        for lhs, op, rhs in constraints:
            if op != "!=":
                continue
            coeffs: Dict[str, float] = {}
            for var, coeff in lhs[0].items():
                coeffs[var] = coeffs.get(var, 0.0) + float(coeff)
            for var, coeff in rhs[0].items():
                coeffs[var] = coeffs.get(var, 0.0) - float(coeff)
            if not coeffs:
                if lhs[1] == rhs[1]:
                    return None
                continue
            expr_min = 0.0
            expr_max = 0.0
            for var, coeff in coeffs.items():
                low, high = domain_bounds[var]
                if coeff >= 0:
                    expr_min += coeff * low
                    expr_max += coeff * high
                else:
                    expr_min += coeff * high
                    expr_max += coeff * low
            rhs_value = float(rhs[1] - lhs[1])
            if rhs_value < expr_min - 1e-9 or rhs_value > expr_max + 1e-9:
                continue
            if abs(expr_min - expr_max) <= 1e-9 and abs(expr_min - rhs_value) <= 1e-9:
                return None
            slack = max(abs(expr_max - rhs_value), abs(rhs_value - expr_min)) + 1.0
            not_equal_specs.append((coeffs, rhs_value, slack))
        not_equal_index = {
            spec_index: len(ordered_ids) + len(decade_ids) + len(forbidden_pairs) + spec_index
            for spec_index in range(len(not_equal_specs))
        }

        total_vars = len(ordered_ids) + len(decade_ids) + len(forbidden_pairs) + len(not_equal_specs)
        if total_vars == 0:
            return {}
        if len(ordered_ids) > 8:
            return None
        if total_vars > 120 or len(forbidden_pairs) > 80:
            return None

        c = np.zeros(total_vars, dtype=float)
        integrality = np.ones(total_vars, dtype=int)
        lower = np.full(total_vars, -np.inf, dtype=float)
        upper = np.full(total_vars, np.inf, dtype=float)

        for tid, idx in index_by_id.items():
            low, high = domain_bounds[tid]
            lower[idx] = low
            upper[idx] = high
            c[idx] = random.uniform(-1e-3, 1e-3)

        for tid, idx in decade_index.items():
            low, high = domain_bounds[tid]
            lower[idx] = int((low + 9) // 10)
            upper[idx] = int(high // 10)

        for idx in forbidden_index.values():
            lower[idx] = 0
            upper[idx] = 1
        for idx in not_equal_index.values():
            lower[idx] = 0
            upper[idx] = 1

        rows = []
        lbs = []
        ubs = []

        for lhs, op, rhs in constraints:
            row = np.zeros(total_vars, dtype=float)
            coeffs: Dict[str, int] = {}
            for var, coeff in lhs[0].items():
                coeffs[var] = coeffs.get(var, 0) + coeff
            for var, coeff in rhs[0].items():
                coeffs[var] = coeffs.get(var, 0) - coeff
            for var, coeff in coeffs.items():
                row[index_by_id[var]] = coeff
            rhs_value = rhs[1] - lhs[1]

            if op in ("=", "=="):
                lb = ub = rhs_value
            elif op == "<":
                lb = -np.inf
                ub = rhs_value - 1
            elif op == "<=":
                lb = -np.inf
                ub = rhs_value
            elif op == ">":
                lb = rhs_value + 1
                ub = np.inf
            elif op == ">=":
                lb = rhs_value
                ub = np.inf
            elif op == "!=":
                continue
            else:
                return None

            rows.append(row)
            lbs.append(lb)
            ubs.append(ub)

        for tid in decade_ids:
            row = np.zeros(total_vars, dtype=float)
            row[index_by_id[tid]] = 1
            row[decade_index[tid]] = -10
            rows.append(row)
            lbs.append(0)
            ubs.append(0)

        for tid, forbidden_year in forbidden_pairs:
            low, high = domain_bounds[tid]
            if low == high == forbidden_year:
                return None
            slack = max(1, high - low + 1)
            binary_idx = forbidden_index[(tid, forbidden_year)]

            upper_cut = np.zeros(total_vars, dtype=float)
            upper_cut[index_by_id[tid]] = 1
            upper_cut[binary_idx] = -slack
            rows.append(upper_cut)
            lbs.append(-np.inf)
            ubs.append(forbidden_year - 1)

            lower_cut = np.zeros(total_vars, dtype=float)
            lower_cut[index_by_id[tid]] = 1
            lower_cut[binary_idx] = -slack
            rows.append(lower_cut)
            lbs.append(forbidden_year + 1 - slack)
            ubs.append(np.inf)

        epsilon = 1e-6
        for spec_index, (coeffs, rhs_value, slack) in enumerate(not_equal_specs):
            binary_idx = not_equal_index[spec_index]
            upper_cut = np.zeros(total_vars, dtype=float)
            for var, coeff in coeffs.items():
                upper_cut[index_by_id[var]] = coeff
            upper_cut[binary_idx] = -slack
            rows.append(upper_cut)
            lbs.append(-np.inf)
            ubs.append(rhs_value - epsilon)

            lower_cut = np.zeros(total_vars, dtype=float)
            for var, coeff in coeffs.items():
                lower_cut[index_by_id[var]] = coeff
            lower_cut[binary_idx] = slack
            rows.append(lower_cut)
            lbs.append(rhs_value + epsilon - slack)
            ubs.append(np.inf)

        linear_constraints = []
        if rows:
            linear_constraints.append(
                LinearConstraint(np.vstack(rows), np.array(lbs, dtype=float), np.array(ubs, dtype=float))
            )

        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(lower, upper),
            constraints=linear_constraints,
            options={"time_limit": 2.0, "disp": False},
        )
        if not getattr(result, "success", False) or result.x is None:
            return None

        assignments = {tid: int(round(float(result.x[index_by_id[tid]]))) for tid in ordered_ids}
        if any(assignments[tid] not in domains[tid] for tid in ordered_ids):
            return None
        if not self._constraints_feasible(
            constraints,
            assignments,
            {tid: (value, value) for tid, value in assignments.items()},
        ):
            return None
        return assignments

    def _collect_temporal_year_constraints(
        self,
        required_temporals: List[tuple],
        rules: List[str],
        existing_entities: Optional[EntityCollection],
    ) -> Optional[List[Tuple[Tuple[Dict[str, int], int], str, Tuple[Dict[str, int], int]]]]:
        required_ids = {tid for tid, _ in required_temporals}
        constraints: List[Tuple[Tuple[Dict[str, int], int], str, Tuple[Dict[str, int], int]]] = []
        for raw in rules:
            cleaned = self._strip_rule_comment(str(raw))
            if "temporal_" not in cleaned:
                continue
            if has_century_function(cleaned):
                continue
            split = self._split_rule(cleaned)
            if not split:
                return None
            lhs, op, rhs = split
            if op not in ("<", "<=", ">", ">=", "=", "==", "!="):
                continue
            left = self._parse_temporal_linear_expr(lhs, required_ids, existing_entities)
            right = self._parse_temporal_linear_expr(rhs, required_ids, existing_entities)
            if left is None or right is None:
                return None
            constraints.append((left, op, right))

        return constraints

    def _ordering_temporal_year_constraints(
        self,
        required_temporals: List[tuple],
        existing_entities: Optional[EntityCollection],
    ) -> List[Tuple[Tuple[Dict[str, int], int], str, Tuple[Dict[str, int], int]]]:
        required_ids = [
            temporal_id for temporal_id, attrs in required_temporals if any(attr in attrs for attr in ("year", "date"))
        ]
        factual_years = {
            temporal_id: factual_year
            for temporal_id in required_ids
            if (factual_year := self._factual_temporal_year(temporal_id)) is not None
        }
        constraints: List[Tuple[Tuple[Dict[str, int], int], str, Tuple[Dict[str, int], int]]] = []

        grouped_required: list[tuple[int, list[str]]] = []
        for temporal_id, factual_year in sorted(factual_years.items(), key=lambda item: (item[1], item[0])):
            if grouped_required and grouped_required[-1][0] == factual_year:
                grouped_required[-1][1].append(temporal_id)
                continue
            grouped_required.append((factual_year, [temporal_id]))
        for (_left_year, left_ids), (_right_year, right_ids) in zip(grouped_required, grouped_required[1:], strict=False):
            for left_id in left_ids:
                for right_id in right_ids:
                    constraints.append((({left_id: 1}, 0), "<", ({right_id: 1}, 0)))

        if existing_entities and existing_entities.temporals:
            kept_years = sorted(
                (
                    other_id,
                    other_year,
                )
                for other_id, other_temporal in existing_entities.temporals.items()
                if other_id not in factual_years
                if (other_year := self._temporal_year_from_entity(other_temporal)) is not None
            )
            for temporal_id in required_ids:
                factual_year = factual_years.get(temporal_id)
                if factual_year is None:
                    continue
                lower_neighbor = max(
                    (other_year for _other_id, other_year in kept_years if other_year < factual_year),
                    default=None,
                )
                upper_neighbor = min(
                    (other_year for _other_id, other_year in kept_years if other_year > factual_year),
                    default=None,
                )
                if lower_neighbor is not None:
                    constraints.append((({temporal_id: 1}, 0), ">", ({}, int(lower_neighbor))))
                if upper_neighbor is not None:
                    constraints.append((({temporal_id: 1}, 0), "<", ({}, int(upper_neighbor))))

        return constraints

    def _temporal_assignments_satisfy_constraints(
        self,
        assignments: Dict[str, int],
        required_temporals: List[tuple],
        rules: List[str],
        existing_entities: Optional[EntityCollection],
    ) -> bool:
        required_ids = {
            tid
            for tid, attrs in required_temporals
            if any(attr in attrs for attr in ("year", "date"))
        }
        if any(tid not in assignments for tid in required_ids):
            return False
        constraints = self._collect_temporal_year_constraints(required_temporals, rules, existing_entities)
        if constraints is None:
            return False
        if not constraints:
            return True
        exact_domains = {tid: (value, value) for tid, value in assignments.items()}
        return self._constraints_feasible(constraints, assignments, exact_domains)
