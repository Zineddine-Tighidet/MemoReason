"""Rule parsing and solving utilities for number generation."""

import math
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.core.century_expressions import has_century_function
from src.core.document_schema import EntityCollection, NumberEntity
from src.core.annotation_runtime import RuleEngine, find_entity_refs
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number
from ..generation_limits import (
    MAX_CSP_NUMBER_VARS,
    MAX_NUMBER_CSP_SECONDS,
    MAX_NUMBER_CSP_STEPS,
    SLOW_SAMPLING_LOG_SECONDS,
    _DEFAULT_NUMBER_MIN,
)

LinearExpr = Tuple[Dict[str, float], float]
LinearConstraintTriple = Tuple[LinearExpr, str, LinearExpr]
_MILP_OPTIONS = {"time_limit": 2.0, "disp": False}


class NumberConstraintSolverMixin:
    """Parses number rules and computes integer assignments satisfying constraints."""

    @staticmethod
    def _strip_rule_comment(rule: str) -> str:
        return rule.split("#", 1)[0].strip()

    @staticmethod
    def _normalize_number_rule(rule: str) -> str:
        normalized = rule
        for attr in (".int", ".str", ".float", ".percent", ".proportion"):
            normalized = normalized.replace(attr, "")
        return normalized

    @staticmethod
    def _split_rule(rule: str) -> Optional[Tuple[str, str, str]]:
        for op in ("<=", ">=", "==", "!=", "=", "<", ">"):
            if op in rule:
                parts = rule.split(op)
                if len(parts) == 2:
                    return parts[0].strip(), op, parts[1].strip()
        return None

    @staticmethod
    def _number_token(expr: str) -> Optional[str]:
        cleaned = expr.strip()
        return cleaned if re.fullmatch(r"number_\d+", cleaned) else None

    @staticmethod
    def _int_literal(expr: str) -> Optional[int]:
        cleaned = expr.strip()
        if re.fullmatch(r"-?\d+", cleaned):
            return int(cleaned)
        return None

    def _number_rule_is_evaluable(
        self,
        rule: str,
        variables: Set[str],
        existing_entities: Optional[EntityCollection],
    ) -> bool:
        refs = find_entity_refs(rule)
        if not refs:
            return False
        has_target_number = False
        for ref in refs:
            base_ref = ref.split(".", 1)[0]
            if base_ref in variables:
                has_target_number = True
                continue
            if existing_entities and RuleEngine._get_entity_value(existing_entities, ref) is not None:
                continue
            return False
        return has_target_number

    def _number_evaluable_rules(
        self,
        rules: List[str],
        variables: Set[str],
        existing_entities: Optional[EntityCollection],
    ) -> List[str]:
        selected: List[str] = []
        for raw_rule in rules:
            cleaned = self._strip_rule_comment(str(raw_rule))
            if not cleaned:
                continue
            if self._number_rule_is_evaluable(cleaned, variables, existing_entities):
                selected.append(str(raw_rule))
        return selected

    def _ordering_number_rules(
        self,
        number_ids: List[str],
        existing_entities: Optional[EntityCollection],
    ) -> List[str]:
        non_integer_ids = [
            number_id for number_id in number_ids if self._factual_number_kind(number_id) not in {"int", "fraction"}
        ]
        if len(non_integer_ids) > 1:
            return []
        ordered: list[tuple[float, str]] = []
        excluded_ids = set(getattr(self, "ordering_excluded_number_ids", set()) or set())
        for number_id in number_ids:
            if number_id in excluded_ids:
                continue
            if self._factual_number_kind(number_id) not in {"int", "fraction"}:
                continue
            factual_entity = self._factual_number_entity(number_id)
            factual_value = self._number_actual_value(factual_entity) if factual_entity is not None else None
            if factual_value is None:
                continue
            ordered.append((factual_value, number_id))
        ordered.sort(key=lambda item: (item[0], item[1]))
        groups: list[tuple[float, list[str]]] = []
        for factual_value, number_id in ordered:
            if not groups or not math.isclose(groups[-1][0], factual_value, abs_tol=1e-9):
                groups.append((factual_value, [number_id]))
                continue
            groups[-1][1].append(number_id)

        rules: list[str] = []
        for left_group_index in range(len(groups) - 1):
            _left_value, left_ids = groups[left_group_index]
            for right_group_index in range(left_group_index + 1, len(groups)):
                _right_value, right_ids = groups[right_group_index]
                for left_id in left_ids:
                    for right_id in right_ids:
                        rules.append(f"{left_id} < {right_id}")
        return rules

    def _parse_linear_expr(
        self,
        expr: str,
        variables: Set[str],
        existing_entities: Optional[EntityCollection],
    ) -> Optional[LinearExpr]:
        import ast

        def resolve_ref(ref: str) -> Optional[float]:
            if not existing_entities:
                return None
            val = RuleEngine._get_entity_value(existing_entities, ref)
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                return None

        def combine(a: LinearExpr, b: LinearExpr, sign: int = 1) -> LinearExpr:
            coeffs = dict(a[0])
            for var, coeff in b[0].items():
                coeffs[var] = coeffs.get(var, 0) + sign * coeff
            return coeffs, a[1] + sign * b[1]

        def scale(linear: LinearExpr, factor: float) -> LinearExpr:
            coeffs, const = linear
            return {key: value * factor for key, value in coeffs.items()}, const * factor

        def to_linear(node) -> Optional[LinearExpr]:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return {}, float(node.value)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                parsed_number = parse_integer_surface_number(node.value)
                if parsed_number is None:
                    parsed_number = parse_word_number(node.value)
                if parsed_number is not None:
                    return {}, float(parsed_number)
                return None
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                inner = to_linear(node.operand)
                if inner is None:
                    return None
                coeffs, const = inner
                if isinstance(node.op, ast.USub):
                    return {k: -v for k, v in coeffs.items()}, -const
                return coeffs, const
            if isinstance(node, ast.Name):
                if node.id in variables:
                    return {node.id: 1}, 0
                parsed_word = parse_word_number(node.id)
                if parsed_word is not None:
                    return {}, float(parsed_word)
                resolved = resolve_ref(node.id)
                if resolved is not None:
                    return {}, resolved
                return None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                base = node.value.id
                attr = node.attr
                if base in variables and attr in {"int", "float", "percent", "proportion"}:
                    return {base: 1}, 0
                ref = f"{base}.{attr}"
                resolved = resolve_ref(ref)
                if resolved is not None:
                    return {}, resolved
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
                if right[0]:
                    return None
                if right[1] == 0:
                    return None
                return scale(left, 1.0 / right[1])
            return None

        try:
            tree = ast.parse(expr, mode="eval").body
        except SyntaxError:
            return None
        return to_linear(tree)

    def _evaluate_linear_expr(
        self,
        linear: LinearExpr,
        assigned: Dict[str, int],
        existing_entities: Optional[EntityCollection],
    ) -> Optional[float]:
        coeffs, const = linear
        total = float(const)
        for var, coeff in coeffs.items():
            if var in assigned:
                total += coeff * float(assigned[var])
                continue
            if existing_entities:
                resolved = RuleEngine._get_entity_value(existing_entities, var)
                if resolved is not None:
                    try:
                        total += coeff * float(resolved)
                        continue
                    except (TypeError, ValueError):
                        return None
            return None
        return total

    def _collect_linear_constraints(
        self,
        rules: List[str],
        variables: Set[str],
        existing_entities: Optional[EntityCollection],
    ) -> Optional[List[LinearConstraintTriple]]:
        constraints: List[LinearConstraintTriple] = []
        for raw_rule in self._number_evaluable_rules(rules, variables, existing_entities):
            rule = self._normalize_number_rule(self._strip_rule_comment(str(raw_rule)))
            if not rule:
                continue
            if has_century_function(rule):
                continue
            split = self._split_rule(rule)
            if not split:
                return None
            lhs, op, rhs = split
            lhs_lin = self._parse_linear_expr(lhs, variables, existing_entities)
            rhs_lin = self._parse_linear_expr(rhs, variables, existing_entities)
            if lhs_lin is None or rhs_lin is None:
                return None
            constraints.append((lhs_lin, op, rhs_lin))
        return constraints

    @staticmethod
    def _expr_bounds(
        expr: Tuple[Dict[str, int], int],
        assignments: Dict[str, int],
        domains: Dict[str, Tuple[int, int]],
    ) -> Tuple[float, float]:
        coeffs, const = expr
        min_val = const
        max_val = const
        for var, coeff in coeffs.items():
            if var in assignments:
                vmin = vmax = assignments[var]
            else:
                vmin, vmax = domains[var]
            if coeff >= 0:
                min_val += coeff * vmin
                max_val += coeff * vmax
            else:
                min_val += coeff * vmax
                max_val += coeff * vmin
        return min_val, max_val

    def _constraints_feasible(
        self,
        constraints: List[LinearConstraintTriple],
        assignments: Dict[str, int],
        domains: Dict[str, Tuple[int, int]],
    ) -> bool:
        eps = 1e-9
        for lhs, op, rhs in constraints:
            lhs_min, lhs_max = self._expr_bounds(lhs, assignments, domains)
            rhs_min, rhs_max = self._expr_bounds(rhs, assignments, domains)
            if op in ("=", "=="):
                if lhs_max < rhs_min - eps or rhs_max < lhs_min - eps:
                    return False
                if (
                    math.isclose(lhs_min, lhs_max, abs_tol=eps)
                    and math.isclose(rhs_min, rhs_max, abs_tol=eps)
                    and not math.isclose(lhs_min, rhs_min, abs_tol=eps)
                ):
                    return False
            elif op == "<":
                if lhs_min >= rhs_max - eps:
                    return False
            elif op == "<=":
                if lhs_min > rhs_max + eps:
                    return False
            elif op == ">":
                if lhs_max <= rhs_min + eps:
                    return False
            elif op == ">=":
                if lhs_max < rhs_min - eps:
                    return False
            elif op == "!=":
                if (
                    math.isclose(lhs_min, lhs_max, abs_tol=eps)
                    and math.isclose(rhs_min, rhs_max, abs_tol=eps)
                    and math.isclose(lhs_min, rhs_min, abs_tol=eps)
                ):
                    return False
        return True

    def _solve_numbers_via_constraints(
        self,
        number_ids: List[str],
        rules: List[str],
        existing_entities: Optional[EntityCollection],
        avoid_values: Dict[str, int],
    ) -> Optional[Dict[str, int]]:
        if not number_ids:
            return {}
        self.last_relaxed_avoid_number_ids = set()
        self.last_number_solution_used_relaxed_avoid = False
        variables = set(number_ids)
        constraints = self._collect_linear_constraints(rules, variables, existing_entities)
        if constraints is None:
            return None

        domains: Dict[str, Tuple[int, int]] = {vid: self._number_base_range(vid) for vid in number_ids}
        domains = self._tighten_number_domains(constraints, domains)
        if domains is None:
            return None
        for _var, (lo, hi) in domains.items():
            if lo > hi:
                return None

        milp_solution = self._solve_numbers_via_milp(constraints, domains, avoid_values)
        if milp_solution is not None:
            return milp_solution

        if len(number_ids) > MAX_CSP_NUMBER_VARS:
            if avoid_values:
                relaxed_solution = self._solve_numbers_via_relaxed_avoid_milp(constraints, domains, avoid_values)
                if relaxed_solution is not None:
                    self.last_number_solution_used_relaxed_avoid = True
                    return relaxed_solution
            return None

        unconstrained = set(number_ids)
        for lhs, _, rhs in constraints:
            unconstrained -= set(lhs[0].keys())
            unconstrained -= set(rhs[0].keys())

        assignments: Dict[str, int] = {}
        for var in sorted(unconstrained):
            lo, hi = domains[var]
            if lo > hi:
                return None
            avoid = avoid_values.get(var)
            assignments[var] = self._sample_int_in_range(lo, hi, avoid=avoid)

        def pick_next_var() -> Optional[str]:
            remaining = [v for v in number_ids if v not in assignments]
            if not remaining:
                return None
            return min(remaining, key=lambda v: domains[v][1] - domains[v][0])

        steps = 0
        start_time = time.monotonic()

        def backtrack() -> Optional[Dict[str, int]]:
            nonlocal steps
            steps += 1
            if steps > MAX_NUMBER_CSP_STEPS:
                return None
            if (time.monotonic() - start_time) > MAX_NUMBER_CSP_SECONDS:
                return None
            if steps % 5000 == 0 and (time.monotonic() - start_time) > SLOW_SAMPLING_LOG_SECONDS:
                print(
                    f"[slow] Number CSP still searching ({steps} steps, {time.monotonic() - start_time:.1f}s, {len(number_ids)} vars).",
                    flush=True,
                )
            var = pick_next_var()
            if var is None:
                return dict(assignments)
            lo, hi = domains[var]
            if lo > hi:
                return None
            avoid = avoid_values.get(var)
            candidates = self._candidate_values(lo, hi, avoid=avoid)
            for val in candidates:
                assignments[var] = val
                if self._constraints_feasible(constraints, assignments, domains):
                    solved = backtrack()
                    if solved is not None:
                        return solved
                assignments.pop(var, None)
            return None

        csp_solution = backtrack()
        if csp_solution is not None:
            return csp_solution
        if avoid_values:
            relaxed_solution = self._solve_numbers_via_relaxed_avoid_milp(constraints, domains, avoid_values)
            if relaxed_solution is not None:
                self.last_number_solution_used_relaxed_avoid = True
                return relaxed_solution
        return None

    def _solve_numbers_via_relaxed_avoid_milp(
        self,
        constraints: List[LinearConstraintTriple],
        domains: Dict[str, Tuple[int, int]],
        avoid_values: Dict[str, int],
    ) -> Optional[Dict[str, int]]:
        ordered_ids = sorted(domains)
        index_by_id = {number_id: idx for idx, number_id in enumerate(ordered_ids)}
        relaxed_pairs = [
            (number_id, forbidden_value)
            for number_id in ordered_ids
            for forbidden_value in sorted(self._coerce_forbidden_number_values(avoid_values.get(number_id)))
            if domains[number_id][0] <= forbidden_value <= domains[number_id][1]
            and domains[number_id][0] != domains[number_id][1]
        ]
        if not relaxed_pairs:
            return None

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
                if math.isclose(lhs[1], rhs[1], abs_tol=1e-9):
                    return None
                continue
            expr_min = 0.0
            expr_max = 0.0
            for var, coeff in coeffs.items():
                low, high = domains[var]
                if coeff >= 0:
                    expr_min += coeff * low
                    expr_max += coeff * high
                else:
                    expr_min += coeff * high
                    expr_max += coeff * low
            rhs_value = float(rhs[1] - lhs[1])
            if rhs_value < expr_min - 1e-9 or rhs_value > expr_max + 1e-9:
                continue
            if math.isclose(expr_min, expr_max, abs_tol=1e-9) and math.isclose(expr_min, rhs_value, abs_tol=1e-9):
                return None
            slack = max(abs(expr_max - rhs_value), abs(rhs_value - expr_min)) + 1.0
            not_equal_specs.append((coeffs, rhs_value, slack))

        relaxed_index = {
            forbidden_pair: (len(ordered_ids) + (2 * idx), len(ordered_ids) + (2 * idx) + 1)
            for idx, forbidden_pair in enumerate(relaxed_pairs)
        }
        not_equal_index = {
            spec_index: len(ordered_ids) + (2 * len(relaxed_pairs)) + spec_index
            for spec_index in range(len(not_equal_specs))
        }

        total_vars = len(ordered_ids) + (2 * len(relaxed_pairs)) + len(not_equal_specs)
        if total_vars == 0:
            return {}

        c = np.zeros(total_vars, dtype=float)
        integrality = np.ones(total_vars, dtype=int)
        lower = np.full(total_vars, -np.inf, dtype=float)
        upper = np.full(total_vars, np.inf, dtype=float)

        for number_id, idx in index_by_id.items():
            low, high = domains[number_id]
            lower[idx] = low
            upper[idx] = high

        for below_idx, above_idx in relaxed_index.values():
            lower[below_idx] = 0
            upper[below_idx] = 1
            lower[above_idx] = 0
            upper[above_idx] = 1
            c[below_idx] = -1.0
            c[above_idx] = -1.0

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

        for number_id, forbidden_value in relaxed_pairs:
            low, high = domains[number_id]
            slack = max(1, high - low + 1)
            below_idx, above_idx = relaxed_index[(number_id, forbidden_value)]

            below_cut = np.zeros(total_vars, dtype=float)
            below_cut[index_by_id[number_id]] = 1
            below_cut[below_idx] = slack
            rows.append(below_cut)
            lbs.append(-np.inf)
            ubs.append(forbidden_value - 1 + slack)

            above_cut = np.zeros(total_vars, dtype=float)
            above_cut[index_by_id[number_id]] = 1
            above_cut[above_idx] = -slack
            rows.append(above_cut)
            lbs.append(forbidden_value + 1 - slack)
            ubs.append(np.inf)

            side_limit = np.zeros(total_vars, dtype=float)
            side_limit[below_idx] = 1
            side_limit[above_idx] = 1
            rows.append(side_limit)
            lbs.append(-np.inf)
            ubs.append(1.0)

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
            options=_MILP_OPTIONS,
        )
        if not getattr(result, "success", False) or result.x is None:
            return None

        assignments = {number_id: int(round(float(result.x[index_by_id[number_id]]))) for number_id in ordered_ids}
        if not self._constraints_feasible(
            constraints,
            assignments,
            {number_id: (value, value) for number_id, value in assignments.items()},
        ):
            return None
        for number_id, value in assignments.items():
            low, high = domains[number_id]
            if value < low or value > high:
                return None

        forced_equal_ids: set[str] = set()
        for number_id in ordered_ids:
            if assignments[number_id] not in self._coerce_forbidden_number_values(avoid_values.get(number_id)):
                continue
            low, high = domains[number_id]
            if low == high == assignments[number_id]:
                forced_equal_ids.add(number_id)
        self.last_relaxed_avoid_number_ids = forced_equal_ids
        return assignments

    def _tighten_number_domains(
        self,
        constraints: List[LinearConstraintTriple],
        domains: Dict[str, Tuple[int, int]],
    ) -> Optional[Dict[str, Tuple[int, int]]]:
        tightened = dict(domains)

        def update(var: str, lo: int, hi: int) -> bool:
            current_lo, current_hi = tightened[var]
            new_lo = max(current_lo, lo)
            new_hi = min(current_hi, hi)
            if new_lo > new_hi:
                raise ValueError(f"Infeasible number domain for {var}: [{new_lo}, {new_hi}]")
            if (new_lo, new_hi) != (current_lo, current_hi):
                tightened[var] = (new_lo, new_hi)
                return True
            return False

        try:
            changed = True
            while changed:
                changed = False
                for lhs, op, rhs in constraints:
                    lhs_vars = list(lhs[0].items())
                    rhs_vars = list(rhs[0].items())
                    if len(lhs_vars) == 1 and lhs_vars[0][1] == 1 and not rhs[0]:
                        var = lhs_vars[0][0]
                        bound = rhs[1] - lhs[1]
                        if op in ("<", "<="):
                            changed |= update(
                                var,
                                tightened[var][0],
                                math.floor(bound - (1 if op == "<" else 0)),
                            )
                        elif op in (">", ">="):
                            changed |= update(
                                var,
                                math.ceil(bound + (1 if op == ">" else 0)),
                                tightened[var][1],
                            )
                        elif op in ("=", "=="):
                            integral_bound = int(round(bound))
                            if not math.isclose(bound, integral_bound, abs_tol=1e-9):
                                continue
                            changed |= update(var, integral_bound, integral_bound)
                        continue
                    if len(rhs_vars) == 1 and rhs_vars[0][1] == 1 and not lhs[0]:
                        var = rhs_vars[0][0]
                        bound = lhs[1] - rhs[1]
                        if op in ("<", "<="):
                            changed |= update(
                                var,
                                math.ceil(bound + (1 if op == "<" else 0)),
                                tightened[var][1],
                            )
                        elif op in (">", ">="):
                            changed |= update(
                                var,
                                tightened[var][0],
                                math.floor(bound - (1 if op == ">" else 0)),
                            )
                        elif op in ("=", "=="):
                            integral_bound = int(round(bound))
                            if not math.isclose(bound, integral_bound, abs_tol=1e-9):
                                continue
                            changed |= update(var, integral_bound, integral_bound)
                        continue
                    if len(lhs_vars) == 1 and lhs_vars[0][1] == 1 and len(rhs_vars) == 1 and rhs_vars[0][1] == 1:
                        left_var = lhs_vars[0][0]
                        right_var = rhs_vars[0][0]
                        delta = rhs[1] - lhs[1]
                        strict = 1 if op in ("<", ">") else 0
                        left_lo, left_hi = tightened[left_var]
                        right_lo, right_hi = tightened[right_var]
                        if op in ("<", "<="):
                            changed |= update(left_var, left_lo, math.floor(right_hi + delta - strict))
                            changed |= update(right_var, math.ceil(left_lo - delta + strict), right_hi)
                        elif op in (">", ">="):
                            changed |= update(left_var, math.ceil(right_lo + delta + strict), left_hi)
                            changed |= update(right_var, right_lo, math.floor(left_hi - delta - strict))
                        elif op in ("=", "=="):
                            changed |= update(left_var, math.ceil(right_lo + delta), math.floor(right_hi + delta))
                            changed |= update(right_var, math.ceil(left_lo - delta), math.floor(left_hi - delta))
        except ValueError:
            return None
        return tightened

    def _solve_numbers_via_milp(
        self,
        constraints: List[LinearConstraintTriple],
        domains: Dict[str, Tuple[int, int]],
        avoid_values: Dict[str, int],
    ) -> Optional[Dict[str, int]]:
        ordered_ids = sorted(domains)
        index_by_id = {number_id: idx for idx, number_id in enumerate(ordered_ids)}
        forbidden_pairs = [
            (number_id, forbidden_value)
            for number_id in ordered_ids
            for forbidden_value in sorted(self._coerce_forbidden_number_values(avoid_values.get(number_id)))
            if domains[number_id][0] <= forbidden_value <= domains[number_id][1]
            and domains[number_id][0] != domains[number_id][1]
        ]
        forbidden_index = {
            forbidden_pair: len(ordered_ids) + idx for idx, forbidden_pair in enumerate(forbidden_pairs)
        }
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
                if math.isclose(lhs[1], rhs[1], abs_tol=1e-9):
                    return None
                continue
            expr_min = 0.0
            expr_max = 0.0
            for var, coeff in coeffs.items():
                low, high = domains[var]
                if coeff >= 0:
                    expr_min += coeff * low
                    expr_max += coeff * high
                else:
                    expr_min += coeff * high
                    expr_max += coeff * low
            rhs_value = float(rhs[1] - lhs[1])
            if rhs_value < expr_min - 1e-9 or rhs_value > expr_max + 1e-9:
                continue
            if math.isclose(expr_min, expr_max, abs_tol=1e-9) and math.isclose(expr_min, rhs_value, abs_tol=1e-9):
                return None
            slack = max(abs(expr_max - rhs_value), abs(rhs_value - expr_min)) + 1.0
            not_equal_specs.append((coeffs, rhs_value, slack))
        not_equal_index = {
            spec_index: len(ordered_ids) + len(forbidden_pairs) + spec_index
            for spec_index in range(len(not_equal_specs))
        }

        total_vars = len(ordered_ids) + len(forbidden_pairs) + len(not_equal_specs)
        if total_vars == 0:
            return {}

        c = np.zeros(total_vars, dtype=float)
        integrality = np.ones(total_vars, dtype=int)
        lower = np.full(total_vars, -np.inf, dtype=float)
        upper = np.full(total_vars, np.inf, dtype=float)

        for number_id, idx in index_by_id.items():
            low, high = domains[number_id]
            lower[idx] = low
            upper[idx] = high

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

        for number_id, forbidden_value in forbidden_pairs:
            low, high = domains[number_id]
            slack = max(1, high - low + 1)
            binary_idx = forbidden_index[(number_id, forbidden_value)]

            upper_cut = np.zeros(total_vars, dtype=float)
            upper_cut[index_by_id[number_id]] = 1
            upper_cut[binary_idx] = -slack
            rows.append(upper_cut)
            lbs.append(-np.inf)
            ubs.append(forbidden_value - 1)

            lower_cut = np.zeros(total_vars, dtype=float)
            lower_cut[index_by_id[number_id]] = 1
            lower_cut[binary_idx] = -slack
            rows.append(lower_cut)
            lbs.append(forbidden_value + 1 - slack)
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
            options=_MILP_OPTIONS,
        )
        if not getattr(result, "success", False) or result.x is None:
            return None

        assignments = {number_id: int(round(float(result.x[index_by_id[number_id]]))) for number_id in ordered_ids}
        if not self._constraints_feasible(
            constraints,
            assignments,
            {number_id: (value, value) for number_id, value in assignments.items()},
        ):
            return None
        for number_id, value in assignments.items():
            low, high = domains[number_id]
            if value < low or value > high:
                return None
        forced_equal_ids: set[str] = set()
        for number_id in ordered_ids:
            forbidden_values = self._coerce_forbidden_number_values(avoid_values.get(number_id))
            if assignments[number_id] not in forbidden_values:
                continue
            low, high = domains[number_id]
            if low == high == assignments[number_id]:
                forced_equal_ids.add(number_id)
                continue
            return None
        self.last_relaxed_avoid_number_ids = forced_equal_ids
        return assignments

    def _get_number_range_from_rules(
        self,
        var: str,
        rules: List[str],
        pre_assigned: Dict[str, int],
        existing_entities: Optional[EntityCollection] = None,
    ) -> Tuple[int, int]:
        min_val, max_val = self._number_base_range(var)

        def get_value(var_name):
            if var_name in pre_assigned:
                return pre_assigned[var_name]
            if existing_entities:
                if var_name in existing_entities.numbers:
                    return self._number_entity_int_value(existing_entities.numbers[var_name])
                v = RuleEngine._get_entity_value(existing_entities, var_name)
                if v is not None:
                    try:
                        return int(v)
                    except (ValueError, TypeError):
                        return None
            return None

        for rule in rules:
            rule = self._normalize_number_rule(self._strip_rule_comment(str(rule)))
            if has_century_function(rule):
                continue
            split = self._split_rule(rule)
            if not split:
                continue
            lhs, op, rhs = split
            lhs_token = self._number_token(lhs)
            rhs_token = self._number_token(rhs)
            lhs_value = self._int_literal(lhs)
            rhs_value = self._int_literal(rhs)
            if lhs_value is None and lhs_token is not None and lhs_token != var:
                lhs_value = get_value(lhs_token)
            if rhs_value is None and rhs_token is not None and rhs_token != var:
                rhs_value = get_value(rhs_token)

            if lhs_token == var and rhs_value is not None:
                bound = int(rhs_value)
                if op == "<":
                    max_val = min(max_val, bound - 1)
                elif op == "<=":
                    max_val = min(max_val, bound)
                elif op == ">":
                    min_val = max(min_val, bound + 1)
                elif op == ">=":
                    min_val = max(min_val, bound)
                elif op in ("=", "=="):
                    min_val = max(min_val, bound)
                    max_val = min(max_val, bound)
            if rhs_token == var and lhs_value is not None:
                bound = int(lhs_value)
                if op == "<":
                    min_val = max(min_val, bound + 1)
                elif op == "<=":
                    min_val = max(min_val, bound)
                elif op == ">":
                    max_val = min(max_val, bound - 1)
                elif op == ">=":
                    max_val = min(max_val, bound)
                elif op in ("=", "=="):
                    min_val = max(min_val, bound)
                    max_val = min(max_val, bound)
        return min_val, max_val

    def _parse_equality_constraints(
        self,
        rules: List[str],
        number_ids: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, Tuple[str, str]]]:
        """Parse equality and reverse-equality constraints from rules."""
        equality_constraints: Dict[str, str] = {}
        reverse_equality_constraints: Dict[str, Tuple[str, str]] = {}
        for rule in rules:
            rule = self._normalize_number_rule(self._strip_rule_comment(str(rule)))
            if has_century_function(rule):
                continue
            split = self._split_rule(rule)
            if not split:
                continue
            lhs, op, rhs = split
            if op not in ("=", "=="):
                continue
            lhs_refs = {ref for ref in re.findall(r"\bnumber_\d+\b", lhs) if ref in number_ids}
            rhs_refs = {ref for ref in re.findall(r"\bnumber_\d+\b", rhs) if ref in number_ids}
            rhs_token = self._number_token(rhs)
            lhs_token = self._number_token(lhs)
            if (
                lhs_token in number_ids
                and rhs_token in number_ids
                and lhs.strip() == lhs_token
                and rhs.strip() == rhs_token
            ):
                equality_constraints[rhs_token] = lhs_token
                continue
            if rhs_token in number_ids:
                equality_constraints[rhs_token] = lhs
                for num_id in lhs_refs - {rhs_token}:
                    reverse_equality_constraints.setdefault(num_id, (lhs, rhs_token))
            if lhs_token in number_ids:
                equality_constraints[lhs_token] = rhs
                for num_id in rhs_refs - {lhs_token}:
                    reverse_equality_constraints.setdefault(num_id, (rhs, lhs_token))
        return equality_constraints, reverse_equality_constraints

    @staticmethod
    def _validate_numbers_against_rules(
        numbers: Dict[str, NumberEntity],
        rules: List[str],
        existing_entities: Optional[EntityCollection] = None,
    ) -> bool:
        test_collection = EntityCollection()
        if existing_entities:
            test_collection.persons = existing_entities.persons.copy()
            test_collection.places = existing_entities.places.copy()
            test_collection.events = existing_entities.events.copy()
            test_collection.organizations = existing_entities.organizations.copy()
            test_collection.temporals = existing_entities.temporals.copy()
            test_collection.numbers = existing_entities.numbers.copy()
            test_collection.numbers.update(numbers)
        else:
            test_collection.numbers = numbers
        validation = RuleEngine.validate_all_rules(rules, test_collection)
        return all(is_valid for _, is_valid in validation)
