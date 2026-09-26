# ruff: noqa: RUF046
"""Rule parsing and solving utilities for number generation."""

import math
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from memoreason.benchmark_definition.document_schema import EntityCollection

from ..generation_limits import (
    MAX_CSP_NUMBER_VARS,
    MAX_NUMBER_CSP_STEPS,
    SLOW_SAMPLING_LOG_SECONDS,
)
from ..linear_constraint_certification import transformed_constraints_use_safe_integer_lattice
from .exact_solve_result import ExactNumberSolveState

LinearExpr = tuple[dict[str, float], float]
LinearConstraintTriple = tuple[LinearExpr, str, LinearExpr]


class NumberAssignmentSolverMixin:
    """Solve constrained number assignments with exact and relaxed strategies."""

    def _solve_numbers_via_constraints(
        self,
        number_ids: list[str],
        rules: list[str],
        existing_entities: EntityCollection | None,
        avoid_values: dict[str, int],
        *,
        allow_relaxed_avoid: bool = True,
    ) -> dict[str, int] | None:
        if not number_ids:
            return {}
        self.last_relaxed_avoid_number_ids = set()
        self.last_number_solution_used_relaxed_avoid = False
        variables = set(number_ids)
        constraints = self._collect_linear_constraints(rules, variables, existing_entities)
        if constraints is None:
            return None

        domains: dict[str, tuple[int, int]] = {vid: self._number_base_range(vid) for vid in number_ids}
        if transformed_constraints_use_safe_integer_lattice(constraints):
            domains = self._tighten_number_domains(constraints, domains)
            if domains is None:
                return None
        for _var, (lo, hi) in domains.items():
            if lo > hi:
                return None

        milp_result = self._solve_numbers_via_milp_with_status(constraints, domains, avoid_values)
        if milp_result.state is ExactNumberSolveState.SOLVED:
            return milp_result.assignments
        if milp_result.state is ExactNumberSolveState.CERTIFIED_INFEASIBLE:
            # HiGHS proved this exact constraint/domain/avoidance system has no
            # solution.  A second exact CSP over the same system is redundant.
            # Relaxed avoidance, when explicitly allowed, is a different
            # system and remains a safe fallback.
            if avoid_values and allow_relaxed_avoid:
                relaxed_solution = self._solve_numbers_via_relaxed_avoid_milp(constraints, domains, avoid_values)
                if relaxed_solution is not None:
                    self.last_number_solution_used_relaxed_avoid = True
                    return relaxed_solution
            return None

        if len(number_ids) > MAX_CSP_NUMBER_VARS:
            if avoid_values and allow_relaxed_avoid:
                relaxed_solution = self._solve_numbers_via_relaxed_avoid_milp(constraints, domains, avoid_values)
                if relaxed_solution is not None:
                    self.last_number_solution_used_relaxed_avoid = True
                    return relaxed_solution
            return None

        unconstrained = set(number_ids)
        for lhs, _, rhs in constraints:
            unconstrained -= set(lhs[0].keys())
            unconstrained -= set(rhs[0].keys())

        assignments: dict[str, int] = {}
        for var in sorted(unconstrained):
            lo, hi = domains[var]
            if lo > hi:
                return None
            avoid = avoid_values.get(var)
            assignments[var] = self._sample_int_in_range(lo, hi, avoid=avoid)

        def pick_next_var() -> str | None:
            remaining = [v for v in number_ids if v not in assignments]
            if not remaining:
                return None
            return min(remaining, key=lambda v: domains[v][1] - domains[v][0])

        steps = 0
        start_time = time.monotonic()

        def backtrack() -> dict[str, int] | None:
            nonlocal steps
            steps += 1
            if steps > MAX_NUMBER_CSP_STEPS:
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
        if avoid_values and allow_relaxed_avoid:
            relaxed_solution = self._solve_numbers_via_relaxed_avoid_milp(constraints, domains, avoid_values)
            if relaxed_solution is not None:
                self.last_number_solution_used_relaxed_avoid = True
                return relaxed_solution
        return None

    def _solve_numbers_via_relaxed_avoid_milp(
        self,
        constraints: list[LinearConstraintTriple],
        domains: dict[str, tuple[int, int]],
        avoid_values: dict[str, int],
    ) -> dict[str, int] | None:
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

        not_equal_specs: list[tuple[dict[str, float], float, float]] = []
        for lhs, op, rhs in constraints:
            if op != "!=":
                continue
            coeffs: dict[str, float] = {}
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
            # Keep the seeded value tie-break strictly below the unit penalty
            # used to maximize avoidance of previous/factual assignments.
            coefficient_budget = 0.1 / max(1, len(ordered_ids))
            c[idx] = random.uniform(-coefficient_budget, coefficient_budget) / max(1, high - low)

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
            coeffs: dict[str, int] = {}
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

        # Match the strict MILP's separation from HiGHS' feasibility
        # tolerance so relaxed avoidance never weakens explicit `!=` rules.
        epsilon = 1e-5
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
            lower_cut[binary_idx] = -slack
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


__all__ = ["NumberAssignmentSolverMixin"]
