# ruff: noqa: RUF046
"""Rule parsing and solving utilities for number generation."""

import math
import random

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from ..linear_constraint_certification import transformed_constraints_use_safe_integer_lattice
from .exact_solve_result import ExactNumberSolveResult

LinearExpr = tuple[dict[str, float], float]
LinearConstraintTriple = tuple[LinearExpr, str, LinearExpr]


class NumberMilpSolverMixin:
    """Tighten number domains and solve the resulting mixed-integer program."""

    def _tighten_number_domains(
        self,
        constraints: list[LinearConstraintTriple],
        domains: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]] | None:
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
        constraints: list[LinearConstraintTriple],
        domains: dict[str, tuple[int, int]],
        avoid_values: dict[str, int],
    ) -> dict[str, int] | None:
        """Backward-compatible solution-only facade for direct callers."""
        return self._solve_numbers_via_milp_with_status(constraints, domains, avoid_values).assignments

    def _solve_numbers_via_milp_with_status(
        self,
        constraints: list[LinearConstraintTriple],
        domains: dict[str, tuple[int, int]],
        avoid_values: dict[str, int],
    ) -> ExactNumberSolveResult:
        ordered_ids = sorted(domains)
        index_by_id = {number_id: idx for idx, number_id in enumerate(ordered_ids)}
        forbidden_pairs = [
            (number_id, forbidden_value)
            for number_id in ordered_ids
            for forbidden_value in sorted(self._coerce_forbidden_number_values(avoid_values.get(number_id)))
            if domains[number_id][0] <= forbidden_value <= domains[number_id][1]
            and domains[number_id][0] != domains[number_id][1]
        ]
        forbidden_index = {forbidden_pair: len(ordered_ids) + idx for idx, forbidden_pair in enumerate(forbidden_pairs)}
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
                    return ExactNumberSolveResult.certified_infeasible()
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
                return ExactNumberSolveResult.certified_infeasible()
            slack = max(abs(expr_max - rhs_value), abs(rhs_value - expr_min)) + 1.0
            not_equal_specs.append((coeffs, rhs_value, slack))
        not_equal_index = {
            spec_index: len(ordered_ids) + len(forbidden_pairs) + spec_index
            for spec_index in range(len(not_equal_specs))
        }

        total_vars = len(ordered_ids) + len(forbidden_pairs) + len(not_equal_specs)
        if total_vars == 0:
            return ExactNumberSolveResult.solved({})

        c = np.zeros(total_vars, dtype=float)
        integrality = np.ones(total_vars, dtype=int)
        lower = np.full(total_vars, -np.inf, dtype=float)
        upper = np.full(total_vars, np.inf, dtype=float)

        for number_id, idx in index_by_id.items():
            low, high = domains[number_id]
            lower[idx] = low
            upper[idx] = high
            # A zero objective leaves every feasible assignment tied and lets
            # HiGHS choose a process-dependent incumbent.  The generator's
            # seeded RNG supplies a stable, effectively unique tie-break.
            c[idx] = random.uniform(-1.0, 1.0) / max(1, high - low)

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
                return ExactNumberSolveResult.unavailable_or_unsupported()

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

        # Stay decisively above HiGHS' feasibility tolerance.  At 1e-6 the
        # solver can accept the equality branch after scaling otherwise small
        # affine expressions against year-/count-sized integer variables.
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
        if not getattr(result, "success", False):
            if getattr(result, "status", None) == 2 and transformed_constraints_use_safe_integer_lattice(constraints):
                return ExactNumberSolveResult.certified_infeasible()
            return ExactNumberSolveResult.unavailable_or_unsupported()
        if result.x is None:
            return ExactNumberSolveResult.unavailable_or_unsupported()

        assignments = {number_id: int(round(float(result.x[index_by_id[number_id]]))) for number_id in ordered_ids}
        if not self._constraints_feasible(
            constraints,
            assignments,
            {number_id: (value, value) for number_id, value in assignments.items()},
        ):
            return ExactNumberSolveResult.unavailable_or_unsupported()
        for number_id, value in assignments.items():
            low, high = domains[number_id]
            if value < low or value > high:
                return ExactNumberSolveResult.unavailable_or_unsupported()
        forced_equal_ids: set[str] = set()
        for number_id in ordered_ids:
            forbidden_values = self._coerce_forbidden_number_values(avoid_values.get(number_id))
            if assignments[number_id] not in forbidden_values:
                continue
            low, high = domains[number_id]
            if low == high == assignments[number_id]:
                forced_equal_ids.add(number_id)
                continue
            return ExactNumberSolveResult.unavailable_or_unsupported()
        self.last_relaxed_avoid_number_ids = forced_equal_ids
        return ExactNumberSolveResult.solved(assignments)


__all__ = ["NumberMilpSolverMixin"]
