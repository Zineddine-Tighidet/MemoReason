"""Temporal year-constraint parsing and solving helpers."""

from __future__ import annotations

import ast
import random
import time
from typing import Any

from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

from ..generation_limits import _MIN_YEAR, SLOW_SAMPLING_LOG_SECONDS
from .exact_solve_result import ExactTemporalSolveResult
from .milp_solver import TemporalMilpSolverMixin


class TemporalConstraintSolverMixin(TemporalMilpSolverMixin):
    """Parse temporal year rules and solve valid year assignments."""

    _EXACT_TEMPORAL_AVOID_EXPANSION_STEPS = (0, 2, 4, 8, 16, 32)

    def _parse_temporal_linear_expr(
        self,
        expr: str,
        required_ids: set[str],
        existing_entities: EntityCollection | None,
    ) -> tuple[dict[str, int], int] | None:
        def combine(a: tuple[dict[str, int], int], b: tuple[dict[str, int], int], sign: int = 1):
            coeffs = dict(a[0])
            for var, coeff in b[0].items():
                coeffs[var] = coeffs.get(var, 0) + sign * coeff
            return coeffs, a[1] + sign * b[1]

        def scale(linear: tuple[dict[str, int], int], factor: float) -> tuple[dict[str, float], float]:
            coeffs, const = linear
            return {key: value * factor for key, value in coeffs.items()}, const * factor

        def coerce_number_entity_value(number: Any, attr: str) -> int | None:
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

        def resolve_ref(ref: str) -> int | None:
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

        def to_linear(node) -> tuple[dict[str, int], int] | None:
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
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "year"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "date"
                and isinstance(node.value.value, ast.Name)
            ):
                entity_id = node.value.value.id
                if not entity_id.startswith("temporal_"):
                    return None
                if entity_id in required_ids:
                    return {entity_id: 1}, 0
                resolved = resolve_ref(f"{entity_id}.date")
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
        required_temporals: list[tuple],
        rules: list[str],
        existing_entities: EntityCollection | None,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
    ) -> dict[str, int] | None:
        """Backward-compatible solution-only facade for direct callers."""
        result = self._solve_temporal_years_with_status(
            required_temporals,
            rules,
            existing_entities,
            excluded_years,
            decade_year_temporal_ids,
        )
        return result.assignments

    def _solve_temporal_years_via_backtracking(
        self,
        constraints: list[Any],
        required_ids: set[str],
        domains: dict[str, list[int]],
        domain_bounds: dict[str, tuple[int, int]],
    ) -> dict[str, int] | None:
        def feasible(assignments: dict[str, int]) -> bool:
            return self._constraints_feasible(constraints, assignments, domain_bounds)

        def choose_var(assignments: dict[str, int]) -> str | None:
            remaining = [tid for tid in required_ids if tid not in assignments]
            if not remaining:
                return None
            random.shuffle(remaining)
            return min(remaining, key=lambda tid: len(domains[tid]))

        steps = 0
        start_time = time.monotonic()
        randomized_step_budget = 1000 if len(required_ids) > 8 else 5000

        def backtrack(assignments: dict[str, int]) -> dict[str, int] | None:
            nonlocal steps
            steps += 1
            if steps > randomized_step_budget:
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

        return backtrack({})

    def _solve_temporal_years_with_status(
        self,
        required_temporals: list[tuple],
        rules: list[str],
        existing_entities: EntityCollection | None,
        excluded_years: set[int],
        decade_year_temporal_ids: set[str],
    ) -> ExactTemporalSolveResult:
        constraints = self._collect_temporal_year_constraints(required_temporals, rules, existing_entities)
        if constraints is None:
            return ExactTemporalSolveResult.unavailable_or_unsupported()
        constraints.extend(self._ordering_temporal_year_constraints(required_temporals, existing_entities))
        if not constraints:
            return ExactTemporalSolveResult.unavailable_or_unsupported()

        required_ids = {tid for tid, attrs in required_temporals if any(attr in attrs for attr in ("year", "date"))}
        if not required_ids:
            return ExactTemporalSolveResult.solved({})
        years_by_id = self.exclude_temporals.get("years_by_id") or {}
        has_intervariant_year_exclusions = any(years_by_id.get(tid) for tid in required_ids)
        expansion_steps = self._EXACT_TEMPORAL_AVOID_EXPANSION_STEPS if has_intervariant_year_exclusions else (0,)
        last_result = ExactTemporalSolveResult.unavailable_or_unsupported()

        for expansion_padding in expansion_steps:
            domains: dict[str, list[int]] = {}
            for tid in required_ids:
                base_lo, base_hi = self._temporal_year_base_range(tid)
                sampling_lo, sampling_hi = self._temporal_year_sampling_bounds(tid)
                domain_lo = max(sampling_lo, int(base_lo) - expansion_padding)
                domain_hi = min(sampling_hi, int(base_hi) + expansion_padding)
                domain = self._temporal_year_domain(
                    tid,
                    domain_lo,
                    domain_hi,
                    excluded_years,
                    decade_year_temporal_ids,
                )
                if not domain and not has_intervariant_year_exclusions:
                    domain = self._expand_temporal_year_domain(
                        tid,
                        _MIN_YEAR,
                        sampling_hi,
                        excluded_years,
                        decade_year_temporal_ids,
                    )
                if not domain:
                    break
                domains[tid] = domain
            if len(domains) != len(required_ids):
                last_result = ExactTemporalSolveResult.unavailable_or_unsupported()
                continue

            domain_bounds = {tid: (min(values), max(values)) for tid, values in domains.items()}
            milp_result = self._solve_temporal_years_via_milp_with_status(
                constraints,
                domains,
                domain_bounds,
                decade_year_temporal_ids,
            )
            last_result = milp_result
            if milp_result.assignments is not None:
                return milp_result
            if milp_result.is_certified_infeasible:
                # Status=2 is final for this exact domain.  A broader domain is
                # a distinct system and is tried only to escape accumulated
                # per-ID intervariant exclusions.
                continue

            solved = self._solve_temporal_years_via_backtracking(
                constraints,
                required_ids,
                domains,
                domain_bounds,
            )
            if solved is not None:
                return ExactTemporalSolveResult.solved(solved)
            last_result = ExactTemporalSolveResult.unavailable_or_unsupported()

        return last_result


__all__ = ["TemporalConstraintSolverMixin"]
