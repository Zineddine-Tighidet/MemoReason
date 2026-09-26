"""Rule parsing and solving utilities for number generation."""

import math
import re


from memoreason.benchmark_definition.century_expressions import has_century_function
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

from .assignment_solver import NumberAssignmentSolverMixin
from .milp_solver import NumberMilpSolverMixin
from .rule_validation import NumberRuleValidationMixin

LinearExpr = tuple[dict[str, float], float]
LinearConstraintTriple = tuple[LinearExpr, str, LinearExpr]


class NumberConstraintSolverMixin(NumberAssignmentSolverMixin, NumberMilpSolverMixin, NumberRuleValidationMixin):
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
    def _split_rule(rule: str) -> tuple[str, str, str] | None:
        for op in ("<=", ">=", "==", "!=", "=", "<", ">"):
            if op in rule:
                parts = rule.split(op)
                if len(parts) == 2:
                    return parts[0].strip(), op, parts[1].strip()
        return None

    @staticmethod
    def _number_token(expr: str) -> str | None:
        cleaned = expr.strip()
        return cleaned if re.fullmatch(r"number_\d+", cleaned) else None

    @staticmethod
    def _int_literal(expr: str) -> int | None:
        cleaned = expr.strip()
        if re.fullmatch(r"-?\d+", cleaned):
            return int(cleaned)
        return None

    def _number_rule_is_evaluable(
        self,
        rule: str,
        variables: set[str],
        existing_entities: EntityCollection | None,
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
        rules: list[str],
        variables: set[str],
        existing_entities: EntityCollection | None,
    ) -> list[str]:
        selected: list[str] = []
        for raw_rule in rules:
            cleaned = self._strip_rule_comment(str(raw_rule))
            if not cleaned:
                continue
            if self._number_rule_is_evaluable(cleaned, variables, existing_entities):
                selected.append(str(raw_rule))
        return selected

    def _ordering_number_rules(
        self,
        number_ids: list[str],
        existing_entities: EntityCollection | None,
    ) -> list[str]:
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
        variables: set[str],
        existing_entities: EntityCollection | None,
    ) -> LinearExpr | None:
        import ast

        def resolve_ref(ref: str) -> float | None:
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

        def to_linear(node) -> LinearExpr | None:
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
        assigned: dict[str, int],
        existing_entities: EntityCollection | None,
    ) -> float | None:
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
        rules: list[str],
        variables: set[str],
        existing_entities: EntityCollection | None,
    ) -> list[LinearConstraintTriple] | None:
        constraints: list[LinearConstraintTriple] = []
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
        expr: tuple[dict[str, int], int],
        assignments: dict[str, int],
        domains: dict[str, tuple[int, int]],
    ) -> tuple[float, float]:
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
        constraints: list[LinearConstraintTriple],
        assignments: dict[str, int],
        domains: dict[str, tuple[int, int]],
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


__all__ = ["NumberConstraintSolverMixin"]
