"""Century-aware rule helpers shared by runtime evaluation and generation."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Callable

from .entity_taxonomy import parse_integer_surface_number, parse_word_number

CENTURY_FUNCTION_NAMES = frozenset({"century_of", "century_start", "century_end"})
_CENTURY_FUNCTION_PATTERN = re.compile(r"\b(?:century_of|century_start|century_end)\s*\(")
_FOUR_DIGIT_YEAR_PATTERN = re.compile(r"\b(\d{4})\b")
_PLAIN_INTEGER_PATTERN = re.compile(r"^\s*-?\d+\s*$")


@dataclass(frozen=True)
class CenturyExpression:
    """Parsed century expression tree."""

    kind: str
    value: int | str | None = None
    function_name: str | None = None
    argument: "CenturyExpression | None" = None


@dataclass(frozen=True)
class CenturyConstraint:
    """One supported century-aware comparison rule."""

    raw_expression: str
    lhs: CenturyExpression
    operator: str
    rhs: CenturyExpression


def has_century_function(text: str) -> bool:
    """Return True when *text* uses one of the supported century helpers."""
    return bool(_CENTURY_FUNCTION_PATTERN.search(str(text or "")))


def century_start(value: Any) -> int | None:
    """Return the first year of the given century number."""
    century = _coerce_positive_int(value)
    if century is None:
        return None
    return (century - 1) * 100 + 1


def century_end(value: Any) -> int | None:
    """Return the last year of the given century number."""
    century = _coerce_positive_int(value)
    if century is None:
        return None
    return century * 100


def century_of(value: Any) -> int | None:
    """Return the century index for a year or year-bearing date string."""
    year = _coerce_year(value)
    if year is None:
        return None
    return ((year - 1) // 100) + 1


def parse_century_constraint(expression: str) -> CenturyConstraint | None:
    """Parse one comparison expression that uses century helper functions."""
    cleaned = str(expression or "").strip()
    if not cleaned or not has_century_function(cleaned):
        return None

    try:
        root = ast.parse(cleaned, mode="eval").body
    except SyntaxError:
        return None

    if not isinstance(root, ast.Compare) or len(root.ops) != 1 or len(root.comparators) != 1:
        return None

    lhs = _parse_expression_node(root.left)
    rhs = _parse_expression_node(root.comparators[0])
    operator = _comparison_operator(root.ops[0])
    if lhs is None or rhs is None or operator is None:
        return None

    return CenturyConstraint(
        raw_expression=cleaned,
        lhs=lhs,
        operator=operator,
        rhs=rhs,
    )


def evaluate_century_expression(
    expression: CenturyExpression,
    resolve_ref: Callable[[str], Any | None],
) -> int | None:
    """Evaluate one parsed century expression via a ref resolver."""
    if expression.kind == "const":
        value = expression.value
        return int(value) if value is not None else None
    if expression.kind == "ref":
        return coerce_numeric_ref_value(str(expression.value or ""), resolve_ref(str(expression.value or "")))
    if expression.kind != "call" or expression.argument is None or expression.function_name is None:
        return None

    inner_value = evaluate_century_expression(expression.argument, resolve_ref)
    if inner_value is None:
        return None

    if expression.function_name == "century_of":
        return century_of(inner_value)
    if expression.function_name == "century_start":
        return century_start(inner_value)
    if expression.function_name == "century_end":
        return century_end(inner_value)
    return None


def coerce_numeric_ref_value(ref: str, value: Any) -> int | None:
    """Convert one numeric/temporal reference value to an integer semantic value."""
    cleaned_ref = str(ref or "").strip()
    if not cleaned_ref:
        return None

    entity_id = cleaned_ref.split(".", 1)[0]
    if entity_id.startswith("temporal_"):
        return _coerce_year(value)
    return _coerce_numeric_value(value)


def century_expression_refs(expression: CenturyExpression) -> set[str]:
    """Collect all entity refs referenced by one parsed century expression."""
    if expression.kind == "ref" and isinstance(expression.value, str):
        return {expression.value}
    if expression.kind == "call" and expression.argument is not None:
        return century_expression_refs(expression.argument)
    return set()


def _comparison_operator(operator_node: ast.cmpop) -> str | None:
    operator_map = {
        ast.Eq: "==",
        ast.NotEq: "!=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Gt: ">",
        ast.GtE: ">=",
    }
    return operator_map.get(type(operator_node))


def _parse_expression_node(node: ast.AST) -> CenturyExpression | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return CenturyExpression(kind="const", value=int(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _parse_expression_node(node.operand)
        if inner is None or inner.kind != "const" or inner.value is None:
            return None
        value = int(inner.value)
        if isinstance(node.op, ast.USub):
            value = -value
        return CenturyExpression(kind="const", value=value)
    if isinstance(node, ast.Name):
        return CenturyExpression(kind="ref", value=node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return CenturyExpression(kind="ref", value=f"{node.value.id}.{node.attr}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        function_name = node.func.id
        if function_name not in CENTURY_FUNCTION_NAMES or len(node.args) != 1 or node.keywords:
            return None
        argument = _parse_expression_node(node.args[0])
        if argument is None:
            return None
        return CenturyExpression(kind="call", function_name=function_name, argument=argument)
    return None


def _coerce_positive_int(value: Any) -> int | None:
    numeric = _coerce_numeric_value(value)
    if numeric is None or numeric < 1:
        return None
    return numeric


def _coerce_numeric_value(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    raw = str(value).strip()
    if not raw:
        return None

    if _PLAIN_INTEGER_PATTERN.fullmatch(raw):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    parsed_integer = parse_integer_surface_number(raw)
    if parsed_integer is not None:
        return parsed_integer

    parsed_words = parse_word_number(raw.lower())
    if parsed_words is not None:
        return parsed_words

    return None


def _coerce_year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 1 else None
    if isinstance(value, float):
        year = int(value)
        return year if year >= 1 else None

    raw = str(value).strip()
    if not raw:
        return None

    explicit_year_match = _FOUR_DIGIT_YEAR_PATTERN.search(raw)
    if explicit_year_match:
        year = int(explicit_year_match.group(1))
        return year if year >= 1 else None

    parsed_numeric = _coerce_numeric_value(raw)
    if parsed_numeric is None or parsed_numeric < 1:
        return None
    return parsed_numeric

