"""Conservative certification helpers for integer linear constraints."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Any


_MAX_EXACT_FLOAT_INTEGER = 2**53
_SUPPORTED_LINEAR_OPERATORS = frozenset({"<", "<=", ">", ">=", "=", "==", "!="})


def _is_safe_integer_lattice_value(value: Any) -> bool:
    """Return whether a coefficient is an exactly represented, safe integer."""
    if isinstance(value, bool):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(numeric) and abs(numeric) <= _MAX_EXACT_FLOAT_INTEGER and numeric.is_integer()


def transformed_constraints_use_safe_integer_lattice(constraints: Iterable[tuple]) -> bool:
    """Prove that the MILP's unit strict gap matches exact integer semantics.

    The solvers encode strict comparisons using a one-unit separation and
    ``!=`` using a much smaller positive separation.  A status-2 result is an
    exact infeasibility certificate only when every transformed affine
    expression lies on an integer lattice.  Fractional systems remain valid
    MILP heuristics, but must retain the exact CSP fallback.
    """
    for lhs, operator, rhs in constraints:
        if operator not in _SUPPORTED_LINEAR_OPERATORS:
            return False
        transformed_coefficients: dict[str, Any] = {}
        for variable, coefficient in lhs[0].items():
            transformed_coefficients[variable] = transformed_coefficients.get(variable, 0) + coefficient
        for variable, coefficient in rhs[0].items():
            transformed_coefficients[variable] = transformed_coefficients.get(variable, 0) - coefficient
        transformed_rhs = rhs[1] - lhs[1]
        if not _is_safe_integer_lattice_value(transformed_rhs):
            return False
        if any(not _is_safe_integer_lattice_value(value) for value in transformed_coefficients.values()):
            return False
    return True


__all__ = ["transformed_constraints_use_safe_integer_lattice"]
