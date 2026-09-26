"""Tri-state result contract for exact integer-number solving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExactNumberSolveState(StrEnum):
    """Outcome of an exact number-solver attempt."""

    SOLVED = "solved"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNAVAILABLE_OR_UNSUPPORTED = "unavailable_or_unsupported"


@dataclass(frozen=True)
class ExactNumberSolveResult:
    """Exact number solution or a precise reason no solution was returned."""

    state: ExactNumberSolveState
    assignments: dict[str, int] | None = None

    def __post_init__(self) -> None:
        has_solution = self.assignments is not None
        if has_solution != (self.state is ExactNumberSolveState.SOLVED):
            raise ValueError("Only a solved exact number result may contain assignments.")

    @classmethod
    def solved(cls, assignments: dict[str, int]) -> ExactNumberSolveResult:
        return cls(ExactNumberSolveState.SOLVED, dict(assignments))

    @classmethod
    def certified_infeasible(cls) -> ExactNumberSolveResult:
        return cls(ExactNumberSolveState.CERTIFIED_INFEASIBLE)

    @classmethod
    def unavailable_or_unsupported(cls) -> ExactNumberSolveResult:
        return cls(ExactNumberSolveState.UNAVAILABLE_OR_UNSUPPORTED)

    @property
    def is_certified_infeasible(self) -> bool:
        return self.state is ExactNumberSolveState.CERTIFIED_INFEASIBLE


__all__ = ["ExactNumberSolveResult", "ExactNumberSolveState"]
