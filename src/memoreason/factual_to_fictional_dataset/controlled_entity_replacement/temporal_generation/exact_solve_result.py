"""Tri-state result contract for exact temporal-year solving."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExactTemporalSolveState(StrEnum):
    """Outcome of an exact solver attempt."""

    SOLVED = "solved"
    CERTIFIED_INFEASIBLE = "certified_infeasible"
    UNAVAILABLE_OR_UNSUPPORTED = "unavailable_or_unsupported"


@dataclass(frozen=True)
class ExactTemporalSolveResult:
    """Exact temporal solution or a precise reason no solution was returned."""

    state: ExactTemporalSolveState
    assignments: dict[str, int] | None = None

    def __post_init__(self) -> None:
        has_solution = self.assignments is not None
        if has_solution != (self.state is ExactTemporalSolveState.SOLVED):
            raise ValueError("Only a solved exact temporal result may contain assignments.")

    @classmethod
    def solved(cls, assignments: dict[str, int]) -> ExactTemporalSolveResult:
        return cls(ExactTemporalSolveState.SOLVED, dict(assignments))

    @classmethod
    def certified_infeasible(cls) -> ExactTemporalSolveResult:
        return cls(ExactTemporalSolveState.CERTIFIED_INFEASIBLE)

    @classmethod
    def unavailable_or_unsupported(cls) -> ExactTemporalSolveResult:
        return cls(ExactTemporalSolveState.UNAVAILABLE_OR_UNSUPPORTED)

    @property
    def is_certified_infeasible(self) -> bool:
        return self.state is ExactTemporalSolveState.CERTIFIED_INFEASIBLE


__all__ = ["ExactTemporalSolveResult", "ExactTemporalSolveState"]
