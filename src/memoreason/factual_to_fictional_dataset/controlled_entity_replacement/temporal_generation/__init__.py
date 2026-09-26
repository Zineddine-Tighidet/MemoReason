"""Backward-compatible temporal generation facade composed from focused helper mixins."""

from .constraint_solver import TemporalConstraintSolverMixin
from .value_strategy import TemporalValueStrategyMixin
from .workflow import TemporalGenerationWorkflowMixin


class TemporalGenerationMixin(
    TemporalGenerationWorkflowMixin,
    TemporalConstraintSolverMixin,
    TemporalValueStrategyMixin,
):
    """Backward-compatible temporal-generation facade."""

    pass


__all__ = ["TemporalGenerationMixin"]
