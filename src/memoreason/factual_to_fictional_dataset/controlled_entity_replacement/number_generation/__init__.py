"""Number generation facade composed from focused helper mixins.

This module intentionally keeps the historical public class name
``NumberGenerationMixin`` for backward compatibility, while delegating
implementation details to smaller, cohesive modules.
"""

from .adjustments import NumberSamplingAdjustmentMixin
from .constraint_solver import NumberConstraintSolverMixin
from .value_strategy import NumberValueStrategyMixin
from .workflow import NumberGenerationWorkflowMixin


class NumberGenerationMixin(
    NumberGenerationWorkflowMixin,
    NumberSamplingAdjustmentMixin,
    NumberConstraintSolverMixin,
    NumberValueStrategyMixin,
):
    """Backward-compatible mixin exposing number-generation behavior."""

    pass


__all__ = ["NumberGenerationMixin"]
