"""Compose numerical and temporal constraints for fictional-entity sampling."""

from __future__ import annotations

from .fictional_entity_age_difference_constraint_enforcement import (
    AgeDifferenceConstraintEnforcementMixin,
)
from .fictional_entity_mixed_temporal_rule_constraint_enforcement import (
    MixedTemporalRuleConstraintEnforcementMixin,
)
from .fictional_entity_numerical_constraints import NumericalRuleConstraintsMixin
from .fictional_entity_numerical_generation import NumericalGenerationMixin
from .fictional_entity_numerical_rule_constraint_enforcement import (
    NumericalRuleConstraintEnforcementMixin,
)
from .fictional_entity_single_difference_constraint_enforcement import (
    SingleDifferenceConstraintEnforcementMixin,
)
from .fictional_entity_temporal_alignment import TemporalRuleAlignmentMixin
from .fictional_entity_temporal_feasibility import TemporalNumberFeasibilityMixin


class FictionalEntitySamplerNumericalMixin(
    NumericalRuleConstraintsMixin,
    NumericalRuleConstraintEnforcementMixin,
    NumericalGenerationMixin,
    TemporalRuleAlignmentMixin,
    TemporalNumberFeasibilityMixin,
    MixedTemporalRuleConstraintEnforcementMixin,
    AgeDifferenceConstraintEnforcementMixin,
    SingleDifferenceConstraintEnforcementMixin,
):
    """MILP-backed number generation plus randomized temporal sampling."""


__all__ = ["FictionalEntitySamplerNumericalMixin"]
