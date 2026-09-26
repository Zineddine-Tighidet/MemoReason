"""Composed rule-expression engine used by MemoReason generation and scoring."""

from . import rule_evaluation as _evaluation_module
from . import temporal_rule_engine as _temporal_module
from .rule_evaluation import RuleEvaluationMixin
from .temporal_rule_engine import TemporalRuleEngineMixin


class RuleEngine(TemporalRuleEngineMixin, RuleEvaluationMixin):
    """Evaluate explicit rules and answer expressions over document entities."""


# Historical static methods call ``RuleEngine`` by name. Bind that name in each
# implementation module after composing the class, preserving the exact call
# graph while avoiding a circular import.
_temporal_module.RuleEngine = RuleEngine
_evaluation_module.RuleEngine = RuleEngine
