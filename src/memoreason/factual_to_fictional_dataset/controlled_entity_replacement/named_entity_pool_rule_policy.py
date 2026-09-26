"""Policy for keeping named-entity pools independent of numeric constraints.

Named-entity cross-reference rules are intentionally ignored; only numerical and
temporal constraints are enforced during generation. This function therefore
returns a defensive copy of the pool without applying any rule-based mutation.
"""

from __future__ import annotations

import copy
from typing import Any


def copy_named_entity_pool_without_rule_based_mutation(
    entity_pool: dict[str, Any],
) -> dict[str, Any]:
    """Return the pool used for sampling without altering its reviewed values."""
    return copy.deepcopy(entity_pool)


__all__ = ["copy_named_entity_pool_without_rule_based_mutation"]
