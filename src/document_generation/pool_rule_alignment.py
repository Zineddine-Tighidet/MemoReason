"""Compatibility wrapper for legacy pool-rule alignment.

Named-entity cross-reference rules are intentionally ignored; only numerical and
temporal constraints are enforced during generation. This function therefore
returns a defensive copy of the pool without applying any rule-based mutation.
"""

from __future__ import annotations

import copy
from typing import Any

def align_pool_values_for_manual_rules(
    entity_pool: dict[str, Any],
    rules: list[str],
) -> dict[str, Any]:
    del rules
    return copy.deepcopy(entity_pool)


__all__ = ["align_pool_values_for_manual_rules"]
