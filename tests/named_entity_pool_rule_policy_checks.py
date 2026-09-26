from __future__ import annotations

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.named_entity_pool_rule_policy import (
    copy_named_entity_pool_without_rule_based_mutation,
)


def test_named_entity_pool_policy_returns_an_unmodified_defensive_copy() -> None:
    pool = {
        "persons": [
            {"full_name": "Person A", "nationality": "Telvaran"},
            {"full_name": "Person B", "nationality": "Prondish"},
            {"full_name": "Person C", "nationality": None},
        ],
        "places": [
            {"country": "Telvara", "demonym": "Telvaran"},
            {"country": "Grothenia", "demonym": "Grothenian"},
            {"country": "Prondal", "demonym": "Prondalian"},
        ],
    }
    aligned = copy_named_entity_pool_without_rule_based_mutation(pool)

    assert aligned == pool
    assert aligned is not pool
