"""Regression checks for the exact-partial projection used by paper curves."""
from __future__ import annotations

import random

import pytest

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.replacement_planning import (
    _select_exact_linked_candidate_indices,
    build_replacement_plan,
    partial_rule_linked_entity_groups_from_rules,
)


@pytest.fixture(autouse=True)
def preserve_random_state():
    original = random.getstate()
    yield
    random.setstate(original)


def test_exact_selection_preserves_linked_rule_components():
    candidates = [("number", f"number_{index}", None) for index in range(1, 5)]
    for seed in range(20):
        random.seed(seed)
        selected = _select_exact_linked_candidate_indices(
            replacement_candidates=candidates,
            target_replacement_count=3,
            linked_entity_groups=[{"number_1", "number_2"}],
        )
        assert len(selected) == 3
        assert (0 in selected) == (1 in selected)


def test_impossible_linked_target_fails_without_splitting():
    candidates = [("number", f"number_{index}", None) for index in range(1, 5)]
    with pytest.raises(ValueError, match="impossible without splitting linked entities"):
        _select_exact_linked_candidate_indices(
            replacement_candidates=candidates,
            target_replacement_count=3,
            linked_entity_groups=[{"number_1", "number_2"}, {"number_3", "number_4"}],
        )


def test_overlapping_rule_groups_form_one_component():
    candidates = [("number", f"number_{index}", None) for index in range(1, 5)]
    with pytest.raises(ValueError, match="impossible without splitting linked entities"):
        _select_exact_linked_candidate_indices(
            replacement_candidates=candidates,
            target_replacement_count=2,
            linked_entity_groups=[{"number_1", "number_2"}, {"number_2", "number_3"}],
        )


def test_projection_selects_only_entities_replaced_in_full_variant():
    plan = build_replacement_plan(
        entity_types={"numbers": {f"number_{i}": object() for i in range(1, 6)}},
        required_entities={},
        replacement_proportion=0.5,
        replace_mode="all",
        eligible_entity_ids={"number": {"number_1", "number_2"}},
    )
    assert plan.eligible_entity_count == 2
    assert plan.target_replacement_count == 1
    assert plan.fully_replaced_entity_ids["number"] <= {"number_1", "number_2"}
    assert len(plan.fully_replaced_entity_ids["number"]) == 1


def test_rule_links_retain_numerical_and_temporal_dependency():
    groups = partial_rule_linked_entity_groups_from_rules(
        rules=["number_1.int == temporal_2.year - temporal_1.year"])
    assert groups == [frozenset({"number_1", "temporal_1", "temporal_2"})]
