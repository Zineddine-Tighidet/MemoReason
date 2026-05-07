"""Plan which factual entities are replaced in one fictional document variant."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from src.core.annotation_runtime import (
    FULL_REPLACE_ENTITY_TYPES,
    PARTIAL_REPLACE_ATTRIBUTES,
    find_entity_refs,
    normalize_entity_ref,
)
from src.core.document_schema import EntityCollection
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number

RequiredEntityMap = Dict[str, List[Tuple[str, List[str]]]]
EntityCollectionsByPluralType = Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ReplacementPlan:
    """Selection of full and partial replacements for one generated variant."""

    fully_replaced_entity_ids: Dict[str, Set[str]]
    partially_replaced_attributes: Dict[str, Dict[str, frozenset[str]]]
    eligible_entity_count: int
    target_replacement_count: int


@dataclass(frozen=True)
class PartialReplacement:
    """One factual entity whose replaced fields are only a subset of its attributes."""

    entity_id: str
    factual_entity: Any
    replaced_attributes: frozenset[str]


@dataclass
class ReplacementLayout:
    """Factual entities grouped by how the current variant will use them."""

    factual_entities_to_replace: Dict[str, List[Tuple[str, Any]]]
    factual_entities_to_keep: Dict[str, List[Tuple[str, Any]]]
    partially_replaced_entities: Dict[str, List[PartialReplacement]]
    initial_hybrid_entities: EntityCollection


def _values_semantically_equal(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if isinstance(lhs, bool) or isinstance(rhs, bool):
        return lhs is rhs
    normalized_candidates = []
    for value in (lhs, rhs):
        normalized = None
        for parser in (parse_word_number, parse_integer_surface_number):
            try:
                normalized = parser(str(value).strip())
            except Exception:
                normalized = None
            if normalized is not None:
                break
        normalized_candidates.append(normalized)
    if normalized_candidates[0] is not None and normalized_candidates[1] is not None:
        return normalized_candidates[0] == normalized_candidates[1]
    for parser in (parse_word_number, parse_integer_surface_number):
        try:
            lhs_parsed = parser(str(lhs).strip())
            rhs_parsed = parser(str(rhs).strip())
        except Exception:
            lhs_parsed = None
            rhs_parsed = None
        if lhs_parsed is not None and rhs_parsed is not None:
            return lhs_parsed == rhs_parsed
    try:
        return abs(float(lhs) - float(rhs)) <= 1e-9
    except (TypeError, ValueError):
        return str(lhs).strip().casefold() == str(rhs).strip().casefold()


def compute_target_replacement_count(proportion: float, eligible_entity_count: int) -> int:
    """Convert a replacement proportion into the exact number of entities to replace."""
    if eligible_entity_count <= 0:
        return 0
    if proportion <= 0.0:
        return 0
    if proportion >= 1.0:
        return eligible_entity_count
    target = int(math.floor((proportion * eligible_entity_count) + 0.5))
    return max(0, min(eligible_entity_count, target))


def build_replacement_plan(
    *,
    entity_types: EntityCollectionsByPluralType,
    required_entities: RequiredEntityMap,
    replacement_proportion: float,
    replace_mode: str,
    rng: random.Random | None = None,
    rules: Iterable[str] | None = None,
) -> ReplacementPlan:
    """Choose which factual entities will be replaced in one fictional variant."""
    full_types = FULL_REPLACE_ENTITY_TYPES.get(replace_mode, set())
    partial_types = PARTIAL_REPLACE_ATTRIBUTES.get(replace_mode, {})

    required_attr_map: Dict[Tuple[str, str], Set[str]] = {}
    for entity_type, specs in required_entities.items():
        for entity_id, attrs in specs:
            required_attr_map[(entity_type, entity_id)] = set(attrs or [])

    replacement_candidates: List[Tuple[str, str, Optional[frozenset[str]]]] = []
    for entity_type_plural, entities_dict in entity_types.items():
        entity_type = entity_type_plural.rstrip("s")
        for entity_id in sorted(entities_dict.keys()):
            if entity_type in full_types:
                replacement_candidates.append((entity_type, entity_id, None))
                continue
            if entity_type in partial_types:
                required_attrs = required_attr_map.get((entity_type, entity_id), set())
                replaceable_attrs = frozenset(attr for attr in required_attrs if attr in partial_types[entity_type])
                if replaceable_attrs:
                    replacement_candidates.append((entity_type, entity_id, replaceable_attrs))

    eligible_entity_count = len(replacement_candidates)
    target_replacement_count = compute_target_replacement_count(
        replacement_proportion,
        eligible_entity_count,
    )
    sampler = rng or random
    selected_indices = (
        set(sampler.sample(range(eligible_entity_count), target_replacement_count))
        if target_replacement_count > 0
        else set()
    )
    if selected_indices and rules:
        selected_indices = _close_selected_indices_over_rule_components(
            selected_indices=selected_indices,
            replacement_candidates=replacement_candidates,
            rules=rules,
        )

    fully_replaced_entity_ids: Dict[str, Set[str]] = {}
    partially_replaced_attributes: Dict[str, Dict[str, frozenset[str]]] = {}
    for candidate_index, (entity_type, entity_id, replaceable_attrs) in enumerate(replacement_candidates):
        if candidate_index not in selected_indices:
            continue
        if replaceable_attrs is None:
            fully_replaced_entity_ids.setdefault(entity_type, set()).add(entity_id)
            continue
        partially_replaced_attributes.setdefault(entity_type, {})[entity_id] = replaceable_attrs

    return ReplacementPlan(
        fully_replaced_entity_ids=fully_replaced_entity_ids,
        partially_replaced_attributes=partially_replaced_attributes,
        eligible_entity_count=eligible_entity_count,
        target_replacement_count=target_replacement_count,
    )


def _close_selected_indices_over_rule_components(
    *,
    selected_indices: set[int],
    replacement_candidates: list[tuple[str, str, Optional[frozenset[str]]]],
    rules: Iterable[str],
) -> set[int]:
    """Expand sampled replacements to whole rule-connected components."""
    index_by_entity_id = {
        entity_id: candidate_index for candidate_index, (_entity_type, entity_id, _attrs) in enumerate(replacement_candidates)
    }
    parent = list(range(len(replacement_candidates)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for rule in rules:
        connected_indices = []
        for ref in find_entity_refs(str(rule)):
            entity_id = normalize_entity_ref(ref.split(".", 1)[0])
            candidate_index = index_by_entity_id.get(entity_id)
            if candidate_index is not None:
                connected_indices.append(candidate_index)
        connected_indices = sorted(set(connected_indices))
        if len(connected_indices) < 2:
            continue
        first_index = connected_indices[0]
        for connected_index in connected_indices[1:]:
            union(first_index, connected_index)

    component_indices: dict[int, set[int]] = {}
    for candidate_index in range(len(replacement_candidates)):
        component_indices.setdefault(find(candidate_index), set()).add(candidate_index)

    closed_indices = set(selected_indices)
    for selected_index in selected_indices:
        closed_indices.update(component_indices.get(find(selected_index), set()))
    return closed_indices


def build_replacement_layout(
    entity_types: EntityCollectionsByPluralType,
    plan: ReplacementPlan,
) -> ReplacementLayout:
    """Split factual entities into replaced, kept, and partially replaced groups."""
    factual_entities_to_replace: Dict[str, List[Tuple[str, Any]]] = {
        entity_type.rstrip("s"): [] for entity_type in entity_types
    }
    factual_entities_to_keep: Dict[str, List[Tuple[str, Any]]] = {
        entity_type.rstrip("s"): [] for entity_type in entity_types
    }
    partially_replaced_entities: Dict[str, List[PartialReplacement]] = {}
    initial_hybrid_entities = EntityCollection()

    for entity_type_plural, entities_dict in entity_types.items():
        entity_type = entity_type_plural.rstrip("s")
        for entity_id in sorted(entities_dict.keys()):
            factual_entity = entities_dict[entity_id]
            if entity_id in plan.fully_replaced_entity_ids.get(entity_type, set()):
                factual_entities_to_replace[entity_type].append((entity_id, factual_entity))
                continue

            replaceable_attrs = plan.partially_replaced_attributes.get(entity_type, {}).get(entity_id)
            if replaceable_attrs:
                partially_replaced_entities.setdefault(entity_type, []).append(
                    PartialReplacement(
                        entity_id=entity_id,
                        factual_entity=factual_entity,
                        replaced_attributes=replaceable_attrs,
                    )
                )
                initial_hybrid_entities.add_entity(entity_type, entity_id, factual_entity)
                continue

            factual_entities_to_keep[entity_type].append((entity_id, factual_entity))
            initial_hybrid_entities.add_entity(entity_type, entity_id, factual_entity)

    return ReplacementLayout(
        factual_entities_to_replace=factual_entities_to_replace,
        factual_entities_to_keep=factual_entities_to_keep,
        partially_replaced_entities=partially_replaced_entities,
        initial_hybrid_entities=initial_hybrid_entities,
    )


def merge_partially_replaced_entity(
    factual_entity: Any,
    fictional_entity: Any,
    replaced_attributes: frozenset[str],
) -> Any:
    """Keep the factual entity but overwrite only the selected fictional attributes."""
    merged = factual_entity.model_copy()
    for attr in replaced_attributes:
        attr_value = getattr(fictional_entity, attr, None)
        if attr_value is not None:
            setattr(merged, attr, attr_value)
    return merged


def entity_attributes_changed(
    factual_entity: Any,
    fictional_entity: Any,
    attrs: Optional[Iterable[str]] = None,
) -> bool:
    """Return True when any relevant factual attribute differs after replacement."""
    if factual_entity is None or fictional_entity is None:
        return True
    if attrs is None:
        if hasattr(factual_entity, "model_dump"):
            attrs = factual_entity.model_dump().keys()
        else:
            return True
    for attr in attrs:
        factual_value = getattr(factual_entity, attr, None)
        if factual_value is None:
            continue
        if not _values_semantically_equal(getattr(fictional_entity, attr, None), factual_value):
            return True
    return False
