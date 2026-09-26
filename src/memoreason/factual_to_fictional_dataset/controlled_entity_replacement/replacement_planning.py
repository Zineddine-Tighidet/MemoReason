"""Plan which factual entities are replaced in one fictional document variant."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from memoreason.benchmark_definition.annotation_runtime import (
    FULL_REPLACE_ENTITY_TYPES,
    PARTIAL_REPLACE_ATTRIBUTES,
    RuleEngine,
    find_entity_refs,
)
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

RequiredEntityMap = Dict[str, List[Tuple[str, List[str]]]]
EntityCollectionsByPluralType = Dict[str, Dict[str, Any]]

_ANNOTATED_BIRTH_TEMPORAL_PATTERN = re.compile(
    r"\bborn\b[^.\n]{0,120}?\[[^\]]+;\s*(temporal_\d+)\.(?:date|year)\]",
    re.IGNORECASE,
)
_ANNOTATED_AGE_EVENT_PATTERN = re.compile(
    r"\bat age\s+\[[^\]]+;\s*((?:person|number)_\d+)\.(?:age|int|str)\]"
    r"[^.\n]{0,80}?\bin\s+"
    r"(?:\[[^\]]+;\s*temporal_\d+\.month\]\s+)?"
    r"\[[^\]]+;\s*(temporal_\d+)\.(?:date|year)\]",
    re.IGNORECASE,
)


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


def fixed_constant_entity_ids_from_rules(
    *,
    rules: Iterable[str],
    factual_entities: EntityCollection,
) -> Dict[str, Set[str]]:
    """Return number/temporal entities fixed to their factual value by equality rules."""
    fixed: Dict[str, Set[str]] = {}
    for raw_rule in rules or []:
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        if not cleaned:
            continue
        if "==" in cleaned:
            lhs, rhs = cleaned.split("==", 1)
        elif "=" in cleaned and all(op not in cleaned for op in (">=", "<=", "!=")):
            lhs, rhs = cleaned.split("=", 1)
        else:
            continue
        lhs_refs = find_entity_refs(lhs)
        rhs_refs = find_entity_refs(rhs)
        if len(lhs_refs) == 1 and not rhs_refs:
            ref = lhs_refs[0]
            constant = rhs
        elif len(rhs_refs) == 1 and not lhs_refs:
            ref = rhs_refs[0]
            constant = lhs
        else:
            continue
        entity_id, _sep, _attr = ref.partition(".")
        entity_type = entity_id.split("_", 1)[0]
        if entity_type not in {"number", "temporal"}:
            continue
        factual_value = RuleEngine._get_entity_value(factual_entities, ref)
        cleaned_constant = str(constant).strip().strip('"').strip("'")
        if factual_value is not None and _values_semantically_equal(factual_value, cleaned_constant):
            fixed.setdefault(entity_type, set()).add(entity_id)
    return fixed


def partial_rule_locked_entity_ids_from_rules(
    *,
    rules: Iterable[str],
) -> Dict[str, Set[str]]:
    """Return numerical rule components that should not be split across partial variants."""
    locked: Dict[str, Set[str]] = {}
    for raw_rule in rules or []:
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        if not cleaned:
            continue
        refs = find_entity_refs(cleaned)
        entity_ids_by_type: Dict[str, Set[str]] = {}
        for ref in refs:
            entity_id, _sep, attr = ref.partition(".")
            entity_type = entity_id.split("_", 1)[0]
            if entity_type in {"number", "temporal"} or (entity_type == "person" and attr == "age"):
                entity_ids_by_type.setdefault(entity_type, set()).add(entity_id)
        if sum(len(entity_ids) for entity_ids in entity_ids_by_type.values()) > 1:
            for entity_type, entity_ids in entity_ids_by_type.items():
                locked.setdefault(entity_type, set()).update(entity_ids)

    return locked


def partial_rule_linked_entity_groups_from_rules(
    *,
    rules: Iterable[str],
) -> List[frozenset[str]]:
    """Return rule components that must be replaced or retained together."""
    groups: List[frozenset[str]] = []
    for raw_rule in rules or []:
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        if not cleaned:
            continue
        linked_entity_ids = set()
        for ref in find_entity_refs(cleaned):
            entity_id, _sep, attr = ref.partition(".")
            entity_type = entity_id.split("_", 1)[0]
            if entity_type in {"number", "temporal"} or (entity_type == "person" and attr == "age"):
                linked_entity_ids.add(entity_id)
        if len(linked_entity_ids) > 1:
            groups.append(frozenset(linked_entity_ids))
    return groups


def partial_chronology_locked_entity_ids_from_text(
    *,
    annotated_text: str,
) -> Dict[str, Set[str]]:
    """Keep birth, age, and event-year anchors on the same factual side in partial variants."""
    birth_temporal_ids = set(_ANNOTATED_BIRTH_TEMPORAL_PATTERN.findall(annotated_text or ""))
    age_event_matches = _ANNOTATED_AGE_EVENT_PATTERN.findall(annotated_text or "")
    if not birth_temporal_ids or not age_event_matches:
        return {}

    locked: Dict[str, Set[str]] = {"temporal": set(birth_temporal_ids)}
    for age_entity_id, event_temporal_id in age_event_matches:
        age_entity_type = age_entity_id.split("_", 1)[0]
        locked.setdefault(age_entity_type, set()).add(age_entity_id)
        locked["temporal"].add(event_temporal_id)
    return locked


def partial_chronology_linked_entity_groups_from_text(
    *,
    annotated_text: str,
) -> List[frozenset[str]]:
    """Return birth, age, and event anchors that must stay on the same side."""
    locked = partial_chronology_locked_entity_ids_from_text(annotated_text=annotated_text)
    linked_entity_ids = frozenset(
        entity_id
        for entity_ids in locked.values()
        for entity_id in entity_ids
    )
    return [linked_entity_ids] if len(linked_entity_ids) > 1 else []


def _select_exact_linked_candidate_indices(
    *,
    replacement_candidates: List[Tuple[str, str, Optional[frozenset[str]]]],
    target_replacement_count: int,
    linked_entity_groups: Iterable[Iterable[str]],
) -> Set[int]:
    """Select an exact-size random subset without splitting linked components."""
    candidate_count = len(replacement_candidates)
    if target_replacement_count <= 0:
        return set()
    if target_replacement_count >= candidate_count:
        return set(range(candidate_count))

    parent = list(range(candidate_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(lhs: int, rhs: int) -> None:
        lhs_root = find(lhs)
        rhs_root = find(rhs)
        if lhs_root != rhs_root:
            parent[rhs_root] = lhs_root

    candidate_index_by_id = {
        entity_id: candidate_index
        for candidate_index, (_entity_type, entity_id, _replaceable_attrs) in enumerate(replacement_candidates)
    }
    for raw_group in linked_entity_groups:
        group_indices = sorted(
            candidate_index_by_id[entity_id]
            for entity_id in set(raw_group)
            if entity_id in candidate_index_by_id
        )
        for group_index in group_indices[1:]:
            union(group_indices[0], group_index)

    components_by_root: Dict[int, Set[int]] = {}
    for candidate_index in range(candidate_count):
        components_by_root.setdefault(find(candidate_index), set()).add(candidate_index)
    components = list(components_by_root.values())
    random.shuffle(components)

    # The first predecessor found after the seeded shuffle defines a deterministic
    # random exact subset while keeping every connected component atomic.
    predecessor: Dict[int, Tuple[int, int]] = {}
    reachable = {0}
    for component_index, component in enumerate(components):
        component_size = len(component)
        for subtotal in sorted(reachable, reverse=True):
            new_total = subtotal + component_size
            if new_total > target_replacement_count or new_total in reachable:
                continue
            predecessor[new_total] = (subtotal, component_index)
            reachable.add(new_total)

    if target_replacement_count not in reachable:
        component_sizes = sorted(len(component) for component in components)
        raise ValueError(
            "Exact partial replacement target is impossible without splitting linked entities: "
            f"target={target_replacement_count}, eligible={candidate_count}, "
            f"linked_component_sizes={component_sizes}"
        )

    selected_indices: Set[int] = set()
    subtotal = target_replacement_count
    while subtotal:
        previous_subtotal, component_index = predecessor[subtotal]
        selected_indices.update(components[component_index])
        subtotal = previous_subtotal
    return selected_indices


def build_replacement_plan(
    *,
    entity_types: EntityCollectionsByPluralType,
    required_entities: RequiredEntityMap,
    replacement_proportion: float,
    replace_mode: str,
    excluded_entity_ids: Dict[str, Set[str]] | None = None,
    eligible_entity_ids: Dict[str, Set[str]] | None = None,
    linked_entity_groups: Iterable[Iterable[str]] = (),
) -> ReplacementPlan:
    """Choose which factual entities will be replaced in one fictional variant."""
    full_types = FULL_REPLACE_ENTITY_TYPES.get(replace_mode, set())
    partial_types = PARTIAL_REPLACE_ATTRIBUTES.get(replace_mode, {})
    excluded_entity_ids = excluded_entity_ids or {}

    required_attr_map: Dict[Tuple[str, str], Set[str]] = {}
    for entity_type, specs in required_entities.items():
        for entity_id, attrs in specs:
            required_attr_map[(entity_type, entity_id)] = set(attrs or [])

    replacement_candidates: List[Tuple[str, str, Optional[frozenset[str]]]] = []
    for entity_type_plural, entities_dict in entity_types.items():
        entity_type = entity_type_plural.rstrip("s")
        for entity_id in sorted(entities_dict.keys()):
            if entity_id in excluded_entity_ids.get(entity_type, set()):
                continue
            if eligible_entity_ids is not None and entity_id not in eligible_entity_ids.get(entity_type, set()):
                continue
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
    linked_entity_groups = list(linked_entity_groups)
    if linked_entity_groups:
        selected_indices = _select_exact_linked_candidate_indices(
            replacement_candidates=replacement_candidates,
            target_replacement_count=target_replacement_count,
            linked_entity_groups=linked_entity_groups,
        )
    else:
        selected_indices = (
            set(random.sample(range(eligible_entity_count), target_replacement_count))
            if target_replacement_count > 0
            else set()
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
