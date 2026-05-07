"""Manual-entity sampling with cross-entity constraints."""

from __future__ import annotations

import copy
import random
import time
from typing import Any

from .generation_limits import (
    MAX_MANUAL_BACKTRACK_STEPS,
    SLOW_SAMPLING_LOG_SECONDS,
)
from src.core.document_schema import ENTITY_TYPE_TO_CLASS, EntityCollection
from src.core.organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    organization_attribute_value,
    organization_pool_bucket,
)

_PERSON_DERIVED_ATTRS = {
    "gender",
    "subj_pronoun",
    "obj_pronoun",
    "poss_det_pronoun",
    "poss_pro_pronoun",
    "refl_pronoun",
    "relationship",
    "relationships",
    "honorific",
}

_EVENT_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "debate": ("debate",),
    "nomination": ("nomination", "nominat", "primary", "primaries"),
    "election": ("election", "electoral", "ballot", "vote"),
    "pandemic": ("pandemic", "outbreak", "epidemic", "contagion", "plague"),
    "war": ("war", "conflict", "invasion", "incursion", "siege", "campaign", "offensive", "withdrawal"),
}


def parse_manual_constraint(rule: str) -> dict[str, Any] | None:
    cleaned = rule.strip()
    if not cleaned:
        return None
    if " in " in cleaned:
        parts = cleaned.split(" in ")
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            if "." in left and "." in right:
                l_ent, l_attr = left.split(".", 1)
                r_ent, r_attr = right.split(".", 1)
                return {"op": "in", "left": (l_ent, l_attr), "right": (r_ent, r_attr)}
        return None
    for op in ("!=", "==", "="):
        if op in cleaned:
            parts = cleaned.split(op)
            if len(parts) == 2:
                left, right = parts[0].strip(), parts[1].strip()
                if "." in left and "." in right:
                    l_ent, l_attr = left.split(".", 1)
                    r_ent, r_attr = right.split(".", 1)
                    return {"op": op, "left": (l_ent, l_attr), "right": (r_ent, r_attr)}
            return None
    return None


def sample_manual_entities_with_constraints(
    sampler: Any,
    manual_required: dict[str, list[tuple[str, list[str]]]],
    manual_rules: list[str],
) -> EntityCollection | None:
    """Sample manual entities while enforcing parsed manual cross-entity rules."""
    start_time = time.monotonic()
    variables: list[tuple[str, str, list[str]]] = []
    for entity_type, specs in manual_required.items():
        for entity_id, required_attrs in specs:
            variables.append((entity_id, entity_type, required_attrs))

    domains: dict[str, list[Any]] = {}
    for entity_id, entity_type, required_attrs in variables:
        domains[entity_id] = _eligible_domain(sampler, entity_id, entity_type, required_attrs)

    constraints = []
    for raw in manual_rules:
        cleaned = sampler._strip_rule_comment(str(raw))
        parsed = parse_manual_constraint(cleaned)
        if parsed:
            constraints.append(parsed)
    constrained_ids = {
        entity_id for constraint in constraints for entity_id in (constraint["left"][0], constraint["right"][0])
    }
    constrained_variables = [var for var in variables if var[0] in constrained_ids]
    unconstrained_variables = [var for var in variables if var[0] not in constrained_ids]

    used_entities: dict[str, list[Any]] = {}

    def attr_value(entity: Any, attr: str):
        return entity.get(attr) if isinstance(entity, dict) else getattr(entity, attr, None)

    def key_attrs(entity_type: str, required_attrs: list[str]) -> list[str]:
        return sampler._get_key_attributes(entity_type, required_attrs)

    def prune_domains() -> bool:
        changed = True
        while changed:
            changed = False
            for constraint in constraints:
                (l_ent, l_attr) = constraint["left"]
                (r_ent, r_attr) = constraint["right"]
                op = constraint["op"]
                if l_ent not in domains or r_ent not in domains:
                    continue
                left_domain = domains[l_ent]
                right_domain = domains[r_ent]
                if not left_domain or not right_domain:
                    return False
                if op in ("=", "=="):
                    right_vals = {attr_value(e, r_attr) for e in right_domain}
                    left_vals = {attr_value(e, l_attr) for e in left_domain}
                    new_left = [e for e in left_domain if attr_value(e, l_attr) in right_vals]
                    new_right = [e for e in right_domain if attr_value(e, r_attr) in left_vals]
                    if len(new_left) != len(left_domain):
                        domains[l_ent] = new_left
                        left_domain = new_left
                        changed = True
                    if len(new_right) != len(right_domain):
                        domains[r_ent] = new_right
                        right_domain = new_right
                        changed = True
                elif op == "!=":
                    if len(left_domain) == 1:
                        lv = attr_value(left_domain[0], l_attr)
                        new_right = [e for e in right_domain if attr_value(e, r_attr) != lv]
                        if len(new_right) != len(right_domain):
                            domains[r_ent] = new_right
                            changed = True
                    if len(right_domain) == 1:
                        rv = attr_value(right_domain[0], r_attr)
                        new_left = [e for e in left_domain if attr_value(e, l_attr) != rv]
                        if len(new_left) != len(left_domain):
                            domains[l_ent] = new_left
                            changed = True
                elif op == "in":
                    right_vals = [attr_value(e, r_attr) for e in right_domain]
                    left_vals = [attr_value(e, l_attr) for e in left_domain]
                    new_left = [
                        e for e in left_domain if any(str(attr_value(e, l_attr)) in str(rv) for rv in right_vals)
                    ]
                    new_right = [
                        e for e in right_domain if any(str(lv) in str(attr_value(e, r_attr)) for lv in left_vals)
                    ]
                    if len(new_left) != len(left_domain):
                        domains[l_ent] = new_left
                        left_domain = new_left
                        changed = True
                    if len(new_right) != len(right_domain):
                        domains[r_ent] = new_right
                        right_domain = new_right
                        changed = True
                if not domains[l_ent] or not domains[r_ent]:
                    return False
        return True

    if constraints and not prune_domains():
        return None

    for key in domains:
        random.shuffle(domains[key])

    degree = {eid: 0 for eid in domains}
    for constraint in constraints:
        degree[constraint["left"][0]] = degree.get(constraint["left"][0], 0) + 1
        degree[constraint["right"][0]] = degree.get(constraint["right"][0], 0) + 1

    required_lookup = {eid: (etype, attrs) for eid, etype, attrs in variables}
    steps = 0

    def consistent(entity_id: str, candidate: Any, assignments: dict[str, Any]) -> bool:
        entity_type, required_attrs = required_lookup[entity_id]
        if entity_type in used_entities:
            for used in used_entities[entity_type]:
                if sampler._entities_equal(candidate, used, key_attrs(entity_type, required_attrs)):
                    return False

        for constraint in constraints:
            (l_ent, l_attr) = constraint["left"]
            (r_ent, r_attr) = constraint["right"]
            op = constraint["op"]
            if entity_id not in (l_ent, r_ent):
                continue
            other_id = r_ent if entity_id == l_ent else l_ent
            if other_id not in assignments:
                continue
            left_val = (
                attr_value(candidate, l_attr) if entity_id == l_ent else attr_value(assignments[other_id], l_attr)
            )
            right_val = (
                attr_value(assignments[other_id], r_attr) if entity_id == l_ent else attr_value(candidate, r_attr)
            )
            if left_val is None or right_val is None:
                continue
            if op in ("=", "==") and left_val != right_val:
                return False
            if op == "!=" and left_val == right_val:
                return False
            if op == "in" and str(left_val) not in str(right_val):
                return False
        return True

    def choose_var(assignments: dict[str, Any]) -> str | None:
        remaining = [eid for eid, _, _ in constrained_variables if eid not in assignments]
        if not remaining:
            return None
        return min(remaining, key=lambda eid: (len(domains.get(eid, [])), -degree.get(eid, 0)))

    def backtrack(assignments: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal steps
        steps += 1
        if steps > MAX_MANUAL_BACKTRACK_STEPS:
            return None
        if steps % 5000 == 0 and (time.monotonic() - start_time) > SLOW_SAMPLING_LOG_SECONDS:
            print(
                f"[slow] Manual CSP still searching ({steps} steps, {time.monotonic() - start_time:.1f}s, {len(variables)} vars).",
                flush=True,
            )
        var = choose_var(assignments)
        if var is None:
            return assignments
        entity_type, _ = required_lookup[var]
        for candidate in domains.get(var, []):
            if not consistent(var, candidate, assignments):
                continue
            assignments[var] = candidate
            used_entities.setdefault(entity_type, []).append(candidate)
            solved = backtrack(assignments)
            if solved is not None:
                return solved
            used_entities[entity_type].pop()
            if not used_entities[entity_type]:
                del used_entities[entity_type]
            assignments.pop(var, None)
        return None

    assignment = backtrack({})
    if assignment is None:
        return None

    def assign_unconstrained(assignments: dict[str, Any]) -> dict[str, Any] | None:
        result = dict(assignments)
        grouped: dict[str, list[tuple[str, list[str]]]] = {}
        for entity_id, entity_type, required_attrs in unconstrained_variables:
            grouped.setdefault(entity_type, []).append((entity_id, required_attrs))

        for entity_type, specs in grouped.items():
            local_used = list(used_entities.get(entity_type, []))
            local_required = {entity_id: attrs for entity_id, attrs in specs}

            def local_consistent(entity_id: str, candidate: Any, local_assignments: dict[str, Any]) -> bool:
                required_attrs = local_required[entity_id]
                keys = key_attrs(entity_type, required_attrs)
                for used in local_used:
                    if sampler._entities_equal(candidate, used, keys):
                        return False
                for other_id, other_candidate in local_assignments.items():
                    other_keys = key_attrs(entity_type, local_required[other_id])
                    if sampler._entities_equal(candidate, other_candidate, keys) or sampler._entities_equal(
                        candidate,
                        other_candidate,
                        other_keys,
                    ):
                        return False
                return True

            def local_backtrack(local_assignments: dict[str, Any]) -> dict[str, Any] | None:
                remaining = [entity_id for entity_id, _ in specs if entity_id not in local_assignments]
                if not remaining:
                    return dict(local_assignments)
                entity_id = min(remaining, key=lambda current_id: len(domains.get(current_id, [])))
                for candidate in domains.get(entity_id, []):
                    if not local_consistent(entity_id, candidate, local_assignments):
                        continue
                    local_assignments[entity_id] = candidate
                    solved = local_backtrack(local_assignments)
                    if solved is not None:
                        return solved
                    local_assignments.pop(entity_id, None)
                return None

            local_solution = local_backtrack({})
            if local_solution is None:
                return None
            for entity_id, candidate in local_solution.items():
                result[entity_id] = candidate
                local_used.append(candidate)

        return result

    assignment = assign_unconstrained(assignment)
    if assignment is None:
        return None

    elapsed = time.monotonic() - start_time
    if elapsed > SLOW_SAMPLING_LOG_SECONDS:
        print(f"[slow] Manual CSP solved in {elapsed:.1f}s ({steps} steps).", flush=True)

    base = EntityCollection()
    for entity_id, entity_type, required_attrs in variables:
        selected = assignment[entity_id]
        sampled_entity = _to_entity_instance(entity_type, selected)
        if entity_type == "person":
            _reset_person_derived_fields(sampled_entity)
        if entity_type == "person" and "age" in required_attrs:
            _ensure_person_age_within_window(
                sampler=sampler,
                entity_id=entity_id,
                required_attrs=required_attrs,
                sampled_entity=sampled_entity,
            )
        base.add_entity(entity_type, entity_id, sampled_entity)
    return base


def _to_entity_instance(entity_type: str, selected: Any):
    if isinstance(selected, dict):
        entity_class = ENTITY_TYPE_TO_CLASS[entity_type]
        return entity_class(**copy.deepcopy(selected))
    return selected.model_copy(deep=True)


def _reset_person_derived_fields(person_entity: Any) -> None:
    for attr in _PERSON_DERIVED_ATTRS:
        setattr(person_entity, attr, None if attr != "relationships" else {})


def _ensure_person_age_within_window(
    sampler: Any,
    entity_id: str,
    required_attrs: list[str],
    sampled_entity: Any,
) -> None:
    if "age" not in required_attrs:
        return
    min_age, max_age = sampler._get_age_bounds_from_rules()
    low, high = sampler._age_window(entity_id, min_age, max_age)
    current_age = getattr(sampled_entity, "age", None)
    try:
        current_age_int = int(current_age) if current_age is not None else None
    except (TypeError, ValueError):
        current_age_int = None
    factual_age = sampler._factual_person_age(entity_id)
    needs_resample = (
        current_age_int is None
        or current_age_int < low
        or current_age_int > high
        or (factual_age is not None and current_age_int == factual_age)
    )
    if needs_resample:
        sampled_entity.age = sampler._sample_person_age(entity_id, min_age, max_age)


def _event_family_from_text(text: str | None) -> str | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None
    for family, keywords in _EVENT_FAMILY_KEYWORDS.items():
        if any(keyword in raw for keyword in keywords):
            return family
    return None


def _factual_event_family(sampler: Any, entity_id: str) -> str | None:
    if not getattr(sampler, "factual_entities", None):
        return None
    factual_event = getattr(sampler.factual_entities, "events", {}).get(entity_id)
    if factual_event is None:
        return None
    event_type = factual_event.get("type") if isinstance(factual_event, dict) else getattr(factual_event, "type", None)
    event_name = factual_event.get("name") if isinstance(factual_event, dict) else getattr(factual_event, "name", None)
    return _event_family_from_text(event_type) or _event_family_from_text(event_name)


def _candidate_event_family(candidate: Any) -> str | None:
    event_type = candidate.get("type") if isinstance(candidate, dict) else getattr(candidate, "type", None)
    event_name = candidate.get("name") if isinstance(candidate, dict) else getattr(candidate, "name", None)
    return _event_family_from_text(event_type) or _event_family_from_text(event_name)


def _eligible_domain(sampler: Any, entity_id: str, entity_type: str, required_attrs: list[str]) -> list[Any]:
    pool_map = {
        "person": sampler.entity_pool.get("persons", []),
        "place": sampler.entity_pool.get("places", []),
        "event": sampler.entity_pool.get("events", []),
        "award": sampler.entity_pool.get("awards", []),
        "legal": sampler.entity_pool.get("legals", []),
        "product": sampler.entity_pool.get("products", []),
    }
    if entity_type in CANONICAL_ORGANIZATION_TYPES:
        pool_bucket = organization_pool_bucket(entity_type) or "organizations"
        pool = sampler.entity_pool.get(pool_bucket, [])
    else:
        pool = pool_map.get(entity_type, [])
    excluded = _PERSON_DERIVED_ATTRS | {"age"}
    pool_required = [attr for attr in required_attrs if attr not in excluded]
    event_family = _factual_event_family(sampler, entity_id) if entity_type == "event" else None
    cache_key = (entity_type, tuple(sorted(pool_required)), event_family)
    if cache_key in sampler._eligible_cache:
        return list(sampler._eligible_cache[cache_key])
    eligible = []
    for entity in pool:
        if all(_get_attr_value(entity, attr) is not None for attr in pool_required):
            if entity_type == "event" and event_family is not None:
                candidate_family = _candidate_event_family(entity)
                if candidate_family is not None and candidate_family != event_family:
                    continue
            eligible.append(entity)
    sampler._eligible_cache[cache_key] = eligible
    return list(eligible)


def _get_attr_value(entity: Any, attr: str):
    if attr in {"name", "organization_kind"}:
        return organization_attribute_value(entity, attr)
    return entity.get(attr) if isinstance(entity, dict) else getattr(entity, attr, None)
