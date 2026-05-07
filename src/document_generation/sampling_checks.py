"""Validation and invariant helpers for sampled entity collections."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from src.core.document_schema import EntityCollection
from src.core.annotation_runtime import RuleEngine
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number

DEBUG_SAMPLING = os.environ.get("DEBUG_SAMPLING", "").lower() in {"1", "true", "yes"}


def merge_factual_entities(
    collection: EntityCollection,
    factual_entities: Optional[EntityCollection],
) -> None:
    """Copy factual entities into collection for any IDs not already present."""
    if not factual_entities:
        return
    for attr in (
        "persons",
        "places",
        "events",
        "organizations",
        "awards",
        "legals",
        "products",
        "numbers",
        "temporals",
    ):
        factual_dict = getattr(factual_entities, attr)
        target_dict = getattr(collection, attr)
        for entity_id, entity in factual_dict.items():
            if entity_id not in target_dict:
                target_dict[entity_id] = entity.model_copy(deep=True)


def validate_rules_with_details(rules: List[str], entities: EntityCollection, logger) -> Dict[str, Any]:
    try:
        results = RuleEngine.validate_all_rules(rules, entities)
        failed = [rule for rule, ok in results if not ok]
        return {
            "all_valid": len(failed) == 0,
            "failed_rules": failed,
            "passed_rules": [rule for rule, ok in results if ok],
        }
    except Exception as exc:
        logger.debug("Rule validation raised: %s", exc)
        return {"all_valid": False, "failed_rules": [str(exc)], "passed_rules": []}


def verify_ordering_preserved(
    entities: EntityCollection,
    factual_entities: Optional[EntityCollection],
    *,
    preserve_temporal_ordering: bool = True,
    preserve_number_ordering: bool = True,
    forced_equal_entity_ids: Optional[set[str]] = None,
) -> bool:
    if not factual_entities:
        return True
    forced_equal_entity_ids = set(forced_equal_entity_ids or set())
    temporal_fictional: Dict[str, float] = {}
    temporal_factual: Dict[str, float] = {}
    age_fictional: Dict[str, float] = {}
    age_factual: Dict[str, float] = {}
    number_fictional: Dict[str, Dict[str, float]] = {}
    number_factual: Dict[str, Dict[str, float]] = {}
    # Temporal ordering is an invariant for fictional generation. The optional
    # flag is retained for compatibility with older call sites/tests.
    for temp_id in entities.temporals:
        temporal = entities.temporals[temp_id]
        if getattr(temporal, "year", None) is not None:
            temporal_fictional[f"{temp_id}.year"] = float(temporal.year)
            if temp_id in factual_entities.temporals:
                factual_temporal = factual_entities.temporals[temp_id]
                value = getattr(factual_temporal, "year", None)
                if value is not None:
                    temporal_factual[f"{temp_id}.year"] = float(value)
    for person_id in entities.persons:
        person = entities.persons[person_id]
        age = getattr(person, "age", None) if not isinstance(person, dict) else person.get("age")
        if age is not None:
            age_fictional[f"{person_id}.age"] = float(age)
            if person_id in factual_entities.persons:
                factual_person = factual_entities.persons[person_id]
                value = (
                    getattr(factual_person, "age", None)
                    if not isinstance(factual_person, dict)
                    else factual_person.get("age")
                )
                if value is not None:
                    age_factual[f"{person_id}.age"] = float(value)
    for number_id in entities.numbers:
        bucket = _number_order_bucket(entities.numbers[number_id])
        fictional_value = _coerce_number_value(entities.numbers[number_id])
        if fictional_value is None:
            continue
        if bucket is None:
            continue
        number_fictional.setdefault(bucket, {})[f"{number_id}.value"] = fictional_value
        if number_id in factual_entities.numbers:
            factual_bucket = _number_order_bucket(factual_entities.numbers[number_id])
            factual_value = _coerce_number_value(factual_entities.numbers[number_id])
            if factual_value is not None:
                target_bucket = factual_bucket or bucket
                number_factual.setdefault(target_bucket, {})[f"{number_id}.value"] = factual_value
    return (
        _ordering_matches(
            factual_values=temporal_factual,
            fictional_values=temporal_fictional,
            forced_equal_entity_ids=forced_equal_entity_ids,
        )
        and _ordering_matches(
            factual_values=age_factual,
            fictional_values=age_fictional,
            forced_equal_entity_ids=forced_equal_entity_ids,
        )
        and (
            not preserve_number_ordering
            or all(
                _ordering_matches(
                    factual_values=number_factual.get(bucket, {}),
                    fictional_values=number_fictional.get(bucket, {}),
                    forced_equal_entity_ids=forced_equal_entity_ids,
                )
                for bucket in set(number_factual) | set(number_fictional)
            )
        )
    )


def _ordering_matches(
    *,
    factual_values: Dict[str, float],
    fictional_values: Dict[str, float],
    forced_equal_entity_ids: Optional[set[str]] = None,
) -> bool:
    forced_equal_entity_ids = set(forced_equal_entity_ids or set())
    common_keys = [key for key in factual_values if key in fictional_values]
    for i in range(len(common_keys)):
        for j in range(i + 1, len(common_keys)):
            key_i, key_j = common_keys[i], common_keys[j]
            factual_i, factual_j = factual_values[key_i], factual_values[key_j]
            fictional_i, fictional_j = fictional_values[key_i], fictional_values[key_j]
            entity_i = key_i.split(".", 1)[0]
            entity_j = key_j.split(".", 1)[0]
            if (
                entity_i in forced_equal_entity_ids
                and entity_j in forced_equal_entity_ids
                and fictional_i == fictional_j
            ):
                continue
            if (factual_i < factual_j and fictional_i >= fictional_j) or (
                factual_i > factual_j and fictional_i <= fictional_j
            ):
                if DEBUG_SAMPLING:
                    print(
                        "[dbg] ordering violation: "
                        f"{key_i}={fictional_i} vs {key_j}={fictional_j} "
                        f"(factual {factual_i} vs {factual_j})",
                        flush=True,
                    )
                return False
    return True


def _coerce_number_value(number: Any) -> float | None:
    getter = number.get if isinstance(number, dict) else getattr
    for attr in ("float", "percent", "proportion", "int"):
        raw_value = getter(attr, None) if isinstance(number, dict) else getter(number, attr, None)
        if raw_value is None:
            continue
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            continue
    return None


def _number_order_bucket(number: Any) -> str | None:
    getter = number.get if isinstance(number, dict) else getattr
    if (getter("percent", None) if isinstance(number, dict) else getter(number, "percent", None)) is not None:
        return "percent"
    if (getter("proportion", None) if isinstance(number, dict) else getter(number, "proportion", None)) is not None:
        return "proportion"
    if (getter("float", None) if isinstance(number, dict) else getter(number, "float", None)) is not None:
        return "float"
    if (getter("fraction", None) if isinstance(number, dict) else getter(number, "fraction", None)) is not None:
        return "int_like"
    if (getter("int", None) if isinstance(number, dict) else getter(number, "int", None)) is not None:
        return "int_like"
    return None


def verify_required_differences(
    required_entities: Dict[str, List[Tuple[str, List[str]]]],
    sampled_entities: EntityCollection,
    factual_entities: Optional[EntityCollection],
    person_diff_exempt_attrs: frozenset[str],
    forced_equal_entity_ids: Optional[set[str]] = None,
    forced_equal_entity_refs: Optional[set[str]] = None,
) -> bool:
    """Require factual/fictional difference on replaceable required attributes."""
    if not factual_entities:
        return True
    forced_equal_entity_ids = set(forced_equal_entity_ids or set())
    forced_equal_entity_refs = set(forced_equal_entity_refs or set())
    for entity_type, specs in required_entities.items():
        for entity_id, attrs in specs:
            if entity_id in forced_equal_entity_ids:
                continue
            for attr in attrs:
                if f"{entity_id}.{attr}" in forced_equal_entity_refs:
                    continue
                if not _attr_requires_difference(entity_type, attr, person_diff_exempt_attrs):
                    continue
                factual_val = RuleEngine._get_entity_value(factual_entities, f"{entity_id}.{attr}")
                fictional_val = RuleEngine._get_entity_value(sampled_entities, f"{entity_id}.{attr}")
                if factual_val is None or fictional_val is None:
                    continue
                if _values_equal(factual_val, fictional_val):
                    if DEBUG_SAMPLING:
                        print(
                            "[dbg] unchanged required ref: "
                            f"{entity_id}.{attr}={fictional_val!r}",
                            flush=True,
                        )
                    return False
    return True


def resample_equal_person_ages(
    collection: EntityCollection,
    required_entities: Dict[str, List[Tuple[str, List[str]]]],
    factual_entities: Optional[EntityCollection],
    age_bounds_getter,
    age_window_getter,
    sample_person_age,
) -> None:
    """Resample age when it matches factual or falls outside the local window."""
    if not factual_entities:
        return
    person_specs = required_entities.get("person", [])
    if not person_specs:
        return
    min_age, max_age = age_bounds_getter()
    for person_id, attrs in person_specs:
        if "age" not in attrs:
            continue
        person = collection.persons.get(person_id)
        factual_person = factual_entities.persons.get(person_id)
        if person is None or factual_person is None:
            continue
        current_age = getattr(person, "age", None)
        factual_age = getattr(factual_person, "age", None)
        if current_age is None or factual_age is None:
            continue
        low, high = age_window_getter(person_id, min_age, max_age)
        if low <= int(current_age) <= high and int(current_age) != int(factual_age):
            continue
        person.age = sample_person_age(person_id, min_age, max_age)


def preserve_age_ordering(
    collection: EntityCollection,
    factual_entities: Optional[EntityCollection],
) -> None:
    if not factual_entities or not getattr(factual_entities, "persons", None):
        return
    factual_ages = {
        pid: (getattr(person, "age", None) or (person.get("age") if isinstance(person, dict) else None))
        for pid, person in factual_entities.persons.items()
    }
    factual_ages = {key: value for key, value in factual_ages.items() if value is not None}
    fictional_ages = {
        pid: (getattr(person, "age", None) or (person.get("age") if isinstance(person, dict) else None))
        for pid, person in collection.persons.items()
    }
    fictional_ages = {key: value for key, value in fictional_ages.items() if value is not None}
    common_ids = [pid for pid in factual_ages if pid in fictional_ages]
    if len(common_ids) <= 1:
        return
    factual_sorted = sorted(common_ids, key=lambda pid: factual_ages[pid])
    fictional_vals = sorted([fictional_ages[pid] for pid in factual_sorted])
    for i, person_id in enumerate(factual_sorted):
        new_age = fictional_vals[i]
        person = collection.persons[person_id]
        if isinstance(person, dict):
            person["age"] = new_age
        else:
            person.age = new_age


def _values_equal(lhs: Any, rhs: Any) -> bool:
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
        lhs_float = float(lhs)
        rhs_float = float(rhs)
        if abs(lhs_float - rhs_float) > 1e-9:
            return False
        lhs_int_like = abs(lhs_float - round(lhs_float)) <= 1e-9
        rhs_int_like = abs(rhs_float - round(rhs_float)) <= 1e-9
        if lhs_int_like and rhs_int_like:
            return int(round(lhs_float)) == int(round(rhs_float))
        return True
    except (TypeError, ValueError):
        pass
    return str(lhs).strip().lower() == str(rhs).strip().lower()


def _attr_requires_difference(
    entity_type: str,
    attribute: str,
    person_diff_exempt_attrs: frozenset[str],
) -> bool:
    if not attribute:
        return False
    root = attribute.split(".", 1)[0]
    if entity_type == "person" and root in person_diff_exempt_attrs:
        return False
    return True
