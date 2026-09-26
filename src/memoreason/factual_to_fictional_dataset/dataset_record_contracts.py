"""Question and entity-difference contracts for exported dataset records."""

from __future__ import annotations

from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, find_entity_refs, partition_generation_rules
from memoreason.benchmark_definition.century_expressions import has_century_function
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import (
    ENTITY_TAXONOMY,
    FULL_REPLACE_ENTITY_TYPES,
    LEGACY_ENTITY_ATTRIBUTES,
    PARTIAL_REPLACE_ATTRIBUTES,
    parse_entity_id,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler_common import (
    detect_ordering_excluded_number_ids,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.linear_constraint_certification import (
    transformed_constraints_use_safe_integer_lattice,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)

from .dataset_record_constants import INLINE_ANNOTATION_PATTERN
from .dataset_record_core import _semantically_equal, answer_behavior_label, normalize_question_type


def _drop_nulls(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {key: _drop_nulls(item) for key, item in value.items() if item is not None}
        return {key: item for key, item in cleaned.items() if item not in ({}, [], None)}
    if isinstance(value, list):
        cleaned_list = [_drop_nulls(item) for item in value]
        return [item for item in cleaned_list if item not in ({}, [], None)]
    return value


def _question_entries_from_generated(
    document,
    generated_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    by_question_id = {question.question_id: question for question in document.questions}
    question_entries_for_export: list[dict[str, Any]] = []
    for question_entry in generated_payload.get("questions", []) or []:
        question_id = str(question_entry.get("question_id") or "")
        source_question = by_question_id.get(question_id)
        question_entries_for_export.append(
            {
                "question_id": question_id,
                "question_type": normalize_question_type(
                    question_entry.get("question_type")
                    or (source_question.question_type if source_question is not None else None)
                ),
                "answer_behavior": answer_behavior_label(
                    getattr(source_question, "answer_type", None) if source_question is not None else None,
                ),
                "answer_type": answer_behavior_label(
                    getattr(source_question, "answer_type", None) if source_question is not None else None,
                ),
                "question_text": question_entry.get("question", ""),
                "answer_expression": question_entry.get("answer_expression", ""),
                "answer_entities": question_entry.get("answer_entities"),
                "evaluated_answer": str(question_entry.get("evaluated_answer", "")).strip(),
                "accepted_answer_overrides": list(
                    getattr(source_question, "accepted_answer_overrides", []) if source_question is not None else []
                ),
            }
        )
    return question_entries_for_export


def _contains_inline_annotations(text: str) -> bool:
    return bool(INLINE_ANNOTATION_PATTERN.search(text or ""))


def _required_refs_for_difference_check(document) -> dict[str, list[tuple[str, list[str]]]]:
    return FictionalEntitySampler.extract_required_entities(document, include_questions=True)


def _explicitly_forced_equal_refs(document, factual_entities: EntityCollection) -> set[str]:
    fixed_refs: set[str] = set()
    equality_neighbors: dict[str, set[str]] = {}
    for raw_rule in document.rules or []:
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        if not cleaned:
            continue
        split = NumberTemporalGenerator._split_rule(cleaned)
        if split is None:
            continue
        left, op, right = split
        if op not in {"=", "=="}:
            continue
        left_refs = find_entity_refs(left)
        right_refs = find_entity_refs(right)
        if len(left_refs) == 1 and len(right_refs) == 1:
            left_ref = left_refs[0]
            right_ref = right_refs[0]
            if (left_ref.startswith("number_") or left_ref.startswith("temporal_")) and (
                right_ref.startswith("number_") or right_ref.startswith("temporal_")
            ):
                equality_neighbors.setdefault(left_ref, set()).add(right_ref)
                equality_neighbors.setdefault(right_ref, set()).add(left_ref)
            continue
        if len(left_refs) == 1 and not right_refs:
            ref = left_refs[0]
            constant = right
        elif len(right_refs) == 1 and not left_refs:
            ref = right_refs[0]
            constant = left
        else:
            continue
        if not (ref.startswith("number_") or ref.startswith("temporal_")):
            continue
        factual_value = RuleEngine._get_entity_value(factual_entities, ref)
        if factual_value is None:
            continue
        cleaned_constant = constant.strip().strip('"').strip("'")
        if _semantically_equal(factual_value, cleaned_constant):
            fixed_refs.add(ref)
    queue = list(fixed_refs)
    while queue:
        current_ref = queue.pop()
        current_value = RuleEngine._get_entity_value(factual_entities, current_ref)
        for neighbor_ref in equality_neighbors.get(current_ref, ()):
            if neighbor_ref in fixed_refs:
                continue
            neighbor_value = RuleEngine._get_entity_value(factual_entities, neighbor_ref)
            if current_value is None or neighbor_value is None:
                continue
            if _semantically_equal(current_value, neighbor_value):
                fixed_refs.add(neighbor_ref)
                queue.append(neighbor_ref)
    implicit_generator = NumberTemporalGenerator(
        factual_entities=factual_entities,
        implicit_rules=getattr(document, "implicit_rules", None),
    )
    for entity_ref in sorted(implicit_generator._implicit_rule_lookup):
        entity_id, _separator, attribute = entity_ref.partition(".")
        if not entity_id.startswith("temporal_") or not attribute:
            continue
        implicit_generator._implicit_temporal_range(entity_id, attribute)
    fixed_refs.update(implicit_generator.last_implicit_forced_equal_refs)
    required_number_specs = _required_refs_for_difference_check(document).get("number", [])
    required_number_ids = {number_id for number_id, _attrs in required_number_specs}
    if required_number_ids:
        effective_rules, _dropped_rules = partition_generation_rules(document, include_questions=True)
        numeric_rules = [
            rule
            for rule in effective_rules
            if "number_" in str(rule) and "temporal_" not in str(rule) and not has_century_function(str(rule))
        ]
        ordering_excluded_number_ids = detect_ordering_excluded_number_ids(
            getattr(document, "document_to_annotate", "")
        )
        generator = NumberTemporalGenerator(
            factual_entities=factual_entities,
            implicit_rules=getattr(document, "implicit_rules", None),
            ordering_excluded_number_ids=ordering_excluded_number_ids,
        )
        active_numeric_rules = generator._number_evaluable_rules(
            numeric_rules,
            required_number_ids,
            factual_entities,
        )
        ordering_rules = generator._ordering_number_rules(list(required_number_ids), factual_entities)
        if ordering_rules:
            active_numeric_rules = [*active_numeric_rules, *ordering_rules]
        constraints = generator._collect_linear_constraints(active_numeric_rules, required_number_ids, factual_entities)
        if constraints is not None and transformed_constraints_use_safe_integer_lattice(constraints):
            base_domains = {number_id: generator._number_base_range(number_id) for number_id in required_number_ids}
            tightened_domains = generator._tighten_number_domains(constraints, base_domains)
            if tightened_domains is not None:
                for number_id, (low, high) in tightened_domains.items():
                    if low != high:
                        continue
                    factual_number = factual_entities.numbers.get(number_id)
                    factual_value = (
                        generator._number_entity_int_value(factual_number) if factual_number is not None else None
                    )
                    if factual_value is None or int(low) != int(factual_value):
                        continue
                    fixed_refs.update(generator._forced_equal_refs_for_number(number_id))
    return fixed_refs


def _verify_full_fictional_manual_differences(
    document,
    *,
    factual_entities: EntityCollection,
    fictional_entities: EntityCollection,
    replace_mode: str,
) -> None:
    exempt_person_attrs = FictionalEntitySampler._PERSON_DIFF_EXEMPT_ATTRS
    explicitly_fixed_refs = _explicitly_forced_equal_refs(document, factual_entities)
    unchanged_refs: list[str] = []
    fully_replaced_types = FULL_REPLACE_ENTITY_TYPES.get(replace_mode, frozenset())
    partially_replaced_attrs = PARTIAL_REPLACE_ATTRIBUTES.get(replace_mode, {})
    for entity_type, specs in _required_refs_for_difference_check(document).items():
        for entity_id, attrs in specs:
            for attr in attrs:
                if entity_type == "person" and attr in exempt_person_attrs:
                    continue
                if entity_type in fully_replaced_types:
                    should_change = True
                else:
                    should_change = attr in partially_replaced_attrs.get(entity_type, frozenset())
                if not should_change:
                    continue
                ref = f"{entity_id}.{attr}"
                if ref in explicitly_fixed_refs:
                    continue
                factual_value = RuleEngine._get_entity_value(factual_entities, ref)
                fictional_value = RuleEngine._get_entity_value(fictional_entities, ref)
                if factual_value is None or fictional_value is None:
                    continue
                if _semantically_equal(factual_value, fictional_value):
                    unchanged_refs.append(ref)
    if unchanged_refs:
        rendered = ", ".join(sorted(set(unchanged_refs))[:12])
        raise ValueError(f"Full-fictional export kept factual values for required refs: {rendered}.")


def _verify_declared_replacements_differ(
    *,
    factual_entities: EntityCollection,
    fictional_entities: EntityCollection,
    replaced_factual_entities: dict[str, Any] | None,
    exempt_refs: set[str] | None = None,
) -> None:
    unchanged_entities: list[str] = []
    exempt_refs = set(exempt_refs or set())
    for entity_entries in (replaced_factual_entities or {}).values():
        if not isinstance(entity_entries, dict):
            continue
        for entity_id, attr_map in entity_entries.items():
            if not isinstance(attr_map, dict):
                continue
            entity_type, _ = parse_entity_id(str(entity_id))
            valid_attrs = ENTITY_TAXONOMY.get(entity_type or "", frozenset()) | LEGACY_ENTITY_ATTRIBUTES.get(
                entity_type or "",
                frozenset(),
            )
            any_changed = False
            for attr in attr_map:
                if attr not in valid_attrs:
                    continue
                ref = f"{entity_id}.{attr}"
                if ref in exempt_refs:
                    any_changed = True
                    break
                factual_value = RuleEngine._get_entity_value(factual_entities, ref)
                fictional_value = RuleEngine._get_entity_value(fictional_entities, ref)
                if factual_value is None or fictional_value is None:
                    continue
                if not _semantically_equal(factual_value, fictional_value):
                    any_changed = True
                    break
            if not any_changed:
                unchanged_entities.append(str(entity_id))
    if unchanged_entities:
        rendered = ", ".join(sorted(set(unchanged_entities))[:12])
        raise ValueError(f"Declared replaced entities kept their factual values: {rendered}.")
