"""Plan controlled entity replacements for one fictional document variant."""

from __future__ import annotations

from typing import Any

from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import is_organization_entity_type

from ..fictional_entity_sampler import FictionalEntitySampler
from ..replacement_planning import (
    PartialReplacement,
    ReplacementLayout,
    RequiredEntityMap,
    build_replacement_layout,
    build_replacement_plan,
    entity_attributes_changed,
    fixed_constant_entity_ids_from_rules,
    merge_partially_replaced_entity,
    partial_chronology_linked_entity_groups_from_text,
    partial_rule_linked_entity_groups_from_rules,
)
from .fictional_document_variant_data_contracts import (
    _ENTITY_TYPE_TO_PLURAL_KEY,
    DEBUG_SAMPLING,
    ControlledEntityReplacementContext,
)


def build_sampling_requirements(
    *,
    entity_types,
    required_entities: RequiredEntityMap,
    replacement_layout: ReplacementLayout,
) -> RequiredEntityMap:
    """Describe which fictional attributes still need to be sampled."""
    fictional_requirements: RequiredEntityMap = {}

    for entity_type, specs in required_entities.items():
        layout_entity_type = "organization" if is_organization_entity_type(entity_type) else entity_type
        replace_ids = {
            entity_id
            for entity_id, _entity in replacement_layout.factual_entities_to_replace.get(layout_entity_type, [])
        }

        all_factual_ids = set()
        for plural_type, entities in entity_types.items():
            if plural_type.rstrip("s") == layout_entity_type:
                all_factual_ids = set(entities.keys())
                break

        rule_only_specs = [(entity_id, attrs) for entity_id, attrs in specs if entity_id not in all_factual_ids]
        fully_replaced_specs = [(entity_id, attrs) for entity_id, attrs in specs if entity_id in replace_ids]

        partially_replaced_ids = {
            partial.entity_id for partial in replacement_layout.partially_replaced_entities.get(layout_entity_type, [])
        }
        partial_attr_map = {
            partial.entity_id: partial.replaced_attributes
            for partial in replacement_layout.partially_replaced_entities.get(layout_entity_type, [])
        }
        partially_replaced_specs = []
        for entity_id, attrs in specs:
            if entity_id not in partially_replaced_ids:
                continue
            filtered_attrs = [attr for attr in attrs if attr in partial_attr_map.get(entity_id, set())]
            if filtered_attrs:
                partially_replaced_specs.append((entity_id, filtered_attrs))

        combined_specs = fully_replaced_specs + partially_replaced_specs + rule_only_specs
        if combined_specs:
            fictional_requirements[entity_type] = combined_specs

    return fictional_requirements


def build_sampling_context(
    *,
    factual_entities: EntityCollection,
    replacement_layout: ReplacementLayout,
) -> EntityCollection:
    """Build the factual context available while sampling replacements."""
    sampling_context = EntityCollection()

    sampling_context.temporals = factual_entities.temporals
    sampling_context.persons = factual_entities.persons
    sampling_context.events = factual_entities.events
    sampling_context.numbers = factual_entities.numbers
    sampling_context.awards = factual_entities.awards
    sampling_context.legals = factual_entities.legals
    sampling_context.products = factual_entities.products

    for entity_type, kept_entities in replacement_layout.factual_entities_to_keep.items():
        for entity_id, factual_entity in kept_entities:
            sampling_context.add_entity(entity_type, entity_id, factual_entity)

    return sampling_context


def apply_sampled_fictional_entities(
    *,
    hybrid_entities: EntityCollection,
    sampled_fictional_entities: EntityCollection,
    partial_replacements: dict[str, list[PartialReplacement]],
) -> None:
    """Merge sampled fictional entities into the final hybrid entity set."""
    factual_numbers = {key: value.model_copy(deep=True) for key, value in hybrid_entities.numbers.items()}
    factual_temporals = {key: value.model_copy(deep=True) for key, value in hybrid_entities.temporals.items()}
    hybrid_entities.merge_from(sampled_fictional_entities)

    # A bounded generator may legitimately exhaust a small component domain
    # (seven weekdays is the common case).  Keep every successfully generated
    # component, but fill an unresolved numeric/temporal component from the
    # factual entity.  This is the final, explicitly authorised fallback; the
    # complete hybrid assignment is still checked against all document rules.
    for collection_name, factual_collection in (
        ("numbers", factual_numbers),
        ("temporals", factual_temporals),
    ):
        final_collection = getattr(hybrid_entities, collection_name)
        for entity_id, factual_entity in factual_collection.items():
            generated_entity = final_collection.get(entity_id)
            if generated_entity is None:
                continue
            merged_payload = generated_entity.model_dump()
            changed = False
            for attr, factual_value in factual_entity.model_dump().items():
                if merged_payload.get(attr) is None and factual_value is not None:
                    merged_payload[attr] = factual_value
                    changed = True
            if changed:
                final_collection[entity_id] = type(generated_entity)(**merged_payload)

    for entity_type, partial_specs in partial_replacements.items():
        plural_key = _ENTITY_TYPE_TO_PLURAL_KEY[entity_type]
        final_collection = getattr(hybrid_entities, plural_key)
        fictional_collection = getattr(sampled_fictional_entities, plural_key)
        for partial_spec in partial_specs:
            fictional_entity = fictional_collection.get(partial_spec.entity_id)
            if fictional_entity is None:
                continue
            final_collection[partial_spec.entity_id] = merge_partially_replaced_entity(
                partial_spec.factual_entity,
                fictional_entity,
                partial_spec.replaced_attributes,
            )


def build_replacement_metadata(
    *,
    hybrid_entities: EntityCollection,
    replacement_layout: ReplacementLayout,
) -> dict[str, dict[str, Any]]:
    """Record which factual entities changed in the generated variant."""
    replaced_factual_entities: dict[str, dict[str, Any]] = {}

    for entity_type, entities in replacement_layout.factual_entities_to_replace.items():
        plural_key = _ENTITY_TYPE_TO_PLURAL_KEY[entity_type]
        if not entities:
            continue
        for entity_id, factual_entity in entities:
            fictional_entity = getattr(hybrid_entities, plural_key).get(entity_id)
            if not entity_attributes_changed(factual_entity, fictional_entity):
                continue
            replaced_factual_entities.setdefault(plural_key, {})[entity_id] = factual_entity.model_dump()

    for entity_type, partial_specs in replacement_layout.partially_replaced_entities.items():
        plural_key = _ENTITY_TYPE_TO_PLURAL_KEY[entity_type]
        for partial_spec in partial_specs:
            fictional_entity = getattr(hybrid_entities, plural_key).get(partial_spec.entity_id)
            if not entity_attributes_changed(
                partial_spec.factual_entity,
                fictional_entity,
                partial_spec.replaced_attributes,
            ):
                continue
            factual_attr_dump = {
                attr: getattr(partial_spec.factual_entity, attr, None)
                for attr in partial_spec.replaced_attributes
                if getattr(partial_spec.factual_entity, attr, None) is not None
            }
            if factual_attr_dump:
                replaced_factual_entities.setdefault(plural_key, {})[partial_spec.entity_id] = factual_attr_dump

    return replaced_factual_entities


def plan_variant_replacements(
    *,
    context: ControlledEntityReplacementContext,
    replacement_proportion: float,
    replace_mode: str,
    eligible_entity_ids: dict[str, set[str]] | None = None,
) -> tuple[Any, ReplacementLayout, RequiredEntityMap]:
    """Select replaced entities and derive the fictional requirements for one variant."""
    excluded_entity_ids = fixed_constant_entity_ids_from_rules(
        rules=context.generation_document.rules,
        factual_entities=context.factual_entities_full,
    )
    linked_entity_groups = []
    if 0.0 < replacement_proportion < 1.0:
        linked_entity_groups.extend(
            partial_rule_linked_entity_groups_from_rules(rules=context.generation_document.rules)
        )
        linked_entity_groups.extend(
            partial_chronology_linked_entity_groups_from_text(
                annotated_text=context.generation_document.document_to_annotate,
            )
        )

    replacement_plan = build_replacement_plan(
        entity_types=context.entity_types,
        required_entities=context.required_entities,
        replacement_proportion=replacement_proportion,
        replace_mode=replace_mode,
        excluded_entity_ids=excluded_entity_ids,
        eligible_entity_ids=eligible_entity_ids,
        linked_entity_groups=linked_entity_groups,
    )
    if DEBUG_SAMPLING:
        print(
            f"[dbg] exact replacement target: k={replacement_plan.target_replacement_count} "
            f"over N_eligible={replacement_plan.eligible_entity_count} (p={replacement_proportion})",
            flush=True,
        )

    replacement_layout = build_replacement_layout(context.entity_types, replacement_plan)
    fictional_requirements = build_sampling_requirements(
        entity_types=context.entity_types,
        required_entities=context.required_entities,
        replacement_layout=replacement_layout,
    )
    return replacement_plan, replacement_layout, fictional_requirements


def targeted_decade_year_temporal_ids(
    *,
    context: ControlledEntityReplacementContext,
    fictional_requirements: RequiredEntityMap,
) -> frozenset[str]:
    """Restrict decade-year constraints to temporals sampled in the current variant."""
    decade_year_temporal_ids = FictionalEntitySampler.extract_decade_year_temporal_ids(
        context.generation_document,
        include_questions=True,
    )
    if "temporal" in fictional_requirements:
        targeted_temporal_ids = {temporal_id for temporal_id, _attrs in fictional_requirements["temporal"]}
        decade_year_temporal_ids &= targeted_temporal_ids
    return frozenset(decade_year_temporal_ids)


def build_variant_sampler(
    *,
    context: ControlledEntityReplacementContext,
    entity_pool: dict[str, Any],
    replacement_layout: ReplacementLayout,
    version_seed: int,
    eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None,
    reference_variant_index: int | None = None,
    reference_variant_count: int | None = None,
    used_named_values_by_id: dict[str, set[str]] | None = None,
    used_number_values_by_id: dict[str, set[int | float]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
    prior_relaxed_intervariant_reuse_audit: list[dict[str, Any]] | None = None,
    allow_relaxed_intervariant_number_reuse: bool = False,
    allow_factual_numtemp_values: bool = False,
) -> FictionalEntitySampler:
    """Build a sampler scoped to one variant layout."""
    sampling_context = build_sampling_context(
        factual_entities=context.factual_entities_full,
        replacement_layout=replacement_layout,
    )
    return FictionalEntitySampler(
        entity_pool,
        seed=version_seed,
        factual_entities=sampling_context,
        eligible_cache=eligible_cache,
        implicit_rules=context.generation_document.implicit_rules,
        reference_variant_index=reference_variant_index,
        reference_variant_count=reference_variant_count,
        used_named_values_by_id=used_named_values_by_id,
        used_number_values_by_id=used_number_values_by_id,
        used_temporal_years_by_id=used_temporal_years_by_id,
        used_temporal_values_by_id=used_temporal_values_by_id,
        prior_relaxed_intervariant_reuse_audit=prior_relaxed_intervariant_reuse_audit,
        allow_relaxed_intervariant_number_reuse=allow_relaxed_intervariant_number_reuse,
        allow_factual_numtemp_values=allow_factual_numtemp_values,
        source_document_text=context.generation_document.document_to_annotate,
    )


__all__ = [
    "apply_sampled_fictional_entities",
    "build_replacement_metadata",
    "build_sampling_context",
    "build_sampling_requirements",
    "build_variant_sampler",
    "plan_variant_replacements",
    "targeted_decade_year_temporal_ids",
]
