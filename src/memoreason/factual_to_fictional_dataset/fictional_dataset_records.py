"""Fictional dataset export built from templates plus Claude-generated entity pools."""

from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Any

import yaml

from memoreason.benchmark_definition.annotation_runtime import (
    load_annotated_document,
    load_entity_pool,
)
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_variant_generation.fictional_document_variant_data_contracts import (
    NamedEntitySample,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_variant_generation.fictional_document_variant_generation_stages import (
    build_controlled_entity_replacement_context,
    generate_named_entities,
    render_and_write_variant,
    sample_named_entities,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_variant_generation.fictional_document_variant_planning import (
    apply_sampled_fictional_entities,
    plan_variant_replacements,
)

from .dataset_paths import (
    document_variant_path,
    format_document_variant_id,
    resolve_template_identity,
)
from .dataset_record_contracts import _drop_nulls, _question_entries_from_generated
from .dataset_record_core import _relative_to_project
from .dataset_settings import FactualToFictionalDatasetSetting
from .fictional_dataset_generation import (
    _should_retry_pool_verification_error,
    generate_fictional_dataset_payload,
)
from .fictional_dataset_record_validation import _verify_generated_payload
from .fictional_dataset_uniqueness import (
    _collect_used_named_values_from_existing_variants,
    _collect_used_number_values_from_existing_variants,
    _collect_used_temporal_values_from_existing_variants,
    _collect_used_temporal_years_from_existing_variants,
    _merge_used_named_values,
    _merge_used_number_values,
    _merge_used_temporal_values,
    _merge_used_temporal_years,
)

logger = logging.getLogger(__name__)
_MAX_VERIFIED_POOL_ATTEMPTS = 2
_STRICT_DATASET_EXPORT = str(os.environ.get("MEMOREASON_STRICT_DATASET_EXPORT", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
}


def build_fictional_dataset_record(
    template_path: Path,
    *,
    setting_spec: FactualToFictionalDatasetSetting,
    pool_path: Path | None,
    seed: int,
    output_path: Path,
    variant_index: int,
    variant_count: int = 1,
    used_named_values_by_id: dict[str, set[str]] | None = None,
    used_number_values_by_id: dict[str, set[int | float]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
    prior_relaxed_intervariant_reuse_audit: list[dict[str, Any]] | None = None,
    include_existing_variant_values: bool = True,
    force_allow_previous_numtemp_reuse: bool = False,
) -> dict[str, Any]:
    """Generate one fictional dataset version and convert it to the exported schema."""
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    theme, document_id = resolve_template_identity(template_path)
    source_mode = "pool"

    if pool_path is None:
        raise FileNotFoundError(
            f"No entity pool found for {theme}/{document_id}. Expected "
            f"data/GENERATED_FICTIONAL_ENTITIES/{theme}/{document_id}_entity_pool.yaml."
        )

    pool_data = load_entity_pool(str(pool_path))
    if include_existing_variant_values:
        used_named_values_by_id = _merge_used_named_values(
            used_named_values_by_id,
            _collect_used_named_values_from_existing_variants(
                output_path=output_path,
                document_id=document_id,
                variant_index=variant_index,
                variant_count=variant_count,
            ),
        )
        used_number_values_by_id = _merge_used_number_values(
            used_number_values_by_id,
            _collect_used_number_values_from_existing_variants(
                output_path=output_path,
                document_id=document_id,
                variant_index=variant_index,
                variant_count=variant_count,
            ),
        )
        used_temporal_years_by_id = _merge_used_temporal_years(
            used_temporal_years_by_id,
            _collect_used_temporal_years_from_existing_variants(
                output_path=output_path,
                document_id=document_id,
                variant_index=variant_index,
                variant_count=variant_count,
            ),
        )
        used_temporal_values_by_id = _merge_used_temporal_values(
            used_temporal_values_by_id,
            _collect_used_temporal_values_from_existing_variants(
                output_path=output_path,
                document_id=document_id,
                variant_index=variant_index,
                variant_count=variant_count,
            ),
        )
    else:
        used_named_values_by_id = _merge_used_named_values(used_named_values_by_id, {})
        used_number_values_by_id = _merge_used_number_values(used_number_values_by_id, {})
        used_temporal_years_by_id = _merge_used_temporal_years(used_temporal_years_by_id, {})
        used_temporal_values_by_id = _merge_used_temporal_values(used_temporal_values_by_id, {})
    last_pool_error: Exception | None = None
    for verification_attempt in range(_MAX_VERIFIED_POOL_ATTEMPTS):
        attempt_seed = seed + (verification_attempt * 100_000)
        staging_output_path = output_path.with_name(f".{output_path.stem}.stage.yaml")
        try:
            generated_payload, successful_seed, generation_metadata = generate_fictional_dataset_payload(
                document,
                setting_spec=setting_spec,
                entity_pool=pool_data,
                seed=attempt_seed,
                output_path=staging_output_path,
                variant_index=variant_index,
                variant_count=variant_count,
                used_named_values_by_id=used_named_values_by_id,
                used_number_values_by_id=used_number_values_by_id,
                used_temporal_years_by_id=used_temporal_years_by_id,
                used_temporal_values_by_id=used_temporal_values_by_id,
                prior_relaxed_intervariant_reuse_audit=prior_relaxed_intervariant_reuse_audit,
                force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                return_metadata=True,
            )
        finally:
            if staging_output_path.exists():
                staging_output_path.unlink()
        try:
            _verify_generated_payload(
                original_document=document,
                render_source_document=document,
                generated_payload=generated_payload,
                setting_spec=setting_spec,
                source_mode="pool",
            )
            break
        except Exception as exc:
            previous_error = last_pool_error
            last_pool_error = exc
            logger.warning(
                "Rejecting pool-based fictional generation for %s/%s on seed %s after verification failure: %s",
                theme,
                document_id,
                attempt_seed,
                exc,
            )
            if not _should_retry_pool_verification_error(exc, previous_error=previous_error):
                raise
            continue
    else:
        raise RuntimeError(
            f"Failed to produce any pool-based fictional generation for {document.document_id}: {last_pool_error}"
        )

    payload = {
        "document_id": document_id,
        "document_theme": theme,
        "document_setting": setting_spec.setting_id,
        "document_setting_family": setting_spec.setting_family,
        "document_variant_id": format_document_variant_id(variant_index),
        "document_variant_index": variant_index,
        "replacement_proportion": setting_spec.replacement_proportion,
        "generation_seed": successful_seed,
        "generation_source": source_mode,
        "source_template_path": _relative_to_project(template_path),
        "document_text": generated_payload.get("generated_document", ""),
        "num_entities_replaced": int(generated_payload.get("num_entities_replaced", 0)),
        "replaced_factual_entities": _drop_nulls(generated_payload.get("replaced_factual_entities", {}) or {}),
        "questions": _question_entries_from_generated(document, generated_payload),
        "entities_used": generated_payload.get("entities_used", {}) or {},
    }
    if pool_path is not None:
        payload["source_entity_pool_path"] = _relative_to_project(pool_path)
    if generation_metadata.get("used_relaxed_intervariant_reuse"):
        payload["_used_relaxed_intervariant_reuse"] = True
    relaxed_reuse_audit = generation_metadata.get("relaxed_intervariant_reuse_audit") or []
    if relaxed_reuse_audit:
        payload["_relaxed_intervariant_reuse_audit"] = [dict(item) for item in relaxed_reuse_audit]
    return payload


def _subset_full_fictional_entities_for_layout(
    *,
    full_entities: EntityCollection,
    factual_entities: EntityCollection,
    replacement_layout,
) -> EntityCollection:
    subset = EntityCollection()
    for entity_type, entities in replacement_layout.factual_entities_to_replace.items():
        full_collection = full_entities.get_collection(entity_type)
        for entity_id, _factual_entity in entities:
            entity = full_collection.get(entity_id)
            if entity is not None:
                subset.add_entity(entity_type, entity_id, entity)
    for entity_type, partial_specs in replacement_layout.partially_replaced_entities.items():
        full_collection = full_entities.get_collection(entity_type)
        for partial_spec in partial_specs:
            entity = full_collection.get(partial_spec.entity_id)
            if entity is not None:
                partial_payload = {
                    attr: getattr(entity, attr, None)
                    for attr in partial_spec.replaced_attributes
                    if getattr(entity, attr, None) is not None
                }
                if partial_payload:
                    subset.add_entity(
                        entity_type,
                        partial_spec.entity_id,
                        entity.__class__.model_validate(partial_payload),
                    )

    for entity_type in ("number", "temporal"):
        factual_collection = factual_entities.get_collection(entity_type)
        full_collection = full_entities.get_collection(entity_type)
        for entity_id, factual_entity in factual_collection.items():
            if entity_id in full_collection and _entity_is_unmaterialized(factual_entity):
                subset.add_entity(entity_type, entity_id, full_collection[entity_id])
    return subset


def _entity_is_unmaterialized(entity: Any) -> bool:
    for value in entity.model_dump().values():
        if value is not None:
            return False
    return True


def build_derived_fictional_dataset_record(
    template_path: Path,
    *,
    setting_spec: FactualToFictionalDatasetSetting,
    seed: int,
    output_path: Path,
    variant_index: int,
    variant_count: int = 1,
) -> dict[str, Any]:
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    theme, document_id = resolve_template_identity(template_path)
    source_fictional_path = document_variant_path(
        theme,
        document_id,
        "fictional",
        variant_index=variant_index,
        variant_count=variant_count,
    )
    if not source_fictional_path.exists():
        raise FileNotFoundError(
            f"Cannot derive {setting_spec.setting_id} for {theme}/{document_id} without full-fictional source "
            f"{source_fictional_path}."
        )

    full_source_bytes = source_fictional_path.read_bytes()
    full_source_sha256 = hashlib.sha256(full_source_bytes).hexdigest()
    full_payload = yaml.safe_load(full_source_bytes) or {}
    full_replacements = full_payload.get("replaced_factual_entities") or {}
    eligible_entity_ids = {
        collection.rstrip("s"): set(entities)
        for collection, entities in full_replacements.items()
        if isinstance(entities, dict)
    }

    context = build_controlled_entity_replacement_context(document)
    # Partial variants must be independently reproducible.  Full generation
    # seeds this global selector before planning; derived projections do the
    # same explicitly instead of inheriting process history.
    random.seed(seed)
    _replacement_plan, replacement_layout, _fictional_requirements = plan_variant_replacements(
        context=context,
        replacement_proportion=setting_spec.replacement_proportion,
        replace_mode=setting_spec.replace_mode,
        eligible_entity_ids=eligible_entity_ids,
    )

    full_entities = EntityCollection.model_validate(full_payload.get("entities_used") or {})
    hybrid_entities = replacement_layout.initial_hybrid_entities.model_copy(deep=True)
    apply_sampled_fictional_entities(
        hybrid_entities=hybrid_entities,
        sampled_fictional_entities=_subset_full_fictional_entities_for_layout(
            full_entities=full_entities,
            factual_entities=context.factual_entities_full,
            replacement_layout=replacement_layout,
        ),
        partial_replacements=replacement_layout.partially_replaced_entities,
    )

    staging_output_path = output_path.with_name(f".{output_path.stem}.stage.yaml")
    staging_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_and_write_variant(
            context=context,
            output_path=staging_output_path,
            document_id=document_id,
            replacement_proportion=setting_spec.replacement_proportion,
            replace_mode=setting_spec.replace_mode,
            hybrid_entities=hybrid_entities,
            replacement_layout=replacement_layout,
        )
        generated_payload = yaml.safe_load(staging_output_path.read_text(encoding="utf-8"))
    finally:
        if staging_output_path.exists():
            staging_output_path.unlink()

    if not isinstance(generated_payload, dict):
        raise ValueError(f"Invalid derived payload for {document.document_id}.")

    try:
        _verify_generated_payload(
            original_document=document,
            render_source_document=document,
            generated_payload=generated_payload,
            setting_spec=setting_spec,
            source_mode="derived_from_full_fictional",
        )
    except Exception as exc:
        if _STRICT_DATASET_EXPORT:
            raise
        logger.warning(
            "Accepting derived %s export for %s/%s despite verification failure: %s",
            setting_spec.setting_id,
            theme,
            document_id,
            exc,
        )

    payload = {
        "document_id": document_id,
        "document_theme": theme,
        "document_setting": setting_spec.setting_id,
        "document_setting_family": setting_spec.setting_family,
        "document_variant_id": format_document_variant_id(variant_index),
        "document_variant_index": variant_index,
        "replacement_proportion": setting_spec.replacement_proportion,
        "generation_seed": int(full_payload.get("generation_seed") or seed),
        "partial_replacement_selection_seed": int(seed),
        "generation_source": "derived_from_full_fictional",
        "source_template_path": _relative_to_project(template_path),
        "source_fictional_document_path": _relative_to_project(source_fictional_path),
        "source_fictional_document_sha256": full_source_sha256,
        "document_text": generated_payload.get("generated_document", ""),
        "num_entities_replaced": int(generated_payload.get("num_entities_replaced", 0)),
        "replaced_factual_entities": _drop_nulls(generated_payload.get("replaced_factual_entities", {}) or {}),
        "questions": _question_entries_from_generated(document, generated_payload),
        "entities_used": generated_payload.get("entities_used", {}) or {},
    }
    source_entity_pool_path = full_payload.get("source_entity_pool_path")
    if source_entity_pool_path:
        payload["source_entity_pool_path"] = source_entity_pool_path
    return payload


def build_named_only_fictional_dataset_record(
    template_path: Path,
    *,
    setting_spec: FactualToFictionalDatasetSetting,
    pool_path: Path | None,
    seed: int,
    output_path: Path,
    variant_index: int,
    variant_count: int = 1,
    used_named_values_by_id: dict[str, set[str]] | None = None,
    include_existing_variant_values: bool = True,
) -> dict[str, Any]:
    """Generate one named-only fictional variant without numeric/temporal synthesis."""
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    theme, document_id = resolve_template_identity(template_path)
    if pool_path is None:
        raise FileNotFoundError(
            f"No entity pool found for {theme}/{document_id}. Expected "
            f"data/GENERATED_FICTIONAL_ENTITIES/{theme}/{document_id}_entity_pool.yaml."
        )

    context = build_controlled_entity_replacement_context(document)
    pool_data = load_entity_pool(str(pool_path))
    if include_existing_variant_values:
        used_named_values_by_id = _merge_used_named_values(
            used_named_values_by_id,
            _collect_used_named_values_from_existing_variants(
                output_path=output_path,
                document_id=document_id,
                variant_index=variant_index,
                variant_count=variant_count,
            ),
        )
    else:
        used_named_values_by_id = _merge_used_named_values(used_named_values_by_id, {})
    named_entities = generate_named_entities(context=context, entity_pool=pool_data, seed=seed)

    successful_seed = seed
    named_entity_sample: NamedEntitySample | None = sample_named_entities(
        context=context,
        named_entities=named_entities,
        replacement_proportion=setting_spec.replacement_proportion,
        version_seed=seed,
        replace_mode=setting_spec.replace_mode,
        reference_variant_index=variant_index - 1,
        reference_variant_count=variant_count,
        used_named_values_by_id=used_named_values_by_id,
    )
    if named_entity_sample is None:
        raise RuntimeError(f"Named-only fictional sampling returned no assignment for {document_id} on seed {seed}.")

    staging_output_path = output_path.with_name(f".{output_path.stem}.stage.yaml")
    staging_output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_and_write_variant(
            context=context,
            output_path=staging_output_path,
            document_id=document_id,
            replacement_proportion=setting_spec.replacement_proportion,
            replace_mode=setting_spec.replace_mode,
            hybrid_entities=named_entity_sample.entities.model_copy(deep=True),
            replacement_layout=named_entity_sample.replacement_layout,
        )
        generated_payload = yaml.safe_load(staging_output_path.read_text(encoding="utf-8"))
    finally:
        if staging_output_path.exists():
            staging_output_path.unlink()

    if not isinstance(generated_payload, dict):
        raise ValueError(f"Invalid named-only generated payload for {document_id}.")

    try:
        _verify_generated_payload(
            original_document=document,
            render_source_document=document,
            generated_payload=generated_payload,
            setting_spec=setting_spec,
            source_mode="pool",
        )
    except Exception as exc:
        if _STRICT_DATASET_EXPORT:
            raise
        logger.warning(
            "Accepting named-only fictional export for %s/%s despite verification failure: %s",
            theme,
            document_id,
            exc,
        )

    payload = {
        "document_id": document_id,
        "document_theme": theme,
        "document_setting": setting_spec.setting_id,
        "document_setting_family": setting_spec.setting_family,
        "document_variant_id": format_document_variant_id(variant_index),
        "document_variant_index": variant_index,
        "replacement_proportion": setting_spec.replacement_proportion,
        "generation_seed": successful_seed,
        "generation_source": "pool",
        "source_template_path": _relative_to_project(template_path),
        "source_entity_pool_path": _relative_to_project(pool_path),
        "document_text": generated_payload.get("generated_document", ""),
        "num_entities_replaced": int(generated_payload.get("num_entities_replaced", 0)),
        "replaced_factual_entities": _drop_nulls(generated_payload.get("replaced_factual_entities", {}) or {}),
        "questions": _question_entries_from_generated(document, generated_payload),
        "entities_used": generated_payload.get("entities_used", {}) or {},
    }
    return payload
