"""Fictional dataset export built from templates plus Claude-generated entity pools."""

from __future__ import annotations

import json
import logging
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from src.core.annotation_runtime import (
    AnnotationParser,
    partition_generation_rules,
    load_annotated_document,
    load_entity_pool,
    REPLACE_MODE_ALL,
)
from src.core.entity_taxonomy import REPLACE_MODE_NON_NUMERICAL, REPLACE_MODE_NUMERICAL_TEMPORAL
from src.core.document_schema import EntityCollection, NumberEntity
from src.document_generation.fictional_entity_sampler import FictionalEntitySampler
from src.document_generation.fictional_entity_sampler_common import detect_ordering_excluded_number_ids
from src.document_generation.fictional_generation.planning import (
    apply_sampled_fictional_entities,
    plan_variant_replacements,
)
from src.document_generation.fictional_generation.stages import (
    build_fictional_generation_context,
    generate_named_entities,
    render_and_write_variant,
    sample_named_entities,
)
from src.document_generation.fictional_generation.types import NamedEntitySample
from src.document_generation.fictional_generation_algorithm import (
    FictionalGenerationTemplateInput,
    FictionalVariantRequest,
    fictional_generation,
)
from src.document_generation.number_generation.value_strategy import NumberValueStrategyMixin
from src.document_generation.number_temporal_generator import NumberTemporalGenerator

from .dataset_paths import (
    document_variant_path,
    existing_entity_pool_path,
    format_document_variant_id,
    resolve_template_identity,
)
from .dataset_record_builder import (
    _drop_nulls,
    _question_entries_from_generated,
    _relative_to_project,
    _semantic_payload_issues,
    _verify_generated_payload,
    _write_yaml,
    answer_behavior_label,
    normalize_question_type,
)
from .dataset_settings import DatasetSettingSpec

logger = logging.getLogger(__name__)
_MAX_VERIFIED_POOL_ATTEMPTS = 2
_MAX_BATCH_GENERATION_ATTEMPTS = 2
_NON_RETRYABLE_VERIFICATION_PREFIXES = (
    "Entity pool cannot satisfy the required manual attributes for this document:",
    "Generated payload failed semantic linting:\nreplaced factual literals still appear in output:",
    "Generated payload failed semantic linting:\nplace/self-reference collision:",
)
_INTERVARIANT_NAMED_BUCKETS = (
    "persons",
    "places",
    "events",
    "organizations",
    "awards",
    "legals",
    "products",
)
_INTERVARIANT_NUMTEMP_BUCKETS = ("numbers", "temporals")
_TEMPORAL_FIELDS = ("timestamp", "date", "year", "month", "day_of_month", "day", "decade", "century")
_TEMPORAL_AVOID_FIELDS = ("year", "month", "day", "day_of_month")

__all__ = [
    "_semantic_payload_issues",
    "answer_behavior_label",
    "build_fictional_dataset_record",
    "export_fictional_dataset_documents_batch",
    "export_fictional_dataset_document",
    "generate_fictional_dataset_payload",
    "normalize_question_type",
]


def _should_retry_pool_verification_error(
    exc: Exception,
    *,
    previous_error: Exception | None = None,
) -> bool:
    message = str(exc or "").strip()
    if not message:
        return True
    if message.startswith(_NON_RETRYABLE_VERIFICATION_PREFIXES):
        return False
    if previous_error is not None and str(previous_error or "").strip() == message:
        return False
    return True


def generate_fictional_dataset_payload(
    document,
    *,
    setting_spec: DatasetSettingSpec,
    entity_pool: dict[str, Any],
    seed: int,
    output_path: Path,
    variant_index: int = 1,
    variant_count: int = 1,
    used_number_values_by_id: dict[str, set[int]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
    force_allow_previous_numtemp_reuse: bool = False,
    return_metadata: bool = False,
) -> tuple[dict[str, Any], int] | tuple[dict[str, Any], int, dict[str, Any]]:
    """Generate one fictional dataset document with the active replacement workflow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        algorithm_result = fictional_generation(
            [
                FictionalGenerationTemplateInput(
                    template_document=document,
                    named_entity_pool=entity_pool,
                    variant_requests=(
                        FictionalVariantRequest(
                            base_seed=seed,
                            output_path=output_path,
                            reference_variant_index=variant_index - 1,
                            reference_variant_count=variant_count,
                        ),
                    ),
                    replacement_proportion=setting_spec.replacement_proportion,
                    document_id=document.document_id,
                    replace_mode=setting_spec.replace_mode,
                    named_entities_seed=seed,
                    used_number_values_by_id=used_number_values_by_id,
                    used_temporal_years_by_id=used_temporal_years_by_id,
                    used_temporal_values_by_id=used_temporal_values_by_id,
                    force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                )
            ]
        )[0]
        _generated_path, successful_seed = algorithm_result.generated_variants[0]

    generated_payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    if not isinstance(generated_payload, dict):
        raise ValueError(f"Invalid generated payload for {document.document_id}.")
    if return_metadata:
        return generated_payload, successful_seed, {
            "used_relaxed_intervariant_reuse": bool(
                getattr(algorithm_result, "used_relaxed_intervariant_reuse", False)
            )
        }
    return generated_payload, successful_seed


def _collect_used_number_values_from_existing_variants(
    *,
    output_path: Path,
    document_id: str,
    variant_index: int,
    variant_count: int,
) -> dict[str, set[int | float]]:
    if int(variant_count) <= 1:
        return {}
    used_values_by_id: dict[str, set[int | float]] = {}
    for sibling_index in range(1, int(variant_index)):
        sibling_name = (
            f"{document_id}_{format_document_variant_id(sibling_index)}.yaml"
            if int(variant_count) > 1
            else f"{document_id}.yaml"
        )
        sibling_path = output_path.parent / sibling_name
        if not sibling_path.exists():
            continue
        sibling_payload = yaml.safe_load(sibling_path.read_text(encoding="utf-8"))
        if not isinstance(sibling_payload, dict):
            continue
        number_payloads = ((sibling_payload.get("entities_used") or {}).get("numbers") or {})
        if not isinstance(number_payloads, dict):
            continue
        for number_id, raw_number_payload in number_payloads.items():
            try:
                number_entity = NumberEntity.model_validate(raw_number_payload or {})
            except Exception:
                continue
            numeric_value = None
            for field in ("int", "percent", "proportion", "float"):
                candidate = getattr(number_entity, field, None)
                if candidate is not None:
                    numeric_value = candidate
                    break
            if numeric_value is None and getattr(number_entity, "fraction", None):
                numeric_value = NumberValueStrategyMixin._number_entity_int_value(number_entity)
            if numeric_value is None:
                continue
            used_values_by_id.setdefault(str(number_id), set()).add(numeric_value)
    return used_values_by_id


def _collect_used_temporal_years_from_existing_variants(
    *,
    output_path: Path,
    document_id: str,
    variant_index: int,
    variant_count: int,
) -> dict[str, set[int]]:
    if int(variant_count) <= 1:
        return {}
    used_years_by_id: dict[str, set[int]] = {}
    for sibling_index in range(1, int(variant_index)):
        sibling_name = (
            f"{document_id}_{format_document_variant_id(sibling_index)}.yaml"
            if int(variant_count) > 1
            else f"{document_id}.yaml"
        )
        sibling_path = output_path.parent / sibling_name
        if not sibling_path.exists():
            continue
        sibling_payload = yaml.safe_load(sibling_path.read_text(encoding="utf-8"))
        if not isinstance(sibling_payload, dict):
            continue
        temporal_payloads = ((sibling_payload.get("entities_used") or {}).get("temporals") or {})
        if not isinstance(temporal_payloads, dict):
            continue
        for temporal_id, raw_temporal_payload in temporal_payloads.items():
            try:
                temporal_year = int((raw_temporal_payload or {}).get("year"))
            except (TypeError, ValueError, AttributeError):
                continue
            used_years_by_id.setdefault(str(temporal_id), set()).add(temporal_year)
    return used_years_by_id


def _record_temporal_payload_values(
    used_values_by_id: dict[str, dict[str, set[Any]]],
    temporal_id: str,
    raw_temporal_payload: dict[str, Any],
) -> None:
    if not isinstance(raw_temporal_payload, dict):
        return
    for attr in _TEMPORAL_AVOID_FIELDS:
        value = raw_temporal_payload.get(attr)
        if value in (None, ""):
            continue
        used_values_by_id.setdefault(str(temporal_id), {}).setdefault(attr, set()).add(value)


def _collect_used_temporal_values_from_existing_variants(
    *,
    output_path: Path,
    document_id: str,
    variant_index: int,
    variant_count: int,
) -> dict[str, dict[str, set[Any]]]:
    if int(variant_count) <= 1:
        return {}
    used_values_by_id: dict[str, dict[str, set[Any]]] = {}
    for sibling_index in range(1, int(variant_index)):
        sibling_name = (
            f"{document_id}_{format_document_variant_id(sibling_index)}.yaml"
            if int(variant_count) > 1
            else f"{document_id}.yaml"
        )
        sibling_path = output_path.parent / sibling_name
        if not sibling_path.exists():
            continue
        sibling_payload = yaml.safe_load(sibling_path.read_text(encoding="utf-8"))
        if not isinstance(sibling_payload, dict):
            continue
        temporal_payloads = ((sibling_payload.get("entities_used") or {}).get("temporals") or {})
        if not isinstance(temporal_payloads, dict):
            continue
        for temporal_id, raw_temporal_payload in temporal_payloads.items():
            _record_temporal_payload_values(used_values_by_id, str(temporal_id), raw_temporal_payload or {})
    return used_values_by_id


def _merge_used_number_values(
    existing: dict[str, set[int | float]] | None,
    additional: dict[str, set[int | float]],
) -> dict[str, set[int | float]]:
    merged = {str(entity_id): set(values) for entity_id, values in (existing or {}).items()}
    for entity_id, values in additional.items():
        merged.setdefault(str(entity_id), set()).update(values)
    return merged


def _merge_used_temporal_years(
    existing: dict[str, set[int]] | None,
    additional: dict[str, set[int]],
) -> dict[str, set[int]]:
    merged = {str(entity_id): set(values) for entity_id, values in (existing or {}).items()}
    for entity_id, values in additional.items():
        merged.setdefault(str(entity_id), set()).update(values)
    return merged


def _merge_used_temporal_values(
    existing: dict[str, dict[str, set[Any]]] | None,
    additional: dict[str, dict[str, set[Any]]],
) -> dict[str, dict[str, set[Any]]]:
    merged: dict[str, dict[str, set[Any]]] = {
        str(entity_id): {str(attr): set(values) for attr, values in values_by_attr.items()}
        for entity_id, values_by_attr in (existing or {}).items()
    }
    for entity_id, values_by_attr in additional.items():
        target = merged.setdefault(str(entity_id), {})
        for attr, values in values_by_attr.items():
            target.setdefault(str(attr), set()).update(values)
    return merged


def _replacement_buckets_for_setting(setting_spec: DatasetSettingSpec) -> tuple[str, ...]:
    if setting_spec.replace_mode == REPLACE_MODE_NON_NUMERICAL:
        return _INTERVARIANT_NAMED_BUCKETS
    if setting_spec.replace_mode == REPLACE_MODE_NUMERICAL_TEMPORAL:
        return _INTERVARIANT_NUMTEMP_BUCKETS
    return _INTERVARIANT_NAMED_BUCKETS + _INTERVARIANT_NUMTEMP_BUCKETS


def _normalize_number_uniqueness_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    for field in ("percent", "proportion", "float"):
        value = payload.get(field)
        if value is not None:
            return {"kind": field, "value": round(float(value), 6)}
    if payload.get("fraction") not in (None, ""):
        return {"kind": "fraction", "value": str(payload["fraction"]).strip()}
    if payload.get("int") is not None:
        return {"kind": "int", "value": int(payload["int"])}
    if payload.get("str") not in (None, ""):
        return {"kind": "str", "value": str(payload["str"]).strip()}
    return None


def _normalize_temporal_uniqueness_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    value = {field: payload[field] for field in _TEMPORAL_FIELDS if payload.get(field) not in (None, "")}
    return value or None


def _normalize_named_uniqueness_value(bucket: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if bucket == "persons":
        for keys in (("full_name",), ("first_name", "last_name"), ("first_name",), ("last_name",), ("name",)):
            value = {key: str(payload[key]).strip() for key in keys if payload.get(key) not in (None, "")}
            if len(value) == len(keys) and value:
                return value
        return None
    if bucket == "places":
        for keys in (
            ("natural_site",),
            ("street", "city", "state", "country", "region"),
            ("city", "state"),
            ("city", "country"),
            ("city", "region"),
            ("city",),
            ("country",),
            ("state",),
            ("region",),
        ):
            value = {key: str(payload[key]).strip() for key in keys if payload.get(key) not in (None, "")}
            if value:
                return value
        return None
    if bucket == "events":
        value = {}
        if payload.get("name") not in (None, ""):
            value["name"] = str(payload["name"]).strip()
        if payload.get("type") not in (None, ""):
            value["type"] = str(payload["type"]).strip()
        return value or None
    if bucket == "legals":
        value = {}
        if payload.get("name") not in (None, ""):
            value["name"] = str(payload["name"]).strip()
        if payload.get("reference_code") not in (None, ""):
            value["reference_code"] = str(payload["reference_code"]).strip()
        return value or None
    if bucket in {"organizations", "awards", "products"}:
        if payload.get("name") in (None, ""):
            return None
        return {"name": str(payload["name"]).strip()}
    return None


def _normalize_uniqueness_value(bucket: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if bucket == "numbers":
        return _normalize_number_uniqueness_value(payload)
    if bucket == "temporals":
        return _normalize_temporal_uniqueness_value(payload)
    return _normalize_named_uniqueness_value(bucket, payload)


def _collect_intervariant_duplicates(
    payloads: list[dict[str, Any]],
    *,
    setting_spec: DatasetSettingSpec,
    singleton_number_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    relevant_buckets = set(_replacement_buckets_for_setting(setting_spec))
    singleton_number_ids = {str(number_id) for number_id in (singleton_number_ids or set())}
    values_by_ref: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    normalized_values: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}

    for payload in payloads:
        variant_id = str(payload.get("document_variant_id") or "")
        entities_used = payload.get("entities_used") or {}
        replaced_entities = payload.get("replaced_factual_entities") or {}
        for bucket in relevant_buckets:
            replaced_bucket = replaced_entities.get(bucket) or {}
            if not isinstance(replaced_bucket, dict) or not replaced_bucket:
                continue
            used_bucket = entities_used.get(bucket) or {}
            if not isinstance(used_bucket, dict):
                continue
            for entity_ref in replaced_bucket.keys():
                if bucket == "numbers" and str(entity_ref) in singleton_number_ids:
                    continue
                entity_payload = used_bucket.get(entity_ref)
                if not isinstance(entity_payload, dict):
                    continue
                normalized = _normalize_uniqueness_value(bucket, entity_payload)
                if normalized is None:
                    continue
                signature = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
                values_by_ref[(bucket, str(entity_ref))][signature].append(variant_id)
                normalized_values.setdefault((bucket, str(entity_ref)), {})[signature] = normalized

    duplicates: list[dict[str, Any]] = []
    for (bucket, entity_ref), variants_by_signature in sorted(values_by_ref.items()):
        repeated_values = []
        for signature, variants in sorted(variants_by_signature.items()):
            if len(variants) < 2:
                continue
            repeated_values.append(
                {
                    "value": normalized_values[(bucket, entity_ref)][signature],
                    "variants": sorted(variants),
                }
            )
        if repeated_values:
            duplicates.append(
                {
                    "entity_bucket": bucket,
                    "entity_ref": entity_ref,
                    "repeated_values": repeated_values,
                }
            )
    return duplicates


def _singleton_domain_number_ids_for_template(
    template_path: Path,
    *,
    variant_count: int = 1,
) -> set[str]:
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_numbers = FictionalEntitySampler.extract_required_entities(document, include_questions=True).get("number", [])
    if not required_numbers:
        return set()
    effective_rules, _dropped_rules = partition_generation_rules(document, include_questions=True)
    numeric_rules = [rule for rule in effective_rules if "number_" in str(rule) and "temporal_" not in str(rule)]
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
        ordering_excluded_number_ids=detect_ordering_excluded_number_ids(document.document_to_annotate),
    )
    number_ids = {number_id for number_id, _attrs in required_numbers}
    target_variant_count = max(int(variant_count), 1)
    exempt_number_ids: set[str] = set()
    for number_id in sorted(number_ids):
        low, high = generator._number_base_range(number_id)
        if int(high) < int(low):
            continue
        factual_value = generator._factual_number_int(number_id)
        available_count = int(high) - int(low) + 1
        if factual_value is not None and int(low) <= int(factual_value) <= int(high):
            available_count -= 1
        if available_count < target_variant_count:
            exempt_number_ids.add(number_id)
    active_rules = generator._number_evaluable_rules(
        numeric_rules,
        number_ids,
        factual_entities.model_copy(deep=True),
    )
    active_rules = [*active_rules, *generator._ordering_number_rules(sorted(number_ids), factual_entities.model_copy(deep=True))]
    constraints = generator._collect_linear_constraints(
        active_rules,
        number_ids,
        factual_entities.model_copy(deep=True),
    )
    if constraints is None:
        return set()
    domains = {number_id: generator._number_base_range(number_id) for number_id in sorted(number_ids)}
    tightened = generator._tighten_number_domains(constraints, domains)
    if tightened is None:
        return exempt_number_ids
    for number_id, (low, high) in tightened.items():
        domain_size = int(high) - int(low) + 1
        if domain_size < target_variant_count:
            exempt_number_ids.add(number_id)
    return exempt_number_ids


def build_fictional_dataset_record(
    template_path: Path,
    *,
    setting_spec: DatasetSettingSpec,
    pool_path: Path | None,
    seed: int,
    output_path: Path,
    variant_index: int,
    variant_count: int = 1,
    used_number_values_by_id: dict[str, set[int | float]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
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
        used_number_values_by_id = _merge_used_number_values(used_number_values_by_id, {})
        used_temporal_years_by_id = _merge_used_temporal_years(used_temporal_years_by_id, {})
        used_temporal_values_by_id = _merge_used_temporal_values(used_temporal_values_by_id, {})
    last_pool_error: Exception | None = None
    for verification_attempt in range(_MAX_VERIFIED_POOL_ATTEMPTS):
        attempt_seed = seed + (verification_attempt * 100_000)
        staging_output_path = output_path.with_name(f".{output_path.stem}.stage.yaml")
        try:
            generation_result = generate_fictional_dataset_payload(
                document,
                setting_spec=setting_spec,
                entity_pool=pool_data,
                seed=attempt_seed,
                output_path=staging_output_path,
                variant_index=variant_index,
                variant_count=variant_count,
                used_number_values_by_id=used_number_values_by_id,
                used_temporal_years_by_id=used_temporal_years_by_id,
                used_temporal_values_by_id=used_temporal_values_by_id,
                force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                return_metadata=True,
            )
            if len(generation_result) == 2:
                generated_payload, successful_seed = generation_result
                generation_metadata = {}
            else:
                generated_payload, successful_seed, generation_metadata = generation_result
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


def _is_partial_all_entity_setting(setting_spec: DatasetSettingSpec) -> bool:
    return (
        setting_spec.replace_mode == REPLACE_MODE_ALL
        and 0.0 < setting_spec.replacement_proportion < 1.0
        and setting_spec.setting_family == "fictional"
    )


def build_derived_fictional_dataset_record(
    template_path: Path,
    *,
    setting_spec: DatasetSettingSpec,
    seed: int,
    output_path: Path,
    variant_index: int,
    variant_count: int = 1,
    strict_verification: bool = False,
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

    context = build_fictional_generation_context(document)
    _replacement_plan, replacement_layout, _fictional_requirements = plan_variant_replacements(
        context=context,
        replacement_proportion=setting_spec.replacement_proportion,
        replace_mode=setting_spec.replace_mode,
        rng=random.Random(seed),
    )

    full_payload = yaml.safe_load(source_fictional_path.read_text(encoding="utf-8")) or {}
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
        if strict_verification:
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
    setting_spec: DatasetSettingSpec,
    pool_path: Path | None,
    seed: int,
    output_path: Path,
    variant_index: int,
    variant_count: int = 1,
) -> dict[str, Any]:
    """Generate one named-only fictional variant without numeric/temporal synthesis."""
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    theme, document_id = resolve_template_identity(template_path)
    if pool_path is None:
        raise FileNotFoundError(
            f"No entity pool found for {theme}/{document_id}. Expected "
            f"data/GENERATED_FICTIONAL_ENTITIES/{theme}/{document_id}_entity_pool.yaml."
        )

    context = build_fictional_generation_context(document)
    pool_data = load_entity_pool(str(pool_path))
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
    )
    if named_entity_sample is None:
        raise RuntimeError(
            f"Named-only fictional sampling returned no assignment for {document_id} on seed {seed}."
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


def export_fictional_dataset_document(
    template_path: Path,
    *,
    setting_spec: DatasetSettingSpec,
    seed: int,
    variant_index: int = 1,
    variant_count: int = 1,
    overwrite: bool = False,
    pool_path: Path | None = None,
) -> Path:
    """Write one fictional dataset document version for one template."""
    theme, document_id = resolve_template_identity(template_path)
    output_path = document_variant_path(
        theme,
        document_id,
        setting_spec.setting_id,
        variant_index=variant_index,
        variant_count=variant_count,
    )
    if output_path.exists() and not overwrite:
        return output_path

    resolved_pool_path = pool_path or existing_entity_pool_path(theme, document_id)
    full_fictional_source_path = document_variant_path(
        theme,
        document_id,
        "fictional",
        variant_index=variant_index,
        variant_count=variant_count,
    )

    if _is_partial_all_entity_setting(setting_spec):
        try:
            payload = build_derived_fictional_dataset_record(
                template_path,
                setting_spec=setting_spec,
                seed=seed,
                output_path=output_path,
                variant_index=variant_index,
                variant_count=variant_count,
                strict_verification=True,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to direct %s generation for %s/%s v%02d after derived partial export failed: %s",
                setting_spec.setting_id,
                theme,
                document_id,
                variant_index,
                exc,
            )
            payload = build_fictional_dataset_record(
                template_path,
                setting_spec=setting_spec,
                pool_path=resolved_pool_path,
                seed=seed,
                output_path=output_path,
                variant_index=variant_index,
                variant_count=variant_count,
                include_existing_variant_values=False,
                force_allow_previous_numtemp_reuse=True,
            )
    elif setting_spec.replace_mode == REPLACE_MODE_ALL:
        payload = build_fictional_dataset_record(
            template_path,
            setting_spec=setting_spec,
            pool_path=resolved_pool_path,
            seed=seed,
            output_path=output_path,
            variant_index=variant_index,
            variant_count=variant_count,
            include_existing_variant_values=not overwrite,
        )
    elif full_fictional_source_path.exists():
        try:
            payload = build_derived_fictional_dataset_record(
                template_path,
                setting_spec=setting_spec,
                seed=seed,
                output_path=output_path,
                variant_index=variant_index,
                variant_count=variant_count,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to direct %s generation for %s/%s v%02d after derived export failed: %s",
                setting_spec.setting_id,
                theme,
                document_id,
                variant_index,
                exc,
            )
            payload = build_fictional_dataset_record(
                template_path,
                setting_spec=setting_spec,
                pool_path=resolved_pool_path,
                seed=seed,
                output_path=output_path,
                variant_index=variant_index,
                variant_count=variant_count,
                include_existing_variant_values=not overwrite,
            )
    else:
        resolved_pool_path = pool_path or existing_entity_pool_path(theme, document_id)
        payload = build_fictional_dataset_record(
            template_path,
            setting_spec=setting_spec,
            pool_path=resolved_pool_path,
            seed=seed,
            output_path=output_path,
            variant_index=variant_index,
            variant_count=variant_count,
            include_existing_variant_values=not overwrite,
        )
    return _write_yaml(payload, output_path)


def export_fictional_dataset_documents_batch(
    template_path: Path,
    *,
    setting_spec: DatasetSettingSpec,
    seed: int,
    variant_count: int = 1,
    overwrite: bool = False,
    pool_path: Path | None = None,
) -> list[Path]:
    """Write all fictional variants for one template/setting atomically."""
    if int(variant_count) < 1:
        raise ValueError(f"variant_count must be >= 1, got {variant_count!r}.")

    theme, document_id = resolve_template_identity(template_path)
    output_paths = [
        document_variant_path(
            theme,
            document_id,
            setting_spec.setting_id,
            variant_index=variant_index,
            variant_count=variant_count,
        )
        for variant_index in range(1, int(variant_count) + 1)
    ]
    if all(path.exists() for path in output_paths) and not overwrite:
        return output_paths

    resolved_pool_path = pool_path or existing_entity_pool_path(theme, document_id)
    singleton_number_ids = _singleton_domain_number_ids_for_template(
        template_path,
        variant_count=variant_count,
    )
    batch_error: Exception | None = None

    for batch_attempt in range(_MAX_BATCH_GENERATION_ATTEMPTS):
        print(
            f"[batch] start theme={theme} doc={document_id} setting={setting_spec.setting_id} "
            f"attempt={batch_attempt + 1}/{_MAX_BATCH_GENERATION_ATTEMPTS}",
            flush=True,
        )
        batch_used_number_values: dict[str, set[int | float]] = {}
        batch_used_temporal_years: dict[str, set[int]] = {}
        batch_used_temporal_values: dict[str, dict[str, set[Any]]] = {}
        payloads: list[dict[str, Any]] = []
        force_allow_previous_numtemp_reuse = False
        batch_used_relaxed_intervariant_reuse = False
        failed = False

        for variant_index, output_path in enumerate(output_paths, start=1):
            print(
                f"[batch] generating theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                f"variant={variant_index}/{variant_count}",
                flush=True,
            )
            version_seed = seed + ((variant_index - 1) * 1_000_000) + (batch_attempt * 10_000_000)
            try:
                if _is_partial_all_entity_setting(setting_spec):
                    try:
                        payload = build_derived_fictional_dataset_record(
                            template_path,
                            setting_spec=setting_spec,
                            seed=version_seed,
                            output_path=output_path,
                            variant_index=variant_index,
                            variant_count=variant_count,
                            strict_verification=True,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Falling back to direct %s generation for %s/%s v%02d after derived partial export failed: %s",
                            setting_spec.setting_id,
                            theme,
                            document_id,
                            variant_index,
                            exc,
                        )
                        payload = build_fictional_dataset_record(
                            template_path,
                            setting_spec=setting_spec,
                            pool_path=resolved_pool_path,
                            seed=version_seed,
                            output_path=output_path,
                            variant_index=variant_index,
                            variant_count=variant_count,
                            used_number_values_by_id=None,
                            used_temporal_years_by_id=None,
                            used_temporal_values_by_id=None,
                            include_existing_variant_values=False,
                            force_allow_previous_numtemp_reuse=True,
                        )
                elif setting_spec.replace_mode == REPLACE_MODE_ALL:
                    payload = build_fictional_dataset_record(
                        template_path,
                        setting_spec=setting_spec,
                        pool_path=resolved_pool_path,
                        seed=version_seed,
                        output_path=output_path,
                        variant_index=variant_index,
                        variant_count=variant_count,
                        used_number_values_by_id=batch_used_number_values,
                        used_temporal_years_by_id=batch_used_temporal_years,
                        used_temporal_values_by_id=batch_used_temporal_values,
                        include_existing_variant_values=False,
                        force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                    )
                elif document_variant_path(
                    theme,
                    document_id,
                    "fictional",
                    variant_index=variant_index,
                    variant_count=variant_count,
                ).exists():
                    payload = build_derived_fictional_dataset_record(
                        template_path,
                        setting_spec=setting_spec,
                        seed=version_seed,
                        output_path=output_path,
                        variant_index=variant_index,
                        variant_count=variant_count,
                    )
                else:
                    payload = build_fictional_dataset_record(
                        template_path,
                        setting_spec=setting_spec,
                        pool_path=resolved_pool_path,
                        seed=version_seed,
                        output_path=output_path,
                        variant_index=variant_index,
                        variant_count=variant_count,
                        used_number_values_by_id=batch_used_number_values,
                        used_temporal_years_by_id=batch_used_temporal_years,
                        used_temporal_values_by_id=batch_used_temporal_values,
                        include_existing_variant_values=False,
                        force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                    )
            except Exception as exc:
                failed = True
                batch_error = exc
                logger.warning(
                    "Retrying batch fictional generation for %s/%s setting=%s after variant v%02d failed on batch attempt %d: %s",
                    theme,
                    document_id,
                    setting_spec.setting_id,
                    variant_index,
                    batch_attempt + 1,
                    exc,
                )
                print(
                    f"[batch] retry theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                    f"after failed variant={variant_index}/{variant_count}: {exc}",
                    flush=True,
                )
                break

            used_relaxed_intervariant_reuse = bool(payload.pop("_used_relaxed_intervariant_reuse", False))
            if used_relaxed_intervariant_reuse:
                force_allow_previous_numtemp_reuse = True
                batch_used_relaxed_intervariant_reuse = True
            payloads.append(payload)
            _write_yaml(payload, output_path)
            entities_used = payload.get("entities_used") or {}
            number_payloads = entities_used.get("numbers") or {}
            if isinstance(number_payloads, dict):
                for number_id, raw_number_payload in number_payloads.items():
                    if str(number_id) in singleton_number_ids:
                        continue
                    normalized = _normalize_number_uniqueness_value(raw_number_payload)
                    if normalized is None:
                        continue
                    avoid_value = normalized["value"]
                    if normalized.get("kind") == "fraction":
                        try:
                            number_entity = NumberEntity.model_validate(raw_number_payload or {})
                        except Exception:
                            number_entity = None
                        denominator = NumberValueStrategyMixin._number_entity_int_value(number_entity)
                        if denominator is not None:
                            avoid_value = denominator
                    batch_used_number_values.setdefault(str(number_id), set()).add(avoid_value)
            temporal_payloads = entities_used.get("temporals") or {}
            if isinstance(temporal_payloads, dict):
                for temporal_id, raw_temporal_payload in temporal_payloads.items():
                    _record_temporal_payload_values(
                        batch_used_temporal_values,
                        str(temporal_id),
                        raw_temporal_payload or {},
                    )
                    try:
                        temporal_year = int((raw_temporal_payload or {}).get("year"))
                    except (TypeError, ValueError, AttributeError):
                        continue
                    batch_used_temporal_years.setdefault(str(temporal_id), set()).add(temporal_year)

        if failed:
            continue

        duplicates = _collect_intervariant_duplicates(
            payloads,
            setting_spec=setting_spec,
            singleton_number_ids=singleton_number_ids,
        )
        if duplicates and not batch_used_relaxed_intervariant_reuse:
            batch_error = RuntimeError(
                f"Inter-variant value reuse detected for {theme}/{document_id} "
                f"setting={setting_spec.setting_id}: {json.dumps(duplicates[:4], ensure_ascii=True)}"
            )
            logger.warning(
                "Retrying batch fictional generation for %s/%s setting=%s after duplicate values on batch attempt %d.",
                theme,
                document_id,
                setting_spec.setting_id,
                batch_attempt + 1,
            )
            print(
                f"[batch] retry theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                f"after duplicate values: {json.dumps(duplicates[:4], ensure_ascii=True)}",
                flush=True,
            )
            continue
        if duplicates:
            logger.warning(
                "Accepting inter-variant value reuse for %s/%s setting=%s on batch attempt %d.",
                theme,
                document_id,
                setting_spec.setting_id,
                batch_attempt + 1,
            )
            print(
                f"[batch] warning theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                f"duplicate values accepted: {json.dumps(duplicates[:4], ensure_ascii=True)}",
                flush=True,
            )

        written_paths: list[Path] = list(output_paths)
        print(
            f"[batch] done theme={theme} doc={document_id} setting={setting_spec.setting_id} "
            f"variants={len(written_paths)}",
            flush=True,
        )
        return written_paths

    if batch_error is not None:
        raise batch_error
    raise RuntimeError(
        f"Failed to generate {theme}/{document_id} setting={setting_spec.setting_id} after "
        f"{_MAX_BATCH_GENERATION_ATTEMPTS} batch attempts."
    )
