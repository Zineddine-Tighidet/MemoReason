"""Fictional dataset export built from templates plus Claude-generated entity pools."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from memoreason.benchmark_definition.entity_taxonomy import REPLACE_MODE_NON_NUMERICAL, REPLACE_MODE_NUMERICAL_TEMPORAL
from memoreason.benchmark_definition.document_schema import NumberEntity
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.named_entity_uniqueness import (
    named_entity_uniqueness_signature,
    normalize_named_entity_uniqueness_value,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_generation.value_strategy import (
    NumberValueStrategyMixin,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_uniqueness import (
    number_entity_uniqueness_payload,
)
from .dataset_paths import (
    format_document_variant_id,
)
from .dataset_settings import FactualToFictionalDatasetSetting

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
_TEMPORAL_AVOID_FIELDS = ("year", "month", "day", "day_of_month", "timestamp")


def _record_named_payload_value(
    used_values_by_id: dict[str, set[str]],
    bucket: str,
    entity_id: str,
    raw_named_payload: dict[str, Any],
) -> None:
    signature = named_entity_uniqueness_signature(bucket, raw_named_payload)
    if signature is not None:
        used_values_by_id.setdefault(str(entity_id), set()).add(signature)


def _collect_used_named_values_from_existing_variants(
    *,
    output_path: Path,
    document_id: str,
    variant_index: int,
    variant_count: int,
) -> dict[str, set[str]]:
    if int(variant_count) <= 1:
        return {}
    used_values_by_id: dict[str, set[str]] = {}
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
        entities_used = sibling_payload.get("entities_used") or {}
        replaced_entities = sibling_payload.get("replaced_factual_entities") or {}
        for bucket in _INTERVARIANT_NAMED_BUCKETS:
            replaced_bucket = replaced_entities.get(bucket) or {}
            used_bucket = entities_used.get(bucket) or {}
            if not isinstance(replaced_bucket, dict) or not isinstance(used_bucket, dict):
                continue
            for entity_id in replaced_bucket:
                raw_named_payload = used_bucket.get(entity_id)
                if isinstance(raw_named_payload, dict):
                    _record_named_payload_value(
                        used_values_by_id,
                        bucket,
                        str(entity_id),
                        raw_named_payload,
                    )
    return used_values_by_id


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
        number_payloads = (sibling_payload.get("entities_used") or {}).get("numbers") or {}
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
        temporal_payloads = (sibling_payload.get("entities_used") or {}).get("temporals") or {}
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
        temporal_payloads = (sibling_payload.get("entities_used") or {}).get("temporals") or {}
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


def _merge_used_named_values(
    existing: dict[str, set[str]] | None,
    additional: dict[str, set[str]],
) -> dict[str, set[str]]:
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


def _replacement_buckets_for_setting(setting_spec: FactualToFictionalDatasetSetting) -> tuple[str, ...]:
    if setting_spec.replace_mode == REPLACE_MODE_NON_NUMERICAL:
        return _INTERVARIANT_NAMED_BUCKETS
    if setting_spec.replace_mode == REPLACE_MODE_NUMERICAL_TEMPORAL:
        return _INTERVARIANT_NUMTEMP_BUCKETS
    return _INTERVARIANT_NAMED_BUCKETS + _INTERVARIANT_NUMTEMP_BUCKETS


def _normalize_number_uniqueness_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    return number_entity_uniqueness_payload(payload)


def _normalize_temporal_uniqueness_value(payload: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    value = {field: payload[field] for field in _TEMPORAL_FIELDS if payload.get(field) not in (None, "")}
    return value or None


def _normalize_named_uniqueness_value(bucket: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return normalize_named_entity_uniqueness_value(bucket, payload)


def _normalize_uniqueness_value(bucket: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    if bucket == "numbers":
        return _normalize_number_uniqueness_value(payload)
    if bucket == "temporals":
        return _normalize_temporal_uniqueness_value(payload)
    return _normalize_named_uniqueness_value(bucket, payload)


def _collect_intervariant_duplicates(
    payloads: list[dict[str, Any]],
    *,
    setting_spec: FactualToFictionalDatasetSetting,
) -> list[dict[str, Any]]:
    relevant_buckets = set(_replacement_buckets_for_setting(setting_spec))
    values_by_ref: dict[tuple[str, str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    normalized_values: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}

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
            for entity_ref, replaced_payload in replaced_bucket.items():
                entity_payload = used_bucket.get(entity_ref)
                if not isinstance(entity_payload, dict):
                    continue
                if bucket == "temporals":
                    replaced_attrs = set(replaced_payload) if isinstance(replaced_payload, dict) else set()
                    if "date" in replaced_attrs:
                        replaced_attrs.update({"year", "month", "day_of_month"})
                        if entity_payload.get("day") not in (None, ""):
                            replaced_attrs.add("day")
                    for attr in _TEMPORAL_AVOID_FIELDS:
                        if attr not in replaced_attrs:
                            continue
                        value = entity_payload.get(attr)
                        if value in (None, ""):
                            continue
                        normalized = {attr: value}
                        signature = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
                        key = (bucket, str(entity_ref), attr)
                        values_by_ref[key][signature].append(variant_id)
                        normalized_values.setdefault(key, {})[signature] = normalized
                    continue
                normalized = _normalize_uniqueness_value(bucket, entity_payload)
                if normalized is None:
                    continue
                signature = (
                    named_entity_uniqueness_signature(bucket, entity_payload)
                    if bucket in _INTERVARIANT_NAMED_BUCKETS
                    else json.dumps(normalized, sort_keys=True, ensure_ascii=True)
                )
                if signature is None:
                    continue
                key = (bucket, str(entity_ref), "")
                values_by_ref[key][signature].append(variant_id)
                normalized_values.setdefault(key, {})[signature] = normalized

    duplicates: list[dict[str, Any]] = []
    for (bucket, entity_ref, entity_attr), variants_by_signature in sorted(values_by_ref.items()):
        repeated_values = []
        for signature, variants in sorted(variants_by_signature.items()):
            if len(variants) < 2:
                continue
            repeated_values.append(
                {
                    "value": normalized_values[(bucket, entity_ref, entity_attr)][signature],
                    "variants": sorted(variants),
                }
            )
        if repeated_values:
            all_normalized_values = normalized_values[(bucket, entity_ref, entity_attr)].values()
            duplicate = {
                "entity_bucket": bucket,
                "entity_ref": entity_ref,
                "distinct_count": len(variants_by_signature),
                "observed_variant_count": sum(len(variants) for variants in variants_by_signature.values()),
                "observed_value_fields": sorted(
                    {field for normalized_value in all_normalized_values for field in normalized_value}
                ),
                "distinct_values": [
                    normalized_values[(bucket, entity_ref, entity_attr)][signature]
                    for signature in sorted(variants_by_signature)
                ],
                "repeated_values": repeated_values,
            }
            if entity_attr:
                duplicate["entity_attr"] = entity_attr
            duplicates.append(duplicate)
    return duplicates
