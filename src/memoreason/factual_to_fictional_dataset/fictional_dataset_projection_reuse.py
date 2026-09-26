"""Fail-closed reuse validation for deterministic partial projections."""

from __future__ import annotations

import json
from typing import Any

from memoreason.factual_to_fictional_dataset.dataset_settings import parse_dataset_setting
from memoreason.factual_to_fictional_dataset.fictional_dataset_reuse_certificates import (
    CapacityCertificateKey,
    _validate_intervariant_reuse,
    _value_signatures,
)
from memoreason.factual_to_fictional_dataset.fictional_dataset_uniqueness import (
    _collect_intervariant_duplicates,
)


def _duplicate_identity(duplicate: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(duplicate.get("entity_bucket") or ""),
        str(duplicate.get("entity_ref") or ""),
        str(duplicate.get("entity_attr") or ""),
    )


def _payloads_by_variant(
    payloads: list[dict[str, Any]],
    *,
    label: str,
    variant_count: int,
) -> dict[str, dict[str, Any]]:
    expected_variant_ids = {f"v{variant_index:02d}" for variant_index in range(1, int(variant_count) + 1)}
    indexed: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        if not isinstance(payload, dict):
            raise ValueError(f"{label} contains a non-mapping payload.")
        variant_id = str(payload.get("document_variant_id") or "")
        if variant_id in indexed:
            raise ValueError(f"{label} repeats document_variant_id={variant_id!r}.")
        indexed[variant_id] = payload
    if set(indexed) != expected_variant_ids:
        raise ValueError(
            f"{label} variant ids do not match the exact batch: "
            f"expected={sorted(expected_variant_ids)!r} observed={sorted(indexed)!r}."
        )
    return indexed


def _validate_projection_payload_lineage(
    *,
    partial_by_variant: dict[str, dict[str, Any]],
    full_by_variant: dict[str, dict[str, Any]],
    full_source_paths_by_variant: dict[str, str],
    full_source_sha256_by_variant: dict[str, str],
) -> None:
    for variant_id, partial_payload in sorted(partial_by_variant.items()):
        full_payload = full_by_variant[variant_id]
        if partial_payload.get("generation_source") != "derived_from_full_fictional":
            raise ValueError(f"Partial projection {variant_id} has invalid generation_source.")
        expected_source_path = full_source_paths_by_variant.get(variant_id)
        if partial_payload.get("source_fictional_document_path") != expected_source_path:
            raise ValueError(f"Partial projection {variant_id} has invalid full-fictional source path.")
        expected_source_sha256 = full_source_sha256_by_variant.get(variant_id)
        if partial_payload.get("source_fictional_document_sha256") != expected_source_sha256:
            raise ValueError(f"Partial projection {variant_id} has invalid full-fictional source SHA-256.")

        partial_entities = partial_payload.get("entities_used") or {}
        full_entities = full_payload.get("entities_used") or {}
        replaced_entities = partial_payload.get("replaced_factual_entities") or {}
        if not all(isinstance(value, dict) for value in (partial_entities, full_entities, replaced_entities)):
            raise ValueError(f"Partial projection {variant_id} has invalid entity mappings.")
        for bucket, replaced_bucket in replaced_entities.items():
            if not isinstance(replaced_bucket, dict):
                raise ValueError(f"Partial projection {variant_id} has invalid replaced bucket {bucket!r}.")
            partial_bucket = partial_entities.get(bucket) or {}
            full_bucket = full_entities.get(bucket) or {}
            if not isinstance(partial_bucket, dict) or not isinstance(full_bucket, dict):
                raise ValueError(f"Partial projection {variant_id} has invalid source bucket {bucket!r}.")
            for entity_ref in replaced_bucket:
                if partial_bucket.get(entity_ref) != full_bucket.get(entity_ref):
                    raise ValueError(
                        f"Partial projection {variant_id} does not inherit {bucket}/{entity_ref} "
                        "byte-semantically from its full-fictional source."
                    )


def _validate_partial_projection_intervariant_reuse(
    duplicates: list[dict[str, Any]],
    *,
    partial_payloads: list[dict[str, Any]],
    full_payloads: list[dict[str, Any]],
    full_source_paths_by_variant: dict[str, str],
    full_source_sha256_by_variant: dict[str, str],
    capacity_certificates: dict[CapacityCertificateKey, dict[str, Any]],
    variant_count: int,
) -> list[dict[str, Any]]:
    """Validate partial reuse as an exact projection of a certified full batch."""
    partial_by_variant = _payloads_by_variant(
        partial_payloads,
        label="partial projection batch",
        variant_count=variant_count,
    )
    full_by_variant = _payloads_by_variant(
        full_payloads,
        label="full-fictional source batch",
        variant_count=variant_count,
    )
    _validate_projection_payload_lineage(
        partial_by_variant=partial_by_variant,
        full_by_variant=full_by_variant,
        full_source_paths_by_variant=full_source_paths_by_variant,
        full_source_sha256_by_variant=full_source_sha256_by_variant,
    )

    full_duplicates = _collect_intervariant_duplicates(
        list(full_by_variant.values()),
        setting_spec=parse_dataset_setting("fictional"),
    )
    full_runtime_reuse_audit: list[dict[str, Any]] = []
    for variant_id, payload in sorted(full_by_variant.items()):
        raw_audit = payload.get("intervariant_reuse_audit") or []
        if not isinstance(raw_audit, list) or not all(isinstance(item, dict) for item in raw_audit):
            raise ValueError(f"Full-fictional source {variant_id} has invalid intervariant_reuse_audit.")
        for item in raw_audit:
            if str(item.get("variant_id") or "") != variant_id:
                raise ValueError(f"Full-fictional source {variant_id} has a cross-variant reuse audit record.")
            full_runtime_reuse_audit.append(dict(item))
    accepted_full = _validate_intervariant_reuse(
        full_duplicates,
        capacity_certificates=capacity_certificates,
        variant_count=variant_count,
        runtime_reuse_audit=full_runtime_reuse_audit,
    )
    accepted_full_by_identity = {_duplicate_identity(item): item for item in accepted_full}

    accepted_partial: list[dict[str, Any]] = []
    rejected_partial: list[dict[str, Any]] = []
    for duplicate in duplicates:
        full_duplicate = accepted_full_by_identity.get(_duplicate_identity(duplicate))
        if full_duplicate is None:
            rejected_partial.append(
                {
                    **duplicate,
                    "rejection_reason": "partial_duplicate_not_capacity_certified_in_full_source_batch",
                }
            )
            continue

        full_distinct_signatures = _value_signatures(full_duplicate.get("distinct_values") or [])
        partial_distinct_signatures = _value_signatures(duplicate.get("distinct_values") or [])
        if not partial_distinct_signatures.issubset(full_distinct_signatures):
            rejected_partial.append(
                {
                    **duplicate,
                    "rejection_reason": "partial_distinct_value_not_present_in_full_source_batch",
                }
            )
            continue

        full_repetitions = {
            json.dumps(item.get("value") or {}, sort_keys=True, ensure_ascii=True): set(item.get("variants") or [])
            for item in full_duplicate.get("repeated_values") or []
        }
        invalid_repetition = False
        for repeated_value in duplicate.get("repeated_values") or []:
            signature = json.dumps(repeated_value.get("value") or {}, sort_keys=True, ensure_ascii=True)
            partial_variants = set(repeated_value.get("variants") or [])
            if not partial_variants.issubset(full_repetitions.get(signature, set())):
                invalid_repetition = True
                break
        if invalid_repetition:
            rejected_partial.append(
                {
                    **duplicate,
                    "rejection_reason": "partial_repetition_not_inherited_from_same_full_variants",
                }
            )
            continue

        runtime_certificate = full_duplicate.get("runtime_reuse_certificate")
        policy_certificate = full_duplicate.get("policy_certificate")
        accepted_partial.append(
            {
                **duplicate,
                "projection_reuse_contract": (
                    "exact_subset_of_runtime_certified_full_fictional_v1"
                    if runtime_certificate
                    else (
                        "exact_subset_of_policy_accepted_full_fictional_v1"
                        if policy_certificate
                        else "exact_subset_of_capacity_certified_full_fictional_v1"
                    )
                ),
                "full_source_distinct_count": int(full_duplicate.get("distinct_count") or 0),
                "full_source_observed_variant_count": int(full_duplicate.get("observed_variant_count") or 0),
                "capacity_certificate": full_duplicate.get("capacity_certificate"),
                "runtime_reuse_certificate": runtime_certificate,
                "policy_certificate": policy_certificate,
            }
        )

    if rejected_partial:
        raise ValueError(
            "Partial projection inter-variant reuse audit failed before dataset writes: "
            + json.dumps(rejected_partial, sort_keys=True, ensure_ascii=True)
        )
    return accepted_partial
