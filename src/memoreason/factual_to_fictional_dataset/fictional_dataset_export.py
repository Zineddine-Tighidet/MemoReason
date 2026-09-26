"""Fictional dataset export built from templates plus Claude-generated entity pools."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from memoreason.benchmark_definition.annotation_runtime import (
    REPLACE_MODE_ALL,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_uniqueness import (
    number_entity_uniqueness_value,
)

from .dataset_paths import (
    document_variant_path,
    existing_entity_pool_path,
    resolve_template_identity,
)
from .dataset_record_core import _relative_to_project, _write_yaml
from .dataset_settings import FactualToFictionalDatasetSetting
from .fictional_dataset_records import (
    build_derived_fictional_dataset_record,
    build_fictional_dataset_record,
    build_named_only_fictional_dataset_record,
)
from .fictional_dataset_projection_reuse import (
    _validate_partial_projection_intervariant_reuse,
)
from .fictional_dataset_reuse_certificates import (
    _validate_intervariant_reuse,
)
from .fictional_dataset_uniqueness import (
    _INTERVARIANT_NAMED_BUCKETS,
    _collect_intervariant_duplicates,
    _record_named_payload_value,
    _record_temporal_payload_values,
)

logger = logging.getLogger(__name__)
_MAX_BATCH_GENERATION_ATTEMPTS = int(os.environ.get("FICTIONAL_BATCH_GENERATION_ATTEMPTS", "1"))


def _is_partial_full_projection(setting_spec: FactualToFictionalDatasetSetting) -> bool:
    return setting_spec.replace_mode == REPLACE_MODE_ALL and 0.0 < setting_spec.replacement_proportion < 1.0


def _persist_runtime_reuse_audit(
    payloads: list[dict[str, Any]],
    accepted_reuse: list[dict[str, Any]],
) -> None:
    """Persist only validated runtime reuse occurrences for later projections."""
    records_by_variant: dict[str, list[dict[str, Any]]] = {}
    for accepted in accepted_reuse:
        certificate = accepted.get("runtime_reuse_certificate")
        if not isinstance(certificate, dict):
            continue
        repeated_variants_by_value = {
            json.dumps(item.get("value") or {}, sort_keys=True, ensure_ascii=True): sorted(
                str(variant) for variant in (item.get("variants") or [])
            )
            for item in accepted.get("repeated_values") or []
            if isinstance(item, dict)
        }
        for raw_record in certificate.get("reused_occurrences") or []:
            if not isinstance(raw_record, dict):
                continue
            record = dict(raw_record)
            signature = json.dumps(record.get("value") or {}, sort_keys=True, ensure_ascii=True)
            record["repeated_variants"] = repeated_variants_by_value.get(signature, [])
            records_by_variant.setdefault(str(record.get("variant_id") or ""), []).append(record)
    for payload in payloads:
        records = records_by_variant.get(str(payload.get("document_variant_id") or ""), [])
        if records:
            payload["intervariant_reuse_audit"] = sorted(
                records,
                key=lambda item: (
                    str(item.get("entity_bucket") or ""),
                    str(item.get("entity_ref") or ""),
                    str(item.get("entity_attr") or ""),
                    json.dumps(item.get("value") or {}, sort_keys=True, ensure_ascii=True),
                ),
            )


def export_fictional_dataset_document(
    template_path: Path,
    *,
    setting_spec: FactualToFictionalDatasetSetting,
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

    if setting_spec.replace_mode == REPLACE_MODE_ALL and setting_spec.replacement_proportion == 1.0:
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
    elif setting_spec.setting_id == "fictional_named":
        payload = build_named_only_fictional_dataset_record(
            template_path,
            setting_spec=setting_spec,
            pool_path=resolved_pool_path,
            seed=seed,
            output_path=output_path,
            variant_index=variant_index,
            variant_count=variant_count,
            include_existing_variant_values=not overwrite,
        )
    elif _is_partial_full_projection(setting_spec):
        if not full_fictional_source_path.exists():
            raise FileNotFoundError(
                f"Cannot export {setting_spec.setting_id} for {theme}/{document_id} without its "
                f"full-fictional source: {full_fictional_source_path}"
            )
        payload = build_derived_fictional_dataset_record(
            template_path,
            setting_spec=setting_spec,
            seed=seed,
            output_path=output_path,
            variant_index=variant_index,
            variant_count=variant_count,
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
    used_relaxed_reuse = bool(payload.pop("_used_relaxed_intervariant_reuse", False))
    runtime_reuse_audit = payload.pop("_relaxed_intervariant_reuse_audit", [])
    if used_relaxed_reuse or runtime_reuse_audit:
        raise ValueError(
            "Single-document export cannot certify inter-variant reuse; use the atomic batch exporter."
        )
    return _write_yaml(payload, output_path)


def export_fictional_dataset_documents_batch(
    template_path: Path,
    *,
    setting_spec: FactualToFictionalDatasetSetting,
    seed: int,
    variant_count: int = 1,
    overwrite: bool = False,
    pool_path: Path | None = None,
) -> list[Path]:
    """Generate and audit a full template/setting batch before any writes."""
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
    existing_output_paths = [path for path in output_paths if path.exists()]
    if existing_output_paths and not overwrite:
        if len(existing_output_paths) == len(output_paths):
            return output_paths
        rendered = ", ".join(path.name for path in existing_output_paths)
        raise FileExistsError(
            f"Refusing to resume an incomplete fictional batch for {theme}/{document_id} "
            f"setting={setting_spec.setting_id}; preserve this attempt and use a fresh output root. "
            f"Existing files: {rendered}"
        )

    resolved_pool_path = pool_path or existing_entity_pool_path(theme, document_id)
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
        batch_used_named_values: dict[str, set[str]] = {}
        payloads: list[dict[str, Any]] = []
        runtime_reuse_audit: list[dict[str, Any]] = []
        force_allow_previous_numtemp_reuse = False
        failed = False

        for variant_index, output_path in enumerate(output_paths, start=1):
            print(
                f"[batch] generating theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                f"variant={variant_index}/{variant_count}",
                flush=True,
            )
            version_seed = seed + ((variant_index - 1) * 1_000_000) + (batch_attempt * 10_000_000)
            try:
                if setting_spec.replace_mode == REPLACE_MODE_ALL and setting_spec.replacement_proportion == 1.0:
                    payload = build_fictional_dataset_record(
                        template_path,
                        setting_spec=setting_spec,
                        pool_path=resolved_pool_path,
                        seed=version_seed,
                        output_path=output_path,
                        variant_index=variant_index,
                        variant_count=variant_count,
                        used_named_values_by_id=batch_used_named_values,
                        used_number_values_by_id=batch_used_number_values,
                        used_temporal_years_by_id=batch_used_temporal_years,
                        used_temporal_values_by_id=batch_used_temporal_values,
                        prior_relaxed_intervariant_reuse_audit=runtime_reuse_audit,
                        include_existing_variant_values=False,
                        force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                    )
                elif setting_spec.setting_id == "fictional_named":
                    payload = build_named_only_fictional_dataset_record(
                        template_path,
                        setting_spec=setting_spec,
                        pool_path=resolved_pool_path,
                        seed=version_seed,
                        output_path=output_path,
                        variant_index=variant_index,
                        variant_count=variant_count,
                        used_named_values_by_id=batch_used_named_values,
                        include_existing_variant_values=False,
                    )
                elif _is_partial_full_projection(setting_spec):
                    full_fictional_source_path = document_variant_path(
                        theme,
                        document_id,
                        "fictional",
                        variant_index=variant_index,
                        variant_count=variant_count,
                    )
                    if not full_fictional_source_path.exists():
                        raise FileNotFoundError(
                            f"Cannot export {setting_spec.setting_id} for {theme}/{document_id} without its "
                            f"full-fictional source: {full_fictional_source_path}"
                        )
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
                        used_named_values_by_id=batch_used_named_values,
                        used_number_values_by_id=batch_used_number_values,
                        used_temporal_years_by_id=batch_used_temporal_years,
                        used_temporal_values_by_id=batch_used_temporal_values,
                        prior_relaxed_intervariant_reuse_audit=runtime_reuse_audit,
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
            raw_runtime_reuse_audit = payload.pop("_relaxed_intervariant_reuse_audit", [])
            if raw_runtime_reuse_audit:
                if not isinstance(raw_runtime_reuse_audit, list) or not all(
                    isinstance(item, dict) for item in raw_runtime_reuse_audit
                ):
                    raise ValueError("Invalid internal relaxed inter-variant reuse audit payload.")
                runtime_reuse_audit.extend(dict(item) for item in raw_runtime_reuse_audit)
            if used_relaxed_intervariant_reuse:
                # Every subsequent variant still gets a complete strict pass;
                # runtime reuse is certified per occurrence, never enabled as
                # a batch-wide waiver.
                force_allow_previous_numtemp_reuse = False
            payloads.append(payload)
            entities_used = payload.get("entities_used") or {}
            replaced_entities = payload.get("replaced_factual_entities") or {}
            for bucket in _INTERVARIANT_NAMED_BUCKETS:
                replaced_bucket = replaced_entities.get(bucket) or {}
                used_bucket = entities_used.get(bucket) or {}
                if not isinstance(replaced_bucket, dict) or not isinstance(used_bucket, dict):
                    continue
                for entity_id in replaced_bucket:
                    raw_named_payload = used_bucket.get(entity_id)
                    if isinstance(raw_named_payload, dict):
                        _record_named_payload_value(
                            batch_used_named_values,
                            bucket,
                            str(entity_id),
                            raw_named_payload,
                        )
            number_payloads = entities_used.get("numbers") or {}
            if isinstance(number_payloads, dict):
                for number_id, raw_number_payload in number_payloads.items():
                    avoid_value = number_entity_uniqueness_value(raw_number_payload)
                    if avoid_value is None:
                        continue
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
        )
        # Reuse is a bounded-search policy outcome, not a validity failure.
        # Do not run the former finite-domain certificate solvers on every
        # setting: rule validity is already checked per variant, and the
        # duplicate audit below remains deterministic and explicit.
        capacity_certificates = {}
        try:
            if duplicates and _is_partial_full_projection(setting_spec):
                full_payloads: list[dict[str, Any]] = []
                full_source_paths_by_variant: dict[str, str] = {}
                full_source_sha256_by_variant: dict[str, str] = {}
                for variant_index in range(1, int(variant_count) + 1):
                    variant_id = f"v{variant_index:02d}"
                    full_source_path = document_variant_path(
                        theme,
                        document_id,
                        "fictional",
                        variant_index=variant_index,
                        variant_count=variant_count,
                    )
                    full_source_bytes = full_source_path.read_bytes()
                    full_payload = yaml.safe_load(full_source_bytes)
                    if not isinstance(full_payload, dict):
                        raise ValueError(f"Invalid full-fictional source payload: {full_source_path}")
                    full_payloads.append(full_payload)
                    full_source_paths_by_variant[variant_id] = _relative_to_project(full_source_path)
                    full_source_sha256_by_variant[variant_id] = hashlib.sha256(full_source_bytes).hexdigest()
                accepted_reuse = _validate_partial_projection_intervariant_reuse(
                    duplicates,
                    partial_payloads=payloads,
                    full_payloads=full_payloads,
                    full_source_paths_by_variant=full_source_paths_by_variant,
                    full_source_sha256_by_variant=full_source_sha256_by_variant,
                    capacity_certificates=capacity_certificates,
                    variant_count=variant_count,
                )
            else:
                accepted_reuse = _validate_intervariant_reuse(
                    duplicates,
                    capacity_certificates=capacity_certificates,
                    variant_count=variant_count,
                    runtime_reuse_audit=runtime_reuse_audit,
                )
        except ValueError as exc:
            batch_error = exc
            logger.warning(
                "Rejecting fictional batch for %s/%s setting=%s on attempt %d before writes: %s",
                theme,
                document_id,
                setting_spec.setting_id,
                batch_attempt + 1,
                exc,
            )
            print(
                f"[batch] reject theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                f"intervariant_audit={exc}",
                flush=True,
            )
            continue
        if accepted_reuse:
            logger.info(
                "Accepting capacity-certified inter-variant reuse for %s/%s setting=%s on batch attempt %d: %s",
                theme,
                document_id,
                setting_spec.setting_id,
                batch_attempt + 1,
                json.dumps(accepted_reuse, sort_keys=True, ensure_ascii=True),
            )
            print(
                f"[batch] certified-reuse theme={theme} doc={document_id} setting={setting_spec.setting_id} "
                f"audit={json.dumps(accepted_reuse, sort_keys=True, ensure_ascii=True)}",
                flush=True,
            )
            _persist_runtime_reuse_audit(payloads, accepted_reuse)

        for payload, output_path in zip(payloads, output_paths, strict=True):
            _write_yaml(payload, output_path)
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
