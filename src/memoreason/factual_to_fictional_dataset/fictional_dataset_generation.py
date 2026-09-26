"""Fictional dataset export built from templates plus Claude-generated entity pools."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.controlled_entity_replacement_algorithm import (
    ReviewedTemplateFictionalGenerationInput,
    FictionalDocumentVariantRequest,
    generate_fictional_document_variants,
)

from .dataset_settings import FactualToFictionalDatasetSetting

_NON_RETRYABLE_VERIFICATION_PREFIXES = (
    "Entity pool cannot satisfy the required manual attributes for this document:",
    "Generated payload failed semantic linting:\nreplaced factual literals still appear in output:",
    "Generated payload failed semantic linting:\nplace/self-reference collision:",
)


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
    setting_spec: FactualToFictionalDatasetSetting,
    entity_pool: dict[str, Any],
    seed: int,
    output_path: Path,
    variant_index: int = 1,
    variant_count: int = 1,
    used_named_values_by_id: dict[str, set[str]] | None = None,
    used_number_values_by_id: dict[str, set[int]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
    prior_relaxed_intervariant_reuse_audit: list[dict[str, Any]] | None = None,
    force_allow_previous_numtemp_reuse: bool = False,
    return_metadata: bool = False,
) -> tuple[dict[str, Any], int] | tuple[dict[str, Any], int, dict[str, Any]]:
    """Generate one fictional dataset document with the active replacement workflow."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
        algorithm_result = generate_fictional_document_variants(
            [
                ReviewedTemplateFictionalGenerationInput(
                    template_document=document,
                    named_entity_pool=entity_pool,
                    variant_requests=(
                        FictionalDocumentVariantRequest(
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
                    used_named_values_by_id=used_named_values_by_id,
                    used_number_values_by_id=used_number_values_by_id,
                    used_temporal_years_by_id=used_temporal_years_by_id,
                    used_temporal_values_by_id=used_temporal_values_by_id,
                    prior_relaxed_intervariant_reuse_audit=tuple(
                        dict(item) for item in (prior_relaxed_intervariant_reuse_audit or [])
                    ),
                    force_allow_previous_numtemp_reuse=force_allow_previous_numtemp_reuse,
                )
            ]
        )[0]
        _generated_path, successful_seed = algorithm_result.generated_variants[0]

    generated_payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    if not isinstance(generated_payload, dict):
        raise ValueError(f"Invalid generated payload for {document.document_id}.")
    if return_metadata:
        return (
            generated_payload,
            successful_seed,
            {
                "used_relaxed_intervariant_reuse": bool(algorithm_result.used_relaxed_intervariant_reuse),
                "relaxed_intervariant_reuse_audit": [
                    dict(item)
                    for item in getattr(algorithm_result, "relaxed_intervariant_reuse_audit", ())
                ],
            },
        )
    return generated_payload, successful_seed
