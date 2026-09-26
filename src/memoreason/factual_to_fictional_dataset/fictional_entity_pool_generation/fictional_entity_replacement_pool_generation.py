"""Generate validated document-specific fictional entity replacement pools."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any


from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.generation_requirements import (
    extract_required_entities,
)
from memoreason.benchmark_definition.document_schema import AnnotatedDocument
from memoreason.benchmark_definition.organization_types import CANONICAL_ORGANIZATION_TYPES
from memoreason.benchmark_definition.annotation_runtime import (
    load_annotated_document,
    load_entity_pool,
)
from .pool_normalization import (
    _empty_normalized_pool,
    _extract_mapping_payload,
    _normalize_pool_dict,
    _pool_bucket_shortages as _pool_bucket_shortages_impl,
    _pool_support_shortages,
)
from .fictional_entity_replacement_pool_prompt_planning import (
    FictionalEntityReplacementPoolGenerationConfiguration,
    build_fictional_entity_replacement_pool_generation_prompt as _build_pool_generation_prompt_impl,
)
from .fictional_entity_replacement_pool_target_planning import (
    FictionalEntityReplacementPoolTargetPlan,
    build_fictional_entity_replacement_pool_target_plan,
    compute_required_fictional_entity_candidates_per_pool_bucket,
)
from .pool_validation import (
    _finalize_pool_candidate,
    _pool_wikipedia_hits,
)
from memoreason.model_providers.text_generation import TextGenerationRequest, generate_text
from ..dataset_paths import (
    generated_entity_pool_path,
    resolve_template_identity,
)

from .pool_candidate_validation import (
    _pool_banned_suffix_hits,
    _pool_factual_literal_hits,
    _pool_factual_literals,
    _prune_generation_unit_variants,
    _reference_shortage_messages,
    _validate_existing_pool,
)
from .pool_generation_planning import (
    _augment_required_entities_with_rule_attrs,
    _generation_units,
    _manual_named_required_entities,
    _merge_normalized_pools,
)
from .fictional_entity_replacement_pool_prompt_alignment import (
    _generation_unit_alignment_failures,
    _is_non_retryable_provider_error,
)
from .pool_persistence import (
    _file_sha256,
    _serializable_pool_payload as _serializable_pool_payload,
    load_saved_entity_pool as load_saved_entity_pool,
    save_entity_pool as save_entity_pool,
)
from .pool_generation_reproducibility import generation_timestamp_utc as _generation_timestamp_utc

logger = logging.getLogger(__name__)
_BANNED_POOL_SUFFIX_TOKENS = frozenset({"Alt", "Astra", "Nova", "Prime", "Sigma"})
_BANNED_POOL_SUFFIX_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in sorted(_BANNED_POOL_SUFFIX_TOKENS)) + r")\b"
)
_MANUAL_POOL_ENTITY_TYPES = {
    "person",
    "place",
    "event",
    *CANONICAL_ORGANIZATION_TYPES,
    "award",
    "legal",
    "product",
}
_AUTO_GENERATED_PERSON_ATTRIBUTES = frozenset(
    {
        "age",
        "gender",
        "subj_pronoun",
        "obj_pronoun",
        "poss_det_pronoun",
        "poss_pro_pronoun",
        "refl_pronoun",
        "honorific",
        "relationship",
    }
)
__all__ = [
    "FictionalEntityReplacementPoolGenerationConfiguration",
    "FictionalEntityReplacementPoolTargetPlan",
    "build_fictional_entity_replacement_pool_generation_prompt",
    "build_fictional_entity_replacement_pool_target_plan",
    "compute_required_fictional_entity_candidates_per_pool_bucket",
    "generate_fictional_entity_replacement_pool",
    "generate_fictional_entity_replacement_pool_for_template",
    "load_saved_entity_pool",
    "resolve_or_generate_entity_pool",
    "save_entity_pool",
]


def build_fictional_entity_replacement_pool_generation_prompt(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    theme: str,
    feedback: str | None = None,
    forbidden_values: set[str] | None = None,
    accepted_pool: dict[str, Any] | None = None,
    shortage_targets: dict[str, dict[str, tuple[int, int]]] | None = None,
) -> tuple[str, str]:
    """Build the provider prompts from the explicit per-reference pool targets."""
    return _build_pool_generation_prompt_impl(
        document,
        required_entities,
        theme=theme,
        target_plan_builder=build_fictional_entity_replacement_pool_target_plan,
        feedback=feedback,
        forbidden_values=forbidden_values,
        accepted_pool=accepted_pool,
        shortage_targets=shortage_targets,
    )


def _pool_bucket_shortages(
    pool: dict[str, list[dict[str, str]]],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, tuple[int, int]]:
    """Return bucket shortages using the currently bound target-plan function."""
    return _pool_bucket_shortages_impl(
        pool,
        required_entities,
        target_plan_builder=build_fictional_entity_replacement_pool_target_plan,
    )


def _reference_pool_shortages(
    pool: dict[str, Any],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, dict[str, tuple[int, int]]]:
    target_plan = build_fictional_entity_replacement_pool_target_plan(required_entities)
    reference_targets = getattr(target_plan, "reference_targets", {}) or {}
    if not reference_targets:
        return {}
    reference_pools = pool.get("_reference_pools", {}) if isinstance(pool, dict) else {}
    shortages: dict[str, dict[str, tuple[int, int]]] = {}
    for bucket, bucket_targets in reference_targets.items():
        bucket_refs = reference_pools.get(bucket, {}) if isinstance(reference_pools, dict) else {}
        bucket_shortages: dict[str, tuple[int, int]] = {}
        for entity_id, target_count in bucket_targets.items():
            ref_payload = bucket_refs.get(entity_id, {}) if isinstance(bucket_refs, dict) else {}
            actual_count = int(ref_payload.get("count", 0)) if isinstance(ref_payload, dict) else 0
            if actual_count < target_count:
                bucket_shortages[entity_id] = (actual_count, int(target_count))
        if bucket_shortages:
            shortages[bucket] = bucket_shortages
    return shortages


def generate_fictional_entity_replacement_pool(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    theme: str,
    config: FictionalEntityReplacementPoolGenerationConfiguration,
    allow_incomplete: bool = False,
    existing_pool: dict[str, Any] | None = None,
    top_up_existing: bool = False,
) -> dict[str, Any]:
    """Generate a normalized pool and validate it before returning or saving."""
    generation_timestamp = _generation_timestamp_utc()
    manual_required_entities = _augment_required_entities_with_rule_attrs(
        document,
        _manual_named_required_entities(required_entities),
    )
    if existing_pool:
        if isinstance(existing_pool, dict) and isinstance(existing_pool.get("_reference_pools"), dict):
            accumulated_pool = _merge_normalized_pools(None, existing_pool)
        else:
            accumulated_pool = _normalize_pool_dict(existing_pool, manual_required_entities)
    else:
        accumulated_pool = _empty_normalized_pool()
    if top_up_existing and existing_pool:
        existing_banned_hits = {
            *_pool_factual_literal_hits(document, manual_required_entities, accumulated_pool),
            *_pool_banned_suffix_hits(accumulated_pool),
        }
        if config.validate_against_wikipedia:
            existing_banned_hits.update(_pool_wikipedia_hits(accumulated_pool))
        if existing_banned_hits:
            accumulated_pool = _prune_generation_unit_variants(
                accumulated_pool,
                manual_required_entities,
                set(existing_banned_hits),
            )
    existing_metadata = (
        dict(existing_pool.get("_metadata", {}))
        if isinstance(existing_pool, dict) and isinstance(existing_pool.get("_metadata"), dict)
        else {}
    )
    accumulated_pool["_metadata"] = {
        **existing_metadata,
        "schema_version": 2,
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "candidates_per_reference": build_fictional_entity_replacement_pool_target_plan(
            required_entities
        ).candidates_per_required_entity,
        "document_id": document.document_id,
        "document_theme": theme,
        "complete": False,
    }
    attempt_counter = 0

    for generation_unit in _generation_units(document, manual_required_entities):
        if top_up_existing:
            unit_reference_shortages = _reference_pool_shortages(accumulated_pool, generation_unit)
            unit_support_shortages = _pool_support_shortages(accumulated_pool, generation_unit)
            if not unit_reference_shortages and not unit_support_shortages:
                continue
        else:
            unit_reference_shortages = {}
        feedback: str | None = None
        last_error: Exception | None = None
        shortage_targets: dict[str, dict[str, tuple[int, int]]] | None = (
            unit_reference_shortages if top_up_existing and unit_reference_shortages else None
        )
        forbidden_values: set[str] = set(_pool_factual_literals(document, generation_unit))

        for _unit_attempt in range(config.max_attempts):
            try:
                system_prompt, user_prompt = build_fictional_entity_replacement_pool_generation_prompt(
                    document,
                    generation_unit,
                    theme=theme,
                    feedback=feedback,
                    forbidden_values=forbidden_values,
                    accepted_pool=accumulated_pool,
                    shortage_targets=shortage_targets,
                )
                attempt_seed = None if config.seed is None else config.seed + attempt_counter
                attempt_counter += 1
                response = generate_text(
                    TextGenerationRequest(
                        provider=config.provider,
                        model=config.model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        seed=attempt_seed,
                    )
                )
                parsed = _extract_mapping_payload(response.text)
                normalized, hits = _finalize_pool_candidate(
                    parsed,
                    generation_unit,
                    validate_against_wikipedia=config.validate_against_wikipedia,
                )
                factual_hits = _pool_factual_literal_hits(document, generation_unit, normalized)
                suffix_hits = _pool_banned_suffix_hits(normalized)
                if factual_hits:
                    hits = sorted({*hits, *factual_hits})
                if suffix_hits:
                    hits = sorted({*hits, *suffix_hits})
                if hits:
                    forbidden_values.update(hits)
                    pruned_normalized = _prune_generation_unit_variants(normalized, generation_unit, set(hits))
                    pruned_candidate_pool = _merge_normalized_pools(accumulated_pool, pruned_normalized)
                    pruned_alignment_failures = _generation_unit_alignment_failures(
                        document,
                        generation_unit,
                        pruned_candidate_pool,
                    )
                    if not pruned_alignment_failures:
                        accumulated_pool = pruned_candidate_pool
                    short_hits = ", ".join(hits[:12])
                    reference_shortages = _reference_pool_shortages(accumulated_pool, generation_unit)
                    shortage_text = (
                        f" Reference shortages after keeping the clean Claude variants: "
                        f"{'; '.join(_reference_shortage_messages(reference_shortages)[:8])}."
                        if reference_shortages
                        else ""
                    )
                    last_error = ValueError(
                        "Pool candidate reused factual or Wikipedia-backed values after validation: " + short_hits
                    )
                    rendered_forbidden = ", ".join(sorted(forbidden_values)[:24])
                    feedback = (
                        "These values still match Wikipedia or reuse factual document literals and must be replaced by "
                        f"fully fictional alternatives: {short_hits}. Do not replace them with other ordinary "
                        "real-world names. Use clearly invented but pronounceable names that do not look like "
                        "standard attested names. "
                        f"Previously rejected exact values are banned for the rest of this run: {rendered_forbidden}."
                        f"{shortage_text}"
                    )
                    logger.warning(
                        "Entity-pool generation attempt hit banned values for %s: %s",
                        document.document_id,
                        short_hits,
                    )
                    continue

                candidate_pool = _merge_normalized_pools(accumulated_pool, normalized)
                alignment_failures = _generation_unit_alignment_failures(document, generation_unit, candidate_pool)
                if alignment_failures:
                    last_error = ValueError(
                        "Pool variant alignment failed after normalization: " + "; ".join(alignment_failures[:6])
                    )
                    feedback = (
                        "Previous attempt produced linked variants that were not internally coherent. "
                        "Regenerate the affected references so that each shared variant index satisfies the relevant rules, "
                        "and make sure every place keeps a demonym or nationality adjective that matches the corresponding "
                        f"fictional place. Failures: {'; '.join(alignment_failures[:6])}."
                    )
                    logger.warning(
                        "Entity-pool generation attempt for %s failed alignment checks: %s",
                        document.document_id,
                        "; ".join(alignment_failures[:4]),
                    )
                    continue

                reference_shortages = _reference_pool_shortages(candidate_pool, generation_unit)
                if reference_shortages:
                    accumulated_pool = candidate_pool
                    last_error = ValueError(
                        "Pool still lacks enough per-reference support after normalization: "
                        + "; ".join(_reference_shortage_messages(reference_shortages)[:8])
                    )
                    feedback = (
                        "Previous attempt still did not provide enough distinct Claude variants for some entity references. "
                        "Generate additional fresh variants without repeating earlier accepted ones. "
                        f"Reference shortages: {'; '.join(_reference_shortage_messages(reference_shortages)[:8])}."
                    )
                    shortage_targets = reference_shortages
                    logger.warning(
                        "Entity-pool generation attempt for %s still lacks per-reference support: %s",
                        document.document_id,
                        "; ".join(_reference_shortage_messages(reference_shortages)[:4]),
                    )
                    continue

                support_shortages = _pool_support_shortages(candidate_pool, generation_unit)
                if not support_shortages:
                    accumulated_pool = candidate_pool
                    break

                accumulated_pool = candidate_pool
                last_error = ValueError(
                    "Pool still lacks enough support after normalization: " + "; ".join(support_shortages[:8])
                )
                feedback = (
                    "Previous attempt still did not provide enough distinct Claude entities for some required groups. "
                    "Generate additional fresh variants without repeating earlier accepted ones. "
                    f"Support shortages: {'; '.join(support_shortages[:8])}."
                )
                logger.warning(
                    "Entity-pool generation attempt for %s still lacks support after merge: %s",
                    document.document_id,
                    "; ".join(support_shortages[:4]),
                )
            except Exception as exc:
                last_error = exc
                if _is_non_retryable_provider_error(exc):
                    raise RuntimeError(
                        f"Claude entity-pool generation failed for {document.document_id}; "
                        f"non-retryable provider error: {exc}"
                    ) from exc
                feedback = f"Previous attempt failed: {exc}"
                logger.warning("Entity-pool generation attempt failed for %s: %s", document.document_id, exc)
        else:
            if last_error is None:
                last_error = RuntimeError("unknown pool generation failure")
            if allow_incomplete:
                accumulated_pool["_metadata"]["generated_at"] = generation_timestamp
                accumulated_pool["_metadata"]["failure_reason"] = str(last_error)
                return accumulated_pool
            raise RuntimeError(
                f"Claude entity-pool generation failed for {document.document_id}; no local fallback is available: {last_error}"
            ) from last_error

    reference_shortages = _reference_pool_shortages(accumulated_pool, manual_required_entities)
    if reference_shortages:
        if allow_incomplete:
            accumulated_pool["_metadata"]["generated_at"] = generation_timestamp
            accumulated_pool["_metadata"]["failure_reason"] = "per-reference shortages: " + "; ".join(
                _reference_shortage_messages(reference_shortages)[:12]
            )
            return accumulated_pool
        raise RuntimeError(
            "Claude entity-pool generation failed for "
            f"{document.document_id}; no local fallback is available: "
            + "; ".join(_reference_shortage_messages(reference_shortages)[:12])
        )
    support_shortages = _pool_support_shortages(accumulated_pool, manual_required_entities)
    if support_shortages:
        if allow_incomplete:
            accumulated_pool["_metadata"]["generated_at"] = generation_timestamp
            accumulated_pool["_metadata"]["failure_reason"] = "support shortages: " + "; ".join(support_shortages[:12])
            return accumulated_pool
        raise RuntimeError(
            "Claude entity-pool generation failed for "
            f"{document.document_id}; no local fallback is available: " + "; ".join(support_shortages[:12])
        )
    accumulated_pool["_metadata"]["generated_at"] = generation_timestamp
    accumulated_pool["_metadata"]["complete"] = True
    return accumulated_pool


def resolve_or_generate_entity_pool(
    document: AnnotatedDocument,
    *,
    theme: str,
    config: FictionalEntityReplacementPoolGenerationConfiguration,
    persist_generated_pool: bool,
) -> tuple[dict[str, Any], str, Path | None]:
    """Return an existing pool or generate a new one when none is available."""
    required_entities = extract_required_entities(document, include_questions=True)
    existing_pool, pool_path = load_saved_entity_pool(theme, document.document_id)
    if existing_pool is not None:
        return _validate_existing_pool(existing_pool, document, required_entities), "existing_pool", pool_path

    generated_pool = generate_fictional_entity_replacement_pool(
        document,
        required_entities,
        theme=theme,
        config=config,
    )
    if not persist_generated_pool:
        return generated_pool, "generated_pool", None

    pool_path = generated_entity_pool_path(theme, document.document_id)
    save_entity_pool(generated_pool, pool_path)
    return generated_pool, "generated_pool", pool_path


def generate_fictional_entity_replacement_pool_for_template(
    template_path: Path,
    *,
    config: FictionalEntityReplacementPoolGenerationConfiguration,
    overwrite: bool = False,
    allow_incomplete_save: bool = False,
    top_up_existing: bool = False,
) -> Path:
    """Generate and save a document-specific fictional entity pool YAML file."""
    if overwrite and top_up_existing:
        raise ValueError("overwrite and top_up_existing are mutually exclusive generation modes")
    theme, document_id = resolve_template_identity(template_path)
    output_path = generated_entity_pool_path(theme, document_id)
    output_existed_before = output_path.exists()
    if top_up_existing and not output_existed_before:
        raise FileNotFoundError(f"Shortage-only top-up requires an existing entity pool: {output_path}")
    existing_pool_sha256 = _file_sha256(output_path) if output_existed_before else None
    document = load_annotated_document(str(template_path), validate_question_scope=False)
    required_entities = extract_required_entities(document, include_questions=True)
    existing_pool: dict[str, Any] | None = None
    should_top_up_existing = False
    if output_existed_before:
        existing_pool = load_entity_pool(str(output_path)) or {}
        if not overwrite:
            try:
                _validate_existing_pool(existing_pool, document, required_entities)
            except ValueError:
                if not top_up_existing:
                    raise
                should_top_up_existing = True
            else:
                return output_path

    pool = generate_fictional_entity_replacement_pool(
        document,
        required_entities,
        theme=theme,
        config=config,
        allow_incomplete=allow_incomplete_save,
        existing_pool=existing_pool if should_top_up_existing else None,
        top_up_existing=should_top_up_existing,
    )
    return save_entity_pool(
        pool,
        output_path,
        overwrite=output_existed_before,
        expected_existing_sha256=existing_pool_sha256,
    )
