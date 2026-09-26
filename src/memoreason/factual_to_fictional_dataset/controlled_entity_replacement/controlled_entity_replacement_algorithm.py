"""Paper-facing implementation of the fictional-generation algorithm.

This module is the publication reference for Algorithm 1 in
``docs/NeurIPS 2026/inputs/controlled_entity_replacement_algorithm.tex``.
The top-level :func:`generate_fictional_document_variants` function is intentionally written in
the same template-loop / variant-loop structure as the paper so it can be
pointed to directly when describing the benchmark construction procedure.
"""

from __future__ import annotations

import logging
from pathlib import Path

from memoreason.benchmark_definition.annotation_runtime import find_rule_sanity_errors

from .fictional_document_variant_generation.fictional_document_variant_generation_stages import (
    build_controlled_entity_replacement_context,
    generate_named_entities,
    generate_numerical_entities,
    replace_factual_entities,
    sample_named_entities,
)
from .fictional_document_variant_generation.fictional_document_variant_data_contracts import (
    DEBUG_SAMPLING,
    MAX_VARIANT_GENERATION_RETRIES,
    ControlledEntityReplacementContext,
    ReviewedTemplateFictionalGenerationInput,
    ReviewedTemplateFictionalGenerationResult,
    FictionalDocumentVariantRequest,
    NamedEntitySample,
    NumericalEntitySample,
)
from .generation_exceptions import StrictInterVariantUniquenessInfeasible
from .named_entity_uniqueness import named_entity_uniqueness_signature
from .number_uniqueness import number_entity_uniqueness_value

logger = logging.getLogger(__name__)


def _has_recorded_values(values_by_id: dict[str, set]) -> bool:
    return any(bool(values) for values in values_by_id.values())


def _has_recorded_temporal_values(values_by_id: dict[str, dict[str, set]]) -> bool:
    return any(any(bool(values) for values in values_by_attr.values()) for values_by_attr in values_by_id.values())


def _record_used_numerical_values(
    numerical_entity_sample: NumericalEntitySample,
    *,
    used_number_values_by_id: dict[str, set[int | float]],
    used_temporal_years_by_id: dict[str, set[int]],
    used_temporal_values_by_id: dict[str, dict[str, set]],
) -> None:
    for number_id, number_entity in numerical_entity_sample.entities.numbers.items():
        number_value = number_entity_uniqueness_value(number_entity)
        if number_value is not None:
            used_number_values_by_id.setdefault(number_id, set()).add(number_value)
    for temporal_id, temporal_entity in numerical_entity_sample.entities.temporals.items():
        temporal_year = getattr(temporal_entity, "year", None)
        if temporal_year is not None:
            used_temporal_years_by_id.setdefault(temporal_id, set()).add(int(temporal_year))
        for attr in ("year", "month", "day", "day_of_month", "timestamp"):
            value = getattr(temporal_entity, attr, None)
            if value in (None, ""):
                continue
            used_temporal_values_by_id.setdefault(temporal_id, {}).setdefault(attr, set()).add(value)


def _record_used_named_values(
    named_entity_sample: NamedEntitySample,
    *,
    used_named_values_by_id: dict[str, set[str]],
) -> None:
    for entity_type, specs in named_entity_sample.fictional_requirements.items():
        if entity_type in {"number", "temporal"}:
            continue
        collection = named_entity_sample.entities.get_collection(entity_type)
        for entity_id, _required_attrs in specs:
            entity = collection.get(entity_id)
            signature = named_entity_uniqueness_signature(entity_type, entity)
            if signature is not None:
                used_named_values_by_id.setdefault(str(entity_id), set()).add(signature)


def _attempt_variant_generation(
    *,
    template_input: ReviewedTemplateFictionalGenerationInput,
    context: ControlledEntityReplacementContext,
    document_id: str,
    reusable_cache: dict,
    named_entities: dict,
    variant_request: FictionalDocumentVariantRequest,
    used_named_values_by_id: dict[str, set[str]],
    used_number_values_by_id: dict[str, set[int | float]],
    used_temporal_years_by_id: dict[str, set[int]],
    used_temporal_values_by_id: dict[str, dict[str, set]],
    allow_previous_numtemp_reuse: bool,
    allow_factual_numtemp_values: bool,
    relaxed_reuse_audit: list[dict],
) -> tuple[Path, int] | None:
    """Try one requested variant without closing over either algorithm loop."""
    attempt_limit = template_input.max_generation_retries
    if not allow_previous_numtemp_reuse and (
        _has_recorded_values(used_number_values_by_id)
        or _has_recorded_values(used_temporal_years_by_id)
        or _has_recorded_temporal_values(used_temporal_values_by_id)
    ):
        # A single strict call already performs an internal multi-attempt solve.
        # If that fails, fall through to the scoped reuse path instead of
        # burning extra seeds on a saturated domain.
        attempt_limit = min(attempt_limit, 1)
    for attempt_index in range(attempt_limit):
        attempt_seed = variant_request.base_seed + (attempt_index * 10_000)
        shifted_reference_variant_index = (
            None
            if variant_request.reference_variant_index is None
            else variant_request.reference_variant_index
            + (
                attempt_index
                * (
                    variant_request.reference_variant_count
                    if variant_request.reference_variant_count is not None
                    else 1
                )
            )
        )

        # e_named_{i,j} ~ Uniform(e_named_i)
        named_entity_sample: NamedEntitySample | None = sample_named_entities(
            context=context,
            named_entities=named_entities,
            replacement_proportion=template_input.replacement_proportion,
            replace_mode=template_input.replace_mode,
            version_seed=attempt_seed,
            eligible_cache=reusable_cache,
            reference_variant_index=shifted_reference_variant_index,
            reference_variant_count=variant_request.reference_variant_count,
            used_named_values_by_id=used_named_values_by_id,
        )
        if named_entity_sample is None:
            continue

        # e_numerical <- MixedIntegerLinearProgramming(r_i)
        try:
            numerical_entity_sample: NumericalEntitySample | None = generate_numerical_entities(
                context=context,
                named_entity_sample=named_entity_sample,
                named_entities=named_entities,
                version_seed=attempt_seed,
                eligible_cache=reusable_cache,
                reference_variant_index=shifted_reference_variant_index,
                reference_variant_count=variant_request.reference_variant_count,
                used_number_values_by_id=used_number_values_by_id,
                used_temporal_years_by_id=used_temporal_years_by_id,
                used_temporal_values_by_id=used_temporal_values_by_id,
                prior_relaxed_intervariant_reuse_audit=list(
                    template_input.prior_relaxed_intervariant_reuse_audit
                ),
                allow_relaxed_intervariant_number_reuse=allow_previous_numtemp_reuse,
                allow_factual_numtemp_values=allow_factual_numtemp_values,
            )
        except StrictInterVariantUniquenessInfeasible:
            if allow_previous_numtemp_reuse:
                return None
            raise
        if numerical_entity_sample is None:
            continue

        # c_{i,j}^fictional <- e_numerical U e_named_{i,j}
        # (d_{i,j}^fictional, q_{i,j}^fictional, a_{i,j}^fictional)
        # <- Replace((d_i^template, q_i^template, a_i^template), c_i^factual, c_{i,j}^fictional)
        generated_variant_path = replace_factual_entities(
            context=context,
            named_entity_sample=named_entity_sample,
            numerical_entity_sample=numerical_entity_sample,
            output_path=variant_request.output_path,
            document_id=document_id,
            replacement_proportion=template_input.replacement_proportion,
            replace_mode=template_input.replace_mode,
        )
        if generated_variant_path is None:
            continue
        for item in numerical_entity_sample.relaxed_intervariant_reuse_audit:
            relaxed_reuse_audit.append(
                {
                    **dict(item),
                    "variant_index": (
                        int(variant_request.reference_variant_index) + 1
                        if variant_request.reference_variant_index is not None
                        else None
                    ),
                    "variant_id": (
                        f"v{int(variant_request.reference_variant_index) + 1:02d}"
                        if variant_request.reference_variant_index is not None
                        else ""
                    ),
                    "generation_seed": int(attempt_seed),
                }
            )
        _record_used_named_values(
            named_entity_sample,
            used_named_values_by_id=used_named_values_by_id,
        )
        _record_used_numerical_values(
            numerical_entity_sample,
            used_number_values_by_id=used_number_values_by_id,
            used_temporal_years_by_id=used_temporal_years_by_id,
            used_temporal_values_by_id=used_temporal_values_by_id,
        )
        if attempt_index > 0:
            logger.warning(
                "Variant generation for %s succeeded after %d retry(ies).",
                document_id,
                attempt_index,
            )
        return generated_variant_path, attempt_seed
    return None


def generate_fictional_document_variants(
    template_inputs: list[ReviewedTemplateFictionalGenerationInput],
) -> list[ReviewedTemplateFictionalGenerationResult]:
    """Implement the paper's ``FictionalGeneration(D_template, K)`` algorithm."""
    fictional_dataset: list[ReviewedTemplateFictionalGenerationResult] = []  # D^fictional <- {}

    for template_input in template_inputs:
        context = build_controlled_entity_replacement_context(template_input.template_document)
        sanity_errors = find_rule_sanity_errors(context.generation_document.rules)
        if sanity_errors:
            raise ValueError("\n".join(sanity_errors))

        document_id = template_input.document_id or context.source_document.document_id
        fictional_variants_for_template: list[tuple[Path, int]] = []  # S_i^fictional <- {}
        reusable_cache = template_input.eligible_cache if template_input.eligible_cache is not None else {}
        used_relaxed_intervariant_reuse = False
        relaxed_intervariant_reuse_audit: list[dict] = []
        used_named_values_by_id = (
            template_input.used_named_values_by_id if template_input.used_named_values_by_id is not None else {}
        )
        used_number_values_by_id = (
            template_input.used_number_values_by_id if template_input.used_number_values_by_id is not None else {}
        )
        used_temporal_years_by_id = (
            template_input.used_temporal_years_by_id if template_input.used_temporal_years_by_id is not None else {}
        )
        used_temporal_values_by_id = (
            template_input.used_temporal_values_by_id if template_input.used_temporal_values_by_id is not None else {}
        )

        # e_named_i <- GenerateNamedEntities(d_i^template)
        named_entities = (
            generate_named_entities(
                context=context,
                entity_pool=template_input.named_entity_pool,
                seed=template_input.named_entities_seed,
            )
            if template_input.replacement_proportion > 0.0
            else {}
        )

        for variant_request in template_input.variant_requests:
            try:
                generated_variant = _attempt_variant_generation(
                    template_input=template_input,
                    context=context,
                    document_id=document_id,
                    reusable_cache=reusable_cache,
                    named_entities=named_entities,
                    variant_request=variant_request,
                    used_named_values_by_id=used_named_values_by_id,
                    used_number_values_by_id=used_number_values_by_id,
                    used_temporal_years_by_id=used_temporal_years_by_id,
                    used_temporal_values_by_id=used_temporal_values_by_id,
                    allow_previous_numtemp_reuse=False,
                    allow_factual_numtemp_values=False,
                    relaxed_reuse_audit=relaxed_intervariant_reuse_audit,
                )
            except StrictInterVariantUniquenessInfeasible:
                generated_variant = None
            if generated_variant is None:
                logger.warning(
                    "Strict fictional generation failed for %s; retrying while allowing prior fictional "
                    "or factual numeric/temporal values.",
                    document_id,
                )
                generated_variant = _attempt_variant_generation(
                    template_input=template_input,
                    context=context,
                    document_id=document_id,
                    reusable_cache=reusable_cache,
                    named_entities=named_entities,
                    variant_request=variant_request,
                    used_named_values_by_id=used_named_values_by_id,
                    used_number_values_by_id=used_number_values_by_id,
                    used_temporal_years_by_id=used_temporal_years_by_id,
                    used_temporal_values_by_id=used_temporal_values_by_id,
                    allow_previous_numtemp_reuse=True,
                    allow_factual_numtemp_values=False,
                    relaxed_reuse_audit=relaxed_intervariant_reuse_audit,
                )
                if generated_variant is not None:
                    used_relaxed_intervariant_reuse = True

            if generated_variant is None:
                logger.warning(
                    "Fictional reuse fallback failed for %s; allowing factual numeric/temporal values.",
                    document_id,
                )
                generated_variant = _attempt_variant_generation(
                    template_input=template_input,
                    context=context,
                    document_id=document_id,
                    reusable_cache=reusable_cache,
                    named_entities=named_entities,
                    variant_request=variant_request,
                    used_named_values_by_id=used_named_values_by_id,
                    used_number_values_by_id=used_number_values_by_id,
                    used_temporal_years_by_id=used_temporal_years_by_id,
                    used_temporal_values_by_id=used_temporal_values_by_id,
                    allow_previous_numtemp_reuse=True,
                    allow_factual_numtemp_values=True,
                    relaxed_reuse_audit=relaxed_intervariant_reuse_audit,
                )
                if generated_variant is not None:
                    used_relaxed_intervariant_reuse = True

            if generated_variant is None:
                raise RuntimeError(
                    f"Failed to generate {document_id} (p={template_input.replacement_proportion:.1f}) "
                    f"after {template_input.max_generation_retries} attempts."
                )
            fictional_variants_for_template.append(generated_variant)

        fictional_dataset.append(
            ReviewedTemplateFictionalGenerationResult(
                document_id=document_id,
                context=context,
                named_entities=named_entities,
                generated_variants=tuple(fictional_variants_for_template),
                used_relaxed_intervariant_reuse=used_relaxed_intervariant_reuse,
                relaxed_intervariant_reuse_audit=tuple(relaxed_intervariant_reuse_audit),
            )
        )

    return fictional_dataset


__all__ = [
    "DEBUG_SAMPLING",
    "MAX_VARIANT_GENERATION_RETRIES",
    "ControlledEntityReplacementContext",
    "FictionalDocumentVariantRequest",
    "NamedEntitySample",
    "NumericalEntitySample",
    "ReviewedTemplateFictionalGenerationInput",
    "ReviewedTemplateFictionalGenerationResult",
    "build_controlled_entity_replacement_context",
    "generate_fictional_document_variants",
    "generate_named_entities",
    "generate_numerical_entities",
    "replace_factual_entities",
    "sample_named_entities",
]
