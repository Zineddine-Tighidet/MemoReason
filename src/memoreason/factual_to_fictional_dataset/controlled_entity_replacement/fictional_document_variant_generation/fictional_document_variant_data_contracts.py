"""Data contracts for controlled generation of fictional document variants."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import REPLACE_MODE_ALL
from memoreason.benchmark_definition.document_schema import AnnotatedDocument, EntityCollection

from ..replacement_planning import EntityCollectionsByPluralType, ReplacementLayout, RequiredEntityMap

DEBUG_SAMPLING = os.environ.get("DEBUG_SAMPLING", "").lower() in {"1", "true", "yes"}
MAX_SAMPLING_ATTEMPTS_PER_VARIANT = 1
MAX_VARIANT_GENERATION_RETRIES = int(os.environ.get("FICTIONAL_MAX_VARIANT_GENERATION_RETRIES", "5"))

_ENTITY_TYPE_TO_PLURAL_KEY = {
    "person": "persons",
    "place": "places",
    "event": "events",
    "award": "awards",
    "legal": "legals",
    "product": "products",
    "organization": "organizations",
    "temporal": "temporals",
    "number": "numbers",
}


@dataclass(frozen=True)
class ControlledEntityReplacementContext:
    """Static inputs derived once from one reviewed template document."""

    source_document: AnnotatedDocument
    generation_document: AnnotatedDocument
    required_entities: RequiredEntityMap
    factual_entities_full: EntityCollection
    entity_types: EntityCollectionsByPluralType
    dropped_rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class NamedEntitySample:
    """Named-entity assignment sampled from the template-level fictional pool."""

    entities: EntityCollection
    replacement_layout: ReplacementLayout
    fictional_requirements: RequiredEntityMap
    decade_year_temporal_ids: frozenset[str]


@dataclass(frozen=True)
class NumericalEntitySample:
    """Generated numerical entities for one fictional variant."""

    entities: EntityCollection
    relaxed_intervariant_reuse_audit: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class FictionalDocumentVariantRequest:
    """One requested fictional variant output in the paper's inner loop."""

    base_seed: int
    output_path: Path
    reference_variant_index: int | None = None
    reference_variant_count: int | None = None


@dataclass(frozen=True)
class ReviewedTemplateFictionalGenerationInput:
    """One template-level input item for the paper's fictional-generation loop."""

    template_document: AnnotatedDocument
    named_entity_pool: dict[str, Any]
    variant_requests: tuple[FictionalDocumentVariantRequest, ...]
    replacement_proportion: float
    document_id: str | None = None
    replace_mode: str = REPLACE_MODE_ALL
    named_entities_seed: int | None = None
    eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None = None
    used_named_values_by_id: dict[str, set[str]] | None = None
    used_number_values_by_id: dict[str, set[int | float]] | None = None
    used_temporal_years_by_id: dict[str, set[int]] | None = None
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None
    prior_relaxed_intervariant_reuse_audit: tuple[dict[str, Any], ...] = ()
    force_allow_previous_numtemp_reuse: bool = False
    max_generation_retries: int = MAX_VARIANT_GENERATION_RETRIES


@dataclass(frozen=True)
class ReviewedTemplateFictionalGenerationResult:
    """One template-level output item produced by the paper algorithm."""

    document_id: str
    context: ControlledEntityReplacementContext
    named_entities: dict[str, Any]
    generated_variants: tuple[tuple[Path, int], ...]
    used_relaxed_intervariant_reuse: bool = False
    relaxed_intervariant_reuse_audit: tuple[dict[str, Any], ...] = ()


__all__ = [
    "DEBUG_SAMPLING",
    "MAX_SAMPLING_ATTEMPTS_PER_VARIANT",
    "MAX_VARIANT_GENERATION_RETRIES",
    "ControlledEntityReplacementContext",
    "FictionalDocumentVariantRequest",
    "NamedEntitySample",
    "NumericalEntitySample",
    "ReviewedTemplateFictionalGenerationInput",
    "ReviewedTemplateFictionalGenerationResult",
]
