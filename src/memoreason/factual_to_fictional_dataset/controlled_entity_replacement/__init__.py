"""Fictional document generation and constrained entity replacement."""

from .fictional_document_renderer import FictionalDocumentRenderer
from .fictional_entity_sampler import FictionalEntitySampler
from .controlled_entity_replacement_algorithm import (
    ControlledEntityReplacementContext,
    ReviewedTemplateFictionalGenerationInput,
    ReviewedTemplateFictionalGenerationResult,
    FictionalDocumentVariantRequest,
    NamedEntitySample,
    NumericalEntitySample,
    build_controlled_entity_replacement_context,
    generate_fictional_document_variants,
    generate_numerical_entities,
    generate_named_entities,
    replace_factual_entities,
    sample_named_entities,
)
from .number_temporal_generator import NumberTemporalGenerator

__all__ = [
    "FictionalDocumentRenderer",
    "FictionalEntitySampler",
    "ControlledEntityReplacementContext",
    "ReviewedTemplateFictionalGenerationInput",
    "ReviewedTemplateFictionalGenerationResult",
    "FictionalDocumentVariantRequest",
    "NamedEntitySample",
    "NumberTemporalGenerator",
    "NumericalEntitySample",
    "build_controlled_entity_replacement_context",
    "generate_fictional_document_variants",
    "generate_named_entities",
    "generate_numerical_entities",
    "replace_factual_entities",
    "sample_named_entities",
]
