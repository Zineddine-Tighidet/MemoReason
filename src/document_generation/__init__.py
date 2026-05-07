"""Fictional document generation and constrained entity replacement."""

from .fictional_document_renderer import FictionalDocumentRenderer
from .fictional_entity_sampler import FictionalEntitySampler
from .fictional_generation_algorithm import (
    FictionalGenerationContext,
    FictionalGenerationTemplateInput,
    FictionalGenerationTemplateResult,
    FictionalVariantRequest,
    NamedEntitySample,
    NumericalEntitySample,
    build_fictional_generation_context,
    fictional_generation,
    generate_numerical_entities,
    generate_named_entities,
    replace_factual_entities,
    sample_named_entities,
)
from .number_temporal_generator import NumberTemporalGenerator
from .fictional_document_generation import (
    create_entity_pool_template,
    generate_fictional_documents,
    run_fictional_document_generation,
)

__all__ = [
    "FictionalDocumentRenderer",
    "FictionalEntitySampler",
    "FictionalGenerationContext",
    "FictionalGenerationTemplateInput",
    "FictionalGenerationTemplateResult",
    "FictionalVariantRequest",
    "NamedEntitySample",
    "NumberTemporalGenerator",
    "NumericalEntitySample",
    "build_fictional_generation_context",
    "create_entity_pool_template",
    "fictional_generation",
    "generate_fictional_documents",
    "generate_named_entities",
    "generate_numerical_entities",
    "replace_factual_entities",
    "run_fictional_document_generation",
    "sample_named_entities",
]
