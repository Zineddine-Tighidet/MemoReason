"""Sampling of manual entities and auto-generated numeric/temporal entities."""

from __future__ import annotations

from typing import ClassVar

from src.core.document_schema import EntityCollection
from src.core.organization_types import CANONICAL_ORGANIZATION_TYPES

from .fictional_entity_sampler_common import DEBUG_SAMPLING, FictionalEntitySamplerCommonMixin
from .fictional_entity_sampler_named import FictionalEntitySamplerNamedMixin
from .fictional_entity_sampler_numerical import FictionalEntitySamplerNumericalMixin
from .generation_requirements import extract_decade_year_temporal_ids, extract_required_entities


class FictionalEntitySampler(
    FictionalEntitySamplerNamedMixin,
    FictionalEntitySamplerNumericalMixin,
    FictionalEntitySamplerCommonMixin,
):
    """Sample entities from manual pools and auto-generate numbers/temporals."""

    _MANUAL_ENTITY_TYPES: ClassVar[tuple[str, ...]] = (
        "person",
        "place",
        "event",
        *CANONICAL_ORGANIZATION_TYPES,
        "award",
        "legal",
        "product",
    )
    _AUTO_ENTITY_TYPES: ClassVar[tuple[str, ...]] = ("number", "temporal")

    _PERSON_DIFF_EXEMPT_ATTRS = frozenset(
        {
            "gender",
            "subj_pronoun",
            "obj_pronoun",
            "poss_det_pronoun",
            "poss_pro_pronoun",
            "refl_pronoun",
            "honorific",
            "relationship",
            "relationships",
        }
    )
    def sample_fictional_entities(
        self,
        required_entities: dict[str, list[tuple[str, list[str]]]],
        rules: list[str],
        max_attempts: int = 10,
        decade_year_temporal_ids: set[str] | None = None,
    ) -> EntityCollection | None:
        """Sample one coherent fictional entity assignment for a template variant."""
        decade_year_temporal_ids = set(decade_year_temporal_ids or set())
        named_entities = self.sample_named_entities(required_entities, rules)
        if named_entities is None:
            return None
        numerical_entities = self.generate_numerical_entities(
            required_entities=required_entities,
            rules=rules,
            named_entities=named_entities,
            max_attempts=max_attempts,
            decade_year_temporal_ids=decade_year_temporal_ids,
        )
        if numerical_entities is None:
            return None
        named_entities.merge_from(numerical_entities)
        return named_entities

    extract_required_entities = staticmethod(extract_required_entities)
    extract_decade_year_temporal_ids = staticmethod(extract_decade_year_temporal_ids)


__all__ = ["DEBUG_SAMPLING", "FictionalEntitySampler"]
