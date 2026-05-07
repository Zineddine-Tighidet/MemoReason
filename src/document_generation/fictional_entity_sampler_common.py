"""Shared helpers for fictional entity sampling."""

from __future__ import annotations

import logging
import os
import random
import re
from typing import Any

from src.core.annotation_runtime import find_entity_refs
from src.core.century_expressions import has_century_function
from src.core.document_schema import EntityCollection
from src.core.implicit_numeric_rules import implicit_rule_bounds_lookup
from src.core.organization_types import CANONICAL_ORGANIZATION_TYPES, organization_attribute_value

from .manual_entity_constraints import sample_manual_entities_with_constraints

logger = logging.getLogger(__name__)
DEBUG_SAMPLING = os.environ.get("DEBUG_SAMPLING", "").lower() in {"1", "true", "yes"}
_EVENT_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "debate": ("debate",),
    "nomination": ("nomination", "nominat", "primary", "primaries"),
    "election": ("election", "electoral", "ballot", "vote"),
    "pandemic": ("pandemic", "outbreak", "epidemic", "contagion", "plague"),
    "war": ("war", "conflict", "invasion", "incursion", "siege", "campaign", "offensive", "withdrawal"),
}
_INLINE_NUMBER_REF_PATTERN = re.compile(r"\[[^\]]+;\s*(number_\d+)\.[^\]]+\]")
_PRECEDING_NAMED_REF_PATTERN = re.compile(
    r"\[[^\]]+;\s*(?:product|legal|event)_\d+\.name\]\s*$",
    re.IGNORECASE,
)
_AGE_CONTEXT_PATTERN = re.compile(
    r"(?:\byears?\s+old\b|-\s*years?\s*-?\s*old\b|-\s*year\s*-?\s*old\b|\baged\s*$)",
    re.IGNORECASE,
)
_IDENTIFIER_CONTEXT_PATTERN = re.compile(r"(?:\bNo\.\s*$|\bsite\s+no\.\s*$)", re.IGNORECASE)
_FOLLOWING_IDENTIFIER_CONTEXT_PATTERN = re.compile(
    r"^\s*(?:\[[^\]]+;\s*[^\]]+\]\s*)?"
    r"(?:range|site|station|facility|complex|base|launchpad|launch\s+pad)\b",
    re.IGNORECASE,
)
_PRECEDING_SCALED_QUANTITY_CONTEXT_PATTERN = re.compile(
    r"(?:[$€£¥]\s*|(?:USD|EUR|GBP|dollars?|euros?|pounds?)\s*)$",
    re.IGNORECASE,
)
_FOLLOWING_SCALED_QUANTITY_CONTEXT_PATTERN = re.compile(
    r"^\s*(?:billion|million|trillion|thousand|"
    r"USD|EUR|GBP|dollars?|euros?|pounds?)\b",
    re.IGNORECASE,
)


def _detect_age_like_number_ids(source_text: str | None) -> set[str]:
    """Return number refs that denote ages in the annotated surface text."""
    text = str(source_text or "")
    age_like_ids: set[str] = set()
    for match in _INLINE_NUMBER_REF_PATTERN.finditer(text):
        before = text[max(0, match.start() - 24) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 32)]
        if _AGE_CONTEXT_PATTERN.search(f"{before} {after}"):
            age_like_ids.add(match.group(1))
    return age_like_ids


def _detect_identifier_like_number_ids(source_text: str | None) -> set[str]:
    """Return numeric refs that should not participate in raw-number ordering."""
    text = str(source_text or "")
    identifier_like_ids: set[str] = set()
    for match in _INLINE_NUMBER_REF_PATTERN.finditer(text):
        before = text[max(0, match.start() - 96) : match.start()]
        after = text[match.end() : min(len(text), match.end() + 96)]
        if (
            _PRECEDING_NAMED_REF_PATTERN.search(before)
            or _IDENTIFIER_CONTEXT_PATTERN.search(before)
            or _FOLLOWING_IDENTIFIER_CONTEXT_PATTERN.search(after)
            or _PRECEDING_SCALED_QUANTITY_CONTEXT_PATTERN.search(before)
            or _FOLLOWING_SCALED_QUANTITY_CONTEXT_PATTERN.search(after)
        ):
            identifier_like_ids.add(match.group(1))
    return identifier_like_ids


def detect_ordering_excluded_number_ids(source_text: str | None) -> set[str]:
    """Return number IDs whose local surface makes global numeric ordering unsafe."""
    return _detect_age_like_number_ids(source_text) | _detect_identifier_like_number_ids(source_text)


class FictionalEntitySamplerCommonMixin:
    """Stateful helpers shared by named and numerical sampling stages."""

    def __init__(
        self,
        entity_pool: dict[str, Any],
        seed: int | None = None,
        factual_entities: EntityCollection | None = None,
        eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None = None,
        implicit_rules: list[Any] | None = None,
        reference_variant_index: int | None = None,
        reference_variant_count: int | None = None,
        used_number_values_by_id: dict[str, set[int | float]] | None = None,
        used_temporal_years_by_id: dict[str, set[int]] | None = None,
        used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
        allow_relaxed_intervariant_number_reuse: bool = False,
        source_document_text: str | None = None,
    ) -> None:
        self.entity_pool = entity_pool
        self.seed = seed
        self.factual_entities = factual_entities
        self._eligible_cache = eligible_cache if eligible_cache is not None else {}
        self.implicit_rules = list(implicit_rules or [])
        self.reference_variant_index = reference_variant_index
        self.reference_variant_count = reference_variant_count
        self.used_number_values_by_id = used_number_values_by_id if used_number_values_by_id is not None else {}
        self.used_temporal_years_by_id = used_temporal_years_by_id if used_temporal_years_by_id is not None else {}
        self.used_temporal_values_by_id = (
            used_temporal_values_by_id if used_temporal_values_by_id is not None else {}
        )
        self.allow_relaxed_intervariant_number_reuse = bool(allow_relaxed_intervariant_number_reuse)
        self.age_like_number_ids = _detect_age_like_number_ids(source_document_text)
        self.ordering_excluded_number_ids = detect_ordering_excluded_number_ids(source_document_text)
        self._implicit_age_rules = {
            entity_ref.split(".", 1)[0]: rule
            for entity_ref, rule in implicit_rule_bounds_lookup(self.implicit_rules).items()
            if entity_ref.startswith("person_") and entity_ref.endswith(".age")
        }
        if seed is not None:
            random.seed(seed)

    def _reference_variant_candidates(self, pool_size: int) -> list[int]:
        if self.reference_variant_index is None or pool_size <= 0:
            return []
        base_index = int(self.reference_variant_index)
        stride = int(self.reference_variant_count or 1)
        if stride < 1:
            stride = 1
        start_index = base_index % pool_size
        if stride == 1:
            return [start_index]
        primary = list(range(start_index, pool_size, stride))
        secondary = [index for index in range(pool_size) if index not in primary]
        return primary + secondary

    @staticmethod
    def _strip_rule_comment(rule: str) -> str:
        return rule.split("#", 1)[0].strip()

    @staticmethod
    def _apply_gender_profile(person_entity: Any, gender: str) -> None:
        person_entity.gender = gender
        if gender == "male":
            (
                person_entity.subj_pronoun,
                person_entity.obj_pronoun,
                person_entity.poss_det_pronoun,
                person_entity.poss_pro_pronoun,
                person_entity.refl_pronoun,
            ) = ("he", "him", "his", "his", "himself")
            person_entity.honorific = "Mr"
            return
        if gender == "neutral":
            (
                person_entity.subj_pronoun,
                person_entity.obj_pronoun,
                person_entity.poss_det_pronoun,
                person_entity.poss_pro_pronoun,
                person_entity.refl_pronoun,
            ) = ("they", "them", "their", "theirs", "themself")
            person_entity.honorific = "Mx"
            return
        (
            person_entity.subj_pronoun,
            person_entity.obj_pronoun,
            person_entity.poss_det_pronoun,
            person_entity.poss_pro_pronoun,
            person_entity.refl_pronoun,
        ) = ("she", "her", "her", "hers", "herself")
        person_entity.honorific = "Ms"

    def _is_simple_rule(self, rule: str) -> bool:
        """Return ``True`` when the rule can be handled by the fast numeric path."""
        cleaned = self._strip_rule_comment(rule)
        if not cleaned:
            return True
        if has_century_function(cleaned):
            return False
        if "number_" in cleaned:
            refs = find_entity_refs(cleaned)
            has_non_number_ref = any(not ref.startswith("number_") for ref in refs)
            if not has_non_number_ref:
                return True
        if re.search(r"\bperson_\d+\.age\b", cleaned):
            refs = find_entity_refs(cleaned)
            has_non_age_person = any(not re.fullmatch(r"person_\d+\.age", ref) for ref in refs)
            if has_non_age_person:
                return False
            if re.search(r"\bperson_\d+\.age\b\s*(<=|>=|<|>|=|==)\s*\d+", cleaned):
                return True
            if re.search(r"\d+\s*(<=|>=|<|>)\s*\bperson_\d+\.age\b", cleaned):
                return True
        return False

    def _is_manual_rule(self, rule: str) -> bool:
        """Named-entity rules are intentionally ignored.

        Only numerical, temporal, and person-age constraints participate in
        generation-time rule enforcement. Legacy cross-entity named rules may
        still exist in reviewed templates, but they should not affect pool or
        document generation.
        """
        return False

    def _sample_manual_entities_with_constraints(
        self,
        manual_required: dict[str, list[tuple[str, list[str]]]],
        manual_rules: list[str],
    ) -> EntityCollection | None:
        return sample_manual_entities_with_constraints(self, manual_required, manual_rules)

    @staticmethod
    def _event_family_from_text(text: str | None) -> str | None:
        raw = str(text or "").strip().lower()
        if not raw:
            return None
        for family, keywords in _EVENT_FAMILY_KEYWORDS.items():
            if any(keyword in raw for keyword in keywords):
                return family
        return None

    def _factual_event_family(self, entity_id: str | None) -> str | None:
        if not entity_id or not self.factual_entities or not self.factual_entities.events:
            return None
        factual_event = self.factual_entities.events.get(entity_id)
        if factual_event is None:
            return None
        event_type = (
            factual_event.get("type") if isinstance(factual_event, dict) else getattr(factual_event, "type", None)
        )
        event_name = (
            factual_event.get("name") if isinstance(factual_event, dict) else getattr(factual_event, "name", None)
        )
        return self._event_family_from_text(event_type) or self._event_family_from_text(event_name)

    def _candidate_event_family(self, entity: Any) -> str | None:
        event_type = entity.get("type") if isinstance(entity, dict) else getattr(entity, "type", None)
        event_name = entity.get("name") if isinstance(entity, dict) else getattr(entity, "name", None)
        return self._event_family_from_text(event_type) or self._event_family_from_text(event_name)

    def _entities_equal(self, entity1: Any, entity2: Any, attrs: list[str]) -> bool:
        if not attrs:
            return False

        def get_val(ent: Any, attr: str) -> Any:
            if attr in {"name", "organization_kind"}:
                return organization_attribute_value(ent, attr)
            return ent.get(attr) if isinstance(ent, dict) else getattr(ent, attr, None)

        return all(get_val(entity1, attr) == get_val(entity2, attr) for attr in attrs)

    def _get_key_attributes(self, entity_type: str, required_attrs: list[str]) -> list[str]:
        if entity_type == "place":
            attrs = set(required_attrs)
            if "natural_site" in attrs:
                return ["natural_site"]
            if "street" in attrs:
                ordered = ["street", "city", "state", "region", "country"]
                return [attr for attr in ordered if attr in attrs]
            if "city" in attrs:
                if "state" in attrs:
                    return ["city", "state"]
                if "country" in attrs:
                    return ["city", "country"]
                if "region" in attrs:
                    return ["city", "region"]
                return []
            return []
        key_attr_map = {
            "person": ["full_name", "first_name", "last_name"],
            "event": ["name", "type"],
            "award": ["name"],
            "legal": ["name", "reference_code"],
            "product": ["name"],
            **{organization_type: ["name"] for organization_type in CANONICAL_ORGANIZATION_TYPES},
        }
        key_attrs = key_attr_map.get(entity_type, required_attrs)
        selected = [attr for attr in key_attrs if attr in required_attrs]
        if entity_type == "person":
            return selected
        if entity_type in CANONICAL_ORGANIZATION_TYPES:
            return ["name"]
        return selected if selected else list(required_attrs)


__all__ = [
    "DEBUG_SAMPLING",
    "FictionalEntitySamplerCommonMixin",
    "logger",
]
