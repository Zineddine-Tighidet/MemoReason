"""Entity-reference normalization used across MemoReason generation and evaluation."""

from __future__ import annotations

import re
from typing import Any

from .entity_taxonomy import VALID_ENTITY_TYPES, parse_entity_id
from .organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
    LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND,
    LEGACY_ORGANIZATION_TYPE_ALIASES,
    canonicalize_organization_type,
    infer_organization_kind,
)


def _split_entity_ref(entity_ref: str) -> tuple[str, str | None]:
    cleaned = str(entity_ref or "").strip()
    if "." in cleaned:
        return cleaned.split(".", 1)[0], cleaned.split(".", 1)[1]
    return cleaned, None


def _canonical_organization_attribute(attribute: str | None) -> str | None:
    if attribute is None:
        return None
    cleaned = str(attribute).strip()
    if not cleaned:
        return None
    if cleaned == "name":
        return "name"
    if cleaned in {"organization_kind", "kind"}:
        return cleaned
    if cleaned in LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND:
        return "name"
    if cleaned in LEGACY_ORGANIZATION_TYPE_ALIASES:
        return "name"
    if cleaned in CANONICAL_ORGANIZATION_TYPES:
        return "name"
    return cleaned


def _prefer_more_specific_organization_kind(current_kind: str | None, candidate_kind: str | None) -> str | None:
    if candidate_kind is None:
        return current_kind
    if current_kind in {None, "organization"}:
        return candidate_kind
    return current_kind


def _infer_document_organization_id_map(texts: list[str]) -> dict[str, str]:
    inferred_kinds: dict[str, str | None] = {}
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        for raw_ref in find_entity_refs(text):
            entity_id, attribute = _split_entity_ref(raw_ref)
            entity_type, entity_index = parse_entity_id(entity_id)
            if entity_type is None or entity_index is None:
                continue

            normalized_entity_id = entity_id.replace("organisation_", "organization_", 1)
            entity_type, entity_index = parse_entity_id(normalized_entity_id)
            if entity_type is None or entity_index is None:
                continue

            candidate_kind = infer_organization_kind(entity_type=entity_type, attribute=attribute)
            if candidate_kind is None:
                continue

            inferred_kinds[normalized_entity_id] = _prefer_more_specific_organization_kind(
                inferred_kinds.get(normalized_entity_id),
                candidate_kind,
            )

    remap: dict[str, str] = {}
    for original_entity_id, candidate_kind in inferred_kinds.items():
        entity_type, entity_index = parse_entity_id(original_entity_id)
        if entity_type is None or entity_index is None:
            continue
        canonical_type = candidate_kind or canonicalize_organization_type(entity_type) or entity_type
        canonical_entity_id = f"{canonical_type}_{entity_index}"
        remap[original_entity_id] = canonical_entity_id
    return remap


def _normalize_annotation_surface_key(raw_text: str) -> str:
    return re.sub(r"\s+", " ", str(raw_text or "").strip()).casefold()


def normalize_text_entity_refs(text: str, *, entity_id_remap: dict[str, str] | None = None) -> str:
    """Rewrite legacy entity references in one text field to the current taxonomy."""
    if not isinstance(text, str) or not text:
        return text
    remap = entity_id_remap or {}

    def _replace(match: re.Match[str]) -> str:
        return normalize_entity_ref(match.group(0), entity_id_remap=remap)

    return ENTITY_REF_PATTERN.sub(_replace, text)


# --- Entity references and annotation parsing ---
_ENTITY_REF_TYPES = tuple(sorted(set(VALID_ENTITY_TYPES) | {"organisation"}, key=len, reverse=True))
_ENTITY_REF_TYPES_RE = "|".join(re.escape(t) for t in _ENTITY_REF_TYPES)
_PERSON_RELATIONSHIP_REF_RE = r"person_\d+\.relationship\.person_\d+"
_SIMPLE_ENTITY_REF_RE = rf"(?:{_ENTITY_REF_TYPES_RE})_\d+(?:\.[a-z_]+)?"
ENTITY_REF_PATTERN = re.compile(rf"\b(?:{_PERSON_RELATIONSHIP_REF_RE}|{_SIMPLE_ENTITY_REF_RE})\b")
ENTITY_REF_VALIDATION_PATTERN = re.compile(rf"^(?:{_PERSON_RELATIONSHIP_REF_RE}|{_SIMPLE_ENTITY_REF_RE})$")
PERSON_RELATIONSHIP_REF_PATTERN = re.compile(rf"^{_PERSON_RELATIONSHIP_REF_RE}$")


def normalize_entity_ref(ref: str, *, entity_id_remap: dict[str, str] | None = None) -> str:
    """Normalize one entity reference to the current taxonomy."""
    if not ref:
        return ref
    entity_id, attribute = _split_entity_ref(ref)
    normalized_entity_id = entity_id.replace("organisation_", "organization_", 1)
    entity_type, entity_index = parse_entity_id(normalized_entity_id)

    canonical_entity_id = normalized_entity_id
    if entity_type is not None and entity_index is not None:
        # Canonicalize zero-padded ids (e.g., number_09 -> number_9) so rule refs
        # resolve to annotation ids consistently across the pipeline.
        canonical_entity_id = f"{entity_type}_{entity_index}"
    canonical_attribute = attribute
    remap = entity_id_remap or {}
    if normalized_entity_id in remap:
        canonical_entity_id = remap[normalized_entity_id]
    elif canonical_entity_id in remap:
        canonical_entity_id = remap[canonical_entity_id]

    if entity_type is not None and entity_index is not None:
        canonical_organization_type = canonicalize_organization_type(entity_type)
        if canonical_organization_type is not None:
            canonical_organization_id = f"{canonical_organization_type}_{entity_index}"
            canonical_entity_id = remap.get(
                normalized_entity_id, remap.get(canonical_organization_id, canonical_organization_id)
            )
            canonical_attribute = _canonical_organization_attribute(attribute)

    if canonical_attribute:
        return f"{canonical_entity_id}.{canonical_attribute}"
    return canonical_entity_id


def find_entity_refs(text: str) -> list[str]:
    """Return all entity references in text (e.g. number_3, place_2.city)."""
    if not text:
        return []
    return [match.group(0) for match in ENTITY_REF_PATTERN.finditer(str(text))]


def is_valid_entity_ref(s: str) -> bool:
    """True if s is a valid single entity reference."""
    return bool(s and ENTITY_REF_VALIDATION_PATTERN.match(s.strip()))


# --- Relationship mapper (gender-appropriate terms) ---
_GENDERED_RELATIONSHIPS = {
    "sibling": ("brother", "sister", "sibling"),
    "parent": ("father", "mother", "parent"),
    "child": ("son", "daughter", "child"),
    "spouse": ("husband", "wife", "spouse"),
    "grandparent": ("grandfather", "grandmother", "grandparent"),
    "grandchild": ("grandson", "granddaughter", "grandchild"),
    "parent_sibling": ("uncle", "aunt", "parent's sibling"),
    "sibling_child": ("nephew", "niece", "sibling's child"),
}
_RELATIONSHIP_TO_BASE = {}
for _base, (_m, _f, _n) in _GENDERED_RELATIONSHIPS.items():
    _RELATIONSHIP_TO_BASE[_m.lower()] = (_base, "male")
    _RELATIONSHIP_TO_BASE[_f.lower()] = (_base, "female")
    _RELATIONSHIP_TO_BASE[_n.lower()] = (_base, "neutral")
_NON_GENDERED = {"cousin", "friend", "partner", "colleague", "ally", "rival", "enemy", "neighbor"}


def map_relationship_for_gender(original_relationship: str, target_gender: str) -> str:
    """Map relationship to gender-appropriate term (target_gender: 'male'|'female'|'neutral')."""
    orig = original_relationship.lower()
    if orig in _NON_GENDERED:
        return original_relationship
    if orig in _RELATIONSHIP_TO_BASE:
        base, _ = _RELATIONSHIP_TO_BASE[orig]
        male, female, neutral = _GENDERED_RELATIONSHIPS[base]
        return male if target_gender == "male" else female if target_gender == "female" else neutral
    return original_relationship


def get_appropriate_relationship(original_relationship: str, person_entity: Any) -> str:
    """Map relationship to gender-appropriate term from person pronouns."""

    def infer_gender():
        for attr in ("subj_pronoun", "obj_pronoun", "poss_det_pronoun"):
            p = getattr(person_entity, attr, None)
            if not p:
                continue
            p = str(p).lower()
            if p in ("he", "him"):
                return "male"
            if p in ("she", "her", "hers"):
                return "female"
            if p in ("they", "them", "their", "theirs"):
                return "neutral"
        return "neutral"

    return map_relationship_for_gender(original_relationship, infer_gender())
