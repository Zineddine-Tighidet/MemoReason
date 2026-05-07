"""Canonical organization subtype schema and legacy migration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CANONICAL_ORGANIZATION_TYPES: tuple[str, ...] = (
    "organization",
    "military_org",
    "entreprise_org",
    "ngo",
    "government_org",
    "educational_org",
    "media_org",
)

ORGANIZATION_POOL_BUCKET_BY_TYPE: dict[str, str] = {
    "organization": "organizations",
    "military_org": "military_orgs",
    "entreprise_org": "entreprise_orgs",
    "ngo": "ngos",
    "government_org": "government_orgs",
    "educational_org": "educational_orgs",
    "media_org": "media_orgs",
}

ORGANIZATION_TYPE_BY_POOL_BUCKET: dict[str, str] = {
    bucket_name: organization_type
    for organization_type, bucket_name in ORGANIZATION_POOL_BUCKET_BY_TYPE.items()
}

ORGANIZATION_POOL_BUCKETS: tuple[str, ...] = tuple(
    ORGANIZATION_POOL_BUCKET_BY_TYPE[organization_type]
    for organization_type in CANONICAL_ORGANIZATION_TYPES
)

LEGACY_ORGANIZATION_TYPE_ALIASES: dict[str, str] = {
    "organisation": "organization",
    "military_organization": "military_org",
    "entreprise_organization": "entreprise_org",
    "ong": "ngo",
    "government_organization": "government_org",
    "educational_organization": "educational_org",
    "media_organization": "media_org",
}

LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND: dict[str, str] = {
    "is_education": "educational_org",
    "is_journalism": "media_org",
    "is_government": "government_org",
    "is_military": "military_org",
    "is_transportation": "entreprise_org",
    "is_technology": "entreprise_org",
    "is_finance": "entreprise_org",
    "is_medical": "ngo",
    "is_sport": "ngo",
    "is_military_org": "military_org",
    "is_entreprise_org": "entreprise_org",
    "is_ngo": "ngo",
    "is_government_org": "government_org",
    "is_educational_org": "educational_org",
    "is_media_org": "media_org",
    "is_military_organization": "military_org",
    "is_entreprise_organization": "entreprise_org",
    "is_ong": "ngo",
    "is_government_organization": "government_org",
    "is_educational_organization": "educational_org",
    "is_media_organization": "media_org",
}

ORG_ENTITY_TYPES: frozenset[str] = frozenset(
    CANONICAL_ORGANIZATION_TYPES
    + tuple(LEGACY_ORGANIZATION_TYPE_ALIASES.keys())
)


def canonicalize_organization_type(entity_type: str | None) -> str | None:
    """Return the canonical organization entity type."""
    if not entity_type:
        return None
    cleaned = str(entity_type).strip()
    if cleaned in CANONICAL_ORGANIZATION_TYPES:
        return cleaned
    return LEGACY_ORGANIZATION_TYPE_ALIASES.get(cleaned)


def is_organization_entity_type(entity_type: str | None) -> bool:
    """Return whether the entity type denotes an organization subtype."""
    return canonicalize_organization_type(entity_type) is not None


def infer_organization_kind(
    *,
    entity_type: str | None,
    attribute: str | None = None,
) -> str | None:
    """Infer the canonical organization subtype from an entity type and optional attribute."""
    canonical_type = canonicalize_organization_type(entity_type)
    if canonical_type and canonical_type != "organization":
        return canonical_type
    if attribute:
        attribute_root = str(attribute).split(".", 1)[0].strip()
        if attribute_root in LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND:
            return LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND[attribute_root]
    return canonical_type


def organization_pool_bucket(entity_type: str | None) -> str | None:
    """Return the pool bucket name used for one canonical organization type."""
    canonical_type = canonicalize_organization_type(entity_type)
    if canonical_type is None:
        return None
    return ORGANIZATION_POOL_BUCKET_BY_TYPE[canonical_type]


def organization_type_for_pool_bucket(bucket_name: str | None) -> str | None:
    """Return the canonical organization type for one pool bucket name."""
    if not bucket_name:
        return None
    return ORGANIZATION_TYPE_BY_POOL_BUCKET.get(str(bucket_name).strip())


def get_organization_name(value: Any) -> str | None:
    """Return the organization name from a mapping or model."""
    if isinstance(value, Mapping):
        raw_name = value.get("name")
    else:
        raw_name = getattr(value, "name", None)
    if raw_name is None:
        return None
    cleaned = str(raw_name).strip()
    return cleaned or None


def get_organization_kind(value: Any) -> str | None:
    """Return the canonical organization kind from a mapping or model."""
    if isinstance(value, Mapping):
        raw_kind = value.get("organization_kind")
    else:
        raw_kind = getattr(value, "organization_kind", None)
    canonical_kind = canonicalize_organization_type(raw_kind)
    if canonical_kind is not None:
        return canonical_kind

    if isinstance(value, Mapping):
        raw_items = value.items()
    else:
        raw_items = ((field_name, getattr(value, field_name)) for field_name in dir(value))

    for key, raw_value in raw_items:
        if raw_value is None or str(raw_value).strip() == "":
            continue
        legacy_kind = LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND.get(str(key))
        if legacy_kind is not None:
            return legacy_kind
    return None


def organization_matches_kind(value: Any, required_entity_type: str) -> bool:
    """Return whether an organization entry or model can satisfy the required subtype."""
    required_kind = infer_organization_kind(entity_type=required_entity_type)
    if required_kind is None:
        return False
    if get_organization_name(value) is None:
        return False
    if required_kind == "organization":
        return True
    return get_organization_kind(value) == required_kind


def organization_attribute_value(value: Any, attribute: str | None) -> str | None:
    """Resolve an organization attribute against the canonical subtype-based schema."""
    if not attribute:
        return get_organization_name(value)
    attribute_root = str(attribute).split(".", 1)[0].strip()
    if attribute_root == "name":
        return get_organization_name(value)
    if attribute_root in {"organization_kind", "kind"}:
        return get_organization_kind(value)

    expected_kind = None
    if attribute_root in CANONICAL_ORGANIZATION_TYPES:
        expected_kind = attribute_root
    elif attribute_root in LEGACY_ORGANIZATION_TYPE_ALIASES:
        expected_kind = canonicalize_organization_type(attribute_root)
    else:
        expected_kind = LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND.get(attribute_root)

    if expected_kind is None:
        return None
    if expected_kind == "organization":
        return get_organization_name(value)
    return get_organization_name(value) if get_organization_kind(value) == expected_kind else None


def normalize_organization_pool_entry(
    raw_entry: Mapping[str, Any],
    *,
    expected_entity_type: str | None = None,
) -> dict[str, str]:
    """Normalize a raw organization pool entry into ``{organization_kind, name}``."""
    name = None
    kind = canonicalize_organization_type(expected_entity_type) or canonicalize_organization_type(
        raw_entry.get("organization_kind") or raw_entry.get("organization_type") or raw_entry.get("kind")
    )

    raw_name = raw_entry.get("name")
    if raw_name is not None and str(raw_name).strip():
        name = str(raw_name).strip()

    legacy_pairs: set[tuple[str, str]] = set()
    for key, raw_value in raw_entry.items():
        if raw_value is None:
            continue
        cleaned_value = str(raw_value).strip()
        if not cleaned_value:
            continue
        legacy_kind = None
        if str(key) in LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND:
            legacy_kind = LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND[str(key)]
        else:
            canonical_key = canonicalize_organization_type(str(key))
            if canonical_key and canonical_key != "organization":
                legacy_kind = canonical_key
        if legacy_kind is not None:
            legacy_pairs.add((legacy_kind, cleaned_value))

    if len(legacy_pairs) > 1:
        raise ValueError(
            "Organization pool entry mixes multiple organization subtypes: "
            f"{sorted(kind for kind, _ in legacy_pairs)}"
        )
    if legacy_pairs:
        legacy_kind, legacy_name = next(iter(legacy_pairs))
        if kind in {None, "organization"} and legacy_kind != "organization":
            kind = legacy_kind
        else:
            kind = kind or legacy_kind
        name = name or legacy_name

    if kind is None:
        kind = "organization"
    if name is None:
        raise ValueError("Organization pool entry is missing a name.")
    if kind not in CANONICAL_ORGANIZATION_TYPES:
        raise ValueError(f"Unsupported organization kind: {kind!r}")
    return {
        "organization_kind": kind,
        "name": name,
    }


__all__ = [
    "CANONICAL_ORGANIZATION_TYPES",
    "LEGACY_ORGANIZATION_ATTRIBUTE_TO_KIND",
    "LEGACY_ORGANIZATION_TYPE_ALIASES",
    "ORGANIZATION_POOL_BUCKETS",
    "ORGANIZATION_POOL_BUCKET_BY_TYPE",
    "ORGANIZATION_TYPE_BY_POOL_BUCKET",
    "ORG_ENTITY_TYPES",
    "canonicalize_organization_type",
    "get_organization_kind",
    "get_organization_name",
    "infer_organization_kind",
    "is_organization_entity_type",
    "normalize_organization_pool_entry",
    "organization_attribute_value",
    "organization_matches_kind",
    "organization_pool_bucket",
    "organization_type_for_pool_bucket",
]
