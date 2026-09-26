"""Canonical identities used to avoid named-entity reuse across variants."""

from __future__ import annotations

import json
import unicodedata
from typing import Any

from memoreason.benchmark_definition.organization_types import CANONICAL_ORGANIZATION_TYPES


def _named_entity_bucket(entity_type_or_bucket: str) -> str | None:
    value = str(entity_type_or_bucket or "").strip()
    if (
        value in CANONICAL_ORGANIZATION_TYPES
        or (value.endswith("s") and value[:-1] in CANONICAL_ORGANIZATION_TYPES)
        or value in {"organization", "organizations"}
    ):
        return "organizations"
    return {
        "person": "persons",
        "persons": "persons",
        "place": "places",
        "places": "places",
        "event": "events",
        "events": "events",
        "award": "awards",
        "awards": "awards",
        "legal": "legals",
        "legals": "legals",
        "product": "products",
        "products": "products",
    }.get(value)


def _named_entity_payload(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return payload
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return None


def _surface(value: Any) -> str:
    return " ".join(str(value).strip().split())


def normalize_named_entity_uniqueness_value(
    entity_type_or_bucket: str,
    payload: Any,
) -> dict[str, str] | None:
    """Return the stable identity fields used by the inter-variant audit."""
    bucket = _named_entity_bucket(entity_type_or_bucket)
    values = _named_entity_payload(payload)
    if bucket is None or values is None:
        return None

    if bucket == "persons":
        key_groups = (
            ("full_name",),
            ("first_name", "last_name"),
            ("first_name",),
            ("last_name",),
            ("name",),
            ("nationality", "ethnicity"),
            ("nationality",),
            ("ethnicity",),
            ("middle_name",),
        )
    elif bucket == "places":
        key_groups = (
            ("natural_site",),
            ("street", "city", "state", "country", "region"),
            ("city", "state"),
            ("city", "country"),
            ("city", "region"),
            ("city",),
            ("country",),
            ("state",),
            ("region",),
            ("continent", "demonym", "nationality"),
            ("continent",),
            ("demonym",),
            ("nationality",),
        )
    elif bucket == "events":
        key_groups = (("name", "type"),)
    elif bucket == "legals":
        key_groups = (("name", "reference_code"),)
    elif bucket in {"organizations", "awards", "products"}:
        key_groups = (("name",),)
    else:
        return None

    for keys in key_groups:
        normalized = {
            key: _surface(values[key]) for key in keys if values.get(key) not in (None, "") and _surface(values[key])
        }
        if bucket == "persons" and len(normalized) != len(keys):
            continue
        if normalized:
            return normalized
    return None


def named_entity_uniqueness_signature(entity_type_or_bucket: str, payload: Any) -> str | None:
    """Serialize a named identity for deterministic, normalized equality checks."""
    normalized = normalize_named_entity_uniqueness_value(entity_type_or_bucket, payload)
    if normalized is None:
        return None
    comparison_value = {key: unicodedata.normalize("NFKC", value).casefold() for key, value in normalized.items()}
    return json.dumps(comparison_value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


__all__ = [
    "named_entity_uniqueness_signature",
    "normalize_named_entity_uniqueness_value",
]
