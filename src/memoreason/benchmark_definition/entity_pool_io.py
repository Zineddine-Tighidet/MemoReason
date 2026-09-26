"""Load and normalize fictional-entity pools used by MemoReason generation."""

from __future__ import annotations

from typing import Any

import yaml

from .organization_types import (
    ORGANIZATION_POOL_BUCKETS,
    canonicalize_organization_type,
    normalize_organization_pool_entry,
    organization_pool_bucket,
)

# Closures intentionally capture the entry currently being normalized.
# ruff: noqa: B023


def load_entity_pool(yaml_path: str) -> dict[str, Any]:
    """
    Load an entity pool from a YAML file.

    Expected YAML structure:
    persons: [...]
    places: [...]
    events: [...]
    organizations: [...]                 # generic organization_X only
    military_orgs: [...]
    entreprise_orgs: [...]
    ngos: [...]
    government_orgs: [...]
    educational_orgs: [...]
    media_orgs: [...]
    awards: [...]
    legals: [...]
    products: [...]
    numbers: [...]
    temporals: [...]
    """
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Entity pool file {yaml_path} does not contain a mapping at the top level.")
    bucket_names = (
        "persons",
        "places",
        "events",
        "organizations",
        "military_orgs",
        "entreprise_orgs",
        "ngos",
        "government_orgs",
        "educational_orgs",
        "media_orgs",
        "awards",
        "legals",
        "products",
    )
    normalized_pool: dict[str, Any] = {
        **{bucket: [] for bucket in bucket_names},
        "_reference_pools": {bucket: {} for bucket in bucket_names},
        "_coverage": {bucket: {} for bucket in bucket_names},
    }
    pool_variant_metadata_keys = {"old_name"}

    def _dedupe_entries(entries: list[dict[str, str]]) -> list[dict[str, str]]:
        deduped: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for entry in entries:
            entry_key = tuple(sorted(entry.items()))
            if entry_key in seen:
                continue
            seen.add(entry_key)
            deduped.append(entry)
        return deduped

    def _normalize_person_entries(
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        allowed = set(required_attrs or [])
        for raw_person in raw_entries:
            if not isinstance(raw_person, dict):
                continue
            person = {key: str(value).strip() for key, value in raw_person.items() if str(value).strip()}
            provided_keys = set(person.keys())
            full_name = person.get("full_name", "")
            first_name = person.get("first_name", "")
            last_name = person.get("last_name", "")
            if not full_name and first_name and last_name:
                full_name = f"{first_name} {last_name}"
            if full_name and (not first_name or not last_name):
                parts = full_name.split()
                if len(parts) >= 2:
                    first_name = first_name or parts[0]
                    last_name = last_name or parts[-1]
            cleaned: dict[str, str] = {}

            def _keep(attr: str, value: str) -> None:
                if not value:
                    return
                if allowed:
                    if attr in allowed:
                        cleaned[attr] = value
                    return
                if attr in provided_keys:
                    cleaned[attr] = value

            _keep("full_name", full_name)
            if (
                not allowed
                and "full_name" not in provided_keys
                and full_name
                and {"first_name", "last_name"} <= provided_keys
            ):
                cleaned["full_name"] = full_name
            _keep("first_name", first_name)
            _keep("last_name", last_name)
            _keep("middle_name", person.get("middle_name", ""))
            _keep("nationality", person.get("nationality", ""))
            _keep("ethnicity", person.get("ethnicity", ""))
            if cleaned:
                normalized.append(cleaned)
        return _dedupe_entries(normalized)

    def _normalize_simple_entries(
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
        require_name: bool = False,
        keep_type: bool = False,
    ) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        allowed = set(required_attrs or [])
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            cleaned = {key: str(value).strip() for key, value in raw_entry.items() if str(value).strip()}
            if not cleaned:
                continue
            if "nationality" in cleaned and "demonym" not in cleaned:
                cleaned["demonym"] = cleaned["nationality"]
            if allowed:
                cleaned = {
                    key: value
                    for key, value in cleaned.items()
                    if key in allowed or key in pool_variant_metadata_keys or (keep_type and key == "type")
                }
            if require_name and not cleaned.get("name"):
                continue
            normalized.append(cleaned)
        return _dedupe_entries(normalized)

    def _normalize_legal_entries(
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
    ) -> list[dict[str, str]]:
        normalized = _normalize_simple_entries(raw_entries, required_attrs=required_attrs)
        return [entry for entry in normalized if entry.get("name") or entry.get("reference_code")]

    def _normalize_organization_entries(raw_entries: list[Any], *, bucket_name: str) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        expected_type = None
        if bucket_name != "organizations":
            expected_type = canonicalize_organization_type(
                bucket_name[:-1] if bucket_name.endswith("s") else bucket_name
            )
        if bucket_name == "ngos":
            expected_type = "ngo"
        for raw_entry in raw_entries:
            if not isinstance(raw_entry, dict):
                continue
            try:
                normalized_entry = normalize_organization_pool_entry(raw_entry, expected_entity_type=expected_type)
            except ValueError:
                continue
            normalized.append({"name": normalized_entry["name"]})
        return _dedupe_entries(normalized)

    def _normalize_bucket_entries(
        bucket_name: str,
        raw_entries: list[Any],
        *,
        required_attrs: list[str] | None = None,
    ) -> list[dict[str, str]]:
        if bucket_name == "persons":
            return _normalize_person_entries(raw_entries, required_attrs=required_attrs)
        if bucket_name == "places":
            return _normalize_simple_entries(raw_entries, required_attrs=required_attrs)
        if bucket_name == "events":
            require_event_name = "name" in set(required_attrs or [])
            return _normalize_simple_entries(
                raw_entries,
                required_attrs=required_attrs,
                require_name=require_event_name,
                keep_type=True,
            )
        if bucket_name in {"awards", "products"}:
            return _normalize_simple_entries(raw_entries, required_attrs=required_attrs, require_name=True)
        if bucket_name == "legals":
            return _normalize_legal_entries(raw_entries, required_attrs=required_attrs)
        if bucket_name in ORGANIZATION_POOL_BUCKETS:
            return _normalize_organization_entries(raw_entries, bucket_name=bucket_name)
        return []

    def _flatten_reference_bucket(bucket_name: str) -> list[dict[str, str]]:
        flattened: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        bucket_refs = normalized_pool["_reference_pools"].get(bucket_name, {})
        if not isinstance(bucket_refs, dict):
            return []
        for ref_payload in bucket_refs.values():
            if not isinstance(ref_payload, dict):
                continue
            for entry in ref_payload.get("variants", []) or []:
                if not isinstance(entry, dict):
                    continue
                entry_key = tuple(
                    sorted((key, str(value).strip()) for key, value in entry.items() if str(value).strip())
                )
                if not entry_key or entry_key in seen:
                    continue
                seen.add(entry_key)
                flattened.append({key: str(value).strip() for key, value in entry.items() if str(value).strip()})
        return flattened

    seen_organization_entries: dict[str, set[tuple[tuple[str, str], ...]]] = {
        bucket_name: set() for bucket_name in ORGANIZATION_POOL_BUCKETS
    }
    for bucket_name in bucket_names:
        raw_bucket = data.get(bucket_name)
        if not isinstance(raw_bucket, dict):
            continue
        for entity_id, raw_value in raw_bucket.items():
            raw_variants: list[Any] = []
            required_attrs: list[str] = []
            if isinstance(raw_value, dict):
                raw_variants = list(raw_value.get("variants", []) or [])
                required_attrs = [
                    str(attr).strip() for attr in raw_value.get("required_attributes", []) or [] if str(attr).strip()
                ]
            elif isinstance(raw_value, list):
                raw_variants = list(raw_value)
            variants = _normalize_bucket_entries(bucket_name, raw_variants, required_attrs=required_attrs or None)
            normalized_pool["_reference_pools"][bucket_name][str(entity_id)] = {
                "required_attributes": required_attrs,
                "count": len(variants),
                "variants": variants,
            }
            normalized_pool["_coverage"][bucket_name][str(entity_id)] = len(variants)
        normalized_pool[bucket_name] = _flatten_reference_bucket(bucket_name)

    for bucket_name in ORGANIZATION_POOL_BUCKETS:
        if normalized_pool["_reference_pools"].get(bucket_name):
            continue
        expected_type = None
        if bucket_name != "organizations":
            expected_type = canonicalize_organization_type(
                bucket_name[:-1] if bucket_name.endswith("s") else bucket_name
            )
        if bucket_name == "ngos":
            expected_type = "ngo"
        for raw_entry in data.get(bucket_name, []) or []:
            if not isinstance(raw_entry, dict):
                continue
            try:
                normalized_entry = normalize_organization_pool_entry(raw_entry, expected_entity_type=expected_type)
            except ValueError:
                continue
            target_bucket = organization_pool_bucket(normalized_entry["organization_kind"]) or bucket_name
            entry_payload = {"name": normalized_entry["name"]}
            entry_key = tuple(sorted(entry_payload.items()))
            if entry_key in seen_organization_entries[target_bucket]:
                continue
            seen_organization_entries[target_bucket].add(entry_key)
            normalized_pool[target_bucket].append(entry_payload)

    if not normalized_pool["_reference_pools"]["places"]:
        normalized_pool["places"] = _normalize_bucket_entries("places", list(data.get("places", []) or []))
    if not normalized_pool["_reference_pools"]["awards"]:
        normalized_pool["awards"] = _normalize_bucket_entries("awards", list(data.get("awards", []) or []))
    if not normalized_pool["_reference_pools"]["legals"]:
        normalized_pool["legals"] = _normalize_bucket_entries("legals", list(data.get("legals", []) or []))
    if not normalized_pool["_reference_pools"]["products"]:
        normalized_pool["products"] = _normalize_bucket_entries("products", list(data.get("products", []) or []))
    if not normalized_pool["_reference_pools"]["persons"]:
        normalized_pool["persons"] = _normalize_bucket_entries("persons", list(data.get("persons", []) or []))
    if not normalized_pool["_reference_pools"]["events"]:
        normalized_pool["events"] = _normalize_bucket_entries("events", list(data.get("events", []) or []))
    return normalized_pool
