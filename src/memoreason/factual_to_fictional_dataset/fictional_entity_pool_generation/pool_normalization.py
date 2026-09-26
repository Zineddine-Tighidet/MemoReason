# ruff: noqa: B023
"""Pool payload parsing/normalization and support-shortage checks."""

from __future__ import annotations

import json
import re
from typing import Any

import yaml

from memoreason.benchmark_definition.organization_types import (
    ORGANIZATION_POOL_BUCKETS,
    normalize_organization_pool_entry,
)
from .fictional_entity_replacement_pool_target_planning import (
    POOL_BUCKET_TO_ENTITY_TYPES,
    build_fictional_entity_replacement_pool_target_plan,
)
from .pool_support_validation import (
    _distinct_entry_count as _distinct_entry_count,
    _pool_bucket_shortages as _pool_bucket_shortages,
    _pool_support_shortages as _pool_support_shortages,
    _reference_pool_shortages as _reference_pool_shortages,
    _render_bucket_shortages as _render_bucket_shortages,
)

POOL_BUCKETS: tuple[str, ...] = (
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
_REFERENCE_VARIANT_KEYS = {"required_attributes", "count", "variants"}
_YEAR_TOKEN_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b")
_LEADING_YEAR_RE = re.compile(r"^\s*(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b[\s:,\-]+(?=\S)")
_TRAILING_YEAR_RE = re.compile(r"(?<=\S)[\s:,\-]+(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\s*$")
_PAREN_YEAR_RE = re.compile(r"\s*\(\s*(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\s*\)\s*$")
_POOL_VARIANT_METADATA_KEYS = frozenset({"old_name"})
_SINGULAR_ENTITY_TYPE_BY_BUCKET: dict[str, str] = {
    "persons": "person",
    "places": "place",
    "events": "event",
    "organizations": "organization",
    "military_orgs": "military_org",
    "entreprise_orgs": "entreprise_org",
    "ngos": "ngo",
    "government_orgs": "government_org",
    "educational_orgs": "educational_org",
    "media_orgs": "media_org",
    "awards": "award",
    "legals": "legal",
    "products": "product",
}


def _extract_mapping_payload(raw_text: str) -> dict[str, Any]:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("LLM returned an empty entity-pool response.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:yaml|yml|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"(?m)^\s*```(?:yaml|yml|json)?\s*$", "", text)
    text = re.sub(r"(?m)^\s*```\s*$", "", text).strip()

    try:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict):
            return parsed
    except yaml.YAMLError:
        pass

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    bucket_pattern = re.compile(
        r"(?mi)^\s*(?P<bucket>" + "|".join(re.escape(bucket) for bucket in POOL_BUCKET_TO_ENTITY_TYPES) + r"):\s*"
    )
    candidate_starts = [match.start() for match in bucket_pattern.finditer(text)]
    if not candidate_starts:
        fallback_bucket_index = min(
            (text.find(f"{bucket}:") for bucket in POOL_BUCKET_TO_ENTITY_TYPES if text.find(f"{bucket}:") != -1),
            default=-1,
        )
        if fallback_bucket_index != -1:
            candidate_starts = [fallback_bucket_index]

    if not candidate_starts:
        raise ValueError("Could not find a YAML or JSON mapping in the entity-pool response.")

    last_yaml_error: yaml.YAMLError | None = None
    for start_index in candidate_starts:
        extracted_text = re.sub(r"(?m)^\s*```\s*$", "", text[start_index:]).strip()
        try:
            parsed = yaml.safe_load(extracted_text)
        except yaml.YAMLError as exc:
            last_yaml_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed

    if last_yaml_error is not None:
        raise last_yaml_error
    raise ValueError("The extracted entity-pool payload is not a mapping.")


def _empty_normalized_pool() -> dict[str, Any]:
    return {
        **{bucket: [] for bucket in POOL_BUCKETS},
        "_reference_pools": {bucket: {} for bucket in POOL_BUCKETS},
        "_coverage": {bucket: {} for bucket in POOL_BUCKETS},
    }


def _required_attrs_by_entity_id(
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, list[str]]:
    return {entity_id: sorted(set(attrs or [])) for specs in required_entities.values() for entity_id, attrs in specs}


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


def detect_year(named_entity: str) -> int | None:
    text = str(named_entity or "").strip()
    if not text:
        return None
    for pattern in (_LEADING_YEAR_RE, _TRAILING_YEAR_RE, _PAREN_YEAR_RE):
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def strip_detected_year(named_entity: str) -> str:
    text = str(named_entity or "").strip()
    if not text:
        return ""
    normalized = _PAREN_YEAR_RE.sub("", text)
    normalized = _LEADING_YEAR_RE.sub("", normalized)
    normalized = _TRAILING_YEAR_RE.sub("", normalized)
    normalized = re.sub(r"\s{2,}", " ", normalized).strip(" ,:-")
    normalized = re.sub(r"\s{2,}", " ", normalized).strip()
    return normalized


def normalize_named_entity_year_variant(raw_variant: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(raw_variant, dict):
        return {}, False
    variant = {key: value for key, value in raw_variant.items()}
    name = str(variant.get("name") or "").strip()
    if not name:
        return variant, False
    if detect_year(name) is None:
        return variant, False
    stripped_name = strip_detected_year(name)
    if not stripped_name or stripped_name == name:
        return variant, False
    variant["old_name"] = str(variant.get("old_name") or name).strip()
    variant["name"] = stripped_name
    return variant, True


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

        cleaned_person: dict[str, str] = {}

        def _keep(attr: str, value: str) -> None:
            if not value:
                return
            if allowed:
                if attr in allowed:
                    cleaned_person[attr] = value
                return
            if attr in provided_keys:
                cleaned_person[attr] = value

        _keep("full_name", full_name)
        if (
            not allowed
            and "full_name" not in provided_keys
            and full_name
            and {"first_name", "last_name"} <= provided_keys
        ):
            cleaned_person["full_name"] = full_name
        _keep("first_name", first_name)
        _keep("last_name", last_name)
        _keep("middle_name", person.get("middle_name", ""))
        _keep("nationality", person.get("nationality", ""))
        _keep("ethnicity", person.get("ethnicity", ""))
        if not cleaned_person:
            continue
        normalized.append(cleaned_person)
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
        cleaned_entry = {key: str(value).strip() for key, value in raw_entry.items() if str(value).strip()}
        if not cleaned_entry:
            continue
        if "nationality" in cleaned_entry and "demonym" not in cleaned_entry:
            cleaned_entry["demonym"] = cleaned_entry["nationality"]
        if allowed:
            cleaned_entry = {
                key: value
                for key, value in cleaned_entry.items()
                if key in allowed or key in _POOL_VARIANT_METADATA_KEYS or (keep_type and key == "type")
            }
        if require_name and not cleaned_entry.get("name"):
            continue
        normalized.append(cleaned_entry)
    return _dedupe_entries(normalized)


def _normalize_legal_entries(
    raw_entries: list[Any],
    *,
    required_attrs: list[str] | None = None,
) -> list[dict[str, str]]:
    normalized = _normalize_simple_entries(raw_entries, required_attrs=required_attrs)
    kept: list[dict[str, str]] = []
    for entry in normalized:
        if entry.get("name") or entry.get("reference_code"):
            kept.append(entry)
    return kept


def _normalize_organization_entries(raw_entries: list[Any], *, bucket_name: str) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    expected_type = None
    if bucket_name != "organizations":
        expected_type = bucket_name[:-1]
    if bucket_name == "ngos":
        expected_type = "ngo"
    for raw_organization in raw_entries:
        if not isinstance(raw_organization, dict):
            continue
        try:
            organization = normalize_organization_pool_entry(
                raw_organization,
                expected_entity_type=expected_type,
            )
        except ValueError:
            continue
        normalized.append({"name": organization["name"]})
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


def _flatten_reference_pools(reference_pools: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    flattened = {bucket: [] for bucket in POOL_BUCKETS}
    seen = {bucket: set() for bucket in POOL_BUCKETS}
    for bucket in POOL_BUCKETS:
        bucket_refs = reference_pools.get(bucket, {})
        if not isinstance(bucket_refs, dict):
            continue
        for ref_payload in bucket_refs.values():
            if not isinstance(ref_payload, dict):
                continue
            for entry in ref_payload.get("variants", []) or []:
                if not isinstance(entry, dict):
                    continue
                entry_key = tuple(
                    sorted((key, str(value).strip()) for key, value in entry.items() if str(value).strip())
                )
                if not entry_key or entry_key in seen[bucket]:
                    continue
                seen[bucket].add(entry_key)
                flattened[bucket].append(
                    {key: str(value).strip() for key, value in entry.items() if str(value).strip()}
                )
    return flattened


def _normalize_reference_payload(
    bucket_name: str,
    entity_id: str,
    raw_value: Any,
    *,
    required_attrs: list[str],
    global_seen: set[tuple[tuple[str, str], ...]],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    raw_variants: list[Any] = []
    if isinstance(raw_value, dict):
        raw_variants = list(raw_value.get("variants", []) or [])
    elif isinstance(raw_value, list):
        raw_variants = list(raw_value)
    normalized_variants = _normalize_bucket_entries(bucket_name, raw_variants, required_attrs=required_attrs)

    distinct_variants: list[dict[str, str]] = []
    local_seen: set[tuple[tuple[str, str], ...]] = set()
    for entry in normalized_variants:
        entry_key = tuple(sorted(entry.items()))
        if entry_key in local_seen or entry_key in global_seen:
            continue
        local_seen.add(entry_key)
        global_seen.add(entry_key)
        distinct_variants.append(entry)
    ref_payload = {
        "required_attributes": list(required_attrs),
        "count": len(distinct_variants),
        "variants": distinct_variants,
    }
    return ref_payload, distinct_variants


def _normalize_pool_dict(
    pool: dict[str, Any],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, Any]:
    normalized = _empty_normalized_pool()
    target_plan = build_fictional_entity_replacement_pool_target_plan(required_entities)
    required_attrs_by_id = _required_attrs_by_entity_id(required_entities)

    for bucket_name in POOL_BUCKETS:
        raw_bucket = pool.get(bucket_name)
        bucket_reference_targets = target_plan.reference_targets.get(bucket_name, {})
        if isinstance(raw_bucket, dict) and bucket_reference_targets:
            global_seen: set[tuple[tuple[str, str], ...]] = set()
            bucket_refs: dict[str, Any] = {}
            ordered_target_ids = list(bucket_reference_targets)
            unexpected_items = [
                (raw_entity_id, raw_value)
                for raw_entity_id, raw_value in raw_bucket.items()
                if raw_entity_id not in bucket_reference_targets
            ]
            fallback_payload_by_target_id: dict[str, Any] = {}
            missing_target_ids = [entity_id for entity_id in ordered_target_ids if raw_bucket.get(entity_id) is None]
            if unexpected_items and missing_target_ids:
                for target_entity_id, (_raw_entity_id, raw_value) in zip(
                    missing_target_ids, unexpected_items, strict=False
                ):
                    fallback_payload_by_target_id[target_entity_id] = raw_value
            for entity_id in bucket_reference_targets:
                raw_value = raw_bucket.get(entity_id)
                if raw_value is None:
                    raw_value = fallback_payload_by_target_id.get(entity_id)
                if raw_value is None:
                    continue
                ref_payload, _distinct_variants = _normalize_reference_payload(
                    bucket_name,
                    entity_id,
                    raw_value,
                    required_attrs=required_attrs_by_id.get(entity_id, []),
                    global_seen=global_seen,
                )
                bucket_refs[entity_id] = ref_payload
            normalized["_reference_pools"][bucket_name] = bucket_refs
            normalized["_coverage"][bucket_name] = {
                entity_id: int(ref_payload.get("count", 0)) for entity_id, ref_payload in bucket_refs.items()
            }
            normalized[bucket_name] = _flatten_reference_pools({bucket_name: bucket_refs})[bucket_name]
            continue

        normalized[bucket_name] = _normalize_bucket_entries(
            bucket_name,
            list(raw_bucket or []),
            required_attrs=None,
        )

    return normalized


__all__ = [
    "POOL_BUCKETS",
    "_empty_normalized_pool",
    "_extract_mapping_payload",
    "_normalize_pool_dict",
    "_pool_bucket_shortages",
    "_pool_support_shortages",
    "_reference_pool_shortages",
    "_render_bucket_shortages",
    "detect_year",
    "normalize_named_entity_year_variant",
    "strip_detected_year",
]
