"""Shared prompt/config helpers for fictional entity pool generation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import yaml

from memoreason import PROJECT_ROOT_DIRECTORY
from memoreason.benchmark_definition.document_schema import AnnotatedDocument
from memoreason.benchmark_definition.organization_types import (
    CANONICAL_ORGANIZATION_TYPES,
)
from memoreason.benchmark_definition.annotation_runtime import AnnotationParser, find_entity_refs
from memoreason.benchmark_definition.entity_taxonomy import parse_entity_id
from ..dataset_paths import DEFAULT_RANDOM_SEED, prompt_path
from .fictional_entity_replacement_pool_target_planning import (
    build_reference_target_counts,
    build_fictional_entity_replacement_pool_target_plan,
)

_POOL_WIKI_CACHE_PATH = PROJECT_ROOT_DIRECTORY / "output" / "wiki_cache_pool_generation.json"
_POOL_WIKI_TIMEOUT = 8.0
_POOL_WIKI_THROTTLE = 0.5
_POOL_WIKI_MAX_RETRIES = 3
CLAUDE_POOL_PROVIDER = "anthropic"
CLAUDE_POOL_MODEL = "claude-opus-4-6"
CLAUDE_POOL_MODEL_PATTERN = re.compile(r"^claude-opus-(?:4(?:[-.]\d+)?(?:-\d{8})?)$")
_PROMPT_ENTITY_ORDER: tuple[str, ...] = (
    "person",
    "place",
    "event",
    "military_org",
    "entreprise_org",
    "ngo",
    "government_org",
    "educational_org",
    "media_org",
    "award",
    "legal",
    "product",
)
_POOL_PROMPT_TAXONOMY_PATH = PROJECT_ROOT_DIRECTORY / "data" / "WikiEvent" / "entity_taxonomy_extended.yaml"
_POOL_PROMPT_ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "place",
    "event",
    "military_org",
    "entreprise_org",
    "ngo",
    "government_org",
    "educational_org",
    "media_org",
    "award",
    "legal",
    "product",
)
_AUTO_GENERATED_PERSON_ATTRIBUTES: dict[str, str] = {
    "age": "generated later by Python code when age constraints appear in the rules",
    "gender": "assigned later by Python code",
    "subj_pronoun": "generated later from gender",
    "obj_pronoun": "generated later from gender",
    "poss_det_pronoun": "generated later from gender",
    "poss_pro_pronoun": "generated later from gender",
    "refl_pronoun": "generated later from gender",
    "honorific": "generated later from gender",
    "relationship": "mapped later from the factual document after gender assignment",
}


@dataclass(frozen=True)
class FictionalEntityReplacementPoolGenerationConfiguration:
    """Settings for LLM-backed pool generation."""

    provider: str = "anthropic"
    model: str = "claude-opus-4-6"
    temperature: float = 0.0
    max_tokens: int = 32768
    seed: int = DEFAULT_RANDOM_SEED
    max_attempts: int = 12
    validate_against_wikipedia: bool = True

    def __post_init__(self) -> None:
        provider = str(self.provider or "").strip().lower()
        model = str(self.model or "").strip()
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "model", model)
        if provider != CLAUDE_POOL_PROVIDER or not CLAUDE_POOL_MODEL_PATTERN.fullmatch(model):
            raise ValueError(
                "Fictional entity-pool generation is Claude-only and Claude Opus-only. "
                f"Expected provider={CLAUDE_POOL_PROVIDER!r} and a Claude Opus model id; "
                f"got provider={self.provider!r}, model={self.model!r}."
            )


def _target_pool_sizes(required_entities: dict[str, list[tuple[str, list[str]]]]) -> dict[str, int]:
    return build_fictional_entity_replacement_pool_target_plan(required_entities).bucket_sizes


def _reference_target_counts(
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, dict[str, int]]:
    return build_reference_target_counts(required_entities)


def _required_entities_summary(required_entities: dict[str, list[tuple[str, list[str]]]]) -> str:
    lines: list[str] = []
    for entity_type in _PROMPT_ENTITY_ORDER:
        specs = required_entities.get(entity_type, [])
        if not specs:
            continue
        lines.append(f"- {entity_type}: {len(specs)} required ids")
        for entity_id, attributes in specs:
            rendered_attributes = ", ".join(sorted(set(attributes or []))) if attributes else "(no explicit attrs)"
            lines.append(f"  - {entity_id}: {rendered_attributes}")
    return "\n".join(lines)


def _requested_reference_ids(required_entities: dict[str, list[tuple[str, list[str]]]]) -> str:
    reference_ids = [
        entity_id
        for entity_type in _PROMPT_ENTITY_ORDER
        for entity_id, _attributes in required_entities.get(entity_type, [])
    ]
    return yaml.safe_dump(reference_ids, sort_keys=False).rstrip()


def _requested_entity_ids(required_entities: dict[str, list[tuple[str, list[str]]]]) -> list[str]:
    return [
        entity_id
        for entity_type in _PROMPT_ENTITY_ORDER
        for entity_id, _attributes in required_entities.get(entity_type, [])
    ]


def _requested_reference_mentions(
    document: AnnotatedDocument, required_entities: dict[str, list[tuple[str, list[str]]]]
) -> str:
    requested_ids = set(_requested_entity_ids(required_entities))
    mentions: dict[str, list[dict[str, str]]] = {}
    for annotation in AnnotationParser.parse_annotations(document.document_to_annotate or ""):
        if annotation.entity_id not in requested_ids:
            continue
        mentions.setdefault(annotation.entity_id, []).append(
            {
                "text": annotation.original_text,
                "attribute": annotation.attribute or "",
            }
        )
    return yaml.safe_dump(mentions, sort_keys=False, allow_unicode=True).rstrip()


def _document_excerpt_for_requested_refs(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> str:
    text = str(document.document_to_annotate or "")
    if not text:
        return ""
    requested_ids = set(_requested_entity_ids(required_entities))
    excerpts: list[str] = []
    seen: set[str] = set()
    for annotation in AnnotationParser.parse_annotations(text):
        if annotation.entity_id not in requested_ids:
            continue
        start = text.rfind("\n", 0, annotation.start_pos)
        end = text.find("\n", annotation.end_pos)
        if start == -1:
            start = 0
        else:
            start += 1
        if end == -1:
            end = len(text)
        excerpt = text[start:end].strip()
        if not excerpt:
            continue
        if excerpt in seen:
            continue
        seen.add(excerpt)
        excerpts.append(excerpt)
    if excerpts:
        return "\n".join(excerpts)
    return text.strip()


def _extract_reference_pool_subset(
    accepted_pool: dict[str, Any],
    required_entities: dict[str, list[tuple[str, list[str]]]],
) -> dict[str, Any]:
    reference_pools = accepted_pool.get("_reference_pools", {}) if isinstance(accepted_pool, dict) else {}
    if not reference_pools:
        return {
            bucket: accepted_pool.get(bucket, [])
            for bucket in _reference_target_counts(required_entities)
            if isinstance(accepted_pool, dict) and accepted_pool.get(bucket)
        }
    subset: dict[str, Any] = {}
    target_plan = build_fictional_entity_replacement_pool_target_plan(required_entities)
    for bucket, bucket_targets in target_plan.reference_targets.items():
        bucket_refs = reference_pools.get(bucket, {})
        if not isinstance(bucket_refs, dict):
            continue
        rendered_bucket: dict[str, Any] = {}
        for entity_id in bucket_targets:
            ref_payload = bucket_refs.get(entity_id)
            if not isinstance(ref_payload, dict):
                continue
            variants = ref_payload.get("variants", [])
            if not isinstance(variants, list) or not variants:
                continue
            rendered_bucket[entity_id] = {
                "required_attributes": list(ref_payload.get("required_attributes", [])),
                "count": int(ref_payload.get("count", len(variants))),
                "variants": variants,
            }
        if rendered_bucket:
            subset[bucket] = rendered_bucket
    return subset


def _render_reference_shortage_targets(
    shortage_targets: dict[str, Any],
) -> dict[str, dict[str, dict[str, int]]]:
    if shortage_targets and all(isinstance(value, tuple) and len(value) == 2 for value in shortage_targets.values()):
        shortage_targets = {bucket: {"_legacy_bucket": value} for bucket, value in shortage_targets.items()}
    return {
        bucket: {
            entity_id: {
                "current": actual,
                "target": target,
                "need_additional": target - actual,
            }
            for entity_id, (actual, target) in bucket_targets.items()
        }
        for bucket, bucket_targets in shortage_targets.items()
        if bucket_targets
    }


def _pool_relevant_rule_ref(ref: str) -> bool:
    entity_id, attribute = ref.split(".", 1) if "." in ref else (ref, "")
    entity_type, _ = parse_entity_id(entity_id)
    if entity_type in {"number", "temporal"}:
        return False
    if entity_type == "person" and attribute.split(".", 1)[0] in _AUTO_GENERATED_PERSON_ATTRIBUTES:
        return False
    return entity_type in set(_POOL_PROMPT_ENTITY_TYPES) | set(CANONICAL_ORGANIZATION_TYPES)


def _pool_relevant_rules_block(document: AnnotatedDocument) -> str:
    relevant_rules = []
    for raw_rule in document.rules or []:
        refs = find_entity_refs(str(raw_rule))
        if any(_pool_relevant_rule_ref(ref) for ref in refs):
            relevant_rules.append(str(raw_rule))
    return yaml.safe_dump(relevant_rules, sort_keys=False, allow_unicode=True).rstrip()


def _load_prompt_taxonomy() -> dict[str, Any]:
    if _POOL_PROMPT_TAXONOMY_PATH.exists():
        payload = yaml.safe_load(_POOL_PROMPT_TAXONOMY_PATH.read_text(encoding="utf-8")) or {}
        entities = payload.get("entities")
        if isinstance(entities, dict) and entities:
            return entities
    return {}


def _taxonomy_entry_description(entity_type: str, taxonomy: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    entry = taxonomy.get(entity_type, {}) if isinstance(taxonomy, dict) else {}
    description = str(entry.get("description") or "").strip()
    attributes = entry.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}
    return description, attributes


def _attribute_examples_text(attr_data: Any) -> str:
    if not isinstance(attr_data, dict):
        return ""
    examples = attr_data.get("examples", [])
    if not isinstance(examples, list):
        return ""
    rendered_examples = [str(example).strip() for example in examples if str(example).strip()]
    if not rendered_examples:
        return ""
    return ", ".join(f"`{example}`" for example in rendered_examples[:3])


def _render_pool_taxonomy_reference() -> str:
    taxonomy = _load_prompt_taxonomy()
    lines: list[str] = []
    for entity_type in _POOL_PROMPT_ENTITY_TYPES:
        description, attributes = _taxonomy_entry_description(entity_type, taxonomy)
        lines.append(f"- `{entity_type}`")
        if description:
            lines.append(f"  - Meaning: {description}")
        lines.append("  - Attributes:")
        if entity_type == "person":
            for attr_name, attr_data in attributes.items():
                attr_description = str((attr_data or {}).get("description") or "").strip()
                if not attr_description:
                    attr_description = attr_name.replace("_", " ")
                examples_text = _attribute_examples_text(attr_data)
                if attr_name in _AUTO_GENERATED_PERSON_ATTRIBUTES:
                    example_suffix = f" Examples: {examples_text}." if examples_text else ""
                    lines.append(
                        f"    - `{attr_name}`: {attr_description}.{example_suffix} "
                        f"Not part of the pool; {_AUTO_GENERATED_PERSON_ATTRIBUTES[attr_name]}."
                    )
                else:
                    example_suffix = f" Examples: {examples_text}." if examples_text else ""
                    lines.append(f"    - `{attr_name}`: {attr_description}.{example_suffix}")
            continue

        if not attributes:
            lines.append("    - `name`: name.")
            continue

        for attr_name, attr_data in attributes.items():
            attr_description = str((attr_data or {}).get("description") or "").strip()
            if not attr_description:
                attr_description = attr_name.replace("_", " ")
            examples_text = _attribute_examples_text(attr_data)
            example_suffix = f" Examples: {examples_text}." if examples_text else ""
            lines.append(f"    - `{attr_name}`: {attr_description}.{example_suffix}")
    return "\n".join(lines)


def _load_pool_prompt_template() -> str:
    return prompt_path("Fictional Entity Pool Generator.md").read_text(encoding="utf-8")


def build_fictional_entity_replacement_pool_generation_prompt(
    document: AnnotatedDocument,
    required_entities: dict[str, list[tuple[str, list[str]]]],
    *,
    theme: str,
    target_plan_builder=None,
    feedback: str | None = None,
    forbidden_values: set[str] | None = None,
    accepted_pool: dict[str, list[dict[str, str]]] | None = None,
    shortage_targets: dict[str, dict[str, tuple[int, int]]] | None = None,
) -> tuple[str, str]:
    """Build the system and user prompts used for LLM-backed pool generation."""
    builder = target_plan_builder or build_fictional_entity_replacement_pool_target_plan
    target_plan = builder(required_entities)
    prompt_template = _load_pool_prompt_template()
    replacements = {
        "{{DOCUMENT_ID}}": document.document_id,
        "{{DOCUMENT_THEME}}": theme,
        "{{ANNOTATED_DOCUMENT}}": _document_excerpt_for_requested_refs(document, required_entities),
        "{{REQUESTED_REFERENCE_MENTIONS}}": _requested_reference_mentions(document, required_entities),
        "{{POOL_RELEVANT_RULES}}": _pool_relevant_rules_block(document),
        "{{ENTITY_TAXONOMY_REFERENCE}}": _render_pool_taxonomy_reference(),
        "{{REQUIRED_ENTITIES_SUMMARY}}": _required_entities_summary(required_entities),
        "{{REFERENCE_IDS_FOR_THIS_CALL}}": _requested_reference_ids(required_entities),
        "{{REFERENCE_TARGET_COUNTS}}": yaml.safe_dump(
            target_plan.reference_targets,
            sort_keys=False,
        ).rstrip(),
        "{{TARGET_CANDIDATES_PER_ENTITY}}": str(target_plan.candidates_per_required_entity),
    }
    user_prompt = prompt_template
    for placeholder, value in replacements.items():
        user_prompt = user_prompt.replace(placeholder, value)
    banned_values = sorted({str(value).strip() for value in (forbidden_values or set()) if str(value).strip()})
    if banned_values:
        rendered_banned_values = "\n".join(f"- {value}" for value in banned_values)
        user_prompt = (
            f"{user_prompt.rstrip()}\n\n"
            "Forbidden factual or previously rejected values. These are banned and must never appear anywhere in the pool. "
            "Do not output exact copies, translations, aliases, near-spellings, abbreviations, or thin suffix variants of them:\n"
            f"{rendered_banned_values}\n"
        )
    if accepted_pool:
        accepted_subset = _extract_reference_pool_subset(accepted_pool, required_entities)
        if accepted_subset:
            user_prompt = (
                f"{user_prompt.rstrip()}\n\n"
                "Already accepted pool entries. These are accepted reference variants. Keep these values fixed, do not repeat them, and only add fresh variants "
                "that complement this partial pool:\n"
                f"{yaml.safe_dump(accepted_subset, sort_keys=False, allow_unicode=True).rstrip()}\n"
            )
    if shortage_targets:
        remaining_targets = _render_reference_shortage_targets(shortage_targets)
        user_prompt = (
            f"{user_prompt.rstrip()}\n\n"
            "Still-short buckets that must be topped up in this retry. Specifically, these entity references still need variants:\n"
            f"{yaml.safe_dump(remaining_targets, sort_keys=False, allow_unicode=True).rstrip()}\n"
        )
    if feedback:
        user_prompt = f"{user_prompt.rstrip()}\n\nRetry feedback:\n- {feedback}\n"
    system_prompt = (
        "You generate only document-specific fictional entity pools. "
        "Every entity must be non-existing, not merely uncommon. "
        "Do not use ordinary real first names, surnames, places, demonyms, or organization names. "
        "Follow the schema exactly and return YAML only."
    )
    return system_prompt, user_prompt
