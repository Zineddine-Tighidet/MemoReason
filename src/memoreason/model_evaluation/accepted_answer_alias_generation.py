"""Generate accepted answer aliases from document entities and question wording."""

from __future__ import annotations

import re

from memoreason.benchmark_definition.annotation_runtime import find_entity_refs
from memoreason.benchmark_definition.answer_matching import normalize_answer
from memoreason.benchmark_definition.document_schema import EntityCollection, PersonEntity

from .answer_schema_data_contracts import (
    _DEGREE_LABEL_RE,
    _DEGREE_QUESTION_EXCLUSION_KEYWORDS,
    _DEGREE_QUESTION_PREFIXES,
    _DOCUMENT_SURFACE_ORG_SUFFIXES,
    _ENTITY_EXPLANATION_SPLIT_RE,
    _LEADING_ARTICLES,
    _PERSON_ORIGIN_QUALIFIER_RE,
    _PERSON_SUFFIXES,
    _POSSESSIVE_DESCRIPTOR_SUFFIX_RE,
    _SHORT_NAME_QUESTION_KEYWORDS,
    _TRAILING_ACRONYM_RE,
)
from .answer_normalization import (
    _clean_text,
    _dedupe_preserve_order,
    _derive_acronym,
    _entity_type_from_id,
    _should_add_parenthetical_acronym_alias,
    canonicalize_answer,
)


def _surface_form_aliases(value: str, question_text: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []

    aliases = [cleaned]
    lowered_question = str(question_text or "").strip().lower()
    lowered_value = cleaned.lower()

    for prefix in ("the ", "its ", "a ", "an "):
        if lowered_value.startswith(prefix):
            aliases.append(cleaned[len(prefix) :].strip())

    if lowered_value.endswith(" itself"):
        aliases.append(cleaned[:-7].strip())

    if "degree" in lowered_question and re.search(r"\bdegree\b", cleaned, flags=re.IGNORECASE):
        aliases.append(re.sub(r"\bdegree\b\s*", "", cleaned, flags=re.IGNORECASE).strip())

    if "trophy" in lowered_question:
        if lowered_value.endswith(" trophy"):
            aliases.append(cleaned[: -len(" Trophy")].strip())
        else:
            aliases.append(f"{cleaned} Trophy")

    if " " not in cleaned:
        if lowered_value.endswith("s") and len(cleaned) > 3:
            aliases.append(cleaned[:-1])
        elif len(cleaned) > 2:
            aliases.append(f"{cleaned}s")
        if lowered_value.endswith("ially") and len(cleaned) > 5:
            aliases.append(cleaned[:-2])
        elif lowered_value.endswith("ial") and len(cleaned) > 4:
            aliases.append(f"{cleaned}ly")

    if lowered_value.endswith("-shaped"):
        aliases.append(cleaned[: -len("-shaped")].strip())

    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _degree_core_aliases(value: str, question_text: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned or not _question_allows_degree_core_alias(question_text):
        return []

    core = ""
    degree_in_match = re.match(r"^(?P<core>.+?)\s+degree\s+in\s+.+$", cleaned, flags=re.IGNORECASE)
    if degree_in_match:
        candidate = re.sub(r"\s+", " ", degree_in_match.group("core")).strip()
        if _DEGREE_LABEL_RE.match(candidate):
            core = candidate
    if not core:
        plain_in_match = re.match(r"^(?P<core>.+?)\s+in\s+.+$", cleaned, flags=re.IGNORECASE)
        if plain_in_match:
            candidate = re.sub(r"\s+", " ", plain_in_match.group("core")).strip()
            if _DEGREE_LABEL_RE.match(candidate):
                core = candidate
    if not core:
        return []

    if not core or normalize_answer(core) == normalize_answer(cleaned):
        return []
    return [core]


def _question_allows_degree_core_alias(question_text: str) -> bool:
    lowered = str(question_text or "").strip().lower()
    if "degree" not in lowered:
        return False
    if any(keyword in lowered for keyword in _DEGREE_QUESTION_EXCLUSION_KEYWORDS):
        return False
    return any(lowered.startswith(prefix) for prefix in _DEGREE_QUESTION_PREFIXES)


def _document_surface_org_aliases(
    value: str,
    *,
    question_text: str,
    document_text: str,
) -> list[str]:
    cleaned = _clean_text(value)
    source_text = str(document_text or "")
    if not cleaned or not source_text.strip():
        return []
    if not _question_allows_short_name_alias(question_text):
        return []
    if len(cleaned.split()) > 3:
        return []

    escaped_base = re.escape(cleaned)
    candidates: list[str] = []
    for suffix in _DOCUMENT_SURFACE_ORG_SUFFIXES:
        pattern = re.compile(
            rf"(?<!\w){escaped_base}\s+{re.escape(suffix)}(?!\w)",
            flags=re.IGNORECASE,
        )
        for match in pattern.finditer(source_text):
            candidate = re.sub(r"\s+", " ", match.group(0)).strip(" \t\r\n,.;:!?")
            if candidate:
                candidates.append(candidate)

    if not candidates:
        return []

    deduped_candidates: list[str] = []
    seen_canonical: set[str] = set()
    base_canonical = canonicalize_answer("entity_span", cleaned)
    for candidate in candidates:
        candidate_canonical = canonicalize_answer("entity_span", candidate)
        if not candidate_canonical or candidate_canonical == base_canonical:
            continue
        if candidate_canonical in seen_canonical:
            continue
        seen_canonical.add(candidate_canonical)
        deduped_candidates.append(candidate)

    if len(deduped_candidates) != 1:
        return []
    return deduped_candidates


def _entity_ref_value(entity_ref: str, entities_used: EntityCollection) -> str | None:
    entity_id, _, attr = entity_ref.partition(".")
    if not entity_id or not attr:
        return None
    entity_type = _entity_type_from_id(entity_id)
    if entity_type is None:
        return None
    try:
        collection = entities_used.get_collection(entity_type)
    except ValueError:
        return None
    entity = collection.get(entity_id)
    if entity is None:
        return None
    value = getattr(entity, attr, None)
    if value is None:
        return None
    return _clean_text(value)


def _composite_expression_aliases(
    answer_expression: str,
    question_text: str,
    entities_used: EntityCollection,
) -> list[str]:
    stripped_expr = str(answer_expression or "").strip()
    refs = find_entity_refs(stripped_expr)
    if len(refs) < 2:
        return []

    expression_skeleton = stripped_expr
    for ref in refs:
        expression_skeleton = expression_skeleton.replace(ref, "REF")
    collapsed_skeleton = re.sub(r"\s+", "", expression_skeleton)
    if any(operator in collapsed_skeleton for operator in ("+", "-", "*", "/")):
        return []
    if re.search(r"[A-Za-z0-9_]+\s*\(", expression_skeleton):
        return []

    values = [_entity_ref_value(ref, entities_used) for ref in refs]
    if any(not value for value in values):
        return []
    resolved_values = [value for value in values if value]

    aliases: list[str] = []
    if len(resolved_values) == 2:
        first, second = resolved_values
        aliases.extend(
            (
                f"{first} and {second}",
                f"{second} and {first}",
                f"{first}, {second}",
                f"{second}, {first}",
                f"('{first}', '{second}')",
                f"('{second}', '{first}')",
            )
        )
    else:
        aliases.append(", ".join(resolved_values))

    lowered_question = str(question_text or "").strip().lower()
    if "which two" in lowered_question or "what two" in lowered_question:
        aliases.extend(resolved_values)

    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _entity_aliases(entity_ref: str, question_text: str, entities_used: EntityCollection) -> list[str]:
    entity_id, _, attr = entity_ref.partition(".")
    if not entity_id or not attr:
        return []
    entity_type = _entity_type_from_id(entity_id)
    if entity_type is None:
        return []

    try:
        collection = entities_used.get_collection(entity_type)
    except ValueError:
        return []
    entity = collection.get(entity_id)
    if entity is None:
        return []

    if entity_type != "person":
        value = _clean_text(getattr(entity, attr, None))
        aliases = [value] if value else []
        aliases.extend(_name_like_aliases(value))
        if attr == "name" and _question_allows_short_name_alias(question_text):
            leading_alias = _leading_name_alias(value)
            if leading_alias and _is_unique_non_person_leading_alias(collection, leading_alias):
                aliases.append(leading_alias)
        return _dedupe_preserve_order(alias for alias in aliases if alias)

    person = entities_used.persons.get(entity_id)
    if person is None:
        return []
    requested_component = _requested_person_name_component(question_text)
    if requested_component is not None:
        component_value = _person_component_value(person, requested_component)
        return [component_value] if component_value else []

    aliases = []
    if person.full_name:
        aliases.append(person.full_name)
    full_name_without_suffix = _person_full_name_without_suffix(person)
    if full_name_without_suffix:
        aliases.append(full_name_without_suffix)
    first_last = _first_last_name(person)
    if first_last:
        aliases.append(first_last)
    first_token = _person_first_token(person)
    if first_token and _is_unique_person_first_token(entities_used, first_token):
        aliases.append(first_token)
    if person.last_name and _is_unique_person_attribute(entities_used, "last_name", person.last_name):
        aliases.append(person.last_name)
    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _requested_person_name_component(question_text: str) -> str | None:
    lowered = str(question_text or "").strip().lower()
    if "full name" in lowered:
        return "full_name"
    if "first name" in lowered or "given name" in lowered:
        return "first_name"
    if "last name" in lowered or "surname" in lowered or "family name" in lowered:
        return "last_name"
    if "middle name" in lowered:
        return "middle_name"
    return None


def _person_component_value(person: PersonEntity, component: str) -> str | None:
    if component == "full_name":
        if person.full_name:
            return person.full_name
        return _first_last_name(person)
    if component == "first_name":
        return _person_first_token(person)
    value = getattr(person, component, None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _first_last_name(person: PersonEntity) -> str | None:
    first_name = _person_first_token(person)
    if first_name and person.last_name:
        return f"{first_name} {person.last_name}".strip()
    return None


def _person_first_token(person: PersonEntity) -> str | None:
    if person.first_name and person.first_name.strip():
        return person.first_name.strip()
    if not person.full_name:
        return None
    tokens = _person_name_tokens_without_suffix(person.full_name)
    if not tokens:
        return None
    return tokens[0]


def _person_full_name_without_suffix(person: PersonEntity) -> str | None:
    if not person.full_name:
        return None
    tokens = _person_name_tokens_without_suffix(person.full_name)
    if not tokens:
        return None
    candidate = " ".join(tokens).strip()
    if candidate == person.full_name.strip():
        return None
    return candidate


def _person_name_tokens_without_suffix(full_name: str) -> list[str]:
    tokens = [token for token in _clean_text(full_name).split() if token]
    while tokens:
        suffix = tokens[-1].rstrip(".").lower()
        if suffix not in _PERSON_SUFFIXES:
            break
        tokens.pop()
    return tokens


def _is_unique_person_attribute(entities_used: EntityCollection, attribute: str, value: str) -> bool:
    target = normalize_answer(value)
    matches = 0
    for person in entities_used.persons.values():
        candidate = getattr(person, attribute, None)
        if isinstance(candidate, str) and normalize_answer(candidate) == target:
            matches += 1
    return matches == 1


def _is_unique_person_first_token(entities_used: EntityCollection, value: str) -> bool:
    target = normalize_answer(value)
    matches = 0
    for person in entities_used.persons.values():
        candidate = _person_first_token(person)
        if candidate and normalize_answer(candidate) == target:
            matches += 1
    return matches == 1


def _name_like_aliases(value: str) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned:
        return []
    aliases = [cleaned]
    acronym_match = _TRAILING_ACRONYM_RE.match(cleaned)
    if acronym_match:
        aliases.append(acronym_match.group("long").strip())
        aliases.append(acronym_match.group("short").strip())
    else:
        derived_acronym = _derive_acronym(cleaned) if _should_add_parenthetical_acronym_alias(cleaned) else None
        if derived_acronym:
            aliases.append(f"{cleaned} ({derived_acronym})")
    return _dedupe_preserve_order(alias for alias in aliases if alias)


def _question_allows_short_name_alias(question_text: str) -> bool:
    lowered = str(question_text or "").strip().lower()
    return any(keyword in lowered for keyword in _SHORT_NAME_QUESTION_KEYWORDS)


def _leading_name_alias(value: str) -> str | None:
    cleaned = _clean_text(value)
    if not cleaned:
        return None
    tokens = [token for token in cleaned.split() if token]
    while tokens and tokens[0].lower() in _LEADING_ARTICLES:
        tokens.pop(0)
    if len(tokens) < 2:
        return None
    return tokens[0]


def _is_unique_non_person_leading_alias(collection: dict[str, object], alias: str) -> bool:
    target = normalize_answer(alias)
    matches = 0
    for entity in collection.values():
        name = _clean_text(getattr(entity, "name", None))
        candidate = _leading_name_alias(name)
        if candidate and normalize_answer(candidate) == target:
            matches += 1
    return matches == 1


def _profession_modifier_aliases(value: str, question_text: str, entities_used: EntityCollection) -> list[str]:
    cleaned = _clean_text(value)
    if not cleaned or not _question_allows_profession_alias(question_text):
        return []
    if cleaned.lower() != cleaned:
        return []
    if len(cleaned.split()) > 3:
        return []

    modifiers: list[str] = []
    for person in entities_used.persons.values():
        if person.nationality and person.nationality.strip():
            modifiers.append(person.nationality.strip())
    for place in entities_used.places.values():
        for attr in ("demonym", "nationality"):
            modifier = getattr(place, attr, None)
            if isinstance(modifier, str) and modifier.strip():
                modifiers.append(modifier.strip())

    return _dedupe_preserve_order(f"{modifier} {cleaned}" for modifier in modifiers if modifier)


def _question_allows_profession_alias(question_text: str) -> bool:
    lowered = str(question_text or "").strip().lower()
    if "profession" in lowered or "occupation" in lowered or "job" in lowered:
        return True
    return "what was" in lowered and any(keyword in lowered for keyword in ("profession", "occupation", "job"))


def _strip_person_origin_qualifier(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    match = _PERSON_ORIGIN_QUALIFIER_RE.match(cleaned)
    if not match:
        return ""
    base = re.sub(r"\s+", " ", match.group("base")).strip()
    place = re.sub(r"\s+", " ", match.group("place")).strip()
    if len(base.split()) < 2:
        return ""
    if not place:
        return ""
    return base


def _strip_possessive_descriptor_suffix(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    match = _POSSESSIVE_DESCRIPTOR_SUFFIX_RE.match(cleaned)
    if not match:
        return ""
    head = re.sub(r"\s+", " ", match.group("head")).strip()
    tail = re.sub(r"\s+", " ", match.group("tail")).strip()
    if len(head.split()) < 2:
        return ""
    if len(tail.split()) < 2 and not any(character.isdigit() for character in tail):
        return ""
    return head


def _trim_entity_candidate(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    for separator in (",", ";", " — ", " – "):  # noqa: RUF001
        if separator in cleaned:
            head = cleaned.split(separator, 1)[0].strip()
            if 1 <= len(head.split()) <= 8:
                cleaned = head
                break
    split_match = _ENTITY_EXPLANATION_SPLIT_RE.match(cleaned)
    if split_match:
        head = split_match.group("head").strip()
        if 1 <= len(head.split()) <= 8:
            return head
    return cleaned
