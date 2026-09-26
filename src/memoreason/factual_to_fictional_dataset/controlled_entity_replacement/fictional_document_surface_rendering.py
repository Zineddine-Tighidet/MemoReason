# ruff: noqa: RUF013
"""Render entity values into the surface forms found in factual text."""

import re
from decimal import Decimal
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import RuleEngine, get_appropriate_relationship
from memoreason.benchmark_definition.document_schema import EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import (
    infer_int_surface_format,
    infer_str_surface_format,
    render_integer_surface_number,
    render_word_surface_number,
)

from .fictional_document_rendering_constants import (
    _GENDER_ADJECTIVES_PLURAL,
    _GENDER_ADJECTIVES_SINGULAR,
    _GENDER_NOUN_ADULT_PLURAL,
    _GENDER_NOUN_ADULT_SINGULAR,
    _GENDER_NOUN_CHILD_PLURAL,
    _GENDER_NOUN_CHILD_SINGULAR,
)


def _capitalize_first_alpha(value: str) -> str:
    if not value:
        return value
    chars = list(value)
    for idx, ch in enumerate(chars):
        if ch.isalpha():
            chars[idx] = ch.upper()
            break
    return "".join(chars)


def _capitalize_sentence_starts(text: str) -> str:
    if not text:
        return text
    chars = list(text)
    capitalize_next = True
    idx = 0
    while idx < len(chars):
        char = chars[idx]
        if capitalize_next and char.isalpha():
            chars[idx] = char.upper()
            capitalize_next = False
        elif char in ".!?":
            capitalize_next = True
        elif char == "\n":
            next_index = idx + 1
            while next_index < len(chars) and chars[next_index] == "\n":
                capitalize_next = True
                next_index += 1
        elif not char.isspace() and char not in "\"')]}":
            capitalize_next = False
        idx += 1
    return "".join(chars)


def _is_sentence_start_annotation(source_text: str, start_pos: int) -> bool:
    """Return True when an annotation starts at a sentence boundary."""
    if start_pos <= 0:
        return True

    i = start_pos - 1
    while i >= 0 and source_text[i].isspace():
        i -= 1
    if i < 0:
        return True

    # Allow closing punctuation/quotes before a sentence boundary, e.g. ." [X; ...]
    while i >= 0 and source_text[i] in "\"')]}":
        i -= 1
        while i >= 0 and source_text[i].isspace():
            i -= 1
    if i < 0:
        return True

    if source_text[i] in ".!?":
        return True

    gap_text = source_text[i + 1 : start_pos]
    return "\n\n" in gap_text


def _apply_original_casing(template: str, value: str) -> str:
    if template.isupper():
        return value.upper()
    if template.islower():
        return value.lower()
    if template.istitle():
        return value.title()
    return value


def _coerce_age(age_value: Any) -> int | None:
    try:
        return int(age_value) if age_value is not None else None
    except (TypeError, ValueError):
        return None


def _render_gender_surface_form(
    target_gender: str | None,
    original_value: str | None,
    age_value: Any,
) -> str | None:
    if not original_value:
        return None
    target = (target_gender or "").strip().lower()
    if target not in {"male", "female"}:
        return None

    original = str(original_value).strip()
    if not original:
        return None
    original_lower = original.lower()

    if original_lower in _GENDER_ADJECTIVES_SINGULAR:
        return _apply_original_casing(original, target)
    if original_lower in _GENDER_ADJECTIVES_PLURAL:
        return _apply_original_casing(original, f"{target}s")

    all_noun_terms = (
        _GENDER_NOUN_CHILD_SINGULAR
        | _GENDER_NOUN_CHILD_PLURAL
        | _GENDER_NOUN_ADULT_SINGULAR
        | _GENDER_NOUN_ADULT_PLURAL
    )
    if original_lower not in all_noun_terms:
        return None

    age = _coerce_age(age_value)
    if age is not None:
        is_adult = age >= 18
    else:
        is_adult = original_lower in _GENDER_NOUN_ADULT_SINGULAR or original_lower in _GENDER_NOUN_ADULT_PLURAL

    is_plural = original_lower in _GENDER_NOUN_CHILD_PLURAL or original_lower in _GENDER_NOUN_ADULT_PLURAL
    if target == "male":
        replacement = ("men" if is_plural else "man") if is_adult else ("boys" if is_plural else "boy")
    else:
        replacement = ("women" if is_plural else "woman") if is_adult else ("girls" if is_plural else "girl")

    return _apply_original_casing(original, replacement)


def _format_numeric_surface(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        quantized = Decimal(str(value)).quantize(Decimal("0.01"))
        return format(quantized.normalize(), "f").rstrip("0").rstrip(".")
    return str(value)


def _temporal_parts_from_entity(temporal_entity: Any) -> tuple[int | None, str | None, int | None]:
    year = getattr(temporal_entity, "year", None)
    month = getattr(temporal_entity, "month", None)
    day_of_month = getattr(temporal_entity, "day_of_month", None)
    date_value = getattr(temporal_entity, "date", None)
    if (month is None or day_of_month is None) and isinstance(date_value, str):
        month_day_year = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", date_value.strip())
        if month_day_year:
            if day_of_month is None:
                day_of_month = int(month_day_year.group(1))
            if month is None:
                month = month_day_year.group(2)
            if year is None:
                year = int(month_day_year.group(3))
        month_year = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", date_value.strip())
        if month_year:
            if month is None:
                month = month_year.group(1)
            if year is None:
                year = int(month_year.group(2))
    return year, month, day_of_month


def _render_temporal_date_surface(temporal_entity: Any, original_value: str | None) -> str | None:
    if temporal_entity is None:
        return None
    date_value = getattr(temporal_entity, "date", None)
    year, month, day_of_month = _temporal_parts_from_entity(temporal_entity)
    original = str(original_value or "").strip()
    if not original:
        return str(date_value) if date_value is not None else None
    year_range = re.fullmatch(r"(\d{4})\s*[-\u2013\u2014]\s*(\d{2}|\d{4})", original)
    if year_range and year is not None:
        end_text = year_range.group(2)
        span = (int(end_text) - int(year_range.group(1))) if len(end_text) == 4 else int(end_text)
        end_year = year + span
        if len(end_text) == 2:
            return f"{year}\u2013{end_year % 100:02d}"
        return f"{year}\u2013{end_year}"
    if re.fullmatch(r"[A-Za-z]+", original):
        return month or (
            str(date_value).split()[1]
            if isinstance(date_value, str) and len(str(date_value).split()) >= 2
            else str(date_value)
        )
    if re.fullmatch(r"[A-Za-z]+\s+\d{4}", original) and month and year is not None:
        return f"{month} {year}"
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", original) and month and year is not None:
        day = day_of_month if day_of_month is not None else 1
        return f"{day} {month} {year}"
    if re.fullmatch(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}", original) and month and year is not None:
        day = day_of_month if day_of_month is not None else 1
        return f"{month} {day}, {year}"
    if re.fullmatch(r"[A-Za-z]+\s+\d{1,2}", original) and month:
        day = day_of_month if day_of_month is not None else 1
        return f"{month} {day}"
    if re.fullmatch(r"\d{1,2}\s+[A-Za-z]+", original) and month:
        day = day_of_month if day_of_month is not None else 1
        return f"{day} {month}"
    return str(date_value) if date_value is not None else None


def _render_name_variant(original_value: str | None, base_value: str) -> str:
    original = str(original_value or "").strip()
    value = str(base_value or "").strip()
    if not original or not value:
        return value
    original_lower = original.lower()
    value_lower = value.lower()
    if original_lower.startswith("project "):
        if value_lower.startswith("project "):
            return value
        if value_lower.endswith(" program"):
            core = value[:-8].strip()
            return f"Project {core}"
        return f"Project {value}"
    if original_lower.endswith(" program"):
        if value_lower.endswith(" program"):
            return value
        if value_lower.startswith("project "):
            core = value[8:].strip()
            return f"{core} program"
        if "project" in value_lower:
            return value
        return f"{value} program"
    if len(original) > 4 and re.fullmatch(r"[A-Za-z0-9]+", original) and " " in value:
        if value_lower.startswith("project "):
            return value[8:].strip()
        return value.split()[0]
    return value


def _needs_parenthetical_long_form(source_text: str | None, end_pos: int | None) -> bool:
    if source_text is None or end_pos is None:
        return False
    return source_text[end_pos:].lstrip().startswith("(")


def _expand_single_token_name(entity_id: str, value: str) -> str:
    if " " in value:
        return value
    entity_type = entity_id.split("_", 1)[0]
    if entity_type in {
        "organization",
        "entreprise",
        "government",
        "educational",
        "media",
        "military",
        "ngo",
    }:
        return f"{value} Group"
    if entity_type == "product":
        return f"{value} Suite"
    return value


def _get_fictional_value(
    entities: EntityCollection,
    entity_id: str,
    attribute: str | None,
    original_value: str = None,
    *,
    preserve_original_gender: bool = False,
    source_text: str | None = None,
    start_pos: int | None = None,
    end_pos: int | None = None,
    age_anchor_map: dict[str, int] | None = None,
) -> str:
    """Get fictional value for an entity.attribute."""
    del start_pos
    if attribute == "date":
        temporal_entity = entities.temporals.get(entity_id)
        rendered = _render_temporal_date_surface(temporal_entity, original_value)
        if rendered is not None:
            return rendered
    if attribute in {"int", "str"}:
        number_entity = entities.numbers.get(entity_id)
        if number_entity is not None:
            if attribute == "int" and number_entity.int is not None:
                int_surface_format = infer_int_surface_format(original_value or "") or number_entity.int_surface_format
                return render_integer_surface_number(int(number_entity.int), int_surface_format)
            if attribute == "str" and number_entity.int is not None:
                str_surface_format = infer_str_surface_format(original_value or "") or number_entity.str_surface_format
                if str_surface_format is not None:
                    return render_word_surface_number(int(number_entity.int), str_surface_format)
            if attribute == "str" and number_entity.str is not None:
                return str(number_entity.str)
    if attribute == "gender":
        if preserve_original_gender and original_value is not None:
            return str(original_value)
        person_entity = entities.persons.get(entity_id)
        if person_entity is not None:
            rendered = _render_gender_surface_form(
                target_gender=person_entity.gender,
                original_value=original_value,
                age_value=person_entity.age,
            )
            if rendered is not None:
                return rendered
    if attribute == "age":
        person_entity = entities.persons.get(entity_id)
        rendered_age = _coerce_age(getattr(person_entity, "age", None) if person_entity else None)
        original_age = _coerce_age(original_value)
        anchor_age = (age_anchor_map or {}).get(entity_id)
        if rendered_age is not None:
            if original_age is not None and anchor_age is not None:
                return str(rendered_age + (original_age - anchor_age))
            return str(rendered_age)
    if attribute and "." in attribute:
        parts = attribute.split(".")
        if parts[0] == "relationship":
            person_entity = entities.persons.get(entity_id)
            if person_entity and getattr(person_entity, "relationships", None):
                stored = person_entity.relationships.get(parts[1])
                if stored is not None:
                    return str(stored)
            if original_value and person_entity:
                return get_appropriate_relationship(original_value, person_entity)
            entity_ref = f"{entity_id}.relationship.{parts[1]}"
            val = RuleEngine._get_entity_value(entities, entity_ref)
            if val is None:
                return str(original_value) if original_value else ""
            return _format_numeric_surface(val) if isinstance(val, (int, float)) else str(val)
    entity_ref = f"{entity_id}.{attribute}" if attribute else entity_id
    value = RuleEngine._get_entity_value(entities, entity_ref)
    if value is None:
        return ""
    if attribute == "name":
        rendered_name = _render_name_variant(original_value, str(value))
        if _needs_parenthetical_long_form(source_text, end_pos):
            return _expand_single_token_name(entity_id, rendered_name)
        return rendered_name
    return _format_numeric_surface(value) if isinstance(value, (int, float)) else str(value)
