"""Parse MemoReason annotations and recover factual entity values."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .document_schema import (
    AnnotatedDocument,
    AwardEntity,
    EntityCollection,
    EventEntity,
    LegalEntity,
    NumberEntity,
    OrganizationEntity,
    PersonEntity,
    PlaceEntity,
    ProductEntity,
    TemporalEntity,
)
from .entity_taxonomy import infer_int_surface_format, infer_str_surface_format, parse_entity_id
from .organization_types import ORG_ENTITY_TYPES, canonicalize_organization_type, infer_organization_kind
from .annotation_references import find_entity_refs, normalize_entity_ref
from .annotation_values import (
    _coerce_numeric_surface,
    _parse_date_surface_components,
    _weekday_from_date_surface,
)


@dataclass
class Annotation:
    """One annotation in text: [text; entity_id.attribute]."""

    start_pos: int
    end_pos: int
    original_text: str
    entity_id: str
    attribute: str | None = None


class AnnotationParser:
    """Parse annotations from document text and extract factual entities."""

    @staticmethod
    def parse_annotations(text: str) -> list[Annotation]:
        """Parse [text; entity_id.attribute] or [text; entity_id] from text."""
        if not text:
            return []
        annotations = []
        pattern = r"\[([^\]]+);\s*([^\]]+)\]"
        for match in re.finditer(pattern, text):
            original_text = match.group(1).strip()
            entity_ref = match.group(2).strip()
            entity_id, attribute = (
                (entity_ref.split(".", 1)[0], entity_ref.split(".", 1)[1]) if "." in entity_ref else (entity_ref, None)
            )
            annotations.append(
                Annotation(
                    start_pos=match.start(),
                    end_pos=match.end(),
                    original_text=original_text,
                    entity_id=entity_id,
                    attribute=attribute,
                )
            )
        return annotations

    @staticmethod
    def _parse_temporal_year_surface(year_text: str) -> int | None:
        raw = str(year_text or "").strip()
        if not raw:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass

        year_range_match = re.fullmatch(
            r"(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\s*[-\u2013\u2014]\s*(\d{2}|\d{4})",
            raw,
        )
        if year_range_match:
            return int(year_range_match.group(1))
        embedded_year_match = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", raw)
        if embedded_year_match:
            return int(embedded_year_match.group(1))
        return None

    @staticmethod
    def extract_factual_entities(
        doc: AnnotatedDocument,
        *,
        include_questions: bool = False,
    ) -> EntityCollection:
        """Extract factual entities from document text (and optionally questions)."""
        entities = EntityCollection()
        all_annotations = list(AnnotationParser.parse_annotations(doc.document_to_annotate))
        if include_questions:
            for q in doc.questions:
                all_annotations.extend(AnnotationParser.parse_annotations(q.question))
                for ref in find_entity_refs(q.question):
                    entity_id, attr = (ref.split(".", 1)[0], ref.split(".", 1)[1]) if "." in ref else (ref, None)
                    all_annotations.append(Annotation(0, 0, "", entity_id, attr))
                if q.answer:
                    for ref in find_entity_refs(q.answer):
                        entity_id, attr = (ref.split(".", 1)[0], ref.split(".", 1)[1]) if "." in ref else (ref, None)
                        all_annotations.append(Annotation(0, 0, "", entity_id, attr))
        entity_data: dict[str, dict[str, str]] = {}
        for ann in all_annotations:
            # Normalize organisation -> organization for consistent entity keys
            eid = normalize_entity_ref(ann.entity_id)
            if eid not in entity_data:
                entity_data[eid] = {}
            if ann.attribute:
                if ann.original_text:
                    entity_data[eid].setdefault(ann.attribute, ann.original_text)
                else:
                    entity_data[eid].setdefault(ann.attribute, "")
            elif ann.original_text:
                entity_data[eid].setdefault("_default", ann.original_text)
        for entity_id, attrs in entity_data.items():
            entity_type, _ = parse_entity_id(entity_id)
            if not entity_type:
                continue
            if entity_type == "number":
                entity = NumberEntity()
                if "int" in attrs:
                    entity.int_surface_format = infer_int_surface_format(attrs["int"])
                    parsed_int = _coerce_numeric_surface(attrs["int"])
                    if isinstance(parsed_int, (int, float)):
                        entity.int = int(parsed_int)
                if "str" in attrs:
                    entity.str = attrs["str"]
                    entity.str_surface_format = infer_str_surface_format(attrs["str"])
                    if entity.int is None:
                        parsed = _coerce_numeric_surface(attrs["str"])
                        if isinstance(parsed, int):
                            entity.int = parsed
                        elif isinstance(parsed, float) and parsed.is_integer():
                            entity.int = int(parsed)
                for key in ("float", "percent", "proportion"):
                    if key in attrs:
                        parsed_numeric = _coerce_numeric_surface(attrs[key])
                        if parsed_numeric is not None:
                            try:
                                setattr(entity, key, float(parsed_numeric))
                            except (ValueError, TypeError):
                                pass
                if "fraction" in attrs:
                    entity.fraction = attrs["fraction"]
                entities.numbers[entity_id] = entity
            elif entity_type == "person":
                entity = PersonEntity()
                if "full_name" in attrs:
                    entity.full_name = attrs["full_name"]
                if "first_name" in attrs:
                    entity.first_name = attrs["first_name"]
                if "last_name" in attrs:
                    entity.last_name = attrs["last_name"]
                if "middle_name" in attrs:
                    entity.middle_name = attrs["middle_name"]
                if "age" in attrs:
                    try:
                        entity.age = int(attrs["age"])
                    except (ValueError, TypeError):
                        entity.age = attrs["age"]
                for k in (
                    "gender",
                    "ethnicity",
                    "nationality",
                    "honorific",
                    "relationship",
                    "subj_pronoun",
                    "obj_pronoun",
                    "poss_det_pronoun",
                    "poss_pro_pronoun",
                    "refl_pronoun",
                ):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                if entity.relationships is None:
                    entity.relationships = {}
                for k, v in attrs.items():
                    if k.startswith("relationship.") and v:
                        other_id = k.split(".", 1)[1]
                        entity.relationships[other_id] = v
                entities.persons[entity_id] = entity
            elif entity_type == "place":
                entity = PlaceEntity()
                for k in ("city", "street", "region", "country", "state", "natural_site", "continent", "demonym"):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                if "nationality" in attrs and entity.demonym is None:
                    entity.demonym = attrs["nationality"]
                    entity.nationality = attrs["nationality"]
                entities.places[entity_id] = entity
            elif entity_type == "temporal":
                entity = TemporalEntity()
                if "year" in attrs:
                    parsed_year = AnnotationParser._parse_temporal_year_surface(attrs["year"])
                    if parsed_year is not None:
                        entity.year = parsed_year
                if "day_of_month" in attrs:
                    try:
                        entity.day_of_month = int(attrs["day_of_month"])
                    except (ValueError, TypeError):
                        pass
                for k in ("day", "date", "month", "timestamp"):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                if "date" in attrs:
                    # Treat `.date` as the authoritative source for derived temporal
                    # components so stale question annotations cannot override the
                    # factual month/day extracted from the document body.
                    date_components = _parse_date_surface_components(attrs["date"])
                    parsed_year = date_components.get("year")
                    parsed_month = date_components.get("month")
                    parsed_day_of_month = date_components.get("day_of_month")
                    if parsed_year is not None:
                        entity.year = int(parsed_year)
                    if parsed_month is not None:
                        entity.month = parsed_month
                    if parsed_day_of_month is not None:
                        entity.day_of_month = int(parsed_day_of_month)
                    if entity.year is None:
                        m = re.search(r"\b(19|20)\d{2}\b", attrs["date"])
                        if m:
                            entity.year = int(m.group(0))
                    # A `.day` reference in a question may be intentionally
                    # implicit in a fully annotated calendar date.  Materialize
                    # that deterministic dependency instead of leaving the
                    # answer blank or trusting a stale independent weekday.
                    if "day" in attrs:
                        derived_weekday = _weekday_from_date_surface(attrs["date"])
                        if derived_weekday is not None:
                            entity.day = derived_weekday
                entities.temporals[entity_id] = entity
            elif entity_type == "event":
                entity = EventEntity()
                for k in ("name", "type"):
                    if k in attrs:
                        setattr(entity, k, attrs[k])
                entities.events[entity_id] = entity
            elif entity_type == "award":
                entities.awards[entity_id] = AwardEntity(name=attrs.get("name") or attrs.get("_default"))
            elif entity_type == "legal":
                entities.legals[entity_id] = LegalEntity(
                    name=attrs.get("name") or attrs.get("_default"),
                    reference_code=attrs.get("reference_code"),
                )
            elif entity_type == "product":
                entities.products[entity_id] = ProductEntity(name=attrs.get("name") or attrs.get("_default"))
            elif entity_type in ORG_ENTITY_TYPES:
                organization_kind = infer_organization_kind(
                    entity_type=entity_type,
                    attribute=next((k for k in attrs if k != "_default"), None),
                )
                entity = OrganizationEntity(
                    name=attrs.get("name") or attrs.get("_default"),
                    organization_kind=organization_kind,
                )
                if entity.name is None:
                    for attribute_name, attribute_value in attrs.items():
                        if attribute_name == "_default":
                            continue
                        inferred_kind = infer_organization_kind(entity_type=entity_type, attribute=attribute_name)
                        if inferred_kind is not None:
                            entity.organization_kind = entity.organization_kind or inferred_kind
                            entity.name = attribute_value
                            break
                entity.organization_kind = entity.organization_kind or canonicalize_organization_type(entity_type)
                entities.organizations[entity_id] = entity
        return entities
