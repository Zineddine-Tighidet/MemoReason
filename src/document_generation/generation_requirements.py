"""Parse annotated documents into required-entity specifications for sampling."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.core.organization_types import infer_organization_kind
from src.core.annotation_runtime import (
    ORG_ENTITY_TYPES,
    Annotation,
    AnnotationParser,
    find_entity_refs,
    normalize_entity_ref,
    parse_entity_id,
)

_EXCLUDED_PERSON_DERIVED_ATTRS: Set[str] = {
    "gender",
    "subj_pronoun",
    "obj_pronoun",
    "poss_det_pronoun",
    "poss_pro_pronoun",
    "refl_pronoun",
    "relationship",
}

_SUPPORTED_ENTITY_TYPES = {
    "person",
    "place",
    "event",
    "organization",
    "military_org",
    "entreprise_org",
    "ngo",
    "government_org",
    "educational_org",
    "media_org",
    "award",
    "legal",
    "product",
    "number",
    "temporal",
}


def _collect_annotations(annotated_doc, *, include_questions: bool = False) -> List[Annotation]:
    annotations = AnnotationParser.parse_annotations(annotated_doc.document_to_annotate)
    if not include_questions:
        return annotations

    for question in annotated_doc.questions:
        annotations.extend(AnnotationParser.parse_annotations(question.question))
        if not question.answer:
            continue
        for ref in find_entity_refs(question.answer):
            entity_id, attribute = ref.split(".", 1) if "." in ref else (ref, None)
            annotations.append(
                Annotation(
                    start_pos=0,
                    end_pos=0,
                    original_text="",
                    entity_id=entity_id,
                    attribute=attribute,
                )
            )
    return annotations


def _attribute_root(attribute: str | None) -> str | None:
    if not attribute:
        return None
    if "." in attribute:
        return attribute.split(".", 1)[0]
    return attribute


def _build_entity_attribute_map(annotated_doc, *, include_questions: bool = False) -> Dict[str, Set[str]]:
    entity_map: Dict[str, Set[str]] = {}
    for ann in _collect_annotations(annotated_doc, include_questions=include_questions):
        entity_id = normalize_entity_ref(ann.entity_id)
        entity_map.setdefault(entity_id, set())
        attr_root = _attribute_root(ann.attribute)
        if not attr_root or attr_root == "relationship" or attr_root in _EXCLUDED_PERSON_DERIVED_ATTRS:
            continue
        entity_map[entity_id].add(attr_root)

    for rule in annotated_doc.rules:
        for ref in find_entity_refs(rule):
            entity_id, attribute = ref.split(".", 1) if "." in ref else (ref, None)
            entity_id = normalize_entity_ref(entity_id)
            entity_type, _ = parse_entity_id(entity_id)
            if not entity_type:
                continue
            type_key = (
                infer_organization_kind(entity_type=entity_type) if entity_type in ORG_ENTITY_TYPES else entity_type
            )
            if type_key in _SUPPORTED_ENTITY_TYPES:
                entity_map.setdefault(entity_id, set())
                attr_root = _attribute_root(attribute)
                if attr_root:
                    entity_map[entity_id].add(attr_root)
    return entity_map


def extract_required_entities(
    annotated_doc,
    *,
    include_questions: bool = False,
) -> Dict[str, List[Tuple[str, List[str]]]]:
    """Extract required entities and referenced attributes from an annotated doc."""
    entity_map = _build_entity_attribute_map(annotated_doc, include_questions=include_questions)
    required: Dict[str, List[Tuple[str, List[str]]]] = defaultdict(list)
    for entity_id, attrs in entity_map.items():
        entity_type, _ = parse_entity_id(entity_id)
        if not entity_type:
            continue
        type_key = infer_organization_kind(entity_type=entity_type) if entity_type in ORG_ENTITY_TYPES else entity_type
        if type_key is None:
            continue
        attrs_set = set(attrs)
        if type_key in ORG_ENTITY_TYPES:
            attrs_set.add("name")
        required[type_key].append((entity_id, sorted(attrs_set)))
    return dict(required)


def _collect_decade_year_temporal_ids_from_text(text: str) -> Set[str]:
    if not isinstance(text, str) or not text:
        return set()

    decade_year_temporal_ids: Set[str] = set()
    for annotation in AnnotationParser.parse_annotations(text):
        if (annotation.attribute or "").strip() != "year":
            continue
        if annotation.end_pos >= len(text):
            continue
        if text[annotation.end_pos].lower() != "s":
            continue

        normalized_entity_id = normalize_entity_ref(annotation.entity_id)
        entity_type, _ = parse_entity_id(normalized_entity_id)
        if entity_type == "temporal":
            decade_year_temporal_ids.add(normalized_entity_id)
    return decade_year_temporal_ids


def extract_decade_year_temporal_ids(
    annotated_doc,
    *,
    include_questions: bool = False,
) -> Set[str]:
    """Return temporal IDs whose year annotation is immediately followed by ``s``."""
    decade_year_temporal_ids = _collect_decade_year_temporal_ids_from_text(
        getattr(annotated_doc, "document_to_annotate", "") or ""
    )

    if include_questions:
        for question in getattr(annotated_doc, "questions", []) or []:
            decade_year_temporal_ids.update(
                _collect_decade_year_temporal_ids_from_text(getattr(question, "question", "") or "")
            )
            answer = getattr(question, "answer", None)
            if isinstance(answer, str):
                decade_year_temporal_ids.update(_collect_decade_year_temporal_ids_from_text(answer))
            elif isinstance(answer, list):
                for item in answer:
                    if isinstance(item, str):
                        decade_year_temporal_ids.update(_collect_decade_year_temporal_ids_from_text(item))

    return decade_year_temporal_ids
