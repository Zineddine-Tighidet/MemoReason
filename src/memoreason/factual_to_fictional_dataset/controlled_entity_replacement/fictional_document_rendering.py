"""Orchestrate document-level factual-to-fictional mention replacement."""

from memoreason.benchmark_definition.annotation_runtime import AnnotationParser
from memoreason.benchmark_definition.document_schema import AnnotatedDocument, EntityCollection, FictionalDocument

from .fictional_document_surface_rendering import (
    _capitalize_first_alpha,
    _get_fictional_value,
    _is_sentence_start_annotation,
)
from .fictional_document_text_normalization import _rewrite_leftover_factual_literals


def render_document(annotated_doc: AnnotatedDocument, fictional_entities: EntityCollection) -> FictionalDocument:
    """Replace annotated factual mentions in the document and questions."""
    preserve_original_gender_ids = _find_ambiguous_gender_entity_ids(annotated_doc)
    age_anchor_map = _build_age_anchor_map(annotated_doc)
    generated_text = _replace_annotations(
        annotated_doc.document_to_annotate,
        fictional_entities,
        preserve_original_gender_ids=preserve_original_gender_ids,
        age_anchor_map=age_anchor_map,
    )
    generated_questions = []
    for q in annotated_doc.questions:
        gen_question = _replace_annotations(
            q.question,
            fictional_entities,
            preserve_original_gender_ids=preserve_original_gender_ids,
            age_anchor_map=age_anchor_map,
        )
        generated_questions.append(
            {
                "question_id": q.question_id,
                "question": gen_question,
                "answer": q.answer,
                "question_type": q.question_type,
                "answer_type": q.answer_type,
                "reasoning_chain": list(q.reasoning_chain or []),
            }
        )
    generated_text, generated_questions = _rewrite_leftover_factual_literals(
        annotated_doc,
        fictional_entities,
        generated_text,
        generated_questions,
    )
    return FictionalDocument(
        document_id=annotated_doc.document_id,
        document_theme=annotated_doc.document_theme,
        generated_document=generated_text,
        entities_used=fictional_entities,
        questions=generated_questions,
        evaluated_answers=[],
    )


def _find_ambiguous_gender_entity_ids(annotated_doc: AnnotatedDocument) -> set[str]:
    gender_surfaces: dict[str, set[str]] = {}
    for text in [annotated_doc.document_to_annotate, *(q.question for q in annotated_doc.questions)]:
        for ann in AnnotationParser.parse_annotations(text):
            if ann.attribute != "gender":
                continue
            original = str(ann.original_text or "").strip().lower()
            if not original:
                continue
            gender_surfaces.setdefault(ann.entity_id, set()).add(original)
    return {entity_id for entity_id, surfaces in gender_surfaces.items() if len(surfaces) > 1}


def _build_age_anchor_map(annotated_doc: AnnotatedDocument) -> dict[str, int]:
    anchors: dict[str, int] = {}
    for text in [annotated_doc.document_to_annotate, *(q.question for q in annotated_doc.questions)]:
        for ann in AnnotationParser.parse_annotations(text):
            if ann.attribute != "age":
                continue
            try:
                age_value = int(str(ann.original_text or "").strip())
            except (TypeError, ValueError):
                continue
            anchors[ann.entity_id] = min(age_value, anchors.get(ann.entity_id, age_value))
    return anchors


def _replace_annotations(
    text: str,
    entities: EntityCollection,
    *,
    preserve_original_gender_ids: set[str] | None = None,
    age_anchor_map: dict[str, int] | None = None,
) -> str:
    """Replace all annotations in text with fictional values."""
    annotations = AnnotationParser.parse_annotations(text)
    annotations.sort(key=lambda x: x.start_pos, reverse=True)
    result = text
    preserve_original_gender_ids = preserve_original_gender_ids or set()
    for ann in annotations:
        fictional_value = _get_fictional_value(
            entities,
            ann.entity_id,
            ann.attribute,
            ann.original_text,
            preserve_original_gender=ann.attribute == "gender" and ann.entity_id in preserve_original_gender_ids,
            source_text=text,
            start_pos=ann.start_pos,
            end_pos=ann.end_pos,
            age_anchor_map=age_anchor_map,
        )
        if _is_sentence_start_annotation(text, ann.start_pos):
            fictional_value = _capitalize_first_alpha(fictional_value)
        result = result[: ann.start_pos] + fictional_value + result[ann.end_pos :]
    return result
