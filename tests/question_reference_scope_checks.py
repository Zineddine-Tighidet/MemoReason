import pytest

from memoreason.benchmark_definition.annotation_runtime import (
    AnnotationValidationError,
    validate_question_and_answer_entity_scope,
)


def test_question_reference_scope_accepts_entity_level_matches() -> None:
    document_text = "There were [nine; number_3.str] casualties on [7 July 1972; temporal_1.date]."
    questions = [
        {
            "question_id": "q1",
            "question": "How many casualties were there? Use number_3.",
            "answer": "number_3",
        },
        {
            "question_id": "q2",
            "question": "What year was it? temporal_1.year",
            "answer": "temporal_1.year",
        },
        {
            "question_id": "q3",
            "question": "As int: number_3.int",
            "answer": "number_3.int",
        },
    ]

    validate_question_and_answer_entity_scope(
        document_text,
        questions,
        source_label="scope_entity_level_ok",
    )


def test_question_reference_scope_rejects_missing_entity_ids() -> None:
    document_text = "There were [nine; number_3.str] casualties."
    questions = [
        {
            "question_id": "q1",
            "question": "Is number_999 present?",
            "answer": "number_999",
        }
    ]

    with pytest.raises(AnnotationValidationError, match="number_999"):
        validate_question_and_answer_entity_scope(
            document_text,
            questions,
            source_label="scope_missing_entity_id",
        )
