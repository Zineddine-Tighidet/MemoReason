"""Serialize one generated fictional document variant to the on-disk YAML format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml

from src.core.answer_evaluation import AnswerEvaluator
from src.core.document_schema import EntityCollection
from src.core.annotation_runtime import RuleEngine, find_entity_refs


def unwrap_answer_expression(raw_answer: Any) -> str:
    """Normalize stored answer expressions before evaluating them."""
    if isinstance(raw_answer, list) and len(raw_answer) == 1 and isinstance(raw_answer[0], str):
        return raw_answer[0]
    if isinstance(raw_answer, str) and raw_answer.strip().startswith("[") and raw_answer.strip().endswith("]"):
        try:
            import ast

            parsed = ast.literal_eval(raw_answer.strip())
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
                return parsed[0]
        except (ValueError, SyntaxError):
            pass
    return str(raw_answer) if raw_answer is not None else ""


def build_generated_question_payloads(generated_document, entities: EntityCollection) -> list[dict[str, Any]]:
    """Build the exported question payloads for one generated document variant."""
    payloads: list[dict[str, Any]] = []
    for question in generated_document.questions:
        answer_expression = unwrap_answer_expression(question["answer"])
        cleaned_expression = AnswerEvaluator._clean_semicolon_syntax(answer_expression)
        entity_refs = find_entity_refs(str(cleaned_expression))
        answer_entities = {}
        for ref in entity_refs:
            value = RuleEngine._get_entity_value(entities, ref)
            if value is not None:
                answer_entities[ref] = value

        rhs_refs = None
        if ";" in str(cleaned_expression):
            parts = str(cleaned_expression).split(";", 1)
            if len(parts) == 2:
                rhs = parts[1].strip().strip('"').strip("'").strip()
                rhs_refs = find_entity_refs(rhs) if rhs else []

        if rhs_refs and all(ref in answer_entities for ref in rhs_refs):
            evaluated_answer = " ".join(str(answer_entities[ref]) for ref in rhs_refs)
        else:
            evaluated_answer = AnswerEvaluator.evaluate_answer(cleaned_expression, entities)

        payloads.append(
            {
                "question_id": question["question_id"],
                "question_type": question.get("question_type") or "unknown",
                "question": question["question"],
                "answer_expression": cleaned_expression,
                "answer_entities": answer_entities if answer_entities else None,
                "evaluated_answer": str(evaluated_answer).strip() if evaluated_answer is not None else "",
            }
        )

    return payloads


def write_generated_variant_yaml(
    output_path: Path,
    *,
    document_id: str,
    replacement_proportion: float,
    generated_document,
    entities: EntityCollection,
    question_payloads: list[dict[str, Any]],
    num_entities_replaced: int = 0,
    replaced_factual_entities: Optional[dict[str, dict[str, Any]]] = None,
    replace_mode: str,
) -> None:
    """Write one fictional document variant to the historical YAML artifact format."""
    serialized_payload = {
        "document_id": document_id,
        "document_theme": generated_document.document_theme,
        "proportion_replaced": replacement_proportion,
        "replace_mode": replace_mode,
        "num_entities_replaced": num_entities_replaced,
        "replaced_factual_entities": replaced_factual_entities or {},
        "generated_document": generated_document.generated_document,
        "questions": question_payloads,
        "entities_used": {
            "persons": {entity_id: entity.model_dump() for entity_id, entity in entities.persons.items()},
            "places": {entity_id: entity.model_dump() for entity_id, entity in entities.places.items()},
            "events": {entity_id: entity.model_dump() for entity_id, entity in entities.events.items()},
            "organizations": {entity_id: entity.model_dump() for entity_id, entity in entities.organizations.items()},
            "awards": {entity_id: entity.model_dump() for entity_id, entity in entities.awards.items()},
            "legals": {entity_id: entity.model_dump() for entity_id, entity in entities.legals.items()},
            "products": {entity_id: entity.model_dump() for entity_id, entity in entities.products.items()},
            "temporals": {entity_id: entity.model_dump() for entity_id, entity in entities.temporals.items()},
            "numbers": {entity_id: entity.model_dump() for entity_id, entity in entities.numbers.items()},
        },
    }
    with open(output_path, "w", encoding="utf-8") as file_handle:
        yaml.dump(
            serialized_payload,
            file_handle,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=float("inf"),
        )
