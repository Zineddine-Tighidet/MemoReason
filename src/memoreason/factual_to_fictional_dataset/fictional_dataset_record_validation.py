"""Final scientific-contract validation for one generated fictional document."""

from __future__ import annotations

from typing import Any

from memoreason.benchmark_definition.annotation_runtime import (
    RuleEngine,
    find_rule_sanity_errors,
    partition_generation_rules,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_renderer import (
    FictionalDocumentRenderer,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.generated_variant_yaml import (
    build_generated_question_payloads,
)

from .dataset_record_contracts import (
    _contains_inline_annotations,
)
from .dataset_record_core import _deserialize_entities
from .dataset_record_semantic_validation import _semantic_payload_issues
from .dataset_settings import FactualToFictionalDatasetSetting


def _verify_generated_payload(
    *,
    original_document,
    render_source_document,
    generated_payload: dict[str, Any],
    setting_spec: FactualToFictionalDatasetSetting,
    source_mode: str,
) -> None:
    entities = _deserialize_entities(generated_payload.get("entities_used") or {})
    rendered_document = FictionalDocumentRenderer.render_document(render_source_document, entities)
    expected_questions = build_generated_question_payloads(rendered_document, entities)
    actual_document_text = str(generated_payload.get("generated_document") or "")
    if not actual_document_text:
        raise ValueError("Generated payload has empty document text.")
    if actual_document_text != rendered_document.generated_document:
        raise ValueError("Generated payload document_text drifted from a fresh render of the source template.")
    if _contains_inline_annotations(actual_document_text):
        raise ValueError("Generated payload still contains inline annotation syntax in document_text.")

    question_entries = generated_payload.get("questions") or []
    if len(question_entries) != len(expected_questions):
        raise ValueError(
            f"Generated payload has {len(question_entries)} questions but {len(expected_questions)} were expected."
        )

    actual_by_id = {str(entry.get("question_id") or ""): entry for entry in question_entries}
    expected_by_id = {str(entry["question_id"]): entry for entry in expected_questions}
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError("Generated payload question IDs do not match the template question IDs.")

    for question_id, expected_entry in expected_by_id.items():
        actual_entry = actual_by_id[question_id]
        actual_question_text = str(actual_entry.get("question") or "")
        if not actual_question_text:
            raise ValueError(f"{question_id}: generated question text is empty.")
        if actual_question_text != str(expected_entry["question"]):
            raise ValueError(f"{question_id}: question text drifted from rendered output.")
        if _contains_inline_annotations(actual_question_text):
            raise ValueError(f"{question_id}: generated question text still contains annotation syntax.")

        actual_expr = str(actual_entry.get("answer_expression") or "").strip()
        expected_expr = str(expected_entry["answer_expression"] or "").strip()
        if actual_expr != expected_expr:
            raise ValueError(f"{question_id}: answer_expression drifted from the rendered question payload.")

        actual_answer_entities = actual_entry.get("answer_entities")
        expected_answer_entities = expected_entry["answer_entities"]
        if actual_answer_entities != expected_answer_entities:
            raise ValueError(f"{question_id}: answer_entities drifted from the evaluated entity refs.")

        actual_evaluated = str(actual_entry.get("evaluated_answer") or "").strip()
        expected_evaluated = str(expected_entry["evaluated_answer"] or "").strip()
        if actual_evaluated != expected_evaluated:
            raise ValueError(f"{question_id}: evaluated_answer drifted from recomputation.")

        if expected_expr and not actual_evaluated:
            raise ValueError(f"{question_id}: evaluated_answer resolved to an empty string.")

    all_rules = [str(rule) for rule in (original_document.rules or []) if str(rule).split("#", 1)[0].strip()]
    effective_rules, _dropped_rules = partition_generation_rules(original_document, include_questions=True)
    sanity_errors = find_rule_sanity_errors(all_rules)
    if sanity_errors:
        raise ValueError("\n".join(sanity_errors))
    rule_results = RuleEngine.validate_all_rules(effective_rules, entities)
    failed_rules = [rule for rule, is_valid in rule_results if not is_valid]
    if failed_rules:
        rendered_failed = "\n".join(failed_rules[:10])
        raise ValueError(f"Generated payload violates document rules:\n{rendered_failed}")

    semantic_issues = _semantic_payload_issues(generated_payload, source_document=original_document)
    if semantic_issues:
        rendered = "\n".join(semantic_issues[:10])
        raise ValueError(f"Generated payload failed semantic linting:\n{rendered}")

    if source_mode not in {
        "pool",
        "derived_from_full_fictional",
    }:
        raise ValueError(f"Unknown fictional generation source mode: {source_mode!r}")
