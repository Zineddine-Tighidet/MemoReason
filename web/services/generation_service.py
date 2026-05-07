"""Service for generating fictional document previews from annotated templates."""

from __future__ import annotations

import ast
import logging
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.dataset_export.fictional_dataset import generate_fictional_dataset_payload
from src.dataset_export.dataset_settings import fictional_setting
from src.document_generation.fictional_entity_sampler import FictionalEntitySampler
from src.core.document_schema import (
    AnnotatedDocument,
    AwardEntity,
    EntityCollection,
    EventEntity,
    ImplicitRule,
    LegalEntity,
    NumberEntity,
    OrganizationEntity,
    PersonEntity,
    PlaceEntity,
    ProductEntity,
    Question,
    TemporalEntity,
)
from src.core.annotation_runtime import (
    AnnotationParser,
    RuleEngine,
    find_entity_refs,
    is_valid_entity_ref,
    normalize_document_taxonomy,
    normalize_implicit_rules_for_storage,
    normalize_rule_expressions,
)
from web.services.entity_pool_service import get_or_generate_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreviewGenerationContext:
    """Prepared immutable context reused across multiple preview generations."""
    doc: AnnotatedDocument
    required_entities: Dict[str, List[Tuple[str, List[str]]]]
    factual_entities: EntityCollection
    referenced_entity_refs: List[str]
    entity_pool: Dict[str, Any]
    pool_source: str
    pool_path: Path | None

def _build_annotated_document(doc_data: Dict[str, Any]) -> AnnotatedDocument:
    """Build an AnnotatedDocument from a raw doc_data dict (as returned by the API)."""
    doc_data = normalize_document_taxonomy(doc_data)
    questions = []
    for q in doc_data.get("questions", []):
        raw_answer = q.get("answer", "")
        if raw_answer is True:
            answer_str = "Yes"
        elif raw_answer is False:
            answer_str = "No"
        elif isinstance(raw_answer, list) and len(raw_answer) == 1:
            answer_str = str(raw_answer[0])
        else:
            answer_str = str(raw_answer)
        raw_answer_type = q.get("answer_type")
        if raw_answer_type is None:
            invariant_flag = q.get("is_answer_invariant")
            if invariant_flag is True:
                raw_answer_type = "invariant"
            elif invariant_flag is False:
                raw_answer_type = "variant"
        questions.append(Question(
            question_id=q.get("question_id", ""),
            question=q.get("question", ""),
            answer=answer_str,
            question_type=q.get("question_type"),
            answer_type=raw_answer_type,
            reasoning_chain=[
                str(step).strip()
                for step in (q.get("reasoning_chain") or [])
                if str(step).strip()
            ],
        ))
    return AnnotatedDocument(
        document_id=doc_data.get("document_id", ""),
        document_theme=doc_data.get("document_theme", ""),
        original_document=doc_data.get("original_document", ""),
        document_to_annotate=doc_data.get("document_to_annotate", ""),
        fictionalized_annotated_template_document=doc_data.get(
            "fictionalized_annotated_template_document", ""
        ),
        rules=normalize_rule_expressions(doc_data.get("rules", [])) or [],
        questions=questions,
        implicit_rules=[
            ImplicitRule(**rule_data)
            for rule_data in (normalize_implicit_rules_for_storage(doc_data.get("implicit_rules", [])) or [])
        ],
    )


def _build_entity_mapping(
    factual: EntityCollection,
    fictional: EntityCollection,
    referenced_entity_refs: List[str],
) -> List[Dict[str, str]]:
    """Build {entity_id, attribute, factual, fictional} for actually annotated refs only."""
    mapping: List[Dict[str, str]] = []
    seen = set()
    for ref in referenced_entity_refs:
        if ref in seen:
            continue
        seen.add(ref)
        if "." in ref:
            entity_id, attribute = ref.split(".", 1)
        else:
            entity_id, attribute = ref, ""
        factual_val = RuleEngine._get_entity_value(factual, ref)
        fictional_val = RuleEngine._get_entity_value(fictional, ref)
        if factual_val is None and fictional_val is None:
            continue
        factual_str = "" if factual_val is None else str(factual_val)
        fictional_str = "" if fictional_val is None else str(fictional_val)
        # Keep only actual replacements.
        if factual_str == fictional_str:
            continue
        mapping.append({
            "entity_id": entity_id,
            "attribute": attribute,
            "factual": factual_str,
            "fictional": fictional_str,
        })

    mapping.sort(key=lambda m: (m["entity_id"], m["attribute"]))
    return mapping


def _collect_annotated_entity_refs(doc: AnnotatedDocument) -> List[str]:
    """Collect ordered entity refs that are explicitly annotated in doc/questions."""
    refs: List[str] = []
    for ann in AnnotationParser.parse_annotations(doc.document_to_annotate):
        if ann.attribute:
            refs.append(f"{ann.entity_id}.{ann.attribute}")
    for q in doc.questions:
        for ann in AnnotationParser.parse_annotations(q.question):
            if ann.attribute:
                refs.append(f"{ann.entity_id}.{ann.attribute}")
        refs.extend(find_entity_refs(q.question))
    return refs


def _build_entity_collection_from_generated_values(entities_dict: Dict[str, Any]) -> EntityCollection:
    """Rebuild an EntityCollection from serialized `entities_used` data."""
    entities = EntityCollection()

    for person_id, person_data in (entities_dict.get("persons") or {}).items():
        if isinstance(person_data, dict):
            entities.persons[person_id] = PersonEntity(**person_data)

    for place_id, place_data in (entities_dict.get("places") or {}).items():
        if isinstance(place_data, dict):
            entities.places[place_id] = PlaceEntity(**place_data)

    for event_id, event_data in (entities_dict.get("events") or {}).items():
        if isinstance(event_data, dict):
            entities.events[event_id] = EventEntity(**event_data)

    for organization_id, organization_data in (entities_dict.get("organizations") or {}).items():
        if isinstance(organization_data, dict):
            entities.organizations[organization_id] = OrganizationEntity(**organization_data)

    for award_id, award_data in (entities_dict.get("awards") or {}).items():
        if isinstance(award_data, dict):
            entities.awards[award_id] = AwardEntity(**award_data)

    for legal_id, legal_data in (entities_dict.get("legals") or {}).items():
        if isinstance(legal_data, dict):
            entities.legals[legal_id] = LegalEntity(**legal_data)

    for product_id, product_data in (entities_dict.get("products") or {}).items():
        if isinstance(product_data, dict):
            entities.products[product_id] = ProductEntity(**product_data)

    for temporal_id, temporal_data in (entities_dict.get("temporals") or {}).items():
        if isinstance(temporal_data, dict):
            entities.temporals[temporal_id] = TemporalEntity(**temporal_data)

    for number_id, number_data in (entities_dict.get("numbers") or {}).items():
        if isinstance(number_data, dict):
            entities.numbers[number_id] = NumberEntity(**number_data)

    return entities


def _build_annotation_diff_entries(
    annotated_text: str,
    entity_mapping_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Return annotation spans paired with their fictional replacements."""
    rows: List[Dict[str, Any]] = []
    for ann in sorted(AnnotationParser.parse_annotations(annotated_text), key=lambda item: item.start_pos):
        ref = f"{ann.entity_id}.{ann.attribute}" if ann.attribute else ann.entity_id
        mapping = entity_mapping_lookup.get(ref)
        if mapping is None:
            continue
        rows.append(
            {
                "start": ann.start_pos,
                "end": ann.end_pos,
                "entity_id": ann.entity_id,
                "attribute": ann.attribute,
                "factual_text": ann.original_text,
                "fictional_text": mapping["fictional"],
            }
        )
    return rows


def _clean_answer_expression(answer_expr: str) -> str:
    """Normalize stored answer expression into evaluable syntax."""
    if not isinstance(answer_expr, str):
        return ""
    expr = answer_expr.strip()
    if not expr:
        return ""

    if expr.startswith("[") and expr.endswith("]"):
        try:
            parsed = ast.literal_eval(expr)
            if isinstance(parsed, list) and parsed:
                expr = str(parsed[0])
        except (ValueError, SyntaxError):
            expr = expr[1:-1].strip()
            if (expr.startswith("'") and expr.endswith("'")) or (expr.startswith('"') and expr.endswith('"')):
                expr = expr[1:-1]

    # Inline annotation syntax: [text; entity_1.attr] -> entity_1.attr
    if "[" in expr and ";" in expr:
        cleaned = re.sub(r"\[([^\]]+);\s*([^\]]+)\]", r"\2", expr)
        if cleaned != expr:
            expr = cleaned.strip()

    # Composite syntax: literal ; "ref1 ref2"
    if ";" in expr:
        literal, refs_part = expr.split(";", 1)
        literal = literal.strip()
        refs_part = refs_part.strip()
        if is_valid_entity_ref(refs_part):
            return refs_part
        stripped = refs_part.strip('"').strip("'").strip()
        if stripped and find_entity_refs(stripped):
            return stripped
        return literal

    return expr

def _resolve_generated_questions(
    generated_questions: List[Dict[str, Any]],
    factual_entities: EntityCollection,
    fictional_entities: EntityCollection,
) -> List[Dict[str, Any]]:
    """Attach normalized answer-expression metadata for preview rendering."""
    counts = Counter(str(q.get("question_id", "")) for q in generated_questions)
    seen: Dict[str, int] = {}
    resolved: List[Dict[str, Any]] = []

    for idx, q in enumerate(generated_questions):
        row = dict(q)
        qid = str(row.get("question_id", ""))
        seen[qid] = seen.get(qid, 0) + 1

        raw_expr = str(row.get("answer_expression") or row.get("answer") or "")
        answer_expr = _clean_answer_expression(raw_expr)
        refs = find_entity_refs(answer_expr)
        answer_entities: List[Dict[str, str]] = []
        for ref in refs:
            factual_val = RuleEngine._get_entity_value(factual_entities, ref)
            fictional_val = RuleEngine._get_entity_value(fictional_entities, ref)
            answer_entities.append({
                "ref": ref,
                "factual": "" if factual_val is None else str(factual_val),
                "fictional": "" if fictional_val is None else str(fictional_val),
            })

        row["question_ui_key"] = f"{qid}::{idx}"
        row["question_id_display"] = qid if counts[qid] <= 1 else f"{qid}#{seen[qid]}"
        row["answer_expression"] = answer_expr
        row["fictional_answer"] = str(row.get("evaluated_answer") or "")
        row["answer_entities"] = answer_entities
        resolved.append(row)
    return resolved


def _build_rules_evaluation(
    doc: AnnotatedDocument,
    fictional_entities: EntityCollection,
) -> Dict[str, Any]:
    """Evaluate explicit and implicit rules for the preview payload."""
    rows: List[Dict[str, Any]] = []

    for rule_text, satisfied in RuleEngine.validate_all_rules(doc.rules, fictional_entities):
        rows.append(
            {
                "rule": str(rule_text),
                "source": "explicit",
                "satisfied": bool(satisfied),
            }
        )

    for implicit_rule in doc.implicit_rules:
        entity_ref = str(implicit_rule.entity_ref or "").strip()
        value = RuleEngine._get_entity_value(fictional_entities, entity_ref)
        try:
            numeric_value = float(value)
            lower_bound = float(implicit_rule.lower_bound)
            upper_bound = float(implicit_rule.upper_bound)
            satisfied = lower_bound <= numeric_value <= upper_bound
        except (TypeError, ValueError):
            satisfied = False
        rows.append(
            {
                "rule": f"{implicit_rule.lower_bound} <= {entity_ref} <= {implicit_rule.upper_bound}",
                "source": "implicit",
                "rule_kind": implicit_rule.rule_kind,
                "entity_ref": entity_ref,
                "lower_bound": implicit_rule.lower_bound,
                "upper_bound": implicit_rule.upper_bound,
                "satisfied": bool(satisfied),
            }
        )

    satisfied_rows = [row for row in rows if row["satisfied"]]
    failed_rows = [row for row in rows if not row["satisfied"]]
    return {
        "counts": {
            "total": len(rows),
            "satisfied": len(satisfied_rows),
            "failed": len(failed_rows),
        },
        "satisfied": satisfied_rows,
        "failed": failed_rows,
        "all": rows,
    }


def _build_preview_generation_context(
    doc_data: Dict[str, Any],
    *,
    seed: int,
    theme_id: str | None = None,
) -> PreviewGenerationContext:
    """Build and cache generation prerequisites shared across preview attempts."""
    doc = _build_annotated_document(doc_data)
    required_entities = FictionalEntitySampler.extract_required_entities(doc, include_questions=False)
    entity_pool, pool_source, pool_path = get_or_generate_pool(
        doc,
        required_entities,
        seed,
        theme_id=theme_id,
    )
    factual_entities = AnnotationParser.extract_factual_entities(doc, include_questions=False)
    referenced_entity_refs = _collect_annotated_entity_refs(doc)
    return PreviewGenerationContext(
        doc=doc,
        required_entities=required_entities,
        factual_entities=factual_entities,
        referenced_entity_refs=referenced_entity_refs,
        entity_pool=entity_pool,
        pool_source=pool_source,
        pool_path=pool_path,
    )


def _generate_payload_from_context(
    context: PreviewGenerationContext,
    *,
    seed: int,
    output_path: Path,
) -> tuple[dict[str, Any], int]:
    """Run one fictional generation pass using precomputed context."""
    generated_payload, successful_seed = generate_fictional_dataset_payload(
        context.doc,
        setting_spec=fictional_setting(1.0),
        entity_pool=context.entity_pool,
        seed=seed,
        output_path=output_path,
    )
    return generated_payload, successful_seed


def _build_preview_result(
    context: PreviewGenerationContext,
    generated_payload: dict[str, Any],
    *,
    successful_seed: int,
) -> Dict[str, Any]:
    """Assemble one UI preview payload from generated benchmark data."""
    fictional_entities = _build_entity_collection_from_generated_values(generated_payload.get("entities_used", {}))
    entity_mapping = _build_entity_mapping(
        context.factual_entities,
        fictional_entities,
        context.referenced_entity_refs,
    )
    entity_mapping_lookup = {
        f"{item['entity_id']}.{item['attribute']}" if item["attribute"] else item["entity_id"]: item
        for item in entity_mapping
    }

    annotation_info = _build_annotation_diff_entries(context.doc.document_to_annotate, entity_mapping_lookup)
    question_annotations = [
        _build_annotation_diff_entries(question.question, entity_mapping_lookup)
        for question in context.doc.questions
    ]
    resolved_questions = _resolve_generated_questions(
        list(generated_payload.get("questions") or []),
        context.factual_entities,
        fictional_entities,
    )

    return {
        "generated_document": str(generated_payload.get("generated_document") or ""),
        "generated_questions": resolved_questions,
        "entity_mapping": entity_mapping,
        "annotations": annotation_info,
        "question_annotations": question_annotations,
        "rules_evaluation": _build_rules_evaluation(context.doc, fictional_entities),
        "seed": successful_seed,
        "pool_source": str(context.pool_path) if context.pool_path else context.pool_source,
    }


def _preview_signature(preview: Dict[str, Any]) -> str:
    """Stable signature used to deduplicate near-identical generated previews."""
    mapping = preview.get("entity_mapping")
    if not isinstance(mapping, list):
        return repr(preview)
    rows: list[str] = []
    for row in mapping:
        if not isinstance(row, dict):
            continue
        entity_id = str(row.get("entity_id") or "")
        attribute = str(row.get("attribute") or "")
        fictional_value = str(row.get("fictional") or "")
        rows.append(f"{entity_id}.{attribute}={fictional_value}")
    rows.sort()
    return "|".join(rows)


def generate_fictional_preview(
    doc_data: Dict[str, Any],
    seed: int = 42,
    theme_id: str | None = None,
) -> Dict[str, Any]:
    """Generate a fictional document preview from annotated template data.

    Args:
        doc_data: Document data dict (as stored in YAML / returned by API).
        seed: Fixed seed for reproducibility.

    Returns:
        Dict with keys:
          - generated_document: The fictional document text.
          - generated_questions: Questions with fictional text.
          - entity_mapping: List of {entity_id, attribute, factual, fictional}.
          - annotations: List of {start, end, entity_id, attribute, factual_text}
            from the annotated source (for building the diff view).
    """
    context = _build_preview_generation_context(
        doc_data,
        seed=seed,
        theme_id=theme_id,
    )
    with tempfile.TemporaryDirectory(prefix="benchmark_preview_") as tmpdir:
        generated_payload, successful_seed = _generate_payload_from_context(
            context,
            seed=seed,
            output_path=Path(tmpdir) / f"{context.doc.document_id or 'preview'}.yaml",
        )
    return _build_preview_result(
        context,
        generated_payload,
        successful_seed=successful_seed,
    )


def generate_fictional_previews_batch(
    doc_data: Dict[str, Any],
    *,
    seed: int = 23,
    target_version_count: int = 10,
    max_attempts: int = 60,
    seed_stride: int = 1_000_003,
    theme_id: str | None = None,
) -> Dict[str, Any]:
    """Generate multiple unique fictional previews in one backend call."""
    if int(target_version_count) < 1:
        raise ValueError(f"target_version_count must be >= 1, got {target_version_count!r}.")
    if int(max_attempts) < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}.")
    if int(seed_stride) < 1:
        raise ValueError(f"seed_stride must be >= 1, got {seed_stride!r}.")

    context = _build_preview_generation_context(
        doc_data,
        seed=seed,
        theme_id=theme_id,
    )

    versions: list[Dict[str, Any]] = []
    seen_signatures: set[str] = set()
    attempts = 0

    with tempfile.TemporaryDirectory(prefix="benchmark_preview_batch_") as tmpdir:
        output_path = Path(tmpdir) / f"{context.doc.document_id or 'preview'}.yaml"
        while len(versions) < int(target_version_count) and attempts < int(max_attempts):
            candidate_seed = int(seed) + (attempts * int(seed_stride))
            generated_payload, successful_seed = _generate_payload_from_context(
                context,
                seed=candidate_seed,
                output_path=output_path,
            )
            preview = _build_preview_result(
                context,
                generated_payload,
                successful_seed=successful_seed,
            )
            signature = _preview_signature(preview)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                versions.append(preview)
            attempts += 1

    if not versions:
        raise RuntimeError("No fictional preview could be generated.")

    return {
        "versions": versions,
        "generated_version_count": len(versions),
        "target_version_count": int(target_version_count),
        "attempt_count": attempts,
    }
