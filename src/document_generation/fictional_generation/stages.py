"""Paper-facing generation stages for fictional benchmark construction."""

from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Any

from src.core.answer_evaluation import AnswerEvaluator
from src.core.annotation_runtime import (
    REPLACE_MODE_ALL,
    AnnotationParser,
    RuleEngine,
    partition_generation_rules,
)
from src.core.document_schema import AnnotatedDocument, EntityCollection
from src.core.entity_taxonomy import parse_integer_surface_number, parse_word_number

from ..fictional_document_renderer import FictionalDocumentRenderer
from ..fictional_entity_sampler import FictionalEntitySampler
from ..fictional_entity_sampler_common import FictionalEntitySamplerCommonMixin
from ..generated_variant_yaml import build_generated_question_payloads, write_generated_variant_yaml
from ..pool_rule_alignment import align_pool_values_for_manual_rules
from .planning import (
    apply_sampled_fictional_entities,
    build_replacement_metadata,
    build_variant_sampler,
    plan_variant_replacements,
    targeted_decade_year_temporal_ids,
)
from .types import (
    DEBUG_SAMPLING,
    MAX_SAMPLING_ATTEMPTS_PER_VARIANT,
    FictionalGenerationContext,
    NamedEntitySample,
    NumericalEntitySample,
)

logger = logging.getLogger(__name__)
_PERSON_GENDER_LITERAL_RULE_RE = re.compile(
    r'^\s*(person_\d+)\.gender\s*(==|=)\s*["\']?(male|female|neutral)["\']?\s*$',
    re.IGNORECASE,
)


def _normalize_question_type_label(question_type: str | None) -> str:
    return str(question_type or "").strip().lower()


def _format_rule_numeric_literal(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return FictionalDocumentRenderer._format_numeric_surface(float(value))
    cleaned = str(value or "").strip()
    if not cleaned:
        return None
    parsed_integer = parse_integer_surface_number(cleaned)
    if parsed_integer is not None:
        return str(parsed_integer)
    parsed_word = parse_word_number(cleaned)
    if parsed_word is not None:
        return str(parsed_word)
    try:
        numeric = float(cleaned)
    except (TypeError, ValueError):
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return FictionalDocumentRenderer._format_numeric_surface(numeric)


def _answers_semantically_match(lhs: Any, rhs: Any) -> bool:
    lhs_literal = _format_rule_numeric_literal(lhs)
    rhs_literal = _format_rule_numeric_literal(rhs)
    if lhs_literal is not None and rhs_literal is not None:
        return lhs_literal == rhs_literal
    return str(lhs or "").strip().casefold() == str(rhs or "").strip().casefold()


def _variant_question_answers_differ_from_factual(
    *,
    context: FictionalGenerationContext,
    hybrid_entities: EntityCollection,
) -> bool:
    for question in context.generation_document.questions:
        if str(getattr(question, "answer_type", "") or "").strip().lower() != "variant":
            continue
        question_type = _normalize_question_type_label(getattr(question, "question_type", None))
        if question_type not in {"arithmetic", "temporal"}:
            continue
        answer_expression = AnswerEvaluator._clean_semicolon_syntax(str(getattr(question, "answer", "") or ""))
        if not answer_expression or answer_expression == "Cannot be determined":
            continue
        factual_answer = AnswerEvaluator.evaluate_answer(answer_expression, context.factual_entities_full)
        fictional_answer = AnswerEvaluator.evaluate_answer(answer_expression, hybrid_entities)
        if _answers_semantically_match(factual_answer, fictional_answer):
            if DEBUG_SAMPLING:
                print(
                    f"[dbg] variant {question_type} answer stayed factual for {question.question_id}: "
                    f"{factual_answer!r}",
                    flush=True,
                )
            return False
    return True


def variant_rules_hold(
    *,
    generation_document: AnnotatedDocument,
    hybrid_entities: EntityCollection,
) -> bool:
    """Validate the final entity assignment against the reviewed rules."""
    rule_results = RuleEngine.validate_all_rules(generation_document.rules, hybrid_entities)
    if all(is_valid for _rule, is_valid in rule_results):
        return True
    if DEBUG_SAMPLING:
        failed_rules = [rule for rule, is_valid in rule_results if not is_valid]
        print(f"[dbg] rule validation failed: {failed_rules}", flush=True)
    return False


def render_and_write_variant(
    *,
    context: FictionalGenerationContext,
    output_path: Path,
    document_id: str,
    replacement_proportion: float,
    replace_mode: str,
    hybrid_entities: EntityCollection,
    replacement_layout,
) -> Path:
    """Render the variant document/questions and serialize them to YAML."""
    replaced_factual_entities = build_replacement_metadata(
        hybrid_entities=hybrid_entities,
        replacement_layout=replacement_layout,
    )
    num_entities_replaced = sum(len(entity_payload) for entity_payload in replaced_factual_entities.values())

    if DEBUG_SAMPLING:
        print("[dbg] rendering fictional document", flush=True)
    generated_document = FictionalDocumentRenderer.render_document(context.generation_document, hybrid_entities)
    generated_document.evaluated_answers = AnswerEvaluator.evaluate_all_answers(
        generated_document.questions,
        hybrid_entities,
    )
    question_payloads = build_generated_question_payloads(generated_document, hybrid_entities)
    if DEBUG_SAMPLING:
        print("[dbg] writing fictional variant yaml", flush=True)
    write_generated_variant_yaml(
        output_path,
        document_id=document_id,
        replacement_proportion=replacement_proportion,
        generated_document=generated_document,
        entities=hybrid_entities,
        question_payloads=question_payloads,
        num_entities_replaced=num_entities_replaced,
        replaced_factual_entities=replaced_factual_entities,
        replace_mode=replace_mode,
    )
    return output_path


def build_fictional_generation_context(
    doc: AnnotatedDocument,
) -> FictionalGenerationContext:
    """Derive the static generation context for one reviewed template."""
    effective_rules, dropped_rules = partition_generation_rules(doc, include_questions=True)
    if dropped_rules:
        rendered = "\n".join(dropped_rules[:10])
        logger.warning(
            "Proceeding despite factual rule mismatch for %s:\n%s",
            getattr(doc, "document_id", "<unknown>"),
            rendered,
        )
    generation_document = doc.model_copy(deep=True)
    generation_document.rules = [
        *effective_rules,
        *_synthesized_birth_age_chronology_rules(generation_document),
    ]
    factual_entities_full = AnnotationParser.extract_factual_entities(generation_document, include_questions=True)
    _materialize_implicit_person_age_rules(factual_entities_full, generation_document.implicit_rules)
    _materialize_explicit_person_gender_rules(factual_entities_full, generation_document.rules)
    entity_types = {
        "persons": factual_entities_full.persons,
        "places": factual_entities_full.places,
        "events": factual_entities_full.events,
        "organizations": factual_entities_full.organizations,
        "awards": factual_entities_full.awards,
        "legals": factual_entities_full.legals,
        "products": factual_entities_full.products,
        "numbers": factual_entities_full.numbers,
        "temporals": factual_entities_full.temporals,
    }
    return FictionalGenerationContext(
        source_document=doc,
        generation_document=generation_document,
        required_entities=FictionalEntitySampler.extract_required_entities(
            generation_document,
            include_questions=True,
        ),
        factual_entities_full=factual_entities_full,
        entity_types=entity_types,
        dropped_rules=tuple(dropped_rules),
    )


def _materialize_explicit_person_gender_rules(entities: EntityCollection, rules: list[str]) -> None:
    for raw_rule in rules or []:
        cleaned = str(raw_rule).split("#", 1)[0].strip()
        match = _PERSON_GENDER_LITERAL_RULE_RE.fullmatch(cleaned)
        if not match:
            continue
        person_id = str(match.group(1) or "").strip()
        gender = str(match.group(3) or "").strip().lower()
        person_entity = entities.persons.get(person_id)
        if person_entity is None:
            continue
        FictionalEntitySamplerCommonMixin._apply_gender_profile(person_entity, gender)


def _materialize_implicit_person_age_rules(entities: EntityCollection, implicit_rules: list[Any]) -> None:
    for rule in implicit_rules or []:
        entity_ref = str(
            (rule.get("entity_ref") if isinstance(rule, dict) else getattr(rule, "entity_ref", ""))
            or ""
        ).strip()
        if not entity_ref.endswith(".age") or not entity_ref.startswith("person_"):
            continue
        person_id = entity_ref.split(".", 1)[0]
        person_entity = entities.persons.get(person_id)
        if person_entity is None:
            continue
        try:
            factual_value = rule.get("factual_value") if isinstance(rule, dict) else getattr(rule, "factual_value")
            person_entity.age = int(round(float(factual_value)))
        except (TypeError, ValueError):
            continue


def _synthesized_birth_age_chronology_rules(doc: AnnotatedDocument) -> list[str]:
    text = doc.document_to_annotate or ""
    annotations = AnnotationParser.parse_annotations(text)
    birth_temporal_id = None
    for ann in annotations:
        if not str(ann.entity_id).startswith("temporal_"):
            continue
        if ann.attribute not in {"year", "date"}:
            continue
        preceding = text[max(0, ann.start_pos - 32) : ann.start_pos].lower()
        if "born" in preceding:
            birth_temporal_id = ann.entity_id
            break
    if birth_temporal_id is None:
        return []

    age_annotations = []
    for ann in annotations:
        if ann.attribute != "age" or not str(ann.entity_id).startswith("person_"):
            continue
        try:
            original_age = int(str(ann.original_text or "").strip())
        except ValueError:
            continue
        age_annotations.append((ann, original_age))
    if not age_annotations:
        return []

    anchor_age_by_person: dict[str, int] = {}
    for ann, original_age in age_annotations:
        anchor_age_by_person[ann.entity_id] = min(original_age, anchor_age_by_person.get(ann.entity_id, original_age))

    rules: list[str] = []
    for age_ann, original_age in age_annotations:
        sentence_end = text.find(".", age_ann.end_pos)
        if sentence_end < 0:
            sentence_end = len(text)
        temporal_ann = next(
            (
                ann
                for ann in annotations
                if ann.start_pos > age_ann.end_pos
                and ann.start_pos <= sentence_end
                and str(ann.entity_id).startswith("temporal_")
                and ann.attribute in {"year", "date"}
            ),
            None,
        )
        if temporal_ann is None:
            continue
        age_delta = original_age - anchor_age_by_person[age_ann.entity_id]
        rhs = f"{age_ann.entity_id}.age" if age_delta == 0 else f"{age_ann.entity_id}.age + {age_delta}"
        rules.append(f"{temporal_ann.entity_id}.year - {birth_temporal_id}.year == {rhs}")
    return rules


def generate_named_entities(
    *,
    context: FictionalGenerationContext,
    entity_pool: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    """Materialize the paper's ``GenerateNamedEntities`` stage for one template."""
    aligned_entity_pool = align_pool_values_for_manual_rules(
        entity_pool,
        context.generation_document.rules,
    )
    pool_sampler = FictionalEntitySampler(
        aligned_entity_pool,
        seed=seed,
        factual_entities=context.factual_entities_full,
        implicit_rules=context.generation_document.implicit_rules,
    )
    manual_pool_shortages = pool_sampler.find_manual_pool_shortages(context.required_entities)
    if manual_pool_shortages:
        rendered_shortages = "\n".join(manual_pool_shortages[:12])
        raise ValueError(
            "Entity pool cannot satisfy the required manual attributes for this document:\n"
            f"{rendered_shortages}"
        )
    return aligned_entity_pool


def sample_named_entities(
    *,
    context: FictionalGenerationContext,
    named_entities: dict[str, Any],
    replacement_proportion: float,
    version_seed: int,
    replace_mode: str = REPLACE_MODE_ALL,
    eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None = None,
    reference_variant_index: int | None = None,
    reference_variant_count: int | None = None,
) -> NamedEntitySample | None:
    """Sample named entities for one fictional variant."""
    if DEBUG_SAMPLING:
        print(
            f"[dbg] start sample_named_entities seed={version_seed} p={replacement_proportion}",
            flush=True,
        )

    _replacement_plan, replacement_layout, fictional_requirements = plan_variant_replacements(
        context=context,
        replacement_proportion=replacement_proportion,
        replace_mode=replace_mode,
        rng=random.Random(version_seed),
    )
    hybrid_entities = replacement_layout.initial_hybrid_entities
    if not fictional_requirements or not any(fictional_requirements.values()):
        return NamedEntitySample(
            entities=hybrid_entities,
            replacement_layout=replacement_layout,
            fictional_requirements=fictional_requirements,
            decade_year_temporal_ids=frozenset(),
        )

    if DEBUG_SAMPLING:
        print(
            f"[dbg] sampling entities (full/partial): "
            f"{sum(len(specs) for specs in fictional_requirements.values())} specs",
            flush=True,
        )

    sampler = build_variant_sampler(
        context=context,
        entity_pool=named_entities,
        replacement_layout=replacement_layout,
        version_seed=version_seed,
        eligible_cache=eligible_cache,
        reference_variant_index=reference_variant_index,
        reference_variant_count=reference_variant_count,
    )
    named_requirements = {
        entity_type: specs
        for entity_type, specs in fictional_requirements.items()
        if entity_type in set(FictionalEntitySampler._MANUAL_ENTITY_TYPES)
    }
    if named_requirements:
        sampled_named_entities = sampler.sample_named_entities(
            required_entities=named_requirements,
            rules=context.generation_document.rules,
        )
        if sampled_named_entities is None:
            if DEBUG_SAMPLING:
                print("[dbg] named entity sampler returned None", flush=True)
            return None
        apply_sampled_fictional_entities(
            hybrid_entities=hybrid_entities,
            sampled_fictional_entities=sampled_named_entities,
            partial_replacements=replacement_layout.partially_replaced_entities,
        )

    return NamedEntitySample(
        entities=hybrid_entities,
        replacement_layout=replacement_layout,
        fictional_requirements=fictional_requirements,
        decade_year_temporal_ids=targeted_decade_year_temporal_ids(
            context=context,
            fictional_requirements=fictional_requirements,
        ),
    )


def generate_numerical_entities(
    *,
    context: FictionalGenerationContext,
    named_entity_sample: NamedEntitySample,
    named_entities: dict[str, Any],
    version_seed: int,
    eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None = None,
    reference_variant_index: int | None = None,
    reference_variant_count: int | None = None,
    used_number_values_by_id: dict[str, set[int | float]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
    allow_relaxed_intervariant_number_reuse: bool = False,
) -> NumericalEntitySample | None:
    """Generate numerical entities for one fictional variant."""
    sampler = build_variant_sampler(
        context=context,
        entity_pool=named_entities,
        replacement_layout=named_entity_sample.replacement_layout,
        version_seed=version_seed,
        eligible_cache=eligible_cache,
        reference_variant_index=reference_variant_index,
        reference_variant_count=reference_variant_count,
        used_number_values_by_id=used_number_values_by_id,
        used_temporal_years_by_id=used_temporal_years_by_id,
        used_temporal_values_by_id=used_temporal_values_by_id,
        allow_relaxed_intervariant_number_reuse=allow_relaxed_intervariant_number_reuse,
    )
    numerical_entities = sampler.generate_numerical_entities(
        required_entities=named_entity_sample.fictional_requirements,
        rules=context.generation_document.rules,
        named_entities=named_entity_sample.entities,
        max_attempts=MAX_SAMPLING_ATTEMPTS_PER_VARIANT,
        decade_year_temporal_ids=set(named_entity_sample.decade_year_temporal_ids),
    )
    if numerical_entities is None:
        if DEBUG_SAMPLING:
            print("[dbg] numerical entity generation returned None", flush=True)
        logger.warning(
            "Numerical entity generation failed for %s; refusing factual fallback.",
            context.source_document.document_id,
        )
        return None
    return NumericalEntitySample(entities=numerical_entities)


def replace_factual_entities(
    *,
    context: FictionalGenerationContext,
    named_entity_sample: NamedEntitySample,
    numerical_entity_sample: NumericalEntitySample,
    output_path: Path,
    document_id: str,
    replacement_proportion: float,
    replace_mode: str = REPLACE_MODE_ALL,
) -> Path | None:
    """Apply the paper's ``Replace`` step to named and numerical entities."""
    hybrid_entities = named_entity_sample.entities.model_copy(deep=True)
    hybrid_entities.merge_from(numerical_entity_sample.entities)

    if DEBUG_SAMPLING:
        print("[dbg] validating variant rules", flush=True)
    if not variant_rules_hold(
        generation_document=context.generation_document,
        hybrid_entities=hybrid_entities,
    ):
        logger.warning("Variant rule validation failed for %s; writing variant anyway.", document_id)
    if not _variant_question_answers_differ_from_factual(
        context=context,
        hybrid_entities=hybrid_entities,
    ):
        logger.warning("Variant answer-difference check failed for %s; writing variant anyway.", document_id)

    return render_and_write_variant(
        context=context,
        output_path=output_path,
        document_id=document_id,
        replacement_proportion=replacement_proportion,
        replace_mode=replace_mode,
        hybrid_entities=hybrid_entities,
        replacement_layout=named_entity_sample.replacement_layout,
    )

__all__ = [
    "build_fictional_generation_context",
    "generate_named_entities",
    "generate_numerical_entities",
    "replace_factual_entities",
    "sample_named_entities",
]
