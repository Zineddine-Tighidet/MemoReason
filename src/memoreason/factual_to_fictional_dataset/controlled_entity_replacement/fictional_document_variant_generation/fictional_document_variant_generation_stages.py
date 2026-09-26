"""Generate and validate one fictional document variant using the paper algorithm."""

from __future__ import annotations

import ast
import logging
import random
import re
from pathlib import Path
from typing import Any

from memoreason.benchmark_definition.annotation_runtime import (
    REPLACE_MODE_ALL,
    AnnotationParser,
    RuleEngine,
    find_entity_refs,
    partition_generation_rules,
)
from memoreason.benchmark_definition.answer_expression_evaluation import AnswerEvaluator
from memoreason.benchmark_definition.document_schema import AnnotatedDocument, EntityCollection
from memoreason.benchmark_definition.entity_taxonomy import parse_integer_surface_number, parse_word_number

from ..fictional_document_renderer import FictionalDocumentRenderer
from ..fictional_entity_sampler import FictionalEntitySampler
from ..generated_variant_yaml import build_generated_question_payloads, write_generated_variant_yaml
from ..named_entity_pool_rule_policy import (
    copy_named_entity_pool_without_rule_based_mutation,
)
from .fictional_document_variant_data_contracts import (
    DEBUG_SAMPLING,
    MAX_SAMPLING_ATTEMPTS_PER_VARIANT,
    ControlledEntityReplacementContext,
    NamedEntitySample,
    NumericalEntitySample,
)
from .fictional_document_variant_planning import (
    apply_sampled_fictional_entities,
    build_replacement_metadata,
    build_variant_sampler,
    plan_variant_replacements,
    targeted_decade_year_temporal_ids,
)

logger = logging.getLogger(__name__)

_QUOTED_LITERAL_RULE_PATTERN = re.compile(
    r"^\s*(?P<ref>[a-z]+_\d+\.[A-Za-z_][\w]*)\s*(?:==|=)\s*"
    r"(?P<literal>\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')\s*$"
)


def _materialize_missing_literal_rule_values(
    *,
    entities: EntityCollection,
    rules: list[str],
) -> None:
    """Populate unannotated direct attributes fixed by reviewed literal rules."""
    for raw_rule in rules or []:
        cleaned = str(raw_rule or "").split("#", 1)[0].strip()
        match = _QUOTED_LITERAL_RULE_PATTERN.fullmatch(cleaned)
        if match is None:
            continue
        entity_ref = match.group("ref")
        if RuleEngine._get_entity_value(entities, entity_ref) is not None:
            continue
        entity_id, attribute = entity_ref.split(".", 1)
        entity_type = entity_id.split("_", 1)[0]
        literal_value = ast.literal_eval(match.group("literal"))
        if not entities.update_entity_attribute(entity_type, entity_id, attribute, literal_value):
            raise ValueError(
                f"Reviewed literal rule references an entity absent from the annotated document: {cleaned}"
            )


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


def _variant_question_answer_collisions(
    *,
    context: ControlledEntityReplacementContext,
    hybrid_entities: EntityCollection,
    replaced_entity_ids: set[str],
    replace_mode: str,
) -> list[tuple[str, str, str]]:
    collisions: list[tuple[str, str, str]] = []
    for question in context.generation_document.questions:
        if str(getattr(question, "answer_type", "") or "").strip().lower() != "variant":
            continue
        answer_expression = AnswerEvaluator._clean_semicolon_syntax(str(getattr(question, "answer", "") or ""))
        if not answer_expression:
            continue
        answer_ref_ids = {entity_ref.split(".", 1)[0] for entity_ref in find_entity_refs(answer_expression)}
        if replace_mode != REPLACE_MODE_ALL and not answer_ref_ids.intersection(replaced_entity_ids):
            continue
        factual_answer = AnswerEvaluator.evaluate_answer(answer_expression, context.factual_entities_full)
        fictional_answer = AnswerEvaluator.evaluate_answer(answer_expression, hybrid_entities)
        if _answers_semantically_match(factual_answer, fictional_answer):
            collisions.append(
                (
                    str(getattr(question, "question_id", "") or ""),
                    answer_expression,
                    str(factual_answer or "").strip(),
                )
            )
    return collisions


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
    context: ControlledEntityReplacementContext,
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


def build_controlled_entity_replacement_context(
    doc: AnnotatedDocument,
) -> ControlledEntityReplacementContext:
    """Derive the static generation context for one reviewed template."""
    effective_rules, dropped_rules = partition_generation_rules(doc, include_questions=True)
    if dropped_rules:
        rendered = "\n".join(dropped_rules[:10])
        raise ValueError(
            "Reviewed generation rules do not hold on the factual source document "
            f"{getattr(doc, 'document_id', '<unknown>')}:\n{rendered}"
        )
    generation_document = doc.model_copy(deep=True)
    generation_document.rules = list(effective_rules)
    factual_entities_full = AnnotationParser.extract_factual_entities(generation_document, include_questions=True)
    _materialize_missing_literal_rule_values(
        entities=factual_entities_full,
        rules=generation_document.rules,
    )
    failed_enriched_rules = [
        rule
        for rule, is_valid in RuleEngine.validate_all_rules(generation_document.rules, factual_entities_full)
        if not is_valid
    ]
    if failed_enriched_rules:
        rendered = "\n".join(failed_enriched_rules[:10])
        raise ValueError(f"Reviewed generation rules do not hold on the enriched factual source document:\n{rendered}")
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
    return ControlledEntityReplacementContext(
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


def generate_named_entities(
    *,
    context: ControlledEntityReplacementContext,
    entity_pool: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    """Materialize the paper's ``GenerateNamedEntities`` stage for one template."""
    named_entity_pool_for_sampling = copy_named_entity_pool_without_rule_based_mutation(entity_pool)
    pool_sampler = FictionalEntitySampler(
        named_entity_pool_for_sampling,
        seed=seed,
        factual_entities=context.factual_entities_full,
        implicit_rules=context.generation_document.implicit_rules,
    )
    manual_pool_shortages = pool_sampler.find_manual_pool_shortages(context.required_entities)
    if manual_pool_shortages:
        rendered_shortages = "\n".join(manual_pool_shortages[:12])
        raise ValueError(
            f"Entity pool cannot satisfy the required manual attributes for this document:\n{rendered_shortages}"
        )
    return named_entity_pool_for_sampling


def sample_named_entities(
    *,
    context: ControlledEntityReplacementContext,
    named_entities: dict[str, Any],
    replacement_proportion: float,
    version_seed: int,
    replace_mode: str = REPLACE_MODE_ALL,
    eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None = None,
    reference_variant_index: int | None = None,
    reference_variant_count: int | None = None,
    used_named_values_by_id: dict[str, set[str]] | None = None,
) -> NamedEntitySample | None:
    """Sample named entities for one fictional variant."""
    if DEBUG_SAMPLING:
        print(
            f"[dbg] start sample_named_entities seed={version_seed} p={replacement_proportion}",
            flush=True,
        )

    random.seed(version_seed)
    _replacement_plan, replacement_layout, fictional_requirements = plan_variant_replacements(
        context=context,
        replacement_proportion=replacement_proportion,
        replace_mode=replace_mode,
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
        used_named_values_by_id=used_named_values_by_id,
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
    context: ControlledEntityReplacementContext,
    named_entity_sample: NamedEntitySample,
    named_entities: dict[str, Any],
    version_seed: int,
    eligible_cache: dict[tuple[str, tuple[str, ...]], list[Any]] | None = None,
    reference_variant_index: int | None = None,
    reference_variant_count: int | None = None,
    used_number_values_by_id: dict[str, set[int | float]] | None = None,
    used_temporal_years_by_id: dict[str, set[int]] | None = None,
    used_temporal_values_by_id: dict[str, dict[str, set[Any]]] | None = None,
    prior_relaxed_intervariant_reuse_audit: list[dict[str, Any]] | None = None,
    allow_relaxed_intervariant_number_reuse: bool = False,
    allow_factual_numtemp_values: bool = False,
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
        prior_relaxed_intervariant_reuse_audit=prior_relaxed_intervariant_reuse_audit,
        allow_relaxed_intervariant_number_reuse=allow_relaxed_intervariant_number_reuse,
        allow_factual_numtemp_values=allow_factual_numtemp_values,
    )
    numerical_entities = sampler.generate_numerical_entities(
        required_entities=named_entity_sample.fictional_requirements,
        rules=list(context.generation_document.rules),
        named_entities=named_entity_sample.entities,
        max_attempts=MAX_SAMPLING_ATTEMPTS_PER_VARIANT,
        decade_year_temporal_ids=set(named_entity_sample.decade_year_temporal_ids),
    )
    if numerical_entities is None and allow_factual_numtemp_values:
        logger.warning(
            "Numerical search exhausted for %s; using factual numeric/temporal values before final rule validation.",
            context.source_document.document_id,
        )
        numerical_entities = EntityCollection(
            numbers={
                entity_id: entity.model_copy(deep=True)
                for entity_id, entity in context.factual_entities_full.numbers.items()
            },
            temporals={
                entity_id: entity.model_copy(deep=True)
                for entity_id, entity in context.factual_entities_full.temporals.items()
            },
        )
    if numerical_entities is None:
        if DEBUG_SAMPLING:
            print("[dbg] numerical entity generation returned None", flush=True)
        logger.warning(
            "Numerical entity generation failed for %s; refusing factual fallback.",
            context.source_document.document_id,
        )
        return None
    return NumericalEntitySample(
        entities=numerical_entities,
        relaxed_intervariant_reuse_audit=tuple(
            dict(item) for item in getattr(sampler, "last_relaxed_numtemp_reuse_audit", [])
        ),
    )


def replace_factual_entities(
    *,
    context: ControlledEntityReplacementContext,
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

    # Numerical generation may return a partially populated temporal/number
    # entity when a small component domain is exhausted (for example, the
    # eighth requested weekday).  Preserve generated components and fill only
    # unresolved ones from the factual source before rule and answer checks.
    for collection_name in ("numbers", "temporals"):
        final_collection = getattr(hybrid_entities, collection_name)
        factual_collection = getattr(context.factual_entities_full, collection_name)
        for entity_id, factual_entity in factual_collection.items():
            generated_entity = final_collection.get(entity_id)
            if generated_entity is None:
                continue
            merged_payload = generated_entity.model_dump()
            changed = False
            for attr, factual_value in factual_entity.model_dump().items():
                if merged_payload.get(attr) is None and factual_value is not None:
                    merged_payload[attr] = factual_value
                    changed = True
            if changed:
                final_collection[entity_id] = type(generated_entity)(**merged_payload)

    if DEBUG_SAMPLING:
        print("[dbg] validating variant rules", flush=True)
    if not variant_rules_hold(
        generation_document=context.generation_document,
        hybrid_entities=hybrid_entities,
    ):
        logger.warning("Rejecting fictional sample for %s because reviewed generation rules failed.", document_id)
        return None
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
    "build_controlled_entity_replacement_context",
    "generate_named_entities",
    "generate_numerical_entities",
    "replace_factual_entities",
    "sample_named_entities",
]
