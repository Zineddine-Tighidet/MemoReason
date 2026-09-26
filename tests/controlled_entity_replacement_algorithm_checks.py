from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import memoreason.factual_to_fictional_dataset.controlled_entity_replacement.controlled_entity_replacement_algorithm as controlled_entity_replacement_algorithm
import memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_variant_generation.fictional_document_variant_generation_stages as fictional_generation_stages
from memoreason.benchmark_definition.annotation_runtime import load_annotated_document
from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity, TemporalEntity
from memoreason.benchmark_definition.entity_taxonomy import REPLACE_MODE_NON_NUMERICAL
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.controlled_entity_replacement_algorithm import (
    ControlledEntityReplacementContext,
    FictionalDocumentVariantRequest,
    NamedEntitySample,
    NumericalEntitySample,
    ReviewedTemplateFictionalGenerationInput,
    build_controlled_entity_replacement_context,
    generate_named_entities,
    generate_numerical_entities,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.named_entity_uniqueness import (
    named_entity_uniqueness_signature,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)


def _write_template(
    template_path: Path,
    *,
    document_text: str,
    rules: list[str] | None = None,
    questions: list[dict] | None = None,
) -> Path:
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "document": {
                    "document_id": template_path.stem,
                    "document_theme": template_path.parent.name,
                    "original_document": "",
                    "document_to_annotate": document_text,
                    "fictionalized_annotated_template_document": "",
                    "rules": rules or [],
                    "questions": questions or [],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return template_path


def test_build_controlled_entity_replacement_context_rejects_invalid_reviewed_rules(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_01.yaml",
        document_text="[Ada Lovelace; person_1.full_name] served from [2024; temporal_1.year] to [2025; temporal_2.year].",
        rules=["temporal_1.year == temporal_2.year"],
    )
    document = load_annotated_document(str(template_path))

    with pytest.raises(ValueError, match="Reviewed generation rules do not hold on the factual source document"):
        build_controlled_entity_replacement_context(document)


def test_build_controlled_entity_replacement_context_keeps_explicit_person_gender_literal_rule(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_01_gender.yaml",
        document_text="[Roger Federer; person_1.full_name] won [men's singles; event_1.type].",
        rules=['person_1.gender == "male"'],
    )
    document = load_annotated_document(str(template_path))

    context = build_controlled_entity_replacement_context(document)

    assert isinstance(context, ControlledEntityReplacementContext)
    assert context.dropped_rules == ()
    assert context.generation_document.rules == ['person_1.gender == "male"']
    assert context.factual_entities_full.persons["person_1"].gender == "male"


def test_build_controlled_entity_replacement_context_keeps_variant_answer_difference_soft(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_variant_arithmetic.yaml",
        document_text="[Alice; person_1.full_name] scored [7; number_1.int] goals and [4; number_2.int] assists.",
        questions=[
            {
                "question_id": "q1",
                "question": "What is the difference between the goals and assists?",
                "answer": "number_1.int - number_2.int",
                "question_type": "arithmetic",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
            {
                "question_id": "q2",
                "question": "How many assists were recorded?",
                "answer": "number_2.int",
                "question_type": "extractive",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
        ],
    )
    document = load_annotated_document(str(template_path))

    context = build_controlled_entity_replacement_context(document)

    assert "number_1.int - number_2.int != 3" not in context.generation_document.rules
    assert all("number_2.int != 4" not in rule for rule in context.generation_document.rules)


def test_variant_conditional_answer_difference_is_not_a_generation_rule(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_variant_conditional.yaml",
        document_text="The totals were [23; number_1.int] and [19; number_2.int].",
        questions=[
            {
                "question_id": "q1",
                "question": "Was the first total greater than 20?",
                "answer": "Yes if number_1.int > 20 else No",
                "question_type": "inference",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
            {
                "question_id": "q2",
                "question": "Was the second total greater than 20?",
                "answer": "Yes if number_2.int > 20 else No",
                "question_type": "inference",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
        ],
    )
    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))

    assert context.generation_document.rules == []


def test_required_number_factual_values_are_hard_solver_exclusions() -> None:
    sampler = FictionalEntitySampler(
        {},
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=23, str="twenty-three")}),
    )

    assert sampler._required_number_factual_difference_rules([("number_1", ["int"])], fixed_number_ids=set()) == [
        "number_1.int != 23"
    ]
    assert (
        sampler._required_number_factual_difference_rules([("number_1", ["int"])], fixed_number_ids={"number_1"}) == []
    )


def test_named_only_collision_gate_ignores_unreplaced_numeric_answers(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_named_only.yaml",
        document_text="[Alice Smith; person_1.full_name] scored [7; number_1.int] goals.",
        questions=[
            {
                "question_id": "q_named",
                "question": "Who scored?",
                "answer": "person_1.full_name",
                "question_type": "extractive",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
            {
                "question_id": "q_numeric",
                "question": "How many goals?",
                "answer": "number_1.int",
                "question_type": "extractive",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
        ],
    )
    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))
    hybrid_entities = context.factual_entities_full.model_copy(deep=True)
    hybrid_entities.persons["person_1"].full_name = "Mira Solen"

    collisions = fictional_generation_stages._variant_question_answer_collisions(
        context=context,
        hybrid_entities=hybrid_entities,
        replaced_entity_ids={"person_1"},
        replace_mode=REPLACE_MODE_NON_NUMERICAL,
    )

    assert collisions == []


def test_replace_factual_entities_rejects_failed_reviewed_rules(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_rule_failure.yaml",
        document_text="There were [7; number_1.int] wins and [4; number_2.int] losses.",
        rules=["number_1.int > number_2.int"],
    )
    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))
    output_path = tmp_path / "must_not_write.yaml"

    result = fictional_generation_stages.replace_factual_entities(
        context=context,
        named_entity_sample=NamedEntitySample(
            entities=EntityCollection(),
            replacement_layout=SimpleNamespace(),
            fictional_requirements={},
            decade_year_temporal_ids=frozenset(),
        ),
        numerical_entity_sample=NumericalEntitySample(
            entities=EntityCollection(
                numbers={
                    "number_1": NumberEntity(int=2),
                    "number_2": NumberEntity(int=9),
                }
            )
        ),
        output_path=output_path,
        document_id="doc_rule_failure",
        replacement_proportion=1.0,
    )

    assert result is None
    assert not output_path.exists()


@pytest.mark.parametrize(
    "answer",
    [
        "Yes",
        "place_1.region Attorney's Office",
        "temporal_1.year - temporal_1.year",
    ],
)
def test_build_controlled_entity_replacement_context_accepts_static_variant_answers(
    tmp_path: Path,
    answer: str,
) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_static_variant.yaml",
        document_text=(
            "[Alice; person_1.full_name] worked in [Westshire; place_1.region] "
            "from [2020; temporal_1.year] to [2024; temporal_2.year]."
        ),
        questions=[
            {
                "question_id": "q_variant",
                "question": "What changed?",
                "answer": answer,
                "question_type": "inference",
                "answer_type": "variant",
                "reasoning_chain": [],
            }
        ],
    )

    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))

    assert context.generation_document.questions[0].answer == answer


def test_replace_factual_entities_accepts_variant_answer_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_variant_collisions.yaml",
        document_text="[Alice; person_1.full_name] scored [7; number_1.int] goals and [4; number_2.int] assists.",
        questions=[
            {
                "question_id": "q_arithmetic",
                "question": "What is the difference between goals and assists?",
                "answer": "number_1.int - number_2.int",
                "question_type": "arithmetic",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
            {
                "question_id": "q_inference",
                "question": "Was the player productive?",
                "answer": "Yes if number_1.int > number_2.int else No",
                "question_type": "inference",
                "answer_type": "variant",
                "reasoning_chain": [],
            },
        ],
    )
    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))
    named_sample = NamedEntitySample(
        entities=EntityCollection(),
        replacement_layout=SimpleNamespace(),
        fictional_requirements={},
        decade_year_temporal_ids=frozenset(),
    )
    numerical_sample = NumericalEntitySample(
        entities=EntityCollection(
            numbers={
                "number_1": NumberEntity(int=8),
                "number_2": NumberEntity(int=5),
            }
        )
    )
    output_path = tmp_path / "variant.yaml"

    def _write_variant(**kwargs):
        path = kwargs["output_path"]
        path.write_text("accepted: true\n", encoding="utf-8")
        return path

    monkeypatch.setattr(fictional_generation_stages, "render_and_write_variant", _write_variant)

    result = fictional_generation_stages.replace_factual_entities(
        context=context,
        named_entity_sample=named_sample,
        numerical_entity_sample=numerical_sample,
        output_path=output_path,
        document_id="doc_variant_collisions",
        replacement_proportion=1.0,
    )

    assert result == output_path
    assert output_path.read_text(encoding="utf-8") == "accepted: true\n"


def test_fictional_generation_is_defined_in_the_public_algorithm_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_log: list[str] = []

    context = SimpleNamespace(
        source_document=SimpleNamespace(document_id="doc_01"),
        generation_document=SimpleNamespace(rules=[]),
        dropped_rules=(),
    )

    def _build_context(_template_document):
        call_log.append("build_context")
        return context

    def _generate_named_entities(*, context, entity_pool, seed):
        assert context is not None
        assert entity_pool == {"persons": []}
        assert seed == 13
        call_log.append("generate_named_entities")
        return {"prepared_pool": True}

    def _sample_named_entities(**kwargs):
        assert kwargs["named_entities"] == {"prepared_pool": True}
        call_log.append("sample_named_entities")
        assert kwargs["reference_variant_index"] is None
        assert kwargs["reference_variant_count"] is None
        return NamedEntitySample(
            entities=EntityCollection(),
            replacement_layout=SimpleNamespace(),
            fictional_requirements={},
            decade_year_temporal_ids=frozenset(),
        )

    def _generate_numerical_entities(**kwargs):
        assert isinstance(kwargs["named_entity_sample"], NamedEntitySample)
        call_log.append("generate_numerical_entities")
        assert kwargs["reference_variant_index"] is None
        assert kwargs["reference_variant_count"] is None
        return NumericalEntitySample(entities=EntityCollection())

    def _replace_factual_entities(**kwargs):
        assert isinstance(kwargs["named_entity_sample"], NamedEntitySample)
        assert isinstance(kwargs["numerical_entity_sample"], NumericalEntitySample)
        call_log.append("replace_factual_entities")
        return kwargs["output_path"]

    monkeypatch.setattr(
        controlled_entity_replacement_algorithm, "build_controlled_entity_replacement_context", _build_context
    )
    monkeypatch.setattr(controlled_entity_replacement_algorithm, "generate_named_entities", _generate_named_entities)
    monkeypatch.setattr(controlled_entity_replacement_algorithm, "sample_named_entities", _sample_named_entities)
    monkeypatch.setattr(
        controlled_entity_replacement_algorithm, "generate_numerical_entities", _generate_numerical_entities
    )
    monkeypatch.setattr(controlled_entity_replacement_algorithm, "replace_factual_entities", _replace_factual_entities)

    result = controlled_entity_replacement_algorithm.generate_fictional_document_variants(
        [
            ReviewedTemplateFictionalGenerationInput(
                template_document=object(),
                named_entity_pool={"persons": []},
                variant_requests=(
                    FictionalDocumentVariantRequest(base_seed=101, output_path=Path("/tmp/variant_1.yaml")),
                    FictionalDocumentVariantRequest(base_seed=102, output_path=Path("/tmp/variant_2.yaml")),
                ),
                replacement_proportion=1.0,
                document_id="doc_01",
                named_entities_seed=13,
                eligible_cache={},
            )
        ]
    )

    assert controlled_entity_replacement_algorithm.generate_fictional_document_variants.__module__ == (
        "memoreason.factual_to_fictional_dataset.controlled_entity_replacement.controlled_entity_replacement_algorithm"
    )
    assert [entry.document_id for entry in result] == ["doc_01"]
    assert result[0].generated_variants == (
        (Path("/tmp/variant_1.yaml"), 101),
        (Path("/tmp/variant_2.yaml"), 102),
    )
    assert call_log == [
        "build_context",
        "generate_named_entities",
        "sample_named_entities",
        "generate_numerical_entities",
        "replace_factual_entities",
        "sample_named_entities",
        "generate_numerical_entities",
        "replace_factual_entities",
    ]


def test_generate_named_entities_rejects_insufficient_pool(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_02.yaml",
        document_text="[Ada Lovelace; person_1.full_name] spoke in [London; place_1.city].",
    )
    document = load_annotated_document(str(template_path))
    context = build_controlled_entity_replacement_context(document)

    with pytest.raises(ValueError, match="Entity pool cannot satisfy"):
        generate_named_entities(
            context=context,
            entity_pool={
                "persons": [{"full_name": "Mira Solen", "first_name": "Mira", "last_name": "Solen"}],
                "places": [],
                "events": [],
                "organizations": [],
                "awards": [],
                "legals": [],
                "products": [],
                "numbers": [],
                "temporals": [],
            },
        )


def test_generate_numerical_entities_refuses_factual_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_02b.yaml",
        document_text="[Ada Lovelace; person_1.full_name] led [three; number_1.str] teams in [2024; temporal_1.year].",
    )
    document = load_annotated_document(str(template_path))
    context = build_controlled_entity_replacement_context(document)

    class _Sampler:
        def generate_numerical_entities(self, **_kwargs):
            return None

    monkeypatch.setattr(fictional_generation_stages, "build_variant_sampler", lambda **_kwargs: _Sampler())

    sample = generate_numerical_entities(
        context=context,
        named_entity_sample=NamedEntitySample(
            entities=EntityCollection(),
            replacement_layout=SimpleNamespace(),
            fictional_requirements={},
            decade_year_temporal_ids=frozenset(),
        ),
        named_entities={},
        version_seed=7,
    )

    assert sample is None


def test_required_temporal_difference_repair_does_not_break_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2005),
        }
    )
    collection = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=1990),
            "temporal_2": TemporalEntity(year=2005),
        }
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: (1980, 2030))
    monkeypatch.setattr(generator, "_temporal_year_domain", lambda *args, **kwargs: [1990])

    repaired = sampler._repair_single_required_temporal_year_difference(
        generator=generator,
        collection=collection,
        temporal_id="temporal_2",
        factual_value=2005,
        rules_with_ordering=[],
        ordering_exempt_ids=set(),
    )

    assert repaired is False
    assert collection.temporals["temporal_2"].year == 2005


def test_force_simple_required_temporal_difference_preserves_ordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2005),
        }
    )
    collection = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=1990),
            "temporal_2": TemporalEntity(year=2005),
        }
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: (1980, 2030))
    monkeypatch.setattr(generator, "_temporal_year_domain", lambda *args, **kwargs: [1990])

    sampler._force_simple_required_differences(
        generator=generator,
        collection=collection,
        unchanged=[("temporal", "temporal_2", "year", 2005)],
        rules_with_ordering=[],
        number_required_attr_map={},
        ordering_exempt_ids=set(),
    )

    assert collection.temporals["temporal_2"].year == 2005


def test_force_simple_required_number_difference_scopes_prior_reuse_per_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(numbers={"number_3": NumberEntity(int=8, str="eight")})
    sampler = FictionalEntitySampler(
        {},
        factual_entities=factual_entities,
        used_number_values_by_id={"number_3": {7}},
    )
    generator = NumberTemporalGenerator(factual_entities=factual_entities)
    bounds = [7, 9]
    monkeypatch.setattr(generator, "_number_actual_bounds", lambda _number_id, _attrs: tuple(bounds))

    def force(*, allowed_number_reuse_ids: set[str]) -> EntityCollection:
        collection = factual_entities.model_copy(deep=True)
        sampler._force_simple_required_differences(
            generator=generator,
            collection=collection,
            unchanged=[("number", "number_3", "str", "eight")],
            rules_with_ordering=[],
            number_required_attr_map={"number_3": {"str"}},
            ordering_exempt_ids=set(),
            allowed_number_reuse_ids=allowed_number_reuse_ids,
        )
        return collection

    collection = force(allowed_number_reuse_ids=set())
    assert collection.numbers["number_3"].int == 9

    bounds[:] = [7, 7]
    collection = force(allowed_number_reuse_ids={"number_3"})
    assert collection.numbers["number_3"].int == 7

    collection = force(allowed_number_reuse_ids={"number_4"})
    assert collection.numbers["number_3"].int == 8


def test_temporal_product_numbers_are_aligned_before_temporal_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_4": TemporalEntity(year=2004),
            "temporal_5": TemporalEntity(year=2009),
        },
        numbers={},
    )
    collection = EntityCollection(
        numbers={},
        temporals={},
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    collection.numbers["number_14"] = generator._build_number_entity("number_14", 4, required_attrs={"str"})
    collection.numbers["number_12"] = generator._build_number_entity("number_12", 25, required_attrs={"int"})

    monkeypatch.setattr(
        generator,
        "_number_base_range",
        lambda number_id: (2, 6) if number_id == "number_14" else (10, 30),
    )
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    monkeypatch.setattr(generator, "_expand_int_domain_to_escape_forbidden", lambda low, high, avoid: (low, high))
    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: (1990, 2030))
    monkeypatch.setattr(generator, "_temporal_year_domain", lambda *args, **kwargs: list(range(1990, 2031)))

    sampler._align_numbers_for_temporal_product_rules(
        generator=generator,
        collection=collection,
        temporal_rules=["(temporal_5.year - temporal_4.year + 1) * number_14.int == number_12.int"],
        required_attr_map={"number_14": {"str"}, "number_12": {"int"}},
        avoid_numbers={},
    )

    source_value = collection.numbers["number_14"].int
    target_value = collection.numbers["number_12"].int
    assert source_value is not None and target_value is not None
    assert source_value > 0
    assert target_value % source_value == 0


def test_temporal_product_alignment_respects_minimum_order_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_4": TemporalEntity(year=2004),
            "temporal_6": TemporalEntity(year=2006),
            "temporal_7": TemporalEntity(year=2007),
            "temporal_11": TemporalEntity(year=2008),
            "temporal_5": TemporalEntity(year=2009),
        },
        numbers={},
    )
    collection = EntityCollection(numbers={}, temporals={})
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    collection.numbers["number_14"] = generator._build_number_entity("number_14", 7, required_attrs={"str"})
    collection.numbers["number_12"] = generator._build_number_entity("number_12", 28, required_attrs={"int"})

    monkeypatch.setattr(
        generator,
        "_number_base_range",
        lambda number_id: (2, 7) if number_id == "number_14" else (20, 28),
    )
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    monkeypatch.setattr(generator, "_expand_int_domain_to_escape_forbidden", lambda low, high, avoid: (low, high))
    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: (1990, 2030))
    monkeypatch.setattr(generator, "_temporal_year_domain", lambda *args, **kwargs: list(range(1990, 2031)))

    sampler._align_numbers_for_temporal_product_rules(
        generator=generator,
        collection=collection,
        temporal_rules=["(temporal_5.year - temporal_4.year + 1) * number_14.int == number_12.int"],
        required_attr_map={"number_14": {"str"}, "number_12": {"int"}},
        avoid_numbers={"number_14": {4}, "number_12": {24}},
    )

    source_value = collection.numbers["number_14"].int
    target_value = collection.numbers["number_12"].int
    assert source_value is not None and target_value is not None
    assert target_value % source_value == 0
    assert (target_value // source_value) >= 5


def test_temporal_number_joint_feasibility_respects_numeric_ordering_rules() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_2": TemporalEntity(year=1998),
            "temporal_3": TemporalEntity(year=2003),
            "temporal_4": TemporalEntity(year=2004),
            "temporal_6": TemporalEntity(year=2006),
            "temporal_7": TemporalEntity(year=2007),
            "temporal_11": TemporalEntity(year=2008),
            "temporal_5": TemporalEntity(year=2009),
            "temporal_9": TemporalEntity(year=2010),
            "temporal_12": TemporalEntity(year=2012),
            "temporal_10": TemporalEntity(year=2014),
            "temporal_13": TemporalEntity(year=2016),
            "temporal_14": TemporalEntity(year=2017),
            "temporal_15": TemporalEntity(year=2018),
            "temporal_19": TemporalEntity(year=2020),
            "temporal_18": TemporalEntity(year=2021),
            "temporal_22": TemporalEntity(year=2022),
            "temporal_23": TemporalEntity(year=2025),
        },
        numbers={
            "number_11": NumberEntity(int=20),
            "number_12": NumberEntity(int=24),
            "number_14": NumberEntity(int=4, str="four"),
            "number_18": NumberEntity(int=3, str="three"),
            "number_22": NumberEntity(int=19),
        },
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(
        factual_entities=factual_entities,
        exclude_temporals={
            "years": {
                1998,
                2003,
                2004,
                2006,
                2007,
                2008,
                2009,
                2010,
                2012,
                2014,
                2016,
                2017,
                2018,
                2020,
                2021,
                2022,
                2025,
            }
        },
    )
    generator._number_base_range = lambda number_id: {
        "number_11": (20, 20),
        "number_12": (20, 28),
        "number_14": (1, 7),
        "number_18": (3, 3),
        "number_22": (16, 22),
    }[number_id]
    generator._implicit_number_range = lambda _number_id: None
    generator._expand_int_domain_to_escape_forbidden = lambda low, high, avoid: (low, high)
    collection = EntityCollection(
        numbers={
            "number_11": generator._build_number_entity("number_11", 20, required_attrs={"int"}),
            "number_18": generator._build_number_entity("number_18", 3, required_attrs={"str"}),
            "number_12": generator._build_number_entity("number_12", 28, required_attrs={"int"}),
            "number_14": generator._build_number_entity("number_14", 2, required_attrs={"str"}),
            "number_22": generator._build_number_entity("number_22", 22, required_attrs={"int"}),
        },
        temporals={},
    )

    sampler._ensure_temporal_number_joint_feasibility(
        generator=generator,
        collection=collection,
        required_temporals=[
            ("temporal_2", ["year"]),
            ("temporal_3", ["year"]),
            ("temporal_4", ["year"]),
            ("temporal_5", ["year"]),
            ("temporal_6", ["year"]),
            ("temporal_7", ["year"]),
            ("temporal_9", ["year"]),
            ("temporal_10", ["year"]),
            ("temporal_11", ["year"]),
            ("temporal_12", ["year"]),
            ("temporal_13", ["year"]),
            ("temporal_14", ["year"]),
            ("temporal_15", ["year"]),
            ("temporal_18", ["year"]),
            ("temporal_19", ["year"]),
            ("temporal_22", ["year"]),
            ("temporal_23", ["year"]),
        ],
        temporal_rules=[
            "temporal_18.year - temporal_3.year + 1 == number_22.int",
            "(temporal_5.year - temporal_4.year + 1) * number_14.int == number_12.int",
            "temporal_3.year - temporal_2.year != 5",
        ],
        supporting_numeric_rules=[
            "number_18.int < number_14.int",
            "number_22.int < number_11.int",
        ],
        required_attr_map={
            "number_11": {"int"},
            "number_12": {"int"},
            "number_14": {"str"},
            "number_18": {"str"},
            "number_22": {"int"},
        },
        avoid_numbers={"number_12": {24}, "number_14": {4}, "number_22": {19}},
        decade_year_temporal_ids=set(),
    )

    assert collection.numbers["number_14"].int > collection.numbers["number_18"].int
    assert collection.numbers["number_22"].int < collection.numbers["number_11"].int
    assert collection.numbers["number_12"].int % collection.numbers["number_14"].int == 0


def test_temporal_number_joint_feasibility_caches_equivalent_temporal_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=2),
            "number_2": NumberEntity(int=4),
        }
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    collection = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=2),
            "number_2": NumberEntity(int=4),
        }
    )
    monkeypatch.setattr(
        generator,
        "_number_base_range",
        lambda number_id: (1, 3) if number_id == "number_1" else (2, 6),
    )
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    monkeypatch.setattr(generator, "_expand_int_domain_to_escape_forbidden", lambda low, high, avoid: (low, high))
    solver_calls: list[tuple[str, ...]] = []

    def solve_temporals(_required, rules, _collection, _excluded, _decade_ids):
        signature = tuple(rules)
        solver_calls.append(signature)
        return {"temporal_1": 2000, "temporal_2": 2000} if signature[0].endswith("== 1") else None

    monkeypatch.setattr(generator, "_solve_temporal_years", solve_temporals)

    sampler._ensure_temporal_number_joint_feasibility(
        generator=generator,
        collection=collection,
        required_temporals=[("temporal_1", ["year"]), ("temporal_2", ["year"])],
        temporal_rules=["(temporal_2.year - temporal_1.year + 1) * number_1.int == number_2.int"],
        supporting_numeric_rules=[],
        required_attr_map={"number_1": {"int"}, "number_2": {"int"}},
        avoid_numbers={},
        decade_year_temporal_ids=set(),
    )

    assert collection.numbers["number_2"].int == collection.numbers["number_1"].int
    assert len(solver_calls) == len(set(solver_calls))
    assert sum(signature[0].endswith("== 2") for signature in solver_calls) == 1


def test_temporal_number_joint_feasibility_uses_retained_factual_supporting_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=8, str="eight"),
            "number_2": NumberEntity(int=2, str="two"),
        },
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2007),
        },
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    collection = EntityCollection(numbers={"number_1": NumberEntity(int=19, str="nineteen")})
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (7, 9))
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    solver_values: list[int] = []

    def solve_temporals(_required, _rules, candidate_collection, _excluded, _decade_ids):
        candidate = int(candidate_collection.numbers["number_1"].int)
        solver_values.append(candidate)
        if candidate not in {7, 9}:
            return None
        return {"temporal_1": 2000, "temporal_2": 2000 + candidate - 1}

    monkeypatch.setattr(generator, "_solve_temporal_years", solve_temporals)

    sampler._ensure_temporal_number_joint_feasibility(
        generator=generator,
        collection=collection,
        required_temporals=[("temporal_1", ["year"]), ("temporal_2", ["year"])],
        temporal_rules=["temporal_2.year - temporal_1.year + 1 == number_1.str"],
        supporting_numeric_rules=["number_2.str == 2"],
        required_attr_map={"number_1": {"str"}},
        avoid_numbers={"number_1": {8}},
        decade_year_temporal_ids=set(),
    )

    assert solver_values
    assert collection.numbers["number_1"].int in {7, 9}


def test_temporal_number_joint_feasibility_does_not_fallback_for_missing_required_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=8, str="eight"),
            "number_2": NumberEntity(int=2, str="two"),
        },
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2007),
        },
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    collection = EntityCollection(numbers={"number_1": NumberEntity(int=19, str="nineteen")})
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (7, 9))
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    solver_values: list[int] = []
    monkeypatch.setattr(
        generator,
        "_solve_temporal_years",
        lambda *_args, **_kwargs: solver_values.append(1),
    )

    sampler._ensure_temporal_number_joint_feasibility(
        generator=generator,
        collection=collection,
        required_temporals=[("temporal_1", ["year"]), ("temporal_2", ["year"])],
        temporal_rules=["temporal_2.year - temporal_1.year + 1 == number_1.str"],
        supporting_numeric_rules=["number_2.str == 2"],
        required_attr_map={"number_1": {"str"}, "number_2": {"str"}},
        avoid_numbers={"number_1": {8}},
        decade_year_temporal_ids=set(),
    )

    assert solver_values == []
    assert collection.numbers["number_1"].int == 19


def test_temporal_number_joint_feasibility_keeps_mixed_retained_number_factual_and_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=8, str="eight"),
            "number_2": NumberEntity(int=2, str="two"),
        },
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2002),
        },
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    collection = EntityCollection(numbers={"number_1": NumberEntity(int=8, str="eight")})
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (1, 3))
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    solver_values: list[int | None] = []

    def solve_temporals(_required, _rules, candidate_collection, _excluded, _decade_ids):
        retained = candidate_collection.numbers.get("number_2")
        retained_value = int(retained.int) if retained is not None else None
        solver_values.append(retained_value)
        if retained_value != 2:
            return None
        return {"temporal_1": 2000, "temporal_2": 2002}

    monkeypatch.setattr(generator, "_solve_temporal_years", solve_temporals)

    sampler._ensure_temporal_number_joint_feasibility(
        generator=generator,
        collection=collection,
        required_temporals=[("temporal_1", ["year"]), ("temporal_2", ["year"])],
        temporal_rules=["temporal_2.year - temporal_1.year == number_2.str"],
        supporting_numeric_rules=["number_1.str > number_2.str"],
        required_attr_map={"number_1": {"str"}},
        avoid_numbers={},
        decade_year_temporal_ids=set(),
    )

    assert solver_values == [2]
    assert "number_2" not in collection.numbers
    assert collection.numbers["number_1"].int == 8


def test_temporal_number_joint_feasibility_removes_missing_search_assignment_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        numbers={"number_1": NumberEntity(int=2, str="two")},
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2002),
        },
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    collection = EntityCollection()
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (1, 3))
    monkeypatch.setattr(generator, "_implicit_number_range", lambda _number_id: None)
    monkeypatch.setattr(generator, "_solve_temporal_years", lambda *_args, **_kwargs: None)

    sampler._ensure_temporal_number_joint_feasibility(
        generator=generator,
        collection=collection,
        required_temporals=[("temporal_1", ["year"]), ("temporal_2", ["year"])],
        temporal_rules=["temporal_2.year - temporal_1.year == number_1.str"],
        supporting_numeric_rules=[],
        required_attr_map={"number_1": {"str"}},
        avoid_numbers={},
        decade_year_temporal_ids=set(),
    )

    assert "number_1" not in collection.numbers


def test_temporal_product_rule_with_constants_is_concretized() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_4": TemporalEntity(year=2004),
            "temporal_5": TemporalEntity(year=2009),
        }
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    concretized = sampler._concretize_temporal_rules_with_numbers(
        generator=generator,
        temporal_rules=["(temporal_5.year - temporal_4.year + 1) * 7 == 28"],
        collection=EntityCollection(),
    )

    assert concretized == ["temporal_5.year - temporal_4.year + 1 == 4"]


def test_temporal_rule_without_numbers_is_preserved_for_temporal_generation() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="5 July 2011", year=2011),
            "temporal_2": TemporalEntity(date="5 July 2016", year=2016),
        }
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    concretized = sampler._concretize_temporal_rules_with_numbers(
        generator=generator,
        temporal_rules=["temporal_1.date == temporal_2.date - 5"],
        collection=EntityCollection(),
    )

    assert concretized == ["temporal_1.date == temporal_2.date - 5"]


def test_temporal_solver_ignores_month_only_temporals() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2005),
            "temporal_3": TemporalEntity(month="August"),
        }
    )
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    solved = generator._solve_temporal_years(
        required_temporals=[
            ("temporal_1", ["year"]),
            ("temporal_2", ["year"]),
            ("temporal_3", ["month"]),
        ],
        rules=["temporal_2.year - temporal_1.year == 3"],
        existing_entities=factual_entities,
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert solved is not None
    assert "temporal_3" not in solved


def test_order_preserving_temporal_candidates_do_not_fallback_to_full_domain_when_interval_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_2": TemporalEntity(year=1998),
            "temporal_3": TemporalEntity(year=2003),
            "temporal_4": TemporalEntity(year=2004),
        }
    )
    collection = EntityCollection(
        temporals={
            "temporal_2": TemporalEntity(year=1999),
            "temporal_3": TemporalEntity(year=2003),
            "temporal_4": TemporalEntity(year=1998),
        }
    )
    sampler = FictionalEntitySampler({}, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(factual_entities=factual_entities, exclude_temporals={"years": set()})

    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: (1984, 2026))
    monkeypatch.setattr(generator, "_temporal_year_domain", lambda *args, **kwargs: list(range(1984, 2027)))

    candidates = sampler._order_preserving_temporal_year_candidates(
        generator=generator,
        collection=collection,
        temporal_id="temporal_3",
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert candidates == []


def test_fixed_reference_variant_index_uses_distinct_reference_pool_entries() -> None:
    from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
        FictionalEntitySampler,
    )

    entity_pool = {
        "persons": [],
        "places": [],
        "events": [],
        "organizations": [],
        "military_orgs": [],
        "entreprise_orgs": [],
        "ngos": [],
        "government_orgs": [],
        "educational_orgs": [],
        "media_orgs": [],
        "awards": [],
        "legals": [],
        "products": [],
        "_reference_pools": {
            "persons": {
                "person_1": {
                    "required_attributes": ["full_name", "first_name", "last_name"],
                    "count": 3,
                    "variants": [
                        {"full_name": "Mira Solen", "first_name": "Mira", "last_name": "Solen"},
                        {"full_name": "Tarin Voss", "first_name": "Tarin", "last_name": "Voss"},
                        {"full_name": "Lena Quill", "first_name": "Lena", "last_name": "Quill"},
                    ],
                }
            },
            "places": {},
            "events": {},
            "organizations": {},
            "military_orgs": {},
            "entreprise_orgs": {},
            "ngos": {},
            "government_orgs": {},
            "educational_orgs": {},
            "media_orgs": {},
            "awards": {},
            "legals": {},
            "products": {},
        },
    }

    seen_names: list[str] = []
    for variant_index in range(3):
        sampler = FictionalEntitySampler(
            entity_pool=entity_pool,
            reference_variant_index=variant_index,
            reference_variant_count=3,
        )
        sampled = sampler._sample_entity_with_attributes(
            "person",
            ["full_name", "first_name", "last_name"],
            [],
            entity_id="person_1",
        )
        seen_names.append(sampled.full_name)

    assert seen_names == ["Mira Solen", "Tarin Voss", "Lena Quill"]


def test_named_uniqueness_signature_normalizes_surface_and_secondary_attributes() -> None:
    assert named_entity_uniqueness_signature("place", {"demonym": "  VELANTHI  "}) == (
        named_entity_uniqueness_signature("places", {"demonym": "velanthi"})
    )
    assert named_entity_uniqueness_signature(
        "person",
        {"nationality": "Telmoric", "ethnicity": "Dravani"},
    ) != named_entity_uniqueness_signature(
        "person",
        {"nationality": "Telmoric", "ethnicity": "Korathi"},
    )


def test_named_reference_candidates_stay_distinct_after_numeric_retry_seed_shifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template_path = _write_template(
        tmp_path / "theme_a" / "doc_named_batch.yaml",
        document_text="[Ada Lovelace; person_1.full_name] led the expedition.",
    )
    document = load_annotated_document(str(template_path))
    candidates = [{"full_name": f"Invented Person {index:02d}"} for index in range(15)]
    entity_pool = {
        "persons": list(candidates),
        "_reference_pools": {
            "persons": {
                "person_1": {
                    "required_attributes": ["full_name"],
                    "count": len(candidates),
                    "variants": list(candidates),
                }
            }
        },
    }
    retry_seeds = {7_000_023, 8_000_023, 9_000_023}
    sampled_names: list[str] = []

    def _generate_numerical_entities(**kwargs):
        if kwargs["version_seed"] in retry_seeds:
            return None
        return NumericalEntitySample(entities=EntityCollection())

    def _replace_factual_entities(**kwargs):
        sampled_names.append(kwargs["named_entity_sample"].entities.persons["person_1"].full_name)
        return kwargs["output_path"]

    monkeypatch.setattr(
        controlled_entity_replacement_algorithm,
        "generate_numerical_entities",
        _generate_numerical_entities,
    )
    monkeypatch.setattr(
        controlled_entity_replacement_algorithm,
        "replace_factual_entities",
        _replace_factual_entities,
    )
    used_named_values: dict[str, set[str]] = {}
    variant_requests = tuple(
        FictionalDocumentVariantRequest(
            base_seed=23 + (variant_index * 1_000_000),
            output_path=tmp_path / f"doc_named_batch_v{variant_index + 1:02d}.yaml",
            reference_variant_index=variant_index,
            reference_variant_count=10,
        )
        for variant_index in range(10)
    )

    result = controlled_entity_replacement_algorithm.generate_fictional_document_variants(
        [
            ReviewedTemplateFictionalGenerationInput(
                template_document=document,
                named_entity_pool=entity_pool,
                variant_requests=variant_requests,
                replacement_proportion=1.0,
                document_id="doc_named_batch",
                named_entities_seed=23,
                used_named_values_by_id=used_named_values,
            )
        ]
    )[0]

    assert len(result.generated_variants) == 10
    assert sampled_names == [
        "Invented Person 00",
        "Invented Person 01",
        "Invented Person 02",
        "Invented Person 03",
        "Invented Person 04",
        "Invented Person 05",
        "Invented Person 06",
        "Invented Person 12",
        "Invented Person 13",
        "Invented Person 14",
    ]
    assert len(set(sampled_names)) == 10
    assert used_named_values == {
        "person_1": {
            named_entity_uniqueness_signature("person", candidate)
            for candidate in candidates
            if candidate["full_name"] in set(sampled_names)
        }
    }


def test_reference_variant_candidates_walk_non_overlapping_stride_before_fallback() -> None:
    from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
        FictionalEntitySampler,
    )

    sampler = FictionalEntitySampler(
        entity_pool={},
        reference_variant_index=1,
        reference_variant_count=3,
    )

    assert sampler._reference_variant_candidates(8) == [1, 4, 7, 0, 2, 3, 5, 6]


def test_reference_pool_candidates_can_fall_back_to_bucket_support() -> None:
    from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
        FictionalEntitySampler,
    )

    entity_pool = {
        "persons": [],
        "places": [],
        "events": [],
        "organizations": [],
        "military_orgs": [],
        "entreprise_orgs": [],
        "ngos": [],
        "government_orgs": [],
        "educational_orgs": [],
        "media_orgs": [],
        "awards": [],
        "legals": [],
        "products": [
            {"name": "Helion Array"},
            {"name": "Quorix Beacon"},
            {"name": "Vetra Lens"},
        ],
        "_reference_pools": {
            "persons": {},
            "places": {},
            "events": {},
            "organizations": {},
            "military_orgs": {},
            "entreprise_orgs": {},
            "ngos": {},
            "government_orgs": {},
            "educational_orgs": {},
            "media_orgs": {},
            "awards": {},
            "legals": {},
            "products": {
                "product_1": {
                    "required_attributes": ["name"],
                    "count": 1,
                    "variants": [{"name": "Helion Array"}],
                }
            },
        },
    }

    sampler = FictionalEntitySampler(entity_pool=entity_pool, seed=23)

    valid = sampler._valid_pool_entities("product", ["name"], [], entity_id="product_1")

    assert [entry["name"] for entry in valid] == [
        "Helion Array",
        "Quorix Beacon",
        "Vetra Lens",
    ]


def test_age_only_person_requirement_is_synthesized_without_pool_entry() -> None:
    from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
        FictionalEntitySampler,
    )

    sampler = FictionalEntitySampler(entity_pool={}, seed=23)
    sampler._current_rules = ["person_2.age > 12", "person_2.age < 18"]

    sampled = sampler._sample_entity_with_attributes(
        "person",
        ["age"],
        [],
        entity_id="person_2",
    )

    assert sampled.age is not None
    assert 12 < int(sampled.age) < 18
