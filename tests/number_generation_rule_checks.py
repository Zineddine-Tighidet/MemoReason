from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from memoreason.benchmark_definition.annotation_runtime import (
    AnnotationParser,
    AnnotationValidationError,
    RuleEngine,
    load_annotated_document,
    load_entity_pool,
    partition_generation_rules,
)
from memoreason.benchmark_definition.document_schema import (
    AnnotatedDocument,
    EntityCollection,
    ImplicitRule,
    NumberEntity,
    Question,
    TemporalEntity,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_variant_generation.fictional_document_variant_generation_stages import (
    build_controlled_entity_replacement_context,
    generate_named_entities,
    generate_numerical_entities,
    sample_named_entities,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_variant_generation.fictional_document_variant_planning import (
    build_variant_sampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_mixed_temporal_rule_constraint_enforcement import (
    _affine_two_year_candidate_pairs,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_single_difference_constraint_enforcement import (
    SingleDifferenceConstraintEnforcementMixin,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.generation_exceptions import (
    StrictInterVariantUniquenessInfeasible,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.named_entity_pool_rule_policy import (
    copy_named_entity_pool_without_rule_based_mutation,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_uniqueness import (
    number_entity_uniqueness_value,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_generation.exact_solve_result import (
    ExactNumberSolveState,
)
from memoreason.factual_to_fictional_dataset.dataset_settings import parse_dataset_setting


def _load_number_test_document(template_path: Path) -> AnnotatedDocument:
    try:
        return load_annotated_document(str(template_path))
    except AnnotationValidationError:
        payload = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
        document_payload = payload.get("document") or {}
        questions = [
            Question(
                question_id=str(question.get("question_id") or ""),
                question=str(question.get("question") or ""),
                answer=str(question.get("answer") or ""),
                question_type=question.get("question_type"),
                answer_type=question.get("answer_type"),
            )
            for question in (document_payload.get("questions") or [])
            if isinstance(question, dict)
        ]
        return AnnotatedDocument(
            document_id=str(document_payload.get("document_id") or template_path.stem),
            document_theme=str(document_payload.get("document_theme") or template_path.parent.name),
            original_document=str(document_payload.get("original_document") or ""),
            document_to_annotate=str(document_payload.get("document_to_annotate") or ""),
            rules=list(document_payload.get("rules") or []),
            implicit_rules=list(document_payload.get("implicit_rules") or []),
            questions=questions,
            fictionalized_annotated_template_document=str(
                document_payload.get("fictionalized_annotated_template_document") or ""
            ),
        )


def test_generate_numbers_supports_str_numeric_comparison_rules() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=11),
            "number_2": NumberEntity(int=5, str="five"),
            "number_3": NumberEntity(int=11),
            "number_4": NumberEntity(int=5, str="five"),
            "number_5": NumberEntity(int=2, str="two"),
            "number_6": NumberEntity(int=100),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)

    required_numbers = [
        ("number_1", ["int"]),
        ("number_2", ["str"]),
        ("number_3", ["int"]),
        ("number_4", ["str"]),
        ("number_5", ["str"]),
        ("number_6", ["int"]),
    ]
    rules = [
        "number_1.int == number_3.int",
        "number_2.str == number_4.str",
        "number_5.str < number_6.int",
    ]
    avoid_values = {
        number_id: int(number_entity.int)
        for number_id, number_entity in factual_entities.numbers.items()
        if number_entity.int is not None
    }

    generated = generator.generate_numbers(
        required_numbers=required_numbers,
        rules=rules,
        existing_entities=EntityCollection(),
        avoid_values=avoid_values,
    )

    assert generated["number_1"].int == generated["number_3"].int
    assert generated["number_2"].str == generated["number_4"].str
    assert generated["number_5"].int < generated["number_6"].int


def test_number_range_parsing_does_not_match_longer_number_prefixes() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=20)}),
    )

    min_val, max_val = generator._get_number_range_from_rules(
        "number_1",
        rules=["number_15.str + number_16.str < number_14.int"],
        pre_assigned={},
        existing_entities=None,
    )

    assert (min_val, max_val) == (16, 24)


def test_generate_numbers_supports_large_awards_02_constraint_set() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/award_winners/awards_02.yaml")
    document = load_annotated_document(str(template_path))
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=False)
    required_numbers = FictionalEntitySampler.extract_required_entities(document, include_questions=False)["number"]
    numeric_rules = [rule for rule in document.rules if "number_" in str(rule) and "temporal_" not in str(rule)]
    avoid_values = {
        number_id: int(number_entity.int)
        for number_id, number_entity in factual_entities.numbers.items()
        if number_entity.int is not None
    }

    generated = NumberTemporalGenerator(seed=23, factual_entities=factual_entities).generate_numbers(
        required_numbers=required_numbers,
        rules=numeric_rules,
        existing_entities=EntityCollection(),
        avoid_values=avoid_values,
    )

    assert set(generated) == {number_id for number_id, _attrs in required_numbers}
    test_entities = EntityCollection(numbers=generated)
    validation = RuleEngine.validate_all_rules(numeric_rules, test_entities)
    assert all(is_valid for _, is_valid in validation)


def test_number_milp_tie_break_is_seed_deterministic() -> None:
    domains = {
        "number_1": (1, 100),
        "number_2": (1, 100),
        "number_3": (1, 100),
    }
    constraints = [
        (({"number_1": 1, "number_2": 1}, 0), "==", ({"number_3": 1}, 0)),
        (({"number_1": 1}, 0), "<", ({"number_2": 1}, 0)),
    ]

    first = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())._solve_numbers_via_milp(
        constraints,
        domains,
        avoid_values={},
    )
    second = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())._solve_numbers_via_milp(
        constraints,
        domains,
        avoid_values={},
    )

    assert first is not None
    assert first == second


def test_strict_number_milp_not_equal_disjunction_never_returns_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())
    constraints = [
        (({"number_1": 1, "number_2": -1}, 0), "!=", ({}, 1)),
    ]
    domains = {"number_1": (1, 2), "number_2": (1, 2)}
    objective_coefficients = iter((-1.0, 1.0))
    monkeypatch.setattr(
        "memoreason.factual_to_fictional_dataset.controlled_entity_replacement."
        "number_generation.milp_solver.random.uniform",
        lambda *_args: next(objective_coefficients),
    )

    result = generator._solve_numbers_via_milp_with_status(constraints, domains, avoid_values={})

    assert result.state is ExactNumberSolveState.SOLVED
    assert result.assignments is not None
    assert result.assignments["number_1"] - result.assignments["number_2"] != 1


def test_relaxed_number_milp_not_equal_disjunction_never_returns_equality() -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())
    constraints = [
        (({"number_1": 1, "number_2": -1}, 0), "!=", ({}, 1)),
    ]
    domains = {"number_1": (1, 2), "number_2": (1, 2)}

    # Avoidance makes (2, 1), the forbidden equality branch, the unique
    # zero-penalty optimum.  The relaxed solver must instead return a valid
    # one-penalty alternative rather than rely on post-validation to reject it.
    solved = generator._solve_numbers_via_relaxed_avoid_milp(
        constraints,
        domains,
        avoid_values={"number_1": {1}, "number_2": {2}},
    )

    assert solved is not None
    assert solved["number_1"] - solved["number_2"] != 1


def test_strict_number_milp_reports_fixed_not_equal_collision_as_infeasible() -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())
    constraints = [
        (({"number_1": 1, "number_2": -1}, 0), "!=", ({}, 1)),
    ]

    result = generator._solve_numbers_via_milp_with_status(
        constraints,
        {"number_1": (2, 2), "number_2": (1, 1)},
        avoid_values={},
    )

    assert result.state is ExactNumberSolveState.CERTIFIED_INFEASIBLE
    assert result.assignments is None


def test_certified_infeasible_number_milp_skips_redundant_csp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (1, 2))

    states: list[ExactNumberSolveState] = []
    real_solve = generator._solve_numbers_via_milp_with_status

    def tracking_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        states.append(result.state)
        return result

    monkeypatch.setattr(generator, "_solve_numbers_via_milp_with_status", tracking_solve)

    def unexpected_csp_candidates(*_args, **_kwargs):
        raise AssertionError("A status-2 exact system must not be searched again by CSP.")

    monkeypatch.setattr(generator, "_candidate_values", unexpected_csp_candidates)

    solved = generator._solve_numbers_via_constraints(
        ["number_1"],
        ["number_1.int >= 1", "number_1.int <= 2"],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": {1, 2}},
        allow_relaxed_avoid=False,
    )

    assert solved is None
    assert states == [ExactNumberSolveState.CERTIFIED_INFEASIBLE]


def test_number_milp_post_validation_failure_preserves_csp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (1, 2))

    # Simulate a status-0 backend incumbent that violates the exact `!=` rule.
    # This is not a proof of infeasibility, so it must stay tri-state
    # unavailable and allow the exact CSP to recover.
    monkeypatch.setattr(
        "memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_generation.milp_solver.milp",
        lambda **_kwargs: SimpleNamespace(success=True, status=0, x=[1.0, 1.0, 0.0]),
    )
    states: list[ExactNumberSolveState] = []
    real_solve = generator._solve_numbers_via_milp_with_status

    def tracking_solve(*args, **kwargs):
        result = real_solve(*args, **kwargs)
        states.append(result.state)
        return result

    monkeypatch.setattr(generator, "_solve_numbers_via_milp_with_status", tracking_solve)

    solved = generator._solve_numbers_via_constraints(
        ["number_1", "number_2"],
        ["number_1.int != number_2.int"],
        existing_entities=EntityCollection(),
        avoid_values={},
        allow_relaxed_avoid=False,
    )

    assert states == [ExactNumberSolveState.UNAVAILABLE_OR_UNSUPPORTED]
    assert solved is not None
    assert solved["number_1"] != solved["number_2"]


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("number_1.int / 2 > 0", 1),
        ("number_1.int / 1000000 != 0", 1),
    ],
)
def test_fractional_number_milp_status_two_preserves_exact_csp_fallback(
    monkeypatch: pytest.MonkeyPatch,
    rule: str,
    expected: int,
) -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())
    monkeypatch.setattr(generator, "_number_base_range", lambda _number_id: (0, 1))
    constraints = generator._collect_linear_constraints([rule], {"number_1"}, EntityCollection())
    assert constraints is not None

    milp_result = generator._solve_numbers_via_milp_with_status(
        constraints,
        {"number_1": (0, 1)},
        avoid_values={},
    )
    solved = generator._solve_numbers_via_constraints(
        ["number_1"],
        [rule],
        existing_entities=EntityCollection(),
        avoid_values={},
        allow_relaxed_avoid=False,
    )

    assert milp_result.state is ExactNumberSolveState.UNAVAILABLE_OR_UNSUPPORTED
    assert solved == {"number_1": expected}


def test_age_like_inline_numbers_do_not_force_global_numeric_ordering() -> None:
    document = load_annotated_document(
        "data/HUMAN_ANNOTATED_TEMPLATES/public_attacks_news_articles/dev_15.yaml",
        validate_question_scope=False,
    )
    context = build_controlled_entity_replacement_context(document)
    pool = load_entity_pool("data/GENERATED_FICTIONAL_ENTITIES/public_attacks_news_articles/dev_15_entity_pool.yaml")
    prepared_pool = generate_named_entities(context=context, entity_pool=pool, seed=23)
    named_sample = sample_named_entities(
        context=context,
        named_entities=prepared_pool,
        replacement_proportion=1.0,
        version_seed=23,
        reference_variant_index=0,
        reference_variant_count=10,
    )

    assert named_sample is not None

    numerical_sample = generate_numerical_entities(
        context=context,
        named_entity_sample=named_sample,
        named_entities=prepared_pool,
        version_seed=23,
        reference_variant_index=0,
        reference_variant_count=10,
        used_number_values_by_id={},
        used_temporal_years_by_id={},
        used_temporal_values_by_id={},
    )

    assert numerical_sample is not None
    for number_id, generated_number in numerical_sample.entities.numbers.items():
        factual_number = context.factual_entities_full.numbers[number_id]
        assert generated_number.int != factual_number.int


def test_generate_numbers_rejects_incomplete_exact_solver_assignments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23, factual_entities=EntityCollection())

    monkeypatch.setattr(
        generator,
        "_solve_numbers_via_constraints",
        lambda *args, **kwargs: {"number_1": 21},
    )
    monkeypatch.setattr(
        generator,
        "_get_number_range_from_rules",
        lambda *args, **kwargs: (5, 4),
    )

    with pytest.raises(ValueError, match="Unable to generate numbers satisfying rules"):
        generator.generate_numbers(
            required_numbers=[("number_1", ["int"]), ("number_2", ["int"])],
            rules=[],
            existing_entities=EntityCollection(),
        )


def test_generate_numbers_supports_scaled_space_01_constraint_set() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/space_missions/space_01.yaml")
    document = load_annotated_document(str(template_path))
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_numbers = FictionalEntitySampler.extract_required_entities(document, include_questions=True)["number"]
    numeric_rules = [rule for rule in document.rules if "number_" in str(rule) and "temporal_" not in str(rule)]

    generated = NumberTemporalGenerator(seed=23, factual_entities=factual_entities).generate_numbers(
        required_numbers=required_numbers,
        rules=numeric_rules,
        existing_entities=factual_entities.model_copy(deep=True),
        avoid_values={},
    )

    assert set(generated) == {number_id for number_id, _attrs in required_numbers}
    test_entities = factual_entities.model_copy(deep=True)
    test_entities.numbers.update(generated)
    validation = RuleEngine.validate_all_rules(numeric_rules, test_entities)
    assert all(is_valid for _, is_valid in validation)


def test_number_rule_subset_ignores_unavailable_temporal_rules() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=2)}),
    )

    selected = generator._number_evaluable_rules(
        [
            "number_1.int > 1",
            "temporal_1.year + 1 == temporal_2.year",
            "temporal_3.year + number_1.int == temporal_4.year",
        ],
        {"number_1"},
        existing_entities=EntityCollection(),
    )

    assert selected == ["number_1.int > 1"]


def test_exact_solver_accepts_forced_equal_factual_numbers() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=3, str="three")})
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["str"])],
        rules=["number_1.str == 3"],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": 3},
    )

    assert generated["number_1"].int == 3
    assert generator.last_number_forced_equal_refs == {"number_1.int", "number_1.str"}


def test_exact_solver_parses_quoted_number_words() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=4, str="four"),
            "number_2": NumberEntity(int=2, str="two"),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["str"]), ("number_2", ["str"])],
        rules=['number_1.str == "four"', "number_2.str > 1"],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": 4, "number_2": 2},
    )

    assert generated["number_1"].int == 4
    assert generated["number_2"].int != 2


def test_generate_numbers_supports_multiple_forbidden_values_per_number() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=0)}),
    )

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["int"])],
        rules=["number_1.int >= 1", "number_1.int <= 4"],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": {1, 2, 3}},
    )

    assert generated["number_1"].int == 4


def test_generate_numbers_supports_not_equal_expression_constraints() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=7),
            "number_2": NumberEntity(int=4),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["int"]), ("number_2", ["int"])],
        rules=["number_1.int - number_2.int != 3"],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": 7, "number_2": 4},
    )

    assert generated["number_1"].int - generated["number_2"].int != 3


def test_generate_numbers_exact_solver_marks_structurally_forced_forbidden_reuse() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=4, str="four"),
            "number_2": NumberEntity(int=5, str="five"),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["int"]), ("number_2", ["int"])],
        rules=[
            "number_1.int == 4",
            "number_2.int > number_1.int",
            "number_2.int <= 6",
        ],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": 4, "number_2": 5},
    )

    assert generated["number_1"].int == 4
    assert generated["number_2"].int == 6
    assert generator.last_relaxed_avoid_number_ids == {"number_1"}


def test_coupled_number_domains_expand_before_relaxing_avoidance() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=1),
            "number_2": NumberEntity(int=2),
            "number_3": NumberEntity(int=3),
        }
    )
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=[
            ImplicitRule(
                entity_ref=f"number_{index}.int",
                lower_bound=1,
                upper_bound=3,
                factual_value=index,
                percentage=20.0,
                rule_kind="number_range",
            )
            for index in range(1, 4)
        ],
    )
    avoid_values = {
        "number_1": {1, 2},
        "number_2": {1, 2, 3},
        "number_3": {2, 3},
    }

    generated = generator.generate_numbers(
        required_numbers=[
            ("number_1", ["int"]),
            ("number_2", ["int"]),
            ("number_3", ["int"]),
        ],
        rules=[],
        existing_entities=EntityCollection(),
        avoid_values=avoid_values,
    )

    values = {number_id: int(entity.int) for number_id, entity in generated.items()}
    assert values["number_1"] < values["number_2"] < values["number_3"]
    assert all(value not in avoid_values[number_id] for number_id, value in values.items())
    assert generator.last_number_solution_used_relaxed_avoid is False


def test_saturated_nonlinear_number_ref_does_not_relax_unrelated_ref() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=3),
            "number_2": NumberEntity(int=50),
        }
    )
    sampler = FictionalEntitySampler(
        {},
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.int",
                lower_bound=1,
                upper_bound=2,
                factual_value=3,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
        reference_variant_count=10,
        used_number_values_by_id={"number_1": {1, 2}, "number_2": {40}},
        allow_relaxed_intervariant_number_reuse=True,
    )

    generated = sampler.generate_numerical_entities(
        required_entities={
            "number": [
                ("number_1", ["int"]),
                ("number_2", ["int"]),
            ]
        },
        rules=[
            "number_1.int * number_1.int <= 4",
            "number_2.int >= 40",
            "number_2.int <= 60",
        ],
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert generated is not None
    assert generated.numbers["number_1"].int in {1, 2}
    assert generated.numbers["number_2"].int not in {40, 50}


def test_saturated_linear_number_ref_is_available_to_difference_repair() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=2, str="two")})
    sampler = FictionalEntitySampler(
        {},
        seed=0,
        factual_entities=factual_entities,
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.int",
                lower_bound=1,
                upper_bound=2,
                factual_value=2,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
        reference_variant_count=10,
        used_number_values_by_id={"number_1": {1}},
        allow_relaxed_intervariant_number_reuse=True,
    )

    generated = sampler.generate_numerical_entities(
        required_entities={"number": [("number_1", ["int"])]},
        rules=["number_1.int <= 2"],
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert generated is not None
    assert generated.numbers["number_1"].int == 1


def test_relaxed_sampler_allows_required_number_reuse_for_mixed_temporal_feasibility() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=3, str="three"),
            "number_2": NumberEntity(int=50, str="fifty"),
        },
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2001),
        },
    )
    sampler_kwargs = {
        "entity_pool": {},
        "seed": 23,
        "factual_entities": factual_entities,
        "used_number_values_by_id": {"number_1": {2}, "number_2": {40}},
        "reference_variant_count": 10,
    }
    required_entities = {
        "number": [("number_1", ["int"]), ("number_2", ["int"])],
        "temporal": [("temporal_1", ["year"]), ("temporal_2", ["year"])],
    }
    rules = [
        "temporal_1.year == 1998",
        "temporal_2.year == 1999",
        "temporal_2.year - temporal_1.year + 1 == number_1.int",
        "number_2.int >= 40",
        "number_2.int <= 60",
    ]

    strict = FictionalEntitySampler(**sampler_kwargs).generate_numerical_entities(
        required_entities=required_entities,
        rules=rules,
        named_entities=EntityCollection(),
        max_attempts=1,
        decade_year_temporal_ids=set(),
    )
    relaxed_sampler = FictionalEntitySampler(
        **sampler_kwargs,
        allow_relaxed_intervariant_number_reuse=True,
    )
    relaxed = relaxed_sampler.generate_numerical_entities(
        required_entities=required_entities,
        rules=rules,
        named_entities=EntityCollection(),
        max_attempts=1,
        decade_year_temporal_ids=set(),
    )

    assert strict is None
    assert relaxed is not None
    assert relaxed.numbers["number_1"].int == 2
    assert relaxed.numbers["number_1"].int != factual_entities.numbers["number_1"].int
    assert relaxed.numbers["number_2"].int not in {40, 50}
    assert relaxed_sampler.last_relaxed_numtemp_reuse_audit == [
        {
            "entity_bucket": "numbers",
            "entity_ref": "number_1",
            "entity_attr": "",
            "value": {"kind": "int", "value": 2},
            "own_factual_value": {"kind": "int", "value": 3},
            "reason": "strict_sampler_budget_exhausted_then_minimum_reuse_relaxed_attempt",
        }
    ]


def test_expandable_implicit_number_range_is_not_certified_for_reuse() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=3, str="three")})
    sampler = FictionalEntitySampler(
        {},
        seed=0,
        factual_entities=factual_entities,
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.int",
                lower_bound=1,
                upper_bound=2,
                factual_value=3,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
        reference_variant_count=10,
        used_number_values_by_id={"number_1": {1, 2}},
        allow_relaxed_intervariant_number_reuse=True,
    )

    generated = sampler.generate_numerical_entities(
        required_entities={"number": [("number_1", ["int"])]},
        rules=["number_1.int * number_1.int >= 1"],
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert generated is not None
    assert generated.numbers["number_1"].int == 4


def test_saturated_heuristic_nonlinear_base_expands_before_reuse() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=3, str="three")})
    base_candidates = set(range(1, 14)) - {3}
    sampler = FictionalEntitySampler(
        {},
        seed=0,
        factual_entities=factual_entities,
        reference_variant_count=20,
        used_number_values_by_id={"number_1": base_candidates},
        allow_relaxed_intervariant_number_reuse=True,
    )

    generated = sampler.generate_numerical_entities(
        required_entities={"number": [("number_1", ["int"])]},
        rules=["number_1.int * number_1.int >= 1"],
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert generated is not None
    assert generated.numbers["number_1"].int not in base_candidates
    assert generated.numbers["number_1"].int != 3


def test_intervariant_float_uniqueness_uses_clamped_surface_value() -> None:
    first = NumberEntity(int=2, float=1.9)
    second = NumberEntity(int=3, float=1.9)
    sampler = FictionalEntitySampler(
        {},
        used_number_values_by_id={"number_1": {1.9}},
    )

    assert sampler._number_intervariant_value(first) == 1.9
    assert sampler._number_intervariant_value(second) == 1.9
    assert not sampler._intervariant_numtemp_values_are_valid(
        collection=EntityCollection(numbers={"number_1": second}),
        required_numbers=[("number_1", ["float"])],
        required_temporals=[],
        allowed_number_reuse_ids=set(),
        allowed_temporal_reuse_refs=set(),
        fixed_required_refs=set(),
    )


def test_fractional_strict_bound_cannot_certify_saturation_while_unused_integer_exists() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=3, str="three")})
    sampler = FictionalEntitySampler(
        {},
        seed=0,
        factual_entities=factual_entities,
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.int",
                lower_bound=1,
                upper_bound=2,
                factual_value=3,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
        reference_variant_count=10,
        used_number_values_by_id={"number_1": {1}},
        allow_relaxed_intervariant_number_reuse=True,
    )
    generator = NumberTemporalGenerator(seed=0, factual_entities=factual_entities)
    rules = ["number_1.int < 2.5"]

    saturated = sampler._saturated_hard_bounded_linear_number_ids(
        generator=generator,
        required_number_ids={"number_1"},
        rules=rules,
        existing_entities=EntityCollection(),
    )
    generated = sampler.generate_numerical_entities(
        required_entities={"number": [("number_1", ["int"])]},
        rules=rules,
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert saturated == set()
    assert generated is not None
    assert generated.numbers["number_1"].int == 2


def test_unsaturated_nonlinear_low_cardinality_ref_stays_unique() -> None:
    sampler = FictionalEntitySampler(
        {},
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=3)}),
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.int",
                lower_bound=1,
                upper_bound=2,
                factual_value=3,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
        reference_variant_count=10,
        used_number_values_by_id={"number_1": {1}},
        allow_relaxed_intervariant_number_reuse=True,
    )

    generated = sampler.generate_numerical_entities(
        required_entities={"number": [("number_1", ["int"])]},
        rules=["number_1.int * number_1.int <= 4"],
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert generated is not None
    assert generated.numbers["number_1"].int == 2


def test_awards_06_ten_variant_number_domain_stays_unique() -> None:
    document = load_annotated_document(
        "data/HUMAN_ANNOTATED_TEMPLATES/award_winners/awards_06.yaml",
        validate_question_scope=False,
    )
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=False)
    required_entities = FictionalEntitySampler.extract_required_entities(document, include_questions=False)
    numerical_requirements = {
        entity_type: specs for entity_type, specs in required_entities.items() if entity_type in {"number", "temporal"}
    }
    used_numbers: dict[str, set[int | float]] = {}
    used_years: dict[str, set[int]] = {}
    used_temporals: dict[str, dict[str, set]] = {}
    number_values: dict[str, list[int]] = {"number_1": [], "number_5": []}

    for variant_index in range(10):
        sampler_kwargs = {
            "entity_pool": {},
            "seed": 23 + (variant_index * 1_000_000),
            "factual_entities": factual_entities,
            "implicit_rules": document.implicit_rules,
            "reference_variant_count": 10,
            "used_number_values_by_id": used_numbers,
            "used_temporal_years_by_id": used_years,
            "used_temporal_values_by_id": used_temporals,
            "source_document_text": document.document_to_annotate,
        }
        sampler = FictionalEntitySampler(**sampler_kwargs)
        try:
            generated = sampler.generate_numerical_entities(
                required_entities=numerical_requirements,
                rules=document.rules,
                named_entities=EntityCollection(),
                max_attempts=2,
            )
        except StrictInterVariantUniquenessInfeasible:
            generated = None
        if generated is None:
            generated = FictionalEntitySampler(
                **sampler_kwargs,
                allow_relaxed_intervariant_number_reuse=True,
            ).generate_numerical_entities(
                required_entities=numerical_requirements,
                rules=document.rules,
                named_entities=EntityCollection(),
                max_attempts=2,
            )

        assert generated is not None
        for number_id in number_values:
            number_values[number_id].append(int(generated.numbers[number_id].int))
        for number_id, number in generated.numbers.items():
            value = FictionalEntitySampler._number_intervariant_value(number)
            if value is not None:
                used_numbers.setdefault(number_id, set()).add(value)
        for temporal_id, temporal in generated.temporals.items():
            if temporal.year is not None:
                used_years.setdefault(temporal_id, set()).add(int(temporal.year))
            for attr in ("year", "month", "day", "day_of_month", "timestamp"):
                value = getattr(temporal, attr, None)
                if value not in (None, ""):
                    used_temporals.setdefault(temporal_id, {}).setdefault(attr, set()).add(value)

    assert len(set(number_values["number_1"])) == 10
    assert 2 not in number_values["number_1"]
    assert len(set(number_values["number_5"])) == 10
    assert 17 in number_values["number_5"]


def test_space_12_singleton_year_reuse_keeps_unrelated_refs_unique() -> None:
    document = load_annotated_document(
        "data/HUMAN_ANNOTATED_TEMPLATES/space_missions/space_12.yaml",
        validate_question_scope=False,
    )
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=False)
    required_entities = FictionalEntitySampler.extract_required_entities(document, include_questions=False)
    numerical_requirements = {
        entity_type: specs for entity_type, specs in required_entities.items() if entity_type in {"number", "temporal"}
    }
    decade_year_temporal_ids = FictionalEntitySampler.extract_decade_year_temporal_ids(
        document,
        include_questions=False,
    )
    assert "temporal_10" in decade_year_temporal_ids

    first = FictionalEntitySampler(
        {},
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
        reference_variant_count=10,
        source_document_text=document.document_to_annotate,
    ).generate_numerical_entities(
        required_entities=numerical_requirements,
        rules=document.rules,
        named_entities=EntityCollection(),
        max_attempts=2,
        decade_year_temporal_ids=decade_year_temporal_ids,
    )
    assert first is not None
    assert first.temporals["temporal_10"].year == 2020

    used_numbers: dict[str, set[int | float]] = {}
    used_years: dict[str, set[int]] = {}
    used_temporals: dict[str, dict[str, set]] = {}
    for number_id, number in first.numbers.items():
        value = FictionalEntitySampler._number_intervariant_value(number)
        if value is not None:
            used_numbers.setdefault(number_id, set()).add(value)
    for temporal_id, temporal in first.temporals.items():
        if temporal.year is not None:
            used_years.setdefault(temporal_id, set()).add(int(temporal.year))
        for attr in ("year", "month", "day", "day_of_month", "timestamp"):
            value = getattr(temporal, attr, None)
            if value not in (None, ""):
                used_temporals.setdefault(temporal_id, {}).setdefault(attr, set()).add(value)

    sampler_kwargs = {
        "entity_pool": {},
        "seed": 1_000_023,
        "factual_entities": factual_entities,
        "implicit_rules": document.implicit_rules,
        "reference_variant_count": 10,
        "used_number_values_by_id": used_numbers,
        "used_temporal_years_by_id": used_years,
        "used_temporal_values_by_id": used_temporals,
        "source_document_text": document.document_to_annotate,
    }
    strict_second = FictionalEntitySampler(**sampler_kwargs).generate_numerical_entities(
        required_entities=numerical_requirements,
        rules=document.rules,
        named_entities=EntityCollection(),
        max_attempts=2,
        decade_year_temporal_ids=decade_year_temporal_ids,
    )
    assert strict_second is None

    relaxed_second = FictionalEntitySampler(
        **sampler_kwargs,
        allow_relaxed_intervariant_number_reuse=True,
    ).generate_numerical_entities(
        required_entities=numerical_requirements,
        rules=document.rules,
        named_entities=EntityCollection(),
        max_attempts=2,
        decade_year_temporal_ids=decade_year_temporal_ids,
    )
    assert relaxed_second is not None
    assert relaxed_second.temporals["temporal_10"].year == 2020

    for number_id, _attrs in numerical_requirements["number"]:
        value = FictionalEntitySampler._number_intervariant_value(relaxed_second.numbers[number_id])
        assert value not in used_numbers[number_id]
    for temporal_id, attrs in numerical_requirements["temporal"]:
        for attr in FictionalEntitySampler._effective_temporal_attrs(attrs):
            value = getattr(relaxed_second.temporals[temporal_id], attr, None)
            if value in (None, ""):
                continue
            if (temporal_id, attr) == ("temporal_10", "year"):
                continue
            assert value not in used_temporals.get(temporal_id, {}).get(attr, set())


def test_awards_03_numeric_generation_satisfies_reviewed_and_ordering_constraints() -> None:
    document = load_annotated_document(
        "data/HUMAN_ANNOTATED_TEMPLATES/award_winners/awards_03.yaml",
        validate_question_scope=False,
    )
    context = build_controlled_entity_replacement_context(document)
    pool = load_entity_pool("data/GENERATED_FICTIONAL_ENTITIES/award_winners/awards_03_entity_pool.yaml")
    setting = parse_dataset_setting("fictional")
    prepared_pool = generate_named_entities(context=context, entity_pool=pool, seed=23)
    named_sample = sample_named_entities(
        context=context,
        named_entities=prepared_pool,
        replacement_proportion=setting.replacement_proportion,
        version_seed=23,
        replace_mode=setting.replace_mode,
        reference_variant_index=0,
        reference_variant_count=10,
    )

    assert named_sample is not None

    sampler = build_variant_sampler(
        context=context,
        entity_pool=prepared_pool,
        replacement_layout=named_sample.replacement_layout,
        version_seed=23,
        eligible_cache=None,
        reference_variant_index=0,
        reference_variant_count=10,
        used_number_values_by_id={},
        used_temporal_years_by_id={},
    )

    generated = sampler.generate_numerical_entities(
        required_entities=named_sample.fictional_requirements,
        rules=context.generation_document.rules,
        named_entities=named_sample.entities,
        max_attempts=1,
        decade_year_temporal_ids=set(named_sample.decade_year_temporal_ids),
    )

    assert generated is not None
    assert all(
        is_valid
        for _rule, is_valid in RuleEngine.validate_all_rules(
            context.generation_document.rules,
            generated,
        )
    )
    assert generated.numbers["number_22"].int < generated.numbers["number_1"].int


def test_forced_equal_markers_are_bound_to_the_final_factual_value() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=8, str="eight")})
    sampler = FictionalEntitySampler({}, seed=23, factual_entities=factual_entities)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    generator.last_number_forced_equal_refs = {"number_1.int", "number_1.str"}

    repaired_collection = EntityCollection(numbers={"number_1": NumberEntity(int=7, str="seven")})
    assert (
        sampler._current_forced_required_refs(
            generator=generator,
            rules=[],
            collection=repaired_collection,
        )
        == set()
    )

    factual_collection = EntityCollection(numbers={"number_1": NumberEntity(int=8, str="eight")})
    assert sampler._current_forced_required_refs(
        generator=generator,
        rules=[],
        collection=factual_collection,
    ) == {"number_1.int", "number_1.str"}

    missing_collection = EntityCollection()
    assert (
        sampler._current_forced_required_refs(
            generator=generator,
            rules=[],
            collection=missing_collection,
        )
        == set()
    )


def test_successful_final_number_difference_repair_clears_stale_forced_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RepairHarness(SingleDifferenceConstraintEnforcementMixin):
        def __init__(self) -> None:
            self.used_number_values_by_id = {"number_3": {6}}

        @staticmethod
        def _rules_referencing_entity_ids(rules, _entity_ids):
            return list(rules)

        @staticmethod
        def _verify_ordering_preserved(_collection, **_kwargs):
            return True

        @staticmethod
        def _validate_rules_with_details(_rules, collection):
            temporal = collection.temporals.get("temporal_1")
            return {"all_valid": temporal is not None and temporal.year == 1992}

        @staticmethod
        def _order_preserving_temporal_year_candidates(**_kwargs):
            return []

        @staticmethod
        def _concretize_temporal_rules_with_numbers(*, temporal_rules, **_kwargs):
            return list(temporal_rules)

    class RepairGenerator:
        def __init__(self) -> None:
            self.exclude_temporals = {"years": set()}
            self.last_number_forced_equal_refs = {"number_3.int", "number_3.str"}

        @staticmethod
        def _number_actual_bounds(_number_id, _required_attrs):
            return 7, 7

        @staticmethod
        def _set_number_actual_value(_number_id, number_entity, value, **_kwargs):
            number_entity.int = int(value)
            number_entity.str = "seven"

        @staticmethod
        def _forced_equal_refs_for_number(_number_id):
            return {"number_3.int", "number_3.str"}

        @staticmethod
        def _update_temporal_year(temporal, year):
            updated = temporal.model_copy(deep=True)
            updated.year = int(year)
            return updated

        @staticmethod
        def generate_temporals_with_rules(*_args, **_kwargs):
            return {
                "temporal_1": TemporalEntity(year=1992),
                "temporal_2": TemporalEntity(year=1998),
            }

    collection = EntityCollection(
        numbers={"number_3": NumberEntity(int=8, str="eight")},
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2007),
        },
    )
    generator = RepairGenerator()
    monkeypatch.setattr(
        "memoreason.factual_to_fictional_dataset.controlled_entity_replacement."
        "fictional_entity_single_difference_constraint_enforcement._affine_two_year_candidate_pairs",
        lambda **_kwargs: [],
    )
    repaired = RepairHarness()._repair_single_required_number_difference(
        generator=generator,
        collection=collection,
        number_id="number_3",
        attr="str",
        factual_value="eight",
        required_attrs={"str"},
        rules_with_ordering=["number_3.str == temporal_2.year - temporal_1.year + 1"],
        ordering_exempt_ids=set(),
    )

    assert repaired is True
    assert collection.numbers["number_3"].int == 7
    assert generator.last_number_forced_equal_refs == set()


def test_final_number_difference_repair_uses_only_ordered_affine_year_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RepairHarness(SingleDifferenceConstraintEnforcementMixin):
        def __init__(self) -> None:
            self.used_number_values_by_id: dict[str, set[int]] = {}

        @staticmethod
        def _rules_referencing_entity_ids(rules, _entity_ids):
            return list(rules)

        @staticmethod
        def _verify_ordering_preserved(_collection, **_kwargs):
            return True

        @staticmethod
        def _validate_rules_with_details(_rules, collection):
            number = collection.numbers["number_3"].int
            left = collection.temporals["temporal_1"].year
            right = collection.temporals["temporal_2"].year
            return {"all_valid": number == right - left + 1}

        @staticmethod
        def _order_preserving_temporal_year_candidates(*, temporal_id, **_kwargs):
            # Much larger than a production temporal window: a Cartesian
            # implementation would materialise 90,000 candidate pairs.
            assert temporal_id in {"temporal_1", "temporal_2"}
            return list(range(1_000, 1_300))

    factual_entities = EntityCollection(
        numbers={"number_3": NumberEntity(int=8, str="eight")},
        temporals={
            "temporal_1": TemporalEntity(year=1_100),
            "temporal_2": TemporalEntity(year=1_107),
        },
    )
    collection = factual_entities.model_copy(deep=True)
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    monkeypatch.setattr(generator, "_number_actual_bounds", lambda *_args, **_kwargs: (7, 7))

    helper_calls: list[tuple[int, int, int]] = []

    def tracking_affine_pairs(**kwargs):
        pairs = _affine_two_year_candidate_pairs(**kwargs)
        helper_calls.append((len(kwargs["left_domain"]), len(kwargs["right_domain"]), len(pairs or [])))
        return pairs

    monkeypatch.setattr(
        "memoreason.factual_to_fictional_dataset.controlled_entity_replacement."
        "fictional_entity_single_difference_constraint_enforcement._affine_two_year_candidate_pairs",
        tracking_affine_pairs,
    )
    evaluated_pairs: list[tuple[int, int]] = []
    original_evaluate_expression = RuleEngine.evaluate_expression

    def tracking_evaluate_expression(rule, candidate_collection):
        evaluated_pairs.append(
            (
                candidate_collection.temporals["temporal_1"].year,
                candidate_collection.temporals["temporal_2"].year,
            )
        )
        return original_evaluate_expression(rule, candidate_collection)

    monkeypatch.setattr(RuleEngine, "evaluate_expression", tracking_evaluate_expression)

    repaired = RepairHarness()._repair_single_required_number_difference(
        generator=generator,
        collection=collection,
        number_id="number_3",
        attr="str",
        factual_value="eight",
        required_attrs={"str"},
        rules_with_ordering=["number_3.str == temporal_2.year - temporal_1.year + 1"],
        ordering_exempt_ids=set(),
    )

    assert repaired is True
    assert helper_calls == [(300, 300, 294)]
    # The previous Cartesian implementation used the same proximity/span/id
    # ordering.  The affine solver must preserve that deterministic first pair.
    assert evaluated_pairs == [(1_100, 1_106)]
    assert collection.temporals["temporal_1"].year == 1_100
    assert collection.temporals["temporal_2"].year == 1_106


def test_final_number_difference_repair_scopes_prior_variant_reuse_per_number() -> None:
    class RepairHarness(SingleDifferenceConstraintEnforcementMixin):
        def __init__(self) -> None:
            self.used_number_values_by_id = {"number_3": {7}}

        @staticmethod
        def _rules_referencing_entity_ids(rules, _entity_ids):
            return list(rules)

        @staticmethod
        def _verify_ordering_preserved(_collection, **_kwargs):
            return True

        @staticmethod
        def _validate_rules_with_details(_rules, _collection):
            return {"all_valid": True}

    class RepairGenerator:
        def __init__(self, low: int, high: int) -> None:
            self.low = low
            self.high = high
            self.last_number_forced_equal_refs: set[str] = set()

        def _number_actual_bounds(self, _number_id, _required_attrs):
            return self.low, self.high

        @staticmethod
        def _set_number_actual_value(_number_id, number_entity, value, **_kwargs):
            integer_value = int(value)
            number_entity.int = integer_value
            number_entity.str = {7: "seven", 8: "eight", 9: "nine"}[integer_value]

        @staticmethod
        def _forced_equal_refs_for_number(_number_id):
            return {"number_3.int", "number_3.str"}

    def repair_with_bounds(
        low: int,
        high: int,
        allowed_number_reuse_ids: set[str] | None = None,
        initial_forced_equal_refs: set[str] | None = None,
    ):
        collection = EntityCollection(numbers={"number_3": NumberEntity(int=8, str="eight")})
        generator = RepairGenerator(low, high)
        generator.last_number_forced_equal_refs.update(initial_forced_equal_refs or set())
        repaired = RepairHarness()._repair_single_required_number_difference(
            generator=generator,
            collection=collection,
            number_id="number_3",
            attr="str",
            factual_value="eight",
            required_attrs={"str"},
            rules_with_ordering=[],
            ordering_exempt_ids=set(),
            allowed_number_reuse_ids=allowed_number_reuse_ids,
        )
        return repaired, collection, generator

    repaired, collection, generator = repair_with_bounds(7, 9)
    assert repaired is True
    assert collection.numbers["number_3"].int == 9
    assert generator.last_number_forced_equal_refs == set()

    repaired, collection, generator = repair_with_bounds(7, 7)
    assert repaired is False
    assert collection.numbers["number_3"].int == 8
    assert generator.last_number_forced_equal_refs == set()

    repaired, collection, generator = repair_with_bounds(7, 7, {"number_3"})
    assert repaired is True
    assert collection.numbers["number_3"].int == 7
    assert generator.last_number_forced_equal_refs == set()

    repaired, collection, generator = repair_with_bounds(7, 7, {"number_4"})
    assert repaired is False
    assert collection.numbers["number_3"].int == 8
    assert generator.last_number_forced_equal_refs == set()

    repaired, collection, generator = repair_with_bounds(
        7,
        7,
        initial_forced_equal_refs={"number_3.str"},
    )
    assert repaired is False
    assert collection.numbers["number_3"].int == 8
    assert generator.last_number_forced_equal_refs == {"number_3.str"}


def test_exact_solver_parses_unquoted_number_words() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=4, str="four"),
            "number_2": NumberEntity(int=2, str="two"),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["str"]), ("number_2", ["str"])],
        rules=["number_1.str == four", "number_2.str > 1"],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": 4, "number_2": 2},
    )

    assert generated["number_1"].int == 4
    assert generated["number_2"].int != 2


def test_generate_numbers_supports_sport_08_constraint_set() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/sport_events/sport_08.yaml")
    document = load_annotated_document(str(template_path))
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_numbers = FictionalEntitySampler.extract_required_entities(document, include_questions=True)["number"]
    numeric_rules = [rule for rule in document.rules if "number_" in str(rule) and "temporal_" not in str(rule)]
    existing_entities = factual_entities.model_copy(deep=True)
    required_ids = {number_id for number_id, _attrs in required_numbers}
    existing_entities.numbers = {
        number_id: entity for number_id, entity in existing_entities.numbers.items() if number_id not in required_ids
    }
    avoid_values = {
        number_id: int(number_entity.int)
        for number_id, number_entity in factual_entities.numbers.items()
        if number_id in required_ids and number_entity.int is not None
    }

    generated = NumberTemporalGenerator(seed=23, factual_entities=factual_entities).generate_numbers(
        required_numbers=required_numbers,
        rules=numeric_rules,
        existing_entities=existing_entities,
        avoid_values=avoid_values,
    )

    assert set(generated) == {number_id for number_id, _attrs in required_numbers}
    test_entities = existing_entities.model_copy(deep=True)
    test_entities.numbers.update(generated)
    validation = RuleEngine.validate_all_rules(numeric_rules, test_entities)
    assert all(is_valid for _, is_valid in validation)


def test_implicit_number_range_prefers_canonical_numeric_window_when_refs_conflict() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/natural_disasters/disaster_05.yaml")
    document = load_annotated_document(str(template_path))
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)

    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
    )

    assert generator._implicit_number_range("number_22") == (19, 23)


def test_number_base_range_expands_when_forbidden_values_exhaust_window() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=10, str="ten")}),
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.str",
                lower_bound=8.0,
                upper_bound=12.0,
                factual_value=10.0,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
    )

    generator._current_number_avoid_values = {"number_1": {7, 8, 9, 10, 11, 12, 13}}
    try:
        assert generator._number_base_range("number_1") == (6, 14)
    finally:
        generator._current_number_avoid_values = {}


def test_generate_numbers_expands_implicit_window_when_all_original_values_are_forbidden() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=10, str="ten")}),
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.str",
                lower_bound=8.0,
                upper_bound=12.0,
                factual_value=10.0,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
    )

    generated = generator.generate_numbers(
        required_numbers=[("number_1", ["str"])],
        rules=[],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": {8, 9, 10, 11, 12}},
    )

    assert generated["number_1"].int not in {8, 9, 10, 11, 12}


def test_fraction_number_base_range_respects_minimum_valid_denominator() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(int=6, fraction="one sixth")}),
        implicit_rules=[],
    )

    generator._current_number_avoid_values = {"number_1": {2, 3, 4, 5, 6, 7, 8, 9}}
    try:
        low, _high = generator._number_base_range("number_1")
    finally:
        generator._current_number_avoid_values = {}

    assert low >= 2


def test_exact_solver_does_not_repair_back_into_forbidden_values_for_place_01() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/cities_countries_and_regions/place_01.yaml")
    document = load_annotated_document(str(template_path))
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_numbers = FictionalEntitySampler.extract_required_entities(document, include_questions=True)["number"]
    numeric_rules = [rule for rule in document.rules if "number_" in str(rule) and "temporal_" not in str(rule)]
    numeric_rules.extend(
        [
            "number_10.int < number_8.int",
            "number_8.int < number_9.int",
            "number_9.int < number_1.int",
            "number_1.int < number_4.int",
            "number_4.int < number_2.int",
        ]
    )
    avoid_values = {
        "number_1": {23, 24, 27, 31},
        "number_2": {3386604, 3386605, 4233254, 4233255},
        "number_4": {360, 361, 362, 450},
        "number_5": {5, 5.5, 6},
        "number_6": {15, 16, 17.935, 18},
        "number_7": {5},
        "number_8": {17, 18, 20, 21},
        "number_9": {18, 19, 22, 26},
        "number_10": {8, 9, 10, 11},
    }

    generated = NumberTemporalGenerator(
        seed=3_000_023,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
    ).generate_numbers(
        required_numbers=required_numbers,
        rules=numeric_rules,
        existing_entities=EntityCollection(),
        avoid_values=avoid_values,
    )

    for number_id, forbidden in avoid_values.items():
        if number_id not in generated:
            continue
        assert generated[number_id].int not in {int(value) for value in forbidden}


@pytest.mark.parametrize(
    ("theme", "doc_id"),
    [
        ("companies_and_organizations", "company_02"),
        ("companies_and_organizations", "company_04"),
        ("natural_disasters", "disaster_02"),
        ("natural_disasters", "disaster_05"),
        ("retail_banking_regulations_and_policies", "bankreg_07"),
    ],
)
def test_sample_entities_preserves_numeric_order_for_hard_documents(theme: str, doc_id: str) -> None:
    template_path = Path(f"data/HUMAN_ANNOTATED_TEMPLATES/{theme}/{doc_id}.yaml")
    pool_path = Path(f"data/GENERATED_FICTIONAL_ENTITIES/{theme}/{doc_id}_entity_pool.yaml")
    document = _load_number_test_document(template_path)
    include_questions = not (theme == "companies_and_organizations" and doc_id == "company_04")
    kept_rules, _dropped_rules = partition_generation_rules(document, include_questions=include_questions)
    active_document = document.model_copy(update={"rules": kept_rules})
    factual_entities = AnnotationParser.extract_factual_entities(active_document, include_questions=include_questions)
    required_entities = FictionalEntitySampler.extract_required_entities(
        active_document, include_questions=include_questions
    )
    aligned_pool = copy_named_entity_pool_without_rule_based_mutation(load_entity_pool(str(pool_path)))

    sampler = FictionalEntitySampler(
        aligned_pool,
        seed=23,
        factual_entities=factual_entities,
        implicit_rules=active_document.implicit_rules,
    )
    local_shortages = sampler.find_manual_pool_shortages(required_entities)
    if local_shortages:
        pytest.skip(f"local pool is structurally insufficient for {theme}/{doc_id}")
    sampled = sampler.sample_fictional_entities(
        required_entities,
        active_document.rules,
        decade_year_temporal_ids=FictionalEntitySampler.extract_decade_year_temporal_ids(
            active_document,
            include_questions=include_questions,
        ),
    )

    assert sampled is not None


def test_space_08_v03_orbital_repair_derives_strict_solution_beyond_preferred_window() -> None:
    document = load_annotated_document(
        "data/HUMAN_ANNOTATED_TEMPLATES/space_missions/space_08.yaml",
        validate_question_scope=False,
    )
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    generator = NumberTemporalGenerator(
        seed=2_000_023,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
    )
    sampler = FictionalEntitySampler(
        {},
        seed=2_000_023,
        factual_entities=factual_entities,
        implicit_rules=document.implicit_rules,
        reference_variant_index=2,
        reference_variant_count=10,
    )
    required_attr_map = {
        "number_13": {"str"},
        "number_9": {"float"},
        "number_14": {"int"},
        "number_7": {"int"},
        "number_15": {"int"},
    }
    collection = EntityCollection(
        numbers={
            number_id: generator._build_number_entity(number_id, value, required_attrs=required_attr_map[number_id])
            for number_id, value in {
                "number_13": 5,
                "number_9": 131,
                "number_14": 1136,
                "number_7": 1,
                "number_15": 84_000_016,
            }.items()
        }
    )
    prior_and_factual_values = {
        "number_13": {2, 3, 4},
        "number_9": {80, 96, 100},
        "number_14": {1152, 1440, 1728},
        "number_7": {7, 8, 11},
        "number_15": {60_825_600, 70_000_000, 72_576_000},
    }
    orbital_rules = [
        str(rule).split("#", 1)[0].strip()
        for rule in document.rules
        if "number_14.int * number_9.float" in str(rule)
        or "number_7.int * number_9.float" in str(rule)
    ]

    sampler._repair_orbital_product_number_rules(
        generator=generator,
        collection=collection,
        numeric_rules=orbital_rules,
        required_attr_map=required_attr_map,
        avoid_numbers=prior_and_factual_values,
    )

    assert all(is_valid for _rule, is_valid in RuleEngine.validate_all_rules(orbital_rules, collection))
    assert collection.numbers["number_13"].int == 5
    assert collection.numbers["number_9"].float == 79.0
    assert collection.numbers["number_14"].int == 2735
    assert collection.numbers["number_7"].int == 10
    assert collection.numbers["number_15"].int == 129_639_000
    assert collection.numbers["number_14"].int > 1728
    for number_id, forbidden_values in prior_and_factual_values.items():
        assert number_entity_uniqueness_value(collection.numbers[number_id]) not in forbidden_values


def test_relaxed_reuse_score_prefers_least_used_value_before_canonical_order() -> None:
    sampler = FictionalEntitySampler(
        {},
        prior_relaxed_intervariant_reuse_audit=[
            {
                "entity_bucket": "numbers",
                "entity_ref": "number_1",
                "entity_attr": "",
                "value": {"kind": "int", "value": 2},
            }
        ],
    )
    reused_twice = [
        {
            "entity_bucket": "numbers",
            "entity_ref": "number_1",
            "entity_attr": "",
            "value": {"kind": "int", "value": 2},
        }
    ]
    reused_once = [
        {
            "entity_bucket": "numbers",
            "entity_ref": "number_1",
            "entity_attr": "",
            "value": {"kind": "int", "value": 7},
        }
    ]

    assert sampler._runtime_reuse_candidate_score(reused_once) < sampler._runtime_reuse_candidate_score(
        reused_twice
    )
