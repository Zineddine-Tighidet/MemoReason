from datetime import timedelta

import pytest

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_mixed_temporal_rule_constraint_enforcement import (
    _affine_two_year_candidate_pairs,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.temporal_generation.exact_solve_result import (
    ExactTemporalSolveResult,
    ExactTemporalSolveState,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.generation_requirements import (
    extract_decade_year_temporal_ids,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.sampling_checks import (
    verify_ordering_preserved,
)
from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity, PersonEntity, TemporalEntity
from memoreason.benchmark_definition.annotation_runtime import AnnotationParser, load_annotated_document


def test_number_base_range_uses_twenty_percent_window() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=100)})

    generator = NumberTemporalGenerator(factual_entities=factual_entities)

    assert generator._number_base_range("number_1") == (80, 120)


def test_small_number_base_range_uses_fixed_plus_minus_ten_window() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity(int=2)})

    generator = NumberTemporalGenerator(factual_entities=factual_entities)

    assert generator._number_base_range("number_1") == (1, 12)


def test_number_generation_preserves_factual_numeric_ordering() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=4, str="four"),
            "number_2": NumberEntity(int=5, str="five"),
        }
    )
    generator = NumberTemporalGenerator(seed=13, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        [("number_1", ["str"]), ("number_2", ["str"])],
        rules=[],
        existing_entities=EntityCollection(),
        avoid_values={"number_1": 4, "number_2": 5},
    )

    assert generated["number_1"].int < generated["number_2"].int


def test_person_age_window_uses_twenty_percent_window() -> None:
    factual_entities = EntityCollection(persons={"person_1": PersonEntity(age=50)})

    sampler = FictionalEntitySampler(entity_pool={}, factual_entities=factual_entities)

    assert sampler._age_window("person_1", min_age=18, max_age=90) == (40, 60)


def test_temporal_year_window_uses_one_percent_range_with_cap() -> None:
    factual_entities = EntityCollection(temporals={"temporal_1": TemporalEntity(year=2000)})

    generator = NumberTemporalGenerator(factual_entities=factual_entities)

    assert generator._temporal_year_base_range("temporal_1") == (1985, 2015)


def test_temporal_day_of_month_window_is_full_random_domain() -> None:
    factual_entities = EntityCollection(temporals={"temporal_1": TemporalEntity(day_of_month=15)})

    generator = NumberTemporalGenerator(factual_entities=factual_entities)

    assert generator._temporal_day_of_month_base_range("temporal_1") == (1, 28)


def test_temporal_year_window_is_capped_at_2026_when_factual_not_future() -> None:
    factual_entities = EntityCollection(temporals={"temporal_1": TemporalEntity(year=2020)})

    generator = NumberTemporalGenerator(factual_entities=factual_entities)

    assert generator._temporal_year_base_range("temporal_1") == (2005, 2026)


def test_temporal_year_window_allows_future_when_factual_is_future() -> None:
    factual_entities = EntityCollection(temporals={"temporal_1": TemporalEntity(year=2035)})

    generator = NumberTemporalGenerator(factual_entities=factual_entities)

    assert generator._temporal_year_base_range("temporal_1") == (2020, 2050)


def test_generated_temporal_year_without_factual_never_exceeds_2026() -> None:
    generator = NumberTemporalGenerator(seed=11, factual_entities=EntityCollection())

    generated = generator.generate_temporals_with_rules(
        [("temporal_1", ["year"])],
        rules=[],
        existing_entities=None,
    )

    assert generated["temporal_1"].year is not None
    assert int(generated["temporal_1"].year) <= 2026


def test_temporal_generation_builds_non_empty_timestamp_when_requested() -> None:
    factual_entities = EntityCollection(temporals={"temporal_1": TemporalEntity(timestamp="14:46:24 JST")})

    generator = NumberTemporalGenerator(seed=11, factual_entities=factual_entities)
    generated = generator.generate_temporals_with_rules(
        [("temporal_1", ["timestamp"])],
        rules=[],
        existing_entities=None,
    )

    assert generated["temporal_1"].timestamp
    assert generated["temporal_1"].timestamp != "14:46:24 JST"
    assert generated["temporal_1"].timestamp.endswith("JST")


def test_temporal_generation_avoids_previous_timestamp_for_same_ref() -> None:
    factual_entities = EntityCollection(temporals={"temporal_1": TemporalEntity(timestamp="14:46:24 JST")})
    first = (
        NumberTemporalGenerator(seed=11, factual_entities=factual_entities)
        .generate_temporals_with_rules(
            [("temporal_1", ["timestamp"])],
            rules=[],
            existing_entities=None,
        )["temporal_1"]
        .timestamp
    )

    second = (
        NumberTemporalGenerator(
            seed=11,
            factual_entities=factual_entities,
            exclude_temporals={"timestamps_by_id": {"temporal_1": {first}}},
        )
        .generate_temporals_with_rules(
            [("temporal_1", ["timestamp"])],
            rules=[],
            existing_entities=None,
        )["temporal_1"]
        .timestamp
    )

    assert first
    assert second
    assert second != first


def test_temporal_generation_derives_requested_weekday_from_generated_date() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(
                day="Thursday",
                date="27 February 2020",
                year=2020,
                month="February",
                day_of_month=27,
            )
        }
    )
    generator = NumberTemporalGenerator(seed=11, factual_entities=factual_entities)

    generated = generator.generate_temporals_with_rules(
        [("temporal_1", ["day", "date"])],
        rules=[],
        existing_entities=None,
    )

    generated_date = generator._temporal_entity_to_date(generated["temporal_1"])
    assert generated_date is not None
    assert generated["temporal_1"].day == generated_date.strftime("%A")


def test_shifted_temporal_years_may_reuse_other_factual_years() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2001),
        }
    )
    generator = NumberTemporalGenerator(
        seed=5,
        factual_entities=factual_entities,
        exclude_temporals={"years": {2000, 2001}},
    )

    generator._temporal_year_base_range = lambda temporal_id: (
        (2001, 2001) if temporal_id == "temporal_1" else (2002, 2002)
    )

    shifted = generator._generate_shifted_required_years(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        existing_entities=None,
        excluded_years={2000, 2001},
        decade_year_temporal_ids=set(),
    )

    assert shifted == {"temporal_1": 2001, "temporal_2": 2002}


def test_temporal_year_solver_ignores_surface_only_rules_without_preserving_factual_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2005),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    monkeypatch.setattr(
        generator,
        "_temporal_year_base_range",
        lambda temporal_id: (1990, 1990) if temporal_id == "temporal_1" else (1997, 1997),
    )

    solved = generator._solve_temporal_years(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        [
            "temporal_4.timestamp - temporal_3.timestamp == number_1.int minutes",
            "temporal_6.day_of_month - temporal_5.day_of_month == 3",
        ],
        existing_entities=None,
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert solved == {"temporal_1": 1990, "temporal_2": 1997}
    assert solved["temporal_2"] - solved["temporal_1"] != 5


def test_temporal_year_solver_supports_date_year_refs_alongside_month_rules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="23 July 2020", year=2020),
            "temporal_3": TemporalEntity(date="14 May 2021", year=2021),
            "temporal_5": TemporalEntity(year=2021),
        }
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    generated_years = {
        "temporal_1": 2000,
        "temporal_3": 2003,
        "temporal_5": 2003,
    }
    monkeypatch.setattr(
        generator,
        "_temporal_year_base_range",
        lambda temporal_id: (generated_years[temporal_id], generated_years[temporal_id]),
    )

    solved = generator._solve_temporal_years(
        [
            ("temporal_1", ["date"]),
            ("temporal_3", ["date"]),
            ("temporal_5", ["year"]),
        ],
        [
            "temporal_5.year == temporal_3.date.year",
            "temporal_12.month == temporal_3.date.month",
        ],
        existing_entities=None,
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert solved == generated_years
    assert solved["temporal_3"] - solved["temporal_1"] != 1


def test_temporal_order_preservation_is_still_enforced_when_explicit_temporal_rules_exist() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2004),
            "temporal_2": TemporalEntity(year=2008),
        }
    )
    sampled_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2010),
            "temporal_2": TemporalEntity(year=2010),
        }
    )

    assert not verify_ordering_preserved(sampled_entities, factual_entities)
    assert not verify_ordering_preserved(
        sampled_entities,
        factual_entities,
        preserve_temporal_ordering=False,
    )


def test_number_order_preservation_is_enforced_for_comparable_values() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=12),
            "number_2": NumberEntity(int=18),
            "number_3": NumberEntity(int=24),
        }
    )
    sampled_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=20),
            "number_2": NumberEntity(int=18),
            "number_3": NumberEntity(int=19),
        }
    )

    assert not verify_ordering_preserved(sampled_entities, factual_entities)


def test_entity_sampler_builds_number_ordering_rules_from_factual_order() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=4),
            "number_2": NumberEntity(int=7),
            "number_3": NumberEntity(float=1.5),
            "number_4": NumberEntity(float=3.5),
            "number_5": NumberEntity(int=7),
        }
    )
    sampler = FictionalEntitySampler(entity_pool={}, factual_entities=factual_entities)

    ordering_rules = sampler._build_number_ordering_rules(
        [
            ("number_1", ["int"]),
            ("number_2", ["int"]),
            ("number_3", ["float"]),
            ("number_4", ["float"]),
            ("number_5", ["int"]),
        ]
    )

    assert "number_1.int < number_2.int" in ordering_rules
    assert "number_3.float < number_4.float" in ordering_rules
    assert all("number_2.int" not in rule or "number_5.int" not in rule for rule in ordering_rules)


def test_temporal_solver_accepts_number_refs_as_constants() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=1980),
            "temporal_2": TemporalEntity(year=2021),
        }
    )
    existing_entities = EntityCollection(numbers={"number_1": NumberEntity(int=41)})
    generator = NumberTemporalGenerator(
        seed=7,
        factual_entities=factual_entities,
        exclude_temporals={"years": {1980, 2021}},
    )

    solved = generator._solve_temporal_years(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        ["temporal_2.year - temporal_1.year == number_1.int"],
        existing_entities,
        {1980, 2021},
        set(),
    )

    assert solved is not None
    assert solved["temporal_2"] - solved["temporal_1"] == 41


def test_temporal_solver_accepts_word_number_refs_as_constants() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=1980),
            "temporal_2": TemporalEntity(year=2021),
        }
    )
    existing_entities = EntityCollection(numbers={"number_1": NumberEntity(int=41, str="forty-one")})
    generator = NumberTemporalGenerator(
        seed=7,
        factual_entities=factual_entities,
        exclude_temporals={"years": {1980, 2021}},
    )

    solved = generator._solve_temporal_years(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        ["temporal_2.year - temporal_1.year == number_1.str"],
        existing_entities,
        {1980, 2021},
        set(),
    )

    assert solved is not None
    assert solved["temporal_2"] - solved["temporal_1"] == 41


def test_temporal_solver_supports_not_equal_expression_constraints() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2003),
        }
    )
    generator = NumberTemporalGenerator(
        seed=7,
        factual_entities=factual_entities,
        exclude_temporals={"years": {2000, 2003}},
    )

    solved = generator._solve_temporal_years(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        ["temporal_2.year - temporal_1.year != 3"],
        existing_entities=None,
        excluded_years={2000, 2003},
        decade_year_temporal_ids=set(),
    )

    assert solved is not None
    assert solved["temporal_2"] - solved["temporal_1"] != 3


def test_temporal_year_domain_excludes_years_used_by_sibling_variants() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(temporals={"temporal_1": TemporalEntity(year=2000)}),
        exclude_temporals={"years_by_id": {"temporal_1": {2005, 2006}}},
    )

    domain = generator._temporal_year_domain(
        "temporal_1",
        2004,
        2007,
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert domain == [2004, 2007]


def test_temporal_solver_expands_local_windows_without_reusing_sibling_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2001),
        }
    )
    sibling_years = {
        "temporal_1": {1997},
        "temporal_2": {1997},
    }
    generator = NumberTemporalGenerator(
        seed=23,
        factual_entities=factual_entities,
        exclude_temporals={"years_by_id": sibling_years},
    )
    base_ranges = {
        "temporal_1": (1998, 1998),
        "temporal_2": (1999, 1999),
    }
    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda temporal_id: base_ranges[temporal_id])

    milp_states: list[ExactTemporalSolveState] = []
    observed_bounds: list[dict[str, tuple[int, int]]] = []
    real_milp_solve = generator._solve_temporal_years_via_milp_with_status

    def tracking_milp_solve(constraints, domains, domain_bounds, decade_year_temporal_ids):
        observed_bounds.append(dict(domain_bounds))
        result = real_milp_solve(constraints, domains, domain_bounds, decade_year_temporal_ids)
        milp_states.append(result.state)
        return result

    monkeypatch.setattr(generator, "_solve_temporal_years_via_milp_with_status", tracking_milp_solve)

    generated = generator.generate_temporals_with_rules(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        ["temporal_2.year - temporal_1.year == 7"],
        existing_entities=None,
    )

    assert milp_states == [
        ExactTemporalSolveState.CERTIFIED_INFEASIBLE,
        ExactTemporalSolveState.CERTIFIED_INFEASIBLE,
        ExactTemporalSolveState.SOLVED,
    ]
    assert observed_bounds[0] == {"temporal_1": (1998, 1998), "temporal_2": (1999, 1999)}
    assert observed_bounds[2]["temporal_1"][0] == 1994
    assert observed_bounds[2]["temporal_2"][1] == 2003
    generated_years = {temporal_id: temporal.year for temporal_id, temporal in generated.items()}
    assert generated_years["temporal_2"] - generated_years["temporal_1"] == 7
    for temporal_id, sibling_values in sibling_years.items():
        assert generated_years[temporal_id] not in sibling_values
        assert generated_years[temporal_id] != factual_entities.temporals[temporal_id].year


def test_relaxed_sampler_expands_heuristic_year_windows_before_reusing_sibling_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2007),
        }
    )
    sibling_years = {
        "temporal_1": {1998},
        "temporal_2": {2005},
    }
    base_ranges = {
        "temporal_1": (1998, 1998),
        "temporal_2": (2005, 2005),
    }
    monkeypatch.setattr(
        NumberTemporalGenerator,
        "_temporal_year_base_range",
        lambda _generator, temporal_id: base_ranges[temporal_id],
    )

    generated = FictionalEntitySampler(
        {},
        seed=23,
        factual_entities=factual_entities,
        used_temporal_years_by_id=sibling_years,
        used_temporal_values_by_id={
            temporal_id: {"year": set(years)} for temporal_id, years in sibling_years.items()
        },
        allow_relaxed_intervariant_number_reuse=True,
    ).generate_numerical_entities(
        required_entities={
            "temporal": [
                ("temporal_1", ["year"]),
                ("temporal_2", ["year"]),
            ]
        },
        rules=["temporal_2.year - temporal_1.year == 7"],
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert generated is not None
    generated_years = {temporal_id: temporal.year for temporal_id, temporal in generated.temporals.items()}
    assert generated_years["temporal_2"] - generated_years["temporal_1"] == 7
    for temporal_id, sibling_values in sibling_years.items():
        assert generated_years[temporal_id] not in sibling_values
        assert generated_years[temporal_id] != factual_entities.temporals[temporal_id].year


def test_temporal_generation_applies_date_difference_rules() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="1 January 2000", year=2000, month="January", day_of_month=1),
            "temporal_2": TemporalEntity(date="15 January 2000", year=2000, month="January", day_of_month=15),
        }
    )
    existing_entities = EntityCollection(numbers={"number_1": NumberEntity(int=14)})
    generator = NumberTemporalGenerator(seed=19, factual_entities=factual_entities)

    generated = generator.generate_temporals_with_rules(
        [("temporal_1", ["date"]), ("temporal_2", ["date"])],
        rules=["temporal_2.date - temporal_1.date = number_1.int days"],
        existing_entities=existing_entities,
    )

    assert generated["temporal_1"].date is not None
    assert generated["temporal_2"].date is not None
    assert generator._temporal_entity_to_date(generated["temporal_2"]) - generator._temporal_entity_to_date(
        generated["temporal_1"]
    ) == timedelta(days=14)


def test_temporal_generation_applies_chained_literal_date_differences_without_days_suffix() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="5 November 2018", year=2018, month="November", day_of_month=5),
            "temporal_2": TemporalEntity(date="4 November 2019", year=2019, month="November", day_of_month=4),
            "temporal_3": TemporalEntity(date="5 November 2019", year=2019, month="November", day_of_month=5),
        }
    )
    generator = NumberTemporalGenerator(seed=19, factual_entities=factual_entities)

    generated = generator.generate_temporals_with_rules(
        [
            ("temporal_1", ["date"]),
            ("temporal_2", ["date"]),
            ("temporal_3", ["date"]),
        ],
        rules=[
            "temporal_2.date - temporal_1.date == 364",
            "temporal_3.date - temporal_2.date == 1",
        ],
        existing_entities=EntityCollection(),
    )

    date_1 = generator._temporal_entity_to_date(generated["temporal_1"])
    date_2 = generator._temporal_entity_to_date(generated["temporal_2"])
    date_3 = generator._temporal_entity_to_date(generated["temporal_3"])
    assert date_2 - date_1 == timedelta(days=364)
    assert date_3 - date_2 == timedelta(days=1)


def test_final_sampler_rejects_date_repair_that_breaks_factual_chronology() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(
                date="1 January 2003",
                year=2003,
                month="January",
                day_of_month=1,
            ),
            "temporal_2": TemporalEntity(
                date="1 January 2001",
                year=2001,
                month="January",
                day_of_month=1,
            ),
            "temporal_3": TemporalEntity(
                date="1 January 2004",
                year=2004,
                month="January",
                day_of_month=1,
            ),
        }
    )
    required_entities = {
        "temporal": [
            ("temporal_1", ["date"]),
            ("temporal_2", ["date"]),
            ("temporal_3", ["date"]),
        ]
    }
    rules = ["temporal_1.date - temporal_2.date == 730"]

    rejected = FictionalEntitySampler(
        {},
        seed=10,
        factual_entities=factual_entities,
    ).generate_numerical_entities(
        required_entities=required_entities,
        rules=rules,
        named_entities=EntityCollection(),
        max_attempts=1,
    )
    accepted = FictionalEntitySampler(
        {},
        seed=0,
        factual_entities=factual_entities,
    ).generate_numerical_entities(
        required_entities=required_entities,
        rules=rules,
        named_entities=EntityCollection(),
        max_attempts=1,
    )

    assert rejected is None
    assert accepted is not None
    assert verify_ordering_preserved(accepted, factual_entities, preserve_number_ordering=False)


def test_generate_ordered_years_returns_none_when_chronology_is_infeasible() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2001),
        }
    )
    generator = NumberTemporalGenerator(seed=19, factual_entities=factual_entities)
    generator._temporal_year_base_range = lambda _temporal_id: (2000, 2000)

    ordered_years = generator._generate_ordered_years(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert ordered_years is None


def test_generate_ordered_years_reserves_sparse_decade_value_for_later_entity() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2019),
            "temporal_2": TemporalEntity(year=2024),
            "temporal_3": TemporalEntity(year=2026),
            "temporal_4": TemporalEntity(year=2040),
        }
    )
    generator = NumberTemporalGenerator(seed=19, factual_entities=factual_entities)
    bounds = {
        "temporal_1": (2017, 2023),
        "temporal_2": (2018, 2024),
        "temporal_3": (2019, 2025),
        "temporal_4": (2020, 2026),
    }
    generator._temporal_year_base_range = lambda temporal_id: bounds[temporal_id]

    ordered_years = generator._generate_ordered_years(
        [(temporal_id, ["year"]) for temporal_id in factual_entities.temporals],
        excluded_years=set(),
        decade_year_temporal_ids={"temporal_4"},
    )

    assert ordered_years is not None
    assert ordered_years["temporal_4"] == 2020
    assert ordered_years["temporal_1"] < ordered_years["temporal_2"]
    assert ordered_years["temporal_2"] < ordered_years["temporal_3"]
    assert ordered_years["temporal_3"] < ordered_years["temporal_4"]


def test_generate_temporals_raises_when_order_preserving_years_are_impossible() -> None:
    factual_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2000),
            "temporal_2": TemporalEntity(year=2001),
        }
    )
    generator = NumberTemporalGenerator(seed=19, factual_entities=factual_entities)
    generator._temporal_year_base_range = lambda _temporal_id: (2000, 2000)

    with pytest.raises(ValueError, match="preserving factual chronological ordering"):
        generator.generate_temporals_with_rules(
            [("temporal_1", ["year"]), ("temporal_2", ["year"])],
            rules=[],
            existing_entities=None,
        )


def test_entity_sampler_feeds_generated_numbers_into_temporal_solving() -> None:
    sampler = FictionalEntitySampler(entity_pool={}, seed=11)

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "number": [("number_1", ["str"])],
            "temporal": [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        },
        rules=[
            "number_1.str == 3",
            "temporal_1.year + number_1.str - 1 == temporal_2.year",
        ],
        max_attempts=1,
    )

    assert sampled is not None
    assert sampled.numbers["number_1"].int == 3
    assert sampled.temporals["temporal_2"].year == sampled.temporals["temporal_1"].year + 2


def test_mixed_temporal_repair_solves_affine_year_rule_without_cartesian_search() -> None:
    generator = NumberTemporalGenerator(seed=11)
    collection = EntityCollection(numbers={"number_21": NumberEntity(int=6, str="six")})

    pairs = _affine_two_year_candidate_pairs(
        generator=generator,
        rule="temporal_20.year + number_21.str - 1 == temporal_21.year",
        collection=collection,
        left_id="temporal_20",
        right_id="temporal_21",
        left_domain=list(range(1_000, 11_000)),
        right_domain=list(range(1_000, 11_000)),
    )

    assert pairs is not None
    assert len(pairs) == 9_995
    assert pairs[0] == (1_000, 1_005)
    assert pairs[-1] == (10_994, 10_999)
    assert all(right_year == left_year + 5 for left_year, right_year in pairs)


def test_temporal_solver_preserves_factual_ordering_inside_explicit_constraint_solves() -> None:
    document = load_annotated_document("data/HUMAN_ANNOTATED_TEMPLATES/award_winners/awards_02.yaml")
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)
    required_entities = FictionalEntitySampler.extract_required_entities(document, include_questions=True)

    existing_entities = factual_entities.model_copy(deep=True)
    for temporal_id, _ in required_entities["temporal"]:
        existing_entities.temporals.pop(temporal_id, None)
    for number_id, _ in required_entities["number"]:
        existing_entities.numbers.pop(number_id, None)

    generator = NumberTemporalGenerator(
        seed=10023, factual_entities=factual_entities, implicit_rules=document.implicit_rules
    )
    # Keep this regression focused on temporal chronology.  The production
    # sampler jointly searches mixed number/temporal rules; here a feasible,
    # non-factual span is supplied directly so unrelated number-domain
    # expansion cannot make the temporal subproblem impossible.
    existing_entities.numbers["number_21"] = NumberEntity(int=7, str="seven")

    solved_years = generator._solve_temporal_years(
        required_entities["temporal"],
        [rule for rule in document.rules if "temporal_" in str(rule)],
        existing_entities,
        {temporal.year for temporal in factual_entities.temporals.values() if getattr(temporal, "year", None)},
        set(),
    )

    assert solved_years is not None
    sampled_entities = EntityCollection(
        temporals={temporal_id: TemporalEntity(year=year) for temporal_id, year in solved_years.items()},
    )
    assert verify_ordering_preserved(sampled_entities, factual_entities, preserve_number_ordering=False)


def test_number_generation_uses_required_attrs_when_factual_kind_is_missing() -> None:
    factual_entities = EntityCollection(numbers={"number_1": NumberEntity()})
    generator = NumberTemporalGenerator(seed=9, factual_entities=factual_entities)

    generated = generator.generate_numbers(
        [("number_1", ["percent"])],
        rules=["number_1.percent > 0", "number_1.percent <= 100"],
        existing_entities=None,
        avoid_values={},
    )

    assert generated["number_1"].percent is not None


def test_temporal_milp_handles_non_contiguous_domains_from_existing_year_exclusions() -> None:
    document = load_annotated_document("data/HUMAN_ANNOTATED_TEMPLATES/biographies_of_famous_personalities/bio_04.yaml")
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=False)
    required_temporals = FictionalEntitySampler.extract_required_entities(document, include_questions=False)["temporal"]
    existing_entities = factual_entities.model_copy(deep=True)
    for temporal_id, _ in required_temporals:
        existing_entities.temporals.pop(temporal_id, None)

    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    temporal_rules = [rule for rule in document.rules if "temporal_" in str(rule)]
    constraints = generator._collect_temporal_year_constraints(
        required_temporals,
        temporal_rules,
        existing_entities,
    )
    assert constraints is not None

    excluded_years = {
        temporal.year for temporal in factual_entities.temporals.values() if getattr(temporal, "year", None) is not None
    }
    decade_year_temporal_ids = extract_decade_year_temporal_ids(document)
    domains = {}
    for temporal_id, _ in required_temporals:
        low, high = generator._temporal_year_base_range(temporal_id)
        domain = generator._temporal_year_domain(
            temporal_id,
            low,
            high,
            excluded_years,
            decade_year_temporal_ids,
        )
        if not domain:
            _, expanded_max_year = generator._temporal_year_sampling_bounds(temporal_id)
            domain = generator._temporal_year_domain(
                temporal_id,
                1,
                expanded_max_year,
                excluded_years,
                decade_year_temporal_ids,
            )
        domains[temporal_id] = domain

    solved = generator._solve_temporal_years_via_milp(
        constraints,
        domains,
        {temporal_id: (min(domain), max(domain)) for temporal_id, domain in domains.items()},
        decade_year_temporal_ids,
    )

    assert solved is not None


def test_temporal_milp_accepts_release_sized_forbidden_year_set() -> None:
    generator = NumberTemporalGenerator(seed=23)
    domains = {
        "temporal_1": [0, *range(84, 100)],
        "temporal_2": [100],
    }

    solved = generator._solve_temporal_years_via_milp(
        [(({"temporal_1": 1}, 0), "<", ({"temporal_2": 1}, 0))],
        domains,
        {
            "temporal_1": (0, 99),
            "temporal_2": (100, 100),
        },
        set(),
    )

    assert solved is not None
    assert solved["temporal_1"] in domains["temporal_1"]
    assert solved["temporal_2"] == 100


def test_temporal_milp_not_equal_disjunction_never_returns_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23)
    constraints = [
        (({"temporal_1": 1, "temporal_2": -1}, 0), "!=", ({}, 0)),
    ]
    domains = {"temporal_1": [2000, 2001], "temporal_2": [2000, 2001]}
    bounds = {"temporal_1": (2000, 2001), "temporal_2": (2000, 2001)}
    objective_coefficients = iter((-1e-3, -1e-3))
    monkeypatch.setattr(
        "memoreason.factual_to_fictional_dataset.controlled_entity_replacement."
        "temporal_generation.milp_solver.random.uniform",
        lambda *_args: next(objective_coefficients),
    )

    result = generator._solve_temporal_years_via_milp_with_status(
        constraints,
        domains,
        bounds,
        decade_year_temporal_ids=set(),
    )

    assert result.state is ExactTemporalSolveState.SOLVED
    assert result.assignments is not None
    assert result.assignments["temporal_1"] != result.assignments["temporal_2"]


def test_temporal_milp_reports_fixed_not_equal_collision_as_infeasible() -> None:
    generator = NumberTemporalGenerator(seed=23)
    constraints = [
        (({"temporal_1": 1, "temporal_2": -1}, 0), "!=", ({}, 0)),
    ]

    result = generator._solve_temporal_years_via_milp_with_status(
        constraints,
        {"temporal_1": [2000], "temporal_2": [2000]},
        {"temporal_1": (2000, 2000), "temporal_2": (2000, 2000)},
        decade_year_temporal_ids=set(),
    )

    assert result.state is ExactTemporalSolveState.CERTIFIED_INFEASIBLE
    assert result.assignments is None


def test_certified_infeasible_temporal_milp_skips_csp_and_random_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23)
    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: (1990, 1990))

    milp_states: list[ExactTemporalSolveState] = []
    real_milp_solve = generator._solve_temporal_years_via_milp_with_status

    def tracking_milp_solve(*args, **kwargs):
        result = real_milp_solve(*args, **kwargs)
        milp_states.append(result.state)
        return result

    monkeypatch.setattr(generator, "_solve_temporal_years_via_milp_with_status", tracking_milp_solve)

    feasibility_calls = 0
    real_constraints_feasible = generator._constraints_feasible

    def tracking_constraints_feasible(*args, **kwargs):
        nonlocal feasibility_calls
        feasibility_calls += 1
        return real_constraints_feasible(*args, **kwargs)

    monkeypatch.setattr(generator, "_constraints_feasible", tracking_constraints_feasible)

    def unexpected_random_fallback(*_args, **_kwargs):
        raise AssertionError("A certified-infeasible exact system must not be sampled again.")

    monkeypatch.setattr(generator, "_sample_temporal_years_randomly", unexpected_random_fallback)
    shift_calls = 0

    def invalid_shift(*_args, **_kwargs):
        nonlocal shift_calls
        shift_calls += 1
        return {"temporal_1": 1990, "temporal_2": 1990}

    monkeypatch.setattr(generator, "_generate_shifted_required_years", invalid_shift)
    monkeypatch.setattr(generator, "_generate_ordered_years", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="Unable to generate temporals"):
        generator.generate_temporals_with_rules(
            [("temporal_1", ["year"]), ("temporal_2", ["year"])],
            ["temporal_1.year < temporal_2.year"],
            existing_entities=None,
        )

    assert milp_states == [ExactTemporalSolveState.CERTIFIED_INFEASIBLE]
    assert shift_calls == 1
    # The only feasibility check validates and rejects the cheap shift.  The
    # exact CSP would perform additional checks if status=2 were discarded.
    assert feasibility_calls == 1


def test_unavailable_temporal_milp_preserves_exact_csp_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23)
    fixed_years = {"temporal_1": 1990, "temporal_2": 1991}
    monkeypatch.setattr(
        generator,
        "_temporal_year_base_range",
        lambda temporal_id: (fixed_years[temporal_id], fixed_years[temporal_id]),
    )
    monkeypatch.setattr(
        generator,
        "_solve_temporal_years_via_milp_with_status",
        lambda *_args, **_kwargs: ExactTemporalSolveResult.unavailable_or_unsupported(),
    )

    feasibility_calls = 0
    real_constraints_feasible = generator._constraints_feasible

    def tracking_constraints_feasible(*args, **kwargs):
        nonlocal feasibility_calls
        feasibility_calls += 1
        return real_constraints_feasible(*args, **kwargs)

    monkeypatch.setattr(generator, "_constraints_feasible", tracking_constraints_feasible)

    result = generator._solve_temporal_years_with_status(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        ["temporal_1.year < temporal_2.year"],
        existing_entities=None,
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert result.state is ExactTemporalSolveState.SOLVED
    assert result.assignments == fixed_years
    assert feasibility_calls > 0


def test_unavailable_exact_temporal_solver_preserves_random_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = NumberTemporalGenerator(seed=23)
    monkeypatch.setattr(
        generator,
        "_solve_temporal_years_with_status",
        lambda *_args, **_kwargs: ExactTemporalSolveResult.unavailable_or_unsupported(),
    )
    random_calls = 0

    def random_fallback(*_args, **_kwargs):
        nonlocal random_calls
        random_calls += 1
        return {"temporal_1": 1990, "temporal_2": 1991}

    monkeypatch.setattr(generator, "_sample_temporal_years_randomly", random_fallback)

    def unexpected_shift(*_args, **_kwargs):
        raise AssertionError("A valid randomized fallback must be accepted before shifting.")

    monkeypatch.setattr(generator, "_generate_shifted_required_years", unexpected_shift)

    generated = generator.generate_temporals_with_rules(
        [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        ["temporal_1.year < temporal_2.year"],
        existing_entities=None,
    )

    assert random_calls == 1
    assert generated["temporal_1"].year == 1990
    assert generated["temporal_2"].year == 1991


@pytest.mark.parametrize(
    ("rule", "bounds", "expected"),
    [
        ("temporal_1.year / 2 > 1000", (2000, 2001), 2001),
        ("(temporal_1.year - 100) / 1000000 != 0", (100, 101), 101),
    ],
)
def test_fractional_temporal_milp_status_two_preserves_exact_csp_fallback(
    monkeypatch: pytest.MonkeyPatch,
    rule: str,
    bounds: tuple[int, int],
    expected: int,
) -> None:
    generator = NumberTemporalGenerator(seed=23)
    monkeypatch.setattr(generator, "_temporal_year_base_range", lambda _temporal_id: bounds)
    required_temporals = [("temporal_1", ["year"])]
    constraints = generator._collect_temporal_year_constraints(required_temporals, [rule], None)
    assert constraints is not None
    domain = list(range(bounds[0], bounds[1] + 1))

    milp_result = generator._solve_temporal_years_via_milp_with_status(
        constraints,
        {"temporal_1": domain},
        {"temporal_1": bounds},
        decade_year_temporal_ids=set(),
    )
    solved = generator._solve_temporal_years_with_status(
        required_temporals,
        [rule],
        existing_entities=None,
        excluded_years=set(),
        decade_year_temporal_ids=set(),
    )

    assert milp_result.state is ExactTemporalSolveState.UNAVAILABLE_OR_UNSUPPORTED
    assert solved.state is ExactTemporalSolveState.SOLVED
    assert solved.assignments == {"temporal_1": expected}


def test_temporal_constraints_can_resolve_existing_person_age_refs() -> None:
    factual_entities = EntityCollection(
        persons={"person_1": PersonEntity(age=45)},
        temporals={
            "temporal_1": TemporalEntity(year=1977),
            "temporal_23": TemporalEntity(year=2023),
        },
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    required_temporals = [("temporal_1", ["year"]), ("temporal_23", ["year"])]
    existing_entities = EntityCollection(
        persons={"person_1": PersonEntity(age=45)},
        temporals={},
    )

    constraints = generator._collect_temporal_year_constraints(
        required_temporals,
        ["temporal_23.year - temporal_1.year == person_1.age + 1"],
        existing_entities,
    )

    assert constraints is not None
    assert generator._temporal_assignments_satisfy_constraints(
        {"temporal_1": 1977, "temporal_23": 2023},
        required_temporals,
        ["temporal_23.year - temporal_1.year == person_1.age + 1"],
        existing_entities,
    )


def test_temporal_constraints_can_resolve_constant_multipliers_with_existing_number_refs() -> None:
    factual_entities = EntityCollection(
        numbers={"number_1": NumberEntity(int=4, str="four")},
        temporals={
            "temporal_1": TemporalEntity(year=1916),
            "temporal_2": TemporalEntity(year=1940),
            "temporal_3": TemporalEntity(year=1944),
        },
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    required_temporals = [
        ("temporal_1", ["year"]),
        ("temporal_2", ["year"]),
        ("temporal_3", ["year"]),
    ]
    existing_entities = EntityCollection(
        numbers={"number_1": NumberEntity(int=3, str="three")},
        temporals={},
    )
    rules = [
        "temporal_2.year - temporal_1.year == 6 * number_1.int",
        "temporal_3.year - temporal_2.year == number_1.int",
    ]

    constraints = generator._collect_temporal_year_constraints(
        required_temporals,
        rules,
        existing_entities,
    )

    assert constraints is not None
    assert generator._temporal_assignments_satisfy_constraints(
        {"temporal_1": 1921, "temporal_2": 1939, "temporal_3": 1942},
        required_temporals,
        rules,
        existing_entities,
    )


def test_generate_temporals_with_rules_solves_explicit_person_age_year_constraints() -> None:
    factual_entities = EntityCollection(
        persons={"person_1": PersonEntity(age=45)},
        temporals={
            "temporal_1": TemporalEntity(year=1977),
            "temporal_23": TemporalEntity(year=2023),
        },
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    existing_entities = EntityCollection(
        persons={"person_1": PersonEntity(age=45)},
        temporals={},
    )
    required_temporals = [("temporal_1", ["year"]), ("temporal_23", ["year"])]

    generated = generator.generate_temporals_with_rules(
        required_temporals,
        ["temporal_23.year - temporal_1.year == person_1.age + 1"],
        existing_entities,
    )

    assert generated["temporal_23"].year - generated["temporal_1"].year == 46


def test_temporal_year_assignment_validation_ignores_month_only_required_temporals() -> None:
    factual_entities = EntityCollection(
        persons={"person_1": PersonEntity(age=39)},
        temporals={
            "temporal_1": TemporalEntity(date="21 December 1977", year=1977),
            "temporal_9": TemporalEntity(year=2017),
            "temporal_22": TemporalEntity(month="May"),
        },
    )
    generator = NumberTemporalGenerator(seed=23, factual_entities=factual_entities)
    existing_entities = EntityCollection(
        persons={"person_1": PersonEntity(age=39)},
        temporals={},
    )
    required_temporals = [
        ("temporal_1", ["date"]),
        ("temporal_9", ["year"]),
        ("temporal_22", ["month"]),
    ]

    generated = generator.generate_temporals_with_rules(
        required_temporals,
        ["temporal_9.year - temporal_1.year == person_1.age + 1"],
        existing_entities,
    )

    assert generated["temporal_9"].year - generated["temporal_1"].year == 40
    assert generated["temporal_22"].month is not None


def test_generate_temporals_avoids_previous_month_for_same_temporal_when_possible() -> None:
    generator = NumberTemporalGenerator(
        seed=23,
        exclude_temporals={
            "months_by_id": {
                "temporal_22": {
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                }
            }
        },
    )

    generated = generator.generate_temporals_with_rules(
        [("temporal_22", ["month"])],
        rules=[],
        existing_entities=EntityCollection(),
    )

    assert generated["temporal_22"].month == "December"


def test_timestamp_generation_replaces_factual_timezone_suffix() -> None:
    generator = NumberTemporalGenerator(seed=23)

    generated = generator._fictionalize_timestamp_surface("11:56 Nepal Standard Time")

    assert "Nepal" not in generated
    assert generated != "11:56 Nepal Standard Time"
