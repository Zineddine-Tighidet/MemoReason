from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)
from memoreason.benchmark_definition.implicit_numeric_rules import (
    ensure_document_implicit_rules,
    format_implicit_rule_explanation,
    generate_implicit_rules_for_document,
    normalize_implicit_rules_for_storage,
)
from memoreason.benchmark_definition.document_schema import (
    EntityCollection,
    ImplicitRule,
    NumberEntity,
    PersonEntity,
    TemporalEntity,
)
from memoreason.benchmark_definition.annotation_runtime import normalize_document_taxonomy


def test_normalize_document_taxonomy_generates_structured_implicit_rules() -> None:
    normalized = normalize_document_taxonomy(
        {
            "document_id": "sample_01",
            "document_theme": "sample",
            "original_document": "",
            "document_to_annotate": (
                "[Alice Roe; person_1.full_name] was [50; person_1.age] years old in [2000; temporal_1.year]. "
                "The conflict started in the [19th; number_1.int] century and ended on [14 March 1802; temporal_2.date]."
            ),
            "rules": [],
            "questions": [],
        }
    )

    implicit_by_ref = {str(rule["entity_ref"]): rule for rule in normalized.get("implicit_rules", [])}

    assert implicit_by_ref["person_1.age"]["lower_bound"] == 40
    assert implicit_by_ref["person_1.age"]["upper_bound"] == 60
    assert isinstance(implicit_by_ref["person_1.age"]["lower_bound"], int)
    assert implicit_by_ref["temporal_1.year"]["lower_bound"] == 1980
    assert implicit_by_ref["temporal_1.year"]["upper_bound"] == 2020
    assert isinstance(implicit_by_ref["temporal_1.year"]["upper_bound"], int)
    assert implicit_by_ref["number_1.int"]["lower_bound"] == 18
    assert implicit_by_ref["number_1.int"]["upper_bound"] == 20
    assert implicit_by_ref["number_1.int"]["rule_kind"] == "century_range"
    assert implicit_by_ref["temporal_2.year"]["lower_bound"] == 1784
    assert implicit_by_ref["temporal_2.year"]["upper_bound"] == 1820
    assert "temporal_2.day_of_month" not in implicit_by_ref


def test_partial_date_does_not_generate_a_year_rule_from_day_of_month() -> None:
    rules = generate_implicit_rules_for_document(
        {
            "document_to_annotate": (
                "The first fire ended on [27 February 2020; temporal_1.date], "
                "and the second ended by [2 March; temporal_2.date]."
            )
        }
    )

    implicit_by_ref = {str(rule["entity_ref"]): rule for rule in rules}
    assert implicit_by_ref["temporal_1.year"]["factual_value"] == 2020
    assert "temporal_2.year" not in implicit_by_ref


def test_numeric_magnitude_words_are_applied_to_implicit_ranges() -> None:
    rules = generate_implicit_rules_for_document(
        {
            "document_to_annotate": (
                "The fires burned [24 million; number_1.int] hectares, displaced "
                "[three billion; number_2.str] animals, and caused "
                "[US$2.8 billion; number_3.float] in damage."
            )
        }
    )

    implicit_by_ref = {str(rule["entity_ref"]): rule for rule in rules}
    assert implicit_by_ref["number_1.int"]["factual_value"] == 24_000_000
    assert implicit_by_ref["number_2.str"]["factual_value"] == 3_000_000_000
    assert implicit_by_ref["number_3.float"]["factual_value"] == 2_800_000_000
    assert implicit_by_ref["number_1.int"]["lower_bound"] == 19_200_000
    assert implicit_by_ref["number_1.int"]["upper_bound"] == 28_800_000


def test_century_detection_handles_multi_annotation_century_span() -> None:
    normalized = normalize_document_taxonomy(
        {
            "document_id": "sample_02",
            "document_theme": "sample",
            "original_document": "",
            "document_to_annotate": "From the mid-[14th; number_1.int] to the mid-[15th; number_2.int] centuries, the kingdom expanded.",
            "rules": [],
            "questions": [],
        }
    )

    implicit_by_ref = {str(rule["entity_ref"]): rule for rule in normalized.get("implicit_rules", [])}

    assert implicit_by_ref["number_1.int"]["rule_kind"] == "century_range"
    assert implicit_by_ref["number_1.int"]["percentage"] == 10.0
    assert implicit_by_ref["number_2.int"]["rule_kind"] == "century_range"


def test_normalize_implicit_rules_coerces_integer_like_bounds() -> None:
    normalized = normalize_implicit_rules_for_storage(
        [
            {
                "entity_ref": "temporal_1.year",
                "lower_bound": 1979.01,
                "upper_bound": 2018.99,
                "factual_value": 1999.0,
                "percentage": 1.0,
                "rule_kind": "temporal_year_range",
            },
            {
                "entity_ref": "number_1.float",
                "lower_bound": 10.111,
                "upper_bound": 11.999,
                "factual_value": 11.0,
                "percentage": 20.0,
                "rule_kind": "number_range",
            },
        ]
    )

    by_ref = {entry["entity_ref"]: entry for entry in normalized or []}
    assert by_ref["temporal_1.year"]["lower_bound"] == 1980
    assert by_ref["temporal_1.year"]["upper_bound"] == 2018
    assert isinstance(by_ref["temporal_1.year"]["lower_bound"], int)
    assert by_ref["number_1.float"]["lower_bound"] == 10.11
    assert by_ref["number_1.float"]["upper_bound"] == 12.0


def test_normalize_implicit_rules_caps_year_upper_bound_at_2026() -> None:
    normalized = normalize_implicit_rules_for_storage(
        [
            {
                "entity_ref": "temporal_1.year",
                "lower_bound": 2000,
                "upper_bound": 2045,
                "factual_value": 2025,
                "percentage": 1.0,
                "rule_kind": "temporal_year_range",
            }
        ]
    )

    assert normalized is not None
    assert normalized[0]["lower_bound"] == 2000
    assert normalized[0]["upper_bound"] == 2026


def test_normalize_implicit_rules_allows_future_window_when_factual_year_is_future() -> None:
    normalized = normalize_implicit_rules_for_storage(
        [
            {
                "entity_ref": "temporal_1.year",
                "lower_bound": 2020,
                "upper_bound": 2061,
                "factual_value": 2040,
                "percentage": 1.0,
                "rule_kind": "temporal_year_range",
            }
        ]
    )

    assert normalized is not None
    assert normalized[0]["lower_bound"] == 2020
    assert normalized[0]["upper_bound"] == 2061


def test_format_implicit_rule_explanation_mentions_year_cap() -> None:
    explanation = format_implicit_rule_explanation(
        {
            "entity_ref": "temporal_1.year",
            "lower_bound": 1980,
            "upper_bound": 2020,
            "factual_value": 2000,
            "percentage": 1.0,
            "rule_kind": "temporal_year_range",
        }
    )

    assert explanation == (
        "This rule was generated following the range of 1% around factual value with an upper bound at 2026."
    )


def test_format_implicit_rule_explanation_omits_cap_for_future_factual_year() -> None:
    explanation = format_implicit_rule_explanation(
        {
            "entity_ref": "temporal_1.year",
            "lower_bound": 2020,
            "upper_bound": 2061,
            "factual_value": 2040,
            "percentage": 1.0,
            "rule_kind": "temporal_year_range",
        }
    )

    assert explanation == "This rule was generated following the range of 1% around factual value."


def test_generate_implicit_rules_for_small_integers_use_fixed_plus_minus_ten_window() -> None:
    implicit_rules = generate_implicit_rules_for_document(
        {
            "document_id": "sample_03",
            "document_theme": "sample",
            "original_document": "",
            "document_to_annotate": "There were [2; number_3.int] delegates in the room.",
            "rules": [],
            "questions": [],
        }
    )

    by_ref = {entry["entity_ref"]: entry for entry in implicit_rules}
    assert by_ref["number_3.int"]["lower_bound"] == 1
    assert by_ref["number_3.int"]["upper_bound"] == 12
    assert by_ref["number_3.int"]["percentage"] == 20.0


def test_ensure_document_implicit_rules_refreshes_legacy_small_number_defaults() -> None:
    normalized = ensure_document_implicit_rules(
        {
            "document_id": "sample_04",
            "document_theme": "sample",
            "original_document": "",
            "document_to_annotate": "There were [2; number_3.int] delegates in the room.",
            "rules": [],
            "questions": [],
            "implicit_rules": [
                {
                    "entity_ref": "number_3.int",
                    "lower_bound": 2,
                    "upper_bound": 2,
                    "factual_value": 2,
                    "percentage": 20.0,
                    "rule_kind": "number_range",
                }
            ],
        }
    )

    by_ref = {entry["entity_ref"]: entry for entry in normalized.get("implicit_rules", [])}
    assert by_ref["number_3.int"]["lower_bound"] == 1
    assert by_ref["number_3.int"]["upper_bound"] == 12


def test_ensure_document_implicit_rules_refreshes_previous_small_number_window_defaults() -> None:
    normalized = ensure_document_implicit_rules(
        {
            "document_id": "sample_05",
            "document_theme": "sample",
            "original_document": "",
            "document_to_annotate": "There were [2; number_3.int] delegates in the room.",
            "rules": [],
            "questions": [],
            "implicit_rules": [
                {
                    "entity_ref": "number_3.int",
                    "lower_bound": 1,
                    "upper_bound": 5,
                    "factual_value": 2,
                    "percentage": 20.0,
                    "rule_kind": "number_range",
                }
            ],
        }
    )

    by_ref = {entry["entity_ref"]: entry for entry in normalized.get("implicit_rules", [])}
    assert by_ref["number_3.int"]["lower_bound"] == 1
    assert by_ref["number_3.int"]["upper_bound"] == 12


def test_format_implicit_rule_explanation_mentions_fixed_small_number_window() -> None:
    explanation = format_implicit_rule_explanation(
        {
            "entity_ref": "number_3.int",
            "lower_bound": 1,
            "upper_bound": 12,
            "factual_value": 2,
            "percentage": 20.0,
            "rule_kind": "number_range",
        }
    )

    assert explanation == (
        "This rule was generated following the interval of factual value +/- 10, "
        "clamped to the valid domain when needed."
    )


def test_number_temporal_generator_prefers_implicit_rule_bounds() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(
            numbers={"number_1": NumberEntity(int=100)},
            temporals={"temporal_1": TemporalEntity(year=2000)},
        ),
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.int",
                lower_bound=95.0,
                upper_bound=96.0,
                factual_value=100.0,
                percentage=20.0,
                rule_kind="number_range",
            ),
            ImplicitRule(
                entity_ref="temporal_1.year",
                lower_bound=1990.0,
                upper_bound=1992.0,
                factual_value=2000.0,
                percentage=1.0,
                rule_kind="temporal_year_range",
            ),
        ],
    )

    assert generator._number_base_range("number_1") == (95, 96)
    assert generator._temporal_year_base_range("temporal_1") == (1990, 1992)


def test_entity_sampler_uses_implicit_age_bounds() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "persons": [
                {
                    "full_name": "Fictional Person",
                    "first_name": "Fictional",
                    "last_name": "Person",
                    "age": 90,
                }
            ],
            "places": [],
            "events": [],
            "organizations": [],
        },
        seed=17,
        factual_entities=EntityCollection(persons={"person_1": PersonEntity(full_name="Original Person", age=50)}),
        implicit_rules=[
            ImplicitRule(
                entity_ref="person_1.age",
                lower_bound=48.0,
                upper_bound=49.0,
                factual_value=50.0,
                percentage=20.0,
                rule_kind="age_range",
            )
        ],
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={"person": [("person_1", ["full_name", "age"])]},
        rules=[],
        max_attempts=3,
    )

    assert sampled is not None
    assert sampled.persons["person_1"].age in {48, 49}


def test_entity_sampler_applies_explicit_age_rules_during_named_sampling() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "persons": [
                {
                    "full_name": "Fictional Person",
                    "first_name": "Fictional",
                    "last_name": "Person",
                    "age": 90,
                }
            ],
            "places": [],
            "events": [],
            "organizations": [],
        },
        seed=17,
        factual_entities=EntityCollection(persons={"person_2": PersonEntity(full_name="Original Teen", age=16)}),
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={"person": [("person_2", ["age"])]},
        rules=["person_2.age < 18", "person_2.age > 12"],
        max_attempts=3,
    )

    assert sampled is not None
    assert sampled.persons["person_2"].age in {13, 14, 15, 17}
