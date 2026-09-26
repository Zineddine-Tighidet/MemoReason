from memoreason.benchmark_definition.century_expressions import century_end, century_of
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity, TemporalEntity
from memoreason.benchmark_definition.annotation_runtime import RuleEngine


def test_entity_sampler_enforces_century_of_rule_alongside_temporal_gap() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=19),
        },
        temporals={
            "temporal_1": TemporalEntity(year=1875),
            "temporal_2": TemporalEntity(year=1876),
        },
    )
    sampler = FictionalEntitySampler(entity_pool={}, seed=11, factual_entities=factual_entities)

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "number": [("number_1", ["int"])],
            "temporal": [("temporal_1", ["year"]), ("temporal_2", ["year"])],
        },
        rules=[
            "century_of(temporal_1.year) == number_1.int",
            "temporal_2.year - temporal_1.year == 1",
        ],
        max_attempts=5,
    )

    assert sampled is not None
    assert sampled.temporals["temporal_1"].year is not None
    assert sampled.temporals["temporal_2"].year is not None
    assert sampled.numbers["number_1"].int is not None
    assert century_of(sampled.temporals["temporal_1"].year) == sampled.numbers["number_1"].int
    assert sampled.temporals["temporal_2"].year - sampled.temporals["temporal_1"].year == 1
    assert all(
        is_valid
        for _, is_valid in RuleEngine.validate_all_rules(
            [
                "century_of(temporal_1.year) == number_1.int",
                "temporal_2.year - temporal_1.year == 1",
            ],
            sampled,
        )
    )


def test_entity_sampler_enforces_century_end_bound_rule() -> None:
    factual_entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=19),
        },
        temporals={
            "temporal_1": TemporalEntity(year=1932),
        },
    )
    sampler = FictionalEntitySampler(entity_pool={}, seed=29, factual_entities=factual_entities)

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "number": [("number_1", ["int"])],
            "temporal": [("temporal_1", ["year"])],
        },
        rules=["century_end(number_1.int) < temporal_1.year"],
        max_attempts=5,
    )

    assert sampled is not None
    assert sampled.temporals["temporal_1"].year is not None
    assert sampled.numbers["number_1"].int is not None
    assert century_end(sampled.numbers["number_1"].int) < sampled.temporals["temporal_1"].year
    assert all(
        is_valid
        for _, is_valid in RuleEngine.validate_all_rules(
            ["century_end(number_1.int) < temporal_1.year"],
            sampled,
        )
    )
