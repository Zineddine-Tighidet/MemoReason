from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity, TemporalEntity
from memoreason.benchmark_definition.annotation_runtime import RuleEngine


def test_rule_engine_evaluates_century_functions() -> None:
    entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=19),
            "number_2": NumberEntity(int=18),
        },
        temporals={
            "temporal_1": TemporalEntity(year=1875),
            "temporal_2": TemporalEntity(year=1802, date="14 March 1802"),
        },
    )

    assert RuleEngine.evaluate_expression("century_of(temporal_1.year)", entities) == 19
    assert RuleEngine.evaluate_expression("century_of(temporal_2.date)", entities) == 19
    assert RuleEngine.evaluate_expression("century_start(number_1.int)", entities) == 1801
    assert RuleEngine.evaluate_expression("century_end(number_2.int)", entities) == 1800

    validation = dict(
        RuleEngine.validate_all_rules(
            [
                "century_of(temporal_1.year) == number_1.int",
                "century_end(number_2.int) < temporal_2.year",
                "temporal_2.year >= century_start(number_1.int)",
            ],
            entities,
        )
    )

    assert validation["century_of(temporal_1.year) == number_1.int"] is True
    assert validation["century_end(number_2.int) < temporal_2.year"] is True
    assert validation["temporal_2.year >= century_start(number_1.int)"] is True
