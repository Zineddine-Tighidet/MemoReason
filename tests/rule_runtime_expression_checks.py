from pathlib import Path

import yaml

from memoreason.benchmark_definition.document_schema import EntityCollection, NumberEntity, PersonEntity, TemporalEntity
from memoreason.benchmark_definition.annotation_runtime import RuleEngine
from memoreason.benchmark_definition.annotation_runtime import (
    find_entity_refs,
    load_annotated_document,
    normalize_document_taxonomy,
)


def test_normalize_document_taxonomy_preserves_rule_comments() -> None:
    normalized = normalize_document_taxonomy(
        {
            "document_id": "doc_01",
            "document_theme": "theme",
            "document_to_annotate": "[French; place_1.demonym] [17; number_1.int]",
            "rules": [
                {"rule": "person_1.nationality == place_1.demonym", "comment": "keep coherent"},
                "number_1.int > 10 # lower bound",
            ],
            "questions": [],
        }
    )

    assert normalized["rules"] == [
        "person_1.nationality == place_1.demonym # keep coherent",
        "number_1.int > 10 # lower bound",
    ]


def test_load_annotated_document_uses_expression_only_rules(tmp_path: Path) -> None:
    template_path = tmp_path / "doc_with_rule_comments.yaml"
    payload = {
        "document": {
            "document_id": "doc_01",
            "document_theme": "theme",
            "original_document": "",
            "document_to_annotate": "[ten; number_1.str] [20; number_2.int] [2010; temporal_1.year] [2014; temporal_2.year]",
            "questions": [],
            "rules": [
                "number_1.int < number_2.int # keep ordering",
                {
                    "rule": "temporal_2.year >= temporal_1.year",
                    "comment": "timeline progression",
                },
            ],
        }
    }
    template_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=10000),
        encoding="utf-8",
    )

    document = load_annotated_document(str(template_path))

    assert document.rules == [
        "number_1.int < number_2.int",
        "temporal_2.year >= temporal_1.year",
    ]


def test_rule_engine_coerces_numeric_strings_and_approximate_equalities() -> None:
    entities = EntityCollection(
        numbers={
            "number_1": NumberEntity(int=23),
            "number_2": NumberEntity(str="23rd"),
            "number_3": NumberEntity(str="thousands"),
            "number_4": NumberEntity(str="once"),
            "number_5": NumberEntity(float=15.5),
            "number_6": NumberEntity(int=93),
        }
    )

    assert RuleEngine.evaluate_expression("number_1.int == number_2.str", entities) is True
    assert RuleEngine.evaluate_expression("number_3.str > 1", entities) is True
    assert RuleEngine.evaluate_expression("number_4.str == 1", entities) is True
    assert RuleEngine.evaluate_expression('number_4.str == "once"', entities) is True
    assert RuleEngine.evaluate_expression("number_5.float * number_6.int / 60 == 24", entities) is True


def test_rule_engine_supports_temporal_year_day_offsets_and_date_differences() -> None:
    entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=2030),
            "temporal_2": TemporalEntity(year=2000, date="2 November 2000"),
            "temporal_3": TemporalEntity(date="11 July 2021"),
            "temporal_4": TemporalEntity(date="29 June 2021"),
        },
        numbers={
            "number_1": NumberEntity(int=25),
            "number_2": NumberEntity(int=114),
            "number_3": NumberEntity(int=12),
        },
    )

    assert (
        RuleEngine.evaluate_expression(
            "temporal_1.year > temporal_2.year + number_1.int years + number_2 days",
            entities,
        )
        is True
    )
    assert (
        RuleEngine.evaluate_expression(
            "temporal_3.date - temporal_4.date == number_3.int days",
            entities,
        )
        is True
    )


def test_rule_engine_supports_timestamp_differences_in_minutes() -> None:
    entities = EntityCollection(
        temporals={
            "temporal_3": TemporalEntity(timestamp="5:34 p.m. CDT"),
            "temporal_4": TemporalEntity(timestamp="6:12 p.m"),
        },
        numbers={
            "number_3": NumberEntity(int=38),
        },
    )

    assert (
        RuleEngine.evaluate_expression(
            "temporal_4.timestamp - temporal_3.timestamp = number_3.int minutes",
            entities,
        )
        is True
    )


def test_rule_engine_supports_person_relationship_directional_refs() -> None:
    entities = EntityCollection(
        persons={
            "person_1": PersonEntity(relationships={"person_2": "daughter"}),
            "person_2": PersonEntity(relationships={"person_1": "father"}),
        }
    )
    assert RuleEngine.evaluate_expression('person_2.relationship.person_1 == "father"', entities) is True
    assert RuleEngine.evaluate_expression('person_1.relationship.person_2 == "daughter"', entities) is True


def test_find_entity_refs_preserves_directional_relationship_refs() -> None:
    assert find_entity_refs("person_2.relationship.person_1") == ["person_2.relationship.person_1"]
    assert find_entity_refs("[brother; person_2.relationship.person_1]") == ["person_2.relationship.person_1"]
