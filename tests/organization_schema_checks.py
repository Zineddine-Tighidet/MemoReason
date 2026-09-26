from pathlib import Path

import pytest
import yaml

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_renderer import (
    FictionalDocumentRenderer,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.named_entity_pool_rule_policy import (
    copy_named_entity_pool_without_rule_based_mutation,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.generation_requirements import (
    extract_required_entities,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.sampling_checks import merge_factual_entities
from memoreason.benchmark_definition.document_schema import (
    AnnotatedDocument,
    AwardEntity,
    EntityCollection,
    LegalEntity,
    ProductEntity,
    Question,
)
from memoreason.benchmark_definition.annotation_runtime import (
    AnnotationParser,
    AnnotationValidationError,
    RuleEngine,
    load_annotated_document,
    load_entity_pool,
    normalize_document_taxonomy,
)


def test_extract_factual_entities_keeps_organization_subtype() -> None:
    annotated_doc = AnnotatedDocument(
        document_id="doc_1",
        document_theme="theme",
        original_document="",
        document_to_annotate="[European Commission; government_org_1.name] met [Reuters; media_org_1.name].",
        rules=[],
        questions=[],
    )

    entities = AnnotationParser.extract_factual_entities(annotated_doc)

    assert entities.organizations["government_org_1"].organization_kind == "government_org"
    assert entities.organizations["government_org_1"].name == "European Commission"
    assert entities.organizations["media_org_1"].organization_kind == "media_org"
    assert entities.organizations["media_org_1"].name == "Reuters"


def test_rule_engine_resolves_legacy_organization_attribute_from_canonical_kind() -> None:
    annotated_doc = AnnotatedDocument(
        document_id="doc_1",
        document_theme="theme",
        original_document="",
        document_to_annotate="[Massachusetts Institute of Technology; organization_1.is_education]",
        rules=[],
        questions=[Question(question_id="q1", question="", answer="organization_1.is_education")],
    )

    entities = AnnotationParser.extract_factual_entities(annotated_doc)

    assert entities.organizations["organization_1"].organization_kind == "educational_org"
    assert (
        RuleEngine._get_entity_value(entities, "organization_1.is_education") == "Massachusetts Institute of Technology"
    )


def test_load_entity_pool_normalizes_legacy_organization_entry(tmp_path: Path) -> None:
    pool_path = tmp_path / "pool.yaml"
    pool_path.write_text(
        yaml.safe_dump(
            {
                "persons": [],
                "places": [],
                "events": [],
                "organizations": [{"is_government_org": "Civic Ministry"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    pool = load_entity_pool(str(pool_path))

    assert pool["government_orgs"] == [
        {
            "name": "Civic Ministry",
        }
    ]


def test_required_entities_keep_organization_subtypes() -> None:
    annotated_doc = AnnotatedDocument(
        document_id="doc_1",
        document_theme="theme",
        original_document="",
        document_to_annotate="[European Commission; government_org_1.name] met [Reuters; media_org_1.name].",
        rules=[],
        questions=[
            Question(question_id="q1", question="Who met [Reuters; media_org_1.name]?", answer="government_org_1.name"),
        ],
    )

    required = extract_required_entities(annotated_doc)

    assert "organization" not in required
    assert required["government_org"] == [("government_org_1", ["name"])]
    assert required["media_org"] == [("media_org_1", ["name"])]


def test_entity_sampler_filters_organizations_by_subtype() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "government_orgs": [{"name": "Civic Ministry"}],
            "media_orgs": [{"name": "Beacon Times"}],
        },
        seed=0,
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "government_org": [("government_org_1", ["name"])],
            "media_org": [("media_org_1", ["name"])],
        },
        rules=[],
    )

    assert sampled is not None
    assert sampled.organizations["government_org_1"].organization_kind == "government_org"
    assert sampled.organizations["government_org_1"].name == "Civic Ministry"
    assert sampled.organizations["media_org_1"].organization_kind == "media_org"
    assert sampled.organizations["media_org_1"].name == "Beacon Times"


def test_entity_sampler_keeps_award_legal_and_product_entities() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "awards": [{"name": "Aurora Prize"}],
            "legals": [{"name": "Silver Charter", "reference_code": "SC-17"}],
            "products": [{"name": "LatticeOS"}],
        },
        seed=0,
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "award": [("award_1", ["name"])],
            "legal": [("legal_1", ["name", "reference_code"])],
            "product": [("product_1", ["name"])],
        },
        rules=[],
    )

    assert sampled is not None
    assert sampled.awards["award_1"].name == "Aurora Prize"
    assert sampled.legals["legal_1"].name == "Silver Charter"
    assert sampled.legals["legal_1"].reference_code == "SC-17"
    assert sampled.products["product_1"].name == "LatticeOS"


def test_merge_factual_entities_keeps_award_legal_and_product_types() -> None:
    collection = EntityCollection()
    factual_entities = EntityCollection(
        awards={"award_1": AwardEntity(name="Aurora Prize")},
        legals={"legal_1": LegalEntity(name="Silver Charter", reference_code="SC-17")},
        products={"product_1": ProductEntity(name="LatticeOS")},
    )

    merge_factual_entities(collection, factual_entities)

    assert collection.awards["award_1"].name == "Aurora Prize"
    assert collection.legals["legal_1"].reference_code == "SC-17"
    assert collection.products["product_1"].name == "LatticeOS"


def test_entity_sampler_derives_full_gender_pronoun_set() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "persons": [
                {"full_name": "Alex Vale"},
                {"full_name": "Mira Doss"},
            ],
        },
        seed=0,
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "person": [
                ("person_1", ["full_name"]),
                ("person_2", ["full_name"]),
            ],
        },
        rules=[],
    )

    assert sampled is not None
    expected = {
        "male": ("he", "him", "his", "his", "himself", "Mr"),
        "female": ("she", "her", "her", "hers", "herself", "Ms"),
    }
    for person in sampled.persons.values():
        assert person.gender in expected
        assert (
            person.subj_pronoun,
            person.obj_pronoun,
            person.poss_det_pronoun,
            person.poss_pro_pronoun,
            person.refl_pronoun,
            person.honorific,
        ) == expected[person.gender]


def test_entity_sampler_gender_sampling_ignores_factual_pronoun_signals() -> None:
    entity_pool = {
        "persons": [
            {"full_name": "Tivona Bryndis"},
            {"full_name": "Kessara Bryndis"},
        ],
    }
    required_entities = {
        "person": [
            ("person_1", ["full_name"]),
            ("person_2", ["full_name"]),
        ],
    }
    factual_entities = EntityCollection(
        persons={
            "person_1": {
                "gender": "male",
                "subj_pronoun": "She",
                "obj_pronoun": "her",
                "poss_det_pronoun": "her",
                "poss_pro_pronoun": "hers",
                "refl_pronoun": "herself",
            },
            "person_2": {
                "subj_pronoun": "she",
                "honorific": "Ms",
            },
        }
    )

    sampled_without_factual = FictionalEntitySampler(
        entity_pool=entity_pool,
        seed=0,
    ).sample_fictional_entities(
        required_entities=required_entities,
        rules=[],
    )
    sampled_with_factual = FictionalEntitySampler(
        entity_pool=entity_pool,
        seed=0,
        factual_entities=factual_entities,
    ).sample_fictional_entities(
        required_entities=required_entities,
        rules=[],
    )

    assert sampled_without_factual is not None
    assert sampled_with_factual is not None
    for person_id in ("person_1", "person_2"):
        baseline = sampled_without_factual.persons[person_id]
        factual = sampled_with_factual.persons[person_id]
        assert factual.gender == baseline.gender
        assert factual.subj_pronoun == baseline.subj_pronoun
        assert factual.honorific == baseline.honorific


def test_entity_sampler_respects_explicit_person_gender_literal_rule() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "persons": [
                {"full_name": "Alex Vale"},
                {"full_name": "Mira Doss"},
            ],
        },
        seed=0,
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={
            "person": [
                ("person_1", ["full_name"]),
                ("person_2", ["full_name"]),
            ],
        },
        rules=['person_1.gender == "male"'],
    )

    assert sampled is not None
    assert sampled.persons["person_1"].gender == "male"
    assert sampled.persons["person_1"].subj_pronoun == "he"
    assert sampled.persons["person_1"].honorific == "Mr"
    assert sampled.persons["person_2"].gender == "female"


def test_entity_sampler_rejects_conflicting_explicit_person_gender_literal_rules() -> None:
    sampler = FictionalEntitySampler(
        entity_pool={
            "persons": [
                {"full_name": "Alex Vale"},
            ],
        },
        seed=0,
    )

    sampled = sampler.sample_fictional_entities(
        required_entities={"person": [("person_1", ["full_name"])]},
        rules=['person_1.gender == "male"', 'person_1.gender == "female"'],
    )

    assert sampled is None


def test_document_generator_keeps_ambiguous_gender_category_phrases_literal() -> None:
    annotated_doc = AnnotatedDocument(
        document_id="doc_1",
        document_theme="theme",
        original_document="",
        document_to_annotate=(
            "[Casey Vale; person_1.full_name] is the only player, "
            "[male; person_1.gender] or [female; person_1.gender], to win."
        ),
        rules=[],
        questions=[],
    )
    fictional_entities = EntityCollection(
        persons={
            "person_1": {
                "full_name": "Mira Solen",
                "gender": "female",
                "subj_pronoun": "she",
                "obj_pronoun": "her",
                "poss_det_pronoun": "her",
                "poss_pro_pronoun": "hers",
                "refl_pronoun": "herself",
                "honorific": "Ms",
            }
        }
    )

    generated = FictionalDocumentRenderer.render_document(annotated_doc, fictional_entities)

    assert generated.generated_document == "Mira Solen is the only player, male or female, to win."


def test_named_entity_pool_policy_keeps_bio_01_pool_values_unchanged() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/biographies_of_famous_personalities/bio_01.yaml")
    pool_path = Path("data/GENERATED_FICTIONAL_ENTITIES/biographies_of_famous_personalities/bio_01_entity_pool.yaml")
    if not pool_path.exists():
        pytest.skip("requires a generated bio_01 entity pool fixture")
    document = load_annotated_document(str(template_path))
    pool = load_entity_pool(str(pool_path))

    aligned_pool = copy_named_entity_pool_without_rule_based_mutation(pool)

    assert aligned_pool == pool

    sampler = FictionalEntitySampler(
        aligned_pool,
        seed=23,
        factual_entities=AnnotationParser.extract_factual_entities(document),
    )
    manual_types = {
        "person",
        "place",
        "event",
        "organization",
        "military_org",
        "entreprise_org",
        "ngo",
        "government_org",
        "educational_org",
        "media_org",
        "award",
        "legal",
        "product",
    }
    required_entities = extract_required_entities(document)
    manual_required = {
        entity_type: specs for entity_type, specs in required_entities.items() if entity_type in manual_types
    }
    manual_rules = [rule for rule in document.rules if sampler._is_manual_rule(str(rule))]
    assert manual_required
    assert manual_rules == []


def test_sparse_place_keys_do_not_trigger_manual_sampling_for_disaster_05() -> None:
    template_path = Path("data/HUMAN_ANNOTATED_TEMPLATES/natural_disasters/disaster_05.yaml")
    pool_path = Path("data/GENERATED_FICTIONAL_ENTITIES/natural_disasters/disaster_05_entity_pool.yaml")
    if not pool_path.exists():
        pytest.skip("requires a generated disaster_05 entity pool fixture")
    document = load_annotated_document(str(template_path))
    pool = load_entity_pool(str(pool_path))

    aligned_pool = copy_named_entity_pool_without_rule_based_mutation(pool)
    assert aligned_pool == pool
    sampler = FictionalEntitySampler(
        aligned_pool,
        seed=23,
        factual_entities=AnnotationParser.extract_factual_entities(document),
    )
    required_entities = extract_required_entities(document)
    manual_required = {
        entity_type: specs
        for entity_type, specs in required_entities.items()
        if entity_type
        in {
            "person",
            "place",
            "event",
            "organization",
            "military_org",
            "entreprise_org",
            "ngo",
            "government_org",
            "educational_org",
            "media_org",
            "award",
            "legal",
            "product",
        }
    }
    manual_rules = [rule for rule in document.rules if sampler._is_manual_rule(str(rule))]
    assert manual_required
    assert manual_rules == []


def test_normalize_document_taxonomy_rewrites_legacy_organization_refs() -> None:
    normalized = normalize_document_taxonomy(
        {
            "document_id": "doc_legacy",
            "document_theme": "theme",
            "document_to_annotate": (
                "[Paris Bar; government_organization_1.name] admitted "
                "[Massachusetts Institute of Technology; organization_2.is_education]."
            ),
            "rules": [
                "organization_2.is_education != government_organization_1.name",
            ],
            "questions": [
                {
                    "question_id": "q1",
                    "question": "Who joined [MIT; organization_2.is_education]?",
                    "answer": "organization_2.is_education",
                }
            ],
        }
    )

    assert "[Paris Bar; government_org_1.name]" in normalized["document_to_annotate"]
    assert "[Massachusetts Institute of Technology; educational_org_2.name]" in normalized["document_to_annotate"]
    assert normalized["rules"] == ["educational_org_2.name != government_org_1.name"]
    assert normalized["questions"][0]["question"] == "Who joined [MIT; educational_org_2.name]?"
    assert normalized["questions"][0]["answer"] == "educational_org_2.name"


def test_normalize_document_taxonomy_remaps_stale_organization_refs_by_surface() -> None:
    normalized = normalize_document_taxonomy(
        {
            "document_id": "doc_surface_sync",
            "document_theme": "theme",
            "document_to_annotate": "Monitoring group [Airwars; entreprise_org_1.name] published a report.",
            "rules": ["media_org_6.name == entreprise_org_1.name"],
            "questions": [
                {
                    "question_id": "q1",
                    "question": "Was [Airwars; media_org_6.name] cited in the report?",
                    "answer": "media_org_6.name",
                }
            ],
        }
    )

    assert "[Airwars; entreprise_org_1.name]" in normalized["document_to_annotate"]
    assert normalized["rules"] == ["entreprise_org_1.name == entreprise_org_1.name"]
    assert normalized["questions"][0]["question"] == "Was [Airwars; entreprise_org_1.name] cited in the report?"
    assert normalized["questions"][0]["answer"] == "entreprise_org_1.name"


def test_load_annotated_document_rejects_legacy_organization_aliases(tmp_path: Path) -> None:
    template_path = tmp_path / "legacy_template.yaml"
    template_path.write_text(
        yaml.safe_dump(
            {
                "document": {
                    "document_id": "doc_legacy",
                    "document_theme": "theme",
                    "original_document": "",
                    "document_to_annotate": (
                        "[Paris Bar; government_organization_1.name] admitted [MIT; organization_2.is_education]."
                    ),
                    "rules": ["organization_2.is_education != government_organization_1.name"],
                    "questions": [
                        {
                            "question_id": "q1",
                            "question": "Who joined [MIT; organization_2.is_education]?",
                            "answer": "organization_2.is_education",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(AnnotationValidationError):
        load_annotated_document(str(template_path))
