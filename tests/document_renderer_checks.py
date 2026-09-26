from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_document_renderer import (
    FictionalDocumentRenderer,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.number_temporal_generator import (
    NumberTemporalGenerator,
)
from memoreason.benchmark_definition.document_schema import (
    AnnotatedDocument,
    AwardEntity,
    EntityCollection,
    EventEntity,
    ImplicitRule,
    NumberEntity,
    OrganizationEntity,
    PersonEntity,
    PlaceEntity,
    TemporalEntity,
)


def test_document_renderer_preserves_month_only_date_surface() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="In [August; temporal_1.date] [2022; temporal_2.year], the policy changed.",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="17 October 1980", month="October", day_of_month=17, year=1980),
            "temporal_2": TemporalEntity(year=2023),
        }
    )

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "In October 2023, the policy changed."


def test_document_renderer_preserves_month_day_surface_without_year() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="They landed on [July 24; temporal_1.date].",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="1 May 2024", month="May", day_of_month=1, year=2024),
        }
    )

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "They landed on May 1."


def test_document_renderer_preserves_unannotated_literals() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate=(
            "[Serena Williams; person_1.full_name] won her title. Later, Serena Williams celebrated with the "
            "Williams sisters."
        ),
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(
        persons={"person_1": PersonEntity(full_name="Tivona Bryndis", first_name="Tivona", last_name="Bryndis")}
    )

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert "Tivona Bryndis" in generated.generated_document
    assert "Later, Serena Williams celebrated with the Williams sisters." in generated.generated_document


def test_document_renderer_preserves_unannotated_lowercase_and_plural_literals() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate=(
            "[Golden Slam; award_1.name] is rare. Some players dream of golden slam history and multiple Golden Slams."
        ),
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(awards={"award_1": AwardEntity(name="Auric Sweep")})

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == (
        "Auric Sweep is rare. Some players dream of golden slam history and multiple Golden Slams."
    )


def test_document_renderer_preserves_alias_surface_variants() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="The [Apollo program; event_1.name], also known as [Project Apollo; event_1.name], launched.",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(events={"event_1": EventEntity(name="Project Cendris")})
    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "The Cendris program, also known as Project Cendris, launched."


def test_document_renderer_expands_single_token_parenthetical_long_form() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="[Alphabet; entreprise_org_1.name] ([Google; entreprise_org_1.name]) announced earnings.",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(
        organizations={"entreprise_org_1": OrganizationEntity(name="Kessley", organization_kind="entreprise_org")}
    )

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "Kessley Group (Kessley) announced earnings."


def test_document_renderer_collapses_duplicate_parenthetical_names() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="[France; place_1.country] ([French Republic; place_1.country]) signed the treaty.",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(places={"place_1": PlaceEntity(country="Republic of Tovvexzor")})

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "Republic of Tovvexzor signed the treaty."


def test_document_renderer_preserves_age_offsets_for_repeated_person_age_mentions() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate=(
            "[Ari Vale; person_1.full_name] (born [24 June 1987; temporal_1.date]) debuted at age [17; person_1.age] "
            "in [October; temporal_2.month] [2004; temporal_2.year], and later won a title at age [22; person_1.age] "
            "in [2008; temporal_3.year]."
        ),
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(
        persons={"person_1": PersonEntity(full_name="Teral Voss", first_name="Teral", last_name="Voss", age=18)},
        temporals={
            "temporal_1": TemporalEntity(date="24 June 1968", month="June", day_of_month=24, year=1968),
            "temporal_2": TemporalEntity(month="October", year=1984),
            "temporal_3": TemporalEntity(year=1989),
        },
    )

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert "at age 16 in October 1984" in generated.generated_document
    assert "at age 21 in 1989" in generated.generated_document


def test_document_renderer_repairs_birth_age_chronology_when_shifted_years_drift() -> None:
    repaired = FictionalDocumentRenderer._repair_birth_age_chronology(
        "Teral Voss (born 24 June 1968) debuted at age 18 in October 1984."
    )

    assert repaired == "Teral Voss (born 24 June 1968) debuted at age 16 in October 1984."


def test_document_renderer_fixes_indefinite_articles_after_replacement() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="[Microsoft; entreprise_org_1.name] is an [American; place_1.demonym] company.",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(
        organizations={
            "entreprise_org_1": OrganizationEntity(name="Highridge Media", organization_kind="entreprise_org")
        },
        places={"place_1": PlaceEntity(demonym="Selorian")},
    )

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "Highridge is a Selorian company."


def test_document_renderer_fixes_definite_articles_for_plain_country_names() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="She was the president of the [France; place_1.country].",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(places={"place_1": PlaceEntity(country="Zorvia")})

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "She was the president of Zorvia."


def test_document_renderer_preserves_definite_articles_for_geographic_type_names() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="Ships crossed the [Pacific Ocean; place_1.natural_site].",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(places={"place_1": PlaceEntity(natural_site="Velkor Sea")})

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "Ships crossed the Velkor Sea."


def test_document_renderer_collapses_duplicate_surface_suffixes() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="It occurred in the [Tōhoku; place_1.region] region.",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(places={"place_1": PlaceEntity(region="Marnnoxbrim Region")})

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "It occurred in the Marnnoxbrim Region."


def test_document_renderer_collapses_duplicate_definite_articles() -> None:
    document = AnnotatedDocument(
        document_id="doc",
        document_theme="theme",
        original_document="",
        document_to_annotate="The incident became known as the [Fukushima Daiichi nuclear disaster; event_1.name].",
        questions=[],
        relations=[],
        rules=[],
    )
    fictional_entities = EntityCollection(events={"event_1": EventEntity(name="The Yornrathvek Conference")})

    generated = FictionalDocumentRenderer.render_document(document, fictional_entities)

    assert generated.generated_document == "The incident became known as the Yornrathvek Conference."


def test_document_renderer_handles_number_article_surface_changes() -> None:
    assert FictionalDocumentRenderer._fix_indefinite_articles("Residents had more than a eighty evacuation sites.") == (
        "Residents had more than eighty evacuation sites."
    )
    assert FictionalDocumentRenderer._fix_indefinite_articles("Residents stayed within a 8 km radius.") == (
        "Residents stayed within an 8 km radius."
    )


def test_document_renderer_normalizes_float_surfaces() -> None:
    assert FictionalDocumentRenderer._format_numeric_surface(28.600000000000001) == "28.6"
    assert FictionalDocumentRenderer._format_numeric_surface(56.099999999999994) == "56.1"


def test_document_renderer_capitalizes_sentence_start_tokens_after_normalization() -> None:
    assert FictionalDocumentRenderer._normalize_replaced_text("The meeting ended. he left quickly.") == (
        "The meeting ended. He left quickly."
    )


def test_fraction_numbers_preserve_fraction_surface_family() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(fraction="one sixth")})
    )

    generated = generator._build_number_entity("number_1", 5, {"fraction"})

    assert generated.fraction == "one fifth"
    assert generated.int == 5


def test_fraction_number_int_uses_denominator_for_ranges() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(fraction="1/5")})
    )

    assert generator._factual_number_int("number_1") == 5


def test_fraction_implicit_range_maps_back_to_denominator_window() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(numbers={"number_1": NumberEntity(fraction="one sixth")}),
        implicit_rules=[
            ImplicitRule(
                entity_ref="number_1.fraction",
                lower_bound=0.13,
                upper_bound=0.2,
                factual_value=0.17,
                percentage=20.0,
                rule_kind="number_range",
            )
        ],
    )

    assert generator._number_base_range("number_1") == (5, 8)


def test_float_and_percent_numbers_keep_numeric_backing_int() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(
            numbers={
                "number_1": NumberEntity(percent=5.5),
                "number_2": NumberEntity(float=17.935),
            }
        )
    )

    percent_entity = generator._build_number_entity("number_1", 6, {"percent"})
    float_entity = generator._build_number_entity("number_2", 15, {"float"})

    assert percent_entity.percent == 6.0
    assert percent_entity.int == 6
    assert float_entity.float == 15.0
    assert float_entity.int == 15


def test_non_integer_number_values_are_rounded_to_two_decimals() -> None:
    generator = NumberTemporalGenerator(
        factual_entities=EntityCollection(
            numbers={
                "number_1": NumberEntity(float=56.099999999999994),
            }
        )
    )

    float_entity = generator._build_number_entity("number_1", 56, {"float"}, allow_non_integer_adjustment=True)

    assert float_entity.float == 56.2
    assert generator._round_non_integer_surface_value(17.935) == 17.93
