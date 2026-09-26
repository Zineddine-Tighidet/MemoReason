from pathlib import Path

import pytest

from memoreason.benchmark_definition.answer_expression_evaluation import AnswerEvaluator
from memoreason.benchmark_definition.document_schema import (
    EntityCollection,
    LegalEntity,
    NumberEntity,
    OrganizationEntity,
    PersonEntity,
    PlaceEntity,
    TemporalEntity,
)
from memoreason.model_evaluation.answer_normalization import canonicalize_answer
from memoreason.model_evaluation.answer_schema_data_contracts import UNANSWERABLE
from memoreason.model_evaluation.benchmark_document_loading import load_evaluation_document
from memoreason.model_evaluation.ground_truth_answer_specification import (
    build_answer_spec,
)
from memoreason.model_evaluation.schema_aware_answer_matching import (
    parse_schema_answer,
    score_canonical_prediction,
    score_prediction_with_schema,
)


def test_build_answer_spec_expands_generic_person_identity_aliases() -> None:
    entities = EntityCollection(
        persons={
            "person_1": PersonEntity(
                full_name="Alice Marie Quinn",
                first_name="Alice",
                last_name="Quinn",
            ),
            "person_2": PersonEntity(
                full_name="Bob Stone",
                first_name="Bob",
                last_name="Stone",
            ),
        }
    )

    spec = build_answer_spec(
        question_text="Who joined the council?",
        answer_expression="person_1.full_name",
        evaluated_answer="Alice Marie Quinn",
        entities_used=entities,
    )

    assert spec.answer_schema == "entity_span"
    assert spec.accepted_answers == ("Alice Marie Quinn", "Alice Quinn", "Alice", "Quinn")
    assert score_canonical_prediction(
        canonicalize_answer("entity_span", "Quinn"),
        spec.accepted_answers_canonical,
    )


def test_build_answer_spec_derives_short_person_aliases_from_full_name() -> None:
    entities = EntityCollection(
        persons={
            "person_1": PersonEntity(
                full_name="Barack Hussein Obama II",
                last_name="Obama",
            )
        }
    )

    spec = build_answer_spec(
        question_text="Who led the campaign?",
        answer_expression="person_1.full_name",
        evaluated_answer="Barack Hussein Obama II",
        entities_used=entities,
    )

    assert spec.answer_schema == "entity_span"
    assert "Barack Obama" in spec.accepted_answers
    assert "Barack" in spec.accepted_answers
    assert "Obama" in spec.accepted_answers
    assert score_canonical_prediction(
        canonicalize_answer("entity_span", "Barack Obama"),
        spec.accepted_answers_canonical,
    )


def test_build_answer_spec_respects_full_name_requests() -> None:
    entities = EntityCollection(
        persons={
            "person_1": PersonEntity(
                full_name="Alice Marie Quinn",
                first_name="Alice",
                last_name="Quinn",
            )
        }
    )

    spec = build_answer_spec(
        question_text="What is Alice Quinn's full name?",
        answer_expression="person_1.full_name",
        evaluated_answer="Alice Marie Quinn",
        entities_used=entities,
    )

    assert spec.answer_schema == "entity_span"
    assert spec.accepted_answers == ("Alice Marie Quinn",)


def test_build_answer_spec_expands_parenthetical_acronym_aliases() -> None:
    entities = EntityCollection(legals={"legal_1": LegalEntity(name="Property Loan Governance Directive (PLGD)")})

    spec = build_answer_spec(
        question_text="Which directive applies?",
        answer_expression="legal_1.name",
        evaluated_answer="Property Loan Governance Directive (PLGD)",
        entities_used=entities,
    )

    assert "Property Loan Governance Directive" in spec.accepted_answers
    assert "PLGD" in spec.accepted_answers
    assert score_canonical_prediction(
        canonicalize_answer("entity_span", "PLGD"),
        spec.accepted_answers_canonical,
    )


def test_build_answer_spec_accepts_long_form_with_matching_parenthetical_acronym_suffix() -> None:
    spec = build_answer_spec(
        question_text="Which degree did she study for?",
        answer_expression="Philosophy, Politics and Economics",
        evaluated_answer="Philosophy, Politics and Economics",
        entities_used=EntityCollection(),
    )

    assert "Philosophy, Politics and Economics (PPE)" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("span", "Philosophy, Politics and Economics (PPE)"),
        spec.accepted_answers_canonical,
        answer_schema="span",
    )


def test_parse_schema_answer_normalizes_closed_class_outputs() -> None:
    yes_no = parse_schema_answer("ANSWER: Yes.", "yes_no")
    assert yes_no.parsed_output == "YES"
    assert yes_no.canonical_output == "YES"
    assert yes_no.format_compliant is True

    quantity = parse_schema_answer("ANSWER: 17 years", "quantity")
    assert quantity.parsed_output == "17"
    assert quantity.canonical_output == "17"

    quantity_words = parse_schema_answer("ANSWER: seventeen", "quantity")
    assert quantity_words.parsed_output == "17"
    assert quantity_words.canonical_output == "17"

    quantity_decimal = parse_schema_answer("ANSWER: 0.1099999999999943", "quantity")
    assert quantity_decimal.parsed_output == "0.11"
    assert quantity_decimal.canonical_output == "0.11"


def test_parse_schema_answer_normalizes_scaled_and_fractional_quantities() -> None:
    scaled = parse_schema_answer(
        "ANSWER: 348.4 million users",
        "quantity",
        accepted_answers=("348400000",),
    )
    assert scaled.parsed_output == "348400000"
    assert scaled.canonical_output == "348400000"

    currency_scaled = parse_schema_answer(
        "ANSWER: US$1.653 trillion",
        "quantity",
        accepted_answers=("1653000000000",),
    )
    assert currency_scaled.parsed_output == "1653000000000"
    assert currency_scaled.canonical_output == "1653000000000"

    approximate_currency_scaled = parse_schema_answer(
        "ANSWER: approximately US$1.654 trillion",
        "quantity",
        accepted_answers=("1654000000000",),
    )
    assert approximate_currency_scaled.parsed_output == "1654000000000"
    assert approximate_currency_scaled.canonical_output == "1654000000000"

    fraction = parse_schema_answer(
        "ANSWER: 7.6/11",
        "quantity",
        accepted_answers=("0.69",),
    )
    assert fraction.parsed_output == "0.690909090909"
    assert fraction.canonical_output == "0.690909090909"

    implicit_scale = parse_schema_answer(
        "ANSWER: $94.9 billion",
        "quantity",
        accepted_answers=("94.9",),
    )
    assert implicit_scale.parsed_output == "94.9"
    assert implicit_scale.canonical_output == "94.9"


def test_parse_schema_answer_keeps_composite_entity_when_it_is_accepted() -> None:
    result = parse_schema_answer(
        "ANSWER: Redmond, Washington",
        "entity_span",
        accepted_answers=("Redmond, Washington",),
    )
    assert result.parsed_output == "Redmond, Washington"
    assert result.canonical_output == "redmond washington"


def test_quantity_match_respects_ground_truth_display_precision() -> None:
    assert score_prediction_with_schema(
        "0.690909090909",
        ("0.69",),
        answer_schema="quantity",
    )
    assert not score_prediction_with_schema(
        "0.696",
        ("0.69",),
        answer_schema="quantity",
    )


def test_parse_schema_answer_recovers_boxed_nemotron_style_answers() -> None:
    entity = parse_schema_answer(
        "ANSWER: Nobel Peace Prize\n\n\\boxed{\\text{Nobel Peace Prize}}",
        "entity_span",
    )
    assert entity.parsed_output == "Nobel Peace Prize"
    assert entity.canonical_output == "nobel peace prize"

    refusal = parse_schema_answer(
        "The document does not provide the requested vote total.\n\n\\boxed{\\text{ANSWER: Cannot be determined}}",
        "span",
    )
    assert refusal.parsed_output == "UNANSWERABLE"
    assert refusal.canonical_output == UNANSWERABLE

    quantity = parse_schema_answer("Some reasoning\n\\boxed{3}", "quantity")
    assert quantity.parsed_output == "3"
    assert quantity.canonical_output == "3"


def test_parse_schema_answer_keeps_full_unicode_and_year_prefixed_spans() -> None:
    award = parse_schema_answer("ANSWER: 2009 Nobel Peace Prize", "entity_span")
    assert award.parsed_output == "2009 Nobel Peace Prize"
    assert award.canonical_output == "2009 nobel peace prize"

    accented_name = parse_schema_answer("ANSWER: Sébastien Lecornu", "entity_span")
    assert accented_name.parsed_output == "Sébastien Lecornu"
    assert accented_name.canonical_output == "sébastien lecornu"

    dashed_span = parse_schema_answer("ANSWER: Sumatra–Andaman earthquake", "entity_span")
    assert dashed_span.parsed_output == "Sumatra–Andaman earthquake"
    assert dashed_span.canonical_output == "sumatra-andaman earthquake"

    empty_tag = parse_schema_answer("ANSWER:", "yes_no")
    assert empty_tag.parsed_output == ""
    assert empty_tag.canonical_output == ""
    assert empty_tag.parse_status == "empty"


def test_parse_schema_answer_trims_entity_explanatory_clauses() -> None:
    entity = parse_schema_answer(
        "ANSWER: President Barack Obama was the Democratic Party nominee.",
        "entity_span",
    )
    assert entity.parsed_output == "President Barack Obama"
    assert entity.canonical_output == "barack obama"


def test_build_answer_spec_keeps_refusal_schema_and_token_for_refusal_questions() -> None:
    spec = build_answer_spec(
        question_text="When did the conflict start?",
        answer_expression="Cannot be determined",
        evaluated_answer="Cannot be determined",
        entities_used=EntityCollection(),
    )

    assert spec.answer_schema == "date"
    assert spec.ground_truth_canonical == UNANSWERABLE
    assert spec.accepted_answers == (UNANSWERABLE,)


def test_build_answer_spec_prefers_quantity_for_how_many_questions() -> None:
    spec = build_answer_spec(
        question_text="How many people were mentioned in total?",
        answer_expression="number_2.int + 1",
        evaluated_answer="1001",
        entities_used=EntityCollection(),
    )

    assert spec.answer_schema == "quantity"
    assert spec.ground_truth_canonical == "1001"


def test_build_answer_spec_expands_multi_entity_answer_expressions() -> None:
    entities = EntityCollection(
        legals={
            "legal_1": LegalEntity(name="Pakistan"),
            "legal_2": LegalEntity(name="China"),
        }
    )

    spec = build_answer_spec(
        question_text="Which two countries were involved?",
        answer_expression="legal_1.name, legal_2.name",
        evaluated_answer="Pakistan and China",
        entities_used=entities,
    )

    assert "Pakistan and China" in spec.accepted_answers
    assert "China and Pakistan" in spec.accepted_answers
    assert "Pakistan, China" in spec.accepted_answers


def test_build_answer_spec_adds_trophy_alias_when_question_asks_for_trophy() -> None:
    entities = EntityCollection(
        persons={
            "person_1": PersonEntity(
                full_name="Vince Lombardi",
                first_name="Vince",
                last_name="Lombardi",
            )
        }
    )

    spec = build_answer_spec(
        question_text="Which trophy is awarded to the winner?",
        answer_expression="person_1.full_name",
        evaluated_answer="Vince Lombardi",
        entities_used=entities,
    )

    assert "Vince Lombardi Trophy" in spec.accepted_answers


def test_quantity_scoring_accepts_small_numeric_approximation_only_for_quantity() -> None:
    assert score_prediction_with_schema(
        "0.11",
        ("0.1099999999999943",),
        answer_schema="quantity",
    )
    assert not score_prediction_with_schema(
        "0.12",
        ("0.1099999999999943",),
        answer_schema="quantity",
    )
    assert not score_prediction_with_schema(
        "2008",
        ("2009",),
        answer_schema="year",
    )


def test_build_answer_spec_does_not_add_coordination_aliases_for_arithmetic_expression() -> None:
    entities = EntityCollection(
        numbers={
            "number_7": NumberEntity(float=9.69),
            "number_8": NumberEntity(float=9.58),
        }
    )

    spec = build_answer_spec(
        question_text="By how many seconds did Bolt improve the record?",
        answer_expression="number_7.float - number_8.float",
        evaluated_answer="0.1099999999999943",
        entities_used=entities,
    )

    assert spec.answer_schema == "quantity"
    assert "0.11" in spec.accepted_answers
    assert "9.69 and 9.58" not in spec.accepted_answers
    assert "9.58 and 9.69" not in spec.accepted_answers


def test_build_answer_spec_accepts_explicit_answer_overrides() -> None:
    spec = build_answer_spec(
        question_text="Which magazine named the person of the year?",
        answer_expression="media_org_1.name",
        evaluated_answer="Time magazine",
        entities_used=EntityCollection(),
        accepted_answer_overrides=("Time",),
    )

    assert spec.answer_schema == "entity_span"
    assert "Time magazine" in spec.accepted_answers
    assert "Time" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Time"),
        spec.accepted_answers_canonical,
        answer_schema="entity_span",
    )


def test_build_answer_spec_adds_short_unique_org_alias_for_entity_questions() -> None:
    entities = EntityCollection(
        organizations={
            "entreprise_org_1": OrganizationEntity(name="Vantrill Rivermen", organization_kind="entreprise_org"),
            "entreprise_org_2": OrganizationEntity(name="Solcrest Mariners", organization_kind="entreprise_org"),
        }
    )

    spec = build_answer_spec(
        question_text="Which team selected Kelvion Trathmar Oskendel with the first overall pick?",
        answer_expression="entreprise_org_1.name",
        evaluated_answer="Vantrill Rivermen",
        entities_used=entities,
    )

    assert spec.answer_schema == "entity_span"
    assert "Vantrill" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Vantrill"),
        spec.accepted_answers_canonical,
        answer_schema="entity_span",
    )


def test_build_answer_spec_adds_document_grounded_org_suffix_alias() -> None:
    spec = build_answer_spec(
        question_text="Which company later acquired Celvotix?",
        answer_expression="entreprise_org_1.name",
        evaluated_answer="Veloryn",
        document_text="Veloryn traces its corporate history to 1897. Veloryn Group acquired Celvotix in 2015.",
        entities_used=EntityCollection(),
    )

    assert "Veloryn Group" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Veloryn Group"),
        spec.accepted_answers_canonical,
        answer_schema="entity_span",
    )


def test_build_answer_spec_adds_document_grounded_publication_suffix_alias() -> None:
    spec = build_answer_spec(
        question_text="Which publication recognized him among the most influential people?",
        answer_expression="media_org_1.name",
        evaluated_answer="Zephkol",
        document_text="Trovandis was included in Zephkol Tribune's 100 Most Influential People of 2016.",
        entities_used=EntityCollection(),
    )

    assert "Zephkol Tribune" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Zephkol Tribune"),
        spec.accepted_answers_canonical,
        answer_schema="entity_span",
    )


def test_build_answer_spec_adds_document_grounded_lowercase_publication_suffix_alias() -> None:
    spec = build_answer_spec(
        question_text="Which magazine named the athlete of the year?",
        answer_expression="media_org_1.name",
        evaluated_answer="Time",
        document_text="The award was covered by Time magazine in its annual special issue.",
        entities_used=EntityCollection(),
    )

    assert "Time magazine" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Time magazine"),
        spec.accepted_answers_canonical,
        answer_schema="entity_span",
    )


def test_build_answer_spec_adds_degree_core_alias_for_degree_questions() -> None:
    spec = build_answer_spec(
        question_text="What degree did Haroven Zelkuun Pashwen earn from Crestivane College?",
        answer_expression="Bachelor of Arts degree in political science",
        evaluated_answer="Bachelor of Arts degree in political science",
        entities_used=EntityCollection(),
    )

    assert "Bachelor of Arts" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("span", "Bachelor of Arts"),
        spec.accepted_answers_canonical,
        answer_schema="span",
        raw_prediction="Bachelor of Arts",
    )


def test_build_answer_spec_skips_ambiguous_document_grounded_org_suffix_aliases() -> None:
    spec = build_answer_spec(
        question_text="Which company handled the acquisition?",
        answer_expression="entreprise_org_1.name",
        evaluated_answer="Veloryn",
        document_text="Veloryn Group expanded rapidly. Veloryn Holdings later sold the division.",
        entities_used=EntityCollection(),
    )

    assert "Veloryn Group" not in spec.accepted_answers
    assert "Veloryn Holdings" not in spec.accepted_answers


def test_build_answer_spec_adds_profession_modifier_aliases() -> None:
    entities = EntityCollection(
        places={
            "place_1": PlaceEntity(demonym="Telvoran"),
        }
    )

    spec = build_answer_spec(
        question_text="What was Torvek Mbashelu's profession?",
        answer_expression="gynecologist",
        evaluated_answer="gynecologist",
        entities_used=entities,
    )

    assert spec.answer_schema == "span"
    assert "Telvoran gynecologist" in spec.accepted_answers
    assert score_prediction_with_schema(
        canonicalize_answer("span", "Telvoran gynecologist"),
        spec.accepted_answers_canonical,
        answer_schema="span",
    )


def test_score_prediction_with_schema_accepts_person_name_with_origin_qualifier() -> None:
    accepted = (canonicalize_answer("entity_span", "Vedran Polkhari"),)

    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Vedran Polkhari of Velorinth"),
        accepted,
        answer_schema="entity_span",
        raw_prediction="Vedran Polkhari of Velorinth",
    )


def test_score_prediction_with_schema_accepts_publication_possessive_descriptor_suffix() -> None:
    accepted = (
        canonicalize_answer("entity_span", "Time magazine"),
        canonicalize_answer("entity_span", "Time"),
    )

    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Time magazine's 100 Most Influential People of 2016"),
        accepted,
        answer_schema="entity_span",
        raw_prediction="Time magazine's 100 Most Influential People of 2016",
    )


def test_score_prediction_with_schema_accepts_organization_possessive_descriptor_suffix() -> None:
    accepted = (
        canonicalize_answer("entity_span", "European Commission"),
        canonicalize_answer("entity_span", "Commission"),
    )

    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "European Commission's Financial Services Action Plan"),
        accepted,
        answer_schema="entity_span",
        raw_prediction="European Commission's Financial Services Action Plan",
    )


def test_score_prediction_with_schema_accepts_unicode_dash_variant() -> None:
    accepted = (canonicalize_answer("span", "self-driving car division"),)

    assert score_prediction_with_schema(
        canonicalize_answer("span", "self–driving car division"),
        accepted,
        answer_schema="span",
        raw_prediction="self–driving car division",
    )


def test_score_prediction_with_schema_accepts_list_without_and() -> None:
    accepted = (canonicalize_answer("span", "gold, silver, and bronze"),)

    assert score_prediction_with_schema(
        canonicalize_answer("span", "gold, silver, bronze"),
        accepted,
        answer_schema="span",
        raw_prediction="gold, silver, bronze",
    )


def test_score_prediction_with_schema_accepts_trailing_format_suffix() -> None:
    accepted = (canonicalize_answer("span", "best-of-nine"),)

    assert score_prediction_with_schema(
        canonicalize_answer("span", "best-of-nine format"),
        accepted,
        answer_schema="span",
        raw_prediction="best-of-nine format",
    )


def test_score_prediction_with_schema_accepts_year_reordered_title() -> None:
    accepted = (canonicalize_answer("entity_span", "2009 Nobel Peace Prize"),)

    assert score_prediction_with_schema(
        canonicalize_answer("entity_span", "Nobel Peace Prize (2009)"),
        accepted,
        answer_schema="entity_span",
        raw_prediction="Nobel Peace Prize (2009)",
    )


def test_parse_schema_answer_preserves_reference_codes_with_parenthetical_prefix() -> None:
    result = parse_schema_answer(
        "ANSWER: EU No. 575/2013",
        "entity_span",
        accepted_answers=(" (EU) No. 575/2013 ".strip(),),
    )

    assert result.parsed_output == "EU No. 575/2013"
    assert result.canonical_output == canonicalize_answer("entity_span", "(EU) No. 575/2013")


def test_parse_schema_answer_recovers_entity_from_gpt_oss_analysis_only_output() -> None:
    raw = (
        "<|channel|>analysis<|message|>We need year of third consecutive Wyndoric double sprint victory. "
        "He won Wyndoric 100m and 200m titles at three consecutive Wyndoric (2008, 2012, 2016). "
        "So third is 2016. Which publication recognized him among golvantis's most influential people in same year? "
        "Document says: \"Trovandis was included in Zephkol Tribune's 100 Most Influential People of 2016"
    )

    result = parse_schema_answer(
        raw,
        "entity_span",
        accepted_answers=("Zephkol Tribune",),
    )

    assert result.parsed_output == "Zephkol Tribune"
    assert result.canonical_output == "zephkol tribune"


def test_parse_schema_answer_recovers_unanswerable_from_gpt_oss_analysis_only_output() -> None:
    raw = (
        "<|channel|>analysis<|message|>We need to see if document says she used education during defense and "
        "if she represented herself. Document: mentions she studied law, but no mention of her representing "
        "herself. So cannot determine"
    )

    result = parse_schema_answer(
        raw,
        "span",
        accepted_answers=(UNANSWERABLE,),
    )

    assert result.parsed_output == UNANSWERABLE
    assert result.canonical_output == UNANSWERABLE


def test_parse_schema_answer_recovers_answer_from_raw_reasoning_without_final_channel() -> None:
    raw = (
        "We need total elite women's solo championships in Vanthor Period. "
        "Document says 59 Tovaren League championship singles accolades, including 19 elite women's solo championships. "
        "So answer 19."
    )

    result = parse_schema_answer(
        raw,
        "quantity",
        accepted_answers=("19",),
    )

    assert result.parsed_output == "19"
    assert result.canonical_output == "19"


def test_parse_schema_answer_recovers_unanswerable_from_truncated_uncertain_reasoning() -> None:
    raw = (
        "<|channel|>analysis<|message|>We need months served before acquisition. "
        "He became CEO in 2001. Acquisition in 2002. So from 2001 to 2002: unclear months. "
        "Likely 12 months? But could be less."
    )

    result = parse_schema_answer(
        raw,
        "quantity",
        accepted_answers=(UNANSWERABLE,),
    )

    assert result.parsed_output == UNANSWERABLE
    assert result.canonical_output == UNANSWERABLE


def test_parse_schema_answer_recovers_before_after_from_reasoning_only_text() -> None:
    raw = "We need to compare the dates. The earlier event happened in 1993 and the later in 2013. So before."

    result = parse_schema_answer(
        raw,
        "before_after",
        accepted_answers=("BEFORE",),
    )

    assert result.parsed_output == "BEFORE"
    assert result.canonical_output == "BEFORE"


def test_parse_schema_answer_recovers_before_after_when_it_is_the_first_word() -> None:
    raw = "Before, because the first event happened in 1993 and the second in 2013."

    result = parse_schema_answer(
        raw,
        "before_after",
        accepted_answers=("BEFORE",),
    )

    assert result.parsed_output == "BEFORE"
    assert result.canonical_output == "BEFORE"


def test_parse_schema_answer_accepts_before_after_answer_tag_with_trailing_phrase() -> None:
    raw = "ANSWER: After his graduation from École nationale d'administration."

    result = parse_schema_answer(
        raw,
        "before_after",
        accepted_answers=("AFTER",),
    )

    assert result.parsed_output == "AFTER"
    assert result.canonical_output == "AFTER"


def test_parse_schema_answer_accepts_yes_no_answer_tag_with_trailing_phrase() -> None:
    result = parse_schema_answer(
        "ANSWER: Yes, because the document states it explicitly.",
        "yes_no",
    )

    assert result.parsed_output == "YES"
    assert result.canonical_output == "YES"


def test_parse_schema_answer_normalizes_earlier_later_as_before_after() -> None:
    earlier = parse_schema_answer("ANSWER: Earlier than the reference event.", "before_after")
    later = parse_schema_answer("ANSWER: Later.", "before_after")

    assert earlier.canonical_output == "BEFORE"
    assert later.canonical_output == "AFTER"


def test_format_compliant_answer_is_not_replaced_by_accepted_answer_recovery() -> None:
    result = parse_schema_answer(
        "ANSWER: Yes.\nA later duplicate line says ANSWER: No.",
        "yes_no",
        accepted_answers=("YES",),
    )

    assert result.format_compliant is True
    assert result.canonical_output == "NO"


def test_score_prediction_with_schema_retries_without_parenthetical_suffix() -> None:
    accepted = (canonicalize_answer("entity_span", "Revised Payment Services Directive"),)
    predicted_text = "Revised Payment Services Directive (PSD2)"
    predicted_canonical = canonicalize_answer("entity_span", predicted_text)

    assert predicted_canonical != accepted[0]
    assert score_prediction_with_schema(
        predicted_canonical,
        accepted,
        answer_schema="entity_span",
        raw_prediction=predicted_text,
    )


def test_answer_evaluator_resolves_currency_concat_and_symbol_affix() -> None:
    entities = EntityCollection(numbers={"number_2": NumberEntity(int=9)})

    assert AnswerEvaluator.evaluate_answer('"$" + number_2.int', entities) == "$9"
    assert AnswerEvaluator.evaluate_answer("$ number_2.int", entities) == "$9"


def test_answer_evaluator_resolves_temporal_string_concatenation() -> None:
    entities = EntityCollection(temporals={"temporal_4": TemporalEntity(month="September", day_of_month=16)})

    assert (
        AnswerEvaluator.evaluate_answer(
            'temporal_4.month + " " + temporal_4.day_of_month',
            entities,
        )
        == "September 16"
    )


def test_answer_evaluator_treats_number_str_as_numeric_in_arithmetic() -> None:
    entities = EntityCollection(numbers={"number_1": NumberEntity(int=4, str="four")})

    assert AnswerEvaluator.evaluate_answer("number_1.str + 2", entities) == "6"


def test_answer_evaluator_preserves_semicolon_punctuation_in_literal_answer() -> None:
    entities = EntityCollection()
    answer = (
        "the free movement of people, goods, services and capital within the "
        "internal market; enact legislation in justice and home affairs; and "
        "maintain common policies on trade"
    )

    assert AnswerEvaluator._clean_semicolon_syntax(answer) == answer
    assert AnswerEvaluator.evaluate_answer(answer, entities) == answer


def test_answer_evaluator_resolves_weekday_shift_with_entity_offset() -> None:
    entities = EntityCollection(
        temporals={"temporal_1": TemporalEntity(day="Tuesday")},
        numbers={"number_3": NumberEntity(int=3)},
    )

    assert AnswerEvaluator.evaluate_answer("temporal_1.day - number_3.int", entities) == "Saturday"


def test_answer_evaluator_resolves_elapsed_days_between_weekdays() -> None:
    entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(day="Monday"),
            "temporal_2": TemporalEntity(day="Tuesday"),
        }
    )

    assert AnswerEvaluator.evaluate_answer("temporal_2.day - temporal_1.day", entities) == "1"
    assert AnswerEvaluator.evaluate_answer("temporal_1.day - temporal_2.day", entities) == "6"


def test_answer_evaluator_resolves_year_of_date_difference() -> None:
    entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(date="4 October 1957"),
            "temporal_2": TemporalEntity(date="4 January 1958"),
            "temporal_3": TemporalEntity(date="5 October 1958"),
        }
    )

    assert AnswerEvaluator.evaluate_answer("year(temporal_2.date - temporal_1.date)", entities) == "0"
    assert AnswerEvaluator.evaluate_answer("year(temporal_3.date - temporal_1.date)", entities) == "1"


def test_answer_evaluator_resolves_conditional_before_concat() -> None:
    entities = EntityCollection(
        temporals={
            "temporal_1": TemporalEntity(year=1974),
            "temporal_3": TemporalEntity(year=1991),
        }
    )

    assert (
        AnswerEvaluator.evaluate_answer(
            "Yes if temporal_3.year < temporal_1.year + 40 else No",
            entities,
        )
        == "Yes"
    )

    entities.temporals["temporal_1"] = TemporalEntity(year=1960)
    entities.temporals["temporal_3"] = TemporalEntity(year=2009)
    assert (
        AnswerEvaluator.evaluate_answer(
            "Yes if temporal_3.year < temporal_1.year + 40 else No",
            entities,
        )
        == "No"
    )


@pytest.fixture(scope="module")
def generated_documents(tmp_path_factory):
    """Build the small document fixture from the public source templates."""
    import subprocess
    import sys

    output = tmp_path_factory.mktemp("generated_examples") / "dataset"
    subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/dataset.py"),
        "--output", str(output), "--docs", "bio_08", "company_11", "disaster_05",
        "train_175", "space_12", "--settings", "factual", "fictional",
        "--variants", "10", "--workers", "1",
    ], check=True, capture_output=True, text=True)
    return output


def test_load_evaluation_document_recomputes_unresolved_currency_ground_truth(generated_documents) -> None:
    factual_document = load_evaluation_document(
        generated_documents / "FACTUAL_DOCUMENTS/biographies_of_famous_personalities/bio_08.yaml"
    )
    fictional_document = load_evaluation_document(
        generated_documents / "FICTIONAL_DOCUMENTS/fictional/biographies_of_famous_personalities/bio_08_v01.yaml"
    )

    factual_question = next(question for question in factual_document.questions if question.question_id == "bio_08_q04")
    fictional_question = next(
        question for question in fictional_document.questions if question.question_id == "bio_08_q04"
    )

    assert factual_question.ground_truth == "$10"
    assert fictional_question.ground_truth == "$8"


def test_load_evaluation_document_resolves_company_11_temporal_ground_truth(generated_documents) -> None:
    factual_document = load_evaluation_document(
        generated_documents / "FACTUAL_DOCUMENTS/companies_and_organizations/company_11.yaml"
    )

    factual_question = next(
        question for question in factual_document.questions if question.question_id == "company_11_q07"
    )

    assert factual_question.ground_truth == "23"
    assert factual_question.ground_truth_canonical == "23"
    assert factual_question.accepted_answers_canonical == ("23",)


def test_load_evaluation_document_accepts_reviewed_thousands_separator_typography(generated_documents) -> None:
    cases = (
        (
            "data/FACTUAL_DOCUMENTS/natural_disasters/disaster_05.yaml",
            "disaster_05_reviewer09_w3",
            "Did Hurricane Maria's total death toll exceed 3000?",
        ),
        (
            "data/FACTUAL_DOCUMENTS/public_attacks_news_articles/train_175.yaml",
            "train_175_w10",
            "Did the report attribute more than 5000 civilian deaths to coalition strikes?",
        ),
        (
            "data/FACTUAL_DOCUMENTS/space_missions/space_12.yaml",
            "space_12_q06",
            "Given that Voyager 1 travels at 61000 kilometers per hour, what is the difference "
            "between Voyager 1's mass and Voyager 2's mass in kilograms?",
        ),
    )
    for path, question_id, expected_text in cases:
        document = load_evaluation_document(generated_documents / Path(path).relative_to("data"))
        question = next(row for row in document.questions if row.question_id == question_id)
        assert question.question_text == expected_text
        if question_id != "space_12_q06":
            assert question.ground_truth == "Yes"
        else:
            assert question.ground_truth == "Cannot be determined"


def test_load_evaluation_document_uses_benchmark_reference_for_bio_08(generated_documents) -> None:
    factual_document = load_evaluation_document(
        generated_documents / "FACTUAL_DOCUMENTS/biographies_of_famous_personalities/bio_08.yaml"
    )
    factual_question = next(question for question in factual_document.questions if question.question_id == "bio_08_q02")

    assert factual_question.ground_truth == "Cannot be determined"
    assert factual_question.ground_truth_canonical == UNANSWERABLE
