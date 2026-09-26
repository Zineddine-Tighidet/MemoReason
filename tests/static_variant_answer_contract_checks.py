from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from memoreason.benchmark_definition.annotated_document_io import load_annotated_document
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement import (
    build_controlled_entity_replacement_context,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_ROOT = REPO_ROOT / "data" / "HUMAN_ANNOTATED_TEMPLATES"


def _write_template(
    path: Path,
    *,
    document_text: str,
    rules: list[str],
    answer: str,
    question_type: str = "arithmetic",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "document": {
                    "document_id": path.stem,
                    "document_theme": path.parent.name,
                    "original_document": "",
                    "document_to_annotate": document_text,
                    "rules": rules,
                    "questions": [
                        {
                            "question_id": "q_variant",
                            "question": "What is the answer?",
                            "answer": answer,
                            "question_type": question_type,
                            "answer_type": "variant",
                            "reasoning_chain": [],
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("document_text", "rules", "answer", "question_type"),
    [
        (
            "[7; number_1.int] goals and [4; number_2.int] assists.",
            ["number_1.int == number_2.int + 3"],
            "number_1.int - number_2.int",
            "arithmetic",
        ),
        (
            "[7; number_1.int], [4; number_2.int], and [1; number_3.int].",
            ["number_1.int == number_3.int + 6", "number_2.int == number_3.int + 3"],
            "number_1.int - number_2.int",
            "arithmetic",
        ),
        (
            "A score of [7; number_1.int].",
            ["number_1.int > 5"],
            "Yes if number_1.int > 5 else No",
            "inference",
        ),
        (
            "The [Westshire; place_1.region] office.",
            ['place_1.region == "Westshire"'],
            "place_1.region",
            "extractive",
        ),
    ],
)
def test_preflight_accepts_variant_answers_fixed_by_explicit_rules(
    tmp_path: Path,
    document_text: str,
    rules: list[str],
    answer: str,
    question_type: str,
) -> None:
    template_path = _write_template(
        tmp_path / "theme" / "fixed.yaml",
        document_text=document_text,
        rules=rules,
        answer=answer,
        question_type=question_type,
    )

    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))

    assert context.generation_document.questions[0].answer == answer


def test_preflight_accepts_nontrivial_algebraically_constant_variant_answer(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme" / "constant.yaml",
        document_text="[7; number_1.int] goals and [4; number_2.int] assists.",
        rules=[],
        answer="number_1.int + number_2.int - number_1.int - number_2.int",
    )

    context = build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))

    assert context.generation_document.questions[0].answer == (
        "number_1.int + number_2.int - number_1.int - number_2.int"
    )


@pytest.mark.parametrize(
    ("rules", "answer"),
    [
        (["number_1.int > number_2.int"], "number_1.int - number_2.int"),
        (["number_1.int == number_2.int + 3"], "number_1.int"),
    ],
)
def test_preflight_keeps_variant_answers_not_fixed_by_explicit_rules(
    tmp_path: Path,
    rules: list[str],
    answer: str,
) -> None:
    template_path = _write_template(
        tmp_path / "theme" / "variable.yaml",
        document_text="[7; number_1.int] goals and [4; number_2.int] assists.",
        rules=rules,
        answer=answer,
    )

    build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))


def test_preflight_keeps_variable_branch_under_rule_fixed_condition(tmp_path: Path) -> None:
    template_path = _write_template(
        tmp_path / "theme" / "variable_conditional_branch.yaml",
        document_text="[7; number_1.int] goals and [4; number_2.int] assists.",
        rules=["number_1.int > 5"],
        answer="number_2.int if number_1.int > 5 else 0",
        question_type="inference",
    )

    build_controlled_entity_replacement_context(load_annotated_document(str(template_path)))


def test_all_100_active_dataset_templates_pass_static_variant_preflight() -> None:
    template_paths = sorted(path for path in TEMPLATES_ROOT.rglob("*.yaml") if path.stem != "bankreg_11")
    assert len(template_paths) == 100

    for template_path in template_paths:
        document = load_annotated_document(str(template_path), validate_question_scope=True)
        build_controlled_entity_replacement_context(document)
