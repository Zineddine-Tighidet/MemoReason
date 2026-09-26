"""Active benchmark golds must bind to exact immutable inputs, never just IDs."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import gzip
import hashlib
import json

import pytest
import yaml

from memoreason.model_evaluation import benchmark_document_loading as documents
from memoreason.model_evaluation import benchmark_scoring_references as references


DOCUMENT = "Elected in 1910; appointed in 1913."
QUESTION = "How many years after election was he appointed?"


@pytest.fixture
def row():
    return {
        "setting": "factual",
        "document_id": "benchmark_01",
        "variant_id": "v01",
        "question_id": "benchmark_01_q01",
        "document_text_sha256": hashlib.sha256(DOCUMENT.encode()).hexdigest(),
        "question_text_sha256": hashlib.sha256(QUESTION.encode()).hexdigest(),
        "frozen_ground_truth": "63",
        "ground_truth": "3",
        "answer_expression": "temporal_2.year - temporal_1.year",
        "answer_schema": "quantity",
        "accepted_answers": ["3", "3.0"],
        "accepted_answer_overrides": ["3.0"],
    }


def write_references(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for item in rows:
            stream.write(json.dumps(item) + "\n")
    return path


def match(index, **changes):
    arguments = {
        "setting": "factual", "document_id": "benchmark_01", "variant_id": "v01",
        "question_id": "benchmark_01_q01", "document_text": DOCUMENT,
        "question_text": QUESTION, "frozen_ground_truth": "63",
    }
    return index.match(**{**arguments, **changes})


def test_exact_match_preserves_alternatives_and_derives_schema_canonicals(tmp_path, row):
    path = write_references(tmp_path / "refs.jsonl.gz", [row])
    before = path.read_bytes()
    result = match(references.load_scoring_references(path))
    assert result.ground_truth == "3"
    assert result.ground_truth_canonical == "3"
    assert result.answer_schema == "quantity"
    assert result.accepted_answers == ("3", "3.0")
    assert result.accepted_answers_canonical == ("3",)
    assert result.accepted_answer_overrides == ("3.0",)
    assert result.frozen_ground_truth == "63"
    assert path.read_bytes() == before
    with pytest.raises(FrozenInstanceError):
        result.ground_truth = "63"


def test_stored_canonical_fields_are_preserved(tmp_path, row):
    row.update(ground_truth_canonical="3.000", accepted_answers_canonical=["3.000", "3"])
    result = match(references.load_scoring_references(write_references(tmp_path / "refs.jsonl.gz", [row])))
    assert result.ground_truth_canonical == "3.000"
    assert result.accepted_answers_canonical == ("3.000", "3")


def test_empty_accepted_answers_are_preserved(tmp_path, row):
    row.update(ground_truth="same report", answer_schema="yes_no", accepted_answers=[])
    result = match(references.load_scoring_references(write_references(tmp_path / "refs.jsonl.gz", [row])))
    assert result.accepted_answers == ()
    assert result.accepted_answers_canonical == ()


@pytest.mark.parametrize("field,value", [
    ("document_text", DOCUMENT + " "),
    ("question_text", QUESTION + " "),
    ("frozen_ground_truth", " 63 "),
])
def test_same_identifiers_with_changed_input_fail_closed(tmp_path, row, field, value):
    index = references.load_scoring_references(write_references(tmp_path / "refs.jsonl.gz", [row]))
    with pytest.raises(ValueError, match="mismatch"):
        match(index, **{field: value})


@pytest.mark.parametrize("field,value", [
    ("setting", "fictional"), ("variant_id", "v02"), ("question_id", "new_question"),
])
def test_missing_key_for_known_benchmark_document_fails_closed(tmp_path, row, field, value):
    index = references.load_scoring_references(write_references(tmp_path / "refs.jsonl.gz", [row]))
    with pytest.raises(ValueError, match="Missing scoring reference"):
        match(index, **{field: value})


def test_unrelated_document_has_no_benchmark_override(tmp_path, row):
    index = references.load_scoring_references(write_references(tmp_path / "refs.jsonl.gz", [row]))
    assert match(index, document_id="my_new_document", frozen_ground_truth="different") is None


def test_duplicate_reference_is_rejected(tmp_path, row):
    path = write_references(tmp_path / "refs.jsonl.gz", [row, deepcopy(row)])
    with pytest.raises(ValueError, match="Duplicate scoring reference"):
        references.load_scoring_references(path)


def test_missing_file_is_rejected_even_after_cached_load(tmp_path, row):
    path = write_references(tmp_path / "refs.jsonl.gz", [row])
    references.load_scoring_references(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        references.load_scoring_references(path)


def test_incomplete_default_snapshot_is_rejected(tmp_path, row, monkeypatch):
    path = write_references(tmp_path / "refs.jsonl.gz", [row])
    monkeypatch.setattr(references, "DEFAULT_SCORING_REFERENCES", path)
    with pytest.raises(ValueError, match="expected 109200, found 1"):
        references.load_scoring_references()


@pytest.fixture
def document_fixture(tmp_path, row, monkeypatch):
    reference_path = write_references(tmp_path / "refs.jsonl.gz", [row])
    index = references.load_scoring_references(reference_path)
    monkeypatch.setattr(documents, "load_scoring_references", lambda: index)
    template_path = tmp_path / "template.yaml"
    template_path.write_text(yaml.safe_dump({"document": {"questions": [{
        "question_id": row["question_id"], "question": QUESTION,
        "question_type": "temporal", "answer_type": "refusal",
    }]}}), encoding="utf-8")
    payload = {
        "document_id": row["document_id"], "document_theme": "example",
        "document_setting": "factual", "document_variant_id": "v01",
        "document_text": DOCUMENT, "source_template_path": str(template_path),
        "entities_used": {}, "questions": [{
            "question_id": row["question_id"], "question_text": QUESTION,
            "question_type": "temporal", "evaluated_answer": "63",
            "answer_expression": "old unresolved expression",
        }],
    }
    path = tmp_path / "benchmark_01.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path, payload, template_path


def test_document_loader_applies_bound_golds_without_changing_inputs_or_labels(document_fixture):
    path, payload, template_path = document_fixture
    before = (path.read_bytes(), template_path.read_bytes())
    document = documents.load_evaluation_document(path)
    question = document.questions[0]
    assert question.ground_truth == "3"  # Original refusal label must not undo the active gold.
    assert question.answer_expression == "temporal_2.year - temporal_1.year"
    assert question.answer_schema == "quantity"
    assert question.accepted_answers == ("3", "3.0")
    assert question.accepted_answers_canonical == ("3",)
    assert question.answer_behavior == "refusal"
    assert question.question_type == "temporal"
    assert question.question_text == payload["questions"][0]["question_text"]
    assert document.document_text == payload["document_text"]
    assert (path.read_bytes(), template_path.read_bytes()) == before


def test_document_loader_binds_original_raw_gold(document_fixture):
    path, payload, _ = document_fixture
    payload["questions"][0]["evaluated_answer"] = " 63 "
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen-gold mismatch"):
        documents.load_evaluation_document(path)


def test_document_loader_keeps_generated_golds_for_new_documents(document_fixture):
    path, payload, _ = document_fixture
    payload["document_id"] = "new_document"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    question = documents.load_evaluation_document(path).questions[0]
    assert question.ground_truth == "Cannot be determined"
    assert question.answer_behavior == "refusal"
