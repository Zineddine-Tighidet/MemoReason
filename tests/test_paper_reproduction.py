"""CPU-only tests for the public reproduction entry point; never call providers."""

import csv
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import paper
from analysis import entity_counts, paper_inputs, paper_reports
from memoreason.factual_to_fictional_dataset.dataset_record_core import build_factual_dataset_record
from memoreason.model_evaluation.benchmark_scoring_references import load_scoring_references


@pytest.mark.parametrize(
    "name,count", [("main", 9), ("partial", 7), ("effort", 2), ("ablations", 4), ("temperature", 12)]
)
def test_experiment_job_scope(name, count):
    jobs = paper.experiment_jobs(name)
    assert len(jobs) == count
    assert len({job[0] for job in jobs}) == count
    for job in jobs:
        commands = paper.commands_for(job, Path("dataset"), Path("runs"))
        assert commands[0][0] == job[-1]
        assert commands[1][0] == "low"  # Judge effort never follows the answer model.
        assert commands[1][1][-8:] == [
            "--judge-model",
            paper_inputs.JUDGE_MODEL,
            "--judge-max-tokens",
            "256",
            "--judge-temperature",
            "0",
            "--judge-seed",
            "23",
        ]
        for _, command in commands:
            assert ("--generation-seed" in command) == (job[1] != "claude-sonnet-4-6")
            assert command[command.index("--generation-temperature") + 1] == str(job[3])


def test_run_is_dry_by_default(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted a subprocess")

    monkeypatch.setattr(paper.subprocess, "run", forbidden)
    assert paper.main(["run", "effort", "--runs", str(tmp_path / "runs")]) == 0
    assert "GROQ_GPT_OSS_REASONING_EFFORT=high" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_execution_preserves_separate_generation_and_judge_effort(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(paper.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    paper.main(["run", "effort", "--execute", "--runs", str(tmp_path)])
    assert [kwargs["env"]["GROQ_GPT_OSS_REASONING_EFFORT"] for _, kwargs in calls] == ["medium", "low", "high", "low"]
    assert all(kwargs["check"] for _, kwargs in calls)


@pytest.fixture
def evaluated_run(tmp_path, monkeypatch):
    template = paper.ROOT / "data/HUMAN_ANNOTATED_TEMPLATES/companies_and_organizations/company_02.yaml"
    document = build_factual_dataset_record(template, seed=23)
    refs = load_scoring_references()
    selected = {key: ref for key, ref in refs.records.items() if key[:3] == ("factual", "company_02", "v01")}
    monkeypatch.setattr(paper_inputs, "load_scoring_references", lambda: SimpleNamespace(records=selected))
    results = []
    for question in document["questions"]:
        qid = question["question_id"]
        ref = selected["factual", "company_02", "v01", qid]
        text = question["question_text"]
        assert hashlib.sha256(text.encode()).hexdigest() == ref.question_text_sha256
        theme, qtype, behavior = paper_inputs.question_metadata()["company_02", qid]
        results.append(
            dict(
                question_id=qid,
                pair_key=f"{theme}::company_02::{qid}",
                question_text=text,
                question_type=qtype,
                answer_behavior=behavior,
                ground_truth_canonical=ref.ground_truth_canonical,
                accepted_answers_canonical=list(ref.accepted_answers_canonical),
                parsed_output_canonical=ref.ground_truth_canonical,
                exact_match=True,
                judge_match=None,
                final_is_correct=True,
            )
        )
    payload = dict(
        model_name="gpt-oss-20b-groq",
        document_theme=theme,
        document_id="company_02",
        document_setting="factual",
        document_variant_id="v01",
        results=results,
        generation_config=dict(temperature=0.0, seed=23, reasoning_effort="low"),
        judge_model_name=paper_inputs.JUDGE_MODEL,
        judge_config=dict(
            provider="groq", model_name=paper_inputs.JUDGE_MODEL, temperature=0.0, max_tokens=256, seed=23
        ),
    )
    path = tmp_path / "fixture_evaluated_outputs.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path, payload


def read_fixture(path):
    return paper_inputs.load_run(path.parent, ("gpt-oss-20b-groq",), ("factual",))["gpt-oss-20b-groq"]


def test_evaluated_adapter_matches_real_template_and_references(evaluated_run):
    path, _ = evaluated_run
    rows = read_fixture(path)
    assert len(rows) == 12
    assert all(row["new_final_is_correct"] for row in rows)
    assert {row["answer_behavior"] for row in rows} == {"variant", "invariant", "refusal"}


@pytest.mark.parametrize(
    "change,error",
    [
        ({"ground_truth_canonical": "invalid gold"}, "stale reference"),
        ({"question_text": "different question"}, "changed question"),
        ({"exact_match": "true"}, "non-boolean"),
        ({"exact_match": False, "judge_match": None, "final_is_correct": False}, "missing judge"),
        ({"exact_match": True, "final_is_correct": False}, "inconsistent final"),
        ({"pair_key": "invalid"}, "pairing"),
        ({"question_type": "other"}, "taxonomy"),
    ],
)
def test_evaluated_adapter_rejects_invalid_decisions(evaluated_run, change, error):
    path, payload = evaluated_run
    payload["results"][0].update(change)
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=error):
        read_fixture(path)


def test_evaluated_adapter_accepts_schema_incompatible_failures(evaluated_run):
    path, payload = evaluated_run
    payload["results"][0].update(
        exact_match=False,
        judge_match=None,
        final_is_correct=False,
        judge_skip_reason="schema_incompatible_prediction",
        parsed_output_canonical="",
    )
    path.write_text(yaml.safe_dump(payload))
    assert sum(row["new_final_is_correct"] for row in read_fixture(path)) == 11


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("temperature", 0.5, "temperature/seed"),
        ("seed", 24, "temperature/seed"),
        ("reasoning_effort", "high", "effort"),
    ],
)
def test_evaluated_adapter_rejects_mixed_conditions(evaluated_run, field, value, error):
    path, payload = evaluated_run
    payload["generation_config"][field] = value
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=error):
        read_fixture(path)


def test_evaluated_adapter_rejects_wrong_judge(evaluated_run):
    path, payload = evaluated_run
    payload["judge_config"]["temperature"] = 1
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="judge configuration"):
        read_fixture(path)


def test_evaluated_adapter_rejects_duplicates(evaluated_run):
    path, payload = evaluated_run
    (path.parent / "duplicate_evaluated_outputs.yaml").write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="duplicate"):
        read_fixture(path)


def test_evaluated_adapter_rejects_missing_scope(evaluated_run):
    path, payload = evaluated_run
    payload["model_name"] = "not-requested"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="incomplete input"):
        read_fixture(path)


def test_temperature_adapter_aggregates_correctly(evaluated_run, monkeypatch):
    from analysis import temperature

    rows = read_fixture(evaluated_run[0])
    monkeypatch.setattr(temperature, "CONDITIONS", (("t00_s23", 0.0, 23),))
    monkeypatch.setattr(temperature, "MODELS", ("gpt-oss-20b-groq",))
    records = paper_inputs.temperature_records({"t00_s23": {"gpt-oss-20b-groq": rows}})
    assert len(records) == 1
    assert records[0]["final_correct"] == records[0]["total"] == 12


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"count": None},
        {"count": -1},
        {"count": True},
        {"count": 0, "error": "unavailable"},
        {"count": 2, "approx": True},
    ],
)
def test_missing_corpus_counts_never_become_zero(payload):
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    session = SimpleNamespace(post=lambda *args, **kwargs: response)
    with pytest.raises(ValueError, match="No exact count"):
        entity_counts.query_count("Example", session)


def test_explicit_zero_corpus_count_is_valid():
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"count": 0})
    assert entity_counts.query_count("Example", SimpleNamespace(post=lambda *a, **kw: response)) == 0


def test_counts_are_dry_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(entity_counts, "named_entities", lambda _: {("theme", "document"): {"key": "Example"}})
    monkeypatch.setattr(entity_counts.requests, "Session", lambda: pytest.fail("Network session opened in dry run"))
    entity_counts.collect_counts(tmp_path, tmp_path / "counts.csv")
    assert not list(tmp_path.iterdir())


def test_csv_never_overwrites(tmp_path):
    path = tmp_path / "table.csv"
    paper_reports.write_csv(path, [{"value": 1}])
    with pytest.raises(FileExistsError):
        paper_reports.write_csv(path, [{"value": 2}])
    with path.open() as stream:
        assert list(csv.DictReader(stream)) == [{"value": "1"}]
