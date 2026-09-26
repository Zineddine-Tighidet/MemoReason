"""Review commands must not start implicit broad model/provider execution."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVALUATE = ROOT / "scripts/model_evaluation/generate_parse_and_score_model_answers.py"


def test_no_argument_model_evaluation_requires_scope():
    result = subprocess.run([sys.executable, str(EVALUATE)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "required" in result.stderr


def test_model_execution_needs_opt_in(tmp_path):
    result = subprocess.run([
        sys.executable, str(EVALUATE), "--steps", "raw", "--models", "olmo-3-7b-think",
        "--settings", "factual", "--model-eval-dir", str(tmp_path / "outputs"),
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--allow-model-execution" in result.stderr
    assert not (tmp_path / "outputs").exists()


def test_fresh_judge_needs_explicit_provider_and_model(tmp_path):
    result = subprocess.run([
        sys.executable, str(EVALUATE), "--steps", "evaluate", "--models", "olmo-3-7b-think",
        "--settings", "factual", "--model-eval-dir", str(tmp_path / "outputs"),
        "--allow-model-execution",
    ], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--judge-provider and --judge-model" in result.stderr
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("stage", ["parse", "evaluate"])
def test_local_scoring_uses_default_reference_policy_without_execution_opt_in(tmp_path, monkeypatch, stage):
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.chdir(ROOT)
    monkeypatch.setitem(sys.modules, "memoreason.model_evaluation.model_answer_evaluation_pipeline",
                        SimpleNamespace(generate_parse_and_score_model_answers=pipeline))
    monkeypatch.setitem(sys.modules, "memoreason.model_evaluation.exact_and_judge_match_scoring",
                        SimpleNamespace(JudgeMatchConfiguration=None))
    monkeypatch.setattr(sys, "argv", [
        str(EVALUATE), "--steps", stage, "--models", "olmo-3-7b-think",
        "--settings", "factual", "--model-eval-dir", str(tmp_path / "outputs"), "--skip-judge",
    ])
    spec = importlib.util.spec_from_file_location("evaluation_cli", EVALUATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.main() == 0
    assert len(calls) == 1
    assert calls[0]["steps"] == [stage]
    assert calls[0]["judge_config"] is None
    assert not (tmp_path / "outputs").exists()
