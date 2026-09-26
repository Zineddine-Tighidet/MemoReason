"""Public generation needs no internal manifests or evaluated model outputs."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/dataset.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                          capture_output=True, text=True)


def test_generation_refuses_existing_output(tmp_path):
    result = run("--output", tmp_path)
    assert result.returncode == 2
    assert "must not already exist" in result.stderr


def test_generation_preserves_source_templates():
    destination = ROOT / "data/HUMAN_ANNOTATED_TEMPLATES/invalid-new-output"
    result = run("--output", destination)
    assert result.returncode == 2
    assert "must not be inside" in result.stderr
    assert not destination.exists()


def test_generation_rejects_unknown_document_before_writing(tmp_path):
    destination = tmp_path / "dataset"
    result = run("--output", destination, "--docs", "not_a_benchmark_document")
    assert result.returncode == 2
    assert "Unknown document IDs" in result.stderr
    assert not destination.exists()


def test_generation_rejects_invalid_worker_count(tmp_path):
    destination = tmp_path / "dataset"
    result = run("--output", destination, "--workers", "0")
    assert result.returncode == 2
    assert "must be positive" in result.stderr
    assert not destination.exists()
