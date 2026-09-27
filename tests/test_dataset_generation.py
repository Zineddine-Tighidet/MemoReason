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


def test_generation_progress_only_and_worker_count_preserves_output(tmp_path):
    outputs = []
    for workers in (1, 2):
        destination = tmp_path / f"workers-{workers}"
        result = run(
            "--output", destination,
            "--docs", "company_01", "company_02",
            "--settings", "factual", "--workers", workers,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == ""
        progress_lines = [line for line in result.stderr.splitlines() if line.strip()]
        assert progress_lines
        assert all(line.startswith("Generating:") for line in progress_lines)
        assert "100%" in progress_lines[-1]
        assert "2/2" in progress_lines[-1]
        files = {
            str(path.relative_to(destination)): path.read_bytes()
            for path in destination.rglob("*.yaml")
        }
        assert len(files) == 2
        outputs.append(files)
    assert outputs[0] == outputs[1]
