"""Publication-facing dataset statistics with explicit, hashed inputs.

Table 1 is computed from reviewed annotated templates.  The input tree is
selected in the reporting manifest and its deterministic tree digest is
verified before any statistic is computed.  There is deliberately no default
repository data path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from memoreason.paper_results.paper_artifact_serialization import sha256_file, write_csv, write_json, write_text
from memoreason.paper_results.reviewed_template_statistics import (
    Table1ThemeStatistics,
    compute_table_1_theme_statistics,
    render_table_1_latex,
)


PAPER_THEME_IDS = (
    "award_winners",
    "biographies_of_famous_personalities",
    "cities_countries_and_regions",
    "companies_and_organizations",
    "natural_disasters",
    "public_attacks_news_articles",
    "retail_banking_regulations_and_policies",
    "space_missions",
    "sport_events",
)

THEME_LABELS = {
    "award_winners": "Award Winners",
    "biographies_of_famous_personalities": "Biographies",
    "cities_countries_and_regions": "Places",
    "companies_and_organizations": "Companies",
    "natural_disasters": "Natural Disasters",
    "public_attacks_news_articles": "Public Attacks",
    "retail_banking_regulations_and_policies": "Retail Banking",
    "space_missions": "Space Missions",
    "sport_events": "Sport Events",
}


@dataclass(frozen=True)
class Table1ReviewedTemplatesSelection:
    """Frozen Table 1 input selection and publication-scope assertions."""

    manifest_path: Path
    root: Path
    glob: str
    excluded_relative_paths: tuple[str, ...]
    paths: tuple[Path, ...]
    file_hashes: Mapping[str, str]
    tree_sha256: str
    provenance: Mapping[str, Any]
    expected_theme_ids: tuple[str, ...]
    expected_document_count: int
    expected_question_count: int


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(payload).__name__}.")
    return payload


def _resolve_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def compute_reviewed_templates_tree_sha256(root: Path, paths: Sequence[Path]) -> tuple[str, dict[str, str]]:
    """Hash relative paths and file digests in a stable, platform-neutral order."""
    root = root.resolve()
    file_hashes: dict[str, str] = {}
    for path in sorted((candidate.resolve() for candidate in paths), key=lambda item: item.as_posix()):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"Dataset-statistics file is outside its declared root: {path}") from exc
        file_hashes[relative] = sha256_file(path)

    hasher = hashlib.sha256()
    for relative, digest in sorted(file_hashes.items()):
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest(), file_hashes


def load_table_1_reviewed_templates_selection(manifest_path: Path) -> Table1ReviewedTemplatesSelection:
    """Load and verify ``reporting.dataset_statistics`` from one run manifest."""
    manifest_path = manifest_path.expanduser().resolve()
    payload = _read_manifest(manifest_path)
    reporting = payload.get("reporting")
    if not isinstance(reporting, Mapping):
        raise ValueError(f"{manifest_path} has no reporting mapping.")
    raw_spec = reporting.get("dataset_statistics")
    if not isinstance(raw_spec, Mapping):
        raise ValueError("reporting.dataset_statistics must be a mapping.")

    raw_root = str(raw_spec.get("root") or "").strip()
    if not raw_root:
        raise ValueError("reporting.dataset_statistics.root is required; no implicit root is allowed.")
    root = _resolve_path(manifest_path, raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset-statistics root does not exist: {root}")

    pattern = str(raw_spec.get("glob") or "").strip()
    pattern_path = Path(pattern)
    if not pattern or pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError("reporting.dataset_statistics.glob must be a non-empty root-relative pattern.")
    matched_paths = tuple(sorted(candidate.resolve() for candidate in root.glob(pattern) if candidate.is_file()))
    if not matched_paths:
        raise FileNotFoundError(f"No Table 1 inputs match {pattern!r} below {root}.")
    relative_paths: dict[str, Path] = {}
    invalid_layout: list[Path] = []
    for path in matched_paths:
        try:
            relative = path.relative_to(root)
        except ValueError:
            invalid_layout.append(path)
            continue
        if len(relative.parts) != 2 or path.suffix.lower() not in {".yaml", ".yml"}:
            invalid_layout.append(path)
            continue
        relative_paths[relative.as_posix()] = path
    if invalid_layout:
        raise ValueError(
            f"Table 1 expects reviewed templates at <root>/<theme>/<document>.yaml; invalid paths: {invalid_layout[:3]}"
        )

    raw_exclusions = raw_spec.get("exclude_relative_paths", [])
    if not isinstance(raw_exclusions, list):
        raise ValueError("reporting.dataset_statistics.exclude_relative_paths must be a list.")
    excluded_relative_paths: list[str] = []
    for value in raw_exclusions:
        candidate = Path(str(value).strip())
        if (
            not str(value).strip()
            or candidate.is_absolute()
            or ".." in candidate.parts
            or len(candidate.parts) != 2
            or candidate.suffix.lower() not in {".yaml", ".yml"}
        ):
            raise ValueError("Each Table 1 exclusion must be a relative <theme>/<document>.yaml path.")
        excluded_relative_paths.append(candidate.as_posix())
    if len(set(excluded_relative_paths)) != len(excluded_relative_paths):
        raise ValueError("Table 1 exclusions must be distinct.")
    missing_exclusions = sorted(set(excluded_relative_paths) - set(relative_paths))
    if missing_exclusions:
        raise ValueError(f"Table 1 exclusions did not match the selected tree: {missing_exclusions}.")
    excluded = set(excluded_relative_paths)
    paths = tuple(path for relative, path in sorted(relative_paths.items()) if relative not in excluded)
    if not paths:
        raise ValueError("Table 1 exclusions removed every selected template.")

    actual_tree_sha, file_hashes = compute_reviewed_templates_tree_sha256(root, paths)
    expected_tree_sha = str(raw_spec.get("sha256") or "").strip().lower()
    if len(expected_tree_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_tree_sha):
        raise ValueError("reporting.dataset_statistics.sha256 must be an explicit 64-character digest.")
    if actual_tree_sha != expected_tree_sha:
        raise ValueError(
            f"Dataset-statistics tree SHA-256 mismatch for {root}: expected {expected_tree_sha}, got {actual_tree_sha}."
        )

    provenance = raw_spec.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("reporting.dataset_statistics.provenance must be a mapping.")
    for field in ("source", "dataset_revision"):
        if not str(provenance.get(field) or "").strip():
            raise ValueError(f"reporting.dataset_statistics.provenance.{field} is required.")
    try:
        json.dumps(dict(provenance), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Dataset-statistics provenance must be JSON serializable.") from exc

    raw_themes = raw_spec.get("expected_theme_ids")
    if not isinstance(raw_themes, list) or not raw_themes:
        raise ValueError("reporting.dataset_statistics.expected_theme_ids must be a non-empty list.")
    expected_theme_ids = tuple(str(value).strip() for value in raw_themes)
    if any(not value for value in expected_theme_ids) or len(set(expected_theme_ids)) != len(expected_theme_ids):
        raise ValueError("Expected theme ids must be distinct and non-empty.")

    expected_document_count = int(raw_spec.get("expected_document_count") or 0)
    expected_question_count = int(raw_spec.get("expected_question_count") or 0)
    if expected_document_count <= 0 or expected_question_count <= 0:
        raise ValueError("Expected document and question counts must be positive integers.")

    return Table1ReviewedTemplatesSelection(
        manifest_path=manifest_path,
        root=root,
        glob=pattern,
        excluded_relative_paths=tuple(excluded_relative_paths),
        paths=paths,
        file_hashes=file_hashes,
        tree_sha256=actual_tree_sha,
        provenance=dict(provenance),
        expected_theme_ids=expected_theme_ids,
        expected_document_count=expected_document_count,
        expected_question_count=expected_question_count,
    )


def _aggregate_mean_std(rows: Sequence[tuple[int, float, float]]) -> tuple[float, float]:
    total_count = sum(count for count, _mean, _std in rows)
    if total_count <= 0:
        return 0.0, 0.0
    overall_mean = sum(count * mean for count, mean, _std in rows) / total_count
    if total_count == 1:
        return overall_mean, 0.0
    sum_squares = sum(((count - 1) * std**2 if count > 1 else 0.0) + count * mean**2 for count, mean, std in rows)
    variance = max(0.0, (sum_squares - total_count * overall_mean**2) / (total_count - 1))
    return overall_mean, math.sqrt(variance)


def compute_table_1_dataset_statistics(
    reviewed_templates_selection: Table1ReviewedTemplatesSelection,
) -> tuple[list[Table1ThemeStatistics], list[dict[str, Any]]]:
    """Compute Table 1 and enforce the manifest's complete dataset scope."""
    included = {(path.parent.name, path.stem) for path in reviewed_templates_selection.paths}
    statistics = compute_table_1_theme_statistics(
        reviewed_templates_selection.root,
        include_doc_keys=included,
    )
    by_theme = {item.theme_id: item for item in statistics}
    actual_theme_ids = tuple(
        theme_id for theme_id in reviewed_templates_selection.expected_theme_ids if theme_id in by_theme
    )
    missing = sorted(set(reviewed_templates_selection.expected_theme_ids) - set(by_theme))
    unexpected = sorted(set(by_theme) - set(reviewed_templates_selection.expected_theme_ids))
    if missing or unexpected or actual_theme_ids != reviewed_templates_selection.expected_theme_ids:
        raise ValueError(
            "Table 1 theme scope mismatch: "
            f"missing={missing}, unexpected={unexpected}, "
            f"expected_order={reviewed_templates_selection.expected_theme_ids}."
        )

    ordered = [by_theme[theme_id] for theme_id in reviewed_templates_selection.expected_theme_ids]
    document_count = sum(item.documents_count for item in ordered)
    question_counts = [round(item.documents_count * item.questions_avg) for item in ordered]
    question_count = sum(question_counts)
    if document_count != reviewed_templates_selection.expected_document_count:
        raise ValueError(
            "Table 1 document scope mismatch: "
            f"expected {reviewed_templates_selection.expected_document_count}, "
            f"got {document_count}."
        )
    if question_count != reviewed_templates_selection.expected_question_count:
        raise ValueError(
            "Table 1 question scope mismatch: "
            f"expected {reviewed_templates_selection.expected_question_count}, "
            f"got {question_count}."
        )

    entity_total = _aggregate_mean_std(
        [(item.documents_count, item.entities_avg, item.entities_std) for item in ordered]
    )
    rule_total = _aggregate_mean_std([(item.documents_count, item.rules_avg, item.rules_std) for item in ordered])
    rows: list[dict[str, Any]] = []
    for item, theme_questions in zip(ordered, question_counts, strict=True):
        rows.append(
            {
                "theme_id": item.theme_id,
                "theme_label": THEME_LABELS.get(item.theme_id, item.theme_id.replace("_", " ").title()),
                "documents_count": item.documents_count,
                "questions_count": theme_questions,
                "questions_share_percent": 100.0 * theme_questions / question_count,
                "entities_mean": item.entities_avg,
                "entities_std": item.entities_std,
                "rules_mean": item.rules_avg,
                "rules_std": item.rules_std,
            }
        )
    rows.append(
        {
            "theme_id": "total",
            "theme_label": "Total",
            "documents_count": document_count,
            "questions_count": question_count,
            "questions_share_percent": 100.0,
            "entities_mean": entity_total[0],
            "entities_std": entity_total[1],
            "rules_mean": rule_total[0],
            "rules_std": rule_total[1],
        }
    )
    return ordered, rows


def table_1_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "table1_dataset_statistics.csv",
        output_dir / "table1_dataset_statistics.tex",
        output_dir / "table1_dataset_statistics.manifest.json",
    )


def write_table_1_artifacts(
    *,
    reviewed_templates_selection: Table1ReviewedTemplatesSelection,
    statistics: Sequence[Table1ThemeStatistics],
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, Path, Path]:
    """Write Table 1 CSV, LaTeX and a complete provenance manifest."""
    csv_path, tex_path, manifest_path = table_1_output_paths(output_dir)
    if not overwrite:
        existing = [path for path in (csv_path, tex_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing Table 1 artifacts: {existing}")
    write_csv(csv_path, rows, overwrite=overwrite)
    write_text(
        tex_path,
        render_table_1_latex(list(statistics)),
        overwrite=overwrite,
    )
    payload = {
        "schema_version": 1,
        "artifact": "table1_dataset_statistics",
        "reporting_code": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                Path(__file__).resolve(),
                Path(render_table_1_latex.__code__.co_filename).resolve(),
            )
        ],
        "input_manifest": {
            "path": str(reviewed_templates_selection.manifest_path),
            "sha256": sha256_file(reviewed_templates_selection.manifest_path),
        },
        "dataset_statistics_input": {
            "root": str(reviewed_templates_selection.root),
            "glob": reviewed_templates_selection.glob,
            "exclude_relative_paths": list(reviewed_templates_selection.excluded_relative_paths),
            "file_count": len(reviewed_templates_selection.paths),
            "tree_sha256": reviewed_templates_selection.tree_sha256,
            "provenance": dict(reviewed_templates_selection.provenance),
        },
        "scope": {
            "theme_ids": list(reviewed_templates_selection.expected_theme_ids),
            "document_count": reviewed_templates_selection.expected_document_count,
            "question_count": reviewed_templates_selection.expected_question_count,
            "unit_of_analysis": "reviewed_document_template",
            "standard_deviation": "sample",
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (csv_path, tex_path)
        ],
        "statistics": [asdict(item) for item in statistics],
    }
    write_json(manifest_path, payload, overwrite=overwrite)
    return csv_path, tex_path, manifest_path
