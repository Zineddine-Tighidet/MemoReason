"""One strict, manifest-driven build for paper Tables 1--4 and Figures 3/5/6."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from memoreason.paper_results.table_1_dataset_statistics import (
    Table1ReviewedTemplatesSelection,
    PAPER_THEME_IDS,
    compute_table_1_dataset_statistics,
    load_table_1_reviewed_templates_selection,
    table_1_output_paths,
)
from memoreason.paper_results.paper_results_data_model import (
    PAPER_REPLACEMENT_SETTINGS,
    PAPER_FICTIONAL_VARIANT_IDS,
    PaperResultsManifest,
    EvaluatedAnswer,
)
from memoreason.paper_results.evaluated_model_answer_loading import load_evaluated_answers
from memoreason.paper_results.figures_3_5_6_partial_replacement_performance import (
    compute_partial_replacement_accuracy_curves,
)
from memoreason.paper_results.frozen_paper_results_input_loading import (
    load_paper_results_manifest,
)
from memoreason.paper_results.tables_2_and_3_factual_to_fictional_performance import (
    compute_table_2_performance_drop,
    compute_table_3_chain_of_thought_comparison,
)
from memoreason.paper_results.table_4_parametric_shortcut_rate import (
    compute_table_4_parametric_shortcut_rate,
)
from memoreason.paper_results.reviewed_template_statistics import Table1ThemeStatistics


TABLES_2_AND_4_MODELS = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "claude-sonnet-4-6",
)
FIGURES_3_5_6_MODELS = (
    "olmo-3-7b-instruct",
    "gpt-oss-120b-groq",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
)
TABLE_3_CHAIN_OF_THOUGHT_MODELS = ("olmo-3-7b-instruct", "olmo-3-7b-think")
TABLE_4_JUDGE_MODEL = "openai/gpt-oss-120b"


@dataclass(frozen=True)
class PaperResultsInputDirectories:
    """Explicit data locations recorded even when an artifact reads only scores."""

    factual_documents_dir: Path
    fictional_documents_dir: Path
    partial_fictional_documents_dir: Path
    model_eval_dir: Path


@dataclass(frozen=True)
class PaperTablesAndFigures:
    """All validated inputs and computed rows required by the paper artifacts."""

    paper_results_manifest: PaperResultsManifest
    table_1_reviewed_templates_selection: Table1ReviewedTemplatesSelection
    input_directories: PaperResultsInputDirectories
    output_dir: Path
    evaluated_answers: tuple[EvaluatedAnswer, ...]
    table_1_theme_statistics: tuple[Table1ThemeStatistics, ...]
    table_1_rows: tuple[Mapping[str, Any], ...]
    table_2_rows: tuple[Mapping[str, Any], ...]
    table_3_rows: tuple[Mapping[str, Any], ...]
    table_4_rows: tuple[Mapping[str, Any], ...]
    table_4_diagnostics: Mapping[str, Any]
    figure_3_rows: tuple[Mapping[str, Any], ...]
    figure_5_rows: tuple[Mapping[str, Any], ...]
    figure_6_rows: tuple[Mapping[str, Any], ...]


__all__ = [
    "TABLE_3_CHAIN_OF_THOUGHT_MODELS",
    "FIGURES_3_5_6_MODELS",
    "TABLE_4_JUDGE_MODEL",
    "TABLES_2_AND_4_MODELS",
    "PaperTablesAndFigures",
    "PaperResultsInputDirectories",
    "expected_table_and_figure_paths",
    "load_paper_results_input_directories",
    "compute_paper_tables_and_figures",
    "validate_paper_output_directory",
    "validate_paper_results_scope",
    "publish_paper_tables_and_figures",
]


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _resolve_manifest_path(manifest_path: Path, value: Any, *, key: str) -> Path:
    if isinstance(value, Mapping):
        raw_path = str(value.get("path") or "").strip()
    else:
        raw_path = str(value or "").strip()
    if not raw_path:
        raise ValueError(f"reporting.input_roots.{key} is required; no implicit path is allowed.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Configured {key} does not exist: {path}")
    return path


def load_paper_results_input_directories(manifest_path: Path) -> PaperResultsInputDirectories:
    """Load the canonical JZ/local roots without assuming an on-disk layout."""
    manifest_path = manifest_path.expanduser().resolve()
    payload = _read_yaml(manifest_path)
    reporting = payload.get("reporting")
    if not isinstance(reporting, Mapping):
        raise ValueError(f"{manifest_path} has no reporting mapping.")
    roots = reporting.get("input_roots")
    if not isinstance(roots, Mapping):
        raise ValueError("reporting.input_roots must be a mapping.")
    return PaperResultsInputDirectories(
        factual_documents_dir=_resolve_manifest_path(
            manifest_path, roots.get("factual_documents_dir"), key="factual_documents_dir"
        ),
        fictional_documents_dir=_resolve_manifest_path(
            manifest_path, roots.get("fictional_documents_dir"), key="fictional_documents_dir"
        ),
        partial_fictional_documents_dir=_resolve_manifest_path(
            manifest_path,
            roots.get("partial_fictional_documents_dir"),
            key="partial_fictional_documents_dir",
        ),
        model_eval_dir=_resolve_manifest_path(manifest_path, roots.get("model_eval_dir"), key="model_eval_dir"),
    )


def validate_paper_results_scope(
    paper_results_manifest: PaperResultsManifest,
    table_1_reviewed_templates_selection: Table1ReviewedTemplatesSelection,
    input_directories: PaperResultsInputDirectories,
) -> None:
    """Reject silent changes to the submitted paper's settings and model panels."""
    expected_models = {
        "table2": TABLES_2_AND_4_MODELS,
        "table4": TABLES_2_AND_4_MODELS,
        "figures": FIGURES_3_5_6_MODELS,
        "table3": TABLE_3_CHAIN_OF_THOUGHT_MODELS,
    }
    for group, expected in expected_models.items():
        actual = paper_results_manifest.models.get(group)
        if actual != expected:
            raise ValueError(f"Paper scope mismatch for reporting.models.{group}: expected {expected}, got {actual}.")
    if paper_results_manifest.settings != PAPER_REPLACEMENT_SETTINGS:
        raise ValueError(
            f"Paper scope mismatch for replacement settings: expected {PAPER_REPLACEMENT_SETTINGS}, got {paper_results_manifest.settings}."
        )
    if paper_results_manifest.variant_ids != PAPER_FICTIONAL_VARIANT_IDS:
        raise ValueError(
            f"Paper scope mismatch for variant ids: expected {PAPER_FICTIONAL_VARIANT_IDS}, got {paper_results_manifest.variant_ids}."
        )
    if not paper_results_manifest.frozen_evaluated_answer_selections:
        raise ValueError("Paper scope requires manifest-declared evaluated-input selections with SHA-256 digests.")
    unpinned_selections = [
        selection.path
        for selection in paper_results_manifest.frozen_evaluated_answer_selections
        if selection.expected_sha256 is None
    ]
    if unpinned_selections:
        raise ValueError(
            "Paper scope requires a sha256 for every reporting.evaluated_outputs entry; "
            f"unpinned selections: {unpinned_selections}."
        )
    if paper_results_manifest.judge_model != TABLE_4_JUDGE_MODEL:
        raise ValueError(
            f"Paper scope requires judge model {TABLE_4_JUDGE_MODEL!r}, got {paper_results_manifest.judge_model!r}."
        )
    if (
        paper_results_manifest.frozen_judge_cache_path is not None
        and paper_results_manifest.expected_frozen_judge_cache_sha256 is None
    ):
        raise ValueError("Paper scope requires reporting.frozen_judge_cache.sha256 when a cache is configured.")
    if table_1_reviewed_templates_selection.expected_theme_ids != PAPER_THEME_IDS:
        raise ValueError(
            f"Paper scope mismatch for Table 1 themes: expected {PAPER_THEME_IDS}, "
            f"got {table_1_reviewed_templates_selection.expected_theme_ids}."
        )
    if (
        table_1_reviewed_templates_selection.expected_document_count != 87
        or table_1_reviewed_templates_selection.expected_question_count != 1044
    ):
        raise ValueError(
            "Paper Table 1 scope must declare 87 document templates and 1044 question templates; "
            f"got {table_1_reviewed_templates_selection.expected_document_count} and {table_1_reviewed_templates_selection.expected_question_count}."
        )

    outside_eval_root = []
    for path in paper_results_manifest.evaluated_output_paths:
        try:
            path.relative_to(input_directories.model_eval_dir)
        except ValueError:
            outside_eval_root.append(path)
    if outside_eval_root:
        raise ValueError(
            "Evaluated outputs must be selected below reporting.input_roots.model_eval_dir; "
            f"outside paths: {outside_eval_root[:3]}"
        )


def validate_paper_output_directory(
    output_dir: Path,
    table_1_reviewed_templates_selection: Table1ReviewedTemplatesSelection,
    input_directories: PaperResultsInputDirectories,
) -> None:
    """Keep publication outputs disjoint from every immutable input tree."""
    output_dir = output_dir.expanduser().resolve()
    immutable_roots = (
        table_1_reviewed_templates_selection.root,
        input_directories.factual_documents_dir,
        input_directories.fictional_documents_dir,
        input_directories.partial_fictional_documents_dir,
        input_directories.model_eval_dir,
    )
    overlaps = [
        root
        for root in immutable_roots
        if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir)
    ]
    if overlaps:
        raise ValueError(
            "reporting.output_dir must be disjoint from all immutable input roots; "
            f"output={output_dir}, overlaps={overlaps}."
        )


def expected_table_and_figure_paths(output_dir: Path) -> tuple[Path, ...]:
    table1 = table_1_output_paths(output_dir)
    table2 = (
        output_dir / "table2_performance_drop.csv",
        output_dir / "table2_performance_drop.tex",
        output_dir / "table2_performance_drop.manifest.json",
    )
    table3 = (
        output_dir / "table3_cot_effect.csv",
        output_dir / "table3_cot_effect.tex",
        output_dir / "table3_cot_effect.manifest.json",
    )
    table4 = (
        output_dir / "table4_parametric_shortcut_rate.csv",
        output_dir / "table4_parametric_shortcut_rate.tex",
        output_dir / "table4_parametric_shortcut_rate.diagnostics.json",
        output_dir / "table4_parametric_shortcut_rate.manifest.json",
    )
    figure_csvs = (
        output_dir / "figure3_performance_by_replacement.csv",
        output_dir / "figure5_performance_by_question_type.csv",
        output_dir / "figure6_performance_by_answer_type.csv",
    )
    figure_plots = tuple(
        output_dir / f"figure{figure}_{stem}.{suffix}"
        for figure, stem in (
            (3, "performance_by_replacement"),
            (5, "performance_by_question_type"),
            (6, "performance_by_answer_type"),
        )
        for suffix in ("png", "pdf")
    )
    return (
        *table1,
        *table2,
        *table3,
        *table4,
        *figure_csvs,
        *figure_plots,
        output_dir / "figures_3_5_6.manifest.json",
        output_dir / "paper_artifacts.manifest.json",
    )


def compute_paper_tables_and_figures(
    manifest_path: Path,
    *,
    output_dir: Path | None = None,
) -> PaperTablesAndFigures:
    """Load every input and compute every table/curve without writing files."""
    paper_results_manifest = load_paper_results_manifest(manifest_path)
    table_1_reviewed_templates_selection = load_table_1_reviewed_templates_selection(manifest_path)
    input_directories = load_paper_results_input_directories(manifest_path)
    validate_paper_results_scope(paper_results_manifest, table_1_reviewed_templates_selection, input_directories)
    resolved_output_dir = paper_results_manifest.output_dir if output_dir is None else output_dir.expanduser().resolve()
    validate_paper_output_directory(resolved_output_dir, table_1_reviewed_templates_selection, input_directories)
    evaluated_answers = tuple(load_evaluated_answers(paper_results_manifest))
    table_1_theme_statistics, table_1_rows = compute_table_1_dataset_statistics(table_1_reviewed_templates_selection)
    table_2_rows = compute_table_2_performance_drop(
        evaluated_answers,
        models=TABLES_2_AND_4_MODELS,
        comparison_setting="fictional",
        variant_ids=paper_results_manifest.variant_ids,
    )
    table_3_rows = compute_table_3_chain_of_thought_comparison(
        evaluated_answers,
        models=TABLE_3_CHAIN_OF_THOUGHT_MODELS,
        comparison_setting="fictional",
        variant_ids=paper_results_manifest.variant_ids,
    )
    table_4_rows, table_4_diagnostics = compute_table_4_parametric_shortcut_rate(
        evaluated_answers,
        paper_results_manifest=paper_results_manifest,
        models=TABLES_2_AND_4_MODELS,
        comparison_setting="fictional",
        missing_cache_policy="error",
    )
    figure_3_rows = compute_partial_replacement_accuracy_curves(
        evaluated_answers,
        models=FIGURES_3_5_6_MODELS,
        settings=paper_results_manifest.settings,
        variant_ids=paper_results_manifest.variant_ids,
    )
    figure_5_rows = compute_partial_replacement_accuracy_curves(
        evaluated_answers,
        models=FIGURES_3_5_6_MODELS,
        settings=paper_results_manifest.settings,
        group_by="question_type",
        variant_ids=paper_results_manifest.variant_ids,
    )
    figure_6_rows = compute_partial_replacement_accuracy_curves(
        evaluated_answers,
        models=FIGURES_3_5_6_MODELS,
        settings=paper_results_manifest.settings,
        group_by="answer_behavior",
        variant_ids=paper_results_manifest.variant_ids,
    )
    return PaperTablesAndFigures(
        paper_results_manifest=paper_results_manifest,
        table_1_reviewed_templates_selection=table_1_reviewed_templates_selection,
        input_directories=input_directories,
        output_dir=resolved_output_dir,
        evaluated_answers=evaluated_answers,
        table_1_theme_statistics=tuple(table_1_theme_statistics),
        table_1_rows=tuple(table_1_rows),
        table_2_rows=tuple(table_2_rows),
        table_3_rows=tuple(table_3_rows),
        table_4_rows=tuple(table_4_rows),
        table_4_diagnostics=table_4_diagnostics,
        figure_3_rows=tuple(figure_3_rows),
        figure_5_rows=tuple(figure_5_rows),
        figure_6_rows=tuple(figure_6_rows),
    )


def publish_paper_tables_and_figures(
    plan: PaperTablesAndFigures,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the complete artifact set after the shared publication preflight."""
    from .paper_artifact_publication import publish_paper_tables_and_figures as write_validated_artifacts

    return write_validated_artifacts(plan, overwrite=overwrite)
