"""Exclusive writers for MemoReason Tables 1--4 and Figures 3, 5 and 6."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

from memoreason.paper_results.table_1_dataset_statistics import write_table_1_artifacts
from memoreason.paper_results.paper_artifact_serialization import (
    render_table2_latex,
    render_table3_latex,
    render_table4_latex,
    write_artifact_manifest,
    write_csv,
    write_json,
    write_text,
)
from memoreason.paper_results.paper_results_data_model import (
    PAPER_FICTIONAL_VARIANT_IDS,
    PAPER_REPLACEMENT_SETTINGS,
)
from memoreason.paper_results.frozen_paper_results_input_loading import sha256_file

from .tables_1_to_4_and_figures_3_5_6 import (
    TABLE_3_CHAIN_OF_THOUGHT_MODELS,
    FIGURES_3_5_6_MODELS,
    TABLE_4_JUDGE_MODEL,
    TABLES_2_AND_4_MODELS,
    PaperTablesAndFigures,
)


def _assert_targets_available(paths: Sequence[Path], *, overwrite: bool) -> None:
    symbolic_links = [path for path in paths if path.is_symlink()]
    if symbolic_links:
        raise ValueError(f"Refusing paper artifact targets that are symbolic links: {symbolic_links}")
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing paper artifacts: {existing}")


def _write_table_2_performance_drop(
    paper_artifacts: PaperTablesAndFigures,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    paper_results_manifest = paper_artifacts.paper_results_manifest
    csv_path = paper_artifacts.output_dir / "table2_performance_drop.csv"
    tex_path = paper_artifacts.output_dir / "table2_performance_drop.tex"
    manifest_path = paper_artifacts.output_dir / "table2_performance_drop.manifest.json"
    write_csv(csv_path, paper_artifacts.table_2_rows, overwrite=overwrite)
    write_text(
        tex_path,
        render_table2_latex(paper_artifacts.table_2_rows, TABLES_2_AND_4_MODELS),
        overwrite=overwrite,
    )
    write_artifact_manifest(
        paper_results_manifest=paper_results_manifest,
        artifact_name="table2_performance_drop",
        outputs=(csv_path, tex_path),
        metadata={
            "models": list(TABLES_2_AND_4_MODELS),
            "comparison_setting": "fictional",
            "unit_of_analysis": "base_question",
            "fictional_score": "mean_over_expected_variants",
            "test": "two_sided_paired_t_uncorrected",
            "confidence_interval": "paired_t_95",
        },
        output_path=manifest_path,
        overwrite=overwrite,
    )
    return csv_path, tex_path, manifest_path


def _write_table_3_chain_of_thought_comparison(
    paper_artifacts: PaperTablesAndFigures,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    paper_results_manifest = paper_artifacts.paper_results_manifest
    csv_path = paper_artifacts.output_dir / "table3_cot_effect.csv"
    tex_path = paper_artifacts.output_dir / "table3_cot_effect.tex"
    manifest_path = paper_artifacts.output_dir / "table3_cot_effect.manifest.json"
    write_csv(csv_path, paper_artifacts.table_3_rows, overwrite=overwrite)
    write_text(
        tex_path,
        render_table3_latex(paper_artifacts.table_3_rows),
        overwrite=overwrite,
    )
    write_artifact_manifest(
        paper_results_manifest=paper_results_manifest,
        artifact_name="table3_cot_effect",
        outputs=(csv_path, tex_path),
        metadata={
            "models": list(TABLE_3_CHAIN_OF_THOUGHT_MODELS),
            "comparison_setting": "fictional",
            "question_types": ["arithmetic", "temporal", "inference"],
            "unit_of_analysis": "base_question",
            "factual_confidence_interval": "t_95_over_binary_base_questions",
            "fictional_confidence_interval": "t_95_over_per_question_variant_means",
        },
        output_path=manifest_path,
        overwrite=overwrite,
    )
    return csv_path, tex_path, manifest_path


def _write_table_4_parametric_shortcut_rate(
    paper_artifacts: PaperTablesAndFigures,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    paper_results_manifest = paper_artifacts.paper_results_manifest
    csv_path = paper_artifacts.output_dir / "table4_parametric_shortcut_rate.csv"
    tex_path = paper_artifacts.output_dir / "table4_parametric_shortcut_rate.tex"
    diagnostics_path = paper_artifacts.output_dir / "table4_parametric_shortcut_rate.diagnostics.json"
    manifest_path = paper_artifacts.output_dir / "table4_parametric_shortcut_rate.manifest.json"
    write_csv(csv_path, paper_artifacts.table_4_rows, overwrite=overwrite)
    write_text(
        tex_path,
        render_table4_latex(paper_artifacts.table_4_rows, TABLES_2_AND_4_MODELS),
        overwrite=overwrite,
    )
    write_json(diagnostics_path, paper_artifacts.table_4_diagnostics, overwrite=overwrite)
    write_artifact_manifest(
        paper_results_manifest=paper_results_manifest,
        artifact_name="table4_parametric_shortcut_rate",
        outputs=(csv_path, tex_path, diagnostics_path),
        metadata={
            "models": list(TABLES_2_AND_4_MODELS),
            "comparison_setting": "fictional",
            "unit_of_analysis": "macro_mean_of_per_base_question_failure_rates",
            "judge_mode": "offline_only",
            "missing_cache_policy": "error",
        },
        output_path=manifest_path,
        overwrite=overwrite,
    )
    return csv_path, tex_path, diagnostics_path, manifest_path


def _write_figures_3_5_6_partial_replacement_performance(
    paper_artifacts: PaperTablesAndFigures,
    *,
    overwrite: bool,
) -> tuple[Path, ...]:
    # Keep plotting imports lazy so --dry-run does not initialize font or Matplotlib caches.
    from memoreason.paper_results.figures_3_5_6_plot_rendering import build_figure3, build_figure5, build_figure6

    output_dir = paper_artifacts.output_dir
    csv_paths = (
        output_dir / "figure3_performance_by_replacement.csv",
        output_dir / "figure5_performance_by_question_type.csv",
        output_dir / "figure6_performance_by_answer_type.csv",
    )
    stems = (
        output_dir / "figure3_performance_by_replacement",
        output_dir / "figure5_performance_by_question_type",
        output_dir / "figure6_performance_by_answer_type",
    )
    write_csv(csv_paths[0], paper_artifacts.figure_3_rows, overwrite=overwrite)
    write_csv(csv_paths[1], paper_artifacts.figure_5_rows, overwrite=overwrite)
    write_csv(csv_paths[2], paper_artifacts.figure_6_rows, overwrite=overwrite)
    figure3_paths = build_figure3(
        paper_artifacts.figure_3_rows,
        models=FIGURES_3_5_6_MODELS,
        output_stem=stems[0],
    )
    figure5_paths = build_figure5(
        paper_artifacts.figure_5_rows,
        models=FIGURES_3_5_6_MODELS,
        output_stem=stems[1],
    )
    figure6_paths = build_figure6(
        paper_artifacts.figure_6_rows,
        models=FIGURES_3_5_6_MODELS,
        output_stem=stems[2],
    )
    outputs = (*csv_paths, *figure3_paths, *figure5_paths, *figure6_paths)
    manifest_path = output_dir / "figures_3_5_6.manifest.json"
    write_artifact_manifest(
        paper_results_manifest=paper_artifacts.paper_results_manifest,
        artifact_name="figures_3_5_6_partial_replacement",
        outputs=outputs,
        metadata={
            "models": list(FIGURES_3_5_6_MODELS),
            "settings": list(paper_artifacts.paper_results_manifest.settings),
            "renderer_sha256": sha256_file(Path(__file__).with_name("figures_3_5_6_plot_rendering.py")),
            "confidence_interval": "wald_95_over_evaluated_examples",
            "figure3_grouping": "all_questions",
            "figure5_grouping": "question_type",
            "figure6_grouping": "answer_behavior",
        },
        output_path=manifest_path,
        overwrite=overwrite,
    )
    return *outputs, manifest_path


def publish_paper_tables_and_figures(
    paper_artifacts: PaperTablesAndFigures,
    *,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write the complete submitted-paper artifact set after one global preflight."""
    from .tables_1_to_4_and_figures_3_5_6 import expected_table_and_figure_paths

    expected = expected_table_and_figure_paths(paper_artifacts.output_dir)
    _assert_targets_available(expected, overwrite=overwrite)
    table1_paths = write_table_1_artifacts(
        reviewed_templates_selection=paper_artifacts.table_1_reviewed_templates_selection,
        statistics=paper_artifacts.table_1_theme_statistics,
        rows=paper_artifacts.table_1_rows,
        output_dir=paper_artifacts.output_dir,
        overwrite=overwrite,
    )
    table2_paths = _write_table_2_performance_drop(paper_artifacts, overwrite=overwrite)
    table3_paths = _write_table_3_chain_of_thought_comparison(
        paper_artifacts,
        overwrite=overwrite,
    )
    table4_paths = _write_table_4_parametric_shortcut_rate(
        paper_artifacts,
        overwrite=overwrite,
    )
    figure_paths = _write_figures_3_5_6_partial_replacement_performance(
        paper_artifacts,
        overwrite=overwrite,
    )
    generated = (*table1_paths, *table2_paths, *table3_paths, *table4_paths, *figure_paths)
    overall_manifest = paper_artifacts.output_dir / "paper_artifacts.manifest.json"
    overall_payload = {
        "schema_version": 1,
        "artifact_set": "paper_tables_1_4_figures_3_5_6",
        "reporting_code": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                Path(__file__).with_name("tables_1_to_4_and_figures_3_5_6.py").resolve(),
                Path(__file__).with_name("paper_artifact_publication.py").resolve(),
                Path(__file__).with_name("table_1_dataset_statistics.py").resolve(),
                Path(__file__).with_name("figures_3_5_6_plot_rendering.py").resolve(),
            )
        ],
        "input_manifest": {
            "path": str(paper_artifacts.paper_results_manifest.manifest_path),
            "sha256": paper_artifacts.paper_results_manifest.manifest_sha256,
        },
        "evaluated_input_selections": [
            {
                "path": str(selection.path),
                "glob": selection.glob,
                "file_count": selection.file_count,
                "sha256": selection.sha256,
            }
            for selection in paper_artifacts.paper_results_manifest.frozen_evaluated_answer_selections
        ],
        "input_roots": {
            "factual_documents_dir": str(paper_artifacts.input_directories.factual_documents_dir),
            "fictional_documents_dir": str(paper_artifacts.input_directories.fictional_documents_dir),
            "partial_fictional_documents_dir": str(paper_artifacts.input_directories.partial_fictional_documents_dir),
            "model_eval_dir": str(paper_artifacts.input_directories.model_eval_dir),
        },
        "dataset_statistics_tree": {
            "root": str(paper_artifacts.table_1_reviewed_templates_selection.root),
            "sha256": paper_artifacts.table_1_reviewed_templates_selection.tree_sha256,
            "provenance": dict(paper_artifacts.table_1_reviewed_templates_selection.provenance),
        },
        "scope": {
            "tables": [1, 2, 3, 4],
            "figures": [3, 5, 6],
            "table_models": list(TABLES_2_AND_4_MODELS),
            "figure_models": list(FIGURES_3_5_6_MODELS),
            "cot_models": list(TABLE_3_CHAIN_OF_THOUGHT_MODELS),
            "settings": list(PAPER_REPLACEMENT_SETTINGS),
            "variant_ids": list(PAPER_FICTIONAL_VARIANT_IDS),
            "judge_mode": "offline_only",
            "judge_model": TABLE_4_JUDGE_MODEL,
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in generated
        ],
    }
    # Guard against NaN and non-serializable values before the final write.
    json.dumps(overall_payload, allow_nan=False)
    write_json(overall_manifest, overall_payload, overwrite=overwrite)
    return *generated, overall_manifest
