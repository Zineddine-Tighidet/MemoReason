"""Serialize the already-computed paper tables, figures, and manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from memoreason.model_evaluation.document_question_answering_prompt import JUDGE_SYSTEM_PROMPT

from .paper_results_data_model import (
    ANSWER_BEHAVIORS,
    MODEL_LABELS,
    QUESTION_COLUMN_ORDER,
    PaperResultsManifest,
)
from .frozen_paper_results_input_loading import sha256_file

__all__ = [
    "latex_escape",
    "render_table2_latex",
    "render_table3_latex",
    "render_table4_latex",
    "rows_as_dicts",
    "write_artifact_manifest",
    "write_csv",
    "write_json",
    "write_text",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"Refusing to write an artifact through a symbolic link: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    if not rows:
        raise ValueError(f"Refusing to write an empty artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_text(path: Path, content: str, *, overwrite: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"Refusing to write an artifact through a symbolic link: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_json(path: Path, payload: Any, *, overwrite: bool = False) -> Path:
    try:
        encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Artifact payload is not strict JSON: {path}") from exc
    return write_text(path, encoded, overwrite=overwrite)


def write_artifact_manifest(
    *,
    paper_results_manifest: PaperResultsManifest,
    artifact_name: str,
    outputs: Sequence[Path],
    metadata: Mapping[str, Any],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    payload = {
        "schema_version": 1,
        "artifact": artifact_name,
        "reporting_code": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "input_manifest": {
            "path": str(paper_results_manifest.manifest_path),
            "sha256": paper_results_manifest.manifest_sha256,
        },
        "evaluated_input_selections": [
            {
                "path": str(selection.path),
                "glob": selection.glob,
                "file_count": selection.file_count,
                "sha256": selection.sha256,
                "expected_sha256": selection.expected_sha256,
            }
            for selection in paper_results_manifest.frozen_evaluated_answer_selections
        ],
        "evaluated_outputs": [
            {
                "path": str(path),
                "sha256": paper_results_manifest.evaluated_output_hashes[str(path)],
            }
            for path in paper_results_manifest.evaluated_output_paths
        ],
        "human_reviewed_templates": {
            "root": str(paper_results_manifest.reviewed_templates_root),
            "sha256": paper_results_manifest.reviewed_templates_tree_sha256,
            "expected_sha256": (paper_results_manifest.expected_reviewed_templates_tree_sha256),
            "question_contract_count": len(paper_results_manifest.reviewed_questions_by_pair_key),
            "authority": "question_type_and_answer_behavior",
        },
        "frozen_judge_cache": (
            None
            if paper_results_manifest.frozen_judge_cache_path is None
            else {
                "path": str(paper_results_manifest.frozen_judge_cache_path),
                "sha256": paper_results_manifest.frozen_judge_cache_sha256,
                "expected_sha256": (paper_results_manifest.expected_frozen_judge_cache_sha256),
            }
        ),
        "judge_contract": {
            "mode": "offline_only",
            "judge_model": paper_results_manifest.judge_model,
            "system_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        },
        "outputs": [
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in outputs
        ],
        "metadata": dict(metadata),
    }
    return write_json(output_path, payload, overwrite=overwrite)


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _table2_cell_is_statistically_significant(value: Any) -> bool:
    """Read the significance flag from computed rows or CSV-loaded rows."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError(
        "Table 2 significant_paired_t_0_05 must be a boolean or the CSV text "
        f"'True'/'False', got {value!r}."
    )


_TABLE2_MODEL_LABEL_LINES = {
    "olmo-3-7b-think": ("OLMO-3", "7B-THINK"),
    "olmo-3-7b-instruct": ("OLMO-3", "7B-INSTRUCT"),
    "gpt-oss-20b-groq": ("GPT-OSS", "20B"),
    "gemma-4-26b-a4b-it": ("GEMMA-4", "26B-A4B-IT"),
    "gpt-oss-120b-groq": ("GPT-OSS", "120B"),
    "qwen3.5-27b": ("QWEN3.5", "27B"),
    "qwen3.5-35b-a3b": ("QWEN3.5", "35B-A3B"),
    "claude-sonnet-4-6": ("CLAUDE", "SONNET 4.6"),
}


def _render_table2_model_label(model: str) -> str:
    lines = _TABLE2_MODEL_LABEL_LINES.get(model)
    if lines is None:
        return rf"\textbf{{{latex_escape(MODEL_LABELS.get(model, model)).upper()}}}"
    return rf"\shortstack[l]{{\textbf{{{lines[0]}}}\\\textbf{{{lines[1]}}}}}"


def _render_table2_score_stack(mean: float, ci: float, *, bold: bool) -> str:
    formatted_mean = f"{mean:+.1f}"
    if bold:
        formatted_mean = rf"\textbf{{{formatted_mean}}}"
    return rf"\shortstack{{{formatted_mean}\\{{\fontsize{{3.6}}{{3.9}}\selectfont $\pm${ci:.1f}}}}}"


def _render_table2_cell(
    row: Mapping[str, Any],
    *,
    answer_behavior: str,
    question_type: str,
    anchor_reason_column_bottom: bool,
) -> str:
    mean = float(row["mean_difference_pp"])
    ci = float(row["ci95_half_width_pp"])
    significant = _table2_cell_is_statistically_significant(row["significant_paired_t_0_05"])
    color = "riseBlueCell" if mean > 0 else "dropPurpleCell"
    content = _render_table2_score_stack(mean, ci, bold=significant)

    if question_type == "reasoning":
        if anchor_reason_column_bottom:
            anchor = f"reason_{answer_behavior}_bottom"
            content = (
                rf"\makebox[1.12cm][c]{{\tikz[remember picture,baseline=({anchor}.base)]"
                rf"\node[inner sep=2.0pt,minimum width=1.12cm] ({anchor}) {{{content}}};}}"
            )
        else:
            content = rf"\makebox[1.12cm][c]{{{content}}}"

    if significant:
        content = (
            rf"\tikz[baseline=(sig.base)]\node[draw=black!80,fill=white!35!{color},"
            "line width=0.28pt,rounded corners=1.1pt,inner sep=1.0pt,outer sep=0pt] "
            rf"(sig) {{{content}}};"
        )
    return rf"\cellcolor{{{color}}}{content}"


def render_table2_latex(rows: Sequence[Mapping[str, Any]], models: Sequence[str]) -> str:
    lookup = {
        (str(row["model_name"]), str(row["answer_behavior"]), str(row["question_type"])): row for row in rows
    }
    lines = [
        r"\definecolor{dropPurpleCell}{HTML}{E7D0FF}",
        r"\definecolor{riseBlueCell}{HTML}{D6E4FF}",
        r"\begin{table}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \renewcommand{\arraystretch}{1.45}",
        r"    \setlength{\tabcolsep}{2.0pt}",
        r"    \caption{Mean Performance Drop (factual $\rightarrow$ fictional - $\%$) by answer and question type. We report aggregated reasoning question performance (i.e. \textit{Arith} for Arithmetic, \textit{Temp} for Temporal, and \textit{Infer} for Inference) in the \textit{Reason} column. Extractive questions are reported in the \textit{Extr} column. Statistically significant results following a paired t-test are framed and indicated in bold. We also report the corresponding 95\% confidence intervals below each value. Results that decrease and increase from factual to fictional are highlighted in {\setlength{\fboxsep}{1pt}\colorbox{dropPurpleCell}{purple}} and {\setlength{\fboxsep}{1pt}\colorbox{riseBlueCell}{blue}} respectively.}",
        r"    \label{tab:question-answer-type-drop}",
        r"    \resizebox{\textwidth}{!}{%",
        r"    \begin{tabular}{@{}l*{5}{c}@{\hspace{8pt}}*{5}{c}@{\hspace{8pt}}*{5}{c}@{}}",
        r"        \toprule",
        r"        \hspace{0.28cm}{\scriptsize\itshape Answer} & \multicolumn{5}{c}{\textbf{Variant}} & \multicolumn{5}{c}{\textbf{Invariant}} & \multicolumn{5}{c}{\textbf{Refusal}} \\",
        r"        \cmidrule(lr){2-6} \cmidrule(lr){7-11} \cmidrule(lr){12-16}",
        r"        \hspace{0.28cm}{\scriptsize\itshape Question} & \textcolor{black!55}{\textbf{Arith.}} & \textcolor{black!55}{\textbf{Temp.}} & \textcolor{black!55}{\textbf{Infer.}} & \makebox[1.12cm][c]{\tikz[remember picture,baseline=(reason_variant_top.base)]\node[inner sep=2.0pt,minimum width=1.12cm] (reason_variant_top) {\textbf{Reason.}};} & \textbf{Extr.} & \textcolor{black!55}{\textbf{Arith.}} & \textcolor{black!55}{\textbf{Temp.}} & \textcolor{black!55}{\textbf{Infer.}} & \makebox[1.12cm][c]{\tikz[remember picture,baseline=(reason_invariant_top.base)]\node[inner sep=2.0pt,minimum width=1.12cm] (reason_invariant_top) {\textbf{Reason.}};} & \textbf{Extr.} & \textcolor{black!55}{\textbf{Arith.}} & \textcolor{black!55}{\textbf{Temp.}} & \textcolor{black!55}{\textbf{Infer.}} & \makebox[1.12cm][c]{\tikz[remember picture,baseline=(reason_refusal_top.base)]\node[inner sep=2.0pt,minimum width=1.12cm] (reason_refusal_top) {\textbf{Reason.}};} & \textbf{Extr.} \\",
        r"        \midrule",
    ]
    for model_index, model in enumerate(models):
        cells: list[str] = []
        for answer_behavior in ANSWER_BEHAVIORS:
            for question_type in QUESTION_COLUMN_ORDER:
                row = lookup[(model, answer_behavior, question_type)]
                cells.append(
                    _render_table2_cell(
                        row,
                        answer_behavior=answer_behavior,
                        question_type=question_type,
                        anchor_reason_column_bottom=model_index == len(models) - 1,
                    )
                )
        lines.append(f"        {_render_table2_model_label(model)} & " + " & ".join(cells) + r" \\")
        if model_index != len(models) - 1:
            lines.append(
                r"        \arrayrulecolor{black!18}\specialrule{0.25pt}{1.0pt}{1.0pt}\arrayrulecolor{black}"
            )
    lines.extend(
        [
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    \begin{tikzpicture}[remember picture,overlay]",
            r"        \draw[black!55,line width=0.26pt,rounded corners=2.8pt] ([xshift=-2.2pt,yshift=1.2pt]reason_variant_top.north west) rectangle ([xshift=2.6pt,yshift=-1.2pt]reason_variant_bottom.south east);",
            r"        \draw[black!55,line width=0.26pt,rounded corners=2.8pt] ([xshift=-2.2pt,yshift=1.2pt]reason_invariant_top.north west) rectangle ([xshift=2.6pt,yshift=-1.2pt]reason_invariant_bottom.south east);",
            r"        \draw[black!55,line width=0.26pt,rounded corners=2.8pt] ([xshift=-2.2pt,yshift=1.2pt]reason_refusal_top.north west) rectangle ([xshift=2.6pt,yshift=-1.2pt]reason_refusal_bottom.south east);",
            r"    \end{tikzpicture}%",
            r"    }",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_table3_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Factual & Fictional & $\Delta$ \\",
        r"\midrule",
    ]
    best_factual = max(float(row["factual_accuracy_pp"]) for row in rows)
    best_fictional = max(float(row["fictional_accuracy_pp"]) for row in rows)
    best_delta = max(float(row["delta_fictional_minus_factual_pp"]) for row in rows)
    for row in rows:
        factual = rf"{float(row['factual_accuracy_pp']):.2f} $\pm$ {float(row['factual_ci95_half_width_pp']):.2f}"
        fictional = rf"{float(row['fictional_accuracy_pp']):.2f} $\pm$ {float(row['fictional_ci95_half_width_pp']):.2f}"
        delta = rf"{float(row['delta_fictional_minus_factual_pp']):+.2f}"
        if np.isclose(float(row["factual_accuracy_pp"]), best_factual):
            factual = rf"\textbf{{{factual}}}"
        if np.isclose(float(row["fictional_accuracy_pp"]), best_fictional):
            fictional = rf"\textbf{{{fictional}}}"
        if np.isclose(float(row["delta_fictional_minus_factual_pp"]), best_delta):
            delta = rf"\textbf{{{delta}}}"
        lines.append(
            f"{latex_escape(MODEL_LABELS.get(str(row['model_name']), str(row['model_name'])))} & "
            f"{factual} & {fictional} & {delta} " + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def render_table4_latex(rows: Sequence[Mapping[str, Any]], models: Sequence[str]) -> str:
    lookup = {(str(row["model_name"]), str(row["question_type"])): row for row in rows}
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Arith. & Temp. & Infer. & Reason. & Extr. \\",
        r"\midrule",
    ]
    for model in models:
        cells: list[str] = []
        for question_type in QUESTION_COLUMN_ORDER:
            row = lookup[(model, question_type)]
            cells.append(rf"{float(row['mean_shortcut_rate_pp']):.1f} ({int(row['total_failcases'])})")
        lines.append(f"{latex_escape(MODEL_LABELS.get(model, model))} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def rows_as_dicts(rows: Iterable[Any]) -> list[dict[str, Any]]:
    return [asdict(row) if hasattr(row, "__dataclass_fields__") else dict(row) for row in rows]
