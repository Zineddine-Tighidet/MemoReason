#!/usr/bin/env python3
"""Build a LaTeX table for question-type by answer-type performance drops."""

from __future__ import annotations

import argparse
import csv
from math import isfinite
from pathlib import Path

from scipy.stats import t


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "MODEL_EVAL"
    / "PLOTS"
    / "performance_drop_ttest"
    / "paper_figure3_question_answer_type_drop_heatmap_bonferroni.csv"
)
DEFAULT_OUTPUT_TEX = (
    PROJECT_ROOT
    / "data"
    / "MODEL_EVAL"
    / "PLOTS"
    / "performance_drop_ttest"
    / "question_answer_type_drop_table.tex"
)

MODEL_ORDER = (
    "olmo-3-7b-think",
    "olmo-3-7b-instruct",
    "gpt-oss-20b-groq",
    "gemma-4-26b-a4b-it",
    "gpt-oss-120b-groq",
    "qwen3.5-27b",
    "qwen3.5-35b-a3b",
    "claude-sonnet-4-6",
)
MODEL_LABELS = {
    "olmo-3-7b-think": ("OLMO-3", "7B-THINK"),
    "olmo-3-7b-instruct": ("OLMO-3", "7B-INSTRUCT"),
    "gpt-oss-20b-groq": ("GPT-OSS", "20B"),
    "gemma-4-26b-a4b-it": ("GEMMA-4", "26B-A4B-IT"),
    "gpt-oss-120b-groq": ("GPT-OSS", "120B"),
    "qwen3.5-27b": ("QWEN3.5", "27B"),
    "qwen3.5-35b-a3b": ("QWEN3.5", "35B-A3B"),
    "claude-sonnet-4-6": ("CLAUDE", "SONNET 4.6"),
}
ANSWER_GROUPS = (
    ("variant", "Variant"),
    ("invariant", "Invariant"),
    ("refusal", "Refusal"),
)
QUESTION_COLUMNS = (
    ("arithmetic", "Arith."),
    ("temporal", "Temp."),
    ("inference", "Infer."),
    ("reasoning", "Reason."),
    ("extractive", "Extr."),
)
MUTED_QUESTION_TYPES = {"arithmetic", "temporal", "inference"}
ANSWER_BLOCK_GAP = r"@{\hspace{8pt}}"
REASON_NODE_MIN_WIDTH = "1.12cm"
REASON_BACKGROUND_MIN_WIDTH = "1.12cm"
DIRECTION_NEGATIVE_COLOR = "dropPurpleCell"
DIRECTION_POSITIVE_COLOR = "riseBlueCell"
DROP_PURPLE_CELL_HEX = "E7D0FF"
RISE_BLUE_CELL_HEX = "D6E4FF"
SIGNIFICANCE_FRAME_COLOR = "black!80"
SIGNIFICANCE_FRAME_FILL_PREFIX = "white!35"
SIGNIFICANCE_FRAME_SEP = "1.0pt"
SIGNIFICANCE_FRAME_RULE = "0.28pt"
SIGNIFICANCE_FRAME_RADIUS = "1.1pt"
CI_FONT_SIZE = "3.6"
CI_BASELINE_SKIP = "3.9"


def _model_label(model_name: str) -> str:
    label = MODEL_LABELS.get(model_name)
    if label is None:
        return r"\textbf{" + _escape_latex(model_name.replace("-", " ").upper()) + "}"
    return r"\shortstack[l]{\textbf{" + r"}\\\textbf{".join(_escape_latex(part) for part in label) + "}}"


def _escape_latex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _format_pp(value: str) -> str:
    numeric = float(value)
    return f"{numeric:+.1f}"


def _table_node(content: str, node_name: str, *, min_width: str, inner_sep: str = "2.0pt") -> str:
    return (
        rf"\tikz[remember picture,baseline=({node_name}.base)]"
        rf"\node[inner sep={inner_sep},minimum width={min_width}] ({node_name}) {{{content}}};"
    )


def _makebox(content: str, width: str) -> str:
    return rf"\makebox[{width}][c]{{{content}}}"


def _paired_t_ci_half_width_pp(row: dict[str, str]) -> float:
    alpha = float(row.get("significance_alpha") or 0.05)
    degrees_of_freedom = int(float(row["degrees_of_freedom"]))
    standard_error_pp = float(row["standard_error_difference"]) * 100.0
    if degrees_of_freedom <= 0 or standard_error_pp <= 0.0 or not isfinite(standard_error_pp):
        return 0.0
    critical_value = float(t.ppf(1.0 - alpha / 2.0, df=degrees_of_freedom))
    return critical_value * standard_error_pp


def _paired_t_p_value(row: dict[str, str]) -> float:
    degrees_of_freedom = int(float(row["degrees_of_freedom"]))
    t_statistic = float(row["t_statistic"])
    if degrees_of_freedom <= 0 or not isfinite(t_statistic):
        return 1.0
    return 2.0 * float(t.sf(abs(t_statistic), df=degrees_of_freedom))


def _is_significant_with_paired_t_test(row: dict[str, str]) -> bool:
    alpha = float(row.get("significance_alpha") or 0.05)
    return _paired_t_p_value(row) < alpha


def _ci_text(ci_half_width: float) -> str:
    return rf"{{\fontsize{{{CI_FONT_SIZE}}}{{{CI_BASELINE_SKIP}}}\selectfont $\pm${ci_half_width:.1f}}}"


def _significance_frame(content: str, background_color: str) -> str:
    fill_color = "white" if background_color == "white" else rf"{SIGNIFICANCE_FRAME_FILL_PREFIX}!{background_color}"
    return (
        r"\tikz[baseline=(sig.base)]"
        rf"\node[draw={SIGNIFICANCE_FRAME_COLOR},"
        rf"fill={fill_color},"
        rf"line width={SIGNIFICANCE_FRAME_RULE},"
        rf"rounded corners={SIGNIFICANCE_FRAME_RADIUS},"
        rf"inner sep={SIGNIFICANCE_FRAME_SEP},"
        r"outer sep=0pt] (sig) "
        rf"{{{content}}};"
    )


def _direction_background_color(mean_value: float) -> str:
    if mean_value < 0.0:
        return DIRECTION_NEGATIVE_COLOR
    if mean_value > 0.0:
        return DIRECTION_POSITIVE_COLOR
    return "white"


def _format_cell(
    row: dict[str, str],
    question_type: str,
    *,
    is_reason: bool = False,
    node_name: str | None = None,
) -> str:
    mean_value = float(row["mean_difference_pp"])
    mean = _format_pp(row["mean_difference_pp"])
    ci_half_width = _paired_t_ci_half_width_pp(row)
    significant = _is_significant_with_paired_t_test(row)
    if significant:
        mean = rf"\textbf{{{mean}}}"
    ci = _ci_text(ci_half_width)
    content = rf"\shortstack{{{mean}\\{ci}}}"
    if node_name is not None:
        content = _table_node(content, node_name, min_width=REASON_NODE_MIN_WIDTH)
    if is_reason:
        content = _makebox(content, REASON_BACKGROUND_MIN_WIDTH)
    background_color = _direction_background_color(mean_value)
    if significant:
        content = _significance_frame(content, background_color)
    return rf"\cellcolor{{{background_color}}}{content}"


def _load_rows(input_csv: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["model_name"], row["answer_behavior"], row["question_type"])
        lookup[key] = row
    return lookup


def build_table(input_csv: Path, output_tex: Path) -> None:
    lookup = _load_rows(input_csv)
    missing = [
        (model_name, answer_key, question_key)
        for model_name in MODEL_ORDER
        for answer_key, _answer_label in ANSWER_GROUPS
        for question_key, _question_label in QUESTION_COLUMNS
        if (model_name, answer_key, question_key) not in lookup
    ]
    if missing:
        formatted = "\n".join(f"- {model}/{answer}/{question}" for model, answer, question in missing)
        raise RuntimeError(f"Missing table cells in {input_csv}:\n{formatted}")

    group_header = [r"        \hspace{0.28cm}{\scriptsize\itshape Answer Type}"]
    cmidrules = []
    start_column = 2
    for _answer_key, answer_label in ANSWER_GROUPS:
        group_header.append(rf"\multicolumn{{5}}{{c}}{{\textbf{{{answer_label}}}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start_column}-{start_column + 4}}}")
        start_column += 5

    question_header = [r"        \hspace{0.28cm}{\scriptsize\itshape Question Type}"]
    for answer_key, _answer_label in ANSWER_GROUPS:
        for question_key, label in QUESTION_COLUMNS:
            header = rf"\textbf{{{label}}}"
            if question_key in MUTED_QUESTION_TYPES:
                header = rf"\textcolor{{black!55}}{{{header}}}"
            if question_key == "reasoning":
                header = _table_node(header, f"reason_{answer_key}_top", min_width=REASON_NODE_MIN_WIDTH)
                header = _makebox(header, REASON_BACKGROUND_MIN_WIDTH)
            question_header.append(header)

    lines = [
        rf"\definecolor{{{DIRECTION_NEGATIVE_COLOR}}}{{HTML}}{{{DROP_PURPLE_CELL_HEX}}}",
        rf"\definecolor{{{DIRECTION_POSITIVE_COLOR}}}{{HTML}}{{{RISE_BLUE_CELL_HEX}}}",
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \scriptsize",
        r"    \renewcommand{\arraystretch}{1.45}",
        r"    \setlength{\tabcolsep}{2.0pt}",
        r"    \caption{Mean Performance Drop (factual $\rightarrow$ fictional -- \%) by answer and question type. We report aggregated reasoning questions performance (i.e. \textit{Arith.} for Arithmetic, \textit{Temp.} for Temporal, and \textit{Infer.} for Inference) in the framed \textit{Reason.} column. Extractive questions are reported in the \textit{Extr.} column. Statistically significant results following a paired t-test are framed and indicated in bold. We report the corresponding 95\% confidence interval half-width below each value. Results that decrease and increase from factual to fictional are highlighted in {\setlength{\fboxsep}{1pt}\colorbox{dropPurpleCell}{purple}} and {\setlength{\fboxsep}{1pt}\colorbox{riseBlueCell}{blue}}, respectively.}",
        r"    \label{tab:question-answer-type-drop}",
        r"    \resizebox{\textwidth}{!}{%",
        rf"    \begin{{tabular}}{{@{{}}l*{{5}}{{c}}{ANSWER_BLOCK_GAP}*{{5}}{{c}}{ANSWER_BLOCK_GAP}*{{5}}{{c}}@{{}}}}",
        r"        \toprule",
        " & ".join(group_header) + r" \\",
        "        " + " ".join(cmidrules),
        " & ".join(question_header) + r" \\",
        r"        \midrule",
    ]

    for model_index, model_name in enumerate(MODEL_ORDER):
        row_cells = [_model_label(model_name)]
        for answer_key, _answer_label in ANSWER_GROUPS:
            for question_key, _question_label in QUESTION_COLUMNS:
                node_name = None
                if question_key == "reasoning" and model_index == len(MODEL_ORDER) - 1:
                    node_name = f"reason_{answer_key}_bottom"
                row_cells.append(
                    _format_cell(
                        lookup[(model_name, answer_key, question_key)],
                        question_key,
                        is_reason=question_key == "reasoning",
                        node_name=node_name,
                    )
                )
        lines.append("        " + " & ".join(row_cells) + r" \\")
        if model_index != len(MODEL_ORDER) - 1:
            lines.append(r"        \arrayrulecolor{black!18}\specialrule{0.25pt}{1.0pt}{1.0pt}\arrayrulecolor{black}")

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
            r"\end{table*}",
            "",
        ]
    )

    output_tex.parent.mkdir(parents=True, exist_ok=True)
    output_tex.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-tex", type=Path, default=DEFAULT_OUTPUT_TEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_table(args.input_csv, args.output_tex)
    print(f"wrote {args.output_tex}")


if __name__ == "__main__":
    main()
