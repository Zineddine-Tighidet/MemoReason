"""Compute Table 1 statistics from the reviewed template dataset."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import math
from pathlib import Path
from statistics import mean, stdev

import yaml

from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.fictional_entity_sampler import (
    FictionalEntitySampler,
)
from memoreason.factual_to_fictional_dataset.controlled_entity_replacement.generation_requirements import (
    extract_required_entities,
)
from memoreason.benchmark_definition.implicit_numeric_rules import normalize_implicit_rules_for_storage
from memoreason.benchmark_definition.document_schema import AnnotatedDocument, ImplicitRule, Question
from memoreason.benchmark_definition.annotation_runtime import (
    AnnotationParser,
    normalize_document_taxonomy,
    normalize_rule_expressions,
)


def _strip_rule_comment(rule: str) -> str:
    return str(rule or "").split("#", 1)[0].strip()


def _non_empty_rules(rules: list[str] | None) -> list[str]:
    return [rule for rule in (rules or []) if _strip_rule_comment(rule)]


def _sorted_ids_by_value(values_by_id: dict[str, float | int]) -> list[str]:
    return [item_id for item_id, _value in sorted(values_by_id.items(), key=lambda item: (item[1], item[0]))]


def _minimal_order_edges(values_by_id: dict[str, float | int]) -> int:
    if len(values_by_id) <= 1:
        return 0
    ordered_ids = _sorted_ids_by_value(values_by_id)
    edges = 0
    for left_id, right_id in pairwise(ordered_ids):
        if values_by_id[left_id] == values_by_id[right_id]:
            continue
        edges += 1
    return edges


def _count_required_entities(document: AnnotatedDocument) -> int:
    required = extract_required_entities(document, include_questions=True)
    return sum(len(specs) for specs in required.values())


def _count_implicit_order_rules(document: AnnotatedDocument) -> int:
    required = extract_required_entities(document, include_questions=True)
    factual_entities = AnnotationParser.extract_factual_entities(document, include_questions=True)

    # Number ordering: reuse generation logic (already non-redundant adjacency constraints).
    sampler = FictionalEntitySampler(entity_pool={}, factual_entities=factual_entities)
    number_order_count = len(sampler._build_number_ordering_rules(required.get("number", [])))

    # Temporal ordering: minimal chain over required temporal years.
    temporal_values: dict[str, int] = {}
    for temporal_id, attrs in required.get("temporal", []):
        if "year" not in set(attrs):
            continue
        factual_temporal = factual_entities.temporals.get(temporal_id)
        if factual_temporal is None or getattr(factual_temporal, "year", None) is None:
            continue
        temporal_values[f"{temporal_id}.year"] = int(factual_temporal.year)
    temporal_order_count = _minimal_order_edges(temporal_values)

    # Person age ordering: minimal chain over required ages.
    age_values: dict[str, int] = {}
    for person_id, attrs in required.get("person", []):
        if "age" not in set(attrs):
            continue
        factual_person = factual_entities.persons.get(person_id)
        if factual_person is None or getattr(factual_person, "age", None) is None:
            continue
        age_values[f"{person_id}.age"] = int(factual_person.age)
    age_order_count = _minimal_order_edges(age_values)

    return number_order_count + temporal_order_count + age_order_count


@dataclass(frozen=True)
class Table1ThemeStatistics:
    theme_id: str
    documents_count: int
    entities_avg: float
    entities_std: float
    rules_avg: float
    rules_std: float
    rules_explicit_avg: float
    rules_explicit_std: float
    rules_implicit_interval_avg: float
    rules_implicit_interval_std: float
    rules_implicit_order_avg: float
    rules_implicit_order_std: float
    questions_avg: float
    questions_std: float


@dataclass(frozen=True)
class _DocumentStatistics:
    entities_count: int
    rules_explicit_count: int
    rules_implicit_interval_count: int
    rules_implicit_order_count: int
    questions_count: int

    @property
    def rules_total_count(self) -> int:
        return self.rules_explicit_count + self.rules_implicit_interval_count + self.rules_implicit_order_count


def _iter_theme_documents(root: Path) -> dict[str, list[Path]]:
    per_theme: dict[str, list[Path]] = {}
    for theme_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        yaml_paths = sorted(path for path in theme_dir.glob("*.yaml") if path.is_file())
        if not yaml_paths:
            continue
        per_theme[theme_dir.name] = yaml_paths
    return per_theme


def _load_annotated_document_relaxed(path: Path) -> AnnotatedDocument:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    document_data = payload.get("document") if isinstance(payload, dict) else None
    if not isinstance(document_data, dict):
        raise ValueError(f"Invalid document payload in {path}")

    normalized = normalize_document_taxonomy(dict(document_data))
    normalized_rules = normalize_rule_expressions(normalized.get("rules", [])) or []
    implicit_rules_data = normalize_implicit_rules_for_storage(normalized.get("implicit_rules", [])) or []

    questions: list[Question] = []
    for question_data in normalized.get("questions", []) or []:
        if not isinstance(question_data, dict):
            continue
        raw_answer = question_data.get("answer", "")
        if raw_answer is True:
            answer_str = "Yes"
        elif raw_answer is False:
            answer_str = "No"
        elif isinstance(raw_answer, list) and len(raw_answer) == 1 and isinstance(raw_answer[0], str):
            answer_str = raw_answer[0]
        else:
            answer_str = str(raw_answer)
        answer_type = question_data.get("answer_type")
        if answer_type is None:
            invariant_flag = question_data.get("is_answer_invariant")
            if invariant_flag is True:
                answer_type = "invariant"
            elif invariant_flag is False:
                answer_type = "variant"
        questions.append(
            Question(
                question_id=str(question_data.get("question_id") or ""),
                question=str(question_data.get("question") or ""),
                answer=answer_str,
                question_type=question_data.get("question_type"),
                answer_type=answer_type,
                reasoning_chain=[
                    str(step).strip() for step in (question_data.get("reasoning_chain") or []) if str(step).strip()
                ],
            )
        )

    return AnnotatedDocument(
        document_id=str(normalized.get("document_id") or path.stem),
        document_theme=str(normalized.get("document_theme") or path.parent.name),
        original_document=str(normalized.get("original_document") or ""),
        document_to_annotate=str(normalized.get("document_to_annotate") or ""),
        fictionalized_annotated_template_document=str(
            normalized.get("fictionalized_annotated_template_document") or ""
        ),
        rules=normalized_rules,
        questions=questions,
        implicit_rules=[ImplicitRule(**entry) for entry in implicit_rules_data],
        implicit_rule_exclusions=list(normalized.get("implicit_rule_exclusions") or []),
    )


def _document_statistics(path: Path) -> _DocumentStatistics:
    document = _load_annotated_document_relaxed(path)
    entities_count = _count_required_entities(document)
    rules_explicit_count = len(_non_empty_rules(document.rules))
    rules_implicit_interval_count = len(document.implicit_rules or [])
    rules_implicit_order_count = _count_implicit_order_rules(document)
    questions_count = len(document.questions or [])
    return _DocumentStatistics(
        entities_count=entities_count,
        rules_explicit_count=rules_explicit_count,
        rules_implicit_interval_count=rules_implicit_interval_count,
        rules_implicit_order_count=rules_implicit_order_count,
        questions_count=questions_count,
    )


def compute_table_1_theme_statistics(
    root: Path,
    *,
    question_count_overrides: dict[tuple[str, str], int] | None = None,
    default_question_count_for_missing: int | None = None,
    include_doc_keys: set[tuple[str, str]] | None = None,
    exclude_zero_question_documents: bool = False,
) -> list[Table1ThemeStatistics]:
    root = Path(root)
    theme_documents = _iter_theme_documents(root)

    def _safe_std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        return float(stdev(values))

    stats: list[Table1ThemeStatistics] = []
    for theme_id, document_paths in theme_documents.items():
        if include_doc_keys is not None:
            document_paths = [path for path in document_paths if (str(theme_id), str(path.stem)) in include_doc_keys]
            if not document_paths:
                continue
        doc_stats: list[_DocumentStatistics] = []
        for path in document_paths:
            doc_stat = _document_statistics(path)
            if question_count_overrides is not None:
                override_key = (str(theme_id), str(path.stem))
                if override_key in question_count_overrides:
                    override_value = max(0, int(question_count_overrides[override_key]))
                    doc_stat = _DocumentStatistics(
                        entities_count=doc_stat.entities_count,
                        rules_explicit_count=doc_stat.rules_explicit_count,
                        rules_implicit_interval_count=doc_stat.rules_implicit_interval_count,
                        rules_implicit_order_count=doc_stat.rules_implicit_order_count,
                        questions_count=override_value,
                    )
                elif default_question_count_for_missing is not None:
                    default_value = max(0, int(default_question_count_for_missing))
                    doc_stat = _DocumentStatistics(
                        entities_count=doc_stat.entities_count,
                        rules_explicit_count=doc_stat.rules_explicit_count,
                        rules_implicit_interval_count=doc_stat.rules_implicit_interval_count,
                        rules_implicit_order_count=doc_stat.rules_implicit_order_count,
                        questions_count=default_value,
                    )
            if exclude_zero_question_documents and doc_stat.questions_count <= 0:
                continue
            doc_stats.append(doc_stat)
        if not doc_stats:
            continue
        entities = [float(item.entities_count) for item in doc_stats]
        rules_total = [float(item.rules_total_count) for item in doc_stats]
        rules_explicit = [float(item.rules_explicit_count) for item in doc_stats]
        rules_interval = [float(item.rules_implicit_interval_count) for item in doc_stats]
        rules_order = [float(item.rules_implicit_order_count) for item in doc_stats]
        questions = [float(item.questions_count) for item in doc_stats]
        stats.append(
            Table1ThemeStatistics(
                theme_id=theme_id,
                documents_count=len(doc_stats),
                entities_avg=mean(entities),
                entities_std=_safe_std(entities),
                rules_avg=mean(rules_total),
                rules_std=_safe_std(rules_total),
                rules_explicit_avg=mean(rules_explicit),
                rules_explicit_std=_safe_std(rules_explicit),
                rules_implicit_interval_avg=mean(rules_interval),
                rules_implicit_interval_std=_safe_std(rules_interval),
                rules_implicit_order_avg=mean(rules_order),
                rules_implicit_order_std=_safe_std(rules_order),
                questions_avg=mean(questions),
                questions_std=_safe_std(questions),
            )
        )
    return stats


def _theme_label(theme_id: str) -> str:
    mapping = {
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
    return mapping.get(theme_id, theme_id.replace("_", " ").title())


def _fmt(value: float) -> str:
    return f"{value:.1f}"


def render_table_1_latex(stats: list[Table1ThemeStatistics]) -> str:
    order = [
        "award_winners",
        "biographies_of_famous_personalities",
        "cities_countries_and_regions",
        "companies_and_organizations",
        "natural_disasters",
        "public_attacks_news_articles",
        "retail_banking_regulations_and_policies",
        "space_missions",
        "sport_events",
    ]
    order_index = {theme_id: idx for idx, theme_id in enumerate(order)}
    sorted_stats = sorted(stats, key=lambda item: order_index.get(item.theme_id, 999))

    def _mean_with_stacked_std(avg: float, std: float) -> str:
        return rf"\shortstack{{{_fmt(avg)}\\{{\scriptsize $\pm{_fmt(std)}$}}}}"

    def _aggregate_mean_std(rows: list[tuple[int, float, float]]) -> tuple[float, float]:
        total_n = sum(max(0, int(n)) for n, _avg, _std in rows)
        if total_n <= 0:
            return 0.0, 0.0
        weighted_sum = sum(int(n) * float(avg) for n, avg, _std in rows if int(n) > 0)
        overall_mean = weighted_sum / float(total_n)
        if total_n < 2:
            return overall_mean, 0.0

        sum_squares = 0.0
        for n_raw, avg_raw, std_raw in rows:
            n = int(n_raw)
            if n <= 0:
                continue
            avg = float(avg_raw)
            std = float(std_raw)
            if n > 1:
                sum_squares += (n - 1) * (std**2)
            sum_squares += n * (avg**2)

        variance = (sum_squares - total_n * (overall_mean**2)) / float(total_n - 1)
        variance = max(0.0, variance)
        return overall_mean, math.sqrt(variance)

    entities_total_avg, entities_total_std = _aggregate_mean_std(
        [(item.documents_count, item.entities_avg, item.entities_std) for item in sorted_stats]
    )
    rules_all_total_avg, rules_all_total_std = _aggregate_mean_std(
        [(item.documents_count, item.rules_avg, item.rules_std) for item in sorted_stats]
    )
    question_counts = [round(float(item.questions_avg) * int(item.documents_count)) for item in sorted_stats]
    total_questions = sum(question_counts)

    def _count_with_share(count: int, total: int, *, include_share: bool = True) -> str:
        if not include_share:
            return str(count)
        if total <= 0:
            return str(count)
        share = 100.0 * count / total
        return rf"\shortstack{{{count}\\{{\scriptsize ({_fmt(share)}\%)}}}}"

    lines: list[str] = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"    \centering")
    lines.append(r"    \footnotesize")
    lines.append(r"    \renewcommand{\arraystretch}{1.18}")
    lines.append(r"    \setlength{\tabcolsep}{3.2pt}")
    lines.append(
        r"    \caption{\textbf{Dataset statistics by theme category.} We provide the distribution of "
        r"questions and the average number of entities/rules per document.}"
    )
    lines.append(r"    \label{tab:theme-dataset-stats}")
    lines.append(
        r"    \begin{tabular*}{\textwidth}{@{}l@{\extracolsep{\fill}}" + ("c" * (len(sorted_stats) + 1)) + r"@{}}"
    )
    lines.append(r"        \toprule")
    header_cells = [""]
    header_cells.extend(f"\\rotatebox{{60}}{{\\textbf{{{_theme_label(item.theme_id)}}}}}" for item in sorted_stats)
    header_cells.append(r"\rotatebox{60}{\textbf{Total}}")
    lines.append("        " + " & ".join(header_cells) + r" \\")
    lines.append(r"        \midrule")
    lines.append(
        "        "
        + " & ".join(
            [r"\shortstack[l]{\textbf{\#Question}\\\textbf{Templates}}"]
            + [_count_with_share(count, total_questions) for count in question_counts]
            + [_count_with_share(total_questions, total_questions, include_share=True)]
        )
        + r" \\"
    )
    lines.append(r"        \midrule")
    lines.append(
        "        "
        + " & ".join(
            [r"\textbf{Avg \#Entities}"]
            + [_mean_with_stacked_std(item.entities_avg, item.entities_std) for item in sorted_stats]
            + [_mean_with_stacked_std(entities_total_avg, entities_total_std)]
        )
        + r" \\"
    )
    lines.append(r"        \midrule")
    lines.append(
        "        "
        + " & ".join(
            [r"\textbf{Avg \#Rules}"]
            + [_mean_with_stacked_std(item.rules_avg, item.rules_std) for item in sorted_stats]
            + [_mean_with_stacked_std(rules_all_total_avg, rules_all_total_std)]
        )
        + r" \\"
    )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular*}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"
