"""Paper tables and numerical figures from freshly evaluated benchmark runs."""

from collections import defaultdict
import csv
from dataclasses import asdict
import math
from pathlib import Path
import statistics

import numpy as np

from . import ablations, bootstrap, effort, endpoint, flips, frequency, plots, temperature
from .paper_inputs import ROOT, load_run, temperature_records

SEED = 20_260_724
ENDPOINTS = ("factual", "fictional")
PERCENTAGES = (10, 20, 30, 50, 80, 90)
PARTIAL = tuple(f"fictional_{p}pct" for p in PERCENTAGES)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_report(dataset, out):
    from memoreason.paper_results.reviewed_template_statistics import compute_table_1_theme_statistics
    from memoreason.factual_to_fictional_dataset.dataset_record_core import build_factual_dataset_record

    templates = sorted((ROOT / "data/HUMAN_ANNOTATED_TEMPLATES").glob("*/*.yaml"))
    lengths = defaultdict(list)
    for path in templates:
        # Pure factual construction: no model calls or writes to the source data.
        lengths[path.parent.name].append(len(build_factual_dataset_record(path, seed=23)["document_text"]))
    stats = [asdict(row) for row in compute_table_1_theme_statistics(ROOT / "data/HUMAN_ANNOTATED_TEMPLATES")]
    for row in stats:
        values = lengths[row["theme_id"]]
        row.update(
            task_templates=round(row["documents_count"] * row["questions_avg"]),
            characters_mean=statistics.mean(values),
            characters_std=statistics.stdev(values),
        )
    total = dict(
        theme_id="Total",
        documents_count=sum(r["documents_count"] for r in stats),
        task_templates=sum(r["task_templates"] for r in stats),
    )
    n = total["documents_count"]
    for prefix in (
        "entities",
        "rules",
        "rules_explicit",
        "rules_implicit_interval",
        "rules_implicit_order",
        "questions",
    ):
        mean = sum(r["documents_count"] * r[f"{prefix}_avg"] for r in stats) / n
        variance = sum(
            (r["documents_count"] - 1) * r[f"{prefix}_std"] ** 2
            + r["documents_count"] * (r[f"{prefix}_avg"] - mean) ** 2
            for r in stats
        ) / (n - 1)
        total.update({f"{prefix}_avg": mean, f"{prefix}_std": math.sqrt(variance)})
    all_lengths = [value for values in lengths.values() for value in values]
    total.update(characters_mean=statistics.mean(all_lengths), characters_std=statistics.stdev(all_lengths))
    stats.append(total)
    write_csv(out / "table1_dataset_statistics.csv", stats)
    settings = ("factual", "fictional", "fictional_named", "fictional_numtemp", *PARTIAL)
    write_csv(
        out / "table6_settings.csv",
        [
            dict(
                setting=s,
                source_documents=100,
                variants_per_document=1 if s == "factual" else 10,
                questions=1200 if s == "factual" else 12000,
            )
            for s in settings
        ],
    )


def main_report(runs, out, samples):
    loaded = load_run(runs / "main", endpoint.MODEL_ORDER, ENDPOINTS)
    metrics, shortcut = endpoint.endpoint_and_shortcut_rows(loaded)
    write_csv(out / "table4_shortcut_rate.csv", shortcut)
    reasoning = {r["model_name"]: r["shortcut_rate_pp"] for r in shortcut if r["question_type"] == "reasoning"}
    for rank, row in enumerate(sorted(metrics, key=lambda r: -r["fictional_accuracy_percent"]), 1):
        row.update(rank=rank, shortcut_rate_percent=reasoning[row["model_name"]])
    write_csv(out / "table5_leaderboard.csv", sorted(metrics, key=lambda r: r["rank"]))
    changes, cot = [], []
    for model, rows in loaded.items():
        factual = [r for r in rows if r["document_setting"] == "factual"]
        fictitious = [r for r in rows if r["document_setting"] == "fictional"]
        documents, factual_index, fictitious_index = bootstrap.validate_and_index(
            factual, fictitious, expected_documents=100, expected_variants=10
        )
        for bi, behavior in enumerate(bootstrap.ANSWER_BEHAVIORS):
            for qi, question_type in enumerate(("arithmetic", "temporal", "inference", "reasoning", "extractive")):
                fv, vv = bootstrap.cell_values(
                    documents, factual_index, fictitious_index, question_type=question_type, answer_behavior=behavior
                )
                lo, hi = bootstrap.bootstrap_interval(vv - fv, resamples=samples, seed=SEED + bi * 100 + qi)
                changes.append(
                    dict(
                        model_name=model,
                        answer_behavior=behavior,
                        question_type=question_type,
                        factual_accuracy_percent=100 * fv.mean(),
                        fictitious_accuracy_percent=100 * vv.mean(),
                        change_pp=100 * (vv - fv).mean(),
                        ci95_lower_pp=lo,
                        ci95_upper_pp=hi,
                        significant=bool(lo > 0 or hi < 0),
                    )
                )
        if model in ("olmo-3-7b-instruct", "olmo-3-7b-think"):
            accuracies = [
                100
                * np.mean([r["new_final_is_correct"] for r in group if r["question_type"] in endpoint.REASONING_TYPES])
                for group in (factual, fictitious)
            ]
            cot.append(
                dict(
                    model=model,
                    factual_accuracy_percent=accuracies[0],
                    fictitious_accuracy_percent=accuracies[1],
                    gap_pp=accuracies[0] - accuracies[1],
                )
            )
    write_csv(out / "table2_performance_changes.csv", changes)
    write_csv(out / "chain_of_thought_comparison.csv", cot)
    clusters = tuple(sorted({(r["document_theme"], r["document_id"]) for r in next(iter(loaded.values()))}))
    draws = np.random.default_rng(SEED).integers(0, len(clusters), size=(samples, len(clusters)))
    output = []
    for model in flips.MODEL_ORDER:
        report = flips.compute_model(model, loaded[model], draws=draws, clusters=clusters)
        for metric in ("forward", "mirror", "difference"):
            output.append(dict(model=model, metric=metric, **report[metric]))
    write_csv(out / "table8_directional_flips.csv", output)


def partial_report(runs, out, samples):
    models = tuple(m for m, _ in plots.PARTIAL_MODELS)
    ends = load_run(runs / "main", models, ENDPOINTS)
    partial = load_run(runs / "partial", models, PARTIAL)
    for model in models:
        rows = ends[model] + partial[model]
        summaries = []
        for percent in (0, *PERCENTAGES, 100):
            setting = "factual" if percent == 0 else "fictional" if percent == 100 else f"fictional_{percent}pct"
            selected = [r for r in rows if r["document_setting"] == setting]
            correct = sum(r["new_final_is_correct"] for r in selected)
            p = correct / len(selected)
            half = 100 * 1.96 * math.sqrt(p * (1 - p) / len(selected))
            summaries.append(
                dict(
                    model=model,
                    replaced_percent=percent,
                    count_correct=correct,
                    count_total=len(selected),
                    accuracy_percent=100 * p,
                    ci95_lower_percent=100 * p - half,
                    ci95_upper_percent=100 * p + half,
                )
            )
        write_csv(out / "partial" / f"{model}.csv", summaries)
    plots.render_all(out)


def effort_report(runs, out, samples):
    model = effort.MODEL
    levels = {"low": load_run(runs / "main", (model,), ENDPOINTS)[model]}
    for level in ("medium", "high"):
        levels[level] = load_run(runs / "effort" / level, (model,), ENDPOINTS, reasoning_effort=level)[model]
    documents, matrix, audit = effort.build_document_metrics(levels)
    report = effort.build_report(documents, matrix, audit, samples=samples, seed=SEED, chunk_size=1000)
    rows, contrasts = [], []
    for level, metrics in report["metrics"].items():
        for metric, cell in metrics.items():
            rows.append(
                dict(
                    effort=level,
                    metric=metric,
                    value_percent=cell["value_percent"],
                    ci95_lower_percent=cell["ci95_percent"][0],
                    ci95_upper_percent=cell["ci95_percent"][1],
                )
            )
    for name, metrics in report["contrasts"].items():
        for metric, cell in metrics.items():
            contrasts.append(
                dict(
                    contrast=name,
                    metric=metric,
                    change_pp=cell["value_percent"],
                    ci95_lower_pp=cell["ci95_percent"][0],
                    ci95_upper_pp=cell["ci95_percent"][1],
                )
            )
    write_csv(out / "table3_reasoning_effort.csv", rows)
    write_csv(out / "reasoning_effort_contrasts.csv", contrasts)


def ablation_report(runs, out, samples):
    models = (*ablations.MODELS, "claude-sonnet-4-6")
    ends = load_run(runs / "main", models, ENDPOINTS)
    named = load_run(runs / "ablations", ablations.MODELS, ("fictional_named",))
    values = load_run(runs / "ablations", models, ("fictional_numtemp",))
    documents, matrix = ablations.invariant_document_matrix(ends, values, named)
    report = ablations.compute_four_setting(documents, matrix, samples=samples, seed=20_260_826, chunk_size=1000)
    rows = []
    for model in ablations.MODELS:
        for setting in ablations.SETTINGS:
            row = dict(
                model_name=model,
                setting=setting,
                accuracy_percent=report["by_model"][model]["settings"][setting]["percent"],
            )
            if setting != "factual":
                cell = report["by_model"][model]["contrasts"][f"{setting}_minus_factual"]
                row.update(
                    change_pp=cell["percentage_points"],
                    ci95_lower_pp=cell["cluster_bootstrap_95ci_percentage_points"][0],
                    ci95_upper_pp=cell["cluster_bootstrap_95ci_percentage_points"][1],
                )
            rows.append(row)
    write_csv(out / "table7_cue_ablations.csv", rows)
    write_csv(
        out / "table9_value_only_shortcut.csv",
        ablations.compute_numtemp_shortcut(values, ablations.factual_answers(ends), models=models),
    )


def temperature_report(runs, out, samples):
    conditions = {}
    for condition, value, seed in temperature.CONDITIONS:
        path = runs / "main" if value == 0 else runs / "temperature" / condition
        conditions[condition] = load_run(path, temperature.MODELS, ENDPOINTS, temperature=value, seed=seed)
    rows = temperature.compute(temperature_records(conditions), samples=samples, seed=SEED)
    write_csv(out / "table10_temperature.csv", rows)


def frequency_report(runs, out, samples, counts_path):
    from .entity_counts import INDEX

    loaded = load_run(runs / "main", ("olmo-3-7b-think",), ENDPOINTS)["olmo-3-7b-think"]
    grouped = defaultdict(list)
    for row in loaded:
        grouped[(row["document_theme"], row["document_id"], row["question_id"])].append(row)
    questions = []
    for (theme, document, _), rows in sorted(grouped.items()):
        factual = [r for r in rows if r["document_setting"] == "factual"]
        fictitious = [r for r in rows if r["document_setting"] == "fictional"]
        questions.append(
            dict(
                document_theme=theme,
                document_id=document,
                factual_correct=int(factual[0]["new_final_is_correct"]),
                fictional_successes=sum(r["new_final_is_correct"] for r in fictitious),
                fictional_total=len(fictitious),
            )
        )
    with counts_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = {}
    for row in rows:
        key = row["document_theme"], row["document_id"]
        if row["index"] != INDEX or key in counts:
            raise ValueError("Frequency index mismatch or duplicate document")
        counts[key] = {
            name: float(row[name])
            for name in (
                "num_named_entities",
                "mean_named_entity_count",
                "median_named_entity_count",
                "mean_log1p_named_entity_count",
                "zero_count_named_entities",
            )
        }
        if any(not math.isfinite(value) or value < 0 for value in counts[key].values()):
            raise ValueError("Frequency counts must be finite and non-negative")
        if (
            not counts[key]["num_named_entities"]
            or counts[key]["zero_count_named_entities"] > counts[key]["num_named_entities"]
        ):
            raise ValueError("Invalid named-entity count totals")
    documents = frequency._document_rows(questions, counts)
    distributions, _ = frequency._bootstrap_distributions(documents, replicates=samples, seed=20_260_728)
    metrics = frequency._metric_rows(frequency._point_metrics(documents), distributions)
    write_csv(out / "frequency_documents.csv", documents)
    write_csv(out / "frequency_statistics.csv", metrics)
    x = np.array([r["log10_1p_mean_named_entity_count"] for r in documents])
    y = np.array([r["performance_drop_pp"] for r in documents])
    fig, ax = plots.plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.scatter(x, y, s=18, alpha=0.65, color="#608cb5")
    fit = np.polyfit(x, y, 1)
    xx = np.array([min(x), max(x)])
    ax.plot(xx, np.polyval(fit, xx), color="#e53b59", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set(xlabel="Mean named-entity count, log10(1 + count)", ylabel="Factual–fictitious accuracy gap (pp)")  # noqa: RUF001
    labels = {
        "pearson_r": "Pearson r",
        "spearman_r": "Spearman rho",
        "ols_slope_pp_per_log10_decade": "Slope (pp/decade)",
    }
    text = "\n".join(
        f"{labels[r['metric']]} = {r['point']:.3f} [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
        for r in metrics
        if r["metric"] in labels
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="lightgray", boxstyle="round,pad=.3"),
    )
    plots.save(fig, out, "figure4_frequency")
