#!/usr/bin/env python3
"""Audit evaluated model outputs against the current benchmark dataset files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_export.dataset_paths import MODEL_EVAL_RAW_OUTPUTS_DIR, sanitize_model_name
from src.evaluation_workflows.parametric_shortcut.dataset import is_excluded_evaluation_document
from src.evaluation_workflows.parametric_shortcut.version_audit import (
    audit_saved_output_payload,
    blocking_issues,
)

try:
    YAML_LOADER = yaml.CSafeLoader
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader


def _discover_models(stage_suffix: str) -> list[str]:
    models = {
        path.parent.name
        for path in MODEL_EVAL_RAW_OUTPUTS_DIR.glob(f"*/*/*_{stage_suffix}.yaml")
    }
    return sorted(models)


def _iter_paths(model_name: str, stage_suffix: str) -> list[Path]:
    model_dir_name = sanitize_model_name(model_name)
    paths = sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.glob(f"*/{model_dir_name}/*_{stage_suffix}.yaml"))
    if not paths and model_dir_name != model_name:
        paths = sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.glob(f"*/{model_name}/*_{stage_suffix}.yaml"))
    return paths


def _payload_rows(payload: dict) -> int:
    results = payload.get("results") or []
    return len(results) if isinstance(results, list) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, help="Model folder/registry names to audit.")
    parser.add_argument(
        "--stage",
        choices=["raw_outputs", "parsed_outputs", "evaluated_outputs"],
        default="evaluated_outputs",
        help="Artifact stage to audit.",
    )
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["factual", "fictional"],
        help="Document settings to include.",
    )
    parser.add_argument(
        "--output-json",
        default="artifacts/analysis/model_eval_version_consistency_audit.json",
        help="Path for the machine-readable audit summary.",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/analysis/model_eval_version_consistency_audit_examples.csv",
        help="Path for issue examples.",
    )
    parser.add_argument("--max-examples-per-model", type=int, default=20)
    args = parser.parse_args()

    stage_suffix = str(args.stage)
    models = [
        str(model).strip()
        for model in (args.models or _discover_models(stage_suffix))
        if str(model).strip()
    ]
    settings = {str(setting).strip().lower() for setting in args.settings if str(setting).strip()}
    check_score_metadata = stage_suffix != "raw_outputs"

    summary: dict[str, dict] = {}
    examples: list[dict[str, str]] = []

    for model_name in models:
        counters: Counter[str] = Counter()
        issue_counts: Counter[str] = Counter()
        issue_files_by_type: dict[str, set[str]] = defaultdict(set)
        paths = _iter_paths(model_name, stage_suffix)
        for path in paths:
            payload = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER) or {}
            setting = str(payload.get("document_setting") or "").strip().lower()
            if settings and setting not in settings:
                continue
            document_id = str(payload.get("document_id") or "").strip()
            if document_id and is_excluded_evaluation_document(document_id):
                counters["excluded_files"] += 1
                continue

            counters["files"] += 1
            counters[f"{setting}_files"] += 1
            row_count = _payload_rows(payload)
            counters["rows"] += row_count
            counters[f"{setting}_rows"] += row_count

            issues = blocking_issues(
                audit_saved_output_payload(
                    payload,
                    path=path,
                    check_score_metadata=check_score_metadata,
                )
            )
            if not issues:
                continue
            counters["files_with_issues"] += 1
            counters[f"{setting}_files_with_issues"] += 1
            for issue in issues:
                issue_counts[issue.issue_type] += 1
                issue_files_by_type[issue.issue_type].add(issue.path)
            if len([row for row in examples if row["model_name"] == model_name]) < args.max_examples_per_model:
                for issue in issues:
                    if len([row for row in examples if row["model_name"] == model_name]) >= args.max_examples_per_model:
                        break
                    examples.append({"model_name": model_name, "stage": stage_suffix, **issue.to_payload()})

        summary[model_name] = {
            "stage": stage_suffix,
            **dict(counters),
            "issues_by_type": dict(sorted(issue_counts.items())),
            "issue_files_by_type": {
                issue_type: len(files) for issue_type, files in sorted(issue_files_by_type.items())
            },
            "is_publication_safe": counters.get("files", 0) > 0 and counters.get("files_with_issues", 0) == 0,
        }

    output_json = PROJECT_ROOT / str(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    output_csv = PROJECT_ROOT / str(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "model_name",
            "stage",
            "issue_type",
            "severity",
            "path",
            "document_id",
            "document_setting",
            "document_variant_id",
            "question_id",
            "detail",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(examples)

    print(output_json)
    print(output_csv)
    for model_name, payload in summary.items():
        safe = "safe" if payload["is_publication_safe"] else "STALE"
        print(
            f"{model_name}: {safe}, files={payload.get('files', 0)}, rows={payload.get('rows', 0)}, "
            f"files_with_issues={payload.get('files_with_issues', 0)}, "
            f"issues={payload.get('issues_by_type', {})}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
