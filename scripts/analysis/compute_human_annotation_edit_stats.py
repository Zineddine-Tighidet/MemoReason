#!/usr/bin/env python3
"""Compute human edit statistics for QA and rule review campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.review_tools.human_edit_stats import (  # noqa: E402
    campaign_input_dir,
    compare_entity_edits,
    compare_question_edits,
    compare_rule_set_edits,
    iter_template_paths,
    parse_doc_key,
    stats_to_dict,
)


DEFAULT_EXCLUDED_DOCS = ("retail_banking_regulations_and_policies/bankreg_11",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-workspace",
        type=Path,
        default=PROJECT_ROOT / "web" / "data" / "annotation_workspace",
        help="Path to the restored annotation workspace containing _review_campaigns/.",
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "HUMAN_ANNOTATED_TEMPLATES",
        help="Path to the final human-annotated benchmark templates.",
    )
    parser.add_argument(
        "--final-questions-dir",
        type=Path,
        default=None,
        help="Optional final QA directory override. Defaults to --templates-dir.",
    )
    parser.add_argument(
        "--final-rules-dir",
        type=Path,
        default=None,
        help="Optional final rules directory override. Defaults to --templates-dir.",
    )
    parser.add_argument(
        "--entity-source-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing initial AI document annotations for entity-edit stats. "
            "Defaults to ANNOTATION_WORKSPACE/claude-opus-4-6."
        ),
    )
    parser.add_argument(
        "--question-campaign-id",
        type=int,
        default=5,
        help="Review campaign id whose input snapshots are the AI-drafted QA baseline.",
    )
    parser.add_argument(
        "--rule-campaign-id",
        type=int,
        default=2,
        help="Review campaign id whose input snapshots are the AI-drafted rule baseline.",
    )
    parser.add_argument(
        "--exclude-doc",
        action="append",
        default=None,
        metavar="THEME/DOC_ID",
        help="Document to exclude from the paper scope. Can be repeated.",
    )
    parser.add_argument(
        "--no-default-exclusions",
        action="store_true",
        help="Do not exclude the default out-of-scope documents.",
    )
    parser.add_argument(
        "--include-qa-types",
        action="store_true",
        help="Also count question_type/answer_type changes as QA edits.",
    )
    parser.add_argument(
        "--ignore-rule-comments",
        action="store_true",
        help="Compare only the rule expression before '#', ignoring comment edits.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON path for the full reproducibility record.",
    )
    return parser.parse_args()


def _excluded_docs(args: argparse.Namespace) -> set[tuple[str, str]]:
    raw_exclusions = [] if args.no_default_exclusions else list(DEFAULT_EXCLUDED_DOCS)
    for raw in args.exclude_doc or []:
        if raw not in raw_exclusions:
            raw_exclusions.append(raw)
    return {parse_doc_key(raw) for raw in raw_exclusions}


def _print_summary(report: dict[str, object]) -> None:
    entities = report["entity_edits"]
    qa = report["qa_edits"]
    rules = report["rule_set_edits"]
    assert isinstance(entities, dict)
    assert isinstance(qa, dict)
    assert isinstance(rules, dict)

    entity_dist = entities["per_document_edits"]
    assert isinstance(entity_dist, dict)
    print("Entity edits")
    print(
        "  operations: "
        f"{entities['entity_edit_operations']} "
        f"(modified/add/delete: "
        f"{entities['modified_entities']} / {entities['added_entities']} / {entities['deleted_entities']})"
    )
    print(
        "  per document: "
        f"mean {entity_dist['mean_edits_per_doc']:.2f}, "
        f"sample std {entity_dist['sample_std_edits_per_doc']:.2f}"
    )

    print("QA edits")
    print(
        "  interventions: "
        f"{qa['interventions']} / {qa['source_items']} "
        f"({qa['source_item_edit_rate']:.1%})"
    )
    print(
        "  changed/add/delete: "
        f"{qa['changed']} / {qa['added']} / {qa['deleted']}"
    )
    print(
        "  documents with QA edits: "
        f"{qa['docs_with_edits']} / {qa['docs_compared']} "
        f"({qa['doc_edit_rate']:.1%})"
    )

    print("Rule-set edits")
    print(
        "  documents with rule-set edits: "
        f"{rules['docs_with_rule_set_edits']} / {rules['docs_compared']} "
        f"({rules['doc_edit_rate']:.1%})"
    )
    rule_dist = rules["per_document_edit_operations"]
    assert isinstance(rule_dist, dict)
    print(
        "  edit operations per document: "
        f"mean {rule_dist['mean_edits_per_doc']:.2f}, "
        f"sample std {rule_dist['sample_std_edits_per_doc']:.2f}"
    )
    print(
        "  exact rule additions/deletions: "
        f"{rules['added_rules']} / {rules['deleted_rules']}"
    )
    print(
        "  exact rules source/final: "
        f"{rules['source_rules']} / {rules['target_rules']}"
    )


def main() -> int:
    args = parse_args()
    excluded_docs = _excluded_docs(args)

    paper_scope_paths = iter_template_paths(args.templates_dir)
    paper_scope_keys = set(paper_scope_paths) - excluded_docs
    entity_source_dir = args.entity_source_dir or (args.annotation_workspace / "claude-opus-4-6")
    entity_source_paths = {
        key: path
        for key, path in iter_template_paths(entity_source_dir).items()
        if key in paper_scope_keys
    }
    final_question_paths = {
        key: path
        for key, path in iter_template_paths(args.final_questions_dir or args.templates_dir).items()
        if key in paper_scope_keys
    }
    final_rule_paths = {
        key: path
        for key, path in iter_template_paths(args.final_rules_dir or args.templates_dir).items()
        if key in paper_scope_keys
    }
    question_source_paths = iter_template_paths(
        campaign_input_dir(args.annotation_workspace, "questions", args.question_campaign_id)
    )
    rule_source_paths = iter_template_paths(
        campaign_input_dir(args.annotation_workspace, "rules", args.rule_campaign_id)
    )

    entity_stats = compare_entity_edits(
        source_paths=entity_source_paths,
        final_paths=paper_scope_paths,
        excluded_docs=excluded_docs,
    )
    qa_stats = compare_question_edits(
        source_paths=question_source_paths,
        final_paths=final_question_paths,
        excluded_docs=excluded_docs,
        include_types=args.include_qa_types,
    )
    rule_stats = compare_rule_set_edits(
        source_paths=rule_source_paths,
        final_paths=final_rule_paths,
        excluded_docs=excluded_docs,
        ignore_comments=args.ignore_rule_comments,
    )

    report: dict[str, object] = {
        "method": {
            "annotation_workspace": str(args.annotation_workspace),
            "templates_dir": str(args.templates_dir),
            "final_questions_dir": str(args.final_questions_dir or args.templates_dir),
            "final_rules_dir": str(args.final_rules_dir or args.templates_dir),
            "entity_source_dir": str(entity_source_dir),
            "scope_dir": str(args.templates_dir),
            "scope_docs": len(paper_scope_keys),
            "question_campaign_id": args.question_campaign_id,
            "rule_campaign_id": args.rule_campaign_id,
            "excluded_docs": sorted("/".join(key) for key in excluded_docs),
            "qa_entity_surface_normalization": "annotated mentions are compared by entity ref",
            "qa_compared_fields": ["question", "answer"]
            + (["question_type", "answer_type"] if args.include_qa_types else []),
            "entity_edits_are_unique_refs": True,
            "entity_modification_definition": "same entity ref appears in both documents but its surface multiset changed",
            "rules_are_compared_as": "unordered multisets of rule strings",
            "rule_edit_operation_definition": (
                "after exact matches, per-document operations are max(added_rules, deleted_rules), "
                "so one deleted+added pair is counted as one modification"
            ),
            "ignore_rule_comments": bool(args.ignore_rule_comments),
        },
        "entity_edits": stats_to_dict(entity_stats),
        "qa_edits": stats_to_dict(qa_stats),
        "rule_set_edits": stats_to_dict(rule_stats),
    }

    _print_summary(report)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
