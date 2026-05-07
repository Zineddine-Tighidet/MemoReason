"""Reproducible human-edit statistics for reviewed templates.

The comparison is designed for the paper audit: it compares the initial
AI-generated review-campaign inputs against the final benchmark templates while
normalizing annotated entity surfaces such as ``[Barack Obama; person_1.name]``
to ``[person_1.name]``. This avoids counting factual/fictional surface swaps as
human edits when the underlying entity reference is unchanged.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from statistics import mean, pstdev, stdev
from typing import Any

import yaml


DocKey = tuple[str, str]

ANNOTATED_MENTION_PATTERN = re.compile(r"\[([^\]]+);\s*([^\]]+)\]")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class PerDocumentEditStats:
    """Distribution of edit counts over documents."""

    docs_compared: int
    total_edits: int
    mean_edits_per_doc: float
    sample_std_edits_per_doc: float
    population_std_edits_per_doc: float
    min_edits_per_doc: int
    max_edits_per_doc: int
    counts_by_doc: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemEditStats:
    """Edit counts for id-addressable items such as QAs."""

    docs_compared: int
    docs_with_edits: int
    source_items: int
    target_items: int
    unchanged: int
    changed: int
    added: int
    deleted: int
    ignored_source_only_docs: list[str] = field(default_factory=list)
    ignored_target_only_docs: list[str] = field(default_factory=list)

    @property
    def interventions(self) -> int:
        return self.changed + self.added + self.deleted

    @property
    def source_item_edit_rate(self) -> float:
        return self.interventions / self.source_items if self.source_items else 0.0

    @property
    def doc_edit_rate(self) -> float:
        return self.docs_with_edits / self.docs_compared if self.docs_compared else 0.0


@dataclass(frozen=True)
class RuleSetEditStats:
    """Edit counts for unordered document-level rule sets."""

    docs_compared: int
    docs_with_rule_set_edits: int
    source_rules: int
    target_rules: int
    unchanged_rules: int
    added_rules: int
    deleted_rules: int
    edit_operations: int
    per_document_edit_operations: PerDocumentEditStats
    ignored_source_only_docs: list[str] = field(default_factory=list)
    ignored_target_only_docs: list[str] = field(default_factory=list)

    @property
    def doc_edit_rate(self) -> float:
        return self.docs_with_rule_set_edits / self.docs_compared if self.docs_compared else 0.0


@dataclass(frozen=True)
class EntityEditStats:
    """Edit counts for unique inline entity refs in document annotations."""

    docs_compared: int
    docs_with_entity_edits: int
    source_entities: int
    target_entities: int
    unchanged_entities: int
    modified_entities: int
    added_entities: int
    deleted_entities: int
    per_document_edits: PerDocumentEditStats
    ignored_source_only_docs: list[str] = field(default_factory=list)
    ignored_target_only_docs: list[str] = field(default_factory=list)

    @property
    def entity_edit_operations(self) -> int:
        return self.modified_entities + self.added_entities + self.deleted_entities

    @property
    def doc_edit_rate(self) -> float:
        return self.docs_with_entity_edits / self.docs_compared if self.docs_compared else 0.0


def stats_to_dict(stats: ItemEditStats | RuleSetEditStats | EntityEditStats) -> dict[str, Any]:
    out = asdict(stats)
    if isinstance(stats, ItemEditStats):
        out["interventions"] = stats.interventions
        out["source_item_edit_rate"] = stats.source_item_edit_rate
        out["doc_edit_rate"] = stats.doc_edit_rate
    elif isinstance(stats, RuleSetEditStats):
        out["doc_edit_rate"] = stats.doc_edit_rate
    else:
        out["entity_edit_operations"] = stats.entity_edit_operations
        out["doc_edit_rate"] = stats.doc_edit_rate
    return out


def summarize_per_document_counts(counts_by_doc: dict[str, int]) -> PerDocumentEditStats:
    counts = list(counts_by_doc.values())
    if not counts:
        return PerDocumentEditStats(
            docs_compared=0,
            total_edits=0,
            mean_edits_per_doc=0.0,
            sample_std_edits_per_doc=0.0,
            population_std_edits_per_doc=0.0,
            min_edits_per_doc=0,
            max_edits_per_doc=0,
            counts_by_doc={},
        )
    return PerDocumentEditStats(
        docs_compared=len(counts),
        total_edits=sum(counts),
        mean_edits_per_doc=float(mean(counts)),
        sample_std_edits_per_doc=float(stdev(counts)) if len(counts) > 1 else 0.0,
        population_std_edits_per_doc=float(pstdev(counts)),
        min_edits_per_doc=min(counts),
        max_edits_per_doc=max(counts),
        counts_by_doc=dict(sorted(counts_by_doc.items())),
    )


def doc_key_label(key: DocKey) -> str:
    return f"{key[0]}/{key[1]}"


def parse_doc_key(raw: str) -> DocKey:
    if "/" not in raw:
        raise ValueError(f"Expected THEME/DOC_ID, got: {raw}")
    theme, doc_id = raw.split("/", 1)
    theme = theme.strip()
    doc_id = doc_id.strip()
    if not theme or not doc_id:
        raise ValueError(f"Expected THEME/DOC_ID, got: {raw}")
    return theme, doc_id


def iter_template_paths(root: Path) -> dict[DocKey, Path]:
    paths: dict[DocKey, Path] = {}
    if not root.exists():
        raise FileNotFoundError(f"Missing template directory: {root}")
    for path in sorted(root.glob("*/*.yaml")):
        key = (path.parent.name, path.stem)
        paths[key] = path
    return paths


def load_document(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return {}
    document = payload.get("document", payload)
    return document if isinstance(document, dict) else {}


def canonicalize_annotated_text(value: Any) -> str:
    text = "" if value is None else str(value)

    def replace_mention(match: re.Match[str]) -> str:
        ref = WHITESPACE_PATTERN.sub("", match.group(2).strip())
        return f"[{ref}]"

    text = ANNOTATED_MENTION_PATTERN.sub(replace_mention, text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def canonical_question_record(raw_question: dict[str, Any], *, include_types: bool = False) -> dict[str, str]:
    record = {
        "question": canonicalize_annotated_text(raw_question.get("question")),
        "answer": canonicalize_annotated_text(raw_question.get("answer")),
    }
    if include_types:
        record["question_type"] = str(raw_question.get("question_type") or "").strip()
        record["answer_type"] = str(raw_question.get("answer_type") or "").strip()
    return record


def question_map(path: Path, *, include_types: bool = False) -> dict[str, dict[str, str]]:
    document = load_document(path)
    questions = document.get("questions") or []
    out: dict[str, dict[str, str]] = {}
    for index, raw_question in enumerate(questions, start=1):
        if not isinstance(raw_question, dict):
            continue
        question_id = str(raw_question.get("question_id") or f"__index_{index}").strip()
        out[question_id] = canonical_question_record(raw_question, include_types=include_types)
    return out


def entity_surface_map(path: Path) -> dict[str, Counter[str]]:
    document = load_document(path)
    text = str(document.get("document_to_annotate") or "")
    out: dict[str, Counter[str]] = {}
    for match in ANNOTATED_MENTION_PATTERN.finditer(text):
        surface = WHITESPACE_PATTERN.sub(" ", match.group(1).strip())
        surface = surface.replace("\u2018", "'").replace("\u2019", "'")
        surface = surface.replace("\u201c", '"').replace("\u201d", '"')
        ref = WHITESPACE_PATTERN.sub("", match.group(2).strip())
        if not ref or not surface:
            continue
        out.setdefault(ref, Counter())[surface] += 1
    return out


def canonical_rule(value: Any, *, ignore_comments: bool = False) -> str:
    text = "" if value is None else str(value)
    if ignore_comments:
        text = text.split("#", 1)[0]
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def rule_counter(path: Path, *, ignore_comments: bool = False) -> Counter[str]:
    document = load_document(path)
    rules = document.get("rules") or []
    out: Counter[str] = Counter()
    for raw_rule in rules:
        normalized = canonical_rule(raw_rule, ignore_comments=ignore_comments)
        if normalized:
            out[normalized] += 1
    return out


def shared_final_doc_scope(
    *,
    source_paths: dict[DocKey, Path],
    final_paths: dict[DocKey, Path],
    excluded_docs: set[DocKey] | None = None,
) -> tuple[list[DocKey], list[str], list[str]]:
    excluded_docs = excluded_docs or set()
    source_keys = set(source_paths) - excluded_docs
    final_keys = set(final_paths) - excluded_docs
    keys = sorted(source_keys & final_keys)
    ignored_source_only = sorted(doc_key_label(key) for key in source_keys - final_keys)
    ignored_target_only = sorted(doc_key_label(key) for key in final_keys - source_keys)
    return keys, ignored_source_only, ignored_target_only


def compare_question_edits(
    *,
    source_paths: dict[DocKey, Path],
    final_paths: dict[DocKey, Path],
    excluded_docs: set[DocKey] | None = None,
    include_types: bool = False,
) -> ItemEditStats:
    keys, ignored_source_only, ignored_target_only = shared_final_doc_scope(
        source_paths=source_paths,
        final_paths=final_paths,
        excluded_docs=excluded_docs,
    )

    unchanged = changed = added = deleted = docs_with_edits = 0
    for key in keys:
        source_items = question_map(source_paths[key], include_types=include_types)
        target_items = question_map(final_paths[key], include_types=include_types)
        shared_ids = set(source_items) & set(target_items)
        doc_changed = 0
        doc_added = len(set(target_items) - set(source_items))
        doc_deleted = len(set(source_items) - set(target_items))
        doc_unchanged = 0
        for question_id in shared_ids:
            if source_items[question_id] == target_items[question_id]:
                doc_unchanged += 1
            else:
                doc_changed += 1
        unchanged += doc_unchanged
        changed += doc_changed
        added += doc_added
        deleted += doc_deleted
        if doc_changed or doc_added or doc_deleted:
            docs_with_edits += 1

    return ItemEditStats(
        docs_compared=len(keys),
        docs_with_edits=docs_with_edits,
        source_items=unchanged + changed + deleted,
        target_items=unchanged + changed + added,
        unchanged=unchanged,
        changed=changed,
        added=added,
        deleted=deleted,
        ignored_source_only_docs=ignored_source_only,
        ignored_target_only_docs=ignored_target_only,
    )


def compare_entity_edits(
    *,
    source_paths: dict[DocKey, Path],
    final_paths: dict[DocKey, Path],
    excluded_docs: set[DocKey] | None = None,
) -> EntityEditStats:
    keys, ignored_source_only, ignored_target_only = shared_final_doc_scope(
        source_paths=source_paths,
        final_paths=final_paths,
        excluded_docs=excluded_docs,
    )

    unchanged = modified = added = deleted = docs_with_edits = 0
    counts_by_doc: dict[str, int] = {}
    for key in keys:
        source_entities = entity_surface_map(source_paths[key])
        target_entities = entity_surface_map(final_paths[key])
        shared_refs = set(source_entities) & set(target_entities)
        doc_modified = sum(
            1
            for ref in shared_refs
            if source_entities[ref] != target_entities[ref]
        )
        doc_added = len(set(target_entities) - set(source_entities))
        doc_deleted = len(set(source_entities) - set(target_entities))
        doc_unchanged = len(shared_refs) - doc_modified
        doc_total = doc_modified + doc_added + doc_deleted
        unchanged += doc_unchanged
        modified += doc_modified
        added += doc_added
        deleted += doc_deleted
        counts_by_doc[doc_key_label(key)] = doc_total
        if doc_total:
            docs_with_edits += 1

    return EntityEditStats(
        docs_compared=len(keys),
        docs_with_entity_edits=docs_with_edits,
        source_entities=unchanged + modified + deleted,
        target_entities=unchanged + modified + added,
        unchanged_entities=unchanged,
        modified_entities=modified,
        added_entities=added,
        deleted_entities=deleted,
        per_document_edits=summarize_per_document_counts(counts_by_doc),
        ignored_source_only_docs=ignored_source_only,
        ignored_target_only_docs=ignored_target_only,
    )


def compare_rule_set_edits(
    *,
    source_paths: dict[DocKey, Path],
    final_paths: dict[DocKey, Path],
    excluded_docs: set[DocKey] | None = None,
    ignore_comments: bool = False,
) -> RuleSetEditStats:
    keys, ignored_source_only, ignored_target_only = shared_final_doc_scope(
        source_paths=source_paths,
        final_paths=final_paths,
        excluded_docs=excluded_docs,
    )

    unchanged = added = deleted = docs_with_edits = edit_operations = 0
    counts_by_doc: dict[str, int] = {}
    for key in keys:
        source_rules = rule_counter(source_paths[key], ignore_comments=ignore_comments)
        target_rules = rule_counter(final_paths[key], ignore_comments=ignore_comments)
        common = source_rules & target_rules
        added_rules = target_rules - source_rules
        deleted_rules = source_rules - target_rules
        doc_added = sum(added_rules.values())
        doc_deleted = sum(deleted_rules.values())
        doc_edit_operations = max(doc_added, doc_deleted)
        unchanged += sum(common.values())
        added += doc_added
        deleted += doc_deleted
        edit_operations += doc_edit_operations
        counts_by_doc[doc_key_label(key)] = doc_edit_operations
        if added_rules or deleted_rules:
            docs_with_edits += 1

    return RuleSetEditStats(
        docs_compared=len(keys),
        docs_with_rule_set_edits=docs_with_edits,
        source_rules=unchanged + deleted,
        target_rules=unchanged + added,
        unchanged_rules=unchanged,
        added_rules=added,
        deleted_rules=deleted,
        edit_operations=edit_operations,
        per_document_edit_operations=summarize_per_document_counts(counts_by_doc),
        ignored_source_only_docs=ignored_source_only,
        ignored_target_only_docs=ignored_target_only,
    )


def campaign_input_dir(annotation_workspace: Path, review_type: str, campaign_id: int) -> Path:
    return annotation_workspace / "_review_campaigns" / review_type / f"campaign_{campaign_id}" / "input"
