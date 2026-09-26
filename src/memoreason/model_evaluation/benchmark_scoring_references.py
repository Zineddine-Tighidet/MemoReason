"""Load immutable benchmark golds and bind them to the original question inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from memoreason import PROJECT_ROOT_DIRECTORY

from .answer_normalization import canonicalize_answer
from .answer_schema_data_contracts import ANSWER_SCHEMAS


DEFAULT_SCORING_REFERENCES = PROJECT_ROOT_DIRECTORY / "data" / "scoring_references.jsonl.gz"
EXPECTED_REFERENCE_COUNT = 109_200
ReferenceKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class ScoringReference:
    """A scoring contract, independent of any model answer or decision."""

    document_text_sha256: str
    question_text_sha256: str
    frozen_ground_truth: str
    ground_truth: str
    ground_truth_canonical: str
    answer_expression: str
    answer_schema: str
    accepted_answers: tuple[str, ...]
    accepted_answers_canonical: tuple[str, ...]
    accepted_answer_overrides: tuple[str, ...]


@dataclass(frozen=True)
class ScoringReferences:
    records: Mapping[ReferenceKey, ScoringReference]
    document_ids: frozenset[str]

    def match(
        self,
        *,
        setting: str,
        document_id: str,
        variant_id: str,
        question_id: str,
        document_text: str,
        question_text: str,
        frozen_ground_truth: str,
    ) -> ScoringReference | None:
        """Return bound golds, allowing generated golds only for new documents."""
        if document_id not in self.document_ids:
            return None
        key = (setting, document_id, variant_id, question_id)
        reference = self.records.get(key)
        if reference is None:
            raise ValueError(f"Missing scoring reference for benchmark question {key!r}")
        for label, text, expected in (
            ("document content", document_text, reference.document_text_sha256),
            ("question content", question_text, reference.question_text_sha256),
        ):
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
                raise ValueError(f"Scoring reference/{label} mismatch for {key!r}")
        if frozen_ground_truth != reference.frozen_ground_truth:
            raise ValueError(f"Scoring reference/frozen-gold mismatch for {key!r}")
        return reference


def _string(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ValueError(f"Scoring reference field {field!r} must be a string")
    return value


def _strings(row: dict, field: str) -> tuple[str, ...]:
    value = row.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Scoring reference field {field!r} must be a list of strings")
    return tuple(value)


def _reference(row: dict) -> ScoringReference:
    schema = _string(row, "answer_schema")
    if schema not in ANSWER_SCHEMAS:
        raise ValueError(f"Unknown scoring reference answer schema {schema!r}")
    for field in ("document_text_sha256", "question_text_sha256"):
        digest = _string(row, field)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"Invalid scoring reference hash in {field!r}")
    gold = _string(row, "ground_truth")
    accepted = _strings(row, "accepted_answers")
    canonical_gold = (
        _string(row, "ground_truth_canonical")
        if "ground_truth_canonical" in row
        else canonicalize_answer(schema, gold)
    )
    if "accepted_answers_canonical" in row:
        canonical_accepted = _strings(row, "accepted_answers_canonical")
    else:
        canonical_accepted = tuple(dict.fromkeys(
            value for answer in accepted if (value := canonicalize_answer(schema, answer))
        ))
        if not canonical_accepted and canonical_gold:
            canonical_accepted = (canonical_gold,)
    return ScoringReference(
        document_text_sha256=row["document_text_sha256"],
        question_text_sha256=row["question_text_sha256"],
        frozen_ground_truth=_string(row, "frozen_ground_truth"),
        ground_truth=gold,
        ground_truth_canonical=canonical_gold,
        answer_expression=_string(row, "answer_expression"),
        answer_schema=schema,
        accepted_answers=accepted,
        accepted_answers_canonical=canonical_accepted,
        accepted_answer_overrides=_strings(row, "accepted_answer_overrides"),
    )


@lru_cache(maxsize=2)
def _load(path: Path, size: int, mtime_ns: int) -> ScoringReferences:
    # File identity participates in the cache key; no manifest or model outputs
    # are needed to load the active benchmark golds.
    records: dict[ReferenceKey, ScoringReference] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Scoring reference must be an object")
                key = tuple(_string(row, field) for field in ("setting", "document_id", "variant_id", "question_id"))
                if any(not part for part in key):
                    raise ValueError("Scoring reference key contains an empty identifier")
                if key in records:
                    raise ValueError(f"Duplicate scoring reference key {key!r}")
                records[key] = _reference(row)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"Empty scoring reference file: {path}")
    return ScoringReferences(MappingProxyType(records), frozenset(key[1] for key in records))


def load_scoring_references(path: Path | None = None) -> ScoringReferences:
    """Load the active snapshot; explicit paths also support small test fixtures."""
    is_default = path is None
    source = (DEFAULT_SCORING_REFERENCES if path is None else Path(path)).resolve()
    stat = source.stat()  # A missing file must fail even after a previous cached load.
    references = _load(source, stat.st_size, stat.st_mtime_ns)
    if is_default and len(references.records) != EXPECTED_REFERENCE_COUNT:
        raise ValueError(
            f"Incomplete benchmark scoring references: expected {EXPECTED_REFERENCE_COUNT}, "
            f"found {len(references.records)} in {source}"
        )
    return references
