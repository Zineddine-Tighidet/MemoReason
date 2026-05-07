"""Audit saved Parametric Shortcut outputs against the current dataset files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from src.dataset_export.dataset_paths import PROJECT_ROOT

from .dataset import EvaluationDocument, EvaluationQuestion, load_evaluation_document
from .prompting import DOCUMENT_QA_SYSTEM_PROMPT, PROMPT_FORMAT_VERSION, build_document_question_prompt


@dataclass(frozen=True)
class OutputVersionIssue:
    """One consistency issue found in a saved model-output payload."""

    issue_type: str
    severity: str
    path: str
    document_id: str
    document_setting: str
    document_variant_id: str
    question_id: str
    detail: str

    def to_payload(self) -> dict[str, str]:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "path": self.path,
            "document_id": self.document_id,
            "document_setting": self.document_setting,
            "document_variant_id": self.document_variant_id,
            "question_id": self.question_id,
            "detail": self.detail,
        }


@lru_cache(maxsize=None)
def _load_source_document(source_document_path: str) -> EvaluationDocument | None:
    if not source_document_path:
        return None
    path = PROJECT_ROOT / source_document_path
    if not path.exists():
        return None
    return load_evaluation_document(path)


def _rel_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _issue(
    *,
    payload: dict,
    path: Path | None,
    issue_type: str,
    severity: str = "error",
    question_id: str = "",
    detail: str,
) -> OutputVersionIssue:
    return OutputVersionIssue(
        issue_type=issue_type,
        severity=severity,
        path=_rel_path(path),
        document_id=str(payload.get("document_id") or ""),
        document_setting=str(payload.get("document_setting") or ""),
        document_variant_id=str(payload.get("document_variant_id") or ""),
        question_id=question_id,
        detail=detail,
    )


def _metadata_mismatches(result: dict, source_question: EvaluationQuestion) -> list[tuple[str, str]]:
    checks = {
        "question_type": source_question.question_type,
        "answer_behavior": source_question.answer_behavior,
        "question_text": source_question.question_text,
        "ground_truth": source_question.ground_truth,
        "ground_truth_canonical": source_question.ground_truth_canonical,
        "answer_schema": source_question.answer_schema,
        "answer_expression": source_question.answer_expression,
        "accepted_answers": list(source_question.accepted_answers),
        "accepted_answers_canonical": list(source_question.accepted_answers_canonical),
        "accepted_answer_overrides": list(source_question.accepted_answer_overrides),
        "pair_key": source_question.pair_key,
    }
    mismatches: list[tuple[str, str]] = []
    for key, expected in checks.items():
        actual = result.get(key)
        if actual != expected:
            mismatches.append((key, f"expected={expected!r} actual={actual!r}"))
    return mismatches


def audit_saved_output_payload(
    payload: dict,
    *,
    path: Path | None = None,
    check_score_metadata: bool = True,
) -> list[OutputVersionIssue]:
    """Return current-dataset consistency issues for one saved output payload.

    A raw model answer is only reusable if its stored prompt exactly matches the
    prompt rebuilt from the current source document/question. Parsed/evaluated
    outputs additionally need current scoring metadata because ground-truth fixes
    can change exact-match and Judge Match decisions without changing the prompt.
    """

    issues: list[OutputVersionIssue] = []
    source_document_path = str(payload.get("source_document_path") or "").strip()
    source_document = _load_source_document(source_document_path)
    if source_document is None:
        issues.append(
            _issue(
                payload=payload,
                path=path,
                issue_type="missing_source_document",
                question_id="",
                detail=f"source_document_path={source_document_path!r}",
            )
        )
        return issues

    expected_source_path = str(source_document.source_path.relative_to(PROJECT_ROOT))
    if source_document_path != expected_source_path:
        issues.append(
            _issue(
                payload=payload,
                path=path,
                issue_type="source_document_path_mismatch",
                question_id="",
                detail=f"expected={expected_source_path!r} actual={source_document_path!r}",
            )
        )

    payload_checks = {
        "document_id": source_document.document_id,
        "document_theme": source_document.document_theme,
        "document_setting": source_document.document_setting,
        "document_setting_family": source_document.document_setting_family,
        "document_variant_id": source_document.document_variant_id,
        "document_variant_index": source_document.document_variant_index,
        "replacement_proportion": source_document.replacement_proportion,
        "prompt_format_version": PROMPT_FORMAT_VERSION,
        "system_prompt": DOCUMENT_QA_SYSTEM_PROMPT,
    }
    for key, expected in payload_checks.items():
        actual = payload.get(key)
        if actual != expected:
            issues.append(
                _issue(
                    payload=payload,
                    path=path,
                    issue_type=f"{key}_mismatch",
                    question_id="",
                    detail=f"expected={expected!r} actual={actual!r}",
                )
            )

    questions_by_id = {question.question_id: question for question in source_document.questions}
    seen_question_ids: set[str] = set()
    results = payload.get("results") or []
    if not isinstance(results, list):
        issues.append(
            _issue(
                payload=payload,
                path=path,
                issue_type="results_not_list",
                question_id="",
                detail=f"type={type(results).__name__}",
            )
        )
        return issues

    for result in results:
        if not isinstance(result, dict):
            issues.append(
                _issue(
                    payload=payload,
                    path=path,
                    issue_type="result_not_mapping",
                    question_id="",
                    detail=f"type={type(result).__name__}",
                )
            )
            continue
        question_id = str(result.get("question_id") or "").strip()
        source_question = questions_by_id.get(question_id)
        if source_question is None:
            issues.append(
                _issue(
                    payload=payload,
                    path=path,
                    issue_type="question_missing_from_current_source",
                    question_id=question_id,
                    detail="saved output row has no matching current question",
                )
            )
            continue
        seen_question_ids.add(question_id)

        expected_prompt = build_document_question_prompt(
            source_document.document_text,
            source_question.question_text,
            answer_schema=source_question.answer_schema,
        )
        actual_prompt = str(result.get("user_prompt") or "")
        if actual_prompt != expected_prompt:
            issues.append(
                _issue(
                    payload=payload,
                    path=path,
                    issue_type="user_prompt_mismatch",
                    question_id=question_id,
                    detail=(
                        f"expected_len={len(expected_prompt)} actual_len={len(actual_prompt)}; "
                        "raw output was generated from a different document/question version"
                    ),
                )
            )

        if check_score_metadata:
            for key, detail in _metadata_mismatches(result, source_question):
                issues.append(
                    _issue(
                        payload=payload,
                        path=path,
                        issue_type=f"{key}_mismatch",
                        question_id=question_id,
                        detail=detail,
                    )
                )

    missing_questions = sorted(set(questions_by_id) - seen_question_ids)
    for question_id in missing_questions:
        issues.append(
            _issue(
                payload=payload,
                path=path,
                issue_type="question_missing_from_saved_output",
                question_id=question_id,
                detail="current source question has no saved model output row",
            )
        )

    return issues


def blocking_issues(issues: Iterable[OutputVersionIssue]) -> list[OutputVersionIssue]:
    return [issue for issue in issues if issue.severity == "error"]


def summarize_issues(issues: Iterable[OutputVersionIssue]) -> dict[str, int]:
    return dict(Counter(issue.issue_type for issue in issues))


def format_issue_summary(issues: Iterable[OutputVersionIssue], *, max_examples: int = 5) -> str:
    issue_list = list(issues)
    counts = summarize_issues(issue_list)
    parts = [f"{issue_type}={count}" for issue_type, count in sorted(counts.items())]
    examples = issue_list[:max_examples]
    if examples:
        parts.append(
            "examples="
            + "; ".join(
                f"{issue.path}:{issue.question_id or issue.document_id}:{issue.issue_type}"
                for issue in examples
            )
        )
    return ", ".join(parts) if parts else "no issues"


__all__ = [
    "OutputVersionIssue",
    "audit_saved_output_payload",
    "blocking_issues",
    "format_issue_summary",
    "summarize_issues",
]
