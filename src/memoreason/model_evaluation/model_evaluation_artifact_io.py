"""Artifact identity, freshness, and non-overwriting evaluation I/O."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
import hashlib
import logging
import os
from pathlib import Path
import tempfile

import yaml

from memoreason import PROJECT_ROOT_DIRECTORY
from memoreason.factual_to_fictional_dataset.dataset_paths import (
    FACTUAL_DOCUMENTS_DIR,
    FICTIONAL_DOCUMENTS_DIR,
    MODEL_EVAL_RAW_OUTPUTS_DIR,
    sanitize_model_name,
)
from .benchmark_document_loading import load_evaluation_document
from .document_question_answering_prompt import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    PROMPT_FORMAT_VERSION,
    build_document_question_prompt,
)
from .model_evaluation_run_manifest import ModelEvaluationRunManifest
from .model_evaluation_artifact_audit import audit_saved_output_payload, blocking_issues, format_issue_summary

try:
    YAML_LOADER = yaml.CSafeLoader
    YAML_DUMPER = yaml.CSafeDumper
except AttributeError:  # pragma: no cover
    YAML_LOADER = yaml.SafeLoader
    YAML_DUMPER = yaml.SafeDumper


logger = logging.getLogger(__name__)


def _serialized_source_path(path: Path) -> str:
    """Keep repository paths portable while preserving explicit external roots."""
    try:
        return str(path.relative_to(PROJECT_ROOT_DIRECTORY))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _path_is_outside_project(path: Path) -> bool:
    if not path.is_absolute():
        return False
    try:
        path.relative_to(PROJECT_ROOT_DIRECTORY)
    except ValueError:
        return True
    return False


def _external_payload_is_current(
    payload: dict,
    *,
    check_score_metadata: bool,
) -> tuple[bool, str]:
    """Validate outputs whose source document intentionally lives outside the checkout.

    ``version_audit`` predates external immutable dataset roots and serializes
    repository-relative source paths. Keep its richer diagnostics for normal
    repository runs while applying the same consistency contract here.
    """
    source_document_path = str(payload.get("source_document_path") or "").strip()
    if not source_document_path:
        return False, "missing source_document_path"
    source_path = Path(source_document_path)
    if not source_path.is_absolute() or not source_path.is_file():
        return False, f"missing external source document: {source_document_path!r}"

    source_document = load_evaluation_document(source_path)
    expected_payload_fields = {
        "source_document_path": _serialized_source_path(source_document.source_path),
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
    for key, expected in expected_payload_fields.items():
        if payload.get(key) != expected:
            return False, f"{key} mismatch"

    results = payload.get("results") or []
    if not isinstance(results, list):
        return False, "results is not a list"
    questions_by_id = {question.question_id: question for question in source_document.questions}
    seen_question_ids: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            return False, "result is not a mapping"
        question_id = str(result.get("question_id") or "").strip()
        source_question = questions_by_id.get(question_id)
        if source_question is None or question_id in seen_question_ids:
            return False, f"unknown or duplicate question_id: {question_id!r}"
        seen_question_ids.add(question_id)
        expected_prompt = build_document_question_prompt(
            source_document.document_text,
            source_question.question_text,
            answer_schema=source_question.answer_schema,
        )
        if str(result.get("user_prompt") or "") != expected_prompt:
            return False, f"user_prompt mismatch for {question_id}"
        if check_score_metadata:
            expected_score_metadata = {
                "question_type": source_question.question_type,
                "answer_behavior": source_question.answer_behavior,
                "question_text": source_question.question_text,
                "ground_truth": source_question.ground_truth,
                "ground_truth_canonical": source_question.ground_truth_canonical,
                "answer_schema": source_question.answer_schema,
                "answer_expression": source_question.answer_expression,
                "accepted_answer_overrides": list(source_question.accepted_answer_overrides),
                "accepted_answers": list(source_question.accepted_answers),
                "accepted_answers_canonical": list(source_question.accepted_answers_canonical),
                "pair_key": source_question.pair_key,
            }
            for key, expected in expected_score_metadata.items():
                if result.get(key) != expected:
                    return False, f"{key} mismatch for {question_id}"

    if seen_question_ids != set(questions_by_id):
        return False, "saved output does not cover the current question set"
    return True, ""


def _drop_downstream_heavy_fields(result: dict) -> dict:
    """Keep raw-provider payloads in raw outputs only; downstream stages need the answer fields."""
    return {key: value for key, value in result.items() if key != "raw_provider_response"}


def _resolved_source_document_path(source_document_path: str) -> Path:
    source_path = Path(source_document_path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT_DIRECTORY / source_path
    if not source_path.exists():
        # Evaluation artifacts can be moved between machines while retaining the
        # absolute source path recorded by the machine that generated them. Rebase
        # such stale paths onto the configured immutable dataset roots, just as we
        # already do for repository-relative paths.
        path_parts = Path(source_document_path).parts
        for directory_name, configured_root in (
            ("FACTUAL_DOCUMENTS", FACTUAL_DOCUMENTS_DIR),
            ("FICTIONAL_DOCUMENTS", FICTIONAL_DOCUMENTS_DIR),
        ):
            if directory_name not in path_parts:
                continue
            directory_index = path_parts.index(directory_name)
            source_path = configured_root.joinpath(*path_parts[directory_index + 1 :])
            break
    return source_path


@cache
def _source_questions_by_id(source_document_path: str) -> dict[str, object]:
    if not source_document_path:
        return {}
    source_path = _resolved_source_document_path(source_document_path)
    if not source_path.exists():
        return {}
    document = load_evaluation_document(source_path)
    return {question.question_id: question for question in document.questions}


def _resolve_source_question(payload: dict, result: dict) -> object | None:
    source_document_path = str(payload.get("source_document_path") or "").strip()
    question_id = str(result.get("question_id") or "").strip()
    if not source_document_path or not question_id:
        return None
    return _source_questions_by_id(source_document_path).get(question_id)


def _should_skip_deleted_question(source_question: object | None) -> bool:
    # If a question no longer exists in the source template, drop stale raw artifacts
    # instead of silently rescoring them from outdated stored metadata.
    return source_question is None


def _should_preserve_raw_metadata(result: dict, source_question: object | None) -> bool:
    if source_question is None:
        return False
    question_id = str(result.get("question_id") or "").strip()
    raw_question_text = str(result.get("question_text") or "").strip().lower()
    source_question_text = str(getattr(source_question, "question_text", "") or "").strip().lower()
    if question_id == "awards_11_q05" and "including" in raw_question_text and "excluding" in source_question_text:
        # This question changed semantics after the original model calls. Keep the
        # historical raw metadata until the question is rerun.
        return True
    return False


def _write_text_exclusive(text: str, output_path: Path) -> Path:
    """Atomically publish one complete text artifact without overwriting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.tmp.",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite model-evaluation artifact: {output_path}"
            ) from exc
        directory_descriptor = os.open(output_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _write_yaml(payload: dict, output_path: Path) -> Path:
    serialized = yaml.dump(
        payload,
        Dumper=YAML_DUMPER,
        sort_keys=False,
        allow_unicode=True,
        width=10000,
    )
    if os.environ.get("MEMOREASON_EXCLUSIVE_ARTIFACT_WRITES") == "1":
        return _write_text_exclusive(serialized, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def _read_yaml_payload(path: Path) -> dict:
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=YAML_LOADER)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping at {path}, got {type(payload).__name__}.")
    return payload


def _payload_currentness(
    payload: dict,
    *,
    path: Path,
    check_score_metadata: bool,
) -> tuple[bool, str]:
    source_path = Path(str(payload.get("source_document_path") or ""))
    if _path_is_outside_project(source_path):
        return _external_payload_is_current(
            payload,
            check_score_metadata=check_score_metadata,
        )
    issues = blocking_issues(
        audit_saved_output_payload(
            payload,
            path=path,
            check_score_metadata=check_score_metadata,
        )
    )
    if issues:
        return False, format_issue_summary(issues)
    return True, ""


def _saved_payload_is_current(
    path: Path,
    *,
    check_score_metadata: bool,
) -> bool:
    payload = _read_yaml_payload(path)
    is_current, detail = _payload_currentness(
        payload,
        path=path,
        check_score_metadata=check_score_metadata,
    )
    if not is_current:
        logger.warning("Will refresh stale %s: %s", path, detail)
    return is_current


def _remove_downstream_stage_artifacts(raw_output_path: Path) -> None:
    """Remove parse/evaluate siblings after a raw prompt-version refresh."""
    parsed_output_path = raw_output_path.with_name(
        raw_output_path.name.replace("_raw_outputs.yaml", "_parsed_outputs.yaml")
    )
    evaluated_output_path = _evaluated_output_path(parsed_output_path)
    for path in (parsed_output_path, evaluated_output_path):
        if path.exists():
            path.unlink()


def _evaluated_output_path(parsed_output_path: Path) -> Path:
    return parsed_output_path.with_name(
        parsed_output_path.name.replace("_parsed_outputs.yaml", "_evaluated_outputs.yaml")
    )


def _parsed_output_path(evaluated_output_path: Path) -> Path:
    if not evaluated_output_path.name.endswith("_evaluated_outputs.yaml"):
        raise ValueError(f"Not an evaluated-output artifact path: {evaluated_output_path}")
    return evaluated_output_path.with_name(
        evaluated_output_path.name.replace("_evaluated_outputs.yaml", "_parsed_outputs.yaml")
    )


def _remove_evaluated_stage_artifact(parsed_output_path: Path) -> None:
    evaluated_output_path = _evaluated_output_path(parsed_output_path)
    if evaluated_output_path.exists():
        evaluated_output_path.unlink()


def _attach_reproducibility_manifest_reference(
    payload: dict,
    *,
    reproducibility_manifest: ModelEvaluationRunManifest | None,
    stage_name: str,
) -> dict:
    if reproducibility_manifest is None:
        return payload
    return {
        **payload,
        "reproducibility_manifest": reproducibility_manifest.stage_reference(stage_name),
    }


def _iter_stage_paths(
    *,
    stage_suffix: str,
    model_names: Sequence[str] | None = None,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
    settings: Sequence[str] | None = None,
) -> list[Path]:
    model_filter = {sanitize_model_name(name) for name in (model_names or [])}
    theme_filter = set(themes or [])
    document_filter = set(document_ids or [])
    setting_filter = {str(setting).strip().lower() for setting in (settings or []) if str(setting).strip()}

    matched_paths: list[Path] = []
    for artifact_path in sorted(MODEL_EVAL_RAW_OUTPUTS_DIR.rglob(f"*_{stage_suffix}.yaml")):
        theme_name = artifact_path.parents[1].name
        model_folder = artifact_path.parent.name
        if theme_filter and theme_name not in theme_filter:
            continue
        if model_filter and model_folder not in model_filter:
            continue
        if document_filter:
            if not any(artifact_path.name.startswith(f"{document_id}_") for document_id in document_filter):
                continue
        if setting_filter:
            payload = _read_yaml_payload(artifact_path)
            document_setting = str(payload.get("document_setting") or "").strip().lower()
            if document_setting not in setting_filter:
                continue
        matched_paths.append(artifact_path)
    return matched_paths
