"""Load and hash frozen manifests, reviewed contracts, and evaluated outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re
from typing import Any

import yaml

from .paper_results_data_model import (
    ANSWER_BEHAVIORS,
    PAPER_REPLACEMENT_SETTINGS,
    PAPER_FICTIONAL_VARIANT_IDS,
    QUESTION_TYPES,
    FrozenEvaluatedAnswersSelection,
    ReviewedQuestionMetadata,
    PaperResultsManifest,
    EvaluatedAnswer,
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(payload).__name__}.")
    return payload


def _resolve_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


_ANNOTATED_SPAN = re.compile(r"\[([^;\]]+);\s*[^\]]+\]")


def _normalized_text(value: Any) -> str:
    collapsed = " ".join(str(value or "").split())
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", collapsed)


def _render_factual_annotation(value: Any) -> str:
    return _normalized_text(_ANNOTATED_SPAN.sub(r"\1", str(value or "")))


def _load_reviewed_question_metadata(
    manifest_path: Path,
    reporting: Mapping[str, Any],
) -> tuple[dict[str, ReviewedQuestionMetadata], Path, str, str]:
    raw_spec = reporting.get("dataset_statistics")
    if not isinstance(raw_spec, Mapping):
        raise ValueError(
            "reporting.dataset_statistics is required because human-reviewed templates "
            "are the authoritative question contracts."
        )
    raw_root = str(raw_spec.get("root") or "").strip()
    pattern = str(raw_spec.get("glob") or "").strip()
    expected_sha = str(raw_spec.get("sha256") or "").strip().lower()
    if not raw_root or not pattern:
        raise ValueError("reporting.dataset_statistics.root and glob are required.")
    root = _resolve_path(manifest_path, raw_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Human-template root does not exist: {root}")
    pattern_path = Path(pattern)
    if pattern_path.is_absolute() or ".." in pattern_path.parts:
        raise ValueError("reporting.dataset_statistics.glob must be root-relative.")
    excluded = {Path(str(value)).as_posix() for value in (raw_spec.get("exclude_relative_paths") or [])}
    paths = tuple(
        path.resolve()
        for path in sorted(root.glob(pattern))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    if not paths:
        raise FileNotFoundError(f"No human templates match {pattern!r} below {root}.")
    actual_sha, _ = evaluated_output_tree_sha256(root, paths)
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("reporting.dataset_statistics.sha256 must be explicit.")
    if actual_sha != expected_sha:
        raise ValueError(f"Human-template tree SHA-256 mismatch for {root}: expected {expected_sha}, got {actual_sha}.")

    reviewed_questions_by_pair_key: dict[str, ReviewedQuestionMetadata] = {}
    for path in paths:
        payload = _read_yaml(path)
        document = payload.get("document")
        if not isinstance(document, Mapping):
            raise ValueError(f"{path}: reviewed template has no document mapping.")
        relative = path.relative_to(root)
        document_theme = relative.parent.as_posix()
        document_id = str(document.get("document_id") or path.stem).strip()
        questions = document.get("questions")
        if not isinstance(questions, list):
            raise ValueError(f"{path}: reviewed template questions must be a list.")
        for index, question in enumerate(questions):
            if not isinstance(question, Mapping):
                raise ValueError(f"{path}: reviewed question {index} is not a mapping.")
            question_id = str(question.get("question_id") or "").strip()
            question_type = str(question.get("question_type") or "").strip().lower()
            answer_behavior = str(question.get("answer_type") or "").strip().lower()
            if not question_id:
                raise ValueError(f"{path}: reviewed question {index} has no question_id.")
            if question_type not in QUESTION_TYPES:
                raise ValueError(f"{path}: invalid human question_type {question_type!r} for {question_id}.")
            if answer_behavior not in ANSWER_BEHAVIORS:
                raise ValueError(f"{path}: invalid human answer_type {answer_behavior!r} for {question_id}.")
            key = f"{document_theme}::{document_id}::{question_id}"
            if key in reviewed_questions_by_pair_key:
                raise ValueError(f"Duplicate human question contract: {key}.")
            reviewed_questions_by_pair_key[key] = ReviewedQuestionMetadata(
                source_path=path,
                document_theme=document_theme,
                document_id=document_id,
                question_id=question_id,
                question_text_factual=_render_factual_annotation(question.get("question")),
                question_type=question_type,
                answer_behavior=answer_behavior,
            )
    return reviewed_questions_by_pair_key, root, actual_sha, expected_sha


def evaluated_output_tree_sha256(
    root: Path,
    paths: Sequence[Path],
) -> tuple[str, dict[str, str]]:
    """Hash a selected evaluated-output tree by relative path and file digest."""
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Evaluated-output root does not exist: {resolved_root}")
    file_hashes: dict[str, str] = {}
    for candidate in sorted(paths, key=lambda item: item.as_posix()):
        path = candidate.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Evaluated output does not exist: {path}")
        try:
            relative_path = path.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Evaluated output is outside its declared selection root {resolved_root}: {path}"
            ) from exc
        if relative_path in file_hashes:
            raise ValueError(f"Evaluated output selected more than once below {resolved_root}: {path}")
        file_hashes[relative_path] = sha256_file(path)

    if not file_hashes:
        raise ValueError(f"Cannot hash an empty evaluated-output selection below {resolved_root}.")
    hasher = hashlib.sha256()
    for relative_path, digest in sorted(file_hashes.items()):
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
    return hasher.hexdigest(), file_hashes


def _optional_sha256(value: Any, *, field: str) -> str | None:
    digest = str(value or "").strip().lower() or None
    if digest is not None and (len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)):
        raise ValueError(f"{field} must be a 64-character hexadecimal digest.")
    return digest


def _expand_evaluated_input(
    manifest_path: Path,
    entry: str | Mapping[str, Any],
) -> tuple[list[tuple[Path, str]], FrozenEvaluatedAnswersSelection]:
    if isinstance(entry, str):
        raw_path = entry
        expected_sha = None
        pattern = None
    elif isinstance(entry, Mapping):
        raw_path = str(entry.get("path") or "").strip()
        expected_sha = _optional_sha256(entry.get("sha256"), field="reporting.evaluated_outputs[].sha256")
        pattern = str(entry.get("glob") or "").strip() or None
    else:
        raise ValueError("Each reporting.evaluated_outputs entry must be a path or mapping.")
    if not raw_path:
        raise ValueError("An evaluated-output entry is missing its path.")

    path = _resolve_path(manifest_path, raw_path)
    if path.is_dir():
        if not pattern:
            raise ValueError(f"Directory input {path} requires an explicit glob pattern.")
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise ValueError(f"Directory input {path} requires a root-relative glob pattern.")
        matches = sorted(candidate.resolve() for candidate in path.glob(pattern) if candidate.is_file())
        if not matches:
            raise FileNotFoundError(f"No evaluated outputs match {pattern!r} below {path}.")
        selection_sha, file_hashes = evaluated_output_tree_sha256(path, matches)
        if expected_sha and selection_sha != expected_sha:
            raise ValueError(
                f"Evaluated-output selection SHA-256 mismatch for {path} ({pattern!r}): "
                f"expected {expected_sha}, got {selection_sha}."
            )
        selection = FrozenEvaluatedAnswersSelection(
            path=path,
            glob=pattern,
            file_count=len(matches),
            sha256=selection_sha,
            expected_sha256=expected_sha,
        )
        return (
            [(candidate, file_hashes[candidate.relative_to(path).as_posix()]) for candidate in matches],
            selection,
        )
    if not path.is_file():
        raise FileNotFoundError(f"Evaluated output does not exist: {path}")
    actual_sha = sha256_file(path)
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha}, got {actual_sha}.")
    selection = FrozenEvaluatedAnswersSelection(
        path=path,
        glob=None,
        file_count=1,
        sha256=actual_sha,
        expected_sha256=expected_sha,
    )
    return [(path, actual_sha)], selection


def load_paper_results_manifest(manifest_path: Path) -> PaperResultsManifest:
    """Load and validate the explicit reporting inputs from one run manifest."""
    manifest_path = manifest_path.expanduser().resolve()
    payload = _read_yaml(manifest_path)
    reporting = payload.get("reporting")
    if not isinstance(reporting, Mapping):
        raise ValueError(f"{manifest_path} has no reporting mapping.")

    (
        reviewed_questions_by_pair_key,
        reviewed_templates_root,
        reviewed_templates_tree_sha256,
        expected_reviewed_templates_tree_sha256,
    ) = _load_reviewed_question_metadata(manifest_path, reporting)

    raw_inputs = reporting.get("evaluated_outputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise ValueError("reporting.evaluated_outputs must be a non-empty list.")

    expanded: list[tuple[Path, str]] = []
    selections: list[FrozenEvaluatedAnswersSelection] = []
    for entry in raw_inputs:
        entry_paths, selection = _expand_evaluated_input(manifest_path, entry)
        expanded.extend(entry_paths)
        selections.append(selection)

    paths: list[Path] = []
    hashes: dict[str, str] = {}
    seen: set[Path] = set()
    for path, actual_sha in expanded:
        if path in seen:
            raise ValueError(f"Evaluated output selected more than once: {path}")
        seen.add(path)
        paths.append(path)
        hashes[str(path)] = actual_sha

    raw_cache = reporting.get("frozen_judge_cache")
    cache_path: Path | None = None
    cache_sha: str | None = None
    expected_cache_sha: str | None = None
    judge_model = "openai/gpt-oss-120b"
    if raw_cache is not None:
        if not isinstance(raw_cache, Mapping) or not str(raw_cache.get("path") or "").strip():
            raise ValueError("reporting.frozen_judge_cache must contain a path.")
        cache_path = _resolve_path(manifest_path, str(raw_cache["path"]))
        if not cache_path.is_file():
            raise FileNotFoundError(f"Frozen judge cache does not exist: {cache_path}")
        cache_sha = sha256_file(cache_path)
        expected_cache_sha = _optional_sha256(raw_cache.get("sha256"), field="reporting.frozen_judge_cache.sha256")
        if expected_cache_sha and cache_sha != expected_cache_sha:
            raise ValueError(f"SHA-256 mismatch for {cache_path}: expected {expected_cache_sha}, got {cache_sha}.")
        judge_model = str(raw_cache.get("judge_model") or judge_model).strip()

    variant_ids = tuple(
        str(value).strip() for value in reporting.get("expected_variant_ids", PAPER_FICTIONAL_VARIANT_IDS)
    )
    if not variant_ids or any(not value for value in variant_ids) or len(set(variant_ids)) != len(variant_ids):
        raise ValueError("reporting.expected_variant_ids must contain distinct non-empty ids.")

    settings = tuple(str(value).strip() for value in reporting.get("settings", PAPER_REPLACEMENT_SETTINGS))
    if not settings or settings[0] != "factual" or len(set(settings)) != len(settings):
        raise ValueError("reporting.settings must be distinct and start with factual.")

    output_dir_raw = str(reporting.get("output_dir") or "").strip()
    if not output_dir_raw:
        raise ValueError("reporting.output_dir is required; implicit output paths are forbidden.")
    output_dir = _resolve_path(manifest_path, output_dir_raw)

    raw_models = reporting.get("models") or {}
    if not isinstance(raw_models, Mapping):
        raise ValueError("reporting.models must be a mapping.")
    models = {
        str(group): tuple(str(model).strip() for model in values if str(model).strip())
        for group, values in raw_models.items()
        if isinstance(values, (list, tuple))
    }

    return PaperResultsManifest(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        evaluated_output_paths=tuple(paths),
        evaluated_output_hashes=hashes,
        frozen_judge_cache_path=cache_path,
        frozen_judge_cache_sha256=cache_sha,
        judge_model=judge_model,
        variant_ids=variant_ids,
        settings=settings,
        output_dir=output_dir,
        models=models,
        reviewed_questions_by_pair_key=reviewed_questions_by_pair_key,
        reviewed_templates_root=reviewed_templates_root,
        reviewed_templates_tree_sha256=reviewed_templates_tree_sha256,
        expected_reviewed_templates_tree_sha256=(expected_reviewed_templates_tree_sha256),
        frozen_evaluated_answer_selections=tuple(selections),
        expected_frozen_judge_cache_sha256=expected_cache_sha,
    )


def resolve_models(
    paper_results_manifest: PaperResultsManifest,
    group: str,
    evaluated_answers: Sequence[EvaluatedAnswer],
) -> tuple[str, ...]:
    configured = paper_results_manifest.models.get(group)
    if configured:
        return configured
    discovered = tuple(sorted({answer.model_name for answer in evaluated_answers}))
    if not discovered:
        raise ValueError(f"No models configured or discovered for {group}.")
    return discovered
