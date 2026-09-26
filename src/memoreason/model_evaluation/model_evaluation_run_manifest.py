"""Evaluation reproducibility manifests for benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import os
import platform
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import traceback
from typing import Any
import uuid

import yaml

from memoreason import PROJECT_ROOT_DIRECTORY
from memoreason.factual_to_fictional_dataset.dataset_settings import order_dataset_settings
from memoreason.factual_to_fictional_dataset.dataset_paths import (
    FACTUAL_DOCUMENTS_DIR,
    FICTIONAL_DOCUMENTS_DIR,
    MODEL_EVAL_DIR,
    reproducibility_manifest_path,
)
from .answer_schema_data_contracts import ANSWER_PARSER_VERSION
from .benchmark_document_loading import (
    EXCLUDED_EVALUATION_DOCUMENT_IDS,
    BenchmarkDocumentForEvaluation,
    iter_evaluation_documents,
)
from .document_question_answering_prompt import (
    DOCUMENT_QA_SYSTEM_PROMPT,
    GENERATION_TOKEN_BUDGET_POLICY_VERSION,
    JUDGE_SYSTEM_PROMPT,
    PROMPT_FORMAT_VERSION,
)
from .paper_model_registry import PaperModelConfiguration, resolve_paper_model_configurations
from .exact_and_judge_match_scoring import (
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MIN_MAX_TOKENS,
    JUDGE_RETRY_DELAY_SECONDS,
    JUDGE_TOKEN_GROWTH_FACTOR,
    SCORING_PROTOCOL_VERSION,
    JudgeMatchConfiguration,
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_timestamp_string(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT_DIRECTORY))
    except ValueError:
        return str(path)


def _input_file_fingerprint(path: Path | None, *, label: str) -> dict[str, Any] | None:
    """Resolve and fingerprint one optional file that defines run inputs."""
    if path is None:
        return None
    expanded_path = path.expanduser()
    if not expanded_path.is_absolute():
        expanded_path = PROJECT_ROOT_DIRECTORY / expanded_path
    try:
        resolved_path = expanded_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {expanded_path}") from exc
    if not resolved_path.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved_path}")
    return {
        "path": _relative_to_repo(resolved_path),
        "sha256": _sha256_file(resolved_path),
        "size_bytes": resolved_path.stat().st_size,
    }


def _git_snapshot() -> dict[str, Any] | None:
    if os.environ.get("MEMOREASON_DISABLE_GIT_SNAPSHOT") == "1":
        manifest_sha256 = str(os.environ.get("MEMOREASON_CODE_MANIFEST_SHA256") or "")
        invalid_character = any(
            character not in "0123456789abcdef" for character in manifest_sha256
        )
        if len(manifest_sha256) != 64 or invalid_character:
            raise RuntimeError(
                "MEMOREASON_DISABLE_GIT_SNAPSHOT=1 requires a lowercase SHA-256 in "
                "MEMOREASON_CODE_MANIFEST_SHA256"
            )
        return {
            "source": "pinned_code_file_manifest",
            "code_manifest_sha256": manifest_sha256,
            "git_probe_performed": False,
        }

    def _run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT_DIRECTORY,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        if completed.returncode != 0:
            return None
        output = completed.stdout.strip()
        return output or None

    commit = _run_git("rev-parse", "HEAD")
    if commit is None:
        return None

    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    dirty_status = _run_git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "is_dirty": bool(dirty_status),
    }


@dataclass(frozen=True)
class ArtifactFingerprint:
    """One saved artifact tracked by a reproducibility manifest."""

    relative_path: str
    sha256: str
    size_bytes: int

    @classmethod
    def from_path(cls, artifact_path: Path) -> ArtifactFingerprint:
        return cls(
            relative_path=_relative_to_repo(artifact_path),
            sha256=_sha256_file(artifact_path),
            size_bytes=artifact_path.stat().st_size,
        )


@dataclass(frozen=True)
class DocumentVariantSnapshot:
    """One evaluated benchmark document used as an input to a run."""

    document_theme: str
    document_id: str
    document_setting: str
    document_setting_family: str
    document_variant_id: str
    document_variant_index: int
    replacement_proportion: float
    question_count: int
    source_path: str
    source_sha256: str

    @classmethod
    def from_document(cls, document: BenchmarkDocumentForEvaluation) -> DocumentVariantSnapshot:
        return cls(
            document_theme=document.document_theme,
            document_id=document.document_id,
            document_setting=document.document_setting,
            document_setting_family=document.document_setting_family,
            document_variant_id=document.document_variant_id,
            document_variant_index=document.document_variant_index,
            replacement_proportion=document.replacement_proportion,
            question_count=len(document.questions),
            source_path=_relative_to_repo(document.source_path),
            source_sha256=_sha256_file(document.source_path),
        )


@dataclass(frozen=True)
class ModelExecutionSnapshot:
    """One evaluated model configuration."""

    registry_name: str
    provider: str
    provider_model_name: str
    temperature: float
    configured_max_tokens_cap: int
    seed: int | None

    @classmethod
    def from_model_configuration(
        cls,
        model_configuration: PaperModelConfiguration,
    ) -> ModelExecutionSnapshot:
        return cls(
            # Keep this serialized key stable so historical MODEL_EVAL
            # manifests remain byte-for-byte interpretable.
            registry_name=model_configuration.model_id,
            provider=model_configuration.provider,
            provider_model_name=model_configuration.model_name,
            temperature=model_configuration.temperature,
            configured_max_tokens_cap=model_configuration.max_tokens,
            seed=model_configuration.seed,
        )


class ModelEvaluationRunManifest:
    """Create and update one reproducibility manifest for an evaluation run."""

    def __init__(
        self,
        *,
        steps: Sequence[str],
        model_names: Sequence[str] | None,
        themes: Sequence[str] | None,
        document_ids: Sequence[str] | None,
        settings: Sequence[str],
        question_ids: Sequence[str] | None = None,
        question_types: Sequence[str] | None = None,
        overwrite: bool,
        refresh_stale_only: bool = False,
        generation_temperature: float | None = None,
        generation_seed: int | None = None,
        judge_config: JudgeMatchConfiguration | None,
        run_label: str | None = None,
        run_notes: str | None = None,
        dataset_revision_manifest: Path | None = None,
        entrypoint: str | None = None,
        invocation_command: Sequence[str] | None = None,
    ) -> None:
        self.run_id = f"eval_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        self.path = reproducibility_manifest_path(self.run_id)

        model_configurations: list[PaperModelConfiguration]
        if model_names or "raw" in steps:
            if generation_temperature is None and generation_seed is None:
                model_configurations = resolve_paper_model_configurations(model_names)
            else:
                model_configurations = resolve_paper_model_configurations(
                    model_names,
                    temperature=generation_temperature,
                    seed=generation_seed,
                )
        else:
            model_configurations = []
        selected_documents = list(
            iter_evaluation_documents(
                settings=settings,
                themes=themes,
                document_ids=document_ids,
            )
        )
        ordered_settings = order_dataset_settings(settings)

        started_at = _utc_now()
        self._manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "run_label": run_label,
            "run_notes": run_notes,
            "status": "running",
            "started_at_utc": _utc_timestamp_string(started_at),
            "completed_at_utc": None,
            "invocation": {
                "entrypoint": entrypoint,
                "command": list(invocation_command or []),
                "working_directory": str(PROJECT_ROOT_DIRECTORY),
                "artifact_root": _relative_to_repo(MODEL_EVAL_DIR),
                "requested_steps": list(steps),
                "question_ids_filter": list(question_ids or []),
                "question_types_filter": list(question_types or []),
                "overwrite": overwrite,
                "refresh_stale_only": refresh_stale_only,
                "generation_temperature_override": generation_temperature,
                "generation_seed_override": generation_seed,
            },
            "dataset": {
                "document_roots": {
                    "factual": _relative_to_repo(FACTUAL_DOCUMENTS_DIR),
                    "fictional": _relative_to_repo(FICTIONAL_DOCUMENTS_DIR),
                },
                "themes_filter": list(themes or []),
                "document_ids_filter": list(document_ids or []),
                "question_ids_filter": list(question_ids or []),
                "question_types_filter": list(question_types or []),
                "excluded_document_ids": sorted(EXCLUDED_EVALUATION_DOCUMENT_IDS),
                "settings": {spec.setting_id: spec.to_payload() for spec in ordered_settings},
                "revision_manifest": _input_file_fingerprint(
                    dataset_revision_manifest,
                    label="Dataset revision manifest",
                ),
                "benchmark_document_count": len(selected_documents),
                "question_count_total": sum(len(document.questions) for document in selected_documents),
                "benchmark_documents": [
                    asdict(DocumentVariantSnapshot.from_document(document)) for document in selected_documents
                ],
            },
            "models": {
                "model_filter": list(model_names or []),
                "resolved_model_specs": [
                    asdict(ModelExecutionSnapshot.from_model_configuration(model_configuration))
                    for model_configuration in model_configurations
                ],
            },
            "evaluation_protocol": {
                "answer_generation": {
                    "prompt_format_version": PROMPT_FORMAT_VERSION,
                    "system_prompt": DOCUMENT_QA_SYSTEM_PROMPT,
                    "system_prompt_sha256": _sha256_text(DOCUMENT_QA_SYSTEM_PROMPT),
                    "user_prompt_builder": "build_document_question_prompt",
                    "configured_max_tokens_field": "models.resolved_model_specs[].configured_max_tokens_cap",
                    "effective_max_tokens_field": "raw.results[].effective_max_tokens",
                    "token_budget_policy": "suggested_generation_max_tokens",
                    "token_budget_policy_version": GENERATION_TOKEN_BUDGET_POLICY_VERSION,
                    "parser": "parse_schema_answer",
                    "parser_version": ANSWER_PARSER_VERSION,
                },
                "scoring": {
                    "protocol_version": SCORING_PROTOCOL_VERSION,
                    "exact_match_function": "accepted_answer_match_is_correct",
                    "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
                    "judge_system_prompt_sha256": _sha256_text(JUDGE_SYSTEM_PROMPT),
                    "judge_config": None
                    if judge_config is None
                    else {
                        "provider": judge_config.provider,
                        "model_name": judge_config.model_name,
                        "temperature": judge_config.temperature,
                        "max_tokens": judge_config.max_tokens,
                        "effective_initial_max_tokens": max(
                            int(judge_config.max_tokens),
                            JUDGE_MIN_MAX_TOKENS,
                        ),
                        "seed": judge_config.seed,
                    },
                    "judge_retry_policy": {
                        "max_attempts": JUDGE_MAX_ATTEMPTS,
                        "token_growth_factor": JUDGE_TOKEN_GROWTH_FACTOR,
                        "retry_delay_seconds": JUDGE_RETRY_DELAY_SECONDS,
                    },
                },
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
            },
            "git": _git_snapshot(),
            "stage_outputs": {},
            "failure": None,
        }
        self.write()

    def write(self) -> Path:
        """Atomically replace the intentionally mutable run-state manifest."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.tmp.",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(
                    yaml.safe_dump(
                        self._manifest,
                        sort_keys=False,
                        allow_unicode=True,
                        width=10000,
                    )
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            directory_descriptor = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)
        return self.path

    def stage_reference(self, stage_name: str) -> dict[str, str]:
        """Return the manifest pointer to attach to saved stage payloads."""
        return {
            "run_id": self.run_id,
            "path": _relative_to_repo(self.path),
            "stage_name": stage_name,
        }

    def record_stage(self, stage_name: str, artifact_paths: Sequence[Path]) -> None:
        """Record the artifacts produced or reused for one stage."""
        self._manifest["stage_outputs"][stage_name] = {
            "recorded_at_utc": _utc_timestamp_string(_utc_now()),
            "artifact_count": len(artifact_paths),
            "artifacts": [asdict(ArtifactFingerprint.from_path(path)) for path in artifact_paths if path.exists()],
        }
        self.write()

    def mark_completed(self) -> Path:
        """Mark the reproducibility manifest as completed."""
        self._manifest["status"] = "completed"
        self._manifest["completed_at_utc"] = _utc_timestamp_string(_utc_now())
        return self.write()

    def mark_failed(self, exc: BaseException) -> Path:
        """Mark the reproducibility manifest as failed and store a short failure record."""
        self._manifest["status"] = "failed"
        self._manifest["completed_at_utc"] = _utc_timestamp_string(_utc_now())
        self._manifest["failure"] = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        return self.write()
