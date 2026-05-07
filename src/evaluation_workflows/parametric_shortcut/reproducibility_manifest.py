"""Evaluation reproducibility manifests for benchmark runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import platform
from pathlib import Path
import socket
import subprocess
import sys
import traceback
from typing import Any
import uuid

import yaml

from src.dataset_export.dataset_settings import order_dataset_settings
from src.dataset_export.dataset_paths import PROJECT_ROOT, reproducibility_manifest_path
from .dataset import (
    EXCLUDED_EVALUATION_DOCUMENT_IDS,
    EvaluationDocument,
    iter_evaluation_documents,
)
from .prompting import DOCUMENT_QA_SYSTEM_PROMPT, JUDGE_SYSTEM_PROMPT, PROMPT_FORMAT_VERSION
from .registry import EvaluatedModelSpec, resolve_model_specs
from .scoring import JudgeConfig


UTC = timezone.utc


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
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _git_snapshot() -> dict[str, Any] | None:
    def _run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
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
    def from_document(cls, document: EvaluationDocument) -> DocumentVariantSnapshot:
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
    max_tokens: int
    seed: int | None

    @classmethod
    def from_model_spec(cls, model_spec: EvaluatedModelSpec) -> ModelExecutionSnapshot:
        return cls(
            registry_name=model_spec.registry_name,
            provider=model_spec.provider,
            provider_model_name=model_spec.model_name,
            temperature=model_spec.temperature,
            max_tokens=model_spec.max_tokens,
            seed=model_spec.seed,
        )


class EvaluationReproducibilityManifest:
    """Create and update one reproducibility manifest for an evaluation run."""

    def __init__(
        self,
        *,
        steps: Sequence[str],
        model_names: Sequence[str] | None,
        themes: Sequence[str] | None,
        document_ids: Sequence[str] | None,
        settings: Sequence[str],
        overwrite: bool,
        judge_config: JudgeConfig | None,
        run_label: str | None = None,
        run_notes: str | None = None,
        entrypoint: str | None = None,
        invocation_command: Sequence[str] | None = None,
    ) -> None:
        self.run_id = f"eval_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
        self.path = reproducibility_manifest_path(self.run_id)

        resolved_model_specs: list[EvaluatedModelSpec]
        if model_names or "raw" in steps:
            resolved_model_specs = resolve_model_specs(model_names)
        else:
            resolved_model_specs = []
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
                "working_directory": str(PROJECT_ROOT),
                "requested_steps": list(steps),
                "overwrite": overwrite,
            },
            "dataset": {
                "themes_filter": list(themes or []),
                "document_ids_filter": list(document_ids or []),
                "excluded_document_ids": sorted(EXCLUDED_EVALUATION_DOCUMENT_IDS),
                "settings": {spec.setting_id: spec.to_payload() for spec in ordered_settings},
                "benchmark_document_count": len(selected_documents),
                "question_count_total": sum(len(document.questions) for document in selected_documents),
                "benchmark_documents": [
                    asdict(DocumentVariantSnapshot.from_document(document)) for document in selected_documents
                ],
            },
            "models": {
                "model_filter": list(model_names or []),
                "resolved_model_specs": [
                    asdict(ModelExecutionSnapshot.from_model_spec(spec)) for spec in resolved_model_specs
                ],
            },
            "evaluation_protocol": {
                "answer_generation": {
                    "prompt_format_version": PROMPT_FORMAT_VERSION,
                    "system_prompt": DOCUMENT_QA_SYSTEM_PROMPT,
                    "system_prompt_sha256": _sha256_text(DOCUMENT_QA_SYSTEM_PROMPT),
                    "user_prompt_builder": "build_document_question_prompt",
                    "parser": "parse_schema_answer",
                },
                "scoring": {
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
                        "seed": judge_config.seed,
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
        """Persist the current manifest state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(self._manifest, sort_keys=False, allow_unicode=True, width=10000),
            encoding="utf-8",
        )
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
