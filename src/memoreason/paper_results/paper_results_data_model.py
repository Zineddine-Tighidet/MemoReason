"""Paper terminology, fixed dimensions, and typed reporting records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


PAPER_FICTIONAL_VARIANT_IDS = tuple(f"v{index:02d}" for index in range(1, 11))


PAPER_REPLACEMENT_SETTINGS = (
    "factual",
    "fictional_10pct",
    "fictional_20pct",
    "fictional_30pct",
    "fictional_50pct",
    "fictional_80pct",
    "fictional_90pct",
    "fictional",
)


REASONING_TYPES = ("arithmetic", "temporal", "inference")


QUESTION_TYPES = (*REASONING_TYPES, "extractive")


ANSWER_BEHAVIORS = ("variant", "invariant", "refusal")


QUESTION_GROUPS: dict[str, tuple[str, ...]] = {
    "arithmetic": ("arithmetic",),
    "temporal": ("temporal",),
    "inference": ("inference",),
    "reasoning": REASONING_TYPES,
    "extractive": ("extractive",),
}


QUESTION_COLUMN_ORDER = tuple(QUESTION_GROUPS)


ANSWER_GROUPS: dict[str, tuple[str, ...]] = {
    "variant": ("variant",),
    "invariant": ("invariant",),
    "refusal": ("refusal",),
}


MODEL_LABELS = {
    "olmo-3-7b-think": "OLMo-3-7B-Think",
    "olmo-3-7b-instruct": "OLMo-3-7B-Instruct",
    "gpt-oss-20b-groq": "GPT-OSS 20B",
    "gemma-4-26b-a4b-it": "Gemma-4 26B-A4B-IT",
    "gpt-oss-120b-groq": "GPT-OSS 120B",
    "qwen3.5-27b": "Qwen3.5 27B",
    "qwen3.5-35b-a3b": "Qwen3.5 35B-A3B",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
}


@dataclass(frozen=True)
class FrozenEvaluatedAnswersSelection:
    """One manifest-declared evaluated-output file or directory selection."""

    path: Path
    glob: str | None
    file_count: int
    sha256: str
    expected_sha256: str | None


@dataclass(frozen=True)
class PaperResultsManifest:
    """Frozen inputs and publication scope loaded from one run manifest."""

    manifest_path: Path
    manifest_sha256: str
    evaluated_output_paths: tuple[Path, ...]
    evaluated_output_hashes: Mapping[str, str]
    frozen_judge_cache_path: Path | None
    frozen_judge_cache_sha256: str | None
    judge_model: str
    variant_ids: tuple[str, ...]
    settings: tuple[str, ...]
    output_dir: Path
    models: Mapping[str, tuple[str, ...]]
    reviewed_questions_by_pair_key: Mapping[str, ReviewedQuestionMetadata]
    reviewed_templates_root: Path
    reviewed_templates_tree_sha256: str
    expected_reviewed_templates_tree_sha256: str
    frozen_evaluated_answer_selections: tuple[FrozenEvaluatedAnswersSelection, ...] = ()
    expected_frozen_judge_cache_sha256: str | None = None


@dataclass(frozen=True)
class ReviewedQuestionMetadata:
    """Human-reviewed question metadata authoritative for reporting."""

    source_path: Path
    document_theme: str
    document_id: str
    question_id: str
    question_text_factual: str
    question_type: str
    answer_behavior: str


@dataclass(frozen=True)
class EvaluatedAnswer:
    """One evaluated answer normalized to the paper's unit of analysis."""

    source_path: Path
    model_name: str
    document_theme: str
    document_id: str
    setting: str
    replacement_proportion: float
    variant_id: str
    pair_key: str
    question_text: str
    question_type: str
    answer_behavior: str
    ground_truth: str
    answer_schema: str
    accepted_answers_canonical: tuple[str, ...]
    parsed_output: str
    parsed_output_canonical: str
    raw_output: str
    exact_match: bool
    judge_match: bool | None
    final_is_correct: bool
    factual_answer_match: bool | None


@dataclass(frozen=True)
class PairedAccuracyStatistics:
    """Mean, uncertainty, and paired t-test summary used in Tables 2 and 3."""

    count: int
    mean: float
    standard_deviation: float
    standard_error: float
    ci_low: float
    ci_high: float
    t_statistic: float
    p_value: float

    @property
    def ci_half_width(self) -> float:
        return max(self.mean - self.ci_low, self.ci_high - self.mean)
