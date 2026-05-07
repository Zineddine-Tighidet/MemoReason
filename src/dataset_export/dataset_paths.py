"""Filesystem paths for templates, dataset exports, pools, and evaluation artifacts."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
import re

from .dataset_settings import parse_dataset_setting


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

HUMAN_ANNOTATED_TEMPLATES_DIR = DATA_DIR / "HUMAN_ANNOTATED_TEMPLATES"
GENERATED_FICTIONAL_ENTITIES_DIR = DATA_DIR / "GENERATED_FICTIONAL_ENTITIES"
FACTUAL_DOCUMENTS_DIR = DATA_DIR / "FACTUAL_DOCUMENTS"
FICTIONAL_DOCUMENTS_DIR = DATA_DIR / "FICTIONAL_DOCUMENTS"
PARTIAL_FICTIONAL_REPLACEMENTS_DIR = FICTIONAL_DOCUMENTS_DIR / "partial_fictional_replacements"
MODEL_EVAL_DIR = DATA_DIR / "MODEL_EVAL"
MODEL_EVAL_RAW_OUTPUTS_DIR = MODEL_EVAL_DIR / "RAW_OUTPUTS"
MODEL_EVAL_METRICS_DIR = MODEL_EVAL_DIR / "METRICS"
MODEL_EVAL_PLOTS_DIR = MODEL_EVAL_DIR / "PLOTS"
MODEL_EVAL_REPRODUCIBILITY_MANIFESTS_DIR = MODEL_EVAL_DIR / "REPRODUCIBILITY_MANIFESTS"

MODEL_EVAL_PROMPT_PATH = PROJECT_ROOT / "src" / "evaluation_workflows" / "parametric_shortcut" / "prompting.py"

DEFAULT_RANDOM_SEED = 23
_DOCUMENT_VARIANT_SUFFIX_PATTERN = re.compile(r"^(?P<document_id>.+)_v(?P<variant_index>[0-9]+)$")


def ensure_dataset_artifact_directories() -> None:
    """Create dataset and evaluation output directories if they do not exist yet."""
    for directory in (
        GENERATED_FICTIONAL_ENTITIES_DIR,
        FACTUAL_DOCUMENTS_DIR,
        FICTIONAL_DOCUMENTS_DIR,
        PARTIAL_FICTIONAL_REPLACEMENTS_DIR,
        MODEL_EVAL_RAW_OUTPUTS_DIR,
        MODEL_EVAL_METRICS_DIR,
        MODEL_EVAL_PLOTS_DIR,
        MODEL_EVAL_REPRODUCIBILITY_MANIFESTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_model_name(model_name: str) -> str:
    """Convert model identifiers into stable folder and file names."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(model_name).strip())
    return cleaned.strip("._-") or "model"


def resolve_template_identity(template_path: Path) -> tuple[str, str]:
    """Return ``(theme, document_id)`` for a template path."""
    return template_path.parent.name, template_path.stem


def iter_template_paths(
    *,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
) -> Iterator[Path]:
    """Yield template YAML paths from ``data/HUMAN_ANNOTATED_TEMPLATES``."""
    theme_filter = set(themes or [])
    document_filter = set(document_ids or [])

    if not HUMAN_ANNOTATED_TEMPLATES_DIR.exists():
        return

    for theme_dir in sorted(HUMAN_ANNOTATED_TEMPLATES_DIR.iterdir()):
        if not theme_dir.is_dir():
            continue
        if theme_filter and theme_dir.name not in theme_filter:
            continue
        for template_path in sorted(theme_dir.glob("*.yaml")):
            if document_filter and template_path.stem not in document_filter:
                continue
            yield template_path


def _document_directory(setting: str) -> Path:
    setting_spec = parse_dataset_setting(setting)
    if setting_spec.is_factual:
        return FACTUAL_DOCUMENTS_DIR
    if 0.0 < setting_spec.replacement_proportion < 1.0 and setting_spec.setting_family == "fictional":
        return PARTIAL_FICTIONAL_REPLACEMENTS_DIR / setting_spec.setting_id
    return FICTIONAL_DOCUMENTS_DIR / setting_spec.setting_id


def format_document_variant_id(variant_index: int) -> str:
    """Return the stored variant label for a 1-based version index."""
    if int(variant_index) < 1:
        raise ValueError(f"Variant index must be >= 1, got {variant_index!r}.")
    return f"v{int(variant_index):02d}"


def split_document_variant_stem(stem: str) -> tuple[str, int | None]:
    """Return the base document id and optional version index from a file stem."""
    match = _DOCUMENT_VARIANT_SUFFIX_PATTERN.fullmatch(str(stem))
    if match is None:
        return str(stem), None
    return match.group("document_id"), int(match.group("variant_index"))


def document_variant_path(
    theme: str,
    document_id: str,
    setting: str,
    *,
    variant_index: int | None = None,
    variant_count: int = 1,
) -> Path:
    """Return the output path for one exported dataset document."""
    if int(variant_count) < 1:
        raise ValueError(f"Variant count must be >= 1, got {variant_count!r}.")
    setting_spec = parse_dataset_setting(setting)
    use_suffix = not setting_spec.is_factual and int(variant_count) > 1
    if use_suffix:
        if variant_index is None:
            raise ValueError("variant_index is required when variant_count > 1 for fictional settings.")
        filename = f"{document_id}_{format_document_variant_id(variant_index)}.yaml"
    else:
        filename = f"{document_id}.yaml"
    return _document_directory(setting) / theme / filename


def generated_entity_pool_path(theme: str, document_id: str) -> Path:
    """Return the dataset path for an entity pool file."""
    return GENERATED_FICTIONAL_ENTITIES_DIR / theme / f"{document_id}_entity_pool.yaml"


def existing_entity_pool_path(theme: str, document_id: str) -> Path | None:
    """Return the entity-pool path when it already exists in the dataset layout."""
    pool_path = generated_entity_pool_path(theme, document_id)
    return pool_path if pool_path.exists() else None


def iter_document_variant_paths(
    *,
    setting: str,
    themes: Sequence[str] | None = None,
    document_ids: Sequence[str] | None = None,
) -> Iterator[Path]:
    """Yield document paths for one dataset setting."""
    theme_filter = set(themes or [])
    document_filter = set(document_ids or [])
    root_dir = _document_directory(setting)

    if not root_dir.exists():
        return

    for theme_dir in sorted(root_dir.iterdir()):
        if not theme_dir.is_dir():
            continue
        if theme_filter and theme_dir.name not in theme_filter:
            continue
        for document_path in sorted(theme_dir.glob("*.yaml")):
            base_document_id, _ = split_document_variant_stem(document_path.stem)
            if document_filter and not any(
                base_document_id == document_id or base_document_id.startswith(f"{document_id}_")
                for document_id in document_filter
            ):
                continue
            yield document_path


def model_eval_artifact_path(
    *,
    theme: str,
    model_name: str,
    document_id: str,
    setting: str,
    stage_suffix: str,
    variant_id: str | None = None,
) -> Path:
    """Return the YAML path for raw, parsed, or evaluated model outputs."""
    model_folder = sanitize_model_name(model_name)
    variant_token = f"_{variant_id}" if variant_id else ""
    filename = f"{document_id}{variant_token}_{setting}_{model_folder}_{stage_suffix}.yaml"
    return MODEL_EVAL_RAW_OUTPUTS_DIR / theme / model_folder / filename


def metrics_output_path(model_name: str) -> Path:
    """Return the metrics YAML path for a model."""
    return MODEL_EVAL_METRICS_DIR / f"{sanitize_model_name(model_name)}_metrics.yaml"


def plot_output_path(filename: str) -> Path:
    """Return the plot path under the evaluation plots directory."""
    return MODEL_EVAL_PLOTS_DIR / filename


def reproducibility_manifest_path(run_id: str) -> Path:
    """Return the reproducibility-manifest path for one evaluation run."""
    return MODEL_EVAL_REPRODUCIBILITY_MANIFESTS_DIR / f"{run_id}.yaml"


def prompt_path(prompt_filename: str) -> Path:
    """Return the path to an agentic annotation prompt file."""
    return AGENTIC_ANNOTATION_PROMPTS_DIR / prompt_filename


def template_path_by_identity(theme: str, document_id: str) -> Path:
    """Return the expected template path for ``theme/document_id``."""
    return HUMAN_ANNOTATED_TEMPLATES_DIR / theme / f"{document_id}.yaml"


def unique_question_key(theme: str, document_id: str, question_id: str) -> str:
    """Build a stable key for pairing factual and fictional question results."""
    return "::".join((theme, document_id, question_id))
