"""YAML I/O service: theme-based source documents with per-user working copies."""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.document_schema import AnnotatedDocument, Question
from src.core.annotation_runtime import (
    ENTITY_TAXONOMY,
    AnnotationParser,
    find_rule_sanity_errors,
    normalize_document_taxonomy,
    validate_annotations,
    validate_question_and_answer_entity_scope,
)
from web.services.persistence import restore_work_file_from_gcs, sync_work_file_to_gcs

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RETIRED_DOCS_FILE = PROJECT_ROOT / "data" / "HUMAN_ANNOTATED_TEMPLATES" / "retired_documents.yaml"
WIKIEVENT_PUBLIC_ATTACKS_DIR = PROJECT_ROOT / "data" / "HUMAN_ANNOTATED_TEMPLATES" / "public_attacks_news_articles"
_TAXONOMY_CACHE_KEY: tuple[float | None, float | None] | None = None
_TAXONOMY_CACHE_VALUE: dict[str, list[str]] | None = None

# Source documents (read-only originals).
# Priority is:
# 1) explicit env override
# 2) required curated wiki set
def _resolve_source_dir() -> Path:
    override = os.getenv("ANNOTATION_SOURCE_DIR")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"ANNOTATION_SOURCE_DIR does not exist: {path}")
        return path

    preferred = PROJECT_ROOT / "data" / "HUMAN_ANNOTATED_TEMPLATES"
    if not preferred.exists():
        raise FileNotFoundError(
            "Required source directory is missing: "
            f"{preferred}. Set ANNOTATION_SOURCE_DIR to point to your own YAML source documents."
        )
    return preferred


SOURCE_DIR = _resolve_source_dir()
# User working copies + completed annotations
WORK_DIR = PROJECT_ROOT / "web" / "data" / "annotation_workspace"
LEGACY_WORK_DIR = PROJECT_ROOT / "data" / "annotation_workspace"
AI_PRE_ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "AI_PRE_ANNOTATIONS"

# Reference annotations (read-only examples, when provided locally)
REFERENCE_DIR = WORK_DIR / "anonymous_reference"

# Canonical theme id for Public Attacks.
PUBLIC_ATTACKS_THEME = "public_attacks_news_articles"
# Backward-compat alias used by earlier runs/history.
LEGACY_THEME = "completed_annotations"
PUBLIC_ATTACKS_THEME_ALIASES = {PUBLIC_ATTACKS_THEME, LEGACY_THEME}


def ensure_workspace_dirs() -> None:
    """Ensure the primary workspace exists and keep a legacy alias available."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    if LEGACY_WORK_DIR.exists() or LEGACY_WORK_DIR.is_symlink():
        return

    try:
        LEGACY_WORK_DIR.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_WORK_DIR.symlink_to(WORK_DIR, target_is_directory=True)
    except OSError:
        # Symlink creation is a compatibility optimization, not a requirement.
        pass


def canonical_theme_id(theme: str) -> str:
    raw = str(theme or "").strip()
    if raw in PUBLIC_ATTACKS_THEME_ALIASES:
        return PUBLIC_ATTACKS_THEME
    return raw


def _load_retired_doc_keys() -> set[tuple[str, str]]:
    """Load optional retired document keys from data/Wikipedia/retired_documents.yaml."""
    if not RETIRED_DOCS_FILE.exists():
        return set()
    try:
        payload = yaml.safe_load(RETIRED_DOCS_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return set()
    entries = payload.get("retired_documents")
    if not isinstance(entries, list):
        return set()

    keys: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        theme = canonical_theme_id(str(entry.get("theme") or "").strip())
        doc_id = str(entry.get("doc_id") or "").strip()
        if not theme or not doc_id:
            continue
        keys.add((theme, doc_id))
    return keys


RETIRED_DOC_KEYS = _load_retired_doc_keys()
SOURCE_METADATA_STEMS = {
    "_index",
    "reproducibility_manifest",
    "retired_documents",
}
EXCLUDED_DOCUMENT_KEYS: set[tuple[str, str]] = {
    ("retail_banking_regulations_and_policies", "bankreg_11"),
}


def is_retired_document(theme: str, doc_id: str) -> bool:
    return (canonical_theme_id(theme), str(doc_id).strip()) in RETIRED_DOC_KEYS


def is_excluded_document(theme: str, doc_id: str) -> bool:
    return (canonical_theme_id(theme), str(doc_id).strip()) in EXCLUDED_DOCUMENT_KEYS


def is_public_attacks_theme(theme: str) -> bool:
    return canonical_theme_id(theme) == PUBLIC_ATTACKS_THEME


def _theme_variants(theme: str) -> list[str]:
    canonical = canonical_theme_id(theme)
    if canonical == PUBLIC_ATTACKS_THEME:
        return [PUBLIC_ATTACKS_THEME, LEGACY_THEME]
    return [canonical]


def _theme_history_paths(theme: str, doc_id: str) -> list[str]:
    return [f"{variant}/{doc_id}" for variant in _theme_variants(theme)]


def _theme_source_dir(theme: str) -> Path | None:
    canonical = canonical_theme_id(theme)
    if canonical == PUBLIC_ATTACKS_THEME:
        if WIKIEVENT_PUBLIC_ATTACKS_DIR.exists():
            return WIKIEVENT_PUBLIC_ATTACKS_DIR
        primary = SOURCE_DIR / PUBLIC_ATTACKS_THEME
        if primary.exists():
            return primary
        legacy = SOURCE_DIR / LEGACY_THEME
        if legacy.exists():
            return legacy
        if REFERENCE_DIR.exists():
            return REFERENCE_DIR
        return None
    path = SOURCE_DIR / canonical
    return path if path.exists() else None


def _canonical_themes_from_source() -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for theme in THEMES:
        theme_id = canonical_theme_id(theme)
        if theme_id in seen:
            continue
        if _theme_source_dir(theme_id) is None:
            continue
        ordered.append(theme_id)
        seen.add(theme_id)
    if PUBLIC_ATTACKS_THEME not in seen and _theme_source_dir(PUBLIC_ATTACKS_THEME) is not None:
        ordered.append(PUBLIC_ATTACKS_THEME)
        seen.add(PUBLIC_ATTACKS_THEME)
    return ordered


# Discover themes from source directory
THEMES = sorted([d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]) if SOURCE_DIR.exists() else []
ALL_THEMES = sorted({canonical_theme_id(t) for t in THEMES})
if PUBLIC_ATTACKS_THEME not in ALL_THEMES:
    ALL_THEMES.append(PUBLIC_ATTACKS_THEME)

# Prettify theme names
THEME_LABELS = {t: t.replace("_", " ").title() for t in THEMES}
THEME_LABELS[PUBLIC_ATTACKS_THEME] = "Public Attacks News Articles"
THEME_LABELS[LEGACY_THEME] = "Public Attacks News Articles"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _represent_str(dumper: yaml.Dumper, data: str) -> Any:
    if "\n" in data or len(data) > 200:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


NoAliasDumper.add_representer(str, _represent_str)

_INLINE_ANNOTATION_PATTERN = re.compile(r"\[[^\]]+;\s*\w+_\d+\.\w+\]")
_ANNOTATED_BY_PATTERN = re.compile(r"^\s*annotated_by\s*:\s*(.+?)\s*$", re.MULTILINE)


def _has_inline_annotations(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    return bool(_INLINE_ANNOTATION_PATTERN.search(text))


# --- Theme & document listing ---

def _is_source_metadata_yaml(path: Path) -> bool:
    stem = str(path.stem or "").strip().lower()
    if not stem:
        return True
    if stem.startswith("_"):
        return True
    return stem in SOURCE_METADATA_STEMS


def _list_source_yaml_docs(
    source_dir: Path | None,
    theme: str | None = None,
    include_retired: bool = True,
) -> list[Path]:
    if source_dir is None or not source_dir.exists():
        return []

    canonical_theme = canonical_theme_id(theme) if theme else None
    inferred_theme = canonical_theme or canonical_theme_id(str(source_dir.name))
    result: list[Path] = []
    for path in sorted(source_dir.glob("*.yaml")):
        if _is_source_metadata_yaml(path):
            continue
        if is_excluded_document(inferred_theme, path.stem):
            continue
        if canonical_theme and not include_retired and is_retired_document(canonical_theme, path.stem):
            continue
        result.append(path)
    return result

def _list_legacy_docs() -> list[Path]:
    """List Public Attacks source docs.

    Preferred location is data/WikiEvent/public_attacks_news_articles. For
    backward compatibility, also supports SOURCE_DIR/public_attacks_news_articles,
    SOURCE_DIR/completed_annotations, and the historical reference folder under
    WORK_DIR/anonymous_reference.
    """
    source_dir = _theme_source_dir(PUBLIC_ATTACKS_THEME)
    if source_dir is None:
        return []
    return _list_source_yaml_docs(source_dir, PUBLIC_ATTACKS_THEME)


def list_themes() -> list[dict[str, Any]]:
    """List all themes with document counts."""
    result = []
    seen_ids = set()
    for theme_id in _canonical_themes_from_source():
        theme_dir = _theme_source_dir(theme_id)
        if theme_dir is None:
            continue
        docs = _list_source_yaml_docs(theme_dir, theme_id, include_retired=False)
        result.append({
            "theme_id": theme_id,
            "theme_label": THEME_LABELS.get(theme_id, theme_id.replace("_", " ").title()),
            "total_docs": len(docs),
        })
        seen_ids.add(theme_id)
    legacy_docs = _list_legacy_docs()
    if legacy_docs and PUBLIC_ATTACKS_THEME not in seen_ids:
        result.append({
            "theme_id": PUBLIC_ATTACKS_THEME,
            "theme_label": THEME_LABELS[PUBLIC_ATTACKS_THEME],
            "total_docs": len(legacy_docs),
        })
    return result


def list_theme_documents(theme: str, include_retired: bool = False) -> list[dict[str, Any]]:
    """List all source documents in a theme."""
    requested_theme = canonical_theme_id(theme)
    if requested_theme == PUBLIC_ATTACKS_THEME and _theme_source_dir(requested_theme) is None:
        return _list_legacy_theme_documents()
    theme_dir = _theme_source_dir(requested_theme)
    if theme_dir is None:
        return []
    result = []
    for yaml_path in _list_source_yaml_docs(theme_dir, requested_theme, include_retired=include_retired):
        raw = _load_raw_yaml(yaml_path)
        doc_data = raw.get("document", {})
        result.append({
            "doc_id": yaml_path.stem,
            "document_id": doc_data.get("document_id", yaml_path.stem),
            "document_theme": doc_data.get("document_theme", ""),
            "source": doc_data.get("source", ""),
            "wikipedia_title": doc_data.get("wikipedia_title", ""),
        })
    return result


def list_theme_doc_ids(theme: str, include_retired: bool = False) -> list[str]:
    """List source doc ids in a theme without parsing YAML payloads."""
    requested_theme = canonical_theme_id(theme)
    theme_dir = _theme_source_dir(requested_theme)
    if theme_dir is None:
        return []
    return [
        yaml_path.stem
        for yaml_path in _list_source_yaml_docs(theme_dir, requested_theme, include_retired=include_retired)
    ]


def _list_legacy_theme_documents() -> list[dict[str, Any]]:
    result = []
    for yaml_path in _list_legacy_docs():
        raw = _load_raw_yaml(yaml_path)
        if raw is None:
            continue
        doc_data = raw.get("document", {})
        result.append({
            "doc_id": yaml_path.stem,
            "document_id": doc_data.get("document_id", yaml_path.stem),
            "document_theme": doc_data.get("document_theme", ""),
            "source": "reference",
            "wikipedia_title": "",
        })
    return result


# --- User working copies ---

def _user_doc_path(username: str, theme: str, doc_id: str) -> Path:
    """Theme-organised working copy path: {username}/{theme}/{doc_id}.yaml"""
    return WORK_DIR / username / theme / f"{doc_id}.yaml"


def _user_doc_path_legacy(username: str, doc_id: str) -> Path:
    """Flat (pre-theme) path kept for backward-compat: {username}/{doc_id}.yaml"""
    return WORK_DIR / username / f"{doc_id}.yaml"


def _find_source_doc(theme: str, doc_id: str) -> Path | None:
    theme_dir = _theme_source_dir(theme)
    if theme_dir is None:
        return None
    p = theme_dir / f"{doc_id}.yaml"
    return p if p.exists() else None


def _iter_source_agent_doc_candidates(theme: str, doc_id: str, source_agent: str):
    agent = str(source_agent or "").strip()
    if not agent:
        return

    seen: set[Path] = set()
    for root in (WORK_DIR, AI_PRE_ANNOTATIONS_DIR):
        if not root.exists():
            continue
        for variant in _theme_variants(theme):
            candidate = root / agent / variant / f"{doc_id}.yaml"
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _find_source_agent_copy(
    theme: str,
    doc_id: str,
    source_agent: str,
    *,
    require_inline_annotations: bool = False,
) -> Path | None:
    for candidate in _iter_source_agent_doc_candidates(theme, doc_id, source_agent):
        if not candidate.exists():
            continue
        if not require_inline_annotations:
            return candidate

        try:
            text = candidate.read_text(encoding="utf-8")
        except Exception:
            text = ""
        if _has_inline_annotations(text):
            return candidate
    return None


def _extract_annotated_by_from_text(raw_text: str) -> str | None:
    if not raw_text:
        return None
    match = _ANNOTATED_BY_PATTERN.search(raw_text)
    if not match:
        return None
    value = match.group(1).strip()
    if value and value[0] in {"'", '"'} and value[-1:] == value[0]:
        value = value[1:-1].strip()
    return value or None


def _find_ai_annotated_copy(theme: str, doc_id: str, exclude_username: str) -> Path | None:
    """Return the path of an AI-annotated copy of a document, if one exists.

    Only considers files that have the ``annotated_by`` field set (written by
    the AI pipeline), so human annotators' copies are never used as bootstrap.
    """
    best_path: Path | None = None
    best_ts: datetime | None = None
    best_mtime: float = -1.0

    variants = _theme_variants(theme)

    for root in (WORK_DIR, AI_PRE_ANNOTATIONS_DIR):
        if not root.exists():
            continue
        for user_dir in sorted(root.iterdir()):
            if not user_dir.is_dir() or user_dir.name == exclude_username:
                continue
            for variant in variants:
                candidate = user_dir / variant / f"{doc_id}.yaml"
                if not candidate.exists():
                    continue
                try:
                    text = candidate.read_text(encoding="utf-8")
                except Exception:
                    continue
                if not _extract_annotated_by_from_text(text):
                    continue
                if not _has_inline_annotations(text):
                    continue

                ts_raw = None
                for line in text.splitlines():
                    if line.strip().startswith("annotated_at:"):
                        ts_raw = line.split(":", 1)[1].strip()
                        if ts_raw and ts_raw[0] in {"'", '"'} and ts_raw[-1:] == ts_raw[0]:
                            ts_raw = ts_raw[1:-1].strip()
                        break
                ts_val: datetime | None = None
                if isinstance(ts_raw, str):
                    try:
                        ts_val = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                    except ValueError:
                        ts_val = None
                mtime = candidate.stat().st_mtime

                is_better = False
                if best_path is None:
                    is_better = True
                elif ts_val is not None and best_ts is not None:
                    is_better = ts_val > best_ts
                elif ts_val is not None and best_ts is None:
                    is_better = True
                elif ts_val is None and best_ts is None:
                    is_better = mtime > best_mtime

                if is_better:
                    best_path = candidate
                    best_ts = ts_val
                    best_mtime = mtime

    return best_path


def ensure_user_copy(username: str, theme: str, doc_id: str) -> Path:
    """Ensure a user has a working copy under {username}/{theme}/{doc_id}.yaml.

    Bootstrap priority:
    1. Existing user copy (theme-organised)  → return as-is (never auto-overwrite)
    2. Legacy flat copy ({username}/{doc_id}.yaml)  → migrate and return
    3. AI-annotated copy from another folder (``annotated_by`` field set)
       → copy so the reviewer starts from the AI draft, not a blank document
    4. Raw source document  → plain copy, no annotations
    """
    ensure_workspace_dirs()
    canonical_theme = canonical_theme_id(theme)
    user_path = _user_doc_path(username, canonical_theme, doc_id)
    if user_path.exists():
        return user_path
    restore_work_file_from_gcs(user_path)
    if user_path.exists():
        return user_path

    # Migrate from alias theme folder if present (Public Attacks legacy ID).
    for variant in _theme_variants(canonical_theme):
        if variant == canonical_theme:
            continue
        alias_path = _user_doc_path(username, variant, doc_id)
        if alias_path.exists():
            user_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(alias_path), str(user_path))
            return user_path
        restore_work_file_from_gcs(alias_path)
        if alias_path.exists():
            user_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(alias_path), str(user_path))
            return user_path

    # Migrate legacy flat copy if present
    legacy = _user_doc_path_legacy(username, doc_id)
    if legacy.exists():
        user_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(user_path))
        return user_path
    restore_work_file_from_gcs(legacy)
    if legacy.exists():
        user_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(user_path))
        return user_path

    # Bootstrap from an AI-annotated copy so reviewers start from a pre-filled draft
    ai_copy = _find_ai_annotated_copy(theme, doc_id, exclude_username=username)
    if ai_copy is not None:
        user_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(ai_copy), str(user_path))
        return user_path

    # Fall back to raw source (no annotations)
    source_path = _find_source_doc(theme, doc_id)
    if source_path is None:
        raise FileNotFoundError(f"Source document not found: {theme}/{doc_id}")
    user_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source_path), str(user_path))
    return user_path


def load_document(username: str, theme: str, doc_id: str) -> dict[str, Any]:
    """Load a user's working copy (auto-creates from source if needed)."""
    user_path = ensure_user_copy(username, theme, doc_id)
    # Keep long-running Cloud Run instances in sync with externally pushed updates
    # (e.g. rules/questions published via maintenance scripts).
    restore_work_file_from_gcs(user_path, max_age_seconds=10.0)
    raw = _load_raw_yaml(user_path)
    if raw is None:
        raise FileNotFoundError(f"Failed to load: {user_path}")
    return normalize_document_taxonomy(raw["document"])


def load_document_fresh(username: str, theme: str, doc_id: str) -> dict[str, Any]:
    """Load a user's working copy after forcing a remote refresh attempt.

    Power-user refreshes can land on a different Cloud Run instance than the save
    request. Bypassing the short work-file restore cache avoids serving a stale
    local copy right after a successful save on another instance.
    """
    user_path = ensure_user_copy(username, theme, doc_id)
    restore_work_file_from_gcs(user_path, max_age_seconds=0)
    raw = _load_raw_yaml(user_path)
    if raw is None:
        raise FileNotFoundError(f"Failed to load: {user_path}")
    return normalize_document_taxonomy(raw["document"])


def load_source_document(theme: str, doc_id: str) -> dict[str, Any]:
    """Load a source document (read-only)."""
    source_path = _find_source_doc(theme, doc_id)
    if source_path is None:
        raise FileNotFoundError(f"Source not found: {theme}/{doc_id}")
    raw = _load_raw_yaml(source_path)
    if raw is None:
        raise FileNotFoundError(f"Failed to load source: {source_path}")
    return normalize_document_taxonomy(raw["document"])


def save_document(
    username: str,
    theme: str,
    doc_id: str,
    doc_data: dict[str, Any],
    *,
    validate_question_scope: bool = True,
) -> None:
    """Save a user's working copy. Creates working copy if needed."""
    # Ensure working copy exists (creates from source if needed)
    user_path = ensure_user_copy(username, theme, doc_id)
    normalized_doc_data = normalize_document_taxonomy(doc_data)
    validation_errors = validate_document(
        normalized_doc_data,
        validate_question_scope=validate_question_scope,
    )
    if validation_errors:
        raise ValueError("\n".join(validation_errors))
    output = {"document": normalized_doc_data}
    with open(user_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, Dumper=NoAliasDumper, default_flow_style=False,
                  allow_unicode=True, sort_keys=False, width=10000)
    sync_work_file_to_gcs(user_path)


def user_has_copy(username: str, theme: str, doc_id: str) -> bool:
    return (
        _user_doc_path(username, theme, doc_id).exists()
        or _user_doc_path_legacy(username, doc_id).exists()
    )


# --- Validation & entity extraction ---

def validate_document(
    doc_data: dict[str, Any],
    *,
    validate_question_scope: bool = True,
) -> list[str]:
    doc_data = normalize_document_taxonomy(doc_data)
    errors: list[str] = []
    doc_id = doc_data.get("document_id", "unknown")
    errors.extend(find_rule_sanity_errors(doc_data.get("rules", []) or []))
    try:
        validate_annotations(doc_data.get("document_to_annotate", ""),
                             source_label=f"{doc_id}/document_to_annotate")
    except Exception as e:
        errors.append(str(e))
    try:
        validate_annotations(
            doc_data.get("fictionalized_annotated_template_document", ""),
            source_label=f"{doc_id}/fictionalized_annotated_template_document",
        )
    except Exception as e:
        errors.append(str(e))
    for q in doc_data.get("questions", []):
        qid = q.get("question_id", "?")
        try:
            validate_annotations(q.get("question", ""), source_label=f"{doc_id}/{qid}/question")
        except Exception as e:
            errors.append(str(e))
        for step_index, reasoning_step in enumerate(q.get("reasoning_chain", []) or [], start=1):
            try:
                validate_annotations(
                    str(reasoning_step),
                    source_label=f"{doc_id}/{qid}/reasoning_chain[{step_index}]",
                )
            except Exception as e:
                errors.append(str(e))
        raw_answer = q.get("answer", "")
        if isinstance(raw_answer, str):
            try:
                validate_annotations(raw_answer, source_label=f"{doc_id}/{qid}/answer")
            except Exception as e:
                errors.append(str(e))
    if validate_question_scope:
        try:
            validate_question_and_answer_entity_scope(
                doc_data.get("document_to_annotate", ""),
                doc_data.get("questions", []),
                source_label=str(doc_id),
            )
        except Exception as e:
            errors.append(str(e))
    return errors


def extract_entities(doc_data: dict[str, Any]) -> dict[str, Any]:
    """Extract entities using the JS-side parser as fallback if Pydantic models fail."""
    doc_data = normalize_document_taxonomy(doc_data)
    try:
        questions = []
        for q in doc_data.get("questions", []):
            raw_answer = q.get("answer", "")
            if raw_answer is True:
                answer_str = "Yes"
            elif raw_answer is False:
                answer_str = "No"
            elif isinstance(raw_answer, list) and len(raw_answer) == 1:
                answer_str = str(raw_answer[0])
            else:
                answer_str = str(raw_answer)
            raw_answer_type = q.get("answer_type")
            if raw_answer_type is None:
                invariant_flag = q.get("is_answer_invariant")
                if invariant_flag is True:
                    raw_answer_type = "invariant"
                elif invariant_flag is False:
                    raw_answer_type = "variant"
            questions.append(Question(
                question_id=q.get("question_id", ""),
                question=q.get("question", ""),
                answer=answer_str,
                question_type=q.get("question_type"),
                answer_type=raw_answer_type,
                reasoning_chain=[
                    str(step).strip()
                    for step in (q.get("reasoning_chain") or [])
                    if str(step).strip()
                ],
            ))
        doc = AnnotatedDocument(
            document_id=doc_data.get("document_id", ""),
            document_theme=doc_data.get("document_theme", ""),
            original_document=doc_data.get("original_document", ""),
            document_to_annotate=doc_data.get("document_to_annotate", ""),
            fictionalized_annotated_template_document=doc_data.get(
                "fictionalized_annotated_template_document", ""
            ),
            rules=doc_data.get("rules", []),
            questions=questions,
        )
        return AnnotationParser.extract_factual_entities(doc, include_questions=False).model_dump()
    except (ValueError, AttributeError):
        return _extract_entities_simple(doc_data)


def _extract_entities_simple(doc_data: dict[str, Any]) -> dict[str, Any]:
    """Lightweight regex-based entity extraction that bypasses Pydantic models."""
    import re
    pattern = re.compile(r'\[([^\]]+);\s*([^\]]+)\]')
    entity_map: dict[str, dict[str, Any]] = {}

    for text_field in ("document_to_annotate",):
        text = doc_data.get(text_field, "")
        for m in pattern.finditer(text):
            ref = m.group(2).strip()
            value = m.group(1).strip()
            if "." in ref:
                eid, attr = ref.split(".", 1)
            else:
                eid, attr = ref, None
            if eid not in entity_map:
                entity_map[eid] = {}
            if attr:
                entity_map[eid][attr] = value

    return {"entities": entity_map}


def get_taxonomy() -> dict[str, list[str]]:
    """Return taxonomy attributes for editor dropdowns.

    Prefer the extended taxonomy file so UI options stay aligned with the
    taxonomy editor; fall back to the static in-code taxonomy on failure.
    """
    try:
        from web.api.taxonomy import TAXONOMY_DOC_FILE, TAXONOMY_FILE, load_extended_taxonomy

        global _TAXONOMY_CACHE_KEY, _TAXONOMY_CACHE_VALUE
        cache_key = (
            TAXONOMY_FILE.stat().st_mtime if TAXONOMY_FILE.exists() else None,
            TAXONOMY_DOC_FILE.stat().st_mtime if TAXONOMY_DOC_FILE.exists() else None,
        )
        if _TAXONOMY_CACHE_KEY == cache_key and _TAXONOMY_CACHE_VALUE is not None:
            return dict(_TAXONOMY_CACHE_VALUE)

        data = load_extended_taxonomy(sync_from_doc=True)
        entities = data.get("entities", {}) or {}
        if isinstance(entities, dict) and entities:
            result: dict[str, list[str]] = {}
            for entity_type, entity_data in entities.items():
                if str(entity_type).strip() == "organization":
                    continue
                attrs = (entity_data or {}).get("attributes", {}) or {}
                if isinstance(attrs, dict):
                    result[entity_type] = sorted(str(k) for k in attrs.keys())
            if result:
                _TAXONOMY_CACHE_KEY = cache_key
                _TAXONOMY_CACHE_VALUE = dict(result)
                return result
    except Exception:
        pass

    return {k: sorted(v) for k, v in ENTITY_TAXONOMY.items() if k != "organization"}


# --- Progress ---

def get_theme_progress(db=None) -> dict[str, Any]:
    """Theme-based progress from document_history table + legacy reference docs."""
    from web.services.db import get_db
    if db is None:
        db = get_db()

    try:
        from web.services import review_campaign_service

        review_status_map = review_campaign_service.get_review_status_map(db=db)
    except Exception:
        review_status_map = {}

    def _doc_review_statuses(theme_id: str, doc_id: str, doc_status: str) -> dict[str, Any]:
        raw = review_status_map.get((canonical_theme_id(theme_id), str(doc_id)), {})
        review_defaults = {
            "status": "draft",
            "reviewed": False,
            "reviewed_at": None,
            "reviewed_by": None,
            "last_edited_by": None,
            "activity_users": [],
            "last_activity_at": None,
            "last_activity_user": None,
            "last_activity_action": None,
            "ignored_qa_count": 0,
            "has_ignored_qas": False,
        }
        canonical_theme = canonical_theme_id(theme_id)
        normalized_doc_id = str(doc_id).strip()
        excluded_from_rules_campaign = (
            canonical_theme in getattr(review_campaign_service, "RULE_REVIEW_EXCLUDED_THEMES", set())
            or is_excluded_document(canonical_theme, normalized_doc_id)
        )
        excluded_from_questions_campaign = (
            canonical_theme in getattr(review_campaign_service, "QUESTION_REVIEW_EXCLUDED_THEMES", set())
            or is_excluded_document(canonical_theme, normalized_doc_id)
        )

        rules_status = raw.get("rules", dict(review_defaults))
        questions_status = raw.get("questions", dict(review_defaults))

        if excluded_from_rules_campaign:
            # Excluded docs are not part of active review assignment; render them as completed.
            rules_status = {
                **dict(review_defaults),
                **(rules_status if isinstance(rules_status, dict) else {}),
                "status": "completed",
                "reviewed": True,
            }
        if excluded_from_questions_campaign:
            # Public-attacks docs are intentionally excluded from QA assignment and should
            # appear done in dashboard progress cards.
            questions_status = {
                **dict(review_defaults),
                **(questions_status if isinstance(questions_status, dict) else {}),
                "status": "completed",
                "reviewed": True,
            }

        return {
            "document_annotation": {
                "status": str(doc_status),
                "reviewed": str(doc_status) in {"completed", "validated"},
            },
            "rules": rules_status,
            "questions": questions_status,
        }

    # If a workflow run is active, completion is task-based:
    # a document is completed only when both stage-1 and stage-2 are completed.
    try:
        from web.services import workflow_service

        active_run = workflow_service.get_active_run(db)
    except Exception:
        active_run = None

    if active_run:
        run_id = int(active_run["id"])
        source_agent = str(active_run.get("source_agent") or "").strip()
        feedback_state_map: dict[tuple[str, str], dict[str, Any]] = {}
        try:
            feedback_state_map = workflow_service.get_run_feedback_acceptance_state_map(run_id, db=db)
        except Exception:
            feedback_state_map = {}
        task_rows = db.execute(
            """
            SELECT t.theme, t.doc_id, t.stage, t.status, u.username
            FROM workflow_tasks t
            JOIN users u ON u.id = t.assignee_user_id
            WHERE t.run_id = ?
            """,
            (run_id,),
        ).fetchall()

        agreement_rows = db.execute(
            """
            SELECT
                a.theme,
                a.doc_id,
                a.status,
                ua.username AS resolved_by
            FROM workflow_agreements a
            LEFT JOIN users ua ON ua.id = a.resolved_by_user_id
            WHERE a.run_id = ?
            """,
            (run_id,),
        ).fetchall()

        history_rows = db.execute(
            """
            SELECT
                h.document_path,
                h.status,
                u.username,
                u.role
            FROM document_history h
            JOIN users u ON u.id = h.user_id
            ORDER BY h.document_path ASC, h.timestamp DESC, h.id DESC
            """
        ).fetchall()

        workflow_docs: dict[tuple[str, str], dict[str, Any]] = {}
        for row in task_rows:
            key = (canonical_theme_id(str(row["theme"])), str(row["doc_id"]))
            bucket = workflow_docs.setdefault(
                key,
                {
                    "statuses": [],
                    "assignees": [],
                },
            )
            bucket["statuses"].append(row["status"])
            bucket["assignees"].append(row["username"])

        workflow_agreements: dict[tuple[str, str], dict[str, Any]] = {}
        for row in agreement_rows:
            key = (canonical_theme_id(str(row["theme"])), str(row["doc_id"]))
            workflow_agreements[key] = {
                "status": str(row["status"] or "pending"),
                "resolved_by": row["resolved_by"],
            }

        latest_history: dict[str, dict[str, Any]] = {}
        for row in history_rows:
            path = str(row["document_path"] or "")
            if not path or path in latest_history:
                continue
            latest_history[path] = {
                "status": str(row["status"] or "draft"),
                "username": row["username"],
                "role": str(row["role"] or ""),
            }

        def _history_doc_status(theme: str, doc_id: str) -> dict[str, Any] | None:
            entry = None
            for path in _theme_history_paths(theme, doc_id):
                entry = latest_history.get(path)
                if entry:
                    break
            if not entry:
                return None
            raw_status = str(entry.get("status") or "draft").lower()
            if raw_status in {"completed", "validated"}:
                status = "completed"
            elif raw_status == "in_progress":
                status = "in_progress"
            else:
                status = "draft"
            return {
                "status": status,
                "raw_status": raw_status,
                "username": entry.get("username"),
                "role": str(entry.get("role") or ""),
            }

        def _workflow_doc_status(theme: str, doc_id: str) -> dict[str, Any]:
            if is_public_attacks_theme(theme):
                # Public-attacks docs are intentionally outside the current
                # document-annotation workflow. Keep dashboard totals aligned
                # with Rules/QA by treating them as completed in this scope.
                history = _history_doc_status(theme, doc_id)
                return {
                    "status": "completed",
                    "last_edited_by": history.get("username") if history else "admin",
                }

            info = workflow_docs.get((theme, doc_id))
            agreement = workflow_agreements.get((theme, doc_id))
            history = _history_doc_status(theme, doc_id)
            feedback_state = feedback_state_map.get((theme, doc_id), {})
            agreement_status = str((agreement or {}).get("status") or "").lower()
            waiting_acceptance = bool(
                agreement_status == "resolved"
                and feedback_state.get("awaiting_reviewer_acceptance", False)
            )

            # Explicit admin completion should be respected even with an active run.
            if history and history["status"] == "completed" and history.get("role") == "power_user" and not waiting_acceptance:
                return {"status": "completed", "last_edited_by": history.get("username")}

            statuses = list(info.get("statuses", [])) if info else []
            assignees = list(info.get("assignees", [])) if info else []
            completed_reviews = sum(1 for s in statuses if s == "completed")

            if agreement:
                if agreement_status == "resolved":
                    status = "awaiting_reviewer_acceptance" if waiting_acceptance else "completed"
                    history_user = history.get("username") if history else None
                    last_editor = agreement.get("resolved_by") or history_user
                    if not last_editor and assignees:
                        last_editor = ", ".join(sorted(set(assignees)))
                    return {"status": status, "last_edited_by": last_editor}
                if agreement_status == "ready":
                    status = "agreement_ready"
                    last_editor = ", ".join(sorted(set(assignees))) if assignees else (history.get("username") if history else None)
                    return {"status": status, "last_edited_by": last_editor}

            # Two submissions are not "completed" until agreement is resolved.
            if completed_reviews >= 2:
                status = "agreement_ready"
                last_editor = ", ".join(sorted(set(assignees))) if assignees else (history.get("username") if history else None)
            elif any(s == "in_progress" for s in statuses) or completed_reviews == 1:
                status = "in_progress"
                last_editor = ", ".join(sorted(set(assignees))) if assignees else (history.get("username") if history else None)
            elif any(s == "available" for s in statuses):
                status = "queued"
                last_editor = ", ".join(sorted(set(assignees))) if assignees else (history.get("username") if history else None)
            elif history:
                status = history["status"]
                last_editor = history.get("username")
            else:
                status = "draft"
                last_editor = None

            return {"status": status, "last_edited_by": last_editor}

        def _source_agent_ai_status(theme: str, doc_id: str) -> dict[str, Any]:
            """Detect whether a source-agent draft exists for this document."""
            if not source_agent:
                return {"ai_annotated": False, "annotated_by": None}

            candidate = _find_source_agent_copy(
                theme,
                doc_id,
                source_agent,
                require_inline_annotations=True,
            )
            if candidate is None:
                return {"ai_annotated": False, "annotated_by": None}

            try:
                raw_text = candidate.read_text(encoding="utf-8")
            except Exception:
                return {"ai_annotated": False, "annotated_by": None}
            if not _has_inline_annotations(raw_text):
                return {"ai_annotated": False, "annotated_by": None}

            annotated_by = _extract_annotated_by_from_text(raw_text) or source_agent
            return {"ai_annotated": True, "annotated_by": annotated_by}

        themes_data = []
        total_completed = 0
        total_docs = 0

        present_themes = _canonical_themes_from_source()
        for theme in present_themes:
            theme_dir = _theme_source_dir(theme)
            docs = _list_source_yaml_docs(theme_dir, theme, include_retired=False)
            docs_info = []
            theme_completed = 0
            for yaml_path in docs:
                doc_id = yaml_path.stem
                status_info = _workflow_doc_status(theme, doc_id)
                ai_info = _source_agent_ai_status(theme, doc_id)
                status = status_info["status"]
                if status == "completed":
                    theme_completed += 1
                docs_info.append(
                    {
                        "doc_id": doc_id,
                        "status": status,
                        "last_edited_by": status_info["last_edited_by"],
                        "review_statuses": _doc_review_statuses(theme, doc_id, status),
                        "ai_annotated": bool(ai_info.get("ai_annotated")),
                        "annotated_by": ai_info.get("annotated_by"),
                    }
                )

            total_completed += theme_completed
            total_docs += len(docs)
            themes_data.append(
                    {
                        "theme_id": theme,
                        "theme_label": THEME_LABELS.get(theme, theme.replace("_", " ").title()),
                        "total": len(docs),
                        "completed": theme_completed,
                        "documents": docs_info,
                    }
                )

        legacy_docs = _list_legacy_docs() if PUBLIC_ATTACKS_THEME not in present_themes else []
        if legacy_docs:
            legacy_info = []
            legacy_completed = 0
            for yaml_path in legacy_docs:
                doc_id = yaml_path.stem
                status_info = _workflow_doc_status(PUBLIC_ATTACKS_THEME, doc_id)
                status = status_info["status"]
                if status == "completed":
                    legacy_completed += 1
                legacy_info.append(
                    {
                        "doc_id": doc_id,
                        "status": status,
                        "last_edited_by": status_info["last_edited_by"],
                        "review_statuses": _doc_review_statuses(PUBLIC_ATTACKS_THEME, doc_id, status),
                    }
                )
            total_completed += legacy_completed
            total_docs += len(legacy_docs)
            themes_data.append(
                {
                    "theme_id": PUBLIC_ATTACKS_THEME,
                    "theme_label": THEME_LABELS[PUBLIC_ATTACKS_THEME],
                    "total": len(legacy_docs),
                    "completed": legacy_completed,
                    "documents": legacy_info,
                }
            )

        return {"total": total_docs, "completed": total_completed, "themes": themes_data}

    themes_data = []
    total_completed = 0
    total_docs = 0

    present_themes = _canonical_themes_from_source()
    for theme in present_themes:
        theme_dir = _theme_source_dir(theme)
        source_docs = _list_source_yaml_docs(theme_dir, theme, include_retired=False)
        docs_info = []
        theme_completed = 0

        def _latest_history_row(theme_id: str, doc_id: str):
            candidates = _theme_history_paths(theme_id, doc_id)
            placeholders = ",".join("?" for _ in candidates)
            return db.execute(
                f"""SELECT u.username, h.action, h.status, h.timestamp
                    FROM document_history h JOIN users u ON h.user_id = u.id
                    WHERE h.document_path IN ({placeholders})
                    ORDER BY h.timestamp DESC, h.id DESC LIMIT 1""",
                tuple(candidates),
            ).fetchone()

        for yaml_path in source_docs:
            doc_id = yaml_path.stem
            row = _latest_history_row(theme, doc_id)

            status = row["status"] if row else "draft"
            last_edited_by = row["username"] if row else None

            # Check if document was AI-annotated
            ai_annotated = False
            annotated_by = None
            try:
                # Look for any user's working copy to check for AI annotation
                ai_copy_path = _find_ai_annotated_copy(theme, doc_id, exclude_username="")
                if ai_copy_path:
                    raw = _load_raw_yaml(ai_copy_path)
                    if raw and raw.get("document", {}).get("annotated_by"):
                        ai_annotated = True
                        annotated_by = raw["document"]["annotated_by"]
            except Exception:
                pass

            if status in ("completed", "validated"):
                theme_completed += 1

            docs_info.append({
                "doc_id": doc_id,
                "status": status,
                "last_edited_by": last_edited_by,
                "review_statuses": _doc_review_statuses(theme, doc_id, status),
                "ai_annotated": ai_annotated,
                "annotated_by": annotated_by,
            })

        total_completed += theme_completed
        total_docs += len(source_docs)
        themes_data.append({
            "theme_id": theme,
            "theme_label": THEME_LABELS.get(theme, theme.replace("_", " ").title()),
            "total": len(source_docs),
            "completed": theme_completed,
            "documents": docs_info,
        })

    # Legacy reference annotations (check database for actual status)
    legacy_docs = _list_legacy_docs() if PUBLIC_ATTACKS_THEME not in present_themes else []
    if legacy_docs:
        legacy_info = []
        legacy_completed = 0
        for yaml_path in legacy_docs:
            doc_id = yaml_path.stem
            candidates = _theme_history_paths(PUBLIC_ATTACKS_THEME, doc_id)
            placeholders = ",".join("?" for _ in candidates)
            row = db.execute(
                f"""SELECT u.username, h.action, h.status, h.timestamp
                    FROM document_history h JOIN users u ON h.user_id = u.id
                    WHERE h.document_path IN ({placeholders})
                    ORDER BY h.timestamp DESC, h.id DESC LIMIT 1""",
                tuple(candidates),
            ).fetchone()
            
            status = row["status"] if row else "validated"
            last_edited_by = row["username"] if row else "anonymous_reference"
            
            if status in ("completed", "validated"):
                legacy_completed += 1
            
            legacy_info.append({
                "doc_id": doc_id,
                "status": status,
                "last_edited_by": last_edited_by,
                "review_statuses": _doc_review_statuses(PUBLIC_ATTACKS_THEME, doc_id, status),
            })
        legacy_count = len(legacy_docs)
        total_completed += legacy_completed
        total_docs += legacy_count
        themes_data.append({
            "theme_id": PUBLIC_ATTACKS_THEME,
            "theme_label": THEME_LABELS[PUBLIC_ATTACKS_THEME],
            "total": legacy_count,
            "completed": legacy_completed,
            "documents": legacy_info,
        })

    return {"total": total_docs, "completed": total_completed, "themes": themes_data}


# --- Migration ---

def migrate_document(from_username: str, doc_id: str, to_dir: str = "anonymous_reference",
                     theme: str | None = None) -> None:
    # Prefer theme-organised path; fall back to legacy flat path
    source = _user_doc_path(from_username, theme, doc_id) if theme else None
    if source is None or not source.exists():
        source = _user_doc_path_legacy(from_username, doc_id)
    if not source.exists():
        raise FileNotFoundError(f"User copy not found for {from_username}/{doc_id}")
    dest_dir = WORK_DIR / to_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(source), str(dest_dir / f"{doc_id}.yaml"))


def _load_raw_yaml(yaml_path: Path) -> dict[str, Any] | None:
    try:
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data and "document" in data:
            return data
    except Exception:
        pass
    return None
