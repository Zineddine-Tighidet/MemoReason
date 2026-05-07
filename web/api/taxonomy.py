"""Taxonomy management API endpoints (power users only)."""

import logging
from pathlib import Path
import re
from typing import Any, Dict, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException

from web.middleware.auth import require_power_user

router = APIRouter(prefix="/api/v1")

TAXONOMY_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "WikiEvent" / "entity_taxonomy_extended.yaml"
TAXONOMY_DOC_FILE = TAXONOMY_FILE.parent.parent / "TAXONOMY.md"
# Backward-compatible alias used by older tests and callers.
README_FILE = TAXONOMY_DOC_FILE

logger = logging.getLogger(__name__)


def _load_taxonomy_yaml_only() -> Dict[str, Any]:
    """Load taxonomy YAML without attempting markdown sync."""
    if not TAXONOMY_FILE.exists():
        return {"entities": {}}

    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {"entities": {}}


def save_extended_taxonomy(data: Dict[str, Any]) -> None:
    """Save the extended taxonomy to YAML file."""
    TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(TAXONOMY_FILE, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _extract_taxonomy_doc_section(markdown_content: str) -> Optional[str]:
    """Extract the taxonomy section bounded by 'Entity Types' and 'Rules' headings."""
    pattern = re.compile(
        r"(?ms)^## Entity Types\s*\n(?P<section>.*?)^## \[Rules\]\(#rules\)\s*$"
    )
    match = pattern.search(markdown_content)
    if not match:
        return None
    section = match.group("section").strip()
    if section:
        return f"## Entity Types\n\n{section}"
    return "## Entity Types"


def _strip_inline_code(text: str) -> str:
    text = text.strip()
    if text.startswith("`") and text.endswith("`") and len(text) >= 2:
        return text[1:-1].strip()
    return text


def _split_markdown_row(row: str) -> list[str]:
    """Split a markdown table row while preserving escaped pipes inside cells."""
    row = row.strip()
    if not row.startswith("|"):
        return []
    row = row[1:]
    if row.endswith("|"):
        row = row[:-1]

    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for ch in row:
        if escaped:
            current.append(ch)
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    return cells


def _parse_examples_cell(examples_cell: str) -> list[str]:
    value = examples_cell.strip()
    if not value or value == "-":
        return []

    backtick_examples = re.findall(r"`([^`]*)`", value)
    if backtick_examples:
        return [ex.strip() for ex in backtick_examples if ex.strip()]

    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_taxonomy_doc_section(section: str) -> Dict[str, Any]:
    """Parse taxonomy markdown back into the extended taxonomy YAML shape."""
    entity_header_pattern = re.compile(
        r"^###\s+(?:\d+\.\s+)?(.+?)\s+\(`(?P<entity_type>[a-z_]+)_X`\)\s*$",
        flags=re.MULTILINE,
    )
    header_matches = list(entity_header_pattern.finditer(section))
    if not header_matches:
        raise ValueError("No entity sections found in TAXONOMY.md taxonomy section.")

    entities: Dict[str, Any] = {}

    for idx, header_match in enumerate(header_matches):
        entity_type = header_match.group("entity_type").strip()
        block_start = header_match.end()
        block_end = header_matches[idx + 1].start() if idx + 1 < len(header_matches) else len(section)
        block = section[block_start:block_end]
        block_lines = block.splitlines()

        table_header_idx = None
        for i, line in enumerate(block_lines):
            if line.strip().startswith("| Attribute |"):
                table_header_idx = i
                break
        if table_header_idx is None:
            continue

        description_lines: list[str] = []
        for line in block_lines[:table_header_idx]:
            stripped = line.strip()
            if not stripped or stripped == "---":
                continue
            description_lines.append(stripped)
        description = " ".join(description_lines).strip()

        attributes: Dict[str, Dict[str, Any]] = {}
        for row in block_lines[table_header_idx + 2:]:
            stripped = row.strip()
            if not stripped or stripped == "---":
                break
            if not stripped.startswith("|"):
                continue
            cells = _split_markdown_row(stripped)
            if len(cells) < 3:
                continue

            attr_name = _strip_inline_code(cells[0])
            if not attr_name or attr_name.lower() == "attribute":
                continue
            attr_description = cells[1].strip()
            if attr_description == "-":
                attr_description = ""
            attributes[attr_name] = {
                "description": attr_description,
                "examples": _parse_examples_cell(cells[2]),
            }

        entities[entity_type] = {
            "description": description,
            "attributes": attributes,
        }

    if not entities:
        raise ValueError("Failed to parse entities from TAXONOMY.md taxonomy section.")

    return {"entities": entities}


def _normalize_markdown(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return "\n".join(lines)


def _sync_taxonomy_yaml_from_doc_if_needed() -> Dict[str, Any]:
    """Sync TAXONOMY.md -> YAML when TAXONOMY.md taxonomy section has newer, divergent content."""
    taxonomy = _load_taxonomy_yaml_only()
    if not README_FILE.exists():
        return taxonomy

    try:
        markdown_content = README_FILE.read_text(encoding="utf-8")
        taxonomy_section = _extract_taxonomy_doc_section(markdown_content)
        if not taxonomy_section:
            return taxonomy
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read TAXONOMY.md for taxonomy sync: %s", exc)
        return taxonomy

    if not TAXONOMY_FILE.exists():
        try:
            parsed = _parse_taxonomy_doc_section(taxonomy_section)
            save_extended_taxonomy(parsed)
            return parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize taxonomy YAML from TAXONOMY.md: %s", exc)
            return taxonomy

    markdown_mtime = README_FILE.stat().st_mtime
    yaml_mtime = TAXONOMY_FILE.stat().st_mtime
    if markdown_mtime <= yaml_mtime:
        return taxonomy

    try:
        yaml_section = _render_taxonomy_markdown_section(taxonomy)
    except Exception:  # noqa: BLE001
        yaml_section = ""

    if _normalize_markdown(taxonomy_section) == _normalize_markdown(yaml_section):
        return taxonomy

    try:
        parsed = _parse_taxonomy_doc_section(taxonomy_section)
        save_extended_taxonomy(parsed)
        return parsed
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to sync taxonomy YAML from TAXONOMY.md: %s", exc)
        return taxonomy


def load_extended_taxonomy(
    sync_from_doc: bool = True,
    *,
    sync_from_readme: Optional[bool] = None,
) -> Dict[str, Any]:
    """Load extended taxonomy, optionally syncing taxonomy markdown edits into YAML first."""
    if sync_from_readme is not None:
        sync_from_doc = bool(sync_from_readme)
    if sync_from_doc:
        return _sync_taxonomy_yaml_from_doc_if_needed()
    return _load_taxonomy_yaml_only()


def _render_readme_taxonomy_section(taxonomy: Dict[str, Any]) -> str:
    """Backward-compatible alias for taxonomy markdown rendering."""
    return _render_taxonomy_markdown_section(taxonomy)


def _parse_readme_taxonomy_section(section: str) -> Dict[str, Any]:
    """Backward-compatible alias for taxonomy markdown parsing."""
    return _parse_taxonomy_doc_section(section)


def _pretty_entity_name(entity_type: str) -> str:
    return " ".join(part.capitalize() for part in entity_type.split("_"))


def _escape_md_cell(text: str) -> str:
    return str(text).replace("|", r"\|").replace("\n", " ").strip()


def _fmt_examples(examples: list[Any]) -> str:
    if not examples:
        return "-"
    return ", ".join(f"`{_escape_md_cell(ex)}`" for ex in examples[:3])


def _render_taxonomy_markdown_section(taxonomy: Dict[str, Any]) -> str:
    entities = taxonomy.get("entities", {})
    if not entities:
        raise ValueError("No entities found in taxonomy.")

    lines: list[str] = []
    lines.append("## Entity Types")
    lines.append("")
    lines.append(f"The data model uses **{len(entities)} entity types**:")
    lines.append("")
    for idx, entity_type in enumerate(entities.keys(), start=1):
        lines.append(f"{idx}. **{_pretty_entity_name(entity_type)}** (`{entity_type}_X`)")
    lines.append("")
    lines.append(
        "Each entity has a unique identifier (e.g., `person_1`, `place_2`, `event_1`) "
        "that must remain consistent across the document, questions, and answers."
    )
    lines.append("")
    lines.append("## Entity Attributes:")
    lines.append("")

    for idx, (entity_type, entity_data) in enumerate(entities.items(), start=1):
        lines.append(f"### {idx}. {_pretty_entity_name(entity_type)} (`{entity_type}_X`)")
        lines.append("")
        description = _escape_md_cell(entity_data.get("description", ""))
        if description:
            lines.append(description)
            lines.append("")
        lines.append("| Attribute | Description | Example(s) |")
        lines.append("|-----------|-------------|------------|")
        attributes = entity_data.get("attributes", {}) or {}
        for attr_name, attr_data in attributes.items():
            attr_desc = _escape_md_cell((attr_data or {}).get("description", "")) or "-"
            examples = (attr_data or {}).get("examples", []) or []
            lines.append(
                f"| `{_escape_md_cell(attr_name)}` | {attr_desc} | {_fmt_examples(examples)} |"
            )
        lines.append("")
        if entity_type == "person" and "relationship" in attributes:
            lines.append(
                "Note: use `person_X.relationship` for generic labels and "
                "`person_X.relationship.person_Y` for directed person-to-person labels."
            )
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).strip()


def _sync_taxonomy_doc_section(markdown_path: Path, taxonomy: Dict[str, Any]) -> None:
    markdown_content = markdown_path.read_text(encoding="utf-8")
    generated = _render_taxonomy_markdown_section(taxonomy)
    pattern = re.compile(
        r"## Entity Types\n.*?\n## \[Rules\]\(#rules\)\s*",
        flags=re.DOTALL,
    )
    replacement = f"{generated}\n\n## [Rules](#rules)\n\n"
    if not pattern.search(markdown_content):
        raise ValueError("Could not locate 'Entity Types' -> 'Rules' section in TAXONOMY.md.")
    updated = pattern.sub(replacement, markdown_content, count=1)
    markdown_path.write_text(updated, encoding="utf-8")


@router.get("/taxonomy/extended")
def api_get_extended_taxonomy(user: Dict = Depends(require_power_user)) -> Dict[str, Any]:
    """Get the full extended taxonomy with descriptions and examples."""
    return load_extended_taxonomy()


@router.put("/taxonomy/extended")
def api_update_extended_taxonomy(
    taxonomy_data: Dict[str, Any],
    user: Dict = Depends(require_power_user)
) -> Dict[str, str]:
    """Update the extended taxonomy."""
    try:
        save_extended_taxonomy(taxonomy_data)
        return {"status": "saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save taxonomy: {str(e)}")


@router.post("/taxonomy/entity")
def api_add_entity_type(
    entity_data: Dict[str, Any],
    user: Dict = Depends(require_power_user)
) -> Dict[str, str]:
    """Add a new entity type to the taxonomy."""
    entity_type = entity_data.get("entity_type")
    description = entity_data.get("description", "")

    if not entity_type:
        raise HTTPException(status_code=400, detail="entity_type is required")

    taxonomy = load_extended_taxonomy()

    if entity_type in taxonomy["entities"]:
        raise HTTPException(status_code=400, detail=f"Entity type '{entity_type}' already exists")

    taxonomy["entities"][entity_type] = {
        "description": description,
        "attributes": {}
    }

    save_extended_taxonomy(taxonomy)
    return {"status": "added", "entity_type": entity_type}


@router.delete("/taxonomy/entity/{entity_type}")
def api_delete_entity_type(
    entity_type: str,
    user: Dict = Depends(require_power_user)
) -> Dict[str, str]:
    """Delete an entity type from the taxonomy."""
    taxonomy = load_extended_taxonomy()

    if entity_type not in taxonomy["entities"]:
        raise HTTPException(status_code=404, detail=f"Entity type '{entity_type}' not found")

    del taxonomy["entities"][entity_type]
    save_extended_taxonomy(taxonomy)
    return {"status": "deleted", "entity_type": entity_type}


@router.post("/taxonomy/entity/{entity_type}/attribute")
def api_add_attribute(
    entity_type: str,
    attr_data: Dict[str, Any],
    user: Dict = Depends(require_power_user)
) -> Dict[str, str]:
    """Add a new attribute to an entity type."""
    attr_name = attr_data.get("attr_name")
    description = attr_data.get("description", "")
    examples = attr_data.get("examples", [])

    if not attr_name:
        raise HTTPException(status_code=400, detail="attr_name is required")

    taxonomy = load_extended_taxonomy()

    if entity_type not in taxonomy["entities"]:
        raise HTTPException(status_code=404, detail=f"Entity type '{entity_type}' not found")

    if attr_name in taxonomy["entities"][entity_type]["attributes"]:
        raise HTTPException(status_code=400, detail=f"Attribute '{attr_name}' already exists")

    taxonomy["entities"][entity_type]["attributes"][attr_name] = {
        "description": description,
        "examples": examples
    }

    save_extended_taxonomy(taxonomy)
    return {"status": "added", "attr_name": attr_name}


@router.put("/taxonomy/entity/{entity_type}/attribute/{attr_name}")
def api_update_attribute(
    entity_type: str,
    attr_name: str,
    attr_data: Dict[str, Any],
    user: Dict = Depends(require_power_user)
) -> Dict[str, str]:
    """Update an attribute's description and examples."""
    taxonomy = load_extended_taxonomy()

    if entity_type not in taxonomy["entities"]:
        raise HTTPException(status_code=404, detail=f"Entity type '{entity_type}' not found")

    if attr_name not in taxonomy["entities"][entity_type]["attributes"]:
        raise HTTPException(status_code=404, detail=f"Attribute '{attr_name}' not found")

    taxonomy["entities"][entity_type]["attributes"][attr_name] = {
        "description": attr_data.get("description", ""),
        "examples": attr_data.get("examples", [])
    }

    save_extended_taxonomy(taxonomy)
    return {"status": "updated", "attr_name": attr_name}


@router.delete("/taxonomy/entity/{entity_type}/attribute/{attr_name}")
def api_delete_attribute(
    entity_type: str,
    attr_name: str,
    user: Dict = Depends(require_power_user)
) -> Dict[str, str]:
    """Delete an attribute from an entity type."""
    taxonomy = load_extended_taxonomy()

    if entity_type not in taxonomy["entities"]:
        raise HTTPException(status_code=404, detail=f"Entity type '{entity_type}' not found")

    if attr_name not in taxonomy["entities"][entity_type]["attributes"]:
        raise HTTPException(status_code=404, detail=f"Attribute '{attr_name}' not found")

    del taxonomy["entities"][entity_type]["attributes"][attr_name]
    save_extended_taxonomy(taxonomy)
    return {"status": "deleted", "attr_name": attr_name}


@router.get("/taxonomy/export/markdown")
def api_export_taxonomy_markdown(user: Dict = Depends(require_power_user)) -> Dict[str, str]:
    """Export taxonomy to markdown format."""
    taxonomy = load_extended_taxonomy()

    markdown = "# Entity Taxonomy\n\n"
    markdown += "This document describes the entity types and their attributes used for annotation.\n\n"

    for entity_type, entity_data in taxonomy["entities"].items():
        markdown += f"## {entity_type}\n\n"
        markdown += f"{entity_data.get('description', '')}\n\n"

        attributes = entity_data.get("attributes", {})
        if attributes:
            markdown += "### Attributes\n\n"
            markdown += "| Attribute | Description | Examples |\n"
            markdown += "|-----------|-------------|----------|\n"

            for attr_name, attr_data in attributes.items():
                desc = attr_data.get("description", "")
                examples = ", ".join([f"`{ex}`" for ex in attr_data.get("examples", [])])
                markdown += f"| `{attr_name}` | {desc} | {examples} |\n"

            markdown += "\n"

    return {"markdown": markdown}


@router.get("/taxonomy/export/readme")
def api_export_taxonomy_readme_alias(user: Dict = Depends(require_power_user)) -> Dict[str, str]:
    """Backward-compatible alias for older frontends."""
    return api_export_taxonomy_markdown(user)


@router.post("/taxonomy/update-doc")
def api_update_taxonomy_doc(user: Dict = Depends(require_power_user)) -> Dict[str, str]:
    """Update TAXONOMY.md taxonomy section from the current taxonomy YAML."""
    if not README_FILE.exists():
        raise HTTPException(status_code=404, detail="TAXONOMY.md not found")

    taxonomy = load_extended_taxonomy()
    try:
        _sync_taxonomy_doc_section(README_FILE, taxonomy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Failed to update TAXONOMY.md: {exc}")

    return {"status": "updated", "message": "TAXONOMY.md taxonomy section synced successfully"}


@router.post("/taxonomy/update-readme")
def api_update_readme_alias(user: Dict = Depends(require_power_user)) -> Dict[str, str]:
    """Backward-compatible alias for older frontends."""
    return api_update_taxonomy_doc(user)
