"""Define prompt instructions and token budgets for each answer schema."""

from __future__ import annotations

from .answer_schema_data_contracts import UNANSWERABLE


def answer_format_instructions(answer_schema: str) -> str:
    """Return the schema-specific output contract used in prompting."""
    if answer_schema == "yes_no":
        return f"Expected answer type: YES_NO\nAllowed values:\n- YES\n- NO\n- {UNANSWERABLE}"
    if answer_schema == "before_after":
        return f"Expected answer type: BEFORE_AFTER\nAllowed values:\n- BEFORE\n- AFTER\n- {UNANSWERABLE}"
    if answer_schema == "quantity":
        return (
            "Expected answer type: QUANTITY\n"
            "Rules:\n"
            "- Output a quantity only.\n"
            "- Digits or number words are both acceptable.\n"
            "- No units or explanation.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    if answer_schema == "year":
        return (
            "Expected answer type: YEAR\n"
            "Rules:\n"
            "- Output one 4-digit year only.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    if answer_schema == "date":
        return (
            "Expected answer type: DATE\n"
            "Rules:\n"
            "- Output one date only.\n"
            "- Prefer the exact date stated in the document.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    if answer_schema == "entity_span":
        return (
            "Expected answer type: ENTITY_SPAN\n"
            "Rules:\n"
            "- Copy the shortest entity name span from the document.\n"
            "- Do not paraphrase.\n"
            f"- If the document does not determine the answer, output {UNANSWERABLE}."
        )
    return (
        "Expected answer type: TEXT_SPAN\n"
        "Rules:\n"
        "- Copy the shortest answer span from the document.\n"
        "- Do not paraphrase.\n"
        f"- If the document does not determine the answer, output {UNANSWERABLE}."
    )


def suggested_max_tokens_for_schema(answer_schema: str) -> int:
    """Return a conservative decoding budget for one answer schema."""
    if answer_schema in {"yes_no", "before_after"}:
        return 4
    if answer_schema in {"quantity", "year"}:
        return 8
    if answer_schema == "date":
        return 16
    return 24
