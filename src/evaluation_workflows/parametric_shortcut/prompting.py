"""Prompt templates used for direct-answer benchmark evaluation."""

from __future__ import annotations

from .answer_handling import suggested_max_tokens_for_schema


PROMPT_FORMAT_VERSION = "direct_answer_line_v1"

DOCUMENT_QA_SYSTEM_PROMPT = """
You answer questions using only the provided document.

Rules:
- Use only the document. Do not use outside knowledge.
- Output exactly one line in the format: ANSWER: <answer>
- Do not explain your reasoning.
- Do not output a calculation or equation.
- Do not restate the question.
- Do not add any extra text before or after the answer line.
- If the document does not determine the answer, output exactly: ANSWER: Cannot be determined
- Otherwise, return only the shortest final answer.
""".strip()


JUDGE_SYSTEM_PROMPT = """
You decide whether a predicted answer should count as correct against a ground-truth answer.

Rules:
- Minor formatting differences are acceptable.
- Equivalent abbreviations and full forms are acceptable.
- Numeric answers with equivalent value are acceptable.
- If the question asks for multiple pieces of information, judge whether the prediction correctly answers all requested parts.
- If the ground-truth string is shorter or partially normalized, do not mark the prediction incorrect just because it spells out the requested parts more explicitly.
- If the prediction includes additional correct information beyond what was asked, it can still be counted as correct as long as it clearly contains the requested answer and does not introduce any contradiction.
- Extra wording is acceptable when it directly answers the question and does not add a contradiction.
- If the prediction is wrong, incomplete, or contradictory, mark it incorrect.
- Respond with exactly one word: CORRECT or INCORRECT.
""".strip()


def build_document_question_prompt(
    document_text: str,
    question_text: str,
    *,
    answer_schema: str | None = None,
) -> str:
    """Build the user prompt for one document-question pair."""
    return (
        "[DOCUMENT]\n"
        f"{document_text}\n\n"
        "[QUESTION]\n"
        f"{question_text}"
    )


def build_judge_prompt(question_text: str, ground_truth: str, predicted_answer: str) -> str:
    """Build the user prompt for judge scoring."""
    return (
        f"Question: {question_text}\n"
        f"Ground truth answer: {ground_truth}\n"
        f"Predicted answer: {predicted_answer}\n\n"
        "Is the prediction correct?"
    )


def suggested_generation_max_tokens(*, answer_schema: str, model_name: str) -> int:
    """Return a schema-aware decoding budget with provider-specific headroom."""
    base_budget = suggested_max_tokens_for_schema(answer_schema)
    lowered = str(model_name).strip().lower()
    if lowered in {
        "magistral-small-2509",
        "mistralai/magistral-small-2509",
        "gpt-oss-20b",
        "gpt-oss-20b-groq",
        "gpt-oss-120b",
        "gpt-oss-120b-groq",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "phi-4-mini-flash-reasoning",
        "microsoft/phi-4-mini-flash-reasoning",
        "olmo-3-7b-instruct",
        "allenai/olmo-3-7b-instruct",
    }:
        return 10000 if lowered in {"gpt-oss-20b", "gpt-oss-20b-groq", "openai/gpt-oss-20b"} else 8192
    if lowered in {
        "olmo-3-7b-think",
        "allenai/olmo-3-7b-think",
    }:
        # OLMo Think can otherwise spend thousands of tokens reasoning for a
        # direct-answer QA. 1024 keeps a genuine thinking budget while making
        # full-corpus open-weight runs tractable.
        return 1024
    if lowered.startswith("claude-opus-4-7"):
        return max(256, base_budget * 8)
    if lowered.startswith("claude-sonnet-4-6"):
        return 512
    if lowered.startswith("gemini-3.1-pro"):
        # Gemini Pro counts hidden thinking tokens inside maxOutputTokens. The
        # schema-level visible-answer budgets are too small and cause empty
        # MAX_TOKENS responses before the final answer is emitted.
        return 768
    if lowered.startswith("gpt-oss-"):
        return max(64, base_budget * 4)
    if lowered.startswith("openreasoning-nemotron-") or lowered.startswith("nvidia/openreasoning-nemotron-"):
        return max(64, base_budget * 4)
    return base_budget


__all__ = [
    "DOCUMENT_QA_SYSTEM_PROMPT",
    "JUDGE_SYSTEM_PROMPT",
    "PROMPT_FORMAT_VERSION",
    "build_document_question_prompt",
    "build_judge_prompt",
    "suggested_generation_max_tokens",
    "suggested_max_tokens_for_schema",
]
