"""Exact-match and judge-based scoring for parsed model outputs."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time

from src.llm.text_generation import TextGenerationRequest, generate_text
from src.dataset_export.dataset_paths import DEFAULT_RANDOM_SEED
from .prompting import JUDGE_SYSTEM_PROMPT, build_judge_prompt
from .answer_handling import score_canonical_prediction, score_prediction_with_schema


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for the optional LLM-as-a-judge pass."""

    provider: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 8
    seed: int | None = DEFAULT_RANDOM_SEED


def exact_match_is_correct(predicted_answer: str, ground_truth: str) -> bool:
    """Backward-compatible two-string scorer for older call sites."""
    return score_canonical_prediction(predicted_answer, (ground_truth,))


def accepted_answer_match_is_correct(
    predicted_canonical: str,
    accepted_answers_canonical: tuple[str, ...],
    *,
    answer_schema: str | None = None,
    raw_prediction: str | None = None,
) -> bool:
    """Apply deterministic exact match against the accepted canonical answer set."""
    if answer_schema:
        return score_prediction_with_schema(
            predicted_canonical,
            accepted_answers_canonical,
            answer_schema=answer_schema,
            raw_prediction=raw_prediction,
        )
    return score_canonical_prediction(predicted_canonical, accepted_answers_canonical)


def judge_match_is_allowed(
    *,
    answer_schema: str | None,
    parsed_output_canonical: str | None,
) -> bool:
    """Return whether a non-exact prediction should be sent to the judge.

    The evaluation contract is intentionally simple: exact-match successes do
    not need LLM-as-a-judge, but every exact-match miss should be judged. This
    avoids silently marking partially parsed but semantically valid answers as
    incorrect.
    """
    return True


def judge_prediction(
    *,
    question_text: str,
    ground_truth: str,
    predicted_answer: str,
    judge_config: JudgeConfig,
) -> tuple[bool, str]:
    """Run the judge model and return ``(is_correct, raw_judge_output)``."""
    def parse_verdict(text: str) -> bool | None:
        normalized = re.sub(r"\s+", " ", str(text or "").strip().upper())
        trimmed = normalized.lstrip(" `\"'([{:-")
        if trimmed.startswith("INCORRECT"):
            return False
        if trimmed.startswith("CORRECT"):
            return True
        match = re.search(r"\bVERDICT\s*[:=-]\s*(INCORRECT|CORRECT)\b", normalized)
        if match:
            return match.group(1) == "CORRECT"
        verdict_tokens = re.findall(r"\b(INCORRECT|CORRECT)\b", normalized)
        if verdict_tokens:
            return verdict_tokens[-1] == "CORRECT"
        return None

    last_raw_output = ""
    for attempt_index in range(4):
        response = generate_text(
            TextGenerationRequest(
                provider=judge_config.provider,
                model=judge_config.model_name,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_prompt=build_judge_prompt(question_text, ground_truth, predicted_answer),
                temperature=judge_config.temperature,
                max_tokens=max(int(judge_config.max_tokens), 64) * (2**attempt_index),
                seed=judge_config.seed,
            )
        )
        last_raw_output = response.text or response.reasoning_text or response.raw_response
        for candidate_text in (response.text, response.reasoning_text):
            verdict = parse_verdict(candidate_text)
            if verdict is not None:
                return verdict, response.text or response.reasoning_text
        if attempt_index < 3:
            time.sleep(0.5 * (attempt_index + 1))
    raise ValueError(f"Judge response is not parseable as CORRECT/INCORRECT: {last_raw_output!r}")
