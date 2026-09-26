"""Separate visible answers from reasoning traces emitted by local models."""

from __future__ import annotations

import re


def extract_reasoning_from_content(content: str) -> tuple[str, str]:
    """
    Extract reasoning content from model response using vLLM-compatible logic.

    vLLM's approach:
    - Everything between <think> and </think> → reasoning_content
    - Everything after </think> → content
    - If no closing tag: everything after opening tag → reasoning_content, content = None (empty)
    - If no opening tag: everything → content, reasoning_content = None (empty)

    Handles multiple <think> blocks by extracting all and combining them.
    Also handles <think> tags and backticked variants for compatibility.

    Returns:
        tuple: (content, reasoning_content)
    """
    import re

    reasoning_parts = []
    cleaned_content = content

    # Try <think>...</think> (vLLM format, used by some models)
    # Extract ALL blocks and remove them from content
    redacted_pattern = r"<think>(.*?)</think>"
    matches = re.findall(redacted_pattern, content, re.DOTALL | re.IGNORECASE)
    if matches:
        reasoning_parts.extend([m.strip() for m in matches])
        # Remove ALL <think>...</think> blocks from content
        cleaned_content = re.sub(redacted_pattern, "", content, flags=re.DOTALL | re.IGNORECASE).strip()
        # Handle unclosed tag at the end
        if "<think>" in cleaned_content:
            # Handle case where last tag is not closed (everything after last <think> is reasoning)
            parts = cleaned_content.rsplit("<think>", 1)
            if len(parts) == 2:
                cleaned_content = parts[0].strip()
                reasoning_parts.append(parts[1].strip())

    # Try <think>...</think> (alternative format)
    # Extract ALL blocks and remove them from content
    think_pattern = r"<think>(.*?)</think>"
    matches = re.findall(think_pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
    if matches:
        reasoning_parts.extend([m.strip() for m in matches])
        # Remove ALL <think>...</think> blocks from content
        cleaned_content = re.sub(think_pattern, "", cleaned_content, flags=re.DOTALL | re.IGNORECASE).strip()
        # Handle unclosed tag at the end
        if "<think>" in cleaned_content:
            parts = cleaned_content.rsplit("<think>", 1)
            if len(parts) == 2:
                cleaned_content = parts[0].strip()
                reasoning_parts.append(parts[1].strip())

    # Try backticked variants `<think>`...`</think>` (alternative format for some models)
    think_backtick_pattern = r"`<think>`(.*?)`</think>`"
    matches = re.findall(think_backtick_pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
    if matches:
        reasoning_parts.extend([m.strip() for m in matches])
        cleaned_content = re.sub(think_backtick_pattern, "", cleaned_content, flags=re.DOTALL | re.IGNORECASE).strip()

    # Try <reasoning>...</reasoning> (fallback)
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    matches = re.findall(reasoning_pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
    if matches:
        reasoning_parts.extend([m.strip() for m in matches])
        cleaned_content = re.sub(reasoning_pattern, "", cleaned_content, flags=re.DOTALL | re.IGNORECASE).strip()

    # Magistral uses [THINK] ... [/THINK] special tokens.
    square_think_pattern = r"\[THINK\](.*?)\[/THINK\]"
    matches = re.findall(square_think_pattern, cleaned_content, re.DOTALL | re.IGNORECASE)
    if matches:
        reasoning_parts.extend([m.strip() for m in matches])
        cleaned_content = re.sub(square_think_pattern, "", cleaned_content, flags=re.DOTALL | re.IGNORECASE).strip()

    # Combine all reasoning parts
    reasoning_content = "\n\n".join(reasoning_parts).strip() if reasoning_parts else ""

    # Some reasoning models emit the final "ANSWER: ..." inside the thinking block
    # before ever closing it. Recover that visible answer when it is explicit.
    if reasoning_content and not cleaned_content:
        answer_matches = list(
            re.finditer(
                r"(?:^|\n)(?:final answer|answer)\s*[:\-]\s*(.+)$",
                reasoning_content,
                flags=re.IGNORECASE | re.DOTALL,
            )
        )
        if answer_matches:
            last_match = answer_matches[-1]
            cleaned_content = last_match.group(1).strip()
            reasoning_content = reasoning_content[: last_match.start()].strip()

    # If no reasoning was extracted but tags exist, handle unclosed tags
    if not reasoning_content:
        # Check for unclosed <think> at the end
        if "<think>" in content and "</think>" not in content:
            parts = content.rsplit("<think>", 1)
            if len(parts) == 2:
                cleaned_content = parts[0].strip()
                reasoning_content = parts[1].strip()
        # Check for unclosed <think> at the end
        elif "<think>" in content and "</think>" not in content and "`<think>`" not in content:
            parts = content.rsplit("<think>", 1)
            if len(parts) == 2:
                cleaned_content = parts[0].strip()
                reasoning_content = parts[1].strip()
        elif "[THINK]" in content and "[/THINK]" not in content:
            parts = re.split(r"\[THINK\]", content, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                cleaned_content = parts[0].strip()
                reasoning_content = parts[1].strip()

    return cleaned_content, reasoning_content


def extract_gpt_oss_harmony_content(content: str) -> tuple[str, str]:
    """
    Extract final and reasoning channels from GPT-OSS Harmony text.

    GPT-OSS emits assistant messages across `analysis` and `final` channels.
    We preserve the final channel as the answer content and aggregate any
    analysis-channel messages as reasoning.
    """
    analysis_matches = re.findall(
        r"<\|channel\|>analysis<\|message\|>(.*?)(?=<\|end\|>|<\|return\|>)",
        content,
        flags=re.DOTALL,
    )
    final_matches = re.findall(
        r"<\|start\|>assistant(?:[^<]*)<\|channel\|>final<\|message\|>(.*?)(?=<\|return\|>|<\|end\|>)",
        content,
        flags=re.DOTALL,
    )

    if final_matches or analysis_matches:
        reasoning = "\n\n".join(m.strip() for m in analysis_matches if m.strip()).strip()
        final = final_matches[-1].strip() if final_matches else ""
        return final, reasoning

    # Fallback for partially stripped Harmony outputs from older runs.
    if content.startswith("analysis"):
        final_match = re.search(
            r"(?:assistantfinalANSWER|Final Answer|ANSWER)\s*:\s*(.+)$",
            content,
            flags=re.DOTALL,
        )
        if final_match:
            final = final_match.group(1).strip()
            reasoning = content[: final_match.start()].strip()
            return final, reasoning

    return extract_reasoning_from_content(content)
