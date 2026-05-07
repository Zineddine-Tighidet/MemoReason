"""Parse raw model outputs into short, directly comparable answers."""

from __future__ import annotations

import re


_ANSWER_PREFIX_RE = re.compile(
    r"^(?:final answer|answer|predicted answer|the answer is|it is)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_INLINE_ANSWER_RE = re.compile(
    r"(?:assistantfinalanswer|final answer|answer)\s*[:\-]\s*(.+?)(?=(?:assistantfinalanswer|final answer|answer)\s*[:\-]|$)",
    re.IGNORECASE | re.DOTALL,
)
_BRACKETED_PREFIX_RE = re.compile(
    r"^(?:\[\s*(?:answer|final answer|response|output)\s*\]\s*)+",
    re.IGNORECASE,
)
_NUMERIC_CUE_RE = re.compile(
    r"(?:answer(?:\s+is)?|final answer|therefore|thus|so|=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
_YES_NO_RE = re.compile(r"^(yes|no|true|false)$", re.IGNORECASE)
_TRAILING_ANSWER_TOKEN_RE = re.compile(r"\s+ANSWER\s*$", re.IGNORECASE)
_DUPLICATED_ANSWER_RE = re.compile(r"^(.+?)\s+ANSWER\s+\1$", re.IGNORECASE)
_ANSWER_SPLIT_RE = re.compile(r"^(.*?)\s+ANSWER\s*:?\s*(.+)$", re.IGNORECASE | re.DOTALL)
_YEARISH_PREFIX_RE = re.compile(
    r"^((?:\d{1,2}\s+[A-Za-z]+\s+\d{4})|(?:[A-Za-z]+\s+\d{1,2},\s+\d{4})|(?:\d{4}(?:[–-]\d{2,4})?))"
)
_ASSISTANT_FINAL_INLINE_RE = re.compile(r"assistantfinalanswer\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
_LATEX_COMMAND_PREFIX_RE = re.compile(r"^\\(?:boxed|text)\s*\{", re.IGNORECASE)
_TRAILING_CHAT_CONTROL_RE = re.compile(
    r"(?:\s*(?:<\|[^<>|]+\|>|<[^<>|>]+\|>))+\s*$"
)


def _extract_balanced_braced_content(text: str, brace_start: int) -> tuple[str, int] | None:
    if brace_start < 0 or brace_start >= len(text) or text[brace_start] != "{":
        return None

    depth = 0
    content_chars: list[str] = []
    for index in range(brace_start, len(text)):
        char = text[index]
        if char == "{":
            if depth > 0:
                content_chars.append(char)
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return "".join(content_chars), index + 1
            if depth < 0:
                break
            content_chars.append(char)
            continue
        if depth > 0:
            content_chars.append(char)
    return None


def _extract_latex_command_contents(text: str, command: str) -> list[str]:
    marker = f"\\{command}"
    start = 0
    contents: list[str] = []
    while True:
        index = text.find(marker, start)
        if index == -1:
            break
        brace_start = index + len(marker)
        while brace_start < len(text) and text[brace_start].isspace():
            brace_start += 1
        extracted = _extract_balanced_braced_content(text, brace_start)
        if extracted is None:
            start = index + len(marker)
            continue
        content, end_index = extracted
        contents.append(content.strip())
        start = end_index
    return contents


def _unwrap_latex_wrappers(text: str) -> str:
    normalized = text.strip()
    for _ in range(4):
        matched_wrapper = False
        for command in ("boxed", "text"):
            command_contents = _extract_latex_command_contents(normalized, command)
            if not command_contents:
                continue
            prefix_match = _LATEX_COMMAND_PREFIX_RE.match(normalized)
            if prefix_match is None:
                continue
            normalized = command_contents[0].strip()
            matched_wrapper = True
            break
        if not matched_wrapper:
            break
    return normalized.strip()


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:\w+)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _sanitize_candidate(text: str) -> str:
    normalized = text.strip().strip("`").strip()
    if not normalized:
        return ""
    lowered = normalized.lower()
    if lowered in {"<answer>", "answer", "final answer"} or "<answer>" in lowered:
        return ""
    assistant_final_match = _ASSISTANT_FINAL_INLINE_RE.search(normalized)
    if assistant_final_match:
        normalized = assistant_final_match.group(1).strip()
    normalized = _BRACKETED_PREFIX_RE.sub("", normalized).strip()
    normalized = _ANSWER_PREFIX_RE.sub("", normalized).strip()
    normalized = re.sub(r"^(?:the answer is|it is)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = _unwrap_latex_wrappers(normalized)
    normalized = _BRACKETED_PREFIX_RE.sub("", normalized).strip()
    normalized = _ANSWER_PREFIX_RE.sub("", normalized).strip()
    normalized = re.sub(r"^(?:the answer is|it is)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = _TRAILING_ANSWER_TOKEN_RE.sub("", normalized).strip()
    normalized = _TRAILING_CHAT_CONTROL_RE.sub("", normalized).strip()
    if not normalized:
        return ""
    normalized = normalized.splitlines()[0].strip()
    normalized = _unwrap_latex_wrappers(normalized)
    normalized = _TRAILING_CHAT_CONTROL_RE.sub("", normalized).strip()
    duplicate_match = _DUPLICATED_ANSWER_RE.match(normalized)
    if duplicate_match:
        normalized = duplicate_match.group(1).strip()

    # If an answer starts with a recognizable year/date and then drifts into
    # punctuation-like garbage, keep only the temporal prefix. Do not clip
    # legitimate spans such as "2009 Nobel Peace Prize".
    yearish_match = _YEARISH_PREFIX_RE.match(normalized)
    if yearish_match:
        prefix = yearish_match.group(1).strip()
        suffix = normalized[len(yearish_match.group(1)) :].strip()
        if suffix and not re.search(r"[A-Za-z0-9\u00C0-\u024F]", suffix):
            normalized = prefix

    normalized = normalized.strip().strip("{}").rstrip("}").strip()
    return normalized.rstrip(".").strip()


def parse_short_answer(raw_text: str) -> str:
    """Extract the short answer from a model output with regex-based heuristics."""
    cleaned = _strip_code_fences(str(raw_text or ""))
    if not cleaned:
        return ""

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    candidates = []
    inline_matches = [match.group(1).strip() for match in _INLINE_ANSWER_RE.finditer(cleaned) if match.group(1).strip()]
    if inline_matches:
        candidates.extend(reversed(inline_matches))
    boxed_matches = [match for match in _extract_latex_command_contents(cleaned, "boxed") if match]
    if boxed_matches:
        candidates.extend(reversed(boxed_matches))
    assistant_final_match = _ASSISTANT_FINAL_INLINE_RE.search(cleaned)
    if assistant_final_match and assistant_final_match.group(1).strip():
        candidates.append(assistant_final_match.group(1).strip())
    split_match = _ANSWER_SPLIT_RE.match(cleaned)
    if split_match:
        left, right = split_match.groups()
        left = left.strip()
        right = right.strip()
        # Prefer the post-ANSWER span; the left side is often a reasoning blob
        # that merely happens to contain the token "ANSWER" later on.
        if right:
            candidates.append(right)
        if left and len(left.split()) <= 6:
            candidates.append(left)
    for line in reversed(lines):
        if _ANSWER_PREFIX_RE.search(line):
            candidates.append(_ANSWER_PREFIX_RE.sub("", line).strip())
    if lines:
        candidates.append(lines[0])
        candidates.append(lines[-1])
    candidates.append(cleaned)

    for candidate in candidates:
        normalized = _sanitize_candidate(candidate)
        if normalized == "/":
            normalized = ""
        if not normalized:
            continue
        if _YES_NO_RE.match(normalized):
            lowered = normalized.lower()
            if lowered in {"true", "yes"}:
                return "Yes"
            return "No"
        return normalized

    numeric_match = _NUMERIC_CUE_RE.search(cleaned)
    if numeric_match:
        return numeric_match.group(1).strip()

    if not _sanitize_candidate(cleaned):
        return ""

    if "<answer>" in cleaned.lower():
        return ""

    return cleaned
