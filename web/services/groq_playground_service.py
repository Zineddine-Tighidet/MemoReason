"""Groq-backed document QA playground service for the web editor."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib import error, request

GROQ_API_BASE = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1").rstrip("/")
DEFAULT_TIMEOUT_SECONDS = 90
MODEL_CACHE_TTL_SECONDS = 300

DEFAULT_SYSTEM_PROMPT = (
    "Answer strictly from the provided document and not from prior knowledge. "
    "If the document does not determine the answer, reply exactly with Cannot be determined. "
    "Keep each answer short and standalone."
)

_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, Any] = {"expires_at": 0.0, "models": []}


def groq_is_configured() -> bool:
    """Return whether a Groq API key is available."""
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def _groq_api_key() -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not configured on the server.")
    return api_key


def _coerce_json_bytes(payload: Any) -> bytes | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _parse_error_body(body: bytes) -> str:
    try:
        data = json.loads(body.decode("utf-8"))
    except Exception:
        return body.decode("utf-8", errors="replace").strip() or "Groq request failed."
    if isinstance(data, dict):
        nested = data.get("error")
        if isinstance(nested, dict):
            return str(nested.get("message") or nested.get("detail") or nested)
        detail = data.get("detail") or data.get("message")
        if detail:
            return str(detail)
    return str(data)


def _groq_request(path: str, *, method: str = "GET", payload: Any = None, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Any:
    data = _coerce_json_bytes(payload)
    req = request.Request(
        f"{GROQ_API_BASE}{path}",
        data=data,
        method=method.upper(),
        headers={
            "Authorization": f"Bearer {_groq_api_key()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ParametricShortcutQuickEval/1.0",
        },
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
    except error.HTTPError as exc:
        body = exc.read() if exc.fp is not None else b""
        detail = _parse_error_body(body)
        raise RuntimeError(detail) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Groq request failed: {exc.reason}") from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Groq returned a non-JSON response.") from exc


def _looks_chat_capable(model_id: str, raw_model: dict[str, Any]) -> bool:
    lowered = model_id.lower()
    blocked_keywords = (
        "whisper",
        "tts",
        "speech",
        "playai",
        "orpheus",
        "transcribe",
        "transcription",
        "prompt-guard",
        "embedding",
        "moderation",
    )
    if any(keyword in lowered for keyword in blocked_keywords):
        return False

    modalities = raw_model.get("input_modalities")
    if isinstance(modalities, list) and modalities and "text" not in {str(item).lower() for item in modalities}:
        return False
    return True


def list_groq_models(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Return cached Groq chat-capable models."""
    if not groq_is_configured():
        return []

    now = time.time()
    with _MODEL_CACHE_LOCK:
        cached_models = _MODEL_CACHE.get("models") or []
        if not force_refresh and cached_models and float(_MODEL_CACHE.get("expires_at") or 0.0) > now:
            return list(cached_models)

    response = _groq_request("/models", timeout=30)
    raw_models = response.get("data") if isinstance(response, dict) else []
    models: list[dict[str, Any]] = []
    for item in raw_models or []:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        if not _looks_chat_capable(model_id, item):
            continue
        models.append(
            {
                "id": model_id,
                "owned_by": str(item.get("owned_by") or "").strip(),
                "created": item.get("created"),
                "context_window": item.get("context_window"),
            }
        )

    models.sort(key=lambda model: model["id"])
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE["models"] = list(models)
        _MODEL_CACHE["expires_at"] = now + MODEL_CACHE_TTL_SECONDS
    return models


def _strip_inline_annotations(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    return re.sub(r"\[([^\]]+);\s*[^\]]+\]", r"\1", text)


def _normalize_question_id(raw_id: str, fallback_index: int, used_ids: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_id or "").strip()).strip("._-")
    if not base:
        base = f"q{fallback_index}"
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


def normalize_questions(raw_questions: Any) -> list[dict[str, str]]:
    """Normalize playground questions into a stable list."""
    normalized: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for idx, raw_question in enumerate(raw_questions or [], start=1):
        if isinstance(raw_question, str):
            question_text = raw_question
            raw_question_id = ""
        elif isinstance(raw_question, dict):
            question_text = raw_question.get("question_text") or raw_question.get("question") or ""
            raw_question_id = raw_question.get("question_id") or raw_question.get("id") or ""
        else:
            continue
        cleaned_text = _strip_inline_annotations(str(question_text or "")).strip()
        if not cleaned_text:
            continue
        question_id = _normalize_question_id(str(raw_question_id or ""), idx, used_ids)
        normalized.append(
            {
                "question_id": question_id,
                "question_text": cleaned_text,
            }
        )
    return normalized


def _normalize_models(raw_models: Any) -> list[str]:
    seen: set[str] = set()
    models: list[str] = []
    for raw_model in raw_models or []:
        model_id = str(raw_model or "").strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(model_id)
    return models


def _coerce_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content or "")


def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace:last_brace + 1])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_answer_map(raw_text: str) -> dict[str, str]:
    parsed = _extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return {}

    answers = parsed.get("answers")
    if isinstance(answers, list):
        result: dict[str, str] = {}
        for row in answers:
            if not isinstance(row, dict):
                continue
            question_id = str(row.get("question_id") or row.get("id") or "").strip()
            if not question_id:
                continue
            raw_answer = row.get("answer")
            if raw_answer is True:
                answer = "Yes"
            elif raw_answer is False:
                answer = "No"
            elif raw_answer is None:
                answer = ""
            else:
                answer = str(raw_answer).strip()
            result[question_id] = answer
        if result:
            return result

    result = {}
    for key, value in parsed.items():
        if key == "answers":
            continue
        if isinstance(value, (str, int, float, bool)):
            if value is True:
                result[str(key)] = "Yes"
            elif value is False:
                result[str(key)] = "No"
            else:
                result[str(key)] = str(value)
    return result


def _build_user_prompt(document_text: str, question: dict[str, str]) -> str:
    question_block = json.dumps([question], ensure_ascii=False, indent=2)
    return (
        "Use only the document below.\n"
        "Return JSON only, with no markdown and no extra commentary.\n"
        "Use exactly this format:\n"
        '{"answers":[{"question_id":"q1","answer":"..."},{"question_id":"q2","answer":"..."}]}\n\n'
        "Questions:\n"
        f"{question_block}\n\n"
        "Document:\n"
        "<<DOCUMENT>>\n"
        f"{document_text}\n"
        "<</DOCUMENT>>"
    )


def _request_chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }
    attempts = [
        {**base_payload, "max_completion_tokens": max_tokens},
        {**base_payload, "max_tokens": max_tokens},
    ]
    last_error: Exception | None = None
    for payload in attempts:
        try:
            return _groq_request("/chat/completions", method="POST", payload=payload)
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            if "max_completion_tokens" in payload and "max_completion_tokens" not in message:
                raise
            if "max_tokens" in payload and "max_tokens" not in message:
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Groq request failed.")


def _extract_reasoning(message: dict[str, Any]) -> str:
    for key in ("reasoning", "reasoning_content", "reasoning_text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list):
            text = _coerce_message_text(value).strip()
            if text:
                return text
    return ""


def _run_single_model(
    *,
    model: str,
    document_text: str,
    questions: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    start = time.perf_counter()

    def run_one(question: dict[str, str]) -> dict[str, Any]:
        question_start = time.perf_counter()
        try:
            response = _request_chat_completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _build_user_prompt(document_text, question)},
                ],
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            choices = response.get("choices") if isinstance(response, dict) else []
            first_choice = choices[0] if choices else {}
            message = first_choice.get("message") if isinstance(first_choice, dict) else {}
            raw_text = _coerce_message_text((message or {}).get("content")).strip()
            answer_map = _extract_answer_map(raw_text)
            answer = answer_map.get(question["question_id"], "")
            if not answer and raw_text:
                answer = raw_text
            return {
                "question_id": question["question_id"],
                "question_text": question["question_text"],
                "answer": answer or "Cannot be determined",
                "reasoning": _extract_reasoning(message or {}),
                "raw_text": raw_text,
                "error": "",
                "latency_ms": int(round((time.perf_counter() - question_start) * 1000)),
            }
        except Exception as exc:
            return {
                "question_id": question["question_id"],
                "question_text": question["question_text"],
                "answer": "",
                "reasoning": "",
                "raw_text": "",
                "error": str(exc),
                "latency_ms": int(round((time.perf_counter() - question_start) * 1000)),
            }

    ordered_answers: list[dict[str, Any]] = [None] * len(questions)
    max_workers = min(4, max(1, len(questions)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_one, question): index
            for index, question in enumerate(questions)
        }
        for future in as_completed(futures):
            ordered_answers[futures[future]] = future.result()

    answer_count = len(ordered_answers)
    failed_count = sum(1 for answer in ordered_answers if answer and answer.get("error"))

    reasoning_sections = []
    raw_sections = []
    for answer in ordered_answers:
        if not answer:
            continue
        question_label = f"[{answer['question_id']}] {answer['question_text']}"
        reasoning = str(answer.get("reasoning") or "").strip()
        raw_text = str(answer.get("raw_text") or "").strip()
        if reasoning:
            reasoning_sections.append(f"{question_label}\n{reasoning}")
        if raw_text:
            raw_sections.append(f"{question_label}\n{raw_text}")

    model_error = ""
    if answer_count > 0 and failed_count == answer_count:
        model_error = "All question requests failed."

    return {
        "model": model,
        "latency_ms": int(round((time.perf_counter() - start) * 1000)),
        "answers": ordered_answers,
        "reasoning": "\n\n".join(reasoning_sections),
        "raw_text": "\n\n".join(raw_sections),
        "error": model_error,
        "failed_questions": failed_count,
        "total_questions": answer_count,
    }


def run_groq_document_questions(
    *,
    document_text: str,
    questions: Any,
    models: Any,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    seed: int = 23,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Run a batch of freeform questions against selected Groq models."""
    if not groq_is_configured():
        raise RuntimeError("Groq is not configured on this deployment.")

    cleaned_document = _strip_inline_annotations(str(document_text or "")).strip()
    if not cleaned_document:
        raise ValueError("Document text is empty.")

    normalized_questions = normalize_questions(questions)
    if not normalized_questions:
        raise ValueError("Add at least one non-empty question.")

    normalized_models = _normalize_models(models)
    if not normalized_models:
        raise ValueError("Select at least one model.")

    prompt = str(system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    max_workers = min(4, max(1, len(normalized_models)))
    by_model: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_single_model,
                model=model,
                document_text=cleaned_document,
                questions=normalized_questions,
                system_prompt=prompt,
                temperature=float(temperature),
                seed=int(seed),
                max_tokens=int(max_tokens),
            ): model
            for model in normalized_models
        }
        for future in as_completed(futures):
            model = futures[future]
            by_model[model] = future.result()

    return {
        "configured": True,
        "document_char_count": len(cleaned_document),
        "questions": normalized_questions,
        "results": [by_model[model] for model in normalized_models if model in by_model],
        "system_prompt": prompt,
        "temperature": float(temperature),
        "seed": int(seed),
        "max_tokens": int(max_tokens),
    }
