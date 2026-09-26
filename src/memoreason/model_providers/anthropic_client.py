"""Explicit Anthropic HTTP client for shared text generation."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import requests


_DEFAULT_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_REQUEST_PACING_LOCK = threading.Lock()
_NEXT_REQUEST_NOT_BEFORE = 0.0


class AnthropicTextGenerationClient:
    """Small Anthropic wrapper used by the shared text-generation layer."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: float = 300.0,
        messages_url: str = _DEFAULT_MESSAGES_URL,
    ) -> None:
        self._api_key = str(api_key).strip()
        self._timeout_seconds = float(timeout_seconds)
        self._messages_url = str(messages_url)
        self._max_retries = self._read_max_retries()
        self._min_interval_seconds = self._read_min_interval_seconds()
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def close(self) -> None:
        """Release the HTTP connection pool owned by this client."""
        self._session.close()

    @staticmethod
    def _read_max_retries() -> int:
        raw_value = str(os.environ.get("ANTHROPIC_MAX_RETRIES", "12")).strip()
        try:
            return max(0, int(raw_value))
        except ValueError:
            return 12

    @staticmethod
    def _read_min_interval_seconds() -> float:
        raw_value = str(os.environ.get("ANTHROPIC_MIN_INTERVAL_SECONDS", "0")).strip()
        try:
            return max(0.0, float(raw_value))
        except ValueError:
            return 0.0

    @staticmethod
    def _retry_delay_seconds(*, attempt_index: int, response: requests.Response | None) -> float:
        if response is not None:
            retry_after = str(response.headers.get("retry-after") or "").strip()
            try:
                if retry_after:
                    return max(0.05, float(retry_after) + 0.05)
            except ValueError:
                pass
        return min(30.0, 1.5 * (2**attempt_index))

    def _wait_for_request_slot(self) -> None:
        if self._min_interval_seconds <= 0:
            return
        global _NEXT_REQUEST_NOT_BEFORE
        while True:
            with _REQUEST_PACING_LOCK:
                now = time.monotonic()
                if now >= _NEXT_REQUEST_NOT_BEFORE:
                    _NEXT_REQUEST_NOT_BEFORE = now + self._min_interval_seconds
                    return
                sleep_for = _NEXT_REQUEST_NOT_BEFORE - now
            time.sleep(sleep_for)

    def generate_raw_response(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model,
            "system": system_prompt,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            "temperature": temperature,
        }
        # Opus 4.7 rejects the legacy temperature field when adaptive thinking is active.
        if str(model).startswith("claude-opus-4-7"):
            payload.pop("temperature")
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "user-agent": "parametric-shortcut-quick-eval/0.1",
        }
        response: requests.Response | None = None
        for attempt_index in range(self._max_retries + 1):
            try:
                self._wait_for_request_slot()
                response = self._session.post(
                    self._messages_url,
                    headers=headers,
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                if response.status_code >= 400:
                    if response.status_code in _RETRYABLE_HTTP_CODES and attempt_index < self._max_retries:
                        time.sleep(
                            self._retry_delay_seconds(
                                attempt_index=attempt_index,
                                response=response,
                            )
                        )
                        continue
                    raise RuntimeError(
                        f"Anthropic request failed with HTTP {response.status_code}: {response.text[:2000]}"
                    )
                parsed = response.json()
                if not isinstance(parsed, dict):
                    raise RuntimeError(f"Anthropic returned an unexpected response type: {type(parsed).__name__}")
                return parsed
            except requests.RequestException as exc:
                if attempt_index < self._max_retries:
                    time.sleep(
                        self._retry_delay_seconds(
                            attempt_index=attempt_index,
                            response=response,
                        )
                    )
                    continue
                raise RuntimeError(f"Anthropic request failed: {exc}") from exc
        raise RuntimeError("Anthropic request retry loop exhausted unexpectedly")

    @staticmethod
    def extract_text(response: Any) -> str:
        chunks: list[str] = []
        if isinstance(response, dict):
            content = response.get("content") or []
        else:
            content = getattr(response, "content", [])
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                chunks.append(str(text))
        return "".join(chunks).strip()
