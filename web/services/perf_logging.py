"""Small helpers for structured latency logs in Cloud Run."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Iterator


logger = logging.getLogger("annotation_perf")
_perf_context: ContextVar[dict[str, object]] = ContextVar("annotation_perf_context", default={})


@contextmanager
def perf_context(**fields: object) -> Iterator[None]:
    """Attach request/user fields to all nested PerfTimer logs."""
    current = dict(_perf_context.get() or {})
    current.update({key: value for key, value in fields.items() if value not in (None, "")})
    token = _perf_context.set(current)
    try:
        yield
    finally:
        _perf_context.reset(token)


class PerfTimer:
    """Collect named substep timings and emit one compact log line."""

    def __init__(self, event: str, **fields: object) -> None:
        self.event = event
        self.fields = dict(_perf_context.get() or {})
        self.fields.update(dict(fields))
        self.started = time.perf_counter()
        self.steps: list[tuple[str, float]] = []

    @contextmanager
    def step(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.steps.append((name, (time.perf_counter() - started) * 1000.0))

    def add_field(self, key: str, value: object) -> None:
        self.fields[key] = value

    def emit(self) -> None:
        total_ms = (time.perf_counter() - self.started) * 1000.0
        parts = [f"event={self.event}", f"total_ms={total_ms:.1f}"]
        for key, value in self.fields.items():
            parts.append(f"{key}={_format_value(value)}")
        for name, elapsed_ms in self.steps:
            parts.append(f"{name}_ms={elapsed_ms:.1f}")
        message = f"perf {' '.join(parts)}"
        logger.info(message)
        print(message, flush=True)


def _format_value(value: object) -> str:
    text = str(value)
    return text.replace(" ", "_")
