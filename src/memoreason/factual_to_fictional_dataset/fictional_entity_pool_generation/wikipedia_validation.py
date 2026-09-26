"""Wikipedia existence checks for entity strings."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
import time
import random
import urllib.parse
import urllib.request
import urllib.error

_WIKIPEDIA_RETRY_SLEEP_CAP_SECONDS = 15.0


@dataclass
class WikipediaHit:
    exists: bool
    title: str | None = None
    page_id: int | None = None
    snippet: str | None = None
    is_disambiguation: bool = False
    normalized: str | None = None


def _build_search_url(query: str) -> str:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
        "srprop": "snippet",
        "format": "json",
    }
    return "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)


def _normalize(value: str) -> str:
    return " ".join(value.split()).strip()


def normalize_query(value: str) -> str:
    return _normalize(value)


def query_wikipedia(query: str, timeout: float = 8.0, max_retries: int = 5) -> WikipediaHit:
    normalized = _normalize(query)
    if not normalized:
        return WikipediaHit(exists=False, normalized=normalized)

    url = _build_search_url(normalized)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ParametricShortcutQuickEval/1.0 (entity-validation)"},
    )
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code == 429 or 500 <= exc.code < 600:
                retry_after = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
                try:
                    retry_after_seconds = float(retry_after) if retry_after is not None else None
                except (TypeError, ValueError):
                    retry_after_seconds = None
                base_sleep = retry_after_seconds if retry_after_seconds is not None else (5 * (2**attempt))
                sleep_for = min(_WIKIPEDIA_RETRY_SLEEP_CAP_SECONDS, base_sleep) + random.uniform(0.25, 1.0)
                time.sleep(sleep_for)
                continue
            raise
        except urllib.error.URLError as exc:
            last_err = exc
            sleep_for = min(_WIKIPEDIA_RETRY_SLEEP_CAP_SECONDS, (2**attempt)) + random.uniform(0.25, 1.0)
            time.sleep(sleep_for)
            continue
    else:
        if last_err:
            raise last_err
        raise RuntimeError("Wikipedia query failed without exception")

    search = payload.get("query", {}).get("search", [])
    if not search:
        return WikipediaHit(exists=False, normalized=normalized)

    for result in search:
        title = result.get("title")
        if not title or title.lower() != normalized.lower():
            continue
        snippet = result.get("snippet")
        page_id = result.get("pageid")
        is_disambiguation = "disambiguation" in (snippet or "").lower() or "(disambiguation)" in title.lower()
        return WikipediaHit(
            exists=True,
            title=title,
            page_id=page_id,
            snippet=snippet,
            is_disambiguation=is_disambiguation,
            normalized=normalized,
        )

    return WikipediaHit(exists=False, normalized=normalized)


def load_cache(path: Path) -> dict[str, WikipediaHit]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    cache: dict[str, WikipediaHit] = {}
    for key, value in raw.items():
        cache[key] = WikipediaHit(**value)
    return cache


def save_cache(path: Path, cache: dict[str, WikipediaHit]) -> None:
    raw = {k: vars(v) for k, v in cache.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(raw, indent=2, sort_keys=True) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def check_entities(
    entities: Iterable[str],
    cache_path: Path,
    timeout: float = 8.0,
    cache: dict[str, WikipediaHit] | None = None,
    throttle: float = 0.0,
    max_retries: int = 5,
    save: bool = True,
    progress: dict | None = None,
    replay_only: bool = False,
) -> dict[str, WikipediaHit]:
    cache = cache if cache is not None else load_cache(cache_path)
    normalized_entities = [normalized for entity in entities if (normalized := _normalize(entity))]
    if replay_only:
        missing = sorted({entity for entity in normalized_entities if entity not in cache})
        if missing:
            rendered = ", ".join(missing[:12])
            if len(missing) > 12:
                rendered += f", ... (+{len(missing) - 12} more)"
            raise FileNotFoundError(
                f"Wikipedia replay cache misses in {cache_path}: {rendered}"
            )
        return cache

    for normalized in normalized_entities:
        if normalized in cache:
            continue
        cache[normalized] = query_wikipedia(normalized, timeout=timeout, max_retries=max_retries)
        if progress is not None:
            progress["count"] = progress.get("count", 0) + 1
            log_every = max(1, int(progress.get("log_every", 50)))
            if progress["count"] == 1 or progress["count"] % log_every == 0:
                elapsed = max(0.001, time.time() - float(progress.get("start", time.time())))
                rate = progress["count"] / elapsed
                total = progress.get("total", 0)
                if total:
                    msg = f"[wiki] {progress['count']}/{total} ({rate:.1f}/s): {normalized}"
                else:
                    msg = f"[wiki] {progress['count']} ({rate:.1f}/s): {normalized}"
                print(msg, flush=True)
        if save:
            save_cache(cache_path, cache)
        if throttle > 0:
            time.sleep(throttle)
    return cache
