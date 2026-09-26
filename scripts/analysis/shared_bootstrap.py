"""Frozen scientific routines used by the submitted result release."""
from __future__ import annotations
import hashlib
import numpy as np

def bootstrap_shared_documents(
    *,
    document_metrics: np.ndarray,
    samples: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    if samples <= 0 or chunk_size <= 0:
        raise ValueError("Bootstrap samples and chunk size must be positive")
    points = document_metrics.mean(axis=1)
    effort_count, document_count, metric_count = document_metrics.shape
    draws = np.empty((samples, effort_count, metric_count), dtype=float)
    draw_digest = hashlib.sha256()
    rng = np.random.default_rng(seed)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = rng.integers(
            0, document_count, size=(stop - start, document_count), endpoint=False
        )
        draw_digest.update(np.asarray(indices, dtype="<i8").tobytes(order="C"))
        draws[start:stop] = document_metrics[:, indices, :].mean(axis=2).transpose(1, 0, 2)
    return points, draws, draw_digest.hexdigest()
