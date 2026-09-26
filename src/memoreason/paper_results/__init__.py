"""Deterministic reproduction of MemoReason Tables 1-4 and Figures 3, 5, and 6."""

from __future__ import annotations

from typing import Any


__all__ = ["compute_paper_tables_and_figures", "publish_paper_tables_and_figures"]


def __getattr__(name: str) -> Any:
    """Load the public full-paper entrypoints only when they are requested.

    Importing a focused utility such as ``frozen_paper_results_input_loading``
    must not initialize the complete paper pipeline and its unrelated dataset
    dependencies.  The lazy attributes preserve the existing public API.
    """

    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .tables_1_to_4_and_figures_3_5_6 import (
        compute_paper_tables_and_figures,
        publish_paper_tables_and_figures,
    )

    globals().update(
        {
            "compute_paper_tables_and_figures": compute_paper_tables_and_figures,
            "publish_paper_tables_and_figures": publish_paper_tables_and_figures,
        }
    )
    return globals()[name]


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
