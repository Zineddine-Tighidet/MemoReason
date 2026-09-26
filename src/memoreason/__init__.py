"""MemoReason benchmark construction, model evaluation, and paper results.

The package follows the same order as the paper:

1. :mod:`memoreason.benchmark_definition` defines templates, entities, questions,
   answers, and replacement rules.
2. :mod:`memoreason.factual_to_fictional_dataset` builds controlled factual and
   fictional document pairs.
3. :mod:`memoreason.model_evaluation` generates, parses, and scores model answers.
4. :mod:`memoreason.paper_results` builds Tables 1-4 and Figures 3, 5, and 6.
"""

from pathlib import Path


MEMOREASON_PACKAGE_DIRECTORY = Path(__file__).resolve().parent
"""Directory containing the importable :mod:`memoreason` package."""

SOURCE_DIRECTORY = MEMOREASON_PACKAGE_DIRECTORY.parent
"""Repository ``src`` directory."""

PROJECT_ROOT_DIRECTORY = SOURCE_DIRECTORY.parent
"""Root of the MemoReason source checkout."""
