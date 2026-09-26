#!/usr/bin/env python3
"""Generate raw answers for the remote models reported in the paper."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from memoreason.model_evaluation.remote_model_answer_generation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
