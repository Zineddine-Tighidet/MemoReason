#!/usr/bin/env python3
"""Generate MemoReason documents from the supplied templates and entity pools."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SETTINGS = (
    "factual", "fictional", "fictional_named", "fictional_numtemp",
    "fictional_10pct", "fictional_20pct", "fictional_30pct",
    "fictional_50pct", "fictional_80pct", "fictional_90pct",
)
ENVIRONMENT = {
    "FICTIONAL_BATCH_GENERATION_ATTEMPTS": "1",
    "FICTIONAL_MAX_VARIANT_GENERATION_RETRIES": "5",
    "MEMOREASON_EXCLUSIVE_ARTIFACT_WRITES": "1",
    "MEMOREASON_STRICT_DATASET_EXPORT": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def generate_one(task: tuple[str, int, tuple[str, ...], int]) -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from memoreason.factual_to_fictional_dataset.paired_factual_and_fictional_dataset_export import (
        export_paired_factual_and_fictional_documents,
    )

    template, seed, settings, variants = task
    return len(export_paired_factual_and_fictional_documents(
        [Path(template)], seed=seed, settings=list(settings),
        fictional_version_count=variants, overwrite=False, skip_missing_pools=False,
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for generated documents")
    parser.add_argument("--docs", nargs="+", help="Optional source document IDs; default: all 100")
    parser.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--variants", type=int, default=10)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    if output.exists() or args.output.is_symlink():
        parser.error("The output directory must not already exist")
    if any(output.is_relative_to(DATA / name) for name in (
        "HUMAN_ANNOTATED_TEMPLATES", "GENERATED_FICTIONAL_ENTITIES",
    )):
        parser.error("The output directory must not be inside the source templates or entity pools")
    if args.workers < 1 or args.variants < 1:
        parser.error("--workers and --variants must be positive")

    templates = sorted((DATA / "HUMAN_ANNOTATED_TEMPLATES").glob("*/*.yaml"))
    if args.docs:
        requested = set(args.docs)
        templates = [p for p in templates if p.stem in requested]
        missing = requested - {p.stem for p in templates}
        if missing:
            parser.error("Unknown document IDs: " + ", ".join(sorted(missing)))
    if not templates:
        parser.error("No source templates found")

    # Hash seed must be set before Python starts; generation uses no provider calls.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                  dict(os.environ, **ENVIRONMENT))
    os.environ.update(ENVIRONMENT)
    os.environ.update({
        "MEMOREASON_HUMAN_ANNOTATED_TEMPLATES_DIR": str(DATA / "HUMAN_ANNOTATED_TEMPLATES"),
        "MEMOREASON_GENERATED_FICTIONAL_ENTITIES_DIR": str(DATA / "GENERATED_FICTIONAL_ENTITIES"),
        "MEMOREASON_FACTUAL_DOCUMENTS_DIR": str(output / "FACTUAL_DOCUMENTS"),
        "MEMOREASON_FICTIONAL_DOCUMENTS_DIR": str(output / "FICTIONAL_DOCUMENTS"),
        "MEMOREASON_MODEL_EVAL_DIR": str(output / "MODEL_EVAL"),
    })
    tasks = [(str(p), args.seed, tuple(dict.fromkeys(args.settings)), args.variants) for p in templates]
    output.mkdir(parents=True, exist_ok=False)
    if args.workers == 1:
        generated = sum(map(generate_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            generated = sum(pool.map(generate_one, tasks))
    expected = len(templates) * sum(1 if s == "factual" else args.variants for s in set(args.settings))
    if generated != expected:
        raise RuntimeError(f"Incomplete generation: expected {expected}, got {generated}")
    print(f"Generated {generated} documents in {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
