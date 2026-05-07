#!/usr/bin/env python3
"""Export benchmark factual and fictional documents from annotated templates."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)


def main() -> int:
    from src.dataset_export.export_workflow import export_dataset_documents
    from src.dataset_export.dataset_settings import resolve_dataset_settings
    from src.dataset_export.dataset_paths import ensure_dataset_artifact_directories, iter_template_paths

    parser = argparse.ArgumentParser(description="Export benchmark factual and fictional documents")
    parser.add_argument("--themes", nargs="+", default=None, help="Theme folders to process")
    parser.add_argument("--docs", nargs="+", default=None, help="Document ids to process")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=None,
        help="Explicit benchmark setting ids, for example: factual fictional fictional_20pct",
    )
    parser.add_argument(
        "--fictional-proportions",
        nargs="+",
        type=float,
        default=None,
        help="Replacement proportions for fictional settings, for example: 0.2 0.5 0.8 1.0",
    )
    parser.add_argument("--skip-factual", action="store_true", help="Do not export the factual document setting")
    parser.add_argument(
        "--fictional-version-count",
        type=int,
        default=1,
        help="Number of fictional document versions to export for each fictional setting",
    )
    parser.add_argument("--seed", type=int, default=23, help="Seed used for fictional document generation")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even when they already exist")
    parser.add_argument(
        "--skip-missing-pools",
        action="store_true",
        help="Skip fictional exports when no entity pool is available",
    )
    args = parser.parse_args()

    ensure_dataset_artifact_directories()
    template_paths = list(iter_template_paths(themes=args.themes, document_ids=args.docs))
    if not template_paths:
        raise FileNotFoundError("No annotated templates matched the requested filters.")

    setting_specs = resolve_dataset_settings(
        explicit_settings=args.settings,
        include_factual=not args.skip_factual,
        fictional_proportions=args.fictional_proportions,
    )
    written_paths = export_dataset_documents(
        template_paths,
        seed=args.seed,
        settings=[spec.setting_id for spec in setting_specs],
        fictional_version_count=args.fictional_version_count,
        overwrite=args.overwrite,
        skip_missing_pools=args.skip_missing_pools,
    )
    for output_path in written_paths:
        print(output_path.relative_to(_ROOT))
    print(f"Generated {len(written_paths)} document files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
