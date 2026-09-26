"""Compute factual-correct to fully-fictional-incorrect flip rates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def _read_payload(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def compute_flip_rates(
    evaluated_root: Path,
    *,
    model_names: set[str] | None = None,
    expected_variant_ids: tuple[str, ...] = tuple(f"v{index:02d}" for index in range(1, 11)),
) -> list[dict[str, Any]]:
    """Return strict per-model flip rates across all fully-fictional variants."""
    factual: dict[tuple[str, str], bool] = {}
    fictional: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    selected_files = 0

    for path in sorted(evaluated_root.rglob("*_evaluated_outputs.yaml")):
        payload = _read_payload(path)
        model_name = str(payload.get("model_name") or "").strip()
        if model_names and model_name not in model_names:
            continue
        setting = str(payload.get("document_setting") or "").strip().lower()
        if setting not in {"factual", "fictional"}:
            continue
        selected_files += 1
        variant_id = str(payload.get("document_variant_id") or "").strip()
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Missing results list: {path}")
        for result in results:
            if not isinstance(result, dict):
                raise ValueError(f"Non-mapping result in {path}")
            pair_key = str(result.get("pair_key") or "").strip()
            is_correct = result.get("final_is_correct")
            if not pair_key or not isinstance(is_correct, bool):
                raise ValueError(f"Invalid pair_key/final_is_correct in {path}")
            key = (model_name, pair_key)
            if setting == "factual":
                if key in factual:
                    raise ValueError(f"Duplicate factual result for {key}")
                factual[key] = is_correct
            else:
                if variant_id not in expected_variant_ids:
                    continue
                if variant_id in fictional[key]:
                    raise ValueError(f"Duplicate fictional result for {key}, {variant_id}")
                fictional[key][variant_id] = is_correct

    if selected_files == 0:
        raise ValueError(f"No evaluated factual/fictional YAML files found below {evaluated_root}")

    rows: list[dict[str, Any]] = []
    selected_models = sorted({model for model, _ in factual})
    for model_name in selected_models:
        model_factual = {pair_key: correct for (model, pair_key), correct in factual.items() if model == model_name}
        factual_correct_keys = [pair_key for pair_key, correct in model_factual.items() if correct]
        missing = {
            pair_key: sorted(set(expected_variant_ids) - set(fictional[(model_name, pair_key)]))
            for pair_key in model_factual
            if set(fictional[(model_name, pair_key)]) != set(expected_variant_ids)
        }
        if missing:
            sample = next(iter(missing.items()))
            raise ValueError(
                f"Incomplete fully-fictional variants for {model_name}: "
                f"{len(missing)} pair keys; example {sample[0]} missing {sample[1]}"
            )
        comparison_count = len(factual_correct_keys) * len(expected_variant_ids)
        flip_count = sum(
            not fictional[(model_name, pair_key)][variant_id]
            for pair_key in factual_correct_keys
            for variant_id in expected_variant_ids
        )
        rows.append(
            {
                "model": model_name,
                "factual_examples": len(model_factual),
                "factual_correct": len(factual_correct_keys),
                "fully_fictional_variants_per_example": len(expected_variant_ids),
                "eligible_comparisons": comparison_count,
                "correct_to_incorrect_flips": flip_count,
                "flip_rate": flip_count / comparison_count if comparison_count else 0.0,
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "flip_rate.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "flip_rate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = "\n".join(
        f"{row['model']} & {row['factual_correct']} / {row['factual_examples']} & "
        f"{row['correct_to_incorrect_flips']} / {row['eligible_comparisons']} & "
        f"{100 * row['flip_rate']:.2f}\\% \\\\" for row in rows
    )
    tex = (
        "\\begin{tabular}{lrrr}\n\\toprule\n"
        "Model & Factual correct & Correct-to-incorrect flips & Flip rate \\\\n"
        "\\midrule\n"
        f"{table_rows}\n"
        "\\bottomrule\n\\end{tabular}\n"
    )
    (output_dir / "flip_rate.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluated-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()
    rows = compute_flip_rates(args.evaluated_root, model_names=set(args.models) if args.models else None)
    write_outputs(rows, args.output_dir)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
