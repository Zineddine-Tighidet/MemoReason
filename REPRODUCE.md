# Reproduce the paper

Generated entity pools are included. Model outputs and numerical results are not.
Use Python 3.11; run from the repository root. The figure/table numbers below
refer to the submission; titles identify them if numbering changes.

```sh
uv sync --extra web --extra llm
uv run python scripts/dataset.py --output data/generated
```

Generation uses seed 23, ten variants, and all ten settings. Existing output
directories are never overwritten. Technical setting IDs retain `fictional`.

## Run and analyze

`run` prints its commands without executing them. Add `--execute` to authorize
local GPU inference and provider calls. Configure `GROQ_API_KEY`,
`ANTHROPIC_API_KEY`, and access to the local model weights as needed.

```sh
uv run python scripts/paper.py run main
uv run python scripts/paper.py run main --execute
uv run python scripts/paper.py analyze main --output output/paper/main
```

For incremental runs, add `--models MODEL_ID` to `run`; analysis requires the
complete experiment. For `partial`, `effort`, `ablations`, and `temperature`, run
`run NAME --execute`, then `analyze NAME --output output/paper/NAME`. Run `main`
first: the other analyses reuse its factual/fully fictitious endpoints.
Dataset statistics need no model run:

```sh
uv run python scripts/paper.py analyze dataset --output output/paper/dataset
```

| Paper item | NAME / source | Produced locally |
|---|---|---|
| Tables 1, 6: dataset statistics/settings | `dataset` | `table1_dataset_statistics.csv`, `table6_settings.csv` |
| Table 2: performance changes | `main` | `table2_performance_changes.csv` |
| Table 3: reasoning effort | `effort` | `table3_reasoning_effort.csv`, paired contrasts |
| Tables 4, 5: Shortcut Rate/leaderboard | `main` | `table4_shortcut_rate.csv`, `table5_leaderboard.csv` |
| Table 7: named/value-only ablations | `ablations` | `table7_cue_ablations.csv` |
| Table 8: directional flips | `main` | `table8_directional_flips.csv` |
| Table 9: value-only Shortcut Rate | `ablations` | `table9_value_only_shortcut.csv` |
| Table 10: temperature robustness | `temperature` | `table10_temperature.csv` |
| Figures 2, 6: partial-replacement curves | `partial` | `partial_accuracy_qwen27` and `partial_accuracy_curves`, PDF/PNG |
| Figure 4: entity frequency versus gap | See below | `figure4_frequency`, PDF/PNG; counts/correlations CSV |
| Figure 1: benchmark illustration | `MemoReason_main_figure.png` | Hand-composed explanatory graphic, not a model result |
| Figure 3: taxonomy diagram | `TAXONOMY.md` | Manually arranged schematic; no numerical experiment |
| Figure 5: annotation interface | README interface command | Open the Toyota template (`company_02`) and capture the UI |
| Figure 7: annotation prompt | Appendix K.3 in the paper | Typeset prompt, not an experimental plot |

The CSV files provide table values; manuscript-specific LaTeX styling is not
required. Section 4.3's CoT comparison is also written by `main`. Qualitative
model responses must be regenerated; they are not included in this repository.

## Protocol

The run plans specify all nine main models, seven partial-sweep models, the
named/value ablation subsets, GPT-OSS-20B low/medium/high effort, and the two
temperature models at T=0/0.5/1 (seeds 23/24/25 for nonzero T). Model identifiers
and token caps are in `src/memoreason/model_evaluation/paper_model_registry.py`
and `reviewer_model_registry.py`. Main decoding uses T=0 and GPT-OSS effort=low.
Each question is evaluated independently with the checked-in prompt.

Scoring uses active reference answers, exact matching, then GPT-OSS-120B Judge
Match for eligible misses (Groq, T=0, seed 23, 256-token initial budget, low
effort). Judge effort stays low even in the reasoning-effort experiment.
`analyze` rejects missing/duplicate questions, stale golds, incomplete scoring,
and mismatched conditions. The README's `--skip-judge` example is only an
exact-match demonstration, **not** the paper-scoring protocol.

Table confidence intervals use 100,000 document-cluster bootstrap draws; partial-curve bands use
1.96 sqrt(p(1-p)/n). Existing kernels retain their seeds and estimators.
Shortcut Rate averages, over variant-answer base questions, the fraction of
failed variants whose canonical final answer matches a factual reference;
questions with no failures contribute zero. The leaderboard uses its reasoning
aggregate. Generation/provider changes can alter fresh-run scores; these commands
reproduce the procedure, not a guarantee of identical API outputs.

## Frequency inputs (Figure 4)

```sh
uv run python scripts/paper.py counts --output output/entity_counts.csv
uv run python scripts/paper.py counts --output output/entity_counts.csv --execute
uv run python scripts/paper.py analyze frequency \
  --counts output/entity_counts.csv --output output/paper/frequency
```

This requires access to Infini-gram index
`v4_olmo-3-7b-pretrain-shared_olmo2`. Counting is resumable and never treats
unavailable counts as zero. If that external index is unavailable, Figure 4
requires previously obtained counts for the same index; other experiments do
not depend on it. No corpus queries or inference run during tests.
