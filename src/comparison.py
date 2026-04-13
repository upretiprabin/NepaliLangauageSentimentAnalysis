"""
comparison.py — load the two models' result CSVs and produce a head-to-head
comparison table + summary CSV.

DONKEY EXPLANATION:
-------------------
After Phase 2 (traditional ML) and Phase 3 (NepBERTa), each model will have
dumped its metrics to a CSV in outputs/results/. This module is the "judge":
it reads both files, lines them up side-by-side, highlights which model wins
which metric, and writes `comparison_summary.csv`.

This is where the research question finally gets a numerical answer:
"Given Nepal's resource constraints, which model is the better deployment
choice?" — accuracy alone doesn't decide; efficiency weighs in.

What this module will expose (Phase 4):
  - load_results(path)           → dict of metrics
  - build_comparison_table(lr, nepberta) → pandas DataFrame, one row per metric
  - pick_winner(row)             → which model wins each metric
  - save_summary(df, path)       → dump to outputs/results/comparison_summary.csv
"""

# TODO (Phase 4).
pass
