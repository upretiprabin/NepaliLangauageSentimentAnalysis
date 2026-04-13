"""
visualizations.py — every plot the project produces, in one place.

DONKEY EXPLANATION:
-------------------
A research paper without graphs is hard to read. Rather than scattering
plt.plot calls across notebooks, we put ALL plotting functions here so that:
  - every figure has the same style (DPI, fonts, colours),
  - notebooks stay short and readable,
  - we can regenerate all figures with one loop if we tweak styling later.

Figures we produce:
  1.  class_distribution_primary.png      — bar chart of label counts (primary)
  2.  class_distribution_secondary.png    — same, secondary dataset
  3.  text_length_distribution.png        — histogram of tweet lengths
  4.  confusion_matrix_lr.png             — LR predictions vs truth
  5.  confusion_matrix_nepberta.png       — NepBERTa predictions vs truth
  6.  accuracy_comparison.png             — grouped bar, all metrics, both models
  7.  efficiency_comparison.png           — bar: inference time + memory
  8.  training_loss_nepberta.png          — train/val loss curves during fine-tune
  9.  sample_predictions.png              — a handful of test examples,
                                            both models' predictions side-by-side

Style rules (from CLAUDE.md):
  - PNG, 300 DPI, tight_layout
  - Add suitable titles and axis labels
  - Save to outputs/figures/
"""

# TODO (Phase 1): stub out all nine functions. Each takes the data it needs
# and a `save_path`, and uses matplotlib/seaborn. Keep styling consistent via
# a small helper that sets rcParams at import time.

pass
