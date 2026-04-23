"""
visualizations.py — every plot the project produces, in one place.

DONKEY EXPLANATION:
-------------------
A research paper without graphs is hard to read. Rather than scattering
`plt.plot` calls across notebooks, we put ALL plotting functions here so:
  - every figure has the same style (DPI, fonts, colours)
  - notebooks stay short and readable (no 40-line plotting blocks)
  - we can regenerate all figures with one loop if we tweak styling later
  - Phase 2 and Phase 3 produce visually consistent figures even though
    they live in different notebooks run in different environments.

Figures we produce (in approximate build order):
  1.  class_distribution.png         — bar chart of label counts (EDA)
  2.  text_length_distribution.png   — histogram of doc lengths  (EDA)
  3.  data_quality.png               — duplicates + conflicts audit (EDA)
  4.  confusion_matrix_<model>.png   — heatmap, true vs predicted (per model)
  5.  training_loss_nepberta.png     — train/val loss curve (Phase 3)
  6.  accuracy_comparison.png        — grouped bars, all metrics, both models
  7.  efficiency_comparison.png      — bars: inference time + memory
  8.  sample_predictions.png         — table of texts + both models' calls

Style rules (from CLAUDE.md):
  - PNG, 300 DPI, tight_layout
  - NO titles — captions live in the paper, not in the figures themselves
  - Clean axis labels + value annotations on bars
  - Saved to outputs/figures/ — never hard-code paths; use `config.FIGURES`

Every public function:
  - takes the data + an optional `save_path`
  - returns the matplotlib `Figure` (so the caller can close / tweak / show)
  - saves at 300 DPI when save_path is given
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────
# `os` for path joining when saving figures.
import os
import sys
from pathlib import Path

# `typing` helpers for cleaner signatures (Optional = "may be None").
from typing import Optional, Sequence

# pandas for the results-DataFrame-based comparison plots.
import pandas as pd
import numpy as np

# matplotlib = base plotting. `plt` = the pyplot API (figure/axes builders).
import matplotlib.pyplot as plt

# seaborn builds on matplotlib with nicer defaults for heatmaps and colour
# palettes. We only use `heatmap` for the confusion matrix; everything else
# is plain matplotlib to keep the dependency footprint small.
import seaborn as sns


# Make `from src import config` importable regardless of how this module
# is called. Same idiom as preprocessing.py / evaluation.py.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config


# ──────────────────────────────────────────────────────────────────────────
# CONSISTENT COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────
# One palette applied everywhere. Reusing the same colour for the same
# concept across figures trains the reader's eye — Positive is always green,
# NepBERTa is always orange, etc.

# Per-sentiment-class: red / grey / green = intuitive for sentiment.
LABEL_COLORS = {
    'Negative': '#d9534f',   # red
    'Neutral':  '#999999',   # grey
    'Positive': '#5cb85c',   # green
}

# Per-model: blue for LR (traditional), orange for NepBERTa (transformer).
# Colour-blind friendly pair.
MODEL_COLORS = {
    'logistic_regression': '#4a90d9',
    'nepberta':            '#f0973a',
}


# ──────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────

def _save_and_return(fig: plt.Figure, save_path: Optional[str]) -> plt.Figure:
    """Apply tight_layout, save (if requested), return the Figure.

    DONKEY: every plot function ends with the same boilerplate — trim
    padding, maybe save, hand back the figure. We centralise it here so
    future style changes (background colour, font family, etc.) only
    need one edit.

    ⚠️ Jupyter "double-display" fix:
      `plt.subplots()` registers a figure with pyplot. In a Jupyter cell
      with %matplotlib inline, the inline backend then auto-renders the
      figure at cell end. If the function ALSO returns the figure and
      the call is the last expression in the cell, Jupyter's rich-repr
      renders it a SECOND time → two identical plots appear.

      `plt.close(fig)` deregisters the figure from pyplot but does NOT
      destroy the underlying Figure object, so:
        - the inline backend skips it (no auto-display)
        - the returned Figure is still rendered once by Jupyter's
          rich-repr when it's the cell's last expression
        - assigning the return value (`_ = viz.plot_...()`) suppresses
          that last display, letting the caller opt out entirely
    """
    fig.tight_layout()
    if save_path is not None:
        # Caller may pass a filename or a full path. If it's a bare
        # filename (no directory), drop it into config.FIGURES.
        if os.path.dirname(save_path) == '':
            os.makedirs(config.FIGURES, exist_ok=True)
            save_path = os.path.join(config.FIGURES, save_path)
        fig.savefig(save_path, dpi=config.FIGURE_DPI, bbox_inches='tight')

    # Deregister from pyplot so the inline backend doesn't auto-show it.
    # See the double-display explanation above.
    plt.close(fig)
    return fig


# ──────────────────────────────────────────────────────────────────────────
# 1) CLASS DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    labels: Sequence[int],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of label counts.

    DONKEY: how many Negative / Neutral / Positive rows do we have?
    Imbalance = some bars much taller than others = we need to handle it
    during training (class_weight='balanced', weighted loss).

    Args:
        labels: iterable of ints in {0, 1, 2} (after normalisation).
        save_path: filename (in config.FIGURES) OR full path. Default: don't save.

    Returns:
        matplotlib Figure.
    """
    # Convert to a Series so we can use .value_counts — handles lists,
    # numpy arrays, pandas Series uniformly.
    counts = pd.Series(labels).value_counts().sort_index()

    # Build bar heights + matching colours in the SAME ORDER as LABEL_NAMES.
    # Using a dict keyed on LABEL_NAMES[i] keeps the colour mapping stable
    # even if a class is missing (would just be height 0).
    bar_heights = [int(counts.get(i, 0)) for i in range(config.NUM_CLASSES)]
    bar_colors  = [LABEL_COLORS[name] for name in config.LABEL_NAMES]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(config.LABEL_NAMES, bar_heights, color=bar_colors,
                  edgecolor='white')

    # Annotate each bar with its count — saves the reader from squinting
    # at the y-axis.
    for bar, h in zip(bars, bar_heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:,}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of rows')
    # Room above the tallest bar for the annotation text.
    ax.set_ylim(0, max(bar_heights) * 1.12)

    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 2) TEXT LENGTH DISTRIBUTION
# ──────────────────────────────────────────────────────────────────────────

def plot_text_length_distribution(
    lengths: Sequence[int],
    save_path: Optional[str] = None,
    x_label: str = 'Tokens per document',
    bins: int = 50,
    vline: Optional[int] = None,
    vline_label: str = 'max_length',
) -> plt.Figure:
    """Histogram of document lengths, optionally with a max_length reference line.

    DONKEY: shows the shape of "how long are the inputs?". A long right
    tail means some docs are outliers and will get truncated; a tight
    distribution means we can pick a small max_length confidently.

    Args:
        lengths: iterable of ints (each = length of one document).
        save_path: filename or full path. Default: don't save.
        x_label: custom x-axis label — useful because the caller knows
                 whether they measured whitespace tokens or BPE tokens.
        bins: number of histogram buckets.
        vline: optional x-coordinate for a vertical reference line
               (e.g., config.NEPBERTA_MAX_LENGTH).
        vline_label: legend label for the vertical line.

    Returns:
        matplotlib Figure.
    """
    lengths = pd.Series(lengths)

    # Clip the top 1% for DISPLAY (not for storage) so one 5000-token outlier
    # doesn't squish 99% of bars into the left edge.
    cutoff = int(lengths.quantile(0.99))
    display_data = lengths.clip(upper=cutoff)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(display_data, bins=bins, color='#4a90d9', edgecolor='white')

    # Optional reference line (the value we care about, e.g., NepBERTa cap).
    if vline is not None:
        ax.axvline(vline, color='red', linestyle='--', linewidth=1,
                   label=f'{vline_label} = {vline}')
        ax.legend()

    ax.set_xlabel(x_label + f' (clipped for display at 99th pct = {cutoff})')
    ax.set_ylabel('Count')
    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 3) CONFUSION MATRIX
# ──────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: Optional[str] = None,
    normalize: bool = False,
) -> plt.Figure:
    """Heatmap of a confusion matrix.

    DONKEY: a confusion matrix is a 3x3 grid where row = TRUE label, col
    = PREDICTED label. The diagonal is "got it right"; off-diagonal is
    "got it wrong, and in which direction". Bright diagonal = good model.

    Why offer normalize=True?
    -------------------------
    Raw counts show absolute mistakes but are biased by class size. Row-
    normalised values show PER-CLASS recall — "of the true Positives,
    what fraction were predicted as each class?". Paper usually shows both.

    Args:
        cm: 2D numpy array of shape (n_classes, n_classes).
        save_path: filename or full path. Default: don't save.
        normalize: if True, convert each row to proportions summing to 1.0.

    Returns:
        matplotlib Figure.
    """
    # Optionally normalise by row (true-class support). `keepdims=True`
    # keeps cm_sum shape (3, 1) so broadcasting divides every row by its
    # own sum. If a row sums to 0, replace with 1 to avoid div-by-zero
    # (that class would just be all zeros anyway).
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm / row_sums
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'  # integer

    fig, ax = plt.subplots(figsize=(5, 4))
    # seaborn's heatmap = matplotlib + nicer annotation defaults. `annot`
    # writes the number into each cell; `fmt` formats it; `cbar` = side bar
    # showing the colour scale.
    sns.heatmap(
        cm_display,
        annot=True, fmt=fmt,
        cmap='Blues',
        xticklabels=config.LABEL_NAMES,
        yticklabels=config.LABEL_NAMES,
        cbar=True,
        ax=ax,
        square=True,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 4) TRAINING LOSS CURVE (for NepBERTa fine-tuning)
# ──────────────────────────────────────────────────────────────────────────

def plot_training_loss(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line plot of training (and optionally validation) loss vs epoch.

    DONKEY: watching the loss go down across epochs tells you if learning
    is happening. Training loss keeps going down — that's expected.
    Validation loss going UP while training loss goes DOWN = overfitting,
    stop training (early-stopping). The gap between the two lines is the
    generalisation story.

    Args:
        train_losses: list of per-epoch training losses.
        val_losses: optional list of validation losses (same length).
        save_path: filename or full path. Default: don't save.

    Returns:
        matplotlib Figure.
    """
    # X-axis is 1-indexed "epoch number" because people count epochs from 1.
    epochs = list(range(1, len(train_losses) + 1))

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(epochs, train_losses, marker='o', color='#4a90d9',
            label='Training loss')
    if val_losses is not None:
        ax.plot(epochs, val_losses, marker='s', color='#d9534f',
                label='Validation loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_xticks(epochs)  # integer ticks, not 0.5-step defaults
    ax.legend()
    ax.grid(True, alpha=0.25)  # faint gridlines help reading curves
    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 5) ACCURACY COMPARISON (both models, multiple metrics)
# ──────────────────────────────────────────────────────────────────────────

def plot_accuracy_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    metrics: Sequence[str] = ('accuracy', 'macro_precision', 'macro_recall',
                              'macro_f1', 'weighted_f1'),
) -> plt.Figure:
    """Grouped bars: one group per metric, bars inside = each model.

    DONKEY: answers "which model wins, AND on which scoreboard?" in one
    glance. If one model is strictly better it towers across all groups;
    if it's nuanced (higher accuracy but lower macro-F1, e.g.), the
    figure shows that too.

    Args:
        results_df: DataFrame with a 'model' column and one column per
                    metric (the flat dict from evaluation.flatten_metrics).
                    One row per model.
        save_path: filename or full path.
        metrics: which metric columns to plot, in order left-to-right.

    Returns:
        matplotlib Figure.
    """
    models = list(results_df['model'])
    n_models = len(models)
    n_metrics = len(metrics)

    # X positions for each metric group (0, 1, 2, ...).
    x_base = np.arange(n_metrics)
    # Width of each bar so that n_models bars fit snugly within one group.
    bar_width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * n_metrics * n_models), 4))

    for i, model_name in enumerate(models):
        row = results_df[results_df['model'] == model_name].iloc[0]
        heights = [float(row[m]) for m in metrics]
        # Offset each model's bars so they sit side-by-side in their group.
        # The `- (n_models - 1) / 2` centering keeps the ensemble centred
        # on each group's x-tick.
        offsets = x_base + (i - (n_models - 1) / 2) * bar_width
        color = MODEL_COLORS.get(model_name, None)
        bars = ax.bar(offsets, heights, width=bar_width, label=model_name,
                      color=color, edgecolor='white')
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xticks(x_base)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)  # accuracy/F1/etc. always in [0, 1]
    ax.legend()
    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 6) EFFICIENCY COMPARISON
# ──────────────────────────────────────────────────────────────────────────

def plot_efficiency_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[str] = None,
    metrics: Sequence[str] = ('inference_ms_per_sample', 'peak_memory_mb',
                              'model_size_mb', 'training_time_s'),
) -> plt.Figure:
    """Bars comparing efficiency metrics across models, one subplot per metric.

    DONKEY: accuracy isn't everything — NepBERTa will be ~100x slower and
    ~500x bigger than LR. This figure makes the trade-off visible and
    feeds directly into the paper's "for deployment in Nepal..." argument.

    Each subplot uses a LOG y-scale because the gap between LR and
    NepBERTa is typically 2-3 orders of magnitude on these metrics —
    linear scale would show "big bar vs invisible bar."

    Args:
        results_df: DataFrame with 'model' + efficiency columns.
        save_path: filename or full path.
        metrics: which efficiency columns to plot (one subplot each).

    Returns:
        matplotlib Figure (with len(metrics) subplots in a row).
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 3.5))
    # When n_metrics == 1, subplots() returns a single Axes, not an array.
    if n_metrics == 1:
        axes = [axes]

    models = list(results_df['model'])
    colors = [MODEL_COLORS.get(m, '#777777') for m in models]

    for ax, metric in zip(axes, metrics):
        heights = [float(results_df[results_df['model'] == m].iloc[0][metric])
                   for m in models]
        bars = ax.bar(models, heights, color=colors, edgecolor='white')
        for bar, h in zip(bars, heights):
            # Format: show full precision for tiny numbers, compact for big.
            label = f'{h:.2f}' if h < 10 else f'{h:,.0f}'
            ax.text(bar.get_x() + bar.get_width() / 2, h, label,
                    ha='center', va='bottom', fontsize=8)
        ax.set_ylabel(metric)
        ax.set_yscale('log')       # wide dynamic range → log scale
        # Tilt model labels so long names don't overlap.
        ax.tick_params(axis='x', rotation=15)

    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 7) SAMPLE PREDICTIONS TABLE
# ──────────────────────────────────────────────────────────────────────────

def plot_sample_predictions(
    samples_df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Table-style figure of texts + true label + each model's prediction.

    DONKEY: a qualitative companion to the accuracy numbers. Shows 8-10
    example rows so the reader can see "here's where they agree" and
    "here's where NepBERTa corrects LR" (or vice versa). More persuasive
    than bare numbers for a non-technical reader.

    Expected columns in samples_df:
        text, true_label, pred_lr, pred_nepberta
    (all as strings after mapping label ints → label names)

    Args:
        samples_df: small DataFrame (typically 8-12 rows) with those cols.
        save_path: filename or full path.

    Returns:
        matplotlib Figure containing a rendered table.
    """
    # matplotlib's table helper expects a 2D list of cell contents + a list
    # of column labels. We hide axes and let the table fill the figure.
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.4 * len(samples_df)))
    ax.axis('off')

    # Truncate long text so rows don't overflow the table width.
    def _trim(s: str, n: int = 80) -> str:
        return (s[:n] + '…') if isinstance(s, str) and len(s) > n else s

    cell_text = [
        [_trim(row['text']), row['true_label'],
         row['pred_lr'], row['pred_nepberta']]
        for _, row in samples_df.iterrows()
    ]
    col_labels = ['Text', 'True', 'LR', 'NepBERTa']

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc='center', cellLoc='left', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)  # taller rows, easier to read

    # Colour-code prediction cells: green if pred matches truth, red otherwise.
    # Column indices: True=1, LR=2, NepBERTa=3.
    for row_idx, (_, row) in enumerate(samples_df.iterrows(), start=1):
        for col_idx, col_name in [(2, 'pred_lr'), (3, 'pred_nepberta')]:
            cell = table[(row_idx, col_idx)]
            if row[col_name] == row['true_label']:
                cell.set_facecolor('#e8f5e9')  # pale green = correct
            else:
                cell.set_facecolor('#ffebee')  # pale red = wrong

    return _save_and_return(fig, save_path)


# ──────────────────────────────────────────────────────────────────────────
# 8) DATA QUALITY (duplicates + conflicts) — used by 01 EDA
# ──────────────────────────────────────────────────────────────────────────

def plot_data_quality(
    n_exact_repeats: int,
    n_conflict_reps: int,
    pair_counts: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Two-panel: (A) dup breakdown bar, (B) label-pair conflict heatmap.

    DONKEY: makes the "noisy labels" EDA finding visible in one figure
    for the paper. Panel A = how many redundant rows are harmless vs
    problematic. Panel B = among the problematic ones, which label-pair
    disagreements occurred (Positive-vs-Negative = worst; Neutral-vs-X
    = milder).

    Args:
        n_exact_repeats: rows that are exact (text+label) dupes (keep='first').
        n_conflict_reps: rows with same text but conflicting labels.
        pair_counts: symmetric DataFrame indexed by RAW labels (e.g., -1/0/1)
                     with off-diagonal cells = count of conflicting-text pairs.
        save_path: filename or full path.

    Returns:
        matplotlib Figure.
    """
    fig, (ax_bar, ax_hm) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A — breakdown of redundant rows
    cats = ['Agreeing\n(text+label match)', 'Conflicting\n(same text, diff label)']
    vals = [n_exact_repeats, n_conflict_reps]
    colors = ['#f0ad4e', '#d9534f']
    bars = ax_bar.bar(cats, vals, color=colors, edgecolor='white')
    for bar, v in zip(bars, vals):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, v, str(v),
                    ha='center', va='bottom', fontsize=10)
    ax_bar.set_ylabel('Redundant rows (keep="first")')
    ax_bar.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)

    # Panel B — label-pair conflict heatmap
    tick_labels = list(pair_counts.index)
    sns.heatmap(pair_counts.values, annot=True, fmt='d', cmap='Reds',
                xticklabels=tick_labels, yticklabels=tick_labels,
                cbar=False, ax=ax_hm, square=True)
    ax_hm.set_xlabel('Label')
    ax_hm.set_ylabel('Label')

    return _save_and_return(fig, save_path)
