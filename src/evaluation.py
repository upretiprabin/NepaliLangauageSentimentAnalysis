"""
evaluation.py — compute every metric the project reports, for either model.

DONKEY EXPLANATION:
-------------------
Once a model makes predictions on the test set, we want to answer the
question: "How good is it?" — and "good" is actually more than one number.

We compute TWO families of metrics:

  === PERFORMANCE (how often is the model right?) ===
    - accuracy    : of all predictions, how many were correct?
    - precision   : of the texts the model called "Positive", how many
                    actually were? (Did it cry wolf too often?)
    - recall      : of the texts that ARE "Positive", how many did the
                    model find? (Did it miss real cases?)
    - F1          : harmonic mean of precision + recall in one number
    - confusion matrix : a 3×3 grid showing every (true, predicted) pair

  === EFFICIENCY (how expensive is the model to run?)  ← our differentiator ===
    - training time (seconds)
    - inference time per sample (milliseconds)
    - total inference time (seconds)
    - model size on disk (MB)
    - peak memory during inference (MB)

Reporting BOTH sets matters: NepBERTa will very likely win on accuracy but
lose on efficiency. For deployment in Nepal's resource-constrained
environments (cheap servers, limited GPUs) the trade-off matters — that's
the central comparison this project makes.

Public API:
  - Profiler                        (context manager: time + peak memory)
  - compute_performance_metrics()   (every sklearn metric at once)
  - print_metrics()                 (pretty console output)
  - flatten_metrics()               (nested dict → flat CSV-friendly dict)
  - model_size_mb()                 (file-size helper)
  - save_results_row()              (append one model's full result to a CSV)
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────
# `os` + `Path` for file sizes and path building. We use os.path here to
# match the rest of the codebase.
import os
import sys
from pathlib import Path

# `time.perf_counter` = the most accurate wall-clock timer in Python.
# Always measures in seconds (float). Prefer it over `time.time` for
# benchmarks because it isn't affected by system clock adjustments.
import time

# `threading` lets us sample memory in the background while the timed
# operation runs on the main thread. `Event` is a thread-safe flag we use
# to tell the sampler "stop now, the block is done".
import threading

# `psutil` = cross-platform OS-process inspection. We use it to read RSS
# (Resident Set Size) — the physical RAM currently in use by our Python
# process. RSS includes memory allocated by C-extensions (PyTorch, sklearn's
# C backends), which tracemalloc would miss.
import psutil

# Typing helpers for cleaner signatures.
from typing import Iterable, Optional

# pandas for the results-CSV append helper.
import pandas as pd

# numpy types show up in sklearn outputs (e.g., confusion_matrix returns
# a 2D np.ndarray). We import it so we can cast / annotate cleanly.
import numpy as np


# Make `from src import config` importable regardless of how this module
# is called (CLI / notebook / other import path). See preprocessing.py for
# the same idiom.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config


# ──────────────────────────────────────────────────────────────────────────
# PROFILER  — time + peak memory context manager
# ──────────────────────────────────────────────────────────────────────────

class Profiler:
    """Context manager that captures wall-clock time AND peak RSS memory.

    DONKEY: imagine putting a stopwatch and a RAM-usage meter on a code
    block. When the block exits, you can read both numbers. That's it.

    Usage:
        with Profiler() as prof:
            y_pred = model.predict(X_test)
        print(prof.elapsed_seconds, prof.peak_memory_mb)

    Implementation note:
        `psutil.Process().memory_info().rss` gives the CURRENT RAM usage.
        A single call before + after the block misses transient peaks in
        the middle. So we spawn a lightweight background thread that
        samples every ~50ms and keeps the max. This adds <<1% overhead.

    Args:
        sample_interval_ms: how often to check memory (default 50 ms).
            Lower = more accurate peak, higher = less overhead.
    """

    def __init__(self, sample_interval_ms: int = 50):
        # Store interval in seconds (what time.sleep wants).
        self._interval = sample_interval_ms / 1000

        # `psutil.Process()` without args refers to THIS process.
        self._process = psutil.Process()

        # Results, populated when the block exits.
        self.elapsed_seconds: float = 0.0
        self.peak_memory_mb: float = 0.0

        # Internal state for the sampler thread.
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None

    def __enter__(self) -> 'Profiler':
        # Seed peak with the current memory so we never return "less than
        # start" due to a sampling gap.
        self.peak_memory_mb = self._process.memory_info().rss / (1024 ** 2)

        # `perf_counter` timestamp — we'll subtract at __exit__.
        self._start_time = time.perf_counter()

        # Start the sampler thread. `daemon=True` means it dies when the
        # main thread dies — no lingering background threads.
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Tell the sampler to stop and wait up to 1 second for it.
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        # Take the final elapsed time.
        self.elapsed_seconds = time.perf_counter() - self._start_time

        # One last memory sample in case the peak was literally in the
        # final ~50ms window (after the sampler thread's last tick).
        final_mb = self._process.memory_info().rss / (1024 ** 2)
        if final_mb > self.peak_memory_mb:
            self.peak_memory_mb = final_mb

    def _sample_loop(self) -> None:
        """Background loop: record max RSS until told to stop."""
        while not self._stop_event.is_set():
            current_mb = self._process.memory_info().rss / (1024 ** 2)
            if current_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_mb
            # Sleep, but wake up immediately if stop_event is set — `.wait()`
            # returns True once the event fires, which we ignore here.
            self._stop_event.wait(self._interval)


# ──────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> dict:
    """Compute every performance metric for one model's predictions.

    DONKEY: this is the one-shot "quiz score report card" for a model.
    Given the correct answers (y_true) and the model's guesses (y_pred),
    it returns a dict with every metric we report.

    We ALWAYS compute macro-averaged metrics too (not just micro / overall)
    because the dataset is imbalanced. A model that ignores the minority
    class can still score high on overall accuracy — macro-averaging
    forces each class to count equally, so a bad Neutral performance
    shows up even if Neutral is rare.

    Args:
        y_true: the correct labels (ints in {0, 1, 2}).
        y_pred: the model's predictions (same length, same label space).

    Returns:
        Dict with keys:
          - accuracy                  : float
          - macro_precision/recall/f1 : float (unweighted per-class average)
          - weighted_f1               : float (support-weighted F1)
          - per_class_precision       : list[float] of length 3
          - per_class_recall          : list[float]
          - per_class_f1              : list[float]
          - per_class_support         : list[int]  (# true samples per class)
          - confusion_matrix          : 2D numpy array, shape (3, 3)
          - classification_report     : sklearn's full text report, as str
    """
    # sklearn metrics live in two submodules; import here (not top-level)
    # because evaluation.py is small enough that a cold import on every
    # call costs nothing, and some notebooks don't want sklearn loaded
    # unless they actually compute metrics.
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report,
    )

    # `labels=[0, 1, 2]` pins the ORDER so the returned arrays always go
    # Negative → Neutral → Positive (not whatever sklearn infers from the
    # data). Otherwise a test set missing one class would return fewer
    # entries and mess up per-class reporting.
    label_ids = list(range(config.NUM_CLASSES))

    # Overall accuracy — simplest metric, "fraction correct".
    accuracy = float(accuracy_score(y_true, y_pred))

    # Per-class precision / recall / F1 / support.
    # `zero_division=0` = if a class has no predictions at all (precision
    # = 0/0), return 0 instead of raising a warning. Avoids noisy output.
    per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, zero_division=0
    )

    # Macro-averaged (unweighted mean across classes). One number per metric.
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average='macro', zero_division=0
    )

    # Weighted F1 — each class contributes proportional to its support.
    # Useful as a single "overall quality" number when classes are imbalanced.
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=label_ids, average='weighted', zero_division=0
    )

    # Confusion matrix — rows = true labels, cols = predicted labels.
    # cm[i, j] = # of samples with true label i predicted as label j.
    # Diagonal = correct; off-diagonal = errors.
    cm = confusion_matrix(y_true, y_pred, labels=label_ids)

    # Pre-formatted text report. Great for paste-into-notebook prints and
    # for inclusion in any results write-up.
    report_str = classification_report(
        y_true, y_pred,
        labels=label_ids,
        target_names=config.LABEL_NAMES,
        zero_division=0,
        digits=4,  # 4 decimal places — sufficient resolution
    )

    # Return everything in ONE flat dict. Callers pick what they need.
    # Explicit float() / .tolist() casts so the dict is pure Python (no
    # numpy / pandas types) — that way pd.DataFrame([metrics]) Just Works.
    return {
        'accuracy':             accuracy,
        'macro_precision':      float(mac_p),
        'macro_recall':         float(mac_r),
        'macro_f1':             float(mac_f1),
        'weighted_f1':          float(weighted_f1),
        'per_class_precision':  per_p.tolist(),
        'per_class_recall':     per_r.tolist(),
        'per_class_f1':         per_f1.tolist(),
        'per_class_support':    per_support.tolist(),
        'confusion_matrix':     cm,          # numpy array, for plotting
        'classification_report': report_str,  # string, for printing
    }


# ──────────────────────────────────────────────────────────────────────────
# PRINTING
# ──────────────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, title: str = 'Performance') -> None:
    """Pretty-print a metrics dict to the console.

    DONKEY: useful in notebooks where you want a quick look without
    loading the CSV into pandas. Keep it tight — 10 lines max.
    """
    print(f'── {title} ──')
    print(f'  accuracy        : {metrics["accuracy"]:.4f}')
    print(f'  macro precision : {metrics["macro_precision"]:.4f}')
    print(f'  macro recall    : {metrics["macro_recall"]:.4f}')
    print(f'  macro F1        : {metrics["macro_f1"]:.4f}')
    print(f'  weighted F1     : {metrics["weighted_f1"]:.4f}')
    print()
    # The classification_report string already has per-class breakdown
    # and averages in a clean grid — just print it.
    print(metrics['classification_report'])


# ──────────────────────────────────────────────────────────────────────────
# FLATTEN — nested dict → CSV-friendly flat dict
# ──────────────────────────────────────────────────────────────────────────

def flatten_metrics(metrics: dict) -> dict:
    """Expand per-class lists into scalar columns for CSV storage.

    Why? A CSV row can't hold a list. So we replace
        per_class_precision = [0.81, 0.73, 0.87]
    with three scalar columns:
        precision_Negative, precision_Neutral, precision_Positive

    The confusion matrix (2D array) and the multi-line classification
    report are DROPPED from the flat dict — they belong in their own
    files (a plot + a .txt).

    Args:
        metrics: output of compute_performance_metrics().

    Returns:
        Flat dict of scalars (str/int/float), safe for pd.DataFrame([...]).
    """
    flat = {
        'accuracy':        metrics['accuracy'],
        'macro_precision': metrics['macro_precision'],
        'macro_recall':    metrics['macro_recall'],
        'macro_f1':        metrics['macro_f1'],
        'weighted_f1':     metrics['weighted_f1'],
    }

    # Expand each per-class list into LABEL_NAMES-indexed scalar columns.
    # `zip(LABEL_NAMES, list)` pairs them up: ('Negative', 0.81), etc.
    for metric_name in ('precision', 'recall', 'f1', 'support'):
        per_class_list = metrics[f'per_class_{metric_name}']
        for class_name, value in zip(config.LABEL_NAMES, per_class_list):
            flat[f'{metric_name}_{class_name}'] = value

    return flat


# ──────────────────────────────────────────────────────────────────────────
# MODEL SIZE ON DISK
# ──────────────────────────────────────────────────────────────────────────

def model_size_mb(path: str) -> float:
    """Return on-disk size in megabytes for a file OR a directory.

    DONKEY: Logistic-regression pickles are ~1 MB. NepBERTa checkpoints
    are ~450 MB. That contrast goes directly into the efficiency-comparison
    figure — one of our key efficiency findings.

    Handles both cases:
      - single file (e.g., logistic_regression.pkl)
      - directory tree (e.g., nepberta_finetuned/ containing many files)

    Args:
        path: file or directory path.

    Returns:
        Size in MB (float). Returns 0.0 if the path doesn't exist.
    """
    if not os.path.exists(path):
        return 0.0

    # Single file — quickest path.
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 ** 2)

    # Directory — walk and sum sizes of every file inside. `os.walk` yields
    # (dirpath, dirnames, filenames) tuples; we only need the files.
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            # Skip broken symlinks / permission errors defensively.
            try:
                total_bytes += os.path.getsize(fpath)
            except OSError:
                pass

    return total_bytes / (1024 ** 2)


# ──────────────────────────────────────────────────────────────────────────
# RESULTS CSV — one row per model run
# ──────────────────────────────────────────────────────────────────────────

def save_results_row(result: dict, filename: str) -> str:
    """Append ONE result row to outputs/results/<filename>.

    DONKEY: each model evaluation produces a flat dict of numbers. We
    append each dict as a row to a CSV so:
      - re-runs don't lose history (we can compare today's run to yesterday's)
      - the comparison phase (Phase 4) just reads one CSV to plot everything.

    If the file doesn't exist yet, it's created with a header. If it does,
    the row is appended WITHOUT writing the header again.

    Args:
        result: flat dict (strings/numbers only — no nested lists/arrays).
                Typically `{'model': '...'} | flatten_metrics(...) | timing dict`.
        filename: relative to config.RESULTS, e.g., 'traditional_ml_results.csv'.

    Returns:
        Full absolute path of the file that was written.
    """
    os.makedirs(config.RESULTS, exist_ok=True)
    path = os.path.join(config.RESULTS, filename)

    # Wrap the dict in a list so pandas treats it as ONE row, not column names.
    df = pd.DataFrame([result])

    # `mode='a'` = append. `header=not os.path.exists(path)` means: write
    # the header row only on the first insertion (when the file is being
    # created). Saves us from ending up with duplicate headers mid-file.
    header = not os.path.exists(path)
    df.to_csv(path, mode='a', header=header, index=False)

    print(f'[save_results_row] appended to {path}')
    return path


# ──────────────────────────────────────────────────────────────────────────
# SELF-TEST — quick sanity run when executed as a script
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Tiny synthetic example to verify all pieces wire together correctly.
    # Real integration is exercised in the 02/03 notebooks.
    print('Self-test: fake 6-row y_true / y_pred...')
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    with Profiler() as prof:
        m = compute_performance_metrics(y_true, y_pred)

    print_metrics(m, title='Self-test metrics')
    print(f'Profiler: {prof.elapsed_seconds:.4f}s, '
          f'peak {prof.peak_memory_mb:.1f} MB')

    flat = flatten_metrics(m)
    print('Flattened keys:', list(flat.keys()))
