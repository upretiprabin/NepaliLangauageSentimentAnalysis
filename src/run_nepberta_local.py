"""
run_nepberta_local.py — fine-tune NepBERTa on this machine instead of Colab.

DONKEY EXPLANATION:
-------------------
This is the local-machine twin of `notebooks/03_nepberta_finetuning.ipynb`.
It skips the Colab-specific bits (upload prompts, zip+download) and just
runs the pipeline directly against the preprocessed CSVs already on disk:

    data/processed/train.csv ─┐
    data/processed/test.csv  ─┤
                              ├─► train → evaluate → save ──► outputs/
              src.models.nepberta ┘

Invoke from the project root:

    python -m src.run_nepberta_local

Apple Silicon note
------------------
On M1/M2/M3 Macs PyTorch uses MPS (Metal Performance Shaders) as the GPU
backend. HuggingFace Trainer picks it automatically when CUDA is absent
and MPS is available. Training is ~3-5x slower than a T4 GPU but still
~10-20x faster than plain CPU — realistic wall time on an M1 + 16 GB RAM
with our config is 30-60 minutes (with early stopping).

If you hit "NotImplementedError: The operator aten::... is not currently
implemented for the MPS device", set PYTORCH_ENABLE_MPS_FALLBACK=1 before
running — that falls back to CPU for the missing op (done for you below).
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS + ENV
# ──────────────────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path

# Some MPS ops aren't implemented yet in stable PyTorch. This env var tells
# the runtime "if you hit an unsupported op, silently run it on CPU instead".
# Needs to be set BEFORE importing torch — so we set it at the very top.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

# Standard project path fix — makes `from src import ...` work.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from collections import defaultdict
import pandas as pd
import torch

from src import config
from src import evaluation as ev
from src import visualizations as viz
from src.models import nepberta as nb


# ──────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    # -------- device report --------
    # MPS = Apple Silicon GPU backend.
    # CUDA = NVIDIA GPU.
    # Fallback = CPU (slow, 6-8h).
    if torch.cuda.is_available():
        device_name = f'CUDA: {torch.cuda.get_device_name(0)}'
    elif torch.backends.mps.is_available():
        device_name = 'MPS (Apple Silicon GPU)'
    else:
        device_name = 'CPU (no accelerator detected)'
    print(f'Device: {device_name}')
    print(f'PyTorch: {torch.__version__}')
    print('=' * 60)

    # -------- load preprocessed data --------
    train_path = os.path.join(config.DATA_PROCESSED, 'train.csv')
    test_path  = os.path.join(config.DATA_PROCESSED, 'test.csv')
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        raise FileNotFoundError(
            f'Missing preprocessed data. Run `python -m src.preprocessing` '
            f'first to regenerate {train_path} + {test_path}.'
        )

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    print(f'Train: {train.shape}   Test: {test.shape}')
    print(f'Label balance (train): '
          f'{dict(train["label"].value_counts().sort_index())}')
    print('=' * 60)

    # -------- train --------
    # Everything inside a Profiler so we record wall-clock + peak memory.
    # Expected wall time on M1: 30-60 min depending on early-stopping kick-in.
    print('Training NepBERTa (this is the slow step)...')
    with ev.Profiler() as train_prof:
        trainer, model, tokenizer = nb.train_model(
            train_texts  = train['text'],
            train_labels = train['label'],
            val_texts    = test['text'],
            val_labels   = test['label'],
        )
    print(f'\nTraining finished in {train_prof.elapsed_seconds/60:.1f} min '
          f'(peak memory {train_prof.peak_memory_mb:.0f} MB)')
    print('=' * 60)

    # -------- evaluate --------
    # Inference-only profiler: measures per-sample cost without training noise.
    print('Evaluating on test set...')
    with ev.Profiler() as infer_prof:
        y_pred = nb.predict(model, tokenizer, test['text'])

    n_test = len(y_pred)
    ms_per_sample = infer_prof.elapsed_seconds * 1000 / n_test
    print(f'Inference: {infer_prof.elapsed_seconds:.1f}s total, '
          f'{ms_per_sample:.2f} ms/sample  '
          f'(peak memory {infer_prof.peak_memory_mb:.0f} MB)')

    metrics = ev.compute_performance_metrics(test['label'], y_pred)
    ev.print_metrics(metrics, title='NepBERTa — local run')

    # -------- figures --------
    # Confusion matrices (raw + normalised) and training-loss curve.
    viz.plot_confusion_matrix(metrics['confusion_matrix'],
                              save_path='confusion_matrix_nepberta.png')
    viz.plot_confusion_matrix(metrics['confusion_matrix'],
                              normalize=True,
                              save_path='confusion_matrix_nepberta_normalized.png')

    # Per-epoch loss curve from Trainer's log history.
    log = trainer.state.log_history
    train_by_epoch = defaultdict(list)
    for entry in log:
        if 'loss' in entry and 'epoch' in entry:
            train_by_epoch[int(entry['epoch']) + 1].append(entry['loss'])
    train_losses = [sum(train_by_epoch[e]) / len(train_by_epoch[e])
                    for e in sorted(train_by_epoch.keys())]
    val_losses   = [e['eval_loss'] for e in log if 'eval_loss' in e]
    # Match lengths in case early stopping truncated one side.
    n_epochs = min(len(train_losses), len(val_losses))
    if n_epochs > 0:
        viz.plot_training_loss(train_losses[:n_epochs], val_losses[:n_epochs],
                               save_path='training_loss_nepberta.png')
    print('Figures saved to outputs/figures/.')

    # -------- save model + results row --------
    model_path = nb.save_model(model, tokenizer)
    size_mb = ev.model_size_mb(model_path)
    print(f'Model directory size: {size_mb:.1f} MB')

    result_row = {
        'model': 'nepberta',
        **ev.flatten_metrics(metrics),
        'training_time_s':         train_prof.elapsed_seconds,
        'inference_time_s':        infer_prof.elapsed_seconds,
        'inference_ms_per_sample': ms_per_sample,
        'peak_memory_mb':          infer_prof.peak_memory_mb,
        'model_size_mb':           size_mb,
    }
    ev.save_results_row(result_row, 'all_models_results.csv')

    # -------- one-line summary --------
    print('=' * 60)
    print(f'DONE. '
          f'accuracy={metrics["accuracy"]:.4f}  '
          f'macro_f1={metrics["macro_f1"]:.4f}  '
          f'train={train_prof.elapsed_seconds/60:.1f}min  '
          f'inference={ms_per_sample:.2f}ms/sample')


if __name__ == '__main__':
    main()
