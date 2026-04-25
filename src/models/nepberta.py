"""
nepberta.py — fine-tune NepBERTa for 3-class Nepali sentiment classification.

DONKEY EXPLANATION:
-------------------
=== What is NepBERTa? ===
NepBERTa is a RoBERTa-architecture transformer pre-trained on a large
corpus of Nepali text. "Pre-trained" means it ALREADY knows Nepali
grammar, common word patterns, and script rules — we don't teach it
Nepali from scratch. Our job is only to teach it the tiny additional
skill of mapping text → sentiment label.

=== Fine-tuning (in one picture) ===
            ┌─────────────────────────┐
   text ───►│ NepBERTa base (all 110M │───► 768-dim "understanding" vector
            │  weights get nudged)    │
            └─────────────────────────┘
                         │
                         ▼
                ┌────────────────┐
                │  3-class head  │───► logits → softmax → probs over
                │ (starts random)│        {Negative, Neutral, Positive}
                └────────────────┘

We update ALL weights (base + head) with a small learning rate (2e-5).
That's the "fine" in fine-tuning — nudge an already-skilled model toward
a specific task, don't retrain from zero.

=== Why HuggingFace Trainer? ===
A manual PyTorch training loop is ~80 lines of "move batch to GPU /
zero grads / forward / loss / backward / step / log / eval / save". The
HF `Trainer` class does all of that, plus:
  - automatic GPU/CPU detection
  - mixed-precision (fp16) on Colab GPU for free
  - logging + checkpointing + early-stopping callbacks
  - plays nicely with our `evaluation.py` via a `compute_metrics` hook
We accept the abstraction because the boilerplate it hides isn't what
this project is about.

⚠️  LOCAL vs COLAB
This module imports `torch` and `transformers`, neither of which is in
`requirements.txt` (CPU-only by choice — NepBERTa weights are ~450 MB
and training needs a GPU). Expect `ImportError` if you try to import
this locally; the matching notebook `03_nepberta_finetuning.ipynb`
runs on Colab where those libs come pre-installed.

⚠️  GPU MEMORY
If Colab's free-tier T4 OOMs during training, drop NEPBERTA_BATCH_SIZE
from 16 → 8 → 4 in config.py until it fits.

Public API (mirrors traditional_ml.py's shape):
  - NepaliSentimentDataset        (PyTorch Dataset wrapping text + labels)
  - build_tokenizer()             → AutoTokenizer for NepBERTa
  - build_model(num_labels)       → pretrained NepBERTa + classification head
  - tokenize_texts(tokenizer,...) → encode strings into model-ready tensors
  - train_model(train_df, val_df) → fine-tune via Trainer, return model
  - predict(model, tokenizer,...) → predicted class ints
  - predict_proba(...)            → per-class probabilities
  - save_model(model, tokenizer)  → persist to config.NEPBERTA_MODEL_DIR
  - load_model()                  → load tokenizer + model back
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# torch + transformers are Colab-only deps. Fail fast with a helpful
# message if someone tries to import this module locally.
try:
    import torch
    from torch.utils.data import Dataset

    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        EarlyStoppingCallback,
    )
except ImportError as e:
    raise ImportError(
        "src.models.nepberta requires `torch` and `transformers`, which are "
        "intentionally NOT in requirements.txt (local env stays CPU-only). "
        "Run this on Colab, or `pip install -r requirements-gpu.txt` on a "
        "machine with a GPU."
    ) from e


# Import-path fix so `from src import config / evaluation` works when this
# file is imported from a notebook OR as a module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src import evaluation                      # metric helpers reused below


# ──────────────────────────────────────────────────────────────────────────
# DATASET — wrap pandas rows in a PyTorch-compatible object
# ──────────────────────────────────────────────────────────────────────────

class NepaliSentimentDataset(Dataset):
    """Wrap tokenised texts + labels in the interface HF Trainer expects.

    DONKEY: PyTorch's `DataLoader` needs objects that support `len(obj)`
    and `obj[i]`. Our tokenizer output is already dict-like
    (`{'input_ids': [...], 'attention_mask': [...]}`) but we still need
    a thin class that, given an integer index, returns a SINGLE example
    with its label attached. That's what this does.

    Args:
        encodings: dict from tokenizer (input_ids, attention_mask, etc.);
                   each value is a LIST of length N (one row per example).
        labels:    iterable of N integer labels.
    """

    def __init__(self, encodings: dict, labels: Iterable[int]):
        # Keep the raw tokenizer output as dict-of-lists. We convert
        # individual rows to tensors lazily in __getitem__ — saves RAM
        # compared to converting everything upfront.
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self) -> int:
        # HF Trainer uses len() to build progress bars and epoch loops.
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Pull the idx-th row from every tokenizer output field and cast
        # to torch tensors (Trainer expects tensors, not Python lists).
        item = {
            key: torch.tensor(vals[idx])
            for key, vals in self.encodings.items()
        }
        # HF models look for a 'labels' key in the batch dict to compute loss.
        item['labels'] = torch.tensor(self.labels[idx])
        return item


# ──────────────────────────────────────────────────────────────────────────
# TOKENIZER + MODEL FACTORIES
# ──────────────────────────────────────────────────────────────────────────

def build_tokenizer():
    """Load the NepBERTa tokenizer.

    DONKEY: the tokenizer knows NepBERTa's BPE vocabulary — the same one
    `colab_bpe_length_eda.ipynb` used to measure doc lengths. First call
    downloads ~20 MB; subsequent calls use the Hugging Face cache.
    """
    return AutoTokenizer.from_pretrained(config.NEPBERTA_MODEL_NAME)


def build_model(num_labels: int = config.NUM_CLASSES):
    """Load pretrained NepBERTa and attach a fresh num_labels classification head.

    DONKEY: `AutoModelForSequenceClassification` does two things in one call:
      1. Download / cache-load NepBERTa's pretrained transformer backbone.
      2. Attach an UNTRAINED linear layer (hidden_size → num_labels) on top.

    The backbone weights are reused (it already knows Nepali). Only the
    new head starts random — that's what `train_model()` will teach.

    ⚠️ TF-only weights on Hugging Face
    The `NepBERTa/NepBERTa` repo ships only TensorFlow weights
    (`tf_model.h5`), not a PyTorch `pytorch_model.bin`. We try PyTorch
    first (cheap, no TF dep needed if it works) and fall back to
    `from_tf=True`, which requires `tensorflow` installed so transformers
    can convert the weights to PyTorch tensors on the fly.

    Args:
        num_labels: how many output classes (3 for us).

    Returns:
        torch.nn.Module ready for the Trainer.
    """
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            config.NEPBERTA_MODEL_NAME,
            num_labels=num_labels,
        )
    except OSError as e:
        # Error message from transformers contains "TensorFlow weights" when
        # only tf_model.h5 is available on the hub. Fall through to from_tf.
        if 'TensorFlow' in str(e) or 'tf_model' in str(e):
            print('[build_model] no PyTorch weights on hub; loading from '
                  'TensorFlow weights (requires `tensorflow` installed)...')
            # `low_cpu_mem_usage=False` is CRITICAL here.
            # In transformers >= 4.50 + torch >= 2.3 the default pre-allocates
            # the PyTorch model on the "meta" device (no real storage, just
            # shape metadata) to save RAM during load. That optimisation is
            # BROKEN on the TF → PyTorch path: the TF weight copies land as
            # no-ops, leaving every parameter as an empty meta tensor. The
            # model then fails to move to GPU/MPS with
            #   "Cannot copy out of meta tensor; no data!"
            # Disabling the optimisation forces real tensor allocation from
            # the start, which the TF-to-PT copy then fills correctly.
            return AutoModelForSequenceClassification.from_pretrained(
                config.NEPBERTA_MODEL_NAME,
                num_labels=num_labels,
                from_tf=True,
                low_cpu_mem_usage=False,
            )
        raise


# ──────────────────────────────────────────────────────────────────────────
# TOKENISING HELPER
# ──────────────────────────────────────────────────────────────────────────

def tokenize_texts(
    tokenizer,
    texts: Iterable[str],
    max_length: int = config.NEPBERTA_MAX_LENGTH,
) -> dict:
    """Batch-tokenise a list of strings; return encodings ready for a Dataset.

    DONKEY: turns each sentence into fixed-length integer arrays:
      - `input_ids`      : one integer per subword token
      - `attention_mask` : 1 for real tokens, 0 for pad — tells the model
                           which positions to ignore

    Args:
        tokenizer: a NepBERTa tokenizer from build_tokenizer().
        texts:     iterable of cleaned sentences.
        max_length: tokens past this position get TRUNCATED; shorter
                    sentences get PADDED up to this. We use 64 because
                    the Colab BPE EDA showed p99 ≈ 61 tokens.

    Returns:
        Dict with keys `input_ids`, `attention_mask`, each mapping to
        a list of lists (outer = rows, inner = max_length).
    """
    return tokenizer(
        list(texts),
        padding='max_length',   # pad short sequences with [PAD] up to max_length
        truncation=True,        # chop anything longer than max_length
        max_length=max_length,
    )


# ──────────────────────────────────────────────────────────────────────────
# METRICS CALLBACK for HF Trainer
# ──────────────────────────────────────────────────────────────────────────

def _compute_metrics_for_trainer(eval_pred) -> dict:
    """Convert Trainer's (logits, labels) pair into our project's metrics.

    The Trainer calls this during evaluation after each epoch. It MUST
    return a dict of SCALAR metrics only — arrays / matrices break the
    logger. We delegate computation to our shared evaluation module, then
    pick out the scalars.
    """
    logits, labels = eval_pred
    # `argmax(axis=-1)` picks the class with the highest logit per row.
    predictions = np.argmax(logits, axis=-1)

    m = evaluation.compute_performance_metrics(labels, predictions)
    return {
        'accuracy':        m['accuracy'],
        'macro_precision': m['macro_precision'],
        'macro_recall':    m['macro_recall'],
        'macro_f1':        m['macro_f1'],
        'weighted_f1':     m['weighted_f1'],
    }


# ──────────────────────────────────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────────────────────────────────

def train_model(
    train_texts: Iterable[str],
    train_labels: Iterable[int],
    val_texts: Iterable[str],
    val_labels: Iterable[int],
    output_dir: Optional[str] = None,
    early_stopping_patience: int = 2,
) -> tuple:
    """Fine-tune NepBERTa; return (trainer, model, tokenizer).

    DONKEY: this is the big one. The Trainer handles the whole loop —
    feed batches, compute loss, backprop, step optimizer, evaluate each
    epoch, checkpoint, stop early if validation stalls. We provide the
    data, hyperparameters, and metric definition; HF does everything else.

    Hyperparameters (all from config.py):
      - learning_rate = 2e-5   : standard BERT-family fine-tune LR.
                                 Much smaller than pretrain (~1e-3)
                                 because we're nudging, not teaching.
      - num_train_epochs = 5   : with early stopping, actual epochs run
                                 may be fewer.
      - per_device_train_batch_size = 16 : comfortable on Colab's T4 GPU
                                 for max_length=64.
      - weight_decay = 0.01    : L2 regularisation on non-bias weights.
      - warmup_ratio = 0.1     : first 10% of steps ramp LR from 0 up to
                                 the target — stabilises early training.

    Args:
        train_texts / train_labels: the 28,568-row training split.
        val_texts  / val_labels:    held-out validation (typically the
                                    test split for this single-shot project).
        output_dir: where checkpoints + final model land. Defaults to
                    config.NEPBERTA_MODEL_DIR.
        early_stopping_patience: stop after N evaluations with no macro_f1
                    improvement.

    Returns:
        (trainer, model, tokenizer)
          - trainer: retains `state.log_history` for loss-curve plots
          - model:   best-scoring weights (Trainer reloads these before return)
          - tokenizer: same as build_tokenizer(), returned for convenience
    """
    if output_dir is None:
        output_dir = config.NEPBERTA_MODEL_DIR

    # --- tokeniser + model ---
    tokenizer = build_tokenizer()
    model     = build_model(num_labels=config.NUM_CLASSES)

    # --- datasets ---
    print(f'[train_model] tokenising {len(train_texts):,} train + '
          f'{len(val_texts):,} val rows (max_length={config.NEPBERTA_MAX_LENGTH})...')
    train_encodings = tokenize_texts(tokenizer, train_texts)
    val_encodings   = tokenize_texts(tokenizer, val_texts)

    train_dataset = NepaliSentimentDataset(train_encodings, train_labels)
    val_dataset   = NepaliSentimentDataset(val_encodings,   val_labels)

    # --- training configuration ---
    # TrainingArguments collects every hyperparameter + IO setting in one
    # object. These are the knobs to touch if anything goes wrong.
    training_args = TrainingArguments(
        output_dir              = output_dir,

        # Core schedule
        num_train_epochs        = config.NEPBERTA_EPOCHS,
        learning_rate           = config.NEPBERTA_LEARNING_RATE,
        weight_decay            = config.NEPBERTA_WEIGHT_DECAY,
        warmup_ratio            = config.NEPBERTA_WARMUP_RATIO,

        # Batch sizes — same for eval (memory is lower there but keep simple).
        per_device_train_batch_size = config.NEPBERTA_BATCH_SIZE,
        per_device_eval_batch_size  = config.NEPBERTA_BATCH_SIZE,

        # Evaluation + checkpointing: both once per epoch.
        eval_strategy           = 'epoch',
        save_strategy           = 'epoch',

        # Early-stopping plumbing: track best model by macro_f1, keep it.
        load_best_model_at_end  = True,
        metric_for_best_model   = 'macro_f1',
        greater_is_better       = True,
        save_total_limit        = 1,   # keep only last checkpoint on disk

        # Logging: every 50 steps is plenty for a 2-3k-step fine-tune.
        logging_steps           = 50,
        report_to               = 'none',   # skip W&B / TensorBoard uploads

        # Seed must match our train/test split seed for determinism.
        seed                    = config.RANDOM_STATE,

        # fp16 mixed precision = ~1.5x speedup + half memory on Colab's T4.
        # torch.cuda.is_available() auto-disables it when no GPU present.
        fp16                    = torch.cuda.is_available(),
    )

    # --- the Trainer ---
    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        tokenizer       = tokenizer,
        compute_metrics = _compute_metrics_for_trainer,
        callbacks       = [EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
        )],
    )

    print('[train_model] starting fine-tuning...')
    trainer.train()
    print('[train_model] done.')

    return trainer, model, tokenizer


# ──────────────────────────────────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────────────────────────────────

def predict(
    model,
    tokenizer,
    texts: Iterable[str],
    batch_size: int = config.NEPBERTA_BATCH_SIZE,
) -> np.ndarray:
    """Return predicted class labels (ints 0/1/2).

    DONKEY: tokenize → model forward pass → argmax over class logits.
    Batched to keep the GPU saturated; `torch.no_grad()` skips gradient
    bookkeeping (saves memory + time — we're not training).

    Args:
        model: fine-tuned NepBERTa.
        tokenizer: matching NepBERTa tokenizer.
        texts: iterable of cleaned sentences.
        batch_size: rows per forward pass.

    Returns:
        1D numpy array of int predictions, same length as `texts`.
    """
    device = next(model.parameters()).device   # whatever device the model is on
    model.eval()                                # disable dropout etc.

    preds = []
    texts = list(texts)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenize_texts(tokenizer, batch)
            # Stack lists into tensors and move to device.
            input_ids      = torch.tensor(enc['input_ids']).to(device)
            attention_mask = torch.tensor(enc['attention_mask']).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # `.logits` shape = (batch_size, num_labels). argmax → class.
            batch_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            preds.extend(batch_preds.tolist())

    return np.array(preds, dtype=np.int64)


def predict_proba(
    model,
    tokenizer,
    texts: Iterable[str],
    batch_size: int = config.NEPBERTA_BATCH_SIZE,
) -> np.ndarray:
    """Return per-class probabilities, shape (n_samples, num_classes).

    DONKEY: identical to `predict()` but softmax the logits instead of
    argmaxing. Useful for the demo UI ("Positive: 78%, Neutral: 15%,
    Negative: 7%").
    """
    device = next(model.parameters()).device
    model.eval()

    probs_all = []
    texts = list(texts)

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenize_texts(tokenizer, batch)
            input_ids      = torch.tensor(enc['input_ids']).to(device)
            attention_mask = torch.tensor(enc['attention_mask']).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # `softmax(dim=-1)` normalises each row to sum to 1 = probabilities.
            batch_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            probs_all.append(batch_probs)

    # Concatenate batch results back into one (n, num_classes) array.
    return np.concatenate(probs_all, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────

def save_model(
    model,
    tokenizer,
    output_dir: Optional[str] = None,
) -> str:
    """Save the fine-tuned model + tokenizer to a directory.

    DONKEY: unlike sklearn's single .pkl, HuggingFace saves a FOLDER of
    files (config.json, model.safetensors, tokenizer vocab, etc.).
    `save_pretrained(directory)` writes everything; `load_model()` reads
    it back.

    Args:
        model: fine-tuned NepBERTa model.
        tokenizer: matching tokenizer.
        output_dir: where to save. Defaults to config.NEPBERTA_MODEL_DIR.

    Returns:
        Absolute path of the directory.
    """
    if output_dir is None:
        output_dir = config.NEPBERTA_MODEL_DIR
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f'[save_model] saved model + tokenizer to {output_dir}')
    return output_dir


def load_model(
    model_dir: Optional[str] = None,
) -> tuple:
    """Load a previously-saved fine-tuned model + tokenizer.

    Args:
        model_dir: directory to load from. Defaults to config.NEPBERTA_MODEL_DIR.

    Returns:
        (tokenizer, model).

    Raises:
        FileNotFoundError: if the directory doesn't exist.
    """
    if model_dir is None:
        model_dir = config.NEPBERTA_MODEL_DIR
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f'No saved model at {model_dir}. '
            f'Run train_model() + save_model() first.'
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model
