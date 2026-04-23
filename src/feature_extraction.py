"""
feature_extraction.py — turn clean text into TF-IDF feature vectors.
(Only used by the traditional-ML pipeline. NepBERTa uses its own tokenizer,
not this.)

DONKEY EXPLANATION:
-------------------
Machine-learning models can't read words — they only crunch NUMBERS. So
before Logistic Regression can learn ANYTHING about a Nepali sentence,
we have to turn each sentence into a fixed-length row of numbers. That
step is called "feature extraction." For text, the classic recipe is
TF-IDF.

=== TF-IDF: Term Frequency × Inverse Document Frequency ===
Imagine we have 10,000 Nepali comments. A word like "को" (of) shows up in
almost every one — so it's useless for distinguishing a happy comment
from an angry one. A word like "भ्रष्टाचार" (corruption) appears in only
a handful — that's a STRONG signal.

  TF  = how often a word appears IN this document.
  IDF = how rare the word is ACROSS all documents.
  TF × IDF gives HIGH weight to "distinctive" words (common here,
  rare elsewhere) and LOW weight to words that are everywhere.

After vectorising, each comment becomes a sparse row: mostly zeros (for
words it doesn't contain) plus a few non-zero TF-IDF scores. Logistic
Regression then draws a decision boundary in this number-space to
separate the sentiment classes.

Public API:
  - build_vectorizer()        → fresh unfitted TfidfVectorizer (config-driven)
  - fit_vectorizer(texts)     → fit on train, return (vec, X_train sparse matrix)
  - transform(vec, texts)     → vectorise new texts with an already-fit vectorizer
  - save_vectorizer(vec)      → persist to disk (default: config.TFIDF_VECTORIZER_PATH)
  - load_vectorizer()         → load from disk
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────
# `os` / `sys` / `pathlib` — standard path handling + the `from src import
# config` import-path fix used across the project.
import os
import sys
from pathlib import Path

# Typing helpers for cleaner signatures.
from typing import Iterable, Optional

# `joblib` = sklearn's preferred save-format. Thin wrapper on pickle
# optimised for numpy arrays (which sklearn objects contain a lot of).
# ~5x faster than plain pickle for big sklearn models.
import joblib

# `scipy.sparse` types — used for type-hinting. TfidfVectorizer outputs a
# CSR (Compressed Sparse Row) matrix: 99%+ of entries in a TF-IDF matrix
# are zero (most docs don't contain most words), so storing zeros
# explicitly would waste gigabytes. CSR stores only non-zero values + their
# positions.
import scipy.sparse as sp

# The actual workhorse: sklearn's batteries-included TF-IDF vectorizer.
# Handles tokenising, counting, IDF weighting, and normalisation in one
# class with sensible defaults.
from sklearn.feature_extraction.text import TfidfVectorizer


# Import-path fix so this file works whether imported from a notebook,
# run as a script, or invoked via `python -m`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config


# ──────────────────────────────────────────────────────────────────────────
# BUILDING the vectorizer
# ──────────────────────────────────────────────────────────────────────────

def build_vectorizer() -> TfidfVectorizer:
    """Construct a fresh (unfitted) TfidfVectorizer with project settings.

    DONKEY: think of this as "buy the blank notebook." The vectorizer
    object is born empty, with its learning rules set (max vocabulary,
    ngram range, etc.) but NO vocabulary yet. You have to FIT it on the
    training texts before it knows which words even exist.

    Hyperparameters (all pulled from config.py, never hardcoded):
      - max_features : cap on vocabulary size. With 10K we keep the most
                       common 10,000 (1-grams + 2-grams). More = slower
                       AND risks learning noise from rare words. Less =
                       may miss useful signal. 10K is the safe middle
                       ground for text classification.
      - ngram_range  : (1, 2) = include both unigrams ("राम्रो") AND
                       bigrams ("राम्रो छैन"). Bigrams catch negations
                       and multi-word phrases that unigrams alone lose.
      - sublinear_tf : log-damps raw term frequencies (see config.py for
                       the full why).

    Things we OVERRIDE:
      - token_pattern : sklearn's default `\\b\\w\\w+\\b` breaks Devanagari
                        words mid-character because it treats vowel signs
                        (े, ा, ो) as NON-word chars. We use a Devanagari-
                        aware pattern — see config.TFIDF_TOKEN_PATTERN for
                        the full story.

    Things we deliberately DON'T change:
      - lowercase : default True; Devanagari has no case so it's a no-op
                    for our data. Safe.
      - stop_words : default None. No reliable Nepali stop-words list
                     exists anyway, and TF-IDF's IDF naturally suppresses
                     over-common words — leaving this off is correct.

    Returns:
        TfidfVectorizer, not yet fitted.
    """
    return TfidfVectorizer(
        max_features  = config.TFIDF_MAX_FEATURES,
        ngram_range   = config.TFIDF_NGRAM_RANGE,
        sublinear_tf  = config.TFIDF_SUBLINEAR_TF,
        token_pattern = config.TFIDF_TOKEN_PATTERN,
    )


# ──────────────────────────────────────────────────────────────────────────
# FITTING on the training corpus
# ──────────────────────────────────────────────────────────────────────────

def fit_vectorizer(
    train_texts: Iterable[str],
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """Build + fit a vectorizer on the training texts; return (vec, X_train).

    DONKEY: two things happen here:
      1. The vectorizer LEARNS the vocabulary from `train_texts` (picks
         the top-10K words / bigrams it observes).
      2. It then TRANSFORMS those SAME texts into numeric feature vectors.

    `.fit_transform(...)` is the single sklearn call that does both — it's
    equivalent to `.fit(texts)` then `.transform(texts)` but more efficient
    because some of the work is reused across the two steps.

    ⚠️ CRITICAL: ONLY fit on TRAINING data. If you fit on train+test
    together (or test alone), the vocabulary reflects words the model
    wasn't supposed to see = data leakage = inflated test accuracy.
    That's why this function is named `fit_vectorizer` — you pass ONLY
    the train texts.

    Args:
        train_texts: iterable of cleaned training-set strings.

    Returns:
        (vectorizer, X_train)
          - vectorizer : the fitted TfidfVectorizer
          - X_train    : scipy.sparse.csr_matrix of shape
                         (n_train_docs, vocabulary_size)
                         ready to feed into sklearn's LogisticRegression.
    """
    vec = build_vectorizer()

    # `list(train_texts)` materialises the iterable — TfidfVectorizer
    # scans it twice (once to build the vocab, once to compute weights),
    # and a generator would be exhausted after the first pass.
    X_train = vec.fit_transform(list(train_texts))

    # Sanity prints so the notebook user sees immediate feedback.
    # `vocabulary_` is the dict {word → column index} populated after fit.
    vocab_size = len(vec.vocabulary_)
    print(f'[fit_vectorizer] vocabulary size: {vocab_size:,} '
          f'(capped at {config.TFIDF_MAX_FEATURES:,})')

    # `.nnz` = number of non-zero entries. For a sparse matrix that's the
    # real storage footprint — dense equivalent would be shape[0]*shape[1].
    n_rows, n_cols = X_train.shape
    density_pct = X_train.nnz / (n_rows * n_cols) * 100
    print(f'[fit_vectorizer] X_train shape: {X_train.shape}, '
          f'nnz={X_train.nnz:,} ({density_pct:.2f}% non-zero)')

    return vec, X_train


# ──────────────────────────────────────────────────────────────────────────
# TRANSFORMING new texts (test set / demo-app input)
# ──────────────────────────────────────────────────────────────────────────

def transform(
    vec: TfidfVectorizer,
    texts: Iterable[str],
) -> sp.csr_matrix:
    """Vectorise new texts with an ALREADY-FITTED vectorizer.

    DONKEY: once the vectorizer has learned its vocabulary during fit,
    we use it to turn any future text (test set, a demo-app input, etc.)
    into the SAME-shape numeric matrix. Words not in the learned vocab
    are silently ignored — which is the correct behaviour for a deployed
    model (you can't retrofit new vocabulary without retraining).

    Args:
        vec: a fitted TfidfVectorizer.
        texts: iterable of cleaned strings.

    Returns:
        Sparse matrix of shape (len(texts), vocabulary_size).
    """
    return vec.transform(list(texts))


# ──────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────

def save_vectorizer(
    vec: TfidfVectorizer,
    path: Optional[str] = None,
) -> str:
    """Write a fitted vectorizer to disk.

    DONKEY: the fitted vectorizer HOLDS STATE — its learned vocabulary
    + IDF weights. Without saving, you'd have to refit every time you
    wanted to make a prediction. Saving lets the Phase 5 demo app load
    the vectorizer in milliseconds and vectorise user input on demand.

    Args:
        vec: fitted TfidfVectorizer.
        path: where to save. Defaults to config.TFIDF_VECTORIZER_PATH.

    Returns:
        Absolute path the file was written to.
    """
    if path is None:
        path = config.TFIDF_VECTORIZER_PATH
    # Make sure the destination directory exists (outputs/models/).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vec, path)
    print(f'[save_vectorizer] saved to {path}')
    return path


def load_vectorizer(path: Optional[str] = None) -> TfidfVectorizer:
    """Load a previously-saved vectorizer from disk.

    Args:
        path: where to load from. Defaults to config.TFIDF_VECTORIZER_PATH.

    Returns:
        Fitted TfidfVectorizer, ready for `.transform(...)`.

    Raises:
        FileNotFoundError: if the expected pickle isn't on disk yet.
    """
    if path is None:
        path = config.TFIDF_VECTORIZER_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No vectorizer at {path}. '
            f'Run fit_vectorizer() + save_vectorizer() first.'
        )
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────
# SELF-TEST  — `python -m src.feature_extraction`
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Minimal end-to-end smoke test with a toy 5-row corpus.
    sample_train = [
        'यो फिल्म राम्रो छ',
        'यो फिल्म बेकार छ',
        'नेपालको राजनीति भ्रष्ट छ',
        'कोभिड ले धेरै मान्छे लाई दुख दियो',
        'राम्रो समाचार हो',
    ]
    sample_test = [
        'फिल्म राम्रो लाग्यो',
        'अज्ञात शब्द',  # contains a word not seen in train → silently ignored
    ]

    print('Self-test: fit on 5-row train corpus...')
    vec, X_train = fit_vectorizer(sample_train)

    print('\nVocabulary (first 10 items by column index):')
    vocab_items = sorted(vec.vocabulary_.items(), key=lambda kv: kv[1])
    for word, idx in vocab_items[:10]:
        print(f'  {idx:3d}  {word}')

    print('\nTransform 2-row test corpus...')
    X_test = transform(vec, sample_test)
    print(f'X_test shape: {X_test.shape}, nnz={X_test.nnz}')

    print('\nSave / load round-trip...')
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp_path = f.name
    save_vectorizer(vec, path=tmp_path)
    loaded = load_vectorizer(path=tmp_path)
    os.unlink(tmp_path)
    # After reload, transforming the same texts should yield the same matrix.
    X_test_reloaded = transform(loaded, sample_test)
    assert (X_test != X_test_reloaded).nnz == 0, 'round-trip mismatch!'
    print('Round-trip OK (loaded vectorizer produces identical output).')
