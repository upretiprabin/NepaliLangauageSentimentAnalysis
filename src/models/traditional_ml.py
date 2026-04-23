"""
traditional_ml.py — the "old-school" sentiment classifier:
TF-IDF features → Logistic Regression → 3-class output.

DONKEY EXPLANATION:
-------------------
=== Why Logistic Regression? ===
Despite the name, Logistic Regression is a CLASSIFIER, not a regression.
For a 3-class problem, it draws two decision boundaries through the
high-dimensional TF-IDF feature space, carving it into
Negative / Neutral / Positive regions. A new text gets vectorised, lands
in one of those regions, and that's its predicted class.

Why start here (before NepBERTa)?
  - trains in SECONDS on a laptop,
  - is a strong, well-understood baseline,
  - gives us a number to beat — if NepBERTa can't clear this bar,
    something is wrong with our fine-tuning.

=== GridSearchCV ===
The regularisation parameter `C` controls how hard the model tries to fit
the training data. Too small → underfits (too simple). Too large →
overfits (memorises noise). We don't GUESS the right value — we try
several and let the data pick the winner via k-fold cross-validation.

  CV flow for each candidate C:
    1. Split train set into k folds (e.g., 5 equal pieces).
    2. For i in 0..k-1:
         - Train LR with that C on the OTHER 4 pieces.
         - Score it on the held-out piece.
    3. Average the 5 scores → that C's CV score.
  The C with the best average CV score wins.
  Then the winning C is refit on the FULL train set.

Public API:
  - build_classifier()         → fresh unfit LogisticRegression
  - tune_and_train(X, y)       → GridSearchCV over C, returns best estimator
  - predict(model, X)          → predicted class labels (ints 0/1/2)
  - predict_proba(model, X)    → per-class probabilities (N, 3)
  - save_model(model)          → persist to config.LR_MODEL_PATH
  - load_model()               → load from disk
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────
import os
import sys
from pathlib import Path
from typing import Optional, Iterable

import numpy as np
import joblib

# `LogisticRegression` = the classifier itself. sklearn handles the 3-class
# case automatically via multinomial / one-vs-rest internally.
from sklearn.linear_model import LogisticRegression

# `GridSearchCV` = cross-validated hyperparameter search. Given a classifier
# and a dict of {param_name: [candidate_values]}, it trains every combo
# with k-fold CV, ranks them by a scoring metric, and exposes the winner
# via `.best_estimator_`.
from sklearn.model_selection import GridSearchCV


# Import-path fix (same idiom as the other modules).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config


# ──────────────────────────────────────────────────────────────────────────
# BUILD
# ──────────────────────────────────────────────────────────────────────────

def build_classifier(C: float = 1.0) -> LogisticRegression:
    """Construct a fresh (unfit) LogisticRegression with project settings.

    DONKEY: like `build_vectorizer()` for TF-IDF — just a factory that
    stamps out blank classifiers with consistent hyperparameters. You'll
    usually not call this directly; `tune_and_train()` uses it internally.

    Hyperparameters (pulled from config.py):
      - max_iter=1000 : LR uses iterative optimisation (L-BFGS by default).
                        The sklearn default is 100 iterations, which is
                        often too few for text-classification problems with
                        10K features — the solver warns "failed to
                        converge". 1000 is generous; it almost always
                        finishes well before the cap.
      - class_weight='balanced' : EDA showed heavy imbalance
                        (Negative ~40%, Neutral ~15%, Positive ~44%).
                        Without this, LR is biased toward the majority
                        classes (Negative + Positive) and under-predicts
                        Neutral. 'balanced' auto-computes inverse-frequency
                        weights per class, levelling the playing field.
      - random_state=42 : LR's L-BFGS solver is deterministic, so this
                        only matters for reproducibility of anything
                        that might use randomness internally (none here,
                        but consistent with the rest of the project).

    Args:
        C: regularisation strength (inverse). Smaller C = stronger
           regularisation = simpler model. GridSearch will pick the best
           from config.LR_C_VALUES; callers rarely pass this manually.

    Returns:
        LogisticRegression, not yet fit.
    """
    return LogisticRegression(
        C=C,
        max_iter=config.LR_MAX_ITER,
        class_weight='balanced',
        random_state=config.RANDOM_STATE,
    )


# ──────────────────────────────────────────────────────────────────────────
# TUNE + TRAIN  (GridSearchCV over C, then refit on full train)
# ──────────────────────────────────────────────────────────────────────────

def tune_and_train(
    X_train,                         # scipy sparse matrix or numpy array
    y_train: Iterable[int],
    cv: int = 5,
    scoring: str = 'f1_macro',
    verbose: int = 1,
) -> LogisticRegression:
    """Grid-search over C values, refit the best on the full training set.

    DONKEY: this is "train the model, but ALSO pick the best version of
    the model before committing to it". The alternative is to guess `C=1.0`
    and hope it's right. Grid search is ~5x more expensive (trains 5 folds
    × 4 C values = 20 classifiers instead of 1) but on our dataset the
    whole thing finishes in under a minute on a laptop.

    Why `scoring='f1_macro'`?
    -------------------------
    F1 (not accuracy) because accuracy is misleading on imbalanced data —
    always predicting "Positive" would score 44%. Macro (not micro/weighted)
    because we want EACH class weighted equally: a model that ignores
    Neutral should be penalised even if Neutral is only 15% of the data.

    Args:
        X_train: TF-IDF feature matrix from feature_extraction.fit_vectorizer.
        y_train: integer labels in {0, 1, 2}.
        cv: number of CV folds (5 is the standard default — trade-off
            between stability and runtime).
        scoring: sklearn scorer name. See docs for the full list.
        verbose: 0 = silent, 1 = prints progress, 2 = detailed.

    Returns:
        The winning LogisticRegression, already refit on the FULL train set
        (sklearn's GridSearchCV does the refit automatically once the best
        C is picked).
    """
    # The "grid" in GridSearchCV — one dict with the param name and its
    # candidate values. For multi-param searches you'd expand this dict.
    param_grid = {'C': config.LR_C_VALUES}

    # Base estimator. We pass in a fresh classifier; GridSearchCV will
    # clone it internally for each candidate C.
    base_lr = build_classifier()

    search = GridSearchCV(
        estimator=base_lr,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        # n_jobs=-1 = use all CPU cores. Each fold×C combo is independent,
        # so sklearn can train them in parallel — big wall-clock speedup.
        n_jobs=-1,
        # refit=True (default) = after finding the best C via CV, train
        # one final model on the FULL train set using that C. This final
        # model is what `.best_estimator_` returns and what we save.
        refit=True,
        verbose=verbose,
    )

    print(f'[tune_and_train] grid-searching C over {config.LR_C_VALUES} '
          f'with {cv}-fold CV (scoring="{scoring}")...')
    search.fit(X_train, y_train)

    # Print the CV score table so the notebook reader can see which C won
    # and by how much.
    print(f'[tune_and_train] best C = {search.best_params_["C"]}, '
          f'best CV {scoring} = {search.best_score_:.4f}')
    print('[tune_and_train] CV results per C:')
    for c_val, mean_score, std_score in zip(
        search.cv_results_['param_C'],
        search.cv_results_['mean_test_score'],
        search.cv_results_['std_test_score'],
    ):
        marker = '  ←' if c_val == search.best_params_['C'] else ''
        print(f'    C={c_val!s:<6}  mean={mean_score:.4f}  '
              f'std={std_score:.4f}{marker}')

    return search.best_estimator_


# ──────────────────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────────────────

def predict(
    model: LogisticRegression,
    X,                               # sparse matrix or numpy array
) -> np.ndarray:
    """Return predicted class labels (ints 0/1/2).

    DONKEY: for each row of X, the model computes probabilities for every
    class and picks the one with the highest probability. Output is a 1D
    numpy array of ints, same length as X.
    """
    return model.predict(X)


def predict_proba(
    model: LogisticRegression,
    X,
) -> np.ndarray:
    """Return per-class probabilities, shape (n_samples, 3).

    DONKEY: useful when we want more than just "the top class" — for the
    Gradio demo we show "Positive: 78%, Neutral: 15%, Negative: 7%",
    which is more informative than a bare label.

    The three columns are in the order model.classes_, which equals
    [0, 1, 2] = [Negative, Neutral, Positive] if trained on our data.
    """
    return model.predict_proba(X)


# ──────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ──────────────────────────────────────────────────────────────────────────

def save_model(
    model: LogisticRegression,
    path: Optional[str] = None,
) -> str:
    """Write a fitted LR model to disk.

    Args:
        model: a fitted LogisticRegression.
        path: optional override; defaults to config.LR_MODEL_PATH.

    Returns:
        Absolute path the file was written to.
    """
    if path is None:
        path = config.LR_MODEL_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f'[save_model] saved to {path}')
    return path


def load_model(path: Optional[str] = None) -> LogisticRegression:
    """Load a previously-saved LR model from disk.

    Args:
        path: optional override; defaults to config.LR_MODEL_PATH.

    Returns:
        Fitted LogisticRegression ready for `.predict(...)`.

    Raises:
        FileNotFoundError: if the expected pickle is missing.
    """
    if path is None:
        path = config.LR_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'No LR model at {path}. '
            f'Run tune_and_train() + save_model() first.'
        )
    return joblib.load(path)


# ──────────────────────────────────────────────────────────────────────────
# SELF-TEST — tiny end-to-end smoke
# ──────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Minimal test with a toy 6-row corpus. We don't test GridSearchCV here
    # (it needs k-fold splits per class, which a 6-row dataset barely
    # supports) — just verify the plain fit/predict path works.
    from src import feature_extraction as fe

    texts = [
        'यो फिल्म राम्रो छ',           # positive
        'एकदमै राम्रो फिल्म',            # positive
        'फिल्म ठीक ठाक छ',             # neutral
        'सामान्य फिल्म',                # neutral
        'यो फिल्म बेकार छ',            # negative
        'धेरै नराम्रो फिल्म',            # negative
    ]
    labels = [2, 2, 1, 1, 0, 0]

    print('Self-test: fit vectorizer + train LR (no grid search)...')
    vec, X = fe.fit_vectorizer(texts)

    # Use a single C and fit directly (skip GridSearch for this tiny set).
    model = build_classifier(C=1.0)
    model.fit(X, labels)

    # Sanity: predict on the training texts themselves (should be perfect).
    y_pred = predict(model, X)
    print(f'train labels: {labels}')
    print(f'predictions:  {y_pred.tolist()}')
    print(f'train accuracy: {(y_pred == np.array(labels)).mean():.2f}')

    # Save / load round-trip
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        tmp = f.name
    save_model(model, path=tmp)
    loaded = load_model(path=tmp)
    os.unlink(tmp)
    assert np.array_equal(predict(loaded, X), y_pred), 'round-trip mismatch'
    print('Round-trip OK.')
