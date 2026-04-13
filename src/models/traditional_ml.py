"""
traditional_ml.py — the "old-school" sentiment classifier:
TF-IDF features → Logistic Regression → 3-class output.

DONKEY EXPLANATION:
-------------------
=== Why Logistic Regression? ===
Despite the name, Logistic Regression is a CLASSIFIER, not a regression. For
a 3-class problem, it draws two "decision boundaries" through the high-
dimensional TF-IDF feature space, carving it into Negative / Neutral /
Positive regions. New text gets vectorised, lands in one of those regions,
and that's its predicted class.

Why start here (before NepBERTa)?
  - trains in seconds on a laptop,
  - is a strong, well-understood baseline,
  - gives us a number to beat — if NepBERTa can't clear this bar, something
    is wrong with our fine-tuning.

=== GridSearchCV ===
The regularisation parameter `C` controls how hard the model tries to fit
the training data. Too small → underfits (too simple). Too large → overfits
(memorises noise). We don't guess: we try several values with cross-
validation and let the data pick the winner.

What this module will expose (Phase 2):
  - build_pipeline()         → sklearn Pipeline(tfidf → lr)
  - tune_hyperparameters(X_train, y_train) → GridSearchCV over C
  - train(...)               → fit best pipeline on full train set
  - predict(pipeline, texts) → predicted labels
  - predict_proba(...)       → class probabilities (for the demo UI)
  - save(pipeline, path)     → joblib.dump
  - load(path)               → joblib.load
"""

# TODO (Phase 2).
pass
