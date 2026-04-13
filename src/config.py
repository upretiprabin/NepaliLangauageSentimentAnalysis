"""
config.py — one single source of truth for ALL paths, hyperparameters,
and constants used across the project.

DONKEY EXPLANATION:
-------------------
In a growing project, "magic numbers" (like `max_features=10000`) start to
creep into every file. Six weeks later, if you want to try 5,000 instead,
you have to grep across the whole repo. Instead, we put every such setting
in ONE file and import it everywhere. Change once, the whole pipeline
follows.

Think of this file as the project's "settings panel".
"""

# TODO (Phase 1): fill in the actual constants. CLAUDE.md has the full
# reference block — copy from there. Expected groups:
#   - Paths (PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, OUTPUTS, FIGURES, ...)
#   - Dataset Kaggle slugs
#   - Label maps + names
#   - Random seed + test-size split ratio
#   - TF-IDF hyperparameters
#   - Logistic Regression hyperparameters
#   - NepBERTa hyperparameters
#   - Figure output settings (DPI, format)

pass
