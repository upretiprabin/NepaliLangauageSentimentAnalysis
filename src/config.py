"""
config.py — ONE single source of truth for ALL paths, hyperparameters,
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

# `os` = standard library module for dealing with file paths in a way that
# works on both macOS/Linux (forward slashes) and Windows (backslashes).
import os


# ──────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────
# We build every path relative to THIS file's location so the project works
# no matter where the user clones the repo. `__file__` is the path to
# config.py itself; `dirname` gives its folder (= src/); going one level up
# with ".." lands us at the project root.

# Absolute path to the `src/` folder (where this file lives).
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = one level above src/.
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

# Raw CSVs downloaded from Kaggle live here. Gitignored — user must download.
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

# Cleaned + split CSVs produced by preprocessing.py land here. Gitignored.
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")

# Everything we output (plots, trained models, result CSVs) goes here.
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
FIGURES = os.path.join(OUTPUTS, "figures")
MODELS = os.path.join(OUTPUTS, "models")
RESULTS = os.path.join(OUTPUTS, "results")

# Specific raw-CSV filename (so loaders don't hard-code strings).
PRIMARY_RAW_CSV = os.path.join(DATA_RAW, "aayamoza.csv")


# ──────────────────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────────────────
# Kaggle slug — used by the download script, kept here for reference.
PRIMARY_DATASET_KAGGLE = "aayamoza/nepali-sentiment-analysis"


# ──────────────────────────────────────────────────────────────────────────
# LABELS
# ──────────────────────────────────────────────────────────────────────────
# Raw CSVs use {-1, 0, 1} for {Negative, Neutral, Positive}. sklearn and
# transformers both prefer non-negative integers starting at 0, so we remap.
# `LABEL_MAP` is the conversion dict; `LABEL_NAMES` is indexed by the NEW
# integer (0→Negative, 1→Neutral, 2→Positive) so we can turn a model's
# prediction back into a human-readable word.
LABEL_MAP = {-1: 0, 0: 1, 1: 2}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
NUM_CLASSES = len(LABEL_NAMES)


# ──────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────────────────
# Any function that involves randomness (shuffling, train/test split, model
# init) should receive this seed. Same seed → same result on every run,
# which is essential for a fair comparison.
RANDOM_STATE = 42

# Fraction of rows held out for the test set. 0.2 = 80/20 split.
TEST_SIZE = 0.2


# ──────────────────────────────────────────────────────────────────────────
# TF-IDF + LOGISTIC REGRESSION
# ──────────────────────────────────────────────────────────────────────────
# Cap vocabulary at 10K most-frequent tokens. More = slower and risks
# overfitting on rare words; less = might miss useful signal. 10K is the
# standard middle ground for text classification.
TFIDF_MAX_FEATURES = 10000

# (1, 2) = use both single words (unigrams) AND two-word phrases (bigrams).
# Bigrams catch things like "not good" that unigrams alone miss.
TFIDF_NGRAM_RANGE = (1, 2)

# `sublinear_tf=True` applies 1 + log(tf) to raw term counts. Without it,
# a word appearing 100 times gets 100x the weight of a word appearing once.
# With it, 100x becomes ~1 + log(100) ≈ 5.6x — a more sensible ratio that
# stops a single spammy post from dominating the signal.
TFIDF_SUBLINEAR_TF = True

# ⚠️ Devanagari-aware token pattern. sklearn's default `\b\w\w+\b` treats
# vowel signs (े, ा, ो — Unicode category Mc/Mn) as NON-word characters,
# which wrongly splits Nepali words MID-CHARACTER: `नेपालको` → fragments
# `लक`, `जन`, etc. We override with a pattern that matches any run of
# ASCII word-chars OR chars in the Devanagari block (U+0900–U+097F).
# Result: `नेपालको` stays as one token, `भ्रष्ट` stays as one token, etc.
TFIDF_TOKEN_PATTERN = r'[\wऀ-ॿ]+'

# Grid-search values for the regularisation strength C.
# Smaller C = stronger regularisation (simpler model, less overfitting).
LR_C_VALUES = [0.01, 0.1, 1, 10]
LR_MAX_ITER = 1000

# Filenames for saved artifacts — composed once here so no module has to
# hardcode these strings.
TFIDF_VECTORIZER_PATH = os.path.join(MODELS, "tfidf_vectorizer.pkl")
LR_MODEL_PATH         = os.path.join(MODELS, "logistic_regression.pkl")
NEPBERTA_MODEL_DIR    = os.path.join(MODELS, "nepberta_finetuned")


# ──────────────────────────────────────────────────────────────────────────
# NepBERTa
# ──────────────────────────────────────────────────────────────────────────
# HuggingFace model id — downloaded automatically by `from_pretrained`.
NEPBERTA_MODEL_NAME = "NepBERTa/NepBERTa"

# ⚠️ NOTE: primary dataset has LONG reviews. 128 may truncate meaning —
# revisit after EDA shows the actual length distribution.
NEPBERTA_MAX_LENGTH = 64
NEPBERTA_BATCH_SIZE = 16
NEPBERTA_LEARNING_RATE = 2e-5
NEPBERTA_EPOCHS = 5
NEPBERTA_WARMUP_RATIO = 0.1
NEPBERTA_WEIGHT_DECAY = 0.01


# ──────────────────────────────────────────────────────────────────────────
# FIGURES
# ──────────────────────────────────────────────────────────────────────────
# 300 DPI = print-quality plots.
FIGURE_DPI = 300
FIGURE_FORMAT = "png"