"""
preprocessing.py — turn raw Nepali CSV rows into clean (text, label) pairs,
ready to feed into either the TF-IDF pipeline or the NepBERTa tokenizer.

DONKEY EXPLANATION:
-------------------
Raw Nepali social-media text is messy. It contains URLs, @mentions, hashtag
symbols, duplicate rows, and (worse) some rows where the SAME text was
annotated with DIFFERENT sentiment labels by different annotators. If we
feed this junk directly to a model, it wastes "learning capacity" on noise,
learns contradictions from the conflicting duplicates, or — most dangerously
— leaks test data into training (if the same sentence lands in both sides
of a random split).

This module is the single "data-cleaning funnel" that fixes all of that:

    raw CSV ─► clean text ─► drop dupes + conflicts ─► remap labels
                                                   ├─► train.csv
                                                   └─► test.csv  (80/20)

CRITICAL: both the TF-IDF pipeline AND NepBERTa read the SAME train.csv /
test.csv produced here. Otherwise their accuracy scores aren't really
comparable — one model might have had an easier (or harder) version of
the data. Same cleaner → fair fight.

Public functions (in the order the pipeline calls them):
  - load_raw()                  → DataFrame with columns ['text', 'label']
  - clean_text(s)               → regex-clean a single string
  - apply_cleaning(df)          → run clean_text over every row, drop empties
  - resolve_duplicates(df)      → drop exact dupes + conflicting-label groups
  - normalize_labels(df)        → map {-1, 0, 1} → {0, 1, 2} via LABEL_MAP
  - build_train_test_split(df)  → (train_df, test_df) stratified 80/20
  - save_processed(train, test) → write train.csv / test.csv
  - run_pipeline()              → one-call end-to-end orchestrator
"""

# ──────────────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────────────
# `re` = standard-library regular expressions. We use it for URL / @mention /
# hashtag stripping. `re.compile` builds a pattern object once; re-using it
# is faster than passing the raw pattern string each call, which matters
# across a 35k-row loop.
import re

# `os` for file-existence checks and directory creation. We don't use
# `pathlib` here only because the project already uses os.path-style strings
# in config.py — keeping the two consistent.
import os

# `sys` + `pathlib.Path` = one-time fix so `from src import config` works
# whether this file is imported from a notebook OR run as a script from the
# project root (`python -m src.preprocessing`).
import sys
from pathlib import Path

# pandas = the DataFrame workhorse. Every CSV read / write goes through it.
import pandas as pd

# `train_test_split` = sklearn's canonical row-shuffler that ALSO supports
# stratified sampling (preserve the label ratio in both halves). Picking the
# right one here is critical — a naive random split on our imbalanced data
# can leave the rare class under-represented in either half.
from sklearn.model_selection import train_test_split


# Make `from src import config` work when this file is run directly
# (e.g. `python src/preprocessing.py`). Path(__file__).parent is src/;
# .parent.parent is the project root, which must be on sys.path for the
# `src` package to be importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Everything lives in config.py — never hardcode paths or hyperparameters.
from src import config


# ──────────────────────────────────────────────────────────────────────────
# REGEX PATTERNS — compiled ONCE at module-load, re-used for every row
# ──────────────────────────────────────────────────────────────────────────
# CONCEPT: compiled regex
# -----------------------
# `re.compile(pattern)` parses the pattern once into an internal state
# machine. Calling `.sub()` on that compiled object is ~5x faster than
# calling `re.sub(pattern_string, ...)` which re-parses the pattern each
# call. For 35k rows × 4 patterns, the difference shows.
#
# Why these four patterns and not more?
# -------------------------------------
# The EDA's noise audit showed these were the common noise sources in our
# data. Other patterns (stripping Latin letters, digits, emojis) would
# destroy real content — code-mixing, dates, and emotion-bearing symbols
# are all signal, not noise. Evidence-based cleaning > kitchen-sink cleaning.

# URL with scheme. `\S+` = one-or-more non-whitespace = the URL body.
# `r''` (raw string) = don't interpret backslashes as Python escapes.
_URL_RE = re.compile(r'https?://\S+')

# URL without scheme (www.foo.com). Separate pattern so we can drop these
# without false-matching "www" inside other text.
_WWW_RE = re.compile(r'www\.\S+')

# @mentions. `\w+` = word characters (letters / digits / underscore).
# In Python's `re` module, `\w` is Unicode-aware by default so it'll also
# match Devanagari word characters — fine for us since mentions are usually
# ASCII anyway.
_MENTION_RE = re.compile(r'@\w+')

# `#hashtag` — strip the `#` symbol but keep the word. The capturing group
# `(\w+)` holds the word; the replacement `\1` puts it back. So
# "#नेपाल राम्रो" becomes "नेपाल राम्रो" — we keep the semantic content
# and only drop the social-media decoration.
_HASHTAG_RE = re.compile(r'#(\w+)')

# Any whitespace run (spaces, tabs, newlines). After the removals above we
# often end up with double-spaces where a URL used to be; this collapses
# them back to single spaces.
_WS_RE = re.compile(r'\s+')


# ──────────────────────────────────────────────────────────────────────────
# TEXT CLEANING — per-row
# ──────────────────────────────────────────────────────────────────────────

def clean_text(s: str) -> str:
    """Strip URLs / @mentions / # symbols and collapse whitespace.

    DONKEY: think of this as a filter funnel. Raw text goes in at the top;
    the filter removes web-scraping noise but PRESERVES Nepali words,
    digits, English code-mixing, and emojis — the parts that actually
    carry meaning. What comes out the bottom is the clean sentence a
    model should see.

    Deliberate NON-removals (decided from EDA evidence, not blog posts):
      - Latin letters  → kept (code-mixed Nepali is real)
      - Digits         → kept (dates, quantities carry context)
      - Emoji          → kept (emotion-bearing in social media)
      - Case           → no-op (Devanagari has no case)

    Args:
        s: Raw text string. NaN or non-string inputs safely return ''.

    Returns:
        Cleaned string. May be empty if input was only noise
        (e.g., a row that was just a URL with nothing else).
    """
    # Defensive early-exit: pandas sometimes reads a malformed row as `nan`
    # (a float) or None. Running regex on that would raise TypeError deep
    # in the loop — cheaper to just return '' and let apply_cleaning drop
    # the row afterwards.
    if not isinstance(s, str):
        return ''

    # Order matters:
    #   - URLs first (they contain '/' and '.' that could interact weirdly
    #     with other patterns).
    #   - Mentions next.
    #   - Hashtags last (simple one-char strip).
    s = _URL_RE.sub('', s)
    s = _WWW_RE.sub('', s)
    s = _MENTION_RE.sub('', s)
    s = _HASHTAG_RE.sub(r'\1', s)  # \1 = keep the captured word

    # Collapse whitespace and strip the ends. If we removed a URL from the
    # middle of a sentence, two spaces are left where it used to be; this
    # normalises to single-space.
    s = _WS_RE.sub(' ', s).strip()

    return s


# ──────────────────────────────────────────────────────────────────────────
# LOAD RAW
# ──────────────────────────────────────────────────────────────────────────

def load_raw() -> pd.DataFrame:
    """Read the raw Kaggle CSV and return a clean 2-column DataFrame.

    DONKEY: the CSV has three columns — an unhelpful leftover index
    (`Unnamed: 0`) saved by whoever exported it, and the real data columns
    `Sentences` / `Sentiment`. We throw away the index and rename the rest
    to the simpler `text` / `label` that every downstream function
    expects. Canonical column names keep the rest of the codebase tidy.

    Returns:
        DataFrame with exactly two columns: `text` (str), `label` (int).

    Raises:
        FileNotFoundError: if data/raw/aayamoza.csv isn't on disk yet.
    """
    # Fail loudly if the user hasn't downloaded the dataset. Silent "0 rows"
    # results are the kind of bug that wastes an afternoon to diagnose.
    if not os.path.exists(config.PRIMARY_RAW_CSV):
        raise FileNotFoundError(
            f'{config.PRIMARY_RAW_CSV} not found. '
            f'Run data/download_data.py or download manually from Kaggle: '
            f'{config.PRIMARY_DATASET_KAGGLE}'
        )

    df = pd.read_csv(config.PRIMARY_RAW_CSV)

    # `errors='ignore'` means: if 'Unnamed: 0' isn't in the DataFrame (e.g.
    # someone re-exported the CSV more cleanly), skip silently instead of
    # raising KeyError. Defensive.
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Rename to canonical names. After this line every other function in
    # the file can assume columns are exactly ['text', 'label'].
    df = df.rename(columns={'Sentences': 'text', 'Sentiment': 'label'})

    print(f'[load_raw] loaded {len(df):,} rows from {config.PRIMARY_RAW_CSV}')
    return df


# ──────────────────────────────────────────────────────────────────────────
# APPLY CLEANING (DataFrame-level wrapper around clean_text)
# ──────────────────────────────────────────────────────────────────────────

def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Run clean_text over every row and drop rows that came out empty.

    DONKEY: if a row was pure noise (e.g., "https://youtu.be/abc123" with
    no words around it), after stripping it becomes ''. That's useless to
    train on — no signal either way — so we drop it.
    """
    n_start = len(df)
    df = df.copy()  # don't mutate the caller's DataFrame

    # `.apply(fn)` runs `fn` once per element — slower than a vectorised op
    # but clean_text is regex-heavy and doesn't vectorise nicely. For 35k
    # short rows the total runtime is ~1 second, which is fine.
    df['text'] = df['text'].apply(clean_text)

    # Drop rows that became empty (or whitespace-only) after cleaning.
    # We re-check with .str.strip() in case an edge-case left '\t' or similar.
    df = df[df['text'].str.strip() != '']
    df = df.reset_index(drop=True)

    n_end = len(df)
    print(f'[apply_cleaning] {n_start:,} → {n_end:,} '
          f'(dropped {n_start - n_end} empty-after-cleaning rows)')
    return df


# ──────────────────────────────────────────────────────────────────────────
# DUPLICATE RESOLUTION
# ──────────────────────────────────────────────────────────────────────────

def resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates AND all rows belonging to conflicting-label texts.

    DONKEY: two kinds of duplicates live in the Kaggle dataset:

      1. Same text + same label (harmless but redundant). Keep one copy,
         throw the rest.
      2. Same text + DIFFERENT labels (annotator disagreement). Poisonous —
         the model would learn contradictions. We drop ALL rows belonging
         to each such conflicting text because there's no principled way
         to pick a winner for only 19 cases.

    Must run BEFORE train/test split, otherwise the same sentence can end
    up on both sides of the split (data leakage → inflated test accuracy).

    Why cleaning first THEN dedup?
    ------------------------------
    Two rows whose raw text differs only by a URL become identical after
    cleaning. Deduping post-clean catches those, pre-clean wouldn't.

    Args:
        df: DataFrame with 'text' and 'label' columns.

    Returns:
        Deduplicated DataFrame with reset index.
    """
    n_start = len(df)

    # Step 1 — drop exact (text + label) duplicates.
    # `keep='first'` means: for each group of identical rows, keep the FIRST
    # one and drop the rest. Standard convention.
    df = df.drop_duplicates(subset=['text', 'label'], keep='first')
    n_after_exact = len(df)

    # Step 2 — find texts with MULTIPLE distinct labels, then drop every
    # row belonging to those texts.
    #
    # Chain explained:
    #   .groupby('text')['label']   → SeriesGroupBy, one group per text
    #   .nunique()                  → number of DISTINCT labels per group
    #   .loc[lambda s: s > 1]       → keep only groups with >1 distinct label
    #   .index                      → the text values themselves
    conflict_texts = (
        df.groupby('text')['label']
        .nunique()
        .loc[lambda s: s > 1]
        .index
    )

    # `~` = logical NOT. Keep rows whose text is NOT in the conflict set.
    df = df[~df['text'].isin(conflict_texts)]
    n_after_conflict = len(df)

    # Reset to a fresh 0..n-1 index. Downstream tools (sklearn, tokenizers)
    # generally don't care about the index, but a clean one avoids nasty
    # surprises like iloc != loc mismatches.
    df = df.reset_index(drop=True)

    print(f'[resolve_duplicates] {n_start:,} → {n_after_exact:,} '
          f'(-{n_start - n_after_exact} exact dupes) → '
          f'{n_after_conflict:,} '
          f'(-{n_after_exact - n_after_conflict} conflict rows across '
          f'{len(conflict_texts)} texts)')
    return df


# ──────────────────────────────────────────────────────────────────────────
# LABEL NORMALIZATION
# ──────────────────────────────────────────────────────────────────────────

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Remap raw {-1, 0, 1} labels to model-friendly {0, 1, 2} via LABEL_MAP.

    DONKEY: sklearn's classification metrics and HuggingFace's loss
    functions both prefer non-negative integer labels starting at 0.
    Our CSV's -1 (Negative) is a pain — the easy fix is to remap:
        -1 → 0   (Negative)
         0 → 1   (Neutral)
         1 → 2   (Positive)
    After this, LABEL_NAMES[label] gives back the human-readable word.

    Args:
        df: DataFrame with an integer 'label' column in {-1, 0, 1}.

    Returns:
        New DataFrame with labels remapped to {0, 1, 2}. Rows with any
        unexpected label value are defensively dropped.
    """
    n_start = len(df)
    df = df.copy()

    # `.map(dict)` replaces each value by looking it up in the dict. Any
    # value NOT in the dict becomes NaN — useful here because it lets us
    # spot (and drop) unexpected labels instead of silently passing them
    # through.
    df['label'] = df['label'].map(config.LABEL_MAP)

    # Drop rows with unmapped labels (shouldn't happen for clean data, but
    # guard anyway). `.dropna(subset=...)` only drops rows with NaN in the
    # named column, leaving other NaNs (if any) alone.
    df = df.dropna(subset=['label'])

    # `.map` returns float64 when NaN is possible; we know the surviving
    # values are now clean ints 0/1/2, so cast back to int for downstream
    # stability (sklearn happily accepts floats but ints are clearer).
    df['label'] = df['label'].astype(int)

    n_end = len(df)
    if n_end < n_start:
        print(f'[normalize_labels] dropped {n_start - n_end} rows '
              f'with unexpected label values')
    else:
        print(f'[normalize_labels] remapped {n_end:,} rows; '
              f'labels now in {sorted(df["label"].unique().tolist())}')
    return df


# ──────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────────────────

def build_train_test_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split with a fixed random seed.

    DONKEY (stratified): means the label PROPORTIONS are preserved in both
    halves. Without it, a random draw on imbalanced data can put most of
    the rare class in training and almost none in test — making test
    accuracy meaningless because the model hardly gets quizzed on the
    rare class.

    DONKEY (random_state=42): fixes the "random" shuffle so every run
    produces the exact same split. Essential for comparing two models
    fairly (both must train and test on identical rows) and for
    reproducibility in the viva.

    Args:
        df: Full DataFrame with 'text' and 'label' columns.

    Returns:
        (train_df, test_df) with the ratio set by config.TEST_SIZE (0.2).
    """
    train_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,          # 0.2 → 80/20 split
        stratify=df['label'],                 # preserve class balance
        random_state=config.RANDOM_STATE,     # 42, same every run
    )

    # Fresh 0..n-1 index on both sides. The train_test_split output keeps
    # the original row indexes which makes downstream iloc/loc behaviour
    # confusing; resetting avoids that class of bug.
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    print(f'[build_train_test_split] train={len(train_df):,}, '
          f'test={len(test_df):,}')
    # `value_counts().sort_index()` gives counts grouped by label, in order
    # 0, 1, 2. We print both sides' counts as a quick sanity check that
    # stratification worked (the ratios should match closely).
    print(f'  train label balance: '
          f'{dict(train_df["label"].value_counts().sort_index())}')
    print(f'  test  label balance: '
          f'{dict(test_df["label"].value_counts().sort_index())}')
    return train_df, test_df


# ──────────────────────────────────────────────────────────────────────────
# SAVE
# ──────────────────────────────────────────────────────────────────────────

def save_processed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Write `train.csv` and `test.csv` into data/processed/.

    DONKEY: we save intermediate artifacts as CSVs (instead of keeping
    them only in a notebook's memory) so Phase 2 and Phase 3 don't have
    to re-run this pipeline every time. The split is deterministic anyway
    (seed=42), but loading a CSV is 10x faster than re-cleaning 35k rows.

    Uses index=False so the CSV doesn't carry the pandas row index —
    keeps the file diff-friendly and avoids re-creating a stale "Unnamed: 0"
    column on the next read.
    """
    # Make sure data/processed/ exists. exist_ok=True means: if it already
    # exists, don't raise — just carry on.
    os.makedirs(config.DATA_PROCESSED, exist_ok=True)

    train_path = os.path.join(config.DATA_PROCESSED, 'train.csv')
    test_path  = os.path.join(config.DATA_PROCESSED, 'test.csv')

    # Default pandas encoding is utf-8, which handles Devanagari correctly.
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f'[save_processed] wrote {train_path} ({len(train_df):,} rows)')
    print(f'[save_processed] wrote {test_path}  ({len(test_df):,} rows)')


# ──────────────────────────────────────────────────────────────────────────
# PIPELINE — one-call orchestrator
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end: load → clean → dedupe → relabel → split → save.

    DONKEY: calling this is the ONLY thing Phase 2 / Phase 3 need to do
    to get their hands on preprocessed data. The function is idempotent
    — safe to call multiple times; each call deterministically rebuilds
    the processed CSVs.

    Step order (and why):
        1. load_raw         — start from the CSV, canonicalise columns.
        2. apply_cleaning   — regex scrub BEFORE dedup, so 'same text
                              but one had a URL' collapses into one row.
        3. resolve_duplicates — drop exact dupes + conflicting-label groups
                              BEFORE splitting, to prevent data leakage.
        4. normalize_labels — remap -1/0/1 → 0/1/2 ONCE the row set is
                              stable.
        5. build_train_test_split — 80/20, stratified, seeded.
        6. save_processed   — persist the CSVs to disk for Phase 2/3.

    Returns:
        (train_df, test_df) — for callers who want to keep them in memory
        right after the pipeline runs (e.g., a notebook exploring the
        distributions).
    """
    print('=' * 60)
    print('PREPROCESSING PIPELINE')
    print('=' * 60)

    df = load_raw()
    df = apply_cleaning(df)
    df = resolve_duplicates(df)
    df = normalize_labels(df)
    train_df, test_df = build_train_test_split(df)
    save_processed(train_df, test_df)

    print('=' * 60)
    print('DONE')
    print('=' * 60)
    return train_df, test_df


# `if __name__ == '__main__'` is the Python idiom for "run this block only
# when the file is executed as a script, not when it's imported". So
# `python -m src.preprocessing` (from the project root) runs the whole
# pipeline; `from src import preprocessing` in a notebook just loads the
# function definitions without side effects.
if __name__ == '__main__':
    run_pipeline()
