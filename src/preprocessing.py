"""
preprocessing.py — turn raw Nepali CSV rows into clean (text, label) pairs,
ready to feed into either the TF-IDF pipeline or the NepBERTa tokenizer.

DONKEY EXPLANATION:
-------------------
Raw Nepali social-media text is messy. It contains URLs ("https://..."),
@mentions, hashtag symbols, emojis, mixed English+Devanagari, repeated
whitespace, and sometimes garbage characters from broken scraping. If we feed
this junk directly to a model, it wastes "learning capacity" on noise instead
of real sentiment signal. So we clean it first — strip the URL boilerplate,
keep the Devanagari words, normalise whitespace.

CRITICAL: both the TF-IDF pipeline AND NepBERTa must receive text that went
through the SAME cleaner. Otherwise their accuracy scores aren't really
comparable — one of them might have had an easier (or harder) version of the
data. Same cleaner → fair fight.

What this module will expose (once implemented in Phase 1):
  - load_primary_dataset()         → (texts, labels) from aayamoza.csv
  - load_secondary_dataset()       → (texts, labels) from nepcov19tweets.csv
  - normalize_label(raw)           → -1/0/1 → 0/1/2 (Negative/Neutral/Positive)
  - clean_text(s)                  → single-string cleaner (URLs, mentions, ws)
  - build_train_test_split(...)    → 80/20 stratified split w/ fixed seed
  - save_processed(df, path)       → write CSV into data/processed/
"""

# TODO (Phase 1):
#   1. Implement `clean_text` with regex for URLs / @mentions / # symbols /
#      whitespace collapse. Do NOT lowercase (Devanagari has no case).
#   2. Write the two `load_*` functions, handling the quirks of each CSV
#      (aayamoza has an index column; nepcov19tweets has malformed rows to
#      filter out — see README for the gotcha).
#   3. Implement label normalisation via config.LABEL_MAP.
#   4. Wrap sklearn's train_test_split with stratify=y, random_state=42.

pass
