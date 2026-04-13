"""
test_preprocessing.py — basic sanity checks for src/preprocessing.py.

DONKEY EXPLANATION:
-------------------
Tests here are NOT about formal correctness proofs. They exist to catch the
kind of bug that silently ruins an ML experiment:
  - cleaner that accidentally drops ALL Devanagari,
  - label map that silently turns "Neutral" into None,
  - train/test split that has different class distributions.

Run with:   pytest tests/
"""

# TODO (Phase 1): write tests for
#   - clean_text strips URLs but preserves Devanagari
#   - clean_text preserves the hashtag word (strips only the '#')
#   - normalize_label maps -1/0/1 → 0/1/2 correctly
#   - train/test split is stratified (class ratios match within ±1%)
#   - loading the primary CSV gives the expected row count (~35,789)

pass
