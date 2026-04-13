"""
feature_extraction.py — turn cleaned Nepali text into numeric vectors using
TF-IDF. (Only used by the traditional-ML pipeline. NepBERTa uses its own
tokenizer, not this.)

DONKEY EXPLANATION:
-------------------
Machine-learning models cannot read words — they can only do math on numbers.
So we need a recipe for turning "यो फिल्म राम्रो छ" into something like
[0, 0.31, 0, 0, 0.78, 0, 0.45, 0, ...] — a long row of numbers that encodes
which words are in the sentence and how "important" each one is.

=== TF-IDF: Term Frequency × Inverse Document Frequency ===
Imagine reading 10,000 Nepali tweets. The word "को" (of) is in almost every
one — it's high-frequency but carries no sentiment signal. Meanwhile,
"भ्रष्टाचार" (corruption) appears in only a handful — and when it does, it
tells you A LOT about the tweet's sentiment.

TF-IDF gives every word a score:
  - TF (Term Frequency):  how often the word appears IN this document
  - IDF (Inverse Doc Freq): penalises words that appear in MANY documents

Multiply them: TF-IDF = TF × IDF. Result: "को" gets a low score everywhere,
"भ्रष्टाचार" gets a high score where it appears. The model can now tell
which words actually matter.
===

What this module will expose (Phase 2):
  - build_tfidf_vectorizer()  → configured but-unfitted TfidfVectorizer
  - fit_transform(texts)      → fit on training texts, return vectorised matrix
  - transform(texts)          → apply the *already-fit* vectorizer to new text
  - save_vectorizer(path)     → pickle it so the app can reuse it
  - load_vectorizer(path)     → load it back later
"""

# TODO (Phase 2): wrap sklearn's TfidfVectorizer using the config constants
#   (max_features=10000, ngram_range=(1,2), sublinear_tf=True).

pass
