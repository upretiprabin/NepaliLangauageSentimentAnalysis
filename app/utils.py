"""
utils.py — small helpers for app.py: load-once caching, timing wrappers,
output formatting.

DONKEY EXPLANATION:
-------------------
Gradio calls our prediction function on every user click. Loading a 400MB
NepBERTa model from disk on every click would take 5+ seconds of waiting
before inference even starts. Instead, we load it ONCE when the app boots
and cache it in a module-level variable. This file holds those cached
loaders plus any formatting helpers the UI needs.
"""

# TODO (Phase 5):
#   - get_lr_pipeline()       → cached joblib.load of LR
#   - get_nepberta_model()    → cached load of NepBERTa + tokenizer
#   - time_prediction(fn, x)  → returns (result, elapsed_ms)
#   - format_confidence(prob) → "73.2%"
pass
