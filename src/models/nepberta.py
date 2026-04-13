"""
nepberta.py — fine-tune NepBERTa (a pre-trained Nepali transformer) for
3-class sentiment classification.

DONKEY EXPLANATION:
-------------------
=== What is NepBERTa? ===
It's a big neural network that was already trained on millions of Nepali
sentences to learn the STRUCTURE of the language — grammar, word meanings,
typical phrases. The authors haven't taught it about sentiment yet. That's
our job.

=== What is "fine-tuning"? ===
Think of hiring a fluent Nepali speaker and saying: "Here are 30,000 labelled
examples. Learn which patterns mean Positive, Negative, or Neutral." We don't
teach the model Nepali from scratch (that would need millions of examples and
weeks of GPU time). We just add a tiny "classification head" (one small
neural layer) on top of NepBERTa and nudge its weights a little with our
sentiment data.

=== Why Google Colab? ===
Fine-tuning needs a GPU. Colab gives us one for free. We prepare this code
locally, upload it to Colab as notebook 03, and run training there.

⚠️  WARNING: training will crash with "CUDA out of memory" if batch_size is
    too large for the free-tier GPU. If that happens, drop batch_size from
    16 → 8 → 4 until it fits.

What this module will expose (Phase 3):
  - class SentimentDataset(torch.utils.data.Dataset)  — wraps (text, label)
  - build_tokenizer()             → AutoTokenizer.from_pretrained(NEPBERTA_MODEL_NAME)
  - build_model(num_labels=3)     → AutoModelForSequenceClassification
  - train(...)                    → HuggingFace Trainer loop
  - predict(model, tokenizer, texts)
  - save(model, tokenizer, dir)
  - load(dir)                     → return (model, tokenizer)
"""

# TODO (Phase 3). Note: keep this file runnable-to-import even without
# torch/transformers installed (do the imports inside functions) so the
# traditional-ML side of the project stays CPU-only.

pass
