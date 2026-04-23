# Nepali Sentiment Analysis — Traditional ML vs NepBERTa

A head-to-head comparison of two approaches for classifying the sentiment of
Nepali (Devanagari-script) text into **Negative / Neutral / Positive**:

1. **Traditional ML** — TF-IDF + Logistic Regression (runs on any laptop CPU).
2. **Transformer** — NepBERTa fine-tuned for 3-class classification (runs on
   Google Colab GPU).

This project goes beyond accuracy: we also measure **training time, inference
latency, model size, and memory usage** — the "efficiency" side of the story
that most Nepali-sentiment papers leave out.

---

## Project status

Phase 1 (skeleton + data pipeline) — in progress. See `CLAUDE.md` for the
full phased plan.

## Quick start (to be filled in as phases complete)

```bash
# 1. Create + activate a virtualenv (Python 3.10+)
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Download datasets from Kaggle — see data/download_data.py
python data/download_data.py

# 4. Explore the data
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Repository layout

See `CLAUDE.md` for the full directory tree + rationale behind every folder.

## Dataset

| Name     | Kaggle slug                          | Rows (raw) | Labels     |
| -------- | ------------------------------------ | ---------- | ---------- |
| aayamoza | `aayamoza/nepali-sentiment-analysis` | ~35,789    | -1 / 0 / 1 |

> Scope note: this project deliberately works on a single dataset. Cross-dataset
> generalisation is future work.
