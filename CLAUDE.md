# CLAUDE.md — Nepali Sentiment Analysis Project

## ⚠️ CRITICAL: Learning Mode — I Am Building This to LEARN

I am a beginner in ML/AI. I understand programming (React, Node.js, Java) but I am NEW to Python ML ecosystem, scikit-learn, transformers, NLP pipelines, and data science workflows.

### What this means for EVERY file you write:

1. **Comment every single line that isn't obvious.** Not just "what" the line does, but "WHY" we're doing it. Bad: `# fit the model`. Good: `# fit() is where the model actually learns from the training data — it finds the best weights/parameters by looking at all the examples we gave it`.

2. **Before each function, write a plain-English "donkey explanation" in the docstring.** Explain what the function does as if explaining to someone who has never seen ML code. Use analogies. Example: "This function turns raw Nepali text into numbers because ML models can't read words — they only understand numbers. Think of it like translating Nepali into 'math language'."

3. **Before each major code block (5+ lines doing one thing), add a comment block explaining the CONCEPT.** Not just the code — the concept behind it. Example before a TF-IDF section:
   ```python
   # === TF-IDF: Term Frequency - Inverse Document Frequency ===
   # Imagine you're reading 1000 tweets. The word "को" (of) appears in almost every tweet
   # — so it's not very useful for telling tweets apart. But "भ्रष्टाचार" (corruption) 
   # appears in only 5 tweets — that's a strong signal!
   # TF-IDF gives HIGH scores to words that are frequent in ONE document but rare across 
   # ALL documents. It's basically asking: "What makes THIS text special compared to others?"
   # ===
   ```

4. **When importing a library, comment what it's for and why we need it.**
   ```python
   import pandas as pd  # pandas = the Excel of Python. Handles tables/spreadsheets (DataFrames)
   from sklearn.model_selection import train_test_split  # Splits data into training set (what the model learns from) and test set (what we quiz it on)
   ```

5. **When a parameter choice matters, explain WHY that value.**
   ```python
   max_features=10000  # Only keep the 10,000 most common words. More = slower + overfitting risk. Less = might miss important words. 10K is a solid middle ground for text classification.
   ```

6. **Print intermediate results often.** After every major step, print something so I can see what just happened — shape of data, sample rows, a count, a score. Don't make me guess what's going on inside.

7. **When something could fail or be confusing, add a WARNING comment.**
   ```python
   # ⚠️ WARNING: If this crashes with "CUDA out of memory", reduce batch_size from 16 to 8
   ```

8. **Use simple variable names that describe what they hold.** Not `X_tr` — use `training_features`. Not `clf` — use `logistic_regression_model`.

### The goal:
After this project, I should be able to explain every line of code in my viva defense. If I can't explain it, it shouldn't be in the code.

## What This Project Is (The Donkey Version)

Imagine you have a huge pile of Nepali text (tweets, comments, reviews) and you want a machine to read each one and say: "This person is happy", "This person is angry", or "This person is meh." That's sentiment analysis — teaching a machine to read emotions in text.

We're building TWO different "emotion readers" and fighting them against each other:
1. **The Old-School Way (TF-IDF + Logistic Regression):** Turn words into numbers using word-frequency math, then draw a line to separate happy/sad/neutral. Simple, fast, runs on a potato.
2. **The Fancy Way (NepBERTa):** Take a pre-trained Nepali language brain (transformer model) and teach it specifically to detect sentiment. Needs a GPU, slower, but (usually) smarter.

Then we compare: which one is more accurate? Which one is faster? Which one should Nepal actually use given limited computing resources?


## Project Structure

```
nepali-sentiment-analysis/
├── CLAUDE.md                    # This file — project brain
├── README.md                    # How to run everything
├── requirements.txt             # Python dependencies (local/CPU)
├── requirements-gpu.txt         # Additional deps for Colab/GPU
├── setup.py                     # Optional package setup
│
├── data/
│   ├── raw/                     # Original downloaded datasets (gitignored)
│   │   ├── aayamoza/            # Primary dataset from Kaggle
│   │   └── nepcov19tweets/      # Secondary dataset from Kaggle
│   ├── processed/               # Cleaned, split datasets (gitignored)
│   │   ├── primary_train.csv
│   │   ├── primary_test.csv
│   │   ├── secondary_train.csv
│   │   └── secondary_test.csv
│   └── download_data.py         # Script to download datasets (needs kaggle API key)
│
├── src/
│   ├── __init__.py
│   ├── config.py                # All hyperparameters, paths, constants in ONE place
│   ├── preprocessing.py         # Text cleaning, tokenization, Devanagari handling
│   ├── feature_extraction.py    # TF-IDF vectorizer wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   ├── traditional_ml.py    # Logistic Regression pipeline (train + predict)
│   │   └── nepberta.py          # NepBERTa fine-tuning + inference wrapper
│   ├── evaluation.py            # Metrics: accuracy, precision, recall, F1, confusion matrix, timing
│   ├── comparison.py            # Head-to-head comparison logic + efficiency metrics
│   └── visualizations.py        # All plots: confusion matrices, bar charts, comparison graphs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA — understand the datasets
│   ├── 02_traditional_ml.ipynb         # Full traditional ML pipeline (runs locally)
│   ├── 03_nepberta_finetuning.ipynb    # NepBERTa fine-tuning (runs on Google Colab)
│   ├── 04_comparison.ipynb             # Side-by-side comparison + all visualizations
│   └── 05_demo_prototype.ipynb         # Interactive demo prototype
│
├── outputs/
│   ├── figures/                 # All generated plots (no titles — captions added separately)
│   ├── models/                  # Saved model files (gitignored)
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── logistic_regression.pkl
│   │   └── nepberta_finetuned/  # Saved transformer checkpoint
│   └── results/                 # CSV files with metrics, predictions
│       ├── traditional_ml_results.csv
│       ├── nepberta_results.csv
│       └── comparison_summary.csv
│
├── app/                         # Simple prototype UI
│   ├── app.py                   # Streamlit or Gradio app
│   └── utils.py                 # App helper functions
│
├── tests/                       # Basic sanity tests
│   ├── test_preprocessing.py
│   └── test_evaluation.py
│
└── .gitignore
```

## Datasets

### Primary Dataset: aayamoza/nepali-sentiment-analysis
- **Source:** Kaggle (`aayamoza/nepali-sentiment-analysis`)
- **What it is:** Publicly available collection combining YouTube comments, social media posts, and movie reviews in Nepali
- **Labels:** Positive (1), Neutral (0), Negative (-1) — 3-class
- **Script:** Devanagari
- **Usage:** Main training and evaluation dataset

### Secondary Dataset: NepCOV19Tweets
- **Source:** Kaggle (`mathew11111/nepcov19tweets`)
- **What it is:** COVID-19 related Nepali tweets collected Feb 2020 – Jan 2021
- **Labels:** Positive, Neutral, Negative — 3-class
- **Note:** The original paper (Sitaula et al., 2021) mentions 33,247 tweets collected, but the Kaggle upload may contain a smaller annotated subset (~10K+). CHECK the actual row count after download.
- **Usage:** Secondary benchmarking dataset to test generalization

### Data Split Strategy
- 80/20 train/test with **stratified sampling** (preserves class distribution)
- Use `sklearn.model_selection.train_test_split` with `stratify=y, random_state=42`
- Same split used for BOTH models — this is critical for fair comparison

## Models

### Model 1: TF-IDF + Logistic Regression (Traditional ML)
Think of it like this: TF-IDF counts how important each word is in a document relative to all documents (a word that appears everywhere like "the" gets a low score; a rare word like "horrible" gets a high score). Then Logistic Regression draws a decision boundary in that number-space to separate the classes.

- **Feature extraction:** `TfidfVectorizer` from scikit-learn
  - `max_features=10000` (cap vocabulary size)
  - `ngram_range=(1, 2)` (use single words AND two-word phrases)
  - `sublinear_tf=True` (use log-scaled term frequencies)
- **Classifier:** `LogisticRegression` from scikit-learn
  - `max_iter=1000`
  - `class_weight='balanced'` (handles imbalanced classes)
  - Hyperparameter tuning via `GridSearchCV` on C parameter: [0.01, 0.1, 1, 10]
- **Runs on:** CPU 

### Model 2: NepBERTa Fine-tuning (Transformer)
Think of it like this: NepBERTa already "speaks" Nepali — it was trained on millions of Nepali texts to understand the language's structure. We're now giving it sentiment-labelled examples and saying "learn which patterns mean positive, negative, or neutral." It's like hiring someone who already speaks Nepali fluently and then training them specifically to detect emotions.

- **Base model:** `NepBERTa/NepBERTa` from HuggingFace (RoBERTa architecture, 110M params)
- **Fine-tuning approach:** Add a classification head (3-class softmax) on top
- **Hyperparameters:**
  - Learning rate: 2e-5
  - Epochs: 3-5 (with early stopping based on validation loss)
  - Batch size: 16 (adjust based on Colab GPU memory)
  - Max sequence length: 128 tokens (tweets are short)
  - Optimizer: AdamW with weight decay 0.01
  - Warmup steps: 10% of total steps
- **Runs on:** Google Colab (free GPU) — prepare the notebook locally, run on Colab
- **Save after training:** Export model + tokenizer to `outputs/models/nepberta_finetuned/`

## Preprocessing Pipeline

Both models share the SAME preprocessing (important for fair comparison):

1. **Load raw data** → pandas DataFrame with columns: `text`, `label`
2. **Normalize labels** → Map to consistent integers: {Negative: 0, Neutral: 1, Positive: 2}
3. **Clean text:**
   - Remove URLs (regex: `https?://\S+`)
   - Remove mentions (@username)
   - Remove hashtag symbols (keep the word: #नेपाल → नेपाल)
   - Remove extra whitespace
   - Remove non-Devanagari/non-punctuation characters (preserve Nepali text)
   - Strip leading/trailing whitespace
4. **DO NOT** lowercase (Devanagari doesn't have case)
5. **DO NOT** remove stop words for TF-IDF initially (let TF-IDF handle it via its own weighting — can experiment with removal later)
6. **For NepBERTa:** Use the model's own tokenizer (`AutoTokenizer.from_pretrained("NepBERTa/NepBERTa")`) — do NOT manually tokenize

## Evaluation Metrics

Both models evaluated on the EXACT SAME test set using:

### Performance Metrics
- **Accuracy** (overall correct predictions / total)
- **Precision** (per-class and macro-averaged)
- **Recall** (per-class and macro-averaged)
- **F1-Score** (per-class, macro-averaged, and weighted)
- **Confusion Matrix** (3x3 heatmap)
- **Classification Report** (sklearn's `classification_report`)

### Efficiency Metrics (THIS IS OUR DIFFERENTIATOR — most Nepali SA papers skip this)
- **Training time** (seconds)
- **Inference time per sample** (milliseconds) — average over test set
- **Total inference time on test set** (seconds)
- **Model size on disk** (MB)
- **Peak memory usage during inference** (MB) — use `tracemalloc` or `psutil`

### Visualization Requirements
- All plots: clean, Add suitable titles
- Use matplotlib + seaborn
- Save as PNG at 300 DPI to `outputs/figures/`
- Required plots:
  1. `class_distribution_primary.png` — Bar chart of label distribution (primary dataset)
  2. `class_distribution_secondary.png` — Same for secondary dataset
  3. `text_length_distribution.png` — Histogram of text lengths
  4. `confusion_matrix_lr.png` — Confusion matrix for Logistic Regression
  5. `confusion_matrix_nepberta.png` — Confusion matrix for NepBERTa
  6. `accuracy_comparison.png` — Grouped bar chart comparing all metrics
  7. `efficiency_comparison.png` — Bar chart comparing inference time + memory
  8. `training_loss_nepberta.png` — Training/validation loss curve for NepBERTa
  9. `sample_predictions.png` — Table-style visualization of example predictions from both models

## Build Phases

### Phase 1: Project Skeleton + Data Pipeline (Local, Claude Code)
**Goal:** Set up the project, download data, explore it, build preprocessing.
- Create the full directory structure
- Write `config.py` with all constants
- Write `download_data.py` (instructions for manual Kaggle download as fallback)
- Write `preprocessing.py` with the full cleaning pipeline
- Build `01_data_exploration.ipynb`:
  - Load both datasets
  - Print shape, columns, sample rows
  - Plot class distributions
  - Plot text length distributions
  - Check for nulls, duplicates
  - Report basic stats
- Write `evaluation.py` with all metric computation functions
- Write `visualizations.py` with all plot functions (no titles)
- Write basic tests

### Phase 2: Traditional ML Pipeline (Local, Claude Code)
**Goal:** Build, train, evaluate Logistic Regression.
- Write `feature_extraction.py` (TF-IDF wrapper)
- Write `models/traditional_ml.py`:
  - Train TF-IDF + Logistic Regression
  - GridSearchCV for hyperparameter tuning
  - Save trained model (pickle)
  - Prediction function
- Build `02_traditional_ml.ipynb`:
  - Load preprocessed data
  - Train model
  - Evaluate on test set
  - Generate all metrics + confusion matrix
  - Time training and inference
  - Save results to CSV
- Run on primary dataset first, then secondary dataset

### Phase 3: NepBERTa Pipeline (Prepare locally, RUN on Colab)
**Goal:** Fine-tune NepBERTa on sentiment data.
- Write `models/nepberta.py`:
  - Dataset class (PyTorch)
  - Fine-tuning training loop
  - Inference function
  - Model save/load
- Build `03_nepberta_finetuning.ipynb`:
  - Colab-ready (includes `!pip install` cells)
  - Mount Google Drive for data/model storage
  - Load preprocessed data
  - Fine-tune NepBERTa
  - Evaluate on test set
  - Generate all metrics
  - Time training and inference
  - Save model checkpoint + results
- **IMPORTANT:** This notebook must be self-contained for Colab (duplicate necessary preprocessing code or import from uploaded src/)

### Phase 4: Comparison + Visualizations (Local)
**Goal:** Head-to-head comparison of both models.
- Write `comparison.py`:
  - Load results from both models
  - Compute comparison metrics
  - Statistical comparison if applicable
- Build `04_comparison.ipynb`:
  - Side-by-side metrics table
  - All comparison visualizations
  - Efficiency comparison
  - Sample predictions from both models
  - Summary findings

### Phase 5: Prototype App (Local)
**Goal:** Simple demo UI that takes Nepali text input and shows sentiment from both models.
- Write `app/app.py` using Gradio (lighter than Streamlit):
  - Text input box
  - "Analyze" button
  - Shows: predicted label + confidence from BOTH models
  - Shows inference time for each model
- Keep it simple — this is for the demo video, not production
- Must work offline with saved models

### Phase 6: Documentation + Polish
- Write `README.md` with full setup instructions
- Ensure `requirements.txt` is complete
- Clean up notebooks (remove debug cells, add markdown explanations)
- Verify all outputs/figures are generated

## Tech Stack

### Core Dependencies (requirements.txt)
```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
joblib>=1.3
psutil>=5.9
tqdm>=4.65
```

### GPU Dependencies (requirements-gpu.txt — for Colab)
```
torch>=2.0
transformers>=4.30
datasets>=2.14
accelerate>=0.21
```

### App Dependencies
```
gradio>=4.0
```

## Coding Conventions

- **Python 3.10+**
- **Type hints** on all function signatures
- **Docstrings** on all public functions (Google style)
- **No hardcoded paths** — everything goes through `config.py`
- **Random seed = 42 everywhere** (reproducibility)
- **Print progress** during long operations (use tqdm for loops)
- **Save intermediate results** as CSVs — don't rely on keeping notebooks running
- **Figures:** No titles, clean axes labels, 300 DPI, tight_layout, save to outputs/figures/

## Config.py Reference

```python
# All paths, hyperparameters, and constants live here
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = os.path.join(PROJECT_ROOT, '..', 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, '..', 'data', 'processed')
OUTPUTS = os.path.join(PROJECT_ROOT, '..', 'outputs')
FIGURES = os.path.join(OUTPUTS, 'figures')
MODELS = os.path.join(OUTPUTS, 'models')
RESULTS = os.path.join(OUTPUTS, 'results')

# Dataset
PRIMARY_DATASET_KAGGLE = 'aayamoza/nepali-sentiment-analysis'
SECONDARY_DATASET_KAGGLE = 'mathew11111/nepcov19tweets'

# Labels
LABEL_MAP = {-1: 0, 0: 1, 1: 2}  # Negative=0, Neutral=1, Positive=2
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']

# Common
RANDOM_STATE = 42
TEST_SIZE = 0.2

# TF-IDF + Logistic Regression
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
LR_C_VALUES = [0.01, 0.1, 1, 10]
LR_MAX_ITER = 1000

# NepBERTa
NEPBERTA_MODEL_NAME = 'NepBERTa/NepBERTa'
NEPBERTA_MAX_LENGTH = 128
NEPBERTA_BATCH_SIZE = 16
NEPBERTA_LEARNING_RATE = 2e-5
NEPBERTA_EPOCHS = 5
NEPBERTA_WARMUP_RATIO = 0.1
NEPBERTA_WEIGHT_DECAY = 0.01

# Visualization
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
```

## Known Gotchas & Warnings

1. **NepBERTa tokenizer:** It's RoBERTa-based, so use `AutoTokenizer` not `BertTokenizer`. The tokenizer handles Devanagari natively.
2. **Label encoding:** The aayamoza dataset might use -1/0/1 or different string labels. CHECK the actual format after download and normalize in preprocessing.
3. **NepCOV19Tweets size:** Proposal says 33,247 but Kaggle listing says 10K+. Download and verify. If it's much smaller, note this in the paper as a limitation.
4. **Class imbalance:** Nepali sentiment datasets tend to be imbalanced (more neutral). Use `class_weight='balanced'` for LR and weighted loss for NepBERTa.
5. **Devanagari stop words:** There's no standard Nepali stop words list in NLTK/spaCy. Either skip stop word removal (let TF-IDF handle via IDF weights) or use a custom list from the NepBERTa project if available.
6. **Colab session timeout:** Save model checkpoints frequently during NepBERTa training. Use Google Drive mounting.
7. **Mixed script text:** Some Nepali social media text mixes Devanagari with Romanized Nepali or English. Decide how to handle (our preprocessing keeps only Devanagari — document this as a limitation).
8. **Memory on Mac:** NepBERTa inference (not training) should work on CPU for the prototype. Expect ~5-10 seconds per prediction vs milliseconds for LR.

## What Success Looks Like

The finished project should:
- Have a clean, runnable codebase that someone can clone and reproduce results
- Show clear comparative results between traditional ML and transformer approaches
- Include efficiency metrics that most Nepali SA papers don't provide
- Have a working prototype that accepts Nepali text and outputs sentiment
- Generate all figures needed for the research paper
- Be documented well enough to defend in a viva

## For the Research Paper (Not Code — Just Context)

The paper needs to argue: "We built a practical system, compared two approaches fairly, and found [X]. For Nepal's resource-constrained environment, [Y] is the better choice because [Z]."

Key narrative points:
- Gap: No existing study provides BOTH accuracy AND efficiency comparison for Nepali SA
- Gap: No publicly available working prototype for Nepali sentiment monitoring
- Contribution 1: Fair head-to-head comparison (same data, same splits, same metrics)
- Contribution 2: Efficiency analysis for practical deployment
- Contribution 3: Working prototype
