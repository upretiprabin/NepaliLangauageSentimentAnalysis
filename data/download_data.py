"""
download_data.py — grab the primary dataset from Kaggle into `data/raw/`.

DONKEY EXPLANATION:
-------------------
Kaggle hosts the Nepali sentiment dataset we need. Rather than committing
the (large, potentially license-restricted) CSV to git, we write a small
script that downloads it on demand. Anyone cloning the repo runs this once
to materialise the data locally.

Uses the official `kaggle` Python package, which looks for credentials at
`~/.kaggle/kaggle.json`. Get that file from your Kaggle account page
(Account → API → "Create New Token").

FALLBACK:
---------
If the Kaggle API is not set up, manually download the dataset from the
Kaggle web UI and unzip it into:
    data/raw/aayamoza/
"""

# TODO (Phase 1): implement downloading via the `kaggle` Python API.
#   - kaggle.api.dataset_download_files(PRIMARY_DATASET_KAGGLE, ...)
#   - unzip into data/raw/<dataset_slug>/
#   - print row count + column names as a sanity check

pass
