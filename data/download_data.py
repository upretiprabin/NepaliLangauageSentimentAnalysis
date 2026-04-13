"""
download_data.py — grab both datasets from Kaggle into `data/raw/`.

DONKEY EXPLANATION:
-------------------
Kaggle hosts the two Nepali sentiment datasets we need. Rather than committing
the (large, potentially license-restricted) CSVs to git, we write a small
script that downloads them on demand. Anyone cloning the repo runs this once
to materialise the data locally.

Uses the official `kaggle` Python package, which looks for credentials at
`~/.kaggle/kaggle.json`. Get that file from your Kaggle account page
(Account → API → "Create New Token").

FALLBACK:
---------
If the Kaggle API is not set up, manually download the two datasets from the
Kaggle web UI and unzip them into:
    data/raw/aayamoza/
    data/raw/nepcov19tweets/
"""

# TODO (Phase 1): implement downloading via the `kaggle` Python API.
#   - kaggle.api.dataset_download_files(PRIMARY_DATASET_KAGGLE, ...)
#   - kaggle.api.dataset_download_files(SECONDARY_DATASET_KAGGLE, ...)
#   - unzip into data/raw/<dataset_slug>/
#   - print row counts + column names as a sanity check

pass
