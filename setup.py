"""
setup.py — optional installable-package definition for this project.

DONKEY EXPLANATION:
-------------------
In Python, the "proper" way to share code across files is to make your project
installable as a package. Once installed (with `pip install -e .`), any script
in the project — including notebooks — can say `from src.config import ...`
without having to mess with sys.path hacks.

We keep this file minimal: we're not publishing to PyPI, we just want our own
code importable from anywhere inside the project. Run `pip install -e .` once
after cloning, and you're set.
"""

from setuptools import setup, find_packages

setup(
    name="nepali_sentiment_analysis",  # The package name you'd `import` as
    version="0.1.0",                   # Bump as the project matures
    description="Nepali sentiment analysis: TF-IDF + LR vs NepBERTa comparison",
    author="Prabin Upreti",
    packages=find_packages(),          # Auto-discovers every folder with __init__.py
    python_requires=">=3.10",          # Newer f-string + typing features we rely on
)
