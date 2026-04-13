"""
src/models/ — one file per model architecture we're comparing.

Keeping them isolated means:
  - the traditional ML file has NO torch imports (stays lightweight on CPU),
  - the NepBERTa file can be copied into a Colab notebook in isolation.
"""
