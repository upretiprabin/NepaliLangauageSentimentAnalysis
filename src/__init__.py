"""
src/ — all reusable Python modules for the project live here.

Keeping logic in importable modules (not buried in notebooks) means:
  - the same preprocessing runs for BOTH models (fair comparison),
  - we can write unit tests against it,
  - the Gradio app in app/app.py can reuse it at inference time.
"""
