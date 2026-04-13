"""
app.py — a tiny Gradio web UI for the demo video / viva.

DONKEY EXPLANATION:
-------------------
The app has one job:
    1. User pastes some Nepali text into a text box.
    2. The page shows:
        - Logistic Regression's prediction + confidence + inference time (ms)
        - NepBERTa's   prediction + confidence + inference time (ms)

It loads both saved models once at startup and reuses them for every request.
This is perfect for a live demo: the reviewer can see the SAME sentence
produce two different answers (or not!) and read the efficiency numbers.

⚠️  WARNING: loading NepBERTa on a laptop CPU can take 5-10 seconds per
    prediction. The UI should stay usable but the demo video may need a
    small "thinking…" spinner to hide the latency.
"""

# TODO (Phase 5): build the Gradio interface.
#   - gr.Interface or gr.Blocks
#   - inputs=[gr.Textbox(lines=3)]
#   - outputs=[gr.Label, gr.Label]  (one per model)
#   - show inference time under each
#   - demo.launch()

pass
