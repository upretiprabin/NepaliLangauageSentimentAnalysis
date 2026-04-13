"""
evaluation.py — compute every metric the project reports, for either model.

DONKEY EXPLANATION:
-------------------
Once a model makes predictions on the test set, we want to answer:
  "How good is it?"  — and "good" is more than one number.

We compute:
  === PERFORMANCE (how often is the model right?) ===
    - accuracy     : out of all predictions, how many were correct?
    - precision    : of the tweets the model called "Positive", how many
                     actually were? (Did it cry wolf too often?)
    - recall       : of the tweets that ARE "Positive", how many did the
                     model find? (Did it miss real cases?)
    - F1           : a balance of precision + recall into one number
    - confusion matrix : a 3×3 grid showing every
                         (true label, predicted label) pair

  === EFFICIENCY (how expensive is the model to run?)  ← our differentiator ===
    - training_time_seconds
    - inference_time_per_sample_ms
    - total_inference_time_seconds
    - model_size_mb         (on disk)
    - peak_memory_mb        (during inference, via psutil/tracemalloc)

Reporting BOTH sets matters: NepBERTa will very likely win on accuracy but
lose on efficiency. For deployment in Nepal's resource-constrained environment
(cheap servers, unreliable electricity, limited GPUs) the trade-off matters —
that's our paper's core argument.

What this module will expose (Phase 1/2):
  - compute_classification_metrics(y_true, y_pred)
  - compute_confusion_matrix(y_true, y_pred)
  - measure_inference_time(model, X_test)
  - measure_peak_memory(callable)
  - get_model_size_mb(path)
  - summarize_to_dict(...)    → a flat dict ready for a results CSV row
"""

# TODO (Phase 1): implement the core sklearn-backed metrics functions.
# TODO (Phase 2): add timing + memory helpers once we have real models to time.

pass
