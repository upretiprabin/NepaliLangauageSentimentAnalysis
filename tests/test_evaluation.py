"""
test_evaluation.py — basic sanity checks for src/evaluation.py.

DONKEY EXPLANATION:
-------------------
If evaluate.py is wrong, every number in the paper is wrong. So we test it
against known inputs:
  - perfect predictions → accuracy == 1.0
  - all-wrong predictions → accuracy == 0.0
  - confusion matrix shape is (3, 3) for a 3-class problem
  - timing wrapper returns a positive number
"""

# TODO (Phase 1).
pass
