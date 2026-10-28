"""Hyperparameter optimisation for the EXP-0014 member ladder: one search per
sequence mixer, the same budget for each, run on R only.

- :mod:`.space`: the search space, scrambled-Sobol trials under a parameter cap.
- :mod:`.halving`: synchronous successive halving (budgets, promotions).
- :mod:`.trial`: one trial's config, a training segment to a rung budget, and
  the full metric set for a finalist.
- :mod:`.selection`: the results table and the members chosen from it.

The driver is ``scripts/active_learning/ensemble_hpo.py``.
"""
