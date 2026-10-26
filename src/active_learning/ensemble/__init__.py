"""Ensembles of independently trained ADoTA models, and how far they agree on
which beamlets are hard (EXP-0014: model-independence of the difficulty score).

- :mod:`.members`: one trained model per member, its per-record errors on an
  evaluation set (reused from a CSV or inferred from the checkpoint) and the
  ids it was trained on.
- :mod:`.agreement`: within-cell rank agreement per pair of members, the pair
  level read from the training ids, and a bootstrap over scans.
"""
