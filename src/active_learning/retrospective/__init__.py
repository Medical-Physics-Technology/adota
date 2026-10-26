"""The retrospective active-learning benchmark on the reference HDF5 set.

The labels already exist; the loop treats them as hidden and reveals them only for
the records a strategy selects. No Monte Carlo runs, nothing is generated, and no
labelling cost is measured: the question is how the sampling strategy shapes the
training progress. Design and record: experiment EXP-0009 (the retrospective
benchmark), on top of the loop of :mod:`src.active_learning`.

- :mod:`.dataset`: the exclusion list, the frozen validation set, the cycle-0 set,
  the pool and the growth schedule.
- :mod:`.scoring`: the scorer interface and the input-only difficulty scorer over
  HDF5 records.
- :mod:`.sampling`: the strategy registry and the three strategies.
- :mod:`.validation`: the fixed evaluation subsample and the cycle metric set,
  with dR80.
- :mod:`.trainer`: one cycle of training on top of :mod:`src.training`.
- :mod:`.loop`: the cycle-0 baseline, the strategy runs, their manifests and resume.
- :mod:`.compare`: reading several runs back for the comparison figures.
"""
