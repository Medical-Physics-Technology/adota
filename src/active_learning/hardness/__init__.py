"""What makes a beamlet hard, per dose engine (EXP-0014, the C1 model-independence test).

- :mod:`.cells`: within-cell ranks and Spearman correlations, the scan bootstrap
  and the clustering of near-duplicate metrics.
- :mod:`.fit`: the difficulty score refit per engine (percentile metrics, ridge),
  out of fold by scan, with permutation importance and cross transfer.
- :mod:`.engines`: the dose engines as config entries, and their per-record
  errors on saved beamlet directories.
- :mod:`.tables`: the two analysis tables (HPO validation records, engine
  comparison beamlets), physics metrics joined with every engine's errors.
- :mod:`.analysis`: the seven analysis steps on one table.
"""
