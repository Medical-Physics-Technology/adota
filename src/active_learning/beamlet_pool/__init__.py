"""The labelled beamlet pool: new patients, their split, and batched Monte Carlo.

- :mod:`.registry`: screens every CT of the configured collections, finds the
  ones the training set already used (by their 1 mm grid against the training
  HDF5's scans), applies the claims of other experiments, and splits the rest by
  patient into train / validation / test.
- :mod:`.planning`: draws candidate beamlets per patient (random field angles,
  steering on the lattice, energy uniform in a range), keeps only those whose
  analytic Bragg peak stops inside the patient and the crop, and assigns them to
  batches.
- :mod:`.store`: packs a simulated batch into compressed HDF5 shards (CT and dose
  at 1 mm, the flux recomputed from the metadata on load), checks every beamlet,
  and moves the shards to their final storage.
- :mod:`.batches`: the batch loop, resumable and shardable over processes.
"""
