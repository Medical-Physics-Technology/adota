"""DoseCUDA pencil-beam algorithm (PBA), commissioned against MCsquare.

DoseCUDA (https://github.com/jhu-som-radiation-oncology/DoseCUDA) is a GPU
pencil-beam dose engine whose proton model is entirely table-driven: per energy an
integrated depth dose IDD(z_w), a multiple-Coulomb-scattering width sigma_mcs(z_w),
a quadratic in-air spot width sigma_air(distance to source) and R80, plus the
machine's virtual source distances and an HU -> relative-stopping-power curve. Its
nuclear halo is hard-coded in the kernel.

This package derives all of those tables from MCsquare itself (the engine that
produced ADoTA's training data), so the three engines -- MC, PBA, ADoTA -- describe
the same beamlet:

* :mod:`~src.pencil_beam.rlsp` -- HU -> RLSP from MCsquare's own scanner and
  material files.
* :mod:`~src.pencil_beam.beam_model` -- the commissioned model, read/written in
  DoseCUDA's lookup-table layout, with interpolation to arbitrary energies.
* :mod:`~src.pencil_beam.commissioning` -- MCsquare water-phantom runs and the fits
  that turn them into the tables.
* :mod:`~src.pencil_beam.kernels` / :mod:`~src.pencil_beam.engine` -- the compiled
  CUDA kernels and a CT-grid dose engine in the MC beamlet frame.
* :mod:`~src.pencil_beam.comparison` -- MC vs PBA vs ADoTA for one beamlet on one
  CT grid.

DoseCUDA is an external install, like MCsquare: see ``docs/pencil_beam.md``.
"""
