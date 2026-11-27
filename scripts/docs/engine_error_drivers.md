# `engine_error_drivers.py`: what drives the dose error of ADoTA and the pencil beam

Computes the ADoTA and pencil-beam doses of every stored beamlet of a beamlet-pool split,
scores both against the stored Monte Carlo, and measures which beamlet properties drive each
engine's error: per beamlet (CT features, importance, the difficulty score) and per voxel
(the voxel's HU, beamlet against material). Records: EXP-0023, CHG-0029.

Everything runs on the 2 mm model grid `(D, H, W) = (160, 30, 30)` against the export's
Monte Carlo, through the repository's one error definition (`record_metrics`). The run of
record is `/scratch/mstryja/adota_runs/paper2/engine_drivers/engine_error_drivers_20261125_153857`
(pool test split, 7,552 beamlets, 130 patients).

## Run

```bash
export TMPDIR=/scratch/mstryja/tmp
CFG=scripts/active_learning/config_engine_error_drivers.yaml

# doses, errors and panels (GPU, about 1.2 h for the test split)
uv run python scripts/active_learning/engine_error_drivers.py --config $CFG \
    --stages pba,adota,score,panels

# analyses on a finished run (CPU, minutes each)
RUN=/scratch/mstryja/adota_runs/paper2/engine_drivers/engine_error_drivers_<stamp>
uv run python scripts/active_learning/engine_error_drivers.py --config $CFG --run-dir $RUN \
    --stages analyse,ct_vs_error
for s in dataset_overview voxel_hu_error voxel_decomposition score_table; do
    uv run python scripts/active_learning/engine_error_drivers.py --config $CFG --run-dir $RUN --stages $s
done

# smoke test: the first two fields only
uv run python scripts/active_learning/engine_error_drivers.py --config $CFG \
    --stages pba,adota,score --limit-fields 2
```

`--run-dir` resumes or extends a run; a dose file is never replaced, so an interrupted
`pba` or `adota` stage continues where it stopped. `--ids-file` (CSV with `sample_id`)
restricts the dose stages to those beamlets. `--device-index` picks the GPU.

The last four stages only add files to a finished run. Run alone, each logs inside its own
folder, so the run's log is not touched.

## Stages

| stage | what | needs | writes |
|---|---|---|---|
| `pba` | pencil-beam dose per beamlet: each field (patient, gantry) rebuilt with the pool generator's `field_geometry`, every crop window checked voxel for voxel against the stored CT, the 1 mm window dose averaged to 2 mm | pool store, DICOM | `doses/pba/` |
| `adota` | ADoTA predictions through the EXP-0014 inference path, de-normalised | checkpoint | `doses/adota/` |
| `score` | GPR 2%/2mm and 3%/3mm (10% cutoff, global), MAPE, RDE, signed dR80, peak and integral ratios | doses | `tables/errors_<engine>.csv` |
| `panels` | slide panels in the `single_beam` layout (axial and sagittal columns, one row per engine, IDD with HU, lateral profiles at the Bragg peak, error table) | doses | `figures/panel_*.png` |
| `analyse` | per engine, error and site: Spearman of every feature with patient-bootstrap intervals, grouped permutation importance of a gradient-boosted model (folds grouped by patient), partial dependence, the difficulty score against the error | errors, features | `analysis/*.csv`, `analysis/summary.md` |
| `ct_vs_error` | per-beamlet input-only feature table; nine CT features against GPR 2%/2mm and MAPE, pooled, per site and energy-controlled; summary figures (correlation overview, shape per feature decile) and detailed grids in `appendix/` | errors, features | `analysis/beamlet_features.csv`, `analysis/ct_vs_error/` |
| `dataset_overview` | per-beamlet metadata, HU histograms of the model-grid crops per site (full and above -500 HU), energy, spot positions, gantry, summary | export, pool plan | `analysis/dataset_overview/` |
| `voxel_hu_error` | per voxel with Monte Carlo dose >= 0.1% of the beamlet's maximum: each engine's error against the voxel's HU, per dose band (0.1-10, 10-50, 50-100%) and site, with patient-resampling intervals | doses, `dataset_overview` | `analysis/voxel_hu_error/` |
| `voxel_decomposition` | is the voxel error about the beamlet or the material? R2 of beamlet and HU-bin effects, within-beamlet tissue contrasts, the beamlet effect against the input features | `voxel_hu_error` | `analysis/voxel_hu_error/decomposition/` |
| `score_table` | one row per beamlet: metadata, difficulty score (score v2) with decile and group contributions, both engines' errors | errors, `dataset_overview` | `analysis/score_table/beamlet_score_and_errors.csv` |

## Definitions worth knowing

- **Populations.** Doses and errors cover every stored beamlet (7,552 on the test split).
  Feature-based analyses use the 7,449 whose analytic peak lies inside the crop, where the
  difficulty features are defined.
- **MAPE** (`record_metrics`) is relative to the engine's own dose, over voxels where the
  engine dose exceeds 10% of its maximum. The voxel stage's *local* error is different: it
  divides by the Monte Carlo dose and selects voxels on it.
- **Voxel global error** `e = 100 (D_engine - D_MC) / max(D_MC)`, the normalisation of global
  gamma; signed (positive = engine too high) and absolute.
- **Path features.** `air_gap_mm` is the depth of the skin on the flux-weighted beam-axis HU,
  the skin being the first run of at least 10 mm above -500 HU (a thin couch shell is not
  taken for it); `lowdensity_path_mm` is the length at or below -500 HU between the skin and
  the analytic peak.
- **Spot positions** lie on the pool's 18 x 18 steering lattice in the isocentre plane
  (X lateral in the gantry rotation plane, Y along the patient axis); the spot figure has one
  cell per lattice position. No couch angle (0) and no spot weight exist for pool beamlets.
- **Dose medium.** MCsquare (and ADoTA, trained on it) gives dose to medium; the DoseCUDA
  kernel deposits the water depth-dose at the water-equivalent depth, i.e. dose to water. The
  pencil beam's error in bone contains this difference.

## Config

`config_engine_error_drivers.yaml`: pool store and export, the two engines (checkpoint;
DoseCUDA kernels and beam model), `panels.ids_csv`, and one block per analysis stage:
`analysis` (feature file, score contributions, grouping cut, folds, resamples, seed,
`ct_vs_error_formats`), `dataset_overview` (pool `plan_csv`, gantry bin, formats),
`voxel_hu_error` (dose cutoff, local-error floor, minimum patients per drawn bin, resamples)
and `voxel_decomposition` (resamples, seed). Figures are PNG by default; the summary stages
write PNG and PDF. Every analysis stage writes a `provenance.json` with the commit.

## Reading the results

- Start with `analysis/summary.md` (drivers and importance), then the four summary figures
  in `analysis/ct_vs_error/` and its hand-written `reading.md`.
- `analysis/voxel_hu_error/summary.md` gives the error per tissue class and dose band;
  `decomposition/summary.md` answers whether the material matters once the beamlet is known.
- Generated sentences follow fixed rules; check them against the tables before quoting.
- Every correlation is an association. Features that drive the error in the thorax move
  together, and a range error caused upstream lands in peak voxels whatever their HU.
- No Monte Carlo noise floor is applied: the lowest dose band and ADoTA's pelvic gamma
  failures cannot yet be separated from noise at 1e6 primaries.
