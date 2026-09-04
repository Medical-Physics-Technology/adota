# Energy x gantry smoke test (102.6 / 135 MeV, 3 random gantries)

A small end-to-end pass through the whole MC -> ADoTA -> gamma -> paper-figure
chain, on two **non-nominal** energies and several random field angles per patient.
It is the cheap rehearsal for a full `grid_n: 18` run: same code path, same panels,
1/65th of the beamlets.

| | |
|---|---|
| patients | 3 thoracic (Lung-PET-CT-Dx) + 3 abdominal (StageII-Colorectal) |
| energies | 102.6 and 135 MeV (both off the BDL's 10-MeV nominal grid; MCsquare interpolates) |
| field angles | 3 uniform-random gantries per patient, **shared** by both energies |
| beamlet angles | 5: the 4 corners `(+-2, +-2)` + the centre `(0, 0)` |
| primaries | 1e6 per beamlet |
| total | 6 x 3 x 2 x 5 = 180 beamlets |

Configs: `scripts/mc/config_energy_gantry_smoke.yaml` (generation),
`scripts/mc/config_plot_energy_gantry_smoke.yaml` (inference + gamma + panels).
Everything runs on the shared spine documented in
[patient_set_generation.md](patient_set_generation.md) -- `n_gantry`, the explicit
`angles` list and the z-coverage check are described there.

## Patient selection is geometry-constrained

`theta_x` steers the beamlet along the CT's slice axis (90.2 mm at 2 deg), so the
`+-2` deg corners need a CT covering **>= 240 mm in z**; shorter scans lose the
corners to the `roi_out_of_bounds` QA gate. Many Lung-PET-CT-Dx diagnostic CTs
cover only 160-230 mm, so the thoracic patients were screened with
`--check-geometry` (`run_logs/energy_gantry_smoke_geomscreen.log`) and chosen for
coverage: G0037/G0049/G0056 (400/328/337 mm) and CT-222/225/230 (290/315/465 mm),
all from the leakage-safe `selection: last` tail and unused by the earlier
robustness runs.

## Depth budget at oblique gantries

This run is what exposed the beam-axis placement bug in the random-gantry path
(fixed; see [patient_set_generation.md](patient_set_generation.md)). An oblique beam
crosses a square FOV diagonally, so the expanded canonical grid left 150-190 mm of
air before the patient and cut the Bragg peak off the 320 mm crop -- deposition
ratios of 0.5-0.9 and WET of 42-125 mm, against 0-70 mm of air and 155-288 mm of WET
in the paper's gantry-90 crops. The grid is now trimmed back along the beam axis
with the entrance face 20 mm before the patient.

Beamlets that still do not fit -- a `+-2` deg beamlet is steered 90 mm along the
slice axis and can end up in lung or leave the body altogether -- are dropped by
`min_deposition_ratio: 0.95` and left blank in the panel rather than scored as model
error: a truncated crop is not a fair ground truth. Expect a few empty corner cells,
mostly on scans that are long in z.

## Run

```bash
# 0. screen the selection (loads CTs, no MC)
uv run python scripts/mc/generate_patient_set.py \
    --config scripts/mc/config_energy_gantry_smoke.yaml --check-geometry

# 1. ground truth (MCsquare); resumable, ~180 beamlets
uv run python scripts/mc/generate_patient_set.py \
    --config scripts/mc/config_energy_gantry_smoke.yaml

# 2. ADoTA inference + gamma + the paper's GPR panels
uv run python scripts/mc/plot_angle_robustness.py \
    --config scripts/mc/config_plot_energy_gantry_smoke.yaml
```

Step 1 writes `energy_gantry_smoke_{anatomy}_{patient}_e{E}_g{G}_v1/` under
`/scratch/mstryja/DoTA_dataset_v2` (36 dirs) plus an `energy_gantry_smoke_summary.json`
carrying the per-patient gantries, z extent and per-block saved/skipped counts.

```bash
# 3. the per-site mean panels (one per anatomy x energy), re-rendered from the
#    grids step 2 saved -- no GPU, no gamma, ~10 s
uv run python scripts/mc/plot_angle_robustness.py \
    --config scripts/mc/config_plot_energy_gantry_smoke_aggregate.yaml
```

Step 2 is the same figure as the paper panels -- GPR over (beamlet angle X, Y),
shared colour scale per criterion, gamma at (2%/2mm/10%) and (1%/3mm/0.1%) -- on the
3x3 lattice: 5 filled cells, the 4 edge-midpoints empty. One panel per
(patient, energy, gantry) = 36 panels, which is why the plot config sets
`distinguish_gantry: true`.

Step 3 averages each cell over a site's patients *and* gantries, giving one panel per
(anatomy, energy), named `{site}_aggregate{n}_e{E}_{criterion}`. It reads the
per-cell grids step 2 wrote (`grids_glob`) rather than re-running inference and
gamma -- the GPRs *are* the result of those stages, so re-rendering at a different
scale, criterion subset or mode is exact and takes seconds instead of an hour.

## Result (run of 2026-09-01)

Generation `run_logs/energy_gantry_smoke_gen_20260901_115506.log`, figures
`run_logs/energy_gantry_smoke_plot_20260901_124954.log`.

| | saved / attempted |
|---|---|
| Lung_Dx-G0037 (z 400 mm) | 6/15 at 102.6, 3/15 at 135 |
| Lung_Dx-G0049 (z 328 mm) | 15/15, 13/15 |
| Lung_Dx-G0056 (z 337 mm) | 15/15, 8/15 |
| StageII-Colorectal-CT-222/225/230 | 15/15 each at 102.6; 13/15, 15/15, 15/15 at 135 |
| **total** | **148 / 180** |

The QA drops are concentrated in G0037, a lung-heavy 400 mm thorax where an oblique
135 MeV beamlet deposits 8-33 % of its dose past the 320 mm ROI; one of its blocks
(g137.9, 135 MeV) kept no beamlet at all and is skipped by the plot script. 23 of
the 35 panels carry all 5 cells.

Mean gamma pass rate per panel, Γ(2 %, 2 mm, 10 %):

| site | 102.6 MeV | 135 MeV |
|---|---|---|
| abdominal | 97.4 % (95.3-98.3) | 96.8 % (95.6-97.7) |
| thoracic | 94.7 % (85.0-97.0) | 95.1 % (89.6-96.2) |

Γ(1 %, 3 mm, 0.1 %) averages 83.8 % over the 35 panels. The four per-site mean
panels give Γ(2 %, 2 mm, 10 %) of 97.4 / 96.8 % (abdominal, 102.6 / 135 MeV) and
95.6 / 95.5 % (thoracic), and Γ(1 %, 3 mm, 0.1 %) of 89.1 / 85.4 % and 83.4 / 78.8 %. Both non-nominal energies
and the random field angles therefore behave like the paper's nominal-energy,
gantry-90 panels -- the chain is ready to scale up.

## Scaling up: the full run

`scripts/mc/config_energy_gantry_full.yaml` is the same experiment on the full
18x18 lattice -- same six patients, same seeded gantries, same two energies,
11 664 beamlets -- at **1e6 primaries** for speed, with `min_deposition_ratio: 0.7`
so most cells are kept. Detached:

```bash
setsid nohup uv run python \
    scripts/mc/generate_patient_set.py \
    --config scripts/mc/config_energy_gantry_full.yaml \
    > run_logs/energy_gantry_full_gen_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
```

Budget 18.1 s per beamlet (measured over the smoke run on these same patients), so
**~2.4 days** wall clock and **~185 GB** of ct/ds/flux (plus ~52 GB of predictions)
against 537 GB free. `overwrite: false` makes it resumable: the same command picks
up where a stopped run left off. Figures afterwards come from
`config_plot_energy_gantry_full.yaml` and its `_aggregate` companion.

### What 1e6 costs in the reported number

The MC ground truth carries ~1.71 % per-voxel statistical uncertainty at 1e6 against
~0.61 % at 1e7, and gamma charges that noise to the model. Measured on one field (49
spots, StageII-Colorectal-CT-222, gantry 114.6 deg, 102.6 MeV) with **identical**
ADoTA predictions scored against both ground truths:

| | vs 1e6 GT | vs 1e7 GT | offset |
|---|---|---|---|
| Gamma(2 %, 2 mm, 10 %) | 98.40 % | 98.82 % | **+0.42 pp** |
| Gamma(1 %, 3 mm, 0.1 %) | 89.90 % | 92.10 % | **+2.20 pp** |

Those offsets come from a single abdominal field and are not established for
thoracic anatomy or other energies; treat them as an estimate of the noise penalty,
not a calibrated correction.

The looser gate is worth reading twice: a cell admitted at 0.7 has up to 30 % of its
dose outside the 320 mm crop, so its ground truth is distally truncated and a low GPR
there may be data coverage rather than model error. It only affects the extreme
+-2 deg ring -- which is exactly what the smoke test sampled, while the 18x18 sweep
is mostly interior angles.
