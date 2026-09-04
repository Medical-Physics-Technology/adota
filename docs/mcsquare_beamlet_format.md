# MCsquare per-beamlet dose: data format

Reference for implementing per-beamlet analysis inside a plan. Describes the
output of `reconstructPlanFromPencil.py` in the OpenTPS repo
(`opentps_core/opentps/core/examples/`), which recomputes the MCsquare dose of
**each individual pencil beam** of an already-published plan.

The beamlets are the MCsquare ground truth, per spot. They are the reference that
per-spot ADoTA predictions should be compared against, in the same way
`Dose.mhd` is the reference for the accumulated plan dose today.

## Where the data lives

Inside the plan directory, alongside `PlanPencil.txt` / `CT.mhd` / `Dose.mhd`:

```
<plan-dir>/beamlets_<primaries>/      e.g. beamlets_1e+05/
├── beamlets_raw_csc.npz    the dose matrix (scipy sparse)
├── spot_index.csv          one row per matrix column
├── grid.json               geometry + conventions
├── manifest.json           provenance (inputs, primaries, nnz stats, timings)
└── reconstruct.log
```

The directory name encodes primaries **per beamlet**; several can coexist for the
same plan (`beamlets_1e+05`, `beamlets_1e+07`, ...). Higher = less MC noise per
beamlet, same format. Currently available:

| Plan | Spots | Directory |
|---|---|---|
| `LUNG1-062_Publication_Plan_1` | 285 | `beamlets_1e+05` |
| `LUNG1-195_Publication_Plan_2` | 40 | `beamlets_1e+05` |

## The matrix

```python
import scipy.sparse as sp
M = sp.load_npz(plan_dir / "beamlets_1e+05" / "beamlets_raw_csc.npz")
# <class 'scipy.sparse.csc_matrix'>, dtype float32
# shape = (n_voxels, n_spots) = (Nx*Ny*Nz, n_spots)
```

- **CSC**, so `M[:, i]` (one beamlet) is a cheap slice. Do not convert to dense:
  a single column on a 501x501x303 grid is 300 MB dense, ~1 MB sparse.
- **Column `i` corresponds to row `i` of `spot_index.csv`.** This ordering is the
  order spots appear in `PlanPencil.txt`, which is exactly the order
  `src/loaders/plan_parser.py` yields them (fraction -> field -> control point ->
  spot). So a flat enumeration of the parsed plan lines up index-for-index.
- Typical density is 0.3% (~250k non-zero voxels per beamlet at 1e5 primaries);
  it grows roughly 4x per 100x primaries.

## Units

Columns hold **raw MCsquare output in eV/g/proton** — the same unit as the
`Dose.mhd` in the plan directory, and the unit `src/beamlets/dose_scaling.py`
already expects. Nothing plan-level has been folded in.

To get the physical dose of a single spot in Gy:

```
dose_gy(spot i) = M[:, i] * mu_i * opentps_rescaling_i
```

Both `mu_i` and `opentps_rescaling_i` come from `spot_index.csv`, where
`opentps_rescaling_i = computeMU2Protons(E_i) * 1.602176e-19 * 1000`, i.e. the
per-spot proton conversion for that layer's energy.

This is consistent with the existing plan-level scaling rather than a competing
convention. The plan factor used by `dose_to_gy_factor()` is the sum of the
per-spot weights:

```
delivered_protons * 1.602176e-19 * 1000  ==  sum_i (mu_i * opentps_rescaling_i)
```

So the per-beamlet weights partition the plan factor, and
`w_i / sum(w)` is spot `i`'s share of the total delivered dose — useful for
ranking which spots dominate a plan.

## Voxel ordering (read this before indexing anything)

Each column is a flattened grid in **MCsquare voxel order, with no axis flips
applied**. To recover a 3-D array:

```python
import numpy as np
col = np.asarray(M[:, i].todense()).ravel()      # or M[:, i].toarray().ravel()
vol_xyz = col.reshape(grid["gridSize"], order="F")   # (Nx, Ny, Nz)
vol_zyx = vol_xyz.transpose(2, 1, 0)                 # (Nz, Ny, Nx)
```

`vol_zyx` matches `sitk.GetArrayFromImage(...)` of `Dose.mhd` and `CT.mhd`
**voxel for voxel** — same orientation, no flips, no resampling. This is verified
empirically, not assumed.

> Note this is *not* the OpenTPS DICOM convention. `SparseBeamlets.toDoseImage()`
> additionally flips axes 0 and 1; the stored columns are the pre-flip MCsquare
> arrays. Do not apply those flips.

### Origins

`grid.json` carries two origins and using the wrong one silently misplaces the
dose by up to a voxel and mirrors it in Y:

| Field | Use with |
|---|---|
| `mcsquare_origin` | **the stored arrays** — identical to the `Offset` in `CT.mhd` / `Dose.mhd` |
| `dicom_origin` | only after flipping axes 0 and 1 (OpenTPS DICOM convention) |

Since `mcsquare_origin` equals the `Dose.mhd` offset, the simplest correct
approach is to build beamlet images with `CopyInformation()` from the plan's
existing `Dose.mhd` image, rather than assembling geometry by hand.

## `spot_index.csv`

One row per column, in matrix-column order.

| Column | Meaning |
|---|---|
| `col` | matrix column index (0-based, equals the row number) |
| `beam_idx`, `beam_name` | field this spot belongs to (0-based index) |
| `gantry_angle`, `couch_angle` | field geometry, degrees |
| `layer_idx` | control-point (energy layer) index within the beam, 0-based |
| `spot_idx` | spot index within the layer, 0-based |
| `energy_mev` | nominal layer energy |
| `x_mm`, `y_mm` | spot position in the beam's scanning plane |
| `mu` | delivered monitor units (the optimized weight from `PlanPencil.txt`) |
| `opentps_rescaling` | `computeMU2Protons(energy) * 1.602176e-19 * 1000` |

`energy_mev` is what stratifies beamlets against the 150 MeV thoracic training
limit, and `mu` is what weights them: a spot above the limit carrying negligible
MU matters far less than its presence in the layer list suggests.

## Verifying an implementation

Summing all beamlets must reproduce the reference plan dose. This is the check to
write first, because it catches ordering, unit and orientation errors at once:

```python
w = mu * opentps_rescaling                       # from spot_index.csv
total = M.dot(w).reshape(grid["gridSize"], order="F").transpose(2, 1, 0)
ref   = sitk.GetArrayFromImage(load_dose_gy(plan_dir / "Dose.mhd", plan, bdl))
# integral ratio total.sum() / ref.sum() should be ~0.98-1.00
```

Measured on the available data: **0.990** (LUNG1-062) and **0.978** (LUNG1-195),
with high-dose-voxel correlations of 0.981 and 0.953 at 1e5 primaries.

Two reasons this is not exactly 1.0, both expected:

1. **Sparse threshold.** MCsquare beamlet mode uses
   `Dose_Sparse_Threshold = 20000`, discarding the low-dose tail of every
   beamlet. This costs ~1-2% of the integral and is a systematic floor — summed
   beamlets will never exactly equal `Dose.mhd`. It is a hardcoded OpenTPS
   constant, not a per-run setting.
2. **MC noise.** Beamlets are computed at far fewer primaries per spot than the
   reference plan dose. Compare *integrals* and correlations, never `max()`:
   at low primary counts the per-beamlet maximum is a single-voxel noise
   statistic and can be off by many times without indicating any error.

## Provenance

`manifest.json` records `primaries_per_beamlet`, `rng_seed`, SHA-256 of the
`PlanPencil.txt` and `bdl.txt` consumed, the OpenTPS commit, grid sizes, nnz
statistics, timings, and the PlanPencil round-trip check (must be 0.0 for energy,
MU and XY). Tie any per-beamlet result back to these, since the same plan can
have several beamlet sets at different statistics.

MCsquare is only bit-reproducible single-threaded, so multi-threaded reruns agree
statistically, not exactly.
