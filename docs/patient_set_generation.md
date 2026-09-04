# Multi-patient MC generation with random gantry

General training-set expansion: generate the same beamlet grid (angles x energies)
as the robustness runs for N + M patients across anatomies, each rotated by a
**random gantry angle**. Driver: `scripts/mc/generate_patient_set.py` +
`config_patient_set.yaml`. Reuses the whole generation spine
(`src/mc_generation/robustness.py`); the only new logic is the CT rotation.

## How random gantry works (rotate-to-canonical)

`extract_beamlet_roi` extracts the ROI along a **fixed** axis and assumes the CT is
already gantry-aligned; the ADoTA model likewise consumes a canonical beam's-eye
frame (gantry is metadata, not a geometric model input). So a gantry angle `G` is
realised by:

1. draw `G` per patient (seeded, reproducible) -- `resolve_gantry`;
2. rotate the CT into the canonical frame by `A = -(G - 90)` about the isocenter,
   grid-**expanded** so no anatomy is clipped (`rotate_ct_around_isocenter`,
   `expand=True`);
3. trim the expanded grid back along the **beam axis** to the unrotated extent,
   with the entrance face `beam_entrance_standoff_mm` (default 20) before the
   patient (`beam_entrance_index` + `trim_beam_axis`);
4. simulate at the canonical **90 deg** on that grid;
5. extract axis-aligned exactly as at gantry 90.

Step 3 is not cosmetic. `extract_beamlet_roi` measures the ROI's 320 mm depth from
the grid's `x = 0` face, and an oblique beam crosses a square FOV diagonally, so the
expanded grid left 150-190 mm of air in front of the patient -- against 0-70 mm in
the gantry-90 data the model was trained on. That pushed the Bragg peak out of the
crop: deposition ratios of 0.5-0.9 and 42-125 mm of WET across the crop, where the
paper's gantry-90 crops carry 155-288 mm. The trim restores the original grid
extent (not the 320 mm crop), so the grid still reaches past the ROI and
`min_deposition_ratio` keeps measuring escaped dose instead of reading 1 by
construction. The lateral (y) expansion is kept, and the entrance is looked for only
where the sweep's beamlets actually pass (`sweep_lateral_half_extents`), so a couch
rail cannot define it.

Rotating about the grid-centre isocenter keeps it the centre of the expanded grid,
so the MC and extraction isocenters stay mutually consistent (unit-tested). The
rotation is in the axial x-y plane only; the z (slice) extent is unchanged, so a CT
large enough for the gantry-90 runs is large enough here.

Provenance per beamlet records both angles:
`gantry_angle` = physical field angle `G` (model metadata), `mc_gantry_angle` = 90
(actually simulated), `ct_rotation_deg` = `A`.

## Gantry modes (`robustness.gantry_mode`)

- `fixed` -- `gantry_value` (90 = no rotation; the robustness/phantom runs).
- `uniform_random` -- uniform on `[gantry_min, gantry_max)`, seeded per patient UID.
- `bimodal_random` -- two lobes `gantry_ranges` (seeded).

## Several gantries per patient (`n_gantry`)

`n_gantry: k` draws `k` field angles per patient from the same patient-seeded
stream (`resolve_gantries`). The angles are **shared across the energies** -- the
CT is rotated into the beam's-eye frame once per field angle and reused for every
energy -- so the energies of one patient are comparable at identical geometry. The
first draw is exactly the single-gantry `resolve_gantry` result, so an
`n_gantry: 1` rerun reproduces earlier runs. Draws whose 0.1-deg tag collides are
discarded, keeping the per-gantry output dirs distinct. `fixed` mode has nothing to
sample and always yields the one configured angle.

Output dirs gain a `_g{angle}` segment **only** when `n_gantry > 1`:
`{prefix}_{anatomy}_{patient}_e{E}_g{G}_v{ver}/`. Fractional energies are written
losslessly (`102.6 -> e102p6`); integer energies keep the historical `e140`.
`plot_angle_robustness.py` needs `distinguish_gantry: true` to carry the same
segment into panel filenames (it refuses to overwrite colliding panels). To re-render
those panels later -- a different colour scale, criterion subset, or
`mode: aggregate` -- point its `grids_glob` at the `grids/*_grids.npz` the first pass
saved: the per-cell GPRs are the whole output of inference and gamma, so the rerun is
exact and costs seconds.

## Sparse angle sweeps (`angles`)

Instead of the full `grid_n x grid_n` sweep, `angles: [[tx, ty], ...]` runs an
explicit list -- e.g. the 4 corners + centre of the +-2 deg square as a cheap smoke
test. Each entry must land **exactly** on the `grid_n` lattice (a 3x3 lattice over
[-2, 2] holds the corners and the centre); `build_angle_grid` raises otherwise.
The angles keep their lattice `grid_index`, so a sparse sweep drops straight into
the usual GPR panel with the unvisited cells left NaN.

## CT z coverage limits the theta_x sweep

`theta_x` steers the beamlet along the CT's **slice** axis: at the isocenter plane
the ray sits `d_smy * tan(theta_x)` (= 90.2 mm at 2 deg, HPTC BDL) away from the
isocenter, and the ROI adds half of its 60-voxel lateral window. A CT shorter than
twice that (240 mm for a +-2 deg sweep) cannot hold the outer beamlets: they are
dropped by the `roi_out_of_bounds` QA gate **after** their MC has been paid for.
`theta_y` costs nothing in z (it steers along y, which is the in-plane axis with
several hundred mm of grid).

Screen a selection before running -- this loads each CT but simulates nothing:

```bash
uv run python scripts/mc/generate_patient_set.py --config <cfg> --check-geometry
#   Lung_Dx-G0035   thoracic   z= 218.0 mm  max|theta_x|=1.75 deg  TOO SHORT
#   Lung_Dx-G0037   thoracic   z= 400.0 mm  max|theta_x|=3.76 deg  OK
```

Many Lung-PET-CT-Dx diagnostic CTs cover only 160-230 mm and cannot hold +-2 deg;
NSCLC-Radiomics and the 5 mm StageII-Colorectal series are typically 285-480 mm.
Generation also logs a per-patient warning when the selection does not fit.

## Patient selection ("N and M")

Per-anatomy counts are the `n_patients` knob on each dataset entry; `--n-patients`
overrides all. **`selection: last`** takes patients from the back of each
collection -- training-set generation consumed the *first* samples, so expansion
patients come from the tail, clear of both training and the held-out test patients.
`selection: first | random` and explicit `patient_ids: [...]` are also available.

## Run

```bash
# preview selected patients + their seeded gantry (no MC):
uv run python scripts/mc/generate_patient_set.py --dry-run
# quick QC (3x3 grid, figures):
uv run python scripts/mc/generate_patient_set.py --grid-n 3 --make-figures --num-primaries 1e6
# full run:
uv run python scripts/mc/generate_patient_set.py --config scripts/mc/config_patient_set.yaml
```

Output dirs: `patient_set_{anatomy}_{patient}_e{E}_v{ver}/` (distinct prefix, so
they do not collide with the reviewer `beamlet_angle_robustness_*` dirs) -- drop-in
inputs to `scripts/mc/plot_angle_robustness.py`.

Config `robustness:` block is shared verbatim with the robustness pipeline
(`robustness_config_from_dict`); keep the grid / energies / ROI matched to the
existing runs.
