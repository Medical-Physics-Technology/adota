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
3. simulate at the canonical **90 deg** on the rotated grid;
4. extract axis-aligned exactly as at gantry 90.

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
