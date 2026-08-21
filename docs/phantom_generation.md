# Water-phantom MC dataset generation

A synthetic-phantom counterpart to the real-CT beamlet-angle robustness run
(`docs/mcsquare_engine.md`, `scripts/mc/beamlet_angle_robustness.py`). It sweeps a
synthetic phantom (a homogeneous water box, optionally with an air shell) over the
**same** beamlet-angle grid, energies, ROI, flux construction and QA gates as the
real patient CTs, giving a homogeneous-medium control for the angular study.

## Design: one new CT source, zero pipeline duplication

The per-beamlet pipeline (MC run -> ADoTA ROI crop -> flux channel -> QA gates ->
`{stem}_ct/_ds/_flux.npy` + `_sim_res.json`) is **reused unchanged** from
`src/mc_generation/robustness.py`. Only the *CT source* differs:

| Concern | Real CTs | Phantoms |
|---|---|---|
| Image source | `TCIADataset` (DICOM) | `PhantomDataset` (synthetic) |
| Per-CT record | `CTRecord` | `PhantomRecord` (same duck-typed interface) |
| Sweep + QA + save | `run_generation` / `generate_for_record` | *same functions* |
| Config | `RobustnessConfig` | *same dataclass* |

`PhantomRecord` exposes exactly what the generation spine needs
(`load_image()`, `uid`, `dataset_name`, `anatomy`, `patient_id`, `series_uid`), so
`generate_for_record(rec, runner, bdl, cfg)` runs on a phantom identically to a real
CT. Nothing in `robustness.py`, `mcsquare_runner.py`, `cropping.py` or `flux.py` was
touched.

- **Phantom source:** `src/datasets/phantom.py`
  - `PhantomSpec` -- content-addressable geometry (size, spacing, HU, air-shell
    depth); `content_hash` is the provenance key.
  - `build_phantom_image(spec) -> sitk.Image` -- builds the volume (dispatch on
    `kind`; the extension point for slabs).
  - `PhantomRecord` / `PhantomDataset` / `build_phantom_dataset(cfg)`.
- **CLI:** `scripts/mc/generate_phantom_set.py` (+ `config_phantom.yaml`).

## Run

```bash
uv run python scripts/mc/generate_phantom_set.py --config scripts/mc/config_phantom.yaml
# quick QC: 3x3 grid + per-beamlet figures
uv run python scripts/mc/generate_phantom_set.py --grid-n 3 --make-figures --num-primaries 1e6
```

Output dirs follow the robustness convention:
`{experiment_prefix}_{anatomy}_{patient_id}_e{E}_v{ver}/`, e.g.
`water_phantom_phantom_plain_e140_v2/`. They are drop-in inputs to the analysis /
plotting stage (`scripts/mc/plot_angle_robustness.py`): the phantom appears as an
`anatomy: phantom` "site" alongside the real anatomies.

## Config (`scripts/mc/config_phantom.yaml`)

Two sections that matter for phantoms:

- **`phantoms:`** -- one entry per synthetic phantom. `kind: water` builds a water
  box; `air_layer_depth` (mm) adds an air shell on every face (`0` = plain water,
  matching the pattern in `datagenerator/scripts/water_phantom_grid.py`). `name`
  becomes the `patient_id` slot in the output dir.
- **`robustness:`** -- the sweep, identical knobs to the real-CT config
  (`theta_*_range`, `grid_n`, `energies`, `roi_size`, `num_primaries`, gantry, QA
  `min_deposition_ratio`, ...). Keep these matched to the real run so the phantom is
  directly comparable.

The sweep uses the **beamlet-angle grid** (theta_x, theta_y in [-2, 2] deg), the
same as the real CTs -- *not* the spatial spot grid of the original
`water_phantom_grid.py`, per the "same way as real CT scans" requirement.

## Extending to high-density slabs (future work)

Slab phantoms (a bone/high-density slab placed horizontally or vertically in the
water box) plug in without touching the generation code:

1. Add a `kind` branch (e.g. `"water_slab_h"` / `"water_slab_v"`) in
   `build_phantom_image`, with the slab's position / thickness / HU as new
   `PhantomSpec` fields (add them to `content_hash` for provenance).
2. Add entries under `phantoms:` in the YAML.

`generate_for_record`, the QA gates, the flux channel and the analysis stage all
continue to work unchanged.
