# `compare_engines.py`: MC vs pencil beam vs ADoTA on the same CT grid

Runs the same proton beamlets through three dose engines and compares them voxel by
voxel on **one CT grid**:

- **MC**: MCsquare, the reference.
- **PBA**: the [DoseCUDA](https://github.com/jhu-som-radiation-oncology/DoseCUDA) pencil beam (Bhattacharya et al., J Appl Clin Med Phys 2025 ([doi:10.1002/acm2.70093](https://doi.org/10.1002/acm2.70093))),
  commissioned with [commission_dosecuda.md](commission_dosecuda.md); source, version,
  licence and citation in
  [docs/pencil_beam.md](../../docs/pencil_beam.md#source-version-licence-and-citation).
- **ADoTA**: the trained model.

Background: [docs/pencil_beam.md](../../docs/pencil_beam.md).

```bash
uv run python scripts/pencil_beam/compare_engines.py \
    --config scripts/pencil_beam/config_compare_engines.yaml --dry-run   # list the beamlets
uv run python scripts/pencil_beam/compare_engines.py \
    --config scripts/pencil_beam/config_compare_engines.yaml
```

## How a beamlet is built

Selection reuses the MC generator's own config blocks. `datasets` works as in
`scripts/mc/generate_patient_set.py`. `robustness` supplies the energies, the angle
lattice (`grid_n` plus explicit `angles`), the gantry mode and seed, the ROI, the
primaries and `rotate_to_canonical`. A comparison beamlet is therefore built exactly
like a training beamlet:

1. The CT is resampled to 1 mm with vacuum mapped to air, then rotated into the
   field's canonical beam's-eye frame (`field_geometry`). **This is the common grid.**
2. MCsquare simulates the spot on it.
3. DoseCUDA computes the same spot on the same grid, isocenter and spot position.
4. ADoTA's crop, flux and QA come from the dataset code (`process_beamlet`), then
   batched inference per block. The predicted crop is deposited back onto the grid.
5. The three doses are compared on one window of that grid: ADoTA's ROI widened by
   `lateral_margin_mm` per side, over the full beam axis. The share of MC and PBA
   dose inside it is recorded.

## Options

| Option | Meaning |
|---|---|
| `--config` | YAML (default `scripts/pencil_beam/config_compare_engines.yaml`) |
| `--set KEY=VALUE` | override any key (dotted, YAML-parsed values) |
| `--num-primaries` | MC primaries per beamlet |
| `--no-adota` | MC vs PBA only |
| `--dry-run` | list patients, gantries and beamlet counts, then exit |

## Config keys

`name`, `output_dir`, `datasets`, `robustness` (as in the generators), `engine`
(MCsquare), `pencil_beam.{dosecuda_kernels, model_dir, gpu_id}`,
`adota.{model_dir, fname, device_index, batch_size}`, and
`comparison.{gamma_criteria, gamma_local, gamma_backend, gamma_backend_options,
lateral_margin_mm, make_figures}`.

The default lattice is `grid_n: 19`, so that 0 degrees is a cell; the 18 x 18
dataset lattice has none. Use `--set robustness.grid_n=18` with on-lattice `angles`
to reproduce dataset cells exactly.

Gamma is global, normalised to the MC maximum, with a 10% cutoff by default.
3%/3mm is the headline; 2%/2mm and 1%/1mm are also reported. A PBA failure at a
lateral interface (MC distal shoulder, PBA single smeared peak) can still pass 3%/3mm
globally, so read gamma together with the range deltas, or set `gamma_local: true`.
`compare_engines.py` puts the criterion and global/local in every figure name and title.

## Outputs

```
<output_dir>/
  run_config.json                  resolved config + provenance (MCsquare/BDL sha256, DoseCUDA build,
                                   adota commit, PBA model dir, ADoTA checkpoint)
  engine_comparison.csv            one row per beamlet (rewritten after every block)
  engine_comparison.json           rows + per-engine summary (mean/median/min/max)
  <anatomy>_<patient>_g<gantry>_e<E>/
    a{ix}_{iy}_ct|ds|flux.npy, _sim_res.json, _ds_pred.npy   ADoTA input/output, dataset format
    a{ix}_{iy}_engines.npz         ct, mc, pba on the comparison window + region_start, crp, totals
    a{ix}_{iy}_adota_region.npy    ADoTA on the same window
    a{ix}_{iy}_cell.json           spot, timings, MC uncertainty, PBA model energy / interpolation
    metrics.json
    figures/a{ix}_{iy}_engines_g3_3_10_global.{svg,pdf,png}
```

Per-beamlet metrics, for `pba_*` and `adota_*`, each against MC:

- gamma pass rate per criterion;
- `r{100,90,80,50,20}_delta_mm` and `dfw_delta_mm` from the integrated depth dose;
- peak, integral and IDD-peak ratios;
- mean, max and signed voxel difference in % of the MC maximum, over the voxels
  above 10% of that maximum.

Timings are recorded too: MC seconds, PBA seconds (ray trace plus dose on the full
grid), and ADoTA seconds per beamlet.

The run is resumable per beamlet: a cell with its `_engines.npz` and `_cell.json`
on disk is scored again but not re-simulated.

## Requirements

The MCsquare install, the DoseCUDA kernels and a commissioned model, a CUDA GPU for
the PBA and ADoTA, and the ADoTA checkpoint
(`models/DoTA_v3_grid_search_v11` by default). Every energy must lie inside the
commissioned span, and at most 250 MeV: above that MCsquare's nuclear data ends, so there is
no valid MC reference (the script refuses it).
