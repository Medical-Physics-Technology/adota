# `commission_dosecuda.py`: commission the DoseCUDA pencil beam against MCsquare

Builds a DoseCUDA proton beam model whose beamlets reproduce MCsquare's (BDL
`hptc_beam_model_rsnone.txt`, scanner `default`). It writes the model in DoseCUDA's
own lookup-table layout and validates it against fresh MC in water. Method,
decisions and limitations: [docs/pencil_beam.md](../../docs/pencil_beam.md).

```bash
# once: build the CUDA kernels for adota's Python
scripts/pencil_beam/provision_dosecuda.sh

uv run python scripts/pencil_beam/commission_dosecuda.py \
    --config scripts/pencil_beam/config_commission_dosecuda.yaml \
    --stages rlsp,simulate,fit,validate
```

## Stages

| Stage | Does | Cost (16 MC threads) |
|---|---|---|
| `rlsp` | `HU_RLSP.csv` from MCsquare's HU tables and Geant4 stopping powers | seconds |
| `simulate` | one central spot per energy node x SSD in a water box; one `runs/e{E}_s{s}.npz` each (full dose, IDD, single-Gaussian widths); resumable | 148 runs, ~7-35 s each at 2e6 primaries |
| `fit` | kernel-form width fits (process pool), then IDD / sigma_mcs / sigma_air / R80 -> `model/` | ~minutes |
| `validate` | fresh MC at held-out SSD(s) and between-node energies, 1e7 primaries; PBA on the same phantom; CSV, JSON, figures | ~14 runs, ~1-3 min each |

`simulate` and `validate` are long; launch them with `nohup` and a log on `/scratch`.

## Options

| Option | Meaning |
|---|---|
| `--config` | YAML file (default `scripts/pencil_beam/config_commission_dosecuda.yaml`) |
| `--stages` | comma-separated subset of `rlsp,simulate,fit,validate` (default `rlsp,fit`) |
| `--set KEY=VALUE` | override any key, dotted for nested (`commissioning.num_primaries=2e5`) |
| `--gpu-id` | CUDA device for the pencil beam, as seen through `CUDA_VISIBLE_DEVICES` |

Smoke test (two nodes, one in-between validation energy, about a minute):

```bash
uv run python scripts/pencil_beam/commission_dosecuda.py --stages rlsp,simulate,fit,validate \
    --set output_dir=/scratch/$USER/pencil_beam/smoke --set "commissioning.energies=[120,125]" \
    --set commissioning.num_primaries=2e5 --set "validation.energies=[122.5]" --set validation.num_primaries=1e6
```

## Config keys

- `output_dir`: the run root (MC runs, validation); on `/scratch`.
- `model_dir`: **the deliverable**, the pencil-beam model of our beam, meant for reuse
  beyond the comparison. Default `/home/mstryja/tools/dosecuda/models/hptc_rsnone_mcsquare_v1`
  (about 2 MB, next to the engines; defaults to `<output_dir>/model` if unset). Bump the
  version suffix instead of overwriting a model in use.
- `engine`: MCsquare install, work dir, BDL, scanner (as in the MC generators).
- `pencil_beam.dosecuda_kernels`, `pencil_beam.gpu_id`.
- `rlsp_reference_energy_mev`: energy of the stopping-power ratio (100).
- `commissioning`: energy nodes (default 70-250 MeV every 5; refused above 250 MeV, see
  [docs/pencil_beam.md](../../docs/pencil_beam.md#mcsquare-above-250-mev)), `surface_to_iso_mm`,
  phantom size, primaries, threads, LUT span (`lut_depth_factor`), `z_ref` fraction.
  `air_front_mm + surface_to_iso_mm` must lie in [120, 420] mm, because MCsquare
  misplaces the entrance outside that range; this is guarded.
  `dose_unit` sets the IDD table's unit: `gy_mm2_per_mu` (the default and `_v2`; a
  spot of weight w MU gives Gy) or `mcsquare_per_weight_1000` (`_v1`). With
  `gy_mm2_per_mu` the nodes' protons per MU are stored in `energies.csv`, and between
  nodes the IDD is blended per proton. The two units give the same MCsquare-scale
  dose to float precision (measured on the 14 validation runs: R80 within 6e-6 mm,
  gamma within 8e-6 percentage points).
- `validation`: `energies`, `surface_to_iso_mm`, `num_primaries`, `gamma_criteria`,
  `gamma_backend`, `make_figures`, `results_subdir` (results folder under `output_dir`;
  the validation MC itself is always cached in `validation/runs`, so validating a new
  model reuses it).

## Outputs

```
<output_dir>/
  runs/e{E}_s{s}.npz                     commissioning MC (dose, idd, widths, provenance)
<model_dir>/                             DoseCUDA layout: machine_geometry.csv, HU_RLSP.csv,
                                         RS0/energies.csv, RS0/energy_NNN.csv, commissioning.json,
                                         README.md (model card: beam, units, usage, accuracy),
                                         validation/ (copy of the validation CSV/JSON and figures)
  validation/runs/*.npz                  held-out MC
  validation/water_validation.csv|.json  per run: dR80, IDD peak/integral ratios, width error and
                                         central-axis dose ratio at 25/50/75/90% R80, MC x-y width
                                         asymmetry, dose agreement, gamma per criterion
  validation/figures/water_e{E}_s{s}.{svg,pdf,png}         IDD and width, MC vs PBA
  validation/water_validation_summary_g3_3_10.{svg,pdf,png}  metrics vs energy
```

`commissioning.json` records the provenance (MCsquare and BDL sha256, DoseCUDA build,
adota commit), the commissioning settings, and per-node diagnostics: R80 per SSD,
sigma_air fit residuals, and MC x-y asymmetry.

## Requirements

A CUDA GPU (a few hundred MB), the DoseCUDA kernels, and the MCsquare install. The
pencil beam needs isotropic spacing.
