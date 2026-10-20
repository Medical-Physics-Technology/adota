# `score_saved_beamlets.py`: pencil beam and ADoTA on saved MC beamlet grids

Runs the commissioned pencil beam (PBA) and ADoTA on beamlet directories that the MC
generator already wrote, for example the paper's energy x gantry grids. Both are scored
against the saved MC by gamma pass rate, and the best and worst beamlets of each engine
are rendered. No new MC is run. **The source directories are never written to.**

```bash
uv run python scripts/pencil_beam/score_saved_beamlets.py \
    --config scripts/pencil_beam/config_score_energy_gantry_full.yaml
```

## Stages

| Stage | Does |
|---|---|
| `mirror` | Selects the full-lattice blocks and mirrors each into `output_root/<block>/` as symlinks to its `_ct/_ds/_flux.npy` and `_sim_res.json`. |
| `pba` | Rebuilds each field's canonical CT from the source DICOM with the generation config (`field_geometry`): resample, rotate, trim. It **refuses to continue unless every rebuilt crop equals the saved `_ct.npy` bit for bit**. Then it ray-traces once per field and writes `{id}_ds_pba.npy` per beamlet, computed in that beamlet's crop window. |
| `adota` | Fresh ADoTA inference into the mirror (`{id}_ds_pred.npy`), and a comparison against the source's own predictions when they exist (read only). |
| `score` | Gamma per criterion and dose agreement (range deltas, dose differences) of PBA and ADoTA against MC, per beamlet, on a process pool. |
| `report` | Best and worst `n` beamlets per engine by the first criterion, with ties broken by the stricter criteria and then the mean dose difference. Each gets a three-engine figure. |

All stages are resumable, and `--overwrite` recomputes. `--limit-blocks` and
`--limit-beamlets` make a smoke run.

## Selection

A block is kept when its grid indices span the `grid_n x grid_n` lattice and it holds
at least `min_beamlets` beamlets (the 18 x 18 sweeps lose a few to the QA gates).
Everything else is logged with the reason and written to `run_config.json`; the
4-5-beamlet smoke directories are excluded this way.

## Outputs

```
<output_root>/
  run_config.json                    resolved config, selected and excluded blocks, provenance
  all_beamlets.csv                   one row per beamlet: gamma per criterion and dose agreement, both engines
  summary.json                       median / mean / p5 / min per engine, overall, per anatomy, per energy
  ranking_best_worst_10_g3_3_10_global.csv
  figures/{pba|adota}_{best|worst}NN_<block>_<id>_g3_3_10_global.{svg,pdf,png}
  <block>/
    {id}_ct.npy, _ds.npy, _flux.npy, _sim_res.json    symlinks to the source (read only)
    {id}_ds_pba.npy                  pencil beam, same (H, W, D) layout as _ds.npy
    {id}_ds_pred.npy                 ADoTA, as written by run_inference
    metrics.json, pba_run.json
```

Gamma is global, normalised to the MC maximum, with a 10% cutoff. 3%/3mm is the
ranking key, and 2%/2mm and 1%/1mm are also reported. Figures show dose in Gy
(`to_gy`).

## Requirements

The source DICOM (the generation config's `datasets`), the DoseCUDA kernels and a
commissioned model, the ADoTA checkpoint, and a CUDA GPU (a few GB for the largest
field's ray trace).
