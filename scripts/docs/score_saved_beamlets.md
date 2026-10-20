# `score_saved_beamlets.py`: the pencil beam on saved MC beamlet grids

Runs the commissioned pencil beam (PBA) on beamlet directories that the MC generator
already wrote, for example the paper's energy x gantry grids. It then scores the PBA
and the **saved** ADoTA prediction (`{id}_ds_pred.npy`) against the **saved** MC
(`{id}_ds.npy`) by gamma pass rate, and renders the best and worst beamlets of each
engine. No new MC and no new ADoTA inference are run.

**Source directories:** the only file added is `{id}_ds_pba.npy`, created
exclusively (a temporary file hard-linked to the target, which fails if the target
exists). No existing file there is modified or replaced, by a rerun or by a crash.
Metrics, rankings and figures go to `output_root`.

```bash
uv run python scripts/pencil_beam/score_saved_beamlets.py \
    --config scripts/pencil_beam/config_score_energy_gantry_full.yaml
```

## Stages

| Stage | Does |
|---|---|
| `pba` | Rebuilds each field's canonical CT from the source DICOM with the generation config (`field_geometry`): resample, rotate, trim. It **refuses to continue unless every rebuilt crop equals the saved `_ct.npy` bit for bit**. Then it ray-traces once per field and adds `{id}_ds_pba.npy` per beamlet to the source directory, computed in that beamlet's crop window. Beamlets that already have one are skipped. |
| `score` | Gamma per criterion and dose agreement (range deltas, dose differences) of the PBA and the saved ADoTA against the saved MC, per beamlet, on a process pool. |
| `report` | Best and worst `n` beamlets per engine by the first criterion, with ties broken by the stricter criteria and then the mean dose difference. Each gets a three-engine figure. |

All stages are resumable. `--rescore` recomputes scores; pencil-beam files are never
replaced (delete them by hand to recompute). `--limit-blocks` and `--limit-beamlets`
make a smoke run.

## Selection

A block is kept when its grid indices span the `grid_n x grid_n` lattice and it holds
at least `min_beamlets` beamlets (the 18 x 18 sweeps lose a few to the QA gates), and
every beamlet has all its inputs, including the saved `_ds_pred.npy`.
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
  <block>/metrics.json, pba_run.json
<source block>/{id}_ds_pba.npy       added to the source: pencil beam, same (H, W, D) layout as _ds.npy
```

Gamma is global, normalised to the MC maximum, with a 10% cutoff. 3%/3mm is the
ranking key, and 2%/2mm and 1%/1mm are also reported. Figures show dose in Gy
(`to_gy`).

## Requirements

The source DICOM (the generation config's `datasets`), the DoseCUDA kernels and a
commissioned model, and a CUDA GPU (a few GB for the largest field's ray trace).
