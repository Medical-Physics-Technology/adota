# `score_saved_beamlets.py`: the pencil beam on saved MC beamlet grids

Runs the commissioned pencil beam (PBA) on beamlet directories that the MC generator
already wrote, for example the paper's energy x gantry grids. It then scores the PBA
and the **saved** ADoTA prediction (`{id}_ds_pred.npy`) against the **saved** MC
(`{id}_ds.npy`) by gamma pass rate, and renders the best and worst beamlets of each
engine. No new MC and no new ADoTA inference are run.

**Source directories:** files are only ever added, each created exclusively (a
temporary file hard-linked to the target, which fails if the target exists). No
existing file is modified or replaced, by a rerun or by a crash. What may be added:

- `{id}_ds_pba.npy` per beamlet, plus one `pba_provenance.json` per directory: model
  fingerprint, kernel build, code commit, settings.
- For beamlets that have no ADoTA prediction only: `{id}_ds_pred.npy` and
  `adota_provenance.json`.

A directory whose `pba_provenance.json` names a different model is refused, so two
models never mix in one place. Metrics, rankings and figures go to `output_root`.

```bash
uv run python scripts/pencil_beam/score_saved_beamlets.py \
    --config scripts/pencil_beam/config_score_energy_gantry_full.yaml
```

## Stages

| Stage | Does |
|---|---|
| `pba` | Rebuilds each field's canonical CT from the source DICOM with the generation config (`field_geometry`): resample, rotate, trim. It verifies **all** of a field's crops (each must equal the saved `_ct.npy` bit for bit) **before writing anything**. Then it ray-traces once per field and adds `{id}_ds_pba.npy` per beamlet, computed in that beamlet's crop window. Beamlets that already have one are skipped. |
| `adota` | Only where a beamlet has no `_ds_pred.npy`: runs ADoTA (`adota:` block) and adds the prediction. Existing predictions are used as they are. A no-op for directories that already have them. |
| `score` | Gamma per criterion and dose agreement (range deltas, dose differences) of the PBA and the saved ADoTA against the saved MC, per beamlet, on a process pool. |
| `report` | Best and worst `n` beamlets per engine by the first criterion, with ties broken by the stricter criteria and then the mean dose difference. Each gets a three-engine figure. |

All stages are resumable. `--rescore` recomputes scores. Pencil-beam files are never
replaced; to recompute with another model, delete the directory's `_ds_pba.npy` files
and its `pba_provenance.json` by hand. `--limit-blocks` and `--limit-beamlets` make a
smoke run, and `--dry-run` lists the selection and runs the full geometry check
without writing anything.

## Running it on your own directories

Any directory written by the MC generator (`src.mc_generation.robustness`, through
`scripts/mc/beamlet_angle_robustness.py`, `generate_patient_set.py` or
`generate_phantom_set.py`) works: one block per (patient, energy, field). Each beamlet
`{id}` has `{id}_ct.npy`, `{id}_ds.npy`, `{id}_flux.npy` and `{id}_sim_res.json` (with
`provenance_uid`, `gantry_angle`, `spot_position`, `initial_energy`), and optionally
`{id}_ds_pred.npy`.

1. **Find the generation config.** This is the YAML that produced the directories, for
   example `scripts/mc/config_energy_gantry_full.yaml` for `energy_gantry_full_*`, or
   `scripts/mc/config_thoracic_18x18.yaml` for `beamlet_angle_robustness_thoracic_LUNG1-006_*`.
   Its `experiment_prefix` and `datasets` match the directory names and the
   `provenance_uid` in `_sim_res.json`. The DICOM roots in its `datasets` must be
   readable.
2. **Write a config.** Copy `config_score_energy_gantry_full.yaml` and set:
   - `output_root`: a new directory on `/scratch`;
   - `sources`: globs of your directories;
   - `selection`: `grid_n` of the sweep and `min_beamlets` (set it to 1 to take
     everything);
   - `generation_config`: the YAML from step 1;
   - the `pencil_beam` model;
   - `adota`, if some beamlets lack predictions;
   - `scoring.gamma_criteria`.

   Energies must lie inside the model's span (70-250 MeV for `hptc_rsnone_mcsquare_v1`).
3. **Dry run:**
   `uv run python scripts/pencil_beam/score_saved_beamlets.py --config <yours> --dry-run`.
   It prints every selected and excluded directory (with the reason), checks that every
   CT is in the generation config, rebuilds each field and compares all its crops.
   **A wrong generation config fails here**, not after doses are written.
4. **Smoke:** `--limit-blocks 1 --limit-beamlets 6 --set output_root=<scratch>/smoke`.
   The PBA files it adds are real results; the full run skips them.
5. **Full run**, with `nohup`. The command and log path go into the experiment record.

## Reproducing or auditing a result

- `output_root/run_config.json` (written once, by the run that created `output_root`;
  every later invocation is appended to `invocations.jsonl`) holds the resolved config, the
  selected and excluded directories, and the provenance: the adota commit (and whether the tree had
  uncommitted code), the MCsquare and BDL sha256, the DoseCUDA build, the pencil-beam
  model fingerprint (`src.pencil_beam.beam_model.model_fingerprint`) and the time.
- Every source directory with PBA files has a `pba_provenance.json` with the same
  identity, so a `_ds_pba.npy` found later still says what produced it.
- Rerunning the same config at the same commit recomputes the scores
  (`--rescore`), the ranking and the figures from the saved files. The PBA doses
  themselves are deterministic up to GPU floating-point order.

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
ranking key, and 2%/2mm and 1%/1mm are also reported. Scores are computed on MCsquare's
scale, the unit of the saved files (see [docs/dose_units.md](../../docs/dose_units.md)).
Figures show Gy per MU of each beamlet (`mcsquare_to_gy_per_mu`).

## Requirements

The source DICOM (the generation config's `datasets`), the DoseCUDA kernels and a
commissioned model, a CUDA GPU (a few GB for the largest field's ray trace), and, for
the `adota` stage only, the ADoTA checkpoint.
