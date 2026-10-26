# `ensemble_agreement.py`: do independently trained models find the same records hard?

`scripts/active_learning/ensemble_agreement.py` measures how far the members of an
ensemble of trained ADoTA models agree on **which** records are hard. It is the
agreement analysis of EXP-0014 (model-independence of the difficulty score).
Stage 0b runs it on the nine cycle-5 models of EXP-0012.

```bash
uv run python scripts/active_learning/ensemble_agreement.py \
    --config scripts/active_learning/config_ensemble_agreement.yaml
```

Stage 0b reuses the errors the runs already wrote, so it runs on CPU in about 35 s.

## What it computes

Members reach different accuracy levels, so the script compares their errors as
ranks, not values. It also compares them within **cells** of one scan
(`group_column`, `patient_key` on v3) and one 20 MeV energy bin. Two members do
not count as agreeing merely because both find high energies, or one anatomy,
hard. Per pair of members, for each error metric in `metrics`:

| Column | Meaning |
|---|---|
| `median_cell_spearman` (`_lo`, `_hi`) | Spearman of the errors inside every cell with at least `min_cell_size` complete records, median over cells; the interval comes from a bootstrap over scans |
| `pooled_spearman` | Spearman over all records; keeps the energy and scan structure, so it is secondary |
| `top_jaccard_within_cell`, `top_jaccard_pooled` | overlap of the top `top_fraction` of records |
| `training_jaccard`, `level` | overlap of the two training-id sets; `seed_only` when identical, `data` otherwise, `unknown` without ids |

The level is read from the training ids, never from the run names. In EXP-0012
only the three `score_topk` runs share a training set: its top-k selection did not
depend on the seed. So the seed floor is those 3 pairs, and the other 33 differ in
their data.

Error metrics, larger is worse: `rde_pct`, `gamma_fail` (1 - pass rate; the
gamma criteria go in the file names), `mape_pct`, `abs_dr80_mm`. Rank statistics
do not change under monotone transforms, so `rde_pct` also stands for
log(1 + RDE). A record with an undefined error in any member (no dR80 past a
plateau) is dropped for that metric only.

## Members: any model, any evaluation set

`record_ids` is a CSV of sample ids in the HDF5 at `dataset_path`. Each entry
under `members` takes one of three forms:

```yaml
members:
  - run_dir: /scratch/.../train_<ts>_al_EXP-0012_d30_random_seed1234   # + cycle (or the top-level default)
  - checkpoint: /scratch/.../model.pth
    model_config: /scratch/.../config.yaml    # a retrospective run config or a TrainingConfig YAML
    training_ids: /scratch/.../training_ids.csv   # optional; without it the level is unknown
    label: my_model
  - errors_csv: /scratch/.../per_record_errors.csv   # columns of evaluate_metrics
    training_ids: /scratch/.../training_ids.csv
```

`source` decides where the errors come from:

- `auto` reuses the member's errors CSV when it covers every evaluation record.
  For a run, that CSV is `validation/cycle_<cc>_full_gamma_<criteria>.csv`, so the
  gamma criteria in the config must match. Otherwise the errors are inferred.
- `csv` never infers.
- `infer` always does.

Inference reruns the benchmark's own per-record metric code
(`evaluate_metrics`) on `checkpoints/cycle_<cc>/<checkpoint>.pth`, with
`torch.compile` off. It writes `members/<label>_errors.csv`, which a later run
can pass back as `errors_csv`. Use `--device-index` for a GPU; without CUDA it
falls back to CPU, which is slow: gamma runs on the full crop.

## Guards

- **Held out.** Before any number is computed, `held_out.csv` counts, per member,
  the evaluation records and scans it trained on. The script stops if any count
  is not zero. `allow_training_overlap: true` overrides that for an in-sample
  analysis such as stage 0a.
- **Root disk.** The script refuses an `output_dir` or a `TMPDIR` on the root
  filesystem. Outputs go to `/scratch`.
- **Reproducibility.** Every run gets a timestamped directory with `manifest.json`
  (git commit, argv, dataset fingerprint, the sha256 of the record-id file and of
  every checkpoint and errors CSV used), the resolved `config.yaml`, and a fixed
  `bootstrap_seed`.

## Outputs

```
<output_dir>/ensemble_agreement_<timestamp>/
  members.csv  held_out.csv  summary.json  manifest.json  config.yaml  ensemble_agreement.log
  pairs_<metric>.csv    one row per pair (table above)
  levels_<metric>.csv   mean per level with its interval, and data_minus_seed_only
  cells_<metric>.csv    each cell's Spearman per pair (columns <a>__<b>)
  members/              inferred errors, if any
  figures/agreement_matrix_<metric>.{svg,pdf,png,csv}
```

The figure is a members-by-members matrix of `median_cell_spearman` with the
`seed_only` pairs framed. Its colour scale starts at the smallest value, rounded
down to 0.05.

## Caveat for stage 0b

The nine models share one configuration and one size (d30), and all warm-start
from the same cycle-0 checkpoint. Their agreement is a preview of stage 1, not
the test of model independence.
