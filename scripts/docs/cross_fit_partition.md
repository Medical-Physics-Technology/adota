# `cross_fit_partition.py`: the EXP-0014 scan partition, folds and HPO split

`scripts/active_learning/cross_fit_partition.py` writes, once, the scan-level
partition that EXP-0014 to EXP-0017 share:

- **F / V / R.** F is the EXP-0004 frozen test set and V the 12 EXP-0012
  held-out scans; R is every other scan. F and V may overlap.
- **Cross-fitting folds.** All of D goes into `n_folds` scan folds, stratified
  by anatomy and independent of F / V / R. Each member of the EXP-0014 ladder
  trains once per fold on the other folds and predicts the held-out one, so
  every record gets an out-of-sample error from every member.
- **The HPO split.** Hyperparameter optimisation trains and validates on R
  only, never on F or V: `hpo_val_scans_per_stratum` R scans per anatomy are
  held out for HPO validation.

```bash
uv run python scripts/active_learning/cross_fit_partition.py \
    --config scripts/active_learning/config_cross_fit_partition.yaml
```

It runs on the CPU in under a minute and reads only the HDF5 keys, the
`<stem>_index.csv` beside the HDF5 and the provenance map.

## Rules

| Step | Rule |
|---|---|
| D | every record of `dataset_path` not on `exclude_indexes_path` |
| Scan | `group_column` (`patient_key` on v3: one CT scan) |
| F | every scan with at least one record in `frozen_test_ids`; the whole scan belongs to F |
| V | every scan whose provenance `patient` string is in `held_out_patients` |
| Folds | per anatomy (sorted), scans in a seeded permutation of their sorted keys, each to the fold with the fewest scans of that anatomy, then the fewest records, then the lowest index |
| HPO validation | `select_held_out_groups` from the retrospective patient split, on R only, with `hpo_seed` |

Per anatomy the fold sizes differ by at most one scan. The result depends only
on the seeds and the set of scans, never on the file order. An id or a
patient that is not in D is an error, as is a record without an anatomy.

## Outputs

```
<output_dir>/
  partition.csv        one row per scan: scan, anatomy, n_records, role (F | V | F+V | R), fold, hpo (train | val | none)
  records.csv          one row per record: sample_id, scan, energy_mev, anatomy, role, fold, hpo
  folds/fold_<k>_train_ids.csv   every record outside fold k (column sample_id)
  folds/fold_<k>_eval_ids.csv    every record in fold k
  hpo/hpo_train_ids.csv  hpo/hpo_val_ids.csv
  fold_summary.csv     scans and records per fold and anatomy, with the F and V scans each holds
  summary.json  manifest.json  config.yaml  cross_fit_partition.log
```

`manifest.json` carries the git commit, the dataset fingerprint and the sha256
of the frozen-test, held-out, exclusion and provenance files.

## Guards

- The script refuses an `output_dir` that already holds a partition, since
  later experiments reference it; `--overwrite` replaces it on purpose.
- It refuses an `output_dir` on the root filesystem.
- `check_partition` stops before anything is written if a record has no fold,
  a scan is split across folds, an F or V scan entered the HPO split, or the
  HPO train and validation scans overlap.
