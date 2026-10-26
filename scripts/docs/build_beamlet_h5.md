# build_beamlet_h5.py and check_beamlet_h5.py

> **Reproducibility / portability.** The paths in the examples are the original
> development environment's locations. Point the scripts at your own data via the
> CLI options, and keep large outputs in a directory of your choice.

---

## What v3 is

The training HDF5 v2 stores per record only six attributes: `dose_deposition_ratio`,
`gantry_angle`, `id`, `initial_energy`, `beamlet_angles`, and `stat_uncertainty`.
The v3 format holds exactly the v2 record ids minus the exclusion list, with the CT,
dose and flux datasets reproduced byte for byte, plus full raw provenance as extra
attributes, plus derived patient and spot keys, plus file-level build provenance.
The v3 file carries an index CSV and a skip CSV as side outputs, so the build is
auditable from the outputs alone.

An existing v3 file can be verified byte-for-byte against the v2 source. The
checker compares the id sets, datasets, storage parameters, the six v2 attributes,
the v3-only attributes, and validates that the H5PYGenerator output is identical
for sampled records.

## The build recipe

The recipe is pinned to ensure reproducibility. Per record, starting from the raw
Monte Carlo JSON and the arrays on disk:

1. Load the raw CT (int16), dose (float32), and metadata JSON from the three files.
2. Extract the energy from `simulation_log.energy[0]` (not the top-level
   `initial_energy`), look up the flux-model sigmas for that energy in the BDL
   (nearest MeanEnergy row), and run flux projection with the following mandatory
   settings to match v2 numerically:
   - `flux_projection_gpu_batched(..., spacing=np.asarray([1.0, 1.0, 1.0],
     dtype=np.float64), dtype=torch.float64, return_numpy=True)`. The float64
     spacing is mandatory (under NumPy 2, the default float32 spacing silently
     demotes the sigma in the receiver, up to 148 ulp after pooling). The float64
     dtype is also mandatory for equivalence (float32 dtype adds up to 558 ulp).
3. Normalise CT with the scale's `min_ct` and `max_ct`, dose with `min_ds` and
   `max_ds`, energy with `min_energy` and `max_energy`. The scale values are used as
   the Python numbers they parse to (never numpy scalars): a float32 dose array divided
   by a Python float stays float32, which is what v2 did.
4. Downsample each of CT, dose and flux (still float64 at this point) to half size per
   axis with the chosen `--downsample` method. `average` is the v2 line, verbatim:
   `torch.tensor(grid, dtype=torch.float32)[None, None]`, `F.avg_pool3d(kernel_size=2,
   stride=2)` on the CPU, `.squeeze().numpy()`; the float32 cast comes first and the
   pooling second, so the reduction order is v2's. `linear` (scipy zoom, order 1) and
   `trilinear` (`F.interpolate`, `align_corners=False`) are the old `DoTADataset`
   ports; one method per file, recorded in the file attrs. Only `average` reproduces v2.
5. Apply the skip rules to the pooled arrays. Skip rules, in the order a reason is
   assigned:

| reason | test |
|---|---|
| `excluded` | id on the exclusion list (decided before any file is opened) |
| `load_error` | any exception reading `_ct.npy`, `_ds.npy` or `_metadata.json`, or during preprocessing |
| `json_error` | JSON does not parse or lacks a key the recipe needs |
| `n_spots` | `simulation_log.n_spots != 1` or more than one bixel shift |
| `zero_dose` | pooled normalised dose sums to zero |
| `bad_shape` | any pooled array is not `(40, 40)` laterally |
| `short_depth` | any pooled array has fewer than 160 depth voxels |

Every skipped id gets a row in the skip CSV with its reason; a skip never stops the
build. The two hard errors are a missing exclusion list and an unreadable BDL, both
raised before any record is read.

## Running the builder

### Smoke build (for testing)

A single-source smoke build on 100 records:

```bash
uv run python scripts/build_beamlet_h5.py \
    --raw-root /RadiotherapyData/dataset_v0 \
    --source initial_test_one_ct \
    --out /scratch/test_smoke.h5 \
    --bdl /home/mstryja/tools/mcsquare/BDL/hptc_beam_model_rsnone.txt \
    --exclusion-list /path/to/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt \
    --device cpu \
    --limit 100
```

Or using an ids file to scope to specific records:

```bash
uv run python scripts/build_beamlet_h5.py \
    --raw-root /RadiotherapyData/dataset_v0 \
    --source trainset_pelvis --source initial_test_one_ct \
    --out /scratch/test_smoke.h5 \
    --bdl /home/mstryja/tools/mcsquare/BDL/hptc_beam_model_rsnone.txt \
    --exclusion-list /path/to/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt \
    --device cpu \
    --ids-file /path/to/ids_to_test.txt
```

### Full build

The raw tree is a CIFS mount with high latency and low bandwidth per reader. A single
reader achieves about 3 MB/s; with 32 parallel readers, the throughput reaches about
100 MB/s, making the load step the bottleneck. The builder writes to an intermediate
`.partial` file and renames it to the final location on success. The full build of
79,257 candidates takes roughly 3 to 4 hours with 32 workers:

```bash
nohup uv run python scripts/build_beamlet_h5.py \
    --raw-root /RadiotherapyData/dataset_v0 \
    --source trainset_pelvis --source initial_test_one_ct \
    --out /scratch/mstryja/DoTA_dataset_v3/trainset_pelvis_initial_test_one_ct_downsampled_v3_all_SingleGaussian.h5 \
    --bdl /home/mstryja/tools/mcsquare/BDL/hptc_beam_model_rsnone.txt \
    --exclusion-list /home/mstryja/projects/dota_pytorch/auxilary_files/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt \
    --device cuda:0 \
    --flux-batch 32 \
    --workers 32 \
    > /scratch/build.log 2>&1 & echo "PID: $!"
```

The build writes three outputs next to the final HDF5:

- `<stem>_index.csv`: one row per written record, with columns `sample_id` plus
  every record attribute (vectors flattened as `name_0`, `name_1`, ...). Useful for
  bulk queries and provenance tracing.
- `<stem>_skipped.csv`: one row per skipped record, with columns `sample_id`,
  `source`, `reason` (one of the skip reasons), and `detail` (error message when
  applicable).
- `<stem>_build.log`: the full build log with timestamps and progress every 1,000
  records.

### Resume from a partial build

If a build is interrupted, use `--resume` to reopen the `.partial` file, skip complete
groups, delete incomplete ones, and reprocess the remaining candidates:

```bash
uv run python scripts/build_beamlet_h5.py \
    --raw-root /RadiotherapyData/dataset_v0 \
    --source trainset_pelvis --source initial_test_one_ct \
    --out /scratch/mstryja/DoTA_dataset_v3/trainset_pelvis_initial_test_one_ct_downsampled_v3_all_SingleGaussian.h5 \
    --bdl /home/mstryja/tools/mcsquare/BDL/hptc_beam_model_rsnone.txt \
    --exclusion-list /home/mstryja/projects/dota_pytorch/auxilary_files/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt \
    --resume
```

A CUDA OOM halves the `--flux-batch` automatically and continues. If even a single
record's flux projection exceeds memory, the batch cannot be halved further and the
error propagates.

### Options

| Option | Default | Meaning |
|---|---|---|
| `--raw-root` | `/RadiotherapyData/dataset_v0` | Root of the raw record tree. |
| `--source` | `trainset_pelvis`, `initial_test_one_ct` | Raw source subdirectory, repeatable; order matters for `--limit`. |
| `--out` | required | Final `.h5` path; the build writes `<out>.partial` first. |
| `--bdl` | required | MCsquare beam-data-library file. |
| `--scale-json` | none | JSON with the six scale keys (min/max for ct, ds, energy); defaults to `src.adota.config.DEFAULT_SCALE`. |
| `--exclusion-list` | required, no default | Ids to drop, one per line. A missing file is a hard error. |
| `--device` | `cuda` | `cuda`, `cuda:N` or `cpu`; falls back to `cpu` with a warning if CUDA is unavailable. |
| `--flux-batch` | 32 | Flux projections per batched GPU call; halved on CUDA OOM down to 1. |
| `--downsample` | `average` | `average`, `linear`, or `trilinear`. Only `average` reproduces v2. |
| `--limit` | none | Truncate the candidate list after source concatenation. |
| `--ids-file` | none | One id per line; restricts the candidate set before the exclusion filter (so listed excluded ids still get skipped). |
| `--resume` | off | Reopen the existing `.partial`, skip complete groups, rebuild incomplete ones. |
| `--workers` | 8 | Process-pool size for the CPU load/screen step; 0 or 1 runs in-process (for debugging). |

## Checking a rebuilt file

Verify that a v3 build is byte-for-byte identical to its v2 source:

```bash
uv run python scripts/check_beamlet_h5.py \
    --old /path/to/v2.h5 \
    --new /path/to/v3.h5 \
    --exclusion-list /path/to/IndexesExclude_...txt \
    --out /scratch/check_report.json
```

### Full check on a complete build

The default checks a random sample of 200 retained records for storage parameters,
array equality, and attribute equivalence:

```bash
uv run python scripts/check_beamlet_h5.py \
    --old /scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5 \
    --new /scratch/mstryja/DoTA_dataset_v3/trainset_pelvis_initial_test_one_ct_downsampled_v3_all_SingleGaussian.h5 \
    --exclusion-list /home/mstryja/projects/dota_pytorch/auxilary_files/IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt \
    --sample 2000 \
    --out /scratch/check_report.json
```

### Checking a smoke (partial) build

When checking a partial build created with `--limit` or `--ids-file`, scope the
comparison to the same ids:

```bash
uv run python scripts/check_beamlet_h5.py \
    --old /path/to/v2.h5 \
    --new /path/to/v3_smoke.h5 \
    --exclusion-list /path/to/IndexesExclude_...txt \
    --ids-file /path/to/smoke_ids.txt \
    --out /scratch/smoke_check_report.json
```

The `--ids-file` option overrides `--sample` and scopes the id-set check to the
intersection of v2 ids and the file, so every id in the file is checked.

`--allow-new <file>` lists ids that may be present in v3 without being in v2. The
2026-09-17 build recovered three records
(`/scratch/mstryja/DoTA_dataset_v3/recovered_not_in_v2.txt`) that the v2 build had lost
to transient CIFS read errors ("Host is down", "Resource temporarily unavailable" in
the v2 build log); they are valid records and were kept, so v3 holds 70,146 records,
v2 minus the 952 excluded plus these three. The report lists them under
`a.allowed_new`.

What the full build's check showed: `ct` and `dose` were `array_equal` on all 2,000
sampled records; `flux` was `array_equal` on 1,991 and differed on 9 by exactly one
voxel each, by 1 to 2 float32 ulp, at values between 1e-41 and 6e-14 against peaks of
about 0.01 (deep Gaussian tail). The cause is CUDA versus CPU float64 round-off in the
kernels (the batched call at batch size 1 on CUDA reproduces v3's value; the NumPy and
CPU torch paths reproduce v2's), not the batching, so `flux_equivalence` is
`allclose` for a CUDA build and `array_equal` for a `--device cpu` build.

### Report format

The report is a JSON file with six top-level checks (a through f) plus a verdict:

- **(a) id-set arithmetic**: validates that the v3 id set equals the v2 id set minus
  the exclusion list. Lists any excluded ids present in v3, any retained v2 ids
  missing from v3, and any unexpected ids in v3.
- **(b) exclusion check**: confirms no exclusion id is present in v3 (subset of (a)).
- **(c) record equality**: for sampled records, compares dataset storage parameters
  (dtype, shape, chunks, compression), array values (CT and dose exact; flux exact or
  passing `allclose(rtol=1e-6)`), and the six v2 attributes (value and type/dtype).
  If flux differs, the max absolute, relative and ULP differences are recorded.
- **(d) v3-only attributes**: checks that every v3-only attribute is present and
  non-empty on every sampled record.
- **(e) spot_key multiplicity**: an informational histogram of how many records share
  a spot_key (the a-priori expectation was a mode of 4; the smoke build showed a mode
  of 1, so read the actual histogram). Never fails. Read from the index CSV when present, otherwise
  from the file attributes.
- **(f) H5PYGenerator equivalence**: for up to 5 sampled records, validates that the
  generator output (CT, energy, dose tensors) is `torch.equal` between v2 and v3.

The top-level fields are:

- `verdict`: `"pass"` only if (a), (b), (c), (d) and (f) all pass; (e) is always pass.
- `flux_equivalence`: `"array_equal"`, `"allclose"`, or `"fail"` (the strongest verdict across all sampled records' flux arrays).
- `seed`: the random seed for sampling (default 0).
- Exit code 1 on a `"fail"` verdict.

## Reading v3 from code

The v3 file is backward compatible with existing code. `H5PYGenerator` and
`load_record_ids` work unchanged (the exclusion list is a no-op on v3). The
`record_metadata` function in `src.active_learning.retrospective.dataset` reads
the new file-level and record-level attributes:

- When an index CSV (`<stem>_index.csv`) sits next to the HDF5 file, it is read
  and preferred over the file attributes for bulk queries (faster and more robust).
- On v3 records (identifiable by the `schema_version` attribute or an index row),
  the returned frame gains the columns `source_dataset`, `patient_key`, `spot_key`,
  `isocenter_x_mm`, `isocenter_y_mm`, `isocenter_z_mm`, and `energy_mev` is read
  from the `energy_mev` attribute rather than being denormalised from `initial_energy`.
- On v2 records, the frame is exactly as today.

## What is not done in this branch

- **Config migration**: switching `scripts/config_al_retro_loop.yaml` and
  `scripts/config_al_train.yaml` to use v3 changes every run manifest's dataset
  fingerprint, so it is a separate task.
- **Duplicate spot detection**: a diagnostic for spot_key duplicates is not yet
  written.
- **Default float64 spacing**: whether `float64` spacing should become the default
  in `src/beamlets/flux.py` is left to a follow-up.
