# `al_build_beamlet_pool.py`: a labelled beamlet pool on new patients

`scripts/al_build_beamlet_pool.py` generates Monte Carlo beamlets on patients the
training set has not used, split by patient into train / validation / test. The
train split is the pool the active-learning loop samples from; validation and test
are held out. It runs in batches: a batch is simulated on fast local storage,
packed into compressed shards, moved to its final storage and deleted locally, so
a multi-day run never holds more than one raw batch per process.

| Run | Config | Beamlets | Primaries | Where |
|---|---|---|---|---|
| Apollo | `scripts/config_beamlet_pool.yaml` | 50,000 | 1e6 | `/scratch/mstryja/adota_pool/v1_apollo_1e6`, later `sync` to `/RadiotherapyData/adota_pool/v1_apollo_1e6` |
| DelftBlue | `scripts/config_beamlet_pool_delftblue.yaml` (template) | 100,000 | 1e7 | set in the config |

Both runs share **one patient registry** (`registry/beamlet_pool_patients_v1.csv`,
committed), so a patient is in the same split on both machines. Their plans differ
(own seed), so no beamlet is simulated twice.

## Stages

Every stage is a separate command and is started by hand.

```bash
export TMPDIR=/scratch/$USER/tmp        # the scripts refuse a root-disk TMPDIR
uv run python scripts/al_build_beamlet_pool.py screen   --config scripts/config_beamlet_pool.yaml --n-workers 8
uv run python scripts/al_build_beamlet_pool.py plan     --config scripts/config_beamlet_pool.yaml
uv run python scripts/al_build_beamlet_pool.py check-energy-mixing --config scripts/config_beamlet_pool.yaml
uv run python scripts/al_build_beamlet_pool.py generate --config scripts/config_beamlet_pool.yaml --shard 0 --n-shards 2
uv run python scripts/al_build_beamlet_pool.py generate --config scripts/config_beamlet_pool.yaml --shard 1 --n-shards 2
uv run python scripts/al_build_beamlet_pool.py status   --config scripts/config_beamlet_pool.yaml
uv run python scripts/al_build_beamlet_pool.py sync     --config scripts/config_beamlet_pool.yaml   # later
uv run python scripts/al_build_beamlet_pool.py stats    --config scripts/config_beamlet_pool.yaml   # reporting
```

1. **`screen`** (DICOM headers only, about 10 minutes with `--n-workers 8`). Every
   CT series of every patient of the four collections (Lung-PET-CT-Dx,
   NSCLC-Radiomics, StageII-Colorectal, Prostate-AEC), from **all** of its headers:
   - a series is the set of files carrying its series UID in their own header, CT
     image objects only (localizers, other series and non-CT files in the same
     directory are set aside), and it must stack into a volume: one orientation
     and in-plane grid, no in-plane shift between slices, no duplicate positions,
     no slice step above `max_slice_gap_ratio` (1.5) times the median step
     (`src/provenance/series_geometry.py`). The reader would otherwise stack a
     missing slice, or two acquisitions under one UID, as if evenly spaced;
   - the image-quality gates of `TCIADataset` (spacing, at least 200 mm of z for
     the +-1.5 deg steering window, 2 x (68 + 30) mm; at least 320 mm in plane for
     the 320 mm crop along the beam), on the **measured** slice count and z extent,
     not the file count and the nominal slice thickness (overlapping
     reconstructions have a thickness larger than their step);
   - the 1 mm grid of **every** CT series of the patient against the 56 scans of
     the training HDF5 (`*_index.csv`): the training set records a scan only by
     that grid, and a training scan can be any series of the patient, so a match
     on any of them excludes the patient. The command prints how many training
     scans were matched; a training scan that matches no patient here comes from a
     collection that is not on this machine, and the leakage check cannot cover
     that collection;
   - the selected series is the largest one that passes; the registry records its
     UID and slice count, and `plan` / `generate` load exactly that series, check it
     again, and refuse it if it changed;
   - claims: plan-validation patients (`opentps_plans`) and the robustness patients
     go to test, the AL validation CTs to validation, the AL pool CTs to train;
   - the rest per collection, seeded, 70 / 15 / 15.

   Writes the registry and refuses to overwrite it (`--overwrite` redraws).
2. **`plan`** (CPU, no Monte Carlo). Per patient, 2 random field angles, steering on
   the 18 x 18 lattice within +-1.5 deg, and energies **uniform in 70-250 MeV**. A
   candidate is kept only if, on the analytic dose of the model grid, its R80 lies at
   least 10 mm before the end of the 320 mm crop and the beam stops inside the
   patient (flux-weighted patient share at the peak at least 0.5). Rejection keeps
   the accepted energies uniform over what each path allows. Each patient gets the
   same quota inside its split; batches hold whole patients, splits interleaved.
   Writes `<store>/plan/` (`plan.csv`, `candidates.csv`, `acceptance.csv`,
   `quotas.csv`, `estimate.json` with the Monte Carlo hours and the storage) and
   refuses to overwrite.
3. **`check-energy-mixing`** (a few minutes of Monte Carlo). Every beamlet has its
   own energy, and a field still runs as one MCsquare beamlet-mode call with one
   control point per energy. This check simulates six planned beamlets of one field
   both ways (one mixed call, single-energy calls) and requires per beamlet a gamma
   pass rate (2%/2 mm) of at least 99%, an integral dose within 1% and R80 within
   1 mm. `generate` refuses to start without a passing check.
4. **`generate`** (days). Per batch: Monte Carlo into `work_root/batch_NNNN/raw`;
   the post-Monte-Carlo checks (Monte Carlo R80 inside the crop with the margin,
   dose maximum inside the patient, 95% of the dose in the crop); packing; moving
   with checksums; a marker in `<store>/state/`; deletion of the raw files. Before
   every batch both disks are checked for the space it needs. Batches with a marker
   are never redone and an interrupted batch resumes beamlet by beamlet, so
   stopping and restarting is safe. `--shard i --n-shards n` splits the batches over
   processes; two processes of 24 threads use the 48 cores better than one of 48.
5. **`status`**: batches done, beamlets stored per split, hours per batch.
6. **`sync`** (later): copies every finished batch and the plan to
   `sync.mirror_root`, each file through a temporary name and checked against the
   sha256 its batch marker recorded; the marker is copied last, so a marker in the
   mirror means the batch is complete there. Files already present with the right
   checksum are skipped, so a sync can be interrupted and repeated, also while
   `generate` is still running. It never deletes from the store.
7. **`stats`** (seconds, index CSVs only): reporting tables and figures into
   `stats.output_dir/pool_stats_<ts>/`, in three views.
   - *Anatomy*: stored beamlets per anatomy, collection and split, the fate of
     every planned beamlet (`fates.csv`), energy, MC R80 and in-body dose spectra.
   - *Patients*: one row per patient (planned, stored, fields, quota, scanner, z
     extent) and the concentration per split: Gini, effective number of patients
     (`1 / sum(p^2)`), share of the largest decile.
   - *Gantry frame* (IEC 61217): beamlets per gantry sector and per spot position
     on the isocentre plane. `theta_x` gives `Y_g` (gantry axis, `d_smy * tan`),
     `theta_y` gives `X_g` (lateral, `d_smx * tan`), the convention of
     `angles_to_spot_position`, whose argument names are crossed.
     `uniformity.csv` compares the spread of the cells with the counting noise of
     a uniform draw (`dispersion` near 1 means noise only). Judge the gantry
     sectors by fields (`gantry_sector_fields`): the angle is drawn per field, so
     beamlet counts per sector vary about five times more than Poisson.

## The noise set (EXP-0014)

Every model's error on a pool beamlet is measured against one Monte Carlo run at
1e6 primaries, so the run's noise is part of every error and is shared by every
model. The noise set measures it on new beamlets simulated twice, at 1e6 and at 1e7
primaries, into two stores of their own:

```bash
uv run python scripts/al_build_beamlet_pool.py noise-plan --config scripts/config_beamlet_pool_noise_1e6.yaml \
    --pair-config scripts/config_beamlet_pool_noise_1e7.yaml
# then, per realisation, the usual two processes:
uv run python scripts/al_build_beamlet_pool.py generate --config scripts/config_beamlet_pool_noise_1e6.yaml \
    --shard 0 --n-shards 2
uv run python scripts/al_build_beamlet_pool.py export --config scripts/config_beamlet_pool_noise_1e6.yaml --split test
```

- **Patients**: `noise.per_anatomy` patients per anatomy, drawn from the test split
  of the registry (out of sample for every engine, and never used by the active
  learning loop).
- **Beamlets**: the pool's own planner (`plan:` with another seed and prefix), so the
  energies, steering and overshoot gate follow the pool's population;
  `noise.beamlets_per_patient` per patient, two fields each. `noise-plan` refuses a
  plan that repeats a beamlet of `noise.source_plan`, and writes the same plan,
  `noise_selection.csv` and the pool's energy-mixing check into both stores.
- **Batches**: one patient per batch, numbered so that each process gets the same
  number of patients of every anatomy.
- **The pair of configs** may differ only in `robustness.num_primaries`,
  `batches.store_root` and `batches.work_root`; `noise-plan` refuses anything else.
- **Seeds**: `robustness.rng_seed` stays 0 in both. In beamlet mode MCsquare runs
  each beamlet on one thread and seeds it at the beamlet's start with `seed + t`
  (thread `t`), so a fixed seed would give every beamlet of a thread the same random
  stream; seed 0 uses `time(NULL) + t`, as the pool did, and the two runs of a
  beamlet start hours apart.
- **Threads**: one beamlet per thread means an MCsquare call lasts at least as long
  as its slowest beamlet on one thread (about 25 minutes at 1e7 for a 200 MeV
  thoracic beamlet). `beamlet_block_size: 42` keeps a field in one call.
- A beamlet counts in the analysis only if both realisations stored it.

## Storage

On Apollo everything lives in one tree, `/scratch/mstryja/adota_pool`:

```
/scratch/mstryja/adota_pool/
  v1_apollo_1e6/      the store (below), about 130 GB at the end
  _work/v1_apollo_1e6/batch_NNNN/   one raw batch per process (about 32 GB), deleted once packed
  _mc_work/           MCsquare's transient run directories (about 16 GB per beamlet block)
  logs/
```

Store and work area share the disk, so a packed shard reaches the store by a rename
(checksummed once), and the free-space check before each batch adds the raw batch,
the transient grids and the packed shards together against the disk's free space.

```
<store>/
  plan/        plan.csv, candidates.csv, acceptance.csv, quotas.csv, estimate.json,
               energy_mixing_check.csv
  train/       batch_NNNN.h5      one group per beamlet (its candidate id)
  validation/  batch_NNNN.h5
  test/        batch_NNNN.h5
  index/       batch_NNNN.csv     every planned beamlet and its fate
  state/       batch_NNNN.json    finished marker: counts, timings, checksums
```

A beamlet group holds `ct` (int16) and `dose` (float32) at 1 mm, `(60, 60, 320)`,
gzip-compressed and lossless, and the generator's `sim_res` as a JSON attribute. The
flux channel is not stored: it is a pure function of the metadata, and
`src.active_learning.beamlet_pool.store.read_record` recomputes it, identical to the
generator's to the last bit (checked on every packed beamlet, and on saved
energy-gantry beamlets). About 2.6 MB per beamlet against 16 MB in the generator's
own format: 50,000 beamlets take about 130 GB.

## DelftBlue

`scripts/config_beamlet_pool_delftblue.yaml` and
`scripts/slurm/beamlet_pool_delftblue.sbatch` are templates. Before the first job:

- every `<...>` path in the config: the DICOM collections, a copy of the v3 index
  CSV, the MCsquare install (`provision_mcsquare.sh`), the work and store
  directories (check which file systems the compute nodes can write to);
- the committed registry (do **not** rerun `screen` there: the registry is the
  shared split);
- `plan` once (it needs the DICOM data), then `check-energy-mixing` at 1e7;
- the account in the job script; `sbatch --array=0-9 ...`. A task that hits the
  time limit is resumed by resubmitting the same array.
