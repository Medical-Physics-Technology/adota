# `al_build_beamlet_pool.py`: a labelled beamlet pool on new patients

`scripts/al_build_beamlet_pool.py` generates Monte Carlo beamlets on patients the
training set has not used, split by patient into train / validation / test. The
train split is the pool the active-learning loop samples from; validation and test
are held out. It runs in batches: a batch is simulated on fast local storage,
packed into compressed shards, moved to its final storage and deleted locally, so
a multi-day run never holds more than one raw batch per process.

| Run | Config | Beamlets | Primaries | Where |
|---|---|---|---|---|
| Apollo | `scripts/config_beamlet_pool.yaml` | 50,000 | 1e6 | `/RadiotherapyData/adota_pool/v1_apollo_1e6` |
| DelftBlue | `scripts/config_beamlet_pool_delftblue.yaml` (template) | 100,000 | 1e7 | set in the config |

Both runs share **one patient registry** (`registry/beamlet_pool_patients_v1.csv`,
committed), so a patient is in the same split on both machines. Their plans differ
(own seed), so no beamlet is simulated twice.

## Stages

Every stage is a separate command and is started by hand.

```bash
export TMPDIR=/scratch/$USER/tmp        # the scripts refuse a root-disk TMPDIR
uv run python scripts/al_build_beamlet_pool.py screen   --config scripts/config_beamlet_pool.yaml
uv run python scripts/al_build_beamlet_pool.py plan     --config scripts/config_beamlet_pool.yaml
uv run python scripts/al_build_beamlet_pool.py check-energy-mixing --config scripts/config_beamlet_pool.yaml
uv run python scripts/al_build_beamlet_pool.py generate --config scripts/config_beamlet_pool.yaml --shard 0 --n-shards 2
uv run python scripts/al_build_beamlet_pool.py generate --config scripts/config_beamlet_pool.yaml --shard 1 --n-shards 2
uv run python scripts/al_build_beamlet_pool.py status   --config scripts/config_beamlet_pool.yaml
```

1. **`screen`** (DICOM headers only, minutes). Every CT of the four collections
   (Lung-PET-CT-Dx, NSCLC-Radiomics, StageII-Colorectal, Prostate-AEC):
   - image-quality gates of `TCIADataset` (spacing, at least 260 mm of z for the
     +-1.5 deg steering window);
   - its 1 mm grid against the 56 scans of the training HDF5 (`*_index.csv`): the
     training set records a scan only by that grid, so a match excludes the patient.
     The command prints how many training scans were matched; a training scan that
     matches no patient here comes from a collection that is not on this machine,
     and the leakage check cannot cover that collection;
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

## Storage

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
