# Patient registry + provenance QC — design

Status: **proposal** (2026-08-21). Scope grew out of the multi-patient robustness /
active-learning work. Goal: make patient selection **leakage-safe, provenance-first,
and multi-dataset** from the start, and extend DICOM quality gating.

## Requirements (from the discussion)

1. **No data leakage** between train / test (and the AL pool) — across *all* datasets
   and *all* experiments, enforced, not by convention.
2. **Multi-dataset / multi-anatomy**, extensible: thoracic (NSCLC-Radiomics,
   Lung-PET-CT-Dx), abdominal (StageII-Colorectal), pelvic (Prostate-AEC), and
   incoming H&N, Prostate, Brain.
3. **Extended quality gating** beyond slice-thickness / pixel-spacing: add kVp, tube
   current, and other acquisition parameters; record them as provenance.
4. **Provenance carried downstream**: every generated beamlet already stamps
   `dataset/anatomy/patient/series`; add the acquisition params so quality is
   traceable per sample.
5. **AL-aware**: an explicit unassigned *pool* the acquisition function draws from
   and marks as consumed.
6. **Start small**, but architected for the above from day one.

## What exists today

- `src/datasets/tcia.py :: _select_ct_series` gates on: dcm count in `[min,max]`,
  `Modality==CT`, `PhotometricInterpretation==MONOCHROME2`. It does **not** check
  spacing/thickness/kVp, and none reach `CTRecord`.
- `CTRecord` provenance = dataset / anatomy / patient_id / series_uid / series_dir /
  n_slices. Stable `uid = dataset/patient/series_uid`.
- The **datagenerator** had a flat "registry": `qualified_indexes_cts.npy`, built by
  `get_qualified_indexes` with `x,y spacing < 1.5 mm`, `z spacing < 2 mm`,
  `z-slices > 70`. **Training used the first indexes; every test / robustness script
  used `qualified_indexes[-16:]` (the last).** This is exactly the train=first /
  test=last split the registry must encode (and supersede).

## Core concepts

- **Global patient key**: `{dataset}::{patient_id}` (e.g. `Lung-PET-CT-Dx::Lung_Dx-G0062`).
  The single leakage primitive. A CT is `key + series_uid` (a patient may have >1 series).
- **Role**: exactly one per key, enforced —
  `train | test_beamlet | test_plan | val | al_pool | excluded`.
  (`excluded` carries a QC-fail or manual reason.)
- **Provenance record** per series: dataset, collection, anatomy, patient_id,
  series_uid, n_slices, spacing (x,y,z), slice_thickness, kVp, tube_current
  (XRayTubeCurrent / Exposure), manufacturer, model, convolution_kernel,
  study/series dates, `qc_pass`, `qc_reasons[]`.

## QC vs. role assignment (authoritative role)

QC has **two separate jobs** that must not be conflated:

1. **Provenance recording** -- always, for every patient/series (informational). We
   record spacing / kVp / tube-current / kernel even for data we keep.
2. **Candidate filtering** -- gates apply only when selecting **new** patients (the
   `al_pool`, a new experiment). They never retroactively exclude an already-used
   patient.

**Role assignment is authoritative and overrides QC.** A patient locked as `train`
(or `test_*`) stays in that role even if it would fail today's gates. Concretely:
the **StageII-Colorectal** training data was sampled **without gating** (uniform
5 mm slices -- a deliberately *homogeneous* training subset); it must remain in
`train`. The registry stores each patient's `role`, its recorded QC provenance, and
a `qc_pass` flag *evaluated but not enforced* for already-assigned patients. So we
can report "this training patient is 5 mm / ungated" without dropping it. Gates are
thus a property of a **selection query**, not of a patient.

Implication for seeding (Phase 2): import the known train patients by role first;
QC provenance is attached but does not filter them. Only the unassigned remainder
becomes the QC-gated `al_pool`.

## Components (new)

1. `src/provenance/dicom_qc.py` — pure functions: `read_acquisition_params(series_dir)`
   → dict; `check_quality(params, gates)` → `(qc_pass, reasons)`. Gates configurable
   (per-dataset overrides): `max_spacing_xy`, `max_spacing_z`, `min_z_slices`
   (datagenerator defaults 1.5 / 2 / 70), plus optional `kvp_range`,
   `min_tube_current`, `exclude_kernels`. Reused by `TCIADataset` **and** the builder,
   so QC lives in one place.
2. `src/registry/patient_registry.py` — `PatientRegistry`: `load/save`;
   `role_of(key)`; `assign(key, role, reason, source)` (refuses to silently move a
   key between train/test); `select(anatomy, n, role='al_pool', exclude={train,test_*})`
   (leakage-safe draw); `upsert(record)`. One-role-per-key invariant enforced.
3. `scripts/registry/build_registry.py` — scan configured datasets → QC + provenance
   → upsert into the registry. Idempotent / incremental (skips scanned series).
4. `scripts/registry/seed_splits.py` — lock the **known** train / test patients:
   recover the datagenerator split (map `qualified_indexes_cts.npy` + the LungDataset
   ordering → patient/series keys; train = head, test_beamlet = `[-16:]`), and pin the
   already-run test patients (thoracic Lung-PET-CT-Dx_768, pelvic
   StageII-Colorectal-CT_168, plan-reconstruction NSCLC-Radiomics patients).

## Store format

**CSV + YAML, under git** (recommended over SQLite): `registry/patients.csv` (one row
per series, provenance + qc + role) and `registry/splits.yaml` (explicit role
assignments / manual overrides, human-authored). Diffable, reviewable, no DB
dependency, reproducible. `PatientRegistry` is the only writer.

## Integration

- **CTRecord**: add optional `provenance: dict = None` (acquisition params + qc).
  Non-breaking. `generate_for_record` already stamps `rec.*` into `sim_res`; add
  `rec.provenance` so every beamlet's `_sim_res.json` carries kVp / spacing / etc.
- **Selection**: add `selection: registry` to the dataset config → the builder calls
  `registry.select(anatomy, n, exclude={train, test_*})`. `first|last|random` stay for
  ad-hoc use. This replaces the interim `selection: last` leakage guard.
- **CI guard**: a test asserting no key holds two roles, and that no experiment output
  dir references a `train` key.

## Phased delivery (start small)

- **Phase 1 (DONE 2026-08-21):** `src/provenance/dicom_qc.py` (`QCGates`,
  `params_from_header`, `check_quality`, `gates_from_dict`) + `CTRecord.provenance`
  + wired into `TCIADataset` (provenance always recorded; spacing/kVp/tube-current
  gates opt-in via `qc:` config) + provenance carried into every beamlet's
  `sim_res["ct_provenance"]` + `selection: registry` deferred. Tests:
  `tests/provenance/test_dicom_qc.py`. Verified on real data: NSCLC-Radiomics is
  natively 3.0 mm (needs `max_spacing_z: 3.5`), Lung-PET-CT-Dx is <2 mm (the thin
  training source), abdominal 5 mm (relaxed to `max_spacing_z: 6.0`).
- **Phase 2:** `PatientRegistry` + `build_registry.py`; scan the 4 current datasets →
  `registry/patients.csv`. Seed known splits (`seed_splits.py`).
- **Phase 3:** `selection: registry` (leakage-safe) in the generation configs;
  deprecate ad-hoc `last` for production runs.
- **Phase 4:** AL hooks — `al_pool` queries + mark-on-acquisition; tie to the
  acquisition-function work.

## Open decisions

- Store format: **CSV+YAML** (recommended) vs SQLite.
- Recover exact datagenerator train/test IDs now (needs `qualified_indexes_cts.npy`;
  not found in the repo yet — may need regeneration via `get_qualified_indexes`) or
  defer to Phase 2.
- Key granularity: `{dataset}::{patient_id}` (recommended) vs including `series_uid`
  in the leakage key (a patient with multiple series → still one person, so leakage
  key should be patient-level; series is provenance).
- QC thresholds per anatomy (abdominal StageII-Colorectal is 5 mm slices — fails the
  2 mm z-gate; it was accepted with pixel<1.5 only). Registry records the actual
  values + a per-dataset gate override.
