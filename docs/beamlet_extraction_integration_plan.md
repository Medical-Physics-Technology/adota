# Plan-level ADoTA pipeline integration plan (extract, infer, accumulate, compare)

Status: **task list awaiting approval** (section 6). Decisions in section 2 are
confirmed by the user (2026-06-11).

Goal: one end-to-end script in this repository that, given an OpenTPS plan
directory (PlanPencil.txt + CT.mhd + bdl.txt + config.txt + Dose.mhd), will:

1. **Extract** per-spot ADoTA inputs (BEV CT crops + Gaussian flux projections),
2. **Infer** the per-spot dose with the ADoTA model (in-process),
3. **Accumulate** predicted doses back into the full grid (per gantry angle,
   derotate, sum), and
4. **Compare** the accumulated dose against the MCsquare reference
   (`Dose.mhd` / `Dose.raw`, READ-ONLY, never modified).

The source functionality lives in a Jupyter notebook plus the external
`datagenerator` repository (`/home/mstryja/projects/datagenerator`, read-only
reference, importable for parity tests). The user suspects a **geometric error**
somewhere in extraction and/or accumulation; finding and fixing it is an
explicit objective, not a side effect.

---

## 1. What already exists in this repo (verified)

- `src/loaders/plan_parser.py` :: `parse_plan`. API-identical to
  datagenerator's parser; all six plans under `/scratch/mstryja/opentps_plans/`
  parse with it.
- `scripts/inference_worker.py` + `src/loaders/dir_based.py`
  (`get_single_record_no_gt`, `save_prediction`): the complete per-spot
  inference path. It consumes exactly the files stage 1 produces
  ((60, 60, 320) crops; permute to (320, 60, 60); trilinear resize to
  (160, 30, 30); model; upsample back to (320, 60, 60); de-normalize).
- `scripts/run_plan_opentps.py` (renamed via `git mv` from `run_model_hptc.py`):
  the end-to-end CLI. Currently a skeleton that loads the model, loads the plan
  directory (`src/loaders/plan_directory.py`: CT + contours via SimpleITK,
  `PlanPencil.txt`, `bdl.txt`, `config.txt`, MC `Dose.mhd` path), parses the
  plan, and prints a preview. The spot-expansion / `--stages` pipeline (T8-T13)
  is not built into it yet. Config: `scripts/config_run_plan_opentps.yaml`.
- `src/loaders/plan_directory.py` :: `load_plan_directory`, `PlanDirectory`,
  `parse_opentps_config`. Unit-tested in `tests/loaders/`.
- `src/utils/serialization.py` :: `NumpyEncoder`.
- `src/evaluation/cli.py` :: `resolve_device`, logging/run-dir helpers,
  `merge_config`.
- All third-party deps (SimpleITK, pandas, pydicom, scipy, torch, typer) are
  already project dependencies.

Reference data (verified on disk):

- Primary test plan: `/scratch/mstryja/opentps_plans/`
  `Prostate-AEC-001_Publication_Prostate-AEC-001_100MperBeam/`
  (1924 spots, gantry -90/90, CT 465x465x358 @ 1 mm, plan-local `bdl.txt`
  and `config.txt` with `Num_Primaries 200000000`, MC `Dose.mhd`/`Dose.raw`).
  The user will provide additional simple test plans later.
- An older extraction exists at `LUNG1-221_.../spot_predictions/` (2272 spots,
  same crop shape/schema, deterministic ids): useful secondary reference, with
  the caveat that it predates possible bug fixes.

## 2. Confirmed decisions

1. **Scope: end-to-end.** All four stages in one script, each stage also
   runnable separately (`--stages extract,infer,accumulate,compare`).
2. **BDL is NOT committed to the repo** (fine-tuned, request-only). The BDL
   path is configuration; the DEFAULT is the plan-local `<plan_dir>/bdl.txt`
   (verified to contain Nozzle/SMX/SMY and the energy table that
   `bdl_extract_geo` parses). README states the BDL must be requested.
3. **Package name:** `src/beamlets/`.
4. **Test plan:** `Prostate-AEC-001_Publication_Prostate-AEC-001_100MperBeam`.
5. **Out-of-bounds crops: pad with air (-1024 HU)** instead of failing or
   silently mis-shaping. (Accumulation correspondingly adds only the
   in-bounds overlap.)
6. **`run_model_hptc.py` is renamed** (`git mv`) to `run_plan_opentps.py` and
   the pipeline is built into it. Its DICOM-CT loader is kept but dormant
   (the pipeline reads `CT.mhd`, which is the grid MCsquare actually used).
7. **Deterministic spot ids:** `b{beam:02d}_l{layer:03d}_s{spot:04d}`
   (beam = field index, layer = control-point index, spot = spot index).
   Reruns become idempotent; matching against references is trivial.
8. **All outputs on `/scratch`** (beamlet files, predictions, accumulated
   dose, run dirs, goldens). Nothing heavy under `/home`.
9. **Working rule unchanged:** one task at a time, each implemented only after
   approval; correctness first, performance later; suspected bugs flagged to
   the user immediately with the decision made together.

## 3. Suspected geometric error sources (flagged immediately, ranked)

These are hypotheses from code reading, NOT confirmed bugs. Each gets a
dedicated experiment in T7; fixes are decided with the user.

- **S1. Wrong BDL used for non-HPTC plans (very likely a real bug).** The
  notebook calls `bdl_extract_geo(None)`; a bare `except` then loads the
  hardcoded HPTC BDL for EVERY plan. Prostate-AEC-001's plan-local `bdl.txt`
  DIFFERS from the HPTC file. Wrong `d_smx`/`d_smy` bends every beamlet ray
  and every `spot_position_to_angles` result; wrong `SpotSize1x/1y` distorts
  every flux sigma. Fix: required explicit BDL path, default plan-local.
- **S2. Physical-pivot frame mismatch in the 2D rotation.** The notebook
  computes the pivot as `(max_range_x - iso_x, iso_y, iso_z)` from
  `size * spacing` WITHOUT the grid origin, then calls
  `rotate_image_batch_torch_physical` with default `origin=(0,0)`,
  `spacing=(1,1)`. With CT origins like (-233, -373, -1244) mm, the pivot the
  function actually rotates around may be offset from the true isocenter
  unless the omissions cancel exactly. Must be pinned by a synthetic test
  (marker at a known physical point, rotate, assert landing voxel).
- **S3. The x-flip convention.** `pivot_x = max_range_x - iso_x` ("grid
  flipped in x") plus the crop taking depth `x in [0, 320)` from the x=0 face
  assumes a specific export orientation. If a plan is exported unflipped, the
  beam enters from the wrong face. Make it an explicit, tested config flag.
- **S4. Half-voxel isocenter shift.** `isocenter_physical = origin + iso -
  spacing/2` is one of three plausible conventions (the notebook contains two
  commented-out alternatives, evidence of past uncertainty). A 0.5 voxel
  systematic shift is exactly the kind of subtle error described.
- **S5. Sign-convention chain.** Plan-side `(-1)*(gantry_angle - 90)`, the
  fixed `-90` inside the `beamlet_ray` call, and the internal negation
  `angle_rad = -angles` in the rotation function must compose to identity.
  Validated end-to-end on a synthetic asymmetric phantom.
- **S6. Entrance-point frame juggling.** `rays_entrence_point` mixes
  full-grid x with crop-relative y/z (via `shifted_origin`), then is permuted
  `[re[1], re[2], re[0]]` before `flux_projection`. Off-by-one or axis-order
  slips here displace the flux Gaussian from the true ray.
- **S7. Dose normalization for the MC comparison.** The notebook's output
  filenames ("new_scaling", "mu_scaling") show the absolute scaling between
  accumulated ADoTA dose (de-normalized eV/g per simulated-primary
  convention of the training data, times `relative_weight`) and the MC grid
  (simulated with `Num_Primaries` from config.txt, `ProtonsMU` in the BDL)
  was never settled. T12 derives it explicitly and the user signs it off.

Parity note: A/B tests against datagenerator prove a FAITHFUL PORT only.
Where T7 confirms a bug and we fix it, the parity test for that function is
inverted into a regression test asserting the documented difference.

## 4. Source-to-target port map

| Source | Functions | Target |
|---|---|---|
| `datagenerator/geometry/geometry_spatial_operations.py` | `beamlet_ray`, `rotate_vector`, `intersect_line_with_cube`, `check_if_point_in_cube` | `src/beamlets/geometry.py` |
| `datagenerator/utils/dataset_generation_utils.py` | `bdl_extract_geo` (as `BeamDataLibrary`) | `src/beamlets/bdl.py` |
| same | `flux_projection`, `flux_spatial_spread`, `spot_position_to_angles`, `angles_to_spot_position` | `src/beamlets/flux.py` |
| same | `cropp_around_spatial_point_with_np_indexes`, `get_roi_with_indexes_ct_only` | `src/beamlets/cropping.py` (air-padded OOB) |
| notebook (inline) | `rotate_image_batch_torch_physical` | `src/image_processing/rotation.py` |
| notebook + `run_plan_opentps.py` (duplicated) | plan -> spot dicts, group by gantry angle | `src/beamlets/plan_spots.py` |
| notebook 2nd half | dose accumulation, derotation, `Dose_adota.mhd` export | `src/beamlets/accumulation.py` |

Not ported: MC simulation runners/datasets, `resample_*`/`instant_rotate`
(superseded by the torch rotation), `get_roi` (CT+dose variant),
`ranges_of_bbox_dc` (dead call), `reduce_vacum_values_to_air` (disabled in the
notebook; can come later behind a flag), matplotlib cells, the notebook's
`NumpyEncoder` (exists in repo).

## 5. Test policy (user requirement: a unit test for EVERY functionality)

Every ported or new function gets unit tests in the same task that adds it.
Three test layers:

1. **Pure unit tests** (synthetic, no external data): mathematical invariants,
   edge cases, axis-order checks, error paths.
2. **A/B parity tests** vs datagenerator (skipif-gated on the repo path):
   faithful-port evidence, function by function.
3. **Geometric ground-truth tests** (the decisive layer, special focus on
   cropping): synthetic sitk phantoms with non-trivial origin/spacing and
   markers at known PHYSICAL coordinates; assertions that markers land at the
   analytically expected voxels after rotate/crop, and that
   extract -> accumulate round-trips a synthetic dose back to the correct
   grid voxels at the correct weight.

End-to-end: golden capture (under `/scratch/mstryja/tmp_adota/golden/`,
existing harness) of a fixed spot slice of the test plan, plus the final
MC-dose comparison metrics as the physical acceptance gate.

---

## 6. Task list (T1 to T15)

Each task: what to do / objective / acceptance criteria. One task at a time,
implemented only after the user approves that task's start; any in-task
surprise is flagged before proceeding.

- **T1. Foundations: conventions doc + test-asset registration.**
  Do: write `src/beamlets/__init__.py` docstring (or a short
  `docs/beamlet_conventions.md`) fixing the coordinate conventions once:
  sitk index order (x, y, z) vs numpy array order (z, y, x), BEV convention
  (grid pre-rotated so the beam axis is -90, along x), gantry adjustment
  `(-1)*(ga - 90)`, id scheme `b/l/s`. Register the test plan path and the
  `/scratch` output policy in the config template.
  Objective: every later task cites one written convention instead of
  re-deriving axis orders.
  Acceptance: doc reviewed by user; conventions match what T2's fixtures
  implement.

- **T2. Synthetic geometric fixture harness.**
  Do: `tests/beamlets/conftest.py` fixtures that build sitk phantoms with
  arbitrary origin/spacing/size, single-voxel and sphere markers at given
  PHYSICAL coordinates, asymmetric patterns (to catch flips), plus a tiny
  in-memory `Plan` builder (1 field, few spots).
  Objective: the ground-truth instrument every geometry test uses.
  Acceptance: fixture self-tests pass (marker voxel index round-trips through
  `TransformPhysicalPointToIndex` for non-trivial origins/spacings).

- **T3. Port `src/beamlets/geometry.py`.**
  Do: port `rotate_vector`, `beamlet_ray`, `check_if_point_in_cube`,
  `intersect_line_with_cube` (cleaned, typed, logged).
  Objective: ray geometry available and trusted.
  Acceptance: unit tests (Rodrigues preserves norm/angle; zero-shift spot ray
  passes through the isocenter; intersection on synthetic cubes incl.
  start-inside and edge cases) + A/B parity vs datagenerator, all green.

- **T4. Port `src/beamlets/bdl.py` + `src/beamlets/flux.py` (fixes S1).**
  Do: `BeamDataLibrary.from_file(path)` (explicit required path, parse once:
  energy table + (d_nozzle, d_smx, d_smy); clear FileNotFoundError);
  `spot_position_to_angles` / `angles_to_spot_position` taking
  (d_smx, d_smy) as arguments (no per-spot file reads);
  `flux_spatial_spread`, `flux_projection`.
  Objective: BDL handling correct per plan; flux math ported and trusted.
  Acceptance: parses both `bdl.txt` (plan-local) and the HPTC file; angle
  round-trip identity; flux peak lies on the ray entrance, sigma matches the
  BDL row of the closest energy; A/B parity (parity for angles/flux is
  checked feeding the SAME BDL values to both implementations); all green.

- **T5. Port 2D physical-pivot rotation (pins S2).**
  Do: add `rotate_image_batch_torch_physical` to
  `src/image_processing/rotation.py`, typed and documented, with explicit
  spacing/origin semantics.
  Objective: rotation primitive whose pivot semantics are PROVEN, not assumed.
  Acceptance: identity at 0 deg; forward+backward returns input (interp
  tolerance); known 90 deg case on an asymmetric phantom; marker at physical
  point P stays FIXED when P is the pivot, for grids with non-zero origin and
  non-unit spacing (this is the S2 experiment); CPU == GPU; A/B parity vs the
  notebook function captured verbatim in the test.

- **T6. Port `src/beamlets/cropping.py` with air padding (fixes notebook OOB
  bug; pins S4; SPECIAL-FOCUS task).**
  Do: `crop_around_spatial_point` returning (crop, crp_indexes) with
  out-of-bounds regions filled with -1024 HU and an `oob_mask`/flag in the
  metadata; `extract_beamlet_roi` (port of `get_roi_with_indexes_ct_only`,
  dead `ranges_of_bbox_dc` call removed).
  Objective: 100% trusted sub-volume extraction.
  Acceptance: marker at known physical point lands at the analytically
  expected crop voxel for in-bounds, edge, and corner cases; OOB regions are
  exactly -1024 and in-bounds content is bit-identical to the unpadded crop;
  crp indexes returned are consistent with where the crop content sits in the
  full grid (verified by re-inserting the crop and diffing); axis-order tests
  (z, y, x vs x, y, z) explicit; A/B parity on in-bounds cases; all green.

- **T7. Geometric chain validation: the error hunt (S1 to S6).**
  Do: dedicated experiments, one per suspect, on synthetic phantoms AND on a
  small slice of the real plan: (a) BDL geometry values plan-local vs HPTC
  and their effect on rays/angles/flux (S1); (b) pivot landing test with the
  real CT origin (S2); (c) flip-flag behavior on both orientations (S3);
  (d) the three isocenter-shift conventions compared, marker-based (S4);
  (e) full sign-chain test: synthetic asymmetric phantom, known gantry angle,
  assert the crop contains the expected anatomy (S5); (f) entrance-point /
  flux placement test: flux maximum must trace the analytic ray inside the
  crop (S6). Write findings to `docs/beamlet_geometry_findings.md`.
  Objective: the suspected geometric error located and characterized.
  Acceptance: findings doc with a verdict per suspect (confirmed bug / not a
  bug / cannot reproduce), reviewed WITH the user; agreed fixes implemented
  in the respective modules with regression tests; parity tests inverted
  where behavior intentionally changed.

- **T8. `src/beamlets/plan_spots.py`.**
  Do: `expand_plan_to_spots(plan)` (the loop currently duplicated in the
  notebook and `run_model_hptc.py`), `group_by_gantry_angle`, deterministic
  ids `b{beam:02d}_l{layer:03d}_s{spot:04d}`.
  Objective: single source of truth for plan expansion.
  Acceptance: unit tests on a synthetic plan (counts, relative weights sum to
  1 over the plan, id uniqueness/stability) and on the real test plan
  (1924 spots, gantry set {-90, 90}); parity with the inline loops.

- **T9. Stage 1: extraction library + CLI section.**
  Do: `run_extraction(plan_dir, output_dir, config) -> manifest` orchestrating
  rotate-per-gantry-angle, per-spot crop + flux + JSON (same schema as today,
  plus `crp_numpy_ct`, `relative_weight`, `oob` flag), timing summary;
  outputs under `/scratch` (default `<plan_dir>/beamlets_adota/`, refuse
  non-empty without `--overwrite`).
  Objective: stage 1 runnable and reproducible.
  Acceptance: smoke run on a fixed slice (one gantry angle, first N spots) of
  the test plan produces N x 3 files with correct shapes/keys; golden capture
  of that slice; deterministic ids stable across reruns; unit tests for the
  orchestration (manifest, overwrite guard, /scratch path resolution).

- **T10. Stage 2: in-process ADoTA inference.**
  Do: `run_inference(spot_dir, model, scale, device, ...)` reusing
  `get_single_record_no_gt` + `save_prediction` (batch = 1, correctness
  first), checkpoint/hyperparams from config.
  Objective: predictions without the subprocess hop, same numerics as the
  existing worker.
  Acceptance: on the same spot files, outputs are numerically identical
  (within float tolerance) to `inference_worker.py` run in this venv;
  `{id}_ds_pred.npy` shapes/units as documented; unit test with a tiny
  random-weight model for the plumbing.

- **T11. Stage 3: accumulation (`src/beamlets/accumulation.py`).**
  Do: port the notebook's second half: per gantry angle add
  `pred * relative_weight` at `crp` indexes into the rotated phantom (with
  the air-padding-aware overlap handling matching T6), derotate, sum angles,
  clip negatives, export `Dose_adota.mhd` with the MC grid's metadata to
  `/scratch`. Remove the dead padding-removal block if confirmed dead.
  Objective: trusted crop-to-grid inverse of stage 1.
  Acceptance: synthetic round-trip: a delta/sphere dose placed in a crop
  lands at the analytically correct grid voxels with the correct weight,
  including edge crops; extract -> accumulate identity test on a synthetic
  phantom (accumulating the CROPS back reproduces the rotated grid region);
  derotation returns markers to pre-rotation voxels; `Dose.mhd`/`Dose.raw`
  verified untouched (checksum before/after).

- **T12. Stage 4: MC comparison (resolves S7 WITH the user).**
  Do: load `Dose.mhd` read-only; derive the normalization chain explicitly
  (training-data dose units, `to_gy`, `relative_weight`, `Num_Primaries`
  from config.txt, `ProtonsMU` from the BDL) and present it to the user for
  sign-off BEFORE coding the comparison; then compute global metrics (RMSE,
  MAPE on thresholded dose, 3D gamma via existing `src/metrics/`), depth/
  lateral profiles, and report (CSV + figures) into the run dir.
  Objective: quantitative, physically meaningful comparison against MC.
  Acceptance: normalization derivation document approved by user; metrics
  computed on the test plan; MC files untouched (checksums); unit tests for
  the metric plumbing on synthetic pairs.

- **T13. The end-to-end CLI: `scripts/run_plan_opentps.py`.**
  Do: `git mv scripts/run_model_hptc.py scripts/run_plan_opentps.py`; build
  the typer CLI per repo conventions (YAML config
  `scripts/config_run_plan_opentps.yaml`, `merge_config`, `resolve_device`,
  run dir + logging on `/scratch`): `--stages extract,infer,accumulate,compare`
  (default all), `--plan-dir`, `--bdl-path` (default plan-local), `--n-spots`
  /`--gantry-angle` subset options for cheap runs; README references updated.
  Objective: one reproducible entry point for the whole pipeline.
  Acceptance: each stage runnable alone and chained; subset run on the test
  plan completes end to end; config copied to run dir; script README section
  drafted; all prior tests still green.

- **T14. End-to-end validation on Prostate-AEC-001.**
  Do: full-plan run (all 1924 spots); compare vs MC with the T12 metrics;
  review results with the user against expectations; iterate on geometric
  fixes if the comparison exposes residual error (back to T7 findings);
  capture the end-to-end golden (subset) for CI-style regression.
  Objective: the integrated pipeline demonstrably reproduces plan dose to an
  accepted accuracy.
  Acceptance: user-approved comparison report; goldens green on rerun;
  acceptance metric thresholds recorded in the findings doc (set by the user
  when the first numbers are on the table).

- **T15. Documentation + changelog + suite.**
  Do: `scripts/README.md` section (style-matched), changelog entry (1.2.0),
  memory/plan-doc status updates, full `pytest` green (unit + parity +
  golden, with clean skips where data/repos are absent).
  Objective: the work is discoverable, documented, and protected.
  Acceptance: docs reviewed; full suite green; changelog entry approved.

## 7. Task-to-suspect coverage

| Suspect | Pinned/fixed in |
|---|---|
| S1 wrong BDL | T4 (fix), T7a (impact assessment) |
| S2 pivot frame | T5 (semantics), T7b (real-CT check) |
| S3 x-flip | T6/T9 (flag), T7c |
| S4 half-voxel shift | T6, T7d |
| S5 sign chain | T7e (synthetic E2E), T11 (derotation) |
| S6 entrance/flux frames | T4, T7f |
| S7 dose normalization | T12 |
