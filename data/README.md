# Data Directory

This directory contains input data for ADoTA model evaluation.

## Downloading Example Data

Example input data can be downloaded from Google Drive:

🔗 **[Download Example Data](https://drive.google.com/drive/folders/186lp3FsIGeJB_36RBqi8f_5mlPjapaz6?usp=sharing)**

### Setup Instructions

1. **Download** the `example_inputs` folder from the Google Drive link above.

2. **Extract/Copy** the contents to this directory:
   ```
   data/
   └── example_inputs/
       ├── <uuid>_ct.npy
       ├── <uuid>_ds.npy
       ├── <uuid>_flux.npy
       ├── <uuid>_sim_res.json
       └── ...
   ```

3. **Verify** the structure by checking that files exist:
   ```bash
   ls data/example_inputs/
   ```

## Data Format

Each sample consists of the following files (identified by a unique UUID):

| File | Description |
|------|-------------|
| `<uuid>_ct.npy` | CT scan data (3D numpy array) |
| `<uuid>_ds.npy` | Ground truth dose distribution (3D numpy array) |
| `<uuid>_flux.npy` | Particle flux data (3D numpy array) |
| `<uuid>_sim_res.json` | Simulation metadata (energy, beamlet angles, etc.) |

## Using Your Own Data

To use your own data, ensure your files follow the naming convention above and contain:

- **CT data**: Hounsfield units, shape `(D, H, W)`
- **Dose data**: Dose values in eV/g/proton, shape `(D, H, W)`
- **Flux data**: fast beamlet shape projection along its central axis, shape `(D, H, W)`
- **Simulation results JSON**: Must contain at least `"energy"` or `"initial_energy"` field

## Beamlet Training HDF5 Format

The training dataset is stored as an HDF5 file holding one group per beamlet record.
Two versions exist: v2 (the original) and v3 (with full provenance).

### v2 Format

Holds 71,095 records; the 952 ids of the exclusion list are still inside it and every
reader drops them at load time. Per record (group `<id>`):

**Datasets** (each float32, shape `(40, 40, D/2)`, gzip level 4, auto-chunked):
- `ct`: CT, min-max normalised with `min_ct`/`max_ct`
- `dose`: dose, min-max normalised with `min_ds`/`max_ds`
- `flux`: the analytical single-Gaussian beamlet flux projection (not normalised)

**Attributes (six):**
- `id`: record id (str)
- `gantry_angle`: patient-frame gantry angle (float64)
- `initial_energy`: normalised energy (float64)
- `beamlet_angles`: beam angles (ndarray float64, shape (2,))
- `dose_deposition_ratio`: fraction of dose inside the body (float64)
- `stat_uncertainty`: Monte Carlo statistical uncertainty (float64, often NaN)

### v3 Format

Holds the same 70,143 retained records (71,095 minus 952 excluded ids), with datasets
byte-for-byte identical to v2 on every retained record. Adds full provenance.

**Additional record attributes:**
- `schema_version`: 3 (int64)
- `source_dataset`: `"trainset_pelvis"` or `"initial_test_one_ct"` (str)
- `energy_mev`: raw beam energy in MeV (float64)
- `gantry_angle_sim`: in-grid beam direction (float64)
- `isocenter_mm`: beam isocenter (ndarray float64, shape (3,))
- `image_origin_mm`: CT origin (ndarray float64, shape (3,))
- `image_size_vox`: CT size in voxels (ndarray int64, shape (3,))
- `image_spacing_mm`: CT spacing in mm (ndarray float64, shape (3,))
- `roi_size_vox`: region of interest size (ndarray int64, shape (3,))
- `bixel_shift_xy_mm`: bixel grid shift (ndarray float64, shape (2,))
- `ray_entrance_mm`: beam entrance point (ndarray float64, shape (3,))
- `ray_entrance_proj_mm`: flux entrance point (ndarray float64, shape (3,))
- `num_primaries`: number of primaries in Monte Carlo (float64)
- `flux_model`: `"SingleGaussian"` (str)
- `flux_compute`: compute path and device (str)
- `bdl_file`: beam data library filename (str)
- `bdl_sha256`: SHA-256 of the BDL file (str)
- `flux_sigma_xy_mm`: beam model sigmas for this energy (ndarray float64, shape (2,))
- `downsample_method`: `"average"`, `"linear"` or `"trilinear"` (str)
- `patient_key`: hash of image geometry (str, 16 chars)
- `spot_key`: hash of spot definition (str, 16 chars)
- `metadata_json`: raw JSON text (str)

**File-level attributes:**
- `schema_version`: 3
- `created_utc`: build timestamp (ISO 8601)
- `generator`: builder command and git hash
- `python`, `numpy`, `torch`, `h5py`, `cuda`: version strings
- `gpu_name`: GPU model or `"none"`
- `scale_json`: normalisation scale (JSON)
- `bdl_file`, `bdl_sha256`: BDL file and hash
- `flux_compute`: compute path used
- `flux_batch`: batch size
- `downsample_method`: downsampling method
- `exclusion_list_path`: path to the exclusion list
- `exclusion_list_sha256`: SHA-256 of the list
- `n_excluded`, `n_excluded_not_found`: excluded id counts
- `sources_json`: per-source build statistics
- `n_records`, `n_skipped`: record counts
- `skip_log_path`: path to the skip CSV
- `index_path`: path to the index CSV
- `flux_equivalence`: always `"unchecked"` in the file; the checker writes its verdict
  to its own report, never into the file

### Side Outputs

Three CSV files sit next to the v3 HDF5:

- `<stem>_index.csv`: one row per written record; columns `sample_id` plus every
  record attribute (vector attributes flattened as `name_0`, `name_1`, ...).
- `<stem>_skipped.csv`: one row per skipped candidate; columns `sample_id`,
  `source`, `reason`, `detail`.
- `<stem>_build.log`: full build log.

### Excluded Records

The list of excluded record ids is stored in `data/excluded_indexes/` with the
filename structure `IndexesExclude_trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.txt`.
One id per line.

### Building and Verifying

To rebuild v3 or verify a build, see [`scripts/docs/build_beamlet_h5.md`](../scripts/docs/build_beamlet_h5.md).

## Note

This directory is excluded from version control (`.gitignore`). You must download the data separately after cloning the repository.
