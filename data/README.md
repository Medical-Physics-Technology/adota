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

## Note

This directory is excluded from version control (`.gitignore`). You must download the data separately after cloning the repository.
