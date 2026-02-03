# Models Directory

This directory contains trained ADoTA model weights and configurations.

## Downloading Pre-trained Models

Pre-trained model weights can be downloaded from Google Drive:

🔗 **[Download Pre-trained Models](https://drive.google.com/drive/folders/1gf5antKnnS7x8QvCgAiPjgF_OizWBxtt?usp=sharing)**

### Setup Instructions

1. **Download** the model folder(s) from the Google Drive link above (e.g., `DoTA_v3_grid_search_v11`).

2. **Extract/Copy** the contents to this directory:
   ```
   models/
   └── DoTA_v3_grid_search_v11/
       ├── best_model.pth
       └── hyperparams.json
   ```

3. **Verify** the structure:
   ```bash
   ls models/DoTA_v3_grid_search_v11/
   # Should show: best_model.pth  hyperparams.json
   ```

## Model Directory Structure

Each model directory must contain:

| File | Description |
|------|-------------|
| `best_model.pth` | PyTorch model weights (state dict) |
| `hyperparams.json` | Model hyperparameters (architecture configuration) |

### Example `hyperparams.json`

```json
{
    "input_shape": [2, 160, 30, 30],
    "num_levels": 4,
    "enc_features": 32,
    "kernel_size": 3,
    "convolutional_steps": 1,
    "conv_hidden_channels": 64,
    "num_transformers": 1,
    "num_heads": 8,
    "dropout_rate": 0.1,
    "causal": true,
    "num_forward": 0
}
```

## Running a Model

After downloading and setting up the model, run evaluation with:

```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs
```

## Note

This directory is excluded from version control (`.gitignore`). You must download the models separately after cloning the repository.
