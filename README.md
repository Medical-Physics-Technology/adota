# ADoTA

**Angle-Dependent Dose Transformer Algorithm**

Official repository of the ADoTA (Angle-Dependent Dose Transformer Algorithm) project.

## About

ADoTA is a deep learning-based algorithm for fast and accurate dose prediction in proton therapy. The model uses a transformer architecture combined with 3D convolutional encoders and decoders to predict dose distributions from CT images and beam configurations.

## Authors

Developed by:
- **Mikołaj Stryja**
- **Zoltan Perkó**
- **Danny Lathouwers**

**Affiliation:** Radiation Science and Technology, Medical Physics and Technology Group, Delft University of Technology (TU Delft)

## Installation

```bash
# Clone the repository
git clone https://github.com/Medical-Physics-Technology/adota.git
cd adota

# Install dependencies using uv
uv sync
```

## Usage

### Prerequisites

Before running the model, please download the required files:

1. **[Download Pre-trained Models](models/README.md)** - Model weights and configuration files
2. **[Download Example Data](data/README.md)** - Input data for evaluation

### Running Evaluation

Run model evaluation on test data:

```bash
uv run python scripts/run_model.py <MODEL_NAME> <TEST_DATA> [OPTIONS]
```

Example:
```bash
uv run python scripts/run_model.py DoTA_v3_grid_search_v11 data/example_inputs
```

For more details, see [scripts/README.md](scripts/README.md).

## Project Structure

```
adota/
├── data/               # Input data (not tracked)
├── models/             # Trained model weights (not tracked)
├── runs/               # Evaluation outputs (not tracked)
├── scripts/            # Utility scripts
├── src/
│   ├── adota/          # Core model implementation
│   ├── figures/        # Visualization utilities
│   ├── loaders/        # Data loading utilities
│   ├── metrics/        # Evaluation metrics
│   ├── tables/         # Results formatting
│   └── utils/          # Helper functions
└── pyproject.toml      # Project configuration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
