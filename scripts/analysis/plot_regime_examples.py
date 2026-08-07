"""Two contrasting beamlets for Section 2.1: a clean Bragg peak vs strong heterogeneity.

Renders the Figure-1 layout (publication_figure) for two thorax beamlets at
clearly different ranges: an easy beamlet whose proton range lands in near-
homogeneous tissue and produces a sharp, well-predicted Bragg peak, and a hard
beamlet whose beam crosses many tissue interfaces and whose Bragg peak is
broadened and mispredicted. Both are drawn from the reference run so the
annotated errors match results.csv.

Run: uv run python scripts/analysis/plot_regime_examples.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/mstryja/projects/adota")
from src.adota.config import get_device
from src.adota.utils import load_model
from src.loaders.generator import H5PYGenerator
from src.utils.scallers import inverse_minmax
from src.utils.unit_conversions import to_gy
from src.metrics.classic import calculate_rmse, calculate_pure_mape
from src.figures.single_beam import publication_figure
from src.schemas.configs import AdvancedAnalysisConfig

H5 = "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
MODEL_DIR = Path("/home/mstryja/projects/adota/models/DoTA_v3_grid_search_v11")
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
BEAMS = [  # (label, uuid, gpr_from_results)
    ("clean_bragg",  "6a5514c6-52c5-4621-9ca9-7c9bec232413", 99.94),
    ("hetero_bragg", "c7702446-6c32-43c1-a9b1-42c2e5413217", 85.80),
]
scale = AdvancedAnalysisConfig().scale
device = get_device(0)
model = load_model(MODEL_DIR / "best_model.pth", MODEL_DIR / "hyperparams.json", device)

for label, uuid, gpr in BEAMS:
    ds = H5PYGenerator(file_path=H5, indexes=[uuid], augmentation=False, cropp=True,
                       normalize=False, normalize_flux_only=True)
    x, energy, y = ds[0]
    x = x.to(device); energy = energy.to(device)
    with torch.no_grad():
        y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]
    gt = inverse_minmax(y.unsqueeze(0).cpu().numpy(), scale["min_ds"], scale["max_ds"]).squeeze()
    pred = inverse_minmax(y_pred.cpu().numpy(), scale["min_ds"], scale["max_ds"]).squeeze()
    rmse = calculate_rmse(to_gy(pred), to_gy(gt))
    mask = gt > 0.1 * gt.max()
    mape = calculate_pure_mape(gt[mask], pred[mask])
    publication_figure(x.cpu().numpy(), 0.0, gt, pred, str(OUT / f"{label}.png"),
                       float(rmse), float(mape), float(gpr), beamlet_shape=True)
    print(f"{label}: {uuid}  MAPE={mape:.2f}%  RMSE={rmse:.4f}  GPR={gpr}")
print("done")
