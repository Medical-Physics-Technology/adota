"""Generate publication_figure for low / medium / high WEPL-spread beamlets."""
import sys
from pathlib import Path
import numpy as np, torch
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
BEAMS = [  # (label, uuid)
    ("wepl_low",  "0e6badd2-0f36-4730-b6b7-836654389769"),
    ("wepl_med",  "c8a09e39-fc25-48e6-b979-2e9c3a3e4403"),
    ("wepl_high", "53cb1b9f-f524-4438-9b9f-77bfa6faf0ea"),
]
scale = AdvancedAnalysisConfig().scale
device = get_device(0)
model = load_model(MODEL_DIR / "best_model.pth", MODEL_DIR / "hyperparams.json", device)

for label, uuid in BEAMS:
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
    e_mev = float(energy.item()) if energy.numel() == 1 else 0.0
    # energy tensor is normalized; use a nominal from filename-independent denorm not needed for the (unused) title
    publication_figure(x.cpu().numpy(), 0.0, gt, pred, str(OUT / f"{label}.png"),
                       float(rmse), float(mape), 0.0, beamlet_shape=True)
    print(f"{label}: {uuid}  MAPE={mape:.2f}%  RMSE={rmse:.4f}")
print("done")
