"""Real low/med/high edge-gradient beamlets: CT -> Sobel edges -> model error."""
import sys
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
sys.path.insert(0, "/home/mstryja/projects/adota")
from src.adota.config import get_device
from src.adota.utils import load_model
from src.loaders.generator import H5PYGenerator
from src.utils.scallers import inverse_minmax
from src.schemas.configs import AdvancedAnalysisConfig

INK, MUTED, BASE, SURF = "#0b0b0b", "#898781", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "font.size": 11, "font.family": "sans-serif"})
H5 = "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
MODEL_DIR = Path("/home/mstryja/projects/adota/models/DoTA_v3_grid_search_v11")
OUT = Path("/home/mstryja/projects/adota/research/figures/acquisition")
DZ = 2.0
ROWS = [  # (label, uuid, sum_sobel_k, gpr)
    ("Low edge",  "8dcc0a2e-7f9f-4882-bf2a-0f0271baa5b5", "12k",  "100%"),
    ("Medium edge", "92bdc9a0-4498-46f7-b35b-94c7886f9bcf", "107k", "88%"),
    ("High edge", "1f99ecad-a349-42d5-99bf-330334241af1", "610k", "87%"),
]
scale = AdvancedAnalysisConfig().scale
device = get_device(0)
model = load_model(MODEL_DIR / "best_model.pth", MODEL_DIR / "hyperparams.json", device)

fig, axes = plt.subplots(3, 3, figsize=(14, 8.2))
fig.subplots_adjust(left=0.11, right=0.93, top=0.86, bottom=0.08, wspace=0.16, hspace=0.22)
col_titles = ["CT (input)", "edge magnitude |∇HU|", "model error |ADoTA − MC| [%]"]
for r, (label, uuid, ssk, gpr) in enumerate(ROWS):
    ds = H5PYGenerator(file_path=H5, indexes=[uuid], augmentation=False, cropp=True,
                       normalize=False, normalize_flux_only=True)
    x, energy, y = ds[0]; x = x.to(device); energy = energy.to(device)
    with torch.no_grad():
        y_pred = model(x.unsqueeze(0), energy.unsqueeze(0))[0]
    ct = inverse_minmax(x[0].cpu().numpy(), scale["min_ct"], scale["max_ct"])
    gt = inverse_minmax(y.unsqueeze(0).cpu().numpy(), scale["min_ds"], scale["max_ds"]).squeeze()
    pred = inverse_minmax(y_pred.cpu().numpy(), scale["min_ds"], scale["max_ds"]).squeeze()
    gmag = np.sqrt(sobel(ct, 0)**2 + sobel(ct, 1)**2 + sobel(ct, 2)**2)
    diff = np.abs(pred - gt) / max(gt.max(), 1e-9) * 100.0
    midy = ct.shape[1] // 2

    def sag(v): return v[:, midy, :].T
    ext = [0, ct.shape[0]*DZ, 0, ct.shape[2]]
    axes[r, 0].imshow(sag(ct), cmap="gray", origin="lower", aspect="auto", vmin=-1000, vmax=800, extent=ext)
    im1 = axes[r, 1].imshow(sag(gmag), cmap="hot", origin="lower", aspect="auto", extent=ext)
    im2 = axes[r, 2].imshow(sag(diff), cmap="magma", origin="lower", aspect="auto", vmin=0, vmax=15, extent=ext)
    axes[r, 0].set_ylabel(f"{label}\nsum_sobel_bp={ssk}\nGPR {gpr}", fontsize=11, color=INK)
    for c in range(3):
        axes[r, c].set_xticks([]) if r < 2 else axes[r, c].set_xlabel("depth [mm]", fontsize=10)
        axes[r, c].set_yticks([])
    if r == 0:
        for c in range(3):
            axes[r, c].set_title(col_titles[c], fontsize=12.5, color=INK)
fig.colorbar(im1, ax=axes[:, 1], fraction=0.025, pad=0.02, label="|∇HU|")
fig.colorbar(im2, ax=axes[:, 2], fraction=0.025, pad=0.02, label="error [%]")
fig.suptitle("Real beamlets ordered by edge burden — more/stronger tissue edges → larger dose-prediction error",
             fontsize=13.5, fontweight="bold", x=0.5, y=0.955)
fig.savefig(OUT / "edge_gradient_examples.png", dpi=150)
print("wrote edge_gradient_examples.png")
