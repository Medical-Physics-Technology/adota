"""Fast per-beamlet MAPE extraction (inference-only, no metric/gamma recompute).

Reuses the shared evaluation engine (same model + load path as the full pipeline)
but a minimal callback that only computes MAPE at two GT-dose masks:

    MAPE@f = mean_{D > f * D_max}  |D_pred - D| / |D|  * 100      (f in {5%, 10%})

MAPE divides by the *local* dose, so it is only meaningful with a dose mask; the
two thresholds match the existing MAPE table (src/tables/results.py). Output is a
CSV ``(sample_id, mape_5pct, mape_10pct)`` to be joined onto ``results.csv`` by
``sample_id`` for the correlation study. No RDE/gamma/heterogeneity recompute, so
this is far cheaper than re-running the full advanced-metrics pipeline.

Run (from repo root):
    uv run python scripts/analysis/extract_mape.py \
        --out /scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/mape_metric.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.adota.config import get_device
from src.adota.utils import load_model
from src.evaluation.engine import evaluate
from src.evaluation.sources import H5Source
from src.loaders.generator import H5PYGenerator
from src.schemas.configs import AdvancedAnalysisConfig

H5 = "/scratch/mstryja/DoTA_dataset_v2/trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5"
MODEL_DIR = Path("/home/mstryja/projects/adota/models/DoTA_v3_grid_search_v11")
THRESHOLDS = [(0.05, "mape_5pct"), (0.10, "mape_10pct")]


def make_mape_fn(scale: dict):
    def per_sample(ctx):
        # zero-flux guard (matches the full pipeline's skip protocol)
        if torch.abs(ctx.x[1]).max().item() < 1e-9:
            return None
        y_np, y_pred_np = ctx.denorm(scale)  # physical dose (Gy)
        gt = np.squeeze(y_np).astype(np.float64)
        pred = np.squeeze(y_pred_np).astype(np.float64)
        gmax = float(gt.max())
        out = {"sample_id": ctx.sample_id}
        if gmax <= 0:
            for _, name in THRESHOLDS:
                out[name] = np.nan
            return out
        for frac, name in THRESHOLDS:
            mask = gt > frac * gmax
            out[name] = (
                float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]) * 100.0)
                if mask.sum()
                else np.nan
            )
        return out

    return per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=H5)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--out", default="/scratch/mstryja/adota_runs/20260707_124010/figures/acquisition/mape_metric.csv")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N records (smoke test)")
    args = ap.parse_args()

    device = get_device(0)
    model = load_model(Path(args.model_dir) / "best_model.pth",
                       Path(args.model_dir) / "hyperparams.json", device)
    config = AdvancedAnalysisConfig()

    gen_kw = dict(augmentation=False, cropp=True, normalize=False, normalize_flux_only=True)
    dataset = H5PYGenerator(file_path=args.h5, **gen_kw)
    record_ids = list(dataset.record_ids)
    if args.limit:
        record_ids = record_ids[: args.limit]
        dataset = H5PYGenerator(file_path=args.h5, indexes=record_ids, **gen_kw)
    source = H5Source(dataset, record_ids=record_ids)

    results = evaluate(model, source, device=device,
                       per_sample_fn=make_mape_fn(config.scale),
                       show_progress=True, desc="MAPE")
    df = pd.DataFrame(results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  rows={len(df)}  "
          f"nan_5={int(df['mape_5pct'].isna().sum())} nan_10={int(df['mape_10pct'].isna().sum())}")
    print(df[["mape_5pct", "mape_10pct"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
