"""Verify the streaming pipeline produces the same dose as the staged pipeline.

For each plan, runs the staged path (`run_extraction -> run_inference ->
run_accumulation`) and the fused `run_streaming_pipeline` with the **same** model,
device and flux setting, and compares the two ``Dose_ADoTA`` arrays. They should be
identical (the streaming path reuses the same crop/flux, the shared pre/post
helpers, and the same deposit + de-rotation). Temp doses + the staged
``adota_beamlets/`` are cleaned up per plan.

Usage:
    python scripts/verify_streaming_equivalence.py [--device-index 0] [--n-spots N]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.utils import load_model
from src.beamlets.accumulation import AccumulationConfig, run_accumulation
from src.beamlets.extraction import ExtractionConfig, run_extraction
from src.beamlets.inference import InferenceConfig, run_inference
from src.beamlets.streaming import StreamingConfig, run_streaming_pipeline
from src.evaluation.cli import resolve_device
from src.loaders.plan_directory import load_plan_directory

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PLANS = [
    "/scratch/mstryja/opentps_plans/Prostate-AEC-004_4mm_target_margin",
    "/scratch/mstryja/opentps_plans/LUNG1-221_LUNG1-221_3beams_positive",
    "/scratch/mstryja/opentps_plans/LUNG1-062_LUNG1-062_3beams",
]
MODEL = "DoTA_v3_grid_search_v11"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--n-spots", type=int, default=None, help="Subset for a quick check.")
    ap.add_argument("--batch-size", type=int, default=56)
    args = ap.parse_args()

    device = resolve_device(args.device_index)
    model = load_model(
        PROJECT_ROOT / "models" / MODEL / "best_model.pth",
        PROJECT_ROOT / "models" / MODEL / "hyperparams.json",
        device,
    )
    flux_dev = str(device)
    print(f"device={device}  model={MODEL}  n_spots={args.n_spots or 'all'}\n")
    print(f"{'plan':<46}{'spots':>7}{'wall staged':>12}{'wall stream':>12}{'max|Δ|':>12}{'max rel':>12}{'equal':>7}")

    for plan in PLANS:
        plan_dir = Path(plan)
        name = plan_dir.name
        pd = load_plan_directory(plan_dir, bdl_path=None)
        beamlets = plan_dir / "adota_beamlets"
        staged_dose = plan_dir / "Dose_staged_verify.mhd"
        stream_dose = plan_dir / "Dose_stream_verify.mhd"

        try:
            # Staged.
            t = perf_counter()
            run_extraction(pd, beamlets, ExtractionConfig(
                n_spots=args.n_spots, save_overlays=False, overwrite=True,
                flux_on_gpu=True, flux_device=flux_dev))
            run_inference(beamlets, model, device, InferenceConfig(batch_size=args.batch_size))
            run_accumulation(pd, beamlets, staged_dose, AccumulationConfig(dose_source="prediction"))
            wall_staged = perf_counter() - t

            # Streaming.
            t = perf_counter()
            summary = run_streaming_pipeline(pd, model, device, stream_dose, StreamingConfig(
                n_spots=args.n_spots, batch_size=args.batch_size,
                flux_on_gpu=True, flux_device=flux_dev))
            wall_stream = perf_counter() - t

            a = sitk.GetArrayFromImage(sitk.ReadImage(str(staged_dose)))
            b = sitk.GetArrayFromImage(sitk.ReadImage(str(stream_dose)))
            d = np.abs(a - b)
            denom = np.abs(a).max() or 1.0
            print(f"{name:<46}{summary['n_spots']:>7}{wall_staged:>11.1f}s{wall_stream:>11.1f}s"
                  f"{d.max():>12.3e}{(d.max()/denom):>12.3e}{str(np.array_equal(a, b)):>7}")
        finally:
            shutil.rmtree(beamlets, ignore_errors=True)
            for f in (staged_dose, stream_dose):
                f.unlink(missing_ok=True)
                f.with_suffix(".raw").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
