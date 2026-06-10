"""ADoTA batch inference worker.

Runs in the ADoTA venv (Python 3.9 + PyTorch). Called via subprocess from
the OpenTPS env. Reads spot patches from disk, runs the DoTA3D_v3 model,
and saves dose predictions as numpy arrays.

Usage:
    /home/mstryja/projects/adota/.venv/bin/python inference_worker.py \
        --config /path/to/worker_config.json

Config JSON schema:
    {
        "spot_ids":        ["b00_l000_s0000", ...],
        "storage_path":    "/path/to/spot_predictions",
        "output_path":     "/path/to/spot_predictions",   # same dir is fine
        "model_path":      "/abs/path/to/best_model.pth",
        "hyperparams_path":"/abs/path/to/hyperparams.json",
        "scale": {
            "min_ds": 0.0, "max_ds": 25277028.0,
            "min_ct": -1024, "max_ct": 3071,
            "min_energy": 70.0, "max_energy": 270.0
        }
    }

Reads per-spot files:
    {id}_ct.npy        shape (60, 60, 320)  – HU values in BEV crop
    {id}_flux.npy      shape (60, 60, 320)  – proton flux channel
    {id}_sim_res.json  – spot metadata (energy, bixelgrid_shifts_xy, …)

Writes per-spot prediction:
    {id}_ds_pred.npy   shape (1, 1, 320, 60, 60) – dose in eV/g/proton
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.adota.utils import load_model
from src.loaders.dir_based import get_single_record_no_gt, save_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ADoTA batch inference worker")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    spot_ids: list = config["spot_ids"]
    storage_path: str = config["storage_path"]
    output_path: str = config.get("output_path", storage_path)
    model_path = Path(config["model_path"])
    hyperparams_path = Path(config["hyperparams_path"])
    scale: dict = config["scale"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Loading model from %s", model_path)

    model = load_model(model_path, hyperparams_path, device)
    logger.info("Model loaded. Running inference for %d spots.", len(spot_ids))

    failed = []
    for spot_id in spot_ids:
        try:
            x, energy = get_single_record_no_gt(
                id=spot_id,
                storage_path=storage_path,
                scale=scale,
                normalize_flux=True,
                downsampling_method="interpolation",
            )
            x = x.unsqueeze(0).to(device)        # (1, 2, 160, 30, 30)
            energy = energy.unsqueeze(0).to(device)   # (1, 1)

            with torch.no_grad():
                pred = model(x, energy)[0]        # (dose, attention) -> dose

            save_prediction(pred, spot_id, output_path, scale=scale)
        except Exception as exc:
            logger.error("Failed for %s: %s", spot_id, exc)
            failed.append(spot_id)

    if failed:
        logger.error("%d spots failed: %s", len(failed), failed)
        sys.exit(1)

    logger.info("Inference complete. %d spots processed.", len(spot_ids))


if __name__ == "__main__":
    main()
