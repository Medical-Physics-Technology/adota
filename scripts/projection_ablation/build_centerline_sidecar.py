"""Precompute the beam-centerline parameters for every record of an H5 dataset.

Writes a parquet sidecar ``{uuid: (a0, a1, b0, b1)}`` (raw-record-frame line
params) built from the source-metadata entrance point + the H5 steering angles,
so the training generator can render the centerline channel with no per-sample
JSON I/O. See src.beamlets.centerline for the geometry conventions.

Run: uv run python scripts/projection_ablation/build_centerline_sidecar.py
"""
import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/mstryja/projects/adota")
from src.beamlets.centerline import beam_line_from_metadata

JDIRS = ["/RadiotherapyData/dataset_v0/trainset_pelvis",
         "/RadiotherapyData/dataset_v0/initial_test_one_ct"]


def find_json(uid):
    for d in JDIRS:
        p = os.path.join(d, uid + "_metadata.json")
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/scratch/mstryja/DoTA_dataset_v2/"
                    "trainset_pelvis_initial_test_one_ct_downsampled_v2_all_SingleGaussian.h5")
    ap.add_argument("--out", default="/scratch/mstryja/DoTA_dataset_v2/"
                    "centerline_params_trainset_pelvis_initial_test_one_ct.csv")
    args = ap.parse_args()

    rows, missing = [], 0
    with h5py.File(args.h5, "r") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys):
            g = f[k]
            jp = find_json(k)
            if jp is None:
                missing += 1
                continue
            ba = np.array(g.attrs["beamlet_angles"], dtype=float)
            lat = int(g["flux"].shape[0])
            meta = json.load(open(jp))
            ds = meta["roi_size"][0] / lat
            line = beam_line_from_metadata(meta["rays_entrence_point"], ba, ds)
            rows.append((k, line.a0, line.a1, line.b0, line.b1, ds))
            if (i + 1) % 10000 == 0:
                print(f"  {i + 1}/{len(keys)} ...", flush=True)

    df = pd.DataFrame(rows, columns=["uuid", "a0", "a1", "b0", "b1", "downsample"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"records={len(df)}  missing_json={missing}  -> {args.out}")


if __name__ == "__main__":
    main()
