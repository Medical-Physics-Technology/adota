"""Per-beamlet ADoTA vs MCsquare analysis (in the BEV crop frame).

Compares the ADoTA per-spot predictions against the MCsquare per-beamlet ground
truth (``beamlets_<primaries>/``), spot by spot, to quantify which pencil beams
ADoTA reproduces well and which it does not, in particular as a function of
energy. The MCsquare beamlet format (matrix, join, units, orientation) is
specified in ``docs/mcsquare_beamlet_format.md`` and is authoritative.

Frame of comparison
-------------------
ADoTA already predicts each beamlet's dose in the field's rotated beam's-eye-view
(BEV) crop frame -- the small subvolume the extraction cropped around the spot
(``{id}_ds_pred.npy``). Rather than depositing that prediction back onto the full
CT grid, the comparison is done **in that crop frame**: the MC beamlet is brought
into the same crop (rotate by the field's extraction angle, then take the crop
window), so a small ADoTA crop is compared against a small MC crop. This is
per-beamlet by construction, avoids any full-grid array, and reuses the same
rotation as extraction/accumulation so it cannot drift from them.

Join
----
Matrix column ``i`` is row ``i`` of ``spot_index.csv``. Its
``(beam_idx, layer_idx, spot_idx)`` enumerate like
:func:`src.beamlets.plan_spots.expand_plan_to_spots` (fraction -> field -> layer
-> spot), so ``spot_id(beam_idx, layer_idx, spot_idx)`` equals the ADoTA record
``id`` (asserted). ``expand_plan_to_spots`` increments the beam across fractions,
so the caller asserts ``n_fractions == 1``.

Normalization (one ADoTA spot vs one MC beamlet, both in Gy)
------------------------------------------------------------
MCsquare columns are raw eV/g/proton; one MC beamlet in Gy is
``mc_gy(i) = M[:, i] * mu_i * opentps_rescaling_i`` (the ``w_i`` whose sum
reproduces ``load_dose_gy(Dose.mhd)``). The ADoTA prediction is de-normalized
BEV dose; the accumulated ``Dose_ADoTA.mhd`` is
``sum_i relative_weight_i * deposit(pred_i)`` scaled to Gy by
``dose_to_gy_factor``. So one ADoTA spot in Gy is
``adota_gy(i) = dose_to_gy_factor * relative_weight_i * pred_i``. Both are Gy and
sum to the two plan doses respectively, so a correctly scaled pair agrees in
**integral**, not merely shape (verified per spot: ratios near 1.0, within model
error and MC noise).

Noise / constraints
-------------------
Beamlets at 1e5 primaries carry MC noise; ``max()`` is a single-voxel noise
statistic and is never used. The high-dose region is set by a robust percentile
peak, metrics are computed there, and the low-dose noise floor is reported. The
CSC matrix is read column by column via its sparse support and never densified;
the MC crop is built from that support, so no full-grid array is ever allocated.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import SimpleITK as sitk

from src.beamlets.accumulation import _load_beamlet_array
from src.beamlets.dose_scaling import dose_to_gy_factor
from src.beamlets.isocenter import isocenter_physical
from src.beamlets.plan_spots import spot_id
from src.beamlets.rotation import rotate_ct_around_isocenter
from src.metrics.classic import calculate_pure_mape, calculate_rmse
from src.metrics.gamma_pass_rate import gamma_index
from src.metrics.range_metrics import (
    compute_range_metrics,
    integrated_depth_dose,
    range_metric_deltas,
)

logger = logging.getLogger(__name__)

ROI_DEPTH_MM = 320.0  # extraction crop depth (x) in mm, for the beam-frame dz
_BBOX_MARGIN = 2      # voxels of margin around the MC support bbox


@dataclass
class BeamletAnalysisConfig:
    energy_split_mev: float = 150.0
    high_dose_frac: float = 0.5          # high-dose mask: dose >= frac * robust_peak
    peak_percentile: float = 99.5        # robust peak (never raw max)
    gamma_criteria: Tuple[Tuple[float, float], ...] = ((2.0, 2.0), (3.0, 3.0))
    gamma_lower_cutoff_pct: float = 10.0
    gamma_interp_fraction: int = 5


@dataclass
class _Beam:
    """Per-field geometry cached across the field's spots (from sim_res)."""

    angle: float
    iso_phys: Tuple[float, float, float]
    image_size: Tuple[int, int, int]      # sitk (x, y, z) of the expanded rotated grid
    image_origin: Tuple[float, float, float]
    image_spacing: Tuple[float, float, float]


def load_beamlet_dir(beamlet_dir: Path) -> Tuple[sp.csc_matrix, pd.DataFrame, dict, dict]:
    beamlet_dir = Path(beamlet_dir)
    M = sp.load_npz(beamlet_dir / "beamlets_raw_csc.npz")
    si = pd.read_csv(beamlet_dir / "spot_index.csv")
    grid = json.loads((beamlet_dir / "grid.json").read_text())
    manifest = json.loads((beamlet_dir / "manifest.json").read_text())
    return M, si, grid, manifest


def default_beamlet_dir(plan_dir: Path) -> Optional[Path]:
    """Newest ``beamlets_*`` directory in the plan dir (by mtime), or ``None``."""
    dirs = sorted((d for d in Path(plan_dir).glob("beamlets_*") if d.is_dir()),
                  key=lambda d: d.stat().st_mtime)
    return dirs[-1] if dirs else None


def assert_join(si: pd.DataFrame, records: List[dict]) -> None:
    if len(si) != len(records):
        raise ValueError(f"spot count mismatch: {len(si)} columns vs {len(records)} plan spots")
    for i in range(len(si)):
        r = si.iloc[i]
        expect = spot_id(int(r.beam_idx), int(r.layer_idx), int(r.spot_idx))
        if expect != records[i]["id"]:
            raise ValueError(
                f"join mismatch at column {i}: spot_index->{expect} plan->{records[i]['id']}"
            )


# ── MC beamlet: sparse support -> small dense bbox (no full-grid densification) ─


def mc_support_bbox_gy(
    M: sp.csc_matrix, si: pd.DataFrame, i: int, grid_size, margin: int = _BBOX_MARGIN
):
    """MC beamlet ``i`` as a small dense bbox in Gy, plus its CT-grid placement.

    Returns ``(mc_bbox (nz,ny,nx), (z0,y0,x0), n_support)``. Reads the CSC
    column's non-zero (row, value) pairs and scatters them into a bbox array; the
    full grid is never allocated.
    """
    nx, ny, nz = grid_size
    s, e = M.indptr[i], M.indptr[i + 1]
    rows = M.indices[s:e]
    vals = M.data[s:e].astype(np.float64) * (float(si.mu.iloc[i]) * float(si.opentps_rescaling.iloc[i]))
    x, y, z = np.unravel_index(rows, (nx, ny, nz), order="F")  # F-order flat -> (x,y,z)
    z0 = max(int(z.min()) - margin, 0); z1 = min(int(z.max()) + margin + 1, nz)
    y0 = max(int(y.min()) - margin, 0); y1 = min(int(y.max()) + margin + 1, ny)
    x0 = max(int(x.min()) - margin, 0); x1 = min(int(x.max()) + margin + 1, nx)
    box = np.zeros((z1 - z0, y1 - y0, x1 - x0), dtype=np.float64)
    box[z - z0, y - y0, x - x0] = vals
    return box, (z0, y0, x0), int(len(vals))


def mc_crop_gy(mc_bbox: np.ndarray, bbox_origin, sim_res: dict, beam: _Beam, ct: sitk.Image) -> np.ndarray:
    """Resample the MC bbox into the field's BEV crop ``(H, W, D)`` (beam along D).

    Rotates by ``+angle`` (as extraction does) into a small reference sized to the
    ADoTA crop window, so the MC crop is voxel-aligned with the ADoTA prediction
    crop. Both the input (MC bbox) and the output (crop window) are small.
    """
    z0, y0, x0 = bbox_origin
    img = sitk.GetImageFromArray(mc_bbox.astype(np.float32))
    img.SetOrigin(ct.TransformIndexToPhysicalPoint([int(x0), int(y0), int(z0)]))
    img.SetSpacing(ct.GetSpacing()); img.SetDirection(ct.GetDirection())

    ex_nx, ex_ny, ex_nz = beam.image_size
    full = sitk.Image(int(ex_nx), int(ex_ny), int(ex_nz), sitk.sitkFloat32)
    full.SetOrigin(beam.image_origin); full.SetSpacing(beam.image_spacing)
    full.SetDirection(ct.GetDirection())
    height, width, depth = sim_res["roi_size"]
    iz, iy, _ix = sim_res["crp_numpy_ct"]
    czlo = max(iz - height // 2, 0); cylo = max(iy - width // 2, 0)
    crop_ref = sitk.RegionOfInterest(full, size=[int(depth), int(width), int(height)],
                                     index=[0, int(cylo), int(czlo)])  # sitk (x,y,z)
    rot = rotate_ct_around_isocenter(img, beam.angle, beam.iso_phys,
                                     reference=crop_ref, default_value=0.0)
    return sitk.GetArrayFromImage(rot).astype(np.float64)  # (H, W, D)


def adota_crop_gy(sim_res: dict, adota_dir: Path, gy_factor: float) -> np.ndarray:
    """ADoTA prediction crop in Gy ``(H, W, D)``: pred * relative_weight * factor."""
    pred = _load_beamlet_array(adota_dir, sim_res["id"], "prediction")  # (H, W, D)
    return pred.astype(np.float64) * float(sim_res["relative_weight"]) * gy_factor


# ── per-spot metrics (crop frame) ────────────────────────────────────────────


def per_spot_metrics(mc: np.ndarray, ad: np.ndarray, cfg: BeamletAnalysisConfig) -> dict:
    """Metrics comparing the MC and ADoTA per-beamlet dose crops (Gy, same grid)."""
    pos = mc > 0
    peak = float(np.percentile(mc[pos], cfg.peak_percentile)) if pos.any() else 0.0
    hd = mc >= cfg.high_dose_frac * peak
    low = pos & (mc < 0.05 * peak)
    out = {
        "mc_peak_gy": round(peak, 5),
        "mc_integral_gy": round(float(mc.sum()), 4),
        "adota_integral_gy": round(float(ad.sum()), 4),
        "integral_ratio": round(float(ad.sum() / mc.sum()), 4) if mc.sum() else float("nan"),
        "n_support_voxels": int(pos.sum()),
        "n_high_dose_voxels": int(hd.sum()),
        "noise_floor_gy": round(float(np.median(mc[low])), 6) if low.any() else 0.0,
    }
    if hd.any():
        out["mape_pct"] = round(float(calculate_pure_mape(ad[hd], mc[hd])), 3)
        out["rmse_gy"] = round(float(calculate_rmse(ad[hd], mc[hd])), 6)
        out["corr_high_dose"] = round(float(np.corrcoef(ad[hd], mc[hd])[0, 1]), 4)

    scale = {"y_max": peak, "y_min": 0.0}
    for dp, dm in cfg.gamma_criteria:
        gp = {"dose_percent_threshold": dp, "distance_mm_threshold": dm,
              "interp_fraction": cfg.gamma_interp_fraction, "max_gamma": 2,
              "lower_percent_dose_cutoff": cfg.gamma_lower_cutoff_pct,
              "random_subset": None, "local_gamma": True, "quiet": True}
        try:
            _, gpr = gamma_index(mc.copy(), ad.copy(), scale, gp, (1.0, 1.0, 1.0))
            out[f"gpr_{int(dp)}pct_{int(dm)}mm"] = round(float(gpr[0]) * 100.0, 2)
        except Exception as exc:  # pragma: no cover
            logger.warning("gamma %s%%/%smm failed: %s", dp, dm, exc)
            out[f"gpr_{int(dp)}pct_{int(dm)}mm"] = float("nan")

    dz = ROI_DEPTH_MM / float(ad.shape[2])  # depth voxels -> mm/voxel
    rm_mc = compute_range_metrics(integrated_depth_dose(np.moveaxis(mc, 2, 0)), dz)
    rm_ad = compute_range_metrics(integrated_depth_dose(np.moveaxis(ad, 2, 0)), dz)
    out["r80_diff_mm"] = round(float(range_metric_deltas(rm_ad, rm_mc)["r80_delta_mm"]), 3)
    return out


# ── driver + aggregation ─────────────────────────────────────────────────────


def run_beamlet_analysis(
    plan_directory, plan_dir: Path, beamlet_dir: Path, adota_dir: Path,
    records: List[dict], cfg: Optional[BeamletAnalysisConfig] = None,
) -> Tuple[pd.DataFrame, dict]:
    cfg = cfg or BeamletAnalysisConfig()
    ct = plan_directory.ct
    from src.beamlets.bdl import BeamDataLibrary
    bdl = BeamDataLibrary.from_file(plan_directory.bdl_path)
    gy_factor = dose_to_gy_factor(plan_directory.plan, bdl)

    M, si, grid, manifest = load_beamlet_dir(beamlet_dir)
    assert_join(si, records)
    grid_size = grid["gridSize"]
    w = (si.mu.values * si.opentps_rescaling.values).astype(np.float64)
    w_sum = float(w.sum())

    beams: Dict[int, _Beam] = {}
    rows: List[dict] = []
    for i in range(len(si)):
        sid = records[i]["id"]
        sr = json.loads((Path(adota_dir) / f"{sid}_sim_res.json").read_text())
        b = sr["beam"]
        if b not in beams:
            beams[b] = _Beam(
                angle=sr["gantry_angle"],
                iso_phys=isocenter_physical(sr["simulation_log"]["isocenter"], ct),
                image_size=tuple(sr["image_size"]),
                image_origin=tuple(sr["image_origin"]),
                image_spacing=tuple(sr["image_spacing"]),
            )
        beam = beams[b]
        mc_bbox, origin, _ = mc_support_bbox_gy(M, si, i, grid_size)
        mc = mc_crop_gy(mc_bbox, origin, sr, beam, ct)
        ad = adota_crop_gy(sr, adota_dir, gy_factor)
        m = per_spot_metrics(mc, ad, cfg)
        rows.append({
            "col": i, "id": sid,
            "beam_idx": int(si.beam_idx.iloc[i]), "layer_idx": int(si.layer_idx.iloc[i]),
            "spot_idx": int(si.spot_idx.iloc[i]), "energy_mev": float(si.energy_mev.iloc[i]),
            "mu": float(si.mu.iloc[i]), "mu_fraction": float(w[i] / w_sum), **m,
        })
        if (i + 1) % 20 == 0 or i == len(si) - 1:
            logger.info("  beamlet metrics %d/%d", i + 1, len(si))

    df = pd.DataFrame(rows)
    return df, _aggregate(df, cfg, manifest, beamlet_dir)


_AGG_METRICS = ["gpr_2pct_2mm", "gpr_3pct_3mm", "mape_pct", "rmse_gy",
                "corr_high_dose", "r80_diff_mm", "integral_ratio"]


def _stats(df: pd.DataFrame) -> dict:
    out = {"n_spots": int(len(df))}
    if df.empty:
        return out
    wt = df["mu_fraction"].to_numpy(dtype=float)
    for col in _AGG_METRICS:
        if col not in df:
            continue
        v = df[col].to_numpy(dtype=float)
        ok = np.isfinite(v)
        wsum = wt[ok].sum()
        out[col] = {
            "mean": round(float(np.nanmean(v)), 4) if ok.any() else None,
            "mu_weighted_mean": round(float(np.sum(v[ok] * wt[ok]) / wsum), 4) if wsum > 0 else None,
            "n": int(ok.sum()),
        }
    return out


def _aggregate(df: pd.DataFrame, cfg: BeamletAnalysisConfig, manifest: dict, beamlet_dir: Path) -> dict:
    split = cfg.energy_split_mev
    hi = df[df.energy_mev >= split]
    return {
        "beamlet_dir": str(beamlet_dir),
        "primaries_per_beamlet": manifest.get("primaries_per_beamlet"),
        "n_spots": int(len(df)),
        "energy_split_mev": split,
        "high_dose_frac": cfg.high_dose_frac,
        "peak_percentile": cfg.peak_percentile,
        "overall": _stats(df),
        f"below_{int(split)}MeV": _stats(df[df.energy_mev < split]),
        f"at_or_above_{int(split)}MeV": _stats(hi),
        "mu_fraction_above_split": round(float(hi["mu_fraction"].sum()), 4) if len(hi) else 0.0,
    }
