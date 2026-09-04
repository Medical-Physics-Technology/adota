"""Anti-diagonal transect through a beamlet-angle robustness grid (appendix A.6).

The 18x18 angular grid of :mod:`src.mc_generation.robustness` sweeps a beamlet
over ``(theta_x, theta_y)``. Because the angle is realised as a spot displacement
on a divergent beam, each grid cell also traverses *different anatomy*: the
anti-diagonal ``(theta_x, theta_y) = (+t, -t)`` walks the beamlet across the
patient. This module pulls the per-position geometry, depth-dose and error
signals off that anti-diagonal so a single figure can show whether the accuracy
variation follows the anatomy rather than the incidence angle.

Everything here reads arrays that already exist in the experiment dir
(``{stem}_ct.npy``, ``_flux.npy``, ``_ds.npy``, ``_ds_pred.npy``) plus the
persisted gamma grids; no Monte Carlo and no inference is re-run.

Conventions
-----------
* Arrays are ``(H, W, D)`` on the *field-aligned* ROI grid, 1 mm isotropic. The
  third axis is the field (depth) axis; a beamlet at ``theta != 0`` crosses that
  grid obliquely, drifting laterally with depth.
* ``depth`` is measured along the field axis from the ROI front face [mm].
* The HU profile is a **flux-weighted lateral average**, not a single central
  column: the proton fluence has sigma ~4 mm here, so the beamlet samples a
  finite tube, and weighting by the (angled) flux follows the beamlet's own
  drift through the grid without any ray tracing. See :func:`flux_weighted_hu`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# HU windows used to classify what the ray traverses.
LUNG_HU_MAX = -300.0      # below this: aerated lung / air
SOFT_HU_MAX = 150.0       # [LUNG_HU_MAX, SOFT_HU_MAX): soft tissue
SURFACE_HU = -500.0       # first crossing marks the patient surface
DISTAL_WINDOW_MM = 20     # window before R80 used for "what does it stop in"


@dataclass
class DiagonalTransect:
    """Per-position signals along the sampled anti-diagonal.

    Rows are diagonal positions ``0 .. n-1``; columns of the 2-D members are
    depth in mm along the field axis.
    """

    positions: np.ndarray        # (n,) integer position index
    stems: List[str]             # (n,) beamlet stem, e.g. "a00_17"
    theta_x: np.ndarray          # (n,) [degrees]
    theta_y: np.ndarray          # (n,) [degrees]
    depth_mm: np.ndarray         # (D,) depth along the field axis [mm]
    hu: np.ndarray               # (n, D) flux-weighted HU [HU]
    ddd_mc: np.ndarray           # (n, D) MC integrated depth dose [arb.]
    ddd_ad: np.ndarray           # (n, D) ADoTA integrated depth dose [arb.]
    gamma: np.ndarray            # (n,) gamma pass rate [%]
    r80_mc: np.ndarray           # (n,) MC distal R80 [mm]
    r80_ad: np.ndarray           # (n,) ADoTA distal R80 [mm]
    r100_mc: np.ndarray          # (n,) MC depth of maximum [mm]
    entry_mm: np.ndarray         # (n,) patient surface along the field axis [mm]
    missing: List[int]           # positions with no usable data
    energy_mev: float
    patient: str
    anatomy: str
    gamma_label: str

    @property
    def n(self) -> int:
        return len(self.positions)


def flux_weighted_hu(ct: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Lateral HU average at each depth, weighted by the beamlet fluence.

    ``ct`` and ``flux`` are ``(H, W, D)``. The flux is normalised per depth
    slice, so the result is the HU the beamlet actually sees at that depth,
    including its lateral drift. Falls back to the geometric centre column for
    any depth slice with no fluence.
    """
    w = np.asarray(flux, dtype=float)
    tot = w.sum(axis=(0, 1))
    out = np.empty(w.shape[2], dtype=float)
    ok = tot > 0
    out[ok] = (ct[:, :, ok] * w[:, :, ok]).sum(axis=(0, 1)) / tot[ok]
    if (~ok).any():
        h, wd = ct.shape[0] // 2, ct.shape[1] // 2
        out[~ok] = ct[h, wd, ~ok]
    return out


def distal_depth(ddd: np.ndarray, depth_mm: np.ndarray, fraction: float) -> float:
    """Distal depth [mm] where ``ddd`` falls through ``fraction`` of its maximum.

    Linear interpolation between the bracketing samples; ``nan`` if the curve
    never drops below the threshold inside the ROI.
    """
    d = np.asarray(ddd, dtype=float)
    if not np.isfinite(d).any() or d.max() <= 0:
        return float("nan")
    k = int(np.argmax(d))
    thr = fraction * d[k]
    below = np.nonzero(d[k:] < thr)[0]
    if below.size == 0:
        return float("nan")
    j = k + int(below[0])
    y0, y1 = d[j - 1], d[j]
    frac = (y0 - thr) / (y0 - y1) if y0 > y1 else 0.0
    return float(depth_mm[j - 1] + frac * (depth_mm[j] - depth_mm[j - 1]))


def surface_depth(hu: np.ndarray, depth_mm: np.ndarray) -> float:
    """Depth [mm] at which the ray first enters the patient (HU > ``SURFACE_HU``)."""
    inside = np.nonzero(np.asarray(hu) > SURFACE_HU)[0]
    return float(depth_mm[inside[0]]) if inside.size else float("nan")


def _load_prediction(path: Path, mc_shape: Tuple[int, ...]) -> np.ndarray:
    """``{stem}_ds_pred.npy`` is ``(1, 1, D, H, W)``; return it as ``(H, W, D)``."""
    ad = np.squeeze(np.load(path)).astype(float)
    ad = np.moveaxis(ad, 0, -1)
    if ad.shape != tuple(mc_shape):
        raise ValueError(f"prediction shape {ad.shape} != MC {tuple(mc_shape)}")
    return ad


def diagonal_stems(grid_n: int) -> List[Tuple[int, int, int]]:
    """``(position, ix, iy)`` for the anti-diagonal ``iy = grid_n - 1 - ix``."""
    return [(k, k, grid_n - 1 - k) for k in range(grid_n)]


def extract_diagonal(
    exp_dir: Path,
    grids_npz: Path,
    criterion_key: str,
    grid_n: int = 18,
    theta_range: Sequence[float] = (-2.0, 2.0),
) -> DiagonalTransect:
    """Build the :class:`DiagonalTransect` for one experiment dir.

    Args:
        exp_dir: Robustness experiment dir holding the per-beamlet arrays.
        grids_npz: ``*_grids.npz`` written by ``save_panel_grids`` (gamma grids).
        criterion_key: Gamma criterion key inside ``grids_npz`` (e.g. ``g1_3_0p1``).
        grid_n: Angular grid size (the diagonal has ``grid_n`` positions).
        theta_range: ``(lo, hi)`` of the swept angle [degrees], for the labels.
    """
    exp_dir = Path(exp_dir)
    grids = np.load(grids_npz)
    grid = grids[criterion_key]
    thetas = np.linspace(float(theta_range[0]), float(theta_range[1]), grid_n)

    stems: List[str] = []
    hu_rows: List[np.ndarray] = []
    mc_rows: List[np.ndarray] = []
    ad_rows: List[np.ndarray] = []
    tx, ty, gam, r80m, r80a, r100m, entry = ([] for _ in range(7))
    missing: List[int] = []
    depth_mm: Optional[np.ndarray] = None
    meta = {"energy": float("nan"), "patient": "?", "anatomy": "?"}

    for pos, ix, iy in diagonal_stems(grid_n):
        stem = f"a{ix:02d}_{iy:02d}"
        stems.append(stem)
        sim = exp_dir / f"{stem}_sim_res.json"
        pred = exp_dir / f"{stem}_ds_pred.npy"
        if not (sim.exists() and pred.exists()):
            missing.append(pos)
            logger.warning("position %02d (%s): missing arrays, left blank", pos, stem)
            tx.append(thetas[ix])
            ty.append(thetas[iy])
            for lst in (gam, r80m, r80a, r100m, entry):
                lst.append(float("nan"))
            hu_rows.append(None)
            mc_rows.append(None)
            ad_rows.append(None)
            continue

        sr = json.loads(sim.read_text())
        ct = np.load(exp_dir / f"{stem}_ct.npy").astype(float)
        flux = np.load(exp_dir / f"{stem}_flux.npy").astype(float)
        mc = np.load(exp_dir / f"{stem}_ds.npy").astype(float)
        ad = _load_prediction(pred, mc.shape)
        if depth_mm is None:
            step = float(sr.get("image_spacing", [1.0, 1.0, 1.0])[2])
            depth_mm = np.arange(mc.shape[2], dtype=float) * step
            meta = {"energy": float(sr["initial_energy"]),
                    "patient": str(sr["patient_id"]), "anatomy": str(sr["anatomy"])}

        hu = flux_weighted_hu(ct, flux)
        d_mc = mc.sum(axis=(0, 1))
        d_ad = ad.sum(axis=(0, 1))
        a_x, a_y = sr["beamlet_angles"]
        tx.append(float(a_x))
        ty.append(float(a_y))
        gam.append(float(grid[ix, iy]))
        r80m.append(distal_depth(d_mc, depth_mm, 0.8))
        r80a.append(distal_depth(d_ad, depth_mm, 0.8))
        r100m.append(float(depth_mm[int(np.argmax(d_mc))]))
        entry.append(surface_depth(hu, depth_mm))
        hu_rows.append(hu)
        mc_rows.append(d_mc)
        ad_rows.append(d_ad)

    if depth_mm is None:
        raise FileNotFoundError(f"no usable beamlets on the diagonal of {exp_dir}")
    blank = np.full(depth_mm.size, np.nan)
    stack = lambda rows: np.stack([r if r is not None else blank for r in rows])  # noqa: E731

    label = criterion_key  # e.g. "g1_3_0p1" -> "Gamma(1%, 3mm, 0.1%)"
    parts = criterion_key.lstrip("g").replace("p", ".").split("_")
    if len(parts) == 3:
        label = f"Γ({parts[0]}%, {parts[1]}mm, {parts[2]}%)"

    return DiagonalTransect(
        positions=np.arange(grid_n),
        stems=stems,
        theta_x=np.asarray(tx), theta_y=np.asarray(ty),
        depth_mm=depth_mm,
        hu=stack(hu_rows), ddd_mc=stack(mc_rows), ddd_ad=stack(ad_rows),
        gamma=np.asarray(gam),
        r80_mc=np.asarray(r80m), r80_ad=np.asarray(r80a), r100_mc=np.asarray(r100m),
        entry_mm=np.asarray(entry),
        missing=missing,
        energy_mev=meta["energy"], patient=meta["patient"], anatomy=meta["anatomy"],
        gamma_label=label,
    )


def path_composition(t: DiagonalTransect) -> dict:
    """Per-position composition of the traversed path, surface -> MC R80.

    Returns arrays keyed ``path_mm`` (physical path length), ``lung_frac`` /
    ``soft_frac`` / ``bone_frac`` [%], ``distal_hu`` (mean HU over the last
    :data:`DISTAL_WINDOW_MM` before R80) and ``peak_width_mm`` (extent over which
    the MC depth dose exceeds 90% of its maximum -- large means a flattened dome
    rather than a sharp Bragg peak).
    """
    n = t.n
    out = {k: np.full(n, np.nan) for k in
           ("path_mm", "lung_frac", "soft_frac", "bone_frac", "distal_hu", "peak_width_mm")}
    for k in range(n):
        if k in t.missing or not np.isfinite(t.r80_mc[k]):
            continue
        d0, d1 = t.entry_mm[k], t.r80_mc[k]
        sel = (t.depth_mm >= d0) & (t.depth_mm < d1)
        seg = t.hu[k][sel]
        if seg.size == 0:
            continue
        out["path_mm"][k] = d1 - d0
        out["lung_frac"][k] = 100.0 * np.mean(seg < LUNG_HU_MAX)
        out["soft_frac"][k] = 100.0 * np.mean((seg >= LUNG_HU_MAX) & (seg < SOFT_HU_MAX))
        out["bone_frac"][k] = 100.0 * np.mean(seg >= SOFT_HU_MAX)
        win = (t.depth_mm >= max(d0, d1 - DISTAL_WINDOW_MM)) & (t.depth_mm < d1)
        out["distal_hu"][k] = float(np.mean(t.hu[k][win])) if win.any() else np.nan
        dd = t.ddd_mc[k]
        step = float(t.depth_mm[1] - t.depth_mm[0])
        out["peak_width_mm"][k] = float((dd > 0.9 * np.nanmax(dd)).sum()) * step
    return out
