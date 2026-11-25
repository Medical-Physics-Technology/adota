"""A beamlet's inputs on the model grid and its profiles along the beam axis.

Everything here is on the 2 mm model grid, depth first ``(D, H, W)``, the grid
the 30 difficulty features are computed on, and the Bragg peak is located with
the analytic surrogate dose (:func:`src.acquisition.surrogate.analytic_dose`),
as the features locate it. The Monte Carlo dose, when stored, is carried along
only to compare the two peaks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import h5py
import numpy as np

from src.acquisition.candidates import model_grid_inputs
from src.acquisition.features import FeatureConfig
from src.acquisition.surrogate import analytic_dose, bragg_peak_index, cumulative_wepl
from src.active_learning.ensemble.engine_grids import downsample2
from src.active_learning.retrospective.scoring import read_stored_inputs
from src.adota.config import DEFAULT_SCALE
from src.processing.mcsquare_calibration import hu_to_rsp_mcsquare
from src.processing.pflugfelder_hi import DEFAULT_ENERGY_MEV
from src.utils.dose_grid_utils import estimate_bp_range


@dataclass(frozen=True)
class BeamletInputs:
    """One beamlet's model-grid inputs. Arrays are ``(D, H, W)``, depth first, 2 mm voxels."""

    beamlet_id: str
    source: str                          # "grid" or "pool"
    ct_hu: np.ndarray
    flux: np.ndarray                     # normalised flux channel
    energy_mev: float
    mc_dose: Optional[np.ndarray] = None  # Monte Carlo dose on the same grid, if stored


@dataclass(frozen=True)
class BeamAxisProfile:
    """Profiles along the beam axis through the analytic peak's lateral voxel.

    All 1-D arrays have length ``D``. Depths are geometric, in mm, at voxel centres.
    """

    depth_mm: np.ndarray
    hu_axis: np.ndarray                  # HU on the axis through peak_yx
    hu_flux_weighted: np.ndarray         # per-slice flux-weighted mean HU (the features' profile)
    rsp_axis: np.ndarray                 # relative stopping power on the axis
    wepl_axis_mm: np.ndarray             # cumulative WEPL at each voxel's exit, on the axis
    idd_analytic: np.ndarray             # laterally integrated analytic dose, max 1
    idd_mc: Optional[np.ndarray]         # laterally integrated MC dose, max 1, if stored
    analytic_dose: np.ndarray            # (D, H, W), max 1
    peak_index: Tuple[int, int, int]     # analytic peak (d, h, w)
    bp_slice: int                        # argmax of idd_analytic (the features' bp_slice)
    bp_zone_mm: Tuple[float, float]      # (bp_range_min_mm, bp_range_max_mm), as the features define them
    mc_bp_slice: Optional[int]           # argmax of idd_mc
    sphere_radius_mm: float              # radius of the edge-metric sphere around the peak
    dz_mm: float


def load_grid_beamlet(block_dir: Path, stem: str, energy_mev: float, beamlet_id: Optional[str] = None
                      ) -> BeamletInputs:
    """A saved grid beamlet (``{stem}_ct.npy``, ``{stem}_flux.npy``, ``{stem}_ds.npy`` on the 1 mm crop).

    CT and flux go through :func:`src.acquisition.candidates.model_grid_inputs`, the path the saved
    feature CSV took. The MC dose is block-averaged 2 x 2 x 2 and transposed with ``(2, 0, 1)`` so
    depth is first and the lateral axes match that output (checked against the flux centre).
    """
    block_dir = Path(block_dir)
    ct = np.load(block_dir / f"{stem}_ct.npy")
    flux = np.load(block_dir / f"{stem}_flux.npy")
    ds = np.load(block_dir / f"{stem}_ds.npy")
    ct_hu, flux_grid = model_grid_inputs(ct, flux, energy_mev)
    mc = np.ascontiguousarray(downsample2(np.asarray(ds, dtype=np.float32)).transpose(2, 0, 1))
    return BeamletInputs(beamlet_id=beamlet_id or f"{block_dir.name}/{stem}", source="grid", ct_hu=ct_hu,
                         flux=flux_grid, energy_mev=float(energy_mev), mc_dose=mc)


def _file_scale(handle: h5py.File, default: Mapping[str, float]) -> Dict[str, float]:
    raw = handle.attrs.get("scale_json")
    if raw is None:
        return dict(default)
    if isinstance(raw, bytes):
        raw = raw.decode()
    return dict(json.loads(raw)) if isinstance(raw, str) else dict(default)


def load_pool_beamlet(h5_path: Path, sample_id: str, scale: Mapping[str, float] = DEFAULT_SCALE) -> BeamletInputs:
    """A record of a beamlet-pool HDF5 export (``ct``, ``flux``, ``dose`` on the 2 mm grid).

    The file's ``scale_json`` attribute wins over ``scale`` when present. The dose is transposed
    like the CT and left in the file's normalised units (profiles are normalised anyway).
    """
    with h5py.File(h5_path, "r") as handle:
        group = handle[sample_id]
        record = read_stored_inputs(group, sample_id, _file_scale(handle, scale))
        dose = np.ascontiguousarray(np.asarray(group["dose"][:], dtype=np.float32).transpose(2, 0, 1))
    return BeamletInputs(beamlet_id=sample_id, source="pool", ct_hu=record.ct_hu, flux=record.flux,
                         energy_mev=float(record.energy_mev), mc_dose=dose)


def flux_weighted_hu(ct_hu: np.ndarray, flux: np.ndarray, frac: float) -> np.ndarray:
    """Per-slice flux-weighted mean HU, the rule of ``analyse_density_regions`` over every slice."""
    weights = np.abs(flux).astype(np.float64)
    f_max = weights.max(axis=(1, 2), keepdims=True)
    mask = (weights >= frac * f_max) & (f_max >= 1e-12)
    w = np.where(mask, weights, 0.0)
    total = w.sum(axis=(1, 2))
    weighted = (w * ct_hu).sum(axis=(1, 2)) / np.where(total > 0, total, 1.0)
    use_mean = ~(mask.any(axis=(1, 2)) & (total > 0))
    return np.where(use_mean, ct_hu.mean(axis=(1, 2)), weighted)


def _normalised_idd(dose: np.ndarray) -> np.ndarray:
    idd = dose.sum(axis=(1, 2)).astype(np.float64)
    peak = idd.max()
    return idd / peak if peak > 0 else idd


def beam_axis_profile(inputs: BeamletInputs, config: FeatureConfig = FeatureConfig()) -> BeamAxisProfile:
    """Profiles along the beam axis, with the peak located as the features locate it.

    ``depth_mm`` is the slice index times ``dz``, the convention of the features'
    ``bp_range_min_mm`` / ``bp_range_max_mm``, so the zone marks line up with the curves.
    """
    dz_mm = float(config.resolution_mm[0])
    ct_hu, flux = inputs.ct_hu, inputs.flux
    dose = analytic_dose(ct_hu, flux, inputs.energy_mev, dz_mm)
    peak = bragg_peak_index(dose)
    h_peak, w_peak = peak[1], peak[2]
    idd_analytic = _normalised_idd(dose)
    z_min, z_max = estimate_bp_range(ct_hu, dose, proximal_fraction=config.proximal_fraction,
                                     fall_fraction=config.fall_fraction)
    hu_axis = ct_hu[:, h_peak, w_peak]
    idd_mc = _normalised_idd(inputs.mc_dose) if inputs.mc_dose is not None else None
    return BeamAxisProfile(
        depth_mm=np.arange(ct_hu.shape[0]) * dz_mm,
        hu_axis=hu_axis,
        hu_flux_weighted=flux_weighted_hu(ct_hu, flux, config.flux_threshold_frac),
        rsp_axis=hu_to_rsp_mcsquare(hu_axis, energy=DEFAULT_ENERGY_MEV),
        wepl_axis_mm=cumulative_wepl(ct_hu, dz_mm)[:, h_peak, w_peak],
        idd_analytic=idd_analytic,
        idd_mc=idd_mc,
        analytic_dose=dose,
        peak_index=(int(peak[0]), int(h_peak), int(w_peak)),
        bp_slice=int(np.argmax(idd_analytic)),
        bp_zone_mm=(float(z_min) * dz_mm, float(z_max) * dz_mm),
        mc_bp_slice=int(np.argmax(idd_mc)) if idd_mc is not None else None,
        sphere_radius_mm=float(config.sphere_radius_mm),
        dz_mm=dz_mm,
    )


__all__ = ["BeamAxisProfile", "BeamletInputs", "beam_axis_profile", "flux_weighted_hu", "load_grid_beamlet",
           "load_pool_beamlet"]
