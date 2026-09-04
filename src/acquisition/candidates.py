"""Scoring candidate beamlets on a full CT grid, before any simulation.

This is the single call the active-learning loop makes. A candidate is the same
six numbers that define a generated beamlet, ``(CT, gantry, energy, steering
angles)``, and it is turned into a model input by exactly the code that would
later produce its label: :func:`src.mc_generation.robustness.field_geometry` puts
the CT into the beam's-eye frame, ``extract_beamlet_roi`` and ``flux_projection``
cut the region of interest and build the flux channel, and
``prepare_input_from_arrays`` resamples to the model grid. The analytic dose of
:mod:`src.acquisition.surrogate` then locates the Bragg peak, the thirty metrics
of :mod:`src.acquisition.features` are computed, and the frozen scorers of
:mod:`src.acquisition.scorer` rank the result. No Monte Carlo, no model.

Candidates whose ray misses the CT, or whose analytic peak leaves the crop, come
back flagged invalid rather than scored: the first is not extractable and the
second is not a valid model input.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src.acquisition.features import FeatureConfig, compute_features
from src.acquisition.scorer import DifficultyScorer
from src.acquisition.surrogate import analytic_dose, peak_inside_crop
from src.adota.config import DEFAULT_SCALE
from src.beamlets.bdl import BeamDataLibrary, angles_to_spot_position, spot_position_to_angles
from src.beamlets.cropping import extract_beamlet_roi
from src.beamlets.flux import flux_projection, flux_spatial_spread
from src.loaders.dir_based import prepare_input_from_arrays
from src.mc_generation.geometry import reduce_vacuum_to_air, resample_to_isotropic
from src.mc_generation.robustness import FieldGeometry, field_geometry
from src.mc_generation.sweep import RobustnessConfig
from src.utils.scallers import inverse_minmax

MODEL_DEPTH_VOXELS = 160


@dataclass(frozen=True)
class BeamletCandidate:
    """One candidate beamlet on a CT: a field angle, an energy and a steering pair.

    ``theta_x_deg`` / ``theta_y_deg`` are the steering angles the generator's
    lattice uses; ``angles_to_spot_position`` turns them into the spot position.
    """

    gantry_deg: float
    energy_mev: float
    theta_x_deg: float = 0.0
    theta_y_deg: float = 0.0
    candidate_id: str = ""


def prepare_ct(image: sitk.Image, iso_spacing_mm: float = 1.0) -> sitk.Image:
    """The generator's CT preprocessing: isotropic resample, vacuum clamped to air."""
    return reduce_vacuum_to_air(resample_to_isotropic(image, iso_spacing_mm))


def extract_candidate_voi(geom: FieldGeometry, candidate: BeamletCandidate, bdl: BeamDataLibrary,
                          roi_size: Tuple[int, int, int], scale: Mapping[str, float] = DEFAULT_SCALE,
                          ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The candidate's CT (HU) and flux on the model grid, ``(D, H, W)`` at 2 mm,
    or ``None`` when its ray misses the CT.

    Built through ``prepare_input_from_arrays`` and de-normalised, so the arrays
    are the ones the model would see and the ones the reference metrics used.
    """
    d_nozzle, d_smx, d_smy = bdl.distances
    spot = angles_to_spot_position(candidate.theta_x_deg, candidate.theta_y_deg, d_smx, d_smy)
    cropped_ct, entrance, _, oob = extract_beamlet_roi(
        geom.ct, d_nozzle, d_smx, d_smy, spot, geom.iso_ext, roi_size, ct_array=geom.ct_array)
    if oob or cropped_ct.shape != tuple(roi_size):
        return None
    beamlet_angles = spot_position_to_angles(spot[0], spot[1], d_smx, d_smy)
    sigmas = flux_spatial_spread(bdl, candidate.energy_mev)
    re_proj = [float(entrance[1]), float(entrance[2]), float(entrance[0])]
    flux = flux_projection(re_proj, beamlet_angles, sigmas, cropped_ct.shape,
                           spacing=np.asarray([1, 1, 1], dtype=np.float32))
    x, _ = prepare_input_from_arrays(cropped_ct, flux, candidate.energy_mev, scale=dict(scale))
    ct_hu = inverse_minmax(x[0].numpy(), scale["min_ct"], scale["max_ct"])
    return ct_hu, x[1].numpy()


def _group_by_gantry(candidates: Iterable[BeamletCandidate]) -> Dict[float, List[BeamletCandidate]]:
    groups: Dict[float, List[BeamletCandidate]] = {}
    for candidate in candidates:
        groups.setdefault(float(candidate.gantry_deg), []).append(candidate)
    return groups


def score_candidates(
    image: sitk.Image,
    candidates: Sequence[BeamletCandidate],
    bdl: BeamDataLibrary,
    scorers: Mapping[str, DifficultyScorer],
    config: RobustnessConfig = RobustnessConfig(),
    feature_config: FeatureConfig = FeatureConfig(),
    prepared: bool = False,
) -> pd.DataFrame:
    """Score every candidate on one CT. One row per candidate, in input order.

    Args:
        image: The patient CT. Preprocessed here unless ``prepared`` says it
            already went through :func:`prepare_ct`.
        candidates: The beamlets to score; grouped by gantry so the CT is put
            into each beam's-eye frame once.
        bdl: The beam model, for spot geometry and flux widths.
        scorers: ``{name: scorer}``; each adds a ``score_<name>`` column.
        config: Generator settings that shape the geometry (rotation, standoff,
            isocenter mode, ROI size); the defaults are the generator's.
        feature_config: Metric settings; the defaults reproduce the study.

    Returns:
        The candidate fields, ``valid`` and ``reason``, the thirty metrics with
        their diagnostics, and one score column per scorer. Invalid rows carry
        NaN metrics and scores.
    """
    ct = image if prepared else prepare_ct(image, config.iso_spacing_mm)
    dz_mm = float(feature_config.resolution_mm[0])
    order = {id(c): i for i, c in enumerate(candidates)}
    rows: List[Dict] = []
    for gantry, group in _group_by_gantry(candidates).items():
        geom = field_geometry(ct, gantry, config, bdl)
        for candidate in group:
            row: Dict = {**asdict(candidate), "_order": order[id(candidate)],
                         "mc_gantry_deg": geom.mc_gantry, "ct_rotation_deg": geom.ct_rotation_deg}
            voi = extract_candidate_voi(geom, candidate, bdl, tuple(config.roi_size))
            if voi is None:
                row.update(valid=False, reason="roi_out_of_bounds")
                rows.append(row)
                continue
            ct_hu, flux = voi
            dose = analytic_dose(ct_hu, flux, candidate.energy_mev, dz_mm)
            features = compute_features(ct_hu, flux, candidate.energy_mev, dose, feature_config)
            inside = peak_inside_crop(features, MODEL_DEPTH_VOXELS, dz_mm)
            row.update(features)
            row.update(valid=inside, reason="" if inside else "peak_outside_crop")
            for name, scorer in scorers.items():
                row[f"score_{name}"] = scorer.score(features) if inside else np.nan
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return frame
