"""Convert MCsquare / ADoTA dose grids from eV/g/proton to Gy (OpenTPS method).

Both the MCsquare ``Dose.mhd`` and our accumulated ``Dose_ADoTA.mhd`` are in
**eV/g/proton** (per simulated primary). This mirrors OpenTPS' MCsquare
``_importDose`` / ``_deliveredProtons`` chain: a raw dose grid is scaled to Gy with

    dose_gy = dose * delivered_protons * 1.602176e-19 * 1000 * n_fractions

where ``delivered_protons`` is summed over the plan's layers

    delivered_protons = Σ_layers  meterset * computeMU2Protons(nominalEnergy)

with the OpenTPS -> ADoTA mapping: a *beam* is a field, a *layer* is a control
point, the layer ``meterset`` is the sum of its spot weights (validated against
``total_msw``), the layer ``nominalEnergy`` is the control-point energy, and
``numberOfFractionsPlanned`` is ``Plan.n_treatment_fractions``.
``computeMU2Protons`` is
:meth:`src.beamlets.bdl.BeamDataLibrary.compute_mu_to_protons`.

The same factor is applied to both dose grids, so their ratio is preserved and
the absolute values become physical Gy. The ``1.602176e-19 * 1000`` converts
eV/g to J/kg = Gy (``1.602176e-19 J/eV * 1000 g/kg``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk

from src.beamlets.bdl import BeamDataLibrary
from src.loaders.plan_parser import Plan

logger = logging.getLogger(__name__)

__all__ = ["delivered_protons", "dose_to_gy_factor", "load_dose_gy"]

# eV/g -> J/kg = Gy, per proton.
_EVG_TO_GY = 1.602176e-19 * 1000.0


def delivered_protons(plan: Plan, bdl: BeamDataLibrary) -> float:
    """Number of protons delivered by the plan (OpenTPS ``_deliveredProtons``).

    Args:
        plan: The parsed plan.
        bdl: The beam data library (for ``computeMU2Protons``).

    Returns:
        ``Σ_layers meterset * computeMU2Protons(nominalEnergy)``, where a layer is
        a control point, its meterset is the sum of its spot weights, and its
        nominal energy is the control-point energy.
    """
    total = 0.0
    for fraction in plan.fractions:
        for field in fraction.fields:
            for control_point in field.control_points:
                meterset = sum(spot.weight for spot in control_point.spots)
                total += meterset * bdl.compute_mu_to_protons(control_point.energy_mev)
    return total


def dose_to_gy_factor(
    plan: Plan,
    bdl: BeamDataLibrary,
    n_fractions: Optional[int] = None,
) -> float:
    """Scalar that converts an eV/g/proton dose grid to Gy.

    Args:
        plan: The parsed plan.
        bdl: The beam data library.
        n_fractions: Number of fractions; defaults to ``plan.n_treatment_fractions``.

    Returns:
        ``delivered_protons * 1.602176e-19 * 1000 * n_fractions``.
    """
    fractions = plan.n_treatment_fractions if n_fractions is None else n_fractions
    return delivered_protons(plan, bdl) * _EVG_TO_GY * fractions


def load_dose_gy(
    dose_path: Path,
    plan: Plan,
    bdl: BeamDataLibrary,
    n_fractions: Optional[int] = None,
) -> sitk.Image:
    """Load an eV/g/proton dose ``.mhd`` and return it in Gy.

    Args:
        dose_path: Path to the dose ``.mhd`` (MCsquare or ADoTA).
        plan: The parsed plan (for the proton count).
        bdl: The beam data library.
        n_fractions: Number of fractions; defaults to ``plan.n_treatment_fractions``.

    Returns:
        A SimpleITK image in Gy, on the source grid (geometry preserved).
    """
    factor = dose_to_gy_factor(plan, bdl, n_fractions)
    image = sitk.ReadImage(str(dose_path))
    arr = sitk.GetArrayFromImage(image).astype(np.float32) * factor
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    logger.info(
        "Loaded dose %s -> Gy (factor=%.4g, max=%.3f Gy)",
        Path(dose_path).name,
        factor,
        float(arr.max()),
    )
    return out
