"""Dataclasses for pure analysis pipelines (no model inference)."""

from __future__ import annotations

from dataclasses import dataclass


# ── Tissue-interface analysis ───────────────────────────────────────────────


@dataclass
class BeamletResult:
    """Container for a single beamlet's analysis results.

    Used by ``training_set_analysis.py``.
    """

    sample_id: str
    energy_mev: float
    bp_idx: tuple  # (depth, y, x) voxel indices of the Bragg peak
    sigma_hu: float  # σ_HU in the spherical neighbourhood
    tv: float  # Total Variation of HU in the spherical neighbourhood
    cv: float  # Coefficient of Variation of HU in the spherical neighbourhood
    label: str  # "interface" or "homogeneous"
    gpr: float  # γ (2%/2mm) pass rate [%]
    rmse: float
    mape: float
    rde: float
    calc_time: float


# ── Threshold sweep ─────────────────────────────────────────────────────────


@dataclass
class BeamletRecord:
    """Lightweight container for a single beamlet read from CSV.

    Used by ``threshold_sweep.py``.
    """

    sample_id: str
    energy_mev: float
    sigma_hu: float
    tv: float
    cv: float
    gpr: float  # GPR [%]
    rmse: float  # RMSE [Gy]
    mape: float  # MAPE [%]
    rde: float  # RDE [%]


@dataclass
class SweepPoint:
    """Metrics at a single threshold value.

    Used by ``threshold_sweep.py``.
    """

    tau: float
    n_homo: int
    n_intf: int
    pct_intf: float
    gpr_homo: float
    gpr_intf: float
    gpr_gap: float  # homo − intf (positive = interface is worse)
    rde_homo: float
    rde_intf: float
    rde_gap: float
    mape_homo: float
    mape_intf: float
    mape_gap: float
    rmse_homo: float
    rmse_intf: float
    rmse_gap: float
