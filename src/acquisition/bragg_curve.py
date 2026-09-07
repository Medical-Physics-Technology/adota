"""Analytic proton depth-dose in water (Bortfeld 1997).

The closed-form Bragg curve of T. Bortfeld, "An analytical approximation of the
Bragg curve for therapeutic proton beams", Med. Phys. 24(12), 2024-2033 (1997).
It gives the depth-dose of a proton pencil beam in water as a function of the
residual range, including range straggling and the nuclear-interaction tail,
with no fitted parameters beyond the published constants.

Used here to place the Bragg peak of a candidate beamlet from its energy alone
(:mod:`src.acquisition.surrogate`), so that the difficulty metrics can be
computed before any Monte Carlo dose exists. Only the *shape* matters for that
purpose; the absolute normalisation is kept as published so the curve is also a
sensible dose proxy.

Units: depths and ranges in **millimetres** at the interface, centimetres
internally because that is what the published constants assume.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import pbdv

ENERGY_SPREAD_TABLE = Path(__file__).parent / "data" / "hptc_energy_spread.csv"

# Bortfeld's constants for water (Table I and Eq. 27-29 of the paper).
ALPHA_CM_MEV = 0.0022     # range-energy: R0[cm] = alpha * E[MeV]^p
P_EXPONENT = 1.77
BETA_PER_CM = 0.012       # slope of the fluence reduction from nuclear interactions
GAMMA_FRACTION = 0.6      # fraction of locally absorbed energy from nuclear interactions
EPSILON_TAIL = 0.1        # fraction of primary fluence in the low-energy tail
STRAGGLING_COEF = 0.012   # sigma_mono[cm] = 0.012 * R0[cm]^0.935
STRAGGLING_EXP = 0.935

# Beyond this many straggling widths past the range the dose is zero; before
# this many widths short of it the straggling-free form is exact to float
# precision and the parabolic cylinder functions overflow anyway.
_ZETA_TAIL = 5.0
_ZETA_HEAD = 10.0


def range_to_energy_bortfeld(r0_cm: float) -> float:
    """Inverse of Bortfeld's own range-energy power law, for the straggling term."""
    return float((r0_cm / ALPHA_CM_MEV) ** (1.0 / P_EXPONENT))


@lru_cache(maxsize=1)
def _energy_spread_table() -> Tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(ENERGY_SPREAD_TABLE, delimiter=",", comments="#", skiprows=4)
    return table[:, 0], table[:, 1]


def beam_energy_spread_mev(energy_mev: float) -> float:
    """The Gaussian energy spread (one sigma, MeV) of the HPTC beam model at
    ``energy_mev``, linearly interpolated from MCsquare's BDL table.

    This is the beam that generated every ground-truth beamlet, so the analytic
    curve straggles like the data rather than like Bortfeld's 1997 beam. The
    BDL's ``EnergySpread`` column is taken as one standard deviation in MeV, the
    MCsquare convention; a FWHM reading would widen the peak by 2.35, which
    moves the peak-to-plateau ratio but not the peak position.
    """
    energies, spreads = _energy_spread_table()
    return float(np.interp(energy_mev, energies, spreads))


def straggling_sigma_cm(r0_cm: float, energy_spread_mev: Optional[float] = None) -> float:
    """Total range-straggling width for a beam of range ``r0_cm``: Bortfeld's
    Eq. 19 for the monoenergetic part plus the energy-spread term of Eq. 20.

    ``energy_spread_mev`` defaults to the HPTC beam model's value at the energy
    that Bortfeld's own range-energy law assigns to ``r0_cm``.
    """
    energy = range_to_energy_bortfeld(r0_cm)
    if energy_spread_mev is None:
        energy_spread_mev = beam_energy_spread_mev(energy)
    sigma_mono = STRAGGLING_COEF * r0_cm**STRAGGLING_EXP
    sigma_e_term = energy_spread_mev * ALPHA_CM_MEV * P_EXPONENT * energy ** (P_EXPONENT - 1.0)
    return float(np.sqrt(sigma_mono**2 + sigma_e_term**2))


def _dose_no_straggling(z_cm: np.ndarray, r0_cm: float) -> np.ndarray:
    """Eq. 27: the curve for a monoenergetic beam, valid for z < R0."""
    residual = np.clip(r0_cm - z_cm, 1e-9, None)
    fluence = 1.0 / (1.0 + BETA_PER_CM * r0_cm)
    inv_p = 1.0 / P_EXPONENT
    norm = 1.0 / (P_EXPONENT * ALPHA_CM_MEV**inv_p)          # 17.93 in the paper
    term_a = norm * residual ** (inv_p - 1.0)
    term_b = norm * (BETA_PER_CM + GAMMA_FRACTION * BETA_PER_CM * P_EXPONENT
                     + EPSILON_TAIL * P_EXPONENT / r0_cm) * residual**inv_p
    return fluence * (term_a + term_b)


def _dose_with_straggling(z_cm: np.ndarray, r0_cm: float, sigma_cm: float) -> np.ndarray:
    """Eq. 28/29: the straggled curve, via parabolic cylinder functions."""
    zeta = (r0_cm - z_cm) / sigma_cm
    d_a = pbdv(-1.0 / P_EXPONENT, -zeta)[0]
    d_b = pbdv(-1.0 / P_EXPONENT - 1.0, -zeta)[0]
    # Gamma(1/p), not Gamma(1/p + 1): the Gaussian convolution of (R0 - z)^(1/p - 1)
    # brings Gamma(1/p), and the 1/p of the second term's Gamma(1/p + 1) is already
    # folded into the bracket's coefficients (beta/p + gamma*beta + eps/R0).
    # Verified against a numerical convolution of Eq. 27 to 0.1 percent.
    prefactor = (np.exp(-(zeta**2) / 4.0) * sigma_cm ** (1.0 / P_EXPONENT) * gamma_fn(1.0 / P_EXPONENT)
                 / (np.sqrt(2.0 * np.pi) * P_EXPONENT * ALPHA_CM_MEV ** (1.0 / P_EXPONENT)
                    * (1.0 + BETA_PER_CM * r0_cm)))
    bracket = (d_a / sigma_cm
               + (BETA_PER_CM / P_EXPONENT + GAMMA_FRACTION * BETA_PER_CM + EPSILON_TAIL / r0_cm) * d_b)
    return prefactor * bracket


def bragg_curve(depth_mm: np.ndarray, r0_mm: float,
                energy_spread_mev: Optional[float] = None) -> np.ndarray:
    """Depth-dose in water of a pencil beam with range ``r0_mm``, at ``depth_mm``.

    ``r0_mm`` is Bortfeld's range, which coincides with the distal 80 percent
    depth to within a fraction of a millimetre, so the Grevillot R80 of
    :mod:`src.processing.range_energy` can be passed directly.
    ``energy_spread_mev`` is the beam's Gaussian energy spread (one sigma); by
    default the HPTC beam model's value.

    Returns the dose in Bortfeld's units (MeV per gram per primary fluence
    unit); the caller normalises. Zero at and beyond ``r0 + 5 sigma``.
    """
    depth = np.asarray(depth_mm, dtype=np.float64) / 10.0
    r0 = float(r0_mm) / 10.0
    if r0 <= 0.0:
        return np.zeros_like(depth)
    sigma = straggling_sigma_cm(r0, energy_spread_mev)
    zeta = (r0 - depth) / sigma

    out = np.zeros_like(depth)
    head = zeta >= _ZETA_HEAD
    body = (zeta < _ZETA_HEAD) & (zeta > -_ZETA_TAIL)
    out[head] = _dose_no_straggling(depth[head], r0)
    out[body] = _dose_with_straggling(depth[body], r0, sigma)
    return np.clip(out, 0.0, None)


def peak_depth_mm(r0_mm: float, step_mm: float = 0.1,
                  energy_spread_mev: Optional[float] = None) -> float:
    """Depth of the Bragg-peak maximum for range ``r0_mm``, by dense evaluation."""
    sigma_mm = 10.0 * straggling_sigma_cm(r0_mm / 10.0, energy_spread_mev)
    z = np.arange(0.0, r0_mm + 5.0 * sigma_mm, step_mm)
    return float(z[np.argmax(bragg_curve(z, r0_mm, energy_spread_mev))])
