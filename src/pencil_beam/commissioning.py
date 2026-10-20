"""Commission the DoseCUDA pencil beam from MCsquare water-phantom beamlets.

This is DoseCUDA's own beam-modelling recipe (``beam_modeling/impt_lookup_tables.py``
in its repository), with MCsquare + our BDL as the reference instead of a clinical
TPS, so that the pencil beam reproduces the beamlets ADoTA was trained on:

1. For every energy node, a single central spot is simulated into a water box at
   several source-to-surface distances (the box surface ``s`` mm upstream of the
   isocenter), always at the canonical MC gantry 90 the generator uses.
2. Per depth slice: the integrated depth dose (lateral sum) and a 2D Gaussian fit
   of the lateral profile (``sigma`` = mean of the two axes; the kernel is
   rotationally symmetric while the BDL spot is not, and that residual is
   recorded rather than hidden).
3. R80 from the IDD; at the reference depth ``0.7 R80`` the width ``sigma_ref``
   of each SSD is fitted with a quadratic in the kernel's distance
   ``d = distance_to_source - z_w + 0.7 R80`` -> ``sigma_air``.
4. ``sigma_mcs(z_w) = sigma(z_w) - sigma_ref`` and ``IDD(z_w)`` averaged over the
   SSDs on a 400-point depth grid.

Depths are water-equivalent: the HU of the "water" box is ``Schneider_AT_AG_SI4``
in MCsquare's default scanner (RLSP 1.017 at 100 MeV, not 1), and the kernel
ray-traces RLSP, so physical depth is converted with the same RLSP curve.
IDD is divided by the plan weight, so a spot of weight ``w`` in the kernel lands
on the MC dose scale of a ``w``-weighted MCsquare spot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.optimize import curve_fit

from src.datasets.phantom import PhantomSpec, build_phantom_image
from src.mc_generation.geometry import mc_isocenter
from src.mc_generation.mcsquare_runner import MCSQUARE_NUCLEAR_DATA_MAX_MEV
from src.pencil_beam.beam_model import LUT_LENGTH, PencilBeamModel

logger = logging.getLogger(__name__)

__all__ = [
    "CommissioningConfig",
    "check_energies_simulable",
    "DepthProfile",
    "commissioning_phantom",
    "depth_profile",
    "distal_r80",
    "fit_energy_node",
    "fit_model",
    "fit_primary_sigma",
    "kernel_halo",
    "kernel_sigma_profile",
    "physical_to_wet",
    "run_commissioning_beamlets",
]


@dataclass
class CommissioningConfig:
    """Geometry, statistics and fit settings of a commissioning run (YAML ``commissioning:``)."""

    energies: List[float] = field(default_factory=lambda: [float(e) for e in range(70, 255, 5)])
    surface_to_iso_mm: List[float] = field(default_factory=lambda: [100.0, 175.0, 250.0, 325.0])
    lateral_size_mm: int = 120
    water_length_mm: int = 480
    air_front_mm: int = 20
    water_hu: int = 0
    air_hu: int = -1024
    num_primaries: float = 2e6
    num_threads: int = 16
    rng_seed: int = 0
    spot_weight: float = 1000.0
    lut_depth_factor: float = 1.3          # LUT spans 0 .. factor * R80
    sigma_reference_fraction: float = 0.7  # z_ref = fraction * R80 (DoseCUDA: 0.7)
    min_slice_fraction: float = 0.01       # fit a slice only above this share of the peak slice
    nozzle_distance_mm: float = 420.0      # BDL "Nozzle exit to Isocenter distance"


# MCsquare starts each proton at the nozzle exit and moves it through vacuum to the
# CT boundary (Transport_to_CT). Measured 2026-10-20 (120 MeV, water box with 20 mm
# front air): when the nozzle exit lies more than ~340 mm upstream of the CT face the
# beam enters late by exactly the excess (60, 35, 10 mm at 400, 375, 350 mm), and
# the skipped water is simply missing. A nozzle inside the grid would skip whatever
# lies upstream of it. Generated datasets put the isocenter at the grid centre (the
# nozzle 160-220 mm upstream of the face), so only commissioning geometries are
# guarded here.
MAX_SAFE_NOZZLE_UPSTREAM_MM = 300.0


@dataclass
class DepthProfile:
    """Per-slice quantities of one MC water-phantom beamlet (beam along x)."""

    x_mm: np.ndarray          # slice centres, mm from the grid's x = 0 face
    idd: np.ndarray           # lateral integral, dose * mm^2 (per unit spot weight)
    sigma_y: np.ndarray       # fitted Gaussian sigma along image y, mm (nan = not fitted)
    sigma_z: np.ndarray
    centre_y: np.ndarray      # fitted centre, mm from the slice's lower corner
    centre_z: np.ndarray

    @property
    def sigma(self) -> np.ndarray:
        return 0.5 * (self.sigma_y + self.sigma_z)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in ("x_mm", "idd", "sigma_y", "sigma_z", "centre_y", "centre_z")}

    @classmethod
    def from_dict(cls, d) -> "DepthProfile":
        return cls(**{k: np.asarray(d[k]) for k in ("x_mm", "idd", "sigma_y", "sigma_z", "centre_y", "centre_z")})


# ---------------------------------------------------------------- phantom + MC
def commissioning_phantom(cfg: CommissioningConfig, surface_to_iso_mm: float
                          ) -> Tuple[sitk.Image, List[float]]:
    """Water box with an air slab in front, and the MC isocenter ``s`` mm past its surface.

    The air is front-only (full width), never an all-faces shell, so the beam
    never grazes a lateral air boundary (CLAUDE.md phantom convention).

    Raises:
        ValueError: If MCsquare's nozzle exit would sit inside the grid or too far
            upstream of it (see ``MAX_SAFE_NOZZLE_UPSTREAM_MM``).
    """
    upstream = cfg.nozzle_distance_mm - (cfg.air_front_mm + surface_to_iso_mm)
    if not 0.0 <= upstream <= MAX_SAFE_NOZZLE_UPSTREAM_MM:
        raise ValueError(
            f"surface_to_iso_mm={surface_to_iso_mm} puts the nozzle exit {upstream:.0f} mm upstream of "
            f"the CT face; MCsquare needs 0..{MAX_SAFE_NOZZLE_UPSTREAM_MM:.0f} mm (air_front_mm + "
            f"surface_to_iso_mm in [{cfg.nozzle_distance_mm - MAX_SAFE_NOZZLE_UPSTREAM_MM:.0f}, "
            f"{cfg.nozzle_distance_mm:.0f}])")
    n_lat = int(cfg.lateral_size_mm)
    spec = PhantomSpec(
        size=(int(cfg.air_front_mm + cfg.water_length_mm), n_lat, n_lat),
        water_hu=int(cfg.water_hu), air_hu=int(cfg.air_hu),
        air_front_mm=int(cfg.air_front_mm), name="commissioning")
    ct = build_phantom_image(spec)
    iso = mc_isocenter(ct)
    iso[0] = float(cfg.air_front_mm + surface_to_iso_mm)
    return ct, iso


def check_energies_simulable(energies: Sequence[float]) -> None:
    """Refuse energies MCsquare cannot simulate correctly (above its nuclear data).

    Raises:
        ValueError: Naming the offending energies and the reason.
    """
    bad = [float(e) for e in energies if float(e) > MCSQUARE_NUCLEAR_DATA_MAX_MEV]
    if bad:
        raise ValueError(
            f"energies {bad} MeV exceed {MCSQUARE_NUCLEAR_DATA_MAX_MEV:.0f} MeV, where MCsquare's ICRU "
            "nuclear tables end; MCsquare's dose there is undefined (and it crashes from ~259 MeV), so "
            "it cannot serve as the reference")


def run_commissioning_beamlets(runner, cfg: CommissioningConfig, out_dir: Path,
                               energies: Optional[Sequence[float]] = None,
                               surfaces: Optional[Sequence[float]] = None) -> List[Path]:
    """Simulate every (energy, SSD) central spot and save its dose + depth profile.

    Resumable: a run whose ``.npz`` exists is skipped. Each file holds the full
    dose grid (float32, compressed), the profile fit, and the run's provenance.
    """
    check_energies_simulable(cfg.energies if energies is None else energies)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in (cfg.surface_to_iso_mm if surfaces is None else surfaces):
        ct, iso = commissioning_phantom(cfg, s)
        for energy in (cfg.energies if energies is None else energies):
            path = out_dir / f"e{energy:07.3f}_s{s:05.1f}.npz"
            written.append(path)
            if path.exists():
                continue
            dose_img, sim_res = runner.run_beamlet(
                ct, energy=float(energy), gantry_angle=90.0, spot_xy=(0.0, 0.0), isocenter=iso,
                num_primaries=cfg.num_primaries, num_threads=cfg.num_threads,
                rng_seed=cfg.rng_seed, weight=cfg.spot_weight)
            dose = sitk.GetArrayFromImage(dose_img).astype(np.float32)
            prof = depth_profile(dose, spacing_mm=1.0, weight=cfg.spot_weight,
                                 min_slice_fraction=cfg.min_slice_fraction)
            np.savez_compressed(
                path, dose=dose, energy=float(energy), surface_to_iso_mm=float(s),
                iso_mc=np.asarray(iso), mc_seconds=float(sim_res["mc_seconds"]),
                stat_uncertainty=float(sim_res["stat_uncertainty"]), **prof.as_dict())
            logger.info("  E=%.1f MeV s=%.0f mm: %.0f s, unc %.2f%% -> %s", energy, s,
                        sim_res["mc_seconds"], sim_res["stat_uncertainty"], path.name)
    return written


# ---------------------------------------------------------------- profile fits
def _gauss2d(yz, amp, y0, z0, sy, sz):
    y, z = yz
    return amp * np.exp(-((y - y0) ** 2 / (2 * sy ** 2) + (z - z0) ** 2 / (2 * sz ** 2)))


def fit_lateral_gaussian(plane: np.ndarray, spacing_mm: float = 1.0) -> Tuple[float, float, float, float]:
    """``(sigma_y, sigma_z, centre_y, centre_z)`` (mm) of a 2D Gaussian fit to a ``(z, y)`` plane.

    Fitted to the whole plane, as DoseCUDA does, so the low nuclear halo barely
    moves the core width the kernel's primary Gaussian represents.
    """
    nz, ny = plane.shape
    peak = float(plane.max())
    if peak <= 0:
        return (np.nan,) * 4
    z, y = np.meshgrid((np.arange(nz) + 0.5) * spacing_mm, (np.arange(ny) + 0.5) * spacing_mm, indexing="ij")
    w = plane / plane.sum()
    y0, z0 = float((w * y).sum()), float((w * z).sum())
    p0 = [1.0, y0, z0, 3.0, 3.0]
    try:
        popt, _ = curve_fit(_gauss2d, (y.ravel(), z.ravel()), (plane / peak).ravel(), p0=p0, maxfev=4000)
    except RuntimeError:
        return (np.nan,) * 4
    return abs(popt[3]), abs(popt[4]), popt[1], popt[2]


def depth_profile(dose: np.ndarray, spacing_mm: float = 1.0, weight: float = 1.0,
                  min_slice_fraction: float = 0.01) -> DepthProfile:
    """IDD and lateral Gaussian width of every x slice of a ``(z, y, x)`` beamlet dose."""
    nx = dose.shape[2]
    idd = dose.sum(axis=(0, 1)) * spacing_mm ** 2 / weight
    slice_max = dose.max(axis=(0, 1))
    fit = np.full((nx, 4), np.nan)
    for i in np.flatnonzero(slice_max > min_slice_fraction * slice_max.max()):
        fit[i] = fit_lateral_gaussian(dose[:, :, i], spacing_mm)
    return DepthProfile(x_mm=(np.arange(nx) + 0.5) * spacing_mm, idd=idd, sigma_y=fit[:, 0],
                        sigma_z=fit[:, 1], centre_y=fit[:, 2], centre_z=fit[:, 3])


def kernel_halo(wet_mm: np.ndarray, r80_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """``(halo_sigma, halo_weight)`` exactly as DoseCUDA's ``IMPTBeam::nuclearHalo``."""
    wet = np.clip(np.asarray(wet_mm, dtype=float), 0.1, r80_mm - 0.1)
    r = float(r80_mm)
    sigma = (2.85 + 0.0014 * r * np.log(wet + 3.0) + 0.06 * wet - 7.4e-5 * wet ** 2
             - 0.22 * r / (wet - r - 5.0) ** 2)
    weight = (0.052 * np.log(1.13 + wet / (11.2 - 0.023 * r))
              + 0.35 * (0.0017 * r ** 2 - r) / ((r + 3.0) ** 2 - wet ** 2)
              - 1.61e-9 * wet * (r + 3.0) ** 2)
    return np.maximum(sigma, 0.0), np.clip(weight, 0.0, 0.9)


def _primary_plus_halo(halo_sigma: float, halo_weight: float):
    def model(yz, amp, y0, z0, sp):
        y, z = yz
        r2 = (y - y0) ** 2 + (z - z0) ** 2
        st2 = sp ** 2 + halo_sigma ** 2
        return amp * ((1 - halo_weight) / sp ** 2 * np.exp(-r2 / (2 * sp ** 2))
                      + halo_weight / st2 * np.exp(-r2 / (2 * st2)))
    return model


def fit_primary_sigma(plane: np.ndarray, halo_sigma: float, halo_weight: float,
                      spacing_mm: float = 1.0, half_window_mm: float = 40.0) -> float:
    """Primary sigma (mm) of a ``(z, y)`` plane in the kernel's own lateral form.

    The kernel deposits ``(1 - w) G(sigma) + w G(hypot(sigma, sigma_halo))`` with
    ``w, sigma_halo`` from its hard-coded halo. Fitting MC with that form (rather
    than a single Gaussian) makes the tabulated sigma mean what the kernel does
    with it; a single-Gaussian width fed to the kernel comes out ~0.1 mm too wide
    with a 3-8% low central-axis dose (measured at 120 MeV).
    """
    peak = float(plane.max())
    if peak <= 0:
        return float("nan")
    iz, iy = np.unravel_index(int(np.argmax(plane)), plane.shape)
    h = int(round(half_window_mm / spacing_mm))
    zs = slice(max(0, iz - h), iz + h + 1)
    ys = slice(max(0, iy - h), iy + h + 1)
    sub = plane[zs, ys] / peak
    z, y = np.meshgrid((np.arange(zs.start, zs.start + sub.shape[0]) + 0.5) * spacing_mm,
                       (np.arange(ys.start, ys.start + sub.shape[1]) + 0.5) * spacing_mm, indexing="ij")
    w = sub / sub.sum()
    p0 = [10.0, float((w * y).sum()), float((w * z).sum()), 3.0]
    try:
        popt, _ = curve_fit(_primary_plus_halo(halo_sigma, halo_weight), (y.ravel(), z.ravel()), sub.ravel(),
                            p0=p0, bounds=([0, -np.inf, -np.inf, 0.2], [np.inf, np.inf, np.inf, 50.0]),
                            maxfev=4000)
    except RuntimeError:
        return float("nan")
    return float(popt[3])


def kernel_sigma_profile(dose: np.ndarray, wet_mm: np.ndarray, r80_mm: float, spacing_mm: float = 1.0,
                         min_slice_fraction: float = 0.01) -> np.ndarray:
    """Kernel-consistent primary sigma of every x slice (nan where too little dose)."""
    halo_sigma, halo_weight = kernel_halo(wet_mm, r80_mm)
    slice_max = dose.max(axis=(0, 1))
    out = np.full(dose.shape[2], np.nan)
    for i in np.flatnonzero(slice_max > min_slice_fraction * slice_max.max()):
        out[i] = fit_primary_sigma(dose[:, :, i], float(halo_sigma[i]), float(halo_weight[i]), spacing_mm)
    return out


def physical_to_wet(x_mm: np.ndarray, air_front_mm: float, rlsp_water: float, rlsp_air: float) -> np.ndarray:
    """Water-equivalent depth (mm) of slice centres in the front-air water box."""
    x = np.asarray(x_mm, dtype=float)
    in_air = np.minimum(x, air_front_mm)
    in_water = np.maximum(x - air_front_mm, 0.0)
    return in_air * rlsp_air + in_water * rlsp_water


def distal_r80(wet_mm: np.ndarray, idd: np.ndarray) -> float:
    """Distal depth where the IDD falls to 80% of its maximum (linear interpolation)."""
    i_max = int(np.argmax(idd))
    level = 0.8 * float(idd[i_max])
    tail = np.flatnonzero(idd[i_max:] < level)
    if tail.size == 0:
        return float("nan")
    j = i_max + int(tail[0])
    return float(np.interp(level, [idd[j], idd[j - 1]], [wet_mm[j], wet_mm[j - 1]]))


# ---------------------------------------------------------------- table fits
def fit_energy_node(
    energy: float,
    profiles: Sequence[DepthProfile],
    wets: Sequence[np.ndarray],
    surfaces_mm: Sequence[float],
    air_front_mm: float,
    source_distance: float,
    cfg: CommissioningConfig,
    sigmas: Optional[Sequence[np.ndarray]] = None,
) -> dict:
    """Fit one energy's tables from its SSD runs (same order in all sequences).

    Args:
        sigmas: Per-run primary sigma per slice (:func:`kernel_sigma_profile`);
            defaults to the single-Gaussian ``profile.sigma``.

    Returns a dict with ``r80``, ``sigma_air_coef``, ``depths``, ``sigmas``,
    ``idds``, ``reference_sigma`` and per-SSD diagnostics.
    """
    sigmas = [p.sigma for p in profiles] if sigmas is None else list(sigmas)
    r80s = np.array([distal_r80(w, p.idd) for p, w in zip(profiles, wets)])
    r80 = float(np.nanmean(r80s))
    z_ref = cfg.sigma_reference_fraction * r80
    sig_ref, d_ref, valid = [], [], []
    for p, sig, w, s in zip(profiles, sigmas, wets, surfaces_mm):
        ok = np.isfinite(sig)
        valid.append(ok)
        sig_ref.append(float(np.interp(z_ref, w[ok], sig[ok])))
        x_ref = float(np.interp(z_ref, w, p.x_mm))
        iso_x = air_front_mm + s
        d_ref.append(source_distance - (iso_x - x_ref))       # = kernel d at the reference voxel
    sig_ref, d_ref = np.array(sig_ref), np.array(d_ref)
    deg = min(2, len(d_ref) - 1)
    coef = np.zeros(3)
    coef[2 - deg:] = np.polyfit(d_ref, sig_ref, deg)

    depths = np.linspace(0.0, cfg.lut_depth_factor * r80, LUT_LENGTH)
    idds = np.mean([np.interp(depths, w, p.idd, right=0.0) for p, w in zip(profiles, wets)], axis=0)
    sigma_mcs = np.mean([np.interp(depths, w[ok], sig[ok] - sr)
                         for sig, w, ok, sr in zip(sigmas, wets, valid, sig_ref)], axis=0)
    d_iso = source_distance + z_ref                           # in air at the isocenter
    reference_sigma = float(np.polyval(coef, d_iso) + sigma_mcs[0])
    asym = [float(np.nanmean(np.abs(p.sigma_y - p.sigma_z))) for p in profiles]
    residual = sig_ref - np.polyval(coef, d_ref)
    return {
        "energy": float(energy), "r80": r80, "sigma_air_coef": coef, "depths": depths,
        "sigmas": sigma_mcs, "idds": idds, "reference_sigma": reference_sigma,
        "diagnostics": {"r80_per_ssd": r80s.tolist(), "sigma_ref_per_ssd": sig_ref.tolist(),
                        "d_ref_per_ssd": d_ref.tolist(), "sigma_air_fit_residual_mm": residual.tolist(),
                        "mean_abs_sigma_y_minus_z_mm": asym},
    }


def fit_model(nodes: Sequence[dict], vsadx: float, vsady: float, meta: dict) -> PencilBeamModel:
    """Assemble fitted energy nodes into a :class:`PencilBeamModel`."""
    nodes = sorted(nodes, key=lambda n: n["energy"])
    meta = dict(meta)
    meta["nodes"] = {f"{n['energy']:.2f}": {"r80_mm": n["r80"], **n["diagnostics"]} for n in nodes}
    return PencilBeamModel(
        vsadx=vsadx, vsady=vsady, energies=np.array([n["energy"] for n in nodes]),
        r80=np.array([n["r80"] for n in nodes]),
        sigma_air_coef=np.array([n["sigma_air_coef"] for n in nodes]),
        depths=np.array([n["depths"] for n in nodes]), sigmas=np.array([n["sigmas"] for n in nodes]),
        idds=np.array([n["idds"] for n in nodes]),
        reference_sigma=np.array([n["reference_sigma"] for n in nodes]), meta=meta)
