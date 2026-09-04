"""Sweep definition for the MC generators: config, angle lattice, gantry draws.

The knobs that describe *what* to simulate -- the beamlet-angle lattice, the field
angles drawn per patient, and the CT coverage those angles require -- separated from
the orchestration in :mod:`src.mc_generation.robustness` that runs them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk

from src.beamlets.bdl import angles_to_spot_position

__all__ = [
    "RobustnessConfig", "robustness_config_from_dict", "energy_tag", "angle_tag",
    "build_angle_grid", "resolve_gantry", "resolve_gantries",
    "sweep_z_half_extent_mm", "sweep_lateral_half_extents", "max_theta_x_deg",
    "sweep_fits_ct",
]


@dataclass
class RobustnessConfig:
    energies: List[float] = field(default_factory=lambda: [90.0, 140.0, 200.0])
    theta_x_range: Tuple[float, float] = (-2.0, 2.0)
    theta_y_range: Tuple[float, float] = (-2.0, 2.0)
    grid_n: int = 18
    angles: Optional[List[Tuple[float, float]]] = None
    """Explicit ``(theta_x, theta_y)`` list instead of the full ``grid_n`` sweep
    (e.g. the 4 corners + centre smoke test). Every entry must sit exactly on the
    ``grid_n`` lattice so the GPR panels stay indexable; unlisted cells stay NaN."""
    gantry_mode: str = "fixed"           # "fixed" | "uniform_random" | "bimodal_random"
    n_gantry: int = 1                    # gantry draws per patient (random modes only)
    gantry_value: float = 90.0
    gantry_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((30.0, 120.0), (240.0, 330.0))
    gantry_min: float = 0.0              # uniform_random bounds [min, max)
    gantry_max: float = 360.0
    gantry_seed: int = 1234
    rotate_to_canonical: bool = True     # rotate CT so beam is axis-aligned (gantry != 90)
    isocenter_mode: str = "grid_center"  # "grid_center" | "body_com" (aim at patient COM)
    body_hu_threshold: float = -500.0    # body mask threshold (includes lung, excludes air)
    border_only: bool = False            # generate only the border (corner/edge) beamlets
    beam_entrance_standoff_mm: float = 20.0
    """Air gap left in front of the patient when a rotated grid is trimmed back to
    the beam-axis window (see :func:`_field_geometry`). The gantry-90 data the model
    was trained on has gaps of 0-70 mm, so the default sits inside that range."""
    roi_size: Tuple[int, int, int] = (60, 60, 320)
    iso_spacing_mm: float = 1.0
    num_primaries: float = 1e7
    num_threads: int = 0
    beamlet_mode: bool = False
    """Run a whole (patient, gantry, energy) block as one MCsquare beamlet-mode
    call instead of one call per beamlet: ~3x faster (measured), same dose, but no
    per-beamlet ``stat_uncertainty`` and the block's dense dose grids sit on disk
    together until they are cropped (see :func:`_generate_energy_block`)."""
    rng_seed: int = 0
    min_deposition_ratio: float = 0.5
    output_root: str = "/scratch/mstryja/DoTA_dataset_v2"
    experiment_prefix: str = "beamlet_angle_robustness"
    experiment_version: Optional[int] = 2
    make_figures: bool = False
    overwrite: bool = False


def robustness_config_from_dict(
    r: dict, *, grid_n=None, num_primaries=None, make_figures=None, overwrite=None,
    n_gantry=None, beamlet_mode=None, default_prefix: str = "beamlet_angle_robustness",
) -> RobustnessConfig:
    """Build a :class:`RobustnessConfig` from a config ``robustness`` sub-dict.

    Shared by the robustness and patient-set CLIs so the (long) YAML->config
    plumbing lives in one place. CLI flags override the matching YAML keys.
    """
    gr = r.get("gantry_ranges")
    return RobustnessConfig(
        energies=[float(e) for e in r.get("energies", [90.0, 140.0, 200.0])],
        theta_x_range=tuple(r.get("theta_x_range", (-2.0, 2.0))),
        theta_y_range=tuple(r.get("theta_y_range", (-2.0, 2.0))),
        grid_n=int(grid_n if grid_n is not None else r.get("grid_n", 18)),
        angles=[(float(a[0]), float(a[1])) for a in r["angles"]] if r.get("angles") else None,
        gantry_mode=r.get("gantry_mode", "fixed"),
        n_gantry=int(n_gantry if n_gantry is not None else r.get("n_gantry", 1)),
        gantry_value=float(r.get("gantry_value", 90.0)),
        gantry_ranges=tuple(tuple(float(x) for x in pair) for pair in gr) if gr
        else ((30.0, 120.0), (240.0, 330.0)),
        gantry_min=float(r.get("gantry_min", 0.0)),
        gantry_max=float(r.get("gantry_max", 360.0)),
        gantry_seed=int(r.get("gantry_seed", 1234)),
        rotate_to_canonical=bool(r.get("rotate_to_canonical", True)),
        isocenter_mode=r.get("isocenter_mode", "grid_center"),
        body_hu_threshold=float(r.get("body_hu_threshold", -500.0)),
        border_only=bool(r.get("border_only", False)),
        beam_entrance_standoff_mm=float(r.get("beam_entrance_standoff_mm", 20.0)),
        roi_size=tuple(r.get("roi_size", (60, 60, 320))),
        iso_spacing_mm=float(r.get("iso_spacing_mm", 1.0)),
        num_primaries=float(num_primaries if num_primaries is not None else r.get("num_primaries", 1e7)),
        num_threads=int(r.get("num_threads", 0)),
        beamlet_mode=bool(beamlet_mode if beamlet_mode is not None
                          else r.get("beamlet_mode", False)),
        rng_seed=int(r.get("rng_seed", 0)),
        min_deposition_ratio=float(r.get("min_deposition_ratio", 0.5)),
        output_root=r.get("output_root", "/scratch/mstryja/DoTA_dataset_v2"),
        experiment_prefix=r.get("experiment_prefix", default_prefix),
        experiment_version=r.get("experiment_version", 2),
        make_figures=bool(make_figures if make_figures is not None else r.get("make_figures", False)),
        overwrite=bool(overwrite if overwrite is not None else r.get("overwrite", False)),
    )


def energy_tag(energy: float) -> str:
    """Filename-safe energy label: ``140.0 -> "140"``, ``102.6 -> "102p6"``.

    Integer energies keep the historical ``e140`` naming, so dirs and panels
    written before fractional energies existed are unchanged.
    """
    return f"{float(energy):g}".replace(".", "p")


def angle_tag(deg: float) -> str:
    """Filename-safe angle label (0.1 deg resolution): ``247.34 -> "247p3"``."""
    return f"{round(float(deg), 1):g}".replace(".", "p").replace("-", "m")


def _lattice_index(value: float, axis: np.ndarray, name: str) -> int:
    """Index of ``value`` on ``axis``; raises if it is not a lattice point."""
    ix = int(np.argmin(np.abs(axis - value)))
    if abs(float(axis[ix]) - value) > 1e-6:
        raise ValueError(
            f"{name}={value:g} is not on the grid_n={axis.size} lattice "
            f"{[round(float(a), 4) for a in axis]}; the GPR panels are indexed by "
            "lattice cell, so every explicit angle must land on one."
        )
    return ix


def build_angle_grid(
    tx_range, ty_range, n, angles: Optional[Sequence[Sequence[float]]] = None,
) -> List[Tuple[int, int, float, float]]:
    """Return the (ix, iy, theta_x, theta_y) sweep (row-major over theta_x).

    With ``angles`` given, only those explicit ``(theta_x, theta_y)`` pairs are
    returned, each carrying its index on the same ``n x n`` lattice (so a sparse
    sweep plots into the usual panel, with the unvisited cells left NaN).
    """
    txs = np.linspace(tx_range[0], tx_range[1], n)
    tys = np.linspace(ty_range[0], ty_range[1], n)
    if angles is None:
        return [(ix, iy, float(tx), float(ty))
                for ix, tx in enumerate(txs) for iy, ty in enumerate(tys)]
    out = []
    for tx, ty in angles:
        tx, ty = float(tx), float(ty)
        out.append((_lattice_index(tx, txs, "theta_x"), _lattice_index(ty, tys, "theta_y"),
                    tx, ty))
    return out


def sweep_z_half_extent_mm(cfg: RobustnessConfig, d_smy: float) -> float:
    """Slice-axis half-extent (mm) the sweep needs around the isocenter.

    ``theta_x`` steers the beamlet along the CT's **slice** axis (the BDL helper
    maps the first angle onto the z spot component, ``d_smy * tan(theta_x)``): at
    the isocenter plane the ray sits that far from the isocenter, and the ROI adds
    half of its 60-voxel lateral window on top. A CT whose z extent is shorter than
    twice this cannot hold the outer beamlets -- the crop runs off the end of the
    scan and the beamlet is dropped by the ``roi_out_of_bounds`` QA gate, after its
    MC has already been paid for.
    """
    grid = build_angle_grid(cfg.theta_x_range, cfg.theta_y_range, cfg.grid_n, cfg.angles)
    max_off = max(abs(angles_to_spot_position(tx, 0.0, 1.0, d_smy)[1]) for _, _, tx, _ in grid)
    return float(max_off) + cfg.roi_size[0] / 2.0


def sweep_lateral_half_extents(cfg: RobustnessConfig, d_smx: float, d_smy: float):
    """Half-extents ``(z, y)`` in mm that the sweep's beamlets can reach laterally.

    The steering offset of the outermost beamlet plus half the ROI window, per axis
    (``theta_x`` steers z, ``theta_y`` steers y). Used to look for the patient only
    where the beamlets actually pass, so a couch rail or distant anatomy does not
    define the beam entrance.
    """
    grid = build_angle_grid(cfg.theta_x_range, cfg.theta_y_range, cfg.grid_n, cfg.angles)
    z_off = max(abs(angles_to_spot_position(tx, 0.0, 1.0, d_smy)[1]) for _, _, tx, _ in grid)
    y_off = max(abs(angles_to_spot_position(0.0, ty, d_smx, 1.0)[0]) for _, _, _, ty in grid)
    return z_off + cfg.roi_size[0] / 2.0, y_off + cfg.roi_size[1] / 2.0


def max_theta_x_deg(z_extent_mm: float, cfg: RobustnessConfig, d_smy: float) -> float:
    """Largest ``|theta_x|`` (deg) a CT of ``z_extent_mm`` can hold for this ROI."""
    reach = z_extent_mm / 2.0 - cfg.roi_size[0] / 2.0
    return float(np.degrees(np.arctan(max(reach, 0.0) / d_smy)))


def sweep_fits_ct(ct: "sitk.Image", cfg: RobustnessConfig, d_smy: float) -> Tuple[bool, float, float]:
    """Return ``(fits, z_extent_mm, max_theta_x_deg)`` for a (resampled) CT."""
    z_extent = float(ct.GetSize()[2] * ct.GetSpacing()[2])
    return (z_extent / 2.0 >= sweep_z_half_extent_mm(cfg, d_smy),
            z_extent, max_theta_x_deg(z_extent, cfg, d_smy))


def _draw_gantry(cfg: RobustnessConfig, rng: random.Random) -> float:
    """One gantry draw from ``rng`` under the configured random mode."""
    if cfg.gantry_mode == "uniform_random":
        return rng.uniform(cfg.gantry_min, cfg.gantry_max)
    if cfg.gantry_mode == "bimodal_random":
        lo1, hi1 = cfg.gantry_ranges[0]
        lo2, hi2 = cfg.gantry_ranges[1]
        return rng.uniform(lo1, hi1) if rng.random() < 0.5 else rng.uniform(lo2, hi2)
    raise ValueError(f"unknown gantry_mode {cfg.gantry_mode!r}")


def resolve_gantry(cfg: RobustnessConfig, patient_uid: str) -> float:
    """Gantry angle for a patient: fixed, or a seeded random draw (reproducible).

    The draw is seeded per patient UID, so a given patient always gets the same
    gantry across reruns (resumable, reproducible).
    """
    if cfg.gantry_mode == "fixed":
        return float(cfg.gantry_value)
    return _draw_gantry(cfg, random.Random(f"{cfg.gantry_seed}:{patient_uid}"))


def resolve_gantries(cfg: RobustnessConfig, patient_uid: str) -> List[float]:
    """The ``n_gantry`` field angles for a patient, shared across all energies.

    Drawn from the single patient-seeded stream, so the first angle is exactly
    :func:`resolve_gantry` (an ``n_gantry: 1`` rerun reproduces earlier runs) and
    the same patient always gets the same set. Draws whose 0.1-deg output-dir tag
    collides are discarded, so the per-gantry dirs stay distinct. ``fixed`` mode
    has nothing to sample and always yields the single configured angle.
    """
    if cfg.gantry_mode == "fixed":
        return [float(cfg.gantry_value)]
    n = max(1, int(cfg.n_gantry))
    rng = random.Random(f"{cfg.gantry_seed}:{patient_uid}")
    draws: List[float] = []
    tags = set()
    for _ in range(100 * n):
        g = _draw_gantry(cfg, rng)
        if angle_tag(g) in tags:
            continue
        tags.add(angle_tag(g))
        draws.append(g)
        if len(draws) == n:
            return draws
    raise RuntimeError(
        f"could not draw {n} distinct gantry angles for {patient_uid} "
        f"(mode={cfg.gantry_mode}, range too narrow?)")
