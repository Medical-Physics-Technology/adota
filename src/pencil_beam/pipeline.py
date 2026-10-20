"""Commissioning stages that run on saved MC beamlets: fit the model, validate it in water.

The MC side (:func:`src.pencil_beam.commissioning.run_commissioning_beamlets`) writes
one ``e{E}_s{s}.npz`` per (energy, source-to-surface) run. :func:`fit_from_runs`
turns a directory of those into a :class:`~src.pencil_beam.beam_model.PencilBeamModel`;
:func:`validate_water` re-simulates held-out geometries (other SSDs, energies
between the nodes, more primaries) and scores the pencil beam against them.
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.mc_generation.sweep import energy_tag
from src.pencil_beam.commissioning import (
    CommissioningConfig,
    DepthProfile,
    commissioning_phantom,
    depth_profile,
    distal_r80,
    fit_energy_node,
    fit_model,
    kernel_sigma_profile,
    physical_to_wet,
    run_commissioning_beamlets,
)
from src.pencil_beam.metrics import GammaCriterion, dose_agreement, dose_bbox, gamma_pass_rates

logger = logging.getLogger(__name__)

__all__ = ["fit_from_runs", "group_runs", "provenance", "validate_water", "write_rows_csv"]

_RUN = re.compile(r"e(?P<e>[0-9.]+)_s(?P<s>[0-9.]+)\.npz$")
SIGMA_DEPTH_FRACTIONS = (0.25, 0.5, 0.75, 0.9)


def group_runs(runs_dir: Path) -> Dict[float, List[Tuple[float, Path]]]:
    """``{energy: [(surface_to_iso_mm, path), ...]}`` of a commissioning run directory."""
    out: Dict[float, List[Tuple[float, Path]]] = {}
    for p in sorted(Path(runs_dir).glob("e*_s*.npz")):
        m = _RUN.search(p.name)
        if m:
            out.setdefault(float(m["e"]), []).append((float(m["s"]), p))
    return {e: sorted(v) for e, v in sorted(out.items())}


def _kernel_sigma_task(task):
    path, wet, r80 = task
    return kernel_sigma_profile(np.load(path)["dose"], wet, r80)


def fit_from_runs(runs_dir: Path, cfg: CommissioningConfig, rlsp_water: float, rlsp_air: float,
                  vsadx: float, vsady: float, meta: dict, n_workers: int = 8) -> "PencilBeamModel":  # noqa: F821
    """Fit every energy node found in ``runs_dir`` (all SSDs of an energy together)."""
    groups = group_runs(runs_dir)
    if not groups:
        raise FileNotFoundError(f"no commissioning runs (e*_s*.npz) in {runs_dir}")
    source_distance = 0.5 * (vsadx + vsady)
    loaded = {}
    tasks = []
    for energy, runs in groups.items():
        profs = [DepthProfile.from_dict(np.load(p)) for _, p in runs]
        wets = [physical_to_wet(pr.x_mm, cfg.air_front_mm, rlsp_water, rlsp_air) for pr in profs]
        r80 = float(np.nanmean([distal_r80(w, pr.idd) for pr, w in zip(profs, wets)]))
        loaded[energy] = (runs, profs, wets)
        tasks += [(p, w, r80) for (_, p), w in zip(runs, wets)]
    logger.info("fit: %d energies, %d runs, kernel-form sigma fits on %d workers", len(groups), len(tasks), n_workers)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        sig_iter = iter(list(ex.map(_kernel_sigma_task, tasks, chunksize=1)))
    nodes = []
    for energy, (runs, profs, wets) in loaded.items():
        sigmas = [next(sig_iter) for _ in runs]
        node = fit_energy_node(energy, profs, wets, [s for s, _ in runs], cfg.air_front_mm,
                               source_distance, cfg, sigmas=sigmas)
        res = node["diagnostics"]["sigma_air_fit_residual_mm"]
        logger.info("  E=%6.1f MeV: R80 %.2f mm (spread %.2f), sigma_air residual max %.3f mm", energy,
                    node["r80"], np.ptp(node["diagnostics"]["r80_per_ssd"]), float(np.max(np.abs(res))))
        nodes.append(node)
    meta = dict(meta, rlsp_water=rlsp_water, rlsp_air=rlsp_air,
                commissioning=dataclasses.asdict(cfg), runs_dir=str(runs_dir))
    return fit_model(nodes, vsadx, vsady, meta)


def _sigma_at(wet: np.ndarray, sigma: np.ndarray, depth: float) -> float:
    ok = np.isfinite(sigma)
    return float(np.interp(depth, wet[ok], sigma[ok])) if ok.any() else float("nan")


def validate_water(
    runner, engine, cfg: CommissioningConfig, energies: Sequence[float], surfaces: Sequence[float],
    out_dir: Path, num_primaries: float, criteria: Sequence[GammaCriterion],
    rlsp_water: float, rlsp_air: float, gamma_backend: str = "pymedphys",
    gamma_backend_options: Optional[dict] = None, make_figures: bool = True,
) -> List[dict]:
    """Score the commissioned pencil beam against fresh MC water beamlets.

    Width comparisons fit a single 2D Gaussian to both engines (like for like),
    so they measure the delivered profile rather than the tabulated sigma.
    """
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    vcfg = dataclasses.replace(cfg, num_primaries=float(num_primaries))
    paths = run_commissioning_beamlets(runner, vcfg, out_dir / "runs", energies=energies, surfaces=surfaces)
    nodes = np.asarray(engine.model.energies)
    rows = []
    for path in paths:
        m = _RUN.search(path.name)
        energy, s = float(m["e"]), float(m["s"])
        data = np.load(path)
        ct, iso = commissioning_phantom(cfg, s)
        pb = engine.dose(ct, energy, [(0.0, 0.0)], iso, weights=[cfg.spot_weight])
        mc = data["dose"]
        prof_mc = DepthProfile.from_dict(data)
        prof_pb = depth_profile(pb.dose, weight=cfg.spot_weight, min_slice_fraction=cfg.min_slice_fraction)
        wet = physical_to_wet(prof_mc.x_mm, cfg.air_front_mm, rlsp_water, rlsp_air)
        r80_mc, r80_pb = distal_r80(wet, prof_mc.idd), distal_r80(wet, prof_pb.idd)
        row = {"energy": energy, "surface_to_iso_mm": s, "node": bool(np.any(np.abs(nodes - energy) < 1e-3)),
               "num_primaries": float(num_primaries), "mc_seconds": float(data["mc_seconds"]),
               "pba_seconds": float(pb.seconds), "r80_mc_mm": r80_mc, "dr80_mm": r80_pb - r80_mc,
               "idd_peak_ratio": float(prof_pb.idd.max() / prof_mc.idd.max()),
               "idd_integral_ratio": float(prof_pb.idd.sum() / prof_mc.idd.sum())}
        dsig = []
        for f in SIGMA_DEPTH_FRACTIONS:
            z = f * r80_mc
            i = int(np.argmin(np.abs(wet - z)))
            ds = _sigma_at(wet, prof_pb.sigma, z) - _sigma_at(wet, prof_mc.sigma, z)
            row[f"dsigma_mm_at_{int(f * 100)}pct_r80"] = ds
            row[f"cax_dose_ratio_at_{int(f * 100)}pct_r80"] = float(pb.dose[:, :, i].max() / mc[:, :, i].max())
            row[f"mc_sigma_y_minus_z_mm_at_{int(f * 100)}pct_r80"] = float(
                np.interp(z, wet, np.nan_to_num(prof_mc.sigma_y - prof_mc.sigma_z)))
            dsig.append(abs(ds))
        row["max_abs_dsigma_mm"] = float(np.nanmax(dsig))
        row.update({f"agreement_{k}": v for k, v in dose_agreement(mc, pb.dose).items()})
        gbox = dose_bbox([mc, pb.dose], low_frac=0.02, margin=6)
        row.update({f"gpr_{k}": v for k, v in gamma_pass_rates(
            mc[gbox], pb.dose[gbox], criteria, backend=gamma_backend,
            backend_options=gamma_backend_options).items()})
        rows.append(row)
        logger.info("  E=%6.1f s=%5.0f: dR80 %+.2f mm, IDD peak %.3f, max|dsigma| %.2f mm, %s %.1f%%",
                    energy, s, row["dr80_mm"], row["idd_peak_ratio"], row["max_abs_dsigma_mm"],
                    criteria[0].key, row[f"gpr_{criteria[0].key}"])
        if make_figures:
            from src.figures.pencil_beam import commissioning_run_figure

            commissioning_run_figure(
                wet, {"MC": prof_mc.idd, "PBA": prof_pb.idd}, {"MC": prof_mc.sigma, "PBA": prof_pb.sigma},
                str(out_dir / "figures" / f"water_e{energy_tag(energy)}_s{energy_tag(s)}"),  # no dots:
                # save_figure_as_publication_formats swaps the suffix
                title=f"water, E = {energy:g} MeV, surface {s:g} mm before isocenter "
                      f"({'node' if row['node'] else 'interpolated'}), {num_primaries:.0e} primaries")
    return rows


def provenance(mcsquare_install: Path, bdl_file: str, scanner: str, kernels_path: Optional[Path] = None) -> dict:
    """What produced a model or a result: engine + BDL hashes, DoseCUDA build, adota commit."""
    import hashlib
    import json
    import subprocess

    def sha256(path: Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    repo = Path(__file__).resolve().parents[2]

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    install = Path(mcsquare_install)
    out = {
        "mcsquare_install": str(install), "mcsquare_sha256": sha256(install / "MCsquare_linux"),
        "bdl_file": bdl_file, "bdl_sha256": sha256(install / "BDL" / bdl_file), "scanner": scanner,
        "adota_commit": git("rev-parse", "HEAD"),
        # uncommitted or untracked code means the commit alone does not identify it
        "adota_uncommitted_changes": bool(git("status", "--porcelain", "--", "src", "scripts")),
    }
    if kernels_path is not None:
        info = Path(kernels_path).resolve().parent / "BUILD_INFO.json"
        out["dosecuda_kernels"] = str(Path(kernels_path).resolve())
        out["dosecuda_build"] = json.loads(info.read_text()) if info.exists() else None
    return out


def write_rows_csv(rows: Sequence[dict], path: Path) -> Path:
    """All rows to one CSV (union of keys, first-seen order)."""
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (",".join(map(str, v)) if isinstance(v, (list, tuple)) else v) for k, v in r.items()})
    return path
